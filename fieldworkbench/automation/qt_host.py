"""The only automation module that knows Qt or Project Manager widgets."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QApplication

from .protocol import MAX_MESSAGE_BYTES, CommandError, dumps, failure
from .server import LocalServer
from .service import CommandService


class ProjectHost:
    def __init__(self, manager, instance_id):
        self.manager, self.instance_id = manager, instance_id

    def pages(self):
        return [self.manager.tabs.widget(i) for i in range(self.manager.tabs.count())]

    def contexts(self):
        return [page.record.context for page in self.pages()]

    def page(self, context):
        for page in self.pages():
            if page.record.context is context:
                return page
        raise CommandError("project_not_found", "Project is no longer open.")

    def context(self, project_id):
        for context in self.contexts():
            if context.project_id == project_id:
                return context
        raise CommandError("project_not_found", "No open project with this ID.")

    def check_write(self, context):
        if self.manager._closing:
            raise CommandError("shutting_down", "FWB is closing.")
        if QApplication.activeModalWidget() is not None:
            raise CommandError(
                "interactive_edit_active",
                "Finish the active FWB dialog before an agent changes the project.",
            )
        if context is not None:
            window = self.page(context).record.model_window
            if window is not None:
                # Commit the existing notes undo group before an external edit;
                # its already-live document values and revision are unchanged.
                window._commit_scene_notes_session()

    def new_project(self):
        from ..studio_adapter import StudioAdapter

        adapter = StudioAdapter()
        adapter.new_empty()
        return self.manager.add_project(adapter).record.context

    def open_project(self, path):
        from ..studio_adapter import StudioAdapter

        target = Path(path).expanduser().resolve()
        page = self.manager._page_for_path(target)
        if page is None:
            adapter = StudioAdapter()
            adapter.load_file(target)  # Raises rather than opening a QMessageBox.
            page = self.manager.add_project(
                adapter, target, replace_page=self.manager._replaceable_startup_page()
            )
        return page.record.context

    def save_project(self, context, path, overwrite):
        page = self.page(context)
        target = Path(path).expanduser().resolve() if path else context.path
        if target is None:
            raise CommandError(
                "path_required", "Provide path when saving an unsaved project."
            )
        existing = self.manager._page_for_path(target)
        if existing is not None and existing is not page:
            raise CommandError(
                "project_already_open", "That path belongs to another open project."
            )
        is_current = context.path is not None and target == context.path
        self.manager.save_project_to_path(
            page, target, overwrite=overwrite or is_current
        )
        return {"path": str(target), "dirty": context.is_dirty()}

    def changed(self, context):
        page = self.page(context)
        window = page.record.model_window
        if window is not None:
            window._sync_scene_notes_dialog()
            window.refresh_scene()
        page._model_project_state_changed()
        self.manager._update_actions()


@dataclass
class _Ticket:
    payload: dict
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = "queued"
    response: dict | None = None


class AutomationController(QObject):
    """Qt queued delivery plus bounded request replay protection.

    A timed-out queued request is cancelled before it can touch live state.
    Started requests may finish; clients get outcome_unknown and their request ID.
    Completed results are replayable for the last 128 requests / 32 MiB.
    """

    submitted = Signal(object)

    def __init__(self, manager, *, directory=None, timeout=25.0):
        super().__init__(manager)
        self.timeout = timeout
        self._closed = False
        self._slots = threading.BoundedSemaphore(32)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.host = ProjectHost(manager, uuid4().hex)
        self.service = CommandService(self.host)
        self.submitted.connect(self._execute, Qt.ConnectionType.QueuedConnection)
        self.server = LocalServer(
            self.dispatch, instance_id=self.host.instance_id, directory=directory
        )
        manager.project_closed.connect(self.service.jobs.cancel_project)
        QApplication.instance().aboutToQuit.connect(self.close)

    def start(self):
        self.server.start()

    def dispatch(self, payload):
        identifier = payload["request_id"]
        if self._closed:
            return failure(CommandError("shutting_down", "FWB is closing."), identifier)
        if not self._slots.acquire(blocking=False):
            return failure(
                CommandError("busy", "The command queue is full."), identifier
            )
        ticket = _Ticket(payload, time.monotonic() + self.timeout)
        self.submitted.emit(ticket)
        if ticket.done.wait(self.timeout):
            return ticket.response
        with ticket.lock:
            if ticket.state == "finished":
                return ticket.response
            unknown = ticket.state == "running"
            if ticket.state == "queued":
                ticket.state = "expired"
        return failure(
            CommandError(
                "command_timeout",
                "FWB did not complete this command in time.",
                outcome_unknown=unknown,
                request_id=identifier,
            ),
            identifier,
        )

    @Slot(object)
    def _execute(self, ticket):
        try:
            with ticket.lock:
                if (
                    ticket.state == "expired"
                    or time.monotonic() >= ticket.deadline
                    or self._closed
                ):
                    ticket.response = failure(
                        CommandError(
                            "command_expired",
                            "Queued command expired without executing.",
                        ),
                        ticket.payload["request_id"],
                    )
                    ticket.state = "finished"
                    ticket.done.set()
                    return
                ticket.state = "running"
            payload = ticket.payload
            identifier = payload["request_id"]
            fingerprint = hashlib.sha256(dumps(payload).encode()).hexdigest()
            if identifier in self._cache:
                original, response, _ = self._cache[identifier]
                if fingerprint != original:
                    response = failure(
                        CommandError(
                            "request_id_conflict",
                            "This request ID was already used with different parameters.",
                        ),
                        identifier,
                    )
            else:
                response = self.service.execute(
                    payload["command"], payload["params"], request_id=identifier
                )
                size = len(dumps(response).encode())
                if size + 1 > MAX_MESSAGE_BYTES:
                    response = failure(
                        CommandError(
                            "response_too_large",
                            "Use project.export or job.export for this result.",
                        ),
                        identifier,
                    )
                    size = len(dumps(response).encode())
                self._cache[identifier] = (fingerprint, response, size)
                self._cache_bytes += size
                while len(self._cache) > 128 or self._cache_bytes > 32 * 1024 * 1024:
                    _, (_, _, discarded_size) = self._cache.popitem(last=False)
                    self._cache_bytes -= discarded_size
            with ticket.lock:
                ticket.response, ticket.state = response, "finished"
                ticket.done.set()
        finally:
            self._slots.release()

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self.service.jobs.shutdown()
        self.server.close()
