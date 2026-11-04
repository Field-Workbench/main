"""End-to-end local sockets, GUI dispatch, discovery, CLI, and timeout semantics."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fwb-test-matplotlib")

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication
from test_automation_commands import MemoryHost

from fieldworkbench.automation.connection import (
    CommandClient,
    discover,
    read_descriptor,
    request,
    select_instance,
)
from fieldworkbench.automation.protocol import CommandError, loads, revision
from fieldworkbench.automation.qt_host import AutomationController
from fieldworkbench.automation.server import LocalServer


class ManagerStub(QObject):
    project_closed = Signal(str)


class TransportHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = ManagerStub()
        self.controller = AutomationController(
            self.manager, directory=self.directory.name, timeout=0.4
        )
        self.host = MemoryHost()
        self.host.instance_id = self.controller.host.instance_id
        self.controller.service.host = self.host
        self.controller.start()
        self.addCleanup(self.controller.close)
        self.client = CommandClient(
            self.host.instance_id, self.directory.name, timeout=2
        )
        self.context = self.host.projects[0]

    def background(self, function):
        result, errors = [], []

        def run():
            try:
                result.append(function())
            except BaseException as error:  # noqa: BLE001 - forward worker failures to the test
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 10
        while thread.is_alive() and time.monotonic() < deadline:
            self.app.processEvents()
            thread.join(0.002)
        self.assertFalse(thread.is_alive(), "Background call timed out")
        if errors:
            raise errors[0]
        return result[0]


class TransportTests(TransportHarness):
    def test_socket_dispatch_executes_on_gui_thread(self):
        expected = threading.get_ident()
        checked = []
        original = self.host.contexts

        def contexts():
            checked.append(threading.get_ident())
            return original()

        self.host.contexts = contexts
        response = self.background(lambda: self.client.call("projects.list"))
        self.assertTrue(response["ok"], response)
        self.assertEqual(checked, [expected])

    def test_authenticated_discovery_and_private_files(self):
        descriptor = read_descriptor(self.controller.server.path)
        if os.name != "nt":
            self.assertEqual(self.controller.server.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(len(self.background(lambda: discover(self.directory.name))), 1)
        descriptor["token"] = "wrong"
        response = request(descriptor, "system.info")
        self.assertEqual(response["error"]["code"], "unauthorized")

    def test_replay_does_not_duplicate_write_and_conflicting_id_is_rejected(self):
        identifier = uuid4().hex
        params = {
            "project_id": self.context.project_id,
            "expected_revision": revision(self.context.adapter),
            "template": "circle",
        }
        first = self.background(
            lambda: self.client.call("object.add", params, request_id=identifier)
        )
        second = self.background(
            lambda: self.client.call("object.add", params, request_id=identifier)
        )
        self.assertEqual(first, second)
        self.assertEqual(len(self.context.adapter.list_objects()), 1)
        changed = dict(params, template="guide_box")
        response = self.background(
            lambda: self.client.call("object.add", changed, request_id=identifier)
        )
        self.assertEqual(response["error"]["code"], "request_id_conflict")

    def test_queued_timeout_cannot_apply_later(self):
        payload = {
            "protocol_version": 1,
            "request_id": uuid4().hex,
            "command": "object.add",
            "params": {
                "project_id": self.context.project_id,
                "expected_revision": revision(self.context.adapter),
                "template": "circle",
            },
        }
        responses = []
        thread = threading.Thread(
            target=lambda: responses.append(self.controller.dispatch(payload))
        )
        thread.start()
        thread.join(2)  # Deliberately do not process Qt events until timeout.
        self.assertFalse(thread.is_alive())
        self.assertEqual(responses[0]["error"]["code"], "command_timeout")
        self.assertFalse(responses[0]["error"]["details"]["outcome_unknown"])
        self.app.processEvents()
        self.assertEqual(self.context.adapter.list_objects(), [])

    def test_cli_round_trip_uses_shared_commands_and_never_prints_token(self):
        def run():
            return subprocess.run(
                [
                    sys.executable,
                    "fwb.py",
                    "--runtime-dir",
                    self.directory.name,
                    "--instance",
                    self.host.instance_id,
                    "call",
                    "projects.list",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )

        completed = self.background(run)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = loads(completed.stdout)
        self.assertEqual(response["result"][0]["id"], self.context.project_id)
        self.assertNotIn(
            read_descriptor(self.controller.server.path)["token"], completed.stdout
        )

    def test_multiple_instances_require_explicit_selection(self):
        second = LocalServer(
            lambda p: {
                "ok": True,
                "request_id": p["request_id"],
                "protocol_version": 1,
                "result": {"instance_id": second.instance_id},
            },
            directory=self.directory.name,
        )
        second.start()
        self.addCleanup(second.close)
        with self.assertRaises(CommandError) as raised:
            self.background(lambda: select_instance(directory=self.directory.name))
        self.assertEqual(raised.exception.code, "instance_required")

    def test_malformed_and_stale_descriptors_are_ignored(self):
        (Path(self.directory.name) / "bad.json").write_text("broken")
        self.assertEqual(len(self.background(lambda: discover(self.directory.name))), 1)

    def test_shutdown_removes_descriptor_and_refuses_commands(self):
        self.controller.close()
        self.assertFalse(self.controller.server.path.exists())
        response = self.controller.dispatch({"request_id": "after-close"})
        self.assertEqual(response["error"]["code"], "shutting_down")

    def test_explicit_project_edits_refresh_an_open_model_editor(self):
        from fieldworkbench.automation.qt_host import ProjectHost
        from fieldworkbench.automation.service import CommandService
        from fieldworkbench.project_manager import ProjectManagerWindow
        from fieldworkbench.studio_adapter import StudioAdapter

        settings = QSettings(
            str(Path(self.directory.name) / "model-settings.ini"),
            QSettings.Format.IniFormat,
        )
        manager = ProjectManagerWindow(settings=settings)
        service = CommandService(ProjectHost(manager, "c" * 32))
        page = manager.add_project(StudioAdapter())
        model = page.ensure_model_window()
        context = page.record.context
        try:
            response = service.execute(
                "object.add",
                {
                    "project_id": context.project_id,
                    "expected_revision": revision(context.adapter),
                    "template": "circle",
                },
            )
            self.assertTrue(response["ok"], response)
            self.assertNotIn("warnings", response)
            self.assertEqual(model.tree.topLevelItemCount(), 1)
            self.assertIn("*", manager.tabs.tabText(0))
            service.execute(
                "project.undo",
                {
                    "project_id": context.project_id,
                    "expected_revision": revision(context.adapter),
                },
            )
            self.assertEqual(model.tree.topLevelItemCount(), 0)
        finally:
            context.mark_saved()
            manager.close()
            manager.deleteLater()
            service.jobs.shutdown()
            self.app.processEvents()

    def test_real_project_manager_saves_the_target_project_without_dialogs(self):
        from fieldworkbench.automation.qt_host import ProjectHost
        from fieldworkbench.automation.service import CommandService
        from fieldworkbench.project_manager import ProjectManagerWindow
        from fieldworkbench.studio_adapter import StudioAdapter

        settings = QSettings(
            str(Path(self.directory.name) / "settings.ini"), QSettings.Format.IniFormat
        )
        manager = ProjectManagerWindow(settings=settings)
        service = CommandService(ProjectHost(manager, "b" * 32))
        try:
            first = manager.add_project(StudioAdapter()).record.context
            second = manager.add_project(StudioAdapter()).record.context
            first.adapter.add_template("circle")
            path = Path(self.directory.name) / "first.magpy.json"
            response = service.execute(
                "project.save",
                {
                    "project_id": first.project_id,
                    "expected_revision": revision(first.adapter),
                    "path": str(path),
                },
            )
            self.assertTrue(response["ok"], response)
            self.assertEqual(first.path, path)
            self.assertFalse(first.is_dirty())
            self.assertIsNone(second.path)
            self.assertEqual(len(loads(path.read_text())["objects"]), 1)
            # An explicit GUI close signals cancellation to the shared job service.
            closed = []
            manager.project_closed.connect(closed.append)
            manager.close_project_tab(0)
            self.assertEqual(closed, [first.project_id])
        finally:
            for i in range(manager.tabs.count()):
                manager.tabs.widget(i).record.context.mark_saved()
            manager.close()
            manager.deleteLater()
            service.jobs.shutdown()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
