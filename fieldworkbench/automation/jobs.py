"""Bounded background jobs operating exclusively on frozen project copies."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .commands import VECTOR, array, obj, text
from .protocol import RUNTIME_SETTINGS, CommandError, json_value, validate

TERMINAL = {"succeeded", "failed", "cancelled"}
CALCULATION_SCHEMAS = {
    "field": obj(
        {
            "points_m": array(VECTOR, 1, 100000),
            "field": {"type": "string", "enum": ["B", "H"]},
        },
        ["points_m"],
    ),
    "measurement": obj({"object_id": text()}, ["object_id"]),
    "drc": obj(),
    "optimizer": {"type": "object"},
    "optimizer_tuning": {"type": "object"},
}


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    project_id: str
    source_revision: str
    kind: str
    settings: dict
    document: dict
    runtime: dict
    created_at: float = field(default_factory=time.time)
    state: str = "queued"
    progress: dict = field(default_factory=dict)
    error: dict | None = None
    result: Any = None
    source_adapter: Any = None
    discard_when_done: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)

    def info(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_revision": self.source_revision,
            "kind": self.kind,
            "created_at": self.created_at,
            "state": self.state,
            "cancel_requested": self.cancel.is_set(),
            "progress": copy.deepcopy(self.progress),
            "error": self.error,
        }


def result_summary(value: Any) -> Any:
    """Bound summaries without truncating the full retained/exportable result."""
    if isinstance(value, dict):
        return {
            k: result_summary(v)
            for k, v in value.items()
            if k
            not in {
                "source_document",
                "field_vectors_t",
                "world_points_m",
                "local_points_m",
                "points",
                "values",
                "magnitude",
                "vectors_t",
                "points_m",
                "figure",
            }
            and not str(k).startswith("_")
        }
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return {
                "count": len(value),
                "preview": [result_summary(v) for v in value[:3]],
            }
        return [result_summary(v) for v in value]
    return json_value(value)


class JobManager:
    def __init__(
        self, *, adapter_factory: Callable | None = None, max_jobs=32, max_pending=8
    ):
        self._factory = adapter_factory
        self._max_jobs, self._max_pending = max_jobs, max_pending
        self._jobs: dict[str, Job] = {}
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._worker: threading.Thread | None = None

    def start(self, context, source_revision, kind, settings):
        validate(settings, CALCULATION_SCHEMAS[kind], "params.settings")
        with self._condition:
            if self._closed:
                raise CommandError("shutting_down", "FWB is shutting down.")
            if len(self._jobs) >= self._max_jobs:
                raise CommandError(
                    "job_limit", "Forget completed jobs before starting more."
                )
            if (
                sum(j.state not in TERMINAL for j in self._jobs.values())
                >= self._max_pending
            ):
                raise CommandError("busy", "The calculation queue is full.")
            adapter = context.adapter
            job = Job(
                uuid4().hex,
                context.project_id,
                source_revision,
                kind,
                copy.deepcopy(settings),
                adapter.document(),
                {key: copy.deepcopy(getattr(adapter, key)) for key in RUNTIME_SETTINGS},
            )
            self._jobs[job.id] = job
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._run, name="FWB-command-jobs", daemon=True
                )
                self._worker.start()
            self._condition.notify_all()
            return job.info()

    def _get(self, project_id, job_id):
        job = self._jobs.get(job_id)
        if job is None or job.project_id != project_id:
            raise CommandError("job_not_found", "No such job in this project.")
        return job

    def list(self, project_id):
        with self._condition:
            return [j.info() for j in self._jobs.values() if j.project_id == project_id]

    def info(self, project_id, job_id):
        with self._condition:
            return self._get(project_id, job_id).info()

    def completed(self, project_id, job_id):
        with self._condition:
            job = self._get(project_id, job_id)
            if job.state != "succeeded":
                raise CommandError(
                    "result_unavailable", "Job has not succeeded.", state=job.state
                )
            # Terminal result/source objects are immutable from this point onward.
            return job

    def cancel(self, project_id, job_id):
        with self._condition:
            job = self._get(project_id, job_id)
            if job.state not in TERMINAL:
                job.cancel.set()
                if job.state == "queued":
                    job.state = "cancelled"
            return job.info()

    def cancel_project(self, project_id):
        """Cancel and release jobs belonging to a closing project."""
        with self._condition:
            for job in list(self._jobs.values()):
                if job.project_id != project_id:
                    continue
                if job.state not in TERMINAL:
                    self.cancel(project_id, job.id)
                if job.state in TERMINAL:
                    del self._jobs[job.id]
                else:
                    job.discard_when_done = True

    def forget(self, project_id, job_id):
        with self._condition:
            job = self._get(project_id, job_id)
            if job.state not in TERMINAL:
                raise CommandError(
                    "job_running", "Cancel and wait for the job before forgetting it."
                )
            del self._jobs[job_id]
            return {"forgotten": job_id}

    def shutdown(self):
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job.state not in TERMINAL:
                    job.cancel.set()
                    if job.state == "queued":
                        job.state = "cancelled"
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._closed
                        or any(j.state == "queued" for j in self._jobs.values())
                    )
                )
                if self._closed:
                    return
                job = next(j for j in self._jobs.values() if j.state == "queued")
                job.state = "running"
            self._execute_job(job)
            with self._condition:
                if job.discard_when_done:
                    self._jobs.pop(job.id, None)
            # Do not retain the last job while the worker is idle.
            job = None

    def _execute_job(self, job):
        try:
            factory = self._factory
            if factory is None:
                from ..studio_adapter import StudioAdapter

                factory = StudioAdapter
            adapter = factory(job.document)
            # Follow StudioAdapter.detached_copy's runtime policy without
            # creating or sharing a live numerical session on the GUI thread.
            adapter.set_coil_modeling_method(job.runtime["coil_modeling_method"])
            adapter.set_solver_memory_budget_mb(job.runtime["solver_memory_budget_mb"])
            adapter.set_optimizer_worker_count(job.runtime["optimizer_worker_count"])
            adapter.set_solver_diagnostics_enabled(
                job.runtime["solver_diagnostics_enabled"]
            )
            adapter.set_fluxline_minimum_field_cutoff(
                job.runtime["fluxline_minimum_field_cutoff_enabled"],
                job.runtime["fluxline_minimum_field_cutoff_percent"],
            )
            adapter.set_fluxline_conductor_cutoff(
                job.runtime["fluxline_conductor_cutoff_enabled"],
                job.runtime["fluxline_conductor_cutoff_mm"],
            )

            def progress(update):
                if job.cancel.is_set():
                    raise JobCancelled()
                with self._condition:
                    job.progress = json_value(update)

            progress({"message": "Starting calculation"})
            result = self._calculate(adapter, job, progress)
            if job.cancel.is_set():
                raise JobCancelled()
        except Exception as error:  # noqa: BLE001 - numerical job boundary
            with self._condition:
                if job.cancel.is_set() or isinstance(error, JobCancelled):
                    job.state = "cancelled"
                else:
                    job.state = "failed"
                    job.error = {
                        "code": "calculation_failed",
                        "message": str(error),
                        "type": type(error).__name__,
                    }
        else:
            with self._condition:
                if job.cancel.is_set():
                    job.state = "cancelled"
                else:
                    job.result, job.source_adapter = result, adapter
                    job.state = "succeeded"
                    job.progress = {"percent": 100, "message": "Completed"}

    @staticmethod
    def _calculate(adapter, job, progress):
        settings = job.settings
        if job.kind == "field":
            return adapter.get_field(
                points=settings["points_m"],
                field=settings.get("field", "B"),
                progress_callback=progress,
            )
        if job.kind == "measurement":
            return adapter.analyze_measurement(
                settings["object_id"], progress_callback=progress
            )
        if job.kind == "drc":
            return adapter.run_drc(progress_callback=progress)
        method = (
            adapter.run_optimizer_tuning
            if job.kind == "optimizer_tuning"
            else adapter.run_optimizer
        )
        return method(settings, progress_callback=progress, cancel_event=job.cancel)
