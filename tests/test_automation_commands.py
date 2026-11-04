"""Automation contract tests against the real numerical adapter, without a GUI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fwb-test-matplotlib")

from fieldworkbench.automation.commands import COMMANDS
from fieldworkbench.automation.protocol import (
    CommandError,
    dumps,
    loads,
    revision,
    validate,
)
from fieldworkbench.automation.service import CommandService
from fieldworkbench.project_context import ProjectContext
from fieldworkbench.studio_adapter import StudioAdapter


class MemoryHost:
    instance_id = "a" * 32

    def __init__(self):
        self.projects = []
        self.refreshes = 0
        self.blocked = False
        self.new_project()

    def contexts(self):
        return self.projects

    def context(self, project_id):
        for context in self.projects:
            if context.project_id == project_id:
                return context
        raise CommandError("project_not_found", "No such project")

    def new_project(self):
        context = ProjectContext(StudioAdapter())
        self.projects.append(context)
        return context

    def open_project(self, path):
        adapter = StudioAdapter()
        adapter.load_file(path)
        context = ProjectContext(adapter, path=path)
        self.projects.append(context)
        return context

    def check_write(self, context):
        if self.blocked:
            raise CommandError("interactive_edit_active", "Busy")

    def changed(self, context):
        self.refreshes += 1

    def save_project(self, context, path, overwrite):
        target = Path(path) if path else context.path
        if target is None:
            raise CommandError("path_required", "Provide path")
        context.adapter.save_file(target, overwrite=overwrite or target == context.path)
        context.mark_saved(target)
        return {"path": str(target)}


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.host = MemoryHost()
        self.context = self.host.projects[0]
        self.adapter = self.context.adapter
        self.service = CommandService(self.host)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.service.jobs.shutdown)

    def call(self, command, **params):
        spec = COMMANDS[command]
        if spec.project:
            params.setdefault("project_id", self.context.project_id)
        if spec.revision_required:
            params.setdefault("expected_revision", revision(self.adapter))
        return self.service.execute(command, params)

    def ok(self, command, **params):
        result = self.call(command, **params)
        self.assertTrue(result["ok"], result)
        return result["result"]

    def add_coil(self):
        return self.ok("object.add", template="circle")["object_id"]

    def wait_job(self, job):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            info = self.ok("job.get", job_id=job["id"])
            if info["state"] in {"succeeded", "failed", "cancelled"}:
                return info
            time.sleep(0.01)
        self.fail("Job timed out")

    def test_catalog_is_schema_valid_and_tool_names_unique(self):
        names = [c.tool_name for c in COMMANDS.values()]
        self.assertEqual(len(names), len(set(names)))
        for command in COMMANDS.values():
            self.assertEqual(command.input_schema["additionalProperties"], False)
        self.assertEqual(len(self.ok("commands.list")["commands"]), len(COMMANDS))

    def test_unknown_commands_and_parameters_do_not_edit(self):
        before = self.adapter.document()
        self.assertEqual(
            self.call("object.add", template="circle", typo=1)["error"]["code"],
            "invalid_params",
        )
        self.assertEqual(
            self.service.execute("eval", {"code": "anything"})["error"]["code"],
            "unknown_command",
        )
        self.assertEqual(before, self.adapter.document())

    def test_project_id_and_revision_are_required(self):
        self.assertFalse(self.service.execute("objects.list")["ok"])
        self.assertFalse(
            self.service.execute(
                "object.add",
                {"project_id": self.context.project_id, "template": "circle"},
            )["ok"]
        )

    def test_stale_revision_rejected_after_external_edit(self):
        old = revision(self.adapter)
        self.adapter.add_template("circle")
        result = self.call("notes.set", notes="lost update", expected_revision=old)
        self.assertEqual(result["error"]["code"], "revision_conflict")
        self.assertEqual(self.adapter.get_scene_notes(), "")

    def test_runtime_policy_changes_revision(self):
        before = revision(self.adapter)
        self.adapter.set_coil_modeling_method("centreline")
        self.assertNotEqual(before, revision(self.adapter))

    def test_multiple_projects_never_follow_active_tab(self):
        other = self.host.new_project()
        self.add_coil()
        self.assertEqual(len(self.ok("objects.list")), 1)
        self.assertEqual(other.adapter.list_objects(), [])

    def test_batch_is_atomic_and_one_undo_step(self):
        oid = self.add_coil()
        before = self.adapter.document()
        self.ok(
            "objects.edit",
            operations=[
                {
                    "method": "set_param",
                    "params": {"object_id": oid, "name": "diameter", "value": 0.2},
                },
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": oid, "drive_current_a": 0.3},
                },
            ],
        )
        self.assertEqual(self.adapter.history()["undo"], 2)
        self.ok("project.undo")
        self.assertEqual(self.adapter.document(), before)
        self.ok("project.redo")
        self.assertEqual(
            self.adapter.coil_physical_properties(oid)["drive_current_a"], 0.3
        )

    def test_failed_batch_rolls_back_earlier_operations(self):
        oid = self.add_coil()
        before, history = self.adapter.document(), self.adapter.history()
        result = self.call(
            "objects.edit",
            operations=[
                {
                    "method": "set_param",
                    "params": {"object_id": oid, "name": "diameter", "value": 0.2},
                },
                {
                    "method": "set_param",
                    "params": {
                        "object_id": oid,
                        "name": "invalid_parameter",
                        "value": 3,
                    },
                },
            ],
        )
        self.assertFalse(result["ok"], result)
        self.assertEqual(self.adapter.document(), before)
        self.assertEqual(self.adapter.history(), history)

    def test_unknown_physical_field_rejected(self):
        oid = self.add_coil()
        result = self.call(
            "objects.edit",
            operations=[
                {"method": "set_coil_physical", "params": {"object_id": oid, "typo": 8}}
            ],
        )
        self.assertEqual(result["error"]["code"], "invalid_params")

    def test_geometry_partial_transform_preserves_rotation(self):
        oid = self.ok("object.add", template="guide_box")["object_id"]
        self.ok(
            "objects.edit",
            operations=[
                {
                    "method": "set_transform",
                    "params": {"object_id": oid, "orientation": [0, 0, 45]},
                }
            ],
        )
        self.ok(
            "objects.edit",
            operations=[
                {
                    "method": "set_transform",
                    "params": {"object_id": oid, "position": [0.1, 0.2, 0.3]},
                }
            ],
        )
        transform = self.ok("object.get", object_id=oid)["transform"]
        self.assertAlmostEqual(transform["orientation"][2], 45)
        self.assertEqual(transform["position"], [0.1, 0.2, 0.3])

    def test_private_modal_edit_blocks_mutation(self):
        self.host.blocked = True
        self.assertEqual(
            self.call("object.add", template="circle")["error"]["code"],
            "interactive_edit_active",
        )
        self.assertTrue(self.call("objects.list")["ok"])

    def test_refresh_failure_does_not_misreport_committed_edit(self):
        def fail(context):
            raise RuntimeError("view unavailable")

        self.host.changed = fail
        result = self.call("object.add", template="circle")
        self.assertTrue(result["ok"])
        self.assertEqual(result["warnings"][0]["code"], "refresh_failed")
        self.assertEqual(len(self.adapter.list_objects()), 1)

    def test_save_export_and_open_preserve_baselines(self):
        self.add_coil()
        target = Path(self.directory.name) / "project.magpy.json"
        self.ok("project.save", path=str(target))
        self.assertFalse(self.context.is_dirty())
        self.ok("notes.set", notes="unsaved")
        exported = Path(self.directory.name) / "copy.magpy.json"
        self.ok("project.export", path=str(exported))
        self.assertEqual(self.context.path, target)
        self.assertTrue(self.context.is_dirty())
        reopened = self.ok("project.open", path=str(exported))
        self.assertEqual(
            self.host.context(reopened["id"]).adapter.get_scene_notes(), "unsaved"
        )
        self.assertEqual(
            self.call("project.export", path=str(exported))["error"]["code"],
            "file_exists",
        )

    def test_no_clobber_save_leaves_existing_file_and_no_temp(self):
        target = Path(self.directory.name) / "existing.json"
        target.write_text("existing")
        self.assertFalse(self.call("project.save", path=str(target))["ok"])
        self.assertEqual(target.read_text(), "existing")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [target])

    def test_real_field_job_uses_frozen_scene_and_exports_provenance(self):
        self.add_coil()
        job = self.ok(
            "calculation.start",
            kind="field",
            settings={"points_m": [[0, 0, 0]], "field": "B"},
        )
        self.ok("notes.set", notes="changed during calculation")
        info = self.wait_job(job)
        self.assertEqual(info["state"], "succeeded", info)
        self.assertTrue(info["stale"])
        result = self.ok("job.result", job_id=job["id"], full=True)
        self.assertGreater(result["data"]["magnitude"][0], 0)
        target = Path(self.directory.name) / "field.json"
        self.ok("job.export", job_id=job["id"], path=str(target))
        exported = json.loads(target.read_text())
        self.assertEqual(exported["job"]["source_revision"], job["source_revision"])
        self.assertNotIn(
            "changed during calculation", json.dumps(exported["source_document"])
        )
        self.assertIn("coil_modeling_method", exported["runtime"])
        self.assertEqual(
            self.call("job.apply", job_id=job["id"])["error"]["code"], "stale_result"
        )
        self.ok("job.forget", job_id=job["id"])
        self.assertEqual(self.ok("jobs.list"), [])

    def test_job_settings_validated_before_queueing(self):
        self.assertFalse(
            self.call(
                "calculation.start", kind="field", settings={"points_m": [[1, 2]]}
            )["ok"]
        )
        self.assertEqual(self.ok("jobs.list"), [])

    def test_jobs_are_project_scoped(self):
        job = self.ok("calculation.start", kind="drc", settings={})
        other = self.host.new_project()
        result = self.call("job.get", project_id=other.project_id, job_id=job["id"])
        self.assertEqual(result["error"]["code"], "job_not_found")
        self.wait_job(job)

    def test_json_nonfinite_inputs_rejected_outputs_normalized(self):
        with self.assertRaises(ValueError):
            loads('{"x": NaN}')
        self.assertEqual(dumps({"value": float("inf")}), '{"value":null}')
        with self.assertRaises(CommandError):
            validate({"settings": {"value": float("nan")}}, {"type": "object"})

    def test_measurement_job_saves_native_snapshot_and_undo_restores_document(self):
        self.add_coil()
        target = self.ok("object.add", template="guide_box")["object_id"]
        self.ok(
            "objects.edit",
            operations=[
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": target, "role": "measurement"},
                }
            ],
        )
        before = self.adapter.document()
        job = self.ok(
            "calculation.start", kind="measurement", settings={"object_id": target}
        )
        info = self.wait_job(job)
        self.assertEqual(info["state"], "succeeded", info)
        result = self.ok("job.apply", job_id=job["id"], name="Automation snapshot")
        snapshots = self.adapter.list_snapshots()
        self.assertEqual(snapshots[0]["id"], result["snapshot_id"])
        self.assertEqual(snapshots[0]["status"], "current")
        self.ok("project.undo")
        self.assertEqual(self.adapter.document(), before)

    def test_optimizer_candidate_application_uses_native_undo_path(self):
        oid = self.add_coil()
        original = self.service.jobs._calculate
        self.service.jobs._calculate = lambda adapter, job, progress: {
            "finalists": [
                {"id": "chosen", "mode": "tune_existing", "currents_a": {oid: 0.42}}
            ]
        }
        job = self.ok("calculation.start", kind="optimizer_tuning", settings={})
        self.assertEqual(self.wait_job(job)["state"], "succeeded")
        before = self.adapter.document()
        self.ok("job.apply", job_id=job["id"], candidate_id="chosen")
        self.assertEqual(
            self.adapter.coil_physical_properties(oid)["drive_current_a"], 0.42
        )
        self.ok("project.undo")
        self.assertEqual(self.adapter.document(), before)
        self.service.jobs._calculate = original

    def test_cancel_running_and_queued_jobs_and_release_retained_data(self):
        import threading

        gate, entered = threading.Event(), threading.Event()

        def blocking(adapter, job, progress):
            entered.set()
            if not gate.wait(5):
                raise RuntimeError("Test did not release worker")
            progress({"percent": 50})
            return {}

        self.service.jobs._calculate = blocking
        first = self.ok("calculation.start", kind="drc", settings={})
        self.assertTrue(entered.wait(2))
        second = self.ok("calculation.start", kind="drc", settings={})
        self.assertEqual(
            self.ok("job.cancel", job_id=second["id"])["state"], "cancelled"
        )
        self.assertEqual(self.ok("job.cancel", job_id=first["id"])["state"], "running")
        gate.set()
        self.assertEqual(self.wait_job(first)["state"], "cancelled")
        self.ok("job.forget", job_id=first["id"])
        self.ok("job.forget", job_id=second["id"])
        self.assertEqual(self.ok("jobs.list"), [])

    def test_closing_project_releases_completed_job_capacity(self):
        self.service.jobs._max_jobs = 1
        job = self.ok("calculation.start", kind="drc", settings={})
        self.assertEqual(self.wait_job(job)["state"], "succeeded")
        self.assertEqual(
            self.call("calculation.start", kind="drc", settings={})["error"]["code"],
            "job_limit",
        )
        self.service.jobs.cancel_project(self.context.project_id)
        self.assertEqual(self.ok("jobs.list"), [])
        next_job = self.ok("calculation.start", kind="drc", settings={})
        self.assertEqual(self.wait_job(next_job)["state"], "succeeded")

    def test_cli_discovery_and_help_do_not_import_numerical_or_gui_packages(self):
        code = "from fieldworkbench.automation.cli import main; import sys; main(['commands']); assert not any(m in sys.modules for m in ['PySide6', 'numpy', 'mcp', 'fieldworkbench.studio_adapter'])"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
