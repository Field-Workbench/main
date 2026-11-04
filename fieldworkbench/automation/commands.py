"""The single catalog consumed by the dispatcher, CLI help, and MCP tools/list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .protocol import CommandError


def obj(properties=None, required=(), *, extra=False):
    return {
        "type": "object",
        "properties": properties or {},
        "required": list(required),
        "additionalProperties": extra,
    }


def text(description=""):
    return {"type": "string", "minLength": 1, "description": description}


def array(items, minimum=0, maximum=10000):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


NUMBER = {"type": "number"}
VECTOR = array(NUMBER, 3, 3)
PROJECT = text("Explicit live project ID from projects.list.")
REVISION = text(
    "Revision from project.get; a mismatch rejects the edit without changing the project."
)


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    input_schema: dict[str, Any]
    project: bool = False
    writes: bool = False
    revision_required: bool = False

    @property
    def tool_name(self):
        return "fwb_" + self.name.replace(".", "_")

    def catalog(self):
        return {
            "name": self.name,
            "tool_name": self.tool_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "project": self.project,
            "writes": self.writes,
            "revision_required": self.revision_required,
        }


COMMANDS: dict[str, Command] = {}


def define(
    name, description, properties=None, required=(), *, project=False, writes=False
):
    properties = dict(properties or {})
    required = list(required)
    revision_required = project and writes
    if project:
        properties["project_id"] = PROJECT
        required.append("project_id")
    if revision_required:
        properties["expected_revision"] = REVISION
        required.append("expected_revision")
    COMMANDS[name] = Command(
        name, description, obj(properties, required), project, writes, revision_required
    )


def get_command(name: str) -> Command:
    try:
        return COMMANDS[name]
    except KeyError:
        raise CommandError("unknown_command", f"Unknown command: {name}.") from None


define("system.info", "Identify this FWB instance and command protocol.")
define("commands.list", "Discover all commands, input schemas, and write semantics.")
define("projects.list", "List open projects, IDs, revisions, and dirty state.")
define(
    "project.get",
    "Inspect a live project, including revision and undo/redo counts.",
    project=True,
)
define("project.new", "Create an empty project tab.", writes=True)
define(
    "project.open",
    "Open a project file without interactive dialogs.",
    {"path": text()},
    ["path"],
    writes=True,
)
define(
    "project.save",
    "Atomically save committed project state. Private editor drafts are not included.",
    {
        "path": text("Required for an unsaved project."),
        "overwrite": {"type": "boolean"},
    },
    project=True,
    writes=True,
)
define(
    "project.document",
    "Get the native project document. Large documents should be exported instead.",
    project=True,
)
define(
    "project.export",
    "Export a project copy without changing its saved baseline or path.",
    {"path": text(), "overwrite": {"type": "boolean"}},
    ["path"],
    project=True,
    writes=True,
)
define("project.undo", "Undo one project operation.", project=True, writes=True)
define("project.redo", "Redo one project operation.", project=True, writes=True)
define("templates.list", "List supported built-in object templates.")
define("objects.list", "List model objects and their stable IDs.", project=True)
define(
    "object.get",
    "Inspect an object's parameters, units, transform, and coil winding data.",
    {"object_id": text()},
    ["object_id"],
    project=True,
)
define(
    "object.add",
    "Add one built-in template as an undoable operation.",
    {"template": text(), "parent_id": text()},
    ["template"],
    project=True,
    writes=True,
)
define(
    "object.duplicate",
    "Duplicate one object as an undoable operation.",
    {"object_id": text()},
    ["object_id"],
    project=True,
    writes=True,
)
define(
    "objects.remove",
    "Remove objects as one undoable operation.",
    {"object_ids": array(text(), 1, 1000)},
    ["object_ids"],
    project=True,
    writes=True,
)
# Deliberately enumerate exposed editing operations; never reflect arbitrary adapter methods.
OPERATION_SCHEMAS = {
    "set_param": obj(
        {"object_id": text(), "name": text(), "value": {}},
        ["object_id", "name", "value"],
    ),
    "set_transform": obj(
        {"object_id": text(), "position": VECTOR, "orientation": VECTOR}, ["object_id"]
    ),
    "set_coil_physical": obj({"object_id": text()}, ["object_id"], extra=True),
    "set_geometry_style": obj(
        {
            "object_id": text(),
            "role": {"type": "string", "enum": ["visual", "exclusion", "measurement"]},
            "colour": text(),
            "opacity": {"type": "number", "minimum": 0.02, "maximum": 1},
        },
        ["object_id"],
    ),
    "set_measurement_settings": obj(
        {
            "object_id": text(),
            "sampling_mode": text(),
            "quality": text(),
            "defined_points_mm": array(VECTOR),
            "target_uT": NUMBER,
            "tolerance_pct": NUMBER,
        },
        ["object_id"],
    ),
    "apply_edit": obj(
        {"object_id": text(), "path": text(), "value": {}},
        ["object_id", "path", "value"],
    ),
}
define(
    "objects.edit",
    "Apply a validated batch atomically as one undo step. SI metres for model parameters and positions; orientation is a rotation vector in degrees. Coil physical fields retain their named units (e.g. drive_current_a, winding_axial_width_mm).",
    {
        "operations": array(
            obj(
                {
                    "method": {"type": "string", "enum": list(OPERATION_SCHEMAS)},
                    "params": {"type": "object"},
                },
                ["method", "params"],
            ),
            1,
            1000,
        )
    },
    ["operations"],
    project=True,
    writes=True,
)
define("notes.get", "Read project notes.", project=True)
define(
    "notes.set",
    "Set project notes as one undoable change.",
    {"notes": {"type": "string"}},
    ["notes"],
    project=True,
    writes=True,
)
define(
    "resources.list",
    "List first-class project resources without returning cached binary payloads.",
    project=True,
)
define(
    "optimizer.inspect",
    "Get eligible targets, existing coils, and default optimizer settings.",
    {"target_id": text()},
    project=True,
)
define(
    "calculation.start",
    "Start a job on a frozen project. Field positions are SI metres and outputs use native SI units. Jobs never automatically change the live model.",
    {
        "kind": {
            "type": "string",
            "enum": ["field", "measurement", "drc", "optimizer", "optimizer_tuning"],
        },
        "settings": {"type": "object"},
    },
    ["kind", "settings"],
    project=True,
    writes=True,
)
define("jobs.list", "List calculation jobs for this project.", project=True)
define(
    "job.get",
    "Get job state, progress, source revision, and current staleness.",
    {"job_id": text()},
    ["job_id"],
    project=True,
)
define(
    "job.cancel",
    "Request cooperative cancellation; uninterruptible solver calls finish before cancellation takes effect.",
    {"job_id": text()},
    ["job_id"],
    project=True,
    writes=True,
)
define(
    "job.result",
    "Get a completed result summary; full=true includes numerical arrays. Export results that exceed the response limit.",
    {"job_id": text(), "full": {"type": "boolean"}},
    ["job_id"],
    project=True,
)
define(
    "job.export",
    "Write a completed numerical result with source provenance to a JSON file.",
    {"job_id": text(), "path": text(), "overwrite": {"type": "boolean"}},
    ["job_id", "path"],
    project=True,
    writes=True,
)
define(
    "job.apply",
    "Apply an optimizer candidate or save a measurement snapshot. Requires the live project to still match the job source revision.",
    {"job_id": text(), "candidate_id": text(), "name": text()},
    ["job_id"],
    project=True,
    writes=True,
)
define(
    "job.forget",
    "Release a terminal job and its retained numerical data.",
    {"job_id": text()},
    ["job_id"],
    project=True,
    writes=True,
)
