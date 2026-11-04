"""One dispatcher for all transports. The host supplies live project lifecycle/UI hooks."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

from . import PROTOCOL_VERSION
from .commands import COMMANDS, OPERATION_SCHEMAS, get_command
from .jobs import JobManager, result_summary
from .protocol import CommandError, dumps, failure, json_value, revision, validate


def write_json(path, value, *, overwrite=False):
    """Atomic export; an existing destination requires an explicit overwrite flag."""
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise CommandError(
            "file_exists",
            "Destination exists; set overwrite=true to replace it.",
            path=str(target),
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(dumps(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # Link-then-unlink provides an atomic no-clobber commit (POSIX/NTFS).
            os.link(temporary, target)
            os.unlink(temporary)
    except FileExistsError:
        raise CommandError(
            "file_exists", "Destination appeared during export.", path=str(target)
        ) from None
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)


def object_transform(adapter, object_id):
    transform = adapter.get_transform(object_id)
    if "orientation" not in transform:
        from scipy.spatial.transform import Rotation

        transform["orientation"] = (
            Rotation.from_euler("xyz", transform["euler"], degrees=True)
            .as_rotvec(degrees=True)
            .tolist()
        )
    return transform


class CommandService:
    def __init__(self, host, *, jobs=None):
        self.host = host
        self.jobs = jobs if jobs is not None else JobManager()

    def execute(self, command: str, params: dict | None = None, *, request_id=None):
        """Called only on the owning GUI thread. Errors never open GUI dialogs."""
        try:
            spec = get_command(command)
            params = {} if params is None else params
            validate(params, spec.input_schema)
            context = self.host.context(params["project_id"]) if spec.project else None
            before = revision(context.adapter) if context else None
            if spec.revision_required and params["expected_revision"] != before:
                raise CommandError(
                    "revision_conflict",
                    "The project changed; inspect it again before editing.",
                    expected=params["expected_revision"],
                    actual=before,
                )
            if spec.writes:
                self.host.check_write(context)
            result = self._dispatch(command, params, context)
            warnings = []
            if (
                context is not None
                and spec.writes
                and revision(context.adapter) != before
            ):
                try:
                    self.host.changed(context)
                except Exception as error:  # noqa: BLE001 - command/presentation boundary
                    # A committed edit must never be reported as a failed edit
                    # just because a presentation refresh failed.
                    warnings.append({"code": "refresh_failed", "message": str(error)})
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": True,
                "result": json_value(result),
            }
            if warnings:
                response["warnings"] = warnings
            if context is not None:
                response.update(
                    project_id=context.project_id, revision=revision(context.adapter)
                )
            return response
        except CommandError as error:
            return failure(error, request_id)
        except (KeyError, ValueError, TypeError) as error:
            return failure(CommandError("invalid_operation", str(error)), request_id)
        except OSError as error:
            return failure(CommandError("io_error", str(error)), request_id)
        except Exception as error:  # noqa: BLE001 - command/presentation boundary
            # The numerical adapter defines its own exception type; keep clients
            # cheap to import and avoid GUI/error-message side effects here.
            return failure(
                CommandError("operation_failed", str(error), type=type(error).__name__),
                request_id,
            )

    def _project_info(self, context):
        return {
            "id": context.project_id,
            "name": context.display_name(),
            "path": str(context.path) if context.path else None,
            "dirty": context.is_dirty(),
            "revision": revision(context.adapter),
            "history": context.adapter.history(),
        }

    def _dispatch(self, name, p, context):
        if name == "system.info":
            from .. import __version__

            return {
                "instance_id": self.host.instance_id,
                "version": __version__,
                "protocol_version": PROTOCOL_VERSION,
            }
        if name == "commands.list":
            return {
                "commands": [c.catalog() for c in COMMANDS.values()],
                "operation_schemas": copy.deepcopy(OPERATION_SCHEMAS),
            }
        if name == "projects.list":
            return [self._project_info(c) for c in self.host.contexts()]
        if name == "project.new":
            return self._project_info(self.host.new_project())
        if name == "project.open":
            return self._project_info(self.host.open_project(p["path"]))
        if name == "templates.list":
            from ..studio_adapter import OBJECT_TEMPLATES, WELL_PLATE_TEMPLATES

            return sorted(
                set(OBJECT_TEMPLATES)
                | set(WELL_PLATE_TEMPLATES)
                | {
                    "generic_bust",
                    "sri24_brain",
                    "guide_box",
                    "guide_cylinder",
                    "guide_sphere",
                }
            )
        adapter = context.adapter
        if name == "project.get":
            return self._project_info(context)
        if name == "project.document":
            return adapter.document()
        if name == "project.save":
            return self.host.save_project(
                context, p.get("path"), p.get("overwrite", False)
            )
        if name == "project.export":
            return {
                "path": write_json(
                    p["path"], adapter.document(), overwrite=p.get("overwrite", False)
                )
            }
        if name in {"project.undo", "project.redo"}:
            getattr(adapter, name.split(".")[1])()
            return adapter.history()
        if name == "objects.list":
            return adapter.list_objects()
        if name == "object.get":
            oid = p["object_id"]
            result = {
                "object": adapter.get_object(oid),
                "parameters": adapter.get_params(oid),
                "transform": object_transform(adapter, oid),
            }
            if adapter.is_coil(oid):
                result["coil_physical"] = adapter.coil_physical_properties(oid)
            if adapter.is_geometry(oid):
                result["geometry"] = adapter.geometry_properties(oid)
            return result
        if name == "object.add":
            if p["template"] not in self._dispatch("templates.list", {}, None):
                raise CommandError("invalid_params", "Unknown object template.")
            if (
                p.get("parent_id")
                and adapter.get_object(p["parent_id"])["type"] != "Collection"
            ):
                raise CommandError("invalid_params", "Parent must be a Collection.")
            return {
                "object_id": adapter.add_template(
                    p["template"], parent=p.get("parent_id")
                )
            }
        if name == "object.duplicate":
            return {"object_id": adapter.duplicate(p["object_id"])}
        if name == "objects.remove":
            for oid in p["object_ids"]:
                adapter.get_object(oid)
            return {"removed": adapter.remove_many(p["object_ids"])}
        if name == "objects.edit":
            for index, operation in enumerate(p["operations"]):
                method, parameters = operation["method"], operation["params"]
                validate(
                    parameters,
                    OPERATION_SCHEMAS[method],
                    f"params.operations[{index}].params",
                )
                if method == "set_coil_physical":
                    from ..studio_adapter import default_coil_spec

                    unknown = set(parameters) - set(default_coil_spec()) - {"object_id"}
                    if unknown:
                        raise CommandError(
                            "invalid_params",
                            f"Unknown coil physical fields: {sorted(unknown)}",
                        )
                if method == "set_transform" and not (
                    {"position", "orientation"} & parameters.keys()
                ):
                    raise CommandError(
                        "invalid_params", "A transform needs position or orientation."
                    )
                adapter.get_object(parameters["object_id"])
                if method == "apply_edit" and parameters["path"] != "label":
                    raise CommandError(
                        "invalid_params",
                        "apply_edit currently exposes only the label path.",
                    )
            operations = copy.deepcopy(p["operations"])
            # Geometry's native set_transform resets omitted components; preserve
            # them explicitly so partial transforms have consistent semantics.
            transforms = {}
            for operation in operations:
                parameters = operation["params"]
                if operation["method"] == "set_transform":
                    oid = parameters["object_id"]
                    if oid not in transforms:
                        transforms[oid] = object_transform(adapter, oid)
                    current = transforms[oid]
                    parameters.setdefault("position", current["position"])
                    parameters.setdefault("orientation", current["orientation"])
                    transforms[oid] = parameters.copy()
            adapter.apply_operations(operations)
            return {"applied": len(p["operations"])}
        if name == "notes.get":
            return {"notes": adapter.get_scene_notes()}
        if name == "notes.set":
            adapter.set_scene_notes(p["notes"])
            return {"notes": adapter.get_scene_notes()}
        if name == "resources.list":
            resources = context.resources.refresh()
            return [
                {
                    "key": r.key,
                    "kind": r.kind,
                    "id": r.native_id,
                    "name": r.name,
                    "persistent": r.persistent,
                }
                for r in resources
            ]
        if name == "optimizer.inspect":
            return {
                "targets": adapter.optimizer_targets(),
                "coils": adapter.optimizer_existing_coils(),
                "defaults": adapter.optimizer_default_settings(p.get("target_id")),
            }
        if name == "calculation.start":
            return self.jobs.start(context, revision(adapter), p["kind"], p["settings"])
        if name == "jobs.list":
            return [
                dict(info, stale=info["source_revision"] != revision(adapter))
                for info in self.jobs.list(context.project_id)
            ]
        if name == "job.get":
            info = self.jobs.info(context.project_id, p["job_id"])
            return dict(info, stale=info["source_revision"] != revision(adapter))
        if name == "job.cancel":
            return self.jobs.cancel(context.project_id, p["job_id"])
        if name == "job.forget":
            return self.jobs.forget(context.project_id, p["job_id"])
        job = self.jobs.completed(context.project_id, p["job_id"])
        if name == "job.result":
            return {
                "job": job.info(),
                "stale": job.source_revision != revision(adapter),
                "data": job.result
                if p.get("full", False)
                else result_summary(job.result),
            }
        if name == "job.export":
            return {
                "path": write_json(
                    p["path"],
                    {
                        "job": job.info(),
                        "settings": job.settings,
                        "runtime": job.runtime,
                        "source_document": job.document,
                        "data": job.result,
                    },
                    overwrite=p.get("overwrite", False),
                )
            }
        if name == "job.apply":
            if revision(adapter) != job.source_revision:
                raise CommandError(
                    "stale_result",
                    "The project no longer matches the calculation source.",
                )
            if job.kind == "measurement":
                return {
                    "snapshot_id": adapter.create_snapshot_from_frozen_analysis(
                        job.result,
                        p.get("name", "Agent measurement"),
                        source_adapter=job.source_adapter,
                    )
                }
            if job.kind in {"optimizer", "optimizer_tuning"}:
                candidate_id = p.get("candidate_id")
                if not candidate_id:
                    raise CommandError(
                        "invalid_params",
                        "Choose an explicit candidate_id from the optimizer result.",
                    )
                candidates = job.result.get("finalists", [])
                candidate = next(
                    (c for c in candidates if str(c.get("id")) == candidate_id), None
                )
                if candidate is None:
                    raise CommandError(
                        "candidate_not_found", "No such candidate in this job."
                    )
                adapter.apply_optimizer_candidate(candidate)
                return {"candidate_id": candidate_id}
            raise CommandError(
                "unsupported_operation",
                "This calculation has no model changes to apply.",
            )
        raise CommandError("unknown_command", name)
