"""Portable saved exposure definitions for Field Workbench scenes.

An exposure is a source/load schedule. Timed steps contain one or more drive
groups; every drive group connects one saved waveform source to one or more
physical coil loads using an explicit series or parallel topology. Multiple
groups may share one source in a step; those groups are parallel routes at the
same source terminals. Electrical source characteristics live with the waveform
and coil R/L lives with the model.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


EXPOSURE_FORMAT_VERSION = 6
EXPOSURE_FILE_SCHEMA = "fieldworkbench.exposure"
EXPOSURE_FILE_VERSION = 7
EXPOSURE_FILE_SUFFIX = ".fwbexposure.json"


class ExposureError(ValueError):
    """Raised when a saved exposure definition is invalid."""


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ExposureError(f"{label} must be numeric.") from error
    if not math.isfinite(number):
        raise ExposureError(f"{label} must be finite.")
    if minimum is not None and number < minimum:
        raise ExposureError(f"{label} must be at least {minimum:g}.")
    return number


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _reject_unknown_fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ExposureError(
            f"Unsupported {label} field(s): " + ", ".join(unknown) + "."
        )


def default_exposure_protocol() -> dict[str, Any]:
    """Return the canonical exposure protocol used by all new projects."""

    return {
        "duration_s": 1.0,
        "sources": [],
        "coil_labels": {},
        "sequence": {
            "loop": True,
            "steps": [],
        },
        "display": {
            "time_unit": "ms",
            "current_unit": "mA",
            "voltage_unit": "V",
        },
    }


def normalize_exposure_protocol(value: Any) -> dict[str, Any]:
    """Validate the canonical source/load exposure protocol.

    Continuous sources run on the exposure clock; once-per-step sources fire one
    waveform cycle from their configured phase whenever a routing step using them
    begins, then command zero for the remainder of that step. Pre-source-catalogue
    drafts are upgraded by sharing one continuous source per waveform.
    """

    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ExposureError("Exposure protocol must be a JSON object.")
    _reject_unknown_fields(value, {"duration_s", "sources", "coil_labels", "sequence", "display"}, "exposure protocol")

    protocol = default_exposure_protocol()
    raw_labels = value.get("coil_labels", {})
    if not isinstance(raw_labels, dict):
        raise ExposureError("Exposure coil labels must be an object.")
    protocol["coil_labels"] = {
        str(key): str(label)
        for key, label in raw_labels.items()
        if str(key).strip() and str(label).strip()
    }

    raw_sequence = value.get("sequence", {})
    if raw_sequence is None:
        raw_sequence = {}
    if not isinstance(raw_sequence, dict):
        raise ExposureError("Exposure sequence settings must be an object.")
    _reject_unknown_fields(raw_sequence, {"loop", "steps"}, "exposure sequence")
    raw_steps = raw_sequence.get("steps", [])
    if raw_steps is None:
        raw_steps = []
    if not isinstance(raw_steps, list):
        raise ExposureError("Exposure sequence steps must be an array.")

    # Older drafts without a source catalogue are upgraded once. The routing
    # structure keeps its sequence/drive_groups keys, but now references sources.
    legacy = "sources" not in value
    raw_sources = value.get("sources", [])
    if legacy:
        raw_sources = [
            {"id": waveform_id, "name": waveform_id, "waveform_id": waveform_id, "playback_mode": "continuous"}
            for waveform_id in dict.fromkeys(
                str(group.get("waveform_id", "") or "")
                for step in raw_steps if isinstance(step, dict)
                for group in step.get("drive_groups", []) if isinstance(group, dict)
            ) if waveform_id
        ]
    if not isinstance(raw_sources, list):
        raise ExposureError("Exposure sources must be an array.")
    sources = []
    source_ids = set()
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise ExposureError("Exposure source must be an object.")
        _reject_unknown_fields(raw_source, {"id", "name", "waveform_id", "playback_mode", "phase_offset_s"}, "source")
        source_id = str(raw_source.get("id", "")).strip()
        waveform_id = str(raw_source.get("waveform_id", "")).strip()
        if not source_id or source_id in source_ids:
            raise ExposureError("Exposure source IDs must be nonempty and unique.")
        if not waveform_id:
            raise ExposureError("Exposure source requires a waveform.")
        mode = str(raw_source.get("playback_mode", "continuous") or "continuous").strip().lower()
        # ``repeat`` was the short-lived source-centric spelling used before the
        # editor exposed source playback mode. Treat it as the same continuous
        # clock rather than making existing in-progress projects unusable.
        if mode == "repeat":
            mode = "continuous"
        if mode not in {"continuous", "once_per_step"}:
            raise ExposureError("Exposure source playback mode must be continuous or once_per_step.")
        source_ids.add(source_id)
        sources.append({"id": source_id, "name": str(raw_source.get("name", source_id)).strip() or source_id,
                        "waveform_id": waveform_id, "playback_mode": mode,
                        "phase_offset_s": _finite(raw_source.get("phase_offset_s", 0.0), "Source phase offset")})
    protocol["sources"] = sources

    sequence_steps: list[dict[str, Any]] = []
    seen_step_ids: set[str] = set()
    for step_index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ExposureError(f"Exposure step {step_index + 1} must be an object.")
        _reject_unknown_fields(
            raw_step, {"id", "duration_s", "drive_groups"}, f"exposure step {step_index + 1}"
        )
        step_id = str(raw_step.get("id", "") or "").strip() or f"step_{step_index + 1}"
        if step_id in seen_step_ids:
            raise ExposureError(f"Exposure contains duplicate step id {step_id!r}.")
        seen_step_ids.add(step_id)

        raw_groups = raw_step.get("drive_groups", [])
        if raw_groups is None:
            raw_groups = []
        if not isinstance(raw_groups, list):
            raise ExposureError(f"Exposure step {step_index + 1} drive groups must be an array.")

        groups: list[dict[str, Any]] = []
        seen_group_ids: set[str] = set()
        driven_coils: set[str] = set()
        for group_index, raw_group in enumerate(raw_groups):
            if not isinstance(raw_group, dict):
                raise ExposureError(
                    f"Exposure step {step_index + 1} drive group {group_index + 1} must be an object."
                )
            _reject_unknown_fields(
                raw_group,
                {
                    "id",
                    "waveform_id" if legacy else "source_id",
                    "coil_ids",
                    "connection",
                    "load_overrides",
                },
                f"exposure step {step_index + 1} drive group {group_index + 1}",
            )
            group_id = str(raw_group.get("id", "") or "").strip() or f"drive_{step_index + 1}_{group_index + 1}"
            if group_id in seen_group_ids:
                raise ExposureError(
                    f"Exposure step {step_index + 1} contains duplicate drive-group id {group_id!r}."
                )
            seen_group_ids.add(group_id)

            source_id = str(raw_group.get("waveform_id" if legacy else "source_id", "") or "").strip()
            coil_ids = _string_list(raw_group.get("coil_ids", []))
            if source_id not in source_ids:
                raise ExposureError(
                    f"Exposure step {step_index + 1} drive group {group_index + 1} requires a waveform source."
                )
            if not coil_ids:
                raise ExposureError(
                    f"Exposure step {step_index + 1} drive group {group_index + 1} requires at least one coil."
                )
            connection = str(raw_group.get("connection", "series") or "series").strip().lower()
            if connection not in {"series", "parallel"}:
                raise ExposureError(
                    f"Exposure step {step_index + 1} drive group {group_index + 1} connection must be series or parallel."
                )
            raw_overrides = raw_group.get("load_overrides", {})
            if raw_overrides is None:
                raw_overrides = {}
            if not isinstance(raw_overrides, dict):
                raise ExposureError(
                    f"Exposure step {step_index + 1} drive group {group_index + 1} load overrides must be an object."
                )
            load_overrides: dict[str, dict[str, float]] = {}
            for coil_id, raw_override in raw_overrides.items():
                coil_key = str(coil_id).strip()
                if coil_key not in coil_ids:
                    raise ExposureError(
                        f"Exposure step {step_index + 1} drive group {group_index + 1} has a load override "
                        f"for coil {coil_key!r}, which is not selected by that drive group."
                    )
                if not isinstance(raw_override, dict):
                    raise ExposureError(
                        f"Exposure load override for coil {coil_key!r} must be an object."
                    )
                _reject_unknown_fields(
                    raw_override,
                    {"resistance_ohm", "inductance_h"},
                    f"load override for coil {coil_key!r}",
                )
                normalized_override: dict[str, float] = {}
                if "resistance_ohm" in raw_override:
                    normalized_override["resistance_ohm"] = _finite(
                        raw_override["resistance_ohm"],
                        f"{coil_key} drive-group resistance override",
                        minimum=0.0,
                    )
                if "inductance_h" in raw_override:
                    inductance = _finite(
                        raw_override["inductance_h"],
                        f"{coil_key} drive-group inductance override",
                        minimum=0.0,
                    )
                    if inductance <= 0.0:
                        raise ExposureError(
                            f"{coil_key} drive-group inductance override must be greater than zero."
                        )
                    normalized_override["inductance_h"] = inductance
                if normalized_override:
                    load_overrides[coil_key] = normalized_override

            overlap = [coil_id for coil_id in coil_ids if coil_id in driven_coils]
            if overlap:
                raise ExposureError(
                    f"Exposure step {step_index + 1} assigns coil {overlap[0]!r} to more than one drive group. "
                    "A coil can be connected to only one source at a time."
                )
            driven_coils.update(coil_ids)
            groups.append(
                {
                    "id": group_id,
                    "source_id": source_id,
                    "coil_ids": coil_ids,
                    "connection": connection,
                    **({"load_overrides": load_overrides} if load_overrides else {}),
                }
            )

        sequence_steps.append(
            {
                "id": step_id,
                "duration_s": _finite(
                    raw_step.get("duration_s", 1.0),
                    f"Exposure step {step_index + 1} duration",
                    minimum=1.0e-12,
                ),
                "drive_groups": groups,
            }
        )

    protocol["sequence"] = {
        "loop": bool(raw_sequence.get("loop", True)),
        "steps": sequence_steps,
    }

    protocol["duration_s"] = _finite(
        value.get("duration_s", sum(step["duration_s"] for step in sequence_steps) or 1.0),
        "Exposure duration", minimum=1e-12,
    )

    raw_display = value.get("display", {})
    if not isinstance(raw_display, dict):
        raise ExposureError("Exposure display settings must be an object.")
    _reject_unknown_fields(
        raw_display, {"time_unit", "current_unit", "voltage_unit"}, "exposure display"
    )
    time_unit = str(raw_display.get("time_unit", "ms"))
    current_unit = str(raw_display.get("current_unit", "mA"))
    voltage_unit = str(raw_display.get("voltage_unit", "V"))
    protocol["display"] = {
        "time_unit": time_unit if time_unit in {"ms", "s"} else "ms",
        "current_unit": current_unit if current_unit in {"mA", "A"} else "mA",
        "voltage_unit": voltage_unit if voltage_unit in {"V", "mV"} else "V",
    }
    return protocol


def normalize_exposure(value: Any, *, require_id: bool = True) -> dict[str, Any]:
    """Validate one saved exposure and return a JSON-safe normalized copy."""

    if not isinstance(value, dict):
        raise ExposureError("Exposure must be a JSON object.")
    _reject_unknown_fields(
        value,
        {"id", "name", "description", "format_version", "created_at", "updated_at", "protocol", "imported_from"},
        "exposure",
    )
    exposure_id = str(value.get("id", "")).strip()
    if require_id and not exposure_id:
        raise ExposureError("Exposure requires a stable id.")
    name = str(value.get("name", "")).strip()
    if not name:
        raise ExposureError("Exposure name cannot be blank.")
    try:
        version = int(value.get("format_version", EXPOSURE_FORMAT_VERSION))
    except (TypeError, ValueError) as error:
        raise ExposureError("Exposure format version must be an integer.") from error
    if version not in {3, 4, 5, EXPOSURE_FORMAT_VERSION}:
        raise ExposureError(
            f"Unsupported exposure format version {version}; expected {EXPOSURE_FORMAT_VERSION}."
        )
    created = str(value.get("created_at", "")).strip() or utc_now_text()
    updated = str(value.get("updated_at", "")).strip() or created
    result = {
        "id": exposure_id,
        "name": name,
        "description": str(value.get("description", "") or "").strip(),
        "format_version": EXPOSURE_FORMAT_VERSION,
        "created_at": created,
        "updated_at": updated,
        "protocol": normalize_exposure_protocol(value.get("protocol", {})),
    }
    origin = value.get("imported_from")
    if origin is not None:
        if not isinstance(origin, dict):
            raise ExposureError("Exposure import provenance must be an object.")
        _reject_unknown_fields(
            origin, {"file_name", "exposure_id", "imported_at"}, "exposure import provenance"
        )
        result["imported_from"] = {
            "file_name": str(origin.get("file_name", "") or ""),
            "exposure_id": str(origin.get("exposure_id", "") or ""),
            "imported_at": str(origin.get("imported_at", "") or ""),
        }
    return result


def new_exposure(name: str = "New exposure", *, exposure_id: str | None = None) -> dict[str, Any]:
    now = utc_now_text()
    return normalize_exposure(
        {
            "id": str(exposure_id or f"exposure_{uuid4().hex}"),
            "name": str(name).strip() or "New exposure",
            "description": "",
            "format_version": EXPOSURE_FORMAT_VERSION,
            "created_at": now,
            "updated_at": now,
            "protocol": default_exposure_protocol(),
        }
    )


def exposure_waveform_ids(protocol: dict[str, Any]) -> list[str]:
    """Return unique waveform ids referenced by an exposure."""

    state = normalize_exposure_protocol(protocol)
    ids = [
        source["waveform_id"] for source in state["sources"]
    ]
    return list(dict.fromkeys(value for value in ids if value))


def exposure_coil_ids(protocol: dict[str, Any]) -> list[str]:
    """Return unique coil ids referenced anywhere in an exposure."""

    state = normalize_exposure_protocol(protocol)
    ids = [
        str(coil_id)
        for step in state["sequence"]["steps"]
        for group in step["drive_groups"]
        for coil_id in group["coil_ids"]
        if str(coil_id)
    ]
    return list(dict.fromkeys(ids))


def exposure_drive_groups(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten drive groups in sequence order."""

    state = normalize_exposure_protocol(protocol)
    return [
        copy.deepcopy(group)
        for step in state["sequence"]["steps"]
        for group in step["drive_groups"]
    ]


def exposure_summary(exposure: dict[str, Any]) -> dict[str, Any]:
    item = normalize_exposure(exposure)
    sequence = item["protocol"]["sequence"]
    steps = sequence["steps"]
    groups = [group for step in steps for group in step["drive_groups"]]
    source_ids = exposure_waveform_ids(item["protocol"])
    coils = list(
        dict.fromkeys(
            str(coil_id)
            for group in groups
            for coil_id in group["coil_ids"]
            if str(coil_id)
        )
    )
    source_label = (
        source_ids[0]
        if len(source_ids) == 1
        else f"{len(source_ids)} sources"
        if source_ids
        else "—"
    )
    return {
        "id": item["id"],
        "name": item["name"],
        "description": item["description"],
        "waveform": source_label,
        "waveform_resource_id": source_ids[0] if len(source_ids) == 1 else "",
        "waveform_resource_ids": source_ids,
        "drive": f"{len(groups)} drive group" + ("" if len(groups) == 1 else "s"),
        "drive_group_count": len(groups),
        "source_count": len(item["protocol"]["sources"]),
        "step_count": len(steps),
        "duration_s": item["protocol"]["duration_s"],
        "routing_duration_s": sum(float(step["duration_s"]) for step in steps),
        "coil_count": len(coils),
        "sequenced": len(steps) > 1 or len(groups) > 1,
        "sequence_loop": bool(sequence.get("loop", False)),
        "updated_at": item["updated_at"],
        "imported_from": copy.deepcopy(item.get("imported_from")),
    }


def exposure_file_payload(
    exposure: dict[str, Any],
    *,
    waveform_resources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a portable exposure payload with all referenced waveform resources."""

    item = normalize_exposure(exposure)
    payload = {
        "schema": EXPOSURE_FILE_SCHEMA,
        "version": EXPOSURE_FILE_VERSION,
        "exposure": copy.deepcopy(item),
    }
    referenced_ids = exposure_waveform_ids(item["protocol"])
    if waveform_resources is None:
        if referenced_ids:
            raise ExposureError(
                "Portable exposure export requires every referenced waveform resource to be bundled."
            )
        return payload
    if not isinstance(waveform_resources, list) or not all(
        isinstance(value, dict) for value in waveform_resources
    ):
        raise ExposureError("Bundled waveform resources must be an array of objects.")
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    bundled_ids: set[str] = set()
    for resource in copy.deepcopy(waveform_resources):
        key = str(resource.get("id", "") or "").strip()
        token = key or json.dumps(resource, sort_keys=True, ensure_ascii=False)
        if token in seen:
            continue
        seen.add(token)
        if key:
            bundled_ids.add(key)
        unique.append(resource)
    missing = [resource_id for resource_id in referenced_ids if resource_id not in bundled_ids]
    if missing:
        raise ExposureError(
            "Portable exposure is missing referenced waveform resource(s): " + ", ".join(missing) + "."
        )
    if unique:
        payload["waveform_resources"] = unique
    return payload


def write_exposure_file(
    path: str | Path,
    exposure: dict[str, Any],
    *,
    waveform_resources: list[dict[str, Any]] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            exposure_file_payload(exposure, waveform_resources=waveform_resources),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def read_exposure_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExposureError(f"Unable to read exposure file {source.name!r}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != EXPOSURE_FILE_SCHEMA:
        raise ExposureError("Not a Field Workbench exposure file.")
    _reject_unknown_fields(
        payload, {"schema", "version", "exposure", "waveform_resources"}, "exposure file"
    )
    try:
        version = int(payload.get("version", 0))
    except (TypeError, ValueError) as error:
        raise ExposureError("Exposure file version must be an integer.") from error
    if version not in {4, 5, 6, EXPOSURE_FILE_VERSION}:
        raise ExposureError(
            f"Unsupported exposure file version {version}; expected {EXPOSURE_FILE_VERSION}."
        )
    raw = payload.get("exposure")
    if not isinstance(raw, dict):
        raise ExposureError("Exposure file payload is missing.")
    result = normalize_exposure(raw)
    bundled = payload.get("waveform_resources")
    if bundled is not None:
        if not isinstance(bundled, list) or not all(isinstance(value, dict) for value in bundled):
            raise ExposureError("Bundled waveform resources must be an array of objects.")
        result["_bundled_waveform_resources"] = copy.deepcopy(bundled)
    referenced_ids = exposure_waveform_ids(result["protocol"])
    bundled_ids = {
        str(item.get("id", "") or "").strip()
        for item in (bundled or [])
        if isinstance(item, dict) and str(item.get("id", "") or "").strip()
    }
    missing = [resource_id for resource_id in referenced_ids if resource_id not in bundled_ids]
    if missing:
        raise ExposureError(
            "Exposure file is missing bundled waveform resource(s): " + ", ".join(missing) + "."
        )
    return result
