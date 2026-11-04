"""Optimizer freedom/fidelity capability metadata.

Optimizer 2.2 intentionally uses Centreline physics for cheap structural
reconnaissance, but not every user-facing freedom is fully observable in that
representation.  This module keeps that knowledge explicit and JSON-friendly so
search policy, reporting, and future convergence checks can reason about the
same capability contract instead of accumulating one-off special cases.

The metadata here describes *representation*, not numerical accuracy.  A model
may represent an effect while still being an approximation of the physical
winding.  Conversely, ``auto`` is deliberately marked dynamic because the
selected representation depends on the coil/observer workload at evaluation
runtime.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable


# Magnetic effects that optimizer freedoms can require from a field model.
MAGNETIC_EFFECT_SOURCE_AMPLITUDE = "source_amplitude"
MAGNETIC_EFFECT_CENTRELINE_PATH = "centreline_path_geometry"
MAGNETIC_EFFECT_POSE = "source_pose"
MAGNETIC_EFFECT_WINDING_DISTRIBUTION = "winding_cross_section_distribution"


# Capability is intentionally about whether an effect exists in the model, not
# whether the model is the most accurate representation available.
_MODEL_MAGNETIC_EFFECTS: dict[str, frozenset[str]] = {
    "centreline": frozenset({
        MAGNETIC_EFFECT_SOURCE_AMPLITUDE,
        MAGNETIC_EFFECT_CENTRELINE_PATH,
        MAGNETIC_EFFECT_POSE,
    }),
    "current_sheet": frozenset({
        MAGNETIC_EFFECT_SOURCE_AMPLITUDE,
        MAGNETIC_EFFECT_CENTRELINE_PATH,
        MAGNETIC_EFFECT_POSE,
        MAGNETIC_EFFECT_WINDING_DISTRIBUTION,
    }),
    "bundled": frozenset({
        MAGNETIC_EFFECT_SOURCE_AMPLITUDE,
        MAGNETIC_EFFECT_CENTRELINE_PATH,
        MAGNETIC_EFFECT_POSE,
        MAGNETIC_EFFECT_WINDING_DISTRIBUTION,
    }),
    "exact": frozenset({
        MAGNETIC_EFFECT_SOURCE_AMPLITUDE,
        MAGNETIC_EFFECT_CENTRELINE_PATH,
        MAGNETIC_EFFECT_POSE,
        MAGNETIC_EFFECT_WINDING_DISTRIBUTION,
    }),
}


_PATH_DIMENSION_KINDS = frozenset({
    "circular_diameter",
    "racetrack_end_diameter",
    "racetrack_straight_length",
    "square_length",
    "square_width",
})
_POSITION_KINDS = frozenset({
    "pair_spacing",
    "position_x",
    "position_y",
    "position_z",
})
_ORIENTATION_KINDS = frozenset({
    # Legacy orientation names remain supported for saved/programmatic jobs.
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "assembly_orientation_x",
    "assembly_orientation_y",
    "assembly_orientation_z",
    "coil_orientation_x",
    "coil_orientation_y",
    "coil_orientation_z",
})


def _base_descriptor(variable: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(variable.get("id", "")),
        "kind": str(variable.get("kind", "")),
        "label": str(variable.get("label", variable.get("id", "freedom"))),
        "group_kind": str(variable.get("group_kind", "")),
        "group": str(variable.get("group", "")),
        "coil_ids": [str(value) for value in list(variable.get("coil_ids", []) or [])],
        "magnetic_effects": [],
        "affects_magnetic_field": True,
        "affects_drc_geometry": False,
        "affects_electrical_limits": False,
        "coupling_families": [],
        "notes": [],
        "known_kind": True,
    }


def optimizer_freedom_descriptor(variable: dict[str, Any]) -> dict[str, Any]:
    """Describe which physical effects one optimizer freedom can change.

    The result is stable, compact metadata suitable for embedding in an optimizer
    result/report.  Dynamic relationships (for example turns becoming an amplitude
    duplicate of a freely solved current group) are intentionally added by the
    caller because they depend on the complete search configuration.
    """
    result = _base_descriptor(variable)
    kind = str(variable.get("kind", ""))

    if kind == "current":
        result.update({
            "magnetic_effects": [MAGNETIC_EFFECT_SOURCE_AMPLITUDE],
            "affects_drc_geometry": False,
            "affects_electrical_limits": True,
            "coupling_families": ["turns"],
        })
    elif kind == "turns":
        # Centreline captures the ampere-turn change, but a physical winding model
        # also sees the changed turn distribution/radial build.  This is therefore
        # deliberately a *partial* Centreline freedom unless a run-specific
        # redundancy proof says the amplitude part is absorbed by solved current.
        result.update({
            "magnetic_effects": [
                MAGNETIC_EFFECT_SOURCE_AMPLITUDE,
                MAGNETIC_EFFECT_WINDING_DISTRIBUTION,
            ],
            "affects_drc_geometry": True,
            "affects_electrical_limits": True,
            "coupling_families": ["current", "winding_distribution"],
            "notes": [
                "Centreline represents ampere-turn scaling but not turn-dependent winding-pack redistribution."
            ],
        })
    elif kind == "axial_winding_length":
        result.update({
            "magnetic_effects": [MAGNETIC_EFFECT_WINDING_DISTRIBUTION],
            "affects_drc_geometry": True,
            # Current optimizer hard limits are resistive/current-density based;
            # axial pack length can alter estimated inductance, but inductance is not
            # presently one of the enforced optimizer electrical constraints.
            "affects_electrical_limits": False,
            "coupling_families": ["pair_spacing", "path_geometry", "turns"],
            "notes": [
                "Axial winding length is magnetically invisible to a zero-thickness Centreline source."
            ],
        })
    elif kind in _PATH_DIMENSION_KINDS:
        result.update({
            "magnetic_effects": [MAGNETIC_EFFECT_CENTRELINE_PATH],
            "affects_drc_geometry": True,
            "affects_electrical_limits": True,
            "coupling_families": ["pair_spacing", "winding_distribution"],
        })
    elif kind in _POSITION_KINDS:
        result.update({
            "magnetic_effects": [MAGNETIC_EFFECT_POSE],
            "affects_drc_geometry": True,
            "affects_electrical_limits": False,
            "coupling_families": ["path_geometry", "winding_distribution", "orientation"],
        })
    elif kind in _ORIENTATION_KINDS:
        result.update({
            "magnetic_effects": [MAGNETIC_EFFECT_POSE],
            "affects_drc_geometry": True,
            "affects_electrical_limits": False,
            "coupling_families": ["position", "pair_spacing", "path_geometry"],
        })
    else:
        # Unknown future freedoms must not become invisible merely because this
        # table predates them.  Mark them conservative/unknown so policy can force
        # a capable physical search rather than silently dropping the coordinate.
        result.update({
            "known_kind": False,
            "magnetic_effects": [
                MAGNETIC_EFFECT_SOURCE_AMPLITUDE,
                MAGNETIC_EFFECT_CENTRELINE_PATH,
                MAGNETIC_EFFECT_POSE,
                MAGNETIC_EFFECT_WINDING_DISTRIBUTION,
            ],
            "affects_drc_geometry": True,
            "affects_electrical_limits": True,
            "coupling_families": ["unknown"],
            "notes": [
                "Unknown optimizer freedom; capability is intentionally conservative."
            ],
        })

    return result


def optimizer_model_visibility(model: str, descriptor: dict[str, Any]) -> str:
    """Return ``full``, ``partial``, ``none``, or ``dynamic`` visibility.

    ``auto`` cannot make a static promise because its runtime selector may choose
    Centreline or a distributed winding representation for each workload.
    """
    requested = str(model or "").strip().lower()
    if requested == "auto":
        return "dynamic"
    available = _MODEL_MAGNETIC_EFFECTS.get(requested)
    if available is None:
        return "unknown"
    required = set(map(str, descriptor.get("magnetic_effects", []) or []))
    if not required:
        return "full"
    represented = required & set(available)
    if represented == required:
        return "full"
    if represented:
        return "partial"
    return "none"


def optimizer_freedom_coverage_entry(
    variable: dict[str, Any],
    *,
    requested_model: str,
    scout_model: str = "centreline",
) -> dict[str, Any]:
    """Return initial per-freedom search/fidelity bookkeeping."""
    descriptor = optimizer_freedom_descriptor(variable)
    model_visibility = {
        model: optimizer_model_visibility(model, descriptor)
        for model in ("centreline", "current_sheet", "bundled", "exact", "auto")
    }
    scout_visibility = optimizer_model_visibility(scout_model, descriptor)
    physical_required = scout_visibility != "full"
    return {
        **copy.deepcopy(descriptor),
        "model_visibility": model_visibility,
        "scout": {
            "model": str(scout_model),
            "visibility": scout_visibility,
            "status": "pending",
            "omission_reason": None,
        },
        "physical_search": {
            "required_for_complete_coverage": bool(physical_required),
            "minimum_capability": (
                "distributed_winding"
                if MAGNETIC_EFFECT_WINDING_DISTRIBUTION
                in set(descriptor.get("magnetic_effects", []) or [])
                else "centreline_geometry"
            ),
            "status": "required" if physical_required else "not_required_by_observability",
            "model": None,
        },
        "authoritative": {
            "requested_model": str(requested_model),
            "visibility": optimizer_model_visibility(requested_model, descriptor),
            "status": "planned",
        },
        "completion_status": "pending",
        "dynamic": {},
    }


def optimizer_freedom_coverage(
    variables: Iterable[dict[str, Any]],
    *,
    requested_model: str,
    scout_model: str = "centreline",
) -> list[dict[str, Any]]:
    """Return initial capability bookkeeping for an ordered variable list."""
    return [
        optimizer_freedom_coverage_entry(
            variable,
            requested_model=requested_model,
            scout_model=scout_model,
        )
        for variable in variables
    ]




def optimizer_apply_source_capability_context(
    variables: Iterable[dict[str, Any]],
    freedom_coverage: Iterable[dict[str, Any]],
    source_support: Iterable[dict[str, Any]],
    *,
    scout_model: str = "centreline",
    requested_model: str = "auto",
) -> list[dict[str, Any]]:
    """Apply source-specific capability facts to the generic freedom table.

    The generic ``turns`` descriptor includes winding-cross-section redistribution
    because changing turns *can* alter a physically specified winding pack.  When
    every coil controlled by a turns freedom has physical winding geometry disabled,
    that distributed magnetic effect does not exist in the active source model.  In
    that case Centreline fully represents the magnetic turns effect (ampere-turn
    scaling), while DRC/electrical consequences remain handled by their own gates.

    Keep the static descriptor conservative; this function annotates a run-specific
    coverage copy so explicit Bundled/Exact/Current-Sheet authority remains strict.
    Missing support records never relax capability requirements.
    """
    variables_by_id = {
        str(item.get("id", "")): item
        for item in variables
        if str(item.get("id", ""))
    }
    support_by_id = {
        str(item.get("id", "")): item
        for item in source_support
        if str(item.get("id", ""))
    }
    result = copy.deepcopy(list(freedom_coverage))
    for entry in result:
        variable_id = str(entry.get("id", ""))
        variable = variables_by_id.get(variable_id)
        if variable is None or str(variable.get("kind", "")) != "turns":
            continue
        coil_ids = [str(value) for value in list(variable.get("coil_ids", []) or []) if str(value)]
        if not coil_ids or any(object_id not in support_by_id for object_id in coil_ids):
            continue
        winding_enabled = [
            bool(support_by_id[object_id].get("physical_winding_available", False))
            for object_id in coil_ids
        ]
        if any(winding_enabled):
            entry.setdefault("dynamic", {})["source_winding_distribution_active"] = True
            continue

        effective_descriptor = copy.deepcopy(entry)
        effective_descriptor["magnetic_effects"] = [MAGNETIC_EFFECT_SOURCE_AMPLITUDE]
        model_visibility = {
            model: optimizer_model_visibility(model, effective_descriptor)
            for model in ("centreline", "current_sheet", "bundled", "exact", "auto")
        }
        scout_visibility = optimizer_model_visibility(scout_model, effective_descriptor)
        physical_required = scout_visibility != "full"
        entry["effective_magnetic_effects"] = [MAGNETIC_EFFECT_SOURCE_AMPLITUDE]
        entry["model_visibility"] = model_visibility
        entry.setdefault("scout", {})["visibility"] = scout_visibility
        physical = entry.setdefault("physical_search", {})
        physical["required_for_complete_coverage"] = bool(physical_required)
        physical["minimum_capability"] = "centreline_geometry"
        physical["status"] = (
            "required" if physical_required else "not_required_by_observability"
        )
        authoritative = entry.setdefault("authoritative", {})
        authoritative["visibility"] = optimizer_model_visibility(
            requested_model, effective_descriptor
        )
        dynamic = entry.setdefault("dynamic", {})
        dynamic["source_winding_distribution_active"] = False
        dynamic["source_capability_adjustment"] = (
            "turns_has_no_distributed_magnetic_effect_without_enabled_physical_winding"
        )
        notes = list(entry.get("notes", []) or [])
        note = (
            "No participating coil has enabled physical winding geometry; for this run, "
            "turn count changes magnetic amplitude but not winding-cross-section distribution."
        )
        if note not in notes:
            notes.append(note)
        entry["notes"] = notes
    return result

def optimizer_validate_requested_model_reports(
    requested_model: str,
    required_coil_ids: Iterable[str],
    reports: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate explicit optimizer coil-model evidence across real field solves.

    Optimizer current solving may evaluate a geometry through several temporary
    adapters (zero-current plus one basis solve per current group).  The fidelity
    guard must inspect the model reports produced by those *actual* field solves
    rather than assuming the parent candidate adapter performed the solve itself.

    Every reported occurrence of a required coil must use the explicit requested
    model.  At least one occurrence must be present for every required coil.
    ``auto`` remains intentionally dynamic and therefore always passes this strict
    explicit-model check.
    """
    requested = str(requested_model or "").strip().lower()
    required = [str(value) for value in required_coil_ids if str(value)]
    if requested == "auto":
        return {
            "ok": True,
            "requested_model": requested,
            "required_coil_ids": required,
            "observed_report_count": 0,
            "failures": [],
        }

    entries_by_id: dict[str, list[dict[str, Any]]] = {value: [] for value in required}
    observed_report_count = 0
    for raw_report in reports:
        if not isinstance(raw_report, dict):
            continue
        observed_report_count += 1
        for raw_item in list(raw_report.get("coils", []) or []):
            if not isinstance(raw_item, dict):
                continue
            item = copy.deepcopy(raw_item)
            item_ids = {str(item.get("id", ""))}
            source_id = str(item.get("source_id", ""))
            if source_id:
                item_ids.add(source_id)
            for object_id in item_ids:
                if object_id in entries_by_id:
                    entries_by_id[object_id].append(item)

    failures: list[dict[str, Any]] = []
    for object_id in required:
        entries = entries_by_id.get(object_id, [])
        if not entries:
            failures.append({
                "id": object_id,
                "code": "missing_model_report",
                "actual": None,
                "reason": "no coil-model result was reported by the field solves",
            })
            continue
        mismatches = [
            item
            for item in entries
            if str(item.get("actual", "centreline")).strip().lower() != requested
        ]
        if mismatches:
            item = mismatches[0]
            actual = str(item.get("actual", "centreline")).strip().lower()
            failures.append({
                "id": object_id,
                "code": "requested_model_fell_back",
                "actual": actual,
                "reason": str(
                    item.get("fallback_reason")
                    or "requested representation was not applied"
                ),
            })

    return {
        "ok": not bool(failures),
        "requested_model": requested,
        "required_coil_ids": required,
        "observed_report_count": int(observed_report_count),
        "failures": failures,
    }

def optimizer_capability_summary(
    coverage: Iterable[dict[str, Any]],
    *,
    requested_model: str,
    scout_model: str = "centreline",
) -> dict[str, Any]:
    """Summarize whether the cheap scout can represent all selected freedoms."""
    entries = list(coverage)
    invisible = [
        str(item.get("id", ""))
        for item in entries
        if str((item.get("scout") or {}).get("visibility", "")) == "none"
    ]
    partial = [
        str(item.get("id", ""))
        for item in entries
        if str((item.get("scout") or {}).get("visibility", "")) == "partial"
    ]
    dynamic_authority = [
        str(item.get("id", ""))
        for item in entries
        if str((item.get("authoritative") or {}).get("visibility", "")) == "dynamic"
    ]
    authority_not_full = [
        str(item.get("id", ""))
        for item in entries
        if str((item.get("authoritative") or {}).get("visibility", ""))
        in {"none", "partial", "unknown"}
    ]
    return {
        "schema_version": 1,
        "scout_model": str(scout_model),
        "requested_model": str(requested_model),
        "freedom_count": int(len(entries)),
        "scout_fully_observable": not bool(invisible or partial),
        "scout_invisible_variable_ids": invisible,
        "scout_partial_variable_ids": partial,
        "physical_search_required_variable_ids": [
            str(item.get("id", ""))
            for item in entries
            if bool((item.get("physical_search") or {}).get("required_for_complete_coverage", False))
        ],
        "authoritative_dynamic_variable_ids": dynamic_authority,
        "authoritative_not_fully_represented_variable_ids": authority_not_full,
    }
