"""Optimizer 2.3 model-resolution policy.

This module turns the user's Preferences coil-model choice into an explicit
three-stage fidelity contract.  The cheap Centreline scout remains a permitted
reconnaissance accelerator, but capable-physics refinement and authoritative
validation may not silently fall back to a model that erases a selected freedom.

Model *capability* is source-aware here.  In particular, Field Workbench's
Current Sheet representation is presently available for closed Polyline coils,
not Circle sources; Bundled/Exact require a physical winding specification.
Auto is allowed to remain dynamic for final authority, while its optimizer
refinement stage receives a deterministic capability floor so candidate-to-
candidate Auto choices cannot make a free coordinate disappear mid-search.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

from .optimizer_effort_policy import optimizer_effort_physical_scope


_VALID_MODELS = {"auto", "exact", "bundled", "current_sheet", "centreline"}


def _model(value: str) -> str:
    model = str(value or "auto").strip().lower()
    if model not in _VALID_MODELS:
        raise ValueError(
            "coil modeling method must be Auto, Exact turns, Bundled, Current sheet, or Centreline"
        )
    return model


def _support_map(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id", "")): copy.deepcopy(item)
        for item in items
        if str(item.get("id", ""))
    }


def _variable_ids(variables: Iterable[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("id", ""))
        for item in variables
        if str(item.get("id", ""))
    ]


def _coil_ids_for_variables(
    variables: Iterable[dict[str, Any]], variable_ids: Iterable[str]
) -> list[str]:
    wanted = set(map(str, variable_ids))
    coil_ids: list[str] = []
    for variable in variables:
        variable_id = str(variable.get("id", ""))
        if variable_id not in wanted:
            continue
        for object_id in list(variable.get("coil_ids", []) or []):
            object_id = str(object_id)
            if object_id and object_id not in coil_ids:
                coil_ids.append(object_id)
    return coil_ids


def _unsupported_coils(
    coil_ids: Iterable[str],
    supports: dict[str, dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    key = {
        "centreline": "centreline_supported",
        "current_sheet": "current_sheet_supported",
        "bundled": "bundled_supported",
        "exact": "exact_supported",
    }.get(str(model))
    if key is None:
        return []
    rows: list[dict[str, Any]] = []
    for object_id in coil_ids:
        info = supports.get(str(object_id))
        if info is None:
            rows.append({
                "id": str(object_id),
                "label": str(object_id),
                "reason": "No optimizer model-capability record is available for this coil.",
            })
            continue
        if bool(info.get(key, False)):
            continue
        reasons = dict(info.get("model_reasons", {}) or {})
        rows.append({
            "id": str(object_id),
            "label": str(info.get("label") or object_id),
            "reason": str(
                reasons.get(str(model))
                or f"{str(model).replace('_', ' ').title()} is not available for this coil."
            ),
        })
    return rows


def optimizer_resolve_model_policy(
    *,
    requested_model: str,
    search_effort: str,
    variables: Iterable[dict[str, Any]],
    structural_variables: Iterable[dict[str, Any]],
    freedom_coverage: Iterable[dict[str, Any]],
    source_support: Iterable[dict[str, Any]],
    participant_coil_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Resolve scout/refinement/authority models for one optimizer run.

    The returned dictionary is JSON-friendly and intentionally contains a
    ``blocking_issues`` list instead of raising.  The Studio adapter decides
    whether to reject a run before expensive reconnaissance starts.
    """
    requested = _model(requested_model)
    all_variables = list(variables)
    structural = list(structural_variables)
    coverage = list(freedom_coverage)
    supports = _support_map(source_support)

    scope = optimizer_effort_physical_scope(
        search_effort=search_effort,
        structural_variables=structural,
        freedom_coverage=coverage,
    )
    active_ids = list(map(str, scope.get("active_variable_ids", []) or []))
    gap_ids = list(map(str, scope.get("capability_gap_variable_ids", []) or []))
    all_variable_ids = _variable_ids(all_variables)
    optimized_coil_ids = _coil_ids_for_variables(all_variables, all_variable_ids)
    participant_ids = list(dict.fromkeys(
        str(value) for value in (participant_coil_ids or optimized_coil_ids) if str(value)
    ))
    authoritative_coil_ids = list(participant_ids or optimized_coil_ids)
    active_variable_coil_ids = _coil_ids_for_variables(structural, active_ids)
    # Field Workbench applies one coil model globally during a solve.  Keep the
    # refinement landscape fidelity-consistent across all participating coils rather
    # than allowing fixed participants to fall back silently while active coils use
    # the declared capable model.
    physical_search_coil_ids = list(authoritative_coil_ids) if active_ids else []

    coverage_by_id = {
        str(item.get("id", "")): item
        for item in coverage
        if str(item.get("id", ""))
    }
    distributed_gap_ids = [
        variable_id
        for variable_id in gap_ids
        if str(
            ((coverage_by_id.get(variable_id, {}).get("physical_search") or {}).get(
                "minimum_capability", ""
            ))
        )
        == "distributed_winding"
    ]
    requires_distributed = bool(distributed_gap_ids)
    blocking: list[dict[str, Any]] = []

    # Explicit non-Auto authority is a promise.  Do not permit the generic field
    # solver's per-coil Centreline fallback to masquerade as optimizer authority.
    if requested in {"current_sheet", "bundled", "exact"}:
        unsupported_authority = _unsupported_coils(
            authoritative_coil_ids, supports, requested
        )
        if unsupported_authority:
            blocking.append({
                "code": "requested_authoritative_model_unavailable",
                "model": requested,
                "coil_ids": [str(item["id"]) for item in unsupported_authority],
                "details": unsupported_authority,
                "message": (
                    f"The requested {requested.replace('_', ' ')} coil model is not available "
                    "for every optimizer-participating coil, so authoritative validation would "
                    "silently fall back to Centreline for part of the participating system."
                ),
            })

    physical_model: str | None = None
    physical_reason = "effort_does_not_require_movable_physical_refinement"
    capability_floor = "distributed_winding" if requires_distributed else "centreline_geometry"

    if active_ids:
        if requested == "centreline":
            if requires_distributed:
                blocking.append({
                    "code": "centreline_cannot_close_capability_gap",
                    "model": requested,
                    "variable_ids": list(distributed_gap_ids),
                    "message": (
                        "Centreline cannot represent the winding-distribution effect of one or more "
                        "selected freedoms required by this effort mode. Choose Auto, Bundled, "
                        "Current sheet (for supported polyline coils), or Exact turns; alternatively "
                        "use Fast if limited physical coverage is intentional."
                    ),
                })
                physical_reason = "requested_centreline_below_required_capability_floor"
            else:
                physical_model = "centreline"
                physical_reason = "requested_centreline_represents_all_active_freedoms"
        elif requested == "current_sheet":
            unsupported = _unsupported_coils(physical_search_coil_ids, supports, "current_sheet")
            if unsupported:
                # Usually already covered by the authoritative check above, but keep
                # the physical-stage diagnosis explicit for report/debug consumers.
                if not any(item.get("code") == "requested_authoritative_model_unavailable" for item in blocking):
                    blocking.append({
                        "code": "current_sheet_refinement_unavailable",
                        "model": requested,
                        "coil_ids": [str(item["id"]) for item in unsupported],
                        "details": unsupported,
                        "message": "Current Sheet cannot represent every active optimizer coil in the physical-refinement stage.",
                    })
                physical_reason = "current_sheet_unavailable_for_active_coils"
            else:
                physical_model = "current_sheet"
                physical_reason = "requested_model_used_for_physical_refinement"
        elif requested == "bundled":
            unsupported = _unsupported_coils(physical_search_coil_ids, supports, "bundled")
            if unsupported:
                physical_reason = "bundled_unavailable_for_active_coils"
            else:
                physical_model = "bundled"
                physical_reason = "requested_model_used_for_physical_refinement"
        elif requested == "exact":
            # Exact is authoritative; Bundled is the deterministic search surrogate
            # because it preserves winding cross-section effects at bounded cost.
            unsupported = _unsupported_coils(physical_search_coil_ids, supports, "bundled")
            if unsupported:
                physical_reason = "bundled_search_surrogate_unavailable_for_active_coils"
            else:
                physical_model = "bundled"
                physical_reason = "exact_authority_uses_bundled_capable_search_surrogate"
        else:  # Auto
            # Auto must not dynamically choose Centreline on some candidates and a
            # distributed model on others when a capability-gap freedom is being
            # optimized.  Bundled provides a stable floor and supports both Circle and
            # Polyline windings.  If no distributed effect is required, prefer Bundled
            # when the active coils have physical specifications; otherwise Centreline
            # remains a valid capability floor for path/pose freedoms.
            unsupported_bundled = _unsupported_coils(physical_search_coil_ids, supports, "bundled")
            if requires_distributed:
                if unsupported_bundled:
                    blocking.append({
                        "code": "auto_capability_floor_unavailable",
                        "model": requested,
                        "variable_ids": list(distributed_gap_ids),
                        "coil_ids": [str(item["id"]) for item in unsupported_bundled],
                        "details": unsupported_bundled,
                        "message": (
                            "Auto cannot establish a distributed-winding capability floor for every "
                            "active optimizer coil. Supply/enable the physical winding specification "
                            "or choose an effort/model combination that does not require that freedom."
                        ),
                    })
                    physical_reason = "auto_distributed_capability_floor_unavailable"
                else:
                    physical_model = "bundled"
                    physical_reason = "auto_uses_bundled_distributed_winding_capability_floor"
            elif not unsupported_bundled and physical_search_coil_ids:
                physical_model = "bundled"
                physical_reason = "auto_prefers_bundled_capable_refinement_when_available"
            else:
                physical_model = "centreline"
                physical_reason = "auto_centreline_capability_floor_sufficient_for_active_freedoms"

    # If an explicit Bundled/Exact authority was accepted above but the active-stage
    # surrogate somehow cannot be built, make that a hard model-policy issue rather
    # than letting the worker silently fall back.
    if active_ids and requested in {"bundled", "exact"} and physical_model is None:
        unsupported = _unsupported_coils(physical_search_coil_ids, supports, "bundled")
        blocking.append({
            "code": "physical_search_model_unavailable",
            "model": "bundled",
            "coil_ids": [str(item["id"]) for item in unsupported],
            "details": unsupported,
            "message": "The optimizer cannot build the capable Bundled physical-search representation for every active coil.",
        })

    # Deduplicate codes that can be reached through both authority and refinement
    # checks while preserving the more detailed first diagnosis.
    deduped_blocking: list[dict[str, Any]] = []
    seen_issue_keys: set[tuple[str, tuple[str, ...]]] = set()
    for issue in blocking:
        key = (
            str(issue.get("code", "")),
            tuple(sorted(map(str, issue.get("coil_ids", []) or []))),
        )
        if key in seen_issue_keys:
            continue
        seen_issue_keys.add(key)
        deduped_blocking.append(issue)

    return {
        "schema_version": 1,
        "requested_model": requested,
        "search_effort": str(search_effort or "full").strip().lower(),
        "scout_model": "centreline",
        "physical_search_model": physical_model,
        "physical_search_reason": physical_reason,
        "physical_capability_floor": capability_floor,
        "physical_search_active_variable_ids": list(active_ids),
        "physical_search_capability_gap_variable_ids": list(gap_ids),
        "distributed_winding_gap_variable_ids": list(distributed_gap_ids),
        "physical_search_active_coil_ids": list(active_variable_coil_ids),
        "physical_search_coil_ids": list(physical_search_coil_ids),
        "authoritative_model": requested,
        "authoritative_mode": "dynamic_auto" if requested == "auto" else "fixed_requested_model",
        "authoritative_fallback_allowed": requested == "auto",
        "optimized_coil_ids": list(optimized_coil_ids),
        "participant_coil_ids": list(participant_ids),
        "authoritative_coil_ids": list(authoritative_coil_ids),
        "source_support": [copy.deepcopy(supports[key]) for key in supports],
        "blocking_issues": deduped_blocking,
        "compatible": not bool(deduped_blocking),
        "effort_scope": copy.deepcopy(scope),
        "notes": [
            "Centreline is a reconnaissance accelerator, not an automatic substitute for a missing capable-physics stage.",
            "Explicit requested models are not allowed to silently fall back to Centreline on optimizer-participating coils.",
            "Auto remains dynamic for final authority but uses a deterministic capability floor during movable physical refinement.",
        ],
    }
