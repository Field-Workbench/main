"""Directional feasibility-boundary recovery for Optimizer 2.4.

Phase 5 turns DRC/electrical feasibility into part of directional closure.
A hard user bound may be physically infeasible; that does not license the
optimizer to stop at an arbitrary last-feasible sample.  This helper bisects on
the variable's legal lattice (or continuous interval) until the nearest safe-side
value next to the measured constraint boundary is found, then reruns the bounded
directional search over the measured feasible interval.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

from .optimizer_directional_closure import (
    CLOSURE_FEASIBILITY_BOUNDARY,
    optimizer_directional_mark_closed,
    optimizer_directional_mark_feasibility_boundary,
    optimizer_directional_record_observation,
)
from .optimizer_directional_search import (
    DirectionalEvaluation,
    optimizer_bounded_directional_search,
    optimizer_directional_search_spec,
)


FEASIBILITY_BOUNDARY_SCHEMA_VERSION = 1
_CONSTRAINT_KINDS = frozenset({"drc", "electrical", "constraint", "feasibility"})


def _coerce(result: Any) -> DirectionalEvaluation:
    if isinstance(result, DirectionalEvaluation):
        return result
    if isinstance(result, dict):
        key = result.get("objective_key")
        if isinstance(key, list):
            key = tuple(key)
        elif key is not None and not isinstance(key, tuple):
            key = (key,)
        return DirectionalEvaluation(
            feasible=bool(result.get("feasible", True)),
            objective_key=key,
            payload=result.get("payload"),
            rejection_reason=(
                str(result.get("rejection_reason"))
                if result.get("rejection_reason") is not None else None
            ),
            feasibility_kind=(
                str(result.get("feasibility_kind"))
                if result.get("feasibility_kind") is not None else None
            ),
        )
    if result is None:
        return DirectionalEvaluation(False, rejection_reason="No evaluation result.", feasibility_kind="evaluation")
    key = tuple(result) if isinstance(result, (tuple, list)) else (result,)
    return DirectionalEvaluation(True, objective_key=key)


def _is_constraint_failure(result: DirectionalEvaluation) -> bool:
    return (not result.feasible) and str(result.feasibility_kind or "").lower() in _CONSTRAINT_KINDS


def _lattice_value(raw: float, *, low: float, high: float, step: float | None) -> float:
    value = min(high, max(low, float(raw)))
    if step is None or step <= 0.0:
        return value
    index = int(round((value - low) / step))
    return float(min(high, max(low, round(low + index * step, 12))))


def _resolved(a: float, b: float, *, step: float | None, tolerance: float) -> bool:
    distance = abs(float(a) - float(b))
    if step is not None and step > 0.0:
        return distance <= step + max(1.0e-12, step * 1.0e-10)
    return distance <= tolerance


def _search_result_observations(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(search_result.get("observations", []) or [])]


def optimizer_resolve_feasibility_boundary(
    variable: dict[str, Any],
    evaluate: Callable[[float], Any],
    search_result: dict[str, Any],
    *,
    side: str,
    closure_entry: dict[str, Any] | None = None,
    evidence_stage: str = "directional_feasibility_boundary",
    max_new_evaluations: int = 16,
) -> dict[str, Any] | None:
    """Measure one safe-side boundary adjacent to an infeasible hard endpoint."""
    normalized = str(side).strip().lower()
    if normalized not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    spec = optimizer_directional_search_spec(variable)
    bounds = dict(spec["bounds"])
    low = float(bounds["effective_min"])
    high = float(bounds["effective_max"])
    step = float(bounds["step"]) if bounds.get("step") is not None else None
    tolerance = float(spec["resolution_tolerance"])
    endpoint = low if normalized == "lower" else high

    observations = _search_result_observations(search_result)
    endpoint_rows = [
        item for item in observations
        if math.isclose(float(item.get("value", 0.0)), endpoint, rel_tol=0.0, abs_tol=max(1.0e-12, tolerance * 1.0e-6))
    ]
    if not endpoint_rows:
        return None
    endpoint_result = _coerce(endpoint_rows[-1])
    if not _is_constraint_failure(endpoint_result):
        return None

    feasible_values = sorted(
        float(item["value"])
        for item in observations
        if bool(item.get("feasible", False)) and item.get("objective_key") is not None
    )
    if normalized == "lower":
        safe = [value for value in feasible_values if value > endpoint]
        if not safe:
            return None
        feasible_value = safe[0]
    else:
        safe = [value for value in feasible_values if value < endpoint]
        if not safe:
            return None
        feasible_value = safe[-1]

    infeasible_value = float(endpoint)
    boundary_kind = endpoint_result.feasibility_kind
    rejection_reason = endpoint_result.rejection_reason
    new_evaluations = 0

    while new_evaluations < max(0, int(max_new_evaluations)) and not _resolved(
        infeasible_value, feasible_value, step=step, tolerance=tolerance
    ):
        if step is not None and step > 0.0:
            i_bad = int(round((infeasible_value - low) / step))
            i_good = int(round((feasible_value - low) / step))
            if normalized == "lower":
                index = (i_bad + i_good + 1) // 2
            else:
                index = (i_bad + i_good) // 2
            proposal = _lattice_value(low + index * step, low=low, high=high, step=step)
        else:
            proposal = 0.5 * (infeasible_value + feasible_value)
        result = _coerce(evaluate(float(proposal)))
        new_evaluations += 1
        if closure_entry is not None:
            optimizer_directional_record_observation(
                closure_entry,
                value=float(proposal),
                feasible=bool(result.feasible),
                objective_key=result.objective_key,
                rejection_reason=result.rejection_reason,
                feasibility_kind=result.feasibility_kind,
                evidence_stage=evidence_stage,
            )
        if result.feasible and result.objective_key is not None:
            feasible_value = float(proposal)
        elif _is_constraint_failure(result):
            infeasible_value = float(proposal)
            boundary_kind = result.feasibility_kind or boundary_kind
            rejection_reason = result.rejection_reason or rejection_reason
        else:
            return None

    if not _resolved(infeasible_value, feasible_value, step=step, tolerance=tolerance):
        return None
    payload = {
        "schema_version": FEASIBILITY_BOUNDARY_SCHEMA_VERSION,
        "side": normalized,
        "feasible_value": float(feasible_value),
        "infeasible_value": float(infeasible_value),
        "feasibility_kind": boundary_kind,
        "rejection_reason": rejection_reason,
        "new_evaluations": int(new_evaluations),
    }
    if closure_entry is not None:
        optimizer_directional_mark_feasibility_boundary(
            closure_entry,
            normalized,
            feasible_value=float(feasible_value),
            infeasible_value=float(infeasible_value),
            feasibility_kind=boundary_kind,
            rejection_reason=rejection_reason,
            evidence_stage=evidence_stage,
        )
    return payload


def optimizer_directional_search_with_feasibility_recovery(
    variable: dict[str, Any],
    evaluate: Callable[[float], Any],
    *,
    incumbent_value: float | None = None,
    closure_entry: dict[str, Any] | None = None,
    evidence_stage: str = "directional_search",
    max_evaluations: int = 33,
    boundary_evaluation_reserve: int = 12,
) -> dict[str, Any]:
    """Run bounded search, recover measured hard boundaries, then re-close inside them."""
    total_limit = max(4, int(max_evaluations))
    reserve = max(0, int(boundary_evaluation_reserve))
    # Feasibility recovery is correctness-floor work, not discretionary search.
    # Do not steal budget from the ordinary directional search to reserve it;
    # Phase 7 formalizes a separate correctness reserve at the stage-planner level.
    first_limit = total_limit
    first = optimizer_bounded_directional_search(
        variable,
        evaluate,
        incumbent_value=incumbent_value,
        closure_entry=closure_entry,
        evidence_stage=evidence_stage,
        max_evaluations=first_limit,
    )
    if bool(first.get("closed", False)) or not bool(first.get("needs_feasibility_boundary", False)):
        first["feasibility_recovery_used"] = False
        first["feasibility_boundaries"] = {}
        return first

    boundaries: dict[str, dict[str, Any]] = {}
    boundary_used = 0
    unresolved_sides = [
        side for side in ("lower", "upper")
        if not bool(dict((first.get("endpoint_status", {}) or {}).get(side, {}) or {}).get("feasible", False))
    ]
    for index, side in enumerate(unresolved_sides):
        remaining_reserve = max(0, reserve - boundary_used)
        remaining_sides = max(1, len(unresolved_sides) - index)
        side_limit = remaining_reserve // remaining_sides
        if side_limit <= 0:
            break
        resolved = optimizer_resolve_feasibility_boundary(
            variable,
            evaluate,
            first,
            side=side,
            closure_entry=closure_entry,
            evidence_stage=f"{evidence_stage}:feasibility_boundary",
            max_new_evaluations=side_limit,
        )
        if resolved is not None:
            boundaries[side] = resolved
            boundary_used += int(resolved.get("new_evaluations", 0) or 0)

    if not boundaries:
        first["feasibility_recovery_used"] = True
        first["feasibility_boundaries"] = {}
        first["feasibility_boundary_evaluations"] = int(boundary_used)
        return first

    narrowed = copy.deepcopy(variable)
    if "lower" in boundaries:
        narrowed["min"] = float(boundaries["lower"]["feasible_value"])
    if "upper" in boundaries:
        narrowed["max"] = float(boundaries["upper"]["feasible_value"])
    second = optimizer_bounded_directional_search(
        narrowed,
        evaluate,
        incumbent_value=(first.get("best_value") if first.get("best_value") is not None else incumbent_value),
        closure_entry=closure_entry,
        evidence_stage=f"{evidence_stage}:feasible_interval",
        max_evaluations=total_limit,
        record_endpoint_markers=False,
    )
    closure_reason = second.get("closure_reason")
    closure_bound = second.get("closure_bound")
    if closure_reason in {"lower_bound_best", "upper_bound_best"}:
        side = "lower" if closure_reason == "lower_bound_best" else "upper"
        if side in boundaries:
            closure_reason = CLOSURE_FEASIBILITY_BOUNDARY
            if closure_entry is not None:
                boundary = boundaries[side]
                optimizer_directional_mark_closed(
                    closure_entry,
                    reason=CLOSURE_FEASIBILITY_BOUNDARY,
                    bound=side,
                    details=(
                        "Best legal value lies on the measured safe side of a hard "
                        f"{boundary.get('feasibility_kind') or 'feasibility'} boundary."
                    ),
                    evidence_stage=f"{evidence_stage}:feasible_interval",
                )

    combined = dict(second)
    combined.update({
        "closure_reason": closure_reason,
        "closure_bound": closure_bound,
        "feasibility_recovery_used": True,
        "feasibility_boundaries": copy.deepcopy(boundaries),
        "feasibility_boundary_evaluations": int(boundary_used),
        "initial_search": copy.deepcopy(first),
        "evaluation_count": int(first.get("evaluation_count", 0) or 0)
        + int(boundary_used)
        + int(second.get("evaluation_count", 0) or 0),
        "needs_feasibility_boundary": False if bool(second.get("closed", False)) else True,
    })
    return combined
