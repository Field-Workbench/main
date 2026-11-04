"""Directional-closure bookkeeping for Optimizer 2.4.

The fidelity/capability layer answers whether a field model can *represent* a
freedom.  Directional closure answers a different question: whether the search
has actually followed that freedom far enough to justify stopping.

A bounded structural freedom starts unresolved.  It becomes closed only when a
later search stage supplies explicit evidence for one of the accepted closure
reasons (interior turnover, a tested best bound, a measured feasibility
boundary, objective saturation, or a real redundancy proof).  Merely observing
an interior candidate, locally polishing it, or sampling the coordinate in a
Sobol cloud never closes the direction by itself.

The Phase-2 bounded 1-D engine populates these records. Phase 4 additionally
archives and reopens conditional closure whenever another coupled structural
freedom moves, because a 1-D optimum is only valid on the slice where it was
proved. Phase 5 makes the same correctness contract govern measured feasibility
boundaries; Phase 6 extends it to capable-physics refinement.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Iterable


DIRECTIONAL_CLOSURE_SCHEMA_VERSION = 4

CLOSURE_UNRESOLVED = "unresolved"
CLOSURE_INTERIOR_TURNOVER = "interior_turnover"
CLOSURE_LOWER_BOUND_BEST = "lower_bound_best"
CLOSURE_UPPER_BOUND_BEST = "upper_bound_best"
CLOSURE_FEASIBILITY_BOUNDARY = "feasibility_boundary"
CLOSURE_OBJECTIVE_SATURATED = "objective_saturated"
CLOSURE_PROVEN_REDUNDANT = "proven_redundant"
CLOSURE_FIXED_RANGE = "fixed_range"

CLOSURE_REASONS = frozenset({
    CLOSURE_UNRESOLVED,
    CLOSURE_INTERIOR_TURNOVER,
    CLOSURE_LOWER_BOUND_BEST,
    CLOSURE_UPPER_BOUND_BEST,
    CLOSURE_FEASIBILITY_BOUNDARY,
    CLOSURE_OBJECTIVE_SATURATED,
    CLOSURE_PROVEN_REDUNDANT,
    CLOSURE_FIXED_RANGE,
})

CLOSED_REASONS = frozenset({
    CLOSURE_INTERIOR_TURNOVER,
    CLOSURE_LOWER_BOUND_BEST,
    CLOSURE_UPPER_BOUND_BEST,
    CLOSURE_FEASIBILITY_BOUNDARY,
    CLOSURE_OBJECTIVE_SATURATED,
    CLOSURE_PROVEN_REDUNDANT,
    CLOSURE_FIXED_RANGE,
})

TREND_DIRECTIONS = frozenset({"unknown", "lower", "upper", "flat", "bracketed"})


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _variable_step(variable: dict[str, Any]) -> float | None:
    """Mirror the optimizer's increment semantics without importing StudioAdapter."""
    raw = variable.get("step")
    if raw is None:
        return 1.0 if bool(variable.get("integer", False)) else None
    step = _finite_float(raw, -1.0)
    if step <= 0.0:
        return None
    return float(step)


def optimizer_directional_lattice_spec(variable: dict[str, Any]) -> dict[str, Any]:
    """Return the effective searchable interval for one bounded freedom.

    Wizard increments are anchored to ``step_origin`` or the source/base value,
    matching the existing optimizer lattice convention.  Requested bounds remain
    visible separately so reports can distinguish a literal user bound from the
    nearest legal lattice point.
    """
    low = _finite_float(variable.get("min"), 0.0)
    high = _finite_float(variable.get("max"), low)
    if high < low:
        low, high = high, low

    base_default = 0.5 * (low + high)
    base = _finite_float(variable.get("base"), base_default)
    step = _variable_step(variable)
    integer = bool(variable.get("integer", False))
    anchor = _finite_float(variable.get("step_origin", variable.get("base", 0.0)), 0.0)

    effective_low = float(low)
    effective_high = float(high)
    lattice_count: int | None = None
    if step is not None and high >= low:
        epsilon = max(1.0e-12, abs(step) * 1.0e-10)
        n_low = int(math.ceil((low - anchor) / step - epsilon))
        n_high = int(math.floor((high - anchor) / step + epsilon))
        if n_low <= n_high:
            effective_low = float(round(anchor + float(n_low) * step, 12))
            effective_high = float(round(anchor + float(n_high) * step, 12))
            lattice_count = int(n_high - n_low + 1)
        else:
            # Degenerate/misaligned range: keep the hard interval visible.  A later
            # execution stage can reject it, but Phase 1 must not invent closure.
            lattice_count = 0

    if integer:
        effective_low = float(int(round(effective_low)))
        effective_high = float(int(round(effective_high)))

    source = min(effective_high, max(effective_low, base)) if effective_high >= effective_low else base
    if step is not None and lattice_count and lattice_count > 0:
        epsilon = max(1.0e-12, abs(step) * 1.0e-10)
        n_low = int(math.ceil((effective_low - anchor) / step - epsilon))
        n_high = int(math.floor((effective_high - anchor) / step + epsilon))
        n_source = min(n_high, max(n_low, int(round((source - anchor) / step))))
        source = float(round(anchor + float(n_source) * step, 12))
    if integer:
        source = float(int(round(source)))

    return {
        "requested_min": float(low),
        "requested_max": float(high),
        "effective_min": float(effective_low),
        "effective_max": float(effective_high),
        "source_value": float(source),
        "step": (float(step) if step is not None else None),
        "step_origin": (float(anchor) if step is not None else None),
        "integer": integer,
        "lattice_count": lattice_count,
        "has_search_span": bool(effective_high > effective_low),
    }


def _endpoint_payload(value: float) -> dict[str, Any]:
    return {
        "value": float(value),
        "tested": False,
        "feasible": None,
        "observation_count": 0,
        "best_objective_key": None,
        "rejection_reason": None,
    }


def optimizer_directional_closure_entry(variable: dict[str, Any]) -> dict[str, Any]:
    """Create an unresolved directional-search record for one structural freedom."""
    lattice = optimizer_directional_lattice_spec(variable)
    has_span = bool(lattice["has_search_span"])
    fixed = not has_span
    closure_reason = CLOSURE_FIXED_RANGE if fixed else CLOSURE_UNRESOLVED
    return {
        "id": str(variable.get("id", "")),
        "kind": str(variable.get("kind", "")),
        "label": str(variable.get("label", variable.get("id", "freedom"))),
        "group_kind": str(variable.get("group_kind", "")),
        "group": str(variable.get("group", "")),
        "coil_ids": [str(value) for value in list(variable.get("coil_ids", []) or [])],
        "unit": str(variable.get("unit", "")),
        "bounds": copy.deepcopy(lattice),
        "source": {
            "value": float(lattice["source_value"]),
            "tested": False,
            "feasible": None,
            "objective_key": None,
        },
        "endpoints": {
            "lower": _endpoint_payload(float(lattice["effective_min"])),
            "upper": _endpoint_payload(float(lattice["effective_max"])),
        },
        "observations": {
            "count": 0,
            "distinct_value_count": 0,
            "minimum_tested_value": None,
            "maximum_tested_value": None,
            "tested_values": [],
        },
        "observation_history": [],
        "feasibility_boundaries": {
            "lower": None,
            "upper": None,
        },
        "search_policy": {
            "engine": "bounded_directional_v1",
            "mandatory_endpoints": True,
            "mandatory_source": True,
            "mandatory_incumbent": True,
            "interior_strategy": "adaptive_bracket_subdivision",
            "objective_order": "caller_lexicographic_key",
            "infeasible_endpoint_requires_boundary_resolution": True,
        },
        "best_observation": None,
        "trend": {
            "direction": "unknown",
            "evidence_count": 0,
            "monotonic": False,
            "open_lower": bool(has_span),
            "open_upper": bool(has_span),
            "last_improving_from": None,
            "last_improving_to": None,
        },
        "bracket": {
            "established": False,
            "lower_value": None,
            "best_value": None,
            "upper_value": None,
        },
        "conditional_context": None,
        "reopen_count": 0,
        "reopen_history": [],
        "closure": {
            "closed": bool(fixed),
            "reason": closure_reason,
            "bound": None,
            "details": (
                "Freedom has no searchable span after applying its increment lattice."
                if fixed else None
            ),
            "evidence_stage": None,
        },
    }


def optimizer_directional_closure_plan(
    structural_variables: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Create the Optimizer-2.4 directional-closure contract.

    This is intentionally independent of Fast/Balanced/Full.  Effort levels may
    later change *how much extra exploration* is done, but a user-declared bounded
    structural freedom begins with the same correctness obligation in every mode.
    """
    entries = [optimizer_directional_closure_entry(item) for item in structural_variables]
    by_id = {
        str(item.get("id", "")): item
        for item in entries
        if str(item.get("id", ""))
    }
    unresolved = [
        variable_id
        for variable_id, item in by_id.items()
        if not bool((item.get("closure") or {}).get("closed", False))
    ]
    closed = [value for value in by_id if value not in set(unresolved)]
    return {
        "schema_version": DIRECTIONAL_CLOSURE_SCHEMA_VERSION,
        "contract": "bounded_structural_freedoms_require_directional_closure",
        "status": "pending" if unresolved else "closed",
        "correctness_floor_applies_to_all_efforts": True,
        "variable_order": [str(item.get("id", "")) for item in entries],
        "entries": entries,
        "summary": {
            "structural_freedom_count": len(entries),
            "closed_count": len(closed),
            "unresolved_count": len(unresolved),
            "closed_variable_ids": closed,
            "unresolved_variable_ids": unresolved,
            "bound_limited_variable_ids": [],
            "feasibility_limited_variable_ids": [],
        },
        "notes": [
            "Fidelity observability and directional closure are independent contracts.",
            "Interior sampling, local polish, or Sobol coverage alone does not close a bounded direction.",
            "A later search stage must provide turnover, tested-bound, feasibility-boundary, saturation, or redundancy evidence.",
            "Phase 2 provides a reusable bounded 1-D engine with mandatory legal endpoints and adaptive bracket subdivision.",
        ],
    }


def optimizer_directional_entries_by_id(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return mutable plan entries keyed by freedom id."""
    return {
        str(item.get("id", "")): item
        for item in list(plan.get("entries", []) or [])
        if str(item.get("id", ""))
    }



def optimizer_directional_record_observation(
    entry: dict[str, Any],
    *,
    value: float,
    feasible: bool,
    objective_key: Any = None,
    rejection_reason: str | None = None,
    feasibility_kind: str | None = None,
    evidence_stage: str | None = None,
) -> None:
    """Record one scalar observation without inferring directional closure.

    Phase 2 uses this helper for every value visited by the bounded 1-D engine.
    The objective key is opaque bookkeeping here; ordering remains the caller's
    responsibility so the directional layer never invents a second objective.
    """
    numeric = float(value)
    observations = entry.setdefault("observations", {})
    observations["count"] = int(observations.get("count", 0) or 0) + 1

    tested_values = list(observations.get("tested_values", []) or [])
    tolerance = 1.0e-12
    if not any(math.isclose(numeric, float(old), rel_tol=0.0, abs_tol=tolerance) for old in tested_values):
        tested_values.append(numeric)
    tested_values.sort()
    observations["tested_values"] = tested_values
    observations["distinct_value_count"] = len(tested_values)
    observations["minimum_tested_value"] = tested_values[0] if tested_values else None
    observations["maximum_tested_value"] = tested_values[-1] if tested_values else None

    source = entry.setdefault("source", {})
    source_value = source.get("value")
    if source_value is not None and math.isclose(
        numeric, float(source_value), rel_tol=0.0, abs_tol=tolerance
    ):
        source["tested"] = True
        source["feasible"] = bool(feasible)
        source["objective_key"] = copy.deepcopy(objective_key)

    history = list(entry.get("observation_history", []) or [])
    history.append({
        "value": numeric,
        "feasible": bool(feasible),
        "objective_key": copy.deepcopy(objective_key),
        "rejection_reason": str(rejection_reason) if rejection_reason else None,
        "feasibility_kind": str(feasibility_kind) if feasibility_kind else None,
        "evidence_stage": str(evidence_stage) if evidence_stage else None,
    })
    entry["observation_history"] = history

    if feasible and objective_key is not None:
        key_tuple = tuple(objective_key) if isinstance(objective_key, (tuple, list)) else (objective_key,)
        previous = entry.get("best_observation")
        previous_key = None
        if isinstance(previous, dict) and previous.get("objective_key") is not None:
            raw_key = previous.get("objective_key")
            previous_key = tuple(raw_key) if isinstance(raw_key, (tuple, list)) else (raw_key,)
        if previous_key is None or key_tuple < previous_key:
            entry["best_observation"] = {
                "value": numeric,
                "objective_key": copy.deepcopy(objective_key),
                "evidence_stage": str(evidence_stage) if evidence_stage else None,
            }

def optimizer_directional_mark_endpoint(
    entry: dict[str, Any],
    endpoint: str,
    *,
    feasible: bool | None,
    objective_key: Any = None,
    rejection_reason: str | None = None,
) -> None:
    """Record evidence that a lower/upper searchable endpoint was evaluated.

    Phase 2 will call this from the real bounded search.  The helper deliberately
    does not infer whether the endpoint is *best*; objective ordering belongs to
    the optimizer objective comparator rather than this bookkeeping module.
    """
    side = str(endpoint).strip().lower()
    if side not in {"lower", "upper"}:
        raise ValueError("endpoint must be 'lower' or 'upper'")
    payload = entry.setdefault("endpoints", {}).setdefault(side, {})
    payload["tested"] = True
    payload["feasible"] = feasible
    payload["observation_count"] = int(payload.get("observation_count", 0) or 0) + 1
    payload["best_objective_key"] = copy.deepcopy(objective_key)
    payload["rejection_reason"] = str(rejection_reason) if rejection_reason else None
    trend = entry.setdefault("trend", {})
    if side == "lower":
        trend["open_lower"] = False
    else:
        trend["open_upper"] = False


def optimizer_directional_set_trend(
    entry: dict[str, Any],
    *,
    direction: str,
    evidence_count: int = 0,
    monotonic: bool = False,
    last_improving_from: float | None = None,
    last_improving_to: float | None = None,
) -> None:
    """Store search-derived directional evidence without deciding closure."""
    normalized = str(direction).strip().lower()
    if normalized not in TREND_DIRECTIONS:
        raise ValueError(f"unsupported trend direction: {direction}")
    trend = entry.setdefault("trend", {})
    trend.update({
        "direction": normalized,
        "evidence_count": max(0, int(evidence_count)),
        "monotonic": bool(monotonic),
        "last_improving_from": (
            float(last_improving_from) if last_improving_from is not None else None
        ),
        "last_improving_to": (
            float(last_improving_to) if last_improving_to is not None else None
        ),
    })


def optimizer_directional_mark_feasibility_boundary(
    entry: dict[str, Any],
    side: str,
    *,
    feasible_value: float,
    infeasible_value: float,
    feasibility_kind: str | None = None,
    rejection_reason: str | None = None,
    evidence_stage: str | None = None,
) -> None:
    """Record a measured safe-side boundary for one constrained direction."""
    normalized = str(side).strip().lower()
    if normalized not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    boundaries = entry.setdefault("feasibility_boundaries", {"lower": None, "upper": None})
    boundaries[normalized] = {
        "side": normalized,
        "feasible_value": float(feasible_value),
        "infeasible_value": float(infeasible_value),
        "feasibility_kind": str(feasibility_kind) if feasibility_kind else None,
        "rejection_reason": str(rejection_reason) if rejection_reason else None,
        "evidence_stage": str(evidence_stage) if evidence_stage else None,
    }


def optimizer_directional_mark_closed(
    entry: dict[str, Any],
    *,
    reason: str,
    bound: str | None = None,
    details: str | None = None,
    evidence_stage: str | None = None,
    bracket: dict[str, Any] | None = None,
) -> None:
    """Close one freedom only with an explicit accepted reason."""
    normalized = str(reason).strip().lower()
    if normalized not in CLOSED_REASONS:
        raise ValueError(f"unsupported directional closure reason: {reason}")
    normalized_bound = str(bound).strip().lower() if bound is not None else None
    if normalized_bound not in {None, "lower", "upper"}:
        raise ValueError("bound must be lower, upper, or None")
    if normalized == CLOSURE_LOWER_BOUND_BEST:
        normalized_bound = "lower"
    elif normalized == CLOSURE_UPPER_BOUND_BEST:
        normalized_bound = "upper"
    entry["closure"] = {
        "closed": True,
        "reason": normalized,
        "bound": normalized_bound,
        "details": str(details) if details else None,
        "evidence_stage": str(evidence_stage) if evidence_stage else None,
    }
    trend = entry.setdefault("trend", {})
    trend["open_lower"] = False
    trend["open_upper"] = False
    if bracket is not None:
        entry["bracket"] = copy.deepcopy(bracket)



def optimizer_directional_set_conditional_context(
    entry: dict[str, Any],
    context: dict[str, float] | None,
) -> None:
    """Record the coupled-coordinate context on which active closure was proved."""
    if context is None:
        entry["conditional_context"] = None
        return
    entry["conditional_context"] = {
        str(key): float(value)
        for key, value in context.items()
    }


def optimizer_directional_reopen(
    entry: dict[str, Any],
    *,
    reason: str,
    triggered_by_variable_id: str | None = None,
    evidence_stage: str | None = None,
    new_context: dict[str, float] | None = None,
) -> None:
    """Invalidate conditional 1-D closure after a coupled coordinate moves.

    Closure evidence is conditional on the other structural coordinates held
    fixed during that axis search.  When a coupled coordinate changes, archive
    the active slice and reset only slice-local evidence.  The append-only
    ``observation_history`` remains intact so reports can explain how the search
    evolved across coupled cycles.
    """
    bounds = dict(entry.get("bounds", {}) or {})
    has_span = bool(bounds.get("has_search_span", False))
    if not has_span:
        return

    archive = {
        "reason": str(reason),
        "triggered_by_variable_id": (
            str(triggered_by_variable_id) if triggered_by_variable_id else None
        ),
        "evidence_stage": str(evidence_stage) if evidence_stage else None,
        "conditional_context": copy.deepcopy(entry.get("conditional_context")),
        "source": copy.deepcopy(entry.get("source", {})),
        "endpoints": copy.deepcopy(entry.get("endpoints", {})),
        "observations": copy.deepcopy(entry.get("observations", {})),
        "best_observation": copy.deepcopy(entry.get("best_observation")),
        "trend": copy.deepcopy(entry.get("trend", {})),
        "bracket": copy.deepcopy(entry.get("bracket", {})),
        "feasibility_boundaries": copy.deepcopy(entry.get("feasibility_boundaries", {})),
        "closure": copy.deepcopy(entry.get("closure", {})),
    }
    history = list(entry.get("reopen_history", []) or [])
    history.append(archive)
    entry["reopen_history"] = history
    entry["reopen_count"] = int(entry.get("reopen_count", 0) or 0) + 1

    source = entry.setdefault("source", {})
    source["tested"] = False
    source["feasible"] = None
    source["objective_key"] = None
    entry["endpoints"] = {
        "lower": _endpoint_payload(float(bounds.get("effective_min", 0.0))),
        "upper": _endpoint_payload(float(bounds.get("effective_max", 0.0))),
    }
    entry["observations"] = {
        "count": 0,
        "distinct_value_count": 0,
        "minimum_tested_value": None,
        "maximum_tested_value": None,
        "tested_values": [],
    }
    entry["best_observation"] = None
    entry["trend"] = {
        "direction": "unknown",
        "evidence_count": 0,
        "monotonic": False,
        "open_lower": True,
        "open_upper": True,
        "last_improving_from": None,
        "last_improving_to": None,
    }
    entry["bracket"] = {
        "established": False,
        "lower_value": None,
        "best_value": None,
        "upper_value": None,
    }
    entry["feasibility_boundaries"] = {"lower": None, "upper": None}
    entry["closure"] = {
        "closed": False,
        "reason": CLOSURE_UNRESOLVED,
        "bound": None,
        "details": (
            "Previous conditional closure was reopened because a coupled structural freedom moved."
        ),
        "evidence_stage": str(evidence_stage) if evidence_stage else None,
    }
    optimizer_directional_set_conditional_context(entry, new_context)

def optimizer_directional_refresh_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """Recompute plan-level closure counts after Phase-2+ search updates."""
    entries = optimizer_directional_entries_by_id(plan)
    closed: list[str] = []
    unresolved: list[str] = []
    bound_limited: list[str] = []
    feasibility_limited: list[str] = []
    for variable_id, entry in entries.items():
        closure = dict(entry.get("closure", {}) or {})
        if bool(closure.get("closed", False)):
            closed.append(variable_id)
            reason = str(closure.get("reason", ""))
            if reason in {CLOSURE_LOWER_BOUND_BEST, CLOSURE_UPPER_BOUND_BEST}:
                bound_limited.append(variable_id)
            if reason == CLOSURE_FEASIBILITY_BOUNDARY:
                feasibility_limited.append(variable_id)
        else:
            unresolved.append(variable_id)
    summary = {
        "structural_freedom_count": len(entries),
        "closed_count": len(closed),
        "unresolved_count": len(unresolved),
        "closed_variable_ids": closed,
        "unresolved_variable_ids": unresolved,
        "bound_limited_variable_ids": bound_limited,
        "feasibility_limited_variable_ids": feasibility_limited,
    }
    plan["summary"] = summary
    plan["status"] = "closed" if not unresolved else "pending"
    return summary
