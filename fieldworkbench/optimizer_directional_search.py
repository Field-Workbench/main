"""Bounded one-dimensional directional search for Optimizer 2.4.

This module provides the correctness-floor primitive introduced in Phase 2 of
Optimizer 2.4.  It is intentionally independent of StudioAdapter and field
physics: callers provide one scalar freedom plus an evaluator that returns the
*existing optimizer objective key* for a proposed legal value.

The search differs from ordinary local polish in three important ways:

* both legal endpoints are mandatory observations;
* an interior incumbent cannot terminate the search merely because nearby
  samples failed to improve -- the interval must be directionally closed;
* when an interior best is found, the containing bracket is adaptively
  subdivided until it is closed at the freedom's lattice resolution (or a
  continuous tolerance).

The engine mutates an Optimizer-2.4 directional-closure entry when one is
provided.  Phase 3 wires this primitive into the scout funnel; Phase 4 adds coupled
cycles and Phase 5 wraps it with measured feasibility-boundary recovery.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .optimizer_directional_closure import (
    CLOSURE_INTERIOR_TURNOVER,
    CLOSURE_LOWER_BOUND_BEST,
    CLOSURE_UPPER_BOUND_BEST,
    optimizer_directional_lattice_spec,
    optimizer_directional_mark_closed,
    optimizer_directional_mark_endpoint,
    optimizer_directional_record_observation,
    optimizer_directional_set_trend,
)


DIRECTIONAL_SEARCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DirectionalEvaluation:
    """One scalar observation returned by a directional-search evaluator.

    ``objective_key`` must use the same ordering as the optimizer's final
    ranking: lexicographically smaller is better.  Infeasible observations may
    omit it.  ``payload`` is carried through untouched so a caller can recover
    the corresponding candidate/row without a second lookup.
    """

    feasible: bool
    objective_key: tuple[Any, ...] | None = None
    payload: Any = None
    rejection_reason: str | None = None
    feasibility_kind: str | None = None


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _key_tuple(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _coerce_evaluation(result: Any) -> DirectionalEvaluation:
    if isinstance(result, DirectionalEvaluation):
        return result
    if result is None:
        return DirectionalEvaluation(
            False, rejection_reason="Evaluator returned no result.", feasibility_kind="evaluation"
        )
    if isinstance(result, dict):
        feasible = bool(result.get("feasible", True))
        key = _key_tuple(result.get("objective_key"))
        if feasible and key is None:
            return DirectionalEvaluation(
                False,
                payload=result.get("payload"),
                rejection_reason="Feasible directional evaluation did not provide an objective key.",
                feasibility_kind="evaluation",
            )
        return DirectionalEvaluation(
            feasible=feasible,
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
    key = _key_tuple(result)
    return DirectionalEvaluation(True, objective_key=key)


def _objective_better(left: DirectionalEvaluation, right: DirectionalEvaluation) -> bool:
    if not left.feasible or left.objective_key is None:
        return False
    if not right.feasible or right.objective_key is None:
        return True
    return tuple(left.objective_key) < tuple(right.objective_key)


def _objective_not_worse(left: DirectionalEvaluation, right: DirectionalEvaluation) -> bool:
    if not left.feasible or left.objective_key is None:
        return False
    if not right.feasible or right.objective_key is None:
        return True
    return tuple(left.objective_key) <= tuple(right.objective_key)


def _lattice_index(value: float, *, low: float, step: float) -> int:
    return int(round((float(value) - float(low)) / float(step)))


def _lattice_value(index: int, *, low: float, step: float, high: float) -> float:
    return float(min(high, max(low, round(low + float(index) * step, 12))))


def _dedupe_values(values: Iterable[float], *, tolerance: float) -> list[float]:
    ordered: list[float] = []
    for value in values:
        numeric = float(value)
        if any(math.isclose(numeric, old, rel_tol=0.0, abs_tol=tolerance) for old in ordered):
            continue
        ordered.append(numeric)
    return ordered


def optimizer_directional_search_spec(
    variable: dict[str, Any],
    *,
    incumbent_value: float | None = None,
    continuous_tolerance_fraction: float = 2.5e-3,
) -> dict[str, Any]:
    """Describe the legal scalar search interval used by the Phase-2 engine."""
    lattice = optimizer_directional_lattice_spec(variable)
    low = float(lattice["effective_min"])
    high = float(lattice["effective_max"])
    source = float(lattice["source_value"])
    step = lattice.get("step")
    incumbent = source if incumbent_value is None else min(high, max(low, float(incumbent_value)))
    if step is not None and float(step) > 0.0 and int(lattice.get("lattice_count") or 0) > 0:
        index = _lattice_index(incumbent, low=low, step=float(step))
        incumbent = _lattice_value(index, low=low, step=float(step), high=high)
        tolerance = 0.5 * float(step)
    else:
        span = max(0.0, high - low)
        tolerance = max(1.0e-12, span * max(1.0e-6, float(continuous_tolerance_fraction)))
    anchor_tolerance = max(1.0e-12, tolerance * 1.0e-6)
    anchors = _dedupe_values([low, source, incumbent, high], tolerance=anchor_tolerance)
    return {
        "schema_version": DIRECTIONAL_SEARCH_SCHEMA_VERSION,
        "bounds": copy.deepcopy(lattice),
        "incumbent_value": float(incumbent),
        "mandatory_anchor_values": anchors,
        "resolution_tolerance": float(tolerance),
        "strategy": "mandatory_endpoints_then_adaptive_bracket_subdivision",
    }


def optimizer_bounded_directional_search(
    variable: dict[str, Any],
    evaluate: Callable[[float], Any],
    *,
    incumbent_value: float | None = None,
    closure_entry: dict[str, Any] | None = None,
    evidence_stage: str = "directional_search",
    max_evaluations: int = 33,
    continuous_tolerance_fraction: float = 2.5e-3,
    record_endpoint_markers: bool = True,
) -> dict[str, Any]:
    """Search one bounded freedom until its direction is explicitly closed.

    The caller's evaluator is invoked with legal scalar values and must return a
    :class:`DirectionalEvaluation`, a compatible dictionary, or directly an
    objective-key tuple/list.  Objective keys are ordered exactly as elsewhere
    in Field Workbench: lexicographically smaller is better.

    Both legal bounds are evaluated before any closure decision.  If the best
    feasible observation is a bound, the freedom closes as bound-limited.  If
    the best observation is interior, the nearest observations on both sides
    define a bracket.  The wider side(s) of that bracket are repeatedly bisected
    (on the legal increment lattice when present) until the best point is
    bracketed at search resolution.

    Infeasible endpoints are recorded but are *not* treated as a measured
    feasibility boundary here; Phase 5 owns locating the nearest legal boundary.
    Such a direction therefore remains unresolved.
    """
    max_evaluations = max(4, int(max_evaluations))
    spec = optimizer_directional_search_spec(
        variable,
        incumbent_value=incumbent_value,
        continuous_tolerance_fraction=continuous_tolerance_fraction,
    )
    bounds = dict(spec["bounds"])
    low = float(bounds["effective_min"])
    high = float(bounds["effective_max"])
    source = float(bounds["source_value"])
    step = bounds.get("step")
    tolerance = float(spec["resolution_tolerance"])
    compare_tolerance = max(1.0e-12, tolerance * 1.0e-6)

    observations: dict[float, DirectionalEvaluation] = {}
    evaluation_order: list[float] = []

    def canonical_value(raw: float) -> float:
        value = min(high, max(low, float(raw)))
        if step is not None and float(step) > 0.0 and int(bounds.get("lattice_count") or 0) > 0:
            index = _lattice_index(value, low=low, step=float(step))
            return _lattice_value(index, low=low, step=float(step), high=high)
        return float(value)

    def existing_value(value: float) -> float | None:
        for old in observations:
            if math.isclose(old, value, rel_tol=0.0, abs_tol=compare_tolerance):
                return old
        return None

    def record(value: float) -> DirectionalEvaluation | None:
        if len(observations) >= max_evaluations:
            return None
        value = canonical_value(value)
        old = existing_value(value)
        if old is not None:
            return observations[old]
        result = _coerce_evaluation(evaluate(float(value)))
        observations[float(value)] = result
        evaluation_order.append(float(value))
        if closure_entry is not None:
            optimizer_directional_record_observation(
                closure_entry,
                value=float(value),
                feasible=bool(result.feasible),
                objective_key=result.objective_key,
                rejection_reason=result.rejection_reason,
                feasibility_kind=result.feasibility_kind,
                evidence_stage=evidence_stage,
            )
            if record_endpoint_markers and math.isclose(value, low, rel_tol=0.0, abs_tol=compare_tolerance):
                optimizer_directional_mark_endpoint(
                    closure_entry,
                    "lower",
                    feasible=bool(result.feasible),
                    objective_key=result.objective_key,
                    rejection_reason=result.rejection_reason,
                )
            if record_endpoint_markers and math.isclose(value, high, rel_tol=0.0, abs_tol=compare_tolerance):
                optimizer_directional_mark_endpoint(
                    closure_entry,
                    "upper",
                    feasible=bool(result.feasible),
                    objective_key=result.objective_key,
                    rejection_reason=result.rejection_reason,
                )
        return result

    # Correctness floor: both legal endpoints, source, and incumbent are explicit
    # observations.  The order evaluates source/incumbent first only so progress
    # UIs can show a useful incumbent quickly; closure is impossible until bounds
    # have also been visited.
    incumbent = float(spec["incumbent_value"])
    initial_values = _dedupe_values(
        [source, incumbent, low, high], tolerance=compare_tolerance
    )
    for value in initial_values:
        record(value)

    def feasible_rows() -> list[tuple[float, DirectionalEvaluation]]:
        return sorted(
            [
                (value, result)
                for value, result in observations.items()
                if result.feasible and result.objective_key is not None
            ],
            key=lambda item: item[0],
        )

    def best_row() -> tuple[float, DirectionalEvaluation] | None:
        # Preserve evaluation order for exact objective-key ties.  Source/incumbent
        # anchors are evaluated before bounds, so a perfectly flat objective is not
        # spuriously labelled bound-limited merely because the lower bound sorts first.
        rows = [
            (value, observations[value])
            for value in evaluation_order
            if observations[value].feasible and observations[value].objective_key is not None
        ]
        if not rows:
            return None
        return min(rows, key=lambda item: tuple(item[1].objective_key or ()))

    def side_rows(best_value: float) -> tuple[
        tuple[float, DirectionalEvaluation] | None,
        tuple[float, DirectionalEvaluation] | None,
    ]:
        rows = feasible_rows()
        left = [item for item in rows if item[0] < best_value - compare_tolerance]
        right = [item for item in rows if item[0] > best_value + compare_tolerance]
        return (left[-1] if left else None, right[0] if right else None)

    def side_is_resolved(a: float, b: float) -> bool:
        distance = abs(float(a) - float(b))
        if step is not None and float(step) > 0.0:
            return distance <= float(step) + compare_tolerance
        return distance <= tolerance

    closure_reason: str | None = None
    closure_bound: str | None = None
    bracket_payload: dict[str, Any] | None = None

    # Adaptive bracket subdivision.  Because both bounds are already present,
    # any improving outward trend necessarily ends either at a tested bound or
    # at an interior best with worse observations on both sides.  We then shrink
    # the interior bracket instead of declaring victory from a local neighbour.
    while len(observations) < max_evaluations:
        best = best_row()
        if best is None:
            break
        best_value, best_result = best

        lower_obs = observations.get(existing_value(low) or low)
        upper_obs = observations.get(existing_value(high) or high)
        lower_feasible = bool(lower_obs and lower_obs.feasible and lower_obs.objective_key is not None)
        upper_feasible = bool(upper_obs and upper_obs.feasible and upper_obs.objective_key is not None)

        if math.isclose(best_value, low, rel_tol=0.0, abs_tol=compare_tolerance):
            # A winning endpoint is not closed from one long jump.  Walk/bracket
            # inward until the bound has a resolution-adjacent non-better neighbour.
            rows = feasible_rows()
            right_candidates = [item for item in rows if item[0] > low + compare_tolerance]
            if not lower_feasible or not right_candidates:
                break
            right_value, right_result = right_candidates[0]
            if side_is_resolved(low, right_value):
                if _objective_not_worse(best_result, right_result):
                    closure_reason = CLOSURE_LOWER_BOUND_BEST
                    closure_bound = "lower"
                break
            if step is not None and float(step) > 0.0:
                low_index = _lattice_index(low, low=low, step=float(step))
                right_index = _lattice_index(right_value, low=low, step=float(step))
                proposal = _lattice_value(
                    (low_index + right_index) // 2,
                    low=low, step=float(step), high=high,
                )
            else:
                proposal = 0.5 * (low + right_value)
            if existing_value(canonical_value(proposal)) is None:
                record(proposal)
                continue
            break

        if math.isclose(best_value, high, rel_tol=0.0, abs_tol=compare_tolerance):
            # Symmetric upper-bound confirmation.  This is the behaviour that
            # prevents an improving 77 -> 78 -> 79 trend from stopping before 85.
            rows = feasible_rows()
            left_candidates = [item for item in rows if item[0] < high - compare_tolerance]
            if not upper_feasible or not left_candidates:
                break
            left_value, left_result = left_candidates[-1]
            if side_is_resolved(left_value, high):
                if _objective_not_worse(best_result, left_result):
                    closure_reason = CLOSURE_UPPER_BOUND_BEST
                    closure_bound = "upper"
                break
            if step is not None and float(step) > 0.0:
                left_index = _lattice_index(left_value, low=low, step=float(step))
                high_index = _lattice_index(high, low=low, step=float(step))
                proposal = _lattice_value(
                    (left_index + high_index + 1) // 2,
                    low=low, step=float(step), high=high,
                )
            else:
                proposal = 0.5 * (left_value + high)
            if existing_value(canonical_value(proposal)) is None:
                record(proposal)
                continue
            break

        left, right = side_rows(best_value)
        # An infeasible hard endpoint leaves a missing feasible side.  Phase 5
        # must locate the actual DRC/electrical boundary before closure.
        if left is None or right is None:
            break

        left_value, left_result = left
        right_value, right_result = right
        left_resolved = side_is_resolved(left_value, best_value)
        right_resolved = side_is_resolved(best_value, right_value)
        if left_resolved and right_resolved:
            if _objective_not_worse(best_result, left_result) and _objective_not_worse(best_result, right_result):
                closure_reason = CLOSURE_INTERIOR_TURNOVER
                bracket_payload = {
                    "established": True,
                    "lower_value": float(left_value),
                    "best_value": float(best_value),
                    "upper_value": float(right_value),
                }
            break

        proposals: list[float] = []
        if not left_resolved:
            if step is not None and float(step) > 0.0:
                left_index = _lattice_index(left_value, low=low, step=float(step))
                best_index = _lattice_index(best_value, low=low, step=float(step))
                if best_index - left_index > 1:
                    proposals.append(_lattice_value(
                        (left_index + best_index) // 2,
                        low=low, step=float(step), high=high,
                    ))
            else:
                proposals.append(0.5 * (left_value + best_value))
        if not right_resolved:
            if step is not None and float(step) > 0.0:
                best_index = _lattice_index(best_value, low=low, step=float(step))
                right_index = _lattice_index(right_value, low=low, step=float(step))
                if right_index - best_index > 1:
                    proposals.append(_lattice_value(
                        (best_index + right_index + 1) // 2,
                        low=low, step=float(step), high=high,
                    ))
            else:
                proposals.append(0.5 * (best_value + right_value))

        proposals = _dedupe_values(proposals, tolerance=compare_tolerance)
        proposals = [value for value in proposals if existing_value(canonical_value(value)) is None]
        if not proposals:
            # There is no new legal point inside the bracket, so an interior best
            # with non-better neighbours is resolution-closed.
            if _objective_not_worse(best_result, left_result) and _objective_not_worse(best_result, right_result):
                closure_reason = CLOSURE_INTERIOR_TURNOVER
                bracket_payload = {
                    "established": True,
                    "lower_value": float(left_value),
                    "best_value": float(best_value),
                    "upper_value": float(right_value),
                }
            break
        for value in proposals:
            if len(observations) >= max_evaluations:
                break
            record(value)

    best = best_row()
    sorted_feasible = feasible_rows()
    trend_direction = "unknown"
    monotonic = False
    evidence_count = 0
    improving_from: float | None = None
    improving_to: float | None = None
    if best is not None and sorted_feasible:
        best_value, _best_result = best
        evidence_count = len(sorted_feasible)
        if math.isclose(best_value, low, rel_tol=0.0, abs_tol=compare_tolerance):
            trend_direction = "lower"
            keys = [tuple(item[1].objective_key or ()) for item in sorted_feasible]
            monotonic = len(keys) >= 2 and all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))
            if len(sorted_feasible) >= 2:
                improving_from = float(sorted_feasible[1][0])
                improving_to = float(sorted_feasible[0][0])
        elif math.isclose(best_value, high, rel_tol=0.0, abs_tol=compare_tolerance):
            trend_direction = "upper"
            keys = [tuple(item[1].objective_key or ()) for item in sorted_feasible]
            monotonic = len(keys) >= 2 and all(keys[i + 1] <= keys[i] for i in range(len(keys) - 1))
            if len(sorted_feasible) >= 2:
                improving_from = float(sorted_feasible[-2][0])
                improving_to = float(sorted_feasible[-1][0])
        elif closure_reason == CLOSURE_INTERIOR_TURNOVER:
            trend_direction = "bracketed"
        else:
            left, right = side_rows(best_value)
            if left is not None and right is None:
                trend_direction = "upper"
            elif right is not None and left is None:
                trend_direction = "lower"

    if closure_entry is not None:
        optimizer_directional_set_trend(
            closure_entry,
            direction=trend_direction,
            evidence_count=evidence_count,
            monotonic=monotonic,
            last_improving_from=improving_from,
            last_improving_to=improving_to,
        )
        if closure_reason is not None:
            details = (
                "Best evaluated legal value is the lower search bound."
                if closure_reason == CLOSURE_LOWER_BOUND_BEST else
                "Best evaluated legal value is the upper search bound."
                if closure_reason == CLOSURE_UPPER_BOUND_BEST else
                "Interior optimum is bracketed by non-better observations at the configured search resolution."
            )
            optimizer_directional_mark_closed(
                closure_entry,
                reason=closure_reason,
                bound=closure_bound,
                details=details,
                evidence_stage=evidence_stage,
                bracket=bracket_payload,
            )

    best_payload = None
    best_value: float | None = None
    best_key: tuple[Any, ...] | None = None
    if best is not None:
        best_value, best_result = best
        best_key = tuple(best_result.objective_key or ())
        best_payload = best_result.payload

    endpoint_status = {
        "lower": {
            "value": low,
            "tested": existing_value(low) is not None,
            "feasible": bool(
                (observations.get(existing_value(low) or low) or DirectionalEvaluation(False)).feasible
            ),
        },
        "upper": {
            "value": high,
            "tested": existing_value(high) is not None,
            "feasible": bool(
                (observations.get(existing_value(high) or high) or DirectionalEvaluation(False)).feasible
            ),
        },
    }
    return {
        "schema_version": DIRECTIONAL_SEARCH_SCHEMA_VERSION,
        "variable_id": str(variable.get("id", "")),
        "variable_kind": str(variable.get("kind", "")),
        "closed": closure_reason is not None,
        "closure_reason": closure_reason,
        "closure_bound": closure_bound,
        "best_value": best_value,
        "best_objective_key": list(best_key) if best_key is not None else None,
        "best_payload": best_payload,
        "evaluation_count": len(observations),
        "evaluation_order": list(evaluation_order),
        "endpoint_status": endpoint_status,
        "trend_direction": trend_direction,
        "bracket": copy.deepcopy(bracket_payload),
        "budget_exhausted": bool(
            closure_reason is None and len(observations) >= max_evaluations
        ),
        "needs_feasibility_boundary": bool(
            closure_reason is None
            and (
                not endpoint_status["lower"]["feasible"]
                or not endpoint_status["upper"]["feasible"]
            )
        ),
        "observations": [
            {
                "value": float(value),
                "feasible": bool(result.feasible),
                "objective_key": (
                    list(result.objective_key) if result.objective_key is not None else None
                ),
                "rejection_reason": result.rejection_reason,
                "feasibility_kind": result.feasibility_kind,
            }
            for value, result in sorted(observations.items(), key=lambda item: item[0])
        ],
    }
