"""Optimizer 2.3 effort-mode search/completion policy.

The capability layer answers *what a model can represent*.  This module answers
*what each user-facing effort mode promises to search*.

Fast intentionally remains reconnaissance-first and may leave physical-only
freedoms unresolved.  Balanced must actively search every freedom that the
Centreline scout cannot fully represent, together with its declared couplings.
Full reopens every structural freedom under capable physical physics and is not
allowed to report convergence until every selected freedom is either genuinely
varied at the required fidelity, analytically solved, or explicitly proven
redundant.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

from .optimizer_physical_refinement import optimizer_physical_active_variable_ids


_VALID_EFFORTS = {"fast", "balanced", "full"}


def _effort(value: str) -> str:
    effort = str(value or "full").strip().lower()
    if effort not in _VALID_EFFORTS:
        raise ValueError("search effort must be Fast, Balanced, or Full")
    return effort


def optimizer_effort_physical_scope(
    *,
    search_effort: str,
    structural_variables: Iterable[dict[str, Any]],
    freedom_coverage: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return the physical-search coordinate contract for one effort mode.

    ``active_variable_ids`` are allowed to move in the physical stage.
    ``full_span_variable_ids`` receive min/mid/max insurance and full-bound Sobol
    coverage rather than only a trust region around a scout basin.
    ``completion_required_variable_ids`` must show real physical variation before
    the effort mode may call its dimensional search complete.
    """
    effort = _effort(search_effort)
    variables = list(structural_variables)
    coverage = list(freedom_coverage)
    structural_ids = [str(item.get("id", "")) for item in variables if str(item.get("id", ""))]
    gap_ids = [
        str(item.get("id", ""))
        for item in coverage
        if str(item.get("id", "")) in set(structural_ids)
        and bool((item.get("physical_search") or {}).get("required_for_complete_coverage", False))
    ]

    if effort == "fast":
        active_ids: list[str] = []
        full_span_ids: list[str] = []
        completion_ids: list[str] = []
        contract = "reconnaissance_and_authoritative_verification"
    elif effort == "balanced":
        active_ids = optimizer_physical_active_variable_ids(variables, coverage)
        full_span_ids = list(gap_ids)
        # Optimizer 2.4 Phase 6: once a coupled coordinate is activated under
        # capable physics it participates in the same directional-correctness
        # floor as the capability-gap freedom that pulled it in.  Balanced may
        # still leave unrelated Centreline-visible coordinates at scout fidelity,
        # but it may not call a coupled physical family complete while one of its
        # active directions remains open.
        completion_ids = list(active_ids)
        contract = "targeted_capability_gap_physical_directional_closure"
    else:
        # Full deliberately reopens *every* structural coordinate.  The cheap map
        # supplies basins, not fixed answers.  All coordinates receive full-bound
        # physical exploration so coupled optima can move away from the scout model.
        active_ids = list(structural_ids)
        full_span_ids = list(structural_ids)
        completion_ids = list(structural_ids)
        contract = "complete_dimensional_physical_refinement"

    return {
        "search_effort": effort,
        "contract": contract,
        "capability_gap_variable_ids": list(gap_ids),
        "active_variable_ids": list(active_ids),
        "full_span_variable_ids": list(full_span_ids),
        "completion_required_variable_ids": list(completion_ids),
        "directional_closure_variable_ids": list(completion_ids),
        "allows_limited_physical_coverage": effort == "fast",
        "requires_capability_gap_completion": effort in {"balanced", "full"},
        "requires_all_structural_physical_completion": effort == "full",
    }


def optimizer_finalize_effort_coverage(
    *,
    search_effort: str,
    variables: Iterable[dict[str, Any]],
    freedom_coverage: Iterable[dict[str, Any]],
    physical_distinct_value_counts: dict[str, int] | None = None,
    physical_directional_closure: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Finalize per-freedom status and return the effort completion contract.

    This function deliberately judges *search coverage*, not whether the objective
    was good.  Authoritative finalist survival is checked separately by the parent
    optimizer before overall convergence is reported.
    """
    effort = _effort(search_effort)
    variables_list = list(variables)
    coverage = copy.deepcopy(list(freedom_coverage))
    by_id = {
        str(item.get("id", "")): item
        for item in coverage
        if str(item.get("id", ""))
    }
    counts = {
        str(key): max(0, int(value))
        for key, value in dict(physical_distinct_value_counts or {}).items()
    }
    structural_variables = [
        item for item in variables_list if str(item.get("kind", "")) != "current"
    ]
    scope = optimizer_effort_physical_scope(
        search_effort=effort,
        structural_variables=structural_variables,
        freedom_coverage=coverage,
    )
    completion_required = set(map(str, scope["completion_required_variable_ids"]))
    capability_gaps = set(map(str, scope["capability_gap_variable_ids"]))
    physical_directional_entries = {
        str(item.get("id", "")): item
        for item in list((physical_directional_closure or {}).get("entries", []) or [])
        if str(item.get("id", ""))
    }
    directional_contract_present = bool(physical_directional_entries)

    satisfied: list[str] = []
    unresolved: list[str] = []
    limited: list[str] = []
    analytically_solved: list[str] = []
    proven_redundant: list[str] = []

    for variable in variables_list:
        variable_id = str(variable.get("id", ""))
        if not variable_id:
            continue
        entry = by_id.get(variable_id)
        if entry is None:
            unresolved.append(variable_id)
            continue
        kind = str(variable.get("kind", ""))
        physical = entry.setdefault("physical_search", {})
        dynamic = entry.setdefault("dynamic", {})
        observed_count = max(
            counts.get(variable_id, 0),
            int(physical.get("distinct_value_count", 0) or 0),
        )
        physical["distinct_value_count"] = int(observed_count)

        redundancy_proven = bool(
            dynamic.get("complete_redundancy_proven", False)
            or str(physical.get("status", "")) == "proven_redundant"
        )
        current_solved = bool(
            kind == "current"
            and (
                str((entry.get("scout") or {}).get("status", "")) == "inner_solved"
                or str(entry.get("completion_status", "")) == "analytically_solved"
            )
        )

        if current_solved:
            entry["completion_status"] = "analytically_solved"
            analytically_solved.append(variable_id)
            satisfied.append(variable_id)
            continue
        if redundancy_proven:
            entry["completion_status"] = "proven_redundant"
            proven_redundant.append(variable_id)
            satisfied.append(variable_id)
            continue

        physically_varied = observed_count >= 2
        directional_entry = physical_directional_entries.get(variable_id)
        directional_closed = bool(
            directional_entry is not None
            and bool((directional_entry.get("closure") or {}).get("closed", False))
        )
        if directional_entry is not None:
            physical["directional_closed"] = bool(directional_closed)
            physical["directional_closure_reason"] = (
                directional_entry.get("closure") or {}
            ).get("reason")
            physical["directional_closure_bound"] = (
                directional_entry.get("closure") or {}
            ).get("bound")
        if physically_varied:
            physical["status"] = "actively_varied"

        if effort == "fast":
            # Fast's promise is deliberately limited.  A physical-only coordinate
            # may be left at source/probe values and still satisfy *Fast* semantics,
            # but it must be labelled as limited rather than optimized.
            if variable_id in capability_gaps and not physically_varied:
                entry["completion_status"] = "limited_fast_search"
                limited.append(variable_id)
            elif physically_varied:
                entry["completion_status"] = "physically_explored"
            else:
                entry["completion_status"] = "scout_explored"
            satisfied.append(variable_id)
            continue

        if variable_id in completion_required:
            if physically_varied and (not directional_contract_present or directional_closed):
                entry["completion_status"] = "physically_directionally_closed"
                satisfied.append(variable_id)
            elif physically_varied and directional_contract_present and not directional_closed:
                entry["completion_status"] = (
                    "incomplete_full_physical_directional_closure"
                    if effort == "full"
                    else "incomplete_balanced_physical_directional_closure"
                )
                unresolved.append(variable_id)
            else:
                entry["completion_status"] = (
                    "incomplete_full_physical_coverage"
                    if effort == "full"
                    else "incomplete_balanced_physical_coverage"
                )
                unresolved.append(variable_id)
            continue

        # Balanced freedoms that Centreline fully represented do not have to be
        # reopened physically.  If they happened to move during coupled refinement,
        # preserve that stronger evidence.
        if physically_varied:
            entry["completion_status"] = "physically_explored"
        else:
            scout_status = str((entry.get("scout") or {}).get("status", ""))
            if scout_status.startswith("actively_varied"):
                entry["completion_status"] = "scout_explored"
            else:
                entry["completion_status"] = "incomplete_balanced_search"
                unresolved.append(variable_id)
                continue
        satisfied.append(variable_id)

    # Any variable missing from coverage is an invariant failure for Balanced/Full;
    # Fast records it as limited rather than pretending it was optimized.
    all_ids = [str(item.get("id", "")) for item in variables_list if str(item.get("id", ""))]
    missing_ids = [variable_id for variable_id in all_ids if variable_id not in by_id]
    if effort == "fast":
        limited.extend(missing_ids)
        satisfied.extend(missing_ids)
    else:
        unresolved.extend(missing_ids)

    unresolved = list(dict.fromkeys(unresolved))
    satisfied = [
        value for value in dict.fromkeys(satisfied)
        if value not in set(unresolved)
    ]
    limited = list(dict.fromkeys(limited))
    complete = not unresolved

    if effort == "fast":
        message = (
            "Fast reconnaissance contract satisfied; physical-only freedoms may remain limited and are not claimed as fully optimized."
        )
    elif complete and effort == "balanced":
        message = (
            "Balanced hybrid-search contract satisfied: every active capable-physics coupling family was physically varied and directionally closed."
        )
    elif complete:
        message = (
            "Full dimensional-search contract satisfied: every selected structural freedom was physically varied and directionally closed under capable physics, while current freedoms were analytically solved or an explicit redundancy proof applied."
        )
    elif effort == "balanced":
        message = (
            "Balanced hybrid-search contract is incomplete because one or more active capable-physics directions were not both physically varied and directionally closed."
        )
    else:
        message = (
            "Full dimensional-search contract is incomplete because one or more selected structural freedoms remain physically unvaried or directionally open, or lack an analytic/redundancy resolution."
        )

    summary = {
        "search_effort": effort,
        "contract": str(scope["contract"]),
        "complete": bool(complete),
        "required_physical_variable_ids": list(scope["completion_required_variable_ids"]),
        "active_physical_variable_ids": list(scope["active_variable_ids"]),
        "full_span_physical_variable_ids": list(scope["full_span_variable_ids"]),
        "capability_gap_variable_ids": list(scope["capability_gap_variable_ids"]),
        "satisfied_variable_ids": satisfied,
        "unresolved_variable_ids": unresolved,
        "limited_variable_ids": limited,
        "analytically_solved_variable_ids": list(dict.fromkeys(analytically_solved)),
        "proven_redundant_variable_ids": list(dict.fromkeys(proven_redundant)),
        "physical_distinct_value_counts": counts,
        "physical_directional_contract_present": bool(directional_contract_present),
        "physical_directionally_closed_variable_ids": [
            variable_id
            for variable_id, entry in physical_directional_entries.items()
            if bool((entry.get("closure") or {}).get("closed", False))
        ],
        "message": message,
    }
    return coverage, summary
