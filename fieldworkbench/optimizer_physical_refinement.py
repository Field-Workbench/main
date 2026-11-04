"""Optimizer 2.3 capable-physics refinement helpers.

This module contains the policy-neutral pieces used by the physical-search stage:
selecting a model that can represent distributed winding effects, expanding the
required capability gaps into coupled structural coordinates, and sizing the
multi-basin seed set.  The actual evaluator lives in :mod:`studio_adapter`
because it owns target sampling, DRC/electrical gates, worker pools and the flat
candidate registry.

The reusable engine remains policy-neutral; :mod:`optimizer_effort_policy` now
defines which coordinates Fast/Balanced/Full are allowed or required to move and
which coverage evidence is sufficient for convergence.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


_PATH_KINDS = frozenset({
    "circular_diameter",
    "racetrack_end_diameter",
    "racetrack_straight_length",
    "square_length",
    "square_width",
})
_POSITION_KINDS = frozenset({"position_x", "position_y", "position_z"})
_ORIENTATION_KINDS = frozenset({
    "orientation_x", "orientation_y", "orientation_z",
    "assembly_orientation_x", "assembly_orientation_y", "assembly_orientation_z",
    "coil_orientation_x", "coil_orientation_y", "coil_orientation_z",
})


def optimizer_variable_families(variable: dict[str, Any]) -> set[str]:
    """Return semantic families a structural freedom belongs to.

    Families intentionally overlap.  They are used only to decide which scout-visible
    coordinates should be allowed to move with a capability-gap coordinate; they are
    not a replacement for numerical coupling tests.
    """
    kind = str(variable.get("kind", ""))
    if kind == "axial_winding_length":
        return {"winding_distribution"}
    if kind == "turns":
        return {"turns", "winding_distribution"}
    if kind == "pair_spacing":
        return {"pair_spacing", "position"}
    if kind in _POSITION_KINDS:
        return {"position"}
    if kind in _PATH_KINDS:
        return {"path_geometry"}
    if kind in _ORIENTATION_KINDS:
        return {"orientation"}
    return {kind} if kind else {"unknown"}


def optimizer_physical_active_variable_ids(
    structural_variables: Iterable[dict[str, Any]],
    freedom_coverage: Iterable[dict[str, Any]],
) -> list[str]:
    """Expand capability-gap freedoms to their explicitly declared couplings.

    Example: axial winding length requires a distributed-winding model and declares
    pair spacing/path geometry as coupling families.  The physical stage therefore
    moves those coordinates together rather than probing axial width at a frozen
    Centreline optimum.
    """
    variables = list(structural_variables)
    coverage = {
        str(item.get("id", "")): item
        for item in freedom_coverage
        if str(item.get("id", ""))
    }
    required_ids = {
        variable_id
        for variable_id, item in coverage.items()
        if bool((item.get("physical_search") or {}).get("required_for_complete_coverage", False))
    }
    if not required_ids:
        return []

    coupled_families: set[str] = set()
    for variable_id in required_ids:
        item = coverage.get(variable_id) or {}
        coupled_families.update(map(str, item.get("coupling_families", []) or []))

    active: list[str] = []
    for variable in variables:
        variable_id = str(variable.get("id", ""))
        families = optimizer_variable_families(variable)
        if variable_id in required_ids or families.intersection(coupled_families):
            active.append(variable_id)
    # Never lose a required coordinate because a future freedom lacks a family tag.
    for variable in variables:
        variable_id = str(variable.get("id", ""))
        if variable_id in required_ids and variable_id not in active:
            active.append(variable_id)
    return active


def optimizer_physical_refinement_model(
    requested_model: str,
    *,
    requires_distributed_winding: bool,
    has_segmented_participants: bool,
) -> tuple[str | None, str]:
    """Legacy compatibility resolver retained for older callers/tests.

    Optimizer 2.3 Phase 5 uses :mod:`optimizer_model_policy` for source-aware model
    resolution.  This helper keeps the earlier context-free behavior available to
    saved/internal callers that do not yet provide coil support metadata.
    """
    requested = str(requested_model or "auto").strip().lower()
    if not requires_distributed_winding:
        if requested in {"current_sheet", "bundled", "exact"}:
            return requested, "requested_model"
        if requested == "auto" and has_segmented_participants:
            return "bundled", "auto_segmented_capable_model"
        return None, "no_physical_capability_gap"
    if requested == "current_sheet":
        return "current_sheet", "requested_model"
    if requested == "bundled":
        return "bundled", "requested_model"
    if requested == "exact":
        return "bundled", "exact_search_uses_bundled_then_exact_authority"
    if requested == "auto":
        return ("bundled", "auto_segmented_capable_model") if has_segmented_participants else (
            "exact", "auto_capable_model"
        )
    return None, "requested_model_cannot_represent_distributed_winding"


def optimizer_physical_seed_limit(
    *,
    evaluation_budget: int,
    active_dimension_count: int,
    available_seed_count: int,
) -> int:
    """Choose a bounded multi-basin seed count without consuming the whole budget."""
    budget = max(0, int(evaluation_budget))
    available = max(0, int(available_seed_count))
    dimensions = max(1, int(active_dimension_count))
    if budget <= 0 or available <= 0:
        return 0
    # Leave at least ~12 evaluations per seed for a small Sobol cloud + local moves.
    budget_limit = max(1, budget // max(12, 3 * dimensions))
    dimension_hint = max(3, int(math.ceil(math.sqrt(dimensions + 1))))
    return max(1, min(8, available, budget_limit, dimension_hint + 2))
