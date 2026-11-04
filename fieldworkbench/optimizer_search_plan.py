"""Optimizer 2.4 search-stage and protected budget planning.

The optimizer has three different jobs that intentionally live at different
fidelities:

* reconnaissance: cheap/global landscape discovery,
* physical refinement: capable-physics movement inside retained basins,
* authoritative validation: final requested-model verification.

Historically Optimizer 2.2 exposed one ``budget`` value in the parent search
while the cheap scout had a separate nested budget.  That made the word budget
mean different things in different branches and encouraged the final verifier
to stand in for a missing physical-search stage.  This module gives each job an
explicit, JSON-friendly plan.  Optimizer 2.4 Phase 7 separates mandatory
correctness work from discretionary exploration so optional Sobol/atlas work
can never consume the capacity reserved for bounds, directional closure,
coupled revisits, or feasibility-boundary recovery.
"""
from __future__ import annotations

from typing import Any, Iterable


_VALID_EFFORTS = {"fast", "balanced", "full"}


def _effort(value: str) -> str:
    effort = str(value or "full").strip().lower()
    if effort not in _VALID_EFFORTS:
        raise ValueError("search effort must be Fast, Balanced, or Full")
    return effort



def _partition_stage_budget(
    total: int,
    *,
    minimum_correctness: int,
    preferred_correctness: int,
) -> tuple[int, int]:
    """Return ``(correctness_reserve, discretionary_allowance)``.

    The reserve is protected capacity for mandatory search-correctness work.
    Discretionary atlas/Sobol/adaptive exploration may consume only the
    remainder.  The total ceiling remains a hard safety ceiling.
    """
    total_i = max(0, int(total))
    if total_i <= 0:
        return 0, 0
    minimum = max(0, min(total_i, int(minimum_correctness)))
    preferred = max(minimum, min(total_i, int(preferred_correctness)))
    return int(preferred), int(max(0, total_i - preferred))


def optimizer_reconnaissance_budget_split(
    *,
    search_effort: str,
    scout_coordinate_count: int,
    total_ceiling: int,
) -> dict[str, int]:
    """Protected reconnaissance budget for Optimizer 2.4.

    Endpoint insurance and directional/coupled closure are correctness work;
    broad relation/Sobol/adaptive probing is discretionary.  The reserve scales
    with dimensionality but deliberately leaves exploration room in ordinary
    searches.
    """
    effort = _effort(search_effort)
    dims = max(0, int(scout_coordinate_count))
    total = max(0, int(total_ceiling))
    if total <= 0 or dims <= 0:
        return {"correctness_reserve": 0, "discretionary_allowance": total}
    per_axis = {"fast": 40, "balanced": 36, "full": 32}[effort]
    minimum = 2 * dims + 8
    preferred = max(minimum, per_axis * dims + 24)
    # Never reserve the entire lane by default: a skeptic/global challenge is
    # still valuable after the correctness floor is paid.
    preferred = min(preferred, max(minimum, int(round(0.55 * total))))
    correctness, discretionary = _partition_stage_budget(
        total, minimum_correctness=minimum, preferred_correctness=preferred
    )
    return {
        "correctness_reserve": int(correctness),
        "discretionary_allowance": int(discretionary),
    }


def optimizer_physical_budget_split(
    *,
    search_effort: str,
    active_physical_freedom_count: int,
    total_ceiling: int,
) -> dict[str, int]:
    """Protected capable-physics budget for directional completion.

    One anchor plus bounded/coupled directional closure is mandatory. Extra
    basin anchors and the physical Sobol skeptic are discretionary.
    """
    effort = _effort(search_effort)
    active = max(0, int(active_physical_freedom_count))
    total = max(0, int(total_ceiling))
    if total <= 0 or active <= 0:
        return {"correctness_reserve": 0, "discretionary_allowance": total}
    per_axis = {"fast": 0, "balanced": 56, "full": 64}[effort]
    minimum = 1 + 2 * active
    preferred = max(minimum, 1 + per_axis * active)
    # Physical correctness is the main point of Balanced/Full, so protect a
    # larger share here than in the cheap reconnaissance lane.
    preferred = min(preferred, max(minimum, int(round(0.72 * total))))
    correctness, discretionary = _partition_stage_budget(
        total, minimum_correctness=minimum, preferred_correctness=preferred
    )
    return {
        "correctness_reserve": int(correctness),
        "discretionary_allowance": int(discretionary),
    }

def optimizer_authoritative_validation_ceiling(
    *,
    search_effort: str,
    finalist_count: int,
    explicit_evaluation_budget: int | None = None,
    max_evaluations: int | None = None,
) -> tuple[int, str]:
    """Return the parent/requested-model evaluation ceiling and its provenance.

    ``evaluation_budget`` remains a compatibility escape hatch for saved and
    programmatic jobs.  In automatic Optimizer 2.4 runs it is interpreted as an
    authoritative/legacy exact ceiling, never as the reconnaissance budget.
    """
    effort = _effort(search_effort)
    if explicit_evaluation_budget is not None:
        return int(explicit_evaluation_budget), "explicit_evaluation_budget"

    requested_finalists = max(1, int(finalist_count))
    if effort == "fast":
        default_cap = min(8, max(6, requested_finalists))
    elif effort == "balanced":
        default_cap = min(14, max(10, requested_finalists + 4))
    else:
        default_cap = min(24, max(14, 2 * requested_finalists + 2))
    if max_evaluations is not None:
        return int(max_evaluations), "max_evaluations_override"
    return int(default_cap), "effort_default"


def optimizer_reconnaissance_budget(
    *,
    search_effort: str,
    scout_coordinate_count: int,
    override: int | None = None,
) -> tuple[int, int, str]:
    """Return ``(ceiling, floor, provenance)`` for the cheap scout map."""
    effort = _effort(search_effort)
    dimensions = max(0, int(scout_coordinate_count))
    if dimensions <= 1:
        default_budget = {"fast": 96, "balanced": 128, "full": 192}[effort]
        floor = 64
    elif dimensions == 2:
        default_budget = {"fast": 320, "balanced": 640, "full": 1280}[effort]
        floor = 192
    else:
        default_budget = min(12288, max(4096, 768 * max(4, dimensions)))
        if effort == "fast":
            default_budget = min(1024, max(640, 96 * max(4, dimensions)))
            floor = 384
        elif effort == "balanced":
            default_budget = min(4096, max(2560, 192 * max(4, dimensions)))
            floor = 1536
        else:
            floor = 512
    requested = default_budget if override is None else int(override)
    budget = max(int(floor), min(16384, int(requested)))
    return int(budget), int(floor), "override" if override is not None else "effort_default"


def optimizer_physical_refinement_reserve(
    *,
    search_effort: str,
    structural_freedom_count: int,
    required_physical_freedom_count: int,
    override: int | None = None,
) -> tuple[int, str]:
    """Return the distinct budget for capable-physics basin refinement.

    The physical-search stage spends this allowance independently of both cheap
    reconnaissance and authoritative verification. Effort-specific completion
    guarantees are enforced by ``optimizer_effort_policy``.
    """
    effort = _effort(search_effort)
    structural = max(0, int(structural_freedom_count))
    required = max(0, int(required_physical_freedom_count))
    if override is not None:
        return max(0, min(50000, int(override))), "override"
    if structural <= 0:
        return 0, "no_structural_freedoms"
    if effort == "fast":
        # Fast is intentionally verification-oriented; a distinct zero reserve
        # makes that contract explicit in stage accounting.
        return 0, "effort_default"
    if effort == "balanced":
        # Balanced only owes a physical stage when the Centreline scout has a
        # capability gap. Fully represented searches can proceed scout → authority.
        if required <= 0:
            return 0, "no_capability_gap"
        active_hint = required
        return min(1536, max(192, 96 * active_hint)), "effort_default"
    # Full will eventually reopen every structural freedom under capable physics.
    # Reserve enough room for a multi-basin adaptive search while keeping a hard
    # internal ceiling until Phase 3/4 benchmark data tunes the policy.
    return min(8192, max(768, 256 * max(1, structural))), "effort_default"


def optimizer_search_stage_plan(
    *,
    search_effort: str,
    finalist_count: int,
    requested_model: str,
    structural_variable_ids: Iterable[str],
    freedom_coverage: Iterable[dict[str, Any]],
    scout_coordinate_count: int,
    explicit_evaluation_budget: int | None = None,
    max_evaluations: int | None = None,
    reconnaissance_override: int | None = None,
    physical_refinement_override: int | None = None,
    model_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the three-stage Optimizer 2.4 execution/budget contract."""
    effort = _effort(search_effort)
    structural_ids = {str(value) for value in structural_variable_ids}
    coverage = list(freedom_coverage)
    physical_required_ids = [
        str(item.get("id", ""))
        for item in coverage
        if str(item.get("id", "")) in structural_ids
        and bool((item.get("physical_search") or {}).get("required_for_complete_coverage", False))
    ]
    reconnaissance_budget, reconnaissance_floor, reconnaissance_source = (
        optimizer_reconnaissance_budget(
            search_effort=effort,
            scout_coordinate_count=int(scout_coordinate_count),
            override=reconnaissance_override,
        )
    )
    physical_budget, physical_source = optimizer_physical_refinement_reserve(
        search_effort=effort,
        structural_freedom_count=len(structural_ids),
        required_physical_freedom_count=len(physical_required_ids),
        override=physical_refinement_override,
    )
    if effort == "fast":
        effort_physical_ids: list[str] = []
    elif effort == "balanced":
        effort_physical_ids = list(physical_required_ids)
    else:
        # Full reopens every structural coordinate under capable physics; the
        # capability-gap list remains separately available for model selection.
        effort_physical_ids = sorted(structural_ids)
    reconnaissance_split = optimizer_reconnaissance_budget_split(
        search_effort=effort,
        scout_coordinate_count=int(scout_coordinate_count),
        total_ceiling=int(reconnaissance_budget),
    )
    # Balanced capability gaps may activate an entire coupled structural family
    # during Phase-6 physical closure.  Budget the correctness reserve against
    # the full structural count as a safe upper hint whenever such a gap exists;
    # runtime accounting will still spend only on actually activated coordinates.
    physical_active_budget_hint = (
        len(structural_ids)
        if effort == "balanced" and physical_required_ids
        else len(effort_physical_ids)
    )
    physical_split = optimizer_physical_budget_split(
        search_effort=effort,
        active_physical_freedom_count=physical_active_budget_hint,
        total_ceiling=int(physical_budget),
    )
    authoritative_budget, authoritative_source = optimizer_authoritative_validation_ceiling(
        search_effort=effort,
        finalist_count=int(finalist_count),
        explicit_evaluation_budget=explicit_evaluation_budget,
        max_evaluations=max_evaluations,
    )
    policy = dict(model_policy or {})
    physical_model = policy.get("physical_search_model")
    physical_reason = policy.get("physical_search_reason")
    authoritative_model = str(policy.get("authoritative_model", requested_model))
    authoritative_mode = str(policy.get("authoritative_mode", "fixed_requested_model"))
    return {
        "schema_version": 3,
        "architecture": "reconnaissance_physical_refinement_authoritative_validation",
        "search_effort": effort,
        "requested_model": str(requested_model),
        "structural_freedom_count": int(len(structural_ids)),
        "scout_coordinate_count": int(scout_coordinate_count),
        "physical_search_required_variable_ids": physical_required_ids,
        "effort_physical_completion_variable_ids": effort_physical_ids,
        "stages": {
            "reconnaissance": {
                "role": "global_landscape_discovery",
                "model": "centreline",
                "evaluation_ceiling": int(reconnaissance_budget),
                "evaluation_floor": int(reconnaissance_floor),
                "correctness_reserve": int(reconnaissance_split["correctness_reserve"]),
                "discretionary_allowance": int(reconnaissance_split["discretionary_allowance"]),
                "budget_contract": "protected_correctness_plus_discretionary",
                "correctness_scope": [
                    "legal_bound_endpoints",
                    "directional_bracketing",
                    "coupled_revisits",
                    "feasibility_boundary_recovery",
                ],
                "discretionary_scope": [
                    "relation_atlas_extras",
                    "global_skeptic",
                    "adaptive_basin_exploration",
                ],
                "budget_source": reconnaissance_source,
                "execution_status": "planned",
            },
            "physical_refinement": {
                "role": "capable_physics_multidimensional_refinement",
                "model": (str(physical_model) if physical_model is not None else None),
                "model_reason": (str(physical_reason) if physical_reason is not None else None),
                "evaluation_ceiling": int(physical_budget),
                "correctness_reserve": int(physical_split["correctness_reserve"]),
                "discretionary_allowance": int(physical_split["discretionary_allowance"]),
                "budget_contract": "protected_correctness_plus_discretionary",
                "correctness_scope": [
                    "mandatory_physical_anchor",
                    "directional_bracketing",
                    "coupled_revisits",
                    "feasibility_boundary_recovery",
                ],
                "discretionary_scope": [
                    "additional_basin_anchors",
                    "global_skeptic",
                    "alternate_basin_challenge",
                ],
                "budget_source": physical_source,
                "execution_status": "planned",
                "required_by_observability": bool(physical_required_ids),
                "required_by_effort": bool(effort_physical_ids),
                "completion_variable_ids": list(effort_physical_ids),
                "effort_scope": (
                    "none" if effort == "fast"
                    else "capability_gaps" if effort == "balanced"
                    else "all_structural_freedoms"
                ),
            },
            "authoritative_validation": {
                "role": "requested_model_final_verification",
                "model": authoritative_model,
                "model_mode": authoritative_mode,
                "evaluation_ceiling": int(authoritative_budget),
                "budget_source": authoritative_source,
                "execution_status": "planned",
            },
        },
        "notes": [
            "Stage budgets are independent ceilings; reconnaissance work does not consume authoritative validation capacity.",
            "Reconnaissance and capable-physics lanes protect a correctness reserve that discretionary atlas/Sobol/adaptive work may not consume.",
            "If mandatory correctness work exhausts its protected safety reserve, the stage must report incomplete rather than treating discretionary exploration as equivalent evidence.",
            "Fast is reconnaissance/verification oriented; Balanced must physically close Centreline capability gaps; Full reopens every structural freedom under capable physics.",
            "The capable-physics refinement stage may move structural coordinates independently of the Centreline scout.",
        ],
    }


def optimizer_stage_budget(plan: dict[str, Any], stage: str, default: int = 0) -> int:
    """Read one stage ceiling from a JSON-friendly stage plan."""
    try:
        return int(((plan.get("stages") or {}).get(str(stage)) or {}).get("evaluation_ceiling", default))
    except (TypeError, ValueError):
        return int(default)


def optimizer_stage_budget_component(
    plan: dict[str, Any],
    stage: str,
    component: str,
    default: int = 0,
) -> int:
    """Read a protected stage-budget component.

    ``component`` is normally ``correctness_reserve`` or
    ``discretionary_allowance``.  Keeping this helper beside
    :func:`optimizer_stage_budget` lets later runtime phases consume the split
    without duplicating JSON traversal.
    """
    try:
        return int(
            ((plan.get("stages") or {}).get(str(stage)) or {}).get(
                str(component), default
            )
        )
    except (TypeError, ValueError):
        return int(default)
