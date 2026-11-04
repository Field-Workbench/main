from __future__ import annotations

from pathlib import Path
import time
import unittest

from fieldworkbench.optimizer_directional_closure import (
    CLOSURE_FEASIBILITY_BOUNDARY,
    CLOSURE_INTERIOR_TURNOVER,
    CLOSURE_LOWER_BOUND_BEST,
    CLOSURE_UPPER_BOUND_BEST,
    optimizer_directional_closure_entry,
    optimizer_directional_lattice_spec,
)
from fieldworkbench.optimizer_directional_coupling import (
    optimizer_directional_coupling_components,
)
from fieldworkbench.optimizer_directional_search import (
    DirectionalEvaluation,
    optimizer_bounded_directional_search,
)
from fieldworkbench.optimizer_feasibility_boundary import (
    optimizer_directional_search_with_feasibility_recovery,
)
from fieldworkbench.optimizer_search_plan import (
    optimizer_reconnaissance_budget_split,
    optimizer_physical_budget_split,
)


ROOT = Path(__file__).resolve().parents[1]


def axis_variable(*, low=70.0, high=85.0, step=0.5, base=75.0):
    return {
        "id": "x",
        "kind": "racetrack_end_diameter",
        "label": "Racetrack end diameter",
        "min": low,
        "max": high,
        "step": step,
        "base": base,
        "unit": "mm",
        "coil_ids": ["coil_1", "coil_2"],
    }


class Optimizer24DirectionalPrimitiveTests(unittest.TestCase):
    def test_monotonic_improvement_to_upper_bound_cannot_stop_at_interior_point(self):
        variable = axis_variable()
        entry = optimizer_directional_closure_entry(variable)
        result = optimizer_bounded_directional_search(
            variable,
            lambda x: (85.0 - x,),
            closure_entry=entry,
            max_evaluations=33,
        )
        self.assertTrue(result["closed"])
        self.assertEqual(result["closure_reason"], CLOSURE_UPPER_BOUND_BEST)
        self.assertEqual(result["best_value"], 85.0)
        self.assertIn(85.0, result["evaluation_order"])
        # Regression for the motivating 77 < 78 < 79 failure: if higher is
        # consistently better, 79 may be sampled but can never be final while
        # the legal 85-mm bound remains available.
        self.assertNotEqual(result["best_value"], 79.0)
        self.assertTrue(entry["endpoints"]["upper"]["tested"])

    def test_monotonic_improvement_to_lower_bound_closes_at_lower_bound(self):
        variable = axis_variable()
        result = optimizer_bounded_directional_search(
            variable,
            lambda x: (x - 70.0,),
            max_evaluations=33,
        )
        self.assertEqual(result["closure_reason"], CLOSURE_LOWER_BOUND_BEST)
        self.assertEqual(result["best_value"], 70.0)
        self.assertIn(70.0, result["evaluation_order"])
        self.assertIn(85.0, result["evaluation_order"])

    def test_interior_turnover_is_bracketed_at_increment_resolution(self):
        variable = axis_variable()
        result = optimizer_bounded_directional_search(
            variable,
            lambda x: (abs(x - 82.0),),
            max_evaluations=33,
        )
        self.assertTrue(result["closed"])
        self.assertEqual(result["closure_reason"], CLOSURE_INTERIOR_TURNOVER)
        self.assertEqual(result["best_value"], 82.0)
        bracket = result["bracket"]
        self.assertTrue(bracket["established"])
        self.assertLess(bracket["lower_value"], 82.0)
        self.assertGreater(bracket["upper_value"], 82.0)

    def test_increment_lattice_uses_reachable_effective_bounds(self):
        variable = axis_variable(low=11.2, high=65.0, step=0.5, base=19.5)
        lattice = optimizer_directional_lattice_spec(variable)
        self.assertEqual(lattice["effective_min"], 11.5)
        self.assertEqual(lattice["effective_max"], 65.0)
        result = optimizer_bounded_directional_search(
            variable,
            lambda x: (65.0 - x,),
            max_evaluations=33,
        )
        self.assertIn(11.5, result["evaluation_order"])
        self.assertIn(65.0, result["evaluation_order"])
        self.assertEqual(result["best_value"], 65.0)

    def test_drc_blocked_upper_bound_resolves_nearest_safe_lattice_boundary(self):
        variable = axis_variable(low=70.0, high=85.0, step=0.5, base=75.0)
        entry = optimizer_directional_closure_entry(variable)

        def evaluate(x):
            if x > 83.0:
                return DirectionalEvaluation(
                    False,
                    rejection_reason="coil clearance below required minimum",
                    feasibility_kind="drc",
                )
            return DirectionalEvaluation(True, objective_key=(83.0 - x,), payload={"x": x})

        result = optimizer_directional_search_with_feasibility_recovery(
            variable,
            evaluate,
            closure_entry=entry,
            max_evaluations=33,
            boundary_evaluation_reserve=12,
        )
        self.assertTrue(result["closed"])
        self.assertEqual(result["closure_reason"], CLOSURE_FEASIBILITY_BOUNDARY)
        self.assertEqual(result["best_value"], 83.0)
        upper = result["feasibility_boundaries"]["upper"]
        self.assertEqual(upper["feasible_value"], 83.0)
        self.assertEqual(upper["infeasible_value"], 83.5)
        self.assertEqual(upper["feasibility_kind"], "drc")

    def test_solver_failure_is_not_misreported_as_physical_boundary(self):
        variable = axis_variable(low=70.0, high=85.0, step=0.5, base=75.0)

        def evaluate(x):
            if x >= 85.0:
                return DirectionalEvaluation(
                    False,
                    rejection_reason="solver crashed",
                    feasibility_kind="evaluation",
                )
            return DirectionalEvaluation(True, objective_key=(85.0 - x,))

        result = optimizer_directional_search_with_feasibility_recovery(
            variable,
            evaluate,
            max_evaluations=20,
            boundary_evaluation_reserve=8,
        )
        self.assertFalse(result["closed"])
        self.assertEqual(result.get("feasibility_boundaries", {}), {})


class Optimizer24CouplingAndBudgetTests(unittest.TestCase):
    def test_racetrack_shape_and_spacing_form_one_coupled_component(self):
        variables = [
            {"id": "end", "kind": "racetrack_end_diameter"},
            {"id": "straight", "kind": "racetrack_straight_length"},
            {"id": "spacing", "kind": "pair_spacing"},
        ]
        plan = optimizer_directional_coupling_components(variables)
        coupled = [item for item in plan["components"] if item["coupled"]]
        self.assertEqual(len(coupled), 1)
        self.assertEqual(set(coupled[0]["variable_ids"]), {"end", "straight", "spacing"})

    def test_reconnaissance_budget_split_protects_correctness_capacity(self):
        split = optimizer_reconnaissance_budget_split(
            search_effort="fast", scout_coordinate_count=3, total_ceiling=640
        )
        self.assertGreater(split["correctness_reserve"], 0)
        self.assertGreater(split["discretionary_allowance"], 0)
        self.assertEqual(
            split["correctness_reserve"] + split["discretionary_allowance"], 640
        )

    def test_physical_budget_split_protects_nontrivial_closure_capacity(self):
        for effort in ("balanced", "full"):
            split = optimizer_physical_budget_split(
                search_effort=effort,
                active_physical_freedom_count=4,
                total_ceiling=1024,
            )
            self.assertGreaterEqual(split["correctness_reserve"], 1 + 32 * 4)
            self.assertGreater(split["discretionary_allowance"], 0)
            self.assertEqual(
                split["correctness_reserve"] + split["discretionary_allowance"], 1024
            )


class Optimizer24ReportingIntegrationTests(unittest.TestCase):
    def test_report_and_ui_expose_directional_closure_and_protected_budget(self):
        adapter = (ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        dialogs = (ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        help_text = (ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")
        self.assertIn("DIRECTIONAL CLOSURE — CENTRELINE RECONNAISSANCE", adapter)
        self.assertIn("DIRECTIONAL CLOSURE — CAPABLE PHYSICS", adapter)
        self.assertIn("protected budget correctness", adapter)
        self.assertIn("Mandatory bounds + directional closure", dialogs)
        self.assertIn("Discretionary global Sobol skeptic", dialogs)
        self.assertIn("Bounded + coupled directional closure", dialogs)
        self.assertIn("directional correctness floor", help_text)
        self.assertIn("Protected correctness budgets", help_text)

    def test_no_racetrack_dimension_whitelist_controls_fast_directional_axes(self):
        source = (ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        # Racetrack dimensions naturally appear elsewhere for geometry handling;
        # the integration guard is that directional selection comes from the
        # structural-variable/capability list and the generic engine is invoked.
        self.assertIn("optimizer_directional_search_with_feasibility_recovery(", source)
        self.assertIn("directional_columns", source)
        self.assertIn("optimizer_directional_coupling_components(", source)

    def test_physical_candidate_row_preserves_scout_invisible_structural_values(self):
        source = (ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        start = source.index("def parent_row_from_values(values: dict[str, Any])")
        end = source.index("def registry_best_scout_metrics", start)
        helper = source[start:end]
        self.assertIn("for variable in structural_variables:", helper)
        self.assertNotIn("for variable in scout_structural_variables:", helper)
        self.assertIn("def physical_candidate_row(candidate: dict[str, Any])", source)
        self.assertIn("return parent_row_from_values(", source[source.index("def physical_candidate_row(candidate: dict[str, Any])"):])

    def test_balanced_report_does_not_claim_centreline_directional_closure(self):
        source = (ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        self.assertIn(
            "Optimizer 2.4 Balanced policy: Centreline supplies broad basin reconnaissance",
            source,
        )
        self.assertIn(
            "Diagnostic only for Balanced: Centreline is used for basin reconnaissance",
            source,
        )
        self.assertNotIn(
            "Optimizer 2.4 Balanced policy: Centreline reconnaissance first satisfies bounded directional closure",
            source,
        )


class Optimizer24MicroBenchmarkTests(unittest.TestCase):
    def test_directional_primitive_closes_large_monotonic_range_with_logarithmic_work(self):
        variable = {
            "id": "spacing",
            "kind": "pair_spacing",
            "label": "Pair spacing",
            "min": 1.0,
            "max": 1000.0,
            "step": 1.0,
            "base": 67.0,
        }
        started = time.perf_counter()
        result = optimizer_bounded_directional_search(
            variable,
            lambda x: (1000.0 - x,),
            max_evaluations=40,
        )
        elapsed = time.perf_counter() - started
        self.assertTrue(result["closed"])
        self.assertEqual(result["best_value"], 1000.0)
        self.assertLessEqual(result["evaluation_count"], 20)
        # This is only a dependency-light algorithmic guard, not a hardware SLA.
        self.assertLess(elapsed, 0.25)


if __name__ == "__main__":
    unittest.main()
