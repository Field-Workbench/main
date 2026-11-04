from __future__ import annotations

import copy
from pathlib import Path
import unittest

from fieldworkbench.candidate_registry import OptimizerCandidateRegistry
from fieldworkbench.optimizer_capabilities import (
    optimizer_apply_source_capability_context,
    optimizer_freedom_coverage,
    optimizer_freedom_descriptor,
    optimizer_model_visibility,
    optimizer_validate_requested_model_reports,
)
from fieldworkbench.optimizer_effort_policy import (
    optimizer_effort_physical_scope,
    optimizer_finalize_effort_coverage,
)
from fieldworkbench.optimizer_model_policy import optimizer_resolve_model_policy
from fieldworkbench.optimizer_physical_refinement import (
    optimizer_physical_active_variable_ids,
    optimizer_physical_seed_limit,
)
from fieldworkbench.optimizer_search_plan import optimizer_search_stage_plan
from fieldworkbench.optimizer_authoritative_policy import (
    optimizer_authoritative_queue_ids,
    optimizer_should_expand_authoritative_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def racetrack_variables() -> list[dict]:
    return [
        {
            "id": "current_A",
            "kind": "current",
            "label": "Current Group A",
            "coil_ids": ["coil_1", "coil_2"],
            "min": 0.001,
            "max": 0.4,
            "step": 0.001,
            "base": 0.06379,
        },
        {
            "id": "axial_A",
            "kind": "axial_winding_length",
            "label": "Axial winding length Group A",
            "coil_ids": ["coil_1", "coil_2"],
            "min": 11.2,
            "max": 65.0,
            "step": 0.5,
            "base": 19.5,
        },
        {
            "id": "end_A",
            "kind": "racetrack_end_diameter",
            "label": "Racetrack corner diameter Group A",
            "coil_ids": ["coil_1", "coil_2"],
            "min": 70.0,
            "max": 85.0,
            "step": 0.5,
            "base": 75.0,
        },
        {
            "id": "straight_A",
            "kind": "racetrack_straight_length",
            "label": "Racetrack straight length Group A",
            "coil_ids": ["coil_1", "coil_2"],
            "min": 70.0,
            "max": 85.0,
            "step": 0.5,
            "base": 75.0,
        },
        {
            "id": "spacing_A",
            "kind": "pair_spacing",
            "label": "Pair spacing Group A",
            "coil_ids": ["coil_1", "coil_2"],
            "min": 10.0,
            "max": 500.0,
            "step": 0.5,
            "base": 67.0,
        },
    ]


def polyline_support() -> list[dict]:
    return [
        {
            "id": coil_id,
            "label": coil_id,
            "centreline_supported": True,
            "current_sheet_supported": True,
            "bundled_supported": True,
            "exact_supported": True,
            "model_reasons": {},
        }
        for coil_id in ("coil_1", "coil_2")
    ]


def circle_support() -> list[dict]:
    return [
        {
            "id": "circle_1",
            "label": "Circular coil 1",
            "centreline_supported": True,
            "current_sheet_supported": False,
            "bundled_supported": True,
            "exact_supported": True,
            "model_reasons": {"current_sheet": "Current Sheet is not available for Circle sources."},
        }
    ]


def coverage_with_scout_statuses(variables: list[dict], requested_model: str = "current_sheet") -> list[dict]:
    coverage = optimizer_freedom_coverage(
        variables,
        requested_model=requested_model,
        scout_model="centreline",
    )
    by_id = {item["id"]: item for item in coverage}
    for variable in variables:
        entry = by_id[variable["id"]]
        if variable["kind"] == "current":
            entry["scout"]["status"] = "inner_solved"
            entry["completion_status"] = "analytically_solved"
        elif entry["scout"]["visibility"] == "none":
            entry["scout"]["status"] = "omitted"
            entry["scout"]["omission_reason"] = "centreline_model_invisible"
            entry["completion_status"] = "requires_physical_search"
        else:
            entry["scout"]["status"] = (
                "actively_varied_partial"
                if entry["scout"]["visibility"] == "partial"
                else "actively_varied"
            )
    return coverage


class Optimizer23CapabilityTests(unittest.TestCase):
    def test_axial_width_is_invisible_to_centreline_but_path_and_spacing_are_visible(self):
        variables = racetrack_variables()
        coverage = optimizer_freedom_coverage(
            variables,
            requested_model="current_sheet",
            scout_model="centreline",
        )
        by_id = {item["id"]: item for item in coverage}
        self.assertEqual(by_id["axial_A"]["scout"]["visibility"], "none")
        self.assertTrue(by_id["axial_A"]["physical_search"]["required_for_complete_coverage"])
        self.assertEqual(by_id["end_A"]["scout"]["visibility"], "full")
        self.assertEqual(by_id["straight_A"]["scout"]["visibility"], "full")
        self.assertEqual(by_id["spacing_A"]["scout"]["visibility"], "full")

    def test_turns_are_only_partially_visible_to_centreline(self):
        descriptor = optimizer_freedom_descriptor({
            "id": "turns_A",
            "kind": "turns",
            "coil_ids": ["coil_1"],
        })
        self.assertEqual(optimizer_model_visibility("centreline", descriptor), "partial")
        self.assertEqual(optimizer_model_visibility("bundled", descriptor), "full")

    def test_unknown_future_freedom_is_conservatively_physical(self):
        descriptor = optimizer_freedom_descriptor({"id": "future", "kind": "future_geometry"})
        self.assertFalse(descriptor["known_kind"])
        self.assertEqual(optimizer_model_visibility("centreline", descriptor), "partial")
        self.assertEqual(optimizer_model_visibility("exact", descriptor), "full")


class Optimizer23EffortPolicyTests(unittest.TestCase):
    def setUp(self):
        self.variables = racetrack_variables()
        self.structural = [item for item in self.variables if item["kind"] != "current"]
        self.coverage = coverage_with_scout_statuses(self.variables)

    def test_balanced_expands_axial_gap_to_spacing_and_racetrack_geometry(self):
        active = optimizer_physical_active_variable_ids(self.structural, self.coverage)
        self.assertEqual(set(active), {"axial_A", "end_A", "straight_A", "spacing_A"})
        scope = optimizer_effort_physical_scope(
            search_effort="balanced",
            structural_variables=self.structural,
            freedom_coverage=self.coverage,
        )
        self.assertEqual(set(scope["active_variable_ids"]), set(active))
        self.assertEqual(scope["full_span_variable_ids"], ["axial_A"])
        self.assertEqual(set(scope["completion_required_variable_ids"]), set(active))

    def test_full_reopens_every_structural_coordinate_across_full_bounds(self):
        scope = optimizer_effort_physical_scope(
            search_effort="full",
            structural_variables=self.structural,
            freedom_coverage=self.coverage,
        )
        expected = {"axial_A", "end_A", "straight_A", "spacing_A"}
        self.assertEqual(set(scope["active_variable_ids"]), expected)
        self.assertEqual(set(scope["full_span_variable_ids"]), expected)
        self.assertEqual(set(scope["completion_required_variable_ids"]), expected)

    def test_fast_marks_axial_limited_without_failing_effort_contract(self):
        finalized, completion = optimizer_finalize_effort_coverage(
            search_effort="fast",
            variables=self.variables,
            freedom_coverage=self.coverage,
            physical_distinct_value_counts={},
        )
        by_id = {item["id"]: item for item in finalized}
        self.assertTrue(completion["complete"])
        self.assertIn("axial_A", completion["limited_variable_ids"])
        self.assertEqual(by_id["axial_A"]["completion_status"], "limited_fast_search")
        self.assertEqual(by_id["current_A"]["completion_status"], "analytically_solved")

    def test_balanced_requires_real_variation_across_the_activated_coupling_family(self):
        _, incomplete = optimizer_finalize_effort_coverage(
            search_effort="balanced",
            variables=self.variables,
            freedom_coverage=self.coverage,
            physical_distinct_value_counts={"axial_A": 3},
        )
        self.assertFalse(incomplete["complete"])
        self.assertEqual(
            set(incomplete["unresolved_variable_ids"]),
            {"end_A", "straight_A", "spacing_A"},
        )

        counts = {"axial_A": 3, "end_A": 3, "straight_A": 3, "spacing_A": 3}
        _, complete = optimizer_finalize_effort_coverage(
            search_effort="balanced",
            variables=self.variables,
            freedom_coverage=self.coverage,
            physical_distinct_value_counts=counts,
        )
        self.assertTrue(complete["complete"])

    def test_full_refuses_completion_when_spacing_never_physically_moves(self):
        counts = {
            "axial_A": 3,
            "end_A": 4,
            "straight_A": 4,
            "spacing_A": 1,
        }
        _, incomplete = optimizer_finalize_effort_coverage(
            search_effort="full",
            variables=self.variables,
            freedom_coverage=self.coverage,
            physical_distinct_value_counts=counts,
        )
        self.assertFalse(incomplete["complete"])
        self.assertIn("spacing_A", incomplete["unresolved_variable_ids"])

        counts["spacing_A"] = 5
        _, complete = optimizer_finalize_effort_coverage(
            search_effort="full",
            variables=self.variables,
            freedom_coverage=self.coverage,
            physical_distinct_value_counts=counts,
        )
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["unresolved_variable_ids"], [])


class Optimizer23ModelTelemetryTests(unittest.TestCase):
    def test_basis_solver_reports_from_temporary_adapters_satisfy_strict_model_guard(self):
        zero_report = {
            "requested": "current_sheet",
            "coils": [
                {"id": "coil_1", "actual": "current_sheet"},
                {"id": "coil_2", "actual": "current_sheet"},
            ],
        }
        basis_report = copy.deepcopy(zero_report)
        result = optimizer_validate_requested_model_reports(
            "current_sheet",
            ["coil_1", "coil_2"],
            [zero_report, basis_report],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["observed_report_count"], 2)

    def test_any_actual_fallback_in_basis_evidence_is_rejected(self):
        good = {
            "requested": "current_sheet",
            "coils": [
                {"id": "coil_1", "actual": "current_sheet"},
                {"id": "coil_2", "actual": "current_sheet"},
            ],
        }
        fallback = copy.deepcopy(good)
        fallback["coils"][1] = {
            "id": "coil_2",
            "actual": "centreline",
            "fallback_reason": "physical representation unavailable",
        }
        result = optimizer_validate_requested_model_reports(
            "current_sheet", ["coil_1", "coil_2"], [good, fallback]
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["failures"][0]["id"], "coil_2")
        self.assertEqual(result["failures"][0]["code"], "requested_model_fell_back")

    def test_missing_basis_evidence_remains_a_hard_failure(self):
        result = optimizer_validate_requested_model_reports(
            "current_sheet", ["coil_1", "coil_2"], []
        )
        self.assertFalse(result["ok"])
        self.assertEqual(
            {item["code"] for item in result["failures"]},
            {"missing_model_report"},
        )



class Optimizer23ModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.variables = racetrack_variables()
        self.structural = [item for item in self.variables if item["kind"] != "current"]
        self.coverage = coverage_with_scout_statuses(self.variables)

    def resolve(self, model: str, effort: str = "full") -> dict:
        return optimizer_resolve_model_policy(
            requested_model=model,
            search_effort=effort,
            variables=self.variables,
            structural_variables=self.structural,
            freedom_coverage=self.coverage,
            source_support=polyline_support(),
            participant_coil_ids=["coil_1", "coil_2"],
        )

    def test_racetrack_current_sheet_full_uses_current_sheet_for_search_and_authority(self):
        policy = self.resolve("current_sheet", "full")
        self.assertTrue(policy["compatible"])
        self.assertEqual(policy["physical_search_model"], "current_sheet")
        self.assertEqual(policy["authoritative_model"], "current_sheet")

    def test_exact_uses_bundled_search_surrogate_then_exact_authority(self):
        policy = self.resolve("exact", "full")
        self.assertTrue(policy["compatible"])
        self.assertEqual(policy["physical_search_model"], "bundled")
        self.assertEqual(policy["authoritative_model"], "exact")

    def test_auto_uses_stable_bundled_floor_for_distributed_winding_gap(self):
        policy = self.resolve("auto", "balanced")
        self.assertTrue(policy["compatible"])
        self.assertEqual(policy["physical_search_model"], "bundled")
        self.assertEqual(policy["authoritative_model"], "auto")
        self.assertEqual(policy["authoritative_mode"], "dynamic_auto")

    def test_centreline_balanced_blocks_axial_winding_gap_but_fast_may_leave_it_limited(self):
        balanced = self.resolve("centreline", "balanced")
        self.assertFalse(balanced["compatible"])
        self.assertIn(
            "centreline_cannot_close_capability_gap",
            {item["code"] for item in balanced["blocking_issues"]},
        )
        fast = self.resolve("centreline", "fast")
        self.assertTrue(fast["compatible"])
        self.assertIsNone(fast["physical_search_model"])


    def test_turns_without_enabled_physical_winding_do_not_force_distributed_auto_floor(self):
        variables = [
            {
                "id": "current_A",
                "kind": "current",
                "coil_ids": ["circle_1", "circle_2"],
                "min": 40.0,
                "max": 100.0,
                "base": 70.0,
            },
            {
                "id": "turns_A",
                "kind": "turns",
                "coil_ids": ["circle_1", "circle_2"],
                "min": 250.0,
                "max": 350.0,
                "base": 300.0,
            },
            {
                "id": "position_x_A",
                "kind": "position_x",
                "coil_ids": ["circle_1", "circle_2"],
                "min": -5.0,
                "max": 5.0,
                "base": 0.0,
            },
        ]
        structural = [item for item in variables if item["kind"] != "current"]
        coverage = optimizer_freedom_coverage(
            variables, requested_model="auto", scout_model="centreline"
        )
        support = [
            {
                "id": coil_id,
                "label": coil_id,
                "centreline_supported": True,
                "current_sheet_supported": False,
                "bundled_supported": False,
                "exact_supported": False,
                "physical_winding_available": False,
                "model_reasons": {
                    "bundled": "No physical winding specification is attached to this coil.",
                    "exact": "No physical winding specification is attached to this coil.",
                },
            }
            for coil_id in ("circle_1", "circle_2")
        ]
        contextual = optimizer_apply_source_capability_context(
            variables, coverage, support, requested_model="auto"
        )
        turns = next(item for item in contextual if item["id"] == "turns_A")
        self.assertEqual(turns["scout"]["visibility"], "full")
        self.assertFalse(turns["physical_search"]["required_for_complete_coverage"])
        self.assertEqual(turns["physical_search"]["minimum_capability"], "centreline_geometry")
        self.assertFalse(turns["dynamic"]["source_winding_distribution_active"])

        policy = optimizer_resolve_model_policy(
            requested_model="auto",
            search_effort="full",
            variables=variables,
            structural_variables=structural,
            freedom_coverage=contextual,
            source_support=support,
            participant_coil_ids=["circle_1", "circle_2"],
        )
        self.assertTrue(policy["compatible"])
        self.assertEqual(policy["physical_search_model"], "centreline")
        self.assertEqual(policy["distributed_winding_gap_variable_ids"], [])

    def test_turns_with_enabled_physical_winding_keep_distributed_capability_requirement(self):
        variables = [
            {
                "id": "turns_A",
                "kind": "turns",
                "coil_ids": ["circle_1"],
                "min": 250.0,
                "max": 350.0,
                "base": 300.0,
            }
        ]
        coverage = optimizer_freedom_coverage(
            variables, requested_model="auto", scout_model="centreline"
        )
        support = [{
            "id": "circle_1",
            "label": "circle_1",
            "centreline_supported": True,
            "current_sheet_supported": False,
            "bundled_supported": True,
            "exact_supported": True,
            "physical_winding_available": True,
            "model_reasons": {},
        }]
        contextual = optimizer_apply_source_capability_context(
            variables, coverage, support, requested_model="auto"
        )
        turns = contextual[0]
        self.assertEqual(turns["scout"]["visibility"], "partial")
        self.assertTrue(turns["physical_search"]["required_for_complete_coverage"])
        self.assertEqual(turns["physical_search"]["minimum_capability"], "distributed_winding")
        self.assertTrue(turns["dynamic"]["source_winding_distribution_active"])

    def test_current_sheet_rejects_circle_participant_instead_of_falling_back(self):
        variables = [
            {
                "id": "diameter_A",
                "kind": "circular_diameter",
                "coil_ids": ["circle_1"],
                "min": 40.0,
                "max": 100.0,
                "step": 0.5,
                "base": 70.0,
            }
        ]
        coverage = coverage_with_scout_statuses(variables, requested_model="current_sheet")
        policy = optimizer_resolve_model_policy(
            requested_model="current_sheet",
            search_effort="full",
            variables=variables,
            structural_variables=variables,
            freedom_coverage=coverage,
            source_support=circle_support(),
            participant_coil_ids=["circle_1"],
        )
        self.assertFalse(policy["compatible"])
        self.assertIn(
            "requested_authoritative_model_unavailable",
            {item["code"] for item in policy["blocking_issues"]},
        )


class Optimizer23SearchPlanTests(unittest.TestCase):
    def setUp(self):
        self.variables = racetrack_variables()
        self.structural = [item for item in self.variables if item["kind"] != "current"]
        self.coverage = coverage_with_scout_statuses(self.variables)

    def make_plan(self, effort: str) -> dict:
        model_policy = optimizer_resolve_model_policy(
            requested_model="current_sheet",
            search_effort=effort,
            variables=self.variables,
            structural_variables=self.structural,
            freedom_coverage=self.coverage,
            source_support=polyline_support(),
            participant_coil_ids=["coil_1", "coil_2"],
        )
        return optimizer_search_stage_plan(
            search_effort=effort,
            finalist_count=1,
            requested_model="current_sheet",
            structural_variable_ids=[item["id"] for item in self.structural],
            freedom_coverage=self.coverage,
            scout_coordinate_count=3,
            model_policy=model_policy,
        )

    def test_stage_budgets_are_independent_and_effort_specific(self):
        fast = self.make_plan("fast")
        balanced = self.make_plan("balanced")
        full = self.make_plan("full")
        self.assertEqual(fast["stages"]["physical_refinement"]["evaluation_ceiling"], 0)
        self.assertGreater(balanced["stages"]["physical_refinement"]["evaluation_ceiling"], 0)
        self.assertGreater(full["stages"]["physical_refinement"]["evaluation_ceiling"], balanced["stages"]["physical_refinement"]["evaluation_ceiling"])
        self.assertEqual(full["stages"]["physical_refinement"]["model"], "current_sheet")
        self.assertEqual(full["stages"]["authoritative_validation"]["model"], "current_sheet")
        self.assertEqual(set(full["effort_physical_completion_variable_ids"]), {item["id"] for item in self.structural})
        self.assertEqual(set(balanced["effort_physical_completion_variable_ids"]), {"axial_A"})

    def test_physical_seed_limit_preserves_multiple_basins_without_spending_entire_budget(self):
        self.assertEqual(optimizer_physical_seed_limit(evaluation_budget=0, active_dimension_count=4, available_seed_count=12), 0)
        seed_count = optimizer_physical_seed_limit(evaluation_budget=1024, active_dimension_count=4, available_seed_count=12)
        self.assertGreaterEqual(seed_count, 3)
        self.assertLessEqual(seed_count, 8)
        self.assertLess(seed_count, 1024)


class Optimizer23RegistryAndIntegrationTests(unittest.TestCase):
    def test_physical_refinement_fidelity_outranks_fixed_promotion_but_not_authority(self):
        registry = OptimizerCandidateRegistry([
            {"id": "spacing", "kind": "pair_spacing", "step": 0.5, "base": 67.0}
        ])
        base = {
            "variable_values": {"spacing": 73.0, "current": 0.068},
            "feasible": True,
            "metrics": {"rms_target_error_uT": 8.0},
            "currents_a": {"A": 0.068},
        }
        registry.observe_candidate(base, stage="promotion", fidelity="physical_promotion", model="current_sheet", sample_count=48)
        refined = copy.deepcopy(base)
        refined["metrics"] = {"rms_target_error_uT": 6.0}
        registry.observe_candidate(refined, stage="refinement", fidelity="physical_refinement", model="current_sheet", sample_count=48)
        authoritative = copy.deepcopy(base)
        authoritative["metrics"] = {"rms_target_error_uT": 6.2}
        registry.observe_candidate(authoritative, stage="authority", fidelity="authoritative", model="current_sheet", sample_count=48)
        record = registry.find_by_values({"spacing": 73.0})
        self.assertIsNotNone(record)
        ranks = {item["fidelity"]: item["fidelity_rank"] for item in record["evaluations"]}
        self.assertGreater(ranks["physical_refinement"], ranks["physical_promotion"])
        self.assertGreater(ranks["authoritative"], ranks["physical_refinement"])

    def test_studio_adapter_contains_full_bound_physical_search_and_effort_completion_gate(self):
        source = (ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        self.assertIn('effort_physical_scope.get("full_span_variable_ids"', source)
        self.assertIn("Physical refinement mandatory anchor", source)
        self.assertIn("Physical directional refinement", source)
        self.assertIn("Physical refinement global skeptic", source)
        self.assertIn("optimizer_directional_search_with_feasibility_recovery(", source)
        self.assertIn("physical_correctness_reserve", source)
        self.assertIn("physical_discretionary_allowance", source)
        self.assertIn("optimizer_finalize_effort_coverage(", source)
        self.assertIn("search coverage incomplete", source)
        self.assertIn('"_optimizer_require_requested_model"', source)
        self.assertIn('"coil_modeling_reports": copy.deepcopy(model_reports)', source)
        self.assertIn('reports=list(current_solution.get("coil_modeling_reports", []) or [])', source)

    def test_phase6_reporting_and_help_keep_effort_contract_visible(self):
        adapter = (ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        help_text = (ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")
        dialogs = (ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        self.assertIn("SEARCH / FIDELITY CONTRACT", adapter)
        self.assertIn("FREEDOM COVERAGE", adapter)
        self.assertIn("DIRECTIONAL CLOSURE — CENTRELINE RECONNAISSANCE", adapter)
        self.assertIn("protected budget correctness", adapter)
        self.assertIn("directional correctness floor", help_text)
        self.assertIn("Protected correctness budgets", help_text)
        self.assertIn("Mandatory bounds + directional closure", dialogs)
        self.assertIn("Bounded + coupled directional closure", dialogs)


class Optimizer23AuthoritativeQueueTests(unittest.TestCase):
    def test_physical_ranking_leader_stays_ahead_of_source_insurance(self):
        # Regression for 0.14.0.302: the physically promoted 75%-coverage
        # candidate was ranked ahead of the 66.7%-coverage source baseline,
        # then the protected-source pass incorrectly moved the baseline to slot 0.
        ordered = optimizer_authoritative_queue_ids(
            ["physical_75pct", "source_66pct", "alternate"],
            ["source_66pct"],
            preserve_ranked_leader=True,
        )
        self.assertEqual(ordered, ["physical_75pct", "source_66pct", "alternate"])

    def test_protected_rows_remain_front_loaded_without_higher_fidelity_ranking(self):
        ordered = optimizer_authoritative_queue_ids(
            ["scout_best", "source", "boundary", "other"],
            ["source", "boundary"],
            preserve_ranked_leader=False,
        )
        self.assertEqual(ordered, ["source", "boundary", "scout_best", "other"])

    def test_decisive_progressive_vet_expands_if_ranked_leader_did_not_survive(self):
        self.assertTrue(
            optimizer_should_expand_authoritative_validation(
                validation_count=1,
                validation_cap=3,
                has_survivor=True,  # insurance candidate survived
                intermediate_enabled=True,
                intermediate_decisive=True,
                ranked_leader_required=True,
                ranked_leader_survived=False,
            )
        )

    def test_decisive_progressive_vet_can_stop_after_ranked_leader_survives(self):
        self.assertFalse(
            optimizer_should_expand_authoritative_validation(
                validation_count=1,
                validation_cap=3,
                has_survivor=True,
                intermediate_enabled=True,
                intermediate_decisive=True,
                ranked_leader_required=True,
                ranked_leader_survived=True,
            )
        )

    def test_studio_adapter_uses_rank_preserving_authoritative_policy(self):
        source = (ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        self.assertIn("optimizer_authoritative_queue_ids(", source)
        self.assertIn("preserve_ranked_leader=higher_fidelity_ranked_queue", source)
        self.assertIn("optimizer_should_expand_authoritative_validation(", source)
        self.assertIn("authoritative_ranked_leader_survived", source)


if __name__ == "__main__":
    unittest.main()
