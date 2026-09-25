from __future__ import annotations

import json
import unittest

from fieldworkbench.candidate_registry import OptimizerCandidateRegistry


class OptimizerCandidateRegistryTests(unittest.TestCase):
    def test_same_structure_accumulates_multiple_fidelities(self):
        registry = OptimizerCandidateRegistry([
            {"id": "x", "kind": "position_x", "step": 0.5, "base": 0.0},
            {"id": "spacing", "kind": "pair_spacing", "step": 0.5, "base": 171.0},
        ])
        scout = {
            "variable_values": {"x": 1.0, "spacing": 170.0, "current": 0.012},
            "setting_summary": "x=1; spacing=170",
            "metrics": {"snapshot_vector_rms_uT": 0.25},
            "currents_a": {"coil": 0.012},
            "search_index": 12,
            "feasible": True,
        }
        authoritative = {
            **scout,
            "variable_values": {"x": 1.0, "spacing": 170.0, "current": 0.0123},
            "metrics": {"snapshot_vector_rms_uT": 0.27},
            "search_index": 501,
        }
        first_id = registry.observe_candidate(
            scout, stage="Sobol", fidelity="scout", model="centreline", sample_count=80
        )
        second_id = registry.observe_candidate(
            authoritative,
            stage="Finalist vet",
            fidelity="authoritative",
            model="exact",
            sample_count=1791,
        )
        self.assertEqual(first_id, second_id)
        self.assertEqual(registry.payload()["candidate_count"], 1)
        self.assertEqual(registry.payload()["solution_count"], 1)
        self.assertEqual(registry.payload()["schema_version"], 3)
        record = registry.get(first_id)
        self.assertEqual(len(record["evaluations"]), 2)
        self.assertEqual(
            {item["fidelity"] for item in record["evaluations"]},
            {"scout", "authoritative"},
        )
        # Inner-solved current is evaluation metadata, never canonical solution state.
        self.assertNotIn("current", record["variable_values"])
        self.assertNotIn("current", record["structural_values"])
        by_fidelity = {item["fidelity"]: item for item in record["evaluations"]}
        self.assertAlmostEqual(by_fidelity["scout"]["solved_values"]["current"], 0.012)
        self.assertAlmostEqual(by_fidelity["authoritative"]["solved_values"]["current"], 0.0123)
        self.assertNotIn("current", record["setting_summary"].lower())
        preferred = registry.preferred_evaluations(record)
        self.assertEqual(len(preferred), 1)
        self.assertEqual(preferred[0]["fidelity"], "authoritative")

    def test_boundary_repair_is_flat_peer_with_provenance(self):
        registry = OptimizerCandidateRegistry([{"id": "x", "kind": "position_x", "step": 0.5, "base": 0.0}])
        source_id = registry.observe_candidate(
            {
                "variable_values": {"x": -10.0},
                "metrics": {"snapshot_vector_rms_uT": 0.1},
                "currents_a": {},
                "feasible": True,
            },
            stage="Scout",
            fidelity="scout",
        )
        registry.mark_drc(
            {"x": -10.0},
            {"feasible": False, "minimum_clearance_mm": -2.0, "reasons": ["collision"]},
        )
        repair = registry.ensure(
            {"x": -4.0},
            origin="drc_boundary_repair",
            stage="DRC",
            relation="drc_boundary_repair",
            related_solution_ids=[source_id],
            provenance_details={"from_solution_id": source_id, "clear_fraction": 0.75},
        )
        registry.mark_drc(
            {"x": -4.0},
            {"feasible": True, "minimum_clearance_mm": 0.2, "reasons": []},
            relation="drc_boundary_repair",
            related_solution_ids=[source_id],
        )
        registry.tag(repair["id"], "drc_boundary")

        source = registry.get(source_id)
        self.assertEqual(source["drc"]["status"], "blocked")
        self.assertEqual(repair["drc"]["status"], "clear")
        self.assertNotIn("parent_ids", repair)
        self.assertNotIn("child_ids", source)
        self.assertTrue(any(
            event.get("relation") == "drc_boundary_repair"
            and source_id in event.get("related_solution_ids", [])
            for event in repair["provenance"]
        ))
        self.assertEqual(registry.payload()["candidate_count"], 2)

    def test_periodic_angles_and_step_noise_merge_to_one_physical_solution(self):
        registry = OptimizerCandidateRegistry([
            {
                "id": "rz",
                "kind": "assembly_orientation_z",
                "step": 10.0,
                "base": 0.0,
                "min": -360.0,
                "max": 360.0,
            },
            {"id": "x", "kind": "position_x", "step": 0.5, "base": 0.0},
        ])
        a = registry.ensure({"rz": 0.0, "x": 1.0000000002})
        b = registry.ensure({"rz": 360.0, "x": 1.0})
        c = registry.ensure({"rz": -360.0, "x": 0.9999999998})
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(a["id"], c["id"])
        self.assertEqual(registry.payload()["candidate_count"], 1)
        self.assertEqual(a["structural_values"]["rz"], 0.0)
        self.assertEqual(a["structural_values"]["x"], 1.0)

    def test_import_merges_nested_flat_record_by_canonical_structure(self):
        scout = OptimizerCandidateRegistry([
            {"id": "x", "kind": "position_x", "step": 0.5, "base": 0.0},
            {"id": "rz", "kind": "assembly_orientation_z", "step": 10.0, "base": 0.0},
        ])
        old_id = scout.observe_candidate(
            {
                "variable_values": {"x": 0.0, "rz": 360.0},
                "metrics": {"snapshot_match_pct": 93.75},
                "currents_a": {"R1": 0.0124, "L1": 0.0124},
                "feasible": True,
            },
            stage="Canonical",
            fidelity="scout",
            sample_count=80,
        )
        parent = OptimizerCandidateRegistry([
            {"id": "x", "kind": "position_x", "step": 0.5, "base": 0.0},
            {"id": "rz", "kind": "assembly_orientation_z", "step": 10.0, "base": 0.0},
        ])
        parent.ensure({"x": 0.0, "rz": 0.0}, origin="source", stage="source")
        id_map = parent.import_payload(scout.payload(), stage_prefix="centreline scout")
        self.assertIn(old_id, id_map)
        record = parent.find_by_values({"x": 0.0, "rz": 0.0})
        self.assertIsNotNone(record)
        self.assertEqual(parent.payload()["candidate_count"], 1)
        self.assertEqual(len(record["evaluations"]), 1)
        self.assertEqual(record["evaluations"][0]["fidelity"], "scout")
        self.assertIn("centreline scout", record["evaluations"][0]["stage"])



    def test_targeted_observation_attaches_to_existing_canonical_solution(self):
        registry = OptimizerCandidateRegistry([
            {"id": "rz", "kind": "assembly_orientation_z", "step": 10.0, "base": 0.0},
            {"id": "spacing", "kind": "pair_spacing", "step": 0.5, "base": 171.0},
        ])
        record = registry.ensure(
            {"rz": 330.0, "spacing": 211.0},
            origin="drc_boundary_repair",
            stage="DRC",
        )
        # Simulate a nested numeric evaluator that reports a periodic alias and an
        # incomplete display dictionary. Identity must remain with the known DRC row.
        attached = registry.observe_candidate_for_id(
            record["id"],
            {
                "variable_values": {"rz": -30.0},
                "setting_summary": "rz=-30",
                "metrics": {"snapshot_vector_rms_uT": 0.42},
                "currents_a": {"R1": 0.012},
                "search_index": 900001,
                "feasible": True,
            },
            stage="Centreline DRC re-score",
            fidelity="scout",
            model="centreline",
            sample_count=80,
        )
        self.assertEqual(attached, record["id"])
        self.assertEqual(registry.payload()["candidate_count"], 1)
        self.assertEqual(len(record["evaluations"]), 1)
        self.assertEqual(record["evaluations"][0]["fidelity"], "scout")
        self.assertAlmostEqual(
            record["evaluations"][0]["metrics"]["snapshot_vector_rms_uT"], 0.42
        )

    def test_targeted_authoritative_observation_keeps_inner_current_per_fidelity(self):
        registry = OptimizerCandidateRegistry([
            {"id": "spacing", "kind": "pair_spacing", "label": "Pair spacing Group A", "unit": "mm", "step": 0.5, "base": 171.0},
            {"id": "rz", "kind": "assembly_orientation_z", "label": "ΔRZ Assembly rotation Group A", "unit": "°", "step": 10.0, "base": 0.0},
        ])
        record = registry.ensure({"spacing": 211.0, "rz": -90.0, "current_group_a": 0.023})
        registry.observe_candidate_for_id(
            record["id"],
            {
                "variable_values": {"spacing": 211.0, "rz": -90.0, "current_group_a": 0.0168},
                "metrics": {"snapshot_vector_rms_uT": 1.62},
                "currents_a": {"R1": 0.0168, "L1": 0.0168},
                "feasible": True,
            },
            stage="Finalist vet",
            fidelity="authoritative",
            model="auto",
            sample_count=1791,
        )
        self.assertEqual(registry.payload()["candidate_count"], 1)
        self.assertNotIn("current_group_a", record["variable_values"])
        self.assertEqual(record["evaluations"][0]["solved_values"]["current_group_a"], 0.0168)
        self.assertIn("Pair spacing Group A=211 mm", record["setting_summary"])

    def test_targeted_complete_observation_rejects_wrong_canonical_solution(self):
        registry = OptimizerCandidateRegistry([
            {"id": "spacing", "kind": "pair_spacing", "step": 0.5, "base": 171.0},
            {"id": "rz", "kind": "assembly_orientation_z", "step": 10.0, "base": 0.0},
        ])
        record = registry.ensure({"spacing": 203.5, "rz": -90.0})
        with self.assertRaisesRegex(ValueError, "does not match its canonical solution"):
            registry.observe_candidate_for_id(
                record["id"],
                {
                    "variable_values": {"spacing": 203.5, "rz": -80.0, "current_group_a": 16.1},
                    "metrics": {"snapshot_vector_rms_uT": 0.44},
                    "currents_a": {"R1": 0.0161, "L1": 0.0161},
                    "feasible": True,
                },
                stage="Finalist vet",
                fidelity="authoritative",
                model="auto",
                sample_count=1791,
            )
        self.assertEqual(record["evaluations"], [])

    def test_v1_tree_payload_imports_as_flat_provenance(self):
        registry = OptimizerCandidateRegistry(["x"])
        payload = {
            "schema_version": 1,
            "candidates": [
                {
                    "id": "old_parent",
                    "variable_values": {"x": 1.0},
                    "origins": [{"origin": "sobol", "stage": "Scout"}],
                    "evaluations": [],
                    "drc": {"status": "blocked", "minimum_clearance_mm": -1.0, "reasons": ["hit"]},
                    "parent_ids": [],
                    "child_ids": ["old_child"],
                    "tags": [],
                },
                {
                    "id": "old_child",
                    "variable_values": {"x": 2.0},
                    "origins": [{"origin": "drc_boundary_repair", "stage": "DRC"}],
                    "evaluations": [],
                    "drc": {"status": "clear", "minimum_clearance_mm": 0.1, "reasons": []},
                    "parent_ids": ["old_parent"],
                    "child_ids": [],
                    "tags": ["drc_boundary"],
                },
            ],
        }
        id_map = registry.import_payload(payload, stage_prefix="legacy scout")
        child = registry.get(id_map["old_child"])
        parent = registry.get(id_map["old_parent"])
        self.assertNotIn("parent_ids", child)
        self.assertNotIn("child_ids", parent)
        self.assertTrue(any(
            event.get("relation") == "derived_from"
            and id_map["old_parent"] in event.get("related_solution_ids", [])
            for event in child["provenance"]
        ))

    def test_lifecycle_counts_partition_master_solution_population(self):
        registry = OptimizerCandidateRegistry([
            {"id": "x", "kind": "position_x", "step": 1.0, "base": 0.0}
        ])

        active_id = registry.observe_candidate(
            {
                "variable_values": {"x": 0.0},
                "metrics": {"rms_target_error_uT": 1.0},
                "currents_a": {},
                "feasible": True,
            },
            stage="Scout",
            fidelity="scout",
        )
        retired_id = registry.observe_candidate(
            {
                "variable_values": {"x": 1.0},
                "metrics": {"rms_target_error_uT": 2.0},
                "currents_a": {},
                "feasible": True,
            },
            stage="Scout",
            fidelity="scout",
        )
        registry.set_retired(retired_id, "decimated")

        registry.ensure({"x": 2.0}, origin="sobol", stage="Scout")
        registry.mark_drc(
            {"x": 2.0},
            {"feasible": False, "minimum_clearance_mm": -0.5, "reasons": ["collision"]},
        )

        authoritative_id = registry.observe_candidate(
            {
                "variable_values": {"x": 3.0},
                "metrics": {"rms_target_error_uT": 0.5},
                "currents_a": {},
                "feasible": True,
            },
            stage="Finalist vet",
            fidelity="authoritative",
        )

        self.assertIsNotNone(active_id)
        self.assertIsNotNone(authoritative_id)
        counts = registry.lifecycle_counts()
        self.assertEqual(counts["solutions_discovered"], 4)
        self.assertEqual(counts["solutions_active"], 2)
        self.assertEqual(counts["solutions_retired"], 1)
        self.assertEqual(counts["solutions_blocked"], 1)
        self.assertEqual(counts["solutions_authoritative"], 1)
        self.assertEqual(
            counts["solutions_active"]
            + counts["solutions_retired"]
            + counts["solutions_blocked"],
            counts["solutions_discovered"],
        )
        self.assertEqual(registry.payload()["lifecycle_counts"], counts)

    def test_targeted_drc_update_keeps_canonical_solution_identity(self):
        registry = OptimizerCandidateRegistry([
            {"id": "x", "kind": "position_x", "step": 0.5, "base": 0.0},
        ])
        candidate_id = registry.ensure({"x": -18.5}, origin="scout", stage="Scout")["id"]
        updated_id = registry.mark_drc_for_id(
            candidate_id,
            {"feasible": True, "minimum_clearance_mm": 0.42, "reasons": []},
            stage="Authoritative",
            origin="accepted_candidate_drc",
        )
        self.assertEqual(updated_id, candidate_id)
        self.assertEqual(len(registry.records), 1)
        self.assertEqual(registry.get(candidate_id)["structural_values"]["x"], -18.5)
        self.assertEqual(registry.get(candidate_id)["drc"]["minimum_clearance_mm"], 0.42)

    def test_views_reference_stable_candidate_ids_and_lifecycle_is_derived(self):
        registry = OptimizerCandidateRegistry(["x"])
        candidate_id = registry.observe_candidate(
            {
                "variable_values": {"x": 0.0},
                "metrics": {"rms_target_error_uT": 1.0},
                "currents_a": {},
                "feasible": True,
            },
            stage="Source",
            fidelity="scout",
        )
        registry.tag(candidate_id, "source_baseline", "protected")
        registry.add_to_view("source_baseline", candidate_id)
        registry.add_to_view("source_baseline", candidate_id)
        registry.set_retired(candidate_id, "decimated_after_scout_ranking")
        payload = registry.payload()
        self.assertEqual(payload["views"]["source_baseline"], [candidate_id])
        self.assertIn("protected", payload["candidates"][0]["tags"])
        self.assertFalse(payload["candidates"][0]["lifecycle"]["active_frontier"])
        self.assertEqual(payload["candidates"][0]["preferred_fidelity"], "scout")
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
