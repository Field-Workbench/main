from __future__ import annotations

import copy
import threading
import unittest
from unittest import mock

import numpy as np

from fieldworkbench.optimizer import (
    GeneticSearchRunner,
    OptimizationCancelled,
    SearchCallbacks,
    VectorFieldStamp,
    available_optimizer_workers,
    default_optimizer_settings,
    field_statistics,
    filter_near_duplicates,
    fit_signed_currents,
    pareto_front,
    normalize_optimizer_worker_count,
    optimizer_benchmark_worker_counts,
    recommend_optimizer_worker_count,
    reported_cpu_description,
    reported_logical_cpu_count,
    validate_optimizer_settings,
)


def synthetic_candidate(value: float, genome: dict) -> dict:
    metrics = {
        "target_uT": 200.0,
        "rms_target_error_uT": float(value),
        "in_band_pct": 100.0,
        "total_mass_g": 100.0 + value,
        "coil_count": 1,
        "max_voltage_v": 1.0,
        "total_power_w": 0.1,
    }
    return {
        "coils": [
            {"diameter_mm": 100.0, "turns": 300, "awg": 30, "axial_width_mm": 8.0}
        ],
        "metrics": metrics,
        "fitness": float(value),
        "feasible": True,
        "rejection_reasons": [],
        "genome": copy.deepcopy(genome),
    }


class OptimizerMathTests(unittest.TestCase):
    def test_vector_field_stamp_translates_rotates_and_scales_full_vectors(self):
        axis = np.linspace(-2.0, 2.0, 9)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        # Linear local vector field makes trilinear interpolation exact.
        values = np.stack([x + 2.0 * y, y - z, 3.0 * z - x], axis=3)
        stamp = VectorFieldStamp((axis, axis, axis), values)

        rotation = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        position = np.asarray([4.0, -3.0, 2.0], dtype=float)
        local = np.asarray([[0.5, -0.25, 0.75]], dtype=float)
        world = local @ rotation.T + position
        expected_local = np.asarray([[0.0, -1.0, 1.75]], dtype=float)
        expected_world = expected_local @ rotation.T
        np.testing.assert_allclose(
            stamp.sample_world(
                world,
                position_m=position,
                rotation_matrix=rotation,
            ),
            expected_world,
            atol=1e-12,
        )

        # A uniformly doubled source queried at the doubled local coordinate
        # has half the same-current field magnitude.
        scaled_world = (2.0 * local) @ rotation.T + position
        np.testing.assert_allclose(
            stamp.sample_world(
                scaled_world,
                position_m=position,
                rotation_matrix=rotation,
                geometric_scale=2.0,
            ),
            0.5 * expected_world,
            atol=1e-12,
        )

    def test_defaults_validate_and_preserve_one_to_four_coil_scope(self):
        settings = validate_optimizer_settings(default_optimizer_settings())
        self.assertEqual((settings["coil_count_min"], settings["coil_count_max"]), (1, 4))
        self.assertEqual(settings["direction_mode"], "magnitude")
        self.assertEqual(settings["finite_pack_order"], 3)
        self.assertGreaterEqual(settings["worker_count"], 1)

    def test_reported_cpu_description_prefers_linux_model_name(self):
        cpuinfo = "processor : 0\nmodel name : Intel(R) Example CPU @ 3.40GHz\n"
        with (
            mock.patch("fieldworkbench.optimizer.platform.system", return_value="Linux"),
            mock.patch("builtins.open", mock.mock_open(read_data=cpuinfo)),
        ):
            self.assertEqual(
                reported_cpu_description(),
                "Intel(R) Example CPU @ 3.40GHz",
            )

    def test_worker_counts_allow_benchmark_oversubscription_and_recommend_near_fastest_smallest(self):
        with mock.patch("fieldworkbench.optimizer.os.cpu_count", return_value=4):
            self.assertEqual(reported_logical_cpu_count(), 4)
            self.assertEqual(available_optimizer_workers(), 32)
            self.assertEqual(optimizer_benchmark_worker_counts(), list(range(1, 33)))
            self.assertEqual(normalize_optimizer_worker_count(16), 16)
            self.assertEqual(normalize_optimizer_worker_count(32), 32)
            self.assertEqual(optimizer_benchmark_worker_counts(7), list(range(1, 8)))
            self.assertEqual(optimizer_benchmark_worker_counts(32), list(range(1, 33)))
            with self.assertRaisesRegex(ValueError, "between 1 and 32"):
                normalize_optimizer_worker_count(33)
        with mock.patch("fieldworkbench.optimizer.os.cpu_count", return_value=24):
            self.assertEqual(reported_logical_cpu_count(), 24)
            self.assertEqual(available_optimizer_workers(), 32)
            self.assertEqual(optimizer_benchmark_worker_counts(), list(range(1, 33)))
            self.assertEqual(normalize_optimizer_worker_count(24), 24)
        results = [
            {"workers": 1, "elapsed_seconds": 10.0},
            {"workers": 2, "elapsed_seconds": 5.3},
            {"workers": 3, "elapsed_seconds": 5.0},
            {"workers": 4, "elapsed_seconds": 5.1},
        ]
        self.assertEqual(recommend_optimizer_worker_count(results), 2)

    def test_invalid_reversed_bounds_are_rejected(self):
        settings = default_optimizer_settings()
        settings.update({"diameter_min_mm": 200.0, "diameter_max_mm": 50.0})
        with self.assertRaisesRegex(ValueError, "Minimum mean diameter"):
            validate_optimizer_settings(settings)

    def test_signed_current_fit_hits_uniform_magnitude_target(self):
        basis = np.zeros((2, 31, 3), dtype=float)
        basis[:, :, 2] = 100e-6
        currents, vectors, fit = fit_signed_currents(
            basis, 200.0, np.asarray([1.0, 1.0]), "magnitude"
        )
        stats = field_statistics(vectors, 200.0, 5.0)
        self.assertTrue(fit["success"])
        self.assertAlmostEqual(float(np.sum(currents)), 2.0, places=3)
        self.assertAlmostEqual(stats["mean_uT"], 200.0, places=2)
        self.assertEqual(stats["in_band_pct"], 100.0)

    def test_directional_fit_uses_requested_world_axis(self):
        basis = np.zeros((1, 9, 3), dtype=float)
        basis[0, :, 1] = -50e-6
        currents, vectors, _fit = fit_signed_currents(
            basis, 100.0, np.asarray([3.0]), "y"
        )
        self.assertLess(currents[0], 0.0)
        np.testing.assert_allclose(vectors[:, [0, 2]], 0.0, atol=1e-12)
        np.testing.assert_allclose(vectors[:, 1], 100e-6, rtol=1e-3)

    def test_pareto_removes_dominated_and_duplicate_entries(self):
        candidates = []
        for index, (error, mass) in enumerate(((1.0, 200.0), (2.0, 100.0), (3.0, 250.0))):
            item = synthetic_candidate(error, {"value": index})
            item["id"] = str(index)
            item["metrics"]["total_mass_g"] = mass
            candidates.append(item)
        self.assertEqual({item["id"] for item in pareto_front(candidates)}, {"0", "1"})
        duplicate = copy.deepcopy(candidates[0])
        duplicate["id"] = "duplicate"
        self.assertEqual(len(filter_near_duplicates([candidates[0], duplicate])), 1)


class GeneticRunnerTests(unittest.TestCase):
    @staticmethod
    def callbacks() -> SearchCallbacks:
        def create(rng):
            return {"value": int(rng.integers(0, 1000))}

        def mutate(genome, rng):
            return {"value": max(0, int(genome["value"] + rng.integers(-50, 51)))}

        def crossover(first, second, rng):
            return copy.deepcopy(first if rng.random() < 0.5 else second)

        def evaluate(genome):
            return synthetic_candidate(float(genome["value"]), genome)

        def validate(candidate):
            candidate["metrics"].update(
                {
                    "mean_uT": 200.0,
                    "min_uT": 200.0,
                    "max_uT": 200.0,
                    "coefficient_variation_pct": 0.0,
                    "target_low_uT": 190.0,
                    "target_high_uT": 210.0,
                }
            )
            return candidate

        return SearchCallbacks(
            create=create,
            mutate=mutate,
            crossover=crossover,
            evaluate=evaluate,
            validate=validate,
            signature=lambda genome: str(genome["value"]),
        )

    def test_seed_reproduces_exact_finalists(self):
        settings = default_optimizer_settings()
        settings.update({"population": 8, "generations": 2, "finalist_count": 3, "seed": 42})
        first = GeneticSearchRunner(settings, self.callbacks()).run()
        second = GeneticSearchRunner(settings, self.callbacks()).run()
        self.assertEqual(
            [item["genome_signature"] for item in first["finalists"]],
            [item["genome_signature"] for item in second["finalists"]],
        )

    def test_pre_set_cancel_event_stops_before_evaluation(self):
        settings = default_optimizer_settings()
        settings.update({"population": 8, "generations": 1})
        cancel = threading.Event()
        cancel.set()
        runner = GeneticSearchRunner(settings, self.callbacks(), cancel_event=cancel)
        with self.assertRaises(OptimizationCancelled):
            runner.run()


if __name__ == "__main__":
    unittest.main()
