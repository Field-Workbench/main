from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm as scipy_expm

from fieldworkbench.exposure_electrical import (
    ElectricalResolutionError,
    equivalent_load,
    resolve_drive_group,
    resolve_shared_source_groups,
)


def load(coil_id: str, resistance: float, inductance: float) -> dict:
    return {"coil_id": coil_id, "resistance_ohm": resistance, "inductance_h": inductance}


class ExposureElectricalTests(unittest.TestCase):

    def test_shared_voltage_long_point_delay_history_avoids_per_knot_matrix_exponentials(self):
        groups = [
            {
                "id": f"g{index}",
                "connection": "series",
                "loads": [
                    load(f"{index}a", 255.0, 0.155),
                    load(f"{index}b", 255.0, 0.155),
                ],
            }
            for index in range(3)
        ]
        segment_count = 2000
        point_delay = 0.0033
        starts = np.arange(segment_count, dtype=float) * point_delay
        stops = starts + point_delay
        segments = [
            (float(start), float(stop), 4.08, 0.0)
            for start, stop in zip(starts, stops, strict=True)
        ]
        sample_times = np.arange(0, segment_count + 1, 20, dtype=float) * point_delay
        with patch(
            "fieldworkbench.exposure_electrical.expm",
            side_effect=scipy_expm,
        ) as mocked_expm:
            result = resolve_shared_source_groups(
                [0.0, segment_count * point_delay],
                [4.08, 4.08],
                "voltage",
                groups=groups,
                sample_times_s=sample_times,
                source_segments=segments,
            )
        self.assertLess(
            mocked_expm.call_count,
            100,
            "Long voltage histories should not evaluate one general matrix exponential per source knot.",
        )
        self.assertTrue(np.isfinite(result["source_current_a"]).all())

    def test_shared_voltage_source_supports_parallel_series_strings(self):
        groups = [
            {"id": "g1", "connection": "series", "loads": [load("a", 2.0, 0.1), load("b", 2.0, 0.1)]},
            {"id": "g2", "connection": "series", "loads": [load("c", 2.0, 0.1), load("d", 2.0, 0.1)]},
            {"id": "g3", "connection": "series", "loads": [load("e", 2.0, 0.1), load("f", 2.0, 0.1)]},
        ]
        result = resolve_shared_source_groups(
            [0.0, 1.0], [10.0, 10.0], "voltage",
            groups=groups,
            output_resistance_ohm=0.0,
        )
        expected_branch = 2.5 * (1.0 - math.exp(-20.0))
        for first, second in (("a", "b"), ("c", "d"), ("e", "f")):
            self.assertAlmostEqual(result["coil_currents_a"][first][-1], expected_branch, places=9)
            np.testing.assert_allclose(
                result["coil_currents_a"][first],
                result["coil_currents_a"][second],
            )
        self.assertAlmostEqual(result["source_current_a"][-1], 3.0 * expected_branch, places=9)
        self.assertAlmostEqual(result["equivalent_resistance_ohm"], 4.0 / 3.0)
        self.assertAlmostEqual(result["equivalent_inductance_h"], 0.2 / 3.0)

    def test_shared_voltage_source_output_resistance_is_applied_once(self):
        groups = [
            {"id": f"g{index}", "connection": "series", "loads": [
                load(f"{index}a", 2.0, 0.1), load(f"{index}b", 2.0, 0.1)
            ]}
            for index in range(3)
        ]
        result = resolve_shared_source_groups(
            [0.0, 2.0], [10.0, 10.0], "voltage",
            groups=groups,
            output_resistance_ohm=1.0,
        )
        expected_total = 10.0 / (1.0 + 4.0 / 3.0)
        self.assertAlmostEqual(result["source_current_a"][-1], expected_total, places=9)
        self.assertAlmostEqual(
            result["load_terminal_voltage_v"][-1],
            10.0 - expected_total,
            places=9,
        )
        for index in range(3):
            self.assertAlmostEqual(
                result["coil_currents_a"][f"{index}a"][-1],
                expected_total / 3.0,
                places=9,
            )

    def test_shared_current_source_splits_total_across_drive_groups(self):
        groups = [
            {"id": "g1", "connection": "series", "loads": [load("a", 2.0, 1.0)]},
            {"id": "g2", "connection": "series", "loads": [load("b", 4.0, 2.0)]},
        ]
        result = resolve_shared_source_groups(
            [0.0, 1.0], [3.0, 3.0], "current", groups=groups,
        )
        summed = result["coil_currents_a"]["a"] + result["coil_currents_a"]["b"]
        np.testing.assert_allclose(summed, result["source_current_a"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(result["source_current_a"], [3.0, 3.0])

    def test_single_shared_source_group_matches_existing_group_solver(self):
        loads = [load("a", 2.0, 0.1), load("b", 3.0, 0.2)]
        expected = resolve_drive_group(
            [0.0, 0.5, 1.0], [8.0, 8.0, 8.0], "voltage",
            loads=loads,
            connection="series",
            output_resistance_ohm=1.5,
            sample_times_s=[0.0, 0.2, 0.8, 1.0],
        )
        actual = resolve_shared_source_groups(
            [0.0, 0.5, 1.0], [8.0, 8.0, 8.0], "voltage",
            groups=[{"id": "g", "connection": "series", "loads": loads}],
            output_resistance_ohm=1.5,
            sample_times_s=[0.0, 0.2, 0.8, 1.0],
        )
        np.testing.assert_allclose(actual["source_current_a"], expected["source_current_a"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(actual["load_terminal_voltage_v"], expected["load_terminal_voltage_v"], rtol=1e-11, atol=1e-11)
        for coil_id in ("a", "b"):
            np.testing.assert_allclose(actual["coil_currents_a"][coil_id], expected["coil_currents_a"][coil_id], rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(actual["coil_voltages_v"][coil_id], expected["coil_voltages_v"][coil_id], rtol=1e-11, atol=1e-11)

    def test_series_current_source_applies_same_current_to_every_coil(self):
        result = resolve_drive_group(
            [0.0, 1.0], [1.0, 1.0], "current",
            loads=[load("a", 2.0, 1.0), load("b", 3.0, 2.0)],
            connection="series",
        )
        np.testing.assert_allclose(result["coil_currents_a"]["a"], [1.0, 1.0])
        np.testing.assert_allclose(result["coil_currents_a"]["b"], [1.0, 1.0])
        np.testing.assert_allclose(result["source_current_a"], [1.0, 1.0])
        np.testing.assert_allclose(result["source_voltage_v"], [5.0, 5.0])
        self.assertAlmostEqual(result["equivalent_resistance_ohm"], 5.0)
        self.assertAlmostEqual(result["equivalent_inductance_h"], 3.0)

    def test_series_voltage_source_includes_source_output_resistance(self):
        result = resolve_drive_group(
            [0.0, 1.0], [10.0, 10.0], "voltage",
            loads=[load("a", 4.0, 1.0)],
            connection="series",
            output_resistance_ohm=1.0,
        )
        expected = 2.0 * (1.0 - math.exp(-5.0))
        self.assertAlmostEqual(result["source_current_a"][-1], expected, places=10)
        self.assertAlmostEqual(result["load_terminal_voltage_v"][-1], 10.0 - expected, places=10)
        self.assertAlmostEqual(result["source_output_resistance_ohm"], 1.0)

    def test_parallel_voltage_source_solves_each_rl_branch_and_common_terminal(self):
        result = resolve_drive_group(
            [0.0, 1.0], [10.0, 10.0], "voltage",
            loads=[load("a", 5.0, 1.0), load("b", 10.0, 2.0)],
            connection="parallel",
            output_resistance_ohm=0.0,
        )
        expected_a = 2.0 * (1.0 - math.exp(-5.0))
        expected_b = 1.0 * (1.0 - math.exp(-5.0))
        self.assertAlmostEqual(result["coil_currents_a"]["a"][-1], expected_a, places=10)
        self.assertAlmostEqual(result["coil_currents_a"]["b"][-1], expected_b, places=10)
        np.testing.assert_allclose(result["load_terminal_voltage_v"], [10.0, 10.0])
        self.assertAlmostEqual(result["equivalent_resistance_ohm"], 10.0 / 3.0)
        self.assertAlmostEqual(result["equivalent_inductance_h"], 2.0 / 3.0)

    def test_parallel_voltage_source_resistance_couples_branch_currents(self):
        result = resolve_drive_group(
            [0.0, 0.2, 1.0], [10.0, 10.0, 10.0], "voltage",
            loads=[load("a", 2.0, 0.1), load("b", 4.0, 0.2)],
            connection="parallel",
            output_resistance_ohm=5.0,
        )
        total = result["source_current_a"]
        np.testing.assert_allclose(
            result["load_terminal_voltage_v"],
            result["source_voltage_v"] - 5.0 * total,
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertTrue(np.all(result["load_terminal_voltage_v"] <= 10.0 + 1e-12))

    def test_parallel_current_source_conserves_total_current(self):
        result = resolve_drive_group(
            [0.0, 1.0], [1.0, 1.0], "current",
            loads=[load("a", 2.0, 1.0), load("b", 3.0, 2.0)],
            connection="parallel",
        )
        summed = result["coil_currents_a"]["a"] + result["coil_currents_a"]["b"]
        np.testing.assert_allclose(summed, result["source_current_a"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(result["source_current_a"], [1.0, 1.0])
        self.assertAlmostEqual(result["coil_currents_a"]["a"][0], 2.0 / 3.0)
        self.assertAlmostEqual(result["coil_currents_a"]["b"][0], 1.0 / 3.0)
        self.assertGreaterEqual(result["ideal_current_step_count"], 1)

    def test_voltage_resolution_accepts_initial_inductor_current(self):
        result = resolve_drive_group(
            [0.0, 1.0], [0.0, 0.0], "voltage",
            loads=[load("a", 2.0, 1.0)],
            connection="series",
            initial_currents_a={"a": 3.0},
        )
        self.assertAlmostEqual(result["coil_currents_a"]["a"][0], 3.0, places=12)
        self.assertAlmostEqual(
            result["coil_currents_a"]["a"][-1],
            3.0 * math.exp(-2.0),
            places=10,
        )

    def test_parallel_voltage_resolution_keeps_per_branch_initial_currents(self):
        result = resolve_drive_group(
            [0.0, 0.1], [0.0, 0.0], "voltage",
            loads=[load("a", 2.0, 1.0), load("b", 4.0, 2.0)],
            connection="parallel",
            initial_currents_a={"a": 1.0, "b": 2.0},
        )
        self.assertAlmostEqual(result["coil_currents_a"]["a"][0], 1.0, places=12)
        self.assertAlmostEqual(result["coil_currents_a"]["b"][0], 2.0, places=12)

    def test_requested_sample_order_is_preserved(self):
        result = resolve_drive_group(
            [0.0, 1.0], [2.0, 2.0], "current",
            loads=[load("a", 1.0, 0.1)],
            connection="series",
            sample_times_s=[1.0, 0.0, 0.5],
        )
        np.testing.assert_allclose(result["time_s"], [1.0, 0.0, 0.5])
        np.testing.assert_allclose(result["source_current_a"], [2.0, 2.0, 2.0])

    def test_load_validation_rejects_nonphysical_inductance(self):
        with self.assertRaises(ElectricalResolutionError):
            equivalent_load([load("a", 1.0, 0.0)], "series")


if __name__ == "__main__":
    unittest.main()
