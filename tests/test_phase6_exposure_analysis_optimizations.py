"""Dependency-light regressions for Exposure Analysis Phase-6 optimizations."""
import unittest
from unittest.mock import patch

import numpy as np

from fieldworkbench.brain_analysis import (
    BrainRegionNode,
    BrainSampleSet,
    weighted_rms_coefficients,
)
from fieldworkbench.region_analysis import (
    _hierarchy_weighted_rms_moments,
    _region_spatial_rms_values,
    _region_spatial_rms_values_direct,
)


class Phase6ExposureAnalysisOptimizationTests(unittest.TestCase):
    def test_hierarchy_moments_match_direct_region_rescans(self):
        rng = np.random.default_rng(24601)
        label_ids = np.repeat(np.asarray([11, 12, 21, 22], dtype=np.int32), 17)
        point_count = len(label_ids)
        drive_count = 4
        fixed = rng.normal(size=(point_count, 3)) * 3e-6
        bases = rng.normal(size=(drive_count, point_count, 3)) * 2e-6
        weights = rng.uniform(0.25, 3.0, size=point_count)
        samples = BrainSampleSet(
            points_m=np.zeros((point_count, 3), dtype=float),
            label_ids=label_ids,
            weights_mm3=weights,
            source_indices=np.arange(point_count, dtype=np.int64),
            source_voxel_counts={11: 17, 12: 17, 21: 17, 22: 17},
            sampled_counts={11: 17, 12: 17, 21: 17, 22: 17},
            voxel_volume_mm3=1.0,
        )
        hierarchy = BrainRegionNode(
            "root",
            "Root",
            (11, 12, 21, 22),
            (
                BrainRegionNode("a", "A", (11, 12), (
                    BrainRegionNode("a.left", "Left", (11,)),
                    BrainRegionNode("a.right", "Right", (12,)),
                )),
                BrainRegionNode("b", "B", (21, 22), (
                    BrainRegionNode("b.left", "Left", (21,)),
                    BrainRegionNode("b.right", "Right", (22,)),
                )),
            ),
        )

        optimized = _hierarchy_weighted_rms_moments(fixed, bases, samples, hierarchy)
        for node in hierarchy.walk():
            mask = np.isin(label_ids, node.label_ids)
            expected = weighted_rms_coefficients(fixed[mask], bases[:, mask], weights[mask])
            actual = optimized[node.key]
            self.assertAlmostEqual(actual["constant"], expected[0], places=24)
            np.testing.assert_allclose(actual["linear"], expected[1], rtol=2e-13, atol=1e-24)
            np.testing.assert_allclose(actual["quadratic"], expected[2], rtol=2e-13, atol=1e-24)

    def test_repeated_current_states_are_evaluated_once_and_expanded_exactly(self):
        currents = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, -2.0],
                [1.0, -2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [3.0, 0.5],
                [3.0, 0.5],
            ],
            dtype=np.float64,
        )
        constant = 2e-12
        linear = np.asarray([0.2e-12, -0.1e-12], dtype=np.float64)
        quadratic = np.asarray([[1.5e-12, 0.25e-12], [0.25e-12, 0.8e-12]])
        expected = _region_spatial_rms_values_direct(constant, linear, quadratic, currents)

        original = _region_spatial_rms_values_direct
        evaluated_rows = []

        def recording_direct(c, l, q, values):
            evaluated_rows.append(len(np.asarray(values)))
            return original(c, l, q, values)

        with patch(
            "fieldworkbench.region_analysis._region_spatial_rms_values_direct",
            side_effect=recording_direct,
        ):
            actual = _region_spatial_rms_values(constant, linear, quadratic, currents)

        np.testing.assert_array_equal(actual, expected)
        # Three exact current vectors occur in four plateaus; non-adjacent zero
        # states are shared as well as adjacent repeats.
        self.assertEqual(sum(evaluated_rows), 3)


if __name__ == "__main__":
    unittest.main()
