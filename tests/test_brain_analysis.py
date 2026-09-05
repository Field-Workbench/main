"""LPBA40 hierarchy, sampling, and brain-region statistics tests."""

from __future__ import annotations

import math
import unittest

import numpy as np

from fieldworkbench.brain_analysis import (
    downsample_brain_samples,
    lpba40_frame_region_metrics,
    lpba40_playback_region_metrics,
    lpba40_region_hierarchy,
    lpba40_region_metrics,
    sample_lpba40_voxels,
    weighted_rms_coefficients,
    weighted_rms_from_coefficients,
)
from fieldworkbench.sri24_atlas import load_sri24_parcellation


class BrainAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.atlas = load_sri24_parcellation("lpba40")
        cls.samples = sample_lpba40_voxels(
            cls.atlas,
            maximum_samples_per_label=9,
        )

    def test_frame_region_metrics_precompute_every_rendered_frame(self) -> None:
        fixed = np.zeros((self.samples.point_count, 3), dtype=float)
        bases = np.zeros((1, self.samples.point_count, 3), dtype=float)
        bases[0, :, 0] = 1e-6
        currents = np.asarray([[0.0], [0.5], [1.0]], dtype=float)
        result = lpba40_frame_region_metrics(
            fixed,
            bases,
            currents,
            self.samples,
            lpba40_region_hierarchy(self.atlas),
        )
        values = np.asarray(result["values"], dtype=float)
        root_index = list(result["keys"]).index("whole_brain")
        self.assertEqual(values.shape[0], 3)
        self.assertAlmostEqual(values[0, root_index, 0], 0.0, places=9)
        self.assertAlmostEqual(values[1, root_index, 1], 0.5, places=5)
        self.assertAlmostEqual(values[2, root_index, 0], 1.0, places=5)
        self.assertAlmostEqual(values[2, root_index, 2], 1.0, places=5)
        self.assertAlmostEqual(values[2, root_index, 3], 1.0, places=5)

    def test_playback_summary_integrates_physical_time_and_captures_peak(self) -> None:
        fixed = np.zeros((self.samples.point_count, 3), dtype=float)
        bases = np.zeros((1, self.samples.point_count, 3), dtype=float)
        bases[0, :, 0] = 1e-6
        currents = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
        times = np.asarray([0.0, 1.0, 2.0], dtype=float)
        hierarchy = lpba40_region_hierarchy(self.atlas)
        frames = lpba40_frame_region_metrics(
            fixed, bases, currents, self.samples, hierarchy
        )
        summaries = lpba40_playback_region_metrics(
            fixed,
            bases,
            currents,
            times,
            frames,
            times,
            self.samples,
            hierarchy,
        )
        whole = summaries["whole_brain"]
        self.assertAlmostEqual(whole.playback_rms_uT, math.sqrt(1.5), places=6)
        self.assertAlmostEqual(whole.mean_uT, 1.0, places=6)
        self.assertAlmostEqual(whole.max_p95_uT, 2.0, places=6)
        self.assertAlmostEqual(whole.absolute_peak_uT, 2.0, places=6)
        self.assertAlmostEqual(whole.b_time_uT_s, 2.0, places=6)
        self.assertAlmostEqual(whole.b2_time_uT2_s, 3.0, places=6)
        self.assertAlmostEqual(whole.peak_time_s, 2.0, places=6)

    def test_workbench_hierarchy_is_a_complete_lpba40_partition(self) -> None:
        root = lpba40_region_hierarchy(self.atlas)
        self.assertEqual(root.name, "Whole brain")
        self.assertEqual(len(root.children), 8)
        self.assertEqual(
            [node.name for node in root.children],
            [
                "Frontal lobe",
                "Parietal lobe",
                "Occipital lobe",
                "Temporal lobe",
                "Insula",
                "Limbic structures",
                "Deep gray matter",
                "Cerebellum and brainstem",
            ],
        )
        anatomical_rows = [
            region
            for division in root.children
            for region in division.children
        ]
        assigned = [label for region in anatomical_rows for label in region.label_ids]
        expected = [value for value in self.atlas.present_label_ids if value != 0]
        self.assertEqual(sorted(assigned), sorted(expected))
        self.assertEqual(len(assigned), len(set(assigned)))
        temporal = next(node for node in root.children if node.key == "temporal")
        self.assertEqual(len(temporal.children), 5)
        superior = next(
            node for node in temporal.children if node.key == "superior_temporal_gyrus"
        )
        self.assertEqual([child.name for child in superior.children], ["Left", "Right"])

    def test_sampling_is_deterministic_volume_weighted_and_placed(self) -> None:
        samples = self.samples
        self.assertEqual(set(samples.sampled_counts.values()), {9})
        self.assertEqual(samples.point_count, 56 * 9)
        self.assertEqual(len(np.unique(samples.label_ids)), 56)
        self.assertAlmostEqual(float(np.sum(samples.weights_mm3)), 1_100_018.0)
        self.assertAlmostEqual(
            float(np.sum(samples.weights_mm3)),
            float(np.count_nonzero(self.atlas.volume)),
        )

        angle = np.deg2rad(90.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scale = np.asarray([1.2, 0.8, 1.1])
        position = np.asarray([0.01, -0.02, 0.03])
        placed = sample_lpba40_voxels(
            self.atlas,
            position_m=position,
            rotation=rotation,
            scale_xyz=scale,
            maximum_samples_per_label=9,
        )
        np.testing.assert_array_equal(placed.label_ids, samples.label_ids)
        np.testing.assert_allclose(
            placed.points_m,
            (samples.points_m * scale) @ rotation.T + position,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            float(np.sum(placed.weights_mm3)),
            float(np.sum(samples.weights_mm3)) * float(np.prod(scale)),
        )

    def test_display_downsampling_retains_every_label_and_total_volume(self) -> None:
        display = downsample_brain_samples(self.samples, maximum_points=112)
        self.assertEqual(display.point_count, 112)
        self.assertEqual(len(np.unique(display.label_ids)), 56)
        self.assertEqual(len(np.unique(display.source_indices)), display.point_count)
        self.assertTrue(np.all(display.source_indices < self.samples.point_count))
        self.assertAlmostEqual(
            float(np.sum(display.weights_mm3)),
            float(np.sum(self.samples.weights_mm3)),
        )

    def test_constant_field_produces_exact_weighted_region_metrics(self) -> None:
        vectors_t = np.tile(
            np.asarray([2.0, -3.0, 6.0], dtype=float) * 1e-6,
            (self.samples.point_count, 1),
        )
        metrics = lpba40_region_metrics(
            vectors_t,
            self.samples,
            lpba40_region_hierarchy(self.atlas),
        )
        self.assertGreater(len(metrics), 60)
        for result in metrics.values():
            self.assertAlmostEqual(result.rms_uT, 7.0)
            self.assertAlmostEqual(result.mean_uT, 7.0)
            self.assertAlmostEqual(result.p95_uT, 7.0)
            self.assertAlmostEqual(result.peak_uT, 7.0)
            self.assertAlmostEqual(result.uniformity_pct, 100.0)
            self.assertAlmostEqual(result.directional_consistency_pct, 100.0)
        self.assertAlmostEqual(metrics["whole_brain"].volume_cm3, 1100.018)
        self.assertEqual(
            metrics["whole_brain"].voxel_count,
            int(np.count_nonzero(self.atlas.volume)),
        )


    def test_peak_reports_maximum_retained_sample_magnitude(self) -> None:
        vectors_t = np.tile(
            np.asarray([1.0, 0.0, 0.0], dtype=float) * 1e-6,
            (self.samples.point_count, 1),
        )
        vectors_t[0] = np.asarray([0.0, 0.0, 17.0], dtype=float) * 1e-6
        metrics = lpba40_region_metrics(
            vectors_t,
            self.samples,
            lpba40_region_hierarchy(self.atlas),
        )
        self.assertAlmostEqual(metrics["whole_brain"].peak_uT, 17.0)

    def test_waveform_spatial_rms_quadratic_matches_direct_vectors(self) -> None:
        fixed = np.asarray(
            [[1.0, 2.0, -1.0], [0.0, 3.0, 2.0], [-2.0, 1.0, 4.0]]
        ) * 1e-6
        bases = np.asarray(
            [
                [[2.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 2.0, -2.0]],
                [[0.0, -1.0, 2.0], [3.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            ]
        ) * 1e-6
        weights = np.asarray([1.0, 4.0, 2.0])
        currents = np.asarray([[0.0, 0.0], [1.5, -0.25], [-2.0, 3.0]])
        coefficients = weighted_rms_coefficients(fixed, bases, weights)
        actual = weighted_rms_from_coefficients(currents, coefficients)
        direct_vectors = fixed[None, :, :] + np.einsum(
            "tc,cpk->tpk", currents, bases
        )
        expected = np.sqrt(
            np.sum(weights[None, :] * np.sum(direct_vectors**2, axis=2), axis=1)
            / np.sum(weights)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-18)


if __name__ == "__main__":
    unittest.main()
