"""Dependency-light tests for the generic regional-analysis contract."""
from __future__ import annotations

import unittest

import numpy as np

from fieldworkbench.brain_analysis import (
    BrainRegionNode,
    BrainSampleSet,
    lpba40_frame_region_metrics,
)
from fieldworkbench.spatial_analysis import (
    AnalysisRegionNode,
    AnalysisRegionSet,
    AnalysisSampleSet,
    REGIONAL_ANALYSIS_SCHEMA_VERSION,
    REGIONAL_GEOMETRY_CACHE_SCHEMA_VERSION,
    analysis_region_hierarchy_from_payload,
    analysis_region_hierarchy_payload,
    analysis_region_set_cache_payload,
    analysis_region_set_from_cache_payload,
    analysis_region_set_identity,
    frame_region_metrics,
)


class SpatialAnalysisContractTests(unittest.TestCase):
    def test_brain_api_uses_shared_generic_types(self):
        self.assertIs(BrainRegionNode, AnalysisRegionNode)
        self.assertIs(BrainSampleSet, AnalysisSampleSet)

    def test_explicit_region_volume_is_independent_of_sample_density(self):
        samples = AnalysisSampleSet(
            points_m=np.zeros((3, 3), dtype=float),
            label_ids=np.asarray([1, 2, 2], dtype=np.int32),
            weights_mm3=np.asarray([1000.0, 1000.0, 1000.0]),
            source_indices=np.arange(3),
            source_counts={1: 1, 2: 2},
            sampled_counts={1: 1, 2: 2},
            nominal_element_volume_mm3=1.0,
            label_volumes_mm3={1: 2500.0, 2: 7500.0},
        )
        hierarchy = AnalysisRegionNode(
            "all",
            "All regions",
            (1, 2),
            (
                AnalysisRegionNode("a", "A", (1,)),
                AnalysisRegionNode("b", "B", (2,)),
            ),
        )
        fixed = np.zeros((3, 3), dtype=float)
        bases = np.zeros((1, 3, 3), dtype=float)
        bases[0, :, 0] = 1e-6
        frames = frame_region_metrics(
            fixed, bases, np.asarray([[1.0]]), samples, hierarchy
        )
        values = np.asarray(frames["values"], dtype=float)[0]
        by_key = {key: values[index] for index, key in enumerate(frames["keys"])}
        self.assertAlmostEqual(by_key["all"][6], 10.0, places=6)
        self.assertAlmostEqual(by_key["a"][6], 2.5, places=6)
        self.assertAlmostEqual(by_key["b"][6], 7.5, places=6)

    def test_lpba40_wrapper_and_generic_engine_match_for_explicit_hierarchy(self):
        samples = AnalysisSampleSet(
            np.zeros((2, 3)), np.asarray([1, 1]), np.asarray([1.0, 1.0]),
            np.arange(2), {1: 2}, {1: 2}, 1.0,
        )
        hierarchy = AnalysisRegionNode("r", "Region", (1,))
        fixed = np.zeros((2, 3), dtype=float)
        bases = np.asarray([[[1e-6, 0.0, 0.0], [2e-6, 0.0, 0.0]]])
        currents = np.asarray([[0.0], [1.0]])
        generic = frame_region_metrics(fixed, bases, currents, samples, hierarchy)
        wrapped = lpba40_frame_region_metrics(
            fixed, bases, currents, samples, hierarchy
        )
        self.assertEqual(generic["keys"], wrapped["keys"])
        np.testing.assert_allclose(generic["values"], wrapped["values"])

    def test_region_set_cache_round_trip_preserves_weighted_geometry(self):
        samples = AnalysisSampleSet(
            np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            np.asarray([3, 4], dtype=np.int32),
            np.asarray([12.5, 25.0]),
            np.asarray([10, 20], dtype=np.int64),
            {3: 5, 4: 7}, {3: 1, 4: 1}, 2.5,
            label_volumes_mm3={3: 62.5, 4: 175.0},
        )
        hierarchy = AnalysisRegionNode(
            "all", "All", (3, 4),
            (
                AnalysisRegionNode("a", "A", (3,), metadata={"depth_mm": 2.0}),
                AnalysisRegionNode("b", "B", (4,)),
            ),
            region_kind="derived",
        )
        original = AnalysisRegionSet(
            key="example", name="Example", kind="surface_region_volume",
            hierarchy=hierarchy, samples=samples, sampling_mode="inward_voxel_shell",
            metadata={"provider": "test", "provider_version": 2, "depth_mm": 2.0},
        )
        payload = analysis_region_set_cache_payload(original)
        self.assertEqual(payload["schema_version"], REGIONAL_GEOMETRY_CACHE_SCHEMA_VERSION)
        restored = analysis_region_set_from_cache_payload(payload)
        self.assertEqual(restored.key, original.key)
        self.assertEqual(restored.hierarchy.children[0].metadata["depth_mm"], 2.0)
        np.testing.assert_allclose(restored.samples.points_m, original.samples.points_m)
        np.testing.assert_array_equal(restored.samples.label_ids, original.samples.label_ids)
        np.testing.assert_allclose(restored.samples.weights_mm3, original.samples.weights_mm3)
        self.assertEqual(restored.samples.source_counts, original.samples.source_counts)
        self.assertEqual(restored.samples.label_volumes_mm3, original.samples.label_volumes_mm3)
        self.assertEqual(analysis_region_set_identity(restored), analysis_region_set_identity(original))

    def test_region_set_identity_changes_when_geometry_membership_changes(self):
        base = AnalysisRegionSet(
            key="r", name="R", kind="volume",
            hierarchy=AnalysisRegionNode("r", "R", (1,)),
            samples=AnalysisSampleSet(
                np.asarray([[0.0, 0.0, 0.0]]), np.asarray([1]), np.asarray([1.0]),
                np.asarray([0]), {1: 1}, {1: 1}, 1.0,
            ),
            sampling_mode="test", metadata={"provider": "test"},
        )
        moved = AnalysisRegionSet(
            key="r", name="R", kind="volume", hierarchy=base.hierarchy,
            samples=AnalysisSampleSet(
                np.asarray([[0.001, 0.0, 0.0]]), np.asarray([1]), np.asarray([1.0]),
                np.asarray([0]), {1: 1}, {1: 1}, 1.0,
            ),
            sampling_mode="test", metadata={"provider": "test"},
        )
        self.assertNotEqual(analysis_region_set_identity(base), analysis_region_set_identity(moved))

    def test_region_set_carries_provider_identity_separately_from_numerics(self):
        samples = AnalysisSampleSet(
            np.zeros((1, 3)), np.asarray([7]), np.asarray([5.0]), np.arange(1),
            {7: 1}, {7: 1}, 5.0,
        )
        hierarchy = AnalysisRegionNode("surface", "Surface", (7,))
        region_set = AnalysisRegionSet(
            key="example.surface",
            name="Example surface regions",
            kind="surface_shell",
            hierarchy=hierarchy,
            samples=samples,
            sampling_mode="derived_shell",
            metadata={"provider": "test_provider", "provider_version": 1},
        )
        self.assertEqual(REGIONAL_ANALYSIS_SCHEMA_VERSION, 1)
        self.assertEqual(region_set.kind, "surface_shell")
        self.assertEqual(region_set.samples.point_count, 1)
        self.assertEqual(region_set.metadata["provider"], "test_provider")

    def test_region_hierarchy_ui_payload_round_trip_preserves_surface_metadata(self):
        hierarchy = AnalysisRegionNode(
            "surface_regions",
            "Surface regions",
            (1, 2),
            (
                AnalysisRegionNode(
                    "face",
                    "Face",
                    (1,),
                    region_kind="near_surface_volume",
                    metadata={"sampling_depth_mm": 2.0, "volume_cm3": 41.25},
                ),
                AnalysisRegionNode(
                    "scalp",
                    "Scalp",
                    (2,),
                    region_kind="near_surface_volume",
                    metadata={"sampling_depth_mm": 2.0, "volume_cm3": 87.5},
                ),
            ),
            region_kind="near_surface_volume_set",
            metadata={"sampling_depth_mm": 2.0},
        )
        restored = analysis_region_hierarchy_from_payload(
            analysis_region_hierarchy_payload(hierarchy)
        )
        self.assertEqual(restored, hierarchy)
        self.assertEqual(restored.children[0].metadata["volume_cm3"], 41.25)


if __name__ == "__main__":
    unittest.main()
