"""Dependency-light validation for derived inward surface-region volumes."""
from __future__ import annotations

import math
import unittest

import numpy as np

from fieldworkbench.spatial_analysis import region_metrics
from fieldworkbench.surface_region_geometry import SurfaceRegionGeometry
from fieldworkbench.surface_region_volume import (
    MAXIMUM_SURFACE_DEPTH_MM,
    MINIMUM_SURFACE_DEPTH_MM,
    SurfaceRegionVolumeError,
    SurfaceRegionVolumeSettings,
    build_surface_region_volume_set,
    surface_region_volume_cache_identity,
)


_CUBE_FACES = np.asarray(
    [
        [0, 1, 3], [0, 3, 2],
        [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4],
        [1, 5, 7], [1, 7, 3],
    ],
    dtype=int,
)


def cube_geometry(
    *,
    complete: bool = True,
    half_extent_m: float = 0.1,
    translation_m=(0.0, 0.0, 0.0),
    rotation_matrix: np.ndarray | None = None,
) -> SurfaceRegionGeometry:
    vertices = np.asarray(
        [
            [sx, sy, sz]
            for sx in (-half_extent_m, half_extent_m)
            for sy in (-half_extent_m, half_extent_m)
            for sz in (-half_extent_m, half_extent_m)
        ],
        dtype=float,
    )
    if rotation_matrix is not None:
        vertices = vertices @ np.asarray(rotation_matrix, dtype=float).T
    vertices = vertices + np.asarray(translation_m, dtype=float)
    regions = {
        "left": {"display_name": "Left", "face_indices": [0, 1]},
    }
    if complete:
        regions["other"] = {"display_name": "Other", "face_indices": list(range(2, 12))}
    return SurfaceRegionGeometry(
        vertices,
        _CUBE_FACES,
        {"regions": regions},
        source_key="test_cube",
        source_name="Test cube",
    )


def left_right_cube_geometry() -> SurfaceRegionGeometry:
    geometry = cube_geometry(complete=False)
    return SurfaceRegionGeometry(
        geometry.vertices_m,
        geometry.faces,
        {
            "regions": {
                "left": {"display_name": "Left", "face_indices": [0, 1]},
                "right": {"display_name": "Right", "face_indices": [2, 3]},
            }
        },
        source_key="test_cube_lr",
        source_name="Test cube left/right",
    )


def icosphere_geometry(*, radius_m: float = 0.1, subdivisions: int = 2) -> SurfaceRegionGeometry:
    """Return a small closed triangular approximation of a sphere."""

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    vertices = [np.asarray(value, dtype=float) for value in vertices]
    vertices = [value / np.linalg.norm(value) for value in vertices]

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            key = tuple(sorted((first, second)))
            cached = midpoint_cache.get(key)
            if cached is not None:
                return cached
            value = vertices[first] + vertices[second]
            value = value / np.linalg.norm(value)
            index = len(vertices)
            vertices.append(value)
            midpoint_cache[key] = index
            return index

        refined = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined

    vertex_array = np.asarray(vertices, dtype=float) * radius_m
    face_array = np.asarray(faces, dtype=int)
    return SurfaceRegionGeometry(
        vertex_array,
        face_array,
        {
            "regions": {
                "sphere": {
                    "display_name": "Sphere",
                    "face_indices": list(range(len(face_array))),
                }
            }
        },
        source_key="test_sphere",
        source_name="Test sphere",
    )


class SurfaceRegionVolumeTests(unittest.TestCase):
    def settings(self, **overrides) -> SurfaceRegionVolumeSettings:
        values = {
            "depth_mm": 20.0,
            "grid_spacing_mm": 5.0,
            "maximum_lattice_cells": 100_000,
            "maximum_samples_per_region": 100_000,
        }
        values.update(overrides)
        return SurfaceRegionVolumeSettings(**values)

    def test_user_facing_depth_range_is_bounded_to_thirty_mm(self):
        self.assertEqual(MINIMUM_SURFACE_DEPTH_MM, 0.01)
        self.assertEqual(MAXIMUM_SURFACE_DEPTH_MM, 30.0)
        self.assertEqual(
            SurfaceRegionVolumeSettings(depth_mm=MAXIMUM_SURFACE_DEPTH_MM).validated().depth_mm,
            MAXIMUM_SURFACE_DEPTH_MM,
        )
        for invalid in (0.0, MINIMUM_SURFACE_DEPTH_MM / 2.0, MAXIMUM_SURFACE_DEPTH_MM + 0.001):
            with self.assertRaisesRegex(SurfaceRegionVolumeError, "between 0.01 and 30 mm"):
                SurfaceRegionVolumeSettings(depth_mm=invalid).validated()

    def test_cube_shell_is_inward_only_and_has_expected_voxel_volume(self):
        geometry = cube_geometry()
        result = build_surface_region_volume_set(geometry, self.settings())
        # 0.2 m cube minus its 0.16 m inner cube = 0.003904 m^3.
        # The requested 5 mm cell grid aligns exactly with those boundaries.
        self.assertAlmostEqual(result.metadata["volume_cm3"], 3904.0, places=6)
        self.assertEqual(result.metadata["depth_direction"], "inward only")
        self.assertEqual(result.metadata["unclassified_source_voxel_count"], 0)
        self.assertEqual(result.samples.point_count, result.metadata["source_voxel_count"])
        self.assertTrue(np.all(np.max(np.abs(result.samples.points_m), axis=1) <= 0.1 + 1e-12))
        nearest_depth = np.min(0.1 - np.abs(result.samples.points_m), axis=1)
        self.assertTrue(np.all(nearest_depth <= 0.02 + 1e-12))
        self.assertTrue(np.all(nearest_depth >= -1e-12))

    def test_downsampling_preserves_per_region_physical_volume(self):
        result = build_surface_region_volume_set(
            cube_geometry(), self.settings(maximum_samples_per_region=17)
        )
        self.assertEqual(set(result.samples.source_counts), {1, 2})
        self.assertLessEqual(result.samples.sampled_counts[1], 17)
        self.assertLessEqual(result.samples.sampled_counts[2], 17)
        for label in (1, 2):
            mask = result.samples.label_ids == label
            self.assertAlmostEqual(
                float(np.sum(result.samples.weights_mm3[mask])),
                result.samples.label_volumes_mm3[label],
                places=6,
            )
        self.assertEqual(result.samples.point_count, sum(result.samples.sampled_counts.values()))

    def test_prebuild_cache_identity_tracks_depth_and_sampling_controls(self):
        geometry = cube_geometry()
        shallow, shallow_payload = surface_region_volume_cache_identity(
            geometry, self.settings(depth_mm=10.0, maximum_samples_per_region=10)
        )
        deep, deep_payload = surface_region_volume_cache_identity(
            geometry, self.settings(depth_mm=20.0, maximum_samples_per_region=10)
        )
        denser, _ = surface_region_volume_cache_identity(
            geometry,
            self.settings(
                depth_mm=10.0, grid_spacing_mm=2.5, maximum_samples_per_region=10
            ),
        )
        self.assertNotEqual(shallow, deep)
        self.assertNotEqual(shallow, denser)
        self.assertEqual(shallow_payload["depth_mm"], 10.0)
        self.assertEqual(deep_payload["depth_mm"], 20.0)
        self.assertEqual(shallow_payload["provider_version"], 2)

    def test_depth_is_part_of_derived_geometry_identity(self):
        geometry = cube_geometry()
        shallow = build_surface_region_volume_set(
            geometry, self.settings(depth_mm=10.0, maximum_samples_per_region=10)
        )
        deep = build_surface_region_volume_set(
            geometry, self.settings(depth_mm=20.0, maximum_samples_per_region=10)
        )
        self.assertNotEqual(
            shallow.metadata["derived_geometry_identity"],
            deep.metadata["derived_geometry_identity"],
        )
        shallow_cache, _ = surface_region_volume_cache_identity(
            geometry, self.settings(depth_mm=10.0, maximum_samples_per_region=10)
        )
        self.assertEqual(shallow.metadata["geometry_cache_identity"], shallow_cache)
        self.assertGreater(deep.metadata["volume_cm3"], shallow.metadata["volume_cm3"])
        self.assertEqual(shallow.hierarchy.children[0].metadata["sampling_depth_mm"], 10.0)
        self.assertEqual(deep.hierarchy.children[0].metadata["sampling_depth_mm"], 20.0)

    def test_unclassified_surface_volume_is_not_assigned_to_named_regions(self):
        result = build_surface_region_volume_set(cube_geometry(complete=False), self.settings())
        self.assertEqual(tuple(child.key for child in result.hierarchy.children), ("left",))
        self.assertGreater(result.metadata["unclassified_source_voxel_count"], 0)
        self.assertGreater(result.metadata["unclassified_volume_cm3"], 0.0)
        self.assertEqual(set(result.samples.source_counts), {1})

    def test_region_metadata_reports_surface_area_volume_and_sampling_counts(self):
        result = build_surface_region_volume_set(
            cube_geometry(), self.settings(grid_spacing_mm=10.0, maximum_samples_per_region=8)
        )
        by_key = {node.key: node for node in result.hierarchy.children}
        self.assertAlmostEqual(by_key["left"].metadata["surface_area_cm2"], 400.0)
        self.assertGreater(by_key["left"].metadata["volume_cm3"], 0.0)
        self.assertGreater(by_key["left"].metadata["source_voxel_count"], 0)
        self.assertLessEqual(by_key["left"].metadata["retained_sample_count"], 8)
        self.assertEqual(result.kind, "surface_region_volume")
        self.assertEqual(result.sampling_mode, "inward_voxel_shell")

    def test_root_volume_equals_sum_of_named_child_volumes(self):
        result = build_surface_region_volume_set(cube_geometry(), self.settings())
        child_volume = sum(float(node.metadata["volume_cm3"]) for node in result.hierarchy.children)
        self.assertAlmostEqual(float(result.hierarchy.metadata["volume_cm3"]), child_volume, places=12)
        self.assertAlmostEqual(float(result.metadata["volume_cm3"]), child_volume, places=12)

    def test_translation_does_not_change_lattice_membership_or_volume(self):
        settings = self.settings(maximum_samples_per_region=250)
        base = build_surface_region_volume_set(cube_geometry(), settings)
        shift = np.asarray([0.123456789, -0.222222222, 1.75], dtype=float)
        moved = build_surface_region_volume_set(
            cube_geometry(translation_m=shift), settings
        )
        self.assertEqual(base.metadata["grid_shape"], moved.metadata["grid_shape"])
        self.assertAlmostEqual(base.metadata["volume_cm3"], moved.metadata["volume_cm3"], places=9)
        self.assertNotEqual(
            base.metadata["geometry_cache_identity"],
            moved.metadata["geometry_cache_identity"],
        )
        np.testing.assert_array_equal(base.samples.label_ids, moved.samples.label_ids)
        np.testing.assert_array_equal(base.samples.source_indices, moved.samples.source_indices)
        np.testing.assert_allclose(base.samples.weights_mm3, moved.samples.weights_mm3, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(
            base.samples.points_m,
            moved.samples.points_m - shift,
            atol=2e-14,
            rtol=0.0,
        )

    def test_rotation_preserves_shell_volume_within_voxel_tolerance(self):
        angle = math.radians(33.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        base = build_surface_region_volume_set(cube_geometry(), self.settings())
        rotated = build_surface_region_volume_set(
            cube_geometry(rotation_matrix=rotation), self.settings(maximum_lattice_cells=200_000)
        )
        relative_error = abs(rotated.metadata["volume_cm3"] - base.metadata["volume_cm3"]) / base.metadata["volume_cm3"]
        self.assertLess(relative_error, 0.035)

    def test_uniform_scaling_with_scaled_depth_and_grid_scales_volume_cubically(self):
        base = build_surface_region_volume_set(cube_geometry(), self.settings())
        scale = 1.5
        scaled = build_surface_region_volume_set(
            cube_geometry(half_extent_m=0.1 * scale),
            self.settings(depth_mm=20.0 * scale, grid_spacing_mm=5.0 * scale),
        )
        self.assertAlmostEqual(
            scaled.metadata["volume_cm3"] / base.metadata["volume_cm3"],
            scale**3,
            places=9,
        )

    def test_build_is_exactly_repeatable_for_same_geometry_and_settings(self):
        settings = self.settings(maximum_samples_per_region=23)
        first = build_surface_region_volume_set(cube_geometry(), settings)
        second = build_surface_region_volume_set(cube_geometry(), settings)
        self.assertEqual(first.metadata["derived_geometry_identity"], second.metadata["derived_geometry_identity"])
        np.testing.assert_array_equal(first.samples.points_m, second.samples.points_m)
        np.testing.assert_array_equal(first.samples.label_ids, second.samples.label_ids)
        np.testing.assert_array_equal(first.samples.weights_mm3, second.samples.weights_mm3)
        np.testing.assert_array_equal(first.samples.source_indices, second.samples.source_indices)

    def test_runtime_progress_and_cancellation_interrupt_shell_build(self):
        events = []

        def progress(payload):
            events.append(dict(payload))

        def cancel_requested():
            return bool(events)

        with self.assertRaisesRegex(SurfaceRegionVolumeError, "cancelled"):
            build_surface_region_volume_set(
                cube_geometry(),
                self.settings(grid_spacing_mm=2.5, maximum_lattice_cells=600_000),
                progress_callback=progress,
                cancel_requested=cancel_requested,
            )
        self.assertTrue(events)
        self.assertIn("Building near-surface integration lattice", events[0]["stage"])

    def test_curved_sphere_shell_agrees_with_analytic_volume(self):
        radius_m = 0.1
        depth_m = 0.02
        geometry = icosphere_geometry(radius_m=radius_m, subdivisions=2)
        result = build_surface_region_volume_set(
            geometry,
            self.settings(depth_mm=depth_m * 1000.0, grid_spacing_mm=5.0),
        )
        analytic_cm3 = (4.0 / 3.0) * math.pi * (
            radius_m**3 - (radius_m - depth_m) ** 3
        ) * 1e6
        relative_error = abs(result.metadata["volume_cm3"] - analytic_cm3) / analytic_cm3
        # Error combines the finite triangular sphere approximation and the 5 mm
        # voxel-centre integration grid; this guards gross curved-shell errors.
        self.assertLess(relative_error, 0.08)

    def test_uniform_field_reports_identical_metrics_in_every_region(self):
        result = build_surface_region_volume_set(
            cube_geometry(), self.settings(maximum_samples_per_region=75)
        )
        vectors_t = np.tile([10.0e-6, 0.0, 0.0], (result.samples.point_count, 1))
        metrics = region_metrics(vectors_t, result.samples, result.hierarchy)
        for row in metrics.values():
            self.assertAlmostEqual(row.mean_uT, 10.0, places=10)
            self.assertAlmostEqual(row.rms_uT, 10.0, places=10)
            self.assertAlmostEqual(row.p95_uT, 10.0, places=10)
            self.assertAlmostEqual(row.peak_uT, 10.0, places=10)
            self.assertAlmostEqual(row.uniformity_pct, 100.0, places=10)
            self.assertAlmostEqual(row.directional_consistency_pct, 100.0, places=10)

    def test_known_spatial_gradient_distinguishes_opposite_surface_regions(self):
        result = build_surface_region_volume_set(
            left_right_cube_geometry(), self.settings(maximum_samples_per_region=250)
        )
        # Positive X-directed field with a monotonic X gradient.  Both retained
        # regions are geometrically identical opposite surface layers.
        x = result.samples.points_m[:, 0]
        magnitude_t = (x + 0.2) * 100.0e-6
        vectors_t = np.column_stack((magnitude_t, np.zeros_like(x), np.zeros_like(x)))
        metrics = region_metrics(vectors_t, result.samples, result.hierarchy)
        self.assertGreater(metrics["right"].mean_uT, metrics["left"].mean_uT * 2.0)
        self.assertGreater(metrics["right"].p95_uT, metrics["left"].p95_uT)


if __name__ == "__main__":
    unittest.main()
