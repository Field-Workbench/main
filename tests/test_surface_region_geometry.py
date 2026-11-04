import json
import unittest
from pathlib import Path

import numpy as np

from fieldworkbench.mesh_io import load_mesh
from fieldworkbench.surface_regions import load_surface_region_map
from fieldworkbench.surface_region_geometry import (
    AMBIGUOUS_REGION_LABEL,
    UNCLASSIFIED_REGION_LABEL,
    SurfaceRegionGeometry,
    SurfaceRegionGeometryError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "fieldworkbench" / "assets"


def cube_geometry(region_map=None):
    vertices = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    )
    faces = np.asarray(
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
    if region_map is None:
        region_map = {
            "regions": {
                "left": {"display_name": "Left", "face_indices": [0, 1]},
                "other": {"display_name": "Other", "face_indices": list(range(2, 12))},
            }
        }
    return SurfaceRegionGeometry(vertices, faces, region_map)


class SurfaceRegionGeometryTests(unittest.TestCase):
    def test_closed_region_mesh_resolves_depth_containment_and_region(self):
        geometry = cube_geometry()
        self.assertTrue(geometry.volume_sampling_ready)
        self.assertTrue(geometry.complete_partition)
        self.assertEqual(geometry.region_key(1), "left")
        self.assertEqual(geometry.region_name(2), "Other")

        query = geometry.query_points(
            np.asarray(
                [
                    [-0.9, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            )
        )
        np.testing.assert_allclose(query.distance_m, [0.1, 0.1, 1.0], atol=1e-12)
        np.testing.assert_array_equal(query.inside, [True, True, False])
        np.testing.assert_array_equal(query.region_label_ids, [1, 2, 2])
        np.testing.assert_allclose(query.signed_depth_m, [0.1, 0.1, -1.0], atol=1e-12)
        self.assertAlmostEqual(geometry.surface_area_m2_for_label(1), 4.0, places=12)
        self.assertTrue(query.containment_certified)

    def test_cube_queries_match_analytic_surface_distance_and_containment(self):
        geometry = cube_geometry()
        rng = np.random.default_rng(12345)
        points = rng.uniform(-1.8, 1.8, size=(120, 3))
        query = geometry.query_points(points)
        absolute = np.abs(points)
        outside_delta = np.maximum(absolute - 1.0, 0.0)
        outside_distance = np.linalg.norm(outside_delta, axis=1)
        analytically_inside = np.all(absolute < 1.0, axis=1)
        inside_distance = np.min(1.0 - absolute, axis=1)
        expected_distance = np.where(analytically_inside, inside_distance, outside_distance)
        np.testing.assert_allclose(query.distance_m, expected_distance, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(query.inside, analytically_inside)

    def test_nearest_surface_ties_use_lowest_original_face_index(self):
        geometry = cube_geometry()
        # Cube centre is exactly equidistant from every face. The stable rule
        # chooses triangle 0, which belongs to the first region.
        distance, closest, faces, labels = geometry.nearest_surface([[0.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(distance[0]), 1.0)
        self.assertEqual(int(faces[0]), 0)
        self.assertEqual(int(labels[0]), 1)
        np.testing.assert_allclose(closest[0], [-1.0, 0.0, 0.0], atol=1e-12)

    def test_unclassified_faces_are_explicit_but_do_not_block_closed_geometry(self):
        geometry = cube_geometry(
            {"regions": {"left": {"display_name": "Left", "face_indices": [0, 1]}}}
        )
        self.assertFalse(geometry.complete_partition)
        self.assertTrue(geometry.volume_sampling_ready)
        self.assertEqual(len(geometry.unclassified_face_indices), 10)
        _, _, faces, labels = geometry.nearest_surface([[0.9, 0.0, 0.0]])
        self.assertEqual(int(faces[0]), 2)
        self.assertEqual(int(labels[0]), UNCLASSIFIED_REGION_LABEL)

    def test_overlapping_authored_faces_are_explicit_and_block_volume_sampling(self):
        geometry = cube_geometry(
            {
                "regions": {
                    "one": {"display_name": "One", "face_indices": [0, 1]},
                    "two": {"display_name": "Two", "face_indices": [0, 2]},
                }
            }
        )
        self.assertFalse(geometry.ownership_unambiguous)
        self.assertFalse(geometry.volume_sampling_ready)
        self.assertEqual(geometry.overlap_face_indices, (0,))
        self.assertEqual(int(geometry.face_region_labels[0]), AMBIGUOUS_REGION_LABEL)
        with self.assertRaisesRegex(SurfaceRegionGeometryError, "more than one authored region"):
            geometry.require_volume_sampling_ready()

    def test_open_mesh_keeps_nearest_queries_but_refuses_containment(self):
        closed = cube_geometry()
        vertices = closed.vertices_m
        faces = closed.faces[:-2]
        region_map = {
            "regions": {
                "surface": {"display_name": "Surface", "face_indices": list(range(len(faces)))}
            }
        }
        geometry = SurfaceRegionGeometry(vertices, faces, region_map)
        self.assertFalse(geometry.containment_reliable)
        distance, _closest, _face_indices, labels = geometry.nearest_surface([[0.0, 0.0, 0.0]])
        self.assertTrue(np.isfinite(distance[0]))
        self.assertEqual(int(labels[0]), 1)
        with self.assertRaisesRegex(SurfaceRegionGeometryError, "closed manifold"):
            geometry.points_inside([[0.0, 0.0, 0.0]])
        query = geometry.query_points([[0.0, 0.0, 0.0]], require_containment=False)
        self.assertFalse(query.containment_certified)

    def test_on_surface_points_are_reported_separately_from_inside(self):
        geometry = cube_geometry()
        query = geometry.query_points([[-1.0, 0.0, 0.0]])
        self.assertTrue(bool(query.on_surface[0]))
        self.assertFalse(bool(query.inside[0]))
        self.assertTrue(bool(query.interior_or_surface[0]))
        self.assertEqual(int(query.region_label_ids[0]), 1)

    def test_bundled_generic_bust_is_closed_partitioned_and_queryable(self):
        mesh_path = ASSET_DIR / "generic_bust_v1.stl"
        regions_path = ASSET_DIR / "generic_bust_v1.regions.json"
        # The built-in asset is authored in SI metres, matching StudioAdapter.
        vertices, faces = load_mesh(mesh_path, unit_scale=1.0)
        region_map = load_surface_region_map(regions_path, vertices=vertices, faces=faces)
        geometry = SurfaceRegionGeometry(
            vertices,
            faces,
            region_map,
            source_key="generic_bust_v1",
            source_name="Generic bust surface regions",
        )
        status = geometry.readiness()
        self.assertTrue(status["volume_sampling_ready"], status["reasons"])
        self.assertTrue(status["complete_partition"])
        self.assertEqual(status["region_count"], 5)
        self.assertEqual(
            {region.key for region in geometry.regions},
            {"scalp", "face", "ear_l", "ear_r", "neck_shoulders"},
        )

        centre = (geometry.bounds_min_m + geometry.bounds_max_m) / 2.0
        # This is primarily a smoke test for the prepared 38k-face provider.
        distance, closest, face_indices, labels = geometry.nearest_surface([centre])
        self.assertTrue(np.isfinite(distance[0]))
        self.assertTrue(np.isfinite(closest[0]).all())
        self.assertGreaterEqual(int(face_indices[0]), 0)
        self.assertGreaterEqual(int(labels[0]), 1)


if __name__ == "__main__":
    unittest.main()
