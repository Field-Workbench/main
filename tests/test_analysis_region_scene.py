"""Dependency-light tests for renderable Exposure Analysis region scenes."""
from __future__ import annotations

import unittest

import numpy as np

from fieldworkbench.analysis_region_scene import (
    ANALYSIS_REGION_SCENE_VERSION,
    AnalysisRegionScene,
    AnalysisRegionSceneError,
    analysis_region_node_display_metrics,
    analysis_region_scene_identity,
    resolve_analysis_region_scene,
)
from fieldworkbench.spatial_analysis import AnalysisRegionNode
from fieldworkbench.surface_region_geometry import SurfaceRegionGeometry


class _FakeAdapter:
    def __init__(self, target, *, surface_geometry=None, surface_map=None, mesh_geometry=None):
        self.target = dict(target)
        self._surface_geometry = surface_geometry
        self._surface_map = surface_map
        self._mesh_geometry = mesh_geometry

    def get_object(self, target_id):
        if str(target_id) != str(self.target["id"]):
            raise KeyError(target_id)
        return dict(self.target)

    def supports_surface_region_volume(self, target_id):
        return self._surface_geometry is not None

    def surface_region_geometry(self, target_id):
        return self._surface_geometry

    def surface_region_properties(self, target_id):
        return self._surface_map

    def surface_region_analysis_properties(self, target_id):
        return {"depth_mm": 2.0}

    def supports_measurement(self, target_id):
        return self._mesh_geometry is not None

    def _mesh(self, target_id):
        return dict(self.target)

    def _mesh_world_vertices_faces(self, mesh):
        if self._mesh_geometry is None:
            raise AssertionError("measurement mesh was not configured")
        return self._mesh_geometry

    @staticmethod
    def _mesh_world_transform(mesh):
        return np.zeros(3), np.eye(3)

    @staticmethod
    def _mesh_scale_xyz(mesh):
        return np.ones(3)


def _cube_surface_geometry():
    vertices = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=float,
    ) * 0.01
    faces = np.asarray(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=int,
    )
    regions = {
        "schema": "fieldworkbench.surface_regions",
        "schema_version": 1,
        "regions": {
            "left": {"display_name": "Left", "face_indices": [0, 1]},
            "rest": {"display_name": "Rest", "face_indices": list(range(2, 12))},
        },
    }
    geometry = SurfaceRegionGeometry(vertices, faces, regions, source_key="test")
    region_map = {
        "regions": {
            "left": {"settings": {"colour": "#112233"}},
            "rest": {"settings": {"colour": "#445566"}},
        }
    }
    return geometry, region_map


class AnalysisRegionSceneTests(unittest.TestCase):
    def test_scene_selects_hierarchy_nodes_by_shared_label_ids(self):
        hierarchy = AnalysisRegionNode(
            "all", "All", (1, 2),
            (
                AnalysisRegionNode("one", "One", (1,)),
                AnalysisRegionNode("two", "Two", (2,)),
            ),
        )
        scene = AnalysisRegionScene(
            target_id="x",
            target_name="Example",
            target_kind="test",
            region_set_key="test.regions",
            hierarchy=hierarchy,
            vertices_m=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
            faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=int),
            face_label_ids=np.asarray([1, 2], dtype=int),
        )
        np.testing.assert_array_equal(scene.face_mask_for_node("one"), [True, False])
        np.testing.assert_array_equal(scene.face_mask_for_node("all"), [True, True])
        self.assertEqual(scene.atomic_label_ids, (1, 2))
        self.assertEqual(scene.triangle_count, 2)
        self.assertEqual(scene.object_triangle_count, 2)
        np.testing.assert_array_equal(scene.object_faces, scene.faces)
        self.assertEqual(ANALYSIS_REGION_SCENE_VERSION, 2)

    def test_node_display_metrics_describe_selected_render_surface(self):
        hierarchy = AnalysisRegionNode(
            "all", "All", (1, 2),
            (AnalysisRegionNode("one", "One", (1,)), AnalysisRegionNode("two", "Two", (2,))),
        )
        scene = AnalysisRegionScene(
            target_id="x", target_name="Example", target_kind="test", region_set_key="test",
            hierarchy=hierarchy,
            vertices_m=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
            faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=int),
            face_label_ids=np.asarray([1, 2], dtype=int),
        )
        metrics = analysis_region_node_display_metrics(scene, "one")
        self.assertEqual(metrics["triangle_count"], 1)
        self.assertEqual(metrics["vertex_count"], 3)
        self.assertEqual(metrics["atomic_label_count"], 1)
        self.assertAlmostEqual(metrics["surface_area_cm2"], 5000.0)
        self.assertEqual(metrics["size_mm"], (1000.0, 1000.0, 0.0))

    def test_scene_rejects_hierarchy_labels_missing_from_display_mesh(self):
        with self.assertRaisesRegex(AnalysisRegionSceneError, "missing hierarchy label"):
            AnalysisRegionScene(
                target_id="x",
                target_name="Bad",
                target_kind="test",
                region_set_key="bad",
                hierarchy=AnalysisRegionNode("all", "All", (1, 2)),
                vertices_m=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.asarray([[0, 1, 2]], dtype=int),
                face_label_ids=np.asarray([1], dtype=int),
            )

    def test_scene_identity_follows_geometry_and_region_ownership_not_style(self):
        hierarchy = AnalysisRegionNode("all", "All", (1,))
        kwargs = dict(
            target_id="x", target_name="Example", target_kind="test",
            region_set_key="test", hierarchy=hierarchy,
            vertices_m=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            faces=np.asarray([[0, 1, 2]], dtype=int), face_label_ids=np.asarray([1]),
        )
        first = AnalysisRegionScene(**kwargs, base_colour="#112233")
        restyled = AnalysisRegionScene(**kwargs, base_colour="#abcdef")
        moved = AnalysisRegionScene(
            **{**kwargs, "vertices_m": np.asarray([[0, 0, 0], [1.1, 0, 0], [0, 1, 0]])}
        )
        self.assertEqual(analysis_region_scene_identity(first), analysis_region_scene_identity(restyled))
        self.assertNotEqual(analysis_region_scene_identity(first), analysis_region_scene_identity(moved))

        alternate_object = AnalysisRegionScene(
            **kwargs,
            object_vertices_m=np.asarray([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float),
            object_faces=np.asarray([[0, 1, 2]], dtype=int),
        )
        self.assertNotEqual(
            analysis_region_scene_identity(first),
            analysis_region_scene_identity(alternate_object),
        )

    def test_surface_regions_take_precedence_over_measurement_scene_role(self):
        geometry, region_map = _cube_surface_geometry()
        adapter = _FakeAdapter(
            {
                "id": "bust", "label": "Bust", "role": "measurement",
                "colour": "#778899", "opacity": 0.5,
            },
            surface_geometry=geometry,
            surface_map=region_map,
            mesh_geometry=(geometry.vertices_m, geometry.faces),
        )
        scene = resolve_analysis_region_scene(adapter, "bust")
        self.assertEqual(scene.target_kind, "surface_region_volume")
        self.assertEqual(scene.hierarchy.key, "surface_regions")
        self.assertEqual(tuple(child.key for child in scene.hierarchy.children), ("left", "rest"))
        self.assertEqual(scene.label_colours, {1: "#112233", 2: "#445566"})
        self.assertEqual(scene.metadata["source_role"], "measurement")
        self.assertEqual(int(np.sum(scene.face_mask_for_node("left"))), 2)
        np.testing.assert_array_equal(scene.object_faces, geometry.faces)
        np.testing.assert_allclose(scene.object_vertices_m, geometry.vertices_m)

    def test_plain_measurement_object_uses_whole_surface_as_one_region(self):
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.asarray([[0, 1, 2]], dtype=int)
        adapter = _FakeAdapter(
            {"id": "m", "label": "Measurement", "role": "measurement", "colour": "#123456"},
            mesh_geometry=(vertices, faces),
        )
        scene = resolve_analysis_region_scene(adapter, "m")
        self.assertEqual(scene.target_kind, "measurement_volume")
        self.assertEqual(scene.hierarchy.label_ids, (1,))
        np.testing.assert_array_equal(scene.face_label_ids, [1])

    def test_lpba40_scene_uses_same_hierarchy_labels_as_analysis_provider(self):
        object_vertices = np.asarray(
            [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=float
        )
        object_faces = np.asarray([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=int)
        adapter = _FakeAdapter(
            {
                "id": "brain", "label": "Brain", "role": "visual",
                "builtin_template": "sri24_brain", "colour": "#8b5cf6", "opacity": 0.72,
            },
            mesh_geometry=(object_vertices, object_faces),
        )
        scene = resolve_analysis_region_scene(adapter, "brain")
        self.assertEqual(scene.target_kind, "brain_lpba40")
        self.assertEqual(scene.region_set_key, "sri24.lpba40")
        self.assertEqual(scene.hierarchy.key, "whole_brain")
        self.assertEqual(set(scene.atomic_label_ids), set(scene.hierarchy.label_ids))
        self.assertGreater(scene.vertex_count, 1000)
        self.assertEqual(scene.triangle_count, 206_080)
        self.assertEqual(scene.object_triangle_count, 4)
        np.testing.assert_array_equal(scene.object_faces, object_faces)
        np.testing.assert_allclose(scene.object_vertices_m, object_vertices)
        self.assertTrue(np.isfinite(scene.bounds_min_m).all())
        self.assertTrue(np.isfinite(scene.bounds_max_m).all())


if __name__ == "__main__":
    unittest.main()
