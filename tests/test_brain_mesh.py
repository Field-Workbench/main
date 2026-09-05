from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from fieldworkbench.brain_mesh import (
    BrainMeshError,
    LPBA40_MESH_CACHE_RESOURCE,
    build_lpba40_region_meshes,
    load_lpba40_region_meshes,
)
from fieldworkbench.sri24_atlas import load_sri24_parcellation


class LPBA40BrainMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.atlas = load_sri24_parcellation("lpba40")
        cls.meshes = load_lpba40_region_meshes(cls.atlas)

    def test_checked_in_cache_retains_every_non_background_label(self) -> None:
        expected = {
            int(value) for value in self.atlas.present_label_ids if int(value) != 0
        }
        self.assertEqual(set(self.meshes.regions), expected)
        self.assertEqual(len(self.meshes.regions), 56)
        self.assertEqual(self.meshes.source_stride, 2)
        self.assertEqual(self.meshes.source_shape, (120, 120, 78))
        self.assertEqual(self.meshes.cache_version, 1)
        self.assertEqual(self.meshes.vertex_count, 102_724)
        self.assertEqual(self.meshes.triangle_count, 206_080)
        self.assertGreater(LPBA40_MESH_CACHE_RESOURCE.stat().st_size, 1_000_000)

        for label_id, mesh in self.meshes.regions.items():
            with self.subTest(label_id=label_id):
                self.assertGreater(len(mesh.vertices_mm), 3)
                self.assertGreater(mesh.triangle_count, 1)
                self.assertTrue(np.isfinite(mesh.vertices_mm).all())
                self.assertEqual(mesh.vertices_mm.dtype, np.dtype("float32"))
                self.assertEqual(mesh.triangles.dtype, np.dtype("int32"))
                self.assertFalse(mesh.vertices_mm.flags.writeable)
                self.assertFalse(mesh.triangles.flags.writeable)
                self.assertGreaterEqual(int(mesh.triangles.min()), 0)
                self.assertLess(int(mesh.triangles.max()), len(mesh.vertices_mm))

    def test_cache_uses_compact_flat_arrays_and_offsets(self) -> None:
        with np.load(LPBA40_MESH_CACHE_RESOURCE, allow_pickle=False) as archive:
            self.assertTrue(
                {
                    "label_ids",
                    "vertices",
                    "vertex_offsets",
                    "triangles",
                    "triangle_offsets",
                    "mesh_version",
                    "source_stride",
                    "source_volume_sha256",
                    "source_label_table_sha256",
                }.issubset(archive.files)
            )
            self.assertEqual(archive["label_ids"].shape, (56,))
            self.assertEqual(archive["vertices"].shape, (102_724, 3))
            self.assertEqual(archive["triangles"].shape, (206_080, 3))
            self.assertEqual(archive["vertex_offsets"].shape, (57,))
            self.assertEqual(archive["triangle_offsets"].shape, (57,))

    def test_runtime_module_has_no_vtk_import(self) -> None:
        source = Path(__file__).resolve().parents[1] / "fieldworkbench" / "brain_mesh.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("vtkmodules", text)
        self.assertIn("load_lpba40_region_meshes", text)

    def test_runtime_rejects_a_stride_not_matching_the_bundled_cache(self) -> None:
        with self.assertRaisesRegex(BrainMeshError, "regenerate"):
            load_lpba40_region_meshes(self.atlas, visual_stride=1)

    def test_scene_placement_matches_existing_brain_transform_contract(self) -> None:
        angle = math.radians(31.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        scale = np.asarray([1.08, 0.94, 1.03], dtype=float)
        position_m = np.asarray([0.012, -0.018, 0.027], dtype=float)
        placed = build_lpba40_region_meshes(
            self.atlas,
            position_m=position_m,
            rotation=rotation,
            scale_xyz=scale,
        )
        label_id = sorted(self.meshes.regions)[17]
        base_vertices = self.meshes.regions[label_id].vertices_mm.astype(float)
        expected = (base_vertices * scale) @ rotation.T + position_m * 1000.0
        actual = placed.regions[label_id].vertices_mm.astype(float)
        np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
