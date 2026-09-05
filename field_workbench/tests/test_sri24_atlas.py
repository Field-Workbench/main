"""Integrity and coordinate-frame tests for the bundled SRI24 labels."""

from __future__ import annotations

import unittest

import numpy as np

from fieldworkbench.nifti import read_nifti_volume
from fieldworkbench.mesh_io import load_mesh
from fieldworkbench.sri24_atlas import (
    SRI24_GRID_SHAPE,
    SRI24_LABEL_AFFINE_RAS_MM,
    SRI24_LABEL_ASSET_DIR,
    load_sri24_parcellation,
    load_sri24_supratentorial_mask,
    sri24_parcellation_choices,
    validate_sri24_asset_bundle,
)


class SRI24AtlasTests(unittest.TestCase):
    def test_bundle_integrity_lookup_coverage_and_laterality(self) -> None:
        report = validate_sri24_asset_bundle()
        self.assertEqual(report["atlas_release"], "SRI24 v2.0")
        self.assertEqual(
            report["source_archive_sha256"],
            "ed47fde305baefb687a38e0daa8527896480e4e0159a0a0f56c8cbe09fc97e1f",
        )
        self.assertEqual(report["maps"]["lpba40"]["present_label_count"], 57)
        self.assertEqual(report["maps"]["lpba40"]["non_background_label_count"], 56)
        self.assertEqual(report["maps"]["lpba40"]["laterality_pairs_checked"], 27)
        self.assertEqual(report["maps"]["tzo116plus"]["present_label_count"], 424)
        self.assertEqual(
            report["maps"]["tzo116plus"]["non_background_label_count"], 423
        )
        self.assertEqual(report["maps"]["tzo116plus"]["laterality_pairs_checked"], 54)
        self.assertEqual(report["suptent_values"], [0, 1, 2])
        self.assertEqual(report["lpba40_mesh_cache"]["mesh_version"], 1)
        self.assertEqual(report["lpba40_mesh_cache"]["source_stride"], 2)
        self.assertEqual(report["lpba40_mesh_cache"]["label_count"], 56)
        self.assertEqual(report["lpba40_mesh_cache"]["vertex_count"], 102_724)
        self.assertEqual(report["lpba40_mesh_cache"]["triangle_count"], 206_080)
        self.assertEqual(
            report["lpba40_mesh_cache"]["generator"],
            "VTK SurfaceNets3D 9.6.2",
        )

    def test_generic_nifti_reader_preserves_native_sri24_grid(self) -> None:
        volume, affine = read_nifti_volume(SRI24_LABEL_ASSET_DIR / "lpba40.nii.gz")
        self.assertEqual(volume.shape, SRI24_GRID_SHAPE)
        self.assertEqual(volume.dtype, np.dtype("uint8"))
        np.testing.assert_allclose(affine, SRI24_LABEL_AFFINE_RAS_MM)
        self.assertEqual(len(np.unique(volume)), 57)

    def test_parcellations_expose_stable_choices_and_named_regions(self) -> None:
        self.assertEqual(
            [choice["key"] for choice in sri24_parcellation_choices()],
            ["tzo116plus", "lpba40"],
        )
        lpba40 = load_sri24_parcellation("lpba")
        self.assertEqual(lpba40.labels[165].name, "L_hippocampus")
        self.assertEqual(lpba40.labels[166].name, "R_hippocampus")
        self.assertEqual(lpba40.unused_table_label_ids, ())
        self.assertEqual(lpba40.supplemental_label_ids, ())

    def test_tzo_table_omissions_are_explicitly_and_completely_supplemented(self) -> None:
        tzo = load_sri24_parcellation("aal")
        self.assertEqual(
            tzo.supplemental_label_ids,
            (422, 424, 426, 428, 430, 432, 434, 476, 478),
        )
        self.assertEqual(
            tzo.unused_table_label_ids,
            (201, 203, 205, 219, 221, 227, 413),
        )
        self.assertEqual(tzo.labels[422].name, "LateralVentricle_R_y158")
        self.assertEqual(tzo.labels[422].rgba, (0, 248, 0, 255))
        self.assertEqual(tzo.labels[476].name, "ThirdVentricle_R_y90")
        self.assertEqual(tzo.labels[476].rgba, (0, 130, 130, 255))
        self.assertFalse(set(tzo.present_label_ids) - set(tzo.labels))

    def test_supplied_support_mask_uses_the_same_affine(self) -> None:
        support, affine = load_sri24_supratentorial_mask()
        self.assertEqual(support.shape, SRI24_GRID_SHAPE)
        self.assertEqual(set(int(value) for value in np.unique(support)), {0, 1, 2})
        np.testing.assert_allclose(affine, SRI24_LABEL_AFFINE_RAS_MM)

    def test_label_volume_bounds_align_with_the_bundled_brain_shell(self) -> None:
        lpba40 = load_sri24_parcellation("lpba40")
        indices = np.argwhere(lpba40.volume != 0)
        atlas_ras_mm = (
            indices @ lpba40.affine_ras_mm[:3, :3].T
            + lpba40.affine_ras_mm[:3, 3]
        )
        vertices_mm, _faces = load_mesh(
            SRI24_LABEL_ASSET_DIR.parent / "sri24_brain_gmwm_v1.stl",
            unit_scale=1.0,
        )
        shell_ras_mm = np.asarray(vertices_mm, dtype=float)
        np.testing.assert_allclose(
            atlas_ras_mm.min(axis=0), shell_ras_mm.min(axis=0), atol=3.0
        )
        np.testing.assert_allclose(
            atlas_ras_mm.max(axis=0), shell_ras_mm.max(axis=0), atol=3.0
        )


if __name__ == "__main__":
    unittest.main()
