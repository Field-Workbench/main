from __future__ import annotations

import gzip
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from workflow_fixture import compact_workflow_scene

import fieldworkbench.upenn_gbm as upenn_gbm_module
from fieldworkbench.studio_adapter import StudioAdapter
from fieldworkbench.upenn_gbm import (
    UPENN_GBM_CASES,
    UPENN_GBM_REGIONS_BY_KEY,
    UpennGbmImportError,
    binary_mask_to_surface,
    build_upenn_gbm_region_meshes,
    calculate_upenn_gbm_case_statistics,
    find_upenn_gbm_case,
    load_upenn_gbm_catalogue,
    read_nifti_segmentation,
    refresh_upenn_gbm_catalogue,
)


def _edge_health(faces: np.ndarray) -> tuple[int, int, int]:
    edges: dict[tuple[int, int], list[int]] = {}
    for face in np.asarray(faces, dtype=int):
        for start, stop in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            key = (int(min(start, stop)), int(max(start, stop)))
            edges.setdefault(key, []).append(1 if start < stop else -1)
    boundary = sum(len(values) == 1 for values in edges.values())
    non_manifold = sum(len(values) > 2 for values in edges.values())
    inconsistent = sum(
        len(values) == 2 and values[0] == values[1]
        for values in edges.values()
    )
    return boundary, non_manifold, inconsistent


def _write_synthetic_upenn_nifti(path: Path) -> np.ndarray:
    shape = (240, 240, 155)
    volume = np.zeros(shape, dtype=np.uint8)
    volume[100:104, 110:114, 70:74] = 1
    volume[104:108, 110:114, 70:74] = 2
    volume[100:104, 114:118, 70:74] = 3
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 2)
    struct.pack_into("<h", header, 72, 8)
    struct.pack_into("<8f", header, 76, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, -1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, -1.0, 0.0, 239.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, 1.0, 0.0)
    header[344:348] = b"n+1\x00"
    path.write_bytes(gzip.compress(bytes(header) + volume.tobytes(order="F")))
    return volume


class UpennGbmTests(unittest.TestCase):
    def test_catalogue_has_supported_case_ids_and_regions(self) -> None:
        self.assertEqual(len(UPENN_GBM_CASES), 147)
        case = find_upenn_gbm_case("UPENN-GBM-00002")
        self.assertEqual(case.filename, "sub-002_seg.nii.gz")
        self.assertTrue(case.download_url.startswith("https://github.com/data-nih/tcia/"))
        self.assertEqual(
            UPENN_GBM_REGIONS_BY_KEY["tumor_core"].source_values, (1, 3)
        )
        self.assertEqual(
            UPENN_GBM_REGIONS_BY_KEY["whole_tumor"].source_values, (1, 2, 3)
        )
        with self.assertRaises(UpennGbmImportError):
            find_upenn_gbm_case("UPENN-GBM-00001")

    def test_refresh_discovers_valid_release_assets_and_caches_verified_catalogue(self) -> None:
        payload = {
            "id": 12345,
            "updated_at": "2026-08-13T12:00:00Z",
            "assets": [
                {
                    "name": "sub-006_seg.nii.gz",
                    "browser_download_url": (
                        "https://github.com/data-nih/tcia/releases/download/"
                        "upenn-gbm/sub-006_seg.nii.gz"
                    ),
                    "size": 42000,
                    "updated_at": "2026-08-13T11:00:00Z",
                },
                {
                    "name": "participants.tsv",
                    "browser_download_url": "https://example.invalid/participants.tsv",
                    "size": 1000,
                },
                {
                    "name": "sub-008_seg.nii.gz",
                    "browser_download_url": (
                        "https://github.com/data-nih/tcia/releases/download/"
                        "upenn-gbm/sub-008_seg.nii.gz"
                    ),
                    "size": 43000,
                    "updated_at": "2026-08-13T11:00:00Z",
                },
            ],
        }

        class FakeResponse:
            status = 200

            def __init__(self):
                self.headers = {
                    "Content-Length": str(len(json.dumps(payload).encode("utf-8"))),
                    "ETag": '"catalogue-v1"',
                }
                self._data = json.dumps(payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def getcode(self):
                return self.status

            def read(self, _size=-1):
                data, self._data = self._data, b""
                return data

        def verify(case, _cache, _cancel):
            return replace(
                case,
                compatibility="compatible",
                compatibility_detail="Verified in test.",
                approximate_location="Right · posterior · inferior",
                centroid_head_mm=(10.0, -20.0, 25.0),
                left_fraction=0.1,
                right_fraction=0.9,
                tumor_core_volume_ml=12.5,
                whole_tumor_volume_ml=30.0,
                extent_head_mm=(20.0, 30.0, 40.0),
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                upenn_gbm_module, "_verify_upenn_gbm_case", side_effect=verify
            ):
                catalogue = refresh_upenn_gbm_catalogue(
                    directory, urlopen=lambda *_args, **_kwargs: FakeResponse()
                )
            cached = load_upenn_gbm_catalogue(directory)
            with mock.patch.object(
                upenn_gbm_module, "_verify_upenn_gbm_case"
            ) as verify_again:
                repeated = refresh_upenn_gbm_catalogue(
                    directory, urlopen=lambda *_args, **_kwargs: FakeResponse()
                )
                verify_again.assert_not_called()

        self.assertEqual([case.subject_number for case in catalogue.cases], [6, 8])
        self.assertEqual(catalogue.compatible_count, 2)
        self.assertEqual(catalogue.etag, '"catalogue-v1"')
        self.assertEqual(cached.source, "cache")
        self.assertEqual(repeated.compatible_count, 2)
        self.assertEqual([case.filename for case in cached.cases], [
            "sub-006_seg.nii.gz",
            "sub-008_seg.nii.gz",
        ])
        self.assertTrue(all(case.is_importable for case in cached.cases))
        self.assertTrue(all(case.has_case_statistics for case in cached.cases))
        self.assertEqual(cached.cases[0].approximate_location, "Right · posterior · inferior")
        self.assertEqual(cached.cases[0].extent_head_mm, (20.0, 30.0, 40.0))

    def test_nifti_reader_preserves_fortran_voxel_order_and_sri24_affine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub-002_seg.nii.gz"
            expected = _write_synthetic_upenn_nifti(path)
            actual, affine = read_nifti_segmentation(path)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(
            affine,
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 239.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

    def test_case_statistics_report_volume_centroid_distribution_and_extent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub-002_seg.nii.gz"
            segmentation = _write_synthetic_upenn_nifti(path)
            _, affine = read_nifti_segmentation(path)
        statistics = calculate_upenn_gbm_case_statistics(segmentation, affine)
        self.assertAlmostEqual(statistics["tumor_core_volume_ml"], 0.128)
        self.assertAlmostEqual(statistics["whole_tumor_volume_ml"], 0.192)
        self.assertEqual(len(statistics["centroid_head_mm"]), 3)
        self.assertAlmostEqual(
            statistics["left_fraction"] + statistics["right_fraction"], 1.0
        )
        self.assertTrue(all(value > 0.0 for value in statistics["extent_head_mm"]))
        self.assertRegex(
            statistics["approximate_location"],
            r"^(Left|Right|Mostly left|Mostly right|Bilateral) · ",
        )

    def test_marching_tetrahedra_surface_is_closed_and_consistently_oriented(self) -> None:
        mask = np.zeros((8, 9, 10), dtype=bool)
        mask[2:6, 2:7, 3:8] = True
        mask[6, 4, 5] = True
        vertices, faces = binary_mask_to_surface(mask)
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
        self.assertEqual(_edge_health(faces), (0, 0, 0))
        triangles = vertices[faces]
        signed_volume = float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                )
            )
            / 6.0
        )
        self.assertGreater(signed_volume, 0.0)

    def test_case_regions_use_fixed_head_frame_and_preserve_voxel_volume(self) -> None:
        segmentation = np.zeros((20, 20, 20), dtype=np.uint8)
        segmentation[8:12, 9:13, 7:11] = 1
        segmentation[10:14, 10:14, 8:12] = 3
        affine = np.asarray(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 239.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = build_upenn_gbm_region_meshes(
            UPENN_GBM_CASES[0],
            segmentation,
            affine,
            ["tumor_core"],
        )[0]
        expected_voxels = int(np.count_nonzero(np.isin(segmentation, (1, 3))))
        self.assertEqual(result["voxel_count"], expected_voxels)
        self.assertAlmostEqual(result["segmented_volume_ml"], expected_voxels / 1000.0)
        vertices = np.asarray(result["vertices"])
        self.assertLess(float(np.max(np.abs(vertices))), 1.0)
        self.assertEqual(_edge_health(np.asarray(result["faces"])), (0, 0, 0))

    def test_adapter_adds_case_as_measurement_geometry_in_one_undo_step(self) -> None:
        segmentation = np.zeros((12, 12, 12), dtype=np.uint8)
        segmentation[4:8, 4:8, 4:8] = 1
        affine = np.eye(4)
        definition = build_upenn_gbm_region_meshes(
            UPENN_GBM_CASES[0], segmentation, affine, ["tumor_core"]
        )
        payload = {
            "case_id": "UPENN-GBM-00002",
            "source_filename": "sub-002_seg.nii.gz",
            "source_url": "https://example.invalid/sub-002_seg.nii.gz",
            "dataset_url": "https://example.invalid/upenn-gbm",
            "dataset_doi": "10.7937/TCIA.709X-DN49",
            "license": "CC BY 4.0",
            "coordinate_frame": "NAS/LPA/RPA head frame",
            "source_coordinate_frame": "UPENN-GBM preprocessed SRI24 RAS millimetres",
            "meshes": definition,
        }
        adapter = StudioAdapter(compact_workflow_scene())
        history_before = adapter.history()["undo"]
        object_ids = adapter.add_upenn_gbm_case(payload, role="measurement")
        self.assertEqual(adapter.history()["undo"], history_before + 1)
        self.assertEqual(len(object_ids), 1)
        object_id = object_ids[0]
        self.assertEqual(adapter.object_type_label(object_id), "UPENN-GBM tumor")
        self.assertTrue(adapter.supports_measurement(object_id))
        properties = adapter.geometry_properties(object_id)
        self.assertEqual(properties["role"], "measurement")
        self.assertEqual(properties["upenn_gbm_case"], "UPENN-GBM-00002")
        self.assertEqual(properties["coordinate_frame"], "NAS/LPA/RPA head frame")
        adapter.undo()
        self.assertFalse(any(item["id"] == object_id for item in adapter.list_objects()))


if __name__ == "__main__":
    unittest.main()
