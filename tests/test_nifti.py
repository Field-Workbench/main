"""Focused tests for the shared dependency-free NIfTI reader."""

from __future__ import annotations

import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fieldworkbench.nifti import read_nifti_volume
from fieldworkbench.upenn_gbm import read_nifti_segmentation


def _single_file_nifti_bytes(volume: np.ndarray, affine: np.ndarray) -> bytes:
    data = np.asarray(volume)
    dtype_codes = {
        np.dtype("uint8"): (2, 8),
        np.dtype("int16"): (4, 16),
    }
    datatype, bitpix = dtype_codes[data.dtype]
    shape = tuple(int(value) for value in data.shape)
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, len(shape), *shape, *([1] * (7 - len(shape))))
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into("<8f", header, 76, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 254, 1)
    for row, offset in enumerate((280, 296, 312)):
        struct.pack_into("<4f", header, offset, *np.asarray(affine[row], dtype=float))
    header[344:348] = b"n+1\x00"
    return bytes(header) + data.tobytes(order="F")


class NiftiTests(unittest.TestCase):
    def test_reader_detects_plain_and_gzip_payloads_by_content(self) -> None:
        expected = np.zeros((4, 5, 6), dtype=np.int16)
        expected[1, 2, 3] = 17
        affine = np.asarray(
            [
                [-1.0, 0.0, 0.0, 12.0],
                [0.0, 1.0, 0.0, -4.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        payload = _single_file_nifti_bytes(expected, affine)
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "volume.bin"
            compressed = Path(directory) / "volume.also-not-named-nii"
            plain.write_bytes(payload)
            compressed.write_bytes(gzip.compress(payload))
            for path in (plain, compressed):
                actual, actual_affine = read_nifti_volume(path)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_allclose(actual_affine, affine)

    def test_upenn_wrapper_retains_its_grid_affine_and_label_validation(self) -> None:
        expected = np.zeros((240, 240, 155), dtype=np.uint8)
        expected[100:102, 110:112, 70:72] = 1
        expected[102:104, 110:112, 70:72] = 2
        expected[100:102, 112:114, 70:72] = 3
        affine = np.asarray(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 239.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub-002_seg.nii.gz"
            path.write_bytes(gzip.compress(_single_file_nifti_bytes(expected, affine)))
            actual, actual_affine = read_nifti_segmentation(path)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual_affine, affine)


if __name__ == "__main__":
    unittest.main()
