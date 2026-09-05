"""Small dependency-free NIfTI-1 volume reader.

Field Workbench only needs single-file three-dimensional volumes (optionally
stored with trailing singleton dimensions).  Keeping that focused reader here
avoids adding a full neuroimaging dependency to the desktop release while
allowing atlas labels and UPENN-GBM masks to share one well-tested decoder.
"""

from __future__ import annotations

import gzip
import math
import struct
from pathlib import Path

import numpy as np


class NiftiReadError(ValueError):
    """Raised when a supported NIfTI-1 volume cannot be decoded safely."""


_NIFTI_DTYPES: dict[int, str] = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
}


def _nifti_qform(header: bytes, endian: str, pixdim: tuple[float, ...]) -> np.ndarray:
    b, c, d = struct.unpack_from(endian + "3f", header, 256)
    x, y, z = struct.unpack_from(endian + "3f", header, 268)
    a_squared = 1.0 - (b * b + c * c + d * d)
    a = math.sqrt(max(0.0, a_squared))
    rotation = np.asarray(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
        ],
        dtype=float,
    )
    qfac = -1.0 if pixdim[0] < 0.0 else 1.0
    scales = np.asarray([pixdim[1], pixdim[2], pixdim[3] * qfac], dtype=float)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = rotation * scales[np.newaxis, :]
    affine[:3, 3] = [x, y, z]
    return affine


def read_nifti_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one single-file NIfTI-1 volume and its voxel-to-world affine.

    Plain ``.nii`` and gzip-compressed ``.nii.gz`` files are detected from
    their bytes rather than their filename.  Three spatial dimensions are
    required; additional dimensions are accepted only when they are singleton.
    NIfTI scaling is applied when the header defines a finite non-zero slope.
    """

    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    except (OSError, gzip.BadGzipFile, EOFError) as error:
        raise NiftiReadError(f"Could not read {source.name}: {error}") from error
    if len(payload) < 352:
        raise NiftiReadError("The NIfTI file is incomplete.")

    little = struct.unpack_from("<i", payload, 0)[0]
    big = struct.unpack_from(">i", payload, 0)[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise NiftiReadError("The file is not a NIfTI-1 volume.")
    if payload[344:348] != b"n+1\x00":
        raise NiftiReadError("Only single-file NIfTI-1 volumes are supported.")

    dimensions = struct.unpack_from(endian + "8h", payload, 40)
    rank = int(dimensions[0])
    if rank < 3 or rank > 7:
        raise NiftiReadError("The NIfTI volume has an unsupported rank.")
    shape = tuple(int(value) for value in dimensions[1 : rank + 1])
    if any(value <= 0 for value in shape):
        raise NiftiReadError("The NIfTI volume has invalid dimensions.")
    if any(value != 1 for value in shape[3:]):
        raise NiftiReadError("A single three-dimensional NIfTI volume was expected.")

    datatype = int(struct.unpack_from(endian + "h", payload, 70)[0])
    dtype_code = _NIFTI_DTYPES.get(datatype)
    if dtype_code is None:
        raise NiftiReadError(f"Unsupported NIfTI datatype code {datatype}.")
    dtype = np.dtype(endian + dtype_code)
    bitpix = int(struct.unpack_from(endian + "h", payload, 72)[0])
    if bitpix != dtype.itemsize * 8:
        raise NiftiReadError("The NIfTI datatype and bit depth are inconsistent.")

    voxel_offset = int(round(struct.unpack_from(endian + "f", payload, 108)[0]))
    voxel_count = int(np.prod(shape, dtype=np.int64))
    required = voxel_offset + voxel_count * dtype.itemsize
    if voxel_offset < 352 or required > len(payload):
        raise NiftiReadError("The NIfTI voxel payload is incomplete.")
    volume = np.frombuffer(
        payload,
        dtype=dtype,
        count=voxel_count,
        offset=voxel_offset,
    ).reshape(shape, order="F")
    if rank > 3:
        volume = volume[(slice(None), slice(None), slice(None)) + (0,) * (rank - 3)]

    slope, intercept = struct.unpack_from(endian + "2f", payload, 112)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise NiftiReadError("The NIfTI scaling fields are not finite.")
    if slope != 0.0 and (slope != 1.0 or intercept != 0.0):
        volume = np.asarray(volume, dtype=float) * float(slope) + float(intercept)
    else:
        volume = np.asarray(volume)

    pixdim = struct.unpack_from(endian + "8f", payload, 76)
    qform_code = int(struct.unpack_from(endian + "h", payload, 252)[0])
    sform_code = int(struct.unpack_from(endian + "h", payload, 254)[0])
    if sform_code > 0:
        affine = np.eye(4, dtype=float)
        affine[:3, :] = np.asarray(
            struct.unpack_from(endian + "12f", payload, 280), dtype=float
        ).reshape(3, 4)
    elif qform_code > 0:
        affine = _nifti_qform(payload, endian, pixdim)
    else:
        raise NiftiReadError("The NIfTI file does not define a physical-space affine.")
    if not np.isfinite(affine).all():
        raise NiftiReadError("The NIfTI physical-space affine is not finite.")
    return volume, affine
