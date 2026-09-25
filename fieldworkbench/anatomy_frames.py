"""Canonical anatomy coordinate-frame transforms used by bundled assets.

The SRI24 brain asset is intentionally stored in the atlas's original RAS
millimetre coordinates.  Field Workbench transforms it at insertion time into
the same NAS/LPA/RPA head frame used by the Generic bust.  Keeping this
mapping explicit also provides the transform required by future SRI24-aligned
clinical/atlas targets.
"""

from __future__ import annotations

import numpy as np


# Fiducials marked on SRI24 v2.0 spgr_unstrip.nii in 3D Slicer, using the
# same convention as the Generic bust: nasion plus left/right helix-tragus
# junctions. Coordinates are Slicer/SRI24 RAS millimetres.
SRI24_HEAD_FIDUCIALS_RAS_MM: dict[str, tuple[float, float, float]] = {
    "NAS": (-120.066, 214.028, 50.000),
    "LPA": (-195.809, 113.054, 27.190),
    "RPA": (-36.738, 113.125, 27.439),
}

# Rigid affine mapping SRI24 RAS millimetres -> canonical NAS/LPA/RPA head
# millimetres.  The head frame is defined by:
#   origin = projection of NAS onto the LPA->RPA line
#   +X     = LPA -> RPA (model/patient right)
#   +Y     = anterior, toward NAS and orthogonal to +X
#   +Z     = superior, completing a right-handed frame
# No scale or non-rigid warp is applied.
SRI24_RAS_TO_HEAD_AFFINE_MM = np.asarray(
    [
        [0.999998675, 0.000446341, 0.001565337, 119.89204464],
        [-0.000778793, 0.975651307, 0.219326058, -116.41725317],
        [-0.001429330, -0.219326990, 0.975650360, -2.01201562],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def sri24_ras_mm_to_head_mm(points_mm: np.ndarray | list[list[float]]) -> np.ndarray:
    """Transform one or more SRI24 RAS points into canonical head millimetres."""
    points = np.asarray(points_mm, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("SRI24 points must have three coordinates per point.")
    rotation = SRI24_RAS_TO_HEAD_AFFINE_MM[:3, :3]
    translation = SRI24_RAS_TO_HEAD_AFFINE_MM[:3, 3]
    return points @ rotation.T + translation


def sri24_ras_mm_to_head_m(points_mm: np.ndarray | list[list[float]]) -> np.ndarray:
    """Transform SRI24 RAS millimetres into the Workbench SI-metre head frame."""
    return sri24_ras_mm_to_head_mm(points_mm) / 1000.0
