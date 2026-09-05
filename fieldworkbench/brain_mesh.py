"""Cached LPBA40 visual shells for Brain View.

The field/statistics path remains voxel/sample based. Runtime code only loads
the checked-in canonical-head mesh cache and applies the selected SRI24 brain's
scene transform. Regenerating that cache is an explicit developer operation in
``tools/build_lpba40_mesh_cache.py`` and is the only place VTK is required.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .sri24_atlas import (
    SRI24_LABEL_ASSET_DIR,
    SRI24_LABEL_MANIFEST,
    SRI24Parcellation,
    load_sri24_parcellation,
)


LPBA40_MESH_CACHE_VERSION = 1
LPBA40_MESH_CACHE_STRIDE = 2
LPBA40_MESH_CACHE_FRAME = "field_workbench_canonical_head_mm"
LPBA40_MESH_CACHE_RESOURCE = SRI24_LABEL_ASSET_DIR / "lpba40_meshes.npz"
_LPBA40_SOURCE_VOLUME = SRI24_LABEL_ASSET_DIR / "lpba40.nii.gz"
_LPBA40_SOURCE_LABELS = SRI24_LABEL_ASSET_DIR / "LPBA40-labels.txt"


class BrainMeshError(ValueError):
    """Raised when the bundled LPBA40 display-shell cache is invalid."""


@dataclass(frozen=True)
class BrainRegionMesh:
    """One placed LPBA40 label shell ready for a Plotly ``mesh3d`` trace."""

    label_id: int
    vertices_mm: np.ndarray
    triangles: np.ndarray

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))


@dataclass(frozen=True)
class LPBA40MeshSet:
    """Validated visual shells for all non-background LPBA40 labels."""

    regions: dict[int, BrainRegionMesh]
    source_stride: int
    source_shape: tuple[int, int, int]
    cache_version: int = LPBA40_MESH_CACHE_VERSION
    coordinate_frame: str = LPBA40_MESH_CACHE_FRAME

    @property
    def vertex_count(self) -> int:
        return int(sum(len(mesh.vertices_mm) for mesh in self.regions.values()))

    @property
    def triangle_count(self) -> int:
        return int(sum(mesh.triangle_count for mesh in self.regions.values()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BrainMeshError(f"Could not read the bundled brain mesh asset: {error}") from error
    return digest.hexdigest()


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise BrainMeshError(f"LPBA40 mesh cache field {name!r} is not a string scalar.")
    return str(array.reshape(()).item())


def _scalar_int(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise BrainMeshError(f"LPBA40 mesh cache field {name!r} is not an integer scalar.")
    return int(array.reshape(()).item())


def _manifest_mesh_metadata() -> dict[str, object]:
    try:
        manifest = json.loads(SRI24_LABEL_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BrainMeshError(f"Could not read the SRI24 asset manifest: {error}") from error
    assets = manifest.get("assets")
    metadata = assets.get(LPBA40_MESH_CACHE_RESOURCE.name) if isinstance(assets, dict) else None
    if not isinstance(metadata, dict):
        raise BrainMeshError("The SRI24 asset manifest does not describe lpba40_meshes.npz.")
    return metadata


def _validated_placement(
    position_m: Iterable[float],
    rotation: np.ndarray,
    scale_xyz: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(position_m, dtype=float)
    orientation = np.asarray(rotation, dtype=float)
    scale = np.asarray(scale_xyz, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise BrainMeshError("Brain mesh placement requires a finite XYZ position.")
    if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
        raise BrainMeshError("Brain mesh placement requires a finite 3×3 rotation.")
    if not np.allclose(orientation @ orientation.T, np.eye(3), atol=1e-6):
        raise BrainMeshError("Brain mesh placement rotation is not orthonormal.")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise BrainMeshError("Brain mesh placement scale must contain three positive values.")
    return position, orientation, scale


@lru_cache(maxsize=1)
def _load_canonical_lpba40_meshes() -> LPBA40MeshSet:
    """Load and exhaustively validate the immutable canonical mesh asset once."""

    metadata = _manifest_mesh_metadata()
    expected_cache_hash = metadata.get("sha256")
    if not isinstance(expected_cache_hash, str) or _sha256(LPBA40_MESH_CACHE_RESOURCE) != expected_cache_hash:
        raise BrainMeshError("Bundled LPBA40 mesh-cache checksum mismatch.")

    required = {
        "mesh_version",
        "atlas_key",
        "coordinate_frame",
        "source_stride",
        "source_shape",
        "source_volume_sha256",
        "source_label_table_sha256",
        "label_ids",
        "vertices",
        "vertex_offsets",
        "triangles",
        "triangle_offsets",
    }
    try:
        with np.load(LPBA40_MESH_CACHE_RESOURCE, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise BrainMeshError(
                    "LPBA40 mesh cache is missing fields: " + ", ".join(missing)
                )
            mesh_version = _scalar_int(archive["mesh_version"], "mesh_version")
            atlas_key = _scalar_string(archive["atlas_key"], "atlas_key")
            coordinate_frame = _scalar_string(
                archive["coordinate_frame"], "coordinate_frame"
            )
            source_stride = _scalar_int(archive["source_stride"], "source_stride")
            source_shape_values = np.asarray(archive["source_shape"])
            source_shape = tuple(int(value) for value in source_shape_values.tolist())
            source_volume_sha256 = _scalar_string(
                archive["source_volume_sha256"], "source_volume_sha256"
            )
            source_label_sha256 = _scalar_string(
                archive["source_label_table_sha256"], "source_label_table_sha256"
            )
            label_ids = np.asarray(archive["label_ids"], dtype=np.int32)
            vertices = np.asarray(archive["vertices"], dtype=np.float32)
            vertex_offsets = np.asarray(archive["vertex_offsets"], dtype=np.int64)
            triangles = np.asarray(archive["triangles"], dtype=np.int32)
            triangle_offsets = np.asarray(archive["triangle_offsets"], dtype=np.int64)
    except BrainMeshError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise BrainMeshError(f"Could not load the bundled LPBA40 mesh cache: {error}") from error

    if mesh_version != LPBA40_MESH_CACHE_VERSION:
        raise BrainMeshError(
            f"LPBA40 mesh cache version {mesh_version} is unsupported; "
            f"expected {LPBA40_MESH_CACHE_VERSION}."
        )
    if atlas_key != "lpba40" or coordinate_frame != LPBA40_MESH_CACHE_FRAME:
        raise BrainMeshError("LPBA40 mesh cache atlas or coordinate frame is invalid.")
    if source_stride != LPBA40_MESH_CACHE_STRIDE or source_shape != (120, 120, 78):
        raise BrainMeshError("LPBA40 mesh cache stride or source shape is invalid.")
    if source_volume_sha256 != _sha256(_LPBA40_SOURCE_VOLUME):
        raise BrainMeshError("LPBA40 mesh cache is stale relative to lpba40.nii.gz.")
    if source_label_sha256 != _sha256(_LPBA40_SOURCE_LABELS):
        raise BrainMeshError("LPBA40 mesh cache is stale relative to LPBA40-labels.txt.")
    if label_ids.ndim != 1 or len(label_ids) != 56 or len(np.unique(label_ids)) != len(label_ids):
        raise BrainMeshError("LPBA40 mesh cache does not contain 56 unique label ids.")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise BrainMeshError("LPBA40 mesh-cache vertices are invalid.")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise BrainMeshError("LPBA40 mesh-cache triangles are invalid.")
    if vertex_offsets.shape != (len(label_ids) + 1,) or triangle_offsets.shape != (
        len(label_ids) + 1,
    ):
        raise BrainMeshError("LPBA40 mesh-cache offsets do not match its labels.")
    if (
        vertex_offsets[0] != 0
        or triangle_offsets[0] != 0
        or vertex_offsets[-1] != len(vertices)
        or triangle_offsets[-1] != len(triangles)
        or np.any(np.diff(vertex_offsets) <= 3)
        or np.any(np.diff(triangle_offsets) <= 1)
    ):
        raise BrainMeshError("LPBA40 mesh-cache offsets are invalid.")

    atlas = load_sri24_parcellation("lpba40")
    expected_labels = tuple(
        int(value) for value in atlas.present_label_ids if int(value) != 0
    )
    if tuple(int(value) for value in label_ids) != expected_labels:
        raise BrainMeshError("LPBA40 mesh-cache labels disagree with the bundled atlas.")

    regions: dict[int, BrainRegionMesh] = {}
    for index, raw_label_id in enumerate(label_ids):
        vertex_start, vertex_end = map(int, vertex_offsets[index : index + 2])
        triangle_start, triangle_end = map(int, triangle_offsets[index : index + 2])
        local_vertices = np.ascontiguousarray(vertices[vertex_start:vertex_end])
        local_triangles = np.ascontiguousarray(triangles[triangle_start:triangle_end])
        if np.any(local_triangles < 0) or np.any(local_triangles >= len(local_vertices)):
            raise BrainMeshError(
                f"LPBA40 mesh-cache label {int(raw_label_id)} has invalid triangle indices."
            )
        local_vertices.setflags(write=False)
        local_triangles.setflags(write=False)
        regions[int(raw_label_id)] = BrainRegionMesh(
            label_id=int(raw_label_id),
            vertices_mm=local_vertices,
            triangles=local_triangles,
        )

    mesh_set = LPBA40MeshSet(
        regions=regions,
        source_stride=source_stride,
        source_shape=source_shape,
        cache_version=mesh_version,
        coordinate_frame=coordinate_frame,
    )
    expected_vertices = int(metadata.get("vertex_count", -1))
    expected_triangles = int(metadata.get("triangle_count", -1))
    if (
        mesh_set.vertex_count != expected_vertices
        or mesh_set.triangle_count != expected_triangles
    ):
        raise BrainMeshError("LPBA40 mesh-cache counts disagree with the asset manifest.")
    return mesh_set


def load_lpba40_region_meshes(
    parcellation: SRI24Parcellation | None = None,
    *,
    position_m: Iterable[float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
    scale_xyz: Iterable[float] = (1.0, 1.0, 1.0),
    visual_stride: int = LPBA40_MESH_CACHE_STRIDE,
) -> LPBA40MeshSet:
    """Load cached LPBA40 shells and apply one selected-brain placement.

    The cache is authored in canonical SRI24/head millimetres. ``visual_stride``
    remains in the API to make a stale caller fail explicitly; changing it now
    requires regenerating and checking in the cache with the developer tool.
    """

    atlas = parcellation or load_sri24_parcellation("lpba40")
    if atlas.key != "lpba40":
        raise BrainMeshError("Brain surface loading currently supports LPBA40 only.")
    stride = int(visual_stride)
    if stride != LPBA40_MESH_CACHE_STRIDE:
        raise BrainMeshError(
            f"The bundled LPBA40 mesh cache uses visual stride {LPBA40_MESH_CACHE_STRIDE}; "
            "regenerate the developer asset to use a different stride."
        )
    position, orientation, scale = _validated_placement(
        position_m,
        np.eye(3) if rotation is None else rotation,
        scale_xyz,
    )
    canonical = _load_canonical_lpba40_meshes()
    expected_labels = {
        int(value) for value in atlas.present_label_ids if int(value) != 0
    }
    if set(canonical.regions) != expected_labels:
        raise BrainMeshError("LPBA40 mesh-cache labels disagree with the requested atlas.")

    identity_placement = (
        np.array_equal(position, np.zeros(3))
        and np.array_equal(orientation, np.eye(3))
        and np.array_equal(scale, np.ones(3))
    )
    if identity_placement:
        return canonical

    regions: dict[int, BrainRegionMesh] = {}
    for label_id, mesh in canonical.regions.items():
        placed_vertices = np.asarray(
            (mesh.vertices_mm.astype(float) * scale) @ orientation.T
            + position * 1000.0,
            dtype=np.float32,
        )
        placed_vertices.setflags(write=False)
        regions[label_id] = BrainRegionMesh(
            label_id=label_id,
            vertices_mm=placed_vertices,
            triangles=mesh.triangles,
        )
    return LPBA40MeshSet(
        regions=regions,
        source_stride=canonical.source_stride,
        source_shape=canonical.source_shape,
        cache_version=canonical.cache_version,
        coordinate_frame=canonical.coordinate_frame,
    )


def build_lpba40_region_meshes(*args, **kwargs) -> LPBA40MeshSet:
    """Backward-compatible name for the runtime cache loader/placement path."""

    return load_lpba40_region_meshes(*args, **kwargs)
