"""Derived inward volume sampling for authored surface-region meshes.

A :class:`~fieldworkbench.surface_region_geometry.SurfaceRegionGeometry` owns
surface classification and exact geometric queries.  This module turns that
surface into a deterministic, interior-only voxel-centre integration shell.
The shell is intentionally source-agnostic: once prepared it is represented by
``AnalysisRegionSet``/``AnalysisSampleSet`` and can be consumed by the same
regional analysis engine as atlas or ordinary measurement volumes.

The dense source lattice exists for geometry/volume accounting.  Field
calculation points are deterministically capped per authored region and each
retained point carries the physical volume represented by its source voxels,
mirroring the LPBA40 sampling strategy.  A source voxel belongs to exactly one
region: the nearest authored surface triangle wins, with the geometry provider's
stable original-face tie break.  Unclassified surface ownership is omitted from
named-region analysis and reported explicitly in metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .spatial_analysis import AnalysisRegionNode, AnalysisRegionSet, AnalysisSampleSet
from .surface_region_geometry import (
    SURFACE_REGION_GEOMETRY_VERSION,
    SurfaceRegionGeometry,
    SurfaceRegionGeometryError,
)


SURFACE_REGION_VOLUME_VERSION = 2
DEFAULT_SURFACE_DEPTH_MM = 2.0
MINIMUM_SURFACE_DEPTH_MM = 0.01
MAXIMUM_SURFACE_DEPTH_MM = 30.0
DEFAULT_TARGET_SOURCE_VOXELS = 60_000
DEFAULT_MAXIMUM_LATTICE_CELLS = 8_000_000
DEFAULT_MAXIMUM_SAMPLES_PER_REGION = 900


class SurfaceRegionVolumeError(ValueError):
    """Raised when a near-surface volume cannot be generated reliably."""


@dataclass(frozen=True)
class SurfaceRegionVolumeSettings:
    """Numerical controls for one derived inward surface-region shell.

    ``depth_mm`` is the only setting intended to become a normal user-facing
    control.  The remaining controls bound geometry work and retained field
    samples. ``grid_spacing_mm`` is primarily a validation/testing override;
    when omitted an isotropic target spacing is selected from surface area,
    requested depth, source-voxel budget, and lattice-memory budget.
    """

    depth_mm: float = DEFAULT_SURFACE_DEPTH_MM
    maximum_samples_per_region: int = DEFAULT_MAXIMUM_SAMPLES_PER_REGION
    target_source_voxels: int = DEFAULT_TARGET_SOURCE_VOXELS
    maximum_lattice_cells: int = DEFAULT_MAXIMUM_LATTICE_CELLS
    grid_spacing_mm: float | None = None

    def validated(self) -> "SurfaceRegionVolumeSettings":
        try:
            depth = float(self.depth_mm)
        except (TypeError, ValueError) as error:
            raise SurfaceRegionVolumeError("Surface sampling depth must be numeric.") from error
        if (
            not math.isfinite(depth)
            or depth < MINIMUM_SURFACE_DEPTH_MM
            or depth > MAXIMUM_SURFACE_DEPTH_MM
        ):
            raise SurfaceRegionVolumeError(
                "Surface sampling depth must be between "
                f"{MINIMUM_SURFACE_DEPTH_MM:g} and {MAXIMUM_SURFACE_DEPTH_MM:g} mm."
            )

        try:
            retained = int(self.maximum_samples_per_region)
            target = int(self.target_source_voxels)
            lattice = int(self.maximum_lattice_cells)
        except (TypeError, ValueError) as error:
            raise SurfaceRegionVolumeError("Surface sampling limits must be integers.") from error
        if retained < 1:
            raise SurfaceRegionVolumeError("At least one retained sample per surface region is required.")
        if target < 1:
            raise SurfaceRegionVolumeError("The source-voxel target must be positive.")
        if lattice < 1_000:
            raise SurfaceRegionVolumeError("The surface-volume lattice limit is too small.")

        spacing: float | None
        if self.grid_spacing_mm is None:
            spacing = None
        else:
            try:
                spacing = float(self.grid_spacing_mm)
            except (TypeError, ValueError) as error:
                raise SurfaceRegionVolumeError("Surface-volume grid spacing must be numeric.") from error
            if not math.isfinite(spacing) or spacing <= 0.0:
                raise SurfaceRegionVolumeError(
                    "Surface-volume grid spacing must be finite and greater than zero."
                )

        return SurfaceRegionVolumeSettings(
            depth_mm=depth,
            maximum_samples_per_region=retained,
            target_source_voxels=target,
            maximum_lattice_cells=lattice,
            grid_spacing_mm=spacing,
        )


@dataclass(frozen=True)
class _Lattice:
    counts_xyz: np.ndarray
    cell_size_m: np.ndarray
    bounds_min_m: np.ndarray

    @property
    def cell_volume_m3(self) -> float:
        return float(np.prod(self.cell_size_m))

    @property
    def cell_volume_mm3(self) -> float:
        return self.cell_volume_m3 * 1e9

    @property
    def total_cells(self) -> int:
        return int(np.prod(self.counts_xyz, dtype=np.int64))

    def points_for_indices(self, indices: np.ndarray) -> np.ndarray:
        return self.bounds_min_m + (np.asarray(indices, dtype=float) + 0.5) * self.cell_size_m


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    completed: int,
    total: int,
    detail: str = "",
) -> None:
    if callable(callback):
        callback(
            {
                "stage": str(stage),
                "completed": int(completed),
                "total": max(1, int(total)),
                "unit": "geometry work",
                "detail": str(detail),
            }
        )


def _check_cancel(cancel_requested: Callable[[], bool] | None) -> None:
    if callable(cancel_requested) and cancel_requested():
        raise SurfaceRegionVolumeError("Surface-region volume generation was cancelled.")


def _geometry_sha256(geometry: SurfaceRegionGeometry) -> str:
    """Return a deterministic identity for transformed geometry + authored ownership."""

    digest = hashlib.sha256()
    digest.update(f"surface-region-geometry:{SURFACE_REGION_GEOMETRY_VERSION}\0".encode("ascii"))
    for value, dtype in (
        (geometry.vertices_m, "<f8"),
        (geometry.faces, "<i8"),
        (geometry.face_region_labels, "<i4"),
    ):
        array = np.ascontiguousarray(np.asarray(value).astype(dtype, copy=False))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    for region in geometry.regions:
        digest.update(str(region.label_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(region.key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(region.name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def surface_region_volume_cache_identity(
    geometry: SurfaceRegionGeometry,
    settings: SurfaceRegionVolumeSettings | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the pre-build cache identity for one inward-shell request.

    Unlike ``derived_geometry_identity`` this identity is available *before* the
    expensive lattice is generated.  It therefore acts as the shared geometry
    cache lookup key.  Provider/geometry versions are part of the key so a future
    algorithm change never reuses an older shell silently.
    """

    if not isinstance(geometry, SurfaceRegionGeometry):
        raise SurfaceRegionVolumeError("A prepared SurfaceRegionGeometry provider is required.")
    config = (settings or SurfaceRegionVolumeSettings()).validated()
    payload = {
        "provider": "surface_region_volume",
        "provider_version": SURFACE_REGION_VOLUME_VERSION,
        "geometry_version": SURFACE_REGION_GEOMETRY_VERSION,
        "geometry_sha256": _geometry_sha256(geometry),
        "depth_mm": float(config.depth_mm),
        "maximum_samples_per_region": int(config.maximum_samples_per_region),
        "target_source_voxels": int(config.target_source_voxels),
        "maximum_lattice_cells": int(config.maximum_lattice_cells),
        "grid_spacing_mm": (
            None if config.grid_spacing_mm is None else float(config.grid_spacing_mm)
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _choose_lattice(
    geometry: SurfaceRegionGeometry,
    settings: SurfaceRegionVolumeSettings,
) -> tuple[_Lattice, float]:
    dimensions = np.asarray(geometry.bounds_max_m - geometry.bounds_min_m, dtype=float)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise SurfaceRegionVolumeError("Surface-region geometry has invalid world-space bounds.")

    depth_m = float(settings.depth_mm) / 1000.0
    box_volume = float(np.prod(dimensions))
    surface_area = float(sum(geometry.region_surface_area_m2.values()))
    if surface_area <= 0.0 or not math.isfinite(surface_area):
        raise SurfaceRegionVolumeError("Surface-region geometry has no finite authored surface area.")

    if settings.grid_spacing_mm is not None:
        requested_spacing_m = float(settings.grid_spacing_mm) / 1000.0
    else:
        # area × depth is a useful first-order shell-volume estimate.  The
        # bounding-box term separately prevents an enormous dense lattice for
        # very thin requested shells.
        estimated_shell_volume = min(box_volume, max(surface_area * depth_m, 1e-18))
        source_spacing = (estimated_shell_volume / settings.target_source_voxels) ** (1.0 / 3.0)
        memory_spacing = (box_volume / settings.maximum_lattice_cells) ** (1.0 / 3.0)
        requested_spacing_m = max(source_spacing, memory_spacing, 1e-7)

    counts = _stable_lattice_counts(dimensions, requested_spacing_m)
    total = int(np.prod(counts, dtype=np.int64))
    if total > settings.maximum_lattice_cells:
        if settings.grid_spacing_mm is not None:
            raise SurfaceRegionVolumeError(
                f"The requested {settings.grid_spacing_mm:g} mm grid would require "
                f"{total:,} lattice cells; the configured limit is "
                f"{settings.maximum_lattice_cells:,}."
            )
        # Integer ceil can leave the first estimate just above the memory cap.
        # Expand the requested spacing until the actual rectangular lattice fits.
        scale = (total / settings.maximum_lattice_cells) ** (1.0 / 3.0)
        requested_spacing_m *= max(1.001, scale * 1.001)
        counts = _stable_lattice_counts(dimensions, requested_spacing_m)
        while int(np.prod(counts, dtype=np.int64)) > settings.maximum_lattice_cells:
            requested_spacing_m *= 1.01
            counts = _stable_lattice_counts(dimensions, requested_spacing_m)

    # Divide the exact object bounds into equal rectangular cells.  This keeps
    # the lattice attached to the object under translations instead of to an
    # arbitrary world-origin phase.  Cell dimensions differ slightly by axis but
    # every source voxel has the same volume.
    cell_size = dimensions / counts.astype(float)
    return (
        _Lattice(
            counts_xyz=counts,
            cell_size_m=cell_size,
            bounds_min_m=np.asarray(geometry.bounds_min_m, dtype=float),
        ),
        requested_spacing_m,
    )


def _stable_lattice_counts(dimensions_m: np.ndarray, spacing_m: float) -> np.ndarray:
    """Return ceil-based lattice counts without floating-point boundary flips.

    The lattice is attached to object bounds, so translating a rigid object must
    not change its voxel count.  Subtracting translated floating-point bounds can
    produce a ratio such as ``20.000000000000004`` instead of the mathematically
    exact ``20``; a raw ``ceil`` would then create a 21st cell.  Snap ratios that
    are only round-off away from an integer before applying ceil.  The 1e-12
    relative tolerance is many orders below any meaningful scene/model length
    change but comfortably above ordinary double-precision subtraction noise.
    """

    dimensions = np.asarray(dimensions_m, dtype=float)
    ratio = dimensions / float(spacing_m)
    nearest = np.rint(ratio)
    tolerance = 1.0e-12 * np.maximum(1.0, np.abs(ratio))
    stable_ratio = np.where(np.abs(ratio - nearest) <= tolerance, nearest, ratio)
    return np.maximum(1, np.ceil(stable_ratio).astype(np.int64))


def _points_to_triangle_distance_squared(
    points: np.ndarray,
    triangle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact point-to-triangle squared distance for many points and one face.

    The second result marks points whose closest point is strictly within the
    triangle face.  For consistently oriented meshes those points can be
    classified as inward/outward from the face normal; edge/vertex closest
    points use the provider's parity classifier to avoid concave-seam errors.
    """

    points = np.asarray(points, dtype=float)
    a, b, c = np.asarray(triangle, dtype=float)
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_sq = float(np.dot(normal, normal))
    if normal_sq <= 1e-30:
        return np.full(len(points), math.inf), np.zeros(len(points), dtype=bool)

    ap = points - a
    plane_fraction = (ap @ normal) / normal_sq
    projected = points - plane_fraction[:, None] * normal

    d00 = float(np.dot(ab, ab))
    d01 = float(np.dot(ab, ac))
    d11 = float(np.dot(ac, ac))
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= 1e-30:
        return np.full(len(points), math.inf), np.zeros(len(points), dtype=bool)
    rel = projected - a
    d20 = rel @ ab
    d21 = rel @ ac
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1.0 - v - w
    inclusive = (u >= -1e-12) & (v >= -1e-12) & (w >= -1e-12)
    strict = (u > 1e-9) & (v > 1e-9) & (w > 1e-9)

    squared = np.empty(len(points), dtype=float)
    if np.any(inclusive):
        delta = points[inclusive] - projected[inclusive]
        squared[inclusive] = np.einsum("ij,ij->i", delta, delta)

    outside = ~inclusive
    if np.any(outside):
        query = points[outside]
        best = np.full(len(query), math.inf, dtype=float)
        for start, stop in ((a, b), (b, c), (c, a)):
            direction = stop - start
            direction_sq = float(np.dot(direction, direction))
            fraction = np.clip(((query - start) @ direction) / direction_sq, 0.0, 1.0)
            closest = start + fraction[:, None] * direction
            delta = query - closest
            best = np.minimum(best, np.einsum("ij,ij->i", delta, delta))
        squared[outside] = best
    return squared, strict


def _surface_candidates(
    geometry: SurfaceRegionGeometry,
    lattice: _Lattice,
    depth_m: float,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve every lattice centre whose exact nearest surface is within depth."""

    shape = tuple(int(value) for value in lattice.counts_xyz)
    best_squared = np.full(shape, np.inf, dtype=np.float32)
    best_face = np.full(shape, -1, dtype=np.int32)
    best_face_interior = np.zeros(shape, dtype=bool)
    depth_squared = (depth_m + geometry.surface_tolerance_m) ** 2
    tie_squared = max(float(geometry.surface_tolerance_m) ** 2, 1e-24)
    total_faces = len(geometry.faces)

    _emit_progress(
        progress_callback,
        stage="Building near-surface integration lattice…",
        completed=0,
        total=total_faces,
        detail=f"Scanning {total_faces:,} surface triangles.",
    )
    for face_index, triangle in enumerate(geometry.triangles_m):
        if face_index % 256 == 0:
            _check_cancel(cancel_requested)
            _emit_progress(
                progress_callback,
                stage="Building near-surface integration lattice…",
                completed=face_index,
                total=total_faces,
                detail=f"Scanning surface triangle {face_index + 1:,}/{total_faces:,}.",
            )
        expanded_min = np.min(triangle, axis=0) - depth_m
        expanded_max = np.max(triangle, axis=0) + depth_m
        lower = np.ceil(
            (expanded_min - lattice.bounds_min_m) / lattice.cell_size_m - 0.5
        ).astype(np.int64)
        upper = np.floor(
            (expanded_max - lattice.bounds_min_m) / lattice.cell_size_m - 0.5
        ).astype(np.int64)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, lattice.counts_xyz - 1)
        if np.any(upper < lower):
            continue

        axes = [np.arange(lower[i], upper[i] + 1, dtype=np.int64) for i in range(3)]
        indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        points = lattice.points_for_indices(indices)
        squared, face_interior = _points_to_triangle_distance_squared(points, triangle)
        within = squared <= depth_squared
        if not np.any(within):
            continue
        selected = indices[within]
        candidate_squared = squared[within].astype(np.float32)
        candidate_interior = face_interior[within]
        current = best_squared[selected[:, 0], selected[:, 1], selected[:, 2]]
        # Faces are visited in original-index order.  Equal-distance ties
        # therefore retain the lower original index, matching nearest_surface().
        update = candidate_squared < (current - tie_squared)
        if not np.any(update):
            continue
        selected = selected[update]
        candidate_squared = candidate_squared[update]
        candidate_interior = candidate_interior[update]
        best_squared[selected[:, 0], selected[:, 1], selected[:, 2]] = candidate_squared
        best_face[selected[:, 0], selected[:, 1], selected[:, 2]] = int(face_index)
        best_face_interior[selected[:, 0], selected[:, 1], selected[:, 2]] = candidate_interior

    _check_cancel(cancel_requested)
    _emit_progress(
        progress_callback,
        stage="Classifying inward surface volume…",
        completed=total_faces,
        total=total_faces,
        detail="Surface-distance lattice complete; resolving interior ownership.",
    )
    candidate_indices = np.argwhere(best_face >= 0).astype(np.int64)
    if not len(candidate_indices):
        raise SurfaceRegionVolumeError(
            "No lattice centres fell within the requested surface depth. Increase the depth "
            "or source-voxel density."
        )
    points = lattice.points_for_indices(candidate_indices)
    faces = best_face[candidate_indices[:, 0], candidate_indices[:, 1], candidate_indices[:, 2]]
    face_interior = best_face_interior[
        candidate_indices[:, 0], candidate_indices[:, 1], candidate_indices[:, 2]
    ]
    squared = best_squared[candidate_indices[:, 0], candidate_indices[:, 1], candidate_indices[:, 2]]

    inside = np.zeros(len(points), dtype=bool)
    orientation = str(geometry.health.get("orientation", "unknown"))
    can_use_normals = bool(geometry.health.get("consistently_oriented")) and orientation in {
        "outward",
        "inward",
    }
    if can_use_normals:
        easy = np.asarray(face_interior, dtype=bool)
        if np.any(easy):
            selected_triangles = geometry.triangles_m[faces[easy]]
            normals = np.cross(
                selected_triangles[:, 1] - selected_triangles[:, 0],
                selected_triangles[:, 2] - selected_triangles[:, 0],
            )
            side = np.einsum(
                "ij,ij->i",
                points[easy] - selected_triangles[:, 0],
                normals,
            )
            inside[easy] = side < 0.0 if orientation == "outward" else side > 0.0
        hard = ~easy
        if np.any(hard):
            inside[hard] = geometry.points_inside(points[hard])
    else:
        inside = geometry.points_inside(points)

    on_surface = squared <= float(geometry.surface_tolerance_m) ** 2
    keep = inside | on_surface
    return candidate_indices[keep], points[keep], faces[keep], squared[keep]


def _select_retained_samples(
    source_indices: np.ndarray,
    source_points: np.ndarray,
    source_labels: np.ndarray,
    *,
    maximum_samples_per_region: int,
    cell_volume_mm3: float,
) -> AnalysisSampleSet:
    selected_positions: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    selected_source: list[np.ndarray] = []
    selected_weights: list[np.ndarray] = []
    source_counts: dict[int, int] = {}
    sampled_counts: dict[int, int] = {}
    label_volumes: dict[int, float] = {}

    flat_source = np.asarray(source_indices, dtype=np.int64)
    labels = np.asarray(source_labels, dtype=np.int32)
    for label in sorted(int(value) for value in np.unique(labels)):
        members = np.flatnonzero(labels == label)
        count = int(len(members))
        retained = min(count, int(maximum_samples_per_region))
        if retained == count:
            chosen = members
        else:
            offsets = np.floor(
                (np.arange(retained, dtype=float) + 0.5) * count / retained
            ).astype(np.int64)
            chosen = members[offsets]
        selected_positions.append(source_points[chosen])
        selected_labels.append(np.full(retained, label, dtype=np.int32))
        selected_source.append(flat_source[chosen])
        represented = count * cell_volume_mm3 / retained
        selected_weights.append(np.full(retained, represented, dtype=float))
        source_counts[label] = count
        sampled_counts[label] = retained
        label_volumes[label] = count * cell_volume_mm3

    return AnalysisSampleSet(
        points_m=np.concatenate(selected_positions),
        label_ids=np.concatenate(selected_labels),
        weights_mm3=np.concatenate(selected_weights),
        source_indices=np.concatenate(selected_source),
        source_counts=source_counts,
        sampled_counts=sampled_counts,
        nominal_element_volume_mm3=cell_volume_mm3,
        label_volumes_mm3=label_volumes,
    )


def build_surface_region_volume_set(
    geometry: SurfaceRegionGeometry,
    settings: SurfaceRegionVolumeSettings | None = None,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> AnalysisRegionSet:
    """Resolve authored surface regions into weighted inward volume samples."""

    if not isinstance(geometry, SurfaceRegionGeometry):
        raise SurfaceRegionVolumeError("A prepared SurfaceRegionGeometry provider is required.")
    try:
        geometry.require_volume_sampling_ready()
    except SurfaceRegionGeometryError as error:
        raise SurfaceRegionVolumeError(str(error)) from error
    config = (settings or SurfaceRegionVolumeSettings()).validated()
    lattice, requested_spacing_m = _choose_lattice(geometry, config)
    depth_m = float(config.depth_mm) / 1000.0

    indices_xyz, source_points, source_faces, _source_squared = _surface_candidates(
        geometry,
        lattice,
        depth_m,
        progress_callback=progress_callback,
        cancel_requested=cancel_requested,
    )
    source_labels = geometry.face_region_labels[source_faces].astype(np.int32, copy=False)
    named = source_labels > 0
    unclassified_voxels = int(np.sum(~named))
    indices_xyz = indices_xyz[named]
    source_points = source_points[named]
    source_labels = source_labels[named]
    if not len(source_points):
        raise SurfaceRegionVolumeError(
            "The inward shell contains no voxels owned by an authored surface region."
        )

    available = {int(value) for value in np.unique(source_labels)}
    missing = [region.name for region in geometry.regions if region.label_id not in available]
    if missing:
        raise SurfaceRegionVolumeError(
            "The selected surface-volume lattice did not retain any interior voxel centres for: "
            + ", ".join(missing)
            + ". Increase source-voxel density or use a larger sampling depth."
        )

    flat_source_indices = np.ravel_multi_index(indices_xyz.T, tuple(lattice.counts_xyz)).astype(np.int64)
    order = np.argsort(flat_source_indices, kind="stable")
    flat_source_indices = flat_source_indices[order]
    source_points = source_points[order]
    source_labels = source_labels[order]
    samples = _select_retained_samples(
        flat_source_indices,
        source_points,
        source_labels,
        maximum_samples_per_region=config.maximum_samples_per_region,
        cell_volume_mm3=lattice.cell_volume_mm3,
    )

    children: list[AnalysisRegionNode] = []
    for region in geometry.regions:
        volume_mm3 = samples.label_volumes_mm3[region.label_id]
        children.append(
            AnalysisRegionNode(
                key=region.key,
                name=region.name,
                label_ids=(region.label_id,),
                region_kind="near_surface_volume",
                metadata={
                    "surface_area_cm2": geometry.surface_area_m2_for_label(region.label_id) * 1e4,
                    "sampling_depth_mm": float(config.depth_mm),
                    "volume_cm3": volume_mm3 / 1000.0,
                    "source_voxel_count": samples.source_counts[region.label_id],
                    "retained_sample_count": samples.sampled_counts[region.label_id],
                },
            )
        )
    total_named_volume_cm3 = sum(samples.label_volumes_mm3.values()) / 1000.0
    hierarchy = AnalysisRegionNode(
        key="surface_regions",
        name="Surface regions",
        label_ids=tuple(region.label_id for region in geometry.regions),
        children=tuple(children),
        region_kind="near_surface_volume_set",
        metadata={
            "sampling_depth_mm": float(config.depth_mm),
            "volume_cm3": total_named_volume_cm3,
            "surface_area_cm2": sum(
                geometry.surface_area_m2_for_label(region.label_id) * 1e4
                for region in geometry.regions
            ),
            "source_voxel_count": int(sum(samples.source_counts.values())),
            "retained_sample_count": samples.point_count,
        },
    )

    geometry_cache_identity, cache_request = surface_region_volume_cache_identity(
        geometry, config
    )
    identity_payload = {
        **cache_request,
        "geometry_cache_identity": geometry_cache_identity,
        "requested_spacing_mm": requested_spacing_m * 1000.0,
        "cell_size_mm": (lattice.cell_size_m * 1000.0).tolist(),
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        **identity_payload,
        "derived_geometry_identity": identity_sha256,
        "source_key": geometry.source_key,
        "source_name": geometry.source_name,
        "grid_shape": [int(value) for value in lattice.counts_xyz],
        "lattice_cell_count": lattice.total_cells,
        "cell_volume_mm3": lattice.cell_volume_mm3,
        "source_voxel_count": int(sum(samples.source_counts.values())),
        "retained_sample_count": samples.point_count,
        "unclassified_source_voxel_count": unclassified_voxels,
        "unclassified_volume_cm3": unclassified_voxels * lattice.cell_volume_mm3 / 1000.0,
        "volume_cm3": total_named_volume_cm3,
        "volume_method": "interior voxel-centre integration",
        "ownership_method": "nearest authored surface triangle",
        "depth_direction": "inward only",
    }
    _check_cancel(cancel_requested)
    _emit_progress(
        progress_callback,
        stage="Surface-region volume samples ready",
        completed=1,
        total=1,
        detail=(
            f"Prepared {samples.point_count:,} retained field samples representing "
            f"{metadata['source_voxel_count']:,} interior source voxels."
        ),
    )
    return AnalysisRegionSet(
        key=f"{geometry.source_key}:inward_surface_volume",
        name=f"{geometry.source_name} ({config.depth_mm:g} mm inward)",
        kind="surface_region_volume",
        hierarchy=hierarchy,
        samples=samples,
        sampling_mode="inward_voxel_shell",
        metadata=metadata,
    )
