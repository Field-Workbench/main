"""Geometric queries for triangle meshes with authored surface regions.

This module turns the immutable named-face metadata from :mod:`surface_regions`
into a reusable geometric provider.  Near-surface volume sampling is layered on
top of these queries by :mod:`surface_region_volume`, keeping the Exposure
Analysis engine independent of mesh implementation details.

The provider supports arbitrary triangulated region-aware meshes, not only the
bundled Generic bust.  Nearest-surface ownership is deterministic: ties are
resolved by the lowest original triangle index.  Unassigned triangles remain
explicitly unclassified.  Overlapping authored face assignments are retained as
ambiguous rather than silently choosing one region.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


SURFACE_REGION_GEOMETRY_VERSION = 1
UNCLASSIFIED_REGION_LABEL = -1
AMBIGUOUS_REGION_LABEL = -2


class SurfaceRegionGeometryError(ValueError):
    """Raised when region-aware surface geometry cannot be queried reliably."""


@dataclass(frozen=True)
class SurfaceRegionDefinition:
    """One authored surface region with a stable provider-local integer label."""

    label_id: int
    key: str
    name: str
    face_indices: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceQueryResult:
    """Nearest-surface and containment information for one batch of points."""

    points_m: np.ndarray
    distance_m: np.ndarray
    closest_points_m: np.ndarray
    face_indices: np.ndarray
    region_label_ids: np.ndarray
    inside: np.ndarray
    on_surface: np.ndarray
    containment_certified: bool

    @property
    def interior_or_surface(self) -> np.ndarray:
        return np.asarray(self.inside, dtype=bool) | np.asarray(self.on_surface, dtype=bool)

    @property
    def signed_depth_m(self) -> np.ndarray:
        """Depth below the surface: positive inside, negative outside, zero on-surface."""
        distance = np.asarray(self.distance_m, dtype=float)
        return np.where(
            np.asarray(self.on_surface, dtype=bool),
            0.0,
            np.where(np.asarray(self.inside, dtype=bool), distance, -distance),
        )


@dataclass
class _BVHNode:
    minimum: np.ndarray
    maximum: np.ndarray
    left: int = -1
    right: int = -1
    face_indices: np.ndarray | None = None

    @property
    def leaf(self) -> bool:
        return self.face_indices is not None


def _usable_triangle_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return usable triangle faces, original indices, and rejected face count."""

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=int), 0
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        count = int(len(faces)) if faces.ndim else 0
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=int), count
    valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    valid &= np.asarray([len(set(map(int, face))) == 3 for face in faces], dtype=bool)
    candidate_indices = np.flatnonzero(valid)
    if not len(candidate_indices):
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=int), int(len(faces))
    candidates = faces[candidate_indices]
    triangles = vertices[candidates]
    finite = np.isfinite(triangles).all(axis=(1, 2))
    finite_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    bounds = np.ptp(finite_vertices, axis=0) if len(finite_vertices) else np.zeros(3)
    diagonal = float(np.linalg.norm(bounds))
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    area_tolerance = max(diagonal * diagonal * 1e-12, 1e-24)
    usable_mask = finite & (twice_area > area_tolerance)
    usable_indices = candidate_indices[usable_mask]
    return faces[usable_indices], usable_indices, int(len(faces) - len(usable_indices))


def mesh_health(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Diagnose whether a mesh can support certified inside/outside queries.

    Containment here uses orientation-independent ray parity, so inconsistent
    winding is reported but does not by itself prevent containment.  Open,
    non-manifold, duplicate, or invalid geometry does prevent certification.
    """

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    vertex_count = int(len(vertices)) if vertices.ndim == 2 else 0
    face_count = int(len(faces)) if faces.ndim == 2 else 0
    usable, usable_indices, unusable_count = _usable_triangle_faces(vertices, faces)
    usable_count = int(len(usable))

    face_occurrences: dict[tuple[int, int, int], int] = {}
    edge_directions: dict[tuple[int, int], list[int]] = {}
    for face in usable:
        face_key = tuple(sorted(map(int, face)))
        face_occurrences[face_key] = face_occurrences.get(face_key, 0) + 1
        for start, stop in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = (start, stop) if start < stop else (stop, start)
            edge_directions.setdefault(key, []).append(1 if start < stop else -1)

    duplicate_faces = sum(max(0, count - 1) for count in face_occurrences.values())
    boundary_edges = sum(len(values) == 1 for values in edge_directions.values())
    non_manifold_edges = sum(len(values) > 2 for values in edge_directions.values())
    inconsistent_edges = sum(
        len(values) == 2 and values[0] == values[1]
        for values in edge_directions.values()
    )
    used_vertices = set(map(int, usable.reshape(-1))) if usable_count else set()
    unused_vertices = max(0, vertex_count - len(used_vertices))
    watertight = bool(usable_count and boundary_edges == 0 and non_manifold_edges == 0)
    manifold = bool(usable_count and non_manifold_edges == 0)
    consistently_oriented = bool(usable_count and inconsistent_edges == 0)
    containment_reliable = bool(
        watertight
        and unusable_count == 0
        and duplicate_faces == 0
    )

    signed_volume_m3 = 0.0
    if usable_count:
        triangles = vertices[usable]
        signed_volume_m3 = float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                )
            )
            / 6.0
        )
    orientation = "unknown"
    if watertight and consistently_oriented and abs(signed_volume_m3) > 1e-24:
        orientation = "outward" if signed_volume_m3 > 0.0 else "inward"

    if containment_reliable:
        summary = (
            f"Closed manifold surface with {usable_count:,} usable triangles; "
            "inside/outside classification is available."
        )
    else:
        issues: list[str] = []
        if boundary_edges:
            issues.append(f"{boundary_edges:,} open edge" + ("s" if boundary_edges != 1 else ""))
        if non_manifold_edges:
            issues.append(
                f"{non_manifold_edges:,} non-manifold edge"
                + ("s" if non_manifold_edges != 1 else "")
            )
        if unusable_count:
            issues.append(
                f"{unusable_count:,} degenerate/invalid face"
                + ("s" if unusable_count != 1 else "")
            )
        if duplicate_faces:
            issues.append(
                f"{duplicate_faces:,} duplicate face" + ("s" if duplicate_faces != 1 else "")
            )
        if not issues:
            issues.append("no usable closed surface")
        summary = "; ".join(issues) + ". Inside/outside classification is not certified."

    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "usable_face_count": usable_count,
        "degenerate_or_invalid_faces": int(unusable_count),
        "duplicate_faces": int(duplicate_faces),
        "boundary_edges": int(boundary_edges),
        "non_manifold_edges": int(non_manifold_edges),
        "inconsistent_edges": int(inconsistent_edges),
        "unused_vertices": int(unused_vertices),
        "watertight": watertight,
        "manifold": manifold,
        "consistently_oriented": consistently_oriented,
        "containment_reliable": containment_reliable,
        "orientation": orientation,
        "signed_volume_cm3": signed_volume_m3 * 1e6,
        "summary": summary,
        "_usable_face_indices": usable_indices.tolist(),
    }


def _aabb_distance_squared(point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> float:
    below = np.maximum(minimum - point, 0.0)
    above = np.maximum(point - maximum, 0.0)
    delta = below + above
    return float(np.dot(delta, delta))


def _closest_points_on_segments(
    point: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    directions = stops - starts
    denominator = np.einsum("ij,ij->i", directions, directions)
    fractions = np.zeros(len(starts), dtype=float)
    valid = denominator > 1e-30
    if np.any(valid):
        fractions[valid] = np.clip(
            np.einsum("ij,ij->i", point[None, :] - starts[valid], directions[valid])
            / denominator[valid],
            0.0,
            1.0,
        )
    closest = starts + fractions[:, None] * directions
    squared = np.einsum("ij,ij->i", closest - point[None, :], closest - point[None, :])
    return closest, squared


def _closest_points_on_triangles(
    point: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact closest points/distances from one point to nondegenerate triangles."""

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    normals = np.cross(ab, ac)
    normal_sq = np.einsum("ij,ij->i", normals, normals)
    signed_numerator = np.einsum("ij,ij->i", point[None, :] - a, normals)
    projected = point[None, :] - (signed_numerator / normal_sq)[:, None] * normals

    v2 = projected - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", v2, ab)
    d21 = np.einsum("ij,ij->i", v2, ac)
    denominator = d00 * d11 - d01 * d01
    beta = (d11 * d20 - d01 * d21) / denominator
    gamma = (d00 * d21 - d01 * d20) / denominator
    alpha = 1.0 - beta - gamma
    bary_tolerance = 1e-12
    plane_inside = (
        (alpha >= -bary_tolerance)
        & (beta >= -bary_tolerance)
        & (gamma >= -bary_tolerance)
    )

    result = projected.copy()
    squared = np.einsum("ij,ij->i", projected - point[None, :], projected - point[None, :])
    if np.all(plane_inside):
        return result, squared

    outside = np.flatnonzero(~plane_inside)
    reduced = triangles[outside]
    edge_points: list[np.ndarray] = []
    edge_sq: list[np.ndarray] = []
    for start_index, stop_index in ((0, 1), (1, 2), (2, 0)):
        candidate_points, candidate_sq = _closest_points_on_segments(
            point, reduced[:, start_index], reduced[:, stop_index]
        )
        edge_points.append(candidate_points)
        edge_sq.append(candidate_sq)
    stacked_sq = np.column_stack(edge_sq)
    selected_edge = np.argmin(stacked_sq, axis=1)
    selected_sq = stacked_sq[np.arange(len(outside)), selected_edge]
    selected_points = np.empty((len(outside), 3), dtype=float)
    for edge_index, candidates in enumerate(edge_points):
        mask = selected_edge == edge_index
        selected_points[mask] = candidates[mask]
    result[outside] = selected_points
    squared[outside] = selected_sq
    return result, squared


class SurfaceRegionGeometry:
    """Prepared region-aware triangle geometry with deterministic spatial queries."""

    def __init__(
        self,
        vertices_m: Iterable[Iterable[float]],
        faces: Iterable[Iterable[int]],
        region_map: dict[str, Any],
        *,
        source_key: str = "surface_regions",
        source_name: str = "Surface regions",
        leaf_size: int = 24,
    ) -> None:
        vertices = np.asarray(vertices_m, dtype=float)
        face_array = np.asarray(faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            raise SurfaceRegionGeometryError("Surface-region vertices must be a non-empty N×3 array.")
        if face_array.ndim != 2 or face_array.shape[1:] != (3,) or not len(face_array):
            raise SurfaceRegionGeometryError("Surface-region faces must be a non-empty M×3 triangle array.")
        if not np.isfinite(vertices).all():
            raise SurfaceRegionGeometryError("Surface-region vertices contain non-finite coordinates.")
        if np.any(face_array < 0) or np.any(face_array >= len(vertices)):
            raise SurfaceRegionGeometryError("Surface-region faces reference vertices outside the mesh.")

        health = mesh_health(vertices, face_array)
        usable_indices = np.asarray(health["_usable_face_indices"], dtype=int)
        if len(usable_indices) != len(face_array):
            raise SurfaceRegionGeometryError(
                "Surface-region geometry contains degenerate or invalid triangles; "
                "region ownership cannot safely preserve authored face indices."
            )

        regions_value = region_map.get("regions") if isinstance(region_map, dict) else None
        if not isinstance(regions_value, dict) or not regions_value:
            raise SurfaceRegionGeometryError("Surface-region metadata contains no named regions.")

        definitions: list[SurfaceRegionDefinition] = []
        face_region_labels = np.full(len(face_array), UNCLASSIFIED_REGION_LABEL, dtype=np.int32)
        overlap_faces: set[int] = set()
        for label_id, (raw_key, raw_region) in enumerate(regions_value.items(), start=1):
            key = str(raw_key).strip()
            if not key or not isinstance(raw_region, dict):
                raise SurfaceRegionGeometryError("Surface-region metadata contains a malformed region.")
            name = str(raw_region.get("display_name") or key).strip() or key
            raw_indices = raw_region.get("face_indices")
            if not isinstance(raw_indices, list):
                raise SurfaceRegionGeometryError(f"Surface region {key!r} has no face-index list.")
            normalized: list[int] = []
            for raw_index in raw_indices:
                if isinstance(raw_index, bool):
                    raise SurfaceRegionGeometryError(f"Surface region {key!r} has an invalid face index.")
                try:
                    face_index = int(raw_index)
                except (TypeError, ValueError) as error:
                    raise SurfaceRegionGeometryError(
                        f"Surface region {key!r} has an invalid face index {raw_index!r}."
                    ) from error
                if face_index != raw_index or not 0 <= face_index < len(face_array):
                    raise SurfaceRegionGeometryError(
                        f"Surface region {key!r} references face {face_index}, outside this mesh."
                    )
                normalized.append(face_index)
            if len(normalized) != len(set(normalized)):
                raise SurfaceRegionGeometryError(f"Surface region {key!r} contains duplicate face indices.")
            for face_index in normalized:
                current = int(face_region_labels[face_index])
                if current == UNCLASSIFIED_REGION_LABEL:
                    face_region_labels[face_index] = label_id
                elif current != label_id:
                    face_region_labels[face_index] = AMBIGUOUS_REGION_LABEL
                    overlap_faces.add(face_index)
            definitions.append(
                SurfaceRegionDefinition(
                    label_id=label_id,
                    key=key,
                    name=name,
                    face_indices=tuple(normalized),
                )
            )

        self.source_key = str(source_key)
        self.source_name = str(source_name)
        self.vertices_m = vertices
        self.faces = face_array
        self.triangles_m = vertices[face_array]
        self.regions = tuple(definitions)
        self.face_region_labels = face_region_labels
        self.health = {key: value for key, value in health.items() if not key.startswith("_")}
        self.overlap_face_indices = tuple(sorted(overlap_faces))
        self.unclassified_face_indices = tuple(
            map(int, np.flatnonzero(face_region_labels == UNCLASSIFIED_REGION_LABEL))
        )
        self.bounds_min_m = np.min(vertices, axis=0)
        self.bounds_max_m = np.max(vertices, axis=0)
        self.extent_m = float(np.linalg.norm(self.bounds_max_m - self.bounds_min_m))
        self.surface_tolerance_m = max(1e-12, self.extent_m * 1e-10)
        self._tie_tolerance_sq = self.surface_tolerance_m * self.surface_tolerance_m
        self._leaf_size = max(4, int(leaf_size))

        self._triangle_min = np.min(self.triangles_m, axis=1)
        self._triangle_max = np.max(self.triangles_m, axis=1)
        self._triangle_centroids = np.mean(self.triangles_m, axis=1)
        self.triangle_area_m2 = 0.5 * np.linalg.norm(
            np.cross(
                self.triangles_m[:, 1] - self.triangles_m[:, 0],
                self.triangles_m[:, 2] - self.triangles_m[:, 0],
            ),
            axis=1,
        )
        self.region_surface_area_m2 = {
            region.label_id: float(np.sum(self.triangle_area_m2[list(region.face_indices)]))
            if region.face_indices else 0.0
            for region in self.regions
        }
        self._bvh_nodes: list[_BVHNode] = []
        self._bvh_root = self._build_bvh(np.arange(len(self.faces), dtype=int))

        # Prepared YZ-projection data for +X parity containment rays.
        triangles = self.triangles_m
        self._ray_y0, self._ray_z0 = triangles[:, 0, 1], triangles[:, 0, 2]
        self._ray_y1, self._ray_z1 = triangles[:, 1, 1], triangles[:, 1, 2]
        self._ray_y2, self._ray_z2 = triangles[:, 2, 1], triangles[:, 2, 2]
        self._ray_denominator = (
            (self._ray_z1 - self._ray_z2) * (self._ray_y0 - self._ray_y2)
            + (self._ray_y2 - self._ray_y1) * (self._ray_z0 - self._ray_z2)
        )
        projection_epsilon = max(1e-24, self.extent_m * self.extent_m * 1e-14)
        self._ray_projected = np.abs(self._ray_denominator) > projection_epsilon
        self._ray_min_y = np.min(triangles[:, :, 1], axis=1)
        self._ray_max_y = np.max(triangles[:, :, 1], axis=1)
        self._ray_min_z = np.min(triangles[:, :, 2], axis=1)
        self._ray_max_z = np.max(triangles[:, :, 2], axis=1)
        self._ray_jitter = max(1e-15, self.extent_m * 1e-11)

    @property
    def region_by_label(self) -> dict[int, SurfaceRegionDefinition]:
        return {region.label_id: region for region in self.regions}

    @property
    def region_by_key(self) -> dict[str, SurfaceRegionDefinition]:
        return {region.key: region for region in self.regions}

    @property
    def complete_partition(self) -> bool:
        return not self.overlap_face_indices and not self.unclassified_face_indices

    @property
    def ownership_unambiguous(self) -> bool:
        return not self.overlap_face_indices

    @property
    def containment_reliable(self) -> bool:
        return bool(self.health.get("containment_reliable", False))

    @property
    def volume_sampling_ready(self) -> bool:
        """Whether inward volume generation has the geometry guarantees it needs."""
        return self.containment_reliable and self.ownership_unambiguous

    def region_key(self, label_id: int) -> str | None:
        if int(label_id) < 1:
            return None
        region = self.region_by_label.get(int(label_id))
        return region.key if region is not None else None

    def region_name(self, label_id: int) -> str | None:
        if int(label_id) < 1:
            return None
        region = self.region_by_label.get(int(label_id))
        return region.name if region is not None else None

    def surface_area_m2_for_label(self, label_id: int) -> float:
        return float(self.region_surface_area_m2.get(int(label_id), 0.0))

    def readiness(self) -> dict[str, Any]:
        """Return compact diagnostics for the near-surface volume provider."""
        reasons: list[str] = []
        if not self.containment_reliable:
            reasons.append(str(self.health.get("summary", "The mesh is not a certified closed surface.")))
        if self.overlap_face_indices:
            reasons.append(
                f"{len(self.overlap_face_indices):,} triangle(s) belong to more than one authored region."
            )
        return {
            "geometry_version": SURFACE_REGION_GEOMETRY_VERSION,
            "volume_sampling_ready": not reasons,
            "containment_reliable": self.containment_reliable,
            "ownership_unambiguous": self.ownership_unambiguous,
            "complete_partition": self.complete_partition,
            "region_count": len(self.regions),
            "unclassified_face_count": len(self.unclassified_face_indices),
            "overlap_face_count": len(self.overlap_face_indices),
            "reasons": reasons,
        }

    def require_volume_sampling_ready(self) -> None:
        status = self.readiness()
        if status["volume_sampling_ready"]:
            return
        detail = " ".join(status["reasons"]) or "Surface-region geometry is not ready."
        raise SurfaceRegionGeometryError(detail)

    def _build_bvh(self, face_indices: np.ndarray) -> int:
        minimum = np.min(self._triangle_min[face_indices], axis=0)
        maximum = np.max(self._triangle_max[face_indices], axis=0)
        node_index = len(self._bvh_nodes)
        self._bvh_nodes.append(_BVHNode(minimum=minimum, maximum=maximum))
        if len(face_indices) <= self._leaf_size:
            self._bvh_nodes[node_index].face_indices = np.sort(face_indices)
            return node_index

        centroid_extent = np.ptp(self._triangle_centroids[face_indices], axis=0)
        axis = int(np.argmax(centroid_extent))
        order = np.argsort(self._triangle_centroids[face_indices, axis], kind="stable")
        ordered = face_indices[order]
        split = len(ordered) // 2
        if split <= 0 or split >= len(ordered):
            self._bvh_nodes[node_index].face_indices = np.sort(face_indices)
            return node_index
        left = self._build_bvh(ordered[:split])
        right = self._build_bvh(ordered[split:])
        self._bvh_nodes[node_index].left = left
        self._bvh_nodes[node_index].right = right
        return node_index

    def _nearest_one(self, point: np.ndarray) -> tuple[float, np.ndarray, int]:
        best_sq = math.inf
        best_point = np.full(3, math.nan, dtype=float)
        best_face = -1
        queue: list[tuple[float, int]] = [(0.0, self._bvh_root)]
        while queue:
            bound_sq, node_index = heapq.heappop(queue)
            if bound_sq > best_sq + self._tie_tolerance_sq:
                break
            node = self._bvh_nodes[node_index]
            if node.leaf:
                assert node.face_indices is not None
                face_indices = node.face_indices
                closest, squared = _closest_points_on_triangles(point, self.triangles_m[face_indices])
                local_best_sq = float(np.min(squared))
                candidates = np.flatnonzero(squared <= local_best_sq + self._tie_tolerance_sq)
                # Original triangle index is the stable seam/equidistance tie-break.
                selected_local = min(candidates, key=lambda index: int(face_indices[index]))
                candidate_face = int(face_indices[selected_local])
                candidate_sq = float(squared[selected_local])
                if (
                    candidate_sq < best_sq - self._tie_tolerance_sq
                    or (
                        candidate_sq <= best_sq + self._tie_tolerance_sq
                        and (best_face < 0 or candidate_face < best_face)
                    )
                ):
                    best_sq = candidate_sq
                    best_face = candidate_face
                    best_point = closest[selected_local]
                continue
            for child_index in (node.left, node.right):
                if child_index < 0:
                    continue
                child = self._bvh_nodes[child_index]
                child_bound = _aabb_distance_squared(point, child.minimum, child.maximum)
                if child_bound <= best_sq + self._tie_tolerance_sq:
                    heapq.heappush(queue, (child_bound, child_index))
        if best_face < 0:
            raise SurfaceRegionGeometryError("Nearest-surface lookup failed for valid triangle geometry.")
        return math.sqrt(max(0.0, best_sq)), best_point, best_face

    def nearest_surface(
        self, points_m: Iterable[Iterable[float]] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return distance, closest point, triangle index, and region label per point."""
        points = np.asarray(points_m, dtype=float)
        single = points.shape == (3,)
        if single:
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise SurfaceRegionGeometryError("Query points must be an N×3 array.")
        if not np.isfinite(points).all():
            raise SurfaceRegionGeometryError("Query points contain non-finite coordinates.")

        distances = np.empty(len(points), dtype=float)
        closest = np.empty((len(points), 3), dtype=float)
        face_indices = np.empty(len(points), dtype=np.int64)
        for index, point in enumerate(points):
            distance, closest_point, face_index = self._nearest_one(point)
            distances[index] = distance
            closest[index] = closest_point
            face_indices[index] = face_index
        labels = self.face_region_labels[face_indices].astype(np.int32, copy=True)
        if single:
            return distances[:1], closest[:1], face_indices[:1], labels[:1]
        return distances, closest, face_indices, labels

    def points_inside(self, points_m: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
        """Classify points using deterministic +X ray parity on a certified closed mesh."""
        if not self.containment_reliable:
            raise SurfaceRegionGeometryError(
                "Inside/outside classification requires a closed manifold surface. "
                + str(self.health.get("summary", ""))
            )
        points = np.asarray(points_m, dtype=float)
        single = points.shape == (3,)
        if single:
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise SurfaceRegionGeometryError("Query points must be an N×3 array.")
        if not np.isfinite(points).all():
            raise SurfaceRegionGeometryError("Query points contain non-finite coordinates.")

        inside = np.zeros(len(points), dtype=bool)
        jitter = self._ray_jitter
        for index, point in enumerate(points):
            py = float(point[1]) + jitter * 0.3713906763541037
            pz = float(point[2]) + jitter * 0.6180339887498948
            candidates = np.flatnonzero(
                self._ray_projected
                & (py >= self._ray_min_y - jitter)
                & (py <= self._ray_max_y + jitter)
                & (pz >= self._ray_min_z - jitter)
                & (pz <= self._ray_max_z + jitter)
            )
            if not len(candidates):
                continue
            denominator = self._ray_denominator[candidates]
            alpha = (
                (self._ray_z1[candidates] - self._ray_z2[candidates]) * (py - self._ray_y2[candidates])
                + (self._ray_y2[candidates] - self._ray_y1[candidates]) * (pz - self._ray_z2[candidates])
            ) / denominator
            beta = (
                (self._ray_z2[candidates] - self._ray_z0[candidates]) * (py - self._ray_y2[candidates])
                + (self._ray_y0[candidates] - self._ray_y2[candidates]) * (pz - self._ray_z2[candidates])
            ) / denominator
            gamma = 1.0 - alpha - beta
            bary_tolerance = 1e-10
            hits = (
                (alpha >= -bary_tolerance)
                & (beta >= -bary_tolerance)
                & (gamma >= -bary_tolerance)
            )
            if not np.any(hits):
                continue
            selected = candidates[hits]
            x_intersections = (
                alpha[hits] * self.triangles_m[selected, 0, 0]
                + beta[hits] * self.triangles_m[selected, 1, 0]
                + gamma[hits] * self.triangles_m[selected, 2, 0]
            )
            forward = np.sort(x_intersections[x_intersections > float(point[0]) + jitter])
            if not len(forward):
                continue
            crossings = 0
            start = 0
            while start < len(forward):
                stop = start + 1
                while stop < len(forward) and forward[stop] - forward[start] <= jitter * 10.0:
                    stop += 1
                crossings += 1
                start = stop
            inside[index] = bool(crossings % 2)
        return inside[:1] if single else inside

    def query_points(
        self,
        points_m: Iterable[Iterable[float]] | np.ndarray,
        *,
        require_containment: bool = True,
    ) -> SurfaceQueryResult:
        """Resolve nearest authored region, surface depth, and containment together.

        ``distance_m`` is unsigned distance to the mesh.  ``inside`` provides the
        inward/outward distinction.  The volume provider can therefore select points where
        ``inside`` is true and ``distance_m <= requested_depth`` without relying
        on triangle normal orientation.
        """
        points = np.asarray(points_m, dtype=float)
        if points.shape == (3,):
            points = points.reshape(1, 3)
        distances, closest, face_indices, labels = self.nearest_surface(points)
        on_surface = distances <= self.surface_tolerance_m
        if self.containment_reliable:
            inside = np.zeros(len(points), dtype=bool)
            unresolved = ~on_surface
            if np.any(unresolved):
                inside[unresolved] = self.points_inside(points[unresolved])
            certified = True
        elif require_containment:
            raise SurfaceRegionGeometryError(
                "Near-surface volume ownership requires certified inside/outside geometry. "
                + str(self.health.get("summary", ""))
            )
        else:
            inside = np.zeros(len(points), dtype=bool)
            certified = False
        return SurfaceQueryResult(
            points_m=np.asarray(points, dtype=float),
            distance_m=distances,
            closest_points_m=closest,
            face_indices=face_indices,
            region_label_ids=labels,
            inside=inside,
            on_surface=on_surface,
            containment_certified=certified,
        )
