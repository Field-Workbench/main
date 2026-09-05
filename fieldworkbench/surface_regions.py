"""Surface-region sidecar loading and validation.

Field Workbench can associate named groups of triangle faces with a mesh without
splitting the mesh into separate scene objects.  Region sidecars are deliberately
small, plain JSON files so authoring can happen in Blender (or another mesh tool)
while Workbench remains responsible for validating and displaying the result.

Version 1 sidecars identify faces by their ordered triangle index.  That makes
mesh identity important: changing/re-exporting the mesh can change triangle
ordering even when the visible shape is similar. Built-in assets therefore pair
one fixed mesh with one fixed sidecar. Custom authoring can additionally record
``mesh_sha256`` in the sidecar; Workbench verifies that matched source pair at
import and then embeds the validated map beside an ordered-geometry fingerprint
in the portable scene.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SURFACE_REGION_SCHEMA = "fieldworkbench.surface_regions"
SURFACE_REGION_SCHEMA_VERSION = 1


class SurfaceRegionError(ValueError):
    """Raised when a surface-region sidecar is missing, malformed, or incompatible."""


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triangle_count(faces: Iterable[Iterable[int]]) -> tuple[int, bool]:
    count = 0
    all_triangles = True
    for face in faces:
        count += 1
        try:
            all_triangles = all_triangles and len(face) == 3  # type: ignore[arg-type]
        except TypeError:
            all_triangles = False
    return count, all_triangles


def load_surface_region_map(
    path: str | Path,
    *,
    vertices: Iterable[Iterable[float]] | None = None,
    faces: Iterable[Iterable[int]] | None = None,
    mesh_path: str | Path | None = None,
    expected_asset: str | None = None,
    expected_mesh_file: str | None = None,
    expected_mesh_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and validate a version-1 named-face region sidecar.

    ``vertices`` and ``faces`` are the actual mesh arrays that Workbench will
    render.  When supplied, their counts (and triangle arity) must match the
    topology recorded by the sidecar.  ``mesh_path`` plus
    ``expected_mesh_sha256`` can additionally lock an index-based sidecar to an
    exact source file, which is how the bundled Generic bust is protected.

    The exporter-provided ``ordered_geometry_sha256`` is retained as provenance.
    Version 1 does not prescribe the byte-serialization used to compute that
    authoring fingerprint, so runtime compatibility is established from the
    fixed source-file digest, topology counts, and face-index bounds instead of
    pretending to recompute an unspecified hash algorithm.
    """
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SurfaceRegionError(f"Unable to read surface-region metadata: {error}") from error
    if not isinstance(document, dict):
        raise SurfaceRegionError("Surface-region metadata must be a JSON object.")

    if document.get("schema") != SURFACE_REGION_SCHEMA:
        raise SurfaceRegionError(
            f"Unsupported surface-region schema {document.get('schema')!r}."
        )
    if document.get("schema_version") != SURFACE_REGION_SCHEMA_VERSION:
        raise SurfaceRegionError(
            "Unsupported surface-region schema version "
            f"{document.get('schema_version')!r}; expected {SURFACE_REGION_SCHEMA_VERSION}."
        )

    asset = str(document.get("asset", "")).strip()
    mesh_file = str(document.get("mesh_file", "")).strip()
    if not asset:
        raise SurfaceRegionError("Surface-region metadata does not identify an asset.")
    if not mesh_file:
        raise SurfaceRegionError("Surface-region metadata does not identify its mesh file.")
    if expected_asset is not None and asset != expected_asset:
        raise SurfaceRegionError(
            f"Surface-region asset is {asset!r}, expected {expected_asset!r}."
        )
    if expected_mesh_file is not None and mesh_file != expected_mesh_file:
        raise SurfaceRegionError(
            f"Surface-region mesh is {mesh_file!r}, expected {expected_mesh_file!r}."
        )

    topology = document.get("topology")
    if not isinstance(topology, dict):
        raise SurfaceRegionError("Surface-region metadata has no topology block.")
    try:
        expected_vertices = int(topology["vertex_count"])
        expected_faces = int(topology["face_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise SurfaceRegionError("Surface-region topology counts are invalid.") from error
    if expected_vertices < 0 or expected_faces <= 0:
        raise SurfaceRegionError("Surface-region topology counts must be positive.")
    if not bool(topology.get("all_triangles", False)):
        raise SurfaceRegionError("Surface-region version 1 requires a triangulated mesh.")

    actual_vertex_count: int | None = None
    if vertices is not None:
        try:
            actual_vertex_count = len(vertices)  # type: ignore[arg-type]
        except TypeError:
            vertices = list(vertices)
            actual_vertex_count = len(vertices)
        if actual_vertex_count != expected_vertices:
            raise SurfaceRegionError(
                f"Surface-region vertex count is {expected_vertices:,}, but the mesh has "
                f"{actual_vertex_count:,}."
            )

    actual_face_count: int | None = None
    if faces is not None:
        try:
            actual_face_count = len(faces)  # type: ignore[arg-type]
            face_values = faces
        except TypeError:
            face_values = list(faces)
            actual_face_count = len(face_values)
        counted_faces, all_triangles = _triangle_count(face_values)
        actual_face_count = counted_faces
        if actual_face_count != expected_faces:
            raise SurfaceRegionError(
                f"Surface-region face count is {expected_faces:,}, but the mesh has "
                f"{actual_face_count:,}."
            )
        if not all_triangles:
            raise SurfaceRegionError("The associated mesh is not fully triangulated.")

    declared_mesh_sha256 = str(document.get("mesh_sha256", "")).strip().lower() or None
    if declared_mesh_sha256 is not None and (
        len(declared_mesh_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_mesh_sha256)
    ):
        raise SurfaceRegionError("Surface-region mesh_sha256 is not a valid SHA-256 digest.")

    actual_mesh_sha256: str | None = None
    if mesh_path is not None:
        mesh_source = Path(mesh_path)
        try:
            actual_mesh_sha256 = file_sha256(mesh_source)
        except OSError as error:
            raise SurfaceRegionError(f"Unable to verify the associated mesh file: {error}") from error
        digests: list[tuple[str, str]] = []
        if declared_mesh_sha256 is not None:
            digests.append(("sidecar", declared_mesh_sha256))
        if expected_mesh_sha256 is not None:
            digests.append(("expected", str(expected_mesh_sha256).strip().lower()))
        for source_label, expected_digest in digests:
            if actual_mesh_sha256 != expected_digest:
                raise SurfaceRegionError(
                    "The associated mesh file does not match the surface-region map. "
                    f"The {source_label} SHA-256 is {expected_digest}, got {actual_mesh_sha256}."
                )

    raw_regions = document.get("regions")
    if not isinstance(raw_regions, dict) or not raw_regions:
        raise SurfaceRegionError("Surface-region metadata contains no named regions.")

    normalized_regions: dict[str, dict[str, Any]] = {}
    assigned: list[int] = []
    for raw_key, raw_region in raw_regions.items():
        key = str(raw_key).strip()
        if not key:
            raise SurfaceRegionError("A surface region has an empty key.")
        if not isinstance(raw_region, dict):
            raise SurfaceRegionError(f"Surface region {key!r} is not an object.")
        raw_indices = raw_region.get("face_indices")
        if not isinstance(raw_indices, list):
            raise SurfaceRegionError(f"Surface region {key!r} has no face-index list.")
        indices: list[int] = []
        for raw_index in raw_indices:
            if isinstance(raw_index, bool):
                raise SurfaceRegionError(f"Surface region {key!r} contains a non-integer face index.")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise SurfaceRegionError(
                    f"Surface region {key!r} contains an invalid face index {raw_index!r}."
                ) from error
            if index != raw_index:
                raise SurfaceRegionError(
                    f"Surface region {key!r} contains a non-integral face index {raw_index!r}."
                )
            if index < 0 or index >= expected_faces:
                raise SurfaceRegionError(
                    f"Surface region {key!r} references face {index:,}, outside 0–{expected_faces - 1:,}."
                )
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise SurfaceRegionError(f"Surface region {key!r} contains duplicate face indices.")
        try:
            declared_count = int(raw_region.get("face_count", len(indices)))
        except (TypeError, ValueError) as error:
            raise SurfaceRegionError(f"Surface region {key!r} has an invalid face count.") from error
        if declared_count != len(indices):
            raise SurfaceRegionError(
                f"Surface region {key!r} declares {declared_count:,} faces but lists {len(indices):,}."
            )
        display_name = str(raw_region.get("display_name") or key).strip() or key
        normalized_regions[key] = {
            "display_name": display_name,
            "face_indices": indices,
            "face_count": len(indices),
        }
        assigned.extend(indices)

    unique_assigned = set(assigned)
    overlap_count = len(assigned) - len(unique_assigned)
    missing_count = expected_faces - len(unique_assigned)
    validation = {
        "region_count": len(normalized_regions),
        "assigned_face_references": len(assigned),
        "unique_assigned_faces": len(unique_assigned),
        "overlap_count": overlap_count,
        "missing_face_count": max(0, missing_count),
        "complete_partition": overlap_count == 0 and len(unique_assigned) == expected_faces,
        "mesh_file_sha256": actual_mesh_sha256,
        "mesh_file_sha256_verified": bool(
            actual_mesh_sha256 is not None
            and (expected_mesh_sha256 is not None or declared_mesh_sha256 is not None)
        ),
        "topology_counts_verified": actual_vertex_count is not None and actual_face_count is not None,
    }

    return {
        "schema": SURFACE_REGION_SCHEMA,
        "schema_version": SURFACE_REGION_SCHEMA_VERSION,
        "asset": asset,
        "mesh_file": mesh_file,
        "mesh_sha256": declared_mesh_sha256,
        "authoring": copy.deepcopy(document.get("authoring", {})),
        "topology": copy.deepcopy(topology),
        "regions": normalized_regions,
        "validation": validation,
        "source": source.name,
    }
