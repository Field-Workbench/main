"""Renderable geometry for Exposure Analysis region targets.

The numerical Exposure Analysis contract lives in :mod:`spatial_analysis` and is
sample/weight based.  Visual previews need a different representation: the
actual object surface plus the same region hierarchy/label identities used by
the numerical provider.  This module is that boundary.

``AnalysisRegionScene`` deliberately contains no Qt, Plotly, or WebGL objects.
It is a dependency-light immutable geometry description for regional identity,
metrics and the modeless Region Viewer. Compact Exposure Analysis previews use
the canonical Model Editor object trace directly; the object-surface arrays kept
here provide target bounds/provenance and a consistent display identity.

Visual geometry is descriptive rather than a second analysis definition:

* LPBA40 targets retain the bundled per-label display shells for Region Viewer
  browsing and also expose the selected brain object's ordinary outer scene shell
  for compact whole-object previews.
* Authored surface-region targets use the object's real transformed triangle
  mesh and its authored per-face region ownership.  The inward analysis shell
  remains numerical sampling geometry; the viewer shows the surface that names
  and owns that shell.
* Ordinary Measurement targets use the same transformed object surface that is
  sampled by the Measurement-volume provider.

The hierarchy keys and integer label ids intentionally match the corresponding
Exposure Analysis provider so UI selection can be shared without translation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .brain_analysis import lpba40_region_hierarchy
from .brain_mesh import BrainMeshError, load_lpba40_region_meshes
from .spatial_analysis import AnalysisRegionNode
from .sri24_atlas import load_sri24_parcellation
from .surface_region_geometry import (
    AMBIGUOUS_REGION_LABEL,
    UNCLASSIFIED_REGION_LABEL,
    SurfaceRegionGeometry,
)


ANALYSIS_REGION_SCENE_VERSION = 2
UNCLASSIFIED_SCENE_LABEL = 0


class AnalysisRegionSceneError(ValueError):
    """Raised when an Exposure Analysis target cannot provide display geometry."""


def _hex_from_rgba(rgba: Iterable[int]) -> str:
    values = tuple(int(value) for value in rgba)
    if len(values) < 3:
        return "#64748b"
    return "#{:02x}{:02x}{:02x}".format(*values[:3])


def _normalized_colour(value: Any, fallback: str = "#64748b") -> str:
    text = str(value or "").strip().lower()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            return fallback
        return text
    return fallback


def _node_map(root: AnalysisRegionNode) -> dict[str, AnalysisRegionNode]:
    result: dict[str, AnalysisRegionNode] = {}
    for node in root.walk():
        if node.key in result:
            raise AnalysisRegionSceneError(
                f"Region hierarchy contains duplicate key {node.key!r}."
            )
        result[node.key] = node
    return result


@dataclass(frozen=True, init=False)
class AnalysisRegionScene:
    """One target's complete renderable object surface and regional identity.

    ``vertices_m``/``faces`` form the labelled Region Viewer mesh in world metres.
    ``face_label_ids`` assigns each triangle to one atomic analysis label.
    ``object_vertices_m``/``object_faces`` describe the ordinary unsegmented target
    surface used for display identity/metrics; the compact preview itself receives
    the canonical Model Editor trace from the adapter. Label ``0`` is reserved for visible-but-
    unclassified regional surface geometry. Hierarchy nodes select labelled faces
    by their existing ``label_ids`` tuple.
    """

    target_id: str
    target_name: str
    target_kind: str
    region_set_key: str
    hierarchy: AnalysisRegionNode
    vertices_m: np.ndarray
    faces: np.ndarray
    face_label_ids: np.ndarray
    label_colours: dict[int, str]
    base_colour: str
    base_opacity: float
    object_vertices_m: np.ndarray
    object_faces: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __init__(
        self,
        *,
        target_id: str,
        target_name: str,
        target_kind: str,
        region_set_key: str,
        hierarchy: AnalysisRegionNode,
        vertices_m: Any,
        faces: Any,
        face_label_ids: Any,
        label_colours: dict[int, str] | None = None,
        base_colour: str = "#64748b",
        base_opacity: float = 0.65,
        object_vertices_m: Any | None = None,
        object_faces: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vertices = np.asarray(vertices_m, dtype=np.float32)
        triangles = np.asarray(faces, dtype=np.int32)
        labels = np.asarray(face_label_ids, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            raise AnalysisRegionSceneError("Region-scene vertices must be a non-empty N×3 array.")
        if not np.isfinite(vertices).all():
            raise AnalysisRegionSceneError("Region-scene vertices contain non-finite coordinates.")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,) or not len(triangles):
            raise AnalysisRegionSceneError("Region-scene faces must be a non-empty M×3 triangle array.")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise AnalysisRegionSceneError("Region-scene faces reference vertices outside the mesh.")
        if labels.shape != (len(triangles),):
            raise AnalysisRegionSceneError("Region-scene face labels must match the triangle count.")
        if np.any(labels < 0):
            raise AnalysisRegionSceneError("Region-scene face labels cannot be negative.")
        if not isinstance(hierarchy, AnalysisRegionNode):
            raise AnalysisRegionSceneError("Region-scene hierarchy must be an AnalysisRegionNode.")

        nodes = _node_map(hierarchy)
        hierarchy_labels = {
            int(label)
            for node in nodes.values()
            for label in node.label_ids
        }
        present_labels = {int(value) for value in np.unique(labels) if int(value) > 0}
        missing = sorted(hierarchy_labels - present_labels)
        if missing:
            raise AnalysisRegionSceneError(
                "Region-scene display mesh is missing hierarchy label(s): "
                + ", ".join(map(str, missing))
                + "."
            )
        extra = sorted(present_labels - set(map(int, hierarchy.label_ids)))
        if extra:
            raise AnalysisRegionSceneError(
                "Region-scene display mesh contains labels outside the root region: "
                + ", ".join(map(str, extra))
                + "."
            )

        try:
            opacity = float(base_opacity)
        except (TypeError, ValueError) as error:
            raise AnalysisRegionSceneError("Region-scene opacity must be numeric.") from error
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise AnalysisRegionSceneError("Region-scene opacity must be between 0 and 1.")

        # The compact Exposure Analysis preview shows the target as it appears in
        # the ordinary scene: one unsegmented outer/object shell.  Region Viewer
        # geometry remains the labelled mesh above.  Most targets share the same
        # surface for both purposes; atlas targets can provide a separate outer
        # anatomical shell while retaining their independent per-region meshes.
        object_vertices = vertices if object_vertices_m is None else np.asarray(object_vertices_m, dtype=np.float32)
        object_triangles = triangles if object_faces is None else np.asarray(object_faces, dtype=np.int32)
        if object_vertices.ndim != 2 or object_vertices.shape[1:] != (3,) or not len(object_vertices):
            raise AnalysisRegionSceneError("Region-scene object vertices must be a non-empty N×3 array.")
        if not np.isfinite(object_vertices).all():
            raise AnalysisRegionSceneError("Region-scene object vertices contain non-finite coordinates.")
        if object_triangles.ndim != 2 or object_triangles.shape[1:] != (3,) or not len(object_triangles):
            raise AnalysisRegionSceneError("Region-scene object faces must be a non-empty M×3 triangle array.")
        if np.any(object_triangles < 0) or np.any(object_triangles >= len(object_vertices)):
            raise AnalysisRegionSceneError("Region-scene object faces reference vertices outside the mesh.")

        colours = {
            int(label): _normalized_colour(colour)
            for label, colour in dict(label_colours or {}).items()
            if int(label) > 0
        }
        vertices = np.ascontiguousarray(vertices)
        triangles = np.ascontiguousarray(triangles)
        labels = np.ascontiguousarray(labels)
        object_vertices = np.ascontiguousarray(object_vertices)
        object_triangles = np.ascontiguousarray(object_triangles)
        vertices.setflags(write=False)
        triangles.setflags(write=False)
        labels.setflags(write=False)
        object_vertices.setflags(write=False)
        object_triangles.setflags(write=False)

        object.__setattr__(self, "target_id", str(target_id))
        object.__setattr__(self, "target_name", str(target_name))
        object.__setattr__(self, "target_kind", str(target_kind))
        object.__setattr__(self, "region_set_key", str(region_set_key))
        object.__setattr__(self, "hierarchy", hierarchy)
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", triangles)
        object.__setattr__(self, "face_label_ids", labels)
        object.__setattr__(self, "label_colours", colours)
        object.__setattr__(self, "base_colour", _normalized_colour(base_colour))
        object.__setattr__(self, "base_opacity", opacity)
        object.__setattr__(self, "object_vertices_m", object_vertices)
        object.__setattr__(self, "object_faces", object_triangles)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def bounds_min_m(self) -> np.ndarray:
        return np.min(self.vertices_m, axis=0)

    @property
    def bounds_max_m(self) -> np.ndarray:
        return np.max(self.vertices_m, axis=0)

    @property
    def centre_m(self) -> np.ndarray:
        return (self.bounds_min_m + self.bounds_max_m) * 0.5

    @property
    def extent_m(self) -> np.ndarray:
        return self.bounds_max_m - self.bounds_min_m

    @property
    def triangle_count(self) -> int:
        return int(len(self.faces))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices_m))

    @property
    def object_triangle_count(self) -> int:
        return int(len(self.object_faces))

    @property
    def object_vertex_count(self) -> int:
        return int(len(self.object_vertices_m))

    @property
    def atomic_label_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in np.unique(self.face_label_ids) if int(value) > 0))

    @property
    def has_unclassified_faces(self) -> bool:
        return bool(np.any(self.face_label_ids == UNCLASSIFIED_SCENE_LABEL))

    def node(self, key: str) -> AnalysisRegionNode:
        """Return one hierarchy node by stable analysis-region key."""
        nodes = _node_map(self.hierarchy)
        try:
            return nodes[str(key)]
        except KeyError as error:
            raise AnalysisRegionSceneError(f"Unknown analysis region {key!r}.") from error

    def face_mask_for_node(self, key: str) -> np.ndarray:
        """Return a triangle mask for a hierarchy row, suitable for highlighting."""
        node = self.node(str(key))
        return np.isin(self.face_label_ids, np.asarray(node.label_ids, dtype=np.int32))

    def faces_for_node(self, key: str) -> np.ndarray:
        """Return the selected triangles without changing vertex indices."""
        return self.faces[self.face_mask_for_node(str(key))]


def analysis_region_node_display_metrics(
    scene: AnalysisRegionScene, key: str
) -> dict[str, object]:
    """Return lightweight geometry facts for one hierarchy node's display mesh.

    These values describe the renderable surface only.  They are intentionally
    separate from Exposure Analysis sample/volume statistics so the Region Viewer
    can explain what it is showing without implying that a brain display shell or
    authored bust patch is itself an authoritative dose/volume measurement.
    """
    if not isinstance(scene, AnalysisRegionScene):
        raise AnalysisRegionSceneError("An AnalysisRegionScene is required.")
    node = scene.node(str(key))
    mask = scene.face_mask_for_node(str(key))
    selected_faces = np.asarray(scene.faces[mask], dtype=np.int32)
    if not len(selected_faces):
        raise AnalysisRegionSceneError(f"Region {key!r} has no display triangles.")
    used_indices = np.unique(selected_faces.reshape(-1))
    selected_vertices = np.asarray(scene.vertices_m[used_indices], dtype=np.float64)
    triangle_vertices = np.asarray(scene.vertices_m[selected_faces], dtype=np.float64)
    cross = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    area_m2 = 0.5 * float(np.sum(np.linalg.norm(cross, axis=1)))
    bounds_min = np.min(selected_vertices, axis=0)
    bounds_max = np.max(selected_vertices, axis=0)
    size_mm = (bounds_max - bounds_min) * 1000.0
    return {
        "triangle_count": int(len(selected_faces)),
        "vertex_count": int(len(used_indices)),
        "atomic_label_count": int(len(tuple(node.label_ids))),
        "surface_area_cm2": float(area_m2 * 10_000.0),
        "bounds_min_m": tuple(float(value) for value in bounds_min),
        "bounds_max_m": tuple(float(value) for value in bounds_max),
        "size_mm": tuple(float(value) for value in size_mm),
    }


def analysis_region_scene_identity(scene: AnalysisRegionScene) -> str:
    """Return a deterministic identity for display geometry and region ownership."""
    if not isinstance(scene, AnalysisRegionScene):
        raise AnalysisRegionSceneError("An AnalysisRegionScene is required.")
    hierarchy = [
        {
            "key": node.key,
            "name": node.name,
            "label_ids": list(map(int, node.label_ids)),
            "region_kind": node.region_kind,
        }
        for node in scene.hierarchy.walk()
    ]
    header = {
        "version": ANALYSIS_REGION_SCENE_VERSION,
        "target_id": scene.target_id,
        "target_kind": scene.target_kind,
        "region_set_key": scene.region_set_key,
        "hierarchy": hierarchy,
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, value, dtype in (
        ("vertices_m", scene.vertices_m, np.float32),
        ("faces", scene.faces, np.int32),
        ("face_label_ids", scene.face_label_ids, np.int32),
        ("object_vertices_m", scene.object_vertices_m, np.float32),
        ("object_faces", scene.object_faces, np.int32),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _combine_label_meshes(meshes: dict[int, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine independent per-label triangle meshes into one labelled display mesh."""
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    offset = 0
    for label_id in sorted(int(value) for value in meshes):
        mesh = meshes[label_id]
        current_vertices = np.asarray(mesh.vertices_mm, dtype=np.float32) / 1000.0
        current_faces = np.asarray(mesh.triangles, dtype=np.int32)
        if not len(current_vertices) or not len(current_faces):
            continue
        vertices.append(current_vertices)
        faces.append(current_faces + offset)
        labels.append(np.full(len(current_faces), label_id, dtype=np.int32))
        offset += len(current_vertices)
    if not vertices:
        raise AnalysisRegionSceneError("The selected regional display source has no triangles.")
    return np.vstack(vertices), np.vstack(faces), np.concatenate(labels)


def _brain_scene(adapter: Any, target_id: str, target: dict[str, Any]) -> AnalysisRegionScene:
    atlas = load_sri24_parcellation("lpba40")
    hierarchy = lpba40_region_hierarchy(atlas)
    mesh = adapter._mesh(str(target_id))  # noqa: SLF001 - adapter provider boundary
    position_m, rotation = adapter._mesh_world_transform(mesh)  # noqa: SLF001
    scale_xyz = adapter._mesh_scale_xyz(mesh)  # noqa: SLF001
    try:
        mesh_set = load_lpba40_region_meshes(
            atlas,
            position_m=position_m,
            rotation=rotation,
            scale_xyz=scale_xyz,
        )
    except BrainMeshError as error:
        raise AnalysisRegionSceneError(str(error)) from error
    vertices, faces, face_labels = _combine_label_meshes(mesh_set.regions)
    try:
        object_vertices, object_faces = adapter._mesh_world_vertices_faces(mesh)  # noqa: SLF001
    except Exception:
        # Dependency-light/custom adapters may expose only the atlas display
        # shells.  Falling back keeps the provider usable, while StudioAdapter
        # supplies the actual SRI24 outer object surface in normal application use.
        object_vertices, object_faces = vertices, faces
    colours = {
        int(label_id): _hex_from_rgba(atlas.labels[int(label_id)].rgba)
        for label_id in hierarchy.label_ids
        if int(label_id) in atlas.labels
    }
    return AnalysisRegionScene(
        target_id=str(target_id),
        target_name=str(target.get("label") or "SRI24 brain"),
        target_kind="brain_lpba40",
        region_set_key="sri24.lpba40",
        hierarchy=hierarchy,
        vertices_m=vertices,
        faces=faces,
        face_label_ids=face_labels,
        label_colours=colours,
        base_colour=_normalized_colour(target.get("colour"), "#8b5cf6"),
        base_opacity=float(target.get("opacity", 0.72)),
        object_vertices_m=object_vertices,
        object_faces=object_faces,
        metadata={
            "provider": "sri24_lpba40",
            "atlas": "LPBA40",
            "display_geometry": "bundled_lpba40_region_shells",
            "source_role": str(target.get("role", "visual")),
        },
    )


def _surface_hierarchy(geometry: SurfaceRegionGeometry) -> AnalysisRegionNode:
    children = tuple(
        AnalysisRegionNode(
            key=region.key,
            name=region.name,
            label_ids=(int(region.label_id),),
            region_kind="near_surface_volume",
            metadata={"display_geometry": "authored_surface_patch"},
        )
        for region in geometry.regions
    )
    return AnalysisRegionNode(
        key="surface_regions",
        name="Surface regions",
        label_ids=tuple(int(region.label_id) for region in geometry.regions),
        children=children,
        region_kind="near_surface_volume_set",
        metadata={"display_geometry": "authored_surface"},
    )


def _surface_scene(adapter: Any, target_id: str, target: dict[str, Any]) -> AnalysisRegionScene:
    geometry = adapter.surface_region_geometry(str(target_id))
    if geometry is None:
        raise AnalysisRegionSceneError("This object no longer provides authored surface regions.")
    hierarchy = _surface_hierarchy(geometry)
    label_colours: dict[int, str] = {}
    try:
        region_map = adapter.surface_region_properties(str(target_id)) or {}
    except Exception:  # presentation colour is optional; geometry ownership is authoritative
        region_map = {}
    region_values = region_map.get("regions", {}) if isinstance(region_map, dict) else {}
    for region in geometry.regions:
        settings = {}
        if isinstance(region_values, dict):
            raw = region_values.get(region.key, {})
            if isinstance(raw, dict) and isinstance(raw.get("settings"), dict):
                settings = raw["settings"]
        label_colours[int(region.label_id)] = _normalized_colour(
            settings.get("colour"), target.get("colour", "#64748b")
        )
    face_labels = np.asarray(geometry.face_region_labels, dtype=np.int32).copy()
    # The geometry provider uses negative sentinels for ambiguous/unclassified
    # ownership.  Ambiguity is not eligible for Exposure Analysis; unclassified
    # authored faces remain visible as neutral whole-object geometry.
    face_labels[
        (face_labels == UNCLASSIFIED_REGION_LABEL)
        | (face_labels == AMBIGUOUS_REGION_LABEL)
    ] = UNCLASSIFIED_SCENE_LABEL
    properties = adapter.surface_region_analysis_properties(str(target_id))
    return AnalysisRegionScene(
        target_id=str(target_id),
        target_name=str(target.get("label") or target_id),
        target_kind="surface_region_volume",
        region_set_key=f"surface_regions:{target_id}:inward_surface_volume",
        hierarchy=hierarchy,
        vertices_m=geometry.vertices_m,
        faces=geometry.faces,
        face_label_ids=face_labels,
        label_colours=label_colours,
        base_colour=_normalized_colour(target.get("colour"), "#64748b"),
        base_opacity=float(target.get("opacity", 0.72)),
        object_vertices_m=geometry.vertices_m,
        object_faces=geometry.faces,
        metadata={
            "provider": "surface_region_volume",
            "display_geometry": "authored_surface",
            "sampling_depth_mm": float(properties["depth_mm"]),
            "source_role": str(target.get("role", "visual")),
            "unclassified_face_count": int(np.sum(face_labels == UNCLASSIFIED_SCENE_LABEL)),
        },
    )


def _measurement_scene(adapter: Any, target_id: str, target: dict[str, Any]) -> AnalysisRegionScene:
    if target.get("derived") or str(target.get("role", "")) != "measurement":
        raise AnalysisRegionSceneError("Choose a Measurement-role object from the current model.")
    if not adapter.supports_measurement(str(target_id)):
        raise AnalysisRegionSceneError("The selected Measurement object does not provide volume geometry.")
    mesh = adapter._mesh(str(target_id))  # noqa: SLF001 - adapter provider boundary
    vertices, faces = adapter._mesh_world_vertices_faces(mesh)  # noqa: SLF001
    name = str(target.get("label") or target_id)
    hierarchy = AnalysisRegionNode(
        key=str(target_id),
        name=name,
        label_ids=(1,),
        region_kind="measurement_volume",
        metadata={"display_geometry": "object_surface"},
    )
    return AnalysisRegionScene(
        target_id=str(target_id),
        target_name=name,
        target_kind="measurement_volume",
        region_set_key=f"measurement:{target_id}",
        hierarchy=hierarchy,
        vertices_m=vertices,
        faces=faces,
        face_label_ids=np.ones(len(faces), dtype=np.int32),
        label_colours={1: _normalized_colour(target.get("colour"), "#3b82f6")},
        base_colour=_normalized_colour(target.get("colour"), "#3b82f6"),
        base_opacity=float(target.get("opacity", 0.72)),
        object_vertices_m=vertices,
        object_faces=faces,
        metadata={
            "provider": "measurement_volume",
            "display_geometry": "object_surface",
            "source_role": str(target.get("role", "measurement")),
        },
    )


def resolve_analysis_region_scene(adapter: Any, target_id: str) -> AnalysisRegionScene:
    """Resolve any eligible Exposure Analysis target into display-region geometry.

    Authored surface regions take precedence over the object's scene role.  This
    mirrors the intended Exposure Analysis semantics: assigning the same bust a
    Measurement role must not make its Face/Scalp/etc. region set disappear.
    Objects without authored regions continue to use the whole Measurement-volume
    representation when assigned the Measurement role.
    """
    try:
        target = adapter.get_object(str(target_id))
    except Exception as error:
        raise AnalysisRegionSceneError("The selected analysis target is no longer present.") from error
    if str(target.get("builtin_template", "") or "") == "sri24_brain":
        return _brain_scene(adapter, str(target_id), target)
    if not target.get("derived") and adapter.supports_surface_region_volume(str(target_id)):
        return _surface_scene(adapter, str(target_id), target)
    if (
        not target.get("derived")
        and str(target.get("role", "")) == "measurement"
        and adapter.supports_measurement(str(target_id))
    ):
        return _measurement_scene(adapter, str(target_id), target)
    raise AnalysisRegionSceneError("Choose an eligible Exposure Analysis region target.")
