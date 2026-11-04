"""Shared rendering plan for Exposure Analysis region scenes.

This module converts :class:`AnalysisRegionScene` geometry into Plotly-compatible
trace dictionaries without depending on Qt. Generic surface-region targets use
the shared deterministic renderer here, while LPBA40 delegates shell styling to
the same presentation helper used by Maps → Brain.

The source scene remains authoritative. Built-in Workbench region targets render
their complete display geometry: LPBA40 uses every atlas shell triangle, authored
surface regions use every object face, and Measurement targets use their complete
surface. Optional deterministic face sampling is retained only for explicit future
custom/generic callers. No display path changes numerical Exposure Analysis
geometry or sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lpba40_display import (
    build_lpba40_region_trace,
    lpba40_division_key_by_label,
    lpba40_region_style,
)

from .analysis_region_scene import (
    UNCLASSIFIED_SCENE_LABEL,
    AnalysisRegionScene,
    AnalysisRegionSceneError,
    analysis_region_scene_identity,
)


ANALYSIS_REGION_RENDER_VERSION = 6
SELECTED_REGION_COLOUR = "#facc15"
UNCLASSIFIED_REGION_COLOUR = "#94a3b8"


@dataclass(frozen=True)
class AnalysisRegionRenderPlan:
    """A replayable Plotly figure plus stable label-to-trace identities."""

    figure: dict[str, Any]
    trace_index_by_label: dict[int, int]
    rendered_triangle_count: int
    source_triangle_count: int
    scene_identity: str
    triangle_budget: int | None


def _atomic_label_names(scene: AnalysisRegionScene) -> dict[int, str]:
    names: dict[int, str] = {}
    # Prefer exact one-label leaves; they are the most specific human names.
    for node in scene.hierarchy.walk():
        if len(node.label_ids) != 1:
            continue
        label_id = int(node.label_ids[0])
        if label_id > 0:
            names.setdefault(label_id, str(node.name))
    for label_id in scene.atomic_label_ids:
        names.setdefault(int(label_id), f"Region {int(label_id)}")
    return names


def _allocate_face_budget(counts: dict[int, int], budget: int | None) -> dict[int, int]:
    counts = {int(label): int(count) for label, count in counts.items() if int(count) > 0}
    if not counts:
        return {}
    total = sum(counts.values())
    if budget is None or int(budget) <= 0 or int(budget) >= total:
        return dict(counts)

    requested = max(int(budget), len(counts))
    raw = {label: requested * count / total for label, count in counts.items()}
    allocated = {
        label: min(counts[label], max(1, int(np.floor(value))))
        for label, value in raw.items()
    }
    used = sum(allocated.values())

    # Largest-remainder fill keeps the budget proportional while preserving
    # every atomic region, including tiny ear/brain structures.
    if used < requested:
        order = sorted(
            counts,
            key=lambda label: (raw[label] - np.floor(raw[label]), counts[label], -label),
            reverse=True,
        )
        while used < requested:
            changed = False
            for label in order:
                if allocated[label] >= counts[label]:
                    continue
                allocated[label] += 1
                used += 1
                changed = True
                if used >= requested:
                    break
            if not changed:
                break
    elif used > requested:
        order = sorted(
            counts,
            key=lambda label: (raw[label] - np.floor(raw[label]), counts[label], -label),
        )
        while used > requested:
            changed = False
            for label in order:
                if allocated[label] <= 1:
                    continue
                allocated[label] -= 1
                used -= 1
                changed = True
                if used <= requested:
                    break
            if not changed:
                break
    return allocated


def _even_face_sample(indices: np.ndarray, count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    count = min(max(0, int(count)), len(indices))
    if count >= len(indices):
        return indices
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    positions = np.floor((np.arange(count, dtype=float) + 0.5) * len(indices) / count).astype(np.int64)
    return indices[positions]


def _compact_label_mesh(
    scene: AnalysisRegionScene,
    label_id: int,
    face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(scene.faces[np.asarray(face_indices, dtype=np.int64)], dtype=np.int32)
    used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices_mm = np.asarray(scene.vertices_m[used_vertices], dtype=np.float32) * 1000.0
    compact_faces = inverse.reshape((-1, 3)).astype(np.int32, copy=False)
    return vertices_mm, compact_faces


def _normal_opacity(scene: AnalysisRegionScene, label_id: int) -> float:
    if int(label_id) == UNCLASSIFIED_SCENE_LABEL:
        return min(0.28, max(0.10, float(scene.base_opacity) * 0.35))
    # Authored bust regions form one continuous outer surface.  Showing them
    # semi-transparent makes back-facing triangles and region seams visually
    # compete with the front surface, so the normal Region Viewer presentation
    # is fully opaque.  Explicit region highlighting still subdues the other
    # regions in ``analysis_region_trace_styles`` below.
    if str(scene.target_kind) == "surface_region_volume":
        return 1.0
    return min(0.82, max(0.24, float(scene.base_opacity)))


def analysis_region_trace_styles(
    scene: AnalysisRegionScene,
    trace_index_by_label: dict[int, int],
    selected_region_key: str | None = None,
) -> dict[int, dict[str, object]]:
    """Return scalar Plotly styles for a whole-object or highlighted-region view."""
    selected_labels: set[int] | None = None
    key = str(selected_region_key or "").strip()
    if key and key != scene.hierarchy.key:
        selected_labels = {int(value) for value in scene.node(key).label_ids}

    # LPBA40 deliberately mirrors Map Creator's Brain selector: unselected atlas
    # shells retain their normal division colour/opacity while the selected
    # shell(s) switch to the shared yellow highlight.
    if str(scene.target_kind) == "brain_lpba40":
        division_map = lpba40_division_key_by_label(scene.hierarchy)
        return {
            int(trace_index): lpba40_region_style(
                int(label_id),
                division_key_by_label=division_map,
                selected=(
                    selected_labels is not None
                    and int(label_id) in selected_labels
                ),
            )
            for label_id, trace_index in trace_index_by_label.items()
        }

    updates: dict[int, dict[str, object]] = {}
    for label_id, trace_index in trace_index_by_label.items():
        normal_colour = (
            UNCLASSIFIED_REGION_COLOUR
            if int(label_id) == UNCLASSIFIED_SCENE_LABEL
            else scene.label_colours.get(int(label_id), scene.base_colour)
        )
        if selected_labels is None:
            colour = normal_colour
            opacity = _normal_opacity(scene, int(label_id))
        elif int(label_id) in selected_labels:
            colour = SELECTED_REGION_COLOUR
            opacity = 0.96
        else:
            colour = normal_colour
            opacity = 0.07 if int(label_id) == UNCLASSIFIED_SCENE_LABEL else 0.12
        updates[int(trace_index)] = {"color": colour, "opacity": opacity}
    return updates



def _analysis_region_layout(scene: AnalysisRegionScene) -> dict[str, Any]:
    """Return the shared clean, axes-free 3-D layout for anatomy previews."""
    hidden_axis = {
        "visible": False,
        "showgrid": False,
        "showbackground": False,
        "showspikes": False,
        "showticklabels": False,
        "zeroline": False,
        "ticks": "",
        "title": {"text": ""},
    }
    return {
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "showlegend": False,
        "hovermode": False,
        "paper_bgcolor": "#ffffff",
        "meta": {"field_workbench_hide_3d_axes": True},
        "scene": {
            "aspectmode": "data",
            "dragmode": "orbit",
            "bgcolor": "#ffffff",
            "xaxis": dict(hidden_axis),
            "yaxis": dict(hidden_axis),
            "zaxis": dict(hidden_axis),
            "camera": {"projection": {"type": "perspective"}},
        },
        "uirevision": f"analysis-region-scene:{analysis_region_scene_identity(scene)}",
    }


def _build_lpba40_region_render_plan(
    scene: AnalysisRegionScene,
    *,
    selected_region_key: str | None = None,
) -> AnalysisRegionRenderPlan:
    """Render every LPBA40 shell with the same presentation as Brain View."""
    labels = np.asarray(scene.face_label_ids, dtype=np.int32)
    visible_labels = [int(value) for value in np.unique(labels) if int(value) > 0]
    names = _atomic_label_names(scene)
    division_map = lpba40_division_key_by_label(scene.hierarchy)

    traces: list[dict[str, Any]] = []
    trace_index_by_label: dict[int, int] = {}
    rendered_triangles = 0
    for label_id in visible_labels:
        face_indices = np.flatnonzero(labels == int(label_id))
        if not len(face_indices):
            continue
        vertices_mm, faces = _compact_label_mesh(
            scene,
            int(label_id),
            face_indices,
        )
        trace_index_by_label[int(label_id)] = len(traces)
        traces.append(
            build_lpba40_region_trace(
                label_id=int(label_id),
                vertices_mm=vertices_mm,
                triangles=faces,
                display_name=names.get(int(label_id), f"Region {int(label_id)}"),
                division_key_by_label=division_map,
                selected=False,
                uid=f"analysis-region-{scene.target_id}-{int(label_id)}",
                meta={
                    "field_workbench_object_id": (
                        f"analysis-region:{scene.target_id}:{int(label_id)}"
                    ),
                    "analysis_region_target_id": scene.target_id,
                    "analysis_region_label_id": int(label_id),
                },
                interactive=False,
            )
        )
        rendered_triangles += int(len(faces))

    if not traces:
        raise AnalysisRegionSceneError(
            "The selected LPBA40 region scene has no renderable triangles."
        )

    styles = analysis_region_trace_styles(
        scene,
        trace_index_by_label,
        selected_region_key,
    )
    for trace_index, updates in styles.items():
        traces[trace_index].update(updates)

    return AnalysisRegionRenderPlan(
        figure={"data": traces, "layout": _analysis_region_layout(scene)},
        trace_index_by_label=trace_index_by_label,
        rendered_triangle_count=rendered_triangles,
        source_triangle_count=scene.triangle_count,
        scene_identity=analysis_region_scene_identity(scene),
        triangle_budget=None,
    )


def _build_labelled_region_render_plan(
    scene: AnalysisRegionScene,
    *,
    triangle_budget: int | None,
    selected_region_key: str | None = None,
) -> AnalysisRegionRenderPlan:
    """Render a labelled non-LPBA40 region mesh.

    ``triangle_budget`` is deliberately ``None`` for all built-in Workbench
    targets.  A finite budget is supported only as an explicit fallback for
    future custom region providers where topology-preserving display geometry is
    not yet available.
    """
    labels = np.asarray(scene.face_label_ids, dtype=np.int32)
    visible_labels = [int(value) for value in np.unique(labels)]
    counts = {label: int(np.sum(labels == label)) for label in visible_labels}
    allocation = _allocate_face_budget(counts, triangle_budget)
    names = _atomic_label_names(scene)

    traces: list[dict[str, Any]] = []
    trace_index_by_label: dict[int, int] = {}
    rendered_triangles = 0
    for label_id in visible_labels:
        all_indices = np.flatnonzero(labels == int(label_id))
        selected_indices = _even_face_sample(
            all_indices, allocation.get(int(label_id), len(all_indices))
        )
        if not len(selected_indices):
            continue
        vertices_mm, faces = _compact_label_mesh(scene, int(label_id), selected_indices)
        display_name = (
            "Unclassified surface"
            if int(label_id) == UNCLASSIFIED_SCENE_LABEL
            else names.get(int(label_id), f"Region {int(label_id)}")
        )
        colour = (
            UNCLASSIFIED_REGION_COLOUR
            if int(label_id) == UNCLASSIFIED_SCENE_LABEL
            else scene.label_colours.get(int(label_id), scene.base_colour)
        )
        trace_index_by_label[int(label_id)] = len(traces)
        traces.append(
            {
                "type": "mesh3d",
                "x": vertices_mm[:, 0].tolist(),
                "y": vertices_mm[:, 1].tolist(),
                "z": vertices_mm[:, 2].tolist(),
                "i": faces[:, 0].tolist(),
                "j": faces[:, 1].tolist(),
                "k": faces[:, 2].tolist(),
                "color": colour,
                "opacity": _normal_opacity(scene, int(label_id)),
                "flatshading": False,
                "lighting": {
                    "ambient": 0.58,
                    "diffuse": 0.76,
                    "specular": 0.13,
                    "roughness": 0.82,
                    "fresnel": 0.06,
                },
                "name": display_name,
                # Region browsing is tree-driven. Disabling WebGL hover/picking
                # avoids Plotly maintaining an expensive per-triangle pick buffer.
                "hoverinfo": "skip",
                "hovertemplate": None,
                "showlegend": False,
                "uid": f"analysis-region-{scene.target_id}-{int(label_id)}",
                "meta": {
                    "field_workbench_object_id": f"analysis-region:{scene.target_id}:{int(label_id)}",
                    "analysis_region_target_id": scene.target_id,
                    "analysis_region_label_id": int(label_id),
                    "field_workbench_ignore_picking": True,
                },
            }
        )
        rendered_triangles += int(len(faces))

    if not traces:
        raise AnalysisRegionSceneError("The selected region scene has no renderable triangles.")

    styles = analysis_region_trace_styles(scene, trace_index_by_label, selected_region_key)
    for trace_index, updates in styles.items():
        traces[trace_index].update(updates)

    return AnalysisRegionRenderPlan(
        figure={"data": traces, "layout": _analysis_region_layout(scene)},
        trace_index_by_label=trace_index_by_label,
        rendered_triangle_count=rendered_triangles,
        source_triangle_count=scene.triangle_count,
        scene_identity=analysis_region_scene_identity(scene),
        triangle_budget=triangle_budget,
    )



def build_analysis_region_render_plan(
    scene: AnalysisRegionScene,
    *,
    triangle_budget: int | None = None,
    selected_region_key: str | None = None,
) -> AnalysisRegionRenderPlan:
    """Build the axes-free regional representation for one analysis target.

    Built-in Workbench targets always preserve complete display topology.  The
    optional budget is honored only for unknown/custom target kinds.
    """
    if not isinstance(scene, AnalysisRegionScene):
        raise AnalysisRegionSceneError("An AnalysisRegionScene is required for rendering.")

    target_kind = str(scene.target_kind)
    if target_kind == "brain_lpba40":
        return _build_lpba40_region_render_plan(
            scene,
            selected_region_key=selected_region_key,
        )

    if target_kind in {"surface_region_volume", "measurement_volume"}:
        # Built-in object-backed regional displays are small enough to render in
        # full and must not be perforated by individual-face sampling.
        return _build_labelled_region_render_plan(
            scene,
            triangle_budget=None,
            selected_region_key=selected_region_key,
        )

    return _build_labelled_region_render_plan(
        scene,
        triangle_budget=triangle_budget,
        selected_region_key=selected_region_key,
    )
