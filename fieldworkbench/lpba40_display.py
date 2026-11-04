"""Shared LPBA40 display styling for Brain View and Exposure Analysis.

The LPBA40 atlas shells are a presentation asset shared by Map Creator's Brain
selector and the Exposure Analysis Region Viewer.  Keeping their trace builder
and style constants here prevents the two viewers from drifting into different
colours, opacity, lighting, or selection semantics.
"""
from __future__ import annotations

from typing import Any

import numpy as np


LPBA40_REGION_OPACITY = 0.34
LPBA40_SELECTED_REGION_OPACITY = 1.0
LPBA40_SELECTED_REGION_COLOUR = "#facc15"
LPBA40_DEFAULT_REGION_COLOUR = "#9ca3af"
LPBA40_DIVISION_COLOURS: dict[str, str] = {
    "frontal": "#4e79a7",
    "parietal": "#59a14f",
    "occipital": "#76b7b2",
    "temporal": "#b07aa1",
    "insula": "#edc948",
    "limbic": "#f28e2b",
    "deep_gray": "#e15759",
    "hindbrain": "#9c755f",
}
LPBA40_REGION_LIGHTING: dict[str, float] = {
    "ambient": 0.58,
    "diffuse": 0.72,
    "specular": 0.16,
    "roughness": 0.78,
    "fresnel": 0.08,
}


def lpba40_division_key_by_label(hierarchy: Any) -> dict[int, str]:
    """Return the major LPBA40 division key for every atomic label."""
    return {
        int(label_id): str(division.key)
        for division in hierarchy.children
        for label_id in division.label_ids
    }


def lpba40_region_style(
    label_id: int,
    *,
    division_key_by_label: dict[int, str],
    selected: bool,
) -> dict[str, Any]:
    """Return the canonical colour/opacity for one LPBA40 shell."""
    division_key = division_key_by_label.get(int(label_id), "")
    return {
        "color": (
            LPBA40_SELECTED_REGION_COLOUR
            if selected
            else LPBA40_DIVISION_COLOURS.get(
                division_key,
                LPBA40_DEFAULT_REGION_COLOUR,
            )
        ),
        "opacity": (
            LPBA40_SELECTED_REGION_OPACITY
            if selected
            else LPBA40_REGION_OPACITY
        ),
    }


def build_lpba40_region_trace(
    *,
    label_id: int,
    vertices_mm: Any,
    triangles: Any,
    display_name: str,
    division_key_by_label: dict[int, str],
    selected: bool = False,
    uid: str | None = None,
    meta: dict[str, Any] | None = None,
    interactive: bool = True,
) -> dict[str, Any]:
    """Build the canonical Plotly Mesh3d trace for one LPBA40 region shell."""
    vertices = np.asarray(vertices_mm, dtype=float)
    faces = np.asarray(triangles, dtype=np.int32)
    style = lpba40_region_style(
        int(label_id),
        division_key_by_label=division_key_by_label,
        selected=bool(selected),
    )
    metadata = {
        "brain_region_label_id": int(label_id),
    }
    if meta:
        metadata.update(meta)
    trace: dict[str, Any] = {
        "type": "mesh3d",
        "x": vertices[:, 0].tolist(),
        "y": vertices[:, 1].tolist(),
        "z": vertices[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "color": style["color"],
        "opacity": style["opacity"],
        "flatshading": False,
        "lighting": dict(LPBA40_REGION_LIGHTING),
        "name": str(display_name),
        "showlegend": False,
        "uid": str(uid or f"lpba40-region-{int(label_id)}"),
        "meta": metadata,
    }
    if interactive:
        trace["hovertemplate"] = str(display_name) + "<extra></extra>"
    else:
        trace["hoverinfo"] = "skip"
        trace["hovertemplate"] = None
        metadata["field_workbench_ignore_picking"] = True
    return trace
