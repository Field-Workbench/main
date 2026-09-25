"""Shared 3D field-volume grid definitions for static and GPU viewers."""

from __future__ import annotations

import math
from typing import Any, Iterable


FIELD_VOLUME_AXIS_LABELS = (
    "X (LR, mm)",
    "Y (AP, mm)",
    "Z (SI, mm)",
)

_BOX_EDGE_INDICES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _tick_label(value: float) -> str:
    normalized = 0.0 if abs(float(value)) < 1.0e-12 else float(value)
    return f"{normalized:.6g}"



def field_plane_grid_spec(
    plane: str,
    offset_mm: float,
    span_mm: float,
    *,
    enabled: bool = True,
    tick_count: int = 5,
) -> dict[str, Any]:
    """Return a serializable grid contract for one 2D field-map plane.

    The WebGL 2D viewer consumes explicit in-plane line segments so its grid
    follows the same map coordinates as the static Plotly view while remaining
    independent of the camera pan/zoom state.
    """
    plane_name = str(plane).strip().lower()
    axes_by_plane = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
    if plane_name not in axes_by_plane:
        raise ValueError("The field-grid plane must be XY, XZ, or YZ.")
    offset = float(offset_mm)
    span = float(span_mm)
    if not math.isfinite(offset):
        raise ValueError("The field-grid offset must be finite.")
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("The field-grid span must be finite and greater than zero.")
    count = int(tick_count)
    if count < 2:
        raise ValueError("The field grid requires at least two ticks per axis.")

    first_axis, second_axis, normal_axis = axes_by_plane[plane_name]
    half = span / 2.0
    low = [0.0, 0.0, 0.0]
    high = [0.0, 0.0, 0.0]
    low[first_axis] = low[second_axis] = -half
    high[first_axis] = high[second_axis] = half
    low[normal_axis] = high[normal_axis] = offset

    values = [-half + span * index / float(count - 1) for index in range(count)]
    axes: list[dict[str, Any]] = []
    for index, (key, label) in enumerate(zip("xyz", FIELD_VOLUME_AXIS_LABELS)):
        ticks = values if index in {first_axis, second_axis} else []
        axes.append(
            {
                "key": key,
                "label": label,
                "minimum_mm": low[index],
                "maximum_mm": high[index],
                "tick_values_mm": list(ticks),
                "tick_labels": [_tick_label(value) for value in ticks],
            }
        )

    segments: list[list[list[float]]] = []
    for value in values:
        first_start = [0.0, 0.0, 0.0]
        first_end = [0.0, 0.0, 0.0]
        first_start[first_axis] = first_end[first_axis] = value
        first_start[second_axis], first_end[second_axis] = -half, half
        first_start[normal_axis] = first_end[normal_axis] = offset
        segments.append([first_start, first_end])

        second_start = [0.0, 0.0, 0.0]
        second_end = [0.0, 0.0, 0.0]
        second_start[second_axis] = second_end[second_axis] = value
        second_start[first_axis], second_end[first_axis] = -half, half
        second_start[normal_axis] = second_end[normal_axis] = offset
        segments.append([second_start, second_end])

    return {
        "enabled": bool(enabled),
        "plane": plane_name,
        "plane_axes": [first_axis, second_axis],
        "normal_axis": normal_axis,
        "box_min_mm": low,
        "box_max_mm": high,
        "axes": axes,
        "plane_segments_mm": segments,
    }

def field_volume_grid_spec(
    centre_mm: Iterable[float],
    span_mm: float,
    *,
    enabled: bool = True,
    tick_count: int = 5,
) -> dict[str, Any]:
    """Return one serializable grid contract shared by Plotly and WebGL.

    The contract deliberately fixes the axis titles and tick locations so a
    rendered map, realtime playback, export preview, and captured video all use
    the same field-volume framing rather than reconstructing it independently.
    """
    centre = [float(value) for value in centre_mm]
    if len(centre) != 3 or not all(math.isfinite(value) for value in centre):
        raise ValueError("The field-grid centre must contain three finite values.")
    span = float(span_mm)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("The field-grid span must be finite and greater than zero.")
    count = int(tick_count)
    if count < 2:
        raise ValueError("The field grid requires at least two ticks per axis.")

    half = span / 2.0
    low = [value - half for value in centre]
    high = [value + half for value in centre]
    axes: list[dict[str, Any]] = []
    for index, (key, label) in enumerate(zip("xyz", FIELD_VOLUME_AXIS_LABELS)):
        step = (high[index] - low[index]) / float(count - 1)
        ticks = [low[index] + step * tick for tick in range(count)]
        axes.append(
            {
                "key": key,
                "label": label,
                "minimum_mm": low[index],
                "maximum_mm": high[index],
                "tick_values_mm": ticks,
                "tick_labels": [_tick_label(value) for value in ticks],
            }
        )

    corners = [
        [x, y, z]
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    box_segments = [
        [corners[first], corners[second]]
        for first, second in _BOX_EDGE_INDICES
    ]
    return {
        "enabled": bool(enabled),
        "box_min_mm": low,
        "box_max_mm": high,
        "box_segments_mm": box_segments,
        "axes": axes,
    }
