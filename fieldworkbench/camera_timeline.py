"""Shared camera/playback timeline helpers for animated maps and video export."""

from __future__ import annotations

import math
from typing import Any

from .numeric_inputs import PLAYBACK_SPEED_MINIMUM
from .video_export import DEFAULT_CAMERA_FOV_DEG, validate_video_camera


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) else float(fallback)


def default_2d_camera(*, centre_mm: list[float] | tuple[float, ...] | None = None, span_mm: float = 1.0) -> dict[str, Any]:
    centre = [0.0, 0.0, 0.0]
    if isinstance(centre_mm, (list, tuple)) and len(centre_mm) >= 3:
        centre = [_finite(centre_mm[index], 0.0) for index in range(3)]
    span = max(1.0e-6, abs(_finite(span_mm, 1.0)))
    return {
        "position": [centre[0], centre[1], centre[2] + max(1.0, span * 2.0)],
        "target": centre,
        "fov_deg": DEFAULT_CAMERA_FOV_DEG,
        "zoom_2d": 1.0,
    }


def normalize_timeline_point(
    point: dict[str, Any] | None,
    *,
    default_camera: dict[str, Any],
    duration_s: float,
    view_mode: str = "3d",
) -> dict[str, Any]:
    source = point if isinstance(point, dict) else {}
    camera_source = source.get("camera") if isinstance(source.get("camera"), dict) else source
    camera = validate_video_camera({
        "position": camera_source.get("position", default_camera.get("position")),
        "target": camera_source.get("target", default_camera.get("target")),
        # Perspective FOV is intentionally fixed. Camera distance is the user-facing zoom control.
        "fov_deg": DEFAULT_CAMERA_FOV_DEG,
    })
    if str(view_mode).lower() == "2d":
        camera["zoom_2d"] = max(
            0.01,
            min(100.0, _finite(camera_source.get("zoom_2d", default_camera.get("zoom_2d", 1.0)), 1.0)),
        )
    return {
        # Timeline authoring is intentionally not capped to the current waveform
        # duration. Later control points are preserved for future/longer playback;
        # consumers ignore points beyond their active physical duration.
        "time_s": max(0.0, _finite(source.get("time_s", 0.0), 0.0)),
        "camera": camera,
        "playback_speed": max(
            PLAYBACK_SPEED_MINIMUM,
            min(100.0, _finite(source.get("playback_speed", 1.0), 1.0)),
        ),
    }


def normalize_camera_timeline(
    timeline: dict[str, Any] | None,
    *,
    duration_s: float,
    default_camera: dict[str, Any],
    view_mode: str = "3d",
) -> dict[str, Any]:
    duration = max(0.0, _finite(duration_s, 0.0))
    source = timeline if isinstance(timeline, dict) else {}
    points_source = source.get("points") if isinstance(source.get("points"), list) else []
    points = [
        normalize_timeline_point(
            point if isinstance(point, dict) else {},
            default_camera=default_camera,
            duration_s=duration,
            view_mode=view_mode,
        )
        for point in points_source
    ]
    if not points:
        points = [
            normalize_timeline_point(
                {"time_s": 0.0, "camera": default_camera, "playback_speed": 1.0},
                default_camera=default_camera,
                duration_s=duration,
                view_mode=view_mode,
            )
        ]
    points.sort(key=lambda item: float(item["time_s"]))
    # A time-zero point is mandatory. If the earliest point drifted away from zero,
    # copy its state to zero rather than inventing a different camera.
    if points[0]["time_s"] > 1.0e-9:
        first = points[0]
        points.insert(
            0,
            {
                "time_s": 0.0,
                "camera": dict(first["camera"]),
                "playback_speed": float(first["playback_speed"]),
            },
        )
    else:
        points[0]["time_s"] = 0.0

    # Duplicate times are ambiguous for interpolation. Last edit wins.
    deduped: list[dict[str, Any]] = []
    for point in points:
        if deduped and abs(point["time_s"] - deduped[-1]["time_s"]) <= 1.0e-9:
            deduped[-1] = point
        else:
            deduped.append(point)
    deduped[0]["time_s"] = 0.0
    return {
        "enabled": bool(source.get("enabled", True)),
        "duration_s": duration,
        "view_mode": str(view_mode).lower(),
        "points": deduped,
    }


def _active_camera_points(timeline: dict[str, Any], points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return control points that participate in the current playback duration.

    Authoring may extend beyond the excitation timeline. Those later points are
    retained in the saved timeline, but do not pull the camera toward a cue that
    the current playback can never reach.
    """
    duration = _finite(timeline.get("duration_s"), math.inf)
    if duration < 0.0:
        duration = 0.0
    active = [
        point
        for point in points
        if isinstance(point, dict) and _finite(point.get("time_s"), 0.0) <= duration + 1.0e-12
    ]
    return active or [points[0]]


def _vector(camera: dict[str, Any], key: str, fallback: list[float]) -> list[float]:
    raw = camera.get(key, fallback)
    if not isinstance(raw, (list, tuple)) or len(raw) < len(fallback):
        raw = fallback
    return [_finite(raw[index], fallback[index]) for index in range(len(fallback))]


def _hermite_value(
    previous: float,
    left: float,
    right: float,
    following: float,
    *,
    previous_time: float,
    left_time: float,
    right_time: float,
    following_time: float,
    time_s: float,
    has_previous: bool,
    has_following: bool,
) -> float:
    """Cubic Hermite interpolation with shared keyframe tangents.

    Unlike per-segment smoothstep, the tangent at an intermediate keyframe is
    shared by the segment on either side, so the camera passes through without
    braking to zero velocity. Equal endpoints are treated as an intentional hold.
    """
    span = right_time - left_time
    if span <= 1.0e-12:
        return float(left)
    u = max(0.0, min(1.0, (time_s - left_time) / span))
    if has_previous and right_time > previous_time + 1.0e-12:
        left_tangent = (right - previous) / (right_time - previous_time)
    else:
        left_tangent = (right - left) / span
    if has_following and following_time > left_time + 1.0e-12:
        right_tangent = (following - left) / (following_time - left_time)
    else:
        right_tangent = (right - left) / span
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return (
        h00 * left
        + h10 * span * left_tangent
        + h01 * right
        + h11 * span * right_tangent
    )


def camera_at_physical_time(timeline: dict[str, Any], time_s: float) -> dict[str, Any]:
    raw_points = timeline.get("points") if isinstance(timeline, dict) else None
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("Camera timeline does not contain a control point.")
    points = _active_camera_points(timeline, raw_points)
    t = max(0.0, _finite(time_s, 0.0))
    if len(points) == 1 or t <= _finite(points[0].get("time_s"), 0.0):
        left_index = right_index = 0
    elif t >= _finite(points[-1].get("time_s"), 0.0):
        left_index = right_index = len(points) - 1
    else:
        right_index = 1
        while right_index < len(points) and _finite(points[right_index].get("time_s"), 0.0) < t:
            right_index += 1
        left_index = max(0, right_index - 1)

    left = points[left_index]
    right = points[right_index]
    left_camera = left.get("camera") if isinstance(left.get("camera"), dict) else {}
    right_camera = right.get("camera") if isinstance(right.get("camera"), dict) else left_camera
    if left_index == right_index:
        result = {
            "position": _vector(left_camera, "position", [0.0, 0.0, 1.0]),
            "target": _vector(left_camera, "target", [0.0, 0.0, 0.0]),
            "fov_deg": DEFAULT_CAMERA_FOV_DEG,
        }
        if "zoom_2d" in left_camera:
            result["zoom_2d"] = max(0.01, _finite(left_camera.get("zoom_2d"), 1.0))
        return result

    previous_index = max(0, left_index - 1)
    following_index = min(len(points) - 1, right_index + 1)
    previous = points[previous_index]
    following = points[following_index]
    previous_camera = previous.get("camera") if isinstance(previous.get("camera"), dict) else left_camera
    following_camera = following.get("camera") if isinstance(following.get("camera"), dict) else right_camera
    previous_time = _finite(previous.get("time_s"), 0.0)
    left_time = _finite(left.get("time_s"), 0.0)
    right_time = _finite(right.get("time_s"), left_time)
    following_time = _finite(following.get("time_s"), right_time)

    def interpolate_vector(key: str, fallback: list[float]) -> list[float]:
        p0 = _vector(previous_camera, key, fallback)
        p1 = _vector(left_camera, key, fallback)
        p2 = _vector(right_camera, key, fallback)
        p3 = _vector(following_camera, key, fallback)
        if all(abs(p2[axis] - p1[axis]) <= 1.0e-12 for axis in range(len(fallback))):
            return p1
        return [
            _hermite_value(
                p0[axis], p1[axis], p2[axis], p3[axis],
                previous_time=previous_time,
                left_time=left_time,
                right_time=right_time,
                following_time=following_time,
                time_s=t,
                has_previous=left_index > 0,
                has_following=right_index + 1 < len(points),
            )
            for axis in range(len(fallback))
        ]

    result = {
        "position": interpolate_vector("position", [0.0, 0.0, 1.0]),
        "target": interpolate_vector("target", [0.0, 0.0, 0.0]),
        "fov_deg": DEFAULT_CAMERA_FOV_DEG,
    }
    if "zoom_2d" in left_camera or "zoom_2d" in right_camera:
        p0 = _finite(previous_camera.get("zoom_2d"), 1.0)
        p1 = _finite(left_camera.get("zoom_2d"), 1.0)
        p2 = _finite(right_camera.get("zoom_2d"), p1)
        p3 = _finite(following_camera.get("zoom_2d"), p2)
        result["zoom_2d"] = max(
            0.01,
            _hermite_value(
                p0, p1, p2, p3,
                previous_time=previous_time,
                left_time=left_time,
                right_time=right_time,
                following_time=following_time,
                time_s=t,
                has_previous=left_index > 0,
                has_following=right_index + 1 < len(points),
            ),
        )
    return result


def playback_speed_at_physical_time(timeline: dict[str, Any] | None, time_s: float) -> float:
    if not isinstance(timeline, dict) or not timeline.get("enabled", False):
        return 1.0
    points = timeline.get("points")
    if not isinstance(points, list) or not points:
        return 1.0
    t = max(0.0, _finite(time_s, 0.0))
    speed = max(
        PLAYBACK_SPEED_MINIMUM,
        _finite(points[0].get("playback_speed"), 1.0),
    )
    for point in points:
        if _finite(point.get("time_s"), 0.0) <= t + 1.0e-12:
            speed = max(
                PLAYBACK_SPEED_MINIMUM,
                _finite(point.get("playback_speed"), speed),
            )
        else:
            break
    return speed


def script_time_for_waveform_duration_s(
    timeline: dict[str, Any] | None, *, waveform_duration_s: float
) -> float:
    """Return script time required for one waveform pass to finish.

    Scripted playback-speed cues are piecewise constant. After the last cue, its
    speed continues indefinitely, so a waveform can finish after the final
    authored camera point without requiring an extra timeline row.
    """
    remaining = max(0.0, _finite(waveform_duration_s, 0.0))
    if remaining <= 0.0:
        return 0.0
    if not isinstance(timeline, dict):
        return remaining
    points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
    boundaries = sorted(
        {
            max(0.0, _finite(point.get("time_s"), 0.0))
            for point in points
            if isinstance(point, dict)
        }
    )
    cursor = 0.0
    for boundary in boundaries:
        if boundary <= cursor + 1.0e-12:
            continue
        speed = max(
            PLAYBACK_SPEED_MINIMUM,
            playback_speed_at_physical_time(timeline, cursor),
        )
        capacity = (boundary - cursor) * speed
        if remaining <= capacity + 1.0e-12:
            return cursor + remaining / speed
        remaining -= capacity
        cursor = boundary
    speed = max(
        PLAYBACK_SPEED_MINIMUM,
        playback_speed_at_physical_time(timeline, cursor),
    )
    return cursor + remaining / speed


def scripted_timeline_duration_s(
    timeline: dict[str, Any] | None, *, fallback_s: float = 0.0
) -> float:
    """Return the effective Scripted-mode output duration.

    The last camera point is the authored camera endpoint. With waveform looping
    disabled, output is automatically extended until one waveform pass finishes;
    the final camera pose simply holds during that extension. With looping enabled,
    the authored camera timeline remains the endpoint.
    """
    waveform_duration = max(0.0, _finite(fallback_s, 0.0))
    if not isinstance(timeline, dict):
        return waveform_duration
    points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
    authored_end = max(
        (_finite(point.get("time_s"), 0.0) for point in points if isinstance(point, dict)),
        default=0.0,
    )
    if bool(timeline.get("loop_waveform", False)):
        return authored_end if authored_end > 1.0e-9 else waveform_duration
    waveform_end = script_time_for_waveform_duration_s(
        timeline, waveform_duration_s=waveform_duration
    )
    return max(authored_end, waveform_end)


def waveform_elapsed_for_script_time(
    timeline: dict[str, Any] | None,
    *,
    script_elapsed_s: float,
    base_real_time_multiplier: float = 1.0,
) -> float:
    """Integrate Scripted-mode playback-speed cues into waveform elapsed time.

    Script time controls the camera and output duration. Point playback speeds
    affect only how quickly the waveform/sequencer advances within that script.
    """
    end = max(0.0, _finite(script_elapsed_s, 0.0))
    base = max(1.0e-9, _finite(base_real_time_multiplier, 1.0))
    if end <= 0.0:
        return 0.0
    if not isinstance(timeline, dict) or not timeline.get("enabled", False):
        return end * base
    points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
    boundaries = [0.0]
    boundaries.extend(
        sorted(
            {
                max(0.0, min(end, _finite(point.get("time_s"), 0.0)))
                for point in points
                if isinstance(point, dict)
            }
        )
    )
    boundaries.append(end)
    boundaries = sorted(set(boundaries))
    total = 0.0
    for start, stop in zip(boundaries, boundaries[1:]):
        if stop <= start:
            continue
        total += (stop - start) * base * playback_speed_at_physical_time(timeline, start)
    return total


def projected_video_duration_s(
    timeline: dict[str, Any] | None,
    *,
    physical_duration_s: float,
    base_real_time_multiplier: float = 1.0,
) -> float:
    duration = max(0.0, _finite(physical_duration_s, 0.0))
    base = max(1.0e-9, _finite(base_real_time_multiplier, 1.0))
    if duration <= 0.0:
        return 0.0
    if isinstance(timeline, dict) and timeline.get("enabled", False) and timeline.get("scripted", False):
        return scripted_timeline_duration_s(timeline, fallback_s=duration) / base
    if not isinstance(timeline, dict) or not timeline.get("enabled", False):
        return duration / base
    points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
    boundaries = [0.0]
    boundaries.extend(
        sorted(
            {
                max(0.0, min(duration, _finite(point.get("time_s"), 0.0)))
                for point in points
                if isinstance(point, dict)
            }
        )
    )
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))
    total = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        total += (end - start) / (base * playback_speed_at_physical_time(timeline, start))
    return total


def physical_time_for_video_elapsed(
    timeline: dict[str, Any] | None,
    *,
    video_elapsed_s: float,
    physical_duration_s: float,
    base_real_time_multiplier: float = 1.0,
) -> float:
    """Map output-video time to physical waveform time for stepped speed cues."""
    duration = max(0.0, _finite(physical_duration_s, 0.0))
    base = max(1.0e-9, _finite(base_real_time_multiplier, 1.0))
    remaining = max(0.0, _finite(video_elapsed_s, 0.0))
    if duration <= 0.0:
        return 0.0
    if isinstance(timeline, dict) and timeline.get("enabled", False) and timeline.get("scripted", False):
        return waveform_elapsed_for_script_time(
            timeline,
            script_elapsed_s=remaining,
            base_real_time_multiplier=base,
        )
    if not isinstance(timeline, dict) or not timeline.get("enabled", False):
        return min(duration, remaining * base)
    points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
    boundaries = [0.0]
    boundaries.extend(
        sorted(
            {
                max(0.0, min(duration, _finite(point.get("time_s"), 0.0)))
                for point in points
                if isinstance(point, dict)
            }
        )
    )
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))
    for start, end in zip(boundaries, boundaries[1:]):
        speed = base * playback_speed_at_physical_time(timeline, start)
        segment_video = (end - start) / speed
        if remaining <= segment_video + 1.0e-12:
            return min(duration, start + remaining * speed)
        remaining -= segment_video
    return duration


_PLOTLY_HOME_EYE_NORM = math.sqrt(1.35**2 + 1.35**2 + 1.1**2)


def _plotly_view_updates(view: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the dotted Plotly relayout map from raw or captured view state."""
    if not isinstance(view, dict):
        return None
    updates = view.get("updates")
    if isinstance(updates, dict):
        return updates
    return view


def _plotly_scene_metrics(view: dict[str, Any]) -> tuple[list[list[float]], list[float], list[float], float, float] | None:
    """Return fitted Plotly scene ranges/centres used for camera conversion."""
    updates = _plotly_view_updates(view)
    if updates is None:
        return None
    ranges: list[list[float]] = []
    try:
        for axis in "xyz":
            raw = updates[f"scene.{axis}axis.range"]
            values = [float(raw[0]), float(raw[1])]
            if not all(math.isfinite(value) for value in values):
                raise ValueError
            ranges.append(values)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    spans = [abs(high - low) for low, high in ranges]
    centres = [(low + high) / 2.0 for low, high in ranges]
    max_span = max(1.0e-9, *spans)
    fit_radius = max(1.0, math.sqrt(sum(span * span for span in spans)) / 2.0)
    return ranges, spans, centres, max_span, fit_radius


def plotly_view_to_video_camera(view: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
    """Approximate Plotly's normalized scene camera as an explicit world camera.

    Plotly expresses ``scene.camera.eye`` in fitted scene coordinates rather than
    millimetres. Field Workbench fits axes with equal physical scale, so preserving
    the eye direction and relative eye magnitude against the fitted world radius
    gives the WebGL renderer the same practical viewpoint. Perspective FOV is fixed;
    camera distance is the single user-facing framing/zoom control.
    """
    base = validate_video_camera({**fallback, "fov_deg": DEFAULT_CAMERA_FOV_DEG})
    updates = _plotly_view_updates(view)
    if updates is None:
        return base
    camera = updates.get("scene.camera")
    metrics = _plotly_scene_metrics(updates)
    if not isinstance(camera, dict) or metrics is None:
        return base
    _ranges, _spans, centres, max_span, fit_radius = metrics
    try:
        eye_raw = camera.get("eye", {})
        centre_raw = camera.get("center", {})
        eye = [float(eye_raw.get(axis, 0.0)) for axis in "xyz"]
        centre_offset = [float(centre_raw.get(axis, 0.0)) for axis in "xyz"]
    except (TypeError, ValueError):
        return base
    if not all(math.isfinite(value) for value in [*eye, *centre_offset]):
        return base
    target = [centres[index] + centre_offset[index] * max_span for index in range(3)]
    eye_norm = math.sqrt(sum(value * value for value in eye))
    if eye_norm <= 1.0e-9:
        return base
    plotly_distance = fit_radius * 2.7 * eye_norm / _PLOTLY_HOME_EYE_NORM
    distance = plotly_distance
    direction = [value / eye_norm for value in eye]
    position = [target[index] + direction[index] * distance for index in range(3)]
    result = validate_video_camera({"position": position, "target": target, "fov_deg": DEFAULT_CAMERA_FOV_DEG})
    if "zoom_2d" in fallback:
        result["zoom_2d"] = float(fallback.get("zoom_2d", 1.0))
    return result


def video_camera_to_plotly_view(camera: dict[str, Any], view: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a Plotly relayout state that previews an explicit 3D video camera.

    The selector's current fitted axis ranges are retained. Position and target are
    converted into Plotly's normalized ``eye``/``center`` coordinates. Perspective
    FOV is fixed, so camera distance directly controls framing in both renderers.
    """
    if not isinstance(view, dict):
        return None
    metrics = _plotly_scene_metrics(view)
    if metrics is None:
        return None
    normalized = validate_video_camera({**camera, "fov_deg": DEFAULT_CAMERA_FOV_DEG})
    _ranges, _spans, centres, max_span, fit_radius = metrics
    target = normalized["target"]
    position = normalized["position"]
    delta = [position[index] - target[index] for index in range(3)]
    distance = math.sqrt(sum(value * value for value in delta))
    if distance <= 1.0e-9:
        return None
    direction = [value / distance for value in delta]
    plotly_distance = distance
    eye_norm = plotly_distance * _PLOTLY_HOME_EYE_NORM / (fit_radius * 2.7)
    eye = {axis: direction[index] * eye_norm for index, axis in enumerate("xyz")}
    center = {axis: (target[index] - centres[index]) / max_span for index, axis in enumerate("xyz")}
    return {
        "kind": "3d",
        "updates": {
            "scene.camera": {
                "eye": eye,
                "center": center,
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "projection": {"type": "perspective"},
            }
        },
    }
