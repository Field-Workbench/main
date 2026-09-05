"""Pure helpers for waveform video export."""

from __future__ import annotations

import base64
import math
import shutil
import sys
from pathlib import Path
from typing import Any


PNG_DATA_URL_PREFIX = "data:image/png;base64,"
DEFAULT_CAMERA_FOV_DEG = 45.0


def default_video_camera(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the realtime viewer's home camera as explicit world coordinates."""
    try:
        centre = [float(value) for value in payload.get("centre_mm", [0.0, 0.0, 0.0])]
        span = abs(float(payload.get("span_mm", 1.0)))
    except (TypeError, ValueError):
        centre = [0.0, 0.0, 0.0]
        span = 1.0
    if len(centre) != 3 or not all(math.isfinite(value) for value in centre):
        centre = [0.0, 0.0, 0.0]
    if not math.isfinite(span):
        span = 1.0
    half = max(0.5, span / 2.0)
    bounds_min = [centre[index] - half for index in range(3)]
    bounds_max = [centre[index] + half for index in range(3)]
    scene = payload.get("scene")
    if isinstance(scene, dict):
        scene_min = scene.get("bounds_min")
        scene_max = scene.get("bounds_max")
        if isinstance(scene_min, list) and isinstance(scene_max, list):
            if len(scene_min) == 3 and len(scene_max) == 3:
                for index in range(3):
                    try:
                        low = float(scene_min[index])
                        high = float(scene_max[index])
                        if math.isfinite(low) and math.isfinite(high):
                            bounds_min[index] = min(bounds_min[index], low)
                            bounds_max[index] = max(bounds_max[index], high)
                    except (TypeError, ValueError):
                        pass

    target = [
        (bounds_min[index] + bounds_max[index]) / 2.0 for index in range(3)
    ]
    radius = max(
        1.0,
        math.sqrt(
            sum((bounds_max[index] - bounds_min[index]) ** 2 for index in range(3))
        )
        / 2.0,
    )
    yaw = -math.pi / 4.0
    pitch = 0.42
    distance = radius * 2.7
    horizontal = distance * math.cos(pitch)
    position = [
        target[0] + horizontal * math.cos(yaw),
        target[1] + horizontal * math.sin(yaw),
        target[2] + distance * math.sin(pitch),
    ]
    return {"position": position, "target": target, "fov_deg": DEFAULT_CAMERA_FOV_DEG}


def validate_video_camera(camera: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate one camera description."""
    try:
        position = [float(value) for value in camera["position"]]
        target = [float(value) for value in camera["target"]]
        fov_deg = float(camera.get("fov_deg", DEFAULT_CAMERA_FOV_DEG))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Camera position, look-at target, and field of view must be numeric.") from error
    if len(position) != 3 or len(target) != 3:
        raise ValueError("Camera position and look-at target must each contain X, Y, and Z.")
    if not all(math.isfinite(value) for value in [*position, *target, fov_deg]):
        raise ValueError("Camera values must be finite.")
    separation = math.sqrt(sum((position[index] - target[index]) ** 2 for index in range(3)))
    if separation <= 1.0e-6:
        raise ValueError("Camera position must be different from the look-at target.")
    if not 10.0 <= fov_deg <= 120.0:
        raise ValueError("Vertical field of view must stay between 10° and 120°.")
    return {"position": position, "target": target, "fov_deg": fov_deg}


def orbit_video_camera(
    camera: dict[str, Any],
    *,
    progress: float,
    revolutions: float,
) -> dict[str, Any]:
    """Rotate a camera smoothly around its look-at target about the +Z axis."""
    normalized = validate_video_camera(camera)
    try:
        fraction = float(progress)
        turns = float(revolutions)
    except (TypeError, ValueError) as error:
        raise ValueError("Orbit progress and orbit count must be numeric.") from error
    if not math.isfinite(fraction) or not math.isfinite(turns):
        raise ValueError("Orbit progress and orbit count must be finite.")
    if turns <= 0.0:
        raise ValueError("Orbit count must be greater than zero.")

    position = normalized["position"]
    target = normalized["target"]
    offset_x = position[0] - target[0]
    offset_y = position[1] - target[1]
    horizontal_radius = math.hypot(offset_x, offset_y)
    if horizontal_radius <= 1.0e-6:
        raise ValueError(
            "Orbit camera position must have some horizontal offset from the look-at target."
        )

    angle = math.tau * turns * fraction
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rotated_x = offset_x * cos_angle - offset_y * sin_angle
    rotated_y = offset_x * sin_angle + offset_y * cos_angle
    return {
        "position": [
            target[0] + rotated_x,
            target[1] + rotated_y,
            position[2],
        ],
        "target": list(target),
        "fov_deg": normalized["fov_deg"],
    }


def decode_png_data_url(value: Any) -> bytes:
    """Decode and sanity-check a PNG data URL returned by the WebGL renderer."""
    if not isinstance(value, str) or not value.startswith(PNG_DATA_URL_PREFIX):
        raise ValueError("The GPU renderer did not return a PNG frame.")
    try:
        payload = base64.b64decode(value[len(PNG_DATA_URL_PREFIX) :], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("The GPU renderer returned invalid PNG frame data.") from error
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The GPU renderer returned an invalid PNG frame.")
    return payload


_WINDOWS_FFMPEG_RELATIVE = (
    Path("fieldworkbench")
    / "assets"
    / "ffmpeg"
    / "windows-x86_64"
    / "ffmpeg.exe"
)

BUNDLED_WINDOWS_FFMPEG = Path(__file__).resolve().parent / "assets" / "ffmpeg" / "windows-x86_64" / "ffmpeg.exe"


def _windows_ffmpeg_candidates() -> list[Path]:
    """Return source and frozen-layout locations for the bundled encoder.

    PyInstaller 6 onedir releases normally expose ``sys._MEIPASS`` as the
    ``_internal`` directory, but keeping explicit executable-relative fallbacks
    makes the resolver tolerant of older/newer PyInstaller layouts as well as
    direct source runs on Windows.
    """
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / _WINDOWS_FFMPEG_RELATIVE)

    try:
        executable_dir = Path(sys.executable).resolve().parent
    except (OSError, RuntimeError):
        executable_dir = Path(sys.executable).parent
    candidates.extend(
        (
            executable_dir / "_internal" / _WINDOWS_FFMPEG_RELATIVE,
            executable_dir / _WINDOWS_FFMPEG_RELATIVE,
            BUNDLED_WINDOWS_FFMPEG,
        )
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_ffmpeg_executable() -> str | None:
    """Resolve the platform video encoder used by Field Workbench.

    Windows releases carry one tested static FFmpeg build and always use that
    copy. Linux/source environments preserve the previous behavior: look beside
    the launcher/Python executable, then PATH.
    """
    if sys.platform == "win32":
        for candidate in _windows_ffmpeg_candidates():
            if candidate.is_file():
                return str(candidate.resolve())
        return None

    candidates: list[str] = []
    for directory in {Path(sys.executable).resolve().parent, Path(sys.argv[0]).resolve().parent}:
        candidates.extend(str(directory / name) for name in ("ffmpeg.exe", "ffmpeg"))
    discovered = shutil.which("ffmpeg")
    if discovered:
        candidates.append(discovered)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
    return None


def ffmpeg_h264_arguments(
    frame_pattern: Path,
    *,
    frame_rate: float,
    output_path: Path,
) -> list[str]:
    """Build a conservative, editor-friendly H.264 MP4 command line."""
    fps = float(frame_rate)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Video frame rate must be greater than zero.")
    return [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:.9g}",
        "-start_number",
        "0",
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def unique_directory(path: Path) -> Path:
    """Choose a non-existing sibling directory without overwriting user data."""
    candidate = path
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{counter}")
        counter += 1
    return candidate
