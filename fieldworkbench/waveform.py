"""Quasi-static waveform playback helpers for Field Workbench.

Waveform playback deliberately stays inside the magneto-quasistatic model used by
ordinary Field Workbench maps. A fixed scene is decomposed into a field that is
independent of the playback-controlled coils plus one field-per-ampere basis for
each active drive coil. Time samples can then be synthesized cheaply without
rebuilding or re-solving the scene for every animation frame.
"""

from __future__ import annotations

import base64
import copy
import math
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from .brain_analysis import (
    BrainSampleSet,
    lpba40_frame_region_metrics,
    lpba40_playback_region_metrics,
    lpba40_region_hierarchy,
    weighted_rms_coefficients,
    weighted_rms_from_coefficients,
)
from .field_grid import field_plane_grid_spec, field_volume_grid_spec
from .studio_adapter import (
    FIELD_HEAT_COLOURSCALE,
    FIELD_SIGNED_COLOURSCALE,
    StudioOperationError,
    field_component_is_magnitude,
    field_component_is_signed,
    field_component_label,
    field_component_scalars,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scipy.spatial.transform import Rotation
    from .studio_adapter import StudioAdapter


MAX_ANIMATION_FRAMES = 2_000_000
MAX_ANALYSIS_SAMPLES = 1_000_000
MAX_ANIMATION_SCALAR_VALUES = 12_000_000
MAX_GPU_DRIVE_COILS = 12
MAX_SEQUENCE_EVENTS = 500_000
# Brain Areas playback metrics are useful for the side table, but a metric cube
# with one row for every ultra-slow-motion video frame can otherwise consume
# multiple gigabytes.  Keep that auxiliary cache bounded; the field animation
# itself remains full-frame-rate.
MAX_BRAIN_FRAME_METRIC_CACHE_BYTES = 128 * 1024 * 1024
DAC_TEXT_SUFFIXES = frozenset({".dac", ".txt", ".csv", ".tsv"})


def dac_file_is_text(filename: Any) -> bool:
    """Return whether a waveform file should use the legacy/text DAC parser."""
    try:
        suffix = str(filename).strip().lower().rsplit(".", 1)
    except Exception:  # noqa: BLE001 - filename-like boundary
        return False
    if len(suffix) != 2:
        return False
    return f".{suffix[1]}" in DAC_TEXT_SUFFIXES


def playback_real_time_multiplier(duration_s: float, frame_count: int, playback_fps: float) -> float:
    """Return physical-time / wall-time playback speed for evenly spaced frames."""
    duration = float(duration_s)
    frames = int(frame_count)
    fps = float(playback_fps)
    if not math.isfinite(duration) or duration <= 0.0:
        raise StudioOperationError("Waveform duration must be finite and greater than zero.")
    if not 2 <= frames <= MAX_ANIMATION_FRAMES:
        raise StudioOperationError(
            f"Waveform playback must use between 2 and {MAX_ANIMATION_FRAMES:,} animation frames."
        )
    if not math.isfinite(fps) or fps <= 0.0:
        raise StudioOperationError("Playback FPS must be finite and greater than zero.")
    return duration * fps / float(frames - 1)




def analysis_sample_count(frame_count: int, sample_multiplier: int) -> int:
    """Return the hidden observation-analysis sample count for a playback timeline.

    The multiplier applies to frame *intervals* so both timelines retain the same
    endpoints: 61 frames at 6x becomes 361 analysis samples, not 366.
    """
    frames = int(frame_count)
    multiplier = int(sample_multiplier)
    if not 2 <= frames <= MAX_ANIMATION_FRAMES:
        raise StudioOperationError(
            f"Waveform playback must use between 2 and {MAX_ANIMATION_FRAMES:,} animation frames."
        )
    if multiplier < 1:
        raise StudioOperationError("Waveform analysis sample multiplier must be at least 1x.")
    return min(MAX_ANALYSIS_SAMPLES, 1 + (frames - 1) * multiplier)

def playback_frame_count(duration_s: float, playback_fps: float, real_time_multiplier: float) -> int:
    """Choose the nearest frame count for a target FPS and real-time multiplier."""
    duration = float(duration_s)
    fps = float(playback_fps)
    multiplier = float(real_time_multiplier)
    if not math.isfinite(duration) or duration <= 0.0:
        raise StudioOperationError("Waveform duration must be finite and greater than zero.")
    if not math.isfinite(fps) or fps <= 0.0:
        raise StudioOperationError("Playback FPS must be finite and greater than zero.")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise StudioOperationError("Real-time playback multiplier must be finite and greater than zero.")
    frames = max(2, int(round(duration * fps / multiplier)) + 1)
    return min(MAX_ANIMATION_FRAMES, frames)


@dataclass(frozen=True)
class WaveformTrace:
    """Normalized user-entered drive trace."""

    time_s: np.ndarray
    value: np.ndarray
    kind: str  # "current" (A) or "voltage" (V)


def _strip_optional_dac_point_count(values: np.ndarray) -> np.ndarray:
    """Drop a leading point-count header when it exactly matches the payload length.

    Several legacy DAC formats prefix the sample sequence with its point count.
    We only treat the first value as metadata when there is at least one following
    sample and the value exactly equals the number of remaining entries.
    """
    flat = np.asarray(values).reshape(-1)
    if len(flat) > 1 and int(flat[0]) == len(flat) - 1:
        return flat[1:]
    return flat


def parse_dac_8bit_values(data: bytes, *, text_hint: bool = False) -> np.ndarray:
    """Decode a DAC file into unsigned 8-bit sample values.

    Raw/binary files are interpreted byte-for-byte.  Text-like files may contain
    decimal or ``0x`` hexadecimal values separated by whitespace, commas, or
    semicolons.  A leading point-count value is discarded when it exactly equals
    the number of values that follow it. ``text_hint`` should be used for known
    text extensions so an ASCII-only file is not mistaken for raw byte data.
    """
    payload = bytes(data)
    if not payload:
        raise StudioOperationError("The selected DAC file is empty.")

    if text_hint:
        try:
            text = payload.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise StudioOperationError("The selected text DAC file is not valid UTF-8 text.") from error
        tokens = [token for token in re.split(r"[\s,;]+", text) if token]
        if not tokens:
            raise StudioOperationError("The selected text DAC file does not contain any values.")
        parsed: list[int] = []
        try:
            for token in tokens:
                base = 16 if token.lower().startswith("0x") else 10
                parsed.append(int(token, base))
        except ValueError as error:
            raise StudioOperationError(
                "The selected text DAC file does not contain a valid list of 8-bit values (0 to 255)."
            ) from error
        values = _strip_optional_dac_point_count(np.asarray(parsed, dtype=int))
        if np.any(values < 0) or np.any(values > 255):
            raise StudioOperationError("Text DAC values must be unsigned 8-bit integers from 0 to 255.")
        return values.astype(np.uint8, copy=False)

    values = np.frombuffer(payload, dtype=np.uint8).copy()
    return _strip_optional_dac_point_count(values).astype(np.uint8, copy=False)


def dac_8bit_voltage_trace(
    values: Any,
    *,
    point_delay_ms: float = 3.0,
    cycle_delay_ms: float = 3.0,
    lower_voltage_v: float = -5.0,
    upper_voltage_v: float = 5.0,
) -> WaveformTrace:
    """Convert unsigned 8-bit DAC samples to a voltage command trace.

    DAC code 0 maps to ``lower_voltage_v`` and code 255 maps to
    ``upper_voltage_v``.  Every sample is held for one point delay.  After the
    last sample, the output is commanded to 0 V and held there for the cycle
    delay.  An explicit final 0 V endpoint preserves that dwell in the waveform
    duration.
    """
    try:
        codes = np.asarray(values, dtype=int).reshape(-1)
    except (TypeError, ValueError) as error:
        raise StudioOperationError("DAC values must be 8-bit integers from 0 to 255.") from error
    if not len(codes):
        raise StudioOperationError("The DAC file does not contain any samples.")
    if np.any(codes < 0) or np.any(codes > 255):
        raise StudioOperationError("DAC values must be unsigned 8-bit integers from 0 to 255.")

    point_delay = float(point_delay_ms)
    cycle_delay = float(cycle_delay_ms)
    lower = float(lower_voltage_v)
    upper = float(upper_voltage_v)
    if not math.isfinite(point_delay) or point_delay <= 0.0:
        raise StudioOperationError("DAC point delay must be finite and greater than zero.")
    if not math.isfinite(cycle_delay) or cycle_delay < 0.0:
        raise StudioOperationError("DAC cycle delay must be finite and non-negative.")
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise StudioOperationError("DAC voltage scale requires a finite lower voltage below the upper voltage.")

    scale = (upper - lower) / 255.0
    voltages = lower + codes.astype(float) * scale
    point_delay_s = point_delay / 1000.0
    cycle_delay_s = cycle_delay / 1000.0
    sample_times = np.arange(len(codes), dtype=float) * point_delay_s
    zero_time = float(len(codes)) * point_delay_s

    times = np.concatenate((sample_times, np.asarray([zero_time], dtype=float)))
    output = np.concatenate((voltages, np.asarray([0.0], dtype=float)))
    if cycle_delay_s > 0.0:
        times = np.concatenate((times, np.asarray([zero_time + cycle_delay_s], dtype=float)))
        output = np.concatenate((output, np.asarray([0.0], dtype=float)))
    return validate_trace(times, output, kind="voltage")


def validate_trace(time_s: Any, value: Any, *, kind: str) -> WaveformTrace:
    """Validate and sort one current/voltage DAC command trace.

    Each row is treated as a sample/update time for a zero-order-hold command.
    Duplicate times are rejected so every DAC update has one unambiguous target.
    """
    mode = str(kind).strip().lower()
    if mode not in {"current", "voltage"}:
        raise StudioOperationError("Waveform drive must use current or voltage.")
    try:
        times = np.asarray(time_s, dtype=float).reshape(-1)
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise StudioOperationError("Waveform time and value columns must be numeric.") from error
    if len(times) != len(values) or len(times) < 2:
        raise StudioOperationError("Enter at least two wave points with matching time and value entries.")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise StudioOperationError("Waveform time and value entries must be finite.")
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    if np.any(np.diff(times) <= 0.0):
        raise StudioOperationError("Waveform times must be unique and strictly increasing.")
    if not times[-1] > times[0]:
        raise StudioOperationError("Waveform duration must be greater than zero.")
    return WaveformTrace(time_s=times, value=values, kind=mode)


def _normalized_slew_rate(slew_rate_per_s: float | None) -> float:
    """Return a finite positive slew rate, or infinity for an ideal DAC step."""
    if slew_rate_per_s is None:
        return math.inf
    try:
        rate = float(slew_rate_per_s)
    except (TypeError, ValueError) as error:
        raise StudioOperationError("Waveform slew rate must be numeric.") from error
    if not math.isfinite(rate):
        if rate > 0.0:
            return math.inf
        raise StudioOperationError("Waveform slew rate must be finite and non-negative.")
    if rate < 0.0:
        raise StudioOperationError("Waveform slew rate must be non-negative.")
    # Zero deliberately means an ideal instantaneous DAC step.  This is useful
    # for square-wave playback while a positive value models a finite output
    # slew without pretending to know anything about source bandwidth.
    return math.inf if rate == 0.0 else rate


def _dac_segments(
    trace: WaveformTrace,
    slew_rate_per_s: float | None,
    *,
    initial_value: float | None = None,
) -> list[tuple[float, float, float, float]]:
    """Build continuous linear segments for a sample-and-hold DAC output.

    Returned tuples are ``(start, stop, value_at_start, slope_per_second)``.
    User-entered values are commands applied at their row timestamps and held
    until the next row.  A positive slew limit makes the output move toward each
    new target at no more than that rate; zero/None is an ideal step.
    """
    rate = _normalized_slew_rate(slew_rate_per_s)
    times = trace.time_s
    targets = trace.value
    output = float(targets[0] if initial_value is None else initial_value)
    if not math.isfinite(output):
        raise StudioOperationError("Initial drive value must be finite.")
    segments: list[tuple[float, float, float, float]] = []
    tolerance = max(1e-15, float(times[-1] - times[0]) * 1e-15)

    for index in range(len(times) - 1):
        start = float(times[index])
        stop = float(times[index + 1])
        target = float(targets[index])
        duration = stop - start
        if duration <= 0.0:
            continue

        if math.isinf(rate):
            output = target
            segments.append((start, stop, output, 0.0))
            continue

        delta = target - output
        if abs(delta) <= tolerance:
            output = target
            segments.append((start, stop, output, 0.0))
            continue

        slope = math.copysign(rate, delta)
        slew_duration = abs(delta) / rate
        if slew_duration >= duration - tolerance:
            segments.append((start, stop, output, slope))
            output += slope * duration
            continue

        arrival = start + slew_duration
        segments.append((start, arrival, output, slope))
        output = target
        if stop > arrival + tolerance:
            segments.append((arrival, stop, output, 0.0))

    return segments


def drive_at_times(
    trace: WaveformTrace,
    sample_times_s: Any,
    *,
    slew_rate_per_s: float | None = 0.0,
    initial_value: float | None = None,
) -> np.ndarray:
    """Evaluate the DAC-style commanded current/voltage waveform.

    A row changes the target at its timestamp rather than drawing a straight
    line to the next row. Positive ``slew_rate_per_s`` limits the transition;
    zero means an ideal instantaneous step. Evaluation is vectorized so very
    long GPU playback timelines do not scale through Python one sample at a time.
    """
    requested = np.asarray(sample_times_s, dtype=float).reshape(-1)
    if not len(requested) or not np.isfinite(requested).all():
        raise StudioOperationError("Waveform sample times must be finite.")
    start = float(trace.time_s[0])
    stop = float(trace.time_s[-1])
    tolerance = max(1e-12, abs(stop - start) * 1e-12)
    if np.min(requested) < start - tolerance or np.max(requested) > stop + tolerance:
        raise StudioOperationError("Waveform sample times must stay inside the waveform duration.")
    clipped = np.clip(requested, start, stop)
    ideal_steps = math.isinf(_normalized_slew_rate(slew_rate_per_s))
    if ideal_steps:
        indices = np.searchsorted(trace.time_s, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(trace.value) - 1)
        return np.asarray(trace.value[indices], dtype=float)

    segments = _dac_segments(trace, slew_rate_per_s, initial_value=initial_value)
    if not segments:
        initial = float(trace.value[0] if initial_value is None else initial_value)
        return np.full(len(clipped), initial, dtype=float)
    starts = np.asarray([segment[0] for segment in segments], dtype=float)
    stops = np.asarray([segment[1] for segment in segments], dtype=float)
    values = np.asarray([segment[2] for segment in segments], dtype=float)
    slopes = np.asarray([segment[3] for segment in segments], dtype=float)
    indices = np.searchsorted(starts, clipped, side="right") - 1
    indices = np.clip(indices, 0, len(segments) - 1)
    dt = np.maximum(0.0, np.minimum(clipped, stops[indices]) - starts[indices])
    return values[indices] + slopes[indices] * dt


def current_at_times(
    trace: WaveformTrace,
    sample_times_s: Any,
    *,
    resistance_ohm: float | None = None,
    inductance_h: float | None = None,
    initial_current_a: float = 0.0,
    initial_drive_value: float | None = None,
    slew_rate_per_s: float | None = 0.0,
) -> np.ndarray:
    """Return instantaneous drive current at requested times.

    Current traces use a sample-and-hold/DAC waveform with an optional finite
    slew rate. Voltage traces solve ``V = R I + L dI/dt`` exactly on each
    constant or linear DAC-output segment. The segment solution is vectorized
    across requested times, which keeps million-sample GPU timelines practical.
    Mutual inductance and frequency-dependent impedance remain outside this ELF model.
    """
    requested = np.asarray(sample_times_s, dtype=float).reshape(-1)
    if not len(requested) or not np.isfinite(requested).all():
        raise StudioOperationError("Waveform sample times must be finite.")
    start = float(trace.time_s[0])
    stop = float(trace.time_s[-1])
    tolerance = max(1e-12, abs(stop - start) * 1e-12)
    if np.min(requested) < start - tolerance or np.max(requested) > stop + tolerance:
        raise StudioOperationError("Waveform sample times must stay inside the waveform duration.")
    clipped = np.clip(requested, start, stop)
    if trace.kind == "current":
        return drive_at_times(
            trace, clipped, slew_rate_per_s=slew_rate_per_s, initial_value=initial_drive_value
        )

    try:
        resistance = float(resistance_ohm)
        inductance = float(inductance_h)
        initial = float(initial_current_a)
    except (TypeError, ValueError) as error:
        raise StudioOperationError("Voltage playback requires numeric R, L, and initial-current values.") from error
    if not math.isfinite(resistance) or resistance < 0.0:
        raise StudioOperationError("Waveform resistance must be finite and non-negative.")
    if not math.isfinite(inductance) or inductance <= 0.0:
        raise StudioOperationError("Waveform inductance must be finite and greater than zero.")
    if not math.isfinite(initial):
        raise StudioOperationError("Initial current must be finite.")

    segments = _dac_segments(trace, slew_rate_per_s, initial_value=initial_drive_value)
    if not segments:
        return np.full(len(clipped), initial, dtype=float)

    starts = np.asarray([segment[0] for segment in segments], dtype=float)
    stops = np.asarray([segment[1] for segment in segments], dtype=float)
    voltages = np.asarray([segment[2] for segment in segments], dtype=float)
    slopes = np.asarray([segment[3] for segment in segments], dtype=float)
    start_currents = np.empty(len(segments), dtype=float)

    def advance(current_value: float, voltage_value: float, slope_value: float, dt_value: float) -> float:
        if resistance <= 1e-15:
            return current_value + (voltage_value * dt_value + 0.5 * slope_value * dt_value * dt_value) / inductance
        decay_rate = resistance / inductance
        exponent = -decay_rate * dt_value
        decay = math.exp(exponent)
        one_minus_decay = -math.expm1(exponent)
        return (
            current_value * decay
            + (voltage_value / resistance) * one_minus_decay
            + (slope_value / resistance) * (dt_value - one_minus_decay / decay_rate)
        )

    current_value = initial
    for index in range(len(segments)):
        start_currents[index] = current_value
        current_value = advance(
            current_value,
            float(voltages[index]),
            float(slopes[index]),
            float(stops[index] - starts[index]),
        )

    indices = np.searchsorted(starts, clipped, side="right") - 1
    indices = np.clip(indices, 0, len(segments) - 1)
    dt = np.maximum(0.0, np.minimum(clipped, stops[indices]) - starts[indices])
    base_current = start_currents[indices]
    v0 = voltages[indices]
    slope = slopes[indices]
    if resistance <= 1e-15:
        return base_current + (v0 * dt + 0.5 * slope * dt * dt) / inductance
    decay_rate = resistance / inductance
    exponent = -decay_rate * dt
    decay = np.exp(exponent)
    one_minus_decay = -np.expm1(exponent)
    return (
        base_current * decay
        + (v0 / resistance) * one_minus_decay
        + (slope / resistance) * (dt - one_minus_decay / decay_rate)
    )



def normalize_coil_sequence(
    steps: Any,
    *,
    available_coil_ids: Any,
) -> list[dict[str, Any]]:
    """Validate sequencer rows and normalize durations to seconds.

    Each step is a mapping with ``duration_s`` and ``active_coil_ids``. Empty
    active sets are allowed deliberately so a sequencer can include explicit
    all-off dwell periods.
    """
    available = {str(value) for value in available_coil_ids}
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(list(steps or [])):
        if not isinstance(step, dict):
            raise StudioOperationError(f"Sequencer step {index + 1} is invalid.")
        try:
            duration = float(step.get("duration_s"))
        except (TypeError, ValueError) as error:
            raise StudioOperationError(f"Sequencer step {index + 1} duration must be numeric.") from error
        if not math.isfinite(duration) or duration <= 0.0:
            raise StudioOperationError(f"Sequencer step {index + 1} duration must be greater than zero.")
        active = tuple(dict.fromkeys(str(value) for value in step.get("active_coil_ids", []) if str(value)))
        unknown = [object_id for object_id in active if object_id not in available]
        if unknown:
            raise StudioOperationError(f"Sequencer step {index + 1} contains a coil that is no longer available.")
        normalized.append({"duration_s": duration, "active_coil_ids": active})
    if not normalized:
        raise StudioOperationError("Advanced coil sequencing requires at least one step.")
    return normalized


def _sequence_boundary_times(
    start_s: float,
    stop_s: float,
    steps: list[dict[str, Any]],
    *,
    loop: bool,
) -> np.ndarray:
    """Return step-transition times inside one waveform trace duration."""
    durations = np.asarray([float(step["duration_s"]) for step in steps], dtype=float)
    cycle = float(np.sum(durations))
    if not math.isfinite(cycle) or cycle <= 0.0:
        raise StudioOperationError("The coil sequencer must have a positive total duration.")
    internal = np.cumsum(durations)[:-1]
    duration = float(stop_s - start_s)
    boundaries: list[float] = []
    if loop:
        cycles = int(math.ceil(duration / cycle)) + 1
        estimated = cycles * max(1, len(steps))
        if estimated > MAX_SEQUENCE_EVENTS:
            raise StudioOperationError(
                "The coil sequencer would create too many switching events. Increase step times or shorten the waveform."
            )
        for cycle_index in range(cycles):
            base = float(start_s + cycle_index * cycle)
            if cycle_index and base < stop_s:
                boundaries.append(base)
            for value in internal:
                event = base + float(value)
                if start_s < event < stop_s:
                    boundaries.append(event)
    else:
        for value in internal:
            event = start_s + float(value)
            if start_s < event < stop_s:
                boundaries.append(event)
        end = start_s + cycle
        if start_s < end < stop_s:
            boundaries.append(end)
    if not boundaries:
        return np.empty(0, dtype=float)
    values = np.sort(np.round(np.asarray(boundaries, dtype=float), 15))
    tolerance = max(1e-15, duration * 1e-13)
    keep = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        keep[1:] = np.diff(values) > tolerance
    return values[keep]


def _sequence_step_indices(
    times_s: Any,
    *,
    start_s: float,
    steps: list[dict[str, Any]],
    loop: bool,
) -> np.ndarray:
    """Return zero-based sequencer step indices, or -1 after a non-looping sequence ends."""
    times = np.asarray(times_s, dtype=float).reshape(-1)
    durations = np.asarray([float(step["duration_s"]) for step in steps], dtype=float)
    cumulative = np.cumsum(durations)
    cycle = float(cumulative[-1])
    elapsed = np.maximum(0.0, times - float(start_s))
    tolerance = max(1e-15, cycle * 1e-12)
    if loop:
        phase = np.mod(elapsed, cycle)
        # Floating-point modulo can report a value infinitesimally below the
        # cycle length at an exact loop boundary. Snap all sequencer boundaries
        # before choosing the active step so transitions are deterministic.
        phase[np.isclose(phase, cycle, rtol=0.0, atol=tolerance)] = 0.0
        phase[np.isclose(phase, 0.0, rtol=0.0, atol=tolerance)] = 0.0
        for boundary in cumulative[:-1]:
            phase[np.isclose(phase, boundary, rtol=0.0, atol=tolerance)] = boundary
        indices = np.searchsorted(cumulative, phase, side="right")
        return np.clip(indices, 0, len(steps) - 1).astype(int)

    valid = elapsed < cycle - tolerance
    phase = np.minimum(elapsed, max(0.0, cycle - tolerance))
    for boundary in cumulative[:-1]:
        phase[np.isclose(phase, boundary, rtol=0.0, atol=tolerance)] = boundary
    indices = np.searchsorted(cumulative, phase, side="right")
    indices = np.clip(indices, 0, len(steps) - 1).astype(int)
    return np.where(valid, indices, -1)


def _sequence_active_mask(
    times_s: Any,
    coil_id: str,
    *,
    start_s: float,
    steps: list[dict[str, Any]],
    loop: bool,
) -> np.ndarray:
    """Return whether one coil is active at each sequencer timestamp."""
    indices = _sequence_step_indices(
        times_s, start_s=start_s, steps=steps, loop=loop
    )
    active_by_step = np.asarray(
        [str(coil_id) in set(step["active_coil_ids"]) for step in steps], dtype=bool
    )
    valid = indices >= 0
    safe = np.clip(indices, 0, len(steps) - 1)
    return valid & active_by_step[safe]


def sequenced_drive_trace(
    trace: WaveformTrace,
    *,
    coil_id: str,
    steps: list[dict[str, Any]] | None,
    loop: bool,
) -> WaveformTrace:
    """Gate one DAC command trace with the optional coil sequencer.

    Sequencer transitions become additional DAC command timestamps. An inactive
    coil is commanded to zero. Therefore current-drive slew still applies to
    on/off transitions, while voltage drive naturally lets RL current decay when
    a step commands 0 V.
    """
    if not steps:
        return trace
    start = float(trace.time_s[0])
    stop = float(trace.time_s[-1])
    boundaries = _sequence_boundary_times(start, stop, steps, loop=loop)
    events = np.sort(np.concatenate([np.asarray(trace.time_s, dtype=float), boundaries]))
    tolerance = max(1e-15, (stop - start) * 1e-13)
    if len(events) > 1:
        keep = np.ones(len(events), dtype=bool)
        keep[1:] = np.diff(events) > tolerance
        events = events[keep]
    source_indices = np.searchsorted(trace.time_s, events, side="right") - 1
    source_indices = np.clip(source_indices, 0, len(trace.value) - 1)
    targets = np.asarray(trace.value[source_indices], dtype=float)
    active = _sequence_active_mask(events, str(coil_id), start_s=start, steps=steps, loop=loop)
    values = np.where(active, targets, 0.0)
    return validate_trace(events, values, kind=trace.kind)


def _prepare_multi_coil_timeline(
    trace: WaveformTrace,
    *,
    coil_ids: list[str],
    frame_count: int,
    analysis_samples: int,
    electrical_by_coil: dict[str, dict[str, float | None]] | None,
    initial_drive_value: float | None,
    slew_rate_per_s: float | None,
    sequence_steps: list[dict[str, Any]] | None,
    sequence_loop: bool,
) -> dict[str, np.ndarray]:
    """Resolve one current timeline per driven coil."""
    frames = int(frame_count)
    samples = int(analysis_samples)
    if not 2 <= frames <= MAX_ANIMATION_FRAMES:
        raise StudioOperationError(
            f"Waveform playback must use between 2 and {MAX_ANIMATION_FRAMES:,} animation frames."
        )
    if not 2 <= samples <= MAX_ANALYSIS_SAMPLES:
        raise StudioOperationError(
            f"Waveform analysis must use between 2 and {MAX_ANALYSIS_SAMPLES:,} time samples."
        )
    start, stop = float(trace.time_s[0]), float(trace.time_s[-1])
    frame_times = np.linspace(start, stop, frames)
    analysis_times = np.linspace(start, stop, samples)
    frame_currents = np.empty((frames, len(coil_ids)), dtype=float)
    free_frame_currents = np.empty((frames, len(coil_ids)), dtype=float)
    analysis_currents = np.empty((samples, len(coil_ids)), dtype=float)
    for column, coil_id in enumerate(coil_ids):
        gated = sequenced_drive_trace(
            trace, coil_id=coil_id, steps=sequence_steps, loop=sequence_loop
        )
        active_at_start = True
        if sequence_steps:
            active_at_start = bool(
                _sequence_active_mask(
                    [start], coil_id, start_s=start, steps=sequence_steps, loop=sequence_loop
                )[0]
            )
        coil_initial_drive = initial_drive_value if active_at_start else 0.0
        electrical = (electrical_by_coil or {}).get(coil_id, {})
        resistance = electrical.get("resistance_ohm")
        inductance = electrical.get("inductance_h")
        initial_current = float(electrical.get("initial_current_a") or 0.0) if active_at_start else 0.0
        frame_currents[:, column] = current_at_times(
            gated,
            frame_times,
            resistance_ohm=resistance,
            inductance_h=inductance,
            initial_current_a=initial_current,
            initial_drive_value=coil_initial_drive,
            slew_rate_per_s=slew_rate_per_s,
        )
        # Keep an ungated, per-coil current template for the realtime viewer.
        # The first excitation pass still uses the fully resolved sequenced
        # current above (including voltage-drive decay and slew). If excitation
        # playback loops, the viewer can apply the independently advancing
        # sequencer phase to this template instead of replaying the baked first
        # pass and therefore rewinding the sequence with the waveform.
        if sequence_steps:
            free_initial_current = float(electrical.get("initial_current_a") or 0.0)
            free_frame_currents[:, column] = current_at_times(
                trace,
                frame_times,
                resistance_ohm=resistance,
                inductance_h=inductance,
                initial_current_a=free_initial_current,
                initial_drive_value=initial_drive_value,
                slew_rate_per_s=slew_rate_per_s,
            )
        else:
            free_frame_currents[:, column] = frame_currents[:, column]
        analysis_currents[:, column] = current_at_times(
            gated,
            analysis_times,
            resistance_ohm=resistance,
            inductance_h=inductance,
            initial_current_a=initial_current,
            initial_drive_value=coil_initial_drive,
            slew_rate_per_s=slew_rate_per_s,
        )
    return {
        "frame_times": frame_times,
        "analysis_times": analysis_times,
        "frame_currents": frame_currents,
        "free_frame_currents": free_frame_currents,
        "analysis_currents": analysis_currents,
    }


def time_statistics(time_s: Any, values: Any) -> dict[str, float]:
    """Time-weighted mean/DC, RMS and peak-to-peak statistics."""
    times = np.asarray(time_s, dtype=float).reshape(-1)
    samples = np.asarray(values, dtype=float).reshape(-1)
    if len(times) != len(samples) or len(times) < 2:
        raise StudioOperationError("Waveform statistics require at least two time samples.")
    finite = np.isfinite(times) & np.isfinite(samples)
    if np.count_nonzero(finite) < 2:
        raise StudioOperationError("Waveform statistics require finite time and signal samples.")
    times = times[finite]
    samples = samples[finite]
    order = np.argsort(times, kind="stable")
    times = times[order]
    samples = samples[order]
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise StudioOperationError("Waveform statistics require a positive time span.")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # pragma: no cover - compatibility with older NumPy
        integrate = np.trapz
    mean = float(integrate(samples, times) / duration)
    rms = float(math.sqrt(max(0.0, integrate(samples * samples, times) / duration)))
    return {
        "mean": mean,
        "rms": rms,
        "pk_pk": float(np.max(samples) - np.min(samples)),
    }


def drive_coil_catalog(adapter: "StudioAdapter") -> list[dict[str, Any]]:
    """Return every physical coil that can be driven by playback.

    Scene-disabled coils remain selectable: waveform playback is an independent
    drive context, so selecting one temporarily energizes its unit-current basis
    without changing the source scene's enabled state.
    """
    choices: list[dict[str, Any]] = []
    for item in adapter.optimizer_existing_coils():
        object_id = str(item["id"])
        resistance = None
        try:
            physical = adapter.coil_physical_properties(object_id)
            resistance = float(physical["calculations"]["resistance_operating_ohm"])
        except Exception:  # noqa: BLE001 - optional convenience metadata
            resistance = None
        choices.append(
            {
                "id": object_id,
                "label": str(item.get("label") or object_id),
                "drive_current_a": float(item.get("drive_current_mA", 0.0)) / 1000.0,
                "resistance_ohm": resistance,
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return choices


def observation_catalog(adapter: "StudioAdapter") -> list[dict[str, Any]]:
    """Return ordinary sensors and physical magnetometer probes."""
    result: list[dict[str, Any]] = []
    for sensor in adapter.measurement_sensor_catalog():
        try:
            _x, points, _rotations, _label = adapter._sensor_sample_geometry(str(sensor["id"]))
            sample_count = int(len(points))
        except Exception:  # noqa: BLE001 - stale/unsupported sensor is not selectable
            continue
        result.append(
            {
                "kind": "sensor",
                "id": str(sensor["id"]),
                "label": str(sensor.get("label") or sensor["id"]),
                "sample_count": max(1, sample_count),
            }
        )
    for item in adapter.magnetometer_probes:
        object_id = str(item.get("id", ""))
        if not object_id:
            continue
        definition = item.get("definition", {})
        channels = list(definition.get("channels", [])) if isinstance(definition, dict) else []
        result.append(
            {
                "kind": "probe",
                "id": object_id,
                "label": str(item.get("label") or object_id),
                "channels": [
                    {
                        "name": str(channel.get("name") or f"Channel {index + 1}"),
                        "output": str(channel.get("output") or channel.get("axis_label") or channel.get("name") or f"Channel {index + 1}"),
                    }
                    for index, channel in enumerate(channels)
                ],
            }
        )
    return result


def _copy_adapter_for_basis(adapter: "StudioAdapter") -> "StudioAdapter":
    # Import locally to avoid importing the large adapter module merely to use
    # the pure waveform math helpers in unit tests.
    from .studio_adapter import StudioAdapter

    clone = StudioAdapter(copy.deepcopy(adapter.document()))
    clone.set_coil_modeling_method(adapter.coil_modeling_method)
    clone.set_solver_memory_budget_mb(adapter.solver_memory_budget_mb)
    return clone


def field_bases_for_drives(
    adapter: "StudioAdapter",
    *,
    coil_ids: Any,
    points_m: Any,
    field: str,
    zero_coil_ids: Any | None = None,
    progress_callback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed field plus one per-ampere basis for every driven coil.

    All controlled coils are first zeroed together, so their original scene
    currents cannot leak into the fixed field or another coil's basis. By default
    the controlled set is exactly ``coil_ids``; ``zero_coil_ids`` may include
    additional coils that should remain off for the entire playback. Sources not
    in that controlled set remain exactly as they are in the source scene.
    ``bases`` is returned with shape ``(coil_count, point_count, 3)``.
    """
    object_ids = list(dict.fromkeys(str(value) for value in coil_ids if str(value)))
    if not object_ids:
        raise StudioOperationError("Choose at least one coil for waveform playback.")
    if len(object_ids) > MAX_GPU_DRIVE_COILS:
        raise StudioOperationError(
            f"GPU waveform playback currently supports up to {MAX_GPU_DRIVE_COILS} driven coils at once."
        )
    zero_ids = list(
        dict.fromkeys(
            str(value)
            for value in (object_ids if zero_coil_ids is None else zero_coil_ids)
            if str(value)
        )
    )
    # A driven coil must always be part of the controlled/zeroed set.
    zero_ids = list(dict.fromkeys([*zero_ids, *object_ids]))
    invalid = [object_id for object_id in zero_ids if not adapter.is_coil(object_id)]
    if invalid:
        raise StudioOperationError("Choose only valid coils for waveform playback.")
    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise StudioOperationError("Waveform playback requires finite field sample coordinates.")

    total_phases = len(object_ids) + 1

    def phase_progress(phase: int, prefix: str):
        if progress_callback is None:
            return None

        def relay(update: dict[str, Any]) -> None:
            completed = max(0, int(update.get("completed", 0) or 0))
            total = max(1, int(update.get("total", 1) or 1))
            fraction = min(1.0, max(0.0, completed / total))
            progress_callback(
                {
                    "completed": int(round((phase + fraction) * 1000.0)),
                    "total": total_phases * 1000,
                    "stage": f"{prefix}: {update.get('stage', 'sampling field')}",
                }
            )

        return relay

    zero = _copy_adapter_for_basis(adapter)
    zero.apply_operations(
        [
            {
                "method": "set_coil_physical",
                "params": {"object_id": object_id, "drive_current_a": 0.0},
            }
            for object_id in zero_ids
        ]
    )
    fixed_result = zero.get_field(
        points=points,
        field=str(field).upper(),
        progress_callback=phase_progress(0, "Fixed scene"),
        progress_label="Waveform fixed scene",
    )
    fixed = np.asarray(fixed_result["values"], dtype=float).reshape(-1, 3)

    bases: list[np.ndarray] = []
    for index, object_id in enumerate(object_ids):
        unit = _copy_adapter_for_basis(zero)
        unit.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": object_id, "drive_current_a": 1.0, "enabled": True},
                }
            ]
        )
        unit_result = unit.get_field(
            points=points,
            field=str(field).upper(),
            progress_callback=phase_progress(index + 1, f"{object_id} basis"),
            progress_label="Preparing waveform field",
        )
        unit_vectors = np.asarray(unit_result["values"], dtype=float).reshape(-1, 3)
        bases.append(unit_vectors - fixed)
    return fixed, np.stack(bases, axis=0)


def field_basis_for_drive(
    adapter: "StudioAdapter",
    *,
    coil_id: str,
    points_m: Any,
    field: str,
    progress_callback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible single-coil wrapper around :func:`field_bases_for_drives`."""
    fixed, bases = field_bases_for_drives(
        adapter, coil_ids=[coil_id], points_m=points_m, field=field, progress_callback=progress_callback
    )
    return fixed, bases[0]


def validate_animation_field_basis(
    adapter: "StudioAdapter",
    *,
    coil_ids: Any,
    controlled_coil_ids: Any,
    centre_mm: Any,
    span_mm: float,
    resolution: int,
    field: str = "B",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Numerically validate the animation's linear field-basis reconstruction.

    The test deliberately uses the exact Full Volume sample lattice used by the
    realtime 3D renderer.  It first builds the normal animation decomposition
    ``fixed + sum(current * basis)`` and then performs fresh direct field solves
    for several deterministic current vectors.  The two vector fields are
    compared point-by-point; no rendering or colour/opacity code participates.
    """
    drive_ids = list(dict.fromkeys(str(value) for value in coil_ids if str(value)))
    controlled_ids = list(
        dict.fromkeys(str(value) for value in controlled_coil_ids if str(value))
    )
    controlled_ids = list(dict.fromkeys([*controlled_ids, *drive_ids]))
    if not drive_ids:
        raise StudioOperationError("Choose at least one active coil before validating animation fields.")
    if any(not adapter.is_coil(object_id) for object_id in controlled_ids):
        raise StudioOperationError("Animation validation received an invalid controlled coil.")

    centre = np.asarray(centre_mm, dtype=float).reshape(3)
    span = float(span_mm)
    samples = int(resolution)
    if not np.isfinite(centre).all() or not math.isfinite(span) or span <= 0.0:
        raise StudioOperationError("Animation validation requires a finite positive map volume.")
    if samples < 2:
        raise StudioOperationError("Animation validation requires at least 2 samples per axis.")

    half = span / 2.0
    coordinates = [
        np.linspace(centre[index] - half, centre[index] + half, samples)
        for index in range(3)
    ]
    # Match build_3d_gpu_playback's X-fastest texture/sample ordering exactly.
    z_grid, y_grid, x_grid = np.meshgrid(
        coordinates[2], coordinates[1], coordinates[0], indexing="ij"
    )
    points_mm = np.column_stack(
        [x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)]
    )
    points_m = points_mm / 1000.0

    catalog = {str(item["id"]): item for item in drive_coil_catalog(adapter)}
    scene_currents = np.asarray(
        [float(catalog.get(object_id, {}).get("drive_current_a", 0.0)) for object_id in drive_ids],
        dtype=float,
    )
    nonzero = np.abs(scene_currents[np.abs(scene_currents) > 1e-12])
    fallback = float(np.max(nonzero)) if nonzero.size else 0.05
    magnitudes = np.where(np.abs(scene_currents) > 1e-12, np.abs(scene_currents), fallback)
    alternating = magnitudes * np.asarray(
        [1.0 if index % 2 == 0 else -1.0 for index in range(len(drive_ids))],
        dtype=float,
    )
    fractions = np.asarray(
        [0.37, -0.61, 0.83, -0.29, 0.53, -0.71, 0.19, -0.43, 0.67, -0.11, 0.47, -0.89],
        dtype=float,
    )[: len(drive_ids)]
    fractional = magnitudes * fractions

    raw_states: list[tuple[str, np.ndarray]] = [
        ("Zero", np.zeros(len(drive_ids), dtype=float)),
        ("Scene currents", scene_currents),
        ("Negative scene currents", -scene_currents),
        ("Alternating signs", alternating),
        ("Fractional mixed", fractional),
    ]
    states: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for label, currents in raw_states:
        key = tuple(np.round(np.asarray(currents, dtype=float), 15).tolist())
        if key in seen:
            continue
        seen.add(key)
        states.append((label, np.asarray(currents, dtype=float)))

    basis_share = 0.45

    def basis_progress(update: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        total = max(1, int(update.get("total", 1) or 1))
        completed = max(0, min(total, int(update.get("completed", 0) or 0)))
        progress_callback(
            {
                "completed": int(round(10000 * basis_share * completed / total)),
                "total": 10000,
                "stage": f"Validation basis: {update.get('stage', 'sampling field')}",
            }
        )

    fixed, bases = field_bases_for_drives(
        adapter,
        coil_ids=drive_ids,
        zero_coil_ids=controlled_ids,
        points_m=points_m,
        field=str(field).upper(),
        progress_callback=basis_progress,
    )

    zero_adapter = _copy_adapter_for_basis(adapter)
    zero_adapter.apply_operations(
        [
            {
                "method": "set_coil_physical",
                "params": {"object_id": object_id, "drive_current_a": 0.0},
            }
            for object_id in controlled_ids
        ]
    )

    results: list[dict[str, Any]] = []
    field_name = str(field).upper()
    scale = 1e6 if field_name == "B" else 1.0
    unit = "µT" if field_name == "B" else "A/m"
    direct_share = 1.0 - basis_share

    for state_index, (label, currents) in enumerate(states):
        direct_adapter = _copy_adapter_for_basis(zero_adapter)
        direct_adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": object_id,
                        "drive_current_a": float(currents[index]),
                        "enabled": True,
                    },
                }
                for index, object_id in enumerate(drive_ids)
            ]
        )

        def direct_progress(update: dict[str, Any], *, _index: int = state_index) -> None:
            if progress_callback is None:
                return
            total = max(1, int(update.get("total", 1) or 1))
            completed = max(0, min(total, int(update.get("completed", 0) or 0)))
            local = completed / total
            fraction = basis_share + direct_share * ((_index + local) / max(1, len(states)))
            progress_callback(
                {
                    "completed": int(round(10000 * fraction)),
                    "total": 10000,
                    "stage": f"Direct validation solve {_index + 1}/{len(states)} — {label}",
                }
            )

        direct_result = direct_adapter.get_field(
            points=points_m,
            field=field_name,
            progress_callback=direct_progress,
            progress_label=f"Animation validation — {label}",
        )
        direct = np.asarray(direct_result["values"], dtype=float).reshape(-1, 3)
        reconstructed = fixed + np.einsum("c,cpk->pk", currents, bases, optimize=True)
        difference = reconstructed - direct
        error_norm = np.linalg.norm(difference, axis=1) * scale
        direct_norm = np.linalg.norm(direct, axis=1) * scale
        magnitude_error = (
            np.abs(np.linalg.norm(reconstructed, axis=1) - np.linalg.norm(direct, axis=1))
            * scale
        )
        peak = float(np.max(direct_norm)) if direct_norm.size else 0.0
        relative_mask = direct_norm > max(peak * 1e-6, 1e-12)
        relative = (
            error_norm[relative_mask] / direct_norm[relative_mask]
            if np.any(relative_mask)
            else np.zeros(1, dtype=float)
        )
        results.append(
            {
                "label": label,
                "currents_a": currents.tolist(),
                "peak_field": peak,
                "max_vector_error": float(np.max(error_norm)),
                "rms_vector_error": float(np.sqrt(np.mean(error_norm * error_norm))),
                "max_magnitude_error": float(np.max(magnitude_error)),
                "rms_magnitude_error": float(np.sqrt(np.mean(magnitude_error * magnitude_error))),
                "max_relative_error_pct": float(np.max(relative) * 100.0),
                "rms_relative_error_pct": float(np.sqrt(np.mean(relative * relative)) * 100.0),
            }
        )

    peak_overall = max((float(item["peak_field"]) for item in results), default=0.0)
    max_abs = max((float(item["max_vector_error"]) for item in results), default=0.0)
    max_rel = max((float(item["max_relative_error_pct"]) for item in results), default=0.0)
    # This is deliberately a validation tolerance rather than solver accuracy claim:
    # either 0.01% relative agreement or an extremely small absolute discrepancy
    # is enough to establish that the animation basis reconstruction is faithful.
    abs_tolerance = max(1e-6, peak_overall * 1e-4)
    passed = bool(max_abs <= abs_tolerance or max_rel <= 0.01)
    if progress_callback is not None:
        progress_callback(
            {"completed": 10000, "total": 10000, "stage": "Animation field validation complete"}
        )
    return {
        "passed": passed,
        "field": field_name,
        "unit": unit,
        "point_count": int(len(points_m)),
        "resolution": samples,
        "centre_mm": centre.tolist(),
        "span_mm": span,
        "drive_coil_ids": drive_ids,
        "controlled_coil_ids": controlled_ids,
        "states": results,
        "max_vector_error": max_abs,
        "max_relative_error_pct": max_rel,
        "absolute_tolerance": abs_tolerance,
    }


def _scalar_values(
    vectors: np.ndarray,
    *,
    component: str,
    field: str,
) -> np.ndarray:
    component_name = str(component).strip().lower()
    scalar = field_component_scalars(vectors, component_name)
    if str(field).strip().upper() == "B":
        scalar = scalar * 1e6
    return np.asarray(scalar, dtype=float)


def _display_time_axis(time_s: np.ndarray) -> tuple[np.ndarray, str, float]:
    duration = abs(float(time_s[-1] - time_s[0]))
    if duration < 2.0:
        return time_s * 1000.0, "Time (ms)", 1000.0
    return time_s.copy(), "Time (s)", 1.0


def _format_stat(value: float, unit: str) -> str:
    return f"{float(value):.6g} {unit}"


def _animation_controls(
    frame_times_display: np.ndarray,
    *,
    restyle_frames: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Playback itself is controlled by PlotView's dedicated control bar beneath
    # the plot.  Keeping the controls out of Plotly's updatemenu prevents them
    # from covering axes and lets Workbench present real enabled/disabled Play
    # and Stop states.  The physical-time slider remains a Plotly control so it
    # stays synchronized with animation frames.
    buttons: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    label_stride = max(1, int(math.ceil(len(frame_times_display) / 12.0)))
    for index, value in enumerate(frame_times_display):
        show_label = index == 0 or index == len(frame_times_display) - 1 or index % label_stride == 0
        steps.append(
            {
                "label": f"{float(value):.5g}" if show_label else "",
                # Heatmaps are updated with Plotly.restyle by PlotView.  Using
                # slider method=skip prevents Qt WebEngine/Plotly from briefly
                # tearing down the heatmap during every animated redraw; the
                # sliderchange event still tells PlotView which frame to apply.
                "method": "skip" if restyle_frames else "animate",
                "args": [index] if restyle_frames else [
                    [f"waveform-frame-{index}"],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    },
                ],
            }
        )
    sliders = [
        {
            "active": 0,
            "x": 0.16,
            "len": 0.82,
            "xanchor": "left",
            "y": 0.285,
            "yanchor": "bottom",
            "currentvalue": {"prefix": "Physical time: "},
            "pad": {"t": 2, "b": 0},
            "steps": steps,
        }
    ]
    return buttons, sliders


def _observation_geometry(adapter: "StudioAdapter", observation: dict[str, Any] | None) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not observation:
        return np.empty((0, 3), dtype=float), None
    kind = str(observation.get("kind", "")).lower()
    object_id = str(observation.get("id", ""))
    if kind == "sensor":
        _x, points, rotations, _x_label = adapter._sensor_sample_geometry(object_id)
        index = int(observation.get("sample_index", 0))
        if not 0 <= index < len(points):
            raise StudioOperationError("The selected sensor sample no longer exists.")
        return np.asarray([points[index]], dtype=float), {
            "kind": "sensor",
            "rotation": rotations[index],
            "component": str(observation.get("component", "magnitude")).lower(),
        }
    if kind == "probe":
        samples = adapter.magnetometer_probe_sample_geometry(object_id)
        if not samples:
            raise StudioOperationError("The selected probe does not contain a sensing channel.")
        return np.asarray([sample["world_position_m"] for sample in samples], dtype=float), {
            "kind": "probe",
            "samples": samples,
            "component": str(observation.get("component", "combined")),
        }
    raise StudioOperationError("The selected waveform observation is not supported.")



def _observation_geometries(
    adapter: "StudioAdapter",
    observation: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return concatenated sample points and metadata for one or more trace sources.

    ``observation`` remains backward compatible with the original single mapping,
    while waveform playback may now pass a list so several sensors/probes can be
    plotted together without repeating the expensive field-basis calculation.
    """
    if not observation:
        return np.empty((0, 3), dtype=float), []
    raw_items = [observation] if isinstance(observation, dict) else list(observation)
    point_sets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    offset = 0
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise StudioOperationError("Waveform observation sources must be mappings.")
        points, meta = _observation_geometry(adapter, raw)
        if meta is None or not len(points):
            continue
        item = dict(meta)
        item["point_offset"] = offset
        item["point_count"] = int(len(points))
        item["source_label"] = str(raw.get("label") or raw.get("id") or "Observation")
        point_sets.append(np.asarray(points, dtype=float).reshape(-1, 3))
        metadata.append(item)
        offset += int(len(points))
    if not point_sets:
        return np.empty((0, 3), dtype=float), []
    return np.vstack(point_sets), metadata


def _observation_payloads(
    fixed_vectors: np.ndarray,
    basis_vectors: np.ndarray,
    timeline: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    """Build synchronized graph payloads for all selected observation sources."""
    if not metadata:
        return []
    combined = fixed_vectors[None, :, :] + np.einsum(
        "tc,cpk->tpk", timeline["analysis_currents"], basis_vectors, optimize=True
    )
    field_name = str(field).strip().upper()
    unit = "µT" if field_name == "B" else "A/m"
    times = np.asarray(timeline["analysis_times"], dtype=float)
    payloads: list[dict[str, Any]] = []
    for meta in metadata:
        start = int(meta.get("point_offset", 0))
        count = int(meta.get("point_count", 0))
        if count <= 0:
            continue
        values, signal_label = _observation_values(
            combined[:, start : start + count, :], meta, field=field_name
        )
        source_label = str(meta.get("source_label") or "Observation")
        label = f"{source_label} — {signal_label}"
        stats = time_statistics(times, values)
        payloads.append(
            {
                "time_s": times.tolist(),
                "values": np.asarray(values, dtype=float).tolist(),
                "label": label,
                "unit": unit,
                "axis": "field",
                "stats": {key: float(value) for key, value in stats.items()},
            }
        )
    return payloads


def _excitation_trace_payload(
    trace: WaveformTrace,
    timeline: dict[str, np.ndarray],
    *,
    slew_rate_per_s: float | None,
    initial_drive_value: float | None,
) -> dict[str, Any]:
    """Return the actual DAC/output excitation at the analysis timestamps."""
    times = np.asarray(timeline["analysis_times"], dtype=float)
    values = drive_at_times(
        trace,
        times,
        slew_rate_per_s=slew_rate_per_s,
        initial_value=initial_drive_value,
    )
    if trace.kind == "current":
        values = np.asarray(values, dtype=float) * 1000.0
        label = "Base excitation current"
        unit = "mA"
    else:
        values = np.asarray(values, dtype=float)
        label = "Base excitation voltage"
        unit = "V"
    stats = time_statistics(times, values)
    return {
        "time_s": times.tolist(),
        "values": values.tolist(),
        "label": label,
        "unit": unit,
        "axis": "excitation",
        "stats": {key: float(value) for key, value in stats.items()},
    }


def _sequencer_trace_payload(
    timeline: dict[str, np.ndarray],
    *,
    steps: list[dict[str, Any]],
    loop: bool,
    coil_labels: dict[str, str],
) -> dict[str, Any]:
    """Return the resolved sequencer state on the common analysis timeline."""
    times = np.asarray(timeline["analysis_times"], dtype=float)
    start_s = float(times[0]) if len(times) else 0.0
    indices = _sequence_step_indices(times, start_s=start_s, steps=steps, loop=loop)
    step_numbers = np.where(indices >= 0, indices + 1, 0).astype(int)
    active_coils: list[str] = []
    for index in indices:
        if index < 0:
            active_coils.append("None")
            continue
        ids = [str(value) for value in steps[int(index)].get("active_coil_ids", [])]
        labels = [str(coil_labels.get(object_id, object_id)) for object_id in ids]
        active_coils.append(", ".join(labels) if labels else "None")
    return {
        "time_s": times.tolist(),
        "values": step_numbers.astype(float).tolist(),
        "step_numbers": step_numbers.tolist(),
        "active_coils": active_coils,
        "label": "Sequencer",
        "unit": "",
        "axis": "sequencer",
    }


def _playback_sequence_payload(
    steps: list[dict[str, Any]] | None,
    *,
    loop: bool,
    coil_ids: list[str],
    coil_labels: dict[str, str],
) -> dict[str, Any] | None:
    """Return the compact sequencer contract used by the realtime player.

    Unlike the optional plotted sequencer trace, this contract describes the
    sequencer itself rather than its state over one waveform duration. The web
    player can consequently keep sequencer elapsed time advancing while the
    excitation frame index wraps independently.
    """
    if not steps:
        return None
    index_by_id = {str(object_id): index for index, object_id in enumerate(coil_ids)}
    payload_steps: list[dict[str, Any]] = []
    for step in steps:
        active_ids = [str(value) for value in step.get("active_coil_ids", [])]
        payload_steps.append(
            {
                "duration_s": float(step["duration_s"]),
                "active_coil_indices": [index_by_id[value] for value in active_ids],
                "active_coils": [str(coil_labels.get(value, value)) for value in active_ids],
            }
        )
    return {
        "loop": bool(loop),
        "cycle_s": float(sum(float(step["duration_s"]) for step in steps)),
        "steps": payload_steps,
    }


def _brain_playback_run_metadata(
    trace: WaveformTrace,
    *,
    frame_count: int,
    analysis_sample_count: int,
    summary_sample_count: int,
    atlas_sample_count: int,
    waveform_metadata: dict[str, Any] | None,
    sequence_steps: list[dict[str, Any]] | None,
    sequence_loop: bool,
) -> dict[str, Any]:
    """Describe the finite physical pass used by Brain exposure summaries."""

    duration = float(trace.time_s[-1] - trace.time_s[0])
    source = dict(waveform_metadata or {})
    mode = str(source.get("mode", "custom")).strip().lower() or "custom"
    frequency_hz = source.get("frequency_hz")
    cycle_count = source.get("cycle_count")
    try:
        frequency_hz = float(frequency_hz) if frequency_hz is not None else None
    except (TypeError, ValueError):
        frequency_hz = None
    try:
        cycle_count = float(cycle_count) if cycle_count is not None else None
    except (TypeError, ValueError):
        cycle_count = None
    if frequency_hz is not None and not math.isfinite(frequency_hz):
        frequency_hz = None
    if cycle_count is not None and not math.isfinite(cycle_count):
        cycle_count = None

    steps = list(sequence_steps or [])
    sequence_cycle_s = float(
        sum(max(0.0, float(step.get("duration_s", 0.0))) for step in steps)
    )
    sequence_equivalent_loops = 0.0
    sequence_full_loops = 0
    sequence_partial_loop = 0.0
    sequence_step_activations = 0
    if steps and sequence_cycle_s > 0.0 and duration > 0.0:
        tolerance = max(1e-12, sequence_cycle_s * 1e-10)
        if sequence_loop:
            sequence_equivalent_loops = duration / sequence_cycle_s
            sequence_full_loops = int(math.floor((duration + tolerance) / sequence_cycle_s))
            remainder = duration - sequence_full_loops * sequence_cycle_s
            if remainder < -tolerance:
                sequence_full_loops -= 1
                remainder += sequence_cycle_s
            if abs(remainder) <= tolerance:
                remainder = 0.0
            sequence_partial_loop = max(0.0, min(1.0, remainder / sequence_cycle_s))
            sequence_step_activations = sequence_full_loops * len(steps)
            elapsed = 0.0
            for step in steps:
                if elapsed < remainder - tolerance:
                    sequence_step_activations += 1
                elapsed += max(0.0, float(step.get("duration_s", 0.0)))
        else:
            covered = min(duration, sequence_cycle_s)
            sequence_equivalent_loops = covered / sequence_cycle_s
            sequence_full_loops = 1 if covered >= sequence_cycle_s - tolerance else 0
            sequence_partial_loop = (
                0.0 if sequence_full_loops else max(0.0, covered / sequence_cycle_s)
            )
            elapsed = 0.0
            for step in steps:
                if elapsed < covered - tolerance:
                    sequence_step_activations += 1
                elapsed += max(0.0, float(step.get("duration_s", 0.0)))

    return {
        "exposure_time_s": duration,
        "waveform_mode": mode,
        "waveform_frequency_hz": frequency_hz,
        "waveform_cycle_count": cycle_count,
        "waveform_pass_count": int(source.get("pass_count", 1) or 1),
        "sequencer_enabled": bool(steps),
        "sequencer_step_count": len(steps),
        "sequencer_loop_enabled": bool(steps and sequence_loop),
        "sequencer_cycle_s": sequence_cycle_s if steps else 0.0,
        "sequencer_equivalent_loops": sequence_equivalent_loops,
        "sequencer_full_loops": sequence_full_loops,
        "sequencer_partial_loop": sequence_partial_loop,
        "sequencer_step_activations": sequence_step_activations,
        "frame_count": int(frame_count),
        "analysis_sample_count": int(analysis_sample_count),
        "regional_frame_sample_count": int(summary_sample_count),
        "atlas_sample_count": int(atlas_sample_count),
    }


def _observation_values(vectors: np.ndarray, metadata: dict[str, Any], *, field: str) -> tuple[np.ndarray, str]:
    field_name = str(field).strip().upper()
    scale, unit = (1e6, "µT") if field_name == "B" else (1.0, "A/m")
    component = str(metadata.get("component", "magnitude")).lower()
    if metadata["kind"] == "sensor":
        rotation: "Rotation" = metadata["rotation"]
        local = np.asarray([rotation.inv().apply(vector) for vector in vectors[:, 0, :]], dtype=float)
        if component == "magnitude":
            values = np.linalg.norm(local, axis=1)
            label = f"|{field_name}|"
        elif component in {"x", "y", "z", "abs_x", "abs_y", "abs_z"}:
            axis = component[-1]
            values = local[:, "xyz".index(axis)]
            if component.startswith("abs_"):
                values = np.abs(values)
                label = f"|{field_name}{axis}|"
            else:
                label = f"{field_name}{axis}"
        else:
            raise StudioOperationError(
                "Sensor observation must use magnitude, absolute X/Y/Z, or signed X/Y/Z."
            )
        return values * scale, f"{label} ({unit})"

    samples = metadata["samples"]
    projected = np.empty((vectors.shape[0], len(samples)), dtype=float)
    for channel_index, sample in enumerate(samples):
        axis = np.asarray(sample["world_sensitive_axis"], dtype=float)
        projected[:, channel_index] = vectors[:, channel_index, :] @ axis

    # Map physical probe channels onto X/Y/Z by their documented names/outputs.
    # Built-in probes are triaxial but their channel sensing positions may differ,
    # so each component continues to use the real channel position and axis.
    axis_channels: dict[str, int] = {}
    for channel_index, sample in enumerate(samples):
        name = str(sample.get("name") or "").strip().lower()
        output = str(sample.get("output") or "").strip().lower()
        for axis_name in "xyz":
            if name == axis_name or output in {axis_name, f"b{axis_name}", f"h{axis_name}"}:
                axis_channels.setdefault(axis_name, channel_index)
    for axis_index, axis_name in enumerate("xyz"):
        if axis_name not in axis_channels and axis_index < len(samples):
            axis_channels[axis_name] = axis_index

    if component in {"magnitude", "combined"}:
        values = np.linalg.norm(projected, axis=1)
        label = f"|{field_name}|"
    elif component in {"x", "y", "z", "abs_x", "abs_y", "abs_z"}:
        axis_name = component[-1]
        if axis_name not in axis_channels:
            raise StudioOperationError(f"The selected probe does not provide a {axis_name.upper()} channel.")
        channel_index = axis_channels[axis_name]
        values = projected[:, channel_index]
        if component.startswith("abs_"):
            values = np.abs(values)
            label = f"|{field_name}{axis_name}|"
        else:
            label = f"{field_name}{axis_name}"
    else:
        # Preserve compatibility with older saved playback settings that stored
        # a specific numeric probe-channel index.
        try:
            channel_index = int(component)
        except ValueError as error:
            raise StudioOperationError("The selected probe channel is not available.") from error
        if not 0 <= channel_index < len(samples):
            raise StudioOperationError("The selected probe channel is not available.")
        values = projected[:, channel_index]
        label = str(
            samples[channel_index].get("output")
            or samples[channel_index].get("name")
            or f"Channel {channel_index + 1}"
        )
    return values * scale, f"{label} ({unit})"


def _prepare_timeline(
    trace: WaveformTrace,
    *,
    frame_count: int,
    analysis_samples: int,
    resistance_ohm: float | None,
    inductance_h: float | None,
    initial_current_a: float,
    initial_drive_value: float | None,
    slew_rate_per_s: float | None,
) -> dict[str, np.ndarray]:
    frames = int(frame_count)
    samples = int(analysis_samples)
    if not 2 <= frames <= MAX_ANIMATION_FRAMES:
        raise StudioOperationError(
            f"Waveform playback must use between 2 and {MAX_ANIMATION_FRAMES:,} animation frames."
        )
    if not 2 <= samples <= MAX_ANALYSIS_SAMPLES:
        raise StudioOperationError(
            f"Waveform analysis must use between 2 and {MAX_ANALYSIS_SAMPLES:,} time samples."
        )
    start, stop = float(trace.time_s[0]), float(trace.time_s[-1])
    frame_times = np.linspace(start, stop, frames)
    analysis_times = np.linspace(start, stop, samples)
    frame_currents = current_at_times(
        trace,
        frame_times,
        resistance_ohm=resistance_ohm,
        inductance_h=inductance_h,
        initial_current_a=initial_current_a,
        initial_drive_value=initial_drive_value,
        slew_rate_per_s=slew_rate_per_s,
    )
    analysis_currents = current_at_times(
        trace,
        analysis_times,
        resistance_ohm=resistance_ohm,
        inductance_h=inductance_h,
        initial_current_a=initial_current_a,
        initial_drive_value=initial_drive_value,
        slew_rate_per_s=slew_rate_per_s,
    )
    return {
        "frame_times": frame_times,
        "analysis_times": analysis_times,
        "frame_currents": frame_currents,
        "analysis_currents": analysis_currents,
    }


def _colour_planes(raw_frames: np.ndarray, *, logarithmic: bool) -> tuple[np.ndarray, float, float]:
    values = np.asarray(raw_frames, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise StudioOperationError("Waveform playback did not produce finite field values.")
    if logarithmic:
        positive = finite[finite > 0.0]
        floor = max(float(np.min(positive)) * 1e-6, 1e-300) if positive.size else 1e-300
        colour = np.log10(np.maximum(values, floor))
    else:
        colour = values.copy()
    finite_colour = colour[np.isfinite(colour)]
    lower = float(np.min(finite_colour))
    upper = float(np.max(finite_colour))
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-15):
        pad = max(abs(lower) * 0.01, 1e-12)
        lower -= pad
        upper += pad
    return colour, lower, upper


def build_2d_animation(
    adapter: "StudioAdapter",
    *,
    map_settings: dict[str, Any],
    coil_id: str,
    trace: WaveformTrace,
    frame_count: int,
    analysis_samples: int,
    playback_fps: float,
    resistance_ohm: float | None = None,
    inductance_h: float | None = None,
    initial_current_a: float = 0.0,
    initial_drive_value: float | None = None,
    slew_rate_per_s: float | None = 0.0,
    observation: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Build an animated planar field map with optional synchronized trace."""
    plane = str(map_settings["plane"]).lower()
    offset_mm = float(map_settings["offset_mm"])
    span_mm = float(map_settings["span_mm"])
    resolution = int(map_settings["resolution"])
    field = str(map_settings["field"]).upper()
    component = str(map_settings["component"]).lower()
    logarithmic = bool(map_settings.get("logarithmic", False))
    if logarithmic and not field_component_is_magnitude(component):
        raise StudioOperationError("Log scale is available only for field magnitude or absolute axis-component magnitude.")
    if frame_count * resolution * resolution > MAX_ANIMATION_SCALAR_VALUES:
        raise StudioOperationError(
            "This 2D playback would create too many animated field samples. Reduce animation frames or map resolution."
        )

    half = span_mm / 2.0
    first = np.linspace(-half, half, resolution)
    second = np.linspace(-half, half, resolution)
    first_grid, second_grid = np.meshgrid(first, second)
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}.get(plane)
    if axes is None:
        raise StudioOperationError("The 2D waveform map plane must be XY, XZ, or YZ.")
    map_points_mm = np.zeros((resolution * resolution, 3), dtype=float)
    map_points_mm[:, axes[0]] = first_grid.reshape(-1)
    map_points_mm[:, axes[1]] = second_grid.reshape(-1)
    map_points_mm[:, axes[2]] = offset_mm

    observation_points, observation_meta = _observation_geometry(adapter, observation)
    map_count = len(map_points_mm)
    all_points = np.vstack([map_points_mm / 1000.0, observation_points])
    fixed, basis = field_basis_for_drive(
        adapter, coil_id=coil_id, points_m=all_points, field=field, progress_callback=progress_callback
    )

    timeline = _prepare_timeline(
        trace,
        frame_count=frame_count,
        analysis_samples=analysis_samples,
        resistance_ohm=resistance_ohm,
        inductance_h=inductance_h,
        initial_current_a=initial_current_a,
        initial_drive_value=initial_drive_value,
        slew_rate_per_s=slew_rate_per_s,
    )
    frame_vectors = fixed[:map_count][None, :, :] + timeline["frame_currents"][:, None, None] * basis[:map_count][None, :, :]
    raw_frames = _scalar_values(frame_vectors, component=component, field=field).reshape(frame_count, resolution, resolution)
    colour_frames, automatic_min, automatic_max = _colour_planes(raw_frames, logarithmic=logarithmic)

    colour_min, colour_max = _resolve_gpu_colour_limits(
        automatic_min,
        automatic_max,
        manual_min=map_settings.get("colour_minimum"),
        manual_max=map_settings.get("colour_maximum"),
        component=component,
        logarithmic=logarithmic,
    )

    component_label = field_component_label(field, component)
    unit = "µT" if field == "B" else "A/m"
    colour_title = f"log₁₀ {component_label} ({unit})" if logarithmic else f"{component_label} ({unit})"
    axis_labels = {
        "xy": ("X (LR, mm)", "Y (AP, mm)"),
        "xz": ("X (LR, mm)", "Z (SI, mm)"),
        "yz": ("Y (AP, mm)", "Z (SI, mm)"),
    }[plane]

    heatmap = {
        "type": "heatmap",
        "x": first.tolist(),
        "y": second.tolist(),
        "z": colour_frames[0].tolist(),
        "customdata": raw_frames[0].tolist(),
        "colorscale": copy.deepcopy(
            FIELD_SIGNED_COLOURSCALE if field_component_is_signed(component) else FIELD_HEAT_COLOURSCALE
        ),
        "zmin": colour_min,
        "zmax": colour_max,
        **({"zmid": 0.0} if field_component_is_signed(component) else {}),
        "autocolorscale": False,
        "colorbar": {"title": {"text": colour_title}},
        "hovertemplate": (
            f"{axis_labels[0]}: %{{x:.5g}}<br>{axis_labels[1]}: %{{y:.5g}}<br>"
            f"{component_label}: %{{customdata:.6g}} {unit}<extra></extra>"
        ),
        "name": component_label,
        "uid": "waveform-field-map",
    }
    data: list[dict[str, Any]] = [heatmap]
    if bool(map_settings.get("show_object_outlines", True)):
        data.extend(
            adapter._scene_cross_section_traces_2d(
                plane=plane,
                offset_mm=offset_mm,
                span_mm=span_mm,
                object_ids=map_settings.get("outline_object_ids"),
            )
        )

    observation_values = None
    observation_label = None
    cursor_index = None
    analysis_time_display, time_axis_label, time_scale = _display_time_axis(timeline["analysis_times"])
    frame_time_display = timeline["frame_times"] * time_scale
    annotations: list[dict[str, Any]] = []
    if observation_meta is not None:
        obs_fixed = fixed[map_count:]
        obs_basis = basis[map_count:]
        obs_vectors = obs_fixed[None, :, :] + timeline["analysis_currents"][:, None, None] * obs_basis[None, :, :]
        observation_values, observation_label = _observation_values(obs_vectors, observation_meta, field=field)
        stats = time_statistics(timeline["analysis_times"], observation_values)
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": analysis_time_display.tolist(),
                "y": observation_values.tolist(),
                "xaxis": "x2",
                "yaxis": "y2",
                "name": observation_label,
                "showlegend": False,
                "hovertemplate": f"%{{x:.6g}}<br>{observation_label}: %{{y:.6g}}<extra></extra>",
                "uid": "waveform-observation",
            }
        )
        y_min = float(np.min(observation_values))
        y_max = float(np.max(observation_values))
        if math.isclose(y_min, y_max, rel_tol=1e-12, abs_tol=1e-15):
            pad = max(abs(y_min) * 0.05, 1.0)
            y_min -= pad
            y_max += pad
        cursor_index = len(data)
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [float(frame_time_display[0]), float(frame_time_display[0])],
                "y": [y_min, y_max],
                "xaxis": "x2",
                "yaxis": "y2",
                "line": {"color": "rgba(15,23,42,0.65)", "width": 1.5},
                "hoverinfo": "skip",
                "showlegend": False,
                "uid": "waveform-time-cursor",
            }
        )
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 0.245,
                "xanchor": "right",
                "yanchor": "bottom",
                "showarrow": False,
                "align": "right",
                "text": (
                    f"Mean/DC: {_format_stat(stats['mean'], unit)} &nbsp; "
                    f"RMS: {_format_stat(stats['rms'], unit)} &nbsp; "
                    f"Pk-pk: {_format_stat(stats['pk_pk'], unit)}"
                ),
                "font": {"size": 12},
            }
        )

    buttons, sliders = _animation_controls(frame_time_display, restyle_frames=True)
    layout: dict[str, Any] = {
        "template": "plotly_white",
        "title": {"text": f"Waveform playback — {component_label}", "x": 0.5},
        "margin": {"l": 66, "r": 88, "t": 52, "b": 42},
        "showlegend": False,
        "dragmode": "pan",
        "updatemenus": buttons,
        "sliders": sliders,
        "annotations": annotations,
        "xaxis": {"title": {"text": axis_labels[0]}},
        "yaxis": {"title": {"text": axis_labels[1]}, "scaleanchor": "x", "scaleratio": 1},
        "uirevision": "field-workbench-waveform-2d",
        "meta": {
            "field_workbench_waveform": {
                "playback_fps": float(playback_fps),
                "real_time_multiplier": playback_real_time_multiplier(
                    float(trace.time_s[-1] - trace.time_s[0]), frame_count, playback_fps
                ),
                "frame_count": int(frame_count),
                "update_mode": "restyle",
            }
        },
    }
    if observation_meta is not None:
        layout["yaxis"]["domain"] = [0.36, 1.0]
        layout["xaxis2"] = {"domain": [0.0, 1.0], "anchor": "y2", "title": {"text": time_axis_label}}
        layout["yaxis2"] = {"domain": [0.0, 0.20], "anchor": "x2", "title": {"text": observation_label}}
        heatmap["colorbar"].update({"len": 0.58, "y": 0.68})
    else:
        layout["yaxis"]["domain"] = [0.12, 1.0]
        for slider in layout["sliders"]:
            slider["y"] = 0.075

    frames: list[dict[str, Any]] = []
    for index in range(frame_count):
        frame_data: list[dict[str, Any]] = [
            {"z": colour_frames[index].tolist(), "customdata": raw_frames[index].tolist()}
        ]
        trace_indices = [0]
        if cursor_index is not None:
            frame_data.append({"x": [float(frame_time_display[index]), float(frame_time_display[index])]})
            trace_indices.append(cursor_index)
        frames.append(
            {
                "name": f"waveform-frame-{index}",
                "data": frame_data,
                "traces": trace_indices,
            }
        )
    return {"data": data, "layout": layout, "frames": frames}



def _array_b64(values: Any, dtype: Any) -> str:
    """Encode a contiguous numeric array for compact browser-side typed arrays."""
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


def _emit_render_task_progress(
    callback: Any | None,
    *,
    task_id: str,
    stage: str,
    completed: int,
    total: int,
    unit: str = "render steps",
    detail: str | None = None,
) -> None:
    """Emit one high-level render-plan update around measured solver work."""

    if callback is None:
        return
    maximum = max(1, int(total))
    current = min(maximum, max(0, int(completed)))
    update: dict[str, Any] = {
        "render_task": str(task_id),
        "render_task_complete": current >= maximum,
        "stage": str(stage),
        "completed": current,
        "total": maximum,
        "unit": str(unit),
    }
    if detail:
        update["detail"] = str(detail)
    callback(update)


def _render_task_progress_relay(
    callback: Any | None,
    task_id: str,
    *,
    default_unit: str = "field samples",
):
    """Tag an existing determinate solver callback with its render-plan leaf."""

    if callback is None:
        return None

    def relay(update: dict[str, Any]) -> None:
        tagged = dict(update)
        tagged["render_task"] = str(task_id)
        tagged.setdefault("unit", str(default_unit))
        try:
            completed = int(tagged.get("completed", 0) or 0)
            total = max(1, int(tagged.get("total", 1) or 1))
        except (TypeError, ValueError):
            completed, total = 0, 1
        stage = str(tagged.get("stage") or "").strip().lower()
        if completed >= total and "complete" in stage:
            tagged["render_task_complete"] = True
        callback(tagged)

    return relay


def _css_rgba(value: Any, *, opacity: float = 1.0) -> list[float]:
    """Best-effort Plotly/CSS colour conversion for the lightweight WebGL scene."""
    alpha = min(1.0, max(0.0, float(opacity)))
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    text = str(value or "#64748b").strip().lower()
    named = {
        "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000",
        "blue": "#0000ff", "gray": "#808080", "grey": "#808080", "orange": "#ffa500",
        "yellow": "#ffff00", "purple": "#800080", "cyan": "#00ffff", "magenta": "#ff00ff",
        "transparent": "#00000000",
    }
    text = named.get(text, text)
    if text.startswith("#"):
        raw = text[1:]
        if len(raw) in {3, 4}:
            raw = "".join(ch * 2 for ch in raw)
        if len(raw) in {6, 8}:
            try:
                rgb = [int(raw[index:index + 2], 16) / 255.0 for index in (0, 2, 4)]
                if len(raw) == 8:
                    alpha *= int(raw[6:8], 16) / 255.0
                return [*rgb, alpha]
            except ValueError:
                pass
    match = re.fullmatch(r"rgba?\(([^)]+)\)", text)
    if match:
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) >= 3:
            try:
                rgb = [min(255.0, max(0.0, float(parts[index]))) / 255.0 for index in range(3)]
                if len(parts) >= 4:
                    alpha *= min(1.0, max(0.0, float(parts[3])))
                return [*rgb, alpha]
            except ValueError:
                pass
    return [0.392, 0.455, 0.545, alpha]


def _gpu_scene_payload(figure: dict[str, Any]) -> dict[str, Any]:
    """Reduce Plotly context geometry to the primitives needed by WaveformGLView.

    The animated view intentionally supports only passive triangle meshes, line
    paths, and marker points.  Unsupported Plotly decorations are skipped rather
    than pulling Plotly's renderer back into the realtime path.
    """
    meshes: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    bounds: list[np.ndarray] = []
    for trace in figure.get("data", []):
        if not isinstance(trace, dict) or trace.get("visible", True) in {False, "legendonly"}:
            continue
        trace_type = str(trace.get("type", "")).lower()
        opacity = float(trace.get("opacity", 1.0) or 0.0)
        if opacity <= 0.01:
            continue
        if trace_type == "mesh3d":
            try:
                vertices = np.column_stack([
                    np.asarray(trace.get("x", []), dtype=float),
                    np.asarray(trace.get("y", []), dtype=float),
                    np.asarray(trace.get("z", []), dtype=float),
                ]).astype(np.float32, copy=False)
                i = np.asarray(trace.get("i", []), dtype=np.uint32)
                j = np.asarray(trace.get("j", []), dtype=np.uint32)
                k = np.asarray(trace.get("k", []), dtype=np.uint32)
            except (TypeError, ValueError):
                continue
            if len(vertices) < 3 or not (len(i) == len(j) == len(k)) or not len(i):
                continue
            indices = np.column_stack([i, j, k]).reshape(-1).astype(np.uint32, copy=False)
            colour = trace.get("color") or trace.get("vertexcolor") or trace.get("facecolor") or "#94a3b8"
            meshes.append({
                "positions": _array_b64(vertices, np.float32),
                "indices": _array_b64(indices, np.uint32),
                "vertex_count": int(len(vertices)),
                "index_count": int(len(indices)),
                # Keep Plotly trace opacity explicit in the GPU payload instead of
                # baking it into the colour. Chromium/WebGL then applies the scene
                # object's alpha independently from its RGB/material colour.
                "colour": _css_rgba(colour),
                "opacity": opacity,
            })
            bounds.append(vertices.astype(float, copy=False))
            continue
        if trace_type != "scatter3d":
            continue
        mode = str(trace.get("mode", ""))
        try:
            xs, ys, zs = trace.get("x", []), trace.get("y", []), trace.get("z", [])
            count = min(len(xs), len(ys), len(zs))
        except TypeError:
            continue
        if "lines" in mode:
            segments: list[list[float]] = []
            previous: np.ndarray | None = None
            valid_points: list[np.ndarray] = []
            for index in range(count):
                values = (xs[index], ys[index], zs[index])
                if any(value is None for value in values):
                    previous = None
                    continue
                try:
                    current = np.asarray(values, dtype=float)
                except (TypeError, ValueError):
                    previous = None
                    continue
                if not np.isfinite(current).all():
                    previous = None
                    continue
                valid_points.append(current)
                if previous is not None:
                    segments.extend([previous.tolist(), current.tolist()])
                previous = current
            if segments:
                array = np.asarray(segments, dtype=np.float32)
                line = trace.get("line") if isinstance(trace.get("line"), dict) else {}
                lines.append({
                    "positions": _array_b64(array, np.float32),
                    "count": int(len(array)),
                    "colour": _css_rgba(line.get("color", "#475569")),
                    "opacity": opacity,
                    "width": max(1.0, float(line.get("width", 1.0) or 1.0)),
                })
                bounds.append(np.asarray(valid_points, dtype=float))
        if "markers" in mode:
            marker_points: list[list[float]] = []
            for index in range(count):
                values = (xs[index], ys[index], zs[index])
                if any(value is None for value in values):
                    continue
                try:
                    current = np.asarray(values, dtype=float)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(current).all():
                    marker_points.append(current.tolist())
            if marker_points:
                array = np.asarray(marker_points, dtype=np.float32)
                marker = trace.get("marker") if isinstance(trace.get("marker"), dict) else {}
                points.append({
                    "positions": _array_b64(array, np.float32),
                    "count": int(len(array)),
                    "colour": _css_rgba(marker.get("color", "#334155")),
                    "opacity": opacity,
                    "size": max(2.0, float(marker.get("size", 4.0) or 4.0)),
                })
                bounds.append(array.astype(float, copy=False))
    finite_bounds = [values.reshape(-1, 3) for values in bounds if values.size]
    if finite_bounds:
        joined = np.vstack(finite_bounds)
        minimum = np.min(joined, axis=0).tolist()
        maximum = np.max(joined, axis=0).tolist()
    else:
        minimum = maximum = None
    return {"meshes": meshes, "lines": lines, "points": points, "bounds_min": minimum, "bounds_max": maximum}


def _gpu_colour_limits(
    fixed_vectors: np.ndarray,
    basis_vectors: np.ndarray,
    currents: np.ndarray,
    *,
    component: str,
    field: str,
    logarithmic: bool,
) -> tuple[float, float]:
    """Find stable playback colour limits without materializing every frame."""
    fixed = np.asarray(fixed_vectors, dtype=float).reshape(-1, 3)
    basis = np.asarray(basis_vectors, dtype=float).reshape(-1, 3)
    current_values = np.asarray(currents, dtype=float).reshape(-1)
    if not len(current_values):
        raise StudioOperationError("Waveform playback does not contain current samples.")
    current_min = float(np.min(current_values))
    current_max = float(np.max(current_values))
    component_name = str(component).lower()
    if component_name in {"x", "y", "z", "abs_x", "abs_y", "abs_z"}:
        axis_name = component_name[-1]
        axis = "xyz".index(axis_name)
        endpoint_a = fixed[:, axis] + current_min * basis[:, axis]
        endpoint_b = fixed[:, axis] + current_max * basis[:, axis]
        lower = np.minimum(endpoint_a, endpoint_b)
        upper = np.maximum(endpoint_a, endpoint_b)
        if component_name.startswith("abs_"):
            minimum_abs = np.where(
                (lower <= 0.0) & (upper >= 0.0),
                0.0,
                np.minimum(np.abs(lower), np.abs(upper)),
            )
            maximum_abs = np.maximum(np.abs(lower), np.abs(upper))
            values = np.concatenate([minimum_abs, maximum_abs])
        else:
            values = np.concatenate([lower, upper])
    elif component_name == "magnitude":
        first = fixed + current_min * basis
        second = fixed + current_max * basis
        maxima = np.maximum(np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1))
        basis_norm2 = np.sum(basis * basis, axis=1)
        optimal = np.full(len(fixed), current_min, dtype=float)
        mask = basis_norm2 > 1e-30
        optimal[mask] = -np.sum(fixed[mask] * basis[mask], axis=1) / basis_norm2[mask]
        optimal = np.clip(optimal, current_min, current_max)
        minima = np.linalg.norm(fixed + optimal[:, None] * basis, axis=1)
        values = np.concatenate([minima, maxima])
    else:
        raise StudioOperationError("The waveform map component is not supported.")
    if str(field).upper() == "B":
        values = values * 1e6
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise StudioOperationError("Waveform playback did not produce finite field values.")
    if field_component_is_signed(component_name):
        extent = float(np.max(np.abs(finite)))
        if not math.isfinite(extent) or extent <= 0.0:
            extent = 1e-12
        return -extent, extent
    if logarithmic:
        positive = finite[finite > 0.0]
        floor = max(float(np.min(positive)) * 1e-6, 1e-300) if positive.size else 1e-300
        finite = np.log10(np.maximum(finite, floor))
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-15):
        pad = max(abs(lower) * 0.01, 1e-12)
        lower -= pad
        upper += pad
        if field_component_is_magnitude(component_name) and lower < 0.0:
            lower = 0.0
    return lower, upper

def _gpu_colour_limits_multi(
    fixed_vectors: np.ndarray,
    basis_vectors: np.ndarray,
    currents: np.ndarray,
    *,
    component: str,
    field: str,
    logarithmic: bool,
) -> tuple[float, float]:
    """Find stable limits for several independently gated drive-coil bases."""
    fixed = np.asarray(fixed_vectors, dtype=float).reshape(-1, 3)
    bases = np.asarray(basis_vectors, dtype=float)
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise StudioOperationError("Waveform coil-basis dimensions are invalid.")
    current_values = np.asarray(currents, dtype=float)
    if current_values.ndim != 2 or current_values.shape[1] != bases.shape[0] or not len(current_values):
        raise StudioOperationError("Waveform playback does not contain valid multi-coil current samples.")
    component_name = str(component).lower()
    scale = 1e6 if str(field).upper() == "B" else 1.0

    if component_name in {"x", "y", "z", "abs_x", "abs_y", "abs_z"}:
        axis = "xyz".index(component_name[-1])
        lower = fixed[:, axis].copy()
        upper = fixed[:, axis].copy()
        for coil_index in range(bases.shape[0]):
            basis = bases[coil_index, :, axis]
            current_min = float(np.min(current_values[:, coil_index]))
            current_max = float(np.max(current_values[:, coil_index]))
            a = basis * current_min
            b = basis * current_max
            lower += np.minimum(a, b)
            upper += np.maximum(a, b)
        if component_name.startswith("abs_"):
            minimum_abs = np.where(
                (lower <= 0.0) & (upper >= 0.0),
                0.0,
                np.minimum(np.abs(lower), np.abs(upper)),
            )
            maximum_abs = np.maximum(np.abs(lower), np.abs(upper))
            finite = np.concatenate([minimum_abs, maximum_abs]) * scale
        else:
            finite = np.concatenate([lower, upper]) * scale
    elif component_name == "magnitude":
        # Evaluate representative actual timeline states for the lower end; use
        # a conservative triangle bound for the upper end so a short sequencer
        # step cannot clip the colour scale.
        count = len(current_values)
        indices = set(np.linspace(0, count - 1, min(24, count), dtype=int).tolist())
        for coil_index in range(current_values.shape[1]):
            indices.add(int(np.argmin(current_values[:, coil_index])))
            indices.add(int(np.argmax(current_values[:, coil_index])))
        minimum = math.inf
        actual_maximum = 0.0
        for index in sorted(indices):
            vectors = fixed + np.einsum("c,cpk->pk", current_values[index], bases, optimize=True)
            magnitudes = np.linalg.norm(vectors, axis=1)
            if magnitudes.size:
                minimum = min(minimum, float(np.min(magnitudes)))
                actual_maximum = max(actual_maximum, float(np.max(magnitudes)))
        max_abs = np.max(np.abs(current_values), axis=0)
        conservative = np.linalg.norm(fixed, axis=1)
        for coil_index in range(bases.shape[0]):
            conservative += float(max_abs[coil_index]) * np.linalg.norm(bases[coil_index], axis=1)
        maximum = max(actual_maximum, float(np.max(conservative)))
        finite = np.asarray([minimum if math.isfinite(minimum) else 0.0, maximum], dtype=float) * scale
    else:
        raise StudioOperationError("The waveform map component is not supported.")

    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise StudioOperationError("Waveform playback did not produce finite field values.")
    if field_component_is_signed(component_name):
        extent = float(np.max(np.abs(finite)))
        if not math.isfinite(extent) or extent <= 0.0:
            extent = 1e-12
        return -extent, extent
    if logarithmic:
        positive = finite[finite > 0.0]
        floor = max(float(np.min(positive)) * 1e-6, 1e-300) if positive.size else 1e-300
        finite = np.log10(np.maximum(finite, floor))
    lower_value, upper_value = float(np.min(finite)), float(np.max(finite))
    if math.isclose(lower_value, upper_value, rel_tol=1e-12, abs_tol=1e-15):
        pad = max(abs(lower_value) * 0.01, 1e-12)
        lower_value -= pad
        upper_value += pad
        if field_component_is_magnitude(component_name) and lower_value < 0.0:
            lower_value = 0.0
    return lower_value, upper_value

def _resolve_gpu_colour_limits(
    automatic_min: float,
    automatic_max: float,
    *,
    manual_min: Any,
    manual_max: Any,
    component: str,
    logarithmic: bool,
) -> tuple[float, float]:
    """Resolve optional user limits while preserving zero-centred signed scales."""
    component_name = str(component).strip().lower()
    if field_component_is_signed(component_name):
        supplied = [
            abs(float(value)) for value in (manual_min, manual_max) if value is not None
        ]
        if supplied:
            if not all(math.isfinite(value) for value in supplied):
                raise StudioOperationError("Signed waveform colour limits must be finite.")
            extent = max(supplied)
        else:
            extent = max(abs(float(automatic_min)), abs(float(automatic_max)))
        if not math.isfinite(extent):
            raise StudioOperationError("Signed waveform colour limits must be finite.")
        if extent <= 0.0:
            extent = max(abs(float(automatic_min)), abs(float(automatic_max)), 1e-12)
        return -extent, extent
    lower = manual_min
    upper = manual_max
    if logarithmic:
        if lower is not None:
            if float(lower) <= 0.0:
                raise StudioOperationError("Logarithmic waveform colour limits must be positive.")
            lower = math.log10(float(lower))
        if upper is not None:
            if float(upper) <= 0.0:
                raise StudioOperationError("Logarithmic waveform colour limits must be positive.")
            upper = math.log10(float(upper))
    colour_min = float(automatic_min) if lower is None else float(lower)
    colour_max = float(automatic_max) if upper is None else float(upper)
    if colour_min >= colour_max:
        raise StudioOperationError("The waveform heatmap colour limits do not form a valid range.")
    return colour_min, colour_max


def _gpu_2d_outline_scene(
    adapter: "StudioAdapter",
    *,
    plane: str,
    offset_mm: float,
    span_mm: float,
    object_ids: Any,
) -> dict[str, Any]:
    """Convert the existing 2D cross-section overlays into lightweight GPU lines."""
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[plane]
    lines: list[dict[str, Any]] = []
    bounds: list[np.ndarray] = []
    for trace in adapter._scene_cross_section_traces_2d(
        plane=plane,
        offset_mm=offset_mm,
        span_mm=span_mm,
        object_ids=object_ids,
    ):
        if not isinstance(trace, dict) or str(trace.get("type", "scatter")).lower() != "scatter":
            continue
        if "lines" not in str(trace.get("mode", "lines")):
            continue
        xs, ys = trace.get("x", []), trace.get("y", [])
        try:
            count = min(len(xs), len(ys))
        except TypeError:
            continue
        segments: list[list[float]] = []
        valid: list[np.ndarray] = []
        previous: np.ndarray | None = None
        for index in range(count):
            if xs[index] is None or ys[index] is None:
                previous = None
                continue
            try:
                point = np.zeros(3, dtype=float)
                point[axes[0]] = float(xs[index])
                point[axes[1]] = float(ys[index])
                point[axes[2]] = float(offset_mm)
            except (TypeError, ValueError):
                previous = None
                continue
            if not np.isfinite(point).all():
                previous = None
                continue
            valid.append(point)
            if previous is not None:
                segments.extend([previous.tolist(), point.tolist()])
            previous = point
        if not segments:
            continue
        line = trace.get("line") if isinstance(trace.get("line"), dict) else {}
        array = np.asarray(segments, dtype=np.float32)
        lines.append(
            {
                "positions": _array_b64(array, np.float32),
                "count": int(len(array)),
                "colour": _css_rgba(line.get("color", "#334155"), opacity=float(trace.get("opacity", 1.0) or 1.0)),
                "width": max(1.0, float(line.get("width", 1.5) or 1.5)),
            }
        )
        bounds.append(np.asarray(valid, dtype=float))
    if bounds:
        joined = np.vstack(bounds)
        minimum = np.min(joined, axis=0).tolist()
        maximum = np.max(joined, axis=0).tolist()
    else:
        minimum = maximum = None
    return {"meshes": [], "lines": lines, "points": [], "bounds_min": minimum, "bounds_max": maximum}


def _resolve_gpu_drive_ids(coil_id: str | None, coil_ids: Any | None) -> list[str]:
    values = list(coil_ids or ([] if coil_id is None else [coil_id]))
    result = list(dict.fromkeys(str(value) for value in values if str(value)))
    if not result:
        raise StudioOperationError("Choose at least one coil for waveform playback.")
    if len(result) > MAX_GPU_DRIVE_COILS:
        raise StudioOperationError(
            f"GPU waveform playback currently supports up to {MAX_GPU_DRIVE_COILS} driven coils at once."
        )
    return result


def build_2d_gpu_playback(
    adapter: "StudioAdapter",
    *,
    map_settings: dict[str, Any],
    coil_id: str | None = None,
    coil_ids: Any | None = None,
    controlled_coil_ids: Any | None = None,
    trace: WaveformTrace,
    frame_count: int,
    analysis_samples: int,
    playback_fps: float,
    resistance_ohm: float | None = None,
    inductance_h: float | None = None,
    initial_current_a: float = 0.0,
    initial_drive_value: float | None = None,
    slew_rate_per_s: float | None = 0.0,
    electrical_by_coil: dict[str, dict[str, float | None]] | None = None,
    sequence_steps: list[dict[str, Any]] | None = None,
    sequence_loop: bool = False,
    waveform_metadata: dict[str, Any] | None = None,
    observation: dict[str, Any] | list[dict[str, Any]] | None = None,
    include_excitation_trace: bool = False,
    include_sequencer_trace: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Build the compact WebGL2 package for realtime planar waveform playback."""
    drive_ids = _resolve_gpu_drive_ids(coil_id, coil_ids)
    normalized_sequence = (
        normalize_coil_sequence(sequence_steps, available_coil_ids=drive_ids)
        if sequence_steps is not None
        else None
    )
    plane = str(map_settings["plane"]).lower()
    offset_mm = float(map_settings["offset_mm"])
    span = float(map_settings["span_mm"])
    samples = int(map_settings["resolution"])
    field = str(map_settings["field"]).upper()
    component = str(map_settings["component"]).lower()
    logarithmic = bool(map_settings.get("logarithmic", False))
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}.get(plane)
    if axes is None:
        raise StudioOperationError("The 2D waveform map plane must be XY, XZ, or YZ.")
    if logarithmic and not field_component_is_magnitude(component):
        raise StudioOperationError("Log scale is available only for field magnitude or absolute axis-component magnitude.")
    if samples < 2:
        raise StudioOperationError("2D waveform playback requires at least a 2×2 map grid.")

    half = span / 2.0
    first = np.linspace(-half, half, samples)
    second = np.linspace(-half, half, samples)
    first_grid, second_grid = np.meshgrid(first, second)
    map_points_mm = np.zeros((samples * samples, 3), dtype=float)
    map_points_mm[:, axes[0]] = first_grid.reshape(-1)
    map_points_mm[:, axes[1]] = second_grid.reshape(-1)
    map_points_mm[:, axes[2]] = offset_mm

    observation_points, observation_meta = _observation_geometries(adapter, observation)
    map_count = len(map_points_mm)
    all_points = np.vstack([map_points_mm / 1000.0, observation_points])
    fixed, bases = field_bases_for_drives(
        adapter,
        coil_ids=drive_ids,
        zero_coil_ids=controlled_coil_ids,
        points_m=all_points,
        field=field,
        progress_callback=progress_callback,
    )
    if electrical_by_coil is None:
        electrical_by_coil = {
            object_id: {
                "resistance_ohm": resistance_ohm,
                "inductance_h": inductance_h,
                "initial_current_a": initial_current_a,
            }
            for object_id in drive_ids
        }
    timeline = _prepare_multi_coil_timeline(
        trace,
        coil_ids=drive_ids,
        frame_count=frame_count,
        analysis_samples=analysis_samples,
        electrical_by_coil=electrical_by_coil,
        initial_drive_value=initial_drive_value,
        slew_rate_per_s=slew_rate_per_s,
        sequence_steps=normalized_sequence,
        sequence_loop=bool(sequence_loop),
    )
    automatic_min, automatic_max = _gpu_colour_limits_multi(
        fixed[:map_count], bases[:, :map_count, :], timeline["frame_currents"],
        component=component, field=field, logarithmic=logarithmic,
    )
    colour_min, colour_max = _resolve_gpu_colour_limits(
        automatic_min,
        automatic_max,
        manual_min=map_settings.get("colour_minimum"),
        manual_max=map_settings.get("colour_maximum"),
        component=component,
        logarithmic=logarithmic,
    )

    component_label = field_component_label(field, component)
    unit = "µT" if field == "B" else "A/m"
    axis_labels = {
        "xy": ["X (LR, mm)", "Y (AP, mm)"],
        "xz": ["X (LR, mm)", "Z (SI, mm)"],
        "yz": ["Y (AP, mm)", "Z (SI, mm)"],
    }[plane]
    centre = np.zeros(3, dtype=float)
    centre[axes[2]] = offset_mm
    grid_spec = field_plane_grid_spec(
        plane,
        offset_mm,
        span,
        enabled=bool(map_settings.get("show_grid", True)),
    )
    scene_payload = (
        _gpu_2d_outline_scene(
            adapter,
            plane=plane,
            offset_mm=offset_mm,
            span_mm=span,
            object_ids=map_settings.get("outline_object_ids"),
        )
        if bool(map_settings.get("show_object_outlines", True))
        else {"meshes": [], "lines": [], "points": [], "bounds_min": None, "bounds_max": None}
    )

    observation_payloads = _observation_payloads(
        fixed[map_count:], bases[:, map_count:, :], timeline, observation_meta, field=field
    )
    excitation_payload = (
        _excitation_trace_payload(
            trace, timeline, slew_rate_per_s=slew_rate_per_s, initial_drive_value=initial_drive_value
        )
        if include_excitation_trace
        else None
    )
    coil_labels = (
        {
            str(choice["id"]): str(choice.get("label") or choice["id"])
            for choice in drive_coil_catalog(adapter)
        }
        if normalized_sequence
        else {}
    )
    playback_sequence = _playback_sequence_payload(
        normalized_sequence,
        loop=bool(sequence_loop),
        coil_ids=drive_ids,
        coil_labels=coil_labels,
    )
    sequencer_payload = None
    if include_sequencer_trace and normalized_sequence:
        sequencer_payload = _sequencer_trace_payload(
            timeline, steps=normalized_sequence, loop=bool(sequence_loop), coil_labels=coil_labels
        )

    return {
        "renderer": "field-workbench-webgl2-waveform-v4",
        "view_mode": "2d",
        "title": f"Waveform playback — {component_label}",
        "component": component,
        "component_label": component_label,
        "field": field,
        "unit": unit,
        "logarithmic": logarithmic,
        "colour_min": float(colour_min),
        "colour_max": float(colour_max),
        "colour_scale": copy.deepcopy(
            FIELD_SIGNED_COLOURSCALE if field_component_is_signed(component) else FIELD_HEAT_COLOURSCALE
        ),
        "signed_component": field_component_is_signed(component),
        "centre_mm": centre.tolist(),
        "span_mm": span,
        "plane": plane,
        "show_grid": bool(grid_spec["enabled"]),
        "grid": grid_spec,
        "show_contours": bool(map_settings.get("show_contours", False)),
        "contour_count": max(1, min(50, int(map_settings.get("contour_count", 10)))),
        "contour_labels": bool(map_settings.get("contour_labels", False)),
        "axis_labels": axis_labels,
        "slice_axis": "xyz"[axes[2]],
        "slice_positions_mm": [offset_mm],
        "slice_count": 1,
        "resolution": samples,
        "positions": _array_b64(np.asarray(map_points_mm, dtype=np.float32), np.float32),
        "fixed_vectors": _array_b64(np.asarray(fixed[:map_count], dtype=np.float32), np.float32),
        "basis_vectors": _array_b64(np.asarray(bases[0, :map_count], dtype=np.float32), np.float32),
        "basis_vectors_b64": [
            _array_b64(np.asarray(values[:map_count], dtype=np.float32), np.float32) for values in bases
        ],
        "drive_coil_ids": drive_ids,
        "drive_coil_count": len(drive_ids),
        "frame_count": int(len(timeline["frame_times"])),
        "frame_times_b64": _array_b64(np.asarray(timeline["frame_times"], dtype=np.float64), np.float64),
        "frame_currents_b64": _array_b64(np.asarray(timeline["frame_currents"], dtype=np.float32), np.float32),
        "free_frame_currents_b64": _array_b64(
            np.asarray(timeline["free_frame_currents"], dtype=np.float32), np.float32
        ) if playback_sequence is not None else "",
        "playback_sequence": playback_sequence,
        "playback_fps": float(playback_fps),
        "real_time_multiplier": playback_real_time_multiplier(
            float(trace.time_s[-1] - trace.time_s[0]), frame_count, playback_fps
        ),
        "scene": scene_payload,
        "observation": observation_payloads[0] if observation_payloads else None,
        "observations": observation_payloads,
        "scroll_observation_traces": bool(
            map_settings.get("scroll_observation_traces", False)
        ),
        "excitation": excitation_payload,
        "sequencer": sequencer_payload,
    }



def build_3d_gpu_static(
    adapter: "StudioAdapter",
    *,
    map_settings: dict[str, Any],
    active_coil_ids: Any | None = None,
    initial_camera: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Build a fixed-current 3D map for the shared WebGL renderer.

    Static Full Volume and Slices deliberately use the same GPU geometry,
    colour mapping, opacity transfer, texture sampling, and scene compositing
    as realtime playback.  The field itself is solved once at the scene's
    configured coil currents and stored as the fixed vector field; no playback
    basis or timeline is required.
    """
    centre = np.asarray(map_settings["centre_mm"], dtype=float).reshape(3)
    span = float(map_settings["span_mm"])
    axis_name = str(map_settings["slice_axis"]).lower()
    layers = int(map_settings["slice_count"])
    samples = int(map_settings["resolution"])
    field = str(map_settings["field"]).upper()
    component = str(map_settings["component"]).lower()
    logarithmic = bool(map_settings.get("logarithmic", False))
    render_mode = str(
        map_settings.get("view_type", map_settings.get("waveform_render_mode", "slices"))
    ).strip().lower()
    display_mode = str(map_settings.get("display_mode", "slices")).strip().lower()
    if axis_name not in {"x", "y", "z"}:
        raise StudioOperationError("The 3D map slice axis must be X, Y, or Z.")
    if render_mode not in {"slices", "volume", "points"}:
        raise StudioOperationError(
            "3D rendering must use Slices, Full Volume, or atlas Points mode."
        )
    if display_mode not in {"slices", "flux", "both"}:
        raise StudioOperationError("The 3D field-map display mode must show the field, fluxlines, or both.")
    if logarithmic and not field_component_is_magnitude(component):
        raise StudioOperationError("Log scale is available only for field magnitude or absolute axis-component magnitude.")
    if layers < 1 or samples < 2:
        raise StudioOperationError("3D maps require at least one slice and a 2×2 map grid.")

    opacity_enabled = bool(map_settings.get("intensity_opacity_enabled", True))
    opacity_lower = float(map_settings.get("volume_opacity_lower", 0.15))
    opacity_upper = float(map_settings.get("volume_opacity_upper", 0.40))
    opacity_lower_level = float(map_settings.get("volume_opacity_lower_level", 0.0))
    opacity_upper_level = float(map_settings.get("volume_opacity_upper_level", 1.0))
    if not 0.0 <= opacity_lower <= 1.0 or not 0.0 <= opacity_upper <= 1.0:
        raise StudioOperationError("Volume opacity limits must stay between 0 and 100%.")
    if not 0.0 <= opacity_lower_level <= 1.0 or not 0.0 <= opacity_upper_level <= 1.0:
        raise StudioOperationError("Volume opacity intensity levels must stay between 0 and 100%.")
    if opacity_enabled and opacity_lower_level > opacity_upper_level:
        raise StudioOperationError("Lower opacity intensity level cannot exceed upper opacity intensity level.")

    _emit_render_task_progress(
        progress_callback,
        task_id="grid",
        stage="Preparing 3D sample grid…",
        completed=0,
        total=1,
    )
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    plane_axes = [index for index in range(3) if index != axis_index]
    half = span / 2.0
    coordinates = [
        np.linspace(centre[index] - half, centre[index] + half, samples)
        for index in range(3)
    ]
    if render_mode == "points":
        try:
            map_points_mm = np.asarray(
                map_settings["points_mm"], dtype=float
            ).reshape(-1, 3)
        except (KeyError, TypeError, ValueError) as error:
            raise StudioOperationError(
                "Brain static analysis requires finite atlas sample points."
            ) from error
        if not len(map_points_mm) or not np.isfinite(map_points_mm).all():
            raise StudioOperationError(
                "Brain static analysis requires finite atlas sample points."
            )
        fixed_coordinates = np.empty(0, dtype=float)
        progress_label = "LPBA40 brain regions"
    elif render_mode == "volume":
        z_grid, y_grid, x_grid = np.meshgrid(
            coordinates[2], coordinates[1], coordinates[0], indexing="ij"
        )
        map_points_mm = np.column_stack(
            [x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)]
        )
        fixed_coordinates = np.asarray(coordinates[axis_index], dtype=float)
        progress_label = "3D volume grid"
    else:
        fixed_coordinates = (
            np.asarray([centre[axis_index]], dtype=float)
            if layers == 1
            else np.linspace(centre[axis_index] - half, centre[axis_index] + half, layers)
        )
        horizontal, vertical = np.meshgrid(
            coordinates[plane_axes[0]], coordinates[plane_axes[1]]
        )
        positions: list[np.ndarray] = []
        for fixed_coordinate in fixed_coordinates:
            xyz = [np.empty_like(horizontal) for _ in range(3)]
            xyz[plane_axes[0]] = horizontal
            xyz[plane_axes[1]] = vertical
            xyz[axis_index] = np.full_like(horizontal, fixed_coordinate)
            positions.append(np.column_stack([value.reshape(-1) for value in xyz]))
        map_points_mm = np.concatenate(positions, axis=0)
        progress_label = "3D slice grid"

    analysis_points_mm = np.empty((0, 3), dtype=float)
    if bool(map_settings.get("_return_field_vectors", False)):
        try:
            analysis_points_mm = np.asarray(
                map_settings.get("analysis_points_mm", map_settings.get("points_mm", [])),
                dtype=float,
            ).reshape(-1, 3)
        except (TypeError, ValueError) as error:
            raise StudioOperationError(
                "Brain static analysis requires finite analysis sample points."
            ) from error
        if not len(analysis_points_mm) or not np.isfinite(analysis_points_mm).all():
            raise StudioOperationError(
                "Brain static analysis requires finite analysis sample points."
            )
    overlay_points_mm = np.empty((0, 3), dtype=float)
    overlay_point_labels = np.empty(0, dtype=np.int32)
    raw_overlay_points = map_settings.get(
        "overlay_points_mm",
        map_settings.get("trace_points_mm", []),
    )
    if raw_overlay_points is not None:
        try:
            overlay_points_mm = np.asarray(raw_overlay_points, dtype=float).reshape(-1, 3)
        except (TypeError, ValueError) as error:
            raise StudioOperationError(
                "Brain static analysis requires finite display sample points."
            ) from error
        if len(overlay_points_mm):
            if not np.isfinite(overlay_points_mm).all():
                raise StudioOperationError(
                    "Brain static analysis requires finite display sample points."
                )
            overlay_point_labels = np.asarray(
                map_settings.get(
                    "overlay_point_label_ids",
                    map_settings.get("trace_point_label_ids", np.zeros(len(overlay_points_mm))),
                ),
                dtype=np.int32,
            ).reshape(-1)
            if overlay_point_labels.shape != (len(overlay_points_mm),):
                raise StudioOperationError(
                    "Brain static analysis display sample labels are invalid."
                )
    analysis_count = int(len(analysis_points_mm))
    overlay_count = int(len(overlay_points_mm))
    all_points_mm = np.vstack(
        [block for block in (map_points_mm, analysis_points_mm, overlay_points_mm) if len(block)]
    )

    _emit_render_task_progress(
        progress_callback,
        task_id="grid",
        stage="3D sample grid ready",
        completed=1,
        total=1,
        detail=(
            f"Prepared {len(map_points_mm):,} map sample positions"
            + (
                f", {analysis_count:,} LPBA40 analysis samples"
                if analysis_count
                else ""
            )
            + (
                f", and {overlay_count:,} LPBA40 display samples."
                if overlay_count
                else "."
            )
        ),
    )
    field_progress = _render_task_progress_relay(
        progress_callback,
        "field",
        default_unit="field samples",
    )
    result = adapter.get_field(
        points=all_points_mm / 1000.0,
        field=field,
        progress_callback=field_progress,
        progress_label=progress_label,
        active_coil_ids=active_coil_ids,
    )
    all_fixed = np.asarray(result["values"], dtype=float).reshape(-1, 3)
    map_count = len(map_points_mm)
    fixed = all_fixed[:map_count]
    analysis_fixed = all_fixed[map_count : map_count + analysis_count]
    overlay_fixed = all_fixed[map_count + analysis_count : map_count + analysis_count + overlay_count]
    scalar = field_component_scalars(fixed, component).reshape(-1)
    scale = 1e6 if field == "B" else 1.0
    values = scalar * scale
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise StudioOperationError("The 3D field calculation did not return finite values.")
    if field_component_is_signed(component):
        extent = float(np.max(np.abs(finite)))
        if not math.isfinite(extent) or extent <= 0.0:
            extent = 1e-12
        automatic_min, automatic_max = -extent, extent
    else:
        colour_values = finite
        if logarithmic:
            positive = colour_values[colour_values > 0.0]
            floor = max(float(np.min(positive)) * 1e-6, 1e-300) if positive.size else 1e-300
            colour_values = np.log10(np.maximum(colour_values, floor))
        automatic_min = float(np.min(colour_values))
        automatic_max = float(np.max(colour_values))
        if math.isclose(automatic_min, automatic_max, rel_tol=1e-12, abs_tol=1e-15):
            pad = max(abs(automatic_min) * 0.01, 1e-12)
            automatic_min -= pad
            automatic_max += pad
            if field_component_is_magnitude(component) and automatic_min < 0.0:
                automatic_min = 0.0
    colour_min, colour_max = _resolve_gpu_colour_limits(
        automatic_min,
        automatic_max,
        manual_min=map_settings.get("colour_minimum"),
        manual_max=map_settings.get("colour_maximum"),
        component=component,
        logarithmic=logarithmic,
    )

    component_label = field_component_label(field, component)
    unit = "µT" if field == "B" else "A/m"
    grid_spec = field_volume_grid_spec(
        centre.tolist(), span, enabled=bool(map_settings.get("show_grid", True))
    )
    scene_figure = (
        adapter._field_volume_scene_figure(
            map_settings.get("scene_object_ids"),
            scene_object_opacities=map_settings.get("scene_object_opacities"),
        )
        if bool(map_settings.get("include_scene", True))
        else {"data": [], "layout": {}}
    )
    if display_mode in {"flux", "both"}:
        flux_progress = _render_task_progress_relay(
            progress_callback,
            "flux",
            default_unit="field samples",
        )
        flux_traces = adapter._field_fluxline_traces_3d(
            centre_mm=centre.tolist(),
            span_mm=span,
            slice_axis=axis_name,
            field=field,
            density=int(map_settings.get("fluxline_density", 5)),
            active_coil_ids=active_coil_ids,
            progress_callback=flux_progress,
        )
        scene_figure = copy.deepcopy(scene_figure)
        scene_figure.setdefault("data", []).extend(flux_traces)
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Converting scene geometry for WebGL…",
        completed=0,
        total=3,
    )
    scene_payload = _gpu_scene_payload(scene_figure)
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Scene geometry converted",
        completed=1,
        total=3,
    )
    if grid_spec["enabled"]:
        box_vertices = np.asarray(
            [point for segment in grid_spec["box_segments_mm"] for point in segment],
            dtype=np.float32,
        )
        scene_payload.setdefault("lines", []).append(
            {
                "positions": _array_b64(box_vertices, np.float32),
                "count": int(len(box_vertices)),
                "colour": _css_rgba("rgba(71,85,105,0.55)"),
                "width": 2.0,
            }
        )
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Encoding field arrays for WebGL…",
        completed=2,
        total=3,
    )

    zero_basis = np.zeros_like(fixed, dtype=np.float32)
    frame_times = np.asarray([0.0], dtype=np.float64)
    frame_currents = np.asarray([[0.0]], dtype=np.float32)
    title_mode = (
        str(map_settings.get("atlas_name", "Atlas"))
        if render_mode == "points"
        else ("Full Volume" if render_mode == "volume" else "Slices")
    )
    if display_mode == "flux":
        title_mode = "Fluxlines"
    elif display_mode == "both":
        title_mode = f"{title_mode} + fluxlines"
    payload = {
        "renderer": "field-workbench-webgl2-static-v1",
        "static_view": True,
        "view_mode": "3d",
        "title": (
            f"Brain static analysis — {map_settings.get('atlas_name', 'LPBA40')}"
            if map_settings.get("atlas_name")
            else f"3D field map — {component_label} — {title_mode}"
        ),
        "component": component,
        "component_label": component_label,
        "field": field,
        "unit": unit,
        "logarithmic": logarithmic,
        "colour_min": float(colour_min),
        "colour_max": float(colour_max),
        "colour_scale": copy.deepcopy(
            FIELD_SIGNED_COLOURSCALE if field_component_is_signed(component) else FIELD_HEAT_COLOURSCALE
        ),
        "signed_component": field_component_is_signed(component),
        "centre_mm": centre.tolist(),
        "span_mm": span,
        "show_grid": bool(grid_spec["enabled"]),
        "axis_labels": [axis["label"] for axis in grid_spec["axes"]],
        "grid": grid_spec,
        "slice_axis": axis_name,
        "slice_positions_mm": fixed_coordinates.tolist() if render_mode == "slices" else [],
        "slice_count": layers if render_mode == "slices" else (1 if render_mode == "points" else samples),
        "resolution": samples,
        "render_mode": render_mode,
        "point_count": int(len(map_points_mm)) if render_mode == "points" else 0,
        "point_size": float(map_settings.get("point_size", 4.0)),
        "overlay_point_count": int(overlay_count if render_mode != "points" else 0),
        "overlay_point_size": float(map_settings.get("point_size", 4.0)),
        "overlay_positions": _array_b64(np.asarray(overlay_points_mm, dtype=np.float32), np.float32) if (overlay_count and render_mode != "points") else "",
        "overlay_fixed_vectors": _array_b64(np.asarray(overlay_fixed, dtype=np.float32), np.float32) if (overlay_count and render_mode != "points") else "",
        "overlay_basis_vectors": _array_b64(np.zeros_like(np.asarray(overlay_fixed, dtype=np.float32)), np.float32) if (overlay_count and render_mode != "points") else "",
        "overlay_basis_vectors_b64": ([_array_b64(np.zeros_like(np.asarray(overlay_fixed, dtype=np.float32)), np.float32)] if (overlay_count and render_mode != "points") else []),
        "overlay_point_label_ids_b64": _array_b64(np.asarray(overlay_point_labels, dtype=np.int32), np.int32) if (overlay_count and render_mode != "points") else "",
        "show_field": display_mode != "flux",
        "intensity_opacity_enabled": opacity_enabled,
        "volume_opacity_lower": opacity_lower,
        "volume_opacity_upper": opacity_upper,
        "volume_opacity_lower_level": opacity_lower_level,
        "volume_opacity_upper_level": opacity_upper_level,
        "volume_surface_count": max(12, min(40, samples)),
        "volume_ray_steps": max(48, min(128, int(round(samples * 1.5)))),
        "volume_box_min_mm": (centre - half).tolist(),
        "volume_box_max_mm": (centre + half).tolist(),
        "positions": _array_b64(np.asarray(map_points_mm, dtype=np.float32), np.float32) if render_mode in {"slices", "points"} else "",
        "fixed_vectors": _array_b64(np.asarray(fixed, dtype=np.float32), np.float32) if render_mode in {"slices", "points"} else "",
        "basis_vectors": _array_b64(zero_basis, np.float32) if render_mode in {"slices", "points"} else "",
        "basis_vectors_b64": [_array_b64(zero_basis, np.float32)] if render_mode in {"slices", "points"} else [],
        "drive_coil_ids": ["__static__"],
        "drive_coil_count": 1,
        "volume_fixed_vectors": _array_b64(np.asarray(fixed, dtype=np.float32), np.float32) if render_mode == "volume" else "",
        "volume_basis_vectors": _array_b64(zero_basis, np.float32) if render_mode == "volume" else "",
        "volume_basis_vectors_b64": [_array_b64(zero_basis, np.float32)] if render_mode == "volume" else [],
        "frame_count": 1,
        "frame_times_b64": _array_b64(frame_times, np.float64),
        "frame_currents_b64": _array_b64(frame_currents, np.float32),
        "free_frame_currents_b64": _array_b64(frame_currents, np.float32),
        "playback_sequence": None,
        "playback_fps": 60.0,
        "real_time_multiplier": 1.0,
        "scene": scene_payload,
        "observation": None,
        "observations": [],
        "excitation": None,
        "sequencer": None,
    }
    if initial_camera is not None:
        payload["initial_camera"] = dict(initial_camera)
    if bool(map_settings.get("_return_field_vectors", False)):
        # Private process-to-GUI payload used by Brain View to update the
        # authoritative LPBA40 statistics without performing a second solve.
        # The isolated render process removes this array before constructing
        # the WebGL viewer payload.
        payload["_field_vectors_t"] = np.asarray(analysis_fixed, dtype=float)
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="WebGL render package ready",
        completed=3,
        total=3,
    )
    return payload


def build_3d_gpu_playback(
    adapter: "StudioAdapter",
    *,
    map_settings: dict[str, Any],
    coil_id: str | None = None,
    coil_ids: Any | None = None,
    controlled_coil_ids: Any | None = None,
    trace: WaveformTrace,
    frame_count: int,
    analysis_samples: int,
    playback_fps: float,
    resistance_ohm: float | None = None,
    inductance_h: float | None = None,
    initial_current_a: float = 0.0,
    initial_drive_value: float | None = None,
    slew_rate_per_s: float | None = 0.0,
    electrical_by_coil: dict[str, dict[str, float | None]] | None = None,
    sequence_steps: list[dict[str, Any]] | None = None,
    sequence_loop: bool = False,
    waveform_metadata: dict[str, Any] | None = None,
    observation: dict[str, Any] | list[dict[str, Any]] | None = None,
    include_excitation_trace: bool = False,
    include_sequencer_trace: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Build the compact data package consumed by the realtime WebGL2 viewer.

    Unlike :func:`build_3d_animation`, no field frame tensor or Plotly frame list
    is created. Fixed and per-ampere vectors are uploaded once and combined by
    the GPU shader using one current value per driven coil and displayed instant.
    """
    drive_ids = _resolve_gpu_drive_ids(coil_id, coil_ids)
    normalized_sequence = (
        normalize_coil_sequence(sequence_steps, available_coil_ids=drive_ids)
        if sequence_steps is not None
        else None
    )
    centre = np.asarray(map_settings["centre_mm"], dtype=float).reshape(3)
    span = float(map_settings["span_mm"])
    axis_name = str(map_settings["slice_axis"]).lower()
    layers = int(map_settings["slice_count"])
    samples = int(map_settings["resolution"])
    field = str(map_settings["field"]).upper()
    component = str(map_settings["component"]).lower()
    logarithmic = bool(map_settings.get("logarithmic", False))
    if axis_name not in {"x", "y", "z"}:
        raise StudioOperationError("The 3D waveform slice axis must be X, Y, or Z.")
    if logarithmic and not field_component_is_magnitude(component):
        raise StudioOperationError("Log scale is available only for field magnitude or absolute axis-component magnitude.")
    if layers < 1 or samples < 2:
        raise StudioOperationError("3D waveform playback requires valid rendering dimensions.")

    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    plane_axes = [index for index in range(3) if index != axis_index]
    half = span / 2.0
    coordinates = [np.linspace(centre[index] - half, centre[index] + half, samples) for index in range(3)]
    render_mode = str(
        map_settings.get("view_type", map_settings.get("waveform_render_mode", "slices"))
    ).strip().lower()
    if render_mode not in {"slices", "volume", "points"}:
        raise StudioOperationError(
            "3D waveform rendering must use Slices, Volume, or atlas Points mode."
        )
    opacity_enabled = bool(map_settings.get("intensity_opacity_enabled", True))
    opacity_lower = float(map_settings.get("volume_opacity_lower", 0.15))
    opacity_upper = float(map_settings.get("volume_opacity_upper", 0.40))
    opacity_lower_level = float(map_settings.get("volume_opacity_lower_level", 0.0))
    opacity_upper_level = float(map_settings.get("volume_opacity_upper_level", 1.0))
    if not 0.0 <= opacity_lower <= 1.0 or not 0.0 <= opacity_upper <= 1.0:
        raise StudioOperationError("Volume opacity limits must stay between 0 and 100%.")
    if not 0.0 <= opacity_lower_level <= 1.0 or not 0.0 <= opacity_upper_level <= 1.0:
        raise StudioOperationError("Volume opacity intensity levels must stay between 0 and 100%.")
    if opacity_enabled and opacity_lower_level > opacity_upper_level:
        raise StudioOperationError("Lower opacity intensity level cannot exceed upper opacity intensity level.")

    _emit_render_task_progress(
        progress_callback,
        task_id="grid",
        stage="Preparing waveform sample grid…",
        completed=0,
        total=1,
    )
    if render_mode == "points":
        try:
            map_points_mm = np.asarray(
                map_settings["points_mm"], dtype=float
            ).reshape(-1, 3)
        except (KeyError, TypeError, ValueError) as error:
            raise StudioOperationError(
                "Brain waveform playback requires finite atlas sample points."
            ) from error
        if not len(map_points_mm) or not np.isfinite(map_points_mm).all():
            raise StudioOperationError(
                "Brain waveform playback requires finite atlas sample points."
            )
        fixed_coordinates = np.empty(0, dtype=float)
    elif render_mode == "volume":
        # Texture storage is X-fastest, then Y, then Z.  Keep the point ordering
        # identical so texel (x,y,z) receives the corresponding fixed/basis vector.
        z_grid, y_grid, x_grid = np.meshgrid(
            coordinates[2], coordinates[1], coordinates[0], indexing="ij"
        )
        map_points_mm = np.column_stack(
            [x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)]
        )
        fixed_coordinates = np.asarray(coordinates[axis_index], dtype=float)
    else:
        fixed_coordinates = (
            np.asarray([centre[axis_index]], dtype=float)
            if layers == 1
            else np.linspace(centre[axis_index] - half, centre[axis_index] + half, layers)
        )
        horizontal, vertical = np.meshgrid(coordinates[plane_axes[0]], coordinates[plane_axes[1]])
        positions: list[np.ndarray] = []
        for fixed_coordinate in fixed_coordinates:
            xyz = [np.empty_like(horizontal) for _ in range(3)]
            xyz[plane_axes[0]] = horizontal
            xyz[plane_axes[1]] = vertical
            xyz[axis_index] = np.full_like(horizontal, fixed_coordinate)
            positions.append(np.column_stack([value.reshape(-1) for value in xyz]))
        map_points_mm = np.concatenate(positions, axis=0)
    observation_points, observation_meta = _observation_geometries(adapter, observation)
    trace_points_mm = np.empty((0, 3), dtype=float)
    trace_point_weights = np.empty(0, dtype=float)
    trace_point_labels = np.empty(0, dtype=np.int32)
    raw_trace_points = map_settings.get(
        "trace_points_mm",
        map_settings.get("points_mm", [] if render_mode != "points" else map_points_mm),
    )
    if raw_trace_points is not None:
        try:
            trace_points_mm = np.asarray(raw_trace_points, dtype=float).reshape(-1, 3)
        except (TypeError, ValueError) as error:
            raise StudioOperationError(
                "Brain waveform playback requires finite atlas trace sample points."
            ) from error
        if len(trace_points_mm):
            if not np.isfinite(trace_points_mm).all():
                raise StudioOperationError(
                    "Brain waveform playback requires finite atlas trace sample points."
                )
            trace_point_weights = np.asarray(
                map_settings.get("trace_point_weights_mm3", map_settings.get("point_weights_mm3", np.ones(len(trace_points_mm)))),
                dtype=float,
            ).reshape(-1)
            trace_point_labels = np.asarray(
                map_settings.get("trace_point_label_ids", map_settings.get("point_label_ids", np.zeros(len(trace_points_mm)))),
                dtype=np.int32,
            ).reshape(-1)
            if (
                trace_point_weights.shape != (len(trace_points_mm),)
                or trace_point_labels.shape != (len(trace_points_mm),)
                or not np.isfinite(trace_point_weights).all()
                or np.any(trace_point_weights <= 0.0)
            ):
                raise StudioOperationError(
                    "Brain waveform atlas labels or volume weights are invalid."
                )
    overlay_points_mm = np.empty((0, 3), dtype=float)
    overlay_point_labels = np.empty(0, dtype=np.int32)
    raw_overlay_points = map_settings.get("overlay_points_mm", [])
    if raw_overlay_points is not None:
        try:
            overlay_points_mm = np.asarray(raw_overlay_points, dtype=float).reshape(-1, 3)
        except (TypeError, ValueError) as error:
            raise StudioOperationError(
                "Brain waveform playback requires finite atlas display sample points."
            ) from error
        if len(overlay_points_mm):
            if not np.isfinite(overlay_points_mm).all():
                raise StudioOperationError(
                    "Brain waveform playback requires finite atlas display sample points."
                )
            overlay_point_labels = np.asarray(
                map_settings.get("overlay_point_label_ids", np.zeros(len(overlay_points_mm))),
                dtype=np.int32,
            ).reshape(-1)
            if overlay_point_labels.shape != (len(overlay_points_mm),):
                raise StudioOperationError(
                    "Brain waveform display sample labels are invalid."
                )
    map_count = len(map_points_mm)
    observation_count = len(observation_points)
    overlay_count = len(overlay_points_mm)
    trace_count = len(trace_points_mm)
    all_points = np.vstack(
        [
            map_points_mm / 1000.0,
            observation_points,
            overlay_points_mm / 1000.0,
            trace_points_mm / 1000.0,
        ]
    )
    _emit_render_task_progress(
        progress_callback,
        task_id="grid",
        stage="Waveform sample grid ready",
        completed=1,
        total=1,
        detail=(
            f"Prepared {map_count:,} "
            f"{'atlas' if render_mode == 'points' else 'map'} positions"
            f", {observation_count:,} observation positions"
            + (f", {overlay_count:,} LPBA40 display samples" if overlay_count else "")
            + (f", and {trace_count:,} LPBA40 analysis samples." if trace_count else ".")
        ),
    )
    field_progress = _render_task_progress_relay(
        progress_callback,
        "field",
        default_unit="basis work",
    )
    fixed, bases = field_bases_for_drives(
        adapter,
        coil_ids=drive_ids,
        zero_coil_ids=controlled_coil_ids,
        points_m=all_points,
        field=field,
        progress_callback=field_progress,
    )
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Preparing playback timeline and colour range…",
        completed=0,
        total=4,
    )
    if electrical_by_coil is None:
        electrical_by_coil = {
            object_id: {
                "resistance_ohm": resistance_ohm,
                "inductance_h": inductance_h,
                "initial_current_a": initial_current_a,
            }
            for object_id in drive_ids
        }
    timeline = _prepare_multi_coil_timeline(
        trace,
        coil_ids=drive_ids,
        frame_count=frame_count,
        analysis_samples=analysis_samples,
        electrical_by_coil=electrical_by_coil,
        initial_drive_value=initial_drive_value,
        slew_rate_per_s=slew_rate_per_s,
        sequence_steps=normalized_sequence,
        sequence_loop=bool(sequence_loop),
    )
    automatic_min, automatic_max = _gpu_colour_limits_multi(
        fixed[:map_count], bases[:, :map_count, :], timeline["frame_currents"],
        component=component, field=field, logarithmic=logarithmic,
    )
    colour_min, colour_max = _resolve_gpu_colour_limits(
        automatic_min,
        automatic_max,
        manual_min=map_settings.get("colour_minimum"),
        manual_max=map_settings.get("colour_maximum"),
        component=component,
        logarithmic=logarithmic,
    )
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Playback timeline and colour range ready",
        completed=1,
        total=4,
    )

    component_label = field_component_label(field, component)
    unit = "µT" if field == "B" else "A/m"
    grid_spec = field_volume_grid_spec(
        centre.tolist(),
        span,
        enabled=bool(map_settings.get("show_grid", True)),
    )
    scene_figure = (
        adapter._field_volume_scene_figure(
            map_settings.get("scene_object_ids"),
            scene_object_opacities=map_settings.get("scene_object_opacities"),
        )
        if bool(map_settings.get("include_scene", True))
        else {"data": [], "layout": {}}
    )
    scene_payload = _gpu_scene_payload(scene_figure)
    # The shared grid contract supplies the exact same map-volume box used by
    # the static Plotly render. WebGL adds the matching grid planes, ticks, and
    # labels from the remainder of this contract at display time.
    if grid_spec["enabled"]:
        box_vertices = np.asarray(
            [
                point
                for segment in grid_spec["box_segments_mm"]
                for point in segment
            ],
            dtype=np.float32,
        )
        scene_payload.setdefault("lines", []).append(
            {
                "positions": _array_b64(box_vertices, np.float32),
                "count": int(len(box_vertices)),
                "colour": _css_rgba("rgba(71,85,105,0.55)"),
                "width": 2.0,
            }
        )
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Scene geometry converted for WebGL",
        completed=2,
        total=4,
    )

    map_fixed = fixed[:map_count]
    map_bases = bases[:, :map_count, :]
    observation_start = map_count
    observation_end = observation_start + observation_count
    overlay_end = observation_end + overlay_count
    observation_fixed = fixed[observation_start:observation_end]
    observation_bases = bases[:, observation_start:observation_end, :]
    overlay_fixed = fixed[observation_end:overlay_end]
    overlay_bases = bases[:, observation_end:overlay_end, :]
    trace_fixed = fixed[overlay_end:]
    trace_bases = bases[:, overlay_end:, :]

    observation_payloads = _observation_payloads(
        observation_fixed, observation_bases, timeline, observation_meta, field=field
    )
    raw_trace_regions = map_settings.get("trace_regions")
    if raw_trace_regions is None:
        # Backwards-compatible single-region contract used by older Brain
        # View requests. New Brain View windows always provide trace_regions
        # explicitly, including an empty list when no row is checked.
        raw_trace_regions = [
            {
                "name": str(
                    map_settings.get("selected_region_name") or "Whole brain"
                ),
                "label_ids": list(
                    map_settings.get("selected_region_label_ids", [])
                ),
            }
        ]

    brain_trace_payloads: list[dict[str, Any]] = []
    if len(trace_points_mm):
        for raw_region in raw_trace_regions if isinstance(raw_trace_regions, list) else []:
            if not isinstance(raw_region, dict):
                continue
            region_name = str(raw_region.get("name") or "Brain region")
            region_label_ids = {
                int(value) for value in raw_region.get("label_ids", [])
            }
            region_mask = (
                np.isin(
                    trace_point_labels,
                    np.asarray(sorted(region_label_ids), dtype=np.int32),
                )
                if region_label_ids
                else np.ones(trace_count, dtype=bool)
            )
            if not np.any(region_mask):
                raise StudioOperationError(
                    f"The checked brain region {region_name!r} has no retained waveform samples."
                )
            coefficients = weighted_rms_coefficients(
                trace_fixed[region_mask],
                trace_bases[:, region_mask, :],
                trace_point_weights[region_mask],
            )
            region_rms = weighted_rms_from_coefficients(
                timeline["analysis_currents"], coefficients
            )
            rms_scale = 1e6 if field == "B" else 1.0
            region_rms = np.asarray(region_rms, dtype=float) * rms_scale
            region_stats = time_statistics(
                timeline["analysis_times"], region_rms
            )
            brain_trace_payloads.append(
                {
                    "time_s": np.asarray(
                        timeline["analysis_times"], dtype=float
                    ).tolist(),
                    "values": region_rms.tolist(),
                    "label": f"{region_name} — spatial RMS |{field}|",
                    "unit": unit,
                    "axis": "field",
                    "stats": {
                        key: float(value)
                        for key, value in region_stats.items()
                    },
                }
            )
    observation_payloads[0:0] = brain_trace_payloads

    brain_region_frame_metrics_payload = None
    if (
        trace_count
        and bool(map_settings.get("precompute_brain_region_frames", False))
        and map_settings.get("trace_source_voxel_counts")
    ):
        source_counts = {
            int(key): int(value)
            for key, value in dict(map_settings.get("trace_source_voxel_counts", {})).items()
        }
        unique_labels, sampled_values = np.unique(trace_point_labels, return_counts=True)
        sampled_counts = {
            int(label): int(count)
            for label, count in zip(unique_labels, sampled_values, strict=True)
        }
        trace_samples = BrainSampleSet(
            points_m=np.asarray(trace_points_mm, dtype=float) / 1000.0,
            label_ids=np.asarray(trace_point_labels, dtype=np.int32),
            weights_mm3=np.asarray(trace_point_weights, dtype=float),
            source_indices=np.arange(trace_count, dtype=np.int64),
            source_voxel_counts=source_counts,
            sampled_counts=sampled_counts,
            voxel_volume_mm3=float(map_settings.get("trace_voxel_volume_mm3", 1.0)),
        )

        def brain_metric_progress(completed: int, total: int) -> None:
            _emit_render_task_progress(
                progress_callback,
                task_id="brain_metrics",
                stage="Precomputing Brain Areas metrics for playback frames…",
                completed=completed,
                total=total,
                unit="frames",
                detail=f"Brain Areas frame metrics {completed:,}/{total:,}",
            )

        # Do not allocate a frame × region × metric cube without a ceiling.
        # At extreme slow-motion frame counts this auxiliary Brain Areas table
        # used to dwarf the actual GPU field payload (millions of frames can be
        # several GB).  Sample the table timeline evenly into a bounded cache.
        # The WebGL field still renders every requested frame; only the numeric
        # side-table cache is temporally decimated, which is effectively
        # invisible at the very high frame rates that trigger this path.
        metric_row_count = len(list(lpba40_region_hierarchy().walk()))
        bytes_per_metric_frame = max(1, metric_row_count * 7 * np.dtype(np.float32).itemsize)
        max_metric_frames = max(2, MAX_BRAIN_FRAME_METRIC_CACHE_BYTES // bytes_per_metric_frame)
        source_frame_count = int(len(timeline["frame_currents"]))
        if source_frame_count > max_metric_frames:
            metric_indices = np.unique(
                np.rint(np.linspace(0, source_frame_count - 1, max_metric_frames)).astype(np.int64)
            )
            metric_currents = np.asarray(timeline["frame_currents"], dtype=float)[metric_indices]
        else:
            metric_indices = np.arange(source_frame_count, dtype=np.int64)
            metric_currents = np.asarray(timeline["frame_currents"], dtype=float)

        frame_metrics = lpba40_frame_region_metrics(
            trace_fixed,
            trace_bases,
            metric_currents,
            trace_samples,
            progress_callback=brain_metric_progress,
        )
        metric_values = np.asarray(frame_metrics["values"], dtype=np.float32)
        metric_times = np.asarray(timeline["frame_times"], dtype=float)[metric_indices]
        playback_metrics = lpba40_playback_region_metrics(
            trace_fixed,
            trace_bases,
            np.asarray(timeline["analysis_currents"], dtype=float),
            np.asarray(timeline["analysis_times"], dtype=float),
            frame_metrics,
            metric_times,
            trace_samples,
        )
        summary_values = np.asarray(
            [
                (
                    playback_metrics[key].playback_rms_uT,
                    playback_metrics[key].mean_uT,
                    playback_metrics[key].max_p95_uT,
                    playback_metrics[key].absolute_peak_uT,
                    playback_metrics[key].b_time_uT_s,
                    playback_metrics[key].b2_time_uT2_s,
                    playback_metrics[key].volume_cm3,
                )
                for key in frame_metrics["keys"]
            ],
            dtype=np.float64,
        )
        peak_times = np.asarray(
            [playback_metrics[key].peak_time_s for key in frame_metrics["keys"]],
            dtype=np.float64,
        )
        brain_region_frame_metrics_payload = {
            "keys": list(frame_metrics["keys"]),
            "names": list(frame_metrics["names"]),
            "shape": list(metric_values.shape),
            "values_b64": _array_b64(metric_values, np.float32),
            "sample_counts": list(frame_metrics["sample_counts"]),
            "voxel_counts": list(frame_metrics["voxel_counts"]),
            "source_frame_count": source_frame_count,
            "sampled_frame_indices_b64": _array_b64(metric_indices, np.int32),
            "playback_summary": {
                "shape": list(summary_values.shape),
                "values_b64": _array_b64(summary_values, np.float64),
                "peak_times_s_b64": _array_b64(peak_times, np.float64),
                "playback": _brain_playback_run_metadata(
                    trace,
                    frame_count=source_frame_count,
                    analysis_sample_count=len(timeline["analysis_times"]),
                    summary_sample_count=len(metric_indices),
                    atlas_sample_count=trace_count,
                    waveform_metadata=waveform_metadata,
                    sequence_steps=normalized_sequence,
                    sequence_loop=bool(sequence_loop),
                ),
            },
        }

    excitation_payload = (
        _excitation_trace_payload(
            trace, timeline, slew_rate_per_s=slew_rate_per_s, initial_drive_value=initial_drive_value
        )
        if include_excitation_trace
        else None
    )
    coil_labels = (
        {
            str(choice["id"]): str(choice.get("label") or choice["id"])
            for choice in drive_coil_catalog(adapter)
        }
        if normalized_sequence
        else {}
    )
    playback_sequence = _playback_sequence_payload(
        normalized_sequence,
        loop=bool(sequence_loop),
        coil_ids=drive_ids,
        coil_labels=coil_labels,
    )
    sequencer_payload = None
    if include_sequencer_trace and normalized_sequence:
        sequencer_payload = _sequencer_trace_payload(
            timeline, steps=normalized_sequence, loop=bool(sequence_loop), coil_labels=coil_labels
        )
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Playback traces and sequencer data ready",
        completed=3,
        total=4,
    )

    payload = {
        "renderer": "field-workbench-webgl2-waveform-v4",
        "view_mode": "3d",
        "title": (
            f"Brain waveform playback — {map_settings.get('atlas_name', 'LPBA40')}"
            if map_settings.get("atlas_name")
            else f"Waveform playback — {component_label}"
        ),
        "component": component,
        "component_label": component_label,
        "field": field,
        "unit": unit,
        "logarithmic": logarithmic,
        "colour_min": float(colour_min),
        "colour_max": float(colour_max),
        "colour_scale": copy.deepcopy(
            FIELD_SIGNED_COLOURSCALE if field_component_is_signed(component) else FIELD_HEAT_COLOURSCALE
        ),
        "signed_component": field_component_is_signed(component),
        "centre_mm": centre.tolist(),
        "span_mm": span,
        "show_grid": bool(grid_spec["enabled"]),
        "axis_labels": [axis["label"] for axis in grid_spec["axes"]],
        "grid": grid_spec,
        "slice_axis": axis_name,
        "slice_positions_mm": fixed_coordinates.tolist() if render_mode == "slices" else [],
        "slice_count": layers if render_mode == "slices" else (1 if render_mode == "points" else samples),
        "resolution": samples,
        "render_mode": render_mode,
        "point_count": map_count if render_mode == "points" else 0,
        "point_size": float(map_settings.get("point_size", 4.0)),
        "overlay_point_count": int(overlay_count if render_mode != "points" else 0),
        "overlay_point_size": float(map_settings.get("point_size", 4.0)),
        "overlay_positions": _array_b64(np.asarray(overlay_points_mm, dtype=np.float32), np.float32) if (overlay_count and render_mode != "points") else "",
        "overlay_fixed_vectors": _array_b64(np.asarray(overlay_fixed, dtype=np.float32), np.float32) if (overlay_count and render_mode != "points") else "",
        "overlay_basis_vectors": _array_b64(np.asarray(overlay_bases[0], dtype=np.float32), np.float32) if (overlay_count and render_mode != "points") else "",
        "overlay_basis_vectors_b64": ([_array_b64(np.asarray(values, dtype=np.float32), np.float32) for values in overlay_bases] if (overlay_count and render_mode != "points") else []),
        "overlay_point_label_ids_b64": _array_b64(np.asarray(overlay_point_labels, dtype=np.int32), np.int32) if (overlay_count and render_mode != "points") else "",
        "intensity_opacity_enabled": opacity_enabled,
        "volume_opacity_lower": opacity_lower,
        "volume_opacity_upper": opacity_upper,
        "volume_opacity_lower_level": opacity_lower_level,
        "volume_opacity_upper_level": opacity_upper_level,
        # Plotly's static Volume trace is a stack of evenly spaced iso-surfaces.
        # The realtime renderer uses the same surface count when selecting the
        # dominant visible field level along each ray.
        "volume_surface_count": max(12, min(40, samples)),
        "volume_ray_steps": max(48, min(128, int(round(samples * 1.5)))),
        "volume_box_min_mm": (centre - half).tolist(),
        "volume_box_max_mm": (centre + half).tolist(),
        "positions": (
            _array_b64(np.asarray(map_points_mm, dtype=np.float32), np.float32)
            if render_mode in {"slices", "points"}
            else ""
        ),
        "fixed_vectors": (
            _array_b64(np.asarray(map_fixed, dtype=np.float32), np.float32)
            if render_mode in {"slices", "points"}
            else ""
        ),
        "basis_vectors": (
            _array_b64(np.asarray(map_bases[0], dtype=np.float32), np.float32)
            if render_mode in {"slices", "points"}
            else ""
        ),
        "basis_vectors_b64": (
            [_array_b64(np.asarray(values, dtype=np.float32), np.float32) for values in map_bases]
            if render_mode in {"slices", "points"}
            else []
        ),
        "drive_coil_ids": drive_ids,
        "drive_coil_count": len(drive_ids),
        "volume_fixed_vectors": (
            _array_b64(np.asarray(map_fixed, dtype=np.float32), np.float32)
            if render_mode == "volume"
            else ""
        ),
        "volume_basis_vectors": (
            _array_b64(np.asarray(map_bases[0], dtype=np.float32), np.float32)
            if render_mode == "volume"
            else ""
        ),
        "volume_basis_vectors_b64": (
            [_array_b64(np.asarray(values, dtype=np.float32), np.float32) for values in map_bases]
            if render_mode == "volume"
            else []
        ),
        "frame_count": int(len(timeline["frame_times"])),
        "frame_times_b64": _array_b64(np.asarray(timeline["frame_times"], dtype=np.float64), np.float64),
        "frame_currents_b64": _array_b64(np.asarray(timeline["frame_currents"], dtype=np.float32), np.float32),
        "free_frame_currents_b64": _array_b64(
            np.asarray(timeline["free_frame_currents"], dtype=np.float32), np.float32
        ) if playback_sequence is not None else "",
        "playback_sequence": playback_sequence,
        "playback_fps": float(playback_fps),
        "real_time_multiplier": playback_real_time_multiplier(
            float(trace.time_s[-1] - trace.time_s[0]), frame_count, playback_fps
        ),
        "scene": scene_payload,
        "observation": observation_payloads[0] if observation_payloads else None,
        "observations": observation_payloads,
        "brain_region_frame_metrics": brain_region_frame_metrics_payload,
        "brain_region_video_table": copy.deepcopy(map_settings.get("brain_region_video_table")),
        "scroll_observation_traces": bool(
            map_settings.get("scroll_observation_traces", False)
        ),
        "excitation": excitation_payload,
        "sequencer": sequencer_payload,
    }
    _emit_render_task_progress(
        progress_callback,
        task_id="assemble",
        stage="Waveform WebGL package ready",
        completed=4,
        total=4,
    )
    return payload


def build_prepared_gpu_payload(
    adapter: "StudioAdapter",
    request: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Build a previously validated waveform request without importing Qt."""

    map_kind = str(request.get("map_kind", "3d")).lower()
    builder = build_2d_gpu_playback if map_kind == "2d" else build_3d_gpu_playback
    result = builder(
        adapter,
        map_settings=copy.deepcopy(request["map_settings"]),
        coil_ids=list(request["coil_ids"]),
        controlled_coil_ids=list(request["controlled_coil_ids"]),
        trace=request["trace"],
        frame_count=int(request["frame_count"]),
        analysis_samples=int(request["analysis_samples"]),
        playback_fps=float(request["playback_fps"]),
        electrical_by_coil=copy.deepcopy(request["electrical_by_coil"]),
        initial_drive_value=float(request["initial_drive_value"]),
        slew_rate_per_s=float(request["slew_rate_per_s"]),
        sequence_steps=copy.deepcopy(request.get("sequence_steps")),
        sequence_loop=bool(request.get("sequence_loop", False)),
        waveform_metadata=copy.deepcopy(request.get("waveform_metadata")),
        observation=copy.deepcopy(request.get("observation")),
        include_excitation_trace=bool(
            request.get("include_excitation_trace", False)
        ),
        include_sequencer_trace=bool(
            request.get("include_sequencer_trace", False)
        ),
        progress_callback=progress_callback,
    )
    initial_camera = request.get("initial_camera")
    if isinstance(initial_camera, dict):
        result["initial_camera"] = copy.deepcopy(initial_camera)
    result["waveform_duration_s"] = float(request["waveform_duration_s"])
    result["real_time_multiplier"] = float(request["real_time_multiplier"])
    camera_timeline = request.get("camera_timeline")
    if isinstance(camera_timeline, dict):
        result["camera_timeline"] = copy.deepcopy(camera_timeline)
    return result


def build_3d_animation(
    adapter: "StudioAdapter",
    *,
    map_settings: dict[str, Any],
    coil_id: str,
    trace: WaveformTrace,
    frame_count: int,
    analysis_samples: int,
    playback_fps: float,
    resistance_ohm: float | None = None,
    inductance_h: float | None = None,
    initial_current_a: float = 0.0,
    initial_drive_value: float | None = None,
    slew_rate_per_s: float | None = 0.0,
    observation: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Build an animated 3D slice map with optional synchronized trace."""
    centre = np.asarray(map_settings["centre_mm"], dtype=float).reshape(3)
    span = float(map_settings["span_mm"])
    axis_name = str(map_settings["slice_axis"]).lower()
    layers = int(map_settings["slice_count"])
    samples = int(map_settings["resolution"])
    field = str(map_settings["field"]).upper()
    component = str(map_settings["component"]).lower()
    logarithmic = bool(map_settings.get("logarithmic", False))
    if axis_name not in {"x", "y", "z"}:
        raise StudioOperationError("The 3D waveform slice axis must be X, Y, or Z.")
    if logarithmic and not field_component_is_magnitude(component):
        raise StudioOperationError("Log scale is available only for field magnitude or absolute axis-component magnitude.")
    scalar_count = frame_count * layers * samples * samples
    if scalar_count > MAX_ANIMATION_SCALAR_VALUES:
        raise StudioOperationError(
            "This 3D playback would create too many animated field samples. Reduce animation frames, map resolution, or slice count."
        )

    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    plane_axes = [index for index in range(3) if index != axis_index]
    half = span / 2.0
    coordinates = [np.linspace(centre[index] - half, centre[index] + half, samples) for index in range(3)]
    fixed_coordinates = (
        np.asarray([centre[axis_index]], dtype=float)
        if layers == 1
        else np.linspace(centre[axis_index] - half, centre[axis_index] + half, layers)
    )
    horizontal, vertical = np.meshgrid(coordinates[plane_axes[0]], coordinates[plane_axes[1]])
    plane_coordinates: list[list[np.ndarray]] = []
    point_batches: list[np.ndarray] = []
    for fixed_coordinate in fixed_coordinates:
        xyz = [np.empty_like(horizontal) for _ in range(3)]
        xyz[plane_axes[0]] = horizontal
        xyz[plane_axes[1]] = vertical
        xyz[axis_index] = np.full_like(horizontal, fixed_coordinate)
        plane_coordinates.append(xyz)
        point_batches.append(np.column_stack([value.reshape(-1) for value in xyz]))
    map_points_mm = np.concatenate(point_batches, axis=0)

    observation_points, observation_meta = _observation_geometry(adapter, observation)
    map_count = len(map_points_mm)
    all_points = np.vstack([map_points_mm / 1000.0, observation_points])
    fixed, basis = field_basis_for_drive(
        adapter, coil_id=coil_id, points_m=all_points, field=field, progress_callback=progress_callback
    )
    timeline = _prepare_timeline(
        trace,
        frame_count=frame_count,
        analysis_samples=analysis_samples,
        resistance_ohm=resistance_ohm,
        inductance_h=inductance_h,
        initial_current_a=initial_current_a,
        initial_drive_value=initial_drive_value,
        slew_rate_per_s=slew_rate_per_s,
    )
    frame_vectors = fixed[:map_count][None, :, :] + timeline["frame_currents"][:, None, None] * basis[:map_count][None, :, :]
    raw_frames = _scalar_values(frame_vectors, component=component, field=field).reshape(frame_count, layers, samples, samples)
    colour_frames, automatic_min, automatic_max = _colour_planes(raw_frames, logarithmic=logarithmic)

    colour_min, colour_max = _resolve_gpu_colour_limits(
        automatic_min,
        automatic_max,
        manual_min=map_settings.get("colour_minimum"),
        manual_max=map_settings.get("colour_maximum"),
        component=component,
        logarithmic=logarithmic,
    )

    component_label = field_component_label(field, component)
    unit = "µT" if field == "B" else "A/m"
    colour_title = f"log₁₀ {component_label} ({unit})" if logarithmic else f"{component_label} ({unit})"

    figure = (
        adapter._field_volume_scene_figure(
            map_settings.get("scene_object_ids"),
            scene_object_opacities=map_settings.get("scene_object_opacities"),
        )
        if bool(map_settings.get("include_scene", True))
        else {"data": [], "layout": {}}
    )
    data: list[dict[str, Any]] = figure.setdefault("data", [])

    low, high = centre - half, centre + half
    corners = np.asarray([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float)
    edge_indices = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    edge_points: list[list[float | None]] = []
    for first_index, second_index in edge_indices:
        edge_points.extend([corners[first_index].tolist(), corners[second_index].tolist(), [None, None, None]])
    edges = np.asarray(edge_points, dtype=object)
    data.append(
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": edges[:, 0].tolist(),
            "y": edges[:, 1].tolist(),
            "z": edges[:, 2].tolist(),
            "line": {"color": "rgba(71,85,105,0.55)", "width": 2},
            "hoverinfo": "skip",
            "showlegend": False,
            "name": "Map volume",
            "uid": "waveform-map-volume",
        }
    )

    surface_indices: list[int] = []
    for layer_index, (fixed_coordinate, xyz) in enumerate(zip(fixed_coordinates, plane_coordinates, strict=True)):
        surface_indices.append(len(data))
        data.append(
            {
                "type": "surface",
                "x": xyz[0].tolist(),
                "y": xyz[1].tolist(),
                "z": xyz[2].tolist(),
                "surfacecolor": colour_frames[0, layer_index].tolist(),
                "customdata": raw_frames[0, layer_index].tolist(),
                "coloraxis": "coloraxis",
                "opacity": 0.78,
                "showscale": False,
                "showlegend": False,
                "name": f"{axis_name.upper()} = {fixed_coordinate:.5g} mm",
                "meta": {"component": component_label, "unit": unit, "slice": f"{axis_name.upper()} = {fixed_coordinate:.5g} mm"},
                "hovertemplate": (
                    "X: %{x:.5g} mm<br>Y (AP): %{y:.5g} mm<br>Z (SI): %{z:.5g} mm<br>"
                    "%{meta.component}: %{customdata:.6g} %{meta.unit}<br>%{meta.slice}<extra></extra>"
                ),
                "uid": f"waveform-volume-slice-{axis_name}-{layer_index}",
            }
        )

    observation_label = None
    cursor_index = None
    analysis_time_display, time_axis_label, time_scale = _display_time_axis(timeline["analysis_times"])
    frame_time_display = timeline["frame_times"] * time_scale
    annotations: list[dict[str, Any]] = []
    if observation_meta is not None:
        obs_vectors = fixed[map_count:][None, :, :] + timeline["analysis_currents"][:, None, None] * basis[map_count:][None, :, :]
        observation_values, observation_label = _observation_values(obs_vectors, observation_meta, field=field)
        stats = time_statistics(timeline["analysis_times"], observation_values)
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": analysis_time_display.tolist(),
                "y": observation_values.tolist(),
                "xaxis": "x2",
                "yaxis": "y2",
                "name": observation_label,
                "showlegend": False,
                "hovertemplate": f"%{{x:.6g}}<br>{observation_label}: %{{y:.6g}}<extra></extra>",
                "uid": "waveform-observation",
            }
        )
        y_min, y_max = float(np.min(observation_values)), float(np.max(observation_values))
        if math.isclose(y_min, y_max, rel_tol=1e-12, abs_tol=1e-15):
            pad = max(abs(y_min) * 0.05, 1.0)
            y_min -= pad
            y_max += pad
        cursor_index = len(data)
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [float(frame_time_display[0]), float(frame_time_display[0])],
                "y": [y_min, y_max],
                "xaxis": "x2",
                "yaxis": "y2",
                "line": {"color": "rgba(15,23,42,0.65)", "width": 1.5},
                "hoverinfo": "skip",
                "showlegend": False,
                "uid": "waveform-time-cursor",
            }
        )
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 0.245,
                "xanchor": "right",
                "yanchor": "bottom",
                "showarrow": False,
                "align": "right",
                "text": (
                    f"Mean/DC: {_format_stat(stats['mean'], unit)} &nbsp; "
                    f"RMS: {_format_stat(stats['rms'], unit)} &nbsp; "
                    f"Pk-pk: {_format_stat(stats['pk_pk'], unit)}"
                ),
                "font": {"size": 12},
            }
        )

    buttons, sliders = _animation_controls(frame_time_display)
    layout = figure.setdefault("layout", {})
    layout.update(
        {
            "template": "plotly_white",
            "title": {"text": f"Waveform playback — {component_label}", "x": 0.5},
            "margin": {"l": 54, "r": 92, "t": 52, "b": 42},
            "showlegend": False,
            "uirevision": "field-workbench-waveform-3d",
            "meta": {
                "field_workbench_waveform": {
                    "playback_fps": float(playback_fps),
                    "real_time_multiplier": playback_real_time_multiplier(
                        float(trace.time_s[-1] - trace.time_s[0]), frame_count, playback_fps
                    ),
                    "frame_count": int(frame_count),
                }
            },
            "updatemenus": buttons,
            "sliders": sliders,
            "annotations": annotations,
            "coloraxis": {
                "colorscale": copy.deepcopy(
                    FIELD_SIGNED_COLOURSCALE if field_component_is_signed(component) else FIELD_HEAT_COLOURSCALE
                ),
                "cmin": colour_min,
                "cmax": colour_max,
                **({"cmid": 0.0} if field_component_is_signed(component) else {}),
                "colorbar": {"title": {"text": colour_title}, "thickness": 15, "len": 0.58 if observation_meta is not None else 0.78, "y": 0.68 if observation_meta is not None else 0.55},
            },
        }
    )
    scene = layout.setdefault("scene", {})
    scene.update(
        {
            "xaxis": {"title": {"text": "X (LR, mm)"}},
            "yaxis": {"title": {"text": "Y (AP, mm)"}},
            "zaxis": {"title": {"text": "Z (SI, mm)"}},
            "aspectmode": "cube",
            "uirevision": "field-workbench-waveform-3d-camera",
            "domain": {"x": [0.0, 1.0], "y": [0.36, 1.0] if observation_meta is not None else [0.12, 1.0]},
        }
    )
    if observation_meta is not None:
        layout["xaxis2"] = {"domain": [0.0, 1.0], "anchor": "y2", "title": {"text": time_axis_label}}
        layout["yaxis2"] = {"domain": [0.0, 0.20], "anchor": "x2", "title": {"text": observation_label}}
    else:
        for slider in layout["sliders"]:
            slider["y"] = 0.075

    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        frame_data: list[dict[str, Any]] = []
        trace_indices: list[int] = []
        for layer_index, trace_index in enumerate(surface_indices):
            frame_data.append(
                {
                    "surfacecolor": colour_frames[frame_index, layer_index].tolist(),
                    "customdata": raw_frames[frame_index, layer_index].tolist(),
                }
            )
            trace_indices.append(trace_index)
        if cursor_index is not None:
            frame_data.append({"x": [float(frame_time_display[frame_index]), float(frame_time_display[frame_index])]})
            trace_indices.append(cursor_index)
        frames.append(
            {
                "name": f"waveform-frame-{frame_index}",
                "data": frame_data,
                "traces": trace_indices,
            }
        )
    return {"data": data, "layout": layout, "frames": frames}
