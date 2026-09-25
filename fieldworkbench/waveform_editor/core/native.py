"""Format-neutral waveform construction and import/export boundary helpers.

The Waveform Editor works exclusively with floating-point Field Workbench
waveform samples for playback. Runtime-timed encoded imports retain their raw
source samples and declarative format profile as authoring metadata so timing
and conversion settings can be reopened without making any source encoding part
of the playback representation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ...waveform_io import WaveformIOError, WaveformSeries


class NativeWaveformError(ValueError):
    """Raised when native waveform construction parameters are invalid."""


def _finite(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise NativeWaveformError(f"{label} must be finite.")
    return number


@dataclass(frozen=True)
class GeneratedTrace:
    """One generator result expressed only in physical time/value terms.

    The point density used to carry the curve is an internal implementation
    detail.  User-facing generator controls describe frequency, duration,
    magnitude, duty cycle, and shape instead of sample counts or intervals.
    """

    time_s: tuple[float, ...]
    values: tuple[float, ...]
    settings: dict[str, object]


@dataclass(frozen=True)
class ComposedTrace:
    """The exact physical trace produced by Replace/Append composition.

    Builder chunks keep their own internal point density. Composition never
    resamples either side; it only places the generated chunk on the existing
    physical time axis. ``append_boundary_s`` is the authored join time, not a
    sampling interval or encoding detail.
    """

    time_s: tuple[float, ...]
    values: tuple[float, ...]
    append_boundary_s: float | None = None

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])


def _validated_physical_trace(
    time_s: Sequence[float],
    values: Sequence[float],
    *,
    label: str,
) -> tuple[list[float], list[float]]:
    times = [float(value) for value in time_s]
    samples = [float(value) for value in values]
    if len(times) != len(samples) or len(times) < 2:
        raise NativeWaveformError(f"{label} must contain at least two matching time/value samples.")
    if not all(math.isfinite(value) for value in (*times, *samples)):
        raise NativeWaveformError(f"{label} contains a non-finite value.")
    if any(times[index + 1] <= times[index] for index in range(len(times) - 1)):
        raise NativeWaveformError(f"{label} time must increase strictly.")
    return times, samples


def compose_generated_trace(
    generated: GeneratedTrace,
    *,
    append: bool,
    existing_time_s: Sequence[float] | None = None,
    existing_values: Sequence[float] | None = None,
) -> ComposedTrace:
    """Compose a generated chunk without imposing a shared sample interval.

    Replace normalizes the generated chunk to start at zero. Append preserves
    every existing sample and places the new chunk after the current endpoint.
    When the two chunks meet at different values, a point at the next
    representable time after the boundary preserves both sides of the step
    instead of overwriting the old endpoint. This makes the join independent of
    either chunk's internal sample spacing while keeping time strictly
    increasing for playback.
    """

    new_times, new_values = _validated_physical_trace(
        generated.time_s,
        generated.values,
        label="Generated waveform",
    )
    new_start = new_times[0]
    relative = [value - new_start for value in new_times]

    if not append:
        return ComposedTrace(tuple(relative), tuple(new_values), None)

    if existing_time_s is None or existing_values is None:
        raise NativeWaveformError("Append requires an existing waveform.")
    old_times, old_values = _validated_physical_trace(
        existing_time_s,
        existing_values,
        label="Existing waveform",
    )
    boundary = old_times[-1]
    combined_times = list(old_times)
    combined_values = list(old_values)

    # A discontinuous chunk must not retroactively change the final sample of
    # the preceding chunk. Represent the jump immediately after the authored
    # boundary, using only floating-point ordering rather than a user-visible
    # or inherited sample interval.
    previous_time = boundary
    if new_values[0] != old_values[-1]:
        step_time = math.nextafter(boundary, math.inf)
        if not math.isfinite(step_time):
            raise NativeWaveformError("Waveform duration is too large to append another segment.")
        combined_times.append(step_time)
        combined_values.append(new_values[0])
        previous_time = step_time

    # The first generated point is represented either by the existing endpoint
    # (continuous join) or by the explicit step point above. Add the remaining
    # generated points at their own relative physical times. If floating-point
    # precision at a very large time origin collapses two adjacent values, move
    # only as far as the next representable float; never infer a sample period.
    for relative_time, sample in zip(relative[1:], new_values[1:]):
        candidate = boundary + relative_time
        if candidate <= previous_time:
            candidate = math.nextafter(previous_time, math.inf)
        if not math.isfinite(candidate):
            raise NativeWaveformError("Waveform duration is too large to append another segment.")
        combined_times.append(candidate)
        combined_values.append(sample)
        previous_time = candidate

    return ComposedTrace(tuple(combined_times), tuple(combined_values), boundary)


@dataclass(frozen=True)
class FreeFormPoint:
    """One anchor in a normalized free-form Bézier waveform.

    ``x`` is normalized segment time from 0..1 and ``y`` is the physical
    electrical value in the waveform's current display unit.  One mirrored
    tangent vector is stored per anchor so interior joins stay smooth.
    """

    x: float
    y: float
    handle_dx: float = 0.1
    handle_dy: float = 0.0


_INTERNAL_SAMPLES_PER_CYCLE = 256
_INTERNAL_MIN_INTERVALS = 256
_INTERNAL_SHAPE_INTERVALS = 1024
_INTERNAL_MAX_POINTS = 1_000_001


def _positive(value: float, *, label: str) -> float:
    number = _finite(value, label=label)
    if number <= 0.0:
        raise NativeWaveformError(f"{label} must be greater than zero.")
    return number


def _non_negative(value: float, *, label: str) -> float:
    number = _finite(value, label=label)
    if number < 0.0:
        raise NativeWaveformError(f"{label} cannot be negative.")
    return number


def _automatic_time_axis(
    duration_s: float,
    *,
    frequency_hz: float | None = None,
    minimum_intervals: int = _INTERNAL_MIN_INTERVALS,
) -> np.ndarray:
    """Return an internal time axis without exposing digitisation to the user."""

    duration = _positive(duration_s, label="Duration")
    intervals = max(2, int(minimum_intervals))
    if frequency_hz is not None:
        frequency = _positive(frequency_hz, label="Frequency")
        intervals = max(
            intervals,
            int(math.ceil(duration * frequency * _INTERNAL_SAMPLES_PER_CYCLE)),
        )
    intervals = min(intervals, _INTERNAL_MAX_POINTS - 1)
    return np.linspace(0.0, duration, intervals + 1, dtype=float)


def _generated_trace(time_s: np.ndarray, values: np.ndarray, settings: dict[str, object]) -> GeneratedTrace:
    times = np.asarray(time_s, dtype=float).reshape(-1)
    samples = np.asarray(values, dtype=float).reshape(-1)
    if len(times) != len(samples) or len(times) < 2:
        raise NativeWaveformError("Generated waveform must contain a finite physical trace.")
    if not np.isfinite(times).all() or not np.isfinite(samples).all():
        raise NativeWaveformError("Generated waveform contains a non-finite value.")
    if np.any(np.diff(times) <= 0.0):
        raise NativeWaveformError("Generated waveform time must increase strictly.")
    return GeneratedTrace(
        tuple(float(value) for value in times),
        tuple(float(value) for value in samples),
        dict(settings),
    )


def square_wave_trace(
    *,
    frequency_hz: float,
    duration_s: float,
    peak_to_peak: float,
    offset: float = 0.0,
    duty_cycle_percent: float = 50.0,
) -> GeneratedTrace:
    """Create a square wave from ordinary signal-generator controls."""

    frequency = _positive(frequency_hz, label="Frequency")
    duration = _positive(duration_s, label="Duration")
    magnitude = _non_negative(peak_to_peak, label="Peak-to-peak magnitude")
    centre = _finite(offset, label="Offset")
    duty = _finite(duty_cycle_percent, label="Duty cycle")
    if not 0.0 < duty < 100.0:
        raise NativeWaveformError("Duty cycle must be greater than 0% and less than 100%.")
    times = _automatic_time_axis(duration, frequency_hz=frequency)
    phase = np.mod(times * frequency, 1.0)
    high = centre + magnitude / 2.0
    low = centre - magnitude / 2.0
    values = np.where(phase < duty / 100.0, high, low)
    return _generated_trace(
        times,
        values,
        {
            "method": "signal_generator",
            "shape": "square",
            "frequency_hz": frequency,
            "duration_s": duration,
            "peak_to_peak": magnitude,
            "offset": centre,
            "duty_cycle_percent": duty,
        },
    )


def sine_wave_trace(
    *,
    frequency_hz: float,
    duration_s: float,
    peak_to_peak: float,
    offset: float = 0.0,
    phase_deg: float = 0.0,
) -> GeneratedTrace:
    """Create a sine wave from frequency, magnitude, offset, phase, and duration."""

    frequency = _positive(frequency_hz, label="Frequency")
    duration = _positive(duration_s, label="Duration")
    magnitude = _non_negative(peak_to_peak, label="Peak-to-peak magnitude")
    centre = _finite(offset, label="Offset")
    phase = _finite(phase_deg, label="Phase")
    times = _automatic_time_axis(duration, frequency_hz=frequency)
    radians = 2.0 * math.pi * frequency * times + math.radians(phase)
    values = centre + (magnitude / 2.0) * np.sin(radians)
    return _generated_trace(
        times,
        values,
        {
            "method": "signal_generator",
            "shape": "sine",
            "frequency_hz": frequency,
            "duration_s": duration,
            "peak_to_peak": magnitude,
            "offset": centre,
            "phase_deg": phase,
        },
    )


def dc_trace(*, duration_s: float, level: float) -> GeneratedTrace:
    """Create a constant level segment for holds, pauses, and DC offsets."""
    duration = _positive(duration_s, label="Duration")
    value = _finite(level, label="DC level")
    return _generated_trace(
        np.asarray([0.0, duration], dtype=float),
        np.asarray([value, value], dtype=float),
        {"method": "signal_generator", "shape": "dc", "duration_s": duration, "level": value},
    )


def _smoothstep(start: float, stop: float, position: float) -> float:
    if stop <= start:
        return 1.0
    normalized = max(0.0, min(1.0, (position - start) / (stop - start)))
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _smooth_transition(start_value: float, stop_value: float, start_time: float, stop_time: float, time: float) -> float:
    amount = _smoothstep(start_time, stop_time, time)
    return start_value + (stop_value - start_value) * amount


def action_potential_trace(
    *,
    duration_s: float,
    start_percent: float,
    peak_percent: float,
    trough_percent: float,
    recovery_percent: float,
    resting_level: float,
    peak_level: float,
    trough_level: float,
) -> GeneratedTrace:
    """Create a smooth action-potential-shaped pulse over a physical duration."""

    duration = _positive(duration_s, label="Duration")
    timing = tuple(float(value) for value in (start_percent, peak_percent, trough_percent, recovery_percent))
    if not 0.0 <= timing[0] < timing[1] < timing[2] < timing[3] <= 100.0:
        raise NativeWaveformError("Action-potential timing must satisfy start < peak < trough < recovered.")
    resting = _finite(resting_level, label="Resting level")
    peak = _finite(peak_level, label="Peak level")
    trough = _finite(trough_level, label="Trough level")
    if not peak > resting > trough:
        raise NativeWaveformError("Action-potential levels must satisfy peak > resting > trough.")

    times = _automatic_time_axis(duration, minimum_intervals=_INTERNAL_SHAPE_INTERVALS)
    percentages = times / duration * 100.0
    values: list[float] = []
    for position in percentages:
        if position <= timing[0]:
            value = resting
        elif position < timing[1]:
            value = _smooth_transition(resting, peak, timing[0], timing[1], float(position))
        elif position < timing[2]:
            value = _smooth_transition(peak, trough, timing[1], timing[2], float(position))
        elif position < timing[3]:
            value = _smooth_transition(trough, resting, timing[2], timing[3], float(position))
        else:
            value = resting
        values.append(float(value))
    return _generated_trace(
        times,
        np.asarray(values, dtype=float),
        {
            "method": "signal_generator",
            "shape": "action_potential",
            "duration_s": duration,
            "start_percent": timing[0],
            "peak_percent": timing[1],
            "trough_percent": timing[2],
            "recovery_percent": timing[3],
            "resting_level": resting,
            "peak_level": peak,
            "trough_level": trough,
        },
    )


def _cubic_bezier(a: float, b: float, c: float, d: float, t: float) -> float:
    inverse = 1.0 - t
    return (
        inverse * inverse * inverse * a
        + 3.0 * inverse * inverse * t * b
        + 3.0 * inverse * t * t * c
        + t * t * t * d
    )


def _normalized_free_form_points(
    control_points: Sequence[FreeFormPoint | tuple[float, float] | tuple[float, float, float, float]],
) -> list[FreeFormPoint]:
    """Normalize modern Bézier anchors and the brief Phase-3.1 tuple format."""

    if len(control_points) < 2:
        raise NativeWaveformError("A free-form waveform needs at least two control points.")

    anchors: list[FreeFormPoint] = []
    for index, point in enumerate(control_points):
        if isinstance(point, FreeFormPoint):
            anchor = FreeFormPoint(
                _finite(point.x, label=f"Control point {index + 1} time"),
                _finite(point.y, label=f"Control point {index + 1} value"),
                _finite(point.handle_dx, label=f"Control point {index + 1} handle length"),
                _finite(point.handle_dy, label=f"Control point {index + 1} handle value"),
            )
        else:
            values = tuple(point)
            if len(values) not in (2, 4):
                raise NativeWaveformError(f"Control point {index + 1} is incomplete.")
            # Phase 3.1 briefly exposed 0..100 percentage/value pairs.  Accept
            # them here as an internal compatibility convenience while the UI
            # now works exclusively with direct-manipulation Bézier anchors.
            x = _finite(values[0], label=f"Control point {index + 1} time") / 100.0
            y = _finite(values[1], label=f"Control point {index + 1} value")
            if len(values) == 4:
                dx = _finite(values[2], label=f"Control point {index + 1} handle length") / 100.0
                dy = _finite(values[3], label=f"Control point {index + 1} handle value")
            else:
                dx = 0.0
                dy = 0.0
            anchor = FreeFormPoint(x, y, dx, dy)
        anchors.append(anchor)

    anchors.sort(key=lambda item: item.x)
    if not math.isclose(anchors[0].x, 0.0, abs_tol=1e-9) or not math.isclose(
        anchors[-1].x, 1.0, abs_tol=1e-9
    ):
        raise NativeWaveformError("Free-form endpoints must be at 0% and 100% time.")

    previous_x = -1.0
    for index, point in enumerate(anchors):
        if not 0.0 <= point.x <= 1.0:
            raise NativeWaveformError("Free-form control-point time must be between 0% and 100%.")
        if point.x <= previous_x:
            raise NativeWaveformError("Free-form control points must be strictly left-to-right.")
        if point.handle_dx < 0.0:
            raise NativeWaveformError("Free-form tangent handles cannot point backward in time.")
        previous_x = point.x

        left_span = point.x - anchors[index - 1].x if index > 0 else None
        right_span = anchors[index + 1].x - point.x if index + 1 < len(anchors) else None
        spans = [span for span in (left_span, right_span) if span is not None]
        max_dx = 0.49 * min(spans) if spans else 0.0
        if point.handle_dx > max_dx + 1e-12:
            raise NativeWaveformError("A free-form tangent handle extends past its neighbouring point.")

    return anchors


def free_form_trace(
    control_points: Sequence[FreeFormPoint | tuple[float, float] | tuple[float, float, float, float]],
    *,
    duration_s: float,
) -> GeneratedTrace:
    """Sample a smooth, time-monotonic Bézier waveform over a physical duration.

    The authored object is the Bézier path itself.  Sampling density remains an
    internal playback/detail concern and is never part of the user workflow.
    """

    duration = _positive(duration_s, label="Duration")
    anchors = _normalized_free_form_points(control_points)
    times = _automatic_time_axis(duration, minimum_intervals=_INTERNAL_SHAPE_INTERVALS)
    target_x = times / duration
    values: list[float] = []
    segment_index = 0
    for sample_index, wanted_x in enumerate(target_x):
        while segment_index + 1 < len(anchors) - 1 and wanted_x > anchors[segment_index + 1].x:
            segment_index += 1

        start = anchors[segment_index]
        end = anchors[segment_index + 1]
        if sample_index == 0:
            value = start.y
        elif sample_index == len(target_x) - 1:
            value = end.y
        else:
            c1x = start.x + start.handle_dx
            c1y = start.y + start.handle_dy
            c2x = end.x - end.handle_dx
            c2y = end.y - end.handle_dy
            low = 0.0
            high = 1.0
            for _ in range(32):
                t = (low + high) / 2.0
                curve_x = _cubic_bezier(start.x, c1x, c2x, end.x, t)
                if curve_x < wanted_x:
                    low = t
                else:
                    high = t
            t = (low + high) / 2.0
            value = _cubic_bezier(start.y, c1y, c2y, end.y, t)
        values.append(float(value))

    return _generated_trace(
        times,
        np.asarray(values, dtype=float),
        {
            "method": "signal_generator",
            "shape": "free_form",
            "duration_s": duration,
            "control_points": [
                {
                    "time_percent": float(point.x * 100.0),
                    "value": float(point.y),
                    "handle_time_percent": float(point.handle_dx * 100.0),
                    "handle_value": float(point.handle_dy),
                }
                for point in anchors
            ],
            "curve": "cubic_bezier_mirrored_tangents",
        },
    )

def generated_trace_from_recipe_segment(segment: Mapping[str, object]) -> GeneratedTrace:
    """Rebuild one generator trace from a canonical authoring-recipe segment.

    Recipes store the controls the user authored, not internal samples.  This
    helper is the inverse boundary used when an existing generated segment is
    reopened in the embedded waveform-tool controls.
    """

    if str(segment.get("kind") or "").strip().lower() != "generator":
        raise NativeWaveformError("Only generated recipe segments can be rebuilt in waveform-tool controls.")
    generator = str(segment.get("generator") or "").strip().lower()
    parameters = segment.get("parameters")
    if not isinstance(parameters, Mapping):
        raise NativeWaveformError("Generated recipe segment is missing its parameters.")
    try:
        duration = float(segment.get("duration_s"))
    except (TypeError, ValueError) as error:
        raise NativeWaveformError("Generated recipe segment has an invalid duration.") from error

    values = dict(parameters)
    if generator == "square":
        return square_wave_trace(
            frequency_hz=float(values["frequency_hz"]),
            duration_s=duration,
            peak_to_peak=float(values["peak_to_peak"]),
            offset=float(values["offset"]),
            duty_cycle_percent=float(values["duty_cycle_percent"]),
        )
    if generator == "sine":
        return sine_wave_trace(
            frequency_hz=float(values["frequency_hz"]),
            duration_s=duration,
            peak_to_peak=float(values["peak_to_peak"]),
            offset=float(values["offset"]),
            phase_deg=float(values["phase_deg"]),
        )
    if generator == "action_potential":
        return action_potential_trace(
            duration_s=duration,
            start_percent=float(values["start_percent"]),
            peak_percent=float(values["peak_percent"]),
            trough_percent=float(values["trough_percent"]),
            recovery_percent=float(values["recovery_percent"]),
            resting_level=float(values["resting_level"]),
            peak_level=float(values["peak_level"]),
            trough_level=float(values["trough_level"]),
        )
    if generator == "free_form":
        raw_points = values.get("control_points")
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes, bytearray)):
            raise NativeWaveformError("Free-form recipe segment is missing its Bézier anchors.")
        points: list[FreeFormPoint] = []
        for raw in raw_points:
            if not isinstance(raw, Mapping):
                raise NativeWaveformError("Free-form recipe contains an invalid Bézier anchor.")
            points.append(
                FreeFormPoint(
                    float(raw["time_percent"]) / 100.0,
                    float(raw["value"]),
                    float(raw.get("handle_time_percent", 0.0)) / 100.0,
                    float(raw.get("handle_value", 0.0)),
                )
            )
        return free_form_trace(points, duration_s=duration)
    if generator == "dc":
        return dc_trace(duration_s=duration, level=float(values["level"]))
    raise NativeWaveformError(f"Unsupported generated recipe segment {generator!r}.")


def trace_from_recipe_segment(segment: Mapping[str, object]) -> GeneratedTrace:
    """Reconstruct a generated, sampled, or editable DAC recipe chunk."""
    kind = str(segment.get("kind") or "").strip().lower()
    if kind == "generator":
        return generated_trace_from_recipe_segment(segment)
    if kind == "sampled":
        return _generated_trace(
            np.asarray(segment.get("time_s", ()), dtype=float),
            np.asarray(segment.get("values", ()), dtype=float),
            {"shape": "sampled", "duration_s": float(segment.get("duration_s", 0.0))},
        )
    if kind == "encoded":
        try:
            from ...waveform_formats import WaveformFormatProfile

            raw = np.asarray(segment.get("raw_values", ()), dtype=float).reshape(-1)
            interval = float(segment.get("sample_interval_s"))
            profile = WaveformFormatProfile.from_dict(dict(segment.get("profile") or {}))
        except Exception as error:
            raise NativeWaveformError("Encoded recipe component contains invalid source/profile settings.") from error
        if len(raw) < 2 or interval <= 0.0:
            raise NativeWaveformError("Encoded recipe component needs at least two source samples and a positive sample interval.")
        if profile.conversion_mode == "runtime_linear":
            low = float(segment.get("runtime_value_at_min", profile.runtime_value_at_min))
            high = float(segment.get("runtime_value_at_max", profile.runtime_value_at_max))
            raw_low = float(profile.runtime_raw_min)
            raw_high = float(profile.runtime_raw_max)
            scale = (high - low) / (raw_high - raw_low)
            offset = low - raw_low * scale
            values = raw * scale + offset
        else:
            values = raw * float(profile.scale) + float(profile.offset)
        times = np.arange(len(raw), dtype=float) * interval
        return _generated_trace(
            times,
            values,
            {"shape": "encoded", "duration_s": float(times[-1])},
        )
    raise NativeWaveformError("This legacy compiled segment cannot be reconstructed or reordered.")


def compose_generated_sequence(traces: Sequence[GeneratedTrace]) -> ComposedTrace:
    """Compose a complete generated waveform from ordered authored chunks.

    This uses the same Replace/Append primitive as the live builder, preserving
    each chunk's independent internal time grid.
    """

    if not traces:
        raise NativeWaveformError("A generated waveform recipe must contain at least one segment.")
    composed = compose_generated_trace(traces[0], append=False)
    for trace in traces[1:]:
        composed = compose_generated_trace(
            trace,
            append=True,
            existing_time_s=composed.time_s,
            existing_values=composed.values,
        )
    return ComposedTrace(composed.time_s, composed.values, None)


def linear_map_codes(
    codes: Iterable[float],
    *,
    code_min: float,
    code_max: float,
    value_at_min: float,
    value_at_max: float,
) -> np.ndarray:
    """Translate raw integer-code space into engineering values."""

    lo_code = _finite(code_min, label="Minimum code")
    hi_code = _finite(code_max, label="Maximum code")
    lo_value = _finite(value_at_min, label="Minimum-code engineering value")
    hi_value = _finite(value_at_max, label="Maximum-code engineering value")
    if hi_code <= lo_code:
        raise NativeWaveformError("Maximum code must be greater than minimum code.")
    array = np.asarray(list(codes), dtype=float).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise NativeWaveformError("Imported codes must contain finite samples.")
    scale = (hi_value - lo_value) / (hi_code - lo_code)
    return lo_value + (array - lo_code) * scale


def quantize_to_unsigned_codes(
    values: Iterable[float],
    *,
    bit_depth: int,
    value_at_zero: float,
    value_at_full_scale: float,
    clip: bool = True,
) -> np.ndarray:
    """Quantize native engineering values at an explicit export boundary."""

    depth = int(bit_depth)
    if depth not in (8, 16):
        raise NativeWaveformError("Export bit depth must be 8 or 16 bits.")
    low = _finite(value_at_zero, label="Zero-code engineering value")
    high = _finite(value_at_full_scale, label="Full-scale engineering value")
    if high == low:
        raise NativeWaveformError("Export engineering endpoints must be different.")
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise NativeWaveformError("Waveform values must contain finite samples.")
    maximum = float((1 << depth) - 1)
    scaled = (array - low) * maximum / (high - low)
    if clip:
        scaled = np.clip(scaled, 0.0, maximum)
    elif np.any((scaled < 0.0) | (scaled > maximum)):
        raise NativeWaveformError("Waveform contains values outside the configured export range.")
    return np.rint(scaled).astype(np.uint16 if depth == 16 else np.uint8)


def uniform_sample_interval_s(series: WaveformSeries, *, relative_tolerance: float = 1e-6) -> float:
    differences = np.diff(np.asarray(series.time_s, dtype=float))
    if not len(differences):
        raise WaveformIOError("A waveform needs at least two samples.")
    interval = float(np.median(differences))
    if interval <= 0.0 or not math.isfinite(interval):
        raise WaveformIOError("Waveform sample interval is invalid.")
    tolerance = max(abs(interval) * float(relative_tolerance), 1e-15)
    if np.max(np.abs(differences - interval)) > tolerance:
        raise WaveformIOError("This export requires a uniformly sampled waveform channel.")
    return interval
