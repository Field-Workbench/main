"""Electrical source/load resolution for exposure drive groups.

The resolver is intentionally independent of the Maps playback UI.  A saved
waveform supplies an ideal current or voltage command plus source properties;
physical coils supply their resolved resistance and self-inductance; an exposure
drive group supplies series/parallel wiring and may substitute per-coil R/L
overrides for that route. Mutual inductance is not
part of this first source/load model, so combined inductance is the uncoupled
self-inductance result.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import expm


class ElectricalResolutionError(ValueError):
    """Raised when an exposure drive group cannot be resolved electrically."""


def _validated_trace(time_s: Any, value: Any, source_kind: str) -> tuple[np.ndarray, np.ndarray, str]:
    kind = str(source_kind).strip().lower()
    if kind not in {"current", "voltage"}:
        raise ElectricalResolutionError("Waveform source must represent current or voltage.")
    try:
        times = np.asarray(time_s, dtype=float).reshape(-1)
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ElectricalResolutionError("Waveform source time/value data must be numeric.") from error
    if len(times) != len(values) or len(times) < 2:
        raise ElectricalResolutionError("Waveform source requires at least two matching time/value samples.")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ElectricalResolutionError("Waveform source time/value data must be finite.")
    if np.any(np.diff(times) <= 0.0) or float(times[-1] - times[0]) <= 0.0:
        raise ElectricalResolutionError("Waveform source times must be strictly increasing.")
    return times, values, kind


def _source_segments(
    times: np.ndarray, values: np.ndarray, slew_rate_per_s: float
) -> list[tuple[float, float, float, float]]:
    rate = float(slew_rate_per_s)
    if not math.isfinite(rate) or rate < 0.0:
        raise ElectricalResolutionError("Source slew rate must be finite and non-negative.")
    ideal = rate == 0.0
    output = float(values[0])
    segments: list[tuple[float, float, float, float]] = []
    tolerance = max(1e-15, float(times[-1] - times[0]) * 1e-15)
    for index in range(len(times) - 1):
        start = float(times[index])
        stop = float(times[index + 1])
        target = float(values[index])
        duration = stop - start
        if ideal:
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


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ElectricalResolutionError(f"{label} must be numeric.") from error
    if not math.isfinite(number) or number < 0.0:
        raise ElectricalResolutionError(f"{label} must be finite and non-negative.")
    return number


def _finite_positive(value: Any, label: str) -> float:
    number = _finite_nonnegative(value, label)
    if number <= 0.0:
        raise ElectricalResolutionError(f"{label} must be greater than zero.")
    return number


def normalize_electrical_loads(loads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(loads, list) or not loads:
        raise ElectricalResolutionError("A drive group needs at least one coil load.")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(loads):
        if not isinstance(raw, dict):
            raise ElectricalResolutionError(f"Coil load {index + 1} must be an object.")
        coil_id = str(raw.get("coil_id", raw.get("id", "")) or "").strip()
        if not coil_id:
            raise ElectricalResolutionError(f"Coil load {index + 1} requires a coil id.")
        if coil_id in seen:
            raise ElectricalResolutionError(f"Coil {coil_id!r} appears more than once in one drive group.")
        seen.add(coil_id)
        result.append(
            {
                "coil_id": coil_id,
                "resistance_ohm": _finite_nonnegative(raw.get("resistance_ohm"), f"{coil_id} resistance"),
                "inductance_h": _finite_positive(raw.get("inductance_h"), f"{coil_id} inductance"),
                "resistance_mode": str(raw.get("resistance_mode", "")),
                "inductance_mode": str(raw.get("inductance_mode", "")),
            }
        )
    return result


def equivalent_load(loads: list[dict[str, Any]], connection: str) -> dict[str, float]:
    """Return uncoupled equivalent R/L for a load group."""
    normalized = normalize_electrical_loads(loads)
    mode = str(connection).strip().lower()
    if mode not in {"series", "parallel"}:
        raise ElectricalResolutionError("Drive-group connection must be series or parallel.")
    resistances = np.asarray([row["resistance_ohm"] for row in normalized], dtype=float)
    inductances = np.asarray([row["inductance_h"] for row in normalized], dtype=float)
    if mode == "series":
        return {
            "resistance_ohm": float(np.sum(resistances)),
            "inductance_h": float(np.sum(inductances)),
        }
    resistance = 0.0 if np.any(resistances <= 0.0) else float(1.0 / np.sum(1.0 / resistances))
    inductance = float(1.0 / np.sum(1.0 / inductances))
    return {"resistance_ohm": resistance, "inductance_h": inductance}


def _affine_step(
    state: np.ndarray,
    matrix: np.ndarray,
    constant: np.ndarray,
    linear: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance x'=A x + constant + linear*t exactly over one interval."""
    if dt <= 0.0:
        return np.asarray(state, dtype=float).copy()
    state = np.asarray(state, dtype=float).reshape(-1)
    count = len(state)
    augmented = np.zeros((count + 2, count + 2), dtype=float)
    augmented[:count, :count] = matrix
    augmented[:count, count] = constant
    augmented[:count, count + 1] = linear
    # constant' = 0; local_time' = constant
    augmented[count + 1, count] = 1.0
    vector = np.zeros(count + 2, dtype=float)
    vector[:count] = state
    vector[count] = 1.0
    advanced = expm(augmented * float(dt)) @ vector
    return np.asarray(advanced[:count], dtype=float)


def _ordered_sample_times(times: np.ndarray, sample_times_s: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if sample_times_s is None:
        requested = np.asarray(times, dtype=float).reshape(-1)
    else:
        requested = np.asarray(sample_times_s, dtype=float).reshape(-1)
    if len(requested) < 1 or not np.isfinite(requested).all():
        raise ElectricalResolutionError("Electrical resolution sample times must be finite.")
    start = float(times[0])
    stop = float(times[-1])
    tolerance = max(1e-12, abs(stop - start) * 1e-12)
    if float(np.min(requested)) < start - tolerance or float(np.max(requested)) > stop + tolerance:
        raise ElectricalResolutionError("Electrical resolution sample times must stay inside the waveform duration.")
    clipped = np.clip(requested, start, stop)
    order = np.argsort(clipped, kind="stable")
    return clipped[order], order


def _restore_order(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(values)[inverse]


def _ideal_jump_count(segments: list[tuple[float, float, float, float]]) -> int:
    count = 0
    previous_end = 0.0
    for index, (start, stop, base, slope) in enumerate(segments):
        if index == 0:
            if not math.isclose(base, 0.0, rel_tol=0.0, abs_tol=1e-12):
                count += 1
        elif not math.isclose(base, previous_end, rel_tol=0.0, abs_tol=1e-12):
            count += 1
        previous_end = base + slope * (stop - start)
    return count


def _segment_samples(
    segments: list[tuple[float, float, float, float]],
    sorted_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.asarray([row[0] for row in segments], dtype=float)
    stops = np.asarray([row[1] for row in segments], dtype=float)
    bases = np.asarray([row[2] for row in segments], dtype=float)
    slopes = np.asarray([row[3] for row in segments], dtype=float)
    scale = max(
        1.0,
        float(np.max(np.abs(starts))) if len(starts) else 0.0,
        float(np.max(np.abs(sorted_times))) if len(sorted_times) else 0.0,
    )
    boundary_tolerance = np.finfo(float).eps * scale * 16.0
    indices = np.searchsorted(
        starts, sorted_times + boundary_tolerance, side="right"
    ) - 1
    indices = np.clip(indices, 0, len(segments) - 1)
    dt = np.maximum(0.0, np.minimum(sorted_times, stops[indices]) - starts[indices])
    return bases[indices] + slopes[indices] * dt, slopes[indices], indices


def _cached_transition(
    augmented: np.ndarray,
    dt: float,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Return expm(augmented*dt), caching the repeated physical intervals.

    Exposure sources often repeat the same point delay hundreds of thousands of
    times.  The old shared-source solver recomputed a tiny matrix exponential at
    every waveform knot, which made long multi-branch voltage exposures vastly
    more expensive than the actual field work.  The state-transition matrix
    depends only on the interval length for a fixed load network, so exact
    repeated intervals can safely reuse it.  Keep the cache bounded; uncommon
    residual intervals simply fall back to a one-off exponential.
    """
    interval = float(dt)
    if interval <= 0.0:
        return np.eye(augmented.shape[0], dtype=float)
    key = interval.hex()
    transition = cache.get(key)
    if transition is not None:
        return transition
    transition = expm(augmented * interval)
    if len(cache) < 4096:
        cache[key] = transition
    return transition


def _modal_affine_coefficients(
    eigenvalues: np.ndarray, dt: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exp(A t), constant-input and ramp-input modal factors.

    For one scalar eigenvalue λ and local time t, the exact affine-forcing
    factors are ``exp(λt)``, ``(exp(λt)-1)/λ`` and
    ``(exp(λt)-1-λt)/λ²``.  The series branches avoid cancellation around
    λt=0 and also cover zero-resistance modes exactly.
    """
    lam = np.asarray(eigenvalues, dtype=float).reshape(-1)
    times = np.asarray(dt, dtype=float)
    scalar = times.ndim == 0
    flat = times.reshape(-1, 1)
    z = flat * lam.reshape(1, -1)
    exponential = np.exp(z)
    phi1 = np.empty_like(z)
    phi2 = np.empty_like(z)
    small = np.abs(z) < 1e-6
    if np.any(small):
        zs = z[small]
        # t and t² are broadcast through the corresponding modal columns below.
        row_index, col_index = np.nonzero(small)
        t_small = flat[row_index, 0]
        phi1[small] = t_small * (
            1.0 + zs / 2.0 + zs * zs / 6.0 + zs**3 / 24.0 + zs**4 / 120.0
        )
        phi2[small] = t_small * t_small * (
            0.5 + zs / 6.0 + zs * zs / 24.0 + zs**3 / 120.0 + zs**4 / 720.0
        )
    if np.any(~small):
        row_index, col_index = np.nonzero(~small)
        z_large = z[~small]
        lam_large = lam[col_index]
        phi1[~small] = np.expm1(z_large) / lam_large
        phi2[~small] = (np.expm1(z_large) - z_large) / (lam_large * lam_large)
    if scalar:
        return exponential[0], phi1[0], phi2[0]
    return exponential, phi1, phi2


def _resolve_parallel_voltage(
    segments: list[tuple[float, float, float, float]],
    sample_times: np.ndarray,
    resistances: np.ndarray,
    inductances: np.ndarray,
    source_resistance: float,
    initial_currents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(resistances)

    # A = -L^-1 R_network is similar to the real symmetric matrix
    # -L^-1/2 R_network L^-1/2.  Solving in that orthonormal modal basis avoids
    # a general matrix exponential at every DAC knot while retaining the exact
    # continuous-time RL solution for a piecewise-linear voltage command.
    inv_sqrt_l = 1.0 / np.sqrt(inductances)
    sqrt_l = np.sqrt(inductances)
    resistance_matrix = np.diag(resistances) + source_resistance * np.ones(
        (count, count), dtype=float
    )
    symmetric = -(inv_sqrt_l[:, None] * resistance_matrix) * inv_sqrt_l[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    source_modal = eigenvectors.T @ inv_sqrt_l

    def evaluate_from_state(
        state: np.ndarray, base: float, slope: float, local_times: Any
    ) -> np.ndarray:
        requested = np.asarray(local_times, dtype=float)
        scalar = requested.ndim == 0
        times = requested.reshape(-1)
        exponential, phi1, phi2 = _modal_affine_coefficients(eigenvalues, times)
        modal_state = eigenvectors.T @ (sqrt_l * np.asarray(state, dtype=float))
        modal = (
            exponential * modal_state.reshape(1, -1)
            + phi1 * (source_modal * float(base)).reshape(1, -1)
            + phi2 * (source_modal * float(slope)).reshape(1, -1)
        )
        values = (modal @ eigenvectors.T) * inv_sqrt_l.reshape(1, -1)
        return values[0] if scalar else values

    currents = np.zeros((len(sample_times), count), dtype=float)
    state = np.asarray(initial_currents, dtype=float).reshape(count).copy()
    cursor = 0
    for segment_index, (start, stop, base, slope) in enumerate(segments):
        is_last = segment_index == len(segments) - 1
        end_side = "right" if is_last else "left"
        end_index = int(np.searchsorted(sample_times, stop, side=end_side))
        segment_times = sample_times[cursor:end_index]
        if len(segment_times):
            local = np.maximum(0.0, segment_times - float(start))
            currents[cursor:end_index] = evaluate_from_state(
                state, float(base), float(slope), local
            )
        state = evaluate_from_state(
            state, float(base), float(slope), max(0.0, float(stop - start))
        )
        cursor = end_index

    command, _slopes, _indices = _segment_samples(segments, sample_times)
    total_current = np.sum(currents, axis=1)
    terminal_voltage = command - source_resistance * total_current
    return currents, command, terminal_voltage


def _resolve_parallel_current(
    segments: list[tuple[float, float, float, float]],
    sample_times: np.ndarray,
    resistances: np.ndarray,
    inductances: np.ndarray,
    initial_currents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    count = len(resistances)
    command, slopes_at_samples, _indices = _segment_samples(segments, sample_times)
    if count == 1:
        currents = command.reshape(-1, 1)
        voltage = resistances[0] * command + inductances[0] * slopes_at_samples
        jumps = _ideal_jump_count(segments)
        return currents, command, voltage, jumps

    reduced_count = count - 1
    l_last = float(inductances[-1])
    r_last = float(resistances[-1])
    lhs = np.diag(inductances[:-1]) + l_last * np.ones((reduced_count, reduced_count), dtype=float)
    resistance_matrix = np.diag(resistances[:-1]) + r_last * np.ones(
        (reduced_count, reduced_count), dtype=float
    )
    matrix = -np.linalg.solve(lhs, resistance_matrix)
    q = np.linalg.solve(lhs, np.full(reduced_count, r_last, dtype=float))
    r = np.linalg.solve(lhs, np.full(reduced_count, l_last, dtype=float))

    # For the reduced branch state, x' = A x + q*u + r*slope.  Augment the
    # physical state with source current u and its piecewise-constant slope so
    # repeated waveform point delays can reuse the same exact transition matrix.
    augmented = np.zeros((reduced_count + 2, reduced_count + 2), dtype=float)
    augmented[:reduced_count, :reduced_count] = matrix
    augmented[:reduced_count, reduced_count] = q
    augmented[:reduced_count, reduced_count + 1] = r
    augmented[reduced_count, reduced_count + 1] = 1.0
    transition_cache: dict[str, np.ndarray] = {}

    currents = np.zeros((len(sample_times), count), dtype=float)
    state = np.asarray(initial_currents, dtype=float).reshape(count).copy()
    cursor = 0
    jump_count = 0
    inverse_l = 1.0 / inductances
    jump_weights = inverse_l / float(np.sum(inverse_l))

    for segment_index, (start, stop, base, slope) in enumerate(segments):
        # An ideal current-source command may jump between source segments. A
        # common voltage impulse divides that jump inversely with branch L.
        total_before = float(np.sum(state))
        delta = float(base - total_before)
        if abs(delta) > max(1e-12, abs(base) * 1e-12):
            state = state + jump_weights * delta
            jump_count += 1

        vector = np.zeros(reduced_count + 2, dtype=float)
        vector[:reduced_count] = state[:-1]
        vector[reduced_count] = float(base)
        vector[reduced_count + 1] = float(slope)
        current_time = float(start)
        is_last = segment_index == len(segments) - 1
        end_side = "right" if is_last else "left"
        end_index = int(np.searchsorted(sample_times, stop, side=end_side))
        segment_times = sample_times[cursor:end_index]
        for offset, time_value in enumerate(segment_times):
            dt = max(0.0, float(time_value) - current_time)
            if dt > 0.0:
                vector = _cached_transition(augmented, dt, transition_cache) @ vector
            reduced_at_time = vector[:reduced_count]
            total_at_time = float(vector[reduced_count])
            full = np.empty(count, dtype=float)
            full[:-1] = reduced_at_time
            full[-1] = total_at_time - float(np.sum(reduced_at_time))
            currents[cursor + offset] = full
            current_time = float(time_value)

        tail = max(0.0, float(stop) - current_time)
        if tail > 0.0:
            vector = _cached_transition(augmented, tail, transition_cache) @ vector
        reduced_end = vector[:reduced_count]
        total_end = float(vector[reduced_count])
        state[:-1] = reduced_end
        state[-1] = total_end - float(np.sum(reduced_end))
        cursor = end_index

    # Common branch voltage from the imposed total-current derivative.
    numerator = slopes_at_samples + np.sum(
        currents * (resistances / inductances).reshape(1, -1), axis=1
    )
    voltage = numerator / float(np.sum(1.0 / inductances))
    return currents, command, voltage, jump_count


def resolve_drive_group(
    time_s: Any,
    value: Any,
    source_kind: str,
    *,
    loads: list[dict[str, Any]],
    connection: str,
    slew_rate_per_s: float = 0.0,
    output_resistance_ohm: float = 0.0,
    sample_times_s: Any | None = None,
    initial_currents_a: dict[str, float] | None = None,
    source_segments: list[tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    """Resolve one waveform source into one current trace per connected coil.

    Current sources impose their commanded total/series current. Voltage sources
    include their saved series output resistance. Series inductances add;
    parallel branches are solved as a coupled uncoupled-RL network. Mutual
    inductance is deliberately outside this phase.
    """

    times, values, source_kind = _validated_trace(time_s, value, source_kind)
    normalized_loads = normalize_electrical_loads(loads)
    mode = str(connection).strip().lower()
    if mode not in {"series", "parallel"}:
        raise ElectricalResolutionError("Drive-group connection must be series or parallel.")
    source_resistance = _finite_nonnegative(output_resistance_ohm, "Source output resistance")
    if source_kind == "current":
        source_resistance = 0.0
    try:
        slew = float(slew_rate_per_s)
    except (TypeError, ValueError) as error:
        raise ElectricalResolutionError("Source slew rate must be numeric.") from error
    if not math.isfinite(slew) or slew < 0.0:
        raise ElectricalResolutionError("Source slew rate must be finite and non-negative.")

    segments = _source_segments(times, values, slew) if source_segments is None else source_segments
    if not segments:
        raise ElectricalResolutionError("The waveform source does not contain a usable duration.")
    sorted_times, order = _ordered_sample_times(times, sample_times_s)
    resistances = np.asarray([row["resistance_ohm"] for row in normalized_loads], dtype=float)
    inductances = np.asarray([row["inductance_h"] for row in normalized_loads], dtype=float)
    initial_map = initial_currents_a if isinstance(initial_currents_a, dict) else {}
    try:
        initial_currents = np.asarray(
            [float(initial_map.get(row["coil_id"], 0.0)) for row in normalized_loads],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ElectricalResolutionError("Initial coil currents must be numeric.") from error
    if not np.isfinite(initial_currents).all():
        raise ElectricalResolutionError("Initial coil currents must be finite.")
    equivalent = equivalent_load(normalized_loads, mode)
    command, command_slope, _segment_indices = _segment_samples(segments, sorted_times)
    ideal_current_steps = 0

    if mode == "series":
        if source_kind == "current":
            branch_current = command
            coil_currents = np.repeat(branch_current.reshape(-1, 1), len(normalized_loads), axis=1)
            terminal_voltage = equivalent["resistance_ohm"] * branch_current + equivalent["inductance_h"] * command_slope
            source_current = branch_current
            source_voltage = terminal_voltage
            ideal_current_steps = _ideal_jump_count(segments)
        else:
            total_r = source_resistance + equivalent["resistance_ohm"]
            total_l = equivalent["inductance_h"]
            # Solve the series RL source exactly on the same linear command segments.
            if len(initial_currents) > 1 and not np.allclose(
                initial_currents, initial_currents[0], rtol=1e-9, atol=1e-12
            ):
                raise ElectricalResolutionError(
                    "Series-connected coils must begin with the same current."
                )
            state = float(initial_currents[0])
            branch_current = np.zeros(len(sorted_times), dtype=float)
            cursor = 0
            for segment_index, (start, stop, base, slope) in enumerate(segments):
                is_last = segment_index == len(segments) - 1
                end_side = "right" if is_last else "left"
                end_index = int(np.searchsorted(sorted_times, stop, side=end_side))
                segment_times = sorted_times[cursor:end_index]
                if total_r <= 1e-15:
                    for offset, time_value in enumerate(segment_times):
                        dt = float(time_value - start)
                        branch_current[cursor + offset] = state + (base * dt + 0.5 * slope * dt * dt) / total_l
                    duration = float(stop - start)
                    state = state + (base * duration + 0.5 * slope * duration * duration) / total_l
                else:
                    decay_rate = total_r / total_l
                    for offset, time_value in enumerate(segment_times):
                        dt = float(time_value - start)
                        decay = math.exp(-decay_rate * dt)
                        one_minus = -math.expm1(-decay_rate * dt)
                        branch_current[cursor + offset] = (
                            state * decay
                            + (base / total_r) * one_minus
                            + (slope / total_r) * (dt - one_minus / decay_rate)
                        )
                    duration = float(stop - start)
                    decay = math.exp(-decay_rate * duration)
                    one_minus = -math.expm1(-decay_rate * duration)
                    state = (
                        state * decay
                        + (base / total_r) * one_minus
                        + (slope / total_r) * (duration - one_minus / decay_rate)
                    )
                cursor = end_index
            coil_currents = np.repeat(branch_current.reshape(-1, 1), len(normalized_loads), axis=1)
            source_current = branch_current
            source_voltage = command
            terminal_voltage = command - source_resistance * branch_current
    else:
        if source_kind == "voltage":
            coil_currents, source_voltage, terminal_voltage = _resolve_parallel_voltage(
                segments, sorted_times, resistances, inductances, source_resistance, initial_currents
            )
            source_current = np.sum(coil_currents, axis=1)
        else:
            coil_currents, source_current, terminal_voltage, ideal_current_steps = _resolve_parallel_current(
                segments, sorted_times, resistances, inductances, initial_currents
            )
            source_voltage = terminal_voltage

    if mode == "series":
        # Voltage division over the uncoupled series load. The same dI/dt flows
        # through every winding.
        if source_kind == "voltage":
            di_dt = (terminal_voltage - equivalent["resistance_ohm"] * source_current) / equivalent["inductance_h"]
        else:
            di_dt = command_slope
        coil_voltages = source_current.reshape(-1, 1) * resistances.reshape(1, -1) + di_dt.reshape(-1, 1) * inductances.reshape(1, -1)
    else:
        coil_voltages = np.repeat(terminal_voltage.reshape(-1, 1), len(normalized_loads), axis=1)

    coil_current_map = {
        row["coil_id"]: _restore_order(coil_currents[:, index], order)
        for index, row in enumerate(normalized_loads)
    }
    coil_voltage_map = {
        row["coil_id"]: _restore_order(coil_voltages[:, index], order)
        for index, row in enumerate(normalized_loads)
    }
    return {
        "time_s": _restore_order(sorted_times, order),
        "source_kind": source_kind,
        "connection": mode,
        "source_command_si": _restore_order(command, order),
        "source_current_a": _restore_order(source_current, order),
        "source_voltage_v": _restore_order(source_voltage, order),
        "load_terminal_voltage_v": _restore_order(terminal_voltage, order),
        "coil_currents_a": coil_current_map,
        "coil_voltages_v": coil_voltage_map,
        "loads": normalized_loads,
        "equivalent_resistance_ohm": float(equivalent["resistance_ohm"]),
        "equivalent_inductance_h": float(equivalent["inductance_h"]),
        "source_output_resistance_ohm": float(source_resistance),
        "mutual_inductance_model": "uncoupled",
        "ideal_current_step_count": int(ideal_current_steps),
    }


def resolve_shared_source_groups(
    time_s: Any,
    value: Any,
    source_kind: str,
    *,
    groups: list[dict[str, Any]],
    slew_rate_per_s: float = 0.0,
    output_resistance_ohm: float = 0.0,
    sample_times_s: Any | None = None,
    initial_currents_a: dict[str, float] | None = None,
    source_segments: list[tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    """Resolve several drive groups connected to one physical source.

    Groups sharing a source are parallel routes at the source terminals.  A
    group's own ``connection`` still determines how the coils *inside that
    branch* are wired.  Thus three Series groups are three series strings in
    parallel, which matches common multi-channel coil fixtures.  A voltage
    source's output resistance is shared once by the complete network.

    Mutual inductance remains deliberately outside the electrical model.
    """
    if not isinstance(groups, list) or not groups:
        raise ElectricalResolutionError("A shared source needs at least one drive group.")

    times, values, source_kind = _validated_trace(time_s, value, source_kind)
    source_resistance = _finite_nonnegative(output_resistance_ohm, "Source output resistance")
    if source_kind == "current":
        source_resistance = 0.0
    try:
        slew = float(slew_rate_per_s)
    except (TypeError, ValueError) as error:
        raise ElectricalResolutionError("Source slew rate must be numeric.") from error
    if not math.isfinite(slew) or slew < 0.0:
        raise ElectricalResolutionError("Source slew rate must be finite and non-negative.")

    segments = _source_segments(times, values, slew) if source_segments is None else source_segments
    if not segments:
        raise ElectricalResolutionError("The waveform source does not contain a usable duration.")
    sorted_times, order = _ordered_sample_times(times, sample_times_s)
    initial_map = initial_currents_a if isinstance(initial_currents_a, dict) else {}

    normalized_groups: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    seen_coils: set[str] = set()
    for group_index, raw_group in enumerate(groups):
        if not isinstance(raw_group, dict):
            raise ElectricalResolutionError(f"Drive group {group_index + 1} must be an object.")
        connection = str(raw_group.get("connection", "series") or "series").strip().lower()
        if connection not in {"series", "parallel"}:
            raise ElectricalResolutionError("Drive-group connection must be series or parallel.")
        loads = normalize_electrical_loads(raw_group.get("loads", []))
        overlap = [row["coil_id"] for row in loads if row["coil_id"] in seen_coils]
        if overlap:
            raise ElectricalResolutionError(
                f"Coil {overlap[0]!r} appears in more than one drive group on the same source."
            )
        seen_coils.update(row["coil_id"] for row in loads)
        group_equivalent = equivalent_load(loads, connection)
        normalized_groups.append({
            "drive_group_id": str(raw_group.get("id", "") or ""),
            "connection": connection,
            "loads": loads,
            "equivalent_resistance_ohm": float(group_equivalent["resistance_ohm"]),
            "equivalent_inductance_h": float(group_equivalent["inductance_h"]),
        })
        if connection == "series":
            initial_values = np.asarray(
                [float(initial_map.get(row["coil_id"], 0.0)) for row in loads], dtype=float
            )
            if not np.isfinite(initial_values).all():
                raise ElectricalResolutionError("Initial coil currents must be finite.")
            if len(initial_values) > 1 and not np.allclose(
                initial_values, initial_values[0], rtol=1e-9, atol=1e-12
            ):
                raise ElectricalResolutionError(
                    "Series-connected coils must begin with the same current."
                )
            branches.append({
                "group_index": group_index,
                "kind": "series",
                "loads": loads,
                "resistance_ohm": float(group_equivalent["resistance_ohm"]),
                "inductance_h": float(group_equivalent["inductance_h"]),
                "initial_current_a": float(initial_values[0]),
            })
        else:
            for load in loads:
                initial = float(initial_map.get(load["coil_id"], 0.0))
                if not math.isfinite(initial):
                    raise ElectricalResolutionError("Initial coil currents must be finite.")
                branches.append({
                    "group_index": group_index,
                    "kind": "parallel_coil",
                    "loads": [load],
                    "resistance_ohm": float(load["resistance_ohm"]),
                    "inductance_h": float(load["inductance_h"]),
                    "initial_current_a": initial,
                })

    resistances = np.asarray([row["resistance_ohm"] for row in branches], dtype=float)
    inductances = np.asarray([row["inductance_h"] for row in branches], dtype=float)
    initial_branch_currents = np.asarray([row["initial_current_a"] for row in branches], dtype=float)
    command, _command_slope, _segment_indices = _segment_samples(segments, sorted_times)
    ideal_current_steps = 0
    if source_kind == "voltage":
        branch_currents, source_voltage, terminal_voltage = _resolve_parallel_voltage(
            segments,
            sorted_times,
            resistances,
            inductances,
            source_resistance,
            initial_branch_currents,
        )
        source_current = np.sum(branch_currents, axis=1)
    else:
        branch_currents, source_current, terminal_voltage, ideal_current_steps = _resolve_parallel_current(
            segments,
            sorted_times,
            resistances,
            inductances,
            initial_branch_currents,
        )
        source_voltage = terminal_voltage

    coil_currents: dict[str, np.ndarray] = {}
    coil_voltages: dict[str, np.ndarray] = {}
    branch_index = 0
    for group in normalized_groups:
        connection = group["connection"]
        loads = group["loads"]
        if connection == "series":
            current = branch_currents[:, branch_index]
            req = float(group["equivalent_resistance_ohm"])
            leq = float(group["equivalent_inductance_h"])
            di_dt = (terminal_voltage - req * current) / leq
            for load in loads:
                coil_currents[load["coil_id"]] = current
                coil_voltages[load["coil_id"]] = (
                    float(load["resistance_ohm"]) * current
                    + float(load["inductance_h"]) * di_dt
                )
            branch_index += 1
        else:
            for load in loads:
                current = branch_currents[:, branch_index]
                coil_currents[load["coil_id"]] = current
                coil_voltages[load["coil_id"]] = terminal_voltage
                branch_index += 1

    overall_resistance = 0.0 if np.any(resistances <= 0.0) else float(1.0 / np.sum(1.0 / resistances))
    overall_inductance = float(1.0 / np.sum(1.0 / inductances))
    return {
        "time_s": _restore_order(sorted_times, order),
        "source_kind": source_kind,
        "source_command_si": _restore_order(command, order),
        "source_current_a": _restore_order(source_current, order),
        "source_voltage_v": _restore_order(source_voltage, order),
        "load_terminal_voltage_v": _restore_order(terminal_voltage, order),
        "coil_currents_a": {
            coil_id: _restore_order(values, order) for coil_id, values in coil_currents.items()
        },
        "coil_voltages_v": {
            coil_id: _restore_order(values, order) for coil_id, values in coil_voltages.items()
        },
        "drive_groups": normalized_groups,
        "equivalent_resistance_ohm": overall_resistance,
        "equivalent_inductance_h": overall_inductance,
        "source_output_resistance_ohm": float(source_resistance),
        "mutual_inductance_model": "uncoupled",
        "ideal_current_step_count": int(ideal_current_steps),
    }
