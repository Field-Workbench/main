"""Rigol DG-series RAF and CSV serialization.

The legacy implementation mixed dialogs, generator-limit planning, slew
simulation, CRC construction, and file writes in one callback. This module keeps
the byte-level behavior testable without importing a GUI toolkit or Matplotlib.
"""

from __future__ import annotations

import binascii
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .encoding import validated_points, validate_bit_depth


RAF_MAX_SAMPLES = 16_000_000
RAF_MAX_SAMPLE_RATE_HZ = 50_000_000.0
CSV_MAX_SAMPLE_RATE_HZ = 60_000_000.0


class RigolExportError(ValueError):
    """Raised when an export cannot be represented safely."""


@dataclass(frozen=True)
class RafSettings:
    point_delay_ms: float = 1.0
    cycle_delay_ms: float = 0.0
    rest_value: int = 128
    bit_depth: int = 8
    slew_rate_v_per_us: float = 1.0
    max_samples: int = RAF_MAX_SAMPLES
    max_sample_rate_hz: float = RAF_MAX_SAMPLE_RATE_HZ


@dataclass(frozen=True)
class RafBuild:
    header: bytes
    payload: bytes
    total_points: int
    sample_rate_hz: float
    stretch_factor: int
    cycle_delay_points: int
    warnings: tuple[str, ...] = ()

    @property
    def file_bytes(self) -> bytes:
        return self.header + self.payload


@dataclass(frozen=True)
class CsvSettings:
    point_delay_ms: float = 1.0
    cycle_delay_ms: float = 0.0
    rest_value: int = 128
    bit_depth: int = 8
    slew_rate_v_per_us: float = 1.0
    target_points: int = 8_000_000
    max_sample_rate_hz: float = CSV_MAX_SAMPLE_RATE_HZ


@dataclass(frozen=True)
class CsvPlan:
    period_seconds: float
    sample_rate_hz: float
    stretch_factor: int
    cycle_delay_points: int
    total_points: int


def _validate_common(
    points: Sequence[int],
    *,
    point_delay_ms: float,
    cycle_delay_ms: float,
    rest_value: int,
    slew_rate_v_per_us: float,
    bit_depth: int,
) -> list[int]:
    depth = validate_bit_depth(bit_depth)
    checked = validated_points(points, bit_depth=depth)
    if not checked:
        raise RigolExportError("A Rigol export needs at least one waveform point.")
    if not 0.00002 <= float(point_delay_ms) <= 500.0:
        raise RigolExportError("Point delay must be between 0.00002 and 500 ms.")
    if not 0.0 <= float(cycle_delay_ms) <= 500.0:
        raise RigolExportError("Cycle delay must be between 0 and 500 ms.")
    validated_points([rest_value], bit_depth=depth)
    if float(slew_rate_v_per_us) <= 0.0:
        raise RigolExportError("Slew rate must be greater than zero.")
    return checked


def map_8bit_to_i16(value: int) -> int:
    point = validated_points([value], bit_depth=8)[0]
    return (point * 65_535 // 255) - 32_768


def map_16bit_to_i16(value: int) -> int:
    """Map native unsigned 16-bit samples directly into Rigol signed samples."""

    point = validated_points([value], bit_depth=16)[0]
    return point - 32_768


def map_point_to_i16(value: int, *, bit_depth: int) -> int:
    depth = validate_bit_depth(bit_depth)
    return map_8bit_to_i16(value) if depth == 8 else map_16bit_to_i16(value)


def _append_i16(target: bytearray, value: int) -> None:
    target.extend(int(value).to_bytes(2, byteorder="little", signed=True))


def _binary_transition(
    target: bytearray,
    start: int,
    stop: int,
    *,
    period_seconds: float,
    slew_rate_v_per_us: float,
    max_transition_seconds: float,
) -> tuple[int, int]:
    delta_counts = stop - start
    volts_per_count = 10.0 / 65_535.0
    delta_volts = abs(delta_counts) * volts_per_count
    transit_seconds = delta_volts / (slew_rate_v_per_us * 1_000_000.0)
    transit_seconds = min(transit_seconds, max_transition_seconds)
    count = max(0, math.ceil(transit_seconds / period_seconds) - 1)

    last_value = start
    for index in range(1, count + 1):
        last_value = int(round(start + delta_counts * (index / (count + 1))))
        _append_i16(target, last_value)
    return count, last_value


def build_raf(
    points: Sequence[int],
    *,
    file_name: str,
    settings: RafSettings,
) -> RafBuild:
    checked = _validate_common(
        points,
        point_delay_ms=settings.point_delay_ms,
        cycle_delay_ms=settings.cycle_delay_ms,
        rest_value=settings.rest_value,
        slew_rate_v_per_us=settings.slew_rate_v_per_us,
        bit_depth=settings.bit_depth,
    )
    if settings.max_samples < 1 or settings.max_sample_rate_hz <= 0:
        raise RigolExportError("Generator sample limits must be positive.")

    point_period = settings.point_delay_ms / 1000.0
    cycle_delay = settings.cycle_delay_ms / 1000.0
    total_cycle_time = point_period * len(checked) + cycle_delay
    sample_rate_limit = min(
        settings.max_sample_rate_hz,
        settings.max_samples / total_cycle_time,
    )
    stretch_factor = max(1, int(sample_rate_limit * point_period))
    sample_rate = stretch_factor / point_period
    period = 1.0 / sample_rate
    delay_point_count = int(round(cycle_delay / period))
    nominal_total = len(checked) * stretch_factor + delay_point_count

    while nominal_total > settings.max_samples and stretch_factor > 1:
        stretch_factor -= 1
        sample_rate = stretch_factor / point_period
        period = 1.0 / sample_rate
        delay_point_count = int(round(cycle_delay / period))
        nominal_total = len(checked) * stretch_factor + delay_point_count

    payload = bytearray()
    total_points = 0
    previous = (
        map_point_to_i16(settings.rest_value, bit_depth=settings.bit_depth)
        if cycle_delay > 0
        else map_point_to_i16(checked[-1], bit_depth=settings.bit_depth)
    )

    for point in checked:
        voltage = map_point_to_i16(point, bit_depth=settings.bit_depth)
        transition_count, last_value = _binary_transition(
            payload,
            previous,
            voltage,
            period_seconds=period,
            slew_rate_v_per_us=settings.slew_rate_v_per_us,
            max_transition_seconds=point_period,
        )
        total_points += transition_count
        hold_count = max(0, stretch_factor - transition_count)
        held_voltage = voltage if hold_count else last_value
        for _ in range(hold_count):
            _append_i16(payload, held_voltage)
        total_points += hold_count
        previous = held_voltage

    if cycle_delay > 0:
        rest_voltage = map_point_to_i16(settings.rest_value, bit_depth=settings.bit_depth)
        transition_count, last_value = _binary_transition(
            payload,
            previous,
            rest_voltage,
            period_seconds=period,
            slew_rate_v_per_us=settings.slew_rate_v_per_us,
            max_transition_seconds=cycle_delay,
        )
        total_points += transition_count
        delay_point_count = int(round(cycle_delay / period))
        hold_count = max(0, delay_point_count - transition_count)
        held_voltage = rest_voltage if hold_count else last_value
        for _ in range(hold_count):
            _append_i16(payload, held_voltage)
        total_points += hold_count

    warnings: list[str] = []
    if total_points > settings.max_samples:
        warnings.append(
            f"Total samples {total_points:,} exceed the configured limit "
            f"of {settings.max_samples:,}."
        )
    if sample_rate > settings.max_sample_rate_hz:
        warnings.append(
            f"Sample rate {sample_rate:,.0f} Sa/s exceeds the configured limit "
            f"of {settings.max_sample_rate_hz:,.0f} Sa/s."
        )

    header = bytearray()
    header.extend(total_points.to_bytes(4, byteorder="little", signed=True))
    header.extend(b"\x01\x00\x03")

    encoded_name = Path(file_name).name.encode("utf-8")[:25]
    header.extend(encoded_name.ljust(25, b"\x00"))
    # Rigol's RAF field stores the legacy micro-Sa/s scaled integer.
    header.extend(int(sample_rate * 1_000_000).to_bytes(8, byteorder="little", signed=True))
    header.extend((50_000_000).to_bytes(4, byteorder="little", signed=True))
    header.extend((-50_000_000).to_bytes(4, byteorder="little", signed=True))
    header.extend(binascii.crc_hqx(payload, 0xEBCC).to_bytes(2, "little", signed=False))
    header.extend(binascii.crc_hqx(header, 0xEBCC).to_bytes(2, "little", signed=False))
    header.extend(b"\x00\x00\x00\x00")

    return RafBuild(
        bytes(header),
        bytes(payload),
        total_points,
        sample_rate,
        stretch_factor,
        delay_point_count,
        tuple(warnings),
    )


def write_raf(path: str | Path, points: Sequence[int], settings: RafSettings) -> RafBuild:
    destination = Path(path)
    build = build_raf(points, file_name=destination.name, settings=settings)
    destination.write_bytes(build.file_bytes)
    return build


def plan_csv(points: Sequence[int], settings: CsvSettings) -> CsvPlan:
    checked = _validate_common(
        points,
        point_delay_ms=settings.point_delay_ms,
        cycle_delay_ms=settings.cycle_delay_ms,
        rest_value=settings.rest_value,
        slew_rate_v_per_us=settings.slew_rate_v_per_us,
        bit_depth=settings.bit_depth,
    )
    if settings.target_points < 1 or settings.max_sample_rate_hz <= 0:
        raise RigolExportError("CSV target points and sample-rate limit must be positive.")

    point_period = settings.point_delay_ms / 1000.0
    cycle_delay = settings.cycle_delay_ms / 1000.0
    total_cycle_time = point_period * len(checked) + cycle_delay
    desired_rate = min(settings.max_sample_rate_hz, settings.target_points / total_cycle_time)
    stretch_factor = max(1, int(desired_rate * point_period))
    sample_rate = stretch_factor / point_period
    period = 1.0 / sample_rate
    delay_points = int(round(cycle_delay / period))
    total_points = len(checked) * stretch_factor + delay_points
    return CsvPlan(period, sample_rate, stretch_factor, delay_points, total_points)


def _csv_voltage(point: int, *, bit_depth: int) -> float:
    depth = validate_bit_depth(bit_depth)
    if depth == 8:
        # Preserve the legacy 8-bit WaveBuilder/Rigol conversion exactly.
        return -4.98 + validated_points([point], bit_depth=8)[0] * 0.0390625

    # Native 16-bit pathway: one unsigned source count maps directly into the
    # generator's full 16-bit span before conversion to the CSV voltage scale.
    checked = validated_points([point], bit_depth=16)[0]
    return -5.0 + checked * (10.0 / 65_536.0)


def _iter_slewed_segment(
    start: float,
    stop: float,
    *,
    sample_count: int,
    period_seconds: float,
    slew_rate_v_per_us: float,
) -> Iterator[float]:
    if sample_count <= 0:
        return
    delta = stop - start
    transit_seconds = abs(delta) / (slew_rate_v_per_us * 1_000_000.0)
    transition_count = min(
        max(0, sample_count - 1),
        max(0, int(transit_seconds // period_seconds)),
    )
    for index in range(1, transition_count + 1):
        yield start + delta * (index / (transition_count + 1))
    for _ in range(sample_count - transition_count):
        yield stop


def iter_csv_samples(
    points: Sequence[int],
    settings: CsvSettings,
    plan: CsvPlan | None = None,
) -> Iterator[float]:
    checked = _validate_common(
        points,
        point_delay_ms=settings.point_delay_ms,
        cycle_delay_ms=settings.cycle_delay_ms,
        rest_value=settings.rest_value,
        slew_rate_v_per_us=settings.slew_rate_v_per_us,
        bit_depth=settings.bit_depth,
    )
    actual_plan = plan or plan_csv(checked, settings)
    previous = (
        _csv_voltage(settings.rest_value, bit_depth=settings.bit_depth)
        if actual_plan.cycle_delay_points
        else _csv_voltage(checked[-1], bit_depth=settings.bit_depth)
    )
    for point in checked:
        voltage = _csv_voltage(point, bit_depth=settings.bit_depth)
        yield from _iter_slewed_segment(
            previous,
            voltage,
            sample_count=actual_plan.stretch_factor,
            period_seconds=actual_plan.period_seconds,
            slew_rate_v_per_us=settings.slew_rate_v_per_us,
        )
        previous = voltage

    if actual_plan.cycle_delay_points:
        rest = _csv_voltage(settings.rest_value, bit_depth=settings.bit_depth)
        yield from _iter_slewed_segment(
            previous,
            rest,
            sample_count=actual_plan.cycle_delay_points,
            period_seconds=actual_plan.period_seconds,
            slew_rate_v_per_us=settings.slew_rate_v_per_us,
        )


def csv_header(plan: CsvPlan) -> str:
    return (
        "RIGOL:DG2:CSV DATA FILE,\n"
        "TYPE:Arb,\n"
        "AMP:10.0000 Vpp,\n"
        f"PERIOD:{plan.period_seconds * 1_000_000:.9f} uS,\n"
        f"DOTS:{plan.total_points},\n"
        "MODE:INSERT,\n"
        f"Sample Rate:{plan.sample_rate_hz:.6f},\n"
        "AWG N:0,\n"
        "x,y[V],\n"
    )


def build_csv_text(points: Sequence[int], settings: CsvSettings) -> tuple[str, CsvPlan]:
    plan = plan_csv(points, settings)
    body = "".join(f",{value}\n" for value in iter_csv_samples(points, settings, plan))
    return csv_header(plan) + body, plan


def write_csv(path: str | Path, points: Sequence[int], settings: CsvSettings) -> CsvPlan:
    plan = plan_csv(points, settings)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(csv_header(plan))
        for value in iter_csv_samples(points, settings, plan):
            stream.write(f",{value}\n")
    return plan
