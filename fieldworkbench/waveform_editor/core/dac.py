"""Read and write WaveBuilder's line-oriented DAC files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .encoding import DAC_MAX, DAC_MIN, validated_points


class DacFormatError(ValueError):
    """Raised when a DAC file does not contain parseable numeric points."""


@dataclass(frozen=True)
class DacReadResult:
    points: tuple[int, ...]
    header_removed: bool
    warnings: tuple[str, ...] = ()


def parse_dac_text(text: str, *, source: str = "DAC data") -> DacReadResult:
    values: list[int] = []
    rounded = False
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            number = float(stripped)
        except ValueError as error:
            raise DacFormatError(
                f"{source}: line {line_number} is not numeric: {stripped!r}"
            ) from error
        if not math.isfinite(number):
            raise DacFormatError(
                f"{source}: line {line_number} is not a finite number: {stripped!r}"
            )
        integer = int(round(number, 0))
        rounded = rounded or number != integer
        values.append(integer)

    if not values:
        raise DacFormatError(f"{source}: the file contains no points.")

    warnings: list[str] = []
    if rounded:
        warnings.append("Non-integer DAC values were rounded.")

    header = values[0]
    header_removed = header in (len(values), len(values) - 1)
    if header_removed:
        points = values[1:]
    else:
        points = values
        warnings.append(
            f"Irregular DAC header: {header} does not match the {len(values) - 1} following values; "
            "the first value was retained."
        )

    clipped = False
    for index, point in enumerate(points):
        if point < DAC_MIN:
            points[index] = DAC_MIN
            clipped = True
        elif point > DAC_MAX:
            points[index] = DAC_MAX
            clipped = True
    if clipped:
        warnings.append("Out-of-range DAC values were clipped to 0..255.")

    return DacReadResult(tuple(validated_points(points)), header_removed, tuple(warnings))


def read_dac(path: str | Path) -> DacReadResult:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise DacFormatError(f"{source}: the file is not UTF-8 text.") from error
    return parse_dac_text(text, source=str(source))


def format_dac(points: tuple[int, ...] | list[int]) -> str:
    checked = validated_points(points)
    lines = [str(len(checked)), *(str(point) for point in checked)]
    return "\n".join(lines) + "\n"


def write_dac(path: str | Path, points: tuple[int, ...] | list[int]) -> None:
    Path(path).write_text(format_dac(points), encoding="utf-8", newline="\n")
