"""Integer-code helpers used only at external waveform format boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real


DAC_MIN = 0
DAC_MAX = 255
SUPPORTED_EXPORT_BIT_DEPTHS = (8, 16)


class InvalidCodeError(ValueError):
    """Raised when an external integer-code sequence cannot be represented."""


def validate_bit_depth(bit_depth: int) -> int:
    depth = int(bit_depth)
    if depth not in SUPPORTED_EXPORT_BIT_DEPTHS:
        raise ValueError(f"Unsupported external waveform bit depth: {bit_depth}")
    return depth


def point_max(bit_depth: int = 8) -> int:
    return (1 << validate_bit_depth(bit_depth)) - 1


def validated_points(values: Iterable[Real], *, bit_depth: int = 8) -> list[int]:
    """Validate unsigned external integer codes for one import/export codec."""
    maximum = point_max(bit_depth)
    result: list[int] = []
    for index, value in enumerate(values):
        try:
            integer = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidCodeError(f"Code {index} is not numeric: {value!r}") from error
        if numeric != integer:
            raise InvalidCodeError(f"Code {index} is not an integer: {value!r}")
        if not DAC_MIN <= integer <= maximum:
            raise InvalidCodeError(
                f"Code {index} is outside the {bit_depth}-bit range {DAC_MIN}..{maximum}: {integer}"
            )
        result.append(integer)
    return result


def convert_points_bit_depth(
    values: Iterable[Real], *, source_bit_depth: int, target_bit_depth: int
) -> list[int]:
    """Scale unsigned codes between external 8- and 16-bit formats."""
    source_depth = validate_bit_depth(source_bit_depth)
    target_depth = validate_bit_depth(target_bit_depth)
    checked = validated_points(values, bit_depth=source_depth)
    if source_depth == target_depth:
        return checked
    source_max = point_max(source_depth)
    target_max = point_max(target_depth)
    half = source_max // 2
    return [(point * target_max + half) // source_max for point in checked]
