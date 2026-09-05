"""Pure helpers shared by the Workbench's numeric input widgets."""

from __future__ import annotations

import math


PLAYBACK_SPEED_MINIMUM = 1.0e-6


def compact_fixed_decimal_text(
    text: str, *, decimal_point: str = ".", minimum_decimals: int = 0
) -> str:
    """Remove insignificant zero padding while keeping a display floor.

    ``QDoubleSpinBox`` uses its configured decimal count both as input
    precision and as a display width. Keeping the precision is useful, but
    displaying every unused place makes ordinary values awkward to edit. This
    helper preserves every significant digit, trims only insignificant trailing
    zeroes, and can retain a requested minimum number of decimal places.
    """

    result = str(text)
    separator = str(decimal_point)
    floor = max(0, int(minimum_decimals))
    if not separator or separator not in result:
        if floor <= 0:
            return result
        return result + separator + ("0" * floor)
    whole, fractional = result.rsplit(separator, 1)
    trimmed = fractional.rstrip("0")
    if len(trimmed) < floor:
        trimmed = fractional[:floor].ljust(floor, "0")
    return whole + separator + trimmed if trimmed else (whole or "0")


def _quantized(value: float, decimals: int) -> float:
    places = max(0, min(15, int(decimals)))
    return float(f"{float(value):.{places}f}")


def playback_speed_ladder(
    *,
    minimum: float = PLAYBACK_SPEED_MINIMUM,
    maximum: float = 1_000_000.0,
    decimals: int = 6,
) -> tuple[float, ...]:
    """Return the ordered playback-speed rungs at and below real time."""

    low = max(0.0, float(minimum))
    high = max(low, float(maximum))
    values = [1.0, 0.75, 0.5, 0.25]
    value = 0.125
    for _ in range(128):
        if value <= low:
            break
        values.append(value)
        value *= 0.5
    values.append(low)
    return tuple(
        sorted(
            {
                _quantized(value, decimals)
                for value in values
                if low <= value <= high
            }
        )
    )


def stepped_playback_speed(
    current: float,
    steps: int,
    *,
    minimum: float = PLAYBACK_SPEED_MINIMUM,
    maximum: float = 1_000_000.0,
    decimals: int = 6,
) -> float:
    """Step a playback multiplier through the ratio ladder.

    Values at or below 1x use ``1, 3/4, 1/2, 1/4, 1/8, ...`` down to the
    configured minimum.  Values above 1x retain the existing quarter-step
    behaviour.  An arbitrary typed value snaps to the next rung or quarter step
    in the requested direction, so the arrows immediately rejoin the pattern.
    """

    low = max(0.0, float(minimum))
    high = max(low, float(maximum))
    places = max(0, min(15, int(decimals)))
    try:
        value = float(current)
    except (TypeError, ValueError):
        value = 1.0
    if not math.isfinite(value):
        value = 1.0
    value = min(high, max(low, _quantized(value, places)))
    remaining = int(steps)
    if remaining == 0:
        return value

    direction = 1 if remaining > 0 else -1
    ladder = playback_speed_ladder(
        minimum=low,
        maximum=high,
        decimals=places,
    )
    quantum = 10.0 ** (-places) if places else 1.0
    epsilon = quantum * 0.25

    for _ in range(abs(remaining)):
        previous = value
        if direction < 0:
            if value <= 1.0 + epsilon:
                candidates = [rung for rung in ladder if rung < value - epsilon]
                value = candidates[-1] if candidates else low
            else:
                quarter_index = math.ceil((value - 1.0) / 0.25 - epsilon) - 1
                value = max(1.0, 1.0 + max(0, quarter_index) * 0.25)
        elif value < 1.0 - epsilon:
            candidates = [rung for rung in ladder if rung > value + epsilon]
            value = candidates[0] if candidates else min(high, 1.0)
        else:
            quarter_index = math.floor((value - 1.0) / 0.25 + epsilon) + 1
            value = 1.0 + max(1, quarter_index) * 0.25

        value = min(high, max(low, _quantized(value, places)))
        if abs(value - previous) <= epsilon:
            break
    return value
