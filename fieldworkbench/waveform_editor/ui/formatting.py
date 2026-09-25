"""Small presentation helpers shared by embedded waveform controls."""

from __future__ import annotations


def format_time_span(seconds: float) -> str:
    """Format a physical time span for compact waveform-tool readouts."""
    value = float(seconds)
    magnitude = abs(value)
    if magnitude < 1.0e-9:
        return f"{value * 1.0e12:.6g} ps"
    if magnitude < 1.0e-6:
        return f"{value * 1.0e9:.6g} ns"
    if magnitude < 1.0e-3:
        return f"{value * 1.0e6:.6g} µs"
    if magnitude < 1.0:
        return f"{value * 1.0e3:.6g} ms"
    return f"{value:.6g} s"
