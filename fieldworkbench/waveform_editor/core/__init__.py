"""Waveform Editor domain helpers.

Native waveform construction is format-neutral. Integer-width helpers in
``encoding`` are reserved for import/export codecs.
"""

from .native import NativeWaveformError

__all__ = ["NativeWaveformError"]
