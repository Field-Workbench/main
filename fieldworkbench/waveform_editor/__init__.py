"""Field Workbench Waveform Editor.

The suite editor uses Field Workbench's native floating-point waveform model.
Some DAC/Rigol codec behavior is derived from WaveBuilder 2.0.0.16, but
integer sample widths are restricted to explicit import/export boundaries.
"""

WAVEBUILDER_SOURCE_VERSION = "2.0.0.16"

__all__ = ["WAVEBUILDER_SOURCE_VERSION"]
