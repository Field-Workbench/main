"""Small Waveform Editor bridge onto the global Field Workbench theme.

The integrated editor intentionally carries no private theme or QSettings keys:
all chrome follows the global Field Workbench palette and custom-painted widgets
read the Workbench plot palette directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ...theme import plot_theme_colors


@dataclass(frozen=True)
class FontSet:
    large: QFont
    medium: QFont
    small: QFont
    tiny: QFont

    @classmethod
    def create(cls, app: QApplication) -> "FontSet":
        default = app.font()

        def sized(delta: float, *, bold: bool = False) -> QFont:
            font = QFont(default)
            base_size = default.pointSizeF() if default.pointSizeF() > 0 else 10.0
            font.setPointSizeF(base_size + delta)
            font.setBold(bold)
            return font

        return cls(
            large=sized(4.0, bold=True),
            medium=sized(1.5, bold=True),
            small=sized(0.5),
            tiny=sized(0.0),
        )


def plot_colors():
    """Return an attribute-style view of Field Workbench's plot colours."""
    class Colours:
        pass

    result = Colours()
    for key, value in plot_theme_colors().items():
        setattr(result, key, value)
    return result
