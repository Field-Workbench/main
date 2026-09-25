"""Reusable Qt widgets shared by Waveform Editor windows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from PySide6.QtCore import QSignalBlocker, QSize, QTimer, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


NumberKind = Literal["int", "float"]


@dataclass(frozen=True)
class NumericSpec:
    label: str
    value: int | float
    minimum: int | float
    maximum: int | float
    increment: int | float = 1
    width: int = 7
    kind: NumberKind = "int"
    trim_trailing_zeros: bool = False


class _WheelSafeSpinBox(QSpinBox):
    """A spin box that cannot be changed accidentally by wheel scrolling."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        event.ignore()


class _WheelSafeDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        event.ignore()


class _CompactDoubleSpinBox(_WheelSafeDoubleSpinBox):
    """Double spin box that hides insignificant trailing decimal zeros."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        text = f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")
        return text if text not in {"", "-0"} else "0"


def _decimal_places(value: int | float) -> int:
    decimal = Decimal(str(value)).normalize()
    return max(0, -decimal.as_tuple().exponent)


class NumericField(QWidget):
    """A labeled, range-constrained integer or floating-point control."""

    def __init__(
        self,
        parent: QWidget,
        spec: NumericSpec,
        *,
        on_change: Callable[[], None] | None = None,
        label_position: Literal["top", "left"] = "top",
        fixed_input_width: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self._on_change = on_change
        self._external_invalid = False
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        if label_position == "left":
            layout = QHBoxLayout(self)
            layout.setSpacing(6)
        else:
            layout = QVBoxLayout(self)
            layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(spec.label, self)
        if label_position == "left":
            self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        else:
            self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.label)

        if spec.kind == "int":
            control = _WheelSafeSpinBox(self)
            control.setRange(int(spec.minimum), int(spec.maximum))
            control.setSingleStep(int(spec.increment))
            control.setValue(int(spec.value))
        else:
            control = (
                _CompactDoubleSpinBox(self)
                if spec.trim_trailing_zeros
                else _WheelSafeDoubleSpinBox(self)
            )
            control.setDecimals(max(1, min(9, _decimal_places(spec.increment))))
            control.setRange(float(spec.minimum), float(spec.maximum))
            control.setSingleStep(float(spec.increment))
            control.setValue(float(spec.value))
        control.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_width = max(72, spec.width * 10) if fixed_input_width is None else fixed_input_width
        if fixed_input_width is None:
            control.setMinimumWidth(input_width)
        else:
            control.setFixedWidth(input_width)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        control.setKeyboardTracking(True)
        self.spinbox: QSpinBox | QDoubleSpinBox = control
        layout.addWidget(control)
        control.valueChanged.connect(self._changed)
        self.mark_validity()

    def is_valid(self) -> bool:
        value = float(self.spinbox.value())
        return float(self.spec.minimum) <= value <= float(self.spec.maximum)

    def mark_validity(self) -> bool:
        valid = self.is_valid()
        self.spinbox.setProperty("invalid", (not valid) or self._external_invalid)
        self.spinbox.style().unpolish(self.spinbox)
        self.spinbox.style().polish(self.spinbox)
        return valid and not self._external_invalid

    def set_external_invalid(self, invalid: bool) -> None:
        """Apply relationship-level validation without constraining typed values."""
        self._external_invalid = bool(invalid)
        self.mark_validity()

    def get(self) -> int | float:
        if not self.is_valid():
            raise ValueError(
                f"{self.spec.label.replace(chr(10), ' ')} must be between "
                f"{self.spec.minimum} and {self.spec.maximum}."
            )
        return int(self.spinbox.value()) if self.spec.kind == "int" else float(self.spinbox.value())

    def get_int(self) -> int:
        return int(self.get())

    def get_float(self) -> float:
        return float(self.get())

    def set(self, value: int | float, *, notify: bool = False) -> None:
        blocker = QSignalBlocker(self.spinbox)
        normalized = int(value) if self.spec.kind == "int" else float(value)
        self.spinbox.setValue(normalized)
        del blocker
        self.mark_validity()
        if notify and self._on_change is not None:
            self._on_change()

    def reconfigure(self, spec: NumericSpec, *, value: int | float | None = None) -> None:
        """Update an existing numeric field's label/range without rebuilding its layout."""

        if spec.kind != self.spec.kind:
            raise ValueError("NumericField kind cannot be changed after construction.")
        self.spec = spec
        self.label.setText(spec.label)
        blocker = QSignalBlocker(self.spinbox)
        if spec.kind == "int":
            self.spinbox.setRange(int(spec.minimum), int(spec.maximum))
            self.spinbox.setSingleStep(int(spec.increment))
            self.spinbox.setValue(int(spec.value if value is None else value))
        else:
            assert isinstance(self.spinbox, QDoubleSpinBox)
            self.spinbox.setDecimals(max(1, min(9, _decimal_places(spec.increment))))
            self.spinbox.setRange(float(spec.minimum), float(spec.maximum))
            self.spinbox.setSingleStep(float(spec.increment))
            self.spinbox.setValue(float(spec.value if value is None else value))
        del blocker
        self.mark_validity()

    def _changed(self, _value: int | float) -> None:
        self.mark_validity()
        if self._on_change is not None:
            self._on_change()


def _waveform_icon(kind: str, *, width: int = 76, height: int = 36) -> QIcon:
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor("#2f6f9f"))
    pen.setWidth(3)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    left = 6
    right = width - 6
    top = 7
    bottom = height - 7
    mid = height // 2

    if kind == "square":
        path = [
            (left, mid + 11),
            (width * 0.28, mid + 11),
            (width * 0.28, top + 6),
            (width * 0.72, top + 6),
            (width * 0.72, mid + 11),
            (right, mid + 11),
        ]
    elif kind == "sine":
        path = []
        span = max(1, right - left)
        import math

        amplitude = (bottom - top) * 0.34
        for step in range(span + 1):
            x = left + step
            phase = (step / span) * (2.0 * math.pi)
            y = mid - math.sin(phase) * amplitude
            path.append((x, y))
    elif kind == "action_potential":
        path = [
            (left, mid + 8),
            (width * 0.32, mid + 8),
            (width * 0.42, mid + 2),
            (width * 0.47, top + 2),
            (width * 0.53, bottom - 5),
            (width * 0.63, mid + 12),
            (width * 0.78, mid + 8),
            (right, mid + 8),
        ]
    elif kind == "free_form":
        path = [
            (left, mid + 8),
            (width * 0.22, mid - 5),
            (width * 0.38, mid - 12),
            (width * 0.52, mid + 9),
            (width * 0.70, mid + 3),
            (right, mid - 10),
        ]
    elif kind == "import":
        # Conventional import glyph: an arrow entering an outlined box/tray.
        box_left = int(width * 0.22)
        box_right = int(width * 0.78)
        box_top = int(height * 0.48)
        box_bottom = int(height * 0.82)
        painter.drawLine(box_left, box_top, box_left, box_bottom)
        painter.drawLine(box_left, box_bottom, box_right, box_bottom)
        painter.drawLine(box_right, box_bottom, box_right, box_top)
        arrow_x = width // 2
        arrow_top = int(height * 0.10)
        arrow_tip = int(height * 0.62)
        painter.drawLine(arrow_x, arrow_top, arrow_x, arrow_tip)
        painter.drawLine(arrow_x, arrow_tip, int(width * 0.39), int(height * 0.45))
        painter.drawLine(arrow_x, arrow_tip, int(width * 0.61), int(height * 0.45))
        path = []
    else:
        path = [(left, mid), (right, mid)]

    for start, end in zip(path, path[1:]):
        painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))

    painter.end()
    return QIcon(pixmap)


def waveform_button(
    text: str,
    parent: QWidget,
    callback: Callable[[], None],
    *,
    icon_kind: str,
) -> QToolButton:
    button = QToolButton(parent)
    button.setText(text)
    button.setIcon(_waveform_icon(icon_kind))
    button.setIconSize(QSize(76, 36))
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    button.setMinimumHeight(82)
    button.setProperty("role", "waveform")
    button.clicked.connect(callback)
    return button


class StatusBar(QStatusBar):
    """Readable status text with an optional transient failure highlight."""

    def __init__(self, parent: QWidget, *, initial: str = "Idle") -> None:
        super().__init__(parent)
        self.setSizeGripEnabled(False)
        self._feedback_timer = QTimer(self)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.timeout.connect(self._clear_feedback)
        self.showMessage(initial)

    def _set_state(self, state: str | None) -> None:
        self.setProperty("statusState", state)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _clear_feedback(self) -> None:
        self._set_state(None)

    def set(self, text: str) -> None:
        """Show an ordinary status message and clear any failure highlight."""
        self._feedback_timer.stop()
        self._clear_feedback()
        self.showMessage(text)

    def fail(self, text: str, *, duration_ms: int = 2000) -> None:
        """Show a refused/invalid-operation message with a brief red highlight."""
        self._feedback_timer.stop()
        self.showMessage(text)
        self._set_state("error")
        self._feedback_timer.start(max(0, int(duration_ms)))


class ScrollPanel(QScrollArea):
    """Resizable scroll area with a ready-to-use vertical content layout."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.interior = QWidget(self)
        self.content_layout = QVBoxLayout(self.interior)
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout.setSpacing(5)
        self.content_layout.addStretch(1)
        self.setWidget(self.interior)

    def add_content_widget(self, widget: QWidget) -> None:
        self.content_layout.insertWidget(self.content_layout.count() - 1, widget)
        self.scroll_to_bottom()

    def scroll_to_bottom(self) -> None:
        def scroll() -> None:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())

        QTimer.singleShot(0, scroll)


def action_button(
    text: str,
    parent: QWidget,
    callback: Callable[[], None],
    *,
    role: str | None = None,
) -> QPushButton:
    button = QPushButton(text, parent)
    if role is not None:
        button.setProperty("role", role)
        if role == "primary":
            button.setObjectName("primaryButton")
        elif role == "danger":
            button.setObjectName("dangerButton")
    button.clicked.connect(callback)
    return button
