"""Reusable theme-aware waveform preview canvas.

The main Waveform Editor and the generator/builder dialogs intentionally share
this canvas so waveform authoring has one visual language.  The widget knows
only physical time/value data; sampling density is never exposed as a UI mode.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QSizePolicy, QToolButton, QWidget

from ...theme import plot_theme_colors


class WaveformPreviewCanvas(QWidget):
    """Plot one physical waveform, optionally marking an append boundary."""

    segmentReordered = Signal(int, int)
    segmentDoubleClicked = Signal(int)
    segmentSelected = Signal(int)
    freeFormEdited = Signal(object)
    freeFormSelectionChanged = Signal(int)

    def __init__(self, parent=None, *, empty_text: str = "No waveform loaded") -> None:
        super().__init__(parent)
        self.signal: Any | None = None
        self._time_s: tuple[float, ...] = ()
        self._values: tuple[float, ...] = ()
        self._unit = ""
        self._quantity = ""
        self._empty_text = str(empty_text)
        self._append_boundary_s: float | None = None
        self._segments: list[dict[str, Any]] = []
        self._segments_editable = False
        self._selected_segment = -1
        self._view: tuple[float, float, float, float] | None = None
        self._mode = "select"
        self._press_pos: QPoint | None = None
        self._press_segment = -1
        self._pan_origin: tuple[float, float, float, float] | None = None
        self._zoom_drag_pos: QPointF | None = None
        self._component_selection_available = False
        self._free_form_active = False
        self._free_form_points: list[dict[str, float]] = []
        self._free_form_start_s = 0.0
        self._free_form_duration_s = 1.0
        self._free_form_selected = -1
        self._free_form_drag: tuple[str, int] | None = None
        # Match the Plotly viewers used elsewhere in Field Workbench: the
        # navigation controls live as a compact icon-only modebar inside the
        # figure rather than consuming a labelled toolbar row.  The waveform
        # canvas remains native Qt because component dragging and free-form
        # Bézier editing need direct mouse access to the authored geometry.
        self._modebar = QWidget(self)
        self._modebar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        tools = QHBoxLayout(self._modebar)
        tools.setContentsMargins(2, 2, 4, 2)
        tools.setSpacing(0)
        tools.addStretch(1)
        self._modebar_buttons: dict[str, QToolButton] = {}
        for key, tip, callback, checkable in (
            ("select", "Select and drag waveform components", lambda: self._set_mode("select"), True),
            ("zoom", "Zoom to rectangle", lambda: self._set_mode("zoom"), True),
            ("pan", "Pan", lambda: self._set_mode("pan"), True),
            ("zoom_in", "Zoom in", lambda: self._zoom_by(0.8), False),
            ("zoom_out", "Zoom out", lambda: self._zoom_by(1.25), False),
            ("home", "Reset axes", self.home, False),
            ("camera", "Save plot as PNG", self.save_png, False),
        ):
            button = QToolButton(self._modebar)
            button.setAutoRaise(True)
            button.setCheckable(checkable)
            button.setToolTip(tip)
            button.setFixedSize(27, 27)
            button.setIconSize(QSize(18, 18))
            button.clicked.connect(callback)
            self._modebar_buttons[key] = button
            tools.addWidget(button)
        self._modebar_buttons["select"].setChecked(True)
        self._modebar_buttons["select"].setVisible(False)
        self._modebar_theme_signature: tuple[str, ...] | None = None
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _modebar_icon(kind: str, colour: QColor) -> QIcon:
        """Return a compact monochrome icon in the visual language of Plotly's modebar."""

        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(colour, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if kind == "select":
            painter.drawRect(QRectF(4.0, 4.0, 12.0, 12.0))
            painter.drawLine(QPointF(2.5, 7.0), QPointF(6.5, 7.0))
            painter.drawLine(QPointF(4.5, 5.0), QPointF(4.5, 9.0))
        elif kind == "zoom":
            painter.drawEllipse(QPointF(8.0, 8.0), 5.0, 5.0)
            painter.drawLine(QPointF(11.7, 11.7), QPointF(17.0, 17.0))
        elif kind == "pan":
            painter.drawLine(QPointF(10.0, 2.5), QPointF(10.0, 17.5))
            painter.drawLine(QPointF(2.5, 10.0), QPointF(17.5, 10.0))
            painter.drawLine(QPointF(10.0, 2.5), QPointF(7.5, 5.0))
            painter.drawLine(QPointF(10.0, 2.5), QPointF(12.5, 5.0))
            painter.drawLine(QPointF(10.0, 17.5), QPointF(7.5, 15.0))
            painter.drawLine(QPointF(10.0, 17.5), QPointF(12.5, 15.0))
            painter.drawLine(QPointF(2.5, 10.0), QPointF(5.0, 7.5))
            painter.drawLine(QPointF(2.5, 10.0), QPointF(5.0, 12.5))
            painter.drawLine(QPointF(17.5, 10.0), QPointF(15.0, 7.5))
            painter.drawLine(QPointF(17.5, 10.0), QPointF(15.0, 12.5))
        elif kind in {"zoom_in", "zoom_out"}:
            painter.drawEllipse(QPointF(8.0, 8.0), 5.0, 5.0)
            painter.drawLine(QPointF(11.7, 11.7), QPointF(17.0, 17.0))
            painter.drawLine(QPointF(5.0, 8.0), QPointF(11.0, 8.0))
            if kind == "zoom_in":
                painter.drawLine(QPointF(8.0, 5.0), QPointF(8.0, 11.0))
        elif kind == "home":
            path = QPainterPath()
            path.moveTo(3.0, 9.0)
            path.lineTo(10.0, 3.0)
            path.lineTo(17.0, 9.0)
            path.moveTo(5.0, 8.0)
            path.lineTo(5.0, 16.5)
            path.lineTo(15.0, 16.5)
            path.lineTo(15.0, 8.0)
            painter.drawPath(path)
        elif kind == "camera":
            painter.drawRoundedRect(QRectF(3.0, 6.0, 14.0, 10.0), 1.5, 1.5)
            painter.drawEllipse(QPointF(10.0, 11.0), 3.0, 3.0)
            painter.drawLine(QPointF(6.0, 6.0), QPointF(7.5, 3.8))
            painter.drawLine(QPointF(7.5, 3.8), QPointF(12.0, 3.8))
            painter.drawLine(QPointF(12.0, 3.8), QPointF(13.5, 6.0))
        painter.end()
        return QIcon(pixmap)

    def _refresh_modebar_style(self, colours: dict[str, str]) -> None:
        signature = (
            str(colours.get("text", "")),
            str(colours.get("reference", "")),
            str(colours.get("figure", "")),
            str(colours.get("major_grid", "")),
        )
        if signature == self._modebar_theme_signature:
            return
        self._modebar_theme_signature = signature
        icon_colour = QColor(colours["text"])
        icon_colour.setAlpha(145)
        reference = QColor(colours["reference"])
        checked = f"rgba({reference.red()}, {reference.green()}, {reference.blue()}, 48)"
        hover = QColor(colours["major_grid"])
        hover_css = f"rgba({hover.red()}, {hover.green()}, {hover.blue()}, 90)"
        for key, button in self._modebar_buttons.items():
            button.setIcon(self._modebar_icon(key, icon_colour))
            button.setStyleSheet(
                "QToolButton { border: none; background: transparent; padding: 3px; }"
                f"QToolButton:hover {{ background: {hover_css}; border-radius: 2px; }}"
                f"QToolButton:checked {{ background: {checked}; border-radius: 2px; }}"
            )

    @staticmethod
    def _format_tick(value: float, span: float) -> str:
        magnitude = abs(float(span))
        value = float(value)
        if magnitude == 0.0:
            return f"{value:.4g}"
        if magnitude < 1.0e-3 or magnitude >= 1.0e5:
            return f"{value:.3e}"
        if magnitude < 0.1:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        if magnitude < 10.0:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        if magnitude < 1000.0:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{value:.4g}"

    def set_signal(self, signal: Any | None) -> None:
        """Display a WaveformSeries-like single-signal object."""
        self.signal = signal
        self._time_s = ()
        self._values = ()
        self._unit = ""
        self._quantity = ""
        self._append_boundary_s = None
        self._view = None
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._modebar.setGeometry(0, 4, self.width(), 31)
        self._modebar.raise_()
        super().resizeEvent(event)

    def _plot_rect(self) -> QRectF:
        return QRectF(72.0, 48.0, max(50.0, self.width() - 94.0), max(50.0, self.height() - 100.0))

    def set_segments(self, segments: Sequence[dict[str, Any]]) -> None:
        self._segments = [dict(item) for item in segments]
        if self._selected_segment >= len(self._segments):
            self._selected_segment = -1
        self.update()

    def set_segments_editable(self, editable: bool) -> None:
        self._segments_editable = bool(editable)

    def set_component_selection_available(self, available: bool) -> None:
        """Show component-selection controls only while authoring a waveform."""
        available = bool(available)
        was_available = self._component_selection_available
        self._component_selection_available = available
        button = self._modebar_buttons.get("select")
        if button is not None:
            button.setVisible(available)
        if available and not was_available:
            # Enter authoring in selection mode so component manipulation is
            # immediately available, just as it was before the modebar became
            # context-sensitive.
            self._set_mode("select")
        elif not available and self._mode == "select":
            # Passive navigation has no component-selection affordance.  Box
            # zoom is the closest match to the map viewer's default drag tool.
            self._set_mode("zoom")
        self.update()

    def set_free_form_editor(
        self,
        points: Sequence[dict[str, float]],
        *,
        start_s: float,
        duration_s: float,
        selected_index: int = -1,
    ) -> None:
        """Overlay direct-manipulation controls for one free-form component.

        Coordinates are stored in the authored component's local normalized
        time (0..1) while values remain in the waveform's physical display
        unit.  The waveform itself remains the full composed preview, so other
        components stay visible for scale and context while this one is edited.
        """

        was_active = self._free_form_active
        clean: list[dict[str, float]] = []
        for raw in points:
            clean.append(
                {
                    "x": float(raw.get("x", 0.0)),
                    "y": float(raw.get("y", 0.0)),
                    "handle_dx": max(0.0, float(raw.get("handle_dx", 0.0))),
                    "handle_dy": float(raw.get("handle_dy", 0.0)),
                }
            )
        clean.sort(key=lambda item: item["x"])
        self._free_form_active = len(clean) >= 2 and float(duration_s) > 0.0
        self._free_form_points = clean
        self._free_form_start_s = float(start_s)
        self._free_form_duration_s = max(float(duration_s), 1.0e-12)
        self._free_form_selected = (
            int(selected_index) if 0 <= int(selected_index) < len(clean) else -1
        )
        if self._free_form_active and not was_active:
            self._set_mode("select")
        self.update()

    def clear_free_form_editor(self) -> None:
        self._free_form_active = False
        self._free_form_points = []
        self._free_form_selected = -1
        self._free_form_drag = None
        self.update()

    def _free_form_payload(self) -> list[dict[str, float]]:
        return [dict(point) for point in self._free_form_points]

    def _emit_free_form_edited(self) -> None:
        self.freeFormEdited.emit(self._free_form_payload())
        self.update()

    def _free_form_max_handle_dx(self, index: int) -> float:
        if index < 0 or index >= len(self._free_form_points):
            return 0.0
        point = self._free_form_points[index]
        spans: list[float] = []
        if index > 0:
            spans.append(point["x"] - self._free_form_points[index - 1]["x"])
        if index + 1 < len(self._free_form_points):
            spans.append(self._free_form_points[index + 1]["x"] - point["x"])
        return max(0.0, 0.49 * min(spans)) if spans else 0.0

    def _clamp_free_form_handles(self) -> None:
        for index, point in enumerate(self._free_form_points):
            point["handle_dx"] = min(max(0.0, point["handle_dx"]), self._free_form_max_handle_dx(index))

    def add_free_form_point(self) -> bool:
        """Insert and select a new anchor in the largest free-form gap.

        The preview canvas is the source of truth while direct manipulation is
        active.  Adding through the canvas keeps the overlay, panel recipe, and
        selection synchronized through the same ``freeFormEdited`` signal used
        by mouse edits.
        """
        if not self._free_form_active or len(self._free_form_points) < 2:
            return False
        gap_index = max(
            range(len(self._free_form_points) - 1),
            key=lambda index: self._free_form_points[index + 1]["x"]
            - self._free_form_points[index]["x"],
        )
        left = self._free_form_points[gap_index]
        right = self._free_form_points[gap_index + 1]
        local_x = (left["x"] + right["x"]) / 2.0
        value = (left["y"] + right["y"]) / 2.0
        handle_dx = max(0.0, min(local_x - left["x"], right["x"] - local_x) * 0.22)
        insertion = gap_index + 1
        self._free_form_points.insert(
            insertion,
            {"x": local_x, "y": value, "handle_dx": handle_dx, "handle_dy": 0.0},
        )
        self._free_form_selected = insertion
        # Inserting a point can shorten neighbouring spans. Clamp their old
        # tangents so the newly-authored curve remains valid immediately.
        self._clamp_free_form_handles()
        self.freeFormSelectionChanged.emit(insertion)
        self._emit_free_form_edited()
        return True

    def delete_selected_free_form_point(self) -> bool:
        index = self._free_form_selected
        if index <= 0 or index >= len(self._free_form_points) - 1:
            return False
        del self._free_form_points[index]
        self._free_form_selected = min(index, len(self._free_form_points) - 2)
        self._clamp_free_form_handles()
        self.freeFormSelectionChanged.emit(self._free_form_selected)
        self._emit_free_form_edited()
        return True

    def reset_selected_free_form_tangent(self) -> bool:
        """Restore the selected anchor's mirrored tangent to a sane default."""
        index = self._free_form_selected
        if index < 0 or index >= len(self._free_form_points):
            return False
        point = self._free_form_points[index]
        spans: list[float] = []
        if index > 0:
            spans.append(point["x"] - self._free_form_points[index - 1]["x"])
        if index + 1 < len(self._free_form_points):
            spans.append(self._free_form_points[index + 1]["x"] - point["x"])
        span = min(spans) if spans else 1.0
        point["handle_dx"] = min(self._free_form_max_handle_dx(index), max(0.0, span * 0.16))
        point["handle_dy"] = 0.0
        self._emit_free_form_edited()
        return True

    def fit_free_form_handles(self) -> bool:
        """Expand the current view so every active free-form handle is visible."""
        if not self._free_form_active or not self._free_form_points:
            return False
        times, values, _unit, _quantity = self._data()
        pairs = [
            (float(x), float(y))
            for x, y in zip(times, values)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        for index, point in enumerate(self._free_form_points):
            anchor_x = self._free_form_start_s + point["x"] * self._free_form_duration_s
            pairs.append((anchor_x, point["y"]))
            if index > 0:
                pairs.append(
                    (
                        self._free_form_start_s
                        + (point["x"] - point["handle_dx"]) * self._free_form_duration_s,
                        point["y"] - point["handle_dy"],
                    )
                )
            if index + 1 < len(self._free_form_points):
                pairs.append(
                    (
                        self._free_form_start_s
                        + (point["x"] + point["handle_dx"]) * self._free_form_duration_s,
                        point["y"] + point["handle_dy"],
                    )
                )
        if not pairs:
            return False
        x0, x1 = min(x for x, _ in pairs), max(x for x, _ in pairs)
        y0, y1 = min(y for _, y in pairs), max(y for _, y in pairs)
        if x1 <= x0:
            x0, x1 = x0 - 0.5, x1 + 0.5
        else:
            pad_x = max((x1 - x0) * 0.03, 1.0e-12)
            x0, x1 = x0 - pad_x, x1 + pad_x
        if y1 <= y0:
            pad_y = max(abs(y0) * 0.05, 1.0)
        else:
            pad_y = max((y1 - y0) * 0.06, 1.0e-12)
        self._view = (x0, x1, y0 - pad_y, y1 + pad_y)
        self.update()
        return True

    def set_selected_segment(self, index: int) -> None:
        """Highlight one authored chunk without changing the waveform."""
        value = int(index)
        if value < 0 or value >= len(self._segments):
            value = -1
        if self._selected_segment != value:
            self._selected_segment = value
            self.update()

    def _set_mode(self, mode: str) -> None:
        if mode == "select" and not self._component_selection_available:
            mode = "zoom"
        self._mode = mode
        for key in ("select", "zoom", "pan"):
            button = self._modebar_buttons.get(key)
            if button is not None:
                button.setChecked(key == mode)
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if mode == "pan" else
            Qt.CursorShape.CrossCursor if mode == "zoom" else Qt.CursorShape.ArrowCursor
        )

    def _zoom_by(self, factor: float) -> None:
        ranges = self._data_ranges()
        if ranges is None:
            return
        x0, x1, y0, y1 = ranges
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        scale = max(0.05, float(factor))
        self._view = (
            cx + (x0 - cx) * scale,
            cx + (x1 - cx) * scale,
            cy + (y0 - cy) * scale,
            cy + (y1 - cy) * scale,
        )
        self.update()

    def focus_time_range(
        self,
        start_s: float,
        end_s: float,
        *,
        context_before_s: float = 0.0,
        context_after_s: float = 0.0,
    ) -> None:
        """Temporarily focus the viewer on a physical time window.

        This is used for very short embedded previews, such as a millisecond
        action-potential component appended to a multi-second waveform.  The
        y-range is derived only from samples in the focused time window so the
        prospective component remains inspectable instead of collapsing to a
        few pixels at the far edge of the full-waveform view.
        """

        times, values, _unit, _quantity = self._data()
        pairs = [
            (float(x), float(y))
            for x, y in zip(times, values)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if not pairs:
            return

        data_x0 = min(x for x, _ in pairs)
        data_x1 = max(x for x, _ in pairs)
        requested_x0 = min(float(start_s), float(end_s)) - max(0.0, float(context_before_s))
        requested_x1 = max(float(start_s), float(end_s)) + max(0.0, float(context_after_s))
        x0 = max(data_x0, requested_x0)
        x1 = min(data_x1, requested_x1)
        if x1 <= x0:
            x0, x1 = data_x0, data_x1
        if x1 <= x0:
            x1 = x0 + 1.0

        visible = [(x, y) for x, y in pairs if x0 <= x <= x1]
        if not visible:
            visible = pairs
        y0 = min(y for _, y in visible)
        y1 = max(y for _, y in visible)
        if y1 <= y0:
            pad = max(abs(y0) * 0.05, 1.0)
        else:
            pad = max((y1 - y0) * 0.08, 1.0e-12)
        self._view = (x0, x1, y0 - pad, y1 + pad)
        self.update()

    def home(self) -> None:
        self._view = None
        self.update()

    def save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save waveform plot", "waveform.png", "PNG images (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.grab(self._plot_rect().toRect()).save(path, "PNG")

    def set_trace(
        self,
        time_s: Sequence[float],
        values: Sequence[float],
        *,
        unit: str = "",
        quantity: str = "",
        append_boundary_s: float | None = None,
        empty_text: str | None = None,
    ) -> None:
        """Display explicit physical time/value data without creating a document."""
        self.signal = None
        self._time_s = tuple(float(value) for value in time_s)
        self._values = tuple(float(value) for value in values)
        self._unit = str(unit or "")
        self._quantity = str(quantity or "")
        self._append_boundary_s = None if append_boundary_s is None else float(append_boundary_s)
        if empty_text is not None:
            self._empty_text = str(empty_text)
        self.update()

    def clear(self, *, text: str | None = None) -> None:
        self.signal = None
        self._time_s = ()
        self._values = ()
        self._unit = ""
        self._quantity = ""
        self._append_boundary_s = None
        self._view = None
        if text is not None:
            self._empty_text = str(text)
        self.update()

    def _data(self) -> tuple[Sequence[float], Sequence[float], str, str]:
        if self.signal is not None:
            return (
                getattr(self.signal, "time_s", ()),
                getattr(self.signal, "values", ()),
                str(getattr(self.signal, "unit", "") or ""),
                str(getattr(self.signal, "quantity", "") or ""),
            )
        return self._time_s, self._values, self._unit, self._quantity

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        colours = plot_theme_colors()
        self._refresh_modebar_style(colours)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(colours["figure"]))

        rect = self._plot_rect()
        painter.fillRect(rect, QColor(colours["axes"]))
        text = QColor(colours["text"])
        major = QColor(colours["major_grid"])
        minor = QColor(colours["minor_grid"])

        painter.setPen(QPen(minor, 1.0, Qt.PenStyle.DotLine))
        for index in range(1, 10):
            x = rect.left() + rect.width() * index / 10.0
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for index in range(1, 8):
            y = rect.top() + rect.height() * index / 8.0
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        painter.setPen(QPen(major, 1.0))
        for index in range(0, 6):
            x = rect.left() + rect.width() * index / 5.0
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for index in range(0, 5):
            y = rect.top() + rect.height() * index / 4.0
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(text, 1.0))
        painter.drawRect(rect)

        times, values, unit, quantity = self._data()
        finite_pairs = [
            (float(time_value), float(sample))
            for time_value, sample in zip(times, values)
            if math.isfinite(float(time_value)) and math.isfinite(float(sample))
        ]
        if not finite_pairs:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            painter.end()
            return

        x_min = min(value[0] for value in finite_pairs)
        x_max = max(value[0] for value in finite_pairs)
        y_min = min(value[1] for value in finite_pairs)
        y_max = max(value[1] for value in finite_pairs)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            padding = max(abs(y_min) * 0.05, 1.0)
            y_min -= padding
            y_max += padding
        if self._view is not None:
            x_min, x_max, y_min, y_max = self._view

        painter.save()
        painter.setClipRect(rect)
        for index, segment in enumerate(self._segments):
            left = float(segment.get("start_s", 0.0))
            right = left + float(segment.get("duration_s", 0.0))
            if right < x_min or left > x_max:
                continue
            px1 = rect.left() + (max(left, x_min) - x_min) / (x_max - x_min) * rect.width()
            px2 = rect.left() + (min(right, x_max) - x_min) / (x_max - x_min) * rect.width()
            colour = QColor(colours["reference"])
            colour.setAlpha(48 if index == self._selected_segment else 18)
            painter.fillRect(QRectF(px1, rect.top(), max(1.0, px2 - px1), rect.height()), colour)
            if index:
                painter.setPen(QPen(QColor(colours["reference"]), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(px1, rect.top()), QPointF(px1, rect.bottom()))
        else:
            padding = (y_max - y_min) * 0.05
            y_min -= padding
            y_max += padding

        def xy(time_value: float, sample: float) -> QPointF:
            x = rect.left() + (time_value - x_min) / (x_max - x_min) * rect.width()
            y = rect.bottom() - (sample - y_min) / (y_max - y_min) * rect.height()
            return QPointF(x, y)

        stride = max(1, len(finite_pairs) // max(2_000, int(rect.width() * 2)))
        drawn = finite_pairs[::stride]
        if drawn[-1] != finite_pairs[-1]:
            drawn.append(finite_pairs[-1])
        path = QPainterPath()
        path.moveTo(xy(*drawn[0]))
        for pair in drawn[1:]:
            path.lineTo(xy(*pair))
        painter.setPen(QPen(text, 1.5))
        painter.drawPath(path)

        # The waveform and component shading stay clipped to the axes, but the
        # direct-manipulation overlay is allowed into the surrounding viewer
        # margin.  Tangent handles can therefore remain visible and draggable
        # even when their control values lie beyond the current graph limits.
        painter.restore()
        if self._free_form_active and self._free_form_points:
            reference = QColor(colours["reference"])
            handle_pen = QPen(reference, 1.2)
            painter.setPen(handle_pen)
            for index, point in enumerate(self._free_form_points):
                anchor_time = self._free_form_start_s + point["x"] * self._free_form_duration_s
                anchor = xy(anchor_time, point["y"])
                if index > 0:
                    left = xy(
                        self._free_form_start_s + (point["x"] - point["handle_dx"]) * self._free_form_duration_s,
                        point["y"] - point["handle_dy"],
                    )
                    display_left, left_proxy = self._free_form_display_handle(anchor, left)
                    painter.drawLine(anchor, display_left)
                    painter.setBrush(reference if left_proxy else QColor(colours["axes"]))
                    painter.drawEllipse(display_left, 5.0 if left_proxy else 4.0, 5.0 if left_proxy else 4.0)
                    if left_proxy:
                        vx, vy = left.x() - anchor.x(), left.y() - anchor.y()
                        length = max(math.hypot(vx, vy), 1.0)
                        painter.drawLine(
                            display_left,
                            QPointF(display_left.x() + vx / length * 9.0, display_left.y() + vy / length * 9.0),
                        )
                if index + 1 < len(self._free_form_points):
                    right = xy(
                        self._free_form_start_s + (point["x"] + point["handle_dx"]) * self._free_form_duration_s,
                        point["y"] + point["handle_dy"],
                    )
                    display_right, right_proxy = self._free_form_display_handle(anchor, right)
                    painter.drawLine(anchor, display_right)
                    painter.setBrush(reference if right_proxy else QColor(colours["axes"]))
                    painter.drawEllipse(display_right, 5.0 if right_proxy else 4.0, 5.0 if right_proxy else 4.0)
                    if right_proxy:
                        vx, vy = right.x() - anchor.x(), right.y() - anchor.y()
                        length = max(math.hypot(vx, vy), 1.0)
                        painter.drawLine(
                            display_right,
                            QPointF(display_right.x() + vx / length * 9.0, display_right.y() + vy / length * 9.0),
                        )
                painter.setBrush(reference if index == self._free_form_selected else QColor(colours["axes"]))
                painter.setPen(QPen(reference, 1.5))
                painter.drawEllipse(anchor, 6.0, 6.0)

        painter.save()
        painter.setClipRect(rect)
        boundary = self._append_boundary_s
        if boundary is not None and x_min < boundary < x_max:
            marker_x = xy(boundary, y_min).x()
            painter.setPen(QPen(QColor(colours["reference"]), 1.3, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(marker_x, rect.top()), QPointF(marker_x, rect.bottom()))
            painter.drawText(
                QRectF(marker_x + 5.0, rect.top() + 4.0, max(0.0, rect.right() - marker_x - 8.0), 20.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                "new component",
            )

        painter.restore()

        # Plotly-like major ticks and labels.  The native waveform canvas keeps
        # the direct editing behavior, but its axes now use the same visual
        # rhythm as the map viewers instead of labelling only min/max.
        painter.setPen(QPen(text, 1.0))
        metrics = QFontMetricsF(painter.font())
        x_span = x_max - x_min
        y_span = y_max - y_min
        for index in range(6):
            fraction = index / 5.0
            px = rect.left() + rect.width() * fraction
            value = x_min + x_span * fraction
            painter.drawLine(QPointF(px, rect.bottom()), QPointF(px, rect.bottom() + 4.0))
            painter.drawText(
                QRectF(px - 46.0, rect.bottom() + 6.0, 92.0, metrics.height() + 4.0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                self._format_tick(value, x_span),
            )
        for index in range(5):
            fraction = index / 4.0
            py = rect.bottom() - rect.height() * fraction
            value = y_min + y_span * fraction
            painter.drawLine(QPointF(rect.left() - 4.0, py), QPointF(rect.left(), py))
            painter.drawText(
                QRectF(18.0, py - metrics.height() / 2.0 - 1.0, rect.left() - 26.0, metrics.height() + 4.0),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._format_tick(value, y_span),
            )

        painter.drawText(
            QRectF(rect.left(), rect.bottom() + 25.0, rect.width(), 20.0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Time (s)",
        )
        axis_label = str(quantity or "Signal")
        if unit:
            axis_label += f" ({unit})"
        painter.save()
        painter.translate(4.0, rect.center().y())
        painter.rotate(-90.0)
        painter.drawText(
            QRectF(-rect.height() / 2.0, 0.0, rect.height(), 18.0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            axis_label,
        )
        painter.restore()

        # Plotly-style box-zoom feedback: draw the selection rectangle while
        # the pointer is dragged instead of applying the zoom only on release.
        if self._mode == "zoom" and self._press_pos is not None and self._zoom_drag_pos is not None:
            start = self._clamp_to_plot(QPointF(self._press_pos))
            stop = self._clamp_to_plot(self._zoom_drag_pos)
            rubber = QRectF(start, stop).normalized().intersected(rect)
            if rubber.width() >= 1.0 and rubber.height() >= 1.0:
                reference = QColor(colours["reference"])
                fill = QColor(reference)
                fill.setAlpha(36)
                painter.fillRect(rubber, fill)
                reference.setAlpha(190)
                painter.setPen(QPen(reference, 1.2, Qt.PenStyle.DashLine))
                painter.drawRect(rubber)

        painter.end()

    def _data_ranges(self) -> tuple[float, float, float, float] | None:
        times, values, _unit, _quantity = self._data()
        pairs = [(float(x), float(y)) for x, y in zip(times, values) if math.isfinite(float(x)) and math.isfinite(float(y))]
        if not pairs:
            return None
        x0, x1 = min(x for x, _ in pairs), max(x for x, _ in pairs)
        y0, y1 = min(y for _, y in pairs), max(y for _, y in pairs)
        if x1 <= x0: x1 = x0 + 1.0
        if y1 <= y0: y0, y1 = y0 - 1.0, y1 + 1.0
        else:
            pad = (y1 - y0) * 0.05
            y0, y1 = y0 - pad, y1 + pad
        return self._view or (x0, x1, y0, y1)

    def _position_to_data_unbounded(self, pos: QPointF) -> tuple[float, float] | None:
        """Map a widget position into plot data coordinates without clipping.

        Direct-manipulation controls sometimes need to travel beyond the axes
        rectangle.  In particular, free-form Bézier tangent handles should be
        able to overshoot the currently visible y-range while the waveform
        itself remains clipped to the plot.  The ordinary interaction helper
        below still rejects positions outside the axes for pan/zoom/selection.
        """

        ranges = self._data_ranges()
        if ranges is None:
            return None
        rect = self._plot_rect()
        x0, x1, y0, y1 = ranges
        x = x0 + (pos.x() - rect.left()) / rect.width() * (x1 - x0)
        y = y1 - (pos.y() - rect.top()) / rect.height() * (y1 - y0)
        return x, y

    def _position_to_data(self, pos: QPointF) -> tuple[float, float] | None:
        rect = self._plot_rect()
        if not rect.contains(pos):
            return None
        return self._position_to_data_unbounded(pos)

    def _clamp_to_plot(self, pos: QPointF) -> QPointF:
        """Clamp a pointer position to the drawable axes rectangle."""
        rect = self._plot_rect()
        return QPointF(
            min(max(float(pos.x()), rect.left()), rect.right()),
            min(max(float(pos.y()), rect.top()), rect.bottom()),
        )

    def _data_to_position(self, x: float, y: float) -> QPointF | None:
        ranges = self._data_ranges()
        if ranges is None:
            return None
        rect = self._plot_rect()
        x0, x1, y0, y1 = ranges
        return QPointF(
            rect.left() + (float(x) - x0) / (x1 - x0) * rect.width(),
            rect.bottom() - (float(y) - y0) / (y1 - y0) * rect.height(),
        )

    def _free_form_proxy_point(self, anchor: QPointF, handle: QPointF) -> QPointF | None:
        """Return a visible plot-edge proxy for an off-axis tangent handle.

        Endpoint anchors can sit exactly on the plot boundary.  Treating an
        inset edge as the first ray/edge intersection then creates a proxy only
        a pixel away from the anchor, where the anchor marker paints over it and
        makes the tangent appear to vanish.  Intersect against the *real* plot
        rectangle instead, ignore boundary contacts that are effectively the
        anchor itself, and continue to the next edge crossed by the tangent.
        """
        plot_rect = self._plot_rect()
        # A handle lying on the graph edge is still directly reachable; only
        # create a proxy after it has genuinely left the plot rectangle.
        if plot_rect.adjusted(-0.5, -0.5, 0.5, 0.5).contains(handle):
            return None
        if not plot_rect.adjusted(-1.0, -1.0, 1.0, 1.0).contains(anchor):
            return None

        dx = handle.x() - anchor.x()
        dy = handle.y() - anchor.y()
        if abs(dx) < 1.0e-12 and abs(dy) < 1.0e-12:
            return None

        candidates: list[tuple[float, QPointF]] = []
        epsilon_t = 1.0e-8
        if abs(dx) > 1.0e-12:
            for edge_x in (plot_rect.left(), plot_rect.right()):
                t = (edge_x - anchor.x()) / dx
                if epsilon_t < t <= 1.0:
                    y = anchor.y() + t * dy
                    if plot_rect.top() - 0.5 <= y <= plot_rect.bottom() + 0.5:
                        candidates.append(
                            (
                                t,
                                QPointF(
                                    edge_x,
                                    min(max(y, plot_rect.top()), plot_rect.bottom()),
                                ),
                            )
                        )
        if abs(dy) > 1.0e-12:
            for edge_y in (plot_rect.top(), plot_rect.bottom()):
                t = (edge_y - anchor.y()) / dy
                if epsilon_t < t <= 1.0:
                    x = anchor.x() + t * dx
                    if plot_rect.left() - 0.5 <= x <= plot_rect.right() + 0.5:
                        candidates.append(
                            (
                                t,
                                QPointF(
                                    min(max(x, plot_rect.left()), plot_rect.right()),
                                    edge_y,
                                ),
                            )
                        )

        # Prefer the first *distinct* edge crossing.  This matters when an
        # endpoint anchor lies on the right/left edge: a near-zero contact at
        # that same edge must not hide a later top/bottom proxy under the
        # anchor marker.
        for _t, point in sorted(candidates, key=lambda item: item[0]):
            if math.hypot(point.x() - anchor.x(), point.y() - anchor.y()) >= 8.0:
                return point

        # Rare outward-pointing boundary case: there may be no later crossing.
        # Keep a grab target on the nearest graph edge, but separate it from
        # the anchor so it cannot be painted over or lose the hit test.
        fallback = QPointF(
            min(max(handle.x(), plot_rect.left()), plot_rect.right()),
            min(max(handle.y(), plot_rect.top()), plot_rect.bottom()),
        )
        if math.hypot(fallback.x() - anchor.x(), fallback.y() - anchor.y()) >= 8.0:
            return fallback

        separation = 10.0
        distances = {
            "left": abs(anchor.x() - plot_rect.left()),
            "right": abs(anchor.x() - plot_rect.right()),
            "top": abs(anchor.y() - plot_rect.top()),
            "bottom": abs(anchor.y() - plot_rect.bottom()),
        }
        edge = min(distances, key=distances.get)
        if edge in {"left", "right"}:
            edge_x = plot_rect.left() if edge == "left" else plot_rect.right()
            direction = 1.0 if dy >= 0.0 else -1.0
            y = anchor.y() + direction * separation
            if not (plot_rect.top() + 6.0 <= y <= plot_rect.bottom() - 6.0):
                y = anchor.y() - direction * separation
            return QPointF(edge_x, min(max(y, plot_rect.top() + 6.0), plot_rect.bottom() - 6.0))

        edge_y = plot_rect.top() if edge == "top" else plot_rect.bottom()
        direction = 1.0 if dx >= 0.0 else -1.0
        x = anchor.x() + direction * separation
        if not (plot_rect.left() + 6.0 <= x <= plot_rect.right() - 6.0):
            x = anchor.x() - direction * separation
        return QPointF(min(max(x, plot_rect.left() + 6.0), plot_rect.right() - 6.0), edge_y)

    def _free_form_display_handle(self, anchor: QPointF, handle: QPointF) -> tuple[QPointF, bool]:
        proxy = self._free_form_proxy_point(anchor, handle)
        return (proxy, True) if proxy is not None else (handle, False)

    def _free_form_hit_test(self, pos: QPointF) -> tuple[str, int] | None:
        if not self._free_form_active:
            return None
        candidates: list[tuple[float, str, int]] = []
        for index, point in enumerate(self._free_form_points):
            anchor_x = self._free_form_start_s + point["x"] * self._free_form_duration_s
            anchor = self._data_to_position(anchor_x, point["y"])
            if anchor is None:
                continue
            candidates.append((math.hypot(pos.x() - anchor.x(), pos.y() - anchor.y()), "anchor", index))
            if index > 0:
                left = self._data_to_position(
                    self._free_form_start_s + (point["x"] - point["handle_dx"]) * self._free_form_duration_s,
                    point["y"] - point["handle_dy"],
                )
                if left is not None:
                    display_left, _proxy = self._free_form_display_handle(anchor, left)
                    candidates.append((math.hypot(pos.x() - display_left.x(), pos.y() - display_left.y()), "left", index))
            if index + 1 < len(self._free_form_points):
                right = self._data_to_position(
                    self._free_form_start_s + (point["x"] + point["handle_dx"]) * self._free_form_duration_s,
                    point["y"] + point["handle_dy"],
                )
                if right is not None:
                    display_right, _proxy = self._free_form_display_handle(anchor, right)
                    candidates.append((math.hypot(pos.x() - display_right.x(), pos.y() - display_right.y()), "right", index))
        if not candidates:
            return None
        distance, kind, index = min(candidates, key=lambda item: item[0])
        return (kind, index) if distance <= 9.0 else None

    def _free_form_add_at(self, pos: QPointF) -> bool:
        data = self._position_to_data(pos)
        if data is None or len(self._free_form_points) < 2:
            return False
        x, y = data
        local_x = (x - self._free_form_start_s) / self._free_form_duration_s
        if not 0.0 < local_x < 1.0:
            return False
        for point in self._free_form_points:
            if abs(point["x"] - local_x) < 1.0e-5:
                return False
        insertion = 0
        while insertion < len(self._free_form_points) and self._free_form_points[insertion]["x"] < local_x:
            insertion += 1
        left = self._free_form_points[insertion - 1]["x"]
        right = self._free_form_points[insertion]["x"]
        handle_dx = max(0.0, min(local_x - left, right - local_x) * 0.22)
        self._free_form_points.insert(
            insertion,
            {"x": local_x, "y": float(y), "handle_dx": handle_dx, "handle_dy": 0.0},
        )
        self._free_form_selected = insertion
        self._clamp_free_form_handles()
        self.freeFormSelectionChanged.emit(insertion)
        self._emit_free_form_edited()
        return True

    def _drag_free_form_control(self, pos: QPointF) -> None:
        if self._free_form_drag is None:
            return
        kind, index = self._free_form_drag
        # Anchors remain confined to the graph/component domain, but tangent
        # handles intentionally use an unbounded coordinate transform so they
        # can be dragged above or below the visible axes.  This allows strong
        # overshoot/undershoot without requiring the user to zoom first.
        data = (
            self._position_to_data(pos)
            if kind == "anchor"
            else self._position_to_data_unbounded(pos)
        )
        if data is None:
            return
        x, y = data
        local_x = (x - self._free_form_start_s) / self._free_form_duration_s
        if not (0 <= index < len(self._free_form_points)):
            return
        point = self._free_form_points[index]
        if kind == "anchor":
            if index == 0:
                local_x = 0.0
            elif index == len(self._free_form_points) - 1:
                local_x = 1.0
            else:
                left = self._free_form_points[index - 1]["x"] + 1.0e-6
                right = self._free_form_points[index + 1]["x"] - 1.0e-6
                local_x = min(max(local_x, left), right)
            point["x"] = min(max(local_x, 0.0), 1.0)
            point["y"] = float(y)
            self._clamp_free_form_handles()
        else:
            max_dx = self._free_form_max_handle_dx(index)
            if kind == "right":
                point["handle_dx"] = min(max(local_x - point["x"], 0.0), max_dx)
                point["handle_dy"] = float(y) - point["y"]
            else:
                point["handle_dx"] = min(max(point["x"] - local_x, 0.0), max_dx)
                point["handle_dy"] = point["y"] - float(y)
        self._emit_free_form_edited()

    def _segment_at(self, pos: QPointF) -> int:
        point = self._position_to_data(pos)
        if point is None:
            return -1
        x = point[0]
        for index, segment in enumerate(self._segments):
            start = float(segment.get("start_s", 0.0))
            if start <= x <= start + float(segment.get("duration_s", 0.0)):
                return index
        return -1

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        point = self._position_to_data(event.position())
        ranges = self._data_ranges()
        if point is None or ranges is None:
            return super().wheelEvent(event)
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        x0, x1, y0, y1 = ranges
        cx, cy = point
        self._view = (cx + (x0-cx)*factor, cx + (x1-cx)*factor, cy + (y0-cy)*factor, cy + (y1-cy)*factor)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._free_form_active and self._mode == "select":
            hit = self._free_form_hit_test(event.position())
            if event.button() == Qt.MouseButton.RightButton:
                if hit is not None and hit[0] == "anchor":
                    self._free_form_selected = hit[1]
                    self.freeFormSelectionChanged.emit(self._free_form_selected)
                    self.delete_selected_free_form_point()
                return
            if hit is not None:
                self._free_form_selected = hit[1]
                self._free_form_drag = hit
                self.freeFormSelectionChanged.emit(self._free_form_selected)
                self.update()
                return
            # While the free-form tool is active, ordinary clicks in select
            # mode belong to the drawing editor rather than component reorder.
            self._free_form_selected = -1
            self.freeFormSelectionChanged.emit(-1)
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._plot_rect().contains(event.position()):
            return
        self._press_pos = event.position().toPoint()
        self._press_segment = self._segment_at(event.position())
        self._zoom_drag_pos = QPointF(event.position()) if self._mode == "zoom" else None
        if self._mode == "select":
            self._selected_segment = self._press_segment
            self.segmentSelected.emit(self._selected_segment)
            self.update()
        elif self._mode == "pan":
            self._pan_origin = self._data_ranges()
        elif self._mode == "zoom":
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._free_form_active and self._free_form_drag is not None and self._mode == "select":
            self._drag_free_form_control(event.position())
            return
        if self._mode == "zoom" and self._press_pos is not None:
            self._zoom_drag_pos = self._clamp_to_plot(event.position())
            self.update()
            return
        if self._mode != "pan" or self._press_pos is None or self._pan_origin is None:
            return
        rect = self._plot_rect()
        x0, x1, y0, y1 = self._pan_origin
        dx = (event.position().x() - self._press_pos.x()) / rect.width() * (x1-x0)
        dy = (event.position().y() - self._press_pos.y()) / rect.height() * (y1-y0)
        self._view = (x0-dx, x1-dx, y0+dy, y1+dy)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._free_form_active and self._free_form_drag is not None:
            self._free_form_drag = None
            return
        if self._press_pos is None:
            return
        if self._mode == "select" and self._segments_editable and self._press_segment >= 0:
            target = self._segment_at(event.position())
            if target >= 0 and target != self._press_segment:
                self.segmentReordered.emit(self._press_segment, target)
                self._selected_segment = target
                self.segmentSelected.emit(target)
        elif self._mode == "zoom":
            start_pos = self._clamp_to_plot(QPointF(self._press_pos))
            stop_pos = self._clamp_to_plot(event.position())
            start = self._position_to_data_unbounded(start_pos)
            stop = self._position_to_data_unbounded(stop_pos)
            if (
                start is not None
                and stop is not None
                and abs(stop_pos.x() - start_pos.x()) > 5
                and abs(stop_pos.y() - start_pos.y()) > 5
            ):
                self._view = (
                    min(start[0], stop[0]),
                    max(start[0], stop[0]),
                    min(start[1], stop[1]),
                    max(start[1], stop[1]),
                )
        self._press_pos = None
        self._pan_origin = None
        self._zoom_drag_pos = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._free_form_active and self._mode == "select":
            if self._free_form_hit_test(event.position()) is None and self._free_form_add_at(event.position()):
                return
        index = self._segment_at(event.position())
        if index >= 0:
            self.segmentDoubleClicked.emit(index)
