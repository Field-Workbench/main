"""Embedded waveform-generator controls for the Waveform Editor.

Generator authoring lives entirely inside the Waveform Editor. These panels
own only the editable generator parameters; the persistent main Waveform Editor
viewer renders the non-destructive full-waveform preview and the main window
commits or discards the proposed chunk.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.native import (
    FreeFormPoint,
    GeneratedTrace,
    action_potential_trace,
    dc_trace,
    free_form_trace,
    sine_wave_trace,
    square_wave_trace,
)
from .formatting import format_time_span
from .widgets import NumericField, NumericSpec, action_button


_VALUE_MIN = -1.0e12
_VALUE_MAX = 1.0e12


class EmbeddedGeneratorPanel(QWidget):
    """One inline generator editor shown below the persistent waveform viewer."""

    def __init__(
        self,
        owner,
        *,
        generator: str,
        recipe_segment: Mapping[str, object] | None = None,
        recipe_segment_index: int | None = None,
        on_preview: Callable[[GeneratedTrace, int | None], str],
        on_apply: Callable[[GeneratedTrace, int | None], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__(owner)
        self.owner = owner
        self.generator = str(generator or "").strip().lower()
        self.recipe_segment = dict(recipe_segment or {})
        self.recipe_segment_index = None if recipe_segment_index is None else int(recipe_segment_index)
        parameters = self.recipe_segment.get("parameters")
        self.initial_parameters = dict(parameters) if isinstance(parameters, Mapping) else {}
        self._on_preview = on_preview
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        self._free_form_loading = False
        self._preview_focused = False
        self._periodic_synchronizing = False
        self._periodic_driver = "duration"

        try:
            signal = owner.native_editor.current_signal()
            self.value_unit = str(signal.unit or "")
            self.value_quantity = str(signal.quantity or "")
            low, high = owner.native_editor.current_value_defaults()
        except Exception:
            self.value_unit = ""
            self.value_quantity = ""
            low, high = -1.0, 1.0
        self.default_low = float(low)
        self.default_high = float(high)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(7)

        if self.generator == "free_form":
            heading = QLabel(
                "Double-click inside the free-form component to add a point; drag a point to move it, "
                "and drag its small tangent handles to change the curve.",
                self,
            )
            heading.setWordWrap(True)
            heading.setProperty("role", "muted")
        else:
            heading = QLabel(self._heading_text(), self)
            heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.controls_group = QGroupBox(self._group_title(), self)
        self.controls = QGridLayout(self.controls_group)
        self.controls.setHorizontalSpacing(8)
        self.controls.setVerticalSpacing(6)
        root.addWidget(self.controls_group)

        initial_duration = float(self.recipe_segment.get("duration_s", 1.0) or 1.0)
        if self.generator == "action_potential" and self.recipe_segment_index is None:
            initial_duration = 0.01
        if self.generator == "action_potential":
            self.duration = self._numeric(
                NumericSpec(
                    "Duration (ms)",
                    initial_duration * 1000.0,
                    1.0e-6,
                    1.0e12,
                    0.1,
                    9,
                    "float",
                    True,
                )
            )
        else:
            duration_callback = (
                self._periodic_duration_changed
                if self.generator in {"square", "sine"}
                else self.refresh_preview
            )
            self.duration = self._numeric(
                NumericSpec("Duration (s)", initial_duration, 1.0e-9, 1.0e9, 0.000001, 10, "float", True),
                on_change=duration_callback,
            )

        if self.generator in {"square", "sine"}:
            self._build_periodic_controls()
        elif self.generator == "dc":
            self._build_dc_controls()
        elif self.generator == "action_potential":
            self._build_action_potential_controls()
        elif self.generator == "free_form":
            self._build_free_form_controls()
        else:
            raise ValueError(f"Unsupported embedded waveform generator {self.generator!r}.")

        timing = QHBoxLayout()
        timing.addWidget(self.duration)
        timing_hint = QLabel(
            (
                f"Editing authored component {self.recipe_segment_index + 1}; Apply regenerates it in place."
                if self.recipe_segment_index is not None
                else "Apply appends this component to the working waveform."
            ),
            self,
        )
        timing_hint.setWordWrap(True)
        timing_hint.setProperty("role", "muted")
        timing.addWidget(timing_hint, 1)
        root.addLayout(timing)

        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        self.summary.setProperty("role", "muted")
        root.addWidget(self.summary)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.apply_button = action_button("Apply", self, self.apply, role="primary")
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.cancel_button)
        root.addLayout(actions)

        QTimer.singleShot(0, self.refresh_preview)

    def _heading_text(self) -> str:
        names = {
            "square": "square-wave component",
            "sine": "sine-wave component",
            "dc": "DC / pause component",
            "action_potential": "action-potential component",
            "free_form": "free-form component",
        }
        prefix = "Edit" if self.recipe_segment_index is not None else "Add"
        return f"{prefix} {names.get(self.generator, 'waveform component')}"

    def _group_title(self) -> str:
        return {
            "square": "Square-wave signal",
            "sine": "Sine-wave signal",
            "dc": "DC level",
            "action_potential": "Action-potential shape",
            "free_form": "Free-form drawing",
        }.get(self.generator, "Signal")

    def _unit_label(self, text: str) -> str:
        return f"{text} ({self.value_unit})" if self.value_unit else text

    def _numeric(
        self,
        spec: NumericSpec,
        *,
        on_change: Callable[[], None] | None = None,
    ) -> NumericField:
        return NumericField(
            self.controls_group,
            spec,
            on_change=self.refresh_preview if on_change is None else on_change,
        )

    def _build_periodic_controls(self) -> None:
        midpoint = (self.default_low + self.default_high) / 2.0
        peak_to_peak = abs(self.default_high - self.default_low)
        initial_frequency = float(self.initial_parameters.get("frequency_hz", 20.0))
        self.frequency = self._numeric(
            NumericSpec("Frequency (Hz)", initial_frequency, 1.0e-9, 1.0e9, 0.001, 10, "float", True),
            on_change=self._periodic_frequency_changed,
        )
        initial_cycles = max(1.0e-9, initial_frequency * self._duration_s())
        self.cycle_count = self._numeric(
            NumericSpec("Cycle count", initial_cycles, 1.0e-9, 1.0e12, 0.1, 10, "float", True),
            on_change=self._periodic_cycle_count_changed,
        )
        self.peak_to_peak = self._numeric(
            NumericSpec(
                self._unit_label("Peak-to-peak"),
                float(self.initial_parameters.get("peak_to_peak", peak_to_peak)),
                0.0,
                1.0e12,
                0.001,
                11,
                "float",
                True,
            )
        )
        self.offset = self._numeric(
            NumericSpec(
                self._unit_label("Offset"),
                float(self.initial_parameters.get("offset", midpoint)),
                _VALUE_MIN,
                _VALUE_MAX,
                0.001,
                11,
                "float",
                True,
            )
        )
        if self.generator == "square":
            self.shape_extra = self._numeric(
                NumericSpec(
                    "Duty cycle (%)",
                    float(self.initial_parameters.get("duty_cycle_percent", 50.0)),
                    0.001,
                    99.999,
                    0.1,
                    8,
                    "float",
                    True,
                )
            )
        else:
            self.shape_extra = self._numeric(
                NumericSpec(
                    "Phase (deg)",
                    float(self.initial_parameters.get("phase_deg", 0.0)),
                    -360000.0,
                    360000.0,
                    1.0,
                    9,
                    "float",
                    True,
                )
            )

        # Frequency and cycle count describe the time base; amplitude controls
        # get a second row so the extra authoring option does not compress them.
        self.controls.addWidget(self.frequency, 0, 0)
        self.controls.addWidget(self.cycle_count, 0, 1)
        self.controls.addWidget(self.peak_to_peak, 1, 0)
        self.controls.addWidget(self.offset, 1, 1)
        self.controls.addWidget(self.shape_extra, 1, 2)
        for column in range(3):
            self.controls.setColumnStretch(column, 1)

    def _periodic_duration_changed(self) -> None:
        if self._periodic_synchronizing or not hasattr(self, "cycle_count"):
            return
        self._periodic_driver = "duration"
        try:
            cycles = self.frequency.get_float() * self._duration_s()
        except (AttributeError, ValueError):
            self.refresh_preview()
            return
        self._periodic_synchronizing = True
        try:
            self.cycle_count.set(cycles)
        finally:
            self._periodic_synchronizing = False
        self.refresh_preview()

    def _periodic_cycle_count_changed(self) -> None:
        if self._periodic_synchronizing:
            return
        self._periodic_driver = "cycles"
        try:
            frequency = self.frequency.get_float()
            cycles = self.cycle_count.get_float()
        except ValueError:
            self.refresh_preview()
            return
        self._periodic_synchronizing = True
        try:
            self.duration.set(cycles / frequency)
        finally:
            self._periodic_synchronizing = False
        self.refresh_preview()

    def _periodic_frequency_changed(self) -> None:
        if self._periodic_synchronizing or not hasattr(self, "cycle_count"):
            return
        try:
            frequency = self.frequency.get_float()
        except ValueError:
            self.refresh_preview()
            return
        self._periodic_synchronizing = True
        try:
            if self._periodic_driver == "cycles":
                self.duration.set(self.cycle_count.get_float() / frequency)
            else:
                self.cycle_count.set(frequency * self._duration_s())
        except ValueError:
            pass
        finally:
            self._periodic_synchronizing = False
        self.refresh_preview()

    def _build_dc_controls(self) -> None:
        midpoint = (self.default_low + self.default_high) / 2.0
        self.level = self._numeric(
            NumericSpec(
                self._unit_label("Level"),
                float(self.initial_parameters.get("level", midpoint)),
                _VALUE_MIN,
                _VALUE_MAX,
                0.001,
                11,
                "float",
                True,
            )
        )
        self.controls.addWidget(self.level, 0, 0)
        self.controls.setColumnStretch(0, 1)

    def _build_action_potential_controls(self) -> None:
        timing_group = QGroupBox("Event timing (% of duration)", self.controls_group)
        timing_layout = QGridLayout(timing_group)
        timing_specs = (
            ("Depolarization starts", "start_percent", 15.0),
            ("Spike peak", "peak_percent", 44.0),
            ("Trough", "trough_percent", 64.0),
            ("Recovered", "recovery_percent", 77.0),
        )
        self.action_timing: list[NumericField] = []
        for column, (label, key, default) in enumerate(timing_specs):
            field = NumericField(
                timing_group,
                NumericSpec(label, float(self.initial_parameters.get(key, default)), _VALUE_MIN, _VALUE_MAX, 0.1, 7, "float", True),
                on_change=self.refresh_preview,
            )
            self.action_timing.append(field)
            timing_layout.addWidget(field, 0, column)
            timing_layout.setColumnStretch(column, 1)
        self.controls.addWidget(timing_group, 0, 0)

        levels_group = QGroupBox("Signal levels", self.controls_group)
        levels_layout = QGridLayout(levels_group)
        midpoint = (self.default_low + self.default_high) / 2.0
        level_specs = (
            (self._unit_label("Resting level"), "resting_level", midpoint),
            (self._unit_label("Peak level"), "peak_level", self.default_high),
            (self._unit_label("Trough level"), "trough_level", self.default_low),
        )
        self.action_levels: list[NumericField] = []
        for column, (label, key, default) in enumerate(level_specs):
            field = NumericField(
                levels_group,
                NumericSpec(label, float(self.initial_parameters.get(key, default)), _VALUE_MIN, _VALUE_MAX, 0.001, 11, "float", True),
                on_change=self.refresh_preview,
            )
            self.action_levels.append(field)
            levels_layout.addWidget(field, 0, column)
            levels_layout.setColumnStretch(column, 1)
        self.controls.addWidget(levels_group, 1, 0)
        self.controls.setColumnStretch(0, 1)

    def _validate_action_potential_inputs(self) -> tuple[bool, str]:
        """Mark invalid AP relationships red without constraining typed values."""
        if self.generator != "action_potential":
            return True, ""
        try:
            start, peak, trough, recovery = [field.get_float() for field in self.action_timing]
            resting, peak_level, trough_level = [field.get_float() for field in self.action_levels]
        except ValueError as error:
            return False, str(error)

        timing_invalid = [
            not (0.0 <= start <= 100.0) or start >= peak,
            not (0.0 <= peak <= 100.0) or peak <= start or peak >= trough,
            not (0.0 <= trough <= 100.0) or trough <= peak or trough >= recovery,
            not (0.0 <= recovery <= 100.0) or recovery <= trough,
        ]
        for field, invalid in zip(self.action_timing, timing_invalid):
            field.set_external_invalid(invalid)

        level_invalid = [
            not (peak_level > resting > trough_level),
            peak_level <= resting,
            trough_level >= resting,
        ]
        for field, invalid in zip(self.action_levels, level_invalid):
            field.set_external_invalid(invalid)

        if any(timing_invalid):
            return False, "Event timing must satisfy 0 ≤ depolarization < peak < trough < recovered ≤ 100%."
        if any(level_invalid):
            return False, "Signal levels must satisfy peak > resting > trough."
        return True, ""

    def _build_free_form_controls(self) -> None:
        selected_row = QHBoxLayout()
        self.free_form_selected_info = QLabel("Selected point: none", self.controls_group)
        self.free_form_selected_info.setProperty("role", "muted")
        selected_row.addWidget(self.free_form_selected_info)
        selected_row.addStretch(1)
        self.free_form_selected_value = NumericField(
            self.controls_group,
            NumericSpec(
                self._unit_label("Vertical position"),
                0.0,
                _VALUE_MIN,
                _VALUE_MAX,
                0.001,
                11,
                "float",
                True,
            ),
            on_change=self._selected_point_value_changed,
            label_position="left",
        )
        self.free_form_selected_value.setEnabled(False)
        selected_row.addWidget(self.free_form_selected_value)
        self.controls.addLayout(selected_row, 0, 0, 1, 2)

        row_actions = QHBoxLayout()
        add_button = QPushButton("Add point", self.controls_group)
        add_button.clicked.connect(self._add_anchor)
        self.free_form_delete_button = QPushButton("Delete selected point", self.controls_group)
        self.free_form_delete_button.setEnabled(False)
        delete_button = self.free_form_delete_button
        delete_button.clicked.connect(self._delete_anchor)
        self.free_form_reset_tangent_button = QPushButton("Reset selected tangent", self.controls_group)
        self.free_form_reset_tangent_button.setEnabled(False)
        self.free_form_reset_tangent_button.clicked.connect(self._reset_selected_tangent)
        fit_button = QPushButton("Fit handles", self.controls_group)
        fit_button.setToolTip("Temporarily expand the viewer so every free-form tangent handle is visible.")
        fit_button.clicked.connect(self._fit_free_form_handles)
        reset_button = QPushButton("Reset curve", self.controls_group)
        reset_button.clicked.connect(self._confirm_reset_anchors)
        row_actions.addWidget(add_button)
        row_actions.addWidget(delete_button)
        row_actions.addWidget(self.free_form_reset_tangent_button)
        row_actions.addWidget(fit_button)
        row_actions.addWidget(reset_button)
        row_actions.addStretch(1)
        self.controls.addLayout(row_actions, 1, 0, 1, 2)

        raw_points = self.initial_parameters.get("control_points")
        points: list[tuple[float, float, float, float]] = []
        if isinstance(raw_points, (list, tuple)):
            for raw in raw_points:
                if isinstance(raw, Mapping):
                    points.append(
                        (
                            float(raw.get("time_percent", 0.0)),
                            float(raw.get("value", 0.0)),
                            float(raw.get("handle_time_percent", 0.0)),
                            float(raw.get("handle_value", 0.0)),
                        )
                    )
        if len(points) < 2:
            midpoint = (self.default_low + self.default_high) / 2.0
            points = [(0.0, midpoint, 16.0, 0.0), (100.0, midpoint, 16.0, 0.0)]
        self._free_form_points = sorted(points, key=lambda item: item[0])
        self._free_form_selected = -1
        self.owner.native_editor.plot.freeFormEdited.connect(self._free_form_points_edited)
        self.owner.native_editor.plot.freeFormSelectionChanged.connect(self._free_form_selection_changed)

    def _anchor_values(self) -> list[tuple[float, float, float, float]]:
        return list(self._free_form_points)

    def _update_free_form_selected_controls(self) -> None:
        if self.generator != "free_form":
            return
        index = self._free_form_selected
        if 0 <= index < len(self._free_form_points):
            time_percent, value, _handle_time, _handle_value = self._free_form_points[index]
            self.free_form_selected_info.setText(
                f"Selected point: {index + 1} of {len(self._free_form_points)} · {time_percent:.6g}%"
            )
            self.free_form_selected_value.setEnabled(True)
            self.free_form_selected_value.set(value)
        else:
            self.free_form_selected_info.setText("Selected point: none")
            self.free_form_selected_value.setEnabled(False)

    def _selected_point_value_changed(self) -> None:
        if self.generator != "free_form" or self._free_form_loading:
            return
        index = self._free_form_selected
        if not (0 <= index < len(self._free_form_points)):
            return
        time_percent, _value, handle_time, handle_value = self._free_form_points[index]
        try:
            value = self.free_form_selected_value.get_float()
        except ValueError:
            return
        self._free_form_points[index] = (time_percent, value, handle_time, handle_value)
        self.refresh_preview()

    def _free_form_points_edited(self, payload) -> None:
        if self.generator != "free_form" or self._free_form_loading:
            return
        points: list[tuple[float, float, float, float]] = []
        try:
            for raw in list(payload or []):
                points.append(
                    (
                        float(raw.get("x", 0.0)) * 100.0,
                        float(raw.get("y", 0.0)),
                        float(raw.get("handle_dx", 0.0)) * 100.0,
                        float(raw.get("handle_dy", 0.0)),
                    )
                )
        except Exception:
            return
        if len(points) < 2:
            return
        self._free_form_points = sorted(points, key=lambda item: item[0])
        self._update_free_form_selected_controls()
        self.refresh_preview()

    def _free_form_selection_changed(self, index: int) -> None:
        if self.generator != "free_form":
            return
        self._free_form_selected = int(index)
        self.free_form_delete_button.setEnabled(0 < self._free_form_selected < len(self._free_form_points) - 1)
        self.free_form_reset_tangent_button.setEnabled(0 <= self._free_form_selected < len(self._free_form_points))
        self._update_free_form_selected_controls()

    def _sync_free_form_graphics(self) -> None:
        if self.generator != "free_form":
            return
        duration = self._duration_s()
        try:
            signal = self.owner.native_editor.current_signal()
            if self.recipe_segment_index is None:
                start_s = float(signal.time_s[0]) if self.owner.native_editor.is_placeholder() else float(signal.time_s[-1])
            else:
                recipe = self.owner.native_editor.exact_authoring_recipe() or {}
                segments = list(recipe.get("segments", []))
                start_s = float(signal.time_s[0])
                if 0 <= self.recipe_segment_index < len(segments):
                    start_s += float(segments[self.recipe_segment_index].get("start_s", 0.0) or 0.0)
        except Exception:
            start_s = 0.0
        points = [
            {
                "x": time_percent / 100.0,
                "y": value,
                "handle_dx": handle_time / 100.0,
                "handle_dy": handle_value,
            }
            for time_percent, value, handle_time, handle_value in self._free_form_points
        ]
        self.owner.native_editor.plot.set_free_form_editor(
            points,
            start_s=start_s,
            duration_s=duration,
            selected_index=self._free_form_selected,
        )

    def _add_anchor(self) -> None:
        # Keep the canvas as the source of truth while the graphical editor is
        # active.  It emits the normal freeFormEdited/selection signals, which
        # feed back into this panel and rebuild the live waveform preview.
        if self.owner.native_editor.plot.add_free_form_point():
            return

        # Defensive fallback for the instant before the overlay has been
        # initialized (for example an unusually fast click after opening it).
        points = sorted(self._anchor_values(), key=lambda item: item[0])
        if len(points) < 2:
            self._reset_anchors()
            return
        gap_index = max(range(len(points) - 1), key=lambda index: points[index + 1][0] - points[index][0])
        left = points[gap_index]
        right = points[gap_index + 1]
        x = (left[0] + right[0]) / 2.0
        y = (left[1] + right[1]) / 2.0
        handle = max(0.0, min(16.0, (right[0] - left[0]) * 0.22))
        points.append((x, y, handle, 0.0))
        self._free_form_points = sorted(points, key=lambda item: item[0])
        self._free_form_selected = next(
            (index for index, point in enumerate(self._free_form_points) if abs(point[0] - x) < 1.0e-9),
            -1,
        )
        self.free_form_delete_button.setEnabled(0 < self._free_form_selected < len(self._free_form_points) - 1)
        self.free_form_reset_tangent_button.setEnabled(0 <= self._free_form_selected < len(self._free_form_points))
        self._update_free_form_selected_controls()
        self.refresh_preview()

    def _delete_anchor(self) -> None:
        row = self._free_form_selected
        if row <= 0 or row >= len(self._free_form_points) - 1:
            return
        del self._free_form_points[row]
        self._free_form_selected = min(row, len(self._free_form_points) - 2)
        self.free_form_delete_button.setEnabled(0 < self._free_form_selected < len(self._free_form_points) - 1)
        self.free_form_reset_tangent_button.setEnabled(0 <= self._free_form_selected < len(self._free_form_points))
        self._update_free_form_selected_controls()
        self.refresh_preview()

    def _reset_selected_tangent(self) -> None:
        self.owner.native_editor.plot.reset_selected_free_form_tangent()

    def _fit_free_form_handles(self) -> None:
        self.owner.native_editor.plot.fit_free_form_handles()

    def _confirm_reset_anchors(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset free-form curve",
            "Are you sure you want to reset the curve?\n\n"
            "All current free-form control points and tangent adjustments will be replaced.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._reset_anchors()

    def _reset_anchors(self) -> None:
        midpoint = (self.default_low + self.default_high) / 2.0
        self._free_form_points = [(0.0, midpoint, 16.0, 0.0), (100.0, midpoint, 16.0, 0.0)]
        self._free_form_selected = -1
        self.free_form_delete_button.setEnabled(False)
        self.free_form_reset_tangent_button.setEnabled(False)
        self._update_free_form_selected_controls()
        self.refresh_preview()

    def _duration_s(self) -> float:
        value = self.duration.get_float()
        return value / 1000.0 if self.generator == "action_potential" else value

    def _action_parameters(self) -> dict[str, float]:
        return {
            "duration_s": self._duration_s(),
            "start_percent": self.action_timing[0].get_float(),
            "peak_percent": self.action_timing[1].get_float(),
            "trough_percent": self.action_timing[2].get_float(),
            "recovery_percent": self.action_timing[3].get_float(),
            "resting_level": self.action_levels[0].get_float(),
            "peak_level": self.action_levels[1].get_float(),
            "trough_level": self.action_levels[2].get_float(),
        }

    def build_trace(self) -> GeneratedTrace:
        duration = self._duration_s()
        if self.generator == "square":
            return square_wave_trace(
                frequency_hz=self.frequency.get_float(),
                duration_s=duration,
                peak_to_peak=self.peak_to_peak.get_float(),
                offset=self.offset.get_float(),
                duty_cycle_percent=self.shape_extra.get_float(),
            )
        if self.generator == "sine":
            return sine_wave_trace(
                frequency_hz=self.frequency.get_float(),
                duration_s=duration,
                peak_to_peak=self.peak_to_peak.get_float(),
                offset=self.offset.get_float(),
                phase_deg=self.shape_extra.get_float(),
            )
        if self.generator == "dc":
            return dc_trace(duration_s=duration, level=self.level.get_float())
        if self.generator == "action_potential":
            return action_potential_trace(**self._action_parameters())
        if self.generator == "free_form":
            points = [
                FreeFormPoint(time_percent / 100.0, value, handle_time / 100.0, handle_value)
                for time_percent, value, handle_time, handle_value in self._anchor_values()
            ]
            return free_form_trace(points, duration_s=duration)
        raise ValueError(f"Unsupported waveform generator {self.generator!r}.")

    def _summary_text(self, trace: GeneratedTrace) -> str:
        duration = float(trace.settings.get("duration_s", self._duration_s()))
        suffix = f" {self.value_unit}" if self.value_unit else ""
        if self.generator in {"square", "sine"}:
            frequency = self.frequency.get_float()
            magnitude = self.peak_to_peak.get_float()
            offset = self.offset.get_float()
            low = offset - magnitude / 2.0
            high = offset + magnitude / 2.0
            return (
                f"Period {format_time_span(1.0 / frequency)} · {duration * frequency:.6g} cycles · "
                f"range {low:.6g} to {high:.6g}{suffix}"
            )
        if self.generator == "dc":
            return f"Hold {self.level.get_float():.6g}{suffix} for {format_time_span(duration)}."
        if self.generator == "action_potential":
            p = self._action_parameters()
            start = duration * p["start_percent"] / 100.0
            peak = duration * p["peak_percent"] / 100.0
            trough = duration * p["trough_percent"] / 100.0
            recovery = duration * p["recovery_percent"] / 100.0
            return (
                f"Events: start {format_time_span(start)} · peak {format_time_span(peak)} · "
                f"trough {format_time_span(trough)} · recovered {format_time_span(recovery)}"
            )
        if self.generator == "free_form":
            values = [point[1] for point in self._anchor_values()]
            return (
                f"{len(values)} control points · authored range {min(values):.6g} to {max(values):.6g}{suffix}."
                if values else "Add at least two control points."
            )
        return ""

    def refresh_preview(self, *_args) -> None:
        if self._free_form_loading:
            return
        if self.generator == "action_potential":
            valid, message = self._validate_action_potential_inputs()
            if not valid:
                self.summary.setText(f"Preview unavailable: {message}")
                self.apply_button.setEnabled(False)
                return
        try:
            trace = self.build_trace()
            placement = self._on_preview(trace, self.recipe_segment_index)
            if self.generator == "action_potential" and not self._preview_focused:
                self.owner.native_editor.focus_generated_preview(
                    trace,
                    segment_index=self.recipe_segment_index,
                )
                self._preview_focused = True
            if self.generator == "free_form":
                self._sync_free_form_graphics()
        except Exception as error:  # noqa: BLE001 - live validation boundary
            self.summary.setText(f"Preview unavailable: {error}")
            self.apply_button.setEnabled(False)
            return
        self.apply_button.setEnabled(True)
        text = self._summary_text(trace)
        self.summary.setText(f"{text}\n{placement}" if placement else text)

    def apply(self) -> None:
        try:
            trace = self.build_trace()
            self._on_apply(trace, self.recipe_segment_index)
        except Exception as error:  # noqa: BLE001 - UI validation boundary
            self.summary.setText(f"Unable to apply: {error}")
            self.apply_button.setEnabled(False)

    def cancel(self) -> None:
        self._on_cancel()

    def detach_preview_editor(self) -> None:
        """Disconnect direct-manipulation hooks before this embedded panel is retired."""
        if self.generator != "free_form":
            return
        plot = self.owner.native_editor.plot
        try:
            plot.freeFormEdited.disconnect(self._free_form_points_edited)
        except (RuntimeError, TypeError):
            pass
        try:
            plot.freeFormSelectionChanged.disconnect(self._free_form_selection_changed)
        except (RuntimeError, TypeError):
            pass
