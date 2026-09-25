"""Qt configuration dialog for quasi-static waveform playback."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .camera_timeline import normalize_camera_timeline, scripted_timeline_duration_s
from .exposures import exposure_coil_ids, normalize_exposure_protocol
from .studio_adapter import StudioAdapter, StudioOperationError
from .waveform import (
    MAX_ANALYSIS_SAMPLES,
    analysis_sample_count,
    build_prepared_gpu_payload,
    drive_coil_catalog,
    observation_catalog,
    playback_frame_count,
    validate_trace,
)
from .waveform_video_dialog import WaveformVideoExportDialog
from .window_utils import (
    CompactDoubleSpinBox,
    capture_window_default_settings,
    centre_window_on_parent,
    enable_standard_window_controls,
    reset_window_to_default_settings,
)


def remembered_waveform_timing(map_kind: str) -> tuple[float, float]:
    """Best-effort physical duration/base speed from the last waveform dialog settings."""
    key = f"ui/windows/waveform_playback/{str(map_kind).lower()}"
    settings = QSettings()
    try:
        controls = json.loads(str(settings.value(f"{key}/controls", "") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        controls = {}
    try:
        extra = json.loads(str(settings.value(f"{key}/extra", "") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        extra = {}
    if not isinstance(controls, dict):
        controls = {}
    if not isinstance(extra, dict):
        extra = {}

    def control_value(name: str, fallback: float) -> float:
        state = controls.get(name)
        if not isinstance(state, dict):
            return fallback
        try:
            value = float(state.get("value", fallback))
        except (TypeError, ValueError):
            return fallback
        return value if math.isfinite(value) else fallback

    duration = max(1.0e-6, control_value("waveform_duration", 1.0))
    multiplier = max(1.0e-6, control_value("real_time_multiplier", 1.0))
    return duration, multiplier


class _EditorWheelScrollFilter(QObject):
    """Use the wheel for dialog navigation, never silent editor changes."""

    def __init__(self, scroll_area: QScrollArea):
        super().__init__(scroll_area)
        self.scroll_area = scroll_area

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if event.type() != QEvent.Type.Wheel or not isinstance(
            watched, (QDoubleSpinBox, QSpinBox, QComboBox)
        ):
            return super().eventFilter(watched, event)

        scroll_bar = self.scroll_area.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        if pixel_delta:
            distance = pixel_delta
        else:
            wheel_steps = event.angleDelta().y() / 120.0
            distance = wheel_steps * max(36, scroll_bar.singleStep() * 3)
        scroll_bar.setValue(scroll_bar.value() - round(distance))
        event.accept()
        return True


class _CollapsibleSection(QWidget):
    """Compact disclosure section matching the map and video setup dialogs."""

    def __init__(self, title: str, *, expanded: bool = True, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self.header = QToolButton()
        self.header.setText(str(title))
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.header.setAutoRaise(True)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setStyleSheet(
            "QToolButton { font-weight: 600; text-align: left; padding: 5px 2px; }"
        )
        root.addWidget(self.header)

        self.content = QWidget()
        self.content.setMinimumWidth(0)
        self.content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.content.setVisible(bool(expanded))
        root.addWidget(self.content)
        self.header.toggled.connect(self._set_expanded)

    def setContentLayout(self, layout: QLayout) -> None:  # noqa: N802 - Qt helper
        self.content.setLayout(layout)

    def _set_expanded(self, expanded: bool) -> None:
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(bool(expanded))


def _double_spin(
    value: float,
    minimum: float,
    maximum: float,
    decimals: int,
    suffix: str = "",
) -> QDoubleSpinBox:
    spin = CompactDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    return spin


class WaveformPlaybackDialog(QDialog):
    """Configure a drive waveform and build a cached-field animation."""

    def __init__(
        self,
        adapter: StudioAdapter,
        *,
        map_kind: str,
        map_settings: dict[str, Any],
        camera_timeline: dict[str, Any] | None = None,
        initial_camera: dict[str, Any] | None = None,
        playback_fps: float = 60.0,
        playback_speed: float = 1.0,
        primary_action: str = "playback",
        defer_payload_build: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.map_kind = str(map_kind).lower()
        self._persistent_settings_key = f"waveform_playback/{self.map_kind}"
        self.map_settings = dict(map_settings)
        self.camera_timeline = dict(camera_timeline) if isinstance(camera_timeline, dict) else None
        self.initial_camera = dict(initial_camera) if isinstance(initial_camera, dict) else None
        try:
            requested_fps = float(playback_fps)
        except (TypeError, ValueError):
            requested_fps = 60.0
        try:
            requested_speed = float(playback_speed)
        except (TypeError, ValueError):
            requested_speed = 1.0
        self.playback_fps_value = requested_fps if math.isfinite(requested_fps) and requested_fps > 0 else 60.0
        self.playback_speed_value = requested_speed if math.isfinite(requested_speed) and requested_speed > 0 else 1.0
        self.primary_action = "export" if str(primary_action).lower() == "export" else "playback"
        self.defer_payload_build = bool(
            defer_payload_build and self.primary_action == "playback"
        )
        self.result_figure: dict[str, Any] | None = None
        self.result_gpu_payload: dict[str, Any] | None = None
        self.result_gpu_request: dict[str, Any] | None = None
        self._runtime_exposure_protocol: dict[str, Any] | None = None
        self._runtime_exposure_metadata: dict[str, Any] = {}
        self.result_label = "Waveform"
        self._default_brain_region_trace_keys: set[str] = set()
        self._drive_choices = drive_coil_catalog(adapter)
        available_drive_ids = {str(choice["id"]) for choice in self._drive_choices}
        requested_drive_ids = [
            str(object_id) for object_id in self.map_settings.get("drive_coil_ids", [])
            if str(object_id) in available_drive_ids
        ]
        self._base_drive_ids = (
            requested_drive_ids
            if "drive_coil_ids" in self.map_settings
            else [
                str(choice["id"])
                for choice in self._drive_choices
                if bool(choice.get("enabled", True))
            ]
        )
        self._observation_choices = observation_catalog(adapter)
        self.setWindowTitle(
            "2D waveform playback"
            if self.map_kind == "2d"
            else (
                "Brain waveform playback"
                if self.map_kind == "brain"
                else "3D waveform playback"
            )
        )
        self.resize(860, 720)

        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        heading = QLabel("Waveform playback")
        heading.setObjectName("dialogHeading")
        content_layout.addWidget(heading)

        # Quick current source ---------------------------------------------
        # Saved Exposures own source/load wiring, voltage-source behavior,
        # and sequencing. This dialog keeps only the lightweight one-off
        # current generator used by Maps' Built-in waveform source.
        self.drive_section = _CollapsibleSection("Quick current source", expanded=False)
        drive_layout = QFormLayout()
        self.slew_rate = _double_spin(100.0, 0.0, 1e12, 6, " mA/ms")
        self.slew_rate.setToolTip(
            "Maximum current transition rate. Set 0 for an ideal instantaneous step."
        )
        drive_layout.addRow("Slew limit", self.slew_rate)
        quick_note = QLabel(
            "Quick playback applies one generated current waveform to the active coils selected "
            "in Scene options. Voltage sources, electrical load wiring, and timed multi-source "
            "sequences belong in Exposure Editor."
        )
        quick_note.setWordWrap(True)
        quick_note.setObjectName("hint")
        drive_layout.addRow(quick_note)
        self.drive_section.setContentLayout(drive_layout)
        content_layout.addWidget(self.drive_section)

        # Waveform ---------------------------------------------------------
        self.waveform_section = _CollapsibleSection("Waveform", expanded=True)
        waveform_layout = QGridLayout()
        waveform_layout.setColumnStretch(1, 1)
        self.waveform_choice_group = QButtonGroup(self)
        self.waveform_choice_group.setExclusive(True)
        self.waveform_sine = QCheckBox("Sine")
        self.waveform_square = QCheckBox("Square")
        self.waveform_triangle = QCheckBox("Triangle")
        for button, mode in (
            (self.waveform_sine, "sine"),
            (self.waveform_square, "square"),
            (self.waveform_triangle, "triangle"),
        ):
            self.waveform_choice_group.addButton(button)
            button.setProperty("waveformMode", mode)
            button.setProperty("persistUiState", False)
        self.waveform_sine.setChecked(True)
        waveform_layout.addWidget(self.waveform_sine, 0, 0)
        waveform_layout.addWidget(self.waveform_square, 1, 0)
        waveform_layout.addWidget(self.waveform_triangle, 2, 0)

        self.generated_waveform_controls = QWidget()
        generated_form = QFormLayout(self.generated_waveform_controls)
        generated_form.setContentsMargins(16, 0, 0, 0)
        self.waveform_amplitude = _double_spin(1.0, 0.0, 1e9, 6)
        self.waveform_frequency = _double_spin(1.0, 0.000001, 1e6, 6, " Hz")
        self.waveform_duration = _double_spin(1.0, 0.000001, 1e9, 6, " s")
        self.waveform_frequency.setToolTip("Generated waveform frequency.")
        self.waveform_duration.setToolTip("Generated waveform duration.")
        generated_form.addRow("Amplitude", self.waveform_amplitude)
        generated_form.addRow("Frequency", self.waveform_frequency)
        generated_form.addRow("Duration", self.waveform_duration)
        waveform_layout.addWidget(self.generated_waveform_controls, 0, 1, 3, 1)
        self.waveform_section.setContentLayout(waveform_layout)
        content_layout.addWidget(self.waveform_section)

        # Output traces ----------------------------------------------------
        self.traces_section = _CollapsibleSection("Traces", expanded=False)
        observation_layout = QFormLayout()
        self.base_excitation_trace = QCheckBox("Drive trace")
        self.base_excitation_trace.setToolTip(
            "Plot the source drive signal. Saved exposures can show several simultaneous drive traces."
        )
        observation_layout.addRow(self.base_excitation_trace)
        self.drive_state_trace = QCheckBox("Sequencer trace")
        self.drive_state_trace.setToolTip(
            "Show the active-coil / exposure-step sequence over the playback timeline."
        )
        self.drive_state_trace.setEnabled(bool(self._base_drive_ids))
        observation_layout.addRow(self.drive_state_trace)
        self.sensor_trace = QCheckBox("Sensor trace")
        self.sensor_trace.setToolTip(
            "Overlay selected sensor or probe field signals beneath the scripted map."
        )
        self.sensor_trace.setEnabled(bool(self._observation_choices))
        if not self._observation_choices:
            self.sensor_trace.setToolTip("Add a sensor or probe to the scene to collect sensor traces.")
        observation_layout.addRow(self.sensor_trace)

        self.sensor_trace_table = QTableWidget(len(self._observation_choices), 4)
        self.sensor_trace_table.setHorizontalHeaderLabels(["Trace", "Sensor / probe", "Sample", "Signal"])
        self.sensor_trace_table.setAlternatingRowColors(True)
        self.sensor_trace_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.sensor_trace_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sensor_trace_table.verticalHeader().setVisible(False)
        self.sensor_trace_table.setMinimumHeight(118)
        self.sensor_trace_table.setMaximumHeight(220)
        trace_header = self.sensor_trace_table.horizontalHeader()
        trace_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        trace_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        trace_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        trace_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for row, choice in enumerate(self._observation_choices):
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
            )
            enabled_item.setCheckState(Qt.CheckState.Unchecked)
            self.sensor_trace_table.setItem(row, 0, enabled_item)

            suffix = "sensor" if choice["kind"] == "sensor" else "probe"
            source_item = QTableWidgetItem(f"{choice['label']} — {suffix}")
            source_item.setData(Qt.ItemDataRole.UserRole, choice)
            self.sensor_trace_table.setItem(row, 1, source_item)

            if choice["kind"] == "sensor":
                sample_spin = QSpinBox(self.sensor_trace_table)
                sample_spin.setRange(1, max(1, int(choice.get("sample_count", 1))))
                sample_spin.setValue(1)
                sample_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sample_spin.setToolTip("Sensor sample point to trace")
                self.sensor_trace_table.setCellWidget(row, 2, sample_spin)
            else:
                sample_item = QTableWidgetItem("—")
                sample_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.sensor_trace_table.setItem(row, 2, sample_item)

            signal_combo = QComboBox(self.sensor_trace_table)
            signal_combo.setToolTip("Field signal to capture from this sensor or probe")
            self._populate_observation_signal_combo(signal_combo)
            self.sensor_trace_table.setCellWidget(row, 3, signal_combo)

        observation_layout.addRow(self.sensor_trace_table)

        self.brain_region_trace_tree: QTreeWidget | None = None
        self.brain_region_trace_select_major_button: QPushButton | None = None
        self.brain_region_trace_clear_button: QPushButton | None = None
        self._brain_trace_items_by_key: dict[str, QTreeWidgetItem] = {}
        self._brain_context_initialized = False
        # Build the optional Brain trace controls once even when this editor is
        # initially used for 2D/3D.  The unified Maps scripted mode moves one
        # protocol editor between views, so the Brain-only portion needs to be
        # revealable later without reconstructing the user's waveform settings.
        self.brain_trace_panel = QWidget()
        brain_trace_layout = QVBoxLayout(self.brain_trace_panel)
        brain_trace_layout.setContentsMargins(0, 8, 0, 0)
        brain_trace_layout.setSpacing(4)
        brain_trace_heading = QLabel("Brain region traces")
        brain_trace_heading.setStyleSheet("font-weight: 600;")
        brain_trace_layout.addWidget(brain_trace_heading)
        brain_trace_note = QLabel(
            "Check any LPBA40 rows to add their spatial RMS |B| traces to playback."
        )
        brain_trace_note.setObjectName("hint")
        brain_trace_note.setWordWrap(True)
        brain_trace_layout.addWidget(brain_trace_note)
        self.brain_region_trace_tree = QTreeWidget()
        self.brain_region_trace_tree.setHeaderLabel("Structure")
        self.brain_region_trace_tree.setAlternatingRowColors(True)
        self.brain_region_trace_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.brain_region_trace_tree.setMinimumHeight(220)
        self.brain_region_trace_tree.setMaximumHeight(360)
        brain_trace_layout.addWidget(self.brain_region_trace_tree)
        brain_trace_buttons = QHBoxLayout()
        self.brain_region_trace_select_major_button = QPushButton("Select major")
        self.brain_region_trace_clear_button = QPushButton("Clear")
        self.brain_region_trace_select_major_button.setToolTip(
            "Check the eight major LPBA40 region groups"
        )
        self.brain_region_trace_clear_button.setToolTip(
            "Clear all brain-region waveform traces"
        )
        self.brain_region_trace_select_major_button.clicked.connect(
            self._select_major_brain_region_traces
        )
        self.brain_region_trace_clear_button.clicked.connect(
            self._clear_brain_region_traces
        )
        brain_trace_buttons.addWidget(self.brain_region_trace_select_major_button)
        brain_trace_buttons.addWidget(self.brain_region_trace_clear_button)
        brain_trace_buttons.addStretch(1)
        brain_trace_layout.addLayout(brain_trace_buttons)
        observation_layout.addRow(self.brain_trace_panel)
        self.brain_trace_panel.setVisible(self.map_kind == "brain")
        if self.map_kind == "brain":
            self._populate_brain_region_trace_tree()
            self._brain_context_initialized = True

        self.traces_section.setContentLayout(observation_layout)
        content_layout.addWidget(self.traces_section)


        content_layout.addStretch(1)

        self.status = QLabel("Ready")
        self.status.setObjectName("hint")
        root.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.create_button = buttons.addButton(
            "Continue to export" if self.primary_action == "export" else "Create playback",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.create_button.setObjectName("primaryButton")
        self.create_button.clicked.connect(self._create_playback)
        self.export_video_button = buttons.addButton(
            "Export video…", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.export_video_button.setToolTip(
            "Configure and render a fixed-camera 3D waveform video"
        )
        # Video export now starts from the main 2D/3D map workflow so camera
        # control is shared with animated playback instead of living in this dialog.
        self.export_video_button.setVisible(False)
        self.export_video_button.setEnabled(False)
        self.export_video_button.clicked.connect(self._export_video)
        self.reset_defaults_button = buttons.addButton(
            "Reset defaults", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.reset_defaults_button.setToolTip(
            "Restore the built-in waveform settings and replace the remembered settings for this playback dialog"
        )
        self.reset_defaults_button.clicked.connect(self._reset_defaults)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        for waveform_button in (
            self.waveform_sine,
            self.waveform_square,
            self.waveform_triangle,
        ):
            waveform_button.toggled.connect(self._update_waveform_mode)
        self.waveform_duration.valueChanged.connect(self._sync_frames_from_playback)
        self.waveform_frequency.valueChanged.connect(self._sync_frames_from_playback)
        self.sensor_trace.toggled.connect(self._update_observation_controls)
        self._syncing_playback_controls = False

        self._wheel_filter = _EditorWheelScrollFilter(scroll)
        for editor in content.findChildren(QWidget):
            if isinstance(editor, (QDoubleSpinBox, QSpinBox, QComboBox)):
                editor.installEventFilter(self._wheel_filter)

        self._update_observation_controls()
        self.waveform_amplitude.setValue(self._starter_amplitude_value())
        self._update_waveform_mode()
        self._sync_frames_from_playback()
        if not self._drive_choices:
            self.create_button.setEnabled(False)
            self.export_video_button.setEnabled(False)
            self.status.setText("The scene does not contain a coil to drive.")
        capture_window_default_settings(self)

    def _reset_defaults(self, *_args) -> None:
        if not reset_window_to_default_settings(self):
            return
        if self.brain_region_trace_tree is not None:
            self._set_brain_region_trace_checks(
                set(self._default_brain_region_trace_keys)
            )
        self._update_observation_controls()
        self._update_drive_state_trace_availability()
        self._update_waveform_mode()
        self._sync_frames_from_playback()
        self.status.setText(
            "Defaults restored" if self._drive_choices
            else "The scene does not contain a coil to drive."
        )

    def configure_quick_current_source(
        self,
        *,
        mode: str,
        amplitude_ma: float,
        frequency_hz: float,
        duration_s: float,
        slew_ma_per_ms: float,
        active_coil_ids: list[str] | None = None,
    ) -> None:
        """Configure the lightweight Maps built-in current source.

        This is intentionally not an exposure protocol. Saved Exposures use the
        canonical step/drive-group source-load model; the quick source is only a
        one-off generated current applied to the current Scene-options coil set.
        """
        available_ids = {str(choice["id"]) for choice in self._drive_choices}
        selected_ids = [
            str(value)
            for value in (self._base_drive_ids if active_coil_ids is None else active_coil_ids)
            if str(value) in available_ids
        ]
        self.clear_runtime_exposure_protocol()
        context = dict(self.map_settings)
        context["drive_coil_ids"] = selected_ids
        self.set_playback_context(map_kind=self.map_kind, map_settings=context)
        self._set_waveform_mode(mode)
        self.waveform_amplitude.setValue(max(0.0, float(amplitude_ma)))
        self.waveform_frequency.setValue(float(frequency_hz))
        self.waveform_duration.setValue(float(duration_s))
        self.slew_rate.setValue(max(0.0, float(slew_ma_per_ms)))
        self._update_observation_controls()
        self._sync_frames_from_playback()

    def _persistent_ui_state(self) -> dict[str, Any]:
        """Return quick-source and trace state not covered by widget persistence."""
        return {
            "waveform_mode": self._waveform_mode(),
            "sensor_trace_enabled": bool(self.sensor_trace.isChecked()),
            "sensor_trace_sources": self._persistent_sensor_trace_rows(),
            "brain_region_traces": [
                str(region.get("key", ""))
                for region in self._brain_region_trace_settings()
                if str(region.get("key", ""))
            ],
        }

    def _restore_persistent_ui_state(self, state: dict[str, Any]) -> None:
        """Restore generated quick-source and viewer-local trace selections."""
        saved_mode = str(state.get("waveform_mode", "")).strip().lower()
        if saved_mode in {"sine", "square", "triangle"}:
            self._set_waveform_mode(saved_mode)

        saved_rows = state.get("sensor_trace_sources", [])
        saved_by_key: dict[str, dict[str, Any]] = {}
        if isinstance(saved_rows, list):
            for raw in saved_rows:
                if isinstance(raw, dict) and str(raw.get("key", "")):
                    saved_by_key[str(raw["key"])] = raw
        # Gracefully restore the previous source-only persistence shape when a
        # user already has waveform playback UI settings from an earlier build.
        if not saved_by_key:
            for value in state.get("observation_sources", []):
                saved_by_key[str(value)] = {"key": str(value), "component": "magnitude", "sample_index": 0}

        for row in range(self.sensor_trace_table.rowCount()):
            choice = self._sensor_trace_choice(row)
            key = self._observation_choice_key(choice)
            saved = saved_by_key.get(key)
            enabled_item = self.sensor_trace_table.item(row, 0)
            if enabled_item is not None:
                enabled_item.setCheckState(
                    Qt.CheckState.Checked if saved is not None else Qt.CheckState.Unchecked
                )
            if saved is None:
                continue
            signal_combo = self.sensor_trace_table.cellWidget(row, 3)
            if isinstance(signal_combo, QComboBox):
                index = signal_combo.findData(str(saved.get("component", "magnitude")))
                if index >= 0:
                    signal_combo.setCurrentIndex(index)
            sample_spin = self.sensor_trace_table.cellWidget(row, 2)
            if isinstance(sample_spin, QSpinBox):
                sample_spin.setValue(int(saved.get("sample_index", 0)) + 1)
        self.sensor_trace.setChecked(
            bool(state.get("sensor_trace_enabled", bool(saved_by_key)))
        )
        if self.brain_region_trace_tree is not None and "brain_region_traces" in state:
            self._set_brain_region_trace_checks(
                {str(value) for value in state.get("brain_region_traces", [])}
            )
        self._update_observation_controls()
        self._sync_frames_from_playback()

    def set_playback_context(
        self,
        *,
        map_kind: str,
        map_settings: dict[str, Any],
        camera_timeline: dict[str, Any] | None = None,
        initial_camera: dict[str, Any] | None = None,
        playback_fps: float | None = None,
        playback_speed: float | None = None,
    ) -> None:
        """Retarget the persistent quick-source/exposure renderer at a field view.

        Generator settings stay untouched. Only the viewer context (map settings,
        camera, active-coil default, and optional Brain trace tree) changes.
        """

        had_brain_context = bool(self._brain_context_initialized)
        previous_brain_keys = {
            str(region.get("key", ""))
            for region in self._brain_region_trace_settings()
            if str(region.get("key", ""))
        }
        normalized_kind = str(map_kind).strip().lower()
        if normalized_kind not in {"2d", "3d", "brain"}:
            raise ValueError(f"Unsupported waveform map kind: {map_kind!r}")
        self.map_kind = normalized_kind
        self.map_settings = dict(map_settings)
        self.camera_timeline = (
            copy.deepcopy(camera_timeline) if isinstance(camera_timeline, dict) else None
        )
        self.initial_camera = (
            copy.deepcopy(initial_camera) if isinstance(initial_camera, dict) else None
        )
        if playback_fps is not None:
            try:
                value = float(playback_fps)
            except (TypeError, ValueError):
                value = self.playback_fps_value
            if math.isfinite(value) and value > 0:
                self.playback_fps_value = value
        if playback_speed is not None:
            try:
                value = float(playback_speed)
            except (TypeError, ValueError):
                value = self.playback_speed_value
            if math.isfinite(value) and value > 0:
                self.playback_speed_value = value

        available_drive_ids = {str(choice["id"]) for choice in self._drive_choices}
        requested_drive_ids = [
            str(object_id)
            for object_id in self.map_settings.get("drive_coil_ids", [])
            if str(object_id) in available_drive_ids
        ]
        self._base_drive_ids = (
            requested_drive_ids
            if "drive_coil_ids" in self.map_settings
            else [
                str(choice["id"])
                for choice in self._drive_choices
                if bool(choice.get("enabled", True))
            ]
        )
        if self.brain_region_trace_tree is not None:
            if self.map_kind == "brain":
                self._populate_brain_region_trace_tree()
                if had_brain_context:
                    # Preserve an intentionally empty selection too; otherwise
                    # returning to Brain after Clear would re-check defaults.
                    self._set_brain_region_trace_checks(previous_brain_keys)
                self._brain_context_initialized = True
                self.brain_trace_panel.setVisible(True)
            else:
                self.brain_trace_panel.setVisible(False)
        self._update_observation_controls()
        self._update_drive_state_trace_availability()

    def prepare_playback_request(
        self,
        *,
        error_title: str = "Unable to create waveform playback",
    ) -> tuple[dict[str, Any], list[str]] | None:
        """Validate the persistent protocol controls and freeze one render request."""
        return self._prepare_gpu_request(error_title=error_title)

    def _choice_for_id(self, object_id: str) -> dict[str, Any] | None:
        return next((choice for choice in self._drive_choices if choice["id"] == object_id), None)

    def _base_selected_drive_ids(self) -> list[str]:
        available = {str(choice["id"]) for choice in self._drive_choices}
        return [object_id for object_id in self._base_drive_ids if object_id in available]

    def _selected_drive_ids(self) -> list[str]:
        if self._runtime_exposure_protocol is not None:
            available = {str(choice["id"]) for choice in self._drive_choices}
            return [
                coil_id
                for coil_id in exposure_coil_ids(self._runtime_exposure_protocol)
                if coil_id in available
            ]
        return self._base_selected_drive_ids()

    def _current_drive_choice(self) -> dict[str, Any] | None:
        ids = self._selected_drive_ids()
        return self._choice_for_id(ids[0]) if ids else None

    def _update_drive_state_trace_availability(self) -> None:
        """Allow the viewer Sequencer trace whenever at least one coil is active."""
        self.drive_state_trace.setEnabled(bool(self._selected_drive_ids()))

    def _populate_brain_region_trace_tree(self) -> None:
        tree = self.brain_region_trace_tree
        if tree is None:
            return
        tree.clear()
        self._brain_trace_items_by_key.clear()
        raw_root = self.map_settings.get("trace_region_tree")
        if not isinstance(raw_root, dict):
            return
        initial_keys = {
            str(region.get("key", ""))
            for region in self.map_settings.get("trace_regions", [])
            if isinstance(region, dict) and str(region.get("key", ""))
        }
        self._default_brain_region_trace_keys = set(initial_keys)

        def add_node(raw: dict[str, Any], parent: QTreeWidgetItem | None) -> None:
            key = str(raw.get("key", ""))
            name = str(raw.get("name", key or "Brain region"))
            label_ids = [int(value) for value in raw.get("label_ids", [])]
            item = QTreeWidgetItem([name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                0,
                Qt.CheckState.Checked
                if key in initial_keys
                else Qt.CheckState.Unchecked,
            )
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "key": key,
                "name": name,
                "label_ids": label_ids,
            })
            self._brain_trace_items_by_key[key] = item
            if parent is None:
                tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            children = raw.get("children", [])
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        add_node(child, item)

        add_node(raw_root, None)
        root = tree.topLevelItem(0)
        if root is not None:
            root.setExpanded(True)
            for index in range(root.childCount()):
                root.child(index).setExpanded(True)
        tree.resizeColumnToContents(0)

    def _brain_region_trace_settings(self) -> list[dict[str, Any]]:
        tree = self.brain_region_trace_tree
        if tree is None:
            return []
        selected: list[dict[str, Any]] = []
        for item in self._brain_trace_items_by_key.values():
            if item.checkState(0) != Qt.CheckState.Checked:
                continue
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(data, dict):
                continue
            label_ids = [int(value) for value in data.get("label_ids", [])]
            if not label_ids:
                continue
            selected.append(
                {
                    "key": str(data.get("key", "")),
                    "name": str(data.get("name", "Brain region")),
                    "label_ids": label_ids,
                }
            )
        return selected

    def _set_brain_region_trace_checks(self, checked_keys: set[str]) -> None:
        for key, item in self._brain_trace_items_by_key.items():
            item.setCheckState(
                0,
                Qt.CheckState.Checked
                if key in checked_keys
                else Qt.CheckState.Unchecked,
            )

    def _select_major_brain_region_traces(self, *_args) -> None:
        tree = self.brain_region_trace_tree
        root = tree.topLevelItem(0) if tree is not None else None
        if root is None:
            return
        keys: set[str] = set()
        for index in range(root.childCount()):
            data = root.child(index).data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                keys.add(str(data.get("key", "")))
        self._set_brain_region_trace_checks(keys)

    def _clear_brain_region_traces(self, *_args) -> None:
        self._set_brain_region_trace_checks(set())

    def _gpu_map_settings(self) -> dict[str, Any]:
        """Return the map-window display settings used by the GPU playback builder."""
        settings = dict(self.map_settings)
        if self.map_kind == "brain":
            trace_regions = self._brain_region_trace_settings()
            settings["trace_regions"] = trace_regions
            settings["scroll_observation_traces"] = len(trace_regions) > 1
            settings.pop("trace_region_tree", None)
        if self.map_kind in {"3d", "brain"}:
            settings["waveform_render_mode"] = str(
                settings.get("view_type", settings.get("waveform_render_mode", "slices"))
            )
            lower = float(settings.get("volume_opacity_lower", 0.15))
            upper = float(settings.get("volume_opacity_upper", 0.40))
            lower_level = float(settings.get("volume_opacity_lower_level", 0.0))
            upper_level = float(settings.get("volume_opacity_upper_level", 1.0))
            if bool(settings.get("intensity_opacity_enabled", True)) and lower_level > upper_level:
                raise StudioOperationError(
                    "Lower opacity intensity level cannot be greater than upper opacity intensity level."
                )
        return settings

    def _waveform_mode(self) -> str:
        button = self.waveform_choice_group.checkedButton()
        if button is None:
            return "sine"
        mode = str(button.property("waveformMode") or "sine").strip().lower()
        return mode if mode in {"sine", "square", "triangle"} else "sine"

    def _set_waveform_mode(self, mode: str) -> None:
        wanted = str(mode).strip().lower()
        mapping = {
            "sine": self.waveform_sine,
            "square": self.waveform_square,
            "triangle": self.waveform_triangle,
        }
        mapping.get(wanted, self.waveform_sine).setChecked(True)
        self._update_waveform_mode()

    def _update_waveform_mode(self, *_args) -> None:
        self._runtime_exposure_protocol = None
        self._runtime_exposure_metadata = {}
        self.generated_waveform_controls.setVisible(True)
        self._sync_frames_from_playback()

    def _starter_amplitude_value(self) -> float:
        """Return a useful generated current amplitude in mA."""
        choice = self._current_drive_choice()
        amplitude_a = abs(float(choice.get("drive_current_a", 0.0))) if choice else 0.01
        if not math.isfinite(amplitude_a) or amplitude_a <= 1e-12:
            amplitude_a = 0.01
        return amplitude_a * 1000.0

    def _generated_waveform_trace(self):
        mode = self._waveform_mode()
        frequency_hz = float(self.waveform_frequency.value())
        duration_s = float(self.waveform_duration.value())
        amplitude_display = float(self.waveform_amplitude.value())
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise StudioOperationError("Waveform frequency must be greater than zero.")
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise StudioOperationError("Waveform duration must be greater than zero.")
        if not math.isfinite(amplitude_display) or amplitude_display < 0.0:
            raise StudioOperationError("Waveform amplitude must be finite and non-negative.")

        cycles = frequency_hz * duration_s
        interval_count = max(32, int(math.ceil(cycles * 128.0)))
        if interval_count > 200_000:
            raise StudioOperationError(
                "This generated waveform requires too many internal samples. Reduce its frequency or duration."
            )
        amplitude = amplitude_display * 0.001  # mA -> A
        times = [duration_s * index / interval_count for index in range(interval_count + 1)]
        values: list[float] = []
        for time_s in times:
            phase = 2.0 * math.pi * frequency_hz * time_s
            sine = math.sin(phase)
            if mode == "sine":
                normalized = sine
            elif mode == "square":
                normalized = 1.0 if sine >= 0.0 else -1.0
            else:
                normalized = (2.0 / math.pi) * math.asin(sine)
            values.append(amplitude * normalized)
        return validate_trace(times, values, kind="current")

    @staticmethod
    def _observation_choice_key(choice: dict[str, Any] | None) -> str:
        if not isinstance(choice, dict):
            return ""
        return f"{choice.get('kind', '')}:{choice.get('id', '')}"

    def _sensor_trace_choice(self, row: int) -> dict[str, Any] | None:
        item = self.sensor_trace_table.item(row, 1)
        if item is None:
            return None
        choice = item.data(Qt.ItemDataRole.UserRole)
        return choice if isinstance(choice, dict) else None

    def _populate_observation_signal_combo(
        self, combo: QComboBox, *, preferred: str | None = None
    ) -> None:
        previous = preferred if preferred is not None else combo.currentData()
        combo.clear()
        field = str(self.map_settings.get("field", "B")).upper()
        combo.addItem(f"|{field}|", "magnitude")
        combo.addItem(f"{field}x", "x")
        combo.addItem(f"{field}y", "y")
        combo.addItem(f"{field}z", "z")
        combo.insertSeparator(combo.count())
        combo.addItem(f"|{field}x|", "abs_x")
        combo.addItem(f"|{field}y|", "abs_y")
        combo.addItem(f"|{field}z|", "abs_z")
        if previous is not None:
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)

    def _persistent_sensor_trace_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.sensor_trace_table.rowCount()):
            enabled_item = self.sensor_trace_table.item(row, 0)
            if enabled_item is None or enabled_item.checkState() != Qt.CheckState.Checked:
                continue
            choice = self._sensor_trace_choice(row)
            if choice is None:
                continue
            signal_combo = self.sensor_trace_table.cellWidget(row, 3)
            component = (
                signal_combo.currentData()
                if isinstance(signal_combo, QComboBox)
                else "magnitude"
            ) or "magnitude"
            sample_spin = self.sensor_trace_table.cellWidget(row, 2)
            sample_index = (sample_spin.value() - 1) if isinstance(sample_spin, QSpinBox) else 0
            rows.append(
                {
                    "key": self._observation_choice_key(choice),
                    "component": str(component),
                    "sample_index": int(sample_index),
                }
            )
        return rows

    def _selected_observation_choices(self) -> list[dict[str, Any]]:
        if not self.sensor_trace.isChecked():
            return []
        selected: list[dict[str, Any]] = []
        for row in range(self.sensor_trace_table.rowCount()):
            enabled_item = self.sensor_trace_table.item(row, 0)
            if enabled_item is None or enabled_item.checkState() != Qt.CheckState.Checked:
                continue
            choice = self._sensor_trace_choice(row)
            if choice is None:
                continue
            item = dict(choice)
            signal_combo = self.sensor_trace_table.cellWidget(row, 3)
            item["component"] = str(
                signal_combo.currentData()
                if isinstance(signal_combo, QComboBox)
                else "magnitude"
            )
            sample_spin = self.sensor_trace_table.cellWidget(row, 2)
            if item.get("kind") == "sensor":
                item["sample_index"] = (
                    sample_spin.value() - 1 if isinstance(sample_spin, QSpinBox) else 0
                )
            selected.append(item)
        return selected

    def _update_observation_controls(self, *_args) -> None:
        table_visible = bool(self.sensor_trace.isChecked() and self._observation_choices)
        self.sensor_trace_table.setVisible(table_visible)
        for row in range(self.sensor_trace_table.rowCount()):
            signal_combo = self.sensor_trace_table.cellWidget(row, 3)
            if isinstance(signal_combo, QComboBox):
                self._populate_observation_signal_combo(signal_combo)

    def _normalized_camera_timeline_for_duration(
        self, duration_s: float
    ) -> dict[str, Any] | None:
        if self.camera_timeline is None:
            return None
        default_camera = self.initial_camera
        if default_camera is None:
            points = self.camera_timeline.get("points")
            if isinstance(points, list) and points and isinstance(points[0], dict):
                candidate = points[0].get("camera")
                if isinstance(candidate, dict):
                    default_camera = candidate
        if not isinstance(default_camera, dict):
            return None
        timeline = normalize_camera_timeline(
            self.camera_timeline,
            duration_s=duration_s,
            default_camera=default_camera,
            view_mode=self.map_kind,
        )
        timeline["enabled"] = True
        if bool(self.camera_timeline.get("scripted", False)):
            points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
            authored_end = max(
                (float(point.get("time_s", 0.0)) for point in points if isinstance(point, dict)),
                default=0.0,
            )
            timeline["scripted"] = True
            if self._runtime_exposure_protocol is not None:
                timeline["exposure_preview_duration_s"] = float(duration_s)
            timeline["loop_waveform"] = bool(self.camera_timeline.get("loop_waveform", False))
            # The camera may finish before the waveform. In that case keep the
            # final camera pose and extend the scripted output until one waveform
            # pass reaches its natural end.
            timeline["duration_s"] = scripted_timeline_duration_s(
                timeline, fallback_s=duration_s
            )
        return timeline

    def _sampling_playback_speed(
        self, timeline: dict[str, Any] | None
    ) -> float:
        # Frame samples are uniform in physical waveform time. For a variable-speed
        # timeline, sample densely enough for the slowest segment so target FPS is
        # still available there; faster segments can safely skip source samples.
        if isinstance(timeline, dict) and timeline.get("enabled", False):
            points = timeline.get("points")
            speeds: list[float] = []
            if isinstance(points, list):
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    try:
                        speed = float(point.get("playback_speed", 1.0))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(speed) and speed > 0:
                        speeds.append(speed)
            if speeds:
                return max(1.0e-6, min(speeds))
        return max(1.0e-6, float(self.playback_speed_value))

    @staticmethod
    def _automatic_sample_multiplier(frame_count: int) -> int:
        # Preserve the former 6x analysis density when practical, then reduce it
        # automatically for long/high-FPS runs to stay inside the analysis cap.
        intervals = max(1, int(frame_count) - 1)
        maximum = max(1, (MAX_ANALYSIS_SAMPLES - 1) // intervals)
        return max(1, min(6, maximum))

    def _resolved_sample_counts(
        self, duration_s: float, timeline: dict[str, Any] | None = None
    ) -> tuple[int, int]:
        frames = playback_frame_count(
            duration_s,
            float(self.playback_fps_value),
            self._sampling_playback_speed(timeline),
        )
        multiplier = self._automatic_sample_multiplier(frames)
        samples = analysis_sample_count(frames, multiplier)
        return frames, samples

    def _sync_frames_from_playback(self, *_args) -> None:
        # Target FPS/playback speed now live in the parent map Camera section and
        # analysis sampling is automatic, so this dialog has no render summary to
        # maintain. Keep the hook because waveform editors already call it.
        return

    def set_runtime_exposure_protocol(
        self, protocol: dict[str, Any], *, metadata: dict[str, Any] | None = None
    ) -> None:
        """Use a canonical saved exposure as the Scripted Maps current source."""
        state = normalize_exposure_protocol(protocol)
        steps = state["sequence"]["steps"]
        if not steps:
            raise StudioOperationError("This exposure does not contain a drive step.")
        duration = float(state["duration_s"])
        if not math.isfinite(duration) or duration <= 0.0:
            raise StudioOperationError("Exposure duration must be finite and positive.")
        coil_ids = exposure_coil_ids(state)
        if not coil_ids:
            raise StudioOperationError("This exposure does not drive any coils.")
        available = {str(choice["id"]) for choice in self._drive_choices}
        missing = [coil_id for coil_id in coil_ids if coil_id not in available]
        if missing:
            raise StudioOperationError(
                "This exposure references coil(s) that are unavailable in this playback scene: "
                + ", ".join(missing)
            )
        self._runtime_exposure_protocol = copy.deepcopy(state)
        self._runtime_exposure_metadata = copy.deepcopy(metadata or {})
        self._update_drive_state_trace_availability()
        self._sync_frames_from_playback()

    def clear_runtime_exposure_protocol(self) -> None:
        self._runtime_exposure_protocol = None
        self._runtime_exposure_metadata = {}
        if hasattr(self, "drive_state_trace"):
            self._update_drive_state_trace_availability()

    def _normalized_trace(self):
        if self._runtime_exposure_protocol is not None:
            duration = min(
                float(self._runtime_exposure_protocol["duration_s"]),
                float(self._runtime_exposure_metadata.get("preview_duration_s", self._runtime_exposure_protocol["duration_s"])),
            )
            return validate_trace([0.0, duration], [0.0, 0.0], kind="current")
        return self._generated_waveform_trace()

    def _waveform_run_metadata(self, duration_s: float) -> dict[str, Any]:
        """Describe the finite source pass for quick-source or exposure playback."""

        if self._runtime_exposure_protocol is not None:
            metadata = copy.deepcopy(self._runtime_exposure_metadata)
            sequence = self._runtime_exposure_protocol["sequence"]
            metadata["physical_duration_s"] = self._runtime_exposure_protocol["duration_s"]
            metadata["preview_duration_s"] = float(duration_s)
            metadata.setdefault("mode", "exposure")
            metadata.setdefault("pass_count", 1)
            metadata.setdefault("step_count", len(sequence["steps"]))
            metadata.setdefault(
                "drive_group_count",
                sum(len(step["drive_groups"]) for step in sequence["steps"]),
            )
            metadata.setdefault("sequence_loop", bool(sequence.get("loop", False)))
            return metadata
        frequency_hz = float(self.waveform_frequency.value())
        return {
            "mode": self._waveform_mode(),
            "pass_count": 1,
            "frequency_hz": frequency_hz,
            "cycle_count": frequency_hz * float(duration_s),
        }

    def _slew_rate_per_second(self) -> float:
        if self._runtime_exposure_protocol is not None:
            return 0.0
        # 1 mA/ms is exactly 1 A/s.
        return max(0.0, float(self.slew_rate.value()))

    def _observation_settings(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for choice in self._selected_observation_choices():
            item = {
                "kind": str(choice["kind"]),
                "id": str(choice["id"]),
                "label": str(choice.get("label") or choice["id"]),
                "component": str(choice.get("component") or "magnitude"),
            }
            if choice["kind"] == "sensor":
                item["sample_index"] = int(choice.get("sample_index", 0))
            result.append(item)
        return result

    def _set_calculation_buttons_enabled(self, enabled: bool) -> None:
        available = bool(enabled and self._drive_choices)
        self.create_button.setEnabled(available)
        self.export_video_button.setEnabled(False)

    def _prepare_gpu_request(
        self,
        *,
        error_title: str,
    ) -> tuple[dict[str, Any], list[str]] | None:
        """Validate the controls and freeze everything needed for one GPU build."""

        if not self._drive_choices:
            return None
        try:
            trace = self._normalized_trace()
            coil_ids = self._selected_drive_ids()
            if not coil_ids:
                raise StudioOperationError(
                    "Choose at least one Active coil in the map window's Scene options."
                )
            exposure_protocol = (
                copy.deepcopy(self._runtime_exposure_protocol)
                if self._runtime_exposure_protocol is not None
                else None
            )
            if exposure_protocol is not None:
                electrical_by_coil = {}
                sequence_steps = [
                    {
                        "duration_s": float(step["duration_s"]),
                        "active_coil_ids": list(
                            dict.fromkeys(
                                coil_id
                                for group in step["drive_groups"]
                                for coil_id in group["coil_ids"]
                            )
                        ),
                    }
                    for step in exposure_protocol["sequence"]["steps"]
                ]
                sequence_loop = bool(exposure_protocol["sequence"].get("loop", False))
                initial_drive_value = 0.0
                slew_rate_per_s = 0.0
                include_excitation_trace = bool(self.base_excitation_trace.isChecked())
                include_drive_state_trace = bool(self.drive_state_trace.isChecked())
            else:
                # Quick Maps playback is deliberately a simple current source.
                # Load wiring, voltage sources, and timed sequencing are authored
                # only as saved Exposures.
                electrical_by_coil = {}
                sequence_steps = None
                sequence_loop = False
                initial_drive_value = 0.0
                slew_rate_per_s = self._slew_rate_per_second()
                include_excitation_trace = bool(self.base_excitation_trace.isChecked())
                include_drive_state_trace = bool(self.drive_state_trace.isChecked())
            duration_s = float(trace.time_s[-1] - trace.time_s[0])
            normalized_timeline = self._normalized_camera_timeline_for_duration(
                duration_s
            )
            frame_count, analysis_samples = self._resolved_sample_counts(
                duration_s, normalized_timeline
            )
            request = {
                "map_kind": self.map_kind,
                "map_settings": copy.deepcopy(self._gpu_map_settings()),
                "coil_ids": list(coil_ids),
                "controlled_coil_ids": [
                    str(choice["id"]) for choice in self._drive_choices
                ],
                "trace": trace,
                "frame_count": frame_count,
                "analysis_samples": analysis_samples,
                "playback_fps": float(self.playback_fps_value),
                "electrical_by_coil": copy.deepcopy(electrical_by_coil),
                "initial_drive_value": initial_drive_value,
                "slew_rate_per_s": slew_rate_per_s,
                "sequence_steps": copy.deepcopy(sequence_steps),
                "sequence_loop": sequence_loop,
                "exposure_protocol": exposure_protocol,
                "waveform_metadata": self._waveform_run_metadata(duration_s),
                "observation": copy.deepcopy(self._observation_settings()),
                "include_excitation_trace": include_excitation_trace,
                "include_sequencer_trace": include_drive_state_trace,
                "initial_camera": copy.deepcopy(self.initial_camera),
                "waveform_duration_s": duration_s,
                "real_time_multiplier": float(self.playback_speed_value),
                "camera_timeline": copy.deepcopy(normalized_timeline),
            }
            return request, coil_ids
        except Exception as error:  # noqa: BLE001 - GUI validation boundary
            QMessageBox.warning(self, error_title, str(error))
            self.status.setText("Waveform settings need attention")
            return None

    def _calculate_gpu_payload(
        self,
        *,
        window_title: str,
        error_title: str,
    ) -> tuple[dict[str, Any], list[str]] | None:
        """Build the shared cached-field payload for playback or video export."""
        prepared = self._prepare_gpu_request(error_title=error_title)
        if prepared is None:
            return None
        request, coil_ids = prepared
        self._set_calculation_buttons_enabled(False)
        progress: QProgressDialog | None = None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.status.setText("Preparing waveform playback…")
            progress = QProgressDialog(
                "Preparing waveform playback…",
                "Cancel",
                0,
                2000,
                self,
            )
            progress.setWindowTitle(window_title)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()
            centre_window_on_parent(progress, self)
            progress.raise_()
            progress.activateWindow()
            QApplication.processEvents()

            def update_progress(update: dict[str, Any]) -> None:
                if progress is None:
                    return
                if progress.wasCanceled():
                    raise StudioOperationError("Waveform field calculation cancelled.")
                progress.setMaximum(max(1, int(update.get("total", 2000) or 2000)))
                progress.setValue(max(0, int(update.get("completed", 0) or 0)))
                progress.setLabelText(
                    str(update.get("stage") or "Preparing waveform playback…")
                )
                QApplication.processEvents()
                if progress.wasCanceled():
                    raise StudioOperationError("Waveform field calculation cancelled.")

            result = build_prepared_gpu_payload(
                self.adapter,
                request,
                progress_callback=update_progress,
            )
            self.status.setText("Waveform playback ready")
            return result, coil_ids
        except Exception as error:  # noqa: BLE001 - GUI calculation boundary
            if "cancelled" in str(error).lower():
                self.status.setText("Waveform calculation cancelled")
            else:
                QMessageBox.warning(self, error_title, str(error))
                self.status.setText("Waveform calculation failed")
            return None
        finally:
            if progress is not None:
                progress.close()
            QApplication.restoreOverrideCursor()
            self._set_calculation_buttons_enabled(True)

    def _set_result_label(self, coil_ids: list[str]) -> None:
        if self._runtime_exposure_protocol is not None:
            name = str(self._runtime_exposure_metadata.get("exposure_name", "") or "Exposure")
            self.result_label = f"{name} — {len(coil_ids)} driven coil" + ("s" if len(coil_ids) != 1 else "")
            return
        labels = [
            str(self._choice_for_id(object_id).get("label", object_id))
            for object_id in coil_ids
        ]
        self.result_label = "Waveform — " + " + ".join(labels)

    def _create_playback(self, *_args) -> None:
        export_mode = self.primary_action == "export"
        if self.defer_payload_build and not export_mode:
            prepared = self._prepare_gpu_request(
                error_title="Unable to create waveform playback"
            )
            if prepared is None:
                return
            request, coil_ids = prepared
            self.result_figure = None
            self.result_gpu_payload = None
            self.result_gpu_request = request
            self._set_result_label(coil_ids)
            self.status.setText("Waveform job ready")
            self.accept()
            return
        calculated = self._calculate_gpu_payload(
            window_title="Prepare video export" if export_mode else "Waveform playback",
            error_title="Unable to prepare video export" if export_mode else "Unable to create waveform playback",
        )
        if calculated is None:
            return
        result, coil_ids = calculated
        self.result_figure = None
        self._set_result_label(coil_ids)
        if export_mode:
            dialog = WaveformVideoExportDialog(result, parent=self)
            centre_window_on_parent(dialog, self)
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.output_path is not None:
                self.status.setText(f"Video exported to {dialog.output_path}")
                self.result_gpu_payload = None
                self.accept()
            return
        self.result_gpu_payload = result
        self.accept()

    def _export_video(self, *_args) -> None:
        calculated = self._calculate_gpu_payload(
            window_title="Prepare video export",
            error_title="Unable to prepare video export",
        )
        if calculated is None:
            return
        payload, _coil_ids = calculated
        dialog = WaveformVideoExportDialog(payload, parent=self)
        centre_window_on_parent(dialog, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.output_path is not None:
            self.status.setText(f"Video exported to {dialog.output_path}")
