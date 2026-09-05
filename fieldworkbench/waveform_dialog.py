"""Qt configuration dialog for quasi-static waveform playback."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
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
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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
from .studio_adapter import StudioAdapter, StudioOperationError
from .waveform import (
    MAX_ANALYSIS_SAMPLES,
    analysis_sample_count,
    build_prepared_gpu_payload,
    dac_8bit_voltage_trace,
    dac_file_is_text,
    drive_coil_catalog,
    observation_catalog,
    parse_dac_8bit_values,
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
    if str(extra.get("waveform_mode", "")).lower() == "custom":
        rows = extra.get("trace_rows")
        if isinstance(rows, list):
            times: list[float] = []
            for row in rows:
                if not isinstance(row, list) or not row:
                    continue
                try:
                    times.append(float(row[0]))
                except (TypeError, ValueError):
                    pass
            if len(times) >= 2:
                unit_state = controls.get("time_unit")
                unit = str(unit_state.get("text", "s")) if isinstance(unit_state, dict) else "s"
                scale = {"s": 1.0, "ms": 1.0e-3, "µs": 1.0e-6, "us": 1.0e-6}.get(unit, 1.0)
                candidate = (max(times) - min(times)) * scale
                if candidate > 0 and math.isfinite(candidate):
                    duration = candidate
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


class DACImportDialog(QDialog):
    """Import a legacy/raw 8-bit DAC sequence as waveform points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Import DAC waveform")
        self.resize(560, 300)
        self.result_trace = None

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("Choose an 8-bit DAC file…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        file_layout.addWidget(self.file_path, 1)
        file_layout.addWidget(browse)
        form.addRow("DAC file", file_row)

        self.point_delay = _double_spin(3.0, 0.000001, 1e9, 6, " ms")
        self.cycle_delay = _double_spin(3.0, 0.0, 1e9, 6, " ms")
        self.lower_voltage = _double_spin(-5.0, -1e9, 1e9, 6, " V")
        self.upper_voltage = _double_spin(5.0, -1e9, 1e9, 6, " V")
        form.addRow("Point delay", self.point_delay)
        form.addRow("Cycle delay", self.cycle_delay)
        form.addRow("Lower voltage (code 0)", self.lower_voltage)
        form.addRow("Upper voltage (code 255)", self.upper_voltage)

        note = QLabel(
            "Each 8-bit value becomes one DAC voltage command. Samples are spaced by Point delay; "
            "after the last sample the output is commanded to 0 V for Cycle delay. "
            "Legacy .DAC files are read as text; raw .BIN/.DAT/.RAW files are read byte-for-byte. Text files may contain decimal or 0x hexadecimal values from 0 to 255. "
            "If the first value equals the number of values that follow, it is treated as a point-count header and discarded."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        root.addWidget(note)
        root.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.import_button = buttons.addButton("Import", QDialogButtonBox.ButtonRole.AcceptRole)
        self.import_button.setObjectName("primaryButton")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._import)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse(self, *_args) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose DAC file",
            "",
            "Legacy DAC text (*.dac *.txt *.csv *.tsv);;Raw 8-bit binary (*.bin *.dat *.raw);;All files (*)",
        )
        if not path:
            return
        self.file_path.setText(path)
        self.import_button.setEnabled(True)

    def _import(self, *_args) -> None:
        path_text = self.file_path.text().strip()
        if not path_text:
            return
        try:
            path = Path(path_text)
            payload = path.read_bytes()
            # The original Complex/DAC standard uses ordinary DOS text files
            # with one decimal 0..255 value per line.  Treat .dac as text;
            # otherwise ASCII characters such as spaces/newlines are mistaken
            # for raw DAC bytes and produce a badly distorted waveform.
            text_hint = dac_file_is_text(path.name)
            values = parse_dac_8bit_values(payload, text_hint=text_hint)
            self.result_trace = dac_8bit_voltage_trace(
                values,
                point_delay_ms=float(self.point_delay.value()),
                cycle_delay_ms=float(self.cycle_delay.value()),
                lower_voltage_v=float(self.lower_voltage.value()),
                upper_voltage_v=float(self.upper_voltage.value()),
            )
            self.accept()
        except Exception as error:  # noqa: BLE001 - GUI file/input boundary
            QMessageBox.warning(self, "Unable to import DAC waveform", str(error))


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

        # Drive ------------------------------------------------------------
        self.drive_section = _CollapsibleSection("Drive", expanded=False)
        drive_section_layout = QVBoxLayout()
        drive_layout = QGridLayout()
        self.drive_kind = QComboBox()
        self.drive_kind.addItem("Voltage (RL)", "voltage")
        self.drive_kind.addItem("Current", "current")
        self.time_unit = QComboBox()
        self.time_unit.addItems(["ms", "s"])
        self.value_unit = QComboBox()
        self.value_unit.addItems(["V", "mV"])
        self.slew_rate = _double_spin(100.0, 0.0, 1e12, 6)
        self.slew_rate.setToolTip(
            "Maximum drive transition rate. Set 0 for an ideal instantaneous step."
        )
        self.initial_drive_label = QLabel("Initial voltage")
        self.initial_drive = _double_spin(0.0, -1e9, 1e9, 6)
        self.initial_drive.setToolTip(
            "Drive value immediately before playback starts. In voltage mode it also sets the RL starting current."
        )
        drive_layout.addWidget(QLabel("Drive type"), 0, 0)
        drive_layout.addWidget(self.drive_kind, 0, 1)
        drive_layout.addWidget(QLabel("Time unit"), 0, 2)
        drive_layout.addWidget(self.time_unit, 0, 3)
        drive_layout.addWidget(QLabel("Value unit"), 1, 2)
        drive_layout.addWidget(self.value_unit, 1, 3)
        drive_layout.addWidget(QLabel("Slew rate"), 1, 0)
        drive_layout.addWidget(self.slew_rate, 1, 1)
        drive_layout.addWidget(self.initial_drive_label, 2, 0)
        drive_layout.addWidget(self.initial_drive, 2, 1)
        drive_section_layout.addLayout(drive_layout)

        self.electrical_group = QWidget()
        electrical_layout = QGridLayout(self.electrical_group)
        electrical_layout.setContentsMargins(0, 8, 0, 0)
        electrical_heading = QLabel("<b>Voltage electrical model</b>")
        electrical_layout.addWidget(electrical_heading, 0, 0, 1, 2)
        self.use_model_resistance = QCheckBox("Use model resistance")
        self.use_model_resistance.setChecked(True)
        self.model_resistance_label = QLabel("—")
        self.manual_resistance = _double_spin(1.0, 0.0, 1e9, 6, " Ω")
        self.inductance = _double_spin(10.0, 0.000001, 1e9, 6, " mH")
        electrical_layout.addWidget(self.use_model_resistance, 1, 0)
        electrical_layout.addWidget(self.model_resistance_label, 1, 1)
        electrical_layout.addWidget(QLabel("Manual resistance"), 2, 0)
        electrical_layout.addWidget(self.manual_resistance, 2, 1)
        electrical_layout.addWidget(QLabel("Inductance"), 3, 0)
        electrical_layout.addWidget(self.inductance, 3, 1)
        electrical_note = QLabel(
            "Use each coil's model resistance or enter one shared resistance. Inductance is shared across driven coils. "
            "Mutual inductance is not included."
        )
        electrical_note.setWordWrap(True)
        electrical_note.setObjectName("hint")
        electrical_layout.addWidget(electrical_note, 4, 0, 1, 2)
        drive_section_layout.addWidget(self.electrical_group)
        self.drive_section.setContentLayout(drive_section_layout)
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
        self.waveform_custom = QCheckBox("Custom")
        for button, mode in (
            (self.waveform_sine, "sine"),
            (self.waveform_square, "square"),
            (self.waveform_triangle, "triangle"),
            (self.waveform_custom, "custom"),
        ):
            self.waveform_choice_group.addButton(button)
            button.setProperty("waveformMode", mode)
            # The selected mode is persisted explicitly in the dynamic state so
            # an exclusive checkbox group cannot be restored into an invalid mix.
            button.setProperty("persistUiState", False)
        self.waveform_sine.setChecked(True)
        waveform_layout.addWidget(self.waveform_sine, 0, 0)
        waveform_layout.addWidget(self.waveform_square, 1, 0)
        waveform_layout.addWidget(self.waveform_triangle, 2, 0)
        waveform_layout.addWidget(self.waveform_custom, 3, 0)

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
        waveform_layout.addWidget(self.generated_waveform_controls, 0, 1, 4, 1)
        self.waveform_section.setContentLayout(waveform_layout)
        content_layout.addWidget(self.waveform_section)

        # Custom wave points ----------------------------------------------
        self.wave_points_section = _CollapsibleSection("Wave points", expanded=True)
        wave_points_layout = QVBoxLayout()
        wave_points_help = QLabel(
            "Enter time/value pairs for a custom waveform. Each point updates the drive value; "
            "the value is held until the next point. CSV import uses the first two numeric columns."
        )
        wave_points_help.setWordWrap(True)
        wave_points_help.setObjectName("hint")
        wave_points_layout.addWidget(wave_points_help)
        self.trace_table = QTableWidget(0, 2)
        self.trace_table.setAlternatingRowColors(True)
        self.trace_table.horizontalHeader().setStretchLastSection(True)
        self.trace_table.setMinimumHeight(180)
        wave_points_layout.addWidget(self.trace_table, 1)
        wave_point_buttons = QHBoxLayout()
        add_row = QPushButton("Add row")
        add_row.clicked.connect(self._add_row)
        remove_row = QPushButton("Remove selected")
        remove_row.clicked.connect(self._remove_selected_rows)
        paste = QPushButton("Paste")
        paste.clicked.connect(self._paste_trace)
        load_csv = QPushButton("Load CSV…")
        load_csv.clicked.connect(self._load_csv)
        import_dac = QPushButton("Import DAC…")
        import_dac.setToolTip("Convert an 8-bit DAC sample file into custom waveform points")
        import_dac.clicked.connect(self._import_dac)
        wave_point_buttons.addWidget(add_row)
        wave_point_buttons.addWidget(remove_row)
        wave_point_buttons.addWidget(paste)
        wave_point_buttons.addWidget(load_csv)
        wave_point_buttons.addWidget(import_dac)
        wave_point_buttons.addStretch(1)
        wave_points_layout.addLayout(wave_point_buttons)
        self.wave_points_section.setContentLayout(wave_points_layout)
        content_layout.addWidget(self.wave_points_section)

        # Coil sequencer ---------------------------------------------------
        self.coil_sequencer_section = _CollapsibleSection("Coil sequencer", expanded=False)
        coil_layout = QVBoxLayout()
        base_labels = [
            str(choice.get("label") or choice["id"])
            for choice in self._drive_choices
            if str(choice["id"]) in set(self._base_drive_ids)
        ]
        base_summary = QLabel(
            "Default active coils: " + (", ".join(base_labels) if base_labels else "none selected")
        )
        base_summary.setWordWrap(True)
        base_summary.setObjectName("hint")
        base_summary.setToolTip(
            "The default active-coil selection comes from Scene options in the 2D/3D map window."
        )
        coil_layout.addWidget(base_summary)
        self.advanced_sequencer = QCheckBox("Advanced sequencer")
        self.advanced_sequencer.setToolTip(
            "Use timed steps to choose which coils receive the drive waveform during playback."
        )
        coil_layout.addWidget(self.advanced_sequencer)
        self.sequence_panel = QWidget()
        sequence_layout = QVBoxLayout(self.sequence_panel)
        sequence_layout.setContentsMargins(0, 4, 0, 0)
        sequence_note = QLabel(
            "Each step sets a duration and active coils. Loop sequence repeats the step pattern."
        )
        sequence_note.setWordWrap(True)
        sequence_note.setObjectName("hint")
        sequence_layout.addWidget(sequence_note)
        self.sequence_table = QTableWidget(0, 1 + len(self._drive_choices))
        self.sequence_table.setHorizontalHeaderLabels(
            ["Duration (ms)"] + [str(choice["label"]) for choice in self._drive_choices]
        )
        self.sequence_table.setAlternatingRowColors(True)
        self.sequence_table.setMinimumHeight(145)
        self.sequence_table.horizontalHeader().setStretchLastSection(False)
        sequence_layout.addWidget(self.sequence_table)
        sequence_buttons = QHBoxLayout()
        add_step = QPushButton("Add step")
        add_step.clicked.connect(self._add_sequence_step)
        remove_step = QPushButton("Remove selected")
        remove_step.clicked.connect(self._remove_sequence_steps)
        self.sequence_loop = QCheckBox("Loop sequence")
        self.sequence_loop.setChecked(True)
        sequence_buttons.addWidget(add_step)
        sequence_buttons.addWidget(remove_step)
        sequence_buttons.addWidget(self.sequence_loop)
        sequence_buttons.addStretch(1)
        sequence_layout.addLayout(sequence_buttons)
        self.sequence_panel.setVisible(False)
        self._advanced_sequencer_initialized = False
        coil_layout.addWidget(self.sequence_panel)
        self.coil_sequencer_section.setContentLayout(coil_layout)
        content_layout.addWidget(self.coil_sequencer_section)

        # Output traces ----------------------------------------------------
        self.traces_section = _CollapsibleSection("Traces", expanded=False)
        observation_layout = QFormLayout()
        self.base_excitation_trace = QCheckBox("Base excitation trace")
        self.base_excitation_trace.setToolTip(
            "Plot the actual drive output after the configured slew limit."
        )
        observation_layout.addRow(self.base_excitation_trace)
        self.sequencer_trace = QCheckBox("Sequencer trace")
        self.sequencer_trace.setToolTip(
            "Show the resolved sequencer step and active-coil state over the playback timeline."
        )
        self.sequencer_trace.setEnabled(False)
        observation_layout.addRow(self.sequencer_trace)
        self.observation_sources = QListWidget()
        self.observation_sources.setAlternatingRowColors(True)
        self.observation_sources.setMaximumHeight(132)
        self.observation_sources.setToolTip(
            "Select sensors or physical probes to overlay as synchronized output traces."
        )
        for choice in self._observation_choices:
            suffix = "sensor" if choice["kind"] == "sensor" else "probe"
            item = QListWidgetItem(f"{choice['label']} — {suffix}")
            item.setData(Qt.ItemDataRole.UserRole, choice)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.observation_sources.addItem(item)
        if self._observation_choices:
            observation_layout.addRow("Observer sources", self.observation_sources)
        else:
            self.observation_sources.hide()
            self.no_observation_sources = QLabel("Add a sensor or probe to collect traces")
            self.no_observation_sources.setEnabled(False)
            observation_layout.addRow("Observer sources", self.no_observation_sources)
        self.observation_sample = QSpinBox()
        self.observation_sample.setMinimum(1)
        self.observation_sample.setMaximum(1)
        self.observation_component = QComboBox()
        self.observation_sample_label = QLabel("Sensor sample")
        observation_layout.addRow(self.observation_sample_label, self.observation_sample)
        observation_layout.addRow("Signal", self.observation_component)

        self.brain_region_trace_tree: QTreeWidget | None = None
        self.brain_region_trace_select_major_button: QPushButton | None = None
        self.brain_region_trace_clear_button: QPushButton | None = None
        self._brain_trace_items_by_key: dict[str, QTreeWidgetItem] = {}
        if self.map_kind == "brain":
            brain_trace_panel = QWidget()
            brain_trace_layout = QVBoxLayout(brain_trace_panel)
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
            observation_layout.addRow(brain_trace_panel)
            self._populate_brain_region_trace_tree()

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

        self.drive_kind.currentIndexChanged.connect(self._update_drive_mode)
        for waveform_button in (
            self.waveform_sine,
            self.waveform_square,
            self.waveform_triangle,
            self.waveform_custom,
        ):
            waveform_button.toggled.connect(self._update_waveform_mode)
        self.waveform_duration.valueChanged.connect(self._sync_frames_from_playback)
        self.waveform_frequency.valueChanged.connect(self._sync_frames_from_playback)
        self.advanced_sequencer.toggled.connect(self._toggle_advanced_sequencer)
        self.use_model_resistance.toggled.connect(self._update_resistance_controls)
        self.observation_sources.itemChanged.connect(self._update_observation_controls)
        self.time_unit.currentTextChanged.connect(self._update_table_headers)
        self.time_unit.currentTextChanged.connect(self._update_slew_rate_suffix)
        self.time_unit.currentTextChanged.connect(self._sync_frames_from_playback)
        self.value_unit.currentTextChanged.connect(self._update_table_headers)
        self.value_unit.currentTextChanged.connect(self._update_slew_rate_suffix)
        self.trace_table.itemChanged.connect(self._sync_frames_from_playback)
        self._syncing_playback_controls = False

        self._wheel_filter = _EditorWheelScrollFilter(scroll)
        for editor in content.findChildren(QWidget):
            if isinstance(editor, (QDoubleSpinBox, QSpinBox, QComboBox)):
                editor.installEventFilter(self._wheel_filter)

        self._add_sequence_step(active_ids=self._base_drive_ids)
        self._update_drive_mode()
        self._update_slew_rate_suffix()
        self._update_resistance()
        self._update_observation_controls()
        self._populate_starter_trace()
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
        self._update_drive_mode()
        self._update_slew_rate_suffix()
        self._update_resistance()
        self._update_observation_controls()
        self._update_waveform_mode()
        self._sync_frames_from_playback()
        self.status.setText(
            "Defaults restored" if self._drive_choices
            else "The scene does not contain a coil to drive."
        )

    def _persistent_ui_state(self) -> dict[str, Any]:
        """Return waveform-specific state that ordinary widget persistence cannot capture."""
        trace_rows: list[list[str]] = []
        for row in range(self.trace_table.rowCount()):
            time_item = self.trace_table.item(row, 0)
            value_item = self.trace_table.item(row, 1)
            trace_rows.append([
                time_item.text() if time_item is not None else "",
                value_item.text() if value_item is not None else "",
            ])
        return {
            "waveform_mode": self._waveform_mode(),
            "trace_rows": trace_rows,
            "sequence_steps": [
                {
                    "duration_ms": float(step["duration_s"]) * 1000.0,
                    "active_coil_ids": list(step.get("active_coil_ids", [])),
                }
                for step in self._sequence_steps()
            ],
            "advanced_initialized": bool(self._advanced_sequencer_initialized),
            "observation_sources": [
                f"{choice.get('kind', '')}:{choice.get('id', '')}"
                for choice in self._selected_observation_choices()
            ],
            "brain_region_traces": [
                str(region.get("key", ""))
                for region in self._brain_region_trace_settings()
                if str(region.get("key", ""))
            ],
        }

    def _restore_persistent_ui_state(self, state: dict[str, Any]) -> None:
        """Restore custom points/coils/sequencer after all dynamic controls exist."""
        available_ids = {str(choice["id"]) for choice in self._drive_choices}

        rows = state.get("trace_rows", [])
        saved_mode = str(state.get("waveform_mode", "")).strip().lower()
        if saved_mode in {"sine", "square", "triangle", "custom"}:
            self._set_waveform_mode(saved_mode)
        elif isinstance(rows, list) and len(rows) >= 2:
            # Pre-0.13.0.148 playback dialogs only supported custom point sets.
            # Preserve remembered work when migrating those settings forward.
            self._set_waveform_mode("custom")
        if isinstance(rows, list) and len(rows) >= 2:
            cleaned: list[tuple[str, str]] = []
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    cleaned.append((str(row[0]), str(row[1])))
            if len(cleaned) >= 2:
                self._set_trace_rows(cleaned)

        saved_steps = state.get("sequence_steps", [])
        if isinstance(saved_steps, list) and saved_steps:
            self.sequence_table.setRowCount(0)
            for raw_step in saved_steps:
                if not isinstance(raw_step, dict):
                    continue
                active_ids = [
                    str(object_id) for object_id in raw_step.get("active_coil_ids", [])
                    if str(object_id) in available_ids
                ]
                self._add_sequence_step(active_ids=active_ids)
                row = self.sequence_table.rowCount() - 1
                duration = self.sequence_table.cellWidget(row, 0)
                if isinstance(duration, QDoubleSpinBox):
                    try:
                        duration.setValue(float(raw_step.get("duration_ms", 50.0)))
                    except (TypeError, ValueError):
                        pass
            if self.sequence_table.rowCount() == 0:
                self._add_sequence_step(active_ids=self._base_drive_ids)

        self._advanced_sequencer_initialized = bool(
            state.get("advanced_initialized", self._advanced_sequencer_initialized)
        )
        saved_sources = {str(value) for value in state.get("observation_sources", [])}
        for row in range(self.observation_sources.count()):
            item = self.observation_sources.item(row)
            choice = item.data(Qt.ItemDataRole.UserRole)
            key = f"{choice.get('kind', '')}:{choice.get('id', '')}" if isinstance(choice, dict) else ""
            item.setCheckState(Qt.CheckState.Checked if key in saved_sources else Qt.CheckState.Unchecked)
        if self.brain_region_trace_tree is not None and "brain_region_traces" in state:
            self._set_brain_region_trace_checks(
                {str(value) for value in state.get("brain_region_traces", [])}
            )
        self._update_observation_controls()
        self._update_resistance()
        self._sync_frames_from_playback()

    def _choice_for_id(self, object_id: str) -> dict[str, Any] | None:
        return next((choice for choice in self._drive_choices if choice["id"] == object_id), None)

    def _base_selected_drive_ids(self) -> list[str]:
        available = {str(choice["id"]) for choice in self._drive_choices}
        return [object_id for object_id in self._base_drive_ids if object_id in available]

    def _sequence_steps(self) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for row in range(self.sequence_table.rowCount()):
            duration_widget = self.sequence_table.cellWidget(row, 0)
            if not isinstance(duration_widget, QDoubleSpinBox):
                continue
            active: list[str] = []
            for column, choice in enumerate(self._drive_choices, start=1):
                checkbox = self.sequence_table.cellWidget(row, column)
                if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                    active.append(str(choice["id"]))
            steps.append({"duration_s": float(duration_widget.value()) / 1000.0, "active_coil_ids": active})
        return steps

    def _selected_drive_ids(self) -> list[str]:
        if not self.advanced_sequencer.isChecked():
            return self._base_selected_drive_ids()
        used = {
            str(object_id)
            for step in self._sequence_steps()
            for object_id in step.get("active_coil_ids", [])
        }
        return [str(choice["id"]) for choice in self._drive_choices if str(choice["id"]) in used]

    def _current_drive_choice(self) -> dict[str, Any] | None:
        ids = self._selected_drive_ids()
        return self._choice_for_id(ids[0]) if ids else None

    def _drive_selection_changed(self, *_args) -> None:
        self._update_resistance()

    def _toggle_advanced_sequencer(self, checked: bool) -> None:
        if checked and not self._advanced_sequencer_initialized and self.sequence_table.rowCount() == 1:
            selected = set(self._base_selected_drive_ids())
            for column, choice in enumerate(self._drive_choices, start=1):
                checkbox = self.sequence_table.cellWidget(0, column)
                if isinstance(checkbox, QCheckBox):
                    checkbox.setChecked(str(choice["id"]) in selected)
            self._advanced_sequencer_initialized = True
        self.sequence_panel.setVisible(bool(checked))
        self.sequencer_trace.setEnabled(bool(checked))
        self._update_resistance()

    def _add_sequence_step(self, *_args, active_ids: list[str] | None = None) -> None:
        row = self.sequence_table.rowCount()
        self.sequence_table.insertRow(row)
        self.sequence_table.setVerticalHeaderItem(row, QTableWidgetItem(f"Step {row + 1}"))
        duration = _double_spin(50.0, 0.001, 1e9, 3, " ms")
        duration.setSingleStep(10.0)
        if hasattr(self, "_wheel_filter"):
            duration.installEventFilter(self._wheel_filter)
        self.sequence_table.setCellWidget(row, 0, duration)
        selected = set(active_ids or [])
        for column, choice in enumerate(self._drive_choices, start=1):
            checkbox = QCheckBox()
            checkbox.setChecked(str(choice["id"]) in selected)
            checkbox.setStyleSheet("margin-left:8px;")
            checkbox.stateChanged.connect(self._drive_selection_changed)
            self.sequence_table.setCellWidget(row, column, checkbox)
        self._update_resistance()

    def _remove_sequence_steps(self, *_args) -> None:
        rows = sorted({index.row() for index in self.sequence_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.sequence_table.removeRow(row)
        for row in range(self.sequence_table.rowCount()):
            self.sequence_table.setVerticalHeaderItem(row, QTableWidgetItem(f"Step {row + 1}"))
        self._update_resistance()

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
        return mode if mode in {"sine", "square", "triangle", "custom"} else "sine"

    def _set_waveform_mode(self, mode: str) -> None:
        wanted = str(mode).strip().lower()
        mapping = {
            "sine": self.waveform_sine,
            "square": self.waveform_square,
            "triangle": self.waveform_triangle,
            "custom": self.waveform_custom,
        }
        mapping.get(wanted, self.waveform_sine).setChecked(True)
        self._update_waveform_mode()

    def _update_waveform_mode(self, *_args) -> None:
        custom = self._waveform_mode() == "custom"
        self.wave_points_section.setVisible(custom)
        self.generated_waveform_controls.setVisible(not custom)
        self._sync_frames_from_playback()

    def _starter_amplitude_value(self) -> float:
        """Return a useful generated/custom starter amplitude in the displayed unit."""
        choice = self._current_drive_choice()
        if self.drive_kind.currentData() == "voltage":
            drive_current = abs(float(choice.get("drive_current_a", 0.0))) if choice else 0.0
            resistance = choice.get("resistance_ohm") if choice else None
            amplitude_v = drive_current * float(resistance) if resistance is not None else 0.0
            if not math.isfinite(amplitude_v) or amplitude_v <= 1e-12:
                amplitude_v = 1.0
            return amplitude_v * (1000.0 if self.value_unit.currentText() == "mV" else 1.0)
        amplitude_a = abs(float(choice.get("drive_current_a", 0.0))) if choice else 0.01
        if not math.isfinite(amplitude_a) or amplitude_a <= 1e-12:
            amplitude_a = 0.01
        return amplitude_a * (1000.0 if self.value_unit.currentText() == "mA" else 1.0)

    def _generated_waveform_trace(self):
        mode = self._waveform_mode()
        if mode == "custom":
            raise StudioOperationError("Custom waveform mode requires wave points.")
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
                "This generated waveform requires too many internal wave points. Reduce its frequency or duration."
            )
        value_scale = (
            0.001
            if self.value_unit.currentText() in {"mA", "mV"}
            else 1.0
        )
        amplitude = amplitude_display * value_scale
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
        return validate_trace(times, values, kind=str(self.drive_kind.currentData()))

    def _populate_starter_trace(self) -> None:
        """Provide an editable low-frequency custom example rather than a blank table."""
        amplitude = self._starter_amplitude_value()
        rows = [
            (0.0, 0.0),
            (250.0, amplitude),
            (500.0, 0.0),
            (750.0, -amplitude),
            (1000.0, 0.0),
        ]
        self.trace_table.setRowCount(len(rows))
        for row, (time_value, drive_value) in enumerate(rows):
            self.trace_table.setItem(row, 0, QTableWidgetItem(f"{time_value:.9g}"))
            self.trace_table.setItem(row, 1, QTableWidgetItem(f"{drive_value:.9g}"))
        self._update_table_headers()

    def _update_table_headers(self, *_args) -> None:
        value_name = "Current" if self.drive_kind.currentData() == "current" else "Voltage"
        self.trace_table.setHorizontalHeaderLabels(
            [f"Time ({self.time_unit.currentText()})", f"{value_name} ({self.value_unit.currentText()})"]
        )

    def _update_slew_rate_suffix(self, *_args) -> None:
        self.slew_rate.setSuffix(f" {self.value_unit.currentText()}/{self.time_unit.currentText()}")
        self.initial_drive.setSuffix(f" {self.value_unit.currentText()}")
        self.waveform_amplitude.setSuffix(f" {self.value_unit.currentText()}")

    def _update_drive_mode(self, *_args) -> None:
        voltage = self.drive_kind.currentData() == "voltage"
        old_unit = self.value_unit.currentText()
        self.value_unit.blockSignals(True)
        self.value_unit.clear()
        self.value_unit.addItems(["V", "mV"] if voltage else ["mA", "A"])
        if old_unit in [self.value_unit.itemText(index) for index in range(self.value_unit.count())]:
            self.value_unit.setCurrentText(old_unit)
        self.value_unit.blockSignals(False)
        self.electrical_group.setEnabled(voltage)
        self.initial_drive_label.setText("Initial voltage" if voltage else "Initial current")
        self.initial_drive.setSuffix(f" {self.value_unit.currentText()}")
        self._update_table_headers()
        self._update_slew_rate_suffix()
        self._update_resistance_controls()

    def _update_resistance(self, *_args) -> None:
        choices = [self._choice_for_id(object_id) for object_id in self._selected_drive_ids()]
        choices = [choice for choice in choices if choice is not None]
        resistances: list[float] = []
        all_available = bool(choices)
        for choice in choices:
            resistance = choice.get("resistance_ohm")
            if resistance is None or not math.isfinite(float(resistance)):
                all_available = False
                break
            resistances.append(float(resistance))
        if not all_available:
            self.model_resistance_label.setText(
                "Select driven coils" if not choices else "Model resistance unavailable for one or more selected coils"
            )
            self.use_model_resistance.setChecked(False)
            self.use_model_resistance.setEnabled(False)
        else:
            low, high = min(resistances), max(resistances)
            if len(resistances) == 1:
                text = f"Model: {low:.7g} Ω"
            elif math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-12):
                text = f"Models: {low:.7g} Ω ({len(resistances)} coils)"
            else:
                text = f"Models: {low:.7g}–{high:.7g} Ω across {len(resistances)} coils"
            self.model_resistance_label.setText(text)
            # Keep the user's manual resistance independent from the model value.
            # The generic window-state restore applies the saved checkbox/manual
            # controls before _restore_persistent_ui_state() rebuilds the dynamic
            # sequencer. Rebuilding those rows calls this method, so blindly
            # copying the model resistance here used to overwrite a restored
            # manual value (and could do the same whenever coil selection changed).
            # Seed the inactive manual editor from the model only while model
            # resistance is actually selected.
            if self.use_model_resistance.isChecked():
                self.manual_resistance.setValue(float(sum(resistances) / len(resistances)))
            self.use_model_resistance.setEnabled(True)
        self._update_resistance_controls()

    def _update_resistance_controls(self, *_args) -> None:
        voltage = self.drive_kind.currentData() == "voltage"
        self.manual_resistance.setEnabled(voltage and not self.use_model_resistance.isChecked())

    def _selected_observation_choices(self) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for row in range(self.observation_sources.count()):
            item = self.observation_sources.item(row)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            choice = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(choice, dict):
                selected.append(choice)
        return selected

    def _update_observation_controls(self, *_args) -> None:
        selected = self._selected_observation_choices()
        previous_component = self.observation_component.currentData()
        self.observation_component.clear()
        field = str(self.map_settings.get("field", "B")).upper()
        self.observation_component.addItem(f"Magnitude |{field}|", "magnitude")
        self.observation_component.addItem(f"X magnitude |{field}x|", "abs_x")
        self.observation_component.addItem(f"Y magnitude |{field}y|", "abs_y")
        self.observation_component.addItem(f"Z magnitude |{field}z|", "abs_z")
        self.observation_component.insertSeparator(self.observation_component.count())
        self.observation_component.addItem(f"X signed {field}x", "x")
        self.observation_component.addItem(f"Y signed {field}y", "y")
        self.observation_component.addItem(f"Z signed {field}z", "z")
        if previous_component is not None:
            index = self.observation_component.findData(previous_component)
            if index >= 0:
                self.observation_component.setCurrentIndex(index)
        self.observation_component.setEnabled(bool(selected))

        sensors = [choice for choice in selected if choice.get("kind") == "sensor"]
        if sensors:
            # One sample number is shared across selected ordinary sensors. Limit
            # the range to the smallest selected sensor so every trace is valid.
            sample_count = min(max(1, int(choice.get("sample_count", 1))) for choice in sensors)
            self.observation_sample.setRange(1, sample_count)
            self.observation_sample.setEnabled(sample_count > 1)
            self.observation_sample_label.setEnabled(sample_count > 1)
        else:
            self.observation_sample.setRange(1, 1)
            self.observation_sample.setEnabled(False)
            self.observation_sample_label.setEnabled(False)

    def _trace_duration_seconds(self) -> float | None:
        if self._waveform_mode() != "custom":
            duration = float(self.waveform_duration.value())
            return duration if math.isfinite(duration) and duration > 0.0 else None
        times: list[float] = []
        for row in range(self.trace_table.rowCount()):
            item = self.trace_table.item(row, 0)
            if item is None or not item.text().strip():
                continue
            try:
                times.append(float(item.text().strip()))
            except ValueError:
                return None
        if len(times) < 2:
            return None
        duration = max(times) - min(times)
        if not math.isfinite(duration) or duration <= 0.0:
            return None
        return duration * (0.001 if self.time_unit.currentText() == "ms" else 1.0)

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

    def _add_row(self, *_args) -> None:
        row = self.trace_table.rowCount()
        self.trace_table.insertRow(row)
        if row:
            previous_time = self.trace_table.item(row - 1, 0)
            if previous_time is not None:
                try:
                    guess = float(previous_time.text()) + 1.0
                    self.trace_table.setItem(row, 0, QTableWidgetItem(f"{guess:.9g}"))
                except ValueError:
                    pass
        self.trace_table.setCurrentCell(row, 0)
        self._sync_frames_from_playback()

    def _remove_selected_rows(self, *_args) -> None:
        rows = sorted({index.row() for index in self.trace_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.trace_table.removeRow(row)
        self._sync_frames_from_playback()

    @staticmethod
    def _parse_two_column_text(text: str) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for raw_line in str(text).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = [part.strip() for part in line.split("\t")]
            elif ";" in line:
                parts = [part.strip() for part in line.split(";")]
            else:
                parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                continue
            rows.append((parts[0], parts[1]))
        return rows

    def _set_trace_rows(self, rows: list[tuple[str, str]]) -> None:
        if len(rows) < 2:
            raise StudioOperationError("The imported waveform does not contain at least two numeric rows.")
        self.trace_table.blockSignals(True)
        self.trace_table.setUpdatesEnabled(False)
        try:
            self.trace_table.setRowCount(len(rows))
            for row_index, (time_value, drive_value) in enumerate(rows):
                self.trace_table.setItem(row_index, 0, QTableWidgetItem(str(time_value)))
                self.trace_table.setItem(row_index, 1, QTableWidgetItem(str(drive_value)))
        finally:
            self.trace_table.setUpdatesEnabled(True)
            self.trace_table.blockSignals(False)
        self._sync_frames_from_playback()

    def _paste_trace(self, *_args) -> None:
        try:
            self._set_waveform_mode("custom")
            rows = self._parse_two_column_text(QApplication.clipboard().text())
            self._set_trace_rows(rows)
        except Exception as error:  # noqa: BLE001 - GUI input boundary
            QMessageBox.warning(self, "Unable to paste wave points", str(error))

    def _load_csv(self, *_args) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load wave points",
            "",
            "CSV / delimited text (*.csv *.txt *.tsv);;All files (*)",
        )
        if not path:
            return
        try:
            self._set_waveform_mode("custom")
            text = Path(path).read_text(encoding="utf-8-sig")
            rows = self._parse_two_column_text(text)
            if len(rows) < 2:
                # Retry with csv.reader for quoted/less ordinary CSV files.
                dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
                parsed: list[tuple[str, str]] = []
                for parts in csv.reader(text.splitlines(), dialect):
                    if len(parts) < 2:
                        continue
                    try:
                        float(parts[0])
                        float(parts[1])
                    except ValueError:
                        continue
                    parsed.append((parts[0].strip(), parts[1].strip()))
                rows = parsed
            self._set_trace_rows(rows)
            self.status.setText(f"Loaded {len(rows)} wave points from {Path(path).name}")
        except Exception as error:  # noqa: BLE001 - GUI file boundary
            QMessageBox.warning(self, "Unable to load wave points", str(error))

    def _import_dac(self, *_args) -> None:
        dialog = DACImportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_trace is None:
            return
        trace = dialog.result_trace
        self._set_waveform_mode("custom")
        voltage_index = self.drive_kind.findData("voltage")
        if voltage_index >= 0:
            self.drive_kind.setCurrentIndex(voltage_index)
        self.time_unit.setCurrentText("ms")
        self.value_unit.setCurrentText("V")
        rows = [
            (f"{float(time_s) * 1000.0:.12g}", f"{float(voltage_v):.12g}")
            for time_s, voltage_v in zip(trace.time_s, trace.value)
        ]
        try:
            self._set_trace_rows(rows)
            source = Path(dialog.file_path.text()).name
            imported_samples = max(0, len(trace.value) - (2 if float(dialog.cycle_delay.value()) > 0.0 else 1))
            self.status.setText(
                f"Imported {imported_samples:,} DAC samples from {source} "
                f"({dialog.point_delay.value():.6g} ms/point + {dialog.cycle_delay.value():.6g} ms at 0 V)"
            )
        except Exception as error:  # noqa: BLE001 - GUI input boundary
            QMessageBox.warning(self, "Unable to import DAC waveform", str(error))

    def _normalized_trace(self):
        if self._waveform_mode() != "custom":
            return self._generated_waveform_trace()
        times: list[float] = []
        values: list[float] = []
        for row in range(self.trace_table.rowCount()):
            time_item = self.trace_table.item(row, 0)
            value_item = self.trace_table.item(row, 1)
            time_text = time_item.text().strip() if time_item is not None else ""
            value_text = value_item.text().strip() if value_item is not None else ""
            if not time_text and not value_text:
                continue
            if not time_text or not value_text:
                raise StudioOperationError(f"Waveform row {row + 1} is incomplete.")
            try:
                times.append(float(time_text))
                values.append(float(value_text))
            except ValueError as error:
                raise StudioOperationError(f"Waveform row {row + 1} must contain numeric values.") from error

        time_scale = 0.001 if self.time_unit.currentText() == "ms" else 1.0
        if self.drive_kind.currentData() == "current":
            value_scale = 0.001 if self.value_unit.currentText() == "mA" else 1.0
        else:
            value_scale = 0.001 if self.value_unit.currentText() == "mV" else 1.0
        return validate_trace(
            [value * time_scale for value in times],
            [value * value_scale for value in values],
            kind=str(self.drive_kind.currentData()),
        )

    def _waveform_run_metadata(self, duration_s: float) -> dict[str, Any]:
        """Describe the finite source pass without inferring cycles for custom data."""

        mode = self._waveform_mode()
        metadata: dict[str, Any] = {"mode": mode, "pass_count": 1}
        if mode != "custom":
            frequency_hz = float(self.waveform_frequency.value())
            metadata.update(
                {
                    "frequency_hz": frequency_hz,
                    "cycle_count": frequency_hz * float(duration_s),
                }
            )
        return metadata

    def _slew_rate_per_second(self) -> float:
        value = float(self.slew_rate.value())
        if value == 0.0:
            return 0.0
        time_scale = 0.001 if self.time_unit.currentText() == "ms" else 1.0
        if self.drive_kind.currentData() == "current":
            value_scale = 0.001 if self.value_unit.currentText() == "mA" else 1.0
        else:
            value_scale = 0.001 if self.value_unit.currentText() == "mV" else 1.0
        return value * value_scale / time_scale

    def _observation_settings(self) -> list[dict[str, Any]]:
        component = self.observation_component.currentData() or "magnitude"
        result: list[dict[str, Any]] = []
        for choice in self._selected_observation_choices():
            item = {
                "kind": str(choice["kind"]),
                "id": str(choice["id"]),
                "label": str(choice.get("label") or choice["id"]),
                "component": component,
            }
            if choice["kind"] == "sensor":
                item["sample_index"] = self.observation_sample.value() - 1
            result.append(item)
        return result

    def _initial_drive_value_si(self) -> float:
        value = float(self.initial_drive.value())
        if self.drive_kind.currentData() == "current":
            scale = 0.001 if self.value_unit.currentText() == "mA" else 1.0
        else:
            scale = 0.001 if self.value_unit.currentText() == "mV" else 1.0
        return value * scale

    def _electrical_settings(self, coil_ids: list[str]) -> dict[str, dict[str, float | None]]:
        if self.drive_kind.currentData() != "voltage":
            return {object_id: {} for object_id in coil_ids}
        inductance_h = float(self.inductance.value()) / 1000.0
        initial_voltage = self._initial_drive_value_si()
        result: dict[str, dict[str, float | None]] = {}
        for object_id in coil_ids:
            choice = self._choice_for_id(object_id)
            if self.use_model_resistance.isChecked():
                resistance = choice.get("resistance_ohm") if choice else None
                if resistance is None or not math.isfinite(float(resistance)):
                    raise StudioOperationError(
                        f"{choice['label'] if choice else object_id} does not have an automatic resistance value."
                    )
                resistance_ohm = float(resistance)
            else:
                resistance_ohm = float(self.manual_resistance.value())
            if resistance_ohm <= 1e-15:
                if abs(initial_voltage) > 1e-15:
                    raise StudioOperationError(
                        "A non-zero initial voltage cannot define a steady RL starting current when resistance is zero."
                    )
                initial_current_a = 0.0
            else:
                initial_current_a = initial_voltage / resistance_ohm
            result[object_id] = {
                "resistance_ohm": resistance_ohm,
                "inductance_h": inductance_h,
                "initial_current_a": initial_current_a,
            }
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
                    "Choose at least one Active coil in the map window's Scene options, or add an active coil to the advanced sequencer."
                )
            electrical_by_coil = self._electrical_settings(coil_ids)
            sequence_steps = (
                self._sequence_steps() if self.advanced_sequencer.isChecked() else None
            )
            if self.advanced_sequencer.isChecked() and not sequence_steps:
                raise StudioOperationError(
                    "Advanced coil sequencing requires at least one step."
                )
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
                "initial_drive_value": self._initial_drive_value_si(),
                "slew_rate_per_s": self._slew_rate_per_second(),
                "sequence_steps": copy.deepcopy(sequence_steps),
                "sequence_loop": (
                    bool(self.sequence_loop.isChecked())
                    if sequence_steps is not None
                    else False
                ),
                "waveform_metadata": self._waveform_run_metadata(duration_s),
                "observation": copy.deepcopy(self._observation_settings()),
                "include_excitation_trace": bool(
                    self.base_excitation_trace.isChecked()
                ),
                "include_sequencer_trace": bool(
                    self.advanced_sequencer.isChecked()
                    and self.sequencer_trace.isChecked()
                ),
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
        if self.advanced_sequencer.isChecked():
            self.result_label = (
                f"Waveform — sequenced {len(coil_ids)} "
                f"coil{'s' if len(coil_ids) != 1 else ''}"
            )
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
