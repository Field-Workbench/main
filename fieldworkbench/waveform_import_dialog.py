"""Explicit waveform importer chooser and declarative format builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, QStandardPaths, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .waveform_formats import (
    FORMAT_PROFILE_SUFFIX,
    ParsedWaveformData,
    WaveformFormatProfile,
    WaveformFormatStore,
    builtin_waveform_format_profiles,
    number_format_labels,
    parse_waveform_file,
)
from .waveform_io import (
    DEFAULT_WAVEFORM_IO_REGISTRY,
    NATIVE_WAVEFORM_READER_ID,
    WaveformDocument,
    WaveformIOError,
    WaveformSeries,
    time_scale_to_seconds,
    read_project_waveform_documents,
    waveform_signal_to_payload,
)
from .waveform_recipe import AUTHORING_RECIPE_KEY, encoded_segment, make_authoring_recipe
from .waveform_table_io import (
    TabularSignalSpec,
    TextTable,
    documents_from_text_table,
    parse_text_table,
)
from .window_utils import centre_window_on_parent, enable_standard_window_controls, install_editor_wheel_guard


_PROFILE_ROLE = int(Qt.ItemDataRole.UserRole)
_KIND_ROLE = _PROFILE_ROLE + 1


def _profile_directory() -> Path:
    root = str(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation) or "").strip()
    if not root:
        root = str(Path.home() / ".field_workbench")
    return Path(root) / "waveform_formats"


def _combo_set_data(combo: QComboBox, value: Any) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _number_combo() -> QComboBox:
    combo = QComboBox()
    for label, value in number_format_labels():
        combo.addItem(label, value)
    return combo




class _FormatBuilderWheelScrollFilter(QObject):
    """Route every wheel gesture in the format form to the outer scroll area.

    The builder is intentionally a long configuration page.  Wheel gestures are
    therefore navigation only: combo boxes, spin boxes, the preview table, and
    other child widgets must not consume them or silently change a setting.
    """

    def __init__(self, scroll_area: QScrollArea):
        super().__init__(scroll_area)
        self.scroll_area = scroll_area

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if event.type() != QEvent.Type.Wheel:
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

def _compact_double(value: float, minimum: float = -1.0e15, maximum: float = 1.0e15) -> QDoubleSpinBox:
    control = QDoubleSpinBox()
    control.setDecimals(9)
    control.setRange(minimum, maximum)
    control.setValue(float(value))
    control.setKeyboardTracking(False)
    return control


class WaveformFormatBuilderDialog(QDialog):
    """Create/edit a portable declarative text or binary waveform format."""

    def __init__(self, profile: WaveformFormatProfile | None = None, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Waveform format builder")
        self.resize(820, 760)
        self.result_profile: WaveformFormatProfile | None = None
        self._profile_id = (profile.profile_id if profile is not None else WaveformFormatProfile.new().profile_id)

        outer = QVBoxLayout(self)
        intro = QLabel(
            "Describe how samples are stored in a text or binary file. The format profile contains only "
            "file layout, timing policy, and raw-to-engineering conversion. Timing and conversion values can be "
            "hard-coded in the profile or requested from the user when a file is imported. The result becomes a normal "
            "Current or Voltage waveform before it enters the project catalogue."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        general = QGroupBox("Format identity")
        general_form = QFormLayout(general)
        self.name_edit = QLineEdit()
        self.description_edit = QLineEdit()
        self.suffixes_edit = QLineEdit()
        self.suffixes_edit.setPlaceholderText(".dac, .txt, .bin")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Text records", "text")
        self.mode_combo.addItem("Binary records", "binary")
        general_form.addRow("Name", self.name_edit)
        general_form.addRow("Description", self.description_edit)
        general_form.addRow("Filename suffixes", self.suffixes_edit)
        general_form.addRow("Storage", self.mode_combo)
        layout.addWidget(general)

        representation = QGroupBox("Sample representation")
        rep_form = QFormLayout(representation)
        self.number_combo = _number_combo()
        self.byte_order_combo = QComboBox()
        self.byte_order_combo.addItem("Little endian", "little")
        self.byte_order_combo.addItem("Big endian", "big")
        rep_form.addRow("Raw sample type", self.number_combo)
        rep_form.addRow("Byte order", self.byte_order_combo)
        layout.addWidget(representation)

        self.storage_stack = QStackedWidget()
        layout.addWidget(self.storage_stack)

        text_group = QGroupBox("Text record layout")
        text_form = QFormLayout(text_group)
        self.encoding_combo = QComboBox()
        for encoding in ("utf-8-sig", "utf-8", "ascii", "latin-1"):
            self.encoding_combo.addItem(encoding, encoding)
        self.record_sep_combo = QComboBox()
        for label, value in (
            ("Any newline", "newline"),
            ("LF (\\n)", "lf"),
            ("CRLF (\\r\\n)", "crlf"),
            ("Custom", "custom"),
        ):
            self.record_sep_combo.addItem(label, value)
        self.custom_record_sep = QLineEdit()
        self.custom_record_sep.setPlaceholderText("Use escapes such as \\n or \\t")
        self.field_sep_combo = QComboBox()
        for label, value in (
            ("Whole record is the value", "none"),
            ("Whitespace", "whitespace"),
            ("Comma", "comma"),
            ("Tab", "tab"),
            ("Semicolon", "semicolon"),
            ("Custom", "custom"),
        ):
            self.field_sep_combo.addItem(label, value)
        self.custom_field_sep = QLineEdit()
        self.custom_field_sep.setPlaceholderText("e.g. | or \\t")
        self.comment_prefix_edit = QLineEdit()
        self.comment_prefix_edit.setPlaceholderText("Optional, e.g. #")
        self.skip_records_spin = QSpinBox()
        self.skip_records_spin.setRange(0, 10_000_000)
        self.value_field_spin = QSpinBox()
        self.value_field_spin.setRange(1, 1024)
        text_form.addRow("Text encoding", self.encoding_combo)
        text_form.addRow("Record separator", self.record_sep_combo)
        text_form.addRow("Custom record separator", self.custom_record_sep)
        text_form.addRow("Field separator", self.field_sep_combo)
        text_form.addRow("Custom field separator", self.custom_field_sep)
        text_form.addRow("Comment prefix", self.comment_prefix_edit)
        text_form.addRow("Skip leading records", self.skip_records_spin)
        text_form.addRow("Sample value field", self.value_field_spin)
        self.storage_stack.addWidget(text_group)

        binary_group = QGroupBox("Binary record layout")
        binary_form = QFormLayout(binary_group)
        self.header_bytes_spin = QSpinBox()
        self.header_bytes_spin.setRange(0, 2_000_000_000)
        self.record_bytes_spin = QSpinBox()
        self.record_bytes_spin.setRange(0, 2_000_000_000)
        self.record_bytes_spin.setSpecialValueText("Auto / packed")
        self.value_offset_spin = QSpinBox()
        self.value_offset_spin.setRange(0, 2_000_000_000)
        binary_form.addRow("Header bytes to skip", self.header_bytes_spin)
        binary_form.addRow("Bytes per record", self.record_bytes_spin)
        binary_form.addRow("Sample value byte offset", self.value_offset_spin)
        binary_note = QLabel(
            "A record size of 0 uses the smallest packed record that contains the configured fields. "
            "Offsets are measured from the start of each record."
        )
        binary_note.setWordWrap(True)
        binary_note.setObjectName("hint")
        binary_form.addRow(binary_note)
        self.storage_stack.addWidget(binary_group)

        timing = QGroupBox("Timeline")
        timing_form = QFormLayout(timing)
        self.time_mode_combo = QComboBox()
        self.time_mode_combo.addItem("Fixed sample interval in encoding", "fixed_interval")
        self.time_mode_combo.addItem("User-entered sample interval at import", "runtime_interval")
        self.time_mode_combo.addItem("Time stored in each record", "field")
        fixed_time_row = QWidget()
        fixed_time_layout = QHBoxLayout(fixed_time_row)
        fixed_time_layout.setContentsMargins(0, 0, 0, 0)
        self.sample_interval_spin = _compact_double(1.0, 1e-12, 1e15)
        self.sample_interval_unit = QComboBox()
        self.sample_interval_unit.addItems(["s", "ms", "us", "ns"])
        fixed_time_layout.addWidget(self.sample_interval_spin, 1)
        fixed_time_layout.addWidget(self.sample_interval_unit)
        self.text_time_field_spin = QSpinBox()
        self.text_time_field_spin.setRange(1, 1024)
        binary_time_row = QWidget()
        binary_time_layout = QHBoxLayout(binary_time_row)
        binary_time_layout.setContentsMargins(0, 0, 0, 0)
        self.binary_time_offset_spin = QSpinBox()
        self.binary_time_offset_spin.setRange(0, 2_000_000_000)
        self.binary_time_format_combo = _number_combo()
        binary_time_layout.addWidget(self.binary_time_offset_spin)
        binary_time_layout.addWidget(self.binary_time_format_combo, 1)
        self.explicit_time_unit = QComboBox()
        self.explicit_time_unit.addItems(["s", "ms", "us", "ns"])
        timing_form.addRow("Time source", self.time_mode_combo)
        timing_form.addRow("Interval / default", fixed_time_row)
        timing_form.addRow("Text time field", self.text_time_field_spin)
        timing_form.addRow("Binary time offset/type", binary_time_row)
        timing_form.addRow("Stored time unit", self.explicit_time_unit)
        layout.addWidget(timing)

        count = QGroupBox("Sample count / termination")
        count_form = QFormLayout(count)
        self.count_mode_combo = QComboBox()
        self.count_mode_combo.addItem("Read records until EOF", "eof")
        self.count_mode_combo.addItem("Fixed number of samples", "fixed")
        self.count_mode_combo.addItem("Count stored in a text record", "text_record")
        self.count_mode_combo.addItem("Count stored in binary header", "binary_header")
        self.fixed_count_spin = QSpinBox()
        self.fixed_count_spin.setRange(1, 2_000_000_000)
        text_count_row = QWidget()
        text_count_layout = QHBoxLayout(text_count_row)
        text_count_layout.setContentsMargins(0, 0, 0, 0)
        self.count_record_spin = QSpinBox()
        self.count_record_spin.setRange(1, 10_000_000)
        self.count_field_spin = QSpinBox()
        self.count_field_spin.setRange(1, 1024)
        text_count_layout.addWidget(QLabel("record"))
        text_count_layout.addWidget(self.count_record_spin)
        text_count_layout.addWidget(QLabel("field"))
        text_count_layout.addWidget(self.count_field_spin)
        binary_count_row = QWidget()
        binary_count_layout = QHBoxLayout(binary_count_row)
        binary_count_layout.setContentsMargins(0, 0, 0, 0)
        self.count_offset_spin = QSpinBox()
        self.count_offset_spin.setRange(0, 2_000_000_000)
        self.count_format_combo = _number_combo()
        self.count_order_combo = QComboBox()
        self.count_order_combo.addItem("Little endian", "little")
        self.count_order_combo.addItem("Big endian", "big")
        binary_count_layout.addWidget(QLabel("offset"))
        binary_count_layout.addWidget(self.count_offset_spin)
        binary_count_layout.addWidget(self.count_format_combo, 1)
        binary_count_layout.addWidget(self.count_order_combo)
        count_form.addRow("Count source", self.count_mode_combo)
        count_form.addRow("Fixed sample count", self.fixed_count_spin)
        count_form.addRow("Text count location", text_count_row)
        count_form.addRow("Binary count location", binary_count_row)
        layout.addWidget(count)

        output = QGroupBox("Engineering value")
        output_form = QFormLayout(output)
        self.quantity_combo = QComboBox()
        self.quantity_combo.addItem("Current", "current")
        self.quantity_combo.addItem("Voltage", "voltage")
        self.unit_edit = QComboBox()
        self.quantity_combo.currentIndexChanged.connect(self._update_output_units)
        self._update_output_units()
        self.conversion_mode_combo = QComboBox()
        self.conversion_mode_combo.addItem("Fixed linear scale/offset in encoding", "fixed_linear")
        self.conversion_mode_combo.addItem("User-entered two-point mapping at import", "runtime_linear")
        self.scale_spin = _compact_double(1.0)
        self.offset_spin = _compact_double(0.0)
        self.runtime_raw_min_spin = _compact_double(0.0)
        self.runtime_raw_max_spin = _compact_double(255.0)
        self.runtime_value_min_spin = _compact_double(0.0)
        self.runtime_value_max_spin = _compact_double(1.0)
        self.conversion_label = QLabel("Fixed: output = raw × scale + offset. Runtime: the importer asks what two raw reference values mean.")
        self.conversion_label.setWordWrap(True)
        self.conversion_label.setObjectName("hint")
        output_form.addRow("Output quantity", self.quantity_combo)
        output_form.addRow("Output unit", self.unit_edit)
        output_form.addRow("Conversion", self.conversion_mode_combo)
        output_form.addRow("Scale", self.scale_spin)
        output_form.addRow("Offset", self.offset_spin)
        output_form.addRow("Runtime raw low", self.runtime_raw_min_spin)
        output_form.addRow("Runtime raw high", self.runtime_raw_max_spin)
        output_form.addRow("Default low value", self.runtime_value_min_spin)
        output_form.addRow("Default high value", self.runtime_value_max_spin)
        output_form.addRow(self.conversion_label)
        layout.addWidget(output)

        preview = QGroupBox("Test this format")
        preview_layout = QVBoxLayout(preview)
        file_row = QHBoxLayout()
        self.preview_path = QLineEdit()
        self.preview_path.setPlaceholderText("Optional sample file…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_preview)
        run_preview = QPushButton("Preview decode")
        run_preview.clicked.connect(self._preview)
        file_row.addWidget(self.preview_path, 1)
        file_row.addWidget(browse)
        file_row.addWidget(run_preview)
        preview_layout.addLayout(file_row)
        self.preview_summary = QLabel("Choose a sample file to validate the format before saving it.")
        self.preview_summary.setWordWrap(True)
        self.preview_summary.setObjectName("hint")
        preview_layout.addWidget(self.preview_summary)
        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["Time (s)", "Raw", "Output"])
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.setMinimumHeight(160)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview)
        layout.addStretch(1)

        self.mode_combo.currentIndexChanged.connect(self._update_visibility)
        self.time_mode_combo.currentIndexChanged.connect(self._update_visibility)
        self.conversion_mode_combo.currentIndexChanged.connect(self._update_visibility)
        self.count_mode_combo.currentIndexChanged.connect(self._update_visibility)
        self.record_sep_combo.currentIndexChanged.connect(self._update_visibility)
        self.field_sep_combo.currentIndexChanged.connect(self._update_visibility)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save format")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # The whole builder is a scrolling form.  Install one retained filter on
        # every child widget so the mouse wheel always navigates the page instead
        # of changing a combo/spin value or scrolling the nested preview table.
        self._wheel_scroll_filter = _FormatBuilderWheelScrollFilter(scroll)
        body.installEventFilter(self._wheel_scroll_filter)
        for child in body.findChildren(QWidget):
            child.installEventFilter(self._wheel_scroll_filter)

        self._load_profile(profile or WaveformFormatProfile.new())
        self._update_visibility()
        centre_window_on_parent(self, parent)

    def _load_profile(self, profile: WaveformFormatProfile) -> None:
        self._profile_id = profile.profile_id
        self.name_edit.setText(profile.name)
        self.description_edit.setText(profile.description)
        self.suffixes_edit.setText(", ".join(profile.file_suffixes))
        _combo_set_data(self.mode_combo, profile.data_mode)
        _combo_set_data(self.number_combo, profile.number_format)
        _combo_set_data(self.byte_order_combo, profile.byte_order)
        _combo_set_data(self.encoding_combo, profile.text_encoding)
        _combo_set_data(self.record_sep_combo, profile.text_record_separator)
        self.custom_record_sep.setText(profile.text_custom_record_separator)
        _combo_set_data(self.field_sep_combo, profile.text_field_separator)
        self.custom_field_sep.setText(profile.text_custom_field_separator)
        self.comment_prefix_edit.setText(profile.text_comment_prefix)
        self.skip_records_spin.setValue(profile.text_skip_records)
        self.value_field_spin.setValue(profile.text_value_field)
        self.header_bytes_spin.setValue(profile.binary_header_bytes)
        self.record_bytes_spin.setValue(profile.binary_record_bytes)
        self.value_offset_spin.setValue(profile.binary_value_offset)
        _combo_set_data(self.time_mode_combo, profile.time_mode)
        self.sample_interval_spin.setValue(profile.sample_interval)
        self.sample_interval_unit.setCurrentText(profile.sample_interval_unit)
        self.text_time_field_spin.setValue(profile.text_time_field)
        self.binary_time_offset_spin.setValue(profile.binary_time_offset)
        _combo_set_data(self.binary_time_format_combo, profile.binary_time_format)
        self.explicit_time_unit.setCurrentText(profile.time_unit)
        _combo_set_data(self.count_mode_combo, profile.count_mode)
        self.fixed_count_spin.setValue(max(1, profile.fixed_count))
        self.count_record_spin.setValue(profile.count_text_record)
        self.count_field_spin.setValue(profile.count_text_field)
        self.count_offset_spin.setValue(profile.count_binary_offset)
        _combo_set_data(self.count_format_combo, profile.count_number_format)
        _combo_set_data(self.count_order_combo, profile.count_byte_order)
        quantity = profile.output_quantity if profile.output_quantity in {"current", "voltage"} else ("voltage" if profile.output_unit in {"V", "mV"} else "current")
        _combo_set_data(self.quantity_combo, quantity)
        self._update_output_units(preferred=profile.output_unit)
        _combo_set_data(self.conversion_mode_combo, profile.conversion_mode)
        self.scale_spin.setValue(profile.scale)
        self.offset_spin.setValue(profile.offset)
        self.runtime_raw_min_spin.setValue(profile.runtime_raw_min)
        self.runtime_raw_max_spin.setValue(profile.runtime_raw_max)
        self.runtime_value_min_spin.setValue(profile.runtime_value_at_min)
        self.runtime_value_max_spin.setValue(profile.runtime_value_at_max)

    def profile(self) -> WaveformFormatProfile:
        suffixes = tuple(part.strip() for part in self.suffixes_edit.text().replace(";", ",").split(",") if part.strip())
        return WaveformFormatProfile(
            profile_id=self._profile_id,
            name=self.name_edit.text(),
            description=self.description_edit.text(),
            file_suffixes=suffixes,
            data_mode=str(self.mode_combo.currentData()),
            number_format=str(self.number_combo.currentData()),
            byte_order=str(self.byte_order_combo.currentData()),
            text_encoding=str(self.encoding_combo.currentData()),
            text_record_separator=str(self.record_sep_combo.currentData()),
            text_custom_record_separator=self.custom_record_sep.text(),
            text_field_separator=str(self.field_sep_combo.currentData()),
            text_custom_field_separator=self.custom_field_sep.text(),
            text_comment_prefix=self.comment_prefix_edit.text(),
            text_skip_records=self.skip_records_spin.value(),
            text_value_field=self.value_field_spin.value(),
            binary_header_bytes=self.header_bytes_spin.value(),
            binary_record_bytes=self.record_bytes_spin.value(),
            binary_value_offset=self.value_offset_spin.value(),
            time_mode=str(self.time_mode_combo.currentData()),
            sample_interval=self.sample_interval_spin.value(),
            sample_interval_unit=self.sample_interval_unit.currentText(),
            time_unit=self.explicit_time_unit.currentText(),
            text_time_field=self.text_time_field_spin.value(),
            binary_time_offset=self.binary_time_offset_spin.value(),
            binary_time_format=str(self.binary_time_format_combo.currentData()),
            count_mode=str(self.count_mode_combo.currentData()),
            fixed_count=self.fixed_count_spin.value(),
            count_text_record=self.count_record_spin.value(),
            count_text_field=self.count_field_spin.value(),
            count_binary_offset=self.count_offset_spin.value(),
            count_number_format=str(self.count_format_combo.currentData()),
            count_byte_order=str(self.count_order_combo.currentData()),
            output_quantity=str(self.quantity_combo.currentData()),
            output_unit=self.unit_edit.currentText(),
            conversion_mode=str(self.conversion_mode_combo.currentData()),
            scale=self.scale_spin.value(),
            offset=self.offset_spin.value(),
            runtime_raw_min=self.runtime_raw_min_spin.value(),
            runtime_raw_max=self.runtime_raw_max_spin.value(),
            runtime_value_at_min=self.runtime_value_min_spin.value(),
            runtime_value_at_max=self.runtime_value_max_spin.value(),
        )

    def _update_output_units(self, *_args, preferred: str = "") -> None:
        choices = ("V", "mV") if self.quantity_combo.currentData() == "voltage" else ("mA", "A")
        current = str(preferred or self.unit_edit.currentText() or "")
        self.unit_edit.blockSignals(True)
        self.unit_edit.clear()
        self.unit_edit.addItems(list(choices))
        if current in choices:
            self.unit_edit.setCurrentText(current)
        self.unit_edit.blockSignals(False)

    def _update_visibility(self, *_args) -> None:
        text_mode = self.mode_combo.currentData() == "text"
        self.storage_stack.setCurrentIndex(0 if text_mode else 1)
        self.byte_order_combo.setEnabled(not text_mode)
        self.custom_record_sep.setEnabled(self.record_sep_combo.currentData() == "custom")
        self.custom_field_sep.setEnabled(self.field_sep_combo.currentData() == "custom")
        explicit = self.time_mode_combo.currentData() == "field"
        interval_based = self.time_mode_combo.currentData() in {"fixed_interval", "runtime_interval"}
        self.sample_interval_spin.setEnabled(interval_based)
        self.sample_interval_unit.setEnabled(interval_based)
        self.text_time_field_spin.setEnabled(explicit and text_mode)
        self.binary_time_offset_spin.setEnabled(explicit and not text_mode)
        self.binary_time_format_combo.setEnabled(explicit and not text_mode)
        self.explicit_time_unit.setEnabled(explicit)
        runtime_mapping = self.conversion_mode_combo.currentData() == "runtime_linear"
        self.scale_spin.setEnabled(not runtime_mapping)
        self.offset_spin.setEnabled(not runtime_mapping)
        self.runtime_raw_min_spin.setEnabled(runtime_mapping)
        self.runtime_raw_max_spin.setEnabled(runtime_mapping)
        self.runtime_value_min_spin.setEnabled(runtime_mapping)
        self.runtime_value_max_spin.setEnabled(runtime_mapping)
        count_mode = self.count_mode_combo.currentData()
        self.fixed_count_spin.setEnabled(count_mode == "fixed")
        self.count_record_spin.setEnabled(count_mode == "text_record" and text_mode)
        self.count_field_spin.setEnabled(count_mode == "text_record" and text_mode)
        self.count_offset_spin.setEnabled(count_mode == "binary_header" and not text_mode)
        self.count_format_combo.setEnabled(count_mode == "binary_header" and not text_mode)
        self.count_order_combo.setEnabled(count_mode == "binary_header" and not text_mode)

    def _browse_preview(self, *_args) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose sample waveform file", "", "All files (*)")
        if path:
            self.preview_path.setText(path)

    def _preview(self, *_args) -> None:
        path = self.preview_path.text().strip()
        if not path:
            QMessageBox.information(self, "Preview waveform format", "Choose a sample file first.")
            return
        try:
            profile = self.profile()
            runtime_interval = (
                float(profile.sample_interval) * time_scale_to_seconds(profile.sample_interval_unit)
                if profile.time_mode == "runtime_interval" else None
            )
            parsed = parse_waveform_file(
                path,
                profile,
                runtime_sample_interval_s=runtime_interval,
                runtime_value_at_min=(profile.runtime_value_at_min if profile.conversion_mode == "runtime_linear" else None),
                runtime_value_at_max=(profile.runtime_value_at_max if profile.conversion_mode == "runtime_linear" else None),
            )
            self._show_preview(parsed)
        except Exception as error:  # noqa: BLE001 - file/UI boundary
            self.preview_summary.setText(f"Decode failed: {error}")
            QMessageBox.warning(self, "Unable to decode sample", str(error))

    def _show_preview(self, parsed: ParsedWaveformData) -> None:
        count = len(parsed.output_values)
        duration = float(parsed.time_s[-1] - parsed.time_s[0]) if count > 1 else 0.0
        self.preview_summary.setText(
            f"Decoded {count:,} samples · duration {duration:.9g} s · "
            f"output {parsed.profile.output_quantity} {parsed.profile.output_unit or '(unit unspecified)'}"
        )
        shown = min(count, 24)
        self.preview_table.setRowCount(shown)
        for row in range(shown):
            values = (parsed.time_s[row], parsed.raw_values[row], parsed.output_values[row])
            for column, value in enumerate(values):
                self.preview_table.setItem(row, column, QTableWidgetItem(f"{float(value):.12g}"))

    def _save(self, *_args) -> None:
        try:
            self.result_profile = self.profile()
        except Exception as error:  # noqa: BLE001 - validation boundary
            QMessageBox.warning(self, "Invalid waveform format", str(error))
            return
        self.accept()



class TabularWaveformImportDialog(QDialog):
    """Manual waveform table entry with explicit column interpretation."""

    def __init__(self, parent=None, *, single_signal: bool = False):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.single_signal = bool(single_signal)
        self.setWindowTitle("Manual waveform input")
        self.resize(1040, 720)
        self.result_documents: tuple[WaveformDocument, ...] = ()
        self._conversion_controls: dict[int, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}

        outer = QVBoxLayout(self)
        intro = QLabel(
            "Enter waveform samples directly, or paste a rectangular table from the clipboard. "
            + ("Choose exactly one signal column for the current working waveform. " if self.single_signal else "Each checked signal column becomes its own waveform catalogue entry. ")
            + "This is the raw-table path; encoded waveform files are imported from the format list in the previous window."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        outer.addWidget(intro)

        body = QGridLayout()
        body.setColumnStretch(0, 3)
        body.setColumnStretch(1, 2)
        outer.addLayout(body, 1)

        data_group = QGroupBox("Manual data table")
        data_layout = QVBoxLayout(data_group)
        data_actions = QHBoxLayout()
        self.add_row_button = QPushButton("Add row")
        self.add_column_button = QPushButton("Add column")
        self.paste_button = QPushButton("Paste table")
        self.clear_button = QPushButton("Clear")
        self.add_row_button.clicked.connect(self._add_row)
        self.add_column_button.clicked.connect(self._add_column)
        self.paste_button.clicked.connect(self._paste_table)
        self.clear_button.clicked.connect(self._clear_table)
        data_actions.addWidget(self.add_row_button)
        data_actions.addWidget(self.add_column_button)
        data_actions.addWidget(self.paste_button)
        data_actions.addWidget(self.clear_button)
        data_actions.addStretch(1)
        data_layout.addLayout(data_actions)
        self.data_table = QTableWidget(12, 2)
        self.data_table.setHorizontalHeaderLabels(["Column 1", "Column 2"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.setToolTip("Enter values directly. Blank rows are ignored when importing.")
        data_layout.addWidget(self.data_table, 1)
        self.table_summary = QLabel("Enter at least two data rows, or paste a table from the clipboard.")
        self.table_summary.setWordWrap(True)
        self.table_summary.setObjectName("hint")
        data_layout.addWidget(self.table_summary)
        body.addWidget(data_group, 0, 0)

        mapping_group = QGroupBox("Column mapping")
        mapping_layout = QVBoxLayout(mapping_group)
        timing_form = QFormLayout()
        self.time_column_combo = QComboBox()
        self.time_column_combo.currentIndexChanged.connect(self._time_column_changed)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["s", "ms", "us", "ns"])
        interval_row = QWidget()
        interval_layout = QHBoxLayout(interval_row)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        self.interval_spin = _compact_double(1.0, 1e-12, 1e15)
        self.interval_unit_combo = QComboBox()
        self.interval_unit_combo.addItems(["s", "ms", "us", "ns"])
        self.interval_unit_combo.setCurrentText("ms")
        interval_layout.addWidget(self.interval_spin, 1)
        interval_layout.addWidget(self.interval_unit_combo)
        timing_form.addRow("Time column", self.time_column_combo)
        timing_form.addRow("Time units", self.time_unit_combo)
        timing_form.addRow("If no time column", interval_row)
        mapping_layout.addLayout(timing_form)

        note = QLabel(
            "Check every signal to create. Name, electrical type, units, scale, and offset are applied when the table is converted into waveform resources."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        mapping_layout.addWidget(note)

        self.signal_mapping = QTableWidget(0, 6)
        self.signal_mapping.setHorizontalHeaderLabels(["Import", "Column", "Name", "Type", "Units", "Scale / offset"])
        self.signal_mapping.horizontalHeader().setStretchLastSection(True)
        self.signal_mapping.setMinimumWidth(430)
        mapping_layout.addWidget(self.signal_mapping, 1)
        body.addWidget(mapping_group, 0, 1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setText("Create selected waveforms")
        self.buttons.accepted.connect(self._accept_import)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)
        self._wheel_guard = install_editor_wheel_guard(self)
        self._rebuild_column_mapping()
        centre_window_on_parent(self, parent)

    def _headers(self) -> tuple[str, ...]:
        headers: list[str] = []
        for column in range(self.data_table.columnCount()):
            item = self.data_table.horizontalHeaderItem(column)
            text = item.text().strip() if item is not None else ""
            headers.append(text or f"Column {column + 1}")
        return tuple(headers)

    def _add_row(self, *_args) -> None:
        self.data_table.insertRow(self.data_table.rowCount())

    def _add_column(self, *_args) -> None:
        column = self.data_table.columnCount()
        self.data_table.insertColumn(column)
        self.data_table.setHorizontalHeaderItem(column, QTableWidgetItem(f"Column {column + 1}"))
        self._rebuild_column_mapping()

    def _clear_table(self, *_args) -> None:
        self.data_table.clearContents()
        self.table_summary.setText("Enter at least two data rows, or paste a table from the clipboard.")

    def _paste_table(self, *_args) -> None:
        text = QApplication.clipboard().text()
        if not str(text).strip():
            QMessageBox.information(self, "Manual waveform input", "The clipboard does not contain any table data.")
            return
        try:
            table = parse_text_table(text, delimiter_mode="auto", header_mode="auto")
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to paste waveform table", str(error))
            return
        self.data_table.clear()
        self.data_table.setColumnCount(table.column_count)
        self.data_table.setRowCount(max(12, table.row_count))
        self.data_table.setHorizontalHeaderLabels(list(table.headers))
        for row, values in enumerate(table.rows):
            for column, value in enumerate(values):
                self.data_table.setItem(row, column, QTableWidgetItem(value))
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.table_summary.setText(
            f"Pasted {table.row_count:,} data rows × {table.column_count} columns. Edit any cell directly before creating the waveform(s)."
        )
        self._rebuild_column_mapping()

    def _manual_table(self) -> TextTable:
        headers = self._headers()
        rows: list[tuple[str, ...]] = []
        for row in range(self.data_table.rowCount()):
            values = []
            for column in range(self.data_table.columnCount()):
                item = self.data_table.item(row, column)
                values.append(item.text().strip() if item is not None else "")
            if not any(values):
                continue
            rows.append(tuple(values))
        if len(rows) < 2:
            raise ValueError("Manual waveform input requires at least two non-empty data rows.")
        return TextTable(
            headers=headers,
            rows=tuple(rows),
            delimiter_mode="manual",
            has_header=False,
            source_path="",
        )

    def _rebuild_column_mapping(self) -> None:
        headers = self._headers()
        old_time = self.time_column_combo.currentData() if self.time_column_combo.count() else 0
        self.time_column_combo.blockSignals(True)
        self.time_column_combo.clear()
        self.time_column_combo.addItem("No time column — use fixed interval", -1)
        for index, header in enumerate(headers):
            self.time_column_combo.addItem(f"{index + 1}: {header}", index)
        preferred_time = -1 if len(headers) == 1 else (int(old_time) if old_time is not None and -1 <= int(old_time) < len(headers) else 0)
        self.time_column_combo.setCurrentIndex(self.time_column_combo.findData(preferred_time))
        self.time_column_combo.blockSignals(False)

        self._conversion_controls.clear()
        self.signal_mapping.setRowCount(len(headers))
        for row, header in enumerate(headers):
            include = QTableWidgetItem()
            include.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            include.setCheckState(Qt.CheckState.Checked)
            include.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.signal_mapping.setItem(row, 0, include)
            column_item = QTableWidgetItem(str(row + 1))
            column_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.signal_mapping.setItem(row, 1, column_item)
            self.signal_mapping.setItem(row, 2, QTableWidgetItem(header))

            quantity = QComboBox(self.signal_mapping)
            quantity.addItem("Current", "current")
            quantity.addItem("Voltage", "voltage")
            units = QComboBox(self.signal_mapping)
            units.addItems(["mA", "A"])
            quantity.currentIndexChanged.connect(lambda _i, q=quantity, u=units: self._mapping_quantity_changed(q, u))
            self.signal_mapping.setCellWidget(row, 3, quantity)
            self.signal_mapping.setCellWidget(row, 4, units)

            conversion = QWidget(self.signal_mapping)
            conversion_layout = QHBoxLayout(conversion)
            conversion_layout.setContentsMargins(0, 0, 0, 0)
            scale = _compact_double(1.0)
            offset = _compact_double(0.0)
            scale.setToolTip("Scale")
            offset.setToolTip("Offset")
            conversion_layout.addWidget(QLabel("×"))
            conversion_layout.addWidget(scale)
            conversion_layout.addWidget(QLabel("+"))
            conversion_layout.addWidget(offset)
            self._conversion_controls[row] = (scale, offset)
            self.signal_mapping.setCellWidget(row, 5, conversion)
        self._time_column_changed()
        self.signal_mapping.resizeColumnsToContents()

    @staticmethod
    def _mapping_quantity_changed(quantity: QComboBox, units: QComboBox) -> None:
        current = units.currentText()
        choices = ["V", "mV"] if quantity.currentData() == "voltage" else ["mA", "A"]
        units.blockSignals(True)
        units.clear()
        units.addItems(choices)
        if current in choices:
            units.setCurrentText(current)
        units.blockSignals(False)

    def _time_column_changed(self, *_args) -> None:
        time_column = int(self.time_column_combo.currentData()) if self.time_column_combo.currentData() is not None else -1
        explicit = time_column >= 0
        self.time_unit_combo.setEnabled(explicit)
        self.interval_spin.setEnabled(not explicit)
        self.interval_unit_combo.setEnabled(not explicit)
        for row in range(self.signal_mapping.rowCount()):
            item = self.signal_mapping.item(row, 0)
            if item is None:
                continue
            is_time = row == time_column
            if is_time:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)

    def _signal_specs(self) -> tuple[TabularSignalSpec, ...]:
        specs: list[TabularSignalSpec] = []
        for row in range(self.signal_mapping.rowCount()):
            include = self.signal_mapping.item(row, 0)
            if include is None or include.checkState() != Qt.CheckState.Checked:
                continue
            name_item = self.signal_mapping.item(row, 2)
            quantity = self.signal_mapping.cellWidget(row, 3)
            units = self.signal_mapping.cellWidget(row, 4)
            assert isinstance(quantity, QComboBox) and isinstance(units, QComboBox)
            scale, offset = self._conversion_controls[row]
            specs.append(
                TabularSignalSpec(
                    column=row,
                    name=name_item.text().strip() if name_item is not None else f"Signal {row + 1}",
                    quantity=str(quantity.currentData()),
                    unit=units.currentText(),
                    scale=scale.value(),
                    offset=offset.value(),
                )
            )
        return tuple(specs)

    def _accept_import(self, *_args) -> None:
        try:
            table = self._manual_table()
            time_data = self.time_column_combo.currentData()
            time_column = None if time_data is None or int(time_data) < 0 else int(time_data)
            documents = documents_from_text_table(
                table,
                self._signal_specs(),
                time_column=time_column,
                time_unit=self.time_unit_combo.currentText(),
                sample_interval=self.interval_spin.value(),
                sample_interval_unit=self.interval_unit_combo.currentText(),
            )
            if self.single_signal and len(documents) != 1:
                raise ValueError("Choose exactly one signal column while replacing a working waveform.")
            for document in documents:
                waveform_signal_to_payload(document)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to create waveform(s)", str(error))
            return
        self.result_documents = documents
        self.accept()


class RuntimeFormatImportDialog(QDialog):
    """Collect import-time values requested by a declarative encoding profile."""

    def __init__(self, profile: WaveformFormatProfile, source: Path, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.profile = profile
        self.source = Path(source)
        self.result_document: WaveformDocument | None = None
        self.setWindowTitle(f"Import {profile.name}")
        self.setMinimumWidth(620)

        root = QVBoxLayout(self)
        intro = QLabel(
            "This encoding leaves some values to be supplied when the file is imported. "
            "These settings are stored with the authored waveform component and can be edited later."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        group = QGroupBox("Import-time settings")
        form = QFormLayout(group)
        self.name_edit = QLineEdit(self.source.stem)
        form.addRow("Waveform name", self.name_edit)

        self.interval_spin = _compact_double(profile.sample_interval, 1e-12, 1e15)
        self.interval_unit = QComboBox()
        self.interval_unit.addItems(["s", "ms", "us", "ns"])
        self.interval_unit.setCurrentText(profile.sample_interval_unit)
        interval_host = QWidget()
        interval_layout = QHBoxLayout(interval_host)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_layout.addWidget(self.interval_spin, 1)
        interval_layout.addWidget(self.interval_unit)
        if profile.time_mode == "runtime_interval":
            form.addRow("Sample interval", interval_host)

        self.value_low_spin = _compact_double(profile.runtime_value_at_min)
        self.value_high_spin = _compact_double(profile.runtime_value_at_max)
        if profile.conversion_mode == "runtime_linear":
            form.addRow(f"Raw {profile.runtime_raw_min:g} =", self.value_low_spin)
            form.addRow(f"Raw {profile.runtime_raw_max:g} =", self.value_high_spin)
            form.addRow("Waveform encoding", QLabel(f"{profile.output_quantity.title()} · {profile.output_unit}"))
        root.addWidget(group)

        try:
            parsed = self._parsed()
            raw = parsed.raw_values
            self.summary = QLabel(
                f"{len(raw):,} source samples · raw range {float(min(raw)):.6g}..{float(max(raw)):.6g}"
            )
        except Exception as error:  # noqa: BLE001
            self.summary = QLabel(f"Unable to decode source with this encoding: {error}")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("hint")
        root.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Open).setText("Import waveform")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        centre_window_on_parent(self, parent)

    def _runtime_interval_s(self) -> float | None:
        if self.profile.time_mode != "runtime_interval":
            return None
        return float(self.interval_spin.value()) * time_scale_to_seconds(self.interval_unit.currentText())

    def _parsed(self) -> ParsedWaveformData:
        return parse_waveform_file(
            self.source,
            self.profile,
            runtime_sample_interval_s=self._runtime_interval_s(),
            runtime_value_at_min=(self.value_low_spin.value() if self.profile.conversion_mode == "runtime_linear" else None),
            runtime_value_at_max=(self.value_high_spin.value() if self.profile.conversion_mode == "runtime_linear" else None),
        )

    def _accept(self) -> None:
        try:
            parsed = self._parsed()
            name = self.name_edit.text().strip() or self.source.stem or self.profile.name
            metadata: dict[str, Any] = {
                "format_profile": self.profile.name,
                "import_settings": {
                    "format": "format_profile",
                    "profile": self.profile.to_dict(),
                },
            }
            if self.profile.time_mode == "runtime_interval":
                interval_s = float(self._runtime_interval_s() or 0.0)
                segment = encoded_segment(
                    parsed.raw_values.tolist(),
                    start_s=0.0,
                    sample_interval_s=interval_s,
                    profile=self.profile.to_dict(),
                    runtime_value_at_min=(self.value_low_spin.value() if self.profile.conversion_mode == "runtime_linear" else None),
                    runtime_value_at_max=(self.value_high_spin.value() if self.profile.conversion_mode == "runtime_linear" else None),
                    source_name=self.source.name,
                    source_path=str(self.source),
                )
                metadata[AUTHORING_RECIPE_KEY] = make_authoring_recipe(
                    [segment],
                    quantity=self.profile.output_quantity,
                    unit=self.profile.output_unit,
                )
                metadata["import_settings"]["runtime_sample_interval_s"] = interval_s
            if self.profile.conversion_mode == "runtime_linear":
                metadata["import_settings"]["runtime_value_at_min"] = self.value_low_spin.value()
                metadata["import_settings"]["runtime_value_at_max"] = self.value_high_spin.value()
            self.result_document = WaveformDocument(
                [WaveformSeries(
                    name,
                    parsed.time_s,
                    parsed.output_values,
                    quantity=self.profile.output_quantity,
                    unit=self.profile.output_unit,
                    metadata={"format_profile_id": self.profile.profile_id},
                )],
                title=name,
                metadata=metadata,
                source_format=f"profile:{self.profile.profile_id}",
                source_path=str(self.source),
            )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to import waveform", str(error))
            return
        self.accept()


class WaveformImportDialog(QDialog):
    """Choose an explicit waveform encoding, then decode one file with it."""

    def __init__(self, parent=None, *, store: WaveformFormatStore | None = None, single_signal: bool = False):
        super().__init__(parent)
        self.single_signal = bool(single_signal)
        enable_standard_window_controls(self)
        self.setWindowTitle("Import waveform")
        self.resize(860, 570)
        self.store = store or WaveformFormatStore(_profile_directory())
        self.result_document: WaveformDocument | None = None
        self.result_documents: tuple[WaveformDocument, ...] = ()
        self.result_series: WaveformSeries | None = None
        self._decoded_document: WaveformDocument | None = None
        self._decoded_documents: tuple[WaveformDocument, ...] = ()
        self._decoded_source_kind = "file"

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Choose Manual input to enter or paste raw sample tables directly. For encoded files, select the encoding first, "
            "then choose a file. Field Workbench waveform JSON can also import saved waveforms directly from a project."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        layout.addWidget(intro)

        self.manual_button = QPushButton("Manual input…")
        self.manual_button.setMinimumHeight(38)
        self.manual_button.setToolTip("Enter or paste waveform samples directly into an editable table.")
        self.manual_button.clicked.connect(self._open_manual_input)
        layout.addWidget(self.manual_button)

        main = QGridLayout()
        main.setColumnStretch(0, 2)
        main.setColumnStretch(1, 3)
        layout.addLayout(main, 1)

        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Available encodings</b>"))
        self.format_list = QListWidget()
        self.format_list.currentItemChanged.connect(self._selection_changed)
        left.addWidget(self.format_list, 1)
        profile_buttons = QGridLayout()
        self.new_button = QPushButton("New format…")
        self.edit_button = QPushButton("Edit…")
        self.copy_button = QPushButton("Duplicate…")
        self.delete_button = QPushButton("Delete")
        self.import_profile_button = QPushButton("Import profile…")
        self.export_profile_button = QPushButton("Export profile…")
        self.new_button.clicked.connect(self._new_profile)
        self.edit_button.clicked.connect(self._edit_profile)
        self.copy_button.clicked.connect(self._copy_profile)
        self.delete_button.clicked.connect(self._delete_profile)
        self.import_profile_button.clicked.connect(self._import_profile)
        self.export_profile_button.clicked.connect(self._export_profile)
        for index, button in enumerate((
            self.new_button,
            self.edit_button,
            self.copy_button,
            self.delete_button,
            self.import_profile_button,
            self.export_profile_button,
        )):
            profile_buttons.addWidget(button, index // 2, index % 2)
        left.addLayout(profile_buttons)
        main.addLayout(left, 0, 0)

        right = QVBoxLayout()
        self.description = QLabel("Select an encoding before choosing a file.")
        self.description.setWordWrap(True)
        self.description.setMinimumHeight(72)
        right.addWidget(self.description)

        file_row = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setEnabled(False)
        self.file_path.setPlaceholderText("Select an encoding first…")
        self.browse_button = QPushButton("Browse…")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_path, 1)
        file_row.addWidget(self.browse_button)
        right.addLayout(file_row)

        self.summary = QLabel("No encoded file loaded.")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("hint")
        right.addWidget(self.summary)
        right.addWidget(QLabel("<b>Waveforms to import</b>"))
        self.signal_list = QListWidget()
        self.signal_list.currentItemChanged.connect(self._signal_changed)
        self.signal_list.itemChanged.connect(self._signal_check_changed)
        right.addWidget(self.signal_list, 1)
        main.addLayout(right, 0, 1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setText("Import selected waveforms")
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(False)
        self.buttons.accepted.connect(self._accept_import)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._wheel_guard = install_editor_wheel_guard(self)
        self._reload_formats()
        centre_window_on_parent(self, parent)

    def _reload_formats(self, select_profile_id: str | None = None) -> None:
        self.format_list.blockSignals(True)
        self.format_list.clear()

        default_item = None
        for reader in DEFAULT_WAVEFORM_IO_REGISTRY.readers:
            item = QListWidgetItem(reader.display_name)
            item.setData(_KIND_ROLE, "reader")
            item.setData(_PROFILE_ROLE, reader.format_id)
            item.setToolTip("Built-in encoding")
            self.format_list.addItem(item)
            if reader.format_id == NATIVE_WAVEFORM_READER_ID:
                default_item = item

        for profile in builtin_waveform_format_profiles():
            item = QListWidgetItem(profile.name)
            item.setData(_KIND_ROLE, "builtin_profile")
            item.setData(_PROFILE_ROLE, profile.profile_id)
            item.setToolTip("Bundled format profile")
            self.format_list.addItem(item)
        selected_item = None
        for profile in self.store.profiles():
            item = QListWidgetItem(profile.name)
            item.setData(_KIND_ROLE, "profile")
            item.setData(_PROFILE_ROLE, profile.profile_id)
            item.setToolTip("User format profile")
            self.format_list.addItem(item)
            if select_profile_id and profile.profile_id == select_profile_id:
                selected_item = item
        self.format_list.clearSelection()
        self.format_list.setCurrentItem(selected_item or default_item)
        self.format_list.blockSignals(False)
        self._selection_changed(self.format_list.currentItem(), None)

    def _selected_profile(self) -> WaveformFormatProfile | None:
        item = self.format_list.currentItem()
        if item is None or item.data(_KIND_ROLE) not in {"profile", "builtin_profile"}:
            return None
        profile_id = str(item.data(_PROFILE_ROLE) or "")
        profiles = (*builtin_waveform_format_profiles(), *self.store.profiles())
        for profile in profiles:
            if profile.profile_id == profile_id:
                return profile
        return None

    def _clear_loaded_file(self) -> None:
        self._decoded_document = None
        self._decoded_documents = ()
        self._decoded_source_kind = "file"
        self.result_document = None
        self.result_series = None
        self.result_documents = ()
        self.file_path.clear()
        self.signal_list.blockSignals(True)
        self.signal_list.clear()
        self.signal_list.blockSignals(False)
        self.summary.setText("Choose a file after selecting its encoding.")
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(False)

    def _selection_changed(self, *_args) -> None:
        item = self.format_list.currentItem()
        profile = self._selected_profile()
        is_profile = profile is not None
        kind = item.data(_KIND_ROLE) if item is not None else None
        is_user_profile = kind == "profile"
        self.edit_button.setEnabled(is_user_profile)
        self.copy_button.setEnabled(is_profile)
        self.delete_button.setEnabled(is_user_profile)
        self.export_profile_button.setEnabled(is_profile)
        self._clear_loaded_file()

        has_encoding = item is not None
        self.file_path.setEnabled(has_encoding)
        self.browse_button.setEnabled(has_encoding)
        is_native_reader = (
            item is not None
            and item.data(_KIND_ROLE) == "reader"
            and str(item.data(_PROFILE_ROLE) or "") == NATIVE_WAVEFORM_READER_ID
        )
        if not has_encoding:
            placeholder = "Select an encoding first…"
        elif is_native_reader:
            placeholder = "Choose waveform or Field Workbench project file…"
        else:
            placeholder = "Choose encoded waveform file…"
        self.file_path.setPlaceholderText(placeholder)
        if item is None:
            self.description.setText("Select an encoding before choosing a file.")
            return
        kind = item.data(_KIND_ROLE)
        if profile is not None:
            suffixes = ", ".join(profile.file_suffixes) or "any filename"
            timing = "user-entered timing" if profile.time_mode == "runtime_interval" else ("stored timing" if profile.time_mode == "field" else "fixed timing")
            conversion = "user-entered mapping" if profile.conversion_mode == "runtime_linear" else "fixed mapping"
            protected = "Built-in encoding" if kind == "builtin_profile" else "User format profile"
            self.description.setText(
                f"<b>{profile.name}</b><br>{profile.description or 'Text/binary waveform layout.'}<br>"
                f"{protected} · {profile.data_mode} · {timing} · {conversion} · suffixes: {suffixes}"
            )
        else:
            format_id = str(item.data(_PROFILE_ROLE) or "")
            reader = DEFAULT_WAVEFORM_IO_REGISTRY.reader(format_id)
            suffixes = ", ".join(getattr(reader, "suffixes", ())) or "any filename"
            project_note = (
                " · projects: .magpy.json"
                if format_id == NATIVE_WAVEFORM_READER_ID
                else ""
            )
            self.description.setText(
                f"<b>{reader.display_name}</b><br>Built-in Field Workbench encoding · waveform suffixes: {suffixes}{project_note}"
            )

    def _open_manual_input(self, *_args) -> None:
        dialog = TabularWaveformImportDialog(self, single_signal=self.single_signal)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.result_documents:
            return
        self.result_documents = tuple(dialog.result_documents)
        self.result_document = self.result_documents[0] if len(self.result_documents) == 1 else None
        self.result_series = self.result_documents[0].signal
        self.accept()

    def _new_profile(self, *_args) -> None:
        dialog = WaveformFormatBuilderDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_profile is not None:
            self.store.save(dialog.result_profile)
            self._reload_formats(dialog.result_profile.profile_id)

    def _edit_profile(self, *_args) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        dialog = WaveformFormatBuilderDialog(profile, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_profile is not None:
            self.store.save(dialog.result_profile)
            self._reload_formats(dialog.result_profile.profile_id)

    def _copy_profile(self, *_args) -> None:
        item = self.format_list.currentItem()
        if item is None:
            return
        profile = self._selected_profile()
        if profile is None:
            return
        duplicate = profile.copy_as_new()
        dialog = WaveformFormatBuilderDialog(duplicate, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_profile is not None:
            self.store.save(dialog.result_profile)
            self._reload_formats(dialog.result_profile.profile_id)

    def _delete_profile(self, *_args) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete waveform format",
            f"Delete the saved format profile {profile.name!r}?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.store.delete(profile.profile_id)
            self._reload_formats()

    def _import_profile(self, *_args) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import waveform format profile",
            "",
            f"Field Workbench waveform format (*{FORMAT_PROFILE_SUFFIX});;JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            imported = WaveformFormatProfile.load(path)
            profile = imported.copy_as_new(name=imported.name)
            self.store.save(profile)
            self._reload_formats(profile.profile_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to import format profile", str(error))

    def _export_profile(self, *_args) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        suggested = f"{profile.name.replace(' ', '_')}{FORMAT_PROFILE_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export waveform format profile",
            suggested,
            f"Field Workbench waveform format (*{FORMAT_PROFILE_SUFFIX});;JSON files (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += FORMAT_PROFILE_SUFFIX
        try:
            profile.save(path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to export format profile", str(error))

    def _browse_file(self, *_args) -> None:
        item = self.format_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Import waveform", "Choose an encoding before choosing a file.")
            return
        profile = self._selected_profile()
        if profile is not None:
            suffixes = profile.file_suffixes
            label = profile.name
            patterns = " ".join(f"*{suffix}" for suffix in suffixes) if suffixes else "*"
            file_filter = f"{label} ({patterns});;All files (*)"
            caption = "Choose waveform file"
        else:
            format_id = str(item.data(_PROFILE_ROLE) or "")
            reader = DEFAULT_WAVEFORM_IO_REGISTRY.reader(format_id)
            suffixes = tuple(getattr(reader, "suffixes", ()))
            patterns = " ".join(f"*{suffix}" for suffix in suffixes) if suffixes else "*"
            if format_id == NATIVE_WAVEFORM_READER_ID:
                project_pattern = "*.magpy.json"
                combined = f"{patterns} {project_pattern}"
                file_filter = (
                    f"Field Workbench waveforms / projects ({combined});;"
                    f"{reader.display_name} ({patterns});;"
                    f"Field Workbench projects ({project_pattern});;"
                    "JSON files (*.json);;All files (*)"
                )
                caption = "Choose waveform or project file"
            else:
                file_filter = f"{reader.display_name} ({patterns});;All files (*)"
                caption = "Choose waveform file"
        path, _ = QFileDialog.getOpenFileName(self, caption, "", file_filter)
        if not path:
            return
        self.file_path.setText(path)
        self._decode_selected_file(path)

    def _decode_selected_file(self, path: str) -> None:
        item = self.format_list.currentItem()
        if item is None:
            return
        self._decoded_document = None
        self._decoded_documents = ()
        self._decoded_source_kind = "file"
        self.result_document = None
        self.result_documents = ()
        self.result_series = None
        self.signal_list.clear()
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(False)
        try:
            profile = self._selected_profile()
            if profile is not None:
                if profile.time_mode == "runtime_interval" or profile.conversion_mode == "runtime_linear":
                    dialog = RuntimeFormatImportDialog(profile, Path(path), self)
                    if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_document is None:
                        self.summary.setText("Import-time settings cancelled. Choose the file again to configure them.")
                        return
                    document = dialog.result_document
                else:
                    parsed = parse_waveform_file(path, profile)
                    document = parsed.document(title=Path(path).stem, source_path=path)
                self._populate_decoded_document(document)
                return

            format_id = str(item.data(_PROFILE_ROLE) or "")
            if format_id == NATIVE_WAVEFORM_READER_ID:
                if str(path).lower().endswith(".magpy.json"):
                    documents = read_project_waveform_documents(path)
                    self._populate_decoded_documents(documents, source_kind="project")
                    return
                try:
                    document = DEFAULT_WAVEFORM_IO_REGISTRY.read(path, format_id=format_id)
                except WaveformIOError as waveform_error:
                    # Project Manager accepts ordinary .json names as well as the
                    # canonical .magpy.json suffix. If native-waveform decoding
                    # fails, accept the file only if it is a real FWB project with
                    # saved waveform resources; otherwise keep the native error.
                    try:
                        documents = read_project_waveform_documents(path)
                    except WaveformIOError:
                        raise waveform_error
                    self._populate_decoded_documents(documents, source_kind="project")
                    return
            else:
                document = DEFAULT_WAVEFORM_IO_REGISTRY.read(path, format_id=format_id)
            self._populate_decoded_document(document)
        except Exception as error:  # noqa: BLE001
            self._decoded_document = None
            self._decoded_documents = ()
            self.result_document = None
            self.result_documents = ()
            self.result_series = None
            self.summary.setText(f"Unable to decode with the selected encoding: {error}")
            QMessageBox.warning(self, "Unable to import waveform", str(error))

    def _populate_decoded_document(self, document: WaveformDocument) -> None:
        self._decoded_document = document
        self._populate_decoded_documents(document.split_signals(), source_kind="file")

    def _populate_decoded_documents(
        self,
        documents: tuple[WaveformDocument, ...],
        *,
        source_kind: str,
    ) -> None:
        self._decoded_documents = tuple(documents)
        self._decoded_source_kind = str(source_kind)
        self.result_document = None
        self.result_documents = ()
        self.result_series = None
        self.signal_list.blockSignals(True)
        self.signal_list.clear()
        for index, document in enumerate(self._decoded_documents):
            signal = document.signal
            display_name = (document.title or signal.name) if source_kind == "project" else signal.name
            item = QListWidgetItem(
                f"{display_name} — {signal.quantity.title()} {signal.unit or '(units not specified)'}"
            )
            item.setData(_PROFILE_ROLE, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.signal_list.addItem(item)
        self.signal_list.blockSignals(False)
        if self.signal_list.count():
            self.signal_list.setCurrentRow(0)
            self._show_signal(0)
        self._update_import_enabled()

    def _signal_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        if not self._decoded_documents or current is None:
            return
        self._show_signal(int(current.data(_PROFILE_ROLE)))

    def _signal_check_changed(self, *_args) -> None:
        self._update_import_enabled()

    def _update_import_enabled(self) -> None:
        checked = sum(
            self.signal_list.item(row).checkState() == Qt.CheckState.Checked
            for row in range(self.signal_list.count())
        )
        valid = bool(self._decoded_documents) and checked > 0 and (not self.single_signal or checked == 1)
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(valid)

    def _show_signal(self, index: int) -> None:
        document = self._decoded_documents[index]
        series = document.signal
        self.result_series = series
        dt = series.nominal_sample_interval_s
        dt_text = f" · nominal Δt {dt:.9g} s" if dt is not None else ""
        if self._decoded_source_kind == "project":
            project_count = len(self._decoded_documents)
            prefix = (
                f"Project waveform {index + 1} of {project_count}: "
                f"{document.title or series.name} · "
            )
        else:
            prefix = "Decoded with the selected encoding: "
        self.summary.setText(
            f"{prefix}{series.sample_count:,} samples · duration {series.duration_s:.9g} s{dt_text} · "
            f"{series.quantity} {series.unit or '(unit unspecified)'}"
        )

    def _accept_import(self, *_args) -> None:
        if not self._decoded_documents:
            QMessageBox.information(self, "Import waveform", "Choose an encoding and file first.")
            return
        selected_indexes = [
            int(self.signal_list.item(row).data(_PROFILE_ROLE))
            for row in range(self.signal_list.count())
            if self.signal_list.item(row).checkState() == Qt.CheckState.Checked
        ]
        if not selected_indexes:
            QMessageBox.information(self, "Import waveform", "Check at least one waveform to import.")
            return
        if self.single_signal and len(selected_indexes) != 1:
            QMessageBox.information(self, "Import waveform", "Choose exactly one waveform.")
            return
        documents = tuple(self._decoded_documents[index] for index in selected_indexes)
        try:
            for document in documents:
                waveform_signal_to_payload(document)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to import selected waveforms", str(error))
            return
        self.result_documents = documents
        self.result_document = documents[0] if len(documents) == 1 else None
        self.result_series = documents[0].signal
        self.accept()
