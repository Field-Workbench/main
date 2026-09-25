"""Embedded waveform-import controls for the Waveform Editor.

Import is an authoring tool while a waveform working copy is active.  These
panels deliberately avoid a second preview window: source decoding and
translation controls live below the persistent Waveform Editor viewer, which
renders the complete prospective waveform before Apply commits a lossless
authored component. Encodings tagged for import-time timing retain their raw
source samples, timing, mapping settings, and declarative profile so the
component can be reopened later regardless of which encoding produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import QStandardPaths, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...waveform_formats import (
    WaveformFormatProfile,
    WaveformFormatStore,
    builtin_waveform_format_profiles,
    parse_waveform_file,
)
from ...waveform_io import (
    DEFAULT_WAVEFORM_IO_REGISTRY,
    WaveformDocument,
    WaveformIOError,
    WaveformSeries,
    signal_unit_to_si,
    waveform_signal_to_payload,
)
from ...waveform_table_io import (
    TabularSignalSpec,
    TextTable,
    documents_from_text_table,
    read_text_table,
)
from ...waveform_recipe import encoded_segment
from .widgets import action_button


def _profile_directory() -> Path:
    root = str(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation) or "").strip()
    if not root:
        root = str(Path.home() / ".field_workbench")
    return Path(root) / "waveform_formats"


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class _NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


def _double(
    value: float,
    *,
    minimum: float = -1.0e15,
    maximum: float = 1.0e15,
    decimals: int = 9,
) -> _NoWheelDoubleSpinBox:
    control = _NoWheelDoubleSpinBox()
    control.setDecimals(decimals)
    control.setRange(float(minimum), float(maximum))
    control.setValue(float(value))
    control.setKeyboardTracking(False)
    return control


class EmbeddedRuntimeFormatPanel(QWidget):
    """Create or reopen one editable import from a runtime-timed format profile."""

    def __init__(
        self,
        owner,
        *,
        profile: WaveformFormatProfile | None = None,
        source: Path | None = None,
        recipe_segment: dict | None = None,
        recipe_segment_index: int | None = None,
        on_preview: Callable[[dict, int | None], str],
        on_apply: Callable[[dict, int | None], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__(owner)
        self.owner = owner
        self.recipe_segment_index = recipe_segment_index
        self._recipe_segment = dict(recipe_segment or {})
        if profile is None:
            payload = self._recipe_segment.get("profile")
            if not isinstance(payload, dict):
                raise ValueError("This encoded component is missing its format profile.")
            profile = WaveformFormatProfile.from_dict(payload)
        self.profile = profile
        stored_source = str(self._recipe_segment.get("source_path") or "").strip()
        self.source = Path(source) if source is not None else (Path(stored_source) if stored_source else None)
        self._on_preview = on_preview
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        self._current_segment: dict | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self.refresh_preview)

        try:
            current = owner.native_editor.current_signal()
        except Exception as error:
            raise ValueError("A waveform encoding must be created before importing encoded waveform data.") from error
        self.target_quantity = str(current.quantity)
        self.target_unit = str(current.unit)
        # The bundled unsigned 8-bit DAC format describes raw 0..255 codes, not
        # an electrical quantity.  Its historical current/mA profile values are
        # merely defaults from the old DAC-oriented editor and must not override
        # the waveform encoding that the user has already chosen.
        self._raw_dac_codes = self.profile.profile_id == "builtin.dac_8bit"
        if not self._raw_dac_codes and self.profile.output_quantity != self.target_quantity:
            raise ValueError(
                f"{self.profile.name} decodes {self.profile.output_quantity}, but this waveform is {self.target_quantity}."
            )
        if self._raw_dac_codes:
            self._unit_factor = 1.0
        else:
            _from_q, _from_si, from_scale = signal_unit_to_si(self.profile.output_unit, quantity=self.target_quantity)
            _to_q, _to_si, to_scale = signal_unit_to_si(self.target_unit, quantity=self.target_quantity)
            self._unit_factor = float(from_scale / to_scale)

        if recipe_segment is not None:
            raw_values = self._recipe_segment.get("raw_values", ())
        else:
            if self.source is None:
                raise ValueError("Choose an encoded waveform file first.")
            default_interval_s = (
                float(self.profile.sample_interval) * _time_unit_scale(self.profile.sample_interval_unit)
                if self.profile.time_mode == "runtime_interval" else None
            )
            parsed = parse_waveform_file(
                self.source,
                self.profile,
                runtime_sample_interval_s=default_interval_s,
                runtime_value_at_min=(self.profile.runtime_value_at_min if self.profile.conversion_mode == "runtime_linear" else None),
                runtime_value_at_max=(self.profile.runtime_value_at_max if self.profile.conversion_mode == "runtime_linear" else None),
            )
            raw_values = parsed.raw_values
        self.raw_values = np.asarray(raw_values, dtype=float).reshape(-1)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(7)
        editing = recipe_segment_index is not None
        heading = QLabel(
            f"{'Edit encoded component' if editing else 'Import encoded waveform'} — {self.profile.name}", self
        )
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        controls = QGroupBox("Import-time settings", self)
        form = QFormLayout(controls)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        stored_interval_ms = float(self._recipe_segment.get("sample_interval_s", 0.0) or 0.0) * 1000.0
        if stored_interval_ms <= 0.0:
            stored_interval_ms = float(self.profile.sample_interval) * _time_unit_scale(self.profile.sample_interval_unit) * 1000.0
        self.interval_ms = _double(stored_interval_ms, minimum=1.0e-9, maximum=1.0e12, decimals=10)
        if self.profile.time_mode == "runtime_interval":
            form.addRow("Point delay (ms)" if self._raw_dac_codes else "Sample interval (ms)", self.interval_ms)

        default_low = float(self.profile.runtime_value_at_min) * self._unit_factor
        default_high = float(self.profile.runtime_value_at_max) * self._unit_factor
        self.value_low = _double(float(self._recipe_segment.get("runtime_value_at_min", default_low)), decimals=11)
        self.value_high = _double(float(self._recipe_segment.get("runtime_value_at_max", default_high)), decimals=11)
        form.addRow("Waveform encoding", QLabel(f"{self.target_quantity.title()} · {self.target_unit}", controls))
        if self.profile.conversion_mode == "runtime_linear":
            low_label = f"Code {self.profile.runtime_raw_min:g} =" if self._raw_dac_codes else f"Raw {self.profile.runtime_raw_min:g} ="
            high_label = f"Code {self.profile.runtime_raw_max:g} =" if self._raw_dac_codes else f"Raw {self.profile.runtime_raw_max:g} ="
            form.addRow(low_label, self.value_low)
            form.addRow(high_label, self.value_high)
        root.addWidget(controls)

        self.edit_points_button = QPushButton("Edit source samples…", self)
        self.edit_points_button.setCheckable(True)
        self.edit_points_button.toggled.connect(self._toggle_points_editor)
        root.addWidget(self.edit_points_button)
        self.samples_editor = QPlainTextEdit(self)
        self.samples_editor.setObjectName("waveformEncodedSamplesEditor")
        self.samples_editor.setPlaceholderText("One raw source sample per line")
        self.samples_editor.setPlainText("\n".join(f"{float(value):.12g}" for value in self.raw_values))
        self.samples_editor.setMinimumHeight(130)
        self.samples_editor.setVisible(False)
        self.samples_editor.textChanged.connect(self._schedule_preview)
        root.addWidget(self.samples_editor)

        raw_min = float(np.min(self.raw_values)) if len(self.raw_values) else 0.0
        raw_max = float(np.max(self.raw_values)) if len(self.raw_values) else 0.0
        editable_bits = ["source samples"]
        if self.profile.time_mode == "runtime_interval":
            editable_bits.append("sample interval")
        if self.profile.conversion_mode == "runtime_linear":
            editable_bits.append("value mapping")
        self.summary = QLabel(
            f"{len(self.raw_values):,} source samples · raw range {raw_min:.6g}..{raw_max:.6g}. "
            f"Stored editable settings: {', '.join(editable_bits)}.", self
        )
        self.summary.setWordWrap(True)
        self.summary.setProperty("role", "muted")
        root.addWidget(self.summary)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.apply_button = action_button("Apply", self, self.apply, role="primary")
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self._on_cancel)
        actions.addWidget(self.apply_button)
        actions.addWidget(cancel_button)
        root.addLayout(actions)

        self.interval_ms.valueChanged.connect(self.refresh_preview)
        self.value_low.valueChanged.connect(self.refresh_preview)
        self.value_high.valueChanged.connect(self.refresh_preview)
        QTimer.singleShot(0, self.refresh_preview)

    def _toggle_points_editor(self, visible: bool) -> None:
        self.samples_editor.setVisible(bool(visible))
        self.edit_points_button.setText("Hide source samples" if visible else "Edit source samples…")

    def _schedule_preview(self) -> None:
        self._preview_timer.start()

    def _edited_raw_values(self) -> np.ndarray:
        text = self.samples_editor.toPlainText().strip()
        if not text:
            raise ValueError("Enter at least two source samples.")
        tokens = [token for token in re.split(r"[\s,;]+", text) if token]
        if len(tokens) < 2:
            raise ValueError("An encoded component needs at least two source samples.")
        try:
            values = np.asarray([float(token) for token in tokens], dtype=float)
        except ValueError as error:
            raise ValueError("Every source sample must be numeric.") from error
        if not np.isfinite(values).all():
            raise ValueError("Source samples must be finite.")
        return values

    def _profile_for_waveform(self) -> WaveformFormatProfile:
        # Store the conversion in the waveform's fixed unit so the recipe can be
        # reconstructed without knowing which catalogue/editor path imported it.
        data = self.profile.to_dict()
        raw = dict(data["profile"])
        raw["output_quantity"] = self.target_quantity
        raw["output_unit"] = self.target_unit
        if self.profile.conversion_mode == "fixed_linear":
            raw["scale"] = float(self.profile.scale) * self._unit_factor
            raw["offset"] = float(self.profile.offset) * self._unit_factor
        else:
            raw["runtime_value_at_min"] = float(self.value_low.value())
            raw["runtime_value_at_max"] = float(self.value_high.value())
        data["profile"] = raw
        return WaveformFormatProfile.from_dict(data)

    def build_segment(self) -> dict:
        profile = self._profile_for_waveform()
        interval_s = (
            float(self.interval_ms.value()) / 1000.0
            if profile.time_mode == "runtime_interval"
            else float(profile.sample_interval) * _time_unit_scale(profile.sample_interval_unit)
        )
        source_name = str(self._recipe_segment.get("source_name") or "").strip() or (
            self.source.name if self.source is not None else profile.name
        )
        source_path = str(self.source) if self.source is not None else str(self._recipe_segment.get("source_path") or "")
        return encoded_segment(
            self._edited_raw_values().tolist(),
            start_s=0.0,
            sample_interval_s=interval_s,
            profile=profile.to_dict(),
            runtime_value_at_min=(self.value_low.value() if profile.conversion_mode == "runtime_linear" else None),
            runtime_value_at_max=(self.value_high.value() if profile.conversion_mode == "runtime_linear" else None),
            source_name=source_name,
            source_path=source_path,
        )

    def refresh_preview(self, *_args) -> None:
        try:
            segment = self.build_segment()
            message = self._on_preview(segment, self.recipe_segment_index)
        except Exception as error:  # noqa: BLE001
            self._current_segment = None
            self.apply_button.setEnabled(False)
            self.summary.setText(f"Unable to preview encoded import: {error}")
            return
        self._current_segment = segment
        self.apply_button.setEnabled(True)
        values = segment.get("raw_values", [])
        raw_min = min(values) if values else 0.0
        raw_max = max(values) if values else 0.0
        self.summary.setText(
            f"{len(values):,} stored source samples · raw range {raw_min:.6g}..{raw_max:.6g}. {message}"
        )

    def apply(self) -> None:
        try:
            segment = self.build_segment()
        except Exception as error:  # noqa: BLE001
            self.summary.setText(f"Unable to apply encoded import: {error}")
            return
        self._on_apply(segment, self.recipe_segment_index)


def _time_unit_scale(unit: str) -> float:
    return {"s": 1.0, "ms": 1.0e-3, "us": 1.0e-6, "ns": 1.0e-9}[str(unit)]


class EmbeddedWaveformImportPanel(QWidget):
    """Decode one source signal and append-preview it in the main viewer.

    Catalogue import remains the multi-signal/resource-management workflow.  An
    active waveform edit intentionally imports exactly one signal, matching the
    one-chunk Apply semantics of every other waveform tool.
    """

    def __init__(
        self,
        owner,
        *,
        on_preview: Callable[[WaveformDocument, str], str],
        on_apply: Callable[[WaveformDocument, str], None],
        on_cancel: Callable[[], None],
        on_runtime_profile_source: Callable[[WaveformFormatProfile, Path], None] | None = None,
        store: WaveformFormatStore | None = None,
    ) -> None:
        super().__init__(owner)
        self.owner = owner
        self._on_preview = on_preview
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        self._on_runtime_profile_source = on_runtime_profile_source
        self.store = store or WaveformFormatStore(_profile_directory())
        self._profiles: dict[str, WaveformFormatProfile] = {}
        self._decoded_document: WaveformDocument | None = None
        self._table: TextTable | None = None
        self._current_document: WaveformDocument | None = None
        self._loading = False
        try:
            current = owner.native_editor.current_signal()
        except Exception as error:
            raise ValueError("A waveform encoding must be created before importing waveform data.") from error
        self.target_quantity = str(current.quantity)
        self.target_unit = str(current.unit)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(7)
        heading = QLabel("Import waveform", self)
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        source_group = QGroupBox("Source", self)
        source = QGridLayout(source_group)
        source.setColumnStretch(1, 1)
        self.format_combo = _NoWheelComboBox(source_group)
        self.file_path = QLineEdit(source_group)
        self.file_path.setPlaceholderText("Choose waveform file…")
        browse = QPushButton("Browse…", source_group)
        browse.clicked.connect(self._browse)
        source.addWidget(QLabel("Format", source_group), 0, 0)
        source.addWidget(self.format_combo, 0, 1, 1, 2)
        source.addWidget(QLabel("File", source_group), 1, 0)
        source.addWidget(self.file_path, 1, 1)
        source.addWidget(browse, 1, 2)
        root.addWidget(source_group)

        self.mode_stack = QStackedWidget(self)
        root.addWidget(self.mode_stack)

        # Structured/native/profile imports need only a source-signal choice;
        # the decoded waveform's own physical timeline and units are preserved.
        self.structured_page = QWidget(self.mode_stack)
        structured_layout = QFormLayout(self.structured_page)
        self.signal_combo = _NoWheelComboBox(self.structured_page)
        structured_layout.addRow("Source signal", self.signal_combo)
        structured_hint = QLabel(
            "Native/structured inputs preserve their decoded physical timeline and are converted into the waveform's fixed encoding when compatible. Reusable format profiles are managed from catalogue Import….",
            self.structured_page,
        )
        structured_hint.setWordWrap(True)
        structured_hint.setProperty("role", "muted")
        structured_layout.addRow(structured_hint)
        self.mode_stack.addWidget(self.structured_page)

        # Focused single-signal text/CSV mapping.  The catalogue importer keeps
        # its larger multi-column table UI; inline authoring only needs one
        # prospective chunk and therefore exposes one signal mapping at a time.
        # Text/CSV has enough mapping controls that forcing the page into a
        # short lower workspace made the rows collapse into each other.  Keep
        # the action/summary area fixed and let only the mapping form scroll.
        self.tabular_scroll = QScrollArea(self.mode_stack)
        self.tabular_scroll.setObjectName("waveformInlineImportScroll")
        self.tabular_scroll.setWidgetResizable(True)
        self.tabular_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tabular_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tabular_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.tabular_page = QWidget(self.tabular_scroll)
        table_form = QFormLayout(self.tabular_page)
        table_form.setVerticalSpacing(7)
        self.delimiter = _NoWheelComboBox(self.tabular_page)
        for label, value in (("Auto", "auto"), ("Comma", "comma"), ("Tab", "tab"), ("Whitespace", "whitespace"), ("Semicolon", "semicolon")):
            self.delimiter.addItem(label, value)
        self.header = _NoWheelComboBox(self.tabular_page)
        for label, value in (("Auto", "auto"), ("First row is header", "yes"), ("No header", "no")):
            self.header.addItem(label, value)
        self.encoding = _NoWheelComboBox(self.tabular_page)
        for value in ("utf-8-sig", "utf-8", "ascii", "latin-1"):
            self.encoding.addItem(value, value)
        self.time_column = _NoWheelComboBox(self.tabular_page)
        self.time_units = _NoWheelComboBox(self.tabular_page)
        self.time_units.addItems(["s", "ms", "us", "ns"])
        interval_host = QWidget(self.tabular_page)
        interval_layout = QHBoxLayout(interval_host)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        self.interval = _double(1.0, minimum=1.0e-12, maximum=1.0e15)
        self.interval_units = _NoWheelComboBox(interval_host)
        self.interval_units.addItems(["s", "ms", "us", "ns"])
        self.interval_units.setCurrentText("ms")
        interval_layout.addWidget(self.interval, 1)
        interval_layout.addWidget(self.interval_units)
        self.signal_column = _NoWheelComboBox(self.tabular_page)
        self.signal_name = QLineEdit(self.tabular_page)
        self.scale = _double(1.0)
        self.offset = _double(0.0)
        table_form.addRow("Delimiter", self.delimiter)
        table_form.addRow("Header", self.header)
        table_form.addRow("Encoding", self.encoding)
        table_form.addRow("Time column", self.time_column)
        table_form.addRow("Time units", self.time_units)
        table_form.addRow("If no time column", interval_host)
        table_form.addRow("Signal column", self.signal_column)
        table_form.addRow("Signal name", self.signal_name)
        table_form.addRow(
            "Waveform encoding",
            QLabel(f"{self.target_quantity.title()} · {self.target_unit}", self.tabular_page),
        )
        table_form.addRow("Scale", self.scale)
        table_form.addRow("Offset", self.offset)
        # Preserve the form's natural height so a short Waveform-tools card
        # produces a scrollbar instead of vertically squashing the controls.
        self.tabular_page.setMinimumHeight(max(330, table_form.sizeHint().height() + 12))
        self.tabular_scroll.setWidget(self.tabular_page)
        self.mode_stack.addWidget(self.tabular_scroll)

        self.summary = QLabel(
            f"Choose an import format and source file. The selected signal will be appended as {self.target_quantity.title()} ({self.target_unit}).",
            self,
        )
        self.summary.setWordWrap(True)
        self.summary.setProperty("role", "muted")
        root.addWidget(self.summary)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.apply_button = action_button("Apply", self, self.apply, role="primary")
        self.apply_button.setEnabled(False)
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self._on_cancel)
        actions.addWidget(self.apply_button)
        actions.addWidget(cancel_button)
        root.addLayout(actions)

        self._populate_formats()

        self.format_combo.currentIndexChanged.connect(self._format_changed)
        self.signal_combo.currentIndexChanged.connect(self.refresh_preview)
        self.file_path.editingFinished.connect(self._load_source)
        self.delimiter.currentIndexChanged.connect(self._reload_tabular_source)
        self.header.currentIndexChanged.connect(self._reload_tabular_source)
        self.encoding.currentIndexChanged.connect(self._reload_tabular_source)
        self.time_column.currentIndexChanged.connect(self._time_column_changed)
        self.time_units.currentIndexChanged.connect(self.refresh_preview)
        self.interval.valueChanged.connect(self.refresh_preview)
        self.interval_units.currentIndexChanged.connect(self.refresh_preview)
        self.signal_column.currentIndexChanged.connect(self._signal_column_changed)
        self.signal_name.textChanged.connect(self.refresh_preview)
        self.scale.valueChanged.connect(self.refresh_preview)
        self.offset.valueChanged.connect(self.refresh_preview)
        self._format_changed()

    def _populate_formats(self) -> None:
        self.format_combo.clear()
        self.format_combo.addItem("Text / CSV table", ("tabular", ""))
        for reader in DEFAULT_WAVEFORM_IO_REGISTRY.readers:
            self.format_combo.addItem(str(reader.display_name), ("reader", str(reader.format_id)))
        profiles = (*builtin_waveform_format_profiles(), *self.store.profiles())
        for profile in profiles:
            self._profiles[str(profile.profile_id)] = profile
            self.format_combo.addItem(f"{profile.name} (format profile)", ("profile", str(profile.profile_id)))

    def _mode(self) -> tuple[str, str]:
        data = self.format_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return str(data[0]), str(data[1])
        return "tabular", ""

    def _format_changed(self, *_args) -> None:
        kind, _identifier = self._mode()
        if kind == "tabular":
            page = self.tabular_scroll
        else:
            page = self.structured_page
        self.mode_stack.setCurrentWidget(page)
        self._decoded_document = None
        self._table = None
        self._current_document = None
        self.apply_button.setEnabled(False)
        current_path = self.file_path.text().strip()
        if current_path:
            self._load_source()
        else:
            self.summary.setText(
                f"Choose a source file to preview this import as {self.target_quantity.title()} ({self.target_unit})."
            )

    def _browse(self, *_args) -> None:
        kind, identifier = self._mode()
        if kind == "tabular":
            file_filter = "Text tables (*.csv *.txt *.tsv *.dat);;All files (*)"
        elif kind == "profile":
            profile = self._profiles.get(identifier)
            suffixes = tuple(profile.file_suffixes) if profile is not None else ()
            patterns = " ".join(f"*{suffix}" for suffix in suffixes) if suffixes else "*"
            label = profile.name if profile is not None else "Waveform"
            file_filter = f"{label} ({patterns});;All files (*)"
        else:
            reader = DEFAULT_WAVEFORM_IO_REGISTRY.reader(identifier)
            suffixes = tuple(getattr(reader, "suffixes", ()))
            patterns = " ".join(f"*{suffix}" for suffix in suffixes) if suffixes else "*"
            file_filter = f"{reader.display_name} ({patterns});;All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Choose waveform file", "", file_filter)
        if not path:
            return
        self.file_path.setText(path)
        self._load_source()

    def _reload_tabular_source(self, *_args) -> None:
        if self._loading or self._mode()[0] != "tabular" or not self.file_path.text().strip():
            return
        self._load_source()

    def _load_source(self, *_args) -> None:
        if self._loading:
            return
        path_text = self.file_path.text().strip()
        if not path_text:
            return
        source = Path(path_text)
        kind, identifier = self._mode()
        if kind == "profile":
            profile = self._profiles[identifier]
            if profile.time_mode == "runtime_interval" or profile.conversion_mode == "runtime_linear":
                if self._on_runtime_profile_source is not None:
                    self._on_runtime_profile_source(profile, source)
                return
        self._loading = True
        try:
            if kind == "tabular":
                table = read_text_table(
                    source,
                    delimiter_mode=str(self.delimiter.currentData() or "auto"),
                    header_mode=str(self.header.currentData() or "auto"),
                    encoding=str(self.encoding.currentData() or "utf-8-sig"),
                )
                self._table = table
                self._decoded_document = None
                self._populate_table_columns(table)
            elif kind == "profile":
                profile = self._profiles[identifier]
                parsed = parse_waveform_file(source, profile)
                self._decoded_document = parsed.document(title=source.stem, source_path=str(source))
                self._table = None
                self._populate_signals(self._decoded_document)
            else:
                self._decoded_document = DEFAULT_WAVEFORM_IO_REGISTRY.read(source, format_id=identifier)
                self._table = None
                self._populate_signals(self._decoded_document)
        except Exception as error:  # noqa: BLE001
            self._decoded_document = None
            self._table = None
            self._current_document = None
            self.apply_button.setEnabled(False)
            self.summary.setText(f"Unable to read source: {error}")
            self._loading = False
            return
        self._loading = False
        self.refresh_preview()

    def _populate_signals(self, document: WaveformDocument) -> None:
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        for index, signal in enumerate(document.channels):
            self.signal_combo.addItem(
                f"{signal.name} — {signal.quantity.title()} {signal.unit or '(units unspecified)'}",
                index,
            )
        self.signal_combo.setCurrentIndex(0 if self.signal_combo.count() else -1)
        self.signal_combo.blockSignals(False)

    @staticmethod
    def _preferred_time_column(table: TextTable) -> int:
        if table.column_count <= 1:
            return -1
        preferred = 0
        for index, header in enumerate(table.headers):
            if any(token in header.casefold() for token in ("time", "timestamp", "seconds", "sec")):
                preferred = index
                break
        return preferred

    @staticmethod
    def _time_unit_for_header(header: str) -> str:
        text = str(header).casefold()
        if any(token in text for token in ("(ns)", "[ns]", " nanosecond")):
            return "ns"
        if any(token in text for token in ("(us)", "[us]", "µs", "μs", " microsecond")):
            return "us"
        if any(token in text for token in ("(ms)", "[ms]", " millisecond")):
            return "ms"
        return "s"

    def _populate_table_columns(self, table: TextTable) -> None:
        time_column = self._preferred_time_column(table)
        self.time_column.blockSignals(True)
        self.time_column.clear()
        self.time_column.addItem("No time column — fixed interval", -1)
        for index, header in enumerate(table.headers):
            self.time_column.addItem(f"{index + 1}: {header}", index)
        self.time_column.setCurrentIndex(max(0, self.time_column.findData(time_column)))
        self.time_column.blockSignals(False)
        if time_column >= 0:
            self.time_units.setCurrentText(self._time_unit_for_header(table.headers[time_column]))

        self.signal_column.blockSignals(True)
        self.signal_column.clear()
        for index, header in enumerate(table.headers):
            if index != time_column:
                self.signal_column.addItem(f"{index + 1}: {header}", index)
        self.signal_column.setCurrentIndex(0 if self.signal_column.count() else -1)
        self.signal_column.blockSignals(False)
        self._time_column_changed(update_signal=False)
        self._signal_column_changed(update_preview=False)

    def _time_column_changed(self, *_args, update_signal: bool = True) -> None:
        if self._loading:
            return
        data = self.time_column.currentData()
        time_column = int(data) if data is not None else -1
        explicit = time_column >= 0
        self.time_units.setEnabled(explicit)
        self.interval.setEnabled(not explicit)
        self.interval_units.setEnabled(not explicit)
        if update_signal and self._table is not None:
            wanted = self.signal_column.currentData()
            self.signal_column.blockSignals(True)
            self.signal_column.clear()
            for index, header in enumerate(self._table.headers):
                if index != time_column:
                    self.signal_column.addItem(f"{index + 1}: {header}", index)
            index = self.signal_column.findData(wanted)
            self.signal_column.setCurrentIndex(index if index >= 0 else (0 if self.signal_column.count() else -1))
            self.signal_column.blockSignals(False)
            self._signal_column_changed(update_preview=False)
        self.refresh_preview()

    def _signal_column_changed(self, *_args, update_preview: bool = True) -> None:
        if self._loading or self._table is None:
            return
        data = self.signal_column.currentData()
        if data is not None and int(data) >= 0:
            header = str(self._table.headers[int(data)])
            self.signal_name.blockSignals(True)
            self.signal_name.setText(header)
            self.signal_name.blockSignals(False)
        if update_preview:
            self.refresh_preview()

    @staticmethod
    def _quantity_from_signal(signal: WaveformSeries) -> str:
        quantity = str(signal.quantity or "").strip().lower()
        if quantity in {"current", "voltage"}:
            return quantity
        unit = str(signal.unit or "").strip()
        if not unit:
            return ""
        try:
            detected, _si_unit, _scale = signal_unit_to_si(unit)
        except WaveformIOError:
            return ""
        return detected

    def _structured_document(self) -> WaveformDocument:
        if self._decoded_document is None:
            raise ValueError("Choose and read a source file first.")
        data = self.signal_combo.currentData()
        if data is None:
            raise ValueError("Choose a source signal to import.")
        index = int(data)
        split = self._decoded_document.split_signals()
        if index < 0 or index >= len(split):
            raise ValueError("The selected source signal is no longer available.")
        source = split[index]
        signal = source.signal
        source_quantity = self._quantity_from_signal(signal)
        if source_quantity and source_quantity != self.target_quantity:
            raise ValueError(
                f"The selected source is {source_quantity.title()}, but this waveform is encoded as "
                f"{self.target_quantity.title()}. Edit the waveform encoding first if that is intentional."
            )

        source_unit = str(signal.unit or "").strip()
        factor = 1.0
        if source_unit and source_unit != self.target_unit:
            try:
                _source_quantity, _source_si, source_scale = signal_unit_to_si(
                    source_unit, quantity=self.target_quantity
                )
                _target_quantity, _target_si, target_scale = signal_unit_to_si(
                    self.target_unit, quantity=self.target_quantity
                )
            except WaveformIOError as error:
                raise ValueError(
                    f"The source unit {source_unit!r} cannot be converted automatically to the fixed "
                    f"waveform unit {self.target_unit!r}."
                ) from error
            factor = source_scale / target_scale

        converted = WaveformSeries(
            signal.name,
            signal.time_s,
            [float(value) * factor for value in signal.values],
            quantity=self.target_quantity,
            unit=self.target_unit,
            metadata=dict(signal.metadata),
        )
        metadata = dict(source.metadata)
        settings = metadata.get("import_settings")
        settings = dict(settings) if isinstance(settings, dict) else {}
        settings["target_quantity"] = self.target_quantity
        settings["target_unit"] = self.target_unit
        if source_unit and source_unit != self.target_unit:
            settings["source_unit"] = source_unit
            settings["unit_conversion_factor"] = factor
        metadata["import_settings"] = settings
        return WaveformDocument(
            [converted],
            title=source.title,
            metadata=metadata,
            source_format=source.source_format,
            source_path=source.source_path,
        )

    def _tabular_document(self) -> WaveformDocument:
        if self._table is None:
            raise ValueError("Choose and read a text/CSV source file first.")
        signal_data = self.signal_column.currentData()
        if signal_data is None:
            raise ValueError("Choose a signal column to import.")
        signal_column = int(signal_data)
        time_data = self.time_column.currentData()
        time_column = None if time_data is None or int(time_data) < 0 else int(time_data)
        name = self.signal_name.text().strip() or str(self._table.headers[signal_column])
        specs = (
            TabularSignalSpec(
                column=signal_column,
                name=name,
                quantity=self.target_quantity,
                unit=self.target_unit,
                scale=float(self.scale.value()),
                offset=float(self.offset.value()),
            ),
        )
        documents = documents_from_text_table(
            self._table,
            specs,
            time_column=time_column,
            time_unit=self.time_units.currentText(),
            sample_interval=float(self.interval.value()),
            sample_interval_unit=self.interval_units.currentText(),
        )
        if len(documents) != 1:
            raise ValueError("The inline importer must resolve exactly one signal.")
        return documents[0]

    def build_document(self) -> WaveformDocument:
        document = self._tabular_document() if self._mode()[0] == "tabular" else self._structured_document()
        waveform_signal_to_payload(document)
        return document

    def refresh_preview(self, *_args) -> None:
        if self._loading or not self.file_path.text().strip():
            return
        try:
            document = self.build_document()
            source_name = Path(self.file_path.text().strip()).name
            message = self._on_preview(document, f"Imported waveform: {source_name}")
        except Exception as error:  # noqa: BLE001 - live validation boundary
            self._current_document = None
            self.apply_button.setEnabled(False)
            self.summary.setText(f"Unable to preview import: {error}")
            return
        self._current_document = document
        self.apply_button.setEnabled(True)
        signal = document.signal
        self.summary.setText(
            f"{message} {signal.sample_count:,} samples · appended as {self.target_quantity.title()} ({self.target_unit})."
        )

    def apply(self) -> None:
        try:
            document = self.build_document()
        except Exception as error:  # noqa: BLE001
            self.summary.setText(f"Unable to apply import: {error}")
            return
        source_name = Path(self.file_path.text().strip()).name
        self._on_apply(document, f"Imported waveform: {source_name}")
