"""Saved exposure library and editor dialogs.

Exposures are first-class project data. The editor authors an ordered source/load
sequence: timed steps contain one or more drive groups, and each drive group maps
a saved waveform source onto physical coil loads with explicit wiring topology.
"""

from __future__ import annotations

import copy
import html
import re
import math
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QPlainTextEdit,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .exposures import (
    EXPOSURE_FILE_SUFFIX,
    default_exposure_protocol,
    new_exposure,
    normalize_exposure_protocol,
)
from .project_context import ProjectContext
from .studio_adapter import StudioAdapter, StudioOperationError
from .waveform import drive_coil_catalog
from .window_utils import (
    centre_window_on_parent,
    enable_standard_window_controls,
    install_persistent_window_settings,
    install_editor_wheel_guard,
)


def _exposure_notes_html(notes: str) -> str:
    """Show plain-text notes with safe, clickable HTTP(S) publication links."""
    pieces = []
    end = 0
    for match in re.finditer(r"https?://[^\s<>\"]+", notes):
        url = match.group().rstrip(".,;:!?")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        pieces.append(html.escape(notes[end:match.start()]))
        escaped = html.escape(url, quote=True)
        pieces.append(f'<a href="{escaped}">{escaped}</a>')
        end = match.start() + len(url)
    pieces.append(html.escape(notes[end:]))
    return '<div style="white-space: pre-wrap;">' + "".join(pieces) + '</div>'


def _engineering(value: float, unit: str) -> str:
    number = abs(float(value))
    if number == 0.0:
        return f"0 {unit}"
    if unit == "A":
        scale, suffix = ((1e-6, "µA") if number < 1e-3 else (1e-3, "mA") if number < 1.0 else (1.0, "A"))
    elif unit == "V":
        scale, suffix = ((1e-6, "µV") if number < 1e-3 else (1e-3, "mV") if number < 1.0 else (1.0, "V"))
    elif unit == "H":
        scale, suffix = ((1e-6, "µH") if number < 1e-3 else (1e-3, "mH") if number < 1.0 else (1.0, "H"))
    elif unit == "Ω":
        scale, suffix = ((1e-3, "mΩ") if number < 1.0 else (1.0, "Ω") if number < 1000.0 else (1000.0, "kΩ"))
    elif unit == "W":
        scale, suffix = ((1e-6, "µW") if number < 1e-3 else (1e-3, "mW") if number < 1.0 else (1.0, "W"))
    else:
        scale, suffix = 1.0, unit
    return f"{float(value) / scale:.6g} {suffix}"


def _format_drive_diagnostics(
    diagnostic: dict[str, Any],
    coil_label: Callable[[str], str] | None = None,
    *,
    compact: bool = False,
) -> str:
    label = coil_label or (lambda value: value)
    load = (
        f"Rₑq {_engineering(diagnostic['equivalent_resistance_ohm'], 'Ω')} · "
        f"Lₑq {_engineering(diagnostic['equivalent_inductance_h'], 'H')}"
    )
    peaks = (
        f"source peak |I| {_engineering(diagnostic['peak_source_current_a'], 'A')} · "
        f"|V| {_engineering(diagnostic['peak_source_voltage_v'], 'V')}"
    )
    if compact:
        warning_count = len(diagnostic.get("warnings", []))
        suffix = f" · {warning_count} warning" + ("" if warning_count == 1 else "s") if warning_count else " · OK"
        return load + " · " + peaks + suffix

    source_kind = str(diagnostic.get("source_kind", "")).title() or "Source"
    source_line = f"Source: {source_kind}"
    if str(diagnostic.get("source_kind", "")) == "voltage":
        source_line += f" · output R {_engineering(diagnostic.get('source_output_resistance_ohm', 0.0), 'Ω')}"
    lines = [
        "Preview basis: one saved waveform from zero initial coil current.",
        source_line,
        f"Effective load: {load}",
        f"Expected source peaks: |I| {_engineering(diagnostic['peak_source_current_a'], 'A')} · "
        f"|V| {_engineering(diagnostic['peak_source_voltage_v'], 'V')} · "
        f"|P| {_engineering(diagnostic['peak_source_power_w'], 'W')}",
        f"Peak load-terminal voltage: {_engineering(diagnostic['peak_load_terminal_voltage_v'], 'V')}",
    ]
    if int(diagnostic.get("network_group_count", 1)) > 1:
        lines.insert(
            2,
            f"Source network: {int(diagnostic['network_group_count'])} drive groups in parallel at the source terminals.",
        )
    branches = diagnostic.get("branch_peaks", {})
    if branches:
        lines.append("Branches:")
        for coil_id, values in branches.items():
            resistance_mode = str(values.get("resistance_mode", "")).lower()
            inductance_mode = str(values.get("inductance_mode", "")).lower()
            resistance_source = (
                "drive override" if resistance_mode == "drive_override"
                else "model override" if resistance_mode == "override"
                else "calculated"
            )
            inductance_source = (
                "drive override" if inductance_mode == "drive_override"
                else "model override" if inductance_mode == "override"
                else "estimated"
            )
            lines.append(
                f"  {label(str(coil_id))}: |I| {_engineering(values['peak_current_a'], 'A')} · "
                f"|V| {_engineering(values['peak_voltage_v'], 'V')} · "
                f"R {_engineering(values['resistance_ohm'], 'Ω')} ({resistance_source}) · "
                f"L {_engineering(values['inductance_h'], 'H')} ({inductance_source})"
            )
    for warning in diagnostic.get("warnings", []):
        lines.append("Warning: " + str(warning))
    for note in diagnostic.get("notes", []):
        lines.append("Note: " + str(note))
    return "\n".join(lines)


class _DriveGroupDialog(QDialog):
    """Choose one waveform source and the coil load group it drives."""

    def __init__(
        self,
        adapter: StudioAdapter,
        coil_catalog: list[dict[str, Any]],
        group: dict[str, Any] | None = None,
        parent=None,
        *,
        sources: list[dict[str, Any]] | None = None,
        peer_groups: list[dict[str, Any]] | None = None,
        open_waveform_editor: Callable[[str | None], None] | None = None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.sources = copy.deepcopy(sources or [])
        self.peer_groups = copy.deepcopy(peer_groups or [])
        self.open_waveform_editor = open_waveform_editor
        self._source = copy.deepcopy(group) if isinstance(group, dict) else {}
        self._load_override_drafts: dict[str, dict[str, str]] = {}
        for coil_id, values in (self._source.get("load_overrides", {}) or {}).items():
            if not isinstance(values, dict):
                continue
            draft: dict[str, str] = {}
            if "resistance_ohm" in values:
                draft["resistance_ohm"] = f"{float(values['resistance_ohm']):.12g}"
            if "inductance_h" in values:
                draft["inductance_mh"] = f"{float(values['inductance_h']) * 1000.0:.12g}"
            if draft:
                self._load_override_drafts[str(coil_id)] = draft
        self._override_editors: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        self.result_group: dict[str, Any] | None = None
        self.setWindowTitle("Edit drive group" if group else "Add drive group")
        self.resize(700, 780)
        self._coil_labels = {
            str(choice.get("id", "")): str(choice.get("label") or choice.get("id") or "")
            for choice in coil_catalog
            if str(choice.get("id", ""))
        }

        root = QVBoxLayout(self)
        intro = QLabel(
            "A drive group connects one waveform source to one or more coils. "
            "The connection determines how those coil loads are combined electrically. "
            "If the same source is used by several groups in one step, those groups are parallel routes at the source terminals."
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self.waveform_combo = QComboBox()
        for row in self.sources:
            waveform_id = str(row.get("id", "") or "").strip()
            if waveform_id:
                self.waveform_combo.addItem(str(row.get("name") or "Waveform"), waveform_id)
        existing_id = str(self._source.get("source_id", "") or "").strip()
        if existing_id and self.waveform_combo.findData(existing_id) < 0:
            self.waveform_combo.addItem(
                f"[Missing] {existing_id}", existing_id
            )
        if existing_id:
            index = self.waveform_combo.findData(existing_id)
            if index >= 0:
                self.waveform_combo.setCurrentIndex(index)
        form.addRow("Source", self.waveform_combo)

        self.connection = QComboBox()
        self.connection.addItem("Series", "series")
        self.connection.addItem("Parallel", "parallel")
        connection = str(self._source.get("connection", "series") or "series")
        index = self.connection.findData(connection)
        if index >= 0:
            self.connection.setCurrentIndex(index)
        form.addRow("Connection", self.connection)
        root.addLayout(form)

        coil_label = QLabel("Coils")
        coil_label.setStyleSheet("font-weight: 600;")
        root.addWidget(coil_label)
        coil_hint = QLabel(
            "Select the physical coil loads connected to this source. Model R/L values are used by default; "
            "optional per-coil overrides below apply only to this drive group."
        )
        coil_hint.setObjectName("hint")
        coil_hint.setWordWrap(True)
        root.addWidget(coil_hint)
        self.coil_list = QListWidget()
        self.coil_list.setAlternatingRowColors(True)
        selected = {str(value) for value in self._source.get("coil_ids", [])}
        available = set()
        for choice in coil_catalog:
            coil_id = str(choice.get("id", "") or "")
            if not coil_id:
                continue
            available.add(coil_id)
            item = QListWidgetItem(str(choice.get("label") or coil_id))
            item.setData(Qt.ItemDataRole.UserRole, coil_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if coil_id in selected else Qt.CheckState.Unchecked)
            self.coil_list.addItem(item)
        for missing_id in sorted(selected - available):
            item = QListWidgetItem(f"[Missing] {missing_id}")
            item.setData(Qt.ItemDataRole.UserRole, missing_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.coil_list.addItem(item)
        root.addWidget(self.coil_list, 1)

        override_hint = QLabel(
            "Optional load overrides for selected coils — leave a cell blank to use the model value."
        )
        override_hint.setObjectName("hint")
        override_hint.setWordWrap(True)
        root.addWidget(override_hint)
        self.load_override_table = QTreeWidget()
        self.load_override_table.setHeaderLabels(["Coil", "Resistance override (Ω)", "Inductance override (mH)"])
        self.load_override_table.setRootIsDecorated(False)
        self.load_override_table.setAlternatingRowColors(True)
        self.load_override_table.setMinimumHeight(155)
        self.load_override_table.setMaximumHeight(220)
        self.load_override_table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.load_override_table.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.load_override_table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.load_override_table, 1)

        self.summary = QLabel()
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        diagnostic_heading = QLabel("Electrical preview")
        diagnostic_heading.setStyleSheet("font-weight: 600;")
        root.addWidget(diagnostic_heading)
        self.diagnostics = QLabel()
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.diagnostics)

        self.waveform_combo.currentIndexChanged.connect(self._update_summary)
        self.connection.currentIndexChanged.connect(self._update_summary)
        self.coil_list.itemChanged.connect(self._coils_changed)
        self._refresh_load_override_table()
        self._update_summary()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._wheel_guard = install_editor_wheel_guard(self)

    def refresh_waveform_resources(self) -> None:
        """Refresh source names/metadata without changing the draft drive group."""
        current_id = str(self.waveform_combo.currentData() or "").strip()
        if not current_id:
            current_id = str(self._source.get("source_id", "") or "").strip()
        self.waveform_combo.blockSignals(True)
        self.waveform_combo.clear()
        try:
            rows = self.sources
        except Exception:
            rows = []
        for row in rows:
            waveform_id = str(row.get("id", "") or "").strip()
            if waveform_id:
                self.waveform_combo.addItem(str(row.get("name") or "Waveform"), waveform_id)
        if current_id and self.waveform_combo.findData(current_id) < 0:
            self.waveform_combo.addItem(f"[Missing] {current_id}", current_id)
        index = self.waveform_combo.findData(current_id) if current_id else -1
        if index >= 0:
            self.waveform_combo.setCurrentIndex(index)
        elif self.waveform_combo.count():
            self.waveform_combo.setCurrentIndex(0)
        self.waveform_combo.blockSignals(False)
        self._update_summary()

    def refresh_model_resources(self, coil_catalog: list[dict[str, Any]]) -> None:
        """Refresh coil labels/availability while preserving checked load IDs."""
        selected = set(self._selected_coils())
        self._coil_labels = {
            str(choice.get("id", "")): str(choice.get("label") or choice.get("id") or "")
            for choice in coil_catalog
            if str(choice.get("id", ""))
        }
        self.coil_list.blockSignals(True)
        self.coil_list.clear()
        available: set[str] = set()
        for choice in coil_catalog:
            coil_id = str(choice.get("id", "") or "")
            if not coil_id:
                continue
            available.add(coil_id)
            item = QListWidgetItem(str(choice.get("label") or coil_id))
            item.setData(Qt.ItemDataRole.UserRole, coil_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if coil_id in selected else Qt.CheckState.Unchecked
            )
            self.coil_list.addItem(item)
        for missing_id in sorted(selected - available):
            item = QListWidgetItem(f"[Missing] {missing_id}")
            item.setData(Qt.ItemDataRole.UserRole, missing_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.coil_list.addItem(item)
        self.coil_list.blockSignals(False)
        self._refresh_load_override_table()
        self._update_summary()

    def _selected_coils(self) -> list[str]:
        result: list[str] = []
        for row in range(self.coil_list.count()):
            item = self.coil_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                value = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if value:
                    result.append(value)
        return result

    def _coils_changed(self, *_args) -> None:
        self._refresh_load_override_table()
        self._update_summary()

    def _set_load_override_draft(self, coil_id: str, key: str, text: str) -> None:
        draft = self._load_override_drafts.setdefault(str(coil_id), {})
        value = str(text).strip()
        if value:
            draft[key] = value
        else:
            draft.pop(key, None)
            if not draft:
                self._load_override_drafts.pop(str(coil_id), None)
        self._update_summary()

    def _refresh_load_override_table(self) -> None:
        self.load_override_table.clear()
        self._override_editors = {}
        for coil_id in self._selected_coils():
            item = QTreeWidgetItem([self._coil_label(coil_id), "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, coil_id)
            self.load_override_table.addTopLevelItem(item)
            draft = self._load_override_drafts.get(coil_id, {})

            resistance = QLineEdit(str(draft.get("resistance_ohm", "")))
            inductance = QLineEdit(str(draft.get("inductance_mh", "")))
            try:
                model_load = self.adapter.coil_electrical_load(coil_id)
                resistance.setPlaceholderText(
                    f"model {float(model_load.get('resistance_ohm', 0.0)):.6g}"
                )
                inductance.setPlaceholderText(
                    f"model {float(model_load.get('inductance_h', 0.0)) * 1000.0:.6g}"
                )
            except Exception:
                resistance.setPlaceholderText("model unavailable")
                inductance.setPlaceholderText("model unavailable")
            resistance.setToolTip("Blank uses the model resistance for this coil.")
            inductance.setToolTip("Blank uses the model self-inductance for this coil.")
            resistance.editingFinished.connect(
                lambda cid=coil_id, editor=resistance: self._set_load_override_draft(
                    cid, "resistance_ohm", editor.text()
                )
            )
            inductance.editingFinished.connect(
                lambda cid=coil_id, editor=inductance: self._set_load_override_draft(
                    cid, "inductance_mh", editor.text()
                )
            )
            self.load_override_table.setItemWidget(item, 1, resistance)
            self.load_override_table.setItemWidget(item, 2, inductance)
            self._override_editors[coil_id] = (resistance, inductance)

    def _selected_load_overrides(self) -> dict[str, dict[str, float]]:
        overrides: dict[str, dict[str, float]] = {}
        for coil_id in self._selected_coils():
            draft = self._load_override_drafts.get(coil_id, {})
            row: dict[str, float] = {}
            editors = self._override_editors.get(coil_id)
            resistance_text = (
                str(editors[0].text()).strip()
                if editors is not None
                else str(draft.get("resistance_ohm", "")).strip()
            )
            inductance_text = (
                str(editors[1].text()).strip()
                if editors is not None
                else str(draft.get("inductance_mh", "")).strip()
            )
            if resistance_text:
                try:
                    resistance = float(resistance_text)
                except ValueError as error:
                    raise ValueError(f"{self._coil_label(coil_id)} resistance override must be numeric.") from error
                if not math.isfinite(resistance) or resistance < 0.0:
                    raise ValueError(
                        f"{self._coil_label(coil_id)} resistance override must be finite and non-negative."
                    )
                row["resistance_ohm"] = resistance
            if inductance_text:
                try:
                    inductance_mh = float(inductance_text)
                except ValueError as error:
                    raise ValueError(f"{self._coil_label(coil_id)} inductance override must be numeric.") from error
                if not math.isfinite(inductance_mh) or inductance_mh <= 0.0:
                    raise ValueError(
                        f"{self._coil_label(coil_id)} inductance override must be finite and greater than zero."
                    )
                row["inductance_h"] = inductance_mh / 1000.0
            if row:
                overrides[coil_id] = row
        return overrides

    def _open_waveform(self, *_args) -> None:
        if self.open_waveform_editor is not None:
            self.open_waveform_editor(self._selected_waveform_id() or None)

    def _selected_waveform_id(self) -> str:
        source_id = str(self.waveform_combo.currentData() or "")
        return next((source["waveform_id"] for source in self.sources if source["id"] == source_id), "")

    def _candidate_group(self) -> dict[str, Any]:
        group = {
            "id": str(self._source.get("id") or ""),
            "waveform_id": self._selected_waveform_id(),
            "coil_ids": self._selected_coils(),
            "connection": str(self.connection.currentData() or "series"),
        }
        overrides = self._selected_load_overrides()
        if overrides:
            group["load_overrides"] = overrides
        return group

    def _coil_label(self, coil_id: str) -> str:
        return self._coil_labels.get(str(coil_id), f"[Missing] {coil_id}")

    def _update_summary(self, *_args) -> None:
        waveform_id = self._selected_waveform_id()
        if not waveform_id:
            self.summary.setText("Add an exposure source before adding a route.")
            self.diagnostics.setText("Choose a waveform source and at least one coil to preview the electrical load.")
            return
        try:
            resource = self.adapter.waveform_resource(waveform_id)
            document = self.adapter.waveform_document(waveform_id)
            signal = document.signal
            kind = str(signal.quantity).title()
            unit = str(signal.unit or "")
            electrical = document.metadata.get("electrical", {})
            extra = ""
            if signal.quantity == "voltage":
                extra = f" · source R={float(electrical.get('output_resistance_ohm', 0.0)):.6g} Ω"
            selected_count = len(self._selected_coils())
            self.summary.setText(
                f"{resource.get('name', 'Waveform')} · {kind} ({unit}) · {signal.duration_s:.6g} s{extra}. "
                f"{selected_count} coil(s), {self.connection.currentText().lower()}."
            )
            if not selected_count:
                self.diagnostics.setText("Select at least one coil to preview the resolved electrical load.")
            else:
                source_id = str(self.waveform_combo.currentData() or "")
                peers = [
                    copy.deepcopy(group)
                    for group in self.peer_groups
                    if str(group.get("source_id", "") or "") == source_id
                ]
                diagnostic = self.adapter.exposure_source_network_diagnostics(
                    peers + [self._candidate_group()],
                    waveform_id=waveform_id,
                )
                self.diagnostics.setText(_format_drive_diagnostics(diagnostic, self._coil_label))
        except Exception as error:  # noqa: BLE001
            self.summary.setText(str(error))
            self.diagnostics.setText("Electrical preview unavailable: " + str(error))

    def _accept(self) -> None:
        waveform_id = self._selected_waveform_id().strip()
        if not waveform_id:
            QMessageBox.warning(self, "Drive group", "Choose a waveform source.")
            return
        coil_ids = self._selected_coils()
        if not coil_ids:
            QMessageBox.warning(self, "Drive group", "Choose at least one coil.")
            return
        try:
            self.adapter.waveform_resource(waveform_id)
            self.adapter.waveform_document(waveform_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Drive group", str(error))
            return
        try:
            load_overrides = self._selected_load_overrides()
        except ValueError as error:
            QMessageBox.warning(self, "Drive group", str(error))
            return
        self.result_group = {
            "id": str(self._source.get("id") or f"drive_{uuid4().hex}"),
            "source_id": str(self.waveform_combo.currentData() or ""),
            "coil_ids": coil_ids,
            "connection": str(self.connection.currentData() or "series"),
            **({"load_overrides": load_overrides} if load_overrides else {}),
        }
        self.accept()


class ExposureEditorDialog(QDialog):
    """Author an exposure as timed steps containing source-to-load drive groups."""

    def __init__(
        self,
        adapter: StudioAdapter,
        exposure: dict[str, Any] | None = None,
        parent=None,
        *,
        project_context: ProjectContext | None = None,
        open_waveform_editor: Callable[[str | None], None] | None = None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.adapter = adapter
        self.project_context = project_context
        self.open_waveform_editor = open_waveform_editor
        self._source = copy.deepcopy(exposure) if isinstance(exposure, dict) else new_exposure()
        self.result_exposure: dict[str, Any] | None = None
        self._coil_catalog = drive_coil_catalog(adapter)
        self.setWindowTitle("Edit exposure" if exposure is not None else "New exposure")
        self.resize(900, 700)

        protocol = normalize_exposure_protocol(self._source.get("protocol", {}))
        self._sources = copy.deepcopy(protocol["sources"])
        self._steps = copy.deepcopy(protocol.get("sequence", {}).get("steps", []))
        if not self._steps:
            self._steps = [{"id": f"step_{uuid4().hex}", "duration_s": 1.0, "drive_groups": []}]

        root = QVBoxLayout(self)
        heading = QLabel("Exposure")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        intro = QLabel(
            "Sources can run continuously from exposure time zero or fire once at the start of each "
            "routing step that uses them. Routing steps choose which coils each source drives."
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        root.addWidget(intro)

        identity = QWidget()
        identity_form = QFormLayout(identity)
        identity_form.setContentsMargins(0, 0, 0, 4)
        self.name_edit = QLineEdit(str(self._source.get("name", "Exposure")))
        identity_form.addRow("Name", self.name_edit)
        duration_widget = QWidget()
        duration_layout = QHBoxLayout(duration_widget)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_seconds = float(protocol["duration_s"])
        if duration_seconds >= 3600.0:
            initial_duration_scale = 3600.0
            initial_duration_unit = "hour"
        elif duration_seconds >= 60.0:
            initial_duration_scale = 60.0
            initial_duration_unit = "min"
        else:
            initial_duration_scale = 1.0
            initial_duration_unit = "s"
        self._duration_unit_scale = initial_duration_scale
        self.duration = QDoubleSpinBox()
        self.duration.setRange(1e-12, 1e12)
        self.duration.setDecimals(6)
        self.duration.setValue(duration_seconds / initial_duration_scale)
        self.duration_unit = QComboBox()
        self.duration_unit.addItem("s", 1.0)
        self.duration_unit.addItem("min", 60.0)
        self.duration_unit.addItem("hour", 3600.0)
        self.duration_unit.setCurrentIndex(self.duration_unit.findText(initial_duration_unit))
        self.duration_unit.currentIndexChanged.connect(self._change_duration_unit)
        duration_layout.addWidget(self.duration, 1)
        duration_layout.addWidget(self.duration_unit)
        identity_form.addRow("Exposure duration", duration_widget)
        identity.setMaximumWidth(360)
        identity_form.setAlignment(Qt.AlignmentFlag.AlignTop)
        notes_panel = QWidget()
        notes_layout = QVBoxLayout(notes_panel)
        notes_layout.setContentsMargins(0, 0, 0, 0)
        notes_layout.addWidget(QLabel("Notes"))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setObjectName("exposureNotesEditor")
        self.notes_edit.setPlainText(str(self._source.get("description", "") or ""))
        self.notes_edit.setPlaceholderText("How this exposure was derived, publication links, and other notes…")
        self.notes_edit.setMinimumHeight(95)
        self.notes_edit.setMaximumHeight(160)
        notes_layout.addWidget(self.notes_edit)
        identity_row = QHBoxLayout()
        identity_row.addWidget(identity, 2, Qt.AlignmentFlag.AlignTop)
        identity_row.addWidget(notes_panel, 3)
        root.addLayout(identity_row)

        root.addWidget(QLabel("Sources"))
        source_hint = QLabel(
            "Edit sources directly. Continuous sources follow the exposure clock; Once per step "
            "plays one waveform cycle from the selected phase, then commands zero until the next step."
        )
        source_hint.setObjectName("hint")
        source_hint.setWordWrap(True)
        root.addWidget(source_hint)
        self.source_table = QTreeWidget()
        self.source_table.setHeaderLabels(["Source", "Waveform", "Phase (s)", "Mode"])
        self.source_table.setRootIsDecorated(False)
        self.source_table.setAlternatingRowColors(True)
        self.source_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.source_table.setMaximumHeight(160)
        self.source_table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.source_table.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.source_table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.source_table.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.source_table)
        source_buttons = QHBoxLayout()
        self.add_source_button = QPushButton("Add source")
        self.add_source_button.clicked.connect(self._add_source)
        self.remove_source_button = QPushButton("Remove source")
        self.remove_source_button.clicked.connect(self._remove_source)
        source_buttons.addWidget(self.add_source_button)
        source_buttons.addWidget(self.remove_source_button)
        source_buttons.addStretch(1)
        root.addLayout(source_buttons)
        self._refresh_sources()
        root.addWidget(QLabel("Routing schedule"))

        self.sequence_tree = QTreeWidget()
        self.sequence_tree.setHeaderLabels(["Step / drive", "Duration", "Source", "Coils", "Connection"])
        self.sequence_tree.setRootIsDecorated(True)
        self.sequence_tree.setAlternatingRowColors(True)
        self.sequence_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sequence_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.sequence_tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.sequence_tree.itemSelectionChanged.connect(self._update_buttons)
        self.sequence_tree.itemDoubleClicked.connect(self._edit_selected)
        root.addWidget(self.sequence_tree, 1)

        row = QHBoxLayout()
        self.add_step_button = QPushButton("Add step")
        self.add_step_button.clicked.connect(self._add_step)
        self.duplicate_step_button = QPushButton("Duplicate")
        self.duplicate_step_button.setToolTip(
            "Clone the selected step, or clone the selected drive group within its step."
        )
        self.duplicate_step_button.clicked.connect(self._duplicate_selected)
        self.add_drive_button = QPushButton("Add drive group…")
        self.add_drive_button.setObjectName("primaryButton")
        self.add_drive_button.clicked.connect(self._add_drive_group)
        self.edit_button = QPushButton("Edit…")
        self.edit_button.clicked.connect(self._edit_selected)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setObjectName("dangerButton")
        self.remove_button.clicked.connect(self._remove_selected)
        for button in (
            self.add_step_button,
            self.duplicate_step_button,
            self.add_drive_button,
            self.edit_button,
            self.remove_button,
        ):
            row.addWidget(button)
        self.drive_info_button = QPushButton("Drive info…")
        self.drive_info_button.setToolTip(
            "Show electrical source/load diagnostics for the selected drive group."
        )
        self.drive_info_button.clicked.connect(self._show_drive_group_information)
        row.addWidget(self.drive_info_button)
        row.addStretch(1)
        self.loop_sequence = QCheckBox("Repeat routing schedule until exposure ends")
        self.loop_sequence.setChecked(bool(protocol.get("sequence", {}).get("loop", False)))
        row.addWidget(self.loop_sequence)
        root.addLayout(row)

        self.missing_coil_warning = QLabel()
        self.missing_coil_warning.setObjectName("warningPanel")
        self.missing_coil_warning.setWordWrap(True)
        root.addWidget(self.missing_coil_warning)
        self._update_missing_coil_warning()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("Save exposure")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._wheel_guard = install_editor_wheel_guard(self)

        self._refresh_tree()

    def _resolved_group(self, group):
        source = next((item for item in self._sources if item["id"] == group.get("source_id")), None)
        if source is None:
            raise StudioOperationError("Route references an unavailable exposure source.")
        return {**group, "waveform_id": source["waveform_id"]}

    def _source_name(self, source_id):
        return next((source["name"] for source in self._sources if source["id"] == source_id), "[Missing source]")

    def _change_duration_unit(self, *_args) -> None:
        new_scale = float(self.duration_unit.currentData() or 1.0)
        old_scale = float(getattr(self, "_duration_unit_scale", 1.0) or 1.0)
        seconds = float(self.duration.value()) * old_scale
        self._duration_unit_scale = new_scale
        self.duration.blockSignals(True)
        try:
            self.duration.setValue(seconds / new_scale)
        finally:
            self.duration.blockSignals(False)

    def _source_by_id(self, source_id: str) -> dict[str, Any] | None:
        return next((row for row in self._sources if str(row.get("id", "")) == str(source_id)), None)

    def _selected_source_index(self) -> int:
        item = self.source_table.currentItem()
        if item is None:
            return -1
        source_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        for index, source in enumerate(self._sources):
            if str(source.get("id", "")) == source_id:
                return index
        return -1

    def _refresh_sources(self):
        selected_index = self._selected_source_index() if hasattr(self, "source_table") else -1
        selected_id = (
            str(self._sources[selected_index].get("id", ""))
            if 0 <= selected_index < len(self._sources)
            else ""
        )
        self.source_table.clear()
        waveform_rows = list(self.adapter.list_waveforms())
        waveform_ids = {str(row.get("id", "")) for row in waveform_rows}
        selected_item = None
        for source in self._sources:
            source_id = str(source.get("id", ""))
            item = QTreeWidgetItem(["", "", "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, source_id)
            self.source_table.addTopLevelItem(item)

            name = QLineEdit(str(source.get("name") or source_id))
            name.setMinimumWidth(110)
            name.textEdited.connect(
                lambda _text, row=item: self.source_table.setCurrentItem(row)
            )
            name.editingFinished.connect(
                lambda sid=source_id, editor=name: self._set_source_name(sid, editor.text())
            )
            self.source_table.setItemWidget(item, 0, name)

            waveform = QComboBox()
            for row in waveform_rows:
                waveform.addItem(str(row.get("name") or row.get("id") or "Waveform"), str(row.get("id", "")))
            waveform_id = str(source.get("waveform_id", "") or "")
            if waveform_id and waveform_id not in waveform_ids:
                waveform.addItem(f"[Missing] {waveform_id}", waveform_id)
            index = waveform.findData(waveform_id)
            if index >= 0:
                waveform.setCurrentIndex(index)
            waveform.currentIndexChanged.connect(
                lambda _value, sid=source_id, combo=waveform, row=item: (
                    self.source_table.setCurrentItem(row),
                    self._set_source_waveform(sid, combo.currentData()),
                )
            )
            self.source_table.setItemWidget(item, 1, waveform)

            phase = QDoubleSpinBox()
            phase.setDecimals(6)
            phase.setRange(-1e9, 1e9)
            phase.setValue(float(source.get("phase_offset_s", 0.0)))
            phase.setToolTip("Positive phase starts further into the selected waveform.")
            phase.valueChanged.connect(
                lambda value, sid=source_id, row=item: (
                    self.source_table.setCurrentItem(row),
                    self._set_source_phase(sid, value),
                )
            )
            self.source_table.setItemWidget(item, 2, phase)

            mode = QComboBox()
            mode.addItem("Continuous", "continuous")
            mode.addItem("Once per step", "once_per_step")
            mode_index = mode.findData(str(source.get("playback_mode", "continuous")))
            mode.setCurrentIndex(mode_index if mode_index >= 0 else 0)
            mode.currentIndexChanged.connect(
                lambda _value, sid=source_id, combo=mode, row=item: (
                    self.source_table.setCurrentItem(row),
                    self._set_source_mode(sid, combo.currentData()),
                )
            )
            self.source_table.setItemWidget(item, 3, mode)

            if source_id == selected_id:
                selected_item = item

        if selected_item is not None:
            self.source_table.setCurrentItem(selected_item)
        elif self.source_table.topLevelItemCount():
            self.source_table.setCurrentItem(self.source_table.topLevelItem(0))
        if hasattr(self, "remove_source_button"):
            self.remove_source_button.setEnabled(bool(self._sources))
        # Source rows are rebuilt in place; refresh the window-local wheel
        # guard so newly-created waveform/phase/mode editors inherit it.
        install_editor_wheel_guard(self)

    def _set_source_name(self, source_id: str, text: str) -> None:
        source = self._source_by_id(source_id)
        if source is None:
            return
        value = str(text).strip() or str(source.get("name") or source_id)
        source["name"] = value
        self._refresh_tree()

    def _set_source_waveform(self, source_id: str, waveform_id: Any) -> None:
        source = self._source_by_id(source_id)
        value = str(waveform_id or "").strip()
        if source is None or not value:
            return
        source["waveform_id"] = value
        self._refresh_tree()

    def _set_source_phase(self, source_id: str, value: float) -> None:
        source = self._source_by_id(source_id)
        if source is not None:
            source["phase_offset_s"] = float(value)

    def _set_source_mode(self, source_id: str, mode: Any) -> None:
        source = self._source_by_id(source_id)
        value = str(mode or "continuous")
        if source is not None and value in {"continuous", "once_per_step"}:
            source["playback_mode"] = value

    def _add_source(self, *_):
        waveforms = list(self.adapter.list_waveforms())
        if not waveforms:
            QMessageBox.information(
                self, "Source", "Create a project waveform before adding an exposure source."
            )
            return
        source = {
            "id": f"source_{uuid4().hex}",
            "name": f"Source {len(self._sources) + 1}",
            "waveform_id": str(waveforms[0].get("id", "")),
            "playback_mode": "continuous",
            "phase_offset_s": 0.0,
        }
        if not source["waveform_id"]:
            QMessageBox.warning(self, "Source", "The waveform catalogue contains an invalid entry.")
            return
        self._sources.append(source)
        self._refresh_sources()
        if self.source_table.topLevelItemCount():
            self.source_table.setCurrentItem(self.source_table.topLevelItem(len(self._sources) - 1))
        self._refresh_tree()

    def _remove_source(self, *_):
        index = self._selected_source_index()
        if index < 0:
            return
        source_id = self._sources[index]["id"]
        if any(group.get("source_id") == source_id for step in self._steps for group in step["drive_groups"]):
            QMessageBox.information(self, "Source in use", "Remove or reassign its routes before removing this source.")
            return
        self._sources.pop(index)
        self._refresh_sources()

    def _coil_name(self, coil_id: str) -> str:
        for choice in self._coil_catalog:
            if str(choice.get("id", "")) == str(coil_id):
                return str(choice.get("label") or coil_id)
        return f"[Missing] {coil_id}"

    def _waveform_name(self, waveform_id: str) -> str:
        source_id = str(waveform_id or "").strip()
        if not source_id:
            return "[Choose waveform]"
        try:
            resource = self.adapter.waveform_resource(source_id)
        except Exception:
            return f"[Missing] {source_id}"
        return str(resource.get("name") or source_id)

    def _update_missing_coil_warning(self) -> None:
        referenced = {
            str(coil_id)
            for step in self._steps
            for group in step.get("drive_groups", [])
            for coil_id in group.get("coil_ids", [])
            if str(coil_id)
        }
        available = {str(choice.get("id", "")) for choice in self._coil_catalog}
        missing = sorted(value for value in referenced if value not in available)
        if missing:
            labels = self._source.get("protocol", {}).get("coil_labels", {})
            labels = labels if isinstance(labels, dict) else {}
            names = [str(labels.get(value) or value) for value in missing]
            self.missing_coil_warning.setText(
                "This exposure references coil(s) not present in the current scene: "
                + ", ".join(names)
            )
            self.missing_coil_warning.setVisible(True)
        else:
            self.missing_coil_warning.clear()
            self.missing_coil_warning.setVisible(False)

    def _refresh_tree(self, *, select: tuple[int, int | None] | None = None) -> None:
        self.sequence_tree.clear()
        selected_item = None
        for step_index, step in enumerate(self._steps):
            groups = step.get("drive_groups", [])
            root = QTreeWidgetItem([
                f"Step {step_index + 1}",
                "",
                "",
                f"{len(groups)} drive group" + ("" if len(groups) == 1 else "s"),
                "",
            ])
            root.setData(0, Qt.ItemDataRole.UserRole, ("step", step_index))
            root.setExpanded(True)
            self.sequence_tree.addTopLevelItem(root)

            duration = QDoubleSpinBox()
            duration.setRange(1e-6, 1e9)
            duration.setDecimals(6)
            duration.setSuffix(" s")
            duration.setValue(float(step.get("duration_s", 1.0)))
            duration.setMinimumWidth(105)
            duration.setToolTip("Duration of this routing step in seconds.")
            duration.valueChanged.connect(
                lambda value, index=step_index, row=root: (
                    self.sequence_tree.setCurrentItem(row),
                    self._set_step_duration(index, value),
                )
            )
            self.sequence_tree.setItemWidget(root, 1, duration)
            if select == (step_index, None):
                selected_item = root
            for group_index, group in enumerate(groups):
                coil_names = [self._coil_name(value) for value in group.get("coil_ids", [])]
                child = QTreeWidgetItem([
                    f"Drive {group_index + 1}",
                    "",
                    self._source_name(str(group.get("source_id", "") or "")),
                    ", ".join(coil_names) or "—",
                    str(group.get("connection", "series")).title(),
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, ("drive", step_index, group_index))
                root.addChild(child)
                if select == (step_index, group_index):
                    selected_item = child
        if selected_item is not None:
            self.sequence_tree.setCurrentItem(selected_item)
        elif self.sequence_tree.topLevelItemCount():
            self.sequence_tree.setCurrentItem(self.sequence_tree.topLevelItem(0))
        self._update_missing_coil_warning()
        self._update_buttons()
        # Step-duration editors are rebuilt with the tree, so refresh the
        # window-local wheel guard for the newly-created spin boxes.
        install_editor_wheel_guard(self)

    def _set_step_duration(self, step_index: int, value: float) -> None:
        if 0 <= int(step_index) < len(self._steps):
            self._steps[int(step_index)]["duration_s"] = float(value)

    def _selection(self) -> tuple[str, int, int | None] | None:
        item = self.sequence_tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(data, tuple) or len(data) < 2:
            return None
        return str(data[0]), int(data[1]), int(data[2]) if len(data) > 2 else None

    def _update_buttons(self, *_args) -> None:
        selection = self._selection()
        self.add_drive_button.setEnabled(selection is not None)
        self.duplicate_step_button.setEnabled(
            selection is not None
            and selection[0] in {"step", "drive"}
            and (selection[0] != "drive" or selection[2] is not None)
        )
        self.edit_button.setEnabled(
            selection is not None and selection[0] == "drive" and selection[2] is not None
        )
        self.remove_button.setEnabled(selection is not None)
        self.drive_info_button.setEnabled(
            selection is not None and selection[0] == "drive" and selection[2] is not None
        )

    def _show_drive_group_information(self, *_args) -> None:
        selection = self._selection()
        if selection is None:
            return
        kind, step_index, group_index = selection
        if kind != "drive" or group_index is None:
            return
        step = self._steps[step_index]
        groups = list(step.get("drive_groups", []))
        if group_index < 0 or group_index >= len(groups):
            return
        group = groups[group_index]
        try:
            source_id = str(group.get("source_id", "") or "")
            source = self._source_by_id(source_id)
            if source is None:
                raise StudioOperationError("The selected drive group references a missing exposure source.")
            source_groups = [
                candidate for candidate in groups
                if str(candidate.get("source_id", "") or "") == source_id
            ]
            diagnostic = self.adapter.exposure_source_network_diagnostics(
                source_groups,
                waveform_id=str(source.get("waveform_id", "") or ""),
            )
            text = _format_drive_diagnostics(diagnostic, self._coil_name)
            text += f"\n\nExposure step duration: {float(step.get('duration_s', 0.0)):.6g} s."
            if source is not None and source.get("playback_mode") == "once_per_step":
                text += " This source fires one waveform cycle from its configured phase at the start of this step, then commands zero."
            else:
                text += " This source follows the continuous exposure clock through routing changes."
        except Exception as error:  # noqa: BLE001
            text = "Electrical diagnostics unavailable: " + str(error)

        dialog = QDialog(self)
        enable_standard_window_controls(dialog)
        dialog.setWindowTitle(f"Drive group information — Step {step_index + 1}, Drive {group_index + 1}")
        dialog.resize(680, 460)
        layout = QVBoxLayout(dialog)
        heading = QLabel(
            f"Step {step_index + 1} · Drive {group_index + 1} — "
            f"{self._source_name(str(group.get('source_id', '') or ''))}"
        )
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        details = QLabel(text)
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(details, 1)
        note = QLabel(
            "This is a one-waveform electrical preview from zero initial coil current. "
            "When several drive groups use this source in the same step, they are parallel routes at one source terminal pair. "
            "Actual exposure playback keeps compatible RL current state across routing boundaries; "
            "source playback mode controls only the source waveform clock."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def refresh_waveform_resources(self) -> None:
        """Refresh waveform names/diagnostics while preserving the draft sequence."""
        selection = self._selection()
        selected = None if selection is None else (selection[1], selection[2])
        self._refresh_sources()
        self._refresh_tree(select=selected)
        for dialog in self.findChildren(_DriveGroupDialog):
            if dialog.isVisible():
                dialog.refresh_waveform_resources()

    def refresh_model_resources(self) -> None:
        """Refresh coil choices/labels after model-side load edits."""
        selection = self._selection()
        selected = None if selection is None else (selection[1], selection[2])
        self._coil_catalog = drive_coil_catalog(self.adapter)
        self._refresh_sources()
        self._refresh_tree(select=selected)
        for dialog in self.findChildren(_DriveGroupDialog):
            if dialog.isVisible():
                dialog.refresh_model_resources(self._coil_catalog)

    def refresh_project_resources(self) -> None:
        """Refresh both source and load catalogues after shared-project undo/redo."""
        selection = self._selection()
        selected = None if selection is None else (selection[1], selection[2])
        self._coil_catalog = drive_coil_catalog(self.adapter)
        self._refresh_sources()
        self._refresh_tree(select=selected)
        for dialog in self.findChildren(_DriveGroupDialog):
            if not dialog.isVisible():
                continue
            dialog.refresh_model_resources(self._coil_catalog)
            dialog.refresh_waveform_resources()

    def _selected_step_index(self) -> int | None:
        selection = self._selection()
        if selection is None:
            return None
        return selection[1]

    def _add_step(self, *_args) -> None:
        duration = float(self._steps[-1].get("duration_s", 1.0)) if self._steps else 1.0
        self._steps.append({"id": f"step_{uuid4().hex}", "duration_s": duration, "drive_groups": []})
        self._refresh_tree(select=(len(self._steps) - 1, None))

    def _duplicate_selected(self, *_args) -> None:
        selection = self._selection()
        if selection is None:
            return
        if selection[0] == "step":
            self._duplicate_step()
            return
        if selection[0] == "drive":
            self._duplicate_drive_group()

    def _duplicate_step(self, *_args) -> None:
        selection = self._selection()
        if selection is None or selection[0] != "step":
            return
        step_index = selection[1]
        if step_index < 0 or step_index >= len(self._steps):
            return
        clone = copy.deepcopy(self._steps[step_index])
        clone["id"] = f"step_{uuid4().hex}"
        for group in clone.get("drive_groups", []):
            group["id"] = f"drive_{uuid4().hex}"
        insert_index = step_index + 1
        self._steps.insert(insert_index, clone)
        self._refresh_tree(select=(insert_index, None))

    def _duplicate_drive_group(self, *_args) -> None:
        selection = self._selection()
        if selection is None or selection[0] != "drive" or selection[2] is None:
            return
        _kind, step_index, group_index = selection
        if step_index < 0 or step_index >= len(self._steps):
            return
        groups = self._steps[step_index].setdefault("drive_groups", [])
        if group_index < 0 or group_index >= len(groups):
            return
        clone = copy.deepcopy(groups[group_index])
        clone["id"] = f"drive_{uuid4().hex}"
        insert_index = group_index + 1
        groups.insert(insert_index, clone)
        self._refresh_tree(select=(step_index, insert_index))

    def _add_drive_group(self, *_args) -> None:
        step_index = self._selected_step_index()
        if step_index is None:
            QMessageBox.information(self, "Drive group", "Select a step first.")
            return
        dialog = _DriveGroupDialog(
            self.adapter,
            self._coil_catalog,
            parent=self,
            sources=self._sources,
            peer_groups=list(self._steps[step_index].get("drive_groups", [])),
            open_waveform_editor=self.open_waveform_editor,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_group is None:
            return
        self._steps[step_index].setdefault("drive_groups", []).append(dialog.result_group)
        self._refresh_tree(select=(step_index, len(self._steps[step_index]["drive_groups"]) - 1))

    def _edit_selected(self, *_args) -> None:
        selection = self._selection()
        if selection is None:
            return
        kind, step_index, group_index = selection
        if kind == "step":
            return
        if group_index is None:
            return
        group = self._steps[step_index]["drive_groups"][group_index]
        peer_groups = [
            candidate
            for index, candidate in enumerate(self._steps[step_index]["drive_groups"])
            if index != group_index
        ]
        dialog = _DriveGroupDialog(
            self.adapter, self._coil_catalog, group=group, parent=self, sources=self._sources,
            peer_groups=peer_groups,
            open_waveform_editor=self.open_waveform_editor,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_group is not None:
            self._steps[step_index]["drive_groups"][group_index] = dialog.result_group
            self._refresh_tree(select=(step_index, group_index))

    def _remove_selected(self, *_args) -> None:
        selection = self._selection()
        if selection is None:
            return
        kind, step_index, group_index = selection
        if kind == "drive" and group_index is not None:
            del self._steps[step_index]["drive_groups"][group_index]
            self._refresh_tree(select=(step_index, None))
            return
        if len(self._steps) <= 1:
            QMessageBox.information(self, "Exposure sequence", "An exposure must retain at least one step.")
            return
        del self._steps[step_index]
        self._refresh_tree(select=(min(step_index, len(self._steps) - 1), None))

    def _save(self, *_args) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Save exposure", "Enter a name for this exposure.")
            return
        groups = [group for step in self._steps for group in step.get("drive_groups", [])]
        if not groups:
            QMessageBox.warning(self, "Save exposure", "Add at least one drive group to the exposure.")
            return
        try:
            referenced_coils = {
                str(coil_id)
                for step in self._steps
                for group in step.get("drive_groups", [])
                for coil_id in group.get("coil_ids", [])
                if str(coil_id)
            }
            labels = {
                str(choice.get("id")): str(choice.get("label") or choice.get("id"))
                for choice in self._coil_catalog
                if str(choice.get("id", "")) in referenced_coils
            }
            for step_index, step in enumerate(self._steps):
                if float(step.get("duration_s", 0.0)) <= 0.0:
                    raise StudioOperationError(f"Step {step_index + 1} duration must be positive.")
                for group_index, group in enumerate(step.get("drive_groups", [])):
                    waveform_id = self._resolved_group(group)["waveform_id"]
                    if not waveform_id:
                        raise StudioOperationError(
                            f"Step {step_index + 1}, drive {group_index + 1} needs a waveform source."
                        )
                    self.adapter.waveform_resource(waveform_id)
                    self.adapter.waveform_document(waveform_id)
                    if not group.get("coil_ids"):
                        raise StudioOperationError(
                            f"Step {step_index + 1}, drive {group_index + 1} needs at least one coil."
                        )
            protocol = default_exposure_protocol()
            protocol["duration_s"] = self.duration.value() * self._duration_unit_scale
            protocol["sources"] = copy.deepcopy(self._sources)
            protocol["coil_labels"] = labels
            protocol["sequence"] = {
                "loop": bool(self.loop_sequence.isChecked()),
                "steps": copy.deepcopy(self._steps),
            }
            protocol = normalize_exposure_protocol(protocol)
            # Resolve each complete step while saving so shared-source parallel
            # networks (and ordinary independent routes) are validated exactly
            # as playback will see them. This also catches missing/nonphysical
            # model-side load data before Maps queues a render job.
            for step_index in range(len(protocol["sequence"]["steps"])):
                self.adapter.resolve_exposure_step_electrical(protocol, step_index)
        except Exception as error:  # noqa: BLE001 - user-input validation boundary
            QMessageBox.warning(self, "Save exposure", str(error))
            return
        result = copy.deepcopy(self._source)
        result["name"] = name
        result["description"] = self.notes_edit.toPlainText().strip()
        result["protocol"] = protocol
        self.result_exposure = result
        self.accept()


class ExposureLibraryDialog(QDialog):
    """Manage named exposure protocols stored inside the current scene."""

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        *,
        on_changed: Callable[[], None] | None = None,
        project_context: ProjectContext | None = None,
        open_waveform_editor: Callable[[str | None], None] | None = None,
        embedded: bool = False,
    ):
        super().__init__(parent)
        if not embedded:
            enable_standard_window_controls(self)
        else:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.adapter = adapter
        self.on_changed = on_changed
        self.project_context = project_context
        self.open_waveform_editor = open_waveform_editor
        self.embedded = bool(embedded)
        self.setWindowTitle("Exposures")
        self.resize(1050, 560)

        root = QVBoxLayout(self)
        heading = QLabel("Saved exposures")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        note = QLabel(
            "An exposure sequences waveform sources into physical coil loads. Each step can contain one or more drive groups with explicit series/parallel wiring."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        root.addWidget(note)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Steps", "Duration", "Updated"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemDoubleClicked.connect(lambda _item, _column: self.edit_selected())
        self.tree.headerItem().setToolTip(3, "Last updated (UTC)")
        self.notes_view = QTextBrowser()
        self.notes_view.setObjectName("exposureNotesView")
        self.notes_view.setOpenExternalLinks(True)
        self.notes_view.setMinimumWidth(180)
        notes_panel = QWidget()
        notes_layout = QVBoxLayout(notes_panel)
        notes_layout.setContentsMargins(0, 0, 0, 0)
        self.notes_heading = QLabel("Notes")
        self.notes_heading.setTextFormat(Qt.TextFormat.PlainText)
        self.notes_heading.setWordWrap(True)
        notes_layout.addWidget(self.notes_heading)
        notes_layout.addWidget(self.notes_view, 1)
        self.notes_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.notes_splitter.setObjectName("exposureNotesSplitter")
        self.notes_splitter.setChildrenCollapsible(False)
        self.notes_splitter.addWidget(self.tree)
        self.notes_splitter.addWidget(notes_panel)
        self.notes_splitter.setStretchFactor(0, 3)
        self.notes_splitter.setStretchFactor(1, 2)
        self.notes_splitter.setSizes([600, 400])
        root.addWidget(self.notes_splitter, 1)

        row = QHBoxLayout()
        self.new_button = QPushButton("New…")
        self.new_button.setObjectName("primaryButton")
        self.new_button.clicked.connect(self.new_exposure)
        self.edit_button = QPushButton("Edit…")
        self.edit_button.clicked.connect(self.edit_selected)
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_selected)
        self.analyze_button = QPushButton("Analyze exposure…")
        self.analyze_button.setToolTip(
            "Prepare authoritative Brain Areas exposure statistics independently of Scripted playback"
        )
        self.analyze_button.clicked.connect(self.analyze_selected)
        self.rename_button = QPushButton("Rename…")
        self.rename_button.clicked.connect(self.rename_selected)
        self.import_button = QPushButton("Import…")
        self.import_button.setToolTip("Import a portable .fwbexposure.json file or all exposures from another scene")
        self.import_button.clicked.connect(self.import_exposure)
        self.export_button = QPushButton("Export…")
        self.export_button.setToolTip("Export one selected exposure as a portable file")
        self.export_button.clicked.connect(self.export_selected)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("dangerButton")
        self.delete_button.clicked.connect(self.delete_selected)
        for button in (
            self.new_button,
            self.edit_button,
            self.duplicate_button,
            self.analyze_button,
            self.rename_button,
            self.import_button,
            self.export_button,
            self.delete_button,
        ):
            row.addWidget(button)
        row.addStretch(1)
        if not self.embedded:
            close = QPushButton("Close")
            close.clicked.connect(self.accept)
            row.addWidget(close)
        root.addLayout(row)

        self._last_signature: tuple | None = None
        self._analysis_windows: dict[str, QDialog] = {}
        self.refresh()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self.refresh_if_changed)
        self.refresh_timer.start()

    def _summaries(self) -> list[dict[str, Any]]:
        return self.adapter.list_exposures()

    @staticmethod
    def _signature(items: list[dict[str, Any]]) -> tuple:
        return tuple(
            (item.get("id"), item.get("name"), item.get("description"), item.get("waveform"), item.get("drive_group_count"), item.get("coil_count"), item.get("step_count"), item.get("duration_s"), item.get("updated_at"))
            for item in items
        )

    def refresh_if_changed(self) -> bool:
        items = self._summaries()
        signature = self._signature(items)
        if signature == self._last_signature:
            return False
        self.refresh(preserve_selection=True, items=items)
        return True

    def refresh(self, select_id: str | None = None, *, preserve_selection: bool = False, items: list[dict[str, Any]] | None = None) -> None:
        previous = set(self.selected_ids()) if preserve_selection and not select_id else set()
        items = self._summaries() if items is None else items
        self.tree.clear()
        selected_item = None
        for exposure in items:
            updated = str(exposure.get("updated_at", "")).replace("T", " ").replace("Z", "")
            if "." in updated:
                updated = updated.split(".", 1)[0]
            duration = float(exposure.get("duration_s", 0.0))
            item = QTreeWidgetItem([
                str(exposure.get("name", "Exposure")),
                str(int(exposure.get("step_count", 1))),
                f"{duration:.6g} s",
                updated or "—",
            ])
            exposure_id = str(exposure.get("id", ""))
            item.setData(0, Qt.ItemDataRole.UserRole, exposure_id)
            item.setData(0, Qt.ItemDataRole.UserRole + 1, str(exposure.get("description", "") or ""))
            self.tree.addTopLevelItem(item)
            if exposure_id == str(select_id or ""):
                selected_item = item
            if exposure_id in previous:
                item.setSelected(True)
        if selected_item is None and not previous and self.tree.topLevelItemCount():
            selected_item = self.tree.topLevelItem(0)
        if selected_item is not None:
            self.tree.setCurrentItem(selected_item)
            selected_item.setSelected(True)
        self._last_signature = self._signature(items)
        self._update_buttons()

    def selected_ids(self) -> list[str]:
        values = []
        for item in self.tree.selectedItems():
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if value:
                values.append(str(value))
        return list(dict.fromkeys(values))

    def selected_id(self) -> str | None:
        values = self.selected_ids()
        return values[0] if len(values) == 1 else None

    def _update_buttons(self) -> None:
        selected = self.selected_ids()
        one = len(selected) == 1
        self.edit_button.setEnabled(one)
        self.duplicate_button.setEnabled(one)
        self.analyze_button.setEnabled(one)
        self.rename_button.setEnabled(one)
        self.export_button.setEnabled(one)
        self.delete_button.setEnabled(bool(selected))
        if one:
            item = self.tree.selectedItems()[0]
            self.notes_heading.setText("Notes — " + item.text(0))
            notes = str(item.data(0, Qt.ItemDataRole.UserRole + 1) or "")
            self.notes_view.setHtml(_exposure_notes_html(notes) if notes.strip() else "No notes recorded.")
        else:
            self.notes_heading.setText("Notes")
            self.notes_view.setPlainText("Select one exposure to view its notes.")

    def _changed(self, select_id: str | None = None) -> None:
        if self.on_changed is not None:
            self.on_changed()
        self.refresh(select_id)

    def _run_editor(self, exposure: dict[str, Any] | None) -> str | None:
        dialog = ExposureEditorDialog(
            self.adapter,
            exposure,
            self,
            project_context=self.project_context,
            open_waveform_editor=self.open_waveform_editor,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_exposure is None:
            return None
        try:
            exposure_id = self.adapter.save_exposure(dialog.result_exposure)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Save exposure", str(error))
            return None
        self._changed(exposure_id)
        return exposure_id

    def new_exposure(self, *_args) -> None:
        self._run_editor(None)

    def edit_selected(self, *_args) -> None:
        exposure_id = self.selected_id()
        if exposure_id is None:
            return
        try:
            exposure = self.adapter.exposure(exposure_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Edit exposure", str(error))
            return
        self._run_editor(exposure)

    def duplicate_selected(self, *_args) -> None:
        exposure_id = self.selected_id()
        if exposure_id is None:
            return
        try:
            created = self.adapter.duplicate_exposure(exposure_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Duplicate exposure", str(error))
            return
        self._changed(created)

    def analyze_selected(self, *_args) -> None:
        exposure_id = self.selected_id()
        if exposure_id is None:
            return

        existing = self._analysis_windows.get(str(exposure_id))
        if existing is not None:
            try:
                if existing.isMinimized():
                    existing.showNormal()
                else:
                    existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self._analysis_windows.pop(str(exposure_id), None)

        try:
            exposure = self.adapter.exposure(exposure_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Analyze exposure", str(error))
            return

        project_label = (
            self.project_context.display_name()
            if self.project_context is not None
            else "Current project model"
        )
        from .exposure_workspace import ExposureAnalysisWorkspace

        owner = self.window()
        workspace = ExposureAnalysisWorkspace(
            self.adapter, exposure, project_label=project_label,
            on_cache_changed=lambda _analysis_id: self._changed(exposure_id),
        )
        key = str(exposure_id)
        self._analysis_windows[key] = workspace
        workspace.destroyed.connect(
            lambda _object=None, exposure_key=key: self._analysis_windows.pop(exposure_key, None)
        )
        if self.project_context is not None:
            self.project_context.register_window("exposure_analysis", workspace)
        centre_window_on_parent(workspace, owner)
        workspace.show()
        workspace.raise_()
        workspace.activateWindow()

    def close_analysis_windows(self) -> None:
        """Close every modeless Exposure Analysis window owned by this editor."""
        for window in tuple(self._analysis_windows.values()):
            try:
                window.close()
            except RuntimeError:
                pass
        self._analysis_windows.clear()

    def done(self, result: int) -> None:
        if not self.embedded:
            self.close_analysis_windows()
        super().done(result)

    def rename_selected(self, *_args) -> None:
        exposure_id = self.selected_id()
        item = self.tree.currentItem()
        if exposure_id is None or item is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename exposure", "Name", text=item.text(0))
        if not accepted:
            return
        try:
            self.adapter.rename_exposure(exposure_id, name)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Rename exposure", str(error))
            return
        self._changed(exposure_id)

    def delete_selected(self, *_args) -> None:
        exposure_ids = self.selected_ids()
        if not exposure_ids:
            return
        title = "Delete exposure" if len(exposure_ids) == 1 else "Delete exposures"
        prompt = (
            "Delete the selected exposure? This can be restored with Undo."
            if len(exposure_ids) == 1
            else f"Delete {len(exposure_ids)} selected exposures? The deletion can be restored with Undo."
        )
        answer = QMessageBox.question(
            self, title, prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.adapter.delete_exposures(exposure_ids)
        self._changed()

    def export_selected(self, *_args) -> None:
        exposure_id = self.selected_id()
        if exposure_id is None:
            return
        try:
            exposure = self.adapter.exposure(exposure_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Export exposure", str(error))
            return
        safe_name = "".join(character if character.isalnum() or character in "-_. " else "_" for character in str(exposure.get("name", "Exposure"))).strip() or "Exposure"
        suggested = safe_name + EXPOSURE_FILE_SUFFIX
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export exposure", suggested,
            "Field Workbench exposure (*.fwbexposure.json);;JSON files (*.json);;All files (*)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".json"):
            filename += EXPOSURE_FILE_SUFFIX
        try:
            self.adapter.export_exposure(exposure_id, filename)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Export exposure", str(error))
            return
        QMessageBox.information(self, "Export exposure", f"Exported {Path(filename).name}.")

    def import_exposure(self, *_args) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import exposure",
            "",
            "Exposure or scene (*.fwbexposure.json *.magpy.json *.json);;All files (*)",
        )
        if not filename:
            return
        try:
            if str(filename).lower().endswith(EXPOSURE_FILE_SUFFIX):
                imported = [self.adapter.import_exposure_file(filename)]
            else:
                imported = self.adapter.import_exposures_from_scene_file(filename)
                if not imported:
                    raise StudioOperationError("No saved exposures were found in that scene.")
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Import exposure", str(error))
            return
        self._changed(imported[-1] if imported else None)
        QMessageBox.information(
            self,
            "Import exposure",
            f"Imported {len(imported)} exposure{'s' if len(imported) != 1 else ''}.",
        )


class ExposureEditorWindow(QMainWindow):
    """Project-aware top-level Exposure Editor suite application.

    Exposure management is independent of Model Editor. The library/editor
    controls persist directly through the shared :class:`ProjectContext` adapter.
    """

    projectResourcesChanged = Signal()
    openWaveformRequested = Signal(object)

    def __init__(self, project_context: ProjectContext) -> None:
        super().__init__(None)
        install_persistent_window_settings(self)
        self.project_context = project_context
        self.setObjectName("exposureEditorWindow")
        self.setWindowTitle(f"Exposure Editor — {project_context.display_name()}")
        self.resize(1120, 650)
        self.setMinimumSize(860, 480)
        self.library = ExposureLibraryDialog(
            project_context.adapter,
            self,
            on_changed=self._project_resources_changed,
            project_context=project_context,
            open_waveform_editor=self._request_waveform_editor,
            embedded=True,
        )
        self.setCentralWidget(self.library)
        self.statusBar().showMessage(
            "Build timed source-to-coil drive groups with explicit series/parallel wiring.",
            5000,
        )

    def _project_resources_changed(self) -> None:
        self.project_context.resources.refresh()
        self.projectResourcesChanged.emit()

    def _request_waveform_editor(self, waveform_id: str | None) -> None:
        self.openWaveformRequested.emit(waveform_id)

    def refresh_project_resources(self, exposure_id: str | None = None) -> None:
        self.library.refresh(select_id=exposure_id, preserve_selection=exposure_id is None)

    def waveform_resources_changed(self) -> None:
        """Refresh source names and diagnostics after Waveform Editor changes."""
        self.library.refresh(preserve_selection=True)
        for dialog in self.findChildren(ExposureEditorDialog):
            if dialog.isVisible():
                dialog.refresh_waveform_resources()

    def model_resources_changed(self) -> None:
        """Refresh coil labels/load diagnostics after model-side edits."""
        self.library.refresh(preserve_selection=True)
        for dialog in self.findChildren(ExposureEditorDialog):
            if dialog.isVisible():
                dialog.refresh_model_resources()

    def project_resources_changed(self) -> None:
        """Refresh source/load references after shared-project undo/redo."""
        self.library.refresh(preserve_selection=True)
        for dialog in self.findChildren(ExposureEditorDialog):
            if dialog.isVisible():
                dialog.refresh_project_resources()

    def prepare_project_close(self) -> bool:
        """Refuse project close while an exposure draft is still open.

        ExposureEditorDialog deliberately keeps its draft outside the shared
        project adapter until the user presses Save.  Project Manager therefore
        cannot include that draft in its normal project Save/Discard prompt.
        Rather than silently throwing it away, keep the project open and ask the
        user to finish the explicit Save/Cancel decision in the editor dialog.
        """
        for dialog in self.findChildren(ExposureEditorDialog):
            if not dialog.isVisible():
                continue
            dialog.raise_()
            dialog.activateWindow()
            QMessageBox.information(
                dialog,
                "Exposure edit in progress",
                "Finish the open Exposure Editor dialog with Save or Cancel before closing the project.",
            )
            return False
        return True

    def open_exposure(self, exposure_id: str) -> None:
        self.library.refresh(select_id=str(exposure_id))
        self.library.edit_selected()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if not self.prepare_project_close():
            event.ignore()
            return
        self.library.close_analysis_windows()
        event.accept()
