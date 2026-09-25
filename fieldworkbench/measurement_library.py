"""Project measurement-dataset manager and editor host.

Measurement datasets are project resources backed by the shared StudioAdapter.
The manager owns raw dataset selection/deletion, derived calibration fits, and
calibration-to-coil links; the Create/Edit dataset dialog owns only the raw samples
and their coordinate/provenance metadata.  In suite mode
this module also exposes a project-aware top-level Dataset Manager window so
measurement work no longer depends on Model Editor being open.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dialogs import MeasurementDatasetEditorDialog
from .project_context import ProjectContext
from .studio_adapter import StudioAdapter
from .window_utils import (
    EditorWheelGuard,
    enable_standard_window_controls,
    install_persistent_window_settings,
)


class MeasurementCompositeCalibrationDialog(QDialog):
    """Create/edit one derived calibration recipe from raw measurement datasets."""

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        *,
        initial_calibration_id: str | None = None,
        initial_dataset_ids: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.current_calibration_id = str(initial_calibration_id or "").strip() or None
        self.initial_dataset_ids = set(str(item) for item in (initial_dataset_ids or []) if str(item).strip())
        self.fit_result: dict[str, Any] | None = None
        self._loading_checks = False

        self.setWindowTitle(
            "Edit calibration / composite fit" if self.current_calibration_id else "New calibration / composite fit"
        )
        self.resize(1160, 820)
        root = QVBoxLayout(self)

        heading = QLabel("Calibration / composite fit")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        note = QLabel(
            "Build one derived correction from selected portions of one or more raw measurement datasets. "
            "The raw datasets are not merged or modified; this relation stores only the source selections, "
            "fit coil group, weighting rule, and fitted result."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        root.addWidget(note)

        setup_box = QGroupBox("Relation")
        setup_form = QFormLayout(setup_box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Main racetrack pair calibration")
        setup_form.addRow("Name", self.name_edit)
        self.weighting_combo = QComboBox()
        self.weighting_combo.addItem("Each measurement point equal", "points")
        self.weighting_combo.addItem("Each source selection equal", "sources")
        self.weighting_combo.setToolTip(
            "Point weighting lets dense datasets contribute more observations. Source weighting normalizes "
            "each selected dataset portion to equal total influence before the joint least-squares fit."
        )
        self.weighting_combo.currentIndexChanged.connect(self._invalidate_fit)
        setup_form.addRow("Weighting", self.weighting_combo)
        root.addWidget(setup_box)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        coils_box = QGroupBox("Fit coil group")
        coils_layout = QVBoxLayout(coils_box)
        coils_note = QLabel(
            "Check the coil or coils whose modeled field produced the measurement being fitted. A multi-coil "
            "selection is fitted as one group and therefore produces one shared correction factor. This fit "
            "basis does not restrict which scene coils may later use the resulting factor."
        )
        coils_note.setObjectName("hint")
        coils_note.setWordWrap(True)
        coils_layout.addWidget(coils_note)
        self.coil_tree = QTreeWidget()
        self.coil_tree.setHeaderLabels(["Use / coil", "Current correction"])
        self.coil_tree.setRootIsDecorated(False)
        self.coil_tree.setAlternatingRowColors(True)
        self.coil_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.coil_tree.itemChanged.connect(self._invalidate_fit)
        coils_layout.addWidget(self.coil_tree, 1)
        splitter.addWidget(coils_box)

        sources_box = QGroupBox("Dataset portions")
        sources_layout = QVBoxLayout(sources_box)
        sources_note = QLabel(
            "Check whole datasets or expand them to choose individual measured samples. Parent checkboxes "
            "select/deselect every measured sample in that dataset."
        )
        sources_note.setObjectName("hint")
        sources_note.setWordWrap(True)
        sources_layout.addWidget(sources_note)
        self.source_tree = QTreeWidget()
        self.source_tree.setHeaderLabels(["Use / dataset or sample", "Quantity", "Measured", "Position (mm)"])
        self.source_tree.setAlternatingRowColors(True)
        self.source_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.source_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            self.source_tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.source_tree.itemChanged.connect(self._source_item_changed)
        sources_layout.addWidget(self.source_tree, 1)
        splitter.addWidget(sources_box)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([340, 760])

        fit_box = QGroupBox("Joint scale-only fit")
        fit_layout = QHBoxLayout(fit_box)
        self.fit_summary = QLabel("No joint fit calculated yet.")
        self.fit_summary.setWordWrap(True)
        self.fit_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fit_layout.addWidget(self.fit_summary, 1)
        self.fit_button = QPushButton("Fit / recalculate")
        self.fit_button.setObjectName("fitCompositeMeasurementCalibrationButton")
        self.fit_button.clicked.connect(self.fit_calibration)
        fit_layout.addWidget(self.fit_button)
        self.save_button = QPushButton("Save relation")
        self.save_button.setObjectName("saveCompositeMeasurementCalibrationButton")
        self.save_button.clicked.connect(self.save_relation)
        fit_layout.addWidget(self.save_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        fit_layout.addWidget(close_button)
        root.addWidget(fit_box)

        self._populate_coils()
        self._populate_sources()
        if self.current_calibration_id:
            self._load_calibration(self.current_calibration_id)
        else:
            self._apply_initial_dataset_selection()

    @staticmethod
    def _quantity_label(output: str) -> str:
        return {"B": "|B|", "Bx": "Bx", "By": "By", "Bz": "Bz"}.get(str(output), str(output))

    @staticmethod
    def _format_factor(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{number:.7g} ×" if math.isfinite(number) else "—"

    def _populate_coils(self) -> None:
        self._loading_checks = True
        try:
            self.coil_tree.clear()
            for obj in self.adapter.list_objects():
                object_id = str(obj.get("id", ""))
                if not object_id or obj.get("derived"):
                    continue
                try:
                    if not self.adapter.is_coil(object_id):
                        continue
                    state = self.adapter.coil_correction_state(object_id)
                except Exception:
                    continue
                item = QTreeWidgetItem([
                    str(obj.get("label") or object_id),
                    f"{float(state.get('resolved_factor', 1.0)):.7g} ×",
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, object_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                self.coil_tree.addTopLevelItem(item)
        finally:
            self._loading_checks = False

    def _populate_sources(self) -> None:
        self._loading_checks = True
        try:
            self.source_tree.clear()
            for dataset in self.adapter.list_measurement_datasets():
                dataset_id = str(dataset.get("id", ""))
                if not dataset_id:
                    continue
                output = self._quantity_label(str(dataset.get("output", "B")))
                measured_samples = []
                for row_index, sample in enumerate(dataset.get("samples", [])):
                    if not isinstance(sample, dict):
                        continue
                    measured = sample.get("measured_uT")
                    try:
                        measured_value = float(measured)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(measured_value):
                        continue
                    measured_samples.append((row_index, sample, measured_value))
                parent = QTreeWidgetItem([
                    f"{dataset.get('label') or dataset_id}  ({len(measured_samples)} measured samples)",
                    output,
                    "",
                    "",
                ])
                parent.setData(0, Qt.ItemDataRole.UserRole, ("dataset", dataset_id))
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
                parent.setCheckState(0, Qt.CheckState.Unchecked)
                for row_index, sample, measured_value in measured_samples:
                    sample_index = int(sample.get("sample_index", row_index))
                    position = sample.get("position_m", [0.0, 0.0, 0.0])
                    try:
                        position_mm = [1000.0 * float(value) for value in position]
                        position_text = ", ".join(f"{value:.3g}" for value in position_mm)
                    except Exception:
                        position_text = "—"
                    child = QTreeWidgetItem([
                        f"Sample {sample_index + 1}",
                        output,
                        f"{measured_value:.7g} µT",
                        position_text,
                    ])
                    child.setData(0, Qt.ItemDataRole.UserRole, ("sample", dataset_id, sample_index))
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Unchecked)
                    parent.addChild(child)
                self.source_tree.addTopLevelItem(parent)
        finally:
            self._loading_checks = False

    def _apply_initial_dataset_selection(self) -> None:
        if not self.initial_dataset_ids:
            return
        self._loading_checks = True
        try:
            for index in range(self.source_tree.topLevelItemCount()):
                item = self.source_tree.topLevelItem(index)
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, tuple) and len(data) >= 2 and str(data[1]) in self.initial_dataset_ids:
                    item.setCheckState(0, Qt.CheckState.Checked)
                    for child_index in range(item.childCount()):
                        item.child(child_index).setCheckState(0, Qt.CheckState.Checked)
        finally:
            self._loading_checks = False

    def _load_calibration(self, calibration_id: str) -> None:
        calibration = self.adapter.measurement_calibration(calibration_id)
        self.name_edit.setText(str(calibration.get("label") or calibration_id))
        weighting = str(calibration.get("weighting", "points"))
        index = self.weighting_combo.findData(weighting)
        if index >= 0:
            self.weighting_combo.setCurrentIndex(index)
        selected_coils = set(str(item) for item in calibration.get("coil_ids", []))
        sources = {
            str(source.get("dataset_id", "")): source
            for source in calibration.get("sources", [])
            if isinstance(source, dict)
        }
        self._loading_checks = True
        try:
            for index in range(self.coil_tree.topLevelItemCount()):
                item = self.coil_tree.topLevelItem(index)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if str(item.data(0, Qt.ItemDataRole.UserRole)) in selected_coils
                    else Qt.CheckState.Unchecked,
                )
            for index in range(self.source_tree.topLevelItemCount()):
                parent = self.source_tree.topLevelItem(index)
                data = parent.data(0, Qt.ItemDataRole.UserRole)
                dataset_id = str(data[1]) if isinstance(data, tuple) and len(data) >= 2 else ""
                source = sources.get(dataset_id)
                if source is None:
                    parent.setCheckState(0, Qt.CheckState.Unchecked)
                    continue
                selected_indices = source.get("sample_indices", [])
                selected_set = set(int(item) for item in selected_indices) if isinstance(selected_indices, list) and selected_indices else None
                for child_index in range(parent.childCount()):
                    child = parent.child(child_index)
                    child_data = child.data(0, Qt.ItemDataRole.UserRole)
                    sample_index = int(child_data[2]) if isinstance(child_data, tuple) and len(child_data) >= 3 else -1
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if selected_set is None or sample_index in selected_set
                        else Qt.CheckState.Unchecked,
                    )
        finally:
            self._loading_checks = False
        self.fit_result = calibration.get("fit") if isinstance(calibration.get("fit"), dict) else None
        self._show_fit_summary(self.fit_result, status=self.adapter.measurement_calibration_status(calibration_id))

    def _source_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading_checks or column != 0:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple) and data and data[0] == "dataset":
            state = item.checkState(0)
            if state in {Qt.CheckState.Checked, Qt.CheckState.Unchecked}:
                self._loading_checks = True
                try:
                    for index in range(item.childCount()):
                        item.child(index).setCheckState(0, state)
                finally:
                    self._loading_checks = False
        self._invalidate_fit()

    def _invalidate_fit(self, *_args) -> None:
        if self._loading_checks:
            return
        self.fit_result = None
        self.fit_summary.setText("Inputs changed; fit/recalculate before using this relation for a new correction link.")

    def _selected_coils(self) -> list[str]:
        result: list[str] = []
        for index in range(self.coil_tree.topLevelItemCount()):
            item = self.coil_tree.topLevelItem(index)
            if item.checkState(0) == Qt.CheckState.Checked:
                value = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
                if value:
                    result.append(value)
        return result

    def _selected_sources(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index in range(self.source_tree.topLevelItemCount()):
            parent = self.source_tree.topLevelItem(index)
            data = parent.data(0, Qt.ItemDataRole.UserRole)
            if not (isinstance(data, tuple) and len(data) >= 2):
                continue
            dataset_id = str(data[1])
            selected_indices: list[int] = []
            measured_count = parent.childCount()
            for child_index in range(measured_count):
                child = parent.child(child_index)
                if child.checkState(0) != Qt.CheckState.Checked:
                    continue
                child_data = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(child_data, tuple) and len(child_data) >= 3:
                    selected_indices.append(int(child_data[2]))
            if not selected_indices:
                continue
            # Empty sample_indices is the compact representation for every measured
            # sample currently in this dataset.
            result.append({
                "dataset_id": dataset_id,
                "sample_indices": [] if len(selected_indices) == measured_count else selected_indices,
            })
        return result

    def fit_calibration(self, *_args) -> None:
        coils = self._selected_coils()
        sources = self._selected_sources()
        if not coils:
            QMessageBox.warning(self, "Calibration / composite fit", "Select at least one fit coil.")
            return
        if not sources:
            QMessageBox.warning(self, "Calibration / composite fit", "Select at least one measured dataset portion.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.fit_button.setEnabled(False)
        try:
            self.fit_result = self.adapter.fit_measurement_composite_calibration(
                coil_ids=coils,
                sources=sources,
                weighting=str(self.weighting_combo.currentData() or "points"),
            )
            self._show_fit_summary(self.fit_result, status="Ready")
        except Exception as error:
            self.fit_result = None
            self.fit_summary.setText("Joint calibration fit failed.")
            QMessageBox.warning(self, "Calibration / composite fit", str(error))
        finally:
            QApplication.restoreOverrideCursor()
            self.fit_button.setEnabled(True)

    def _show_fit_summary(self, fit: dict[str, Any] | None, *, status: str = "") -> None:
        if not fit:
            self.fit_summary.setText("No joint fit calculated yet.")
            return
        try:
            factor = float(fit.get("factor"))
        except (TypeError, ValueError):
            self.fit_summary.setText("Saved fit metadata is incomplete; recalculate the relation.")
            return
        parts = [f"Factor {factor:.7g} ×"]
        if fit.get("source_count") is not None and fit.get("sample_count") is not None:
            parts.append(f"{int(fit['source_count'])} sources / {int(fit['sample_count'])} samples")
        if fit.get("rms_before_uT") is not None and fit.get("rms_after_uT") is not None:
            parts.append(f"RMS {float(fit['rms_before_uT']):.5g} → {float(fit['rms_after_uT']):.5g} µT")
        if fit.get("r_squared") is not None:
            parts.append(f"R² {float(fit['r_squared']):.5g}")
        if status:
            parts.append(status)
        self.fit_summary.setText("  •  ".join(parts))

    def _payload(self) -> dict[str, Any]:
        label = self.name_edit.text().strip() or "Calibration / composite fit"
        payload = {
            "id": self.current_calibration_id or "",
            "label": label,
            "coil_ids": self._selected_coils(),
            "sources": self._selected_sources(),
            "weighting": str(self.weighting_combo.currentData() or "points"),
            "fit": self.fit_result,
            "source_versions": {},
            "model_signature": "",
        }
        if self.fit_result:
            payload["source_versions"] = dict(self.fit_result.get("source_versions", {}))
            payload["model_signature"] = str(self.fit_result.get("model_signature", ""))
        return payload

    def save_relation(self, *_args) -> None:
        payload = self._payload()
        if not payload["coil_ids"]:
            QMessageBox.warning(self, "Save calibration / composite fit", "Select at least one fit coil.")
            return
        if not payload["sources"]:
            QMessageBox.warning(self, "Save calibration / composite fit", "Select at least one measured dataset portion.")
            return
        try:
            self.current_calibration_id = self.adapter.save_measurement_calibration(payload)
        except Exception as error:
            QMessageBox.warning(self, "Save calibration / composite fit", str(error))
            return
        status = self.adapter.measurement_calibration_status(self.current_calibration_id)
        self._show_fit_summary(self.fit_result, status=status)


class MeasurementDatasetManagerDialog(QDialog):
    """Manage raw measurement datasets, derived calibrations, and coil links."""

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        *,
        on_changed: Callable[[], None] | None = None,
        project_context: ProjectContext | None = None,
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
        self.embedded = bool(embedded)
        self._last_signature: tuple | None = None
        self._refreshing_links = False
        self._link_wheel_guard = EditorWheelGuard(parent=self)

        self.setWindowTitle("Dataset Manager" if self.embedded else "Measurements")
        self.resize(1260, 860)
        root = QVBoxLayout(self)

        heading = QLabel("Measurement data and calibration")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        note = QLabel(
            "Keep raw measurement datasets as evidence, combine selected portions into reusable Calibration / "
            "Composite fits, then link those fitted relations to the coils they compensate. Raw datasets are "
            "never merged or overwritten by a composite fit."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        root.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("measurementDatasetManagerSplitter")
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # --- 1. Raw datasets -------------------------------------------------
        dataset_panel = QWidget()
        dataset_layout = QVBoxLayout(dataset_panel)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_heading = QLabel("1 · Measurement datasets")
        dataset_heading.setStyleSheet("font-weight: 600;")
        dataset_layout.addWidget(dataset_heading)
        dataset_note = QLabel(
            "Raw observations and their coordinates/provenance only. Fitting is intentionally handled in "
            "Calibration / Composite fits below so each correction has one clear derivation."
        )
        dataset_note.setObjectName("hint")
        dataset_note.setWordWrap(True)
        dataset_layout.addWidget(dataset_note)

        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setObjectName("measurementDatasetManagerTree")
        self.dataset_tree.setHeaderLabels(
            ["Name", "Sensor", "Quantity", "Samples", "Updated (UTC)"]
        )
        self.dataset_tree.setRootIsDecorated(False)
        self.dataset_tree.setAlternatingRowColors(True)
        self.dataset_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.dataset_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.dataset_tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.dataset_tree.itemSelectionChanged.connect(self._update_buttons)
        self.dataset_tree.itemDoubleClicked.connect(lambda _item, _column: self.edit_selected())
        dataset_layout.addWidget(self.dataset_tree, 1)

        dataset_buttons = QHBoxLayout()
        self.new_button = QPushButton("Add dataset…")
        self.new_button.setObjectName("primaryButton")
        self.new_button.clicked.connect(self.new_dataset)
        dataset_buttons.addWidget(self.new_button)
        self.edit_button = QPushButton("Edit…")
        self.edit_button.clicked.connect(self.edit_selected)
        dataset_buttons.addWidget(self.edit_button)
        self.create_sensors_button = QPushButton("Create point sensors")
        self.create_sensors_button.setToolTip(
            "Create a new scene group with one point sensor at each saved sample position and orientation."
        )
        self.create_sensors_button.clicked.connect(self.create_selected_sensors)
        dataset_buttons.addWidget(self.create_sensors_button)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("dangerButton")
        self.delete_button.clicked.connect(self.delete_selected)
        dataset_buttons.addWidget(self.delete_button)
        dataset_buttons.addStretch(1)
        dataset_layout.addLayout(dataset_buttons)
        splitter.addWidget(dataset_panel)

        # --- 2. Derived calibration/composite fits --------------------------
        calibration_panel = QWidget()
        calibration_layout = QVBoxLayout(calibration_panel)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        calibration_heading = QLabel("2 · Calibration / Composite fits")
        calibration_heading.setStyleSheet("font-weight: 600;")
        calibration_layout.addWidget(calibration_heading)
        calibration_note = QLabel(
            "Each relation selects portions of one or more datasets, chooses the coil group used as the fit basis, "
            "and calculates one joint scale-only correction. Equal-source weighting prevents a dense scan from "
            "automatically dominating a smaller measurement set."
        )
        calibration_note.setObjectName("hint")
        calibration_note.setWordWrap(True)
        calibration_layout.addWidget(calibration_note)

        self.calibration_tree = QTreeWidget()
        self.calibration_tree.setObjectName("measurementCalibrationRelationTree")
        self.calibration_tree.setHeaderLabels(
            ["Name", "Sources", "Samples", "Fit factor", "Fit coils", "Weighting", "Status", "Updated (UTC)"]
        )
        self.calibration_tree.setRootIsDecorated(False)
        self.calibration_tree.setAlternatingRowColors(True)
        self.calibration_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.calibration_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 8):
            self.calibration_tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.calibration_tree.itemSelectionChanged.connect(self._update_buttons)
        self.calibration_tree.itemDoubleClicked.connect(lambda _item, _column: self.edit_selected_calibration())
        calibration_layout.addWidget(self.calibration_tree, 1)

        calibration_buttons = QHBoxLayout()
        self.new_calibration_button = QPushButton("Add calibration…")
        self.new_calibration_button.setObjectName("primaryButton")
        self.new_calibration_button.clicked.connect(self.new_calibration)
        calibration_buttons.addWidget(self.new_calibration_button)
        self.edit_calibration_button = QPushButton("Edit…")
        self.edit_calibration_button.clicked.connect(self.edit_selected_calibration)
        calibration_buttons.addWidget(self.edit_calibration_button)
        self.recalculate_calibration_button = QPushButton("Recalculate")
        self.recalculate_calibration_button.clicked.connect(self.recalculate_selected_calibration)
        calibration_buttons.addWidget(self.recalculate_calibration_button)
        self.delete_calibration_button = QPushButton("Delete")
        self.delete_calibration_button.setObjectName("dangerButton")
        self.delete_calibration_button.clicked.connect(self.delete_selected_calibrations)
        calibration_buttons.addWidget(self.delete_calibration_button)
        calibration_buttons.addStretch(1)
        calibration_layout.addLayout(calibration_buttons)
        splitter.addWidget(calibration_panel)

        # --- 3. Coil links ---------------------------------------------------
        link_panel = QWidget()
        link_layout = QVBoxLayout(link_panel)
        link_layout.setContentsMargins(0, 0, 0, 0)
        link_heading = QLabel("3 · Coil correction links")
        link_heading.setStyleSheet("font-weight: 600;")
        link_layout.addWidget(link_heading)
        link_note = QLabel(
            "Attach any ready Calibration / Composite fit to any coil in the scene. The fit-coil group records how "
            "the factor was derived; application is intentionally separate, so equivalent coils can share a factor "
            "even when only one channel was measured. Raw measurement datasets are never linked directly to coils."
        )
        link_note.setObjectName("hint")
        link_note.setWordWrap(True)
        link_layout.addWidget(link_note)

        self.link_tree = QTreeWidget()
        self.link_tree.setObjectName("measurementDatasetLinkTree")
        self.link_tree.setHeaderLabels(["Coil", "Correction mode", "Calibration / source", "Resolved factor"])
        self.link_tree.setRootIsDecorated(False)
        self.link_tree.setAlternatingRowColors(True)
        self.link_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.link_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.link_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.link_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.link_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        link_layout.addWidget(self.link_tree, 1)
        splitter.addWidget(link_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([290, 300, 220])

        if not self.embedded:
            close_row = QHBoxLayout()
            close_row.addStretch(1)
            close = QPushButton("Close")
            close.clicked.connect(self.accept)
            close_row.addWidget(close)
            root.addLayout(close_row)

        self.refresh()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self.refresh_if_changed)
        self.refresh_timer.start()

    @staticmethod
    def _format_updated(value: Any) -> str:
        text = str(value or "").replace("T", " ").replace("Z", "")
        if "." in text:
            text = text.split(".", 1)[0]
        if "+" in text:
            text = text.split("+", 1)[0]
        return text or "—"

    @staticmethod
    def _fit_factor(payload: dict[str, Any]) -> str:
        fit = payload.get("fit") if isinstance(payload.get("fit"), dict) else {}
        raw = fit.get("factor")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return "—"
        return f"{value:.7g} ×" if math.isfinite(value) else "—"

    def _signature(self) -> tuple:
        datasets = self.adapter.list_measurement_datasets()
        dataset_part = tuple(
            (
                item.get("id"), item.get("label"), item.get("sensor_label"), item.get("output"),
                len(item.get("samples", [])) if isinstance(item.get("samples"), list) else 0,
                item.get("updated_at"),
            )
            for item in datasets
        )
        calibrations = self.adapter.list_measurement_calibrations()
        calibration_part = tuple(
            (
                item.get("id"), item.get("label"), repr(item.get("sources")), repr(item.get("fit")),
                tuple(item.get("coil_ids", [])) if isinstance(item.get("coil_ids"), list) else (),
                item.get("weighting"), repr(item.get("source_versions")), item.get("model_signature"), item.get("updated_at"),
            )
            for item in calibrations
        )
        coils = []
        for obj in self.adapter.list_objects():
            object_id = str(obj.get("id", ""))
            if not object_id or obj.get("derived"):
                continue
            try:
                if not self.adapter.is_coil(object_id):
                    continue
                state = self.adapter.coil_correction_state(object_id)
            except Exception:
                continue
            coils.append((
                object_id, obj.get("label"), state.get("mode"), state.get("dataset_id"),
                state.get("calibration_id"), state.get("custom_factor"), state.get("resolved_factor"),
            ))
        return dataset_part, calibration_part, tuple(coils)

    def refresh_if_changed(self) -> bool:
        signature = self._signature()
        if signature == self._last_signature:
            return False
        self.refresh(preserve_selection=True)
        return True

    def refresh(
        self,
        select_id: str | None = None,
        *,
        select_calibration_id: str | None = None,
        preserve_selection: bool = False,
    ) -> None:
        previous_datasets = set(self.selected_ids()) if preserve_selection and select_id is None else set()
        previous_calibrations = set(self.selected_calibration_ids()) if preserve_selection and select_calibration_id is None else set()
        self.dataset_tree.clear()
        selected_item = None
        for dataset in self.adapter.list_measurement_datasets():
            dataset_id = str(dataset.get("id", ""))
            output = str(dataset.get("output", "B"))
            quantity = {"B": "|B|", "Bx": "Bx", "By": "By", "Bz": "Bz"}.get(output, output)
            item = QTreeWidgetItem([
                str(dataset.get("label") or dataset_id),
                str(dataset.get("sensor_label") or dataset.get("sensor_id") or "—"),
                quantity,
                str(len(dataset.get("samples", [])) if isinstance(dataset.get("samples"), list) else 0),
                self._format_updated(dataset.get("updated_at")),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, dataset_id)
            refs = self.adapter.measurement_dataset_calibrations(dataset_id)
            if refs:
                item.setToolTip(0, f"Used by {len(refs)} calibration/composite fit{'s' if len(refs) != 1 else ''}.")
            self.dataset_tree.addTopLevelItem(item)
            if dataset_id == str(select_id or ""):
                selected_item = item
            elif dataset_id in previous_datasets:
                item.setSelected(True)
        if selected_item is not None:
            self.dataset_tree.setCurrentItem(selected_item)
            selected_item.setSelected(True)

        self.calibration_tree.clear()
        selected_calibration_item = None
        for calibration in self.adapter.list_measurement_calibrations():
            calibration_id = str(calibration.get("id", ""))
            fit = calibration.get("fit") if isinstance(calibration.get("fit"), dict) else {}
            source_count = len(calibration.get("sources", [])) if isinstance(calibration.get("sources"), list) else 0
            sample_count = fit.get("sample_count") if fit else None
            raw_coil_ids = calibration.get("coil_ids", []) if isinstance(calibration.get("coil_ids"), list) else []
            coil_labels: list[str] = []
            for coil_id in raw_coil_ids:
                try:
                    coil = self.adapter.get_object(str(coil_id))
                    coil_labels.append(str(coil.get("label") or coil_id))
                except Exception:
                    coil_labels.append(str(coil_id))
            if len(coil_labels) <= 2:
                target_text = ", ".join(coil_labels) or "—"
            else:
                target_text = f"{coil_labels[0]}, {coil_labels[1]} +{len(coil_labels) - 2}"
            weighting = "Equal points" if str(calibration.get("weighting", "points")) == "points" else "Equal sources"
            status = self.adapter.measurement_calibration_status(calibration_id)
            item = QTreeWidgetItem([
                str(calibration.get("label") or calibration_id),
                str(source_count),
                str(sample_count) if sample_count is not None else "—",
                self._fit_factor(calibration),
                target_text,
                weighting,
                status,
                self._format_updated(calibration.get("updated_at")),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, calibration_id)
            source_labels = []
            for source in calibration.get("sources", []):
                try:
                    dataset = self.adapter.measurement_dataset(str(source.get("dataset_id", "")))
                    source_labels.append(str(dataset.get("label") or dataset.get("id")))
                except Exception:
                    source_labels.append(str(source.get("dataset_id", "missing dataset")))
            if source_labels:
                item.setToolTip(1, "Sources: " + ", ".join(source_labels))
            if coil_labels:
                item.setToolTip(4, "Fit coil group: " + ", ".join(coil_labels))
            self.calibration_tree.addTopLevelItem(item)
            if calibration_id == str(select_calibration_id or ""):
                selected_calibration_item = item
            elif calibration_id in previous_calibrations:
                item.setSelected(True)
        if selected_calibration_item is not None:
            self.calibration_tree.setCurrentItem(selected_calibration_item)
            selected_calibration_item.setSelected(True)

        self._refresh_link_table()
        self._last_signature = self._signature()
        self._update_buttons()

    def _refresh_link_table(self) -> None:
        self._refreshing_links = True
        try:
            self.link_tree.clear()
            fitted_calibrations: list[tuple[str, str, float, str]] = []
            for calibration in self.adapter.list_measurement_calibrations():
                calibration_id = str(calibration.get("id", ""))
                if not calibration_id:
                    continue
                try:
                    factor = self.adapter.measurement_calibration_fit_factor(calibration_id)
                except Exception:
                    continue
                fitted_calibrations.append((
                    calibration_id,
                    str(calibration.get("label") or calibration_id),
                    factor,
                    self.adapter.measurement_calibration_status(calibration_id),
                ))

            for obj in self.adapter.list_objects():
                object_id = str(obj.get("id", ""))
                if not object_id or obj.get("derived"):
                    continue
                try:
                    if not self.adapter.is_coil(object_id):
                        continue
                    state = self.adapter.coil_correction_state(object_id)
                except Exception:
                    continue
                row = QTreeWidgetItem([
                    str(obj.get("label") or object_id), "", "", f"{float(state['resolved_factor']):.7g} ×",
                ])
                row.setData(0, Qt.ItemDataRole.UserRole, object_id)
                self.link_tree.addTopLevelItem(row)

                mode_combo = QComboBox(self.link_tree)
                mode_combo.installEventFilter(self._link_wheel_guard)
                mode_combo.addItem("Custom", "custom")
                mode_combo.addItem("Calibration", "calibration")
                mode_combo.setCurrentIndex(max(0, mode_combo.findData(state.get("mode", "custom"))))
                self.link_tree.setItemWidget(row, 1, mode_combo)

                source_combo = QComboBox(self.link_tree)
                source_combo.installEventFilter(self._link_wheel_guard)

                def populate_source_combo(
                    *_args, combo=source_combo, mode=mode_combo, current_state=state, coil_id=object_id
                ) -> None:
                    old_block = combo.blockSignals(True)
                    try:
                        combo.clear()
                        current_mode = str(mode.currentData() or "custom")
                        if current_mode == "calibration":
                            combo.addItem("Select calibration…", "")
                            current_link = str(current_state.get("calibration_id", ""))
                            for calibration_id, label, factor, status in fitted_calibrations:
                                if status != "Ready" and calibration_id != current_link:
                                    continue
                                suffix = "" if status == "Ready" else f" · {status}"
                                combo.addItem(f"{label}  ({factor:.7g} ×{suffix})", calibration_id)
                            target = str(current_state.get("calibration_id", ""))
                            index = combo.findData(target)
                            if index >= 0:
                                combo.setCurrentIndex(index)
                            combo.setEnabled(True)
                        else:
                            combo.addItem("—", "")
                            combo.setEnabled(False)
                    finally:
                        combo.blockSignals(old_block)

                populate_source_combo()
                self.link_tree.setItemWidget(row, 2, source_combo)

                def mode_changed(
                    _index: int,
                    *,
                    coil_id=object_id,
                    mode=mode_combo,
                    source=source_combo,
                    state_ref=state,
                    populate=populate_source_combo,
                ) -> None:
                    if self._refreshing_links:
                        return
                    # Bind this row's population helper explicitly so each combo
                    # refreshes its own saved state rather than the final row's.
                    populate(
                        combo=source,
                        mode=mode,
                        current_state=state_ref,
                        coil_id=coil_id,
                    )
                    try:
                        selected_mode = str(mode.currentData() or "custom")
                        if selected_mode == "custom":
                            self.adapter.set_coil_correction_custom([coil_id])
                        elif selected_mode == "calibration":
                            target = str(source.currentData() or "")
                            if not target:
                                if source.count() <= 1:
                                    raise RuntimeError("Create and fit a Calibration / Composite fit before linking this coil.")
                                source.setCurrentIndex(1)
                                target = str(source.currentData() or "")
                            self.adapter.set_coil_correction_calibration([coil_id], target)
                        else:
                            return
                    except Exception as error:
                        QMessageBox.warning(self, "Coil correction", str(error))
                    self._changed()

                def source_changed(_index: int, *, coil_id=object_id, mode=mode_combo, source=source_combo) -> None:
                    if self._refreshing_links:
                        return
                    target = str(source.currentData() or "")
                    if not target:
                        return
                    try:
                        if mode.currentData() == "calibration":
                            self.adapter.set_coil_correction_calibration([coil_id], target)
                        else:
                            return
                    except Exception as error:
                        QMessageBox.warning(self, "Coil correction", str(error))
                    self._changed()

                mode_combo.currentIndexChanged.connect(mode_changed)
                source_combo.currentIndexChanged.connect(source_changed)
        finally:
            self._refreshing_links = False

    def selected_ids(self) -> list[str]:
        values: list[str] = []
        for item in self.dataset_tree.selectedItems():
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if value:
                values.append(str(value))
        return list(dict.fromkeys(values))

    def selected_id(self) -> str | None:
        values = self.selected_ids()
        return values[0] if len(values) == 1 else None

    def selected_calibration_ids(self) -> list[str]:
        values: list[str] = []
        for item in self.calibration_tree.selectedItems():
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if value:
                values.append(str(value))
        return list(dict.fromkeys(values))

    def selected_calibration_id(self) -> str | None:
        values = self.selected_calibration_ids()
        return values[0] if len(values) == 1 else None

    def _update_buttons(self) -> None:
        selected = self.selected_ids()
        self.edit_button.setEnabled(len(selected) == 1)
        self.create_sensors_button.setEnabled(len(selected) == 1)
        self.delete_button.setEnabled(bool(selected))
        selected_calibrations = self.selected_calibration_ids()
        self.edit_calibration_button.setEnabled(len(selected_calibrations) == 1)
        self.recalculate_calibration_button.setEnabled(len(selected_calibrations) == 1)
        self.delete_calibration_button.setEnabled(bool(selected_calibrations))

    def _changed(
        self,
        select_id: str | None = None,
        *,
        select_calibration_id: str | None = None,
    ) -> None:
        if self.on_changed is not None:
            self.on_changed()
        self.refresh(select_id, select_calibration_id=select_calibration_id)

    def _run_editor(self, dataset_id: str | None) -> None:
        before = self._signature()
        dialog = MeasurementDatasetEditorDialog(self.adapter, self, initial_dataset_id=dataset_id)
        dialog.exec()
        after = self._signature()
        if after != before:
            selected = dialog.current_dataset_id or dataset_id
            self._changed(selected)
        else:
            self.refresh(preserve_selection=True)

    def new_dataset(self, *_args) -> None:
        self._run_editor(None)

    def edit_selected(self, *_args) -> None:
        dataset_id = self.selected_id()
        if dataset_id is not None:
            self._run_editor(dataset_id)

    def create_selected_sensors(self, *_args) -> None:
        dataset_id = self.selected_id()
        if dataset_id is None:
            return
        try:
            sensor_ids = self.adapter.create_measurement_dataset_sensors(dataset_id)
        except Exception as error:
            QMessageBox.warning(self, "Create point sensors", str(error))
            return
        self._changed(dataset_id)
        QMessageBox.information(
            self, "Create point sensors",
            f"Created {len(sensor_ids)} point sensors in a new scene group. "
            "Open Model Editor to view them. This addition can be undone in one step.",
        )

    def delete_selected(self, *_args) -> None:
        dataset_ids = self.selected_ids()
        if not dataset_ids:
            return
        relation_refs = {
            dataset_id: self.adapter.measurement_dataset_calibrations(dataset_id)
            for dataset_id in dataset_ids
        }
        blocked = {key: values for key, values in relation_refs.items() if values}
        if blocked:
            count = sum(len(values) for values in blocked.values())
            QMessageBox.warning(
                self,
                "Delete measurement dataset",
                f"The selected raw data is used by {count} calibration/composite fit reference"
                f"{'s' if count != 1 else ''}. Edit or delete those relations first so their provenance is not broken.",
            )
            return
        title = "Delete measurement dataset" if len(dataset_ids) == 1 else "Delete measurement datasets"
        prompt = (
            "Delete the selected raw measurement dataset? The deletion can be restored with Undo."
            if len(dataset_ids) == 1
            else f"Delete {len(dataset_ids)} selected raw measurement datasets? The deletion can be restored with Undo."
        )
        answer = QMessageBox.question(
            self, title, prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            for dataset_id in dataset_ids:
                self.adapter.delete_measurement_dataset(dataset_id)
        except Exception as error:
            QMessageBox.warning(self, title, str(error))
            return
        self._changed()

    def _run_calibration_editor(self, calibration_id: str | None) -> None:
        before = self._signature()
        dialog = MeasurementCompositeCalibrationDialog(
            self.adapter,
            self,
            initial_calibration_id=calibration_id,
            initial_dataset_ids=self.selected_ids() if calibration_id is None else None,
        )
        dialog.exec()
        after = self._signature()
        selected = dialog.current_calibration_id or calibration_id
        if after != before:
            self._changed(select_calibration_id=selected)
        else:
            self.refresh(preserve_selection=True)

    def new_calibration(self, *_args) -> None:
        self._run_calibration_editor(None)

    def edit_selected_calibration(self, *_args) -> None:
        calibration_id = self.selected_calibration_id()
        if calibration_id:
            self._run_calibration_editor(calibration_id)

    def recalculate_selected_calibration(self, *_args) -> None:
        calibration_id = self.selected_calibration_id()
        if not calibration_id:
            return
        calibration = self.adapter.measurement_calibration(calibration_id)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.recalculate_calibration_button.setEnabled(False)
        try:
            fit = self.adapter.fit_measurement_composite_calibration(
                coil_ids=list(calibration.get("coil_ids", [])),
                sources=list(calibration.get("sources", [])),
                weighting=str(calibration.get("weighting", "points")),
            )
            calibration["fit"] = fit
            calibration["source_versions"] = dict(fit.get("source_versions", {}))
            calibration["model_signature"] = str(fit.get("model_signature", ""))
            self.adapter.save_measurement_calibration(calibration)
        except Exception as error:
            QMessageBox.warning(self, "Recalculate calibration / composite fit", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.recalculate_calibration_button.setEnabled(True)
        self._changed(select_calibration_id=calibration_id)

    def delete_selected_calibrations(self, *_args) -> None:
        calibration_ids = self.selected_calibration_ids()
        if not calibration_ids:
            return
        linked_count = sum(len(self.adapter.measurement_calibration_linked_coils(item)) for item in calibration_ids)
        prompt = (
            f"Delete {len(calibration_ids)} calibration/composite fit{'s' if len(calibration_ids) != 1 else ''}? "
        )
        if linked_count:
            prompt += (
                f"They are linked to {linked_count} coil{'s' if linked_count != 1 else ''}; those coils will keep their "
                "current resolved factors as Custom values. "
            )
        prompt += "The deletion can be restored with Undo."
        answer = QMessageBox.question(
            self,
            "Delete calibration / composite fit",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            for calibration_id in calibration_ids:
                self.adapter.delete_measurement_calibration(calibration_id, unlink=True)
        except Exception as error:
            QMessageBox.warning(self, "Delete calibration / composite fit", str(error))
            return
        self._changed()


class DatasetManagerWindow(QMainWindow):
    """Project-aware top-level Dataset Manager suite application.

    The manager/editor controls operate directly on the adapter owned by
    :class:`ProjectContext`, so Dataset Manager can stay open independently and
    alongside applications from other projects.
    """

    projectResourcesChanged = Signal()

    def __init__(self, project_context: ProjectContext) -> None:
        super().__init__(None)
        self.setObjectName("datasetManagerWindow")
        install_persistent_window_settings(self)
        self.project_context = project_context
        self.setWindowTitle(f"Dataset Manager — {project_context.display_name()}")
        self.resize(1280, 860)
        self.setMinimumSize(960, 640)
        self.manager = MeasurementDatasetManagerDialog(
            project_context.adapter,
            self,
            on_changed=self._project_resources_changed,
            project_context=project_context,
            embedded=True,
        )
        self.setCentralWidget(self.manager)
        self.statusBar().showMessage(
            "Raw datasets → Calibration / Composite fits → coil correction links.",
            5000,
        )

    def _project_resources_changed(self) -> None:
        self.project_context.resources.refresh()
        self.projectResourcesChanged.emit()

    def refresh_project_resources(
        self,
        dataset_id: str | None = None,
        *,
        calibration_id: str | None = None,
    ) -> None:
        self.manager.refresh(
            select_id=dataset_id,
            select_calibration_id=calibration_id,
            preserve_selection=dataset_id is None and calibration_id is None,
        )

    def open_dataset(self, dataset_id: str) -> None:
        """Select and open one raw project dataset in the existing editor dialog."""
        self.manager.refresh(select_id=str(dataset_id))
        self.manager.edit_selected()

    def open_calibration(self, calibration_id: str) -> None:
        """Select and open one derived calibration/composite fit."""
        self.manager.refresh(select_calibration_id=str(calibration_id))
        self.manager.edit_selected_calibration()

