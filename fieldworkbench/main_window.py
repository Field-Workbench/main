"""Main PySide6 window for the Field Workbench workflow prototype."""

from __future__ import annotations

import ast
import base64
import html
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QSignalBlocker, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabBar,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation

from . import __version__
from .dialogs import (
    AiHelpDialog,
    BackgroundFieldDialog,
    CollapsibleSection,
    DrcResultsDialog,
    FieldMapDialog,
    FieldVolumeMapDialog,
    MeasurementAnalysisProgressDialog,
    MeasurementCalibrationDialog,
    MeasurementResultsDialog,
    OptimizerDialog,
    OptimizerWorkerBenchmarkDialog,
    ProbeLibraryDialog,
    SensorPlotDialog,
    SceneExportDialog,
    SceneNotesDialog,
    SceneObjectImportDialog,
    SnapshotComparisonDialog,
    SnapshotManagerDialog,
    UpennGbmImportDialog,
    configure_field_render_worker_limit,
)
from .brain_view import BrainViewDialog
from .help_topics import HelpTopicsDialog
from .optimizer import (
    DEFAULT_OPTIMIZER_WORKER_COUNT,
    available_optimizer_workers,
)
from .plot_view import PlotView
from .portable_scene import (
    translator_filter,
    translator_suffix,
    write_export,
)
from .window_utils import (
    CompactDoubleSpinBox,
    enable_standard_window_controls,
    install_persistent_window_settings,
    show_application_window,
)
from .theme import (
    THEME_CHOICES,
    apply_workbench_theme,
    current_theme,
    load_theme_preference,
    save_theme_preference,
    themed_icon,
)
from .studio_adapter import (
    APP_NAME,
    COIL_MODELING_LABELS,
    DEFAULT_SOLVER_MEMORY_BUDGET_MB,
    MAX_SOLVER_MEMORY_BUDGET_MB,
    MIN_SOLVER_MEMORY_BUDGET_MB,
    DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_ENABLED,
    DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
    MIN_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
    MAX_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
    DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_ENABLED,
    DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_MM,
    MIN_FLUXLINE_CONDUCTOR_CUTOFF_MM,
    MAX_FLUXLINE_CONDUCTOR_CUTOFF_MM,
    GEOMETRY_ROLE_LABELS,
    SCENE_SUFFIX,
    FieldCalculationCancelled,
    StudioAdapter,
    StudioOperationError,
    awg_resistance_20_ohm_per_km,
    scale_nested,
)


USER_ROLE_ID = Qt.ItemDataRole.UserRole


def _setting_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _flatten_finite_plotly_values(value: Any) -> list[float]:
    """Return finite numeric leaves from a Plotly coordinate container."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return []
        return [float(item) for item in array if np.isfinite(item)]
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            values.extend(_flatten_finite_plotly_values(item))
        return values
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return []
    return [numeric] if math.isfinite(numeric) else []


def _figure_3d_bounds(figure: dict[str, Any] | None) -> tuple[float, ...] | None:
    """Return visible 3-D data bounds as xmin/xmax/ymin/ymax/zmin/zmax."""
    if not isinstance(figure, dict):
        return None
    bounds = {axis: [math.inf, -math.inf] for axis in ("x", "y", "z")}
    supported_types = {
        "scatter3d",
        "mesh3d",
        "surface",
        "cone",
        "streamtube",
        "volume",
        "isosurface",
    }
    found = False
    for trace in figure.get("data", []):
        if not isinstance(trace, dict):
            continue
        if trace.get("visible") in {False, "legendonly"}:
            continue
        if str(trace.get("type", "")).strip().lower() not in supported_types:
            continue
        trace_found = False
        for axis in ("x", "y", "z"):
            values = _flatten_finite_plotly_values(trace.get(axis))
            if not values:
                continue
            bounds[axis][0] = min(bounds[axis][0], min(values))
            bounds[axis][1] = max(bounds[axis][1], max(values))
            trace_found = True
        found = found or trace_found
    if not found or any(not all(math.isfinite(v) for v in bounds[axis]) for axis in bounds):
        return None
    return tuple(
        value
        for axis in ("x", "y", "z")
        for value in (bounds[axis][0], bounds[axis][1])
    )


def _figure_3d_bounds_changed(
    before: tuple[float, ...] | None,
    after: tuple[float, ...] | None,
) -> bool:
    """Return whether a rebuilt scene needs its fixed Plotly ranges refitted."""
    if before is None or after is None:
        return before != after
    if len(before) != len(after):
        return True
    return not np.allclose(
        np.asarray(before, dtype=float),
        np.asarray(after, dtype=float),
        rtol=1e-12,
        atol=1e-9,
    )


@dataclass
class SceneDocument:
    """One independently editable scene shown by the shared Workbench UI."""

    adapter: StudioAdapter
    untitled_number: int
    document_id: str = field(default_factory=lambda: uuid4().hex)
    path: Path | None = None
    saved_signature: str = ""
    selected_id: str | None = None
    selected_ids: list[str] = field(default_factory=list)
    analysis_overlay: dict[str, Any] | None = None
    analysis_overlay_owner: object | None = None
    surface_region_highlight: tuple[str, str] | None = None
    base_figure: dict[str, Any] | None = None
    view_state: dict[str, Any] | None = None
    drc_dialog: DrcResultsDialog | None = None
    snapshot_manager_dialog: SnapshotManagerDialog | None = None
    # True only for a packaged default scene that was created as a fresh
    # document.  This is an origin marker, not a content comparison: a user's
    # unrelated scene is never considered disposable merely because it happens
    # to match the packaged demo.
    disposable_default: bool = False
    display_label: str | None = None

    def display_name(self) -> str:
        if self.path is not None:
            return self.path.name
        if self.display_label:
            return self.display_label
        return (
            "Untitled"
            if self.untitled_number == 1
            else f"Untitled {self.untitled_number}"
        )

    def is_dirty(self) -> bool:
        return self.adapter.signature() != self.saved_signature


class SceneTreeWidget(QTreeWidget):
    """Scene hierarchy with controlled drag-and-drop reparent requests."""

    reparent_requested = Signal(str, object)
    drag_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    @staticmethod
    def _drop_destination(source, target, indicator_position):
        """Return ``(valid, parent_id)`` without changing the widget model.

        The Studio adapter, rather than QTreeWidget's internal model, owns the
        hierarchy.  A blank-area drop means the scene root; dropping directly
        on an item is only meaningful when that item is a Group.
        """
        if target is None:
            return True, None

        ancestor = target
        while ancestor is not None:
            if ancestor is source:
                return False, None
            ancestor = ancestor.parent()

        if indicator_position == QAbstractItemView.DropIndicatorPosition.OnItem:
            if target.text(1) != "Group":
                return False, None
            return True, target.data(0, USER_ROLE_ID)

        parent = target.parent()
        return True, parent.data(0, USER_ROLE_ID) if parent is not None else None

    def startDrag(self, supported_actions):  # noqa: N802 - Qt API name
        """Always give the window a chance to reconcile after a drag ends."""
        try:
            super().startDrag(supported_actions)
        finally:
            self.drag_finished.emit()

    def dropEvent(self, event):  # noqa: N802 - Qt API name
        source = self.currentItem()
        if source is None:
            event.ignore()
            return
        object_id = source.data(0, USER_ROLE_ID)
        if not object_id:
            event.ignore()
            return

        target = self.itemAt(event.position().toPoint())
        valid, parent_id = self._drop_destination(
            source,
            target,
            self.dropIndicatorPosition(),
        )
        if not valid:
            event.ignore()
            return
        current_parent = source.parent().data(0, USER_ROLE_ID) if source.parent() else None
        if parent_id != current_parent:
            self.reparent_requested.emit(str(object_id), parent_id)

        # Never accept QTreeWidget's internal MoveAction.  Accepting it makes
        # Qt remove the source row even though the authoritative scene model is
        # updated separately (or has correctly decided there is no move).
        event.ignore()


class InspectorWheelFilter(QObject):
    """Turn wheel gestures over editors into inspector scrolling.

    Qt spin boxes and combo boxes normally consume the mouse wheel even when
    the user is only trying to move through a long form.  In this inspector a
    wheel gesture is navigation; deliberate value changes still work through
    typing, arrow buttons, and opening a combo box.
    """

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


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            # Detach immediately so stale inspector widgets are no longer
            # discoverable through findChildren() while Qt waits to process
            # the deferred deletion event.
            widget.setParent(None)
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _number_spin(value=0.0, *, decimals=4, minimum=-1e9, maximum=1e9, suffix=""):
    spin = CompactDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minimum, maximum)
    spin.setValue(float(value))
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    spin.setSingleStep(1.0)
    spin.setMinimumWidth(120)
    spin.setMaximumWidth(205)
    spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return spin


class MeshImportOptionsDialog(QDialog):
    """Compact unit and origin choices shown after selecting an STL/OBJ file."""

    UNIT_SCALES = {
        "Millimetres": 0.001,
        "Centimetres": 0.01,
        "Metres": 1.0,
    }

    def __init__(self, filename: str, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Mesh import options")
        self.setMinimumWidth(430)
        root = QVBoxLayout(self)

        source = QLabel(f"Importing {Path(filename).name}")
        source.setObjectName("dialogHeading")
        root.addWidget(source)

        sidecar = Path(filename).with_suffix(".regions.json")
        region_note = (
            f" A companion {sidecar.name} file was detected and will be imported as named "
            "surface regions."
            if sidecar.is_file()
            else ""
        )
        note = QLabel(
            "STL and OBJ geometry do not provide an authoritative source-unit scale. "
            "Choose the units represented by the file's numeric coordinates. Centering subtracts "
            "the mesh bounding-box centre so the imported geometry begins at its local object "
            "origin." + region_note
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        root.addWidget(note)

        form = QFormLayout()
        self.units = QComboBox()
        self.units.addItems(list(self.UNIT_SCALES))
        form.addRow("Source coordinates", self.units)
        root.addLayout(form)

        self.center_mesh = QCheckBox("Center geometry on its local origin")
        self.center_mesh.setObjectName("centerImportedMeshCheckBox")
        self.center_mesh.setChecked(False)
        self.center_mesh.setToolTip(
            "Use the centre of the imported mesh's axis-aligned bounding box as X=Y=Z=0"
        )
        root.addWidget(self.center_mesh)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def unit_scale(self) -> float:
        return self.UNIT_SCALES[self.units.currentText()]

    def should_center(self) -> bool:
        return self.center_mesh.isChecked()


class MainWindow(QMainWindow):
    """A narrow but complete save/edit/analyze workflow over Studio."""

    RECENT_FILES_KEY = "files/recentScenes"
    MAX_RECENT_FILES = 10

    def __init__(
        self,
        adapter: StudioAdapter | None = None,
        settings: QSettings | None = None,
    ):
        super().__init__()
        install_persistent_window_settings(self)
        self.settings = settings if settings is not None else QSettings()
        self._documents: list[SceneDocument] = []
        self._active_document_index = 0
        self._untitled_counter = 1
        initial_is_packaged_default = adapter is None
        initial_adapter = adapter or StudioAdapter()
        self._documents.append(
            SceneDocument(
                adapter=initial_adapter,
                untitled_number=self._untitled_counter,
                saved_signature=initial_adapter.signature(),
                disposable_default=initial_is_packaged_default,
            )
        )
        self._tab_change_guard = False
        self._tab_switch_in_progress = False
        self._pending_tab_index: int | None = None
        self._pending_tab_callbacks: list[Any] = []
        saved_model = self.settings.value("coil_modeling/method", None)
        if saved_model is not None:
            try:
                self.adapter.set_coil_modeling_method(str(saved_model))
            except StudioOperationError:
                self.adapter.set_coil_modeling_method("auto")
        saved_memory_budget = self.settings.value(
            "solver/memoryBudgetMb", DEFAULT_SOLVER_MEMORY_BUDGET_MB
        )
        try:
            self.adapter.set_solver_memory_budget_mb(int(saved_memory_budget))
        except (TypeError, ValueError, StudioOperationError):
            self.adapter.set_solver_memory_budget_mb(DEFAULT_SOLVER_MEMORY_BUDGET_MB)
        try:
            self.adapter.set_optimizer_worker_count(
                int(
                    self.settings.value(
                        "optimizer/workerCount", DEFAULT_OPTIMIZER_WORKER_COUNT
                    )
                )
            )
        except (TypeError, ValueError, StudioOperationError):
            self.adapter.set_optimizer_worker_count(DEFAULT_OPTIMIZER_WORKER_COUNT)
        configure_field_render_worker_limit(self.adapter.optimizer_worker_count)
        saved_diagnostics = self.settings.value("solver/diagnosticsEnabled", False)
        self.adapter.set_solver_diagnostics_enabled(_setting_bool(saved_diagnostics, False))

        minimum_field_enabled = _setting_bool(
            self.settings.value(
                "fluxlines/minimumFieldCutoffEnabled",
                DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_ENABLED,
            ),
            DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_ENABLED,
        )
        minimum_field_percent = self.settings.value(
            "fluxlines/minimumFieldCutoffPercent",
            DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
        )
        try:
            self.adapter.set_fluxline_minimum_field_cutoff(
                minimum_field_enabled, float(minimum_field_percent)
            )
        except (TypeError, ValueError, StudioOperationError):
            self.adapter.set_fluxline_minimum_field_cutoff(
                DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_ENABLED,
                DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
            )

        conductor_cutoff_enabled = _setting_bool(
            self.settings.value(
                "fluxlines/conductorDistanceCutoffEnabled",
                DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_ENABLED,
            ),
            DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_ENABLED,
        )
        conductor_cutoff_mm = self.settings.value(
            "fluxlines/conductorDistanceCutoffMm",
            DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_MM,
        )
        try:
            self.adapter.set_fluxline_conductor_cutoff(
                conductor_cutoff_enabled, float(conductor_cutoff_mm)
            )
        except (TypeError, ValueError, StudioOperationError):
            self.adapter.set_fluxline_conductor_cutoff(
                DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_ENABLED,
                DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_MM,
            )
        self.saved_signature = self.adapter.signature()
        self._tree_items: dict[str, QTreeWidgetItem] = {}
        self._pending_export_path: Path | None = None
        self._independent_windows: dict[int, QDialog] = {}
        self._independent_window_documents: dict[int, str] = {}
        self.notes_dialog: SceneNotesDialog | None = None
        self._scene_notes_edit_baseline: str | None = None
        self.parameter_editors: dict[str, dict[str, Any]] = {}
        # Inspector disclosure state is deliberately session-scoped.  The inspector
        # widgets are rebuilt whenever selection changes, but users should not lose
        # the layout they just arranged simply because they clicked another object.
        # Keys are semantic section names so comparable object types share the same
        # disclosure state (for example, collapsing Coil Construction applies to the
        # next coil inspected as well).
        self._inspector_section_states: dict[str, bool] = {}
        self._build_ui()
        self._build_actions()
        self._apply_theme()
        self.refresh_scene(select_id="coil")

    @property
    def active_document(self) -> SceneDocument:
        return self._documents[self._active_document_index]

    @property
    def documents(self) -> tuple[SceneDocument, ...]:
        """Read-only ordered view of the open scene documents."""
        return tuple(self._documents)

    @property
    def adapter(self) -> StudioAdapter:
        return self.active_document.adapter

    @property
    def current_path(self) -> Path | None:
        return self.active_document.path

    @current_path.setter
    def current_path(self, value: str | Path | None) -> None:
        self.active_document.path = None if value is None else Path(value)

    @property
    def saved_signature(self) -> str:
        return self.active_document.saved_signature

    @saved_signature.setter
    def saved_signature(self, value: str) -> None:
        self.active_document.saved_signature = str(value)

    @property
    def selected_id(self) -> str | None:
        return self.active_document.selected_id

    @selected_id.setter
    def selected_id(self, value: str | None) -> None:
        self.active_document.selected_id = value

    @property
    def selected_ids(self) -> list[str]:
        return self.active_document.selected_ids

    @selected_ids.setter
    def selected_ids(self, value: list[str]) -> None:
        self.active_document.selected_ids = list(value)

    @property
    def analysis_overlay(self) -> dict[str, Any] | None:
        return self.active_document.analysis_overlay

    @analysis_overlay.setter
    def analysis_overlay(self, value: dict[str, Any] | None) -> None:
        self.active_document.analysis_overlay = value

    @property
    def _analysis_overlay_owner(self) -> object | None:
        return self.active_document.analysis_overlay_owner

    @_analysis_overlay_owner.setter
    def _analysis_overlay_owner(self, value: object | None) -> None:
        self.active_document.analysis_overlay_owner = value

    @property
    def surface_region_highlight(self) -> tuple[str, str] | None:
        return self.active_document.surface_region_highlight

    @surface_region_highlight.setter
    def surface_region_highlight(self, value: tuple[str, str] | None) -> None:
        self.active_document.surface_region_highlight = value

    @property
    def _base_figure(self) -> dict[str, Any] | None:
        return self.active_document.base_figure

    @_base_figure.setter
    def _base_figure(self, value: dict[str, Any] | None) -> None:
        self.active_document.base_figure = value

    @property
    def drc_dialog(self) -> DrcResultsDialog | None:
        return self.active_document.drc_dialog

    @drc_dialog.setter
    def drc_dialog(self, value: DrcResultsDialog | None) -> None:
        self.active_document.drc_dialog = value

    @property
    def snapshot_manager_dialog(self) -> SnapshotManagerDialog | None:
        return self.active_document.snapshot_manager_dialog

    @snapshot_manager_dialog.setter
    def snapshot_manager_dialog(self, value: SnapshotManagerDialog | None) -> None:
        self.active_document.snapshot_manager_dialog = value

    # --- construction -------------------------------------------------
    def _build_ui(self) -> None:
        self.resize(1460, 860)
        self.setMinimumSize(1050, 680)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.main_splitter)
        splitter = self.main_splitter

        left = QWidget()
        left.setMinimumWidth(230)
        left.setMaximumWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 8, 12)

        title = QLabel("ADD OBJECT")
        title.setObjectName("sectionTitle")
        left_layout.addWidget(title)

        palette = QGroupBox()
        palette_layout = QVBoxLayout(palette)
        palette_categories = [
            ("Coil", [("Circular coil", "circle"), ("Racetrack coil", "racetrack"), ("Square coil", "polyline")]),
            (
                "Magnet",
                [
                    ("Block magnet", "cuboid"),
                    ("Cylinder magnet", "cylinder"),
                    ("Sphere magnet", "sphere"),
                ],
            ),
            (
                "Sensor",
                [
                    ("Point sensor", "sensor"),
                    ("Linear axis sensor", "axis_sensor"),
                    ("Magnetometer probe…", "magnetometer_probe"),
                ],
            ),
            ("Background", "background_field"),
            (
                "Object",
                [
                    ("Box", "guide_box"),
                    ("Cylinder", "guide_cylinder"),
                    ("Sphere", "guide_sphere"),
                    ("6-well plate", "well_plate_6"),
                    ("12-well plate", "well_plate_12"),
                    ("24-well plate", "well_plate_24"),
                    ("48-well plate", "well_plate_48"),
                    ("96-well plate", "well_plate_96"),
                    ("Generic bust", "generic_bust"),
                    ("SRI24 brain", "sri24_brain"),
                ],
            ),
        ]
        self.palette_buttons: dict[str, QPushButton] = {}
        for text, entries in palette_categories:
            button = QPushButton(text)
            button.setMinimumHeight(38)
            if isinstance(entries, str):
                button.clicked.connect(
                    lambda checked=False, kind=entries: self.add_object(kind)
                )
            else:
                menu = QMenu(button)
                for action_text, template in entries:
                    action = menu.addAction(action_text)
                    action.triggered.connect(
                        lambda checked=False, kind=template: self.add_object(kind)
                    )
                button.setMenu(menu)
            self.palette_buttons[text] = button
            palette_layout.addWidget(button)
        import_button = QPushButton("Import object…")
        import_button.setMinimumHeight(38)
        import_button.setToolTip("Import passive STL or OBJ geometry")
        import_button.clicked.connect(self.import_mesh)
        self.palette_buttons["Import object…"] = import_button
        palette_layout.addWidget(import_button)
        upenn_button = QPushButton("Import UPENN-GBM case…")
        upenn_button.setMinimumHeight(38)
        upenn_button.setToolTip(
            "Browse atlas-normalized UPENN-GBM tumor segmentations and add one to the scene"
        )
        upenn_button.clicked.connect(self.import_upenn_gbm_case)
        self.palette_buttons["Import UPENN-GBM case…"] = upenn_button
        palette_layout.addWidget(upenn_button)
        left_layout.addWidget(palette)

        scene_label = QLabel("SCENE")
        scene_label.setObjectName("sectionTitle")
        left_layout.addWidget(scene_label)
        self.tree = SceneTreeWidget()
        self.tree.setHeaderLabels(["Object", "Type"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._tree_selection_changed)
        self.tree.reparent_requested.connect(self.reparent_object)
        self.tree.drag_finished.connect(self._reconcile_scene_tree_after_drag)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_scene_context_menu)
        left_layout.addWidget(self.tree, 1)

        object_buttons = QHBoxLayout()
        self.new_group_button = QToolButton()
        self.new_group_button.setText("+")
        self.new_group_button.setToolTip("New empty group")
        self.new_group_button.setFixedWidth(30)
        self.new_group_button.setMinimumHeight(30)
        self.new_group_button.clicked.connect(self.add_empty_group)
        self.duplicate_button = QPushButton("Clone")
        self.duplicate_button.clicked.connect(self.duplicate_selected)
        self.visibility_button = QPushButton("Hide")
        self.visibility_button.clicked.connect(self.toggle_visibility)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("dangerButton")
        self.delete_button.clicked.connect(self.delete_selected)
        object_buttons.addWidget(self.new_group_button)
        object_buttons.addWidget(self.duplicate_button)
        object_buttons.addWidget(self.visibility_button)
        object_buttons.addWidget(self.delete_button)
        left_layout.addLayout(object_buttons)
        splitter.addWidget(left)

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(6, 12, 6, 12)
        self.scene_tools = QFrame()
        self.scene_tools.setObjectName("sceneTools")
        scene_tools_layout = QHBoxLayout(self.scene_tools)
        scene_tools_layout.setContentsMargins(6, 4, 6, 4)
        scene_tools_layout.setSpacing(5)

        scene_tools_label = QLabel("SCENE VIEW")
        scene_tools_label.setObjectName("sceneToolsLabel")
        scene_tools_layout.addWidget(scene_tools_label)
        scene_tools_layout.addSpacing(4)

        self.export_image_button = QPushButton("Export image…")
        self.export_image_button.setToolTip("Save the current 3D view as a high-resolution PNG")
        self.export_image_button.clicked.connect(self.export_scene_image)
        scene_tools_layout.addWidget(self.export_image_button)

        self.home_view_button = QPushButton("Home")
        self.home_view_button.setToolTip("Fit the whole scene and restore the isometric view")
        self.home_view_button.clicked.connect(self.home_scene_view)
        scene_tools_layout.addWidget(self.home_view_button)

        self.fullscreen_view_button = QPushButton("Fullscreen")
        self.fullscreen_view_button.setToolTip(
            "Open a separate true-fullscreen interactive viewer (Esc/F11 closes it)"
        )
        scene_tools_layout.addWidget(self.fullscreen_view_button)

        self.view_preset_buttons: dict[str, QPushButton] = {}
        for text, preset in (("Front", "front"), ("Right", "right"), ("Top", "top")):
            button = QPushButton(text)
            button.setToolTip(f"Fit the scene and show the {text.lower()} engineering view")
            button.clicked.connect(
                lambda checked=False, view_name=preset: self.plot.set_camera_preset(view_name)
            )
            self.view_preset_buttons[preset] = button
            scene_tools_layout.addWidget(button)
        scene_tools_layout.addStretch(1)
        centre_layout.addWidget(self.scene_tools)
        self.plot = PlotView(image_filename="scene-view")
        self.fullscreen_view_button.clicked.connect(self.plot.toggle_fullscreen)
        self.plot.selection_requested.connect(self._plot_selection_requested)
        self.plot.image_ready.connect(self._scene_image_ready)
        self.plot.image_failed.connect(self._scene_image_failed)
        centre_layout.addWidget(self.plot, 1)
        splitter.addWidget(centre)

        self.inspector_scroll = QScrollArea()
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector_scroll.setMinimumWidth(350)
        self.inspector_scroll.setMaximumWidth(560)
        self.inspector_container = QWidget()
        self.inspector_layout = QVBoxLayout(self.inspector_container)
        self.inspector_layout.setContentsMargins(14, 14, 14, 14)
        self.inspector_scroll.setWidget(self.inspector_container)
        self._inspector_wheel_filter = InspectorWheelFilter(self.inspector_scroll)
        splitter.addWidget(self.inspector_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([250, 790, 420])

        self.statusBar().showMessage("Ready")

    def _build_actions(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setObjectName("mainActionToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        self.main_toolbar = toolbar

        self.new_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon), "New scene", self
        )
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self.new_empty_scene)

        self.open_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open", self
        )
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_scene)

        self.save_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self
        )
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_scene)

        self.undo_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack), "Undo", self
        )
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        self.redo_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward), "Redo", self
        )
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo)
        self._standard_action_icon_specs = (
            ("new", self.new_action, QStyle.StandardPixmap.SP_FileIcon),
            ("open", self.open_action, QStyle.StandardPixmap.SP_DialogOpenButton),
            ("save", self.save_action, QStyle.StandardPixmap.SP_DialogSaveButton),
            ("undo", self.undo_action, QStyle.StandardPixmap.SP_ArrowBack),
            ("redo", self.redo_action, QStyle.StandardPixmap.SP_ArrowForward),
        )

        self.drc_action = QAction("Check design", self)
        self.drc_action.setToolTip(
            "Check physical coil assemblies against exclusion zones and each other"
        )
        self.drc_action.triggered.connect(self.open_drc)
        self.optimizer_action = QAction("Optimize design", self)
        self.optimizer_action.setToolTip(
            "Open the independent optimizer wizard; searches run on a frozen scene copy in the background"
        )
        self.optimizer_action.triggered.connect(self.open_optimizer)
        self.sensor_plot_action = QAction("Sensor plot", self)
        self.sensor_plot_action.triggered.connect(self.open_sensor_plot)
        self.measurement_calibration_action = QAction("Measurements", self)
        self.measurement_calibration_action.setToolTip(
            "Enter real field measurements and fit a linked coil-group correction factor"
        )
        self.measurement_calibration_action.triggered.connect(lambda _checked=False: self.open_measurement_calibration())
        self.field_map_action = QAction("2D field map", self)
        self.field_map_action.triggered.connect(self.open_field_map)
        self.field_volume_map_action = QAction("3D field map", self)
        self.field_volume_map_action.triggered.connect(self.open_field_volume_map)
        self.brain_view_action = QAction("Brain view", self)
        self.brain_view_action.setToolTip(
            "Analyze static and waveform fields across SRI24 LPBA40 brain regions"
        )
        self.brain_view_action.triggered.connect(self.open_brain_view)
        self.measurement_action = QAction("Analyze volume", self)
        self.measurement_action.triggered.connect(self.analyze_selected_measurement)
        self.snapshots_action = QAction("Snapshots", self)
        self.snapshots_action.triggered.connect(self.open_snapshots)
        self.notes_action = QAction("Notes", self)
        self.notes_action.setToolTip("Open notes stored with the current scene")
        self.notes_action.triggered.connect(self.open_notes)

        # Keep the main action row compact and deterministic.  File/history buttons
        # retain a fixed icon-only footprint, while every analysis shortcut shares
        # the remaining width equally.  QSizePolicy.Ignored deliberately allows the
        # text buttons to become narrower than their label instead of pushing later
        # actions into QToolBar's overflow menu on smaller windows.
        toolbar_widget = QWidget(toolbar)
        toolbar_widget.setObjectName("mainActionToolbarContents")
        toolbar_widget.setMinimumWidth(0)
        toolbar_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(2)
        self.main_toolbar_buttons: dict[str, QToolButton] = {}

        def _sync_toolbar_button(button: QToolButton, action: QAction) -> None:
            button.setEnabled(action.isEnabled())
            button.setToolTip(action.toolTip() or action.text())

        def _icon_button(key: str, action: QAction) -> QToolButton:
            button = QToolButton(toolbar_widget)
            button.setObjectName(f"mainToolbar_{key}")
            button.setIcon(action.icon())
            button.setIconSize(QSize(24, 24))
            button.setToolTip(action.toolTip() or action.text())
            button.setAccessibleName(action.text())
            button.setAutoRaise(True)
            button.setFixedSize(36, 36)
            button.clicked.connect(lambda _checked=False, a=action: a.trigger())
            action.changed.connect(lambda b=button, a=action: _sync_toolbar_button(b, a))
            _sync_toolbar_button(button, action)
            self.main_toolbar_buttons[key] = button
            toolbar_layout.addWidget(button, 0)
            return button

        def _toolbar_divider(key: str) -> QFrame:
            divider = QFrame(toolbar_widget)
            divider.setObjectName(f"mainToolbarDivider_{key}")
            divider.setFrameShape(QFrame.Shape.VLine)
            divider.setFrameShadow(QFrame.Shadow.Sunken)
            divider.setFixedSize(7, 24)
            toolbar_layout.addWidget(divider, 0, Qt.AlignmentFlag.AlignVCenter)
            return divider

        def _analysis_button(key: str, label: str, action: QAction) -> QToolButton:
            button = QToolButton(toolbar_widget)
            button.setObjectName(f"mainToolbar_{key}")
            button.setText(label)
            button.setToolTip(action.toolTip() or action.text())
            button.setAccessibleName(action.text())
            button.setAutoRaise(True)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            button.clicked.connect(lambda _checked=False, a=action: a.trigger())
            action.changed.connect(lambda b=button, a=action: _sync_toolbar_button(b, a))
            _sync_toolbar_button(button, action)
            self.main_toolbar_buttons[key] = button
            toolbar_layout.addWidget(button, 1)
            return button

        for key, action in (
            ("open", self.open_action),
            ("save", self.save_action),
            ("undo", self.undo_action),
            ("redo", self.redo_action),
        ):
            _icon_button(key, action)

        self.main_toolbar_dividers: dict[str, QFrame] = {}
        analysis_shortcuts = (
            ("check", "DRC", self.drc_action),
            ("optimize", "Optimize", self.optimizer_action),
            ("sensor_plot", "Sensor plot", self.sensor_plot_action),
            ("measurements", "Measurements", self.measurement_calibration_action),
            ("2d_view", "2D View", self.field_map_action),
            ("3d_view", "3D View", self.field_volume_map_action),
            ("brain_view", "Brain view", self.brain_view_action),
            ("analyze", "Analyze", self.measurement_action),
            ("snapshots", "Snapshots", self.snapshots_action),
            ("notes", "Notes", self.notes_action),
        )
        for key, label, action in analysis_shortcuts:
            _analysis_button(key, label, action)
            if key in {"optimize", "brain_view"}:
                self.main_toolbar_dividers[key] = _toolbar_divider(key)

        toolbar.addWidget(toolbar_widget)
        self.main_toolbar_widget = toolbar_widget

        self.document_toolbar = QToolBar("Open scenes")
        self.document_toolbar.setMovable(False)
        self.document_toolbar.setFloatable(False)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.document_toolbar)
        self.new_scene_tab_button = QToolButton(self.document_toolbar)
        self.new_scene_tab_button.setObjectName("newSceneTabButton")
        self.new_scene_tab_button.setText("+")
        self.new_scene_tab_button.setToolTip("New scene")
        self.new_scene_tab_button.setAccessibleName("New scene")
        self.new_scene_tab_button.setAutoRaise(True)
        self.new_scene_tab_button.setFixedWidth(30)
        self.new_scene_tab_button.clicked.connect(
            lambda _checked=False: self.new_action.trigger()
        )
        self.document_toolbar.addWidget(self.new_scene_tab_button)
        self.document_tabs = QTabBar()
        self.document_tabs.setDocumentMode(True)
        self.document_tabs.setDrawBase(True)
        self.document_tabs.setExpanding(False)
        self.document_tabs.setMovable(False)
        self.document_tabs.setTabsClosable(True)
        self.document_tabs.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.document_tabs.setUsesScrollButtons(True)
        self.document_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.document_tabs.currentChanged.connect(self._document_tab_changed)
        self.document_tabs.tabCloseRequested.connect(self.close_document_tab)
        self.document_toolbar.addWidget(self.document_tabs)
        self._rebuild_document_tabs()

        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.new_action)
        self.new_default_action = self.file_menu.addAction("New default scene")
        self.new_default_action.setToolTip(
            "Open the HM demonstration scene; a pristine startup default tab is reused"
        )
        self.new_default_action.triggered.connect(self.new_default_scene)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.open_action)
        self.recent_menu = self.file_menu.addMenu("Open Recent")
        self._refresh_recent_menu()
        self.import_objects_action = self.file_menu.addAction("Import objects from scene…")
        self.import_objects_action.setToolTip(
            "Choose objects from another saved .magpy.json scene and copy them into this scene"
        )
        self.import_objects_action.triggered.connect(self.import_objects_from_scene)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        save_as_action = self.file_menu.addAction("Save as…")
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_scene_as)
        self.export_scene_action = self.file_menu.addAction("Export…")
        self.export_scene_action.setToolTip(
            "Export selected physical scene objects through a portable scene translator"
        )
        self.export_scene_action.triggered.connect(self.export_scene)
        self.file_menu.addSeparator()
        self.close_scene_action = self.file_menu.addAction("Close scene")
        self.close_scene_action.setShortcut(QKeySequence.StandardKey.Close)
        self.close_scene_action.triggered.connect(self.close_active_document)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Exit", self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        self.group_selected_action = edit_menu.addAction("Group selected objects")
        self.group_selected_action.setShortcut(QKeySequence("Ctrl+G"))
        self.group_selected_action.triggered.connect(self.group_selected)
        edit_menu.addAction("Clone selected", self.duplicate_selected, QKeySequence("Ctrl+D"))
        edit_menu.addAction("Delete selected", self.delete_selected, QKeySequence.StandardKey.Delete)
        edit_menu.addSeparator()
        preferences_action = edit_menu.addAction("Preferences…")
        preferences_action.triggered.connect(self.open_preferences)

        self.analysis_menu = self.menuBar().addMenu("&Analyze")
        self.analysis_menu.addAction(self.drc_action)
        self.analysis_menu.addAction(self.optimizer_action)
        self.analysis_menu.addSeparator()
        self.analysis_menu.addAction(self.sensor_plot_action)
        self.analysis_menu.addAction(self.measurement_calibration_action)
        self.analysis_menu.addAction(self.field_map_action)
        self.analysis_menu.addAction(self.field_volume_map_action)
        self.analysis_menu.addAction(self.brain_view_action)
        self.analysis_menu.addSeparator()
        self.analysis_menu.addAction(self.measurement_action)
        self.analysis_menu.addAction(self.snapshots_action)
        self.analysis_menu.addAction(self.notes_action)

        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_topics_action = self.help_menu.addAction("Help Topics…")
        self.help_topics_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.help_topics_action.triggered.connect(self.open_help_topics)
        self.ai_help_action = self.help_menu.addAction("AI Help…")
        self.ai_help_action.setToolTip(
            "Copy the Scene Recipe guide for an external AI chat, then validate and import returned JSON"
        )
        self.ai_help_action.triggered.connect(self.open_ai_help)
        self.help_menu.addSeparator()
        self.about_action = self.help_menu.addAction("About Field Workbench…")
        self.about_action.triggered.connect(self.show_about)

    # --- scene documents ----------------------------------------------
    def _document_by_id(self, document_id: str) -> SceneDocument | None:
        return next(
            (document for document in self._documents if document.document_id == document_id),
            None,
        )

    def _document_index(self, document_id: str) -> int | None:
        return next(
            (
                index
                for index, document in enumerate(self._documents)
                if document.document_id == document_id
            ),
            None,
        )

    def _document_for_path(self, path: str | Path) -> SceneDocument | None:
        key = self._scene_path_key(path)
        return next(
            (
                document
                for document in self._documents
                if document.path is not None
                and self._scene_path_key(document.path) == key
            ),
            None,
        )

    def _document_tab_text(self, document: SceneDocument) -> str:
        marker = " *" if document.is_dirty() else ""
        return f"{document.display_name()}{marker}"

    def _update_document_tab(self, document: SceneDocument) -> None:
        if not hasattr(self, "document_tabs"):
            return
        index = self._document_index(document.document_id)
        if index is None or index >= self.document_tabs.count():
            return
        self.document_tabs.setTabText(index, self._document_tab_text(document))
        self.document_tabs.setTabToolTip(
            index,
            str(document.path) if document.path is not None else "Unsaved scene",
        )

    def _rebuild_document_tabs(self) -> None:
        if not hasattr(self, "document_tabs"):
            return
        self._tab_change_guard = True
        try:
            while self.document_tabs.count():
                self.document_tabs.removeTab(self.document_tabs.count() - 1)
            for document in self._documents:
                index = self.document_tabs.addTab(self._document_tab_text(document))
                self.document_tabs.setTabData(index, document.document_id)
                self.document_tabs.setTabToolTip(
                    index,
                    str(document.path) if document.path is not None else "Unsaved scene",
                )
            if self._documents:
                self.document_tabs.setCurrentIndex(self._active_document_index)
        finally:
            self._tab_change_guard = False

    def _document_tab_changed(self, index: int) -> None:
        if self._tab_change_guard or not 0 <= index < len(self._documents):
            return
        self._request_document_activation(index)

    def _request_document_activation(self, index: int, on_complete=None) -> None:
        if not 0 <= index < len(self._documents):
            return
        if index == self._active_document_index:
            if on_complete is not None:
                on_complete()
            return
        if self._pending_export_path is not None:
            self._tab_change_guard = True
            try:
                self.document_tabs.setCurrentIndex(self._active_document_index)
            finally:
                self._tab_change_guard = False
            self.statusBar().showMessage(
                "Wait for the current image export to finish before switching scenes.",
                3500,
            )
            return
        if self._tab_switch_in_progress:
            self._pending_tab_index = index
            if on_complete is not None:
                self._pending_tab_callbacks.append(on_complete)
            return

        self._tab_switch_in_progress = True
        target_id = self._documents[index].document_id
        source_id = self.active_document.document_id
        self.document_tabs.setEnabled(False)
        self._tab_change_guard = True
        try:
            self.document_tabs.setCurrentIndex(self._active_document_index)
        finally:
            self._tab_change_guard = False

        def finish_capture(state) -> None:
            self._complete_document_activation(
                source_id,
                target_id,
                state,
                on_complete,
            )

        self.plot.capture_view_state(finish_capture)

    def _complete_document_activation(
        self,
        source_id: str,
        target_id: str,
        view_state,
        on_complete=None,
    ) -> None:
        source = self._document_by_id(source_id)
        target_index = self._document_index(target_id)
        if source is not None and isinstance(view_state, dict):
            source.view_state = view_state
        if target_index is None:
            self._tab_switch_in_progress = False
            self.document_tabs.setEnabled(True)
            return

        self._commit_scene_notes_session()
        self.plot.exit_fullscreen()
        self._active_document_index = target_index
        self._tab_change_guard = True
        try:
            self.document_tabs.setCurrentIndex(target_index)
        finally:
            self._tab_change_guard = False
        self._sync_scene_notes_dialog()
        if self.active_document.view_state is None:
            self.plot.request_home_on_next_figure()
        self.refresh_scene()
        if self.active_document.view_state is not None:
            self.plot.apply_view_state(self.active_document.view_state)
        self.document_tabs.setEnabled(True)
        self._tab_switch_in_progress = False
        if on_complete is not None:
            on_complete()

        pending_index = self._pending_tab_index
        callbacks = self._pending_tab_callbacks
        self._pending_tab_index = None
        self._pending_tab_callbacks = []
        if pending_index is not None and pending_index != self._active_document_index:
            callback = callbacks[-1] if callbacks else None
            self._request_document_activation(pending_index, callback)

    def _activate_document_id(self, document_id: str, on_complete=None) -> None:
        index = self._document_index(document_id)
        if index is not None:
            self._request_document_activation(index, on_complete)

    def _apply_saved_preferences_to_adapter(self, adapter: StudioAdapter) -> None:
        method = str(self.settings.value("coil_modeling/method", "auto"))
        try:
            adapter.set_coil_modeling_method(method)
        except StudioOperationError:
            adapter.set_coil_modeling_method("auto")
        try:
            adapter.set_solver_memory_budget_mb(
                int(
                    self.settings.value(
                        "solver/memoryBudgetMb", DEFAULT_SOLVER_MEMORY_BUDGET_MB
                    )
                )
            )
        except (TypeError, ValueError, StudioOperationError):
            adapter.set_solver_memory_budget_mb(DEFAULT_SOLVER_MEMORY_BUDGET_MB)
        try:
            adapter.set_optimizer_worker_count(
                int(
                    self.settings.value(
                        "optimizer/workerCount", DEFAULT_OPTIMIZER_WORKER_COUNT
                    )
                )
            )
        except (TypeError, ValueError, StudioOperationError):
            adapter.set_optimizer_worker_count(DEFAULT_OPTIMIZER_WORKER_COUNT)
        adapter.set_solver_diagnostics_enabled(
            _setting_bool(self.settings.value("solver/diagnosticsEnabled", False), False)
        )
        try:
            adapter.set_fluxline_minimum_field_cutoff(
                _setting_bool(
                    self.settings.value(
                        "fluxlines/minimumFieldCutoffEnabled",
                        DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_ENABLED,
                    ),
                    DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_ENABLED,
                ),
                float(
                    self.settings.value(
                        "fluxlines/minimumFieldCutoffPercent",
                        DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
                    )
                ),
            )
        except (TypeError, ValueError, StudioOperationError):
            adapter.set_fluxline_minimum_field_cutoff(
                DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_ENABLED,
                DEFAULT_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
            )
        try:
            adapter.set_fluxline_conductor_cutoff(
                _setting_bool(
                    self.settings.value(
                        "fluxlines/conductorDistanceCutoffEnabled",
                        DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_ENABLED,
                    ),
                    DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_ENABLED,
                ),
                float(
                    self.settings.value(
                        "fluxlines/conductorDistanceCutoffMm",
                        DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_MM,
                    )
                ),
            )
        except (TypeError, ValueError, StudioOperationError):
            adapter.set_fluxline_conductor_cutoff(
                DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_ENABLED,
                DEFAULT_FLUXLINE_CONDUCTOR_CUTOFF_MM,
            )

    def _make_document(
        self,
        adapter: StudioAdapter,
        *,
        path: str | Path | None = None,
        selected_id: str | None = None,
        disposable_default: bool = False,
        untitled_number: int | None = None,
    ) -> SceneDocument:
        resolved_path = None if path is None else Path(path).expanduser().resolve()
        if untitled_number is None:
            if resolved_path is None:
                self._untitled_counter += 1
                untitled_number = self._untitled_counter
            else:
                # Named file tabs do not consume an Untitled number.
                untitled_number = self._untitled_counter
        return SceneDocument(
            adapter=adapter,
            untitled_number=untitled_number,
            path=resolved_path,
            saved_signature=adapter.signature(),
            selected_id=selected_id,
            selected_ids=[selected_id] if selected_id is not None else [],
            disposable_default=bool(disposable_default),
        )

    def _active_default_tab_is_disposable(self) -> bool:
        """Return whether the active packaged-default tab is still a throwaway placeholder.

        The marker records how the document was created; signatures/history then
        ensure that a scene the user edited (even if later undone back to the
        same geometry) is not silently discarded. View/selection changes do not
        make the placeholder precious.
        """
        if not self._documents:
            return False
        document = self.active_document
        if (
            not document.disposable_default
            or document.path is not None
            or document.is_dirty()
            or self._tab_switch_in_progress
            or self._pending_export_path is not None
        ):
            return False
        if document.drc_dialog is not None and document.drc_dialog.thread is not None:
            return False
        try:
            history = document.adapter.history()
        except Exception:
            return False
        return int(history.get("undo", 0)) == 0 and int(history.get("redo", 0)) == 0

    def _replace_active_disposable_default(self, document: SceneDocument) -> bool:
        """Reuse the pristine default tab for ``document`` when it is safe to do so."""
        if not self._active_default_tab_is_disposable():
            return False
        old_document = self.active_document
        if document.path is None:
            generated_number = document.untitled_number
            document.untitled_number = old_document.untitled_number
            if generated_number == self._untitled_counter and generated_number > old_document.untitled_number:
                self._untitled_counter -= 1
        self._commit_scene_notes_session()
        self._close_document_windows(old_document.document_id)
        self._documents[self._active_document_index] = document
        self._rebuild_document_tabs()
        self._sync_scene_notes_dialog()
        self.plot.exit_fullscreen()
        self.plot.request_home_on_next_figure()
        self.refresh_scene(select_id=document.selected_id)
        return True

    def _append_or_reuse_default_document(self, document: SceneDocument) -> None:
        if not self._replace_active_disposable_default(document):
            self._append_document(document)

    def _append_document(self, document: SceneDocument) -> None:
        self._documents.append(document)
        self._rebuild_document_tabs()
        self._request_document_activation(len(self._documents) - 1)

    def close_active_document(self) -> None:
        self.close_document_tab(self._active_document_index)

    def close_document_tab(self, index: int) -> bool:
        if not 0 <= index < len(self._documents):
            return False
        if self._tab_switch_in_progress:
            self.statusBar().showMessage(
                "Wait for the current scene switch to finish before closing a tab.",
                2500,
            )
            return False
        document = self._documents[index]
        if (
            document.document_id == self.active_document.document_id
            and self._pending_export_path is not None
        ):
            self.statusBar().showMessage(
                "Wait for the current image export to finish before closing this scene.",
                3500,
            )
            return False
        if document.drc_dialog is not None and document.drc_dialog.thread is not None:
            QMessageBox.information(
                self,
                "Design check running",
                (
                    f"Wait for the design check for {document.display_name()} "
                    "to finish before closing it."
                ),
            )
            return False
        if not self.maybe_save_document(document):
            return False

        if document.document_id == self.active_document.document_id:
            self._commit_scene_notes_session()
        self._close_document_windows(document.document_id)
        active_id = self.active_document.document_id
        self._documents.pop(index)
        if not self._documents:
            adapter = StudioAdapter()
            adapter.new_empty()
            self._apply_saved_preferences_to_adapter(adapter)
            self._untitled_counter += 1
            self._documents.append(
                SceneDocument(
                    adapter=adapter,
                    untitled_number=self._untitled_counter,
                    saved_signature=adapter.signature(),
                )
            )
            self._active_document_index = 0
        else:
            active_index = self._document_index(active_id)
            if active_index is None:
                self._active_document_index = min(index, len(self._documents) - 1)
            else:
                self._active_document_index = active_index
        self._rebuild_document_tabs()
        self._sync_scene_notes_dialog()
        self.plot.exit_fullscreen()
        if self.active_document.view_state is None:
            self.plot.request_home_on_next_figure()
        self.refresh_scene()
        if self.active_document.view_state is not None:
            self.plot.apply_view_state(self.active_document.view_state)
        return True

    def _close_document_windows(self, document_id: str) -> None:
        document = self._document_by_id(document_id)
        if document is not None:
            manager = document.snapshot_manager_dialog
            if manager is not None:
                for dialog in tuple(manager._independent_windows.values()):
                    dialog.close()
            for dialog in (document.drc_dialog, document.snapshot_manager_dialog):
                if dialog is not None:
                    dialog.close()
            document.drc_dialog = None
            document.snapshot_manager_dialog = None
        for key, owner_id in list(self._independent_window_documents.items()):
            if owner_id != document_id:
                continue
            dialog = self._independent_windows.get(key)
            if dialog is not None:
                dialog.close()

    def _apply_theme(self) -> None:
        # Apply at application scope so independent non-modal result windows
        # keep the same styling after they are detached from this main window.
        apply_workbench_theme(self, load_theme_preference(self.settings))

    def apply_workbench_theme(self, theme: str) -> None:
        """Refresh palette-dependent native icons after a global theme change."""

        buttons = getattr(self, "main_toolbar_buttons", {})
        for key, action, standard_pixmap in getattr(
            self, "_standard_action_icon_specs", ()
        ):
            action.setIcon(
                themed_icon(
                    self.style().standardIcon(standard_pixmap),
                    theme,
                )
            )
            button = buttons.get(key)
            if button is not None:
                button.setIcon(action.icon())

    def set_theme(self, theme: str) -> None:
        """Apply and persist one appearance choice across every open window."""

        apply_workbench_theme(self, theme)
        save_theme_preference(theme, self.settings)

    # --- scene refresh -------------------------------------------------
    def refresh_scene(
        self,
        select_id: str | None = None,
        *,
        select_ids: list[str] | None = None,
        refresh_plot: bool = True,
        rebuild_plot: bool = True,
    ) -> None:
        if select_ids is not None:
            desired_ids = list(dict.fromkeys(str(value) for value in select_ids))
        elif select_id is not None:
            desired_ids = [select_id]
        else:
            desired_ids = list(self.selected_ids)
        desired_primary = select_id or self.selected_id
        with QSignalBlocker(self.tree):
            self.tree.clear()
            object_items: dict[str, QTreeWidgetItem] = {}
            roots: list[QTreeWidgetItem] = []
            objects = self.adapter.list_objects()
            for obj in objects:
                type_label = self.adapter.object_type_label(obj["id"])
                if obj["type"] == "Sensor":
                    try:
                        path_length = self.adapter.get_transform(obj["id"])["path_length"]
                        type_label = "Linear axis sensor" if path_length > 1 else "Point sensor"
                    except Exception:
                        type_label = "Sensor"
                if self.adapter.is_coil(obj["id"]) and not self.adapter.coil_enabled(obj["id"]):
                    type_label = f"{type_label} • disabled"
                item = QTreeWidgetItem([obj["label"], type_label])
                item.setData(0, USER_ROLE_ID, obj["id"])
                if self.adapter.is_coil(obj["id"]) and not self.adapter.coil_enabled(obj["id"]):
                    item.setToolTip(1, "Magnetic current disabled; the coil remains visible and keeps its configured drive current.")
                if obj.get("derived"):
                    font = item.font(0)
                    font.setItalic(True)
                    item.setFont(0, font)
                    item.setToolTip(0, f"Generated from {obj['derived']}")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
                elif obj["type"] == "Collection":
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled)
                    item.setToolTip(0, "Drag objects onto this group to organize them as children.")
                else:
                    item.setFlags((item.flags() | Qt.ItemFlag.ItemIsDragEnabled) & ~Qt.ItemFlag.ItemIsDropEnabled)
                if obj.get("workbench_geometry"):
                    role_key = str(obj.get("role", "visual"))
                    role = GEOMETRY_ROLE_LABELS.get(role_key, role_key.capitalize())
                    shape = {
                        "box": "box",
                        "cylinder": "cylinder",
                        "sphere": "sphere",
                        "triangles": "mesh",
                    }.get(str(obj.get("geometry_kind", "")), "geometry")
                    item.setText(1, f"{role} • {shape}")
                    item.setToolTip(0, f"Passive Workbench {shape} • {role}")
                if not obj.get("visible", True):
                    item.setForeground(0, Qt.GlobalColor.gray)
                object_items[obj["id"]] = item
            for obj in objects:
                item = object_items[obj["id"]]
                parent_id = obj.get("parent")
                if parent_id and parent_id in object_items:
                    object_items[parent_id].addChild(item)
                else:
                    roots.append(item)
            self.tree.addTopLevelItems(roots)
            self.tree.expandAll()
            available_ids = [value for value in desired_ids if value in object_items]
            if available_ids:
                primary_id = (
                    desired_primary
                    if desired_primary in available_ids
                    else available_ids[-1]
                )
                self.tree.setCurrentItem(object_items[primary_id])
                for value in available_ids:
                    object_items[value].setSelected(True)
                self.selected_ids = available_ids
                self.selected_id = primary_id
            elif roots:
                self.tree.setCurrentItem(roots[0])
                self.selected_id = roots[0].data(0, USER_ROLE_ID)
                self.selected_ids = [self.selected_id]
            else:
                self.selected_id = None
                self.selected_ids = []
            self._tree_items = object_items

        self.populate_inspector(self.selected_id)
        if refresh_plot:
            self.refresh_plot(rebuild_base=rebuild_plot)
        self._update_actions()
        self._update_title()
        self._update_status()
        if self.drc_dialog is not None:
            self.drc_dialog.refresh_staleness()

    def refresh_plot(self, *, rebuild_base: bool = False) -> None:
        expensive_refresh = rebuild_base or self._base_figure is None
        if expensive_refresh:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if expensive_refresh:
                # The embedded Plotly view deliberately preserves the user's
                # camera and fixed axis ranges across ordinary ``react`` calls.
                # That is ideal for selection/current/name changes, but it also
                # means a geometry edit can grow or move outside those old
                # ranges unless the caller remembered to request a refit.
                #
                # Treat the rebuilt base figure as the authority instead of
                # making every Apply button know which parameters affect scene
                # extents.  If visible 3-D bounds actually changed, refit the
                # next figure while preserving the current camera direction.
                # This catches diameter/length/build edits, transforms, imported
                # geometry changes, visibility changes, optimizer results, etc.,
                # while electrical-only edits leave a manually chosen view alone.
                previous_bounds = _figure_3d_bounds(self._base_figure)
                rebuilt_base = self.adapter.base_figure()
                rebuilt_bounds = _figure_3d_bounds(rebuilt_base)
                if self._base_figure is not None and _figure_3d_bounds_changed(
                    previous_bounds, rebuilt_bounds
                ):
                    self.plot.request_fit_bounds_on_next_figure()
                self._base_figure = rebuilt_base
            self.plot.set_figure(
                self.adapter.compose_figure(
                    self._base_figure,
                    selected_ids=self.selected_ids,
                    analysis_overlay=self.analysis_overlay,
                    surface_region_highlight=self.surface_region_highlight,
                )
            )
        except Exception as error:  # noqa: BLE001 - GUI error boundary
            self.plot.show_message("Unable to render this scene", str(error))
        finally:
            if expensive_refresh:
                QApplication.restoreOverrideCursor()

    def _update_status(self) -> None:
        objects = self.adapter.list_objects()
        base_count = sum(not bool(item.get("derived")) for item in objects)
        try:
            field = self.adapter.field_at_mm([0, 0, 0], "B")
            centre = field["magnitude"][0] * 1e6
            report = field.get("coil_modeling", {})
            requested = str(report.get("requested", self.adapter.coil_modeling_method))
            model_label = COIL_MODELING_LABELS.get(requested, requested)
            fallback_count = sum(
                item.get("actual") == "centreline"
                and requested != "centreline"
                and bool(item.get("fallback_reason"))
                for item in report.get("coils", [])
            )
            fallback_text = (
                f"; {fallback_count} coil fallback" + ("s" if fallback_count != 1 else "")
                if fallback_count
                else ""
            )
            self.statusBar().showMessage(
                f"{base_count} objects   •   centre |B| = {centre:.6g} µT   •   "
                f"{model_label}{fallback_text}"
            )
        except Exception:
            self.statusBar().showMessage(f"{base_count} objects   •   no field source to measure")

    def _update_actions(self) -> None:
        history = self.adapter.history()
        self.undo_action.setEnabled(bool(history["undo"]))
        self.redo_action.setEnabled(bool(history["redo"]))
        has_selection = self.selected_id is not None and len(self.selected_ids) == 1
        derived = False
        if has_selection:
            try:
                derived = bool(self.adapter.get_object(self.selected_id).get("derived"))
            except KeyError:
                has_selection = False
        editable = has_selection and not derived
        editable_selected = self._editable_selected_ids()
        groupable = len(editable_selected) >= 2
        self.group_selected_action.setEnabled(groupable)
        brain_object_id = self._brain_view_sri24_brain_id()
        brain_view_ready = brain_object_id is not None
        # Keep Brain View available so the command can explain what must be
        # selected/added. When the scene contains exactly one SRI24 brain,
        # Brain View can resolve it automatically without changing scene
        # selection. Multiple brains still require an explicit selection.
        self.brain_view_action.setEnabled(True)
        sri24_count = len(self._sri24_brain_ids())
        if brain_view_ready:
            brain_view_hint = (
                "Analyze static and waveform fields across LPBA40 regions aligned to the SRI24 brain"
            )
        elif sri24_count > 1:
            brain_view_hint = "Please select the SRI24 brain to use for Brain View"
        else:
            brain_view_hint = "Brain View requires an SRI24 brain in the scene"
        self.brain_view_action.setToolTip(brain_view_hint)
        self.brain_view_action.setStatusTip(brain_view_hint)
        # Keep Analyze volume available even with an empty or mixed selection so
        # the command can explain what the user needs to select.  The command
        # itself filters the selection to Measurement-role volumes and supports
        # analyzing several of them in one pass.
        self.measurement_action.setEnabled(True)
        self.duplicate_button.setEnabled(editable)
        self.visibility_button.setEnabled(editable)
        self.delete_button.setEnabled(bool(editable_selected))
        if editable:
            obj = self.adapter.get_object(self.selected_id)
            self.visibility_button.setText("Hide" if obj.get("visible", True) else "Show")
        else:
            self.visibility_button.setText("Hide")

    # --- inspector -----------------------------------------------------
    def _tree_selection_changed(self) -> None:
        selected_items = self.tree.selectedItems()
        selected_ids = [
            str(item.data(0, USER_ROLE_ID))
            for item in selected_items
            if item.data(0, USER_ROLE_ID)
        ]
        current = self.tree.currentItem()
        current_id = str(current.data(0, USER_ROLE_ID)) if current else None
        self.selected_ids = selected_ids
        self.selected_id = (
            current_id if current_id in selected_ids else (selected_ids[-1] if selected_ids else None)
        )
        if (
            self.surface_region_highlight is not None
            and (len(selected_ids) != 1 or self.surface_region_highlight[0] != self.selected_id)
        ):
            self.surface_region_highlight = None
        self.populate_inspector(self.selected_id)
        self._update_actions()
        self.refresh_plot(rebuild_base=False)

    def _plot_selection_requested(self, object_id: str, additive: bool) -> None:
        """Mirror a Plotly object click into the scene tree selection."""
        if object_id not in self._tree_items:
            return
        if additive:
            selected = list(self.selected_ids)
            if object_id in selected:
                selected.remove(object_id)
            else:
                selected.append(object_id)
        else:
            selected = [object_id]

        primary = object_id if object_id in selected else (selected[-1] if selected else None)
        with QSignalBlocker(self.tree):
            self.tree.clearSelection()
            if primary is not None:
                self.tree.setCurrentItem(self._tree_items[primary])
            for value in selected:
                item = self._tree_items.get(value)
                if item is not None:
                    item.setSelected(True)
            if primary is not None:
                self.tree.scrollToItem(self._tree_items[primary])
        self.selected_ids = selected
        self.selected_id = primary
        if (
            self.surface_region_highlight is not None
            and (len(selected) != 1 or self.surface_region_highlight[0] != primary)
        ):
            self.surface_region_highlight = None
        self.populate_inspector(primary)
        self._update_actions()
        self.refresh_plot(rebuild_base=False)

    def _show_scene_context_menu(self, position) -> None:
        item = self.tree.itemAt(position)
        if item is not None and not item.isSelected():
            self.tree.clearSelection()
            self.tree.setCurrentItem(item)
            item.setSelected(True)

        menu = QMenu(self)
        editable_ids = self._editable_selected_ids()
        selected_coils = [
            object_id for object_id in editable_ids if self.adapter.is_coil(object_id)
        ]
        if selected_coils and len(selected_coils) == len(editable_ids):
            all_enabled = all(
                self.adapter.coil_enabled(object_id) for object_id in selected_coils
            )
            target_enabled = not all_enabled
            if len(selected_coils) == 1:
                label = "Enable coil" if target_enabled else "Disable coil"
            else:
                label = (
                    "Enable selected coils" if target_enabled else "Disable selected coils"
                )
            magnetic_action = menu.addAction(label)
            magnetic_action.triggered.connect(
                lambda checked=False, value=target_enabled: self.set_selected_coils_enabled(value)
            )
            menu.addSeparator()
        group_action = menu.addAction("Group selected objects")
        group_action.setEnabled(len(self._editable_selected_ids()) >= 2)
        group_action.triggered.connect(self.group_selected)
        if len(self.selected_ids) == 1:
            try:
                selected_object = self.adapter.get_object(self.selected_ids[0])
            except KeyError:
                selected_object = None
            if selected_object and selected_object["type"] == "Collection" and not selected_object.get("derived"):
                ungroup_action = menu.addAction("Ungroup")
                ungroup_action.triggered.connect(self.ungroup_selected)
            menu.addSeparator()
            menu.addAction("Clone", self.duplicate_selected)
            menu.addAction("Hide / show", self.toggle_visibility)
            if editable_ids:
                menu.addAction("Delete", self.delete_selected)
        elif editable_ids:
            menu.addSeparator()
            menu.addAction("Delete selected", self.delete_selected)
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _inspector_section(
        self,
        title: str,
        *,
        expanded: bool = True,
        state_key: str | None = None,
    ) -> CollapsibleSection:
        """Create a disclosure section that remembers its state for this session.

        Inspector widgets are transient because selection changes rebuild the form.
        Persisting the disclosure state here keeps the user's chosen working layout
        stable while moving between objects without turning it into a saved project
        property or a permanent application preference.
        """
        key = str(state_key or title)
        initial = self._inspector_section_states.get(key, bool(expanded))
        section = CollapsibleSection(title, expanded=initial)

        def remember(checked: bool, *, section_key: str = key) -> None:
            self._inspector_section_states[section_key] = bool(checked)

        section.header.toggled.connect(remember)
        return section

    def populate_inspector(self, object_id: str | None) -> None:
        _clear_layout(self.inspector_layout)
        self.parameter_editors.clear()
        self.geometry_role_combo = None
        self.geometry_colour_button: QPushButton | None = None
        self.geometry_drc_enabled_check: QCheckBox | None = None
        self.geometry_drc_setting_explicit = False
        self._geometry_drc_syncing = False
        self.geometry_clearance_override_check = None
        self.geometry_clearance_spin = None
        self.geometry_clearance_spin_label = None
        self.geometry_appearance_section = None
        self.geometry_appearance_form = None
        self.drc_section = None
        self._deferred_surface_regions = None
        self._deferred_mesh_health = None
        self.measurement_section = None
        self.measurement_box = None
        self.measurement_quality_combo = None
        self.measurement_defined_points_edit = None
        self.measurement_defined_points_label = None
        self.measurement_note = None
        self.measurement_target_spin = None
        self.measurement_tolerance_spin = None
        self.measurement_analyze_button = None
        self.coil_object_id = None
        self.coil_widgets: dict[str, Any] = {}
        self.coil_estimate_labels: dict[str, QLabel] = {}
        self.coil_estimate_row_names: dict[str, str] = {}
        self.coil_estimate_copy_button: QPushButton | None = None
        self.coil_envelope_group: QCheckBox | None = None
        self.mesh_dimension_labels: dict[str, QLabel] = {}
        self.mesh_source_dimensions_mm: list[float] | None = None
        self.surface_region_tree: QTreeWidget | None = None
        self.surface_region_clear_button: QPushButton | None = None
        self.surface_region_editors: dict[str, dict[str, Any]] = {}
        self.probe_reading_labels: dict[str, QLabel] = {}
        self.probe_magnitude_label: QLabel | None = None
        if object_id is None:
            heading = QLabel("Nothing selected")
            heading.setObjectName("inspectorHeading")
            self.inspector_layout.addWidget(heading)
            self.inspector_layout.addStretch(1)
            return

        if len(self.selected_ids) > 1:
            heading = QLabel(f"{len(self.selected_ids)} objects selected")
            heading.setObjectName("inspectorHeading")
            self.inspector_layout.addWidget(heading)
            labels = []
            for selected in self.selected_ids[:10]:
                try:
                    labels.append(self.adapter.get_object(selected)["label"])
                except KeyError:
                    continue
            summary = QLabel("\n".join(f"• {label}" for label in labels))
            summary.setWordWrap(True)
            self.inspector_layout.addWidget(summary)
            if len(self.selected_ids) > len(labels):
                more = QLabel(f"…and {len(self.selected_ids) - len(labels)} more")
                more.setObjectName("hint")
                self.inspector_layout.addWidget(more)
            note = QLabel(
                "Ctrl-click in the scene tree or 3D view to adjust the selection. "
                "Properties are edited one object at a time."
            )
            note.setWordWrap(True)
            note.setObjectName("hint")
            self.inspector_layout.addWidget(note)
            group_button = QPushButton("Group selected objects")
            group_button.setObjectName("primaryButton")
            group_button.setEnabled(len(self._editable_selected_ids()) >= 2)
            group_button.clicked.connect(self.group_selected)
            self.inspector_layout.addWidget(group_button)
            self.inspector_layout.addStretch(1)
            return

        obj = self.adapter.get_object(object_id)
        heading = QLabel(obj["label"])
        heading.setObjectName("inspectorHeading")
        self.inspector_layout.addWidget(heading)
        type_label = QLabel(f"{self.adapter.object_type_label(object_id)}  •  {object_id}")
        type_label.setObjectName("hint")
        type_label.setWordWrap(True)
        self.inspector_layout.addWidget(type_label)

        if obj.get("derived"):
            note = QLabel(
                f"This is a generated copy of {obj['derived']}. Edit the source object or its pattern."
            )
            note.setWordWrap(True)
            note.setObjectName("hint")
            self.inspector_layout.addWidget(note)
            self.inspector_layout.addStretch(1)
            return

        identity = QGroupBox("Object")
        identity_form = QFormLayout(identity)
        self.label_edit = QLineEdit(obj["label"])
        identity_form.addRow("Name", self.label_edit)
        self.inspector_layout.addWidget(identity)
        deferred_after_position: list[QWidget] = []

        if obj.get("workbench_magnetometer_probe"):
            properties = self.adapter.magnetometer_probe_properties(object_id)
            definition = properties["definition"]
            probe_housing = definition.get("housing", {})
            sticker_face = str(probe_housing.get("sticker_face") or "").strip()
            sticker_label = str(probe_housing.get("sticker_label") or "STICKER SIDE").strip()
            cable_face = str(probe_housing.get("cable_face") or "").strip()
            model_box = self._inspector_section("Probe model", expanded=False)
            model_form = QFormLayout()
            model_box.setContentLayout(model_form)
            model_form.addRow("Manufacturer", QLabel(str(definition["manufacturer"])))
            model_form.addRow("Model", QLabel(str(definition["model"])))
            status = QLabel(str(definition["verification_status"]).replace("-", " ").title())
            status.setProperty(
                "statusTone",
                "success"
                if definition["verification_status"] == "manufacturer-derived"
                else "muted",
            )
            model_form.addRow("Definition", status)
            for row_name, value in (
                ("Description", definition.get("description", "")),
                ("Local origin", definition.get("origin_description", "")),
                ("Axis convention", definition.get("coordinate_system", "")),
                ("Housing", probe_housing.get("geometry_note", "")),
                (
                    "Sticker/label face",
                    f"{sticker_face} — {sticker_label}" if sticker_face else "",
                ),
                (
                    "Cable connection face",
                    f"{cable_face} — red end face" if cable_face else "",
                ),
            ):
                if not value:
                    continue
                label = QLabel(str(value))
                label.setWordWrap(True)
                label.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                    | Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
                model_form.addRow(row_name, label)
            sources = definition.get("sources", [])
            for index, source in enumerate(sources):
                if not isinstance(source, dict):
                    continue
                title = str(source.get("title") or source.get("url") or "Source")
                url = str(source.get("url") or "")
                source_label = QLabel(
                    f"<a href='{html.escape(url, quote=True)}'>{html.escape(title)}</a>"
                    if url
                    else html.escape(title)
                )
                source_label.setOpenExternalLinks(True)
                source_label.setWordWrap(True)
                model_form.addRow("Source" if index == 0 else "", source_label)
            channel_table = QTreeWidget()
            channel_table.setObjectName("probeChannelGeometryTable")
            channel_table.setHeaderLabels(["Channel", "Local point (mm)", "Sensitive direction"])
            channel_table.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            channel_table.setRootIsDecorated(False)
            channel_table.setAlternatingRowColors(True)
            channel_table.setMaximumHeight(145)
            for channel in definition["channels"]:
                channel_table.addTopLevelItem(
                    QTreeWidgetItem(
                        [
                            str(channel["name"]),
                            ", ".join(f"{float(value):g}" for value in channel["position_mm"]),
                            ", ".join(f"{float(value):g}" for value in channel["sensitive_axis"]),
                        ]
                    )
                )
            model_form.addRow(channel_table)
            deferred_after_position.append(model_box)

            readings = self._inspector_section("Simulated probe reading", expanded=False)
            readings_form = QFormLayout()
            readings.setContentLayout(readings_form)
            for channel in definition["channels"]:
                value = QLabel("Not calculated")
                value.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                    | Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
                self.probe_reading_labels[str(channel["name"])] = value
                readings_form.addRow(str(channel.get("output") or channel["name"]), value)
            self.probe_magnitude_label = QLabel("Not calculated")
            self.probe_magnitude_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            readings_form.addRow("Combined magnitude", self.probe_magnitude_label)
            reading_note = QLabel(
                "Each channel samples its own physical point and projects the complete field "
                "onto that channel's probe-local sensitive direction."
            )
            reading_note.setWordWrap(True)
            reading_note.setObjectName("hint")
            readings_form.addRow(reading_note)
            calculate = QPushButton("Calculate probe reading")
            calculate.setObjectName("calculateMagnetometerProbeButton")
            calculate.clicked.connect(self.calculate_selected_magnetometer_probe)
            readings_form.addRow(calculate)
            deferred_after_position.append(readings)

        if obj.get("workbench_background_field"):
            identity_form.addRow(self._inline_apply_button("Apply Object"))
            properties = self.adapter.background_field_properties(object_id)
            vector = [float(value) for value in properties.get("vector_uT", [0.0, 0.0, 0.0])]
            field_box = self._inspector_section("Uniform static field", expanded=False)
            field_form = QFormLayout()
            field_box.setContentLayout(field_form)
            for axis, value in zip(("Bx", "By", "Bz"), vector, strict=True):
                value_label = QLabel(f"{value:.6g} µT")
                value_label.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                    | Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
                field_form.addRow(axis, value_label)
            magnitude = math.sqrt(sum(value * value for value in vector))
            field_form.addRow("Magnitude", QLabel(f"{magnitude:.6g} µT"))
            mode = str(properties.get("mode", "manual"))
            source = properties.get("source", {})
            field_form.addRow("Source", QLabel("WMM2025 Earth field" if mode == "wmm2025" else "Manual vector"))
            if mode == "wmm2025" and isinstance(source, dict):
                location = QLabel(
                    f"{float(source.get('latitude_deg', 0.0)):.6f}°, "
                    f"{float(source.get('longitude_deg', 0.0)):.6f}° • "
                    f"{float(source.get('altitude_m', 0.0)):.1f} m • "
                    f"{source.get('date', '—')}"
                )
                location.setWordWrap(True)
                location.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                    | Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
                field_form.addRow("Location / date", location)
                orientation = QLabel(
                    f"heading {float(source.get('heading_deg', 0.0)):.2f}° • "
                    f"pitch {float(source.get('pitch_deg', 0.0)):.2f}° • "
                    f"roll {float(source.get('roll_deg', 0.0)):.2f}°"
                )
                orientation.setWordWrap(True)
                field_form.addRow("Orientation", orientation)
                geographic = QLabel(
                    f"N {float(source.get('north_uT', 0.0)):.4f} µT • "
                    f"E {float(source.get('east_uT', 0.0)):.4f} µT • "
                    f"Down {float(source.get('down_uT', 0.0)):.4f} µT"
                )
                geographic.setWordWrap(True)
                field_form.addRow("Geographic field", geographic)
            note = QLabel(
                "This vector is added uniformly to every scene field evaluation. "
                "WMM values are frozen until you explicitly edit/recalculate this object."
            )
            note.setWordWrap(True)
            note.setObjectName("hint")
            field_form.addRow(note)
            edit_field = QPushButton("Edit field…")
            edit_field.setObjectName("primaryButton")
            edit_field.clicked.connect(self.edit_background_field)
            field_form.addRow(edit_field)
            transform = self.adapter.get_transform(object_id)
            display_box = self._inspector_section("Position", expanded=True)
            display_form = QFormLayout()
            display_box.setContentLayout(display_form)
            self.background_position_spins = []
            for axis, value in zip("XYZ", transform.get("position", [0.0, 0.0, 0.0]), strict=True):
                spin = _number_spin(value * 1000.0, decimals=3, suffix=" mm")
                self.background_position_spins.append(spin)
                display_form.addRow(f"{axis} position", spin)
            display_note = QLabel(
                "Display only: this moves the vector glyph in the 3D scene. "
                "The background field remains uniform everywhere."
            )
            display_note.setWordWrap(True)
            display_note.setObjectName("hint")
            display_form.addRow(display_note)
            apply_name = QPushButton("Apply Position")
            apply_name.setObjectName("inlineApplyButton")
            apply_name.clicked.connect(self.apply_background_field_name)
            display_form.addRow(apply_name)
            self.inspector_layout.addWidget(display_box)
            self.inspector_layout.addWidget(field_box)
            self.inspector_layout.addStretch(1)
            return

        geometry_properties_for_inspector: dict[str, Any] | None = None
        if obj.get("workbench_geometry"):
            properties = self.adapter.geometry_properties(object_id)
            geometry_properties_for_inspector = properties

            # Object identity and semantic role stay together at the top of the inspector.
            self.geometry_role_combo = QComboBox()
            for role in properties["eligible_roles"]:
                self.geometry_role_combo.addItem(GEOMETRY_ROLE_LABELS[role], role)
            index = self.geometry_role_combo.findData(properties["role"])
            self.geometry_role_combo.setCurrentIndex(max(0, index))
            identity_form.addRow("Role", self.geometry_role_combo)
            identity_form.addRow(self._inline_apply_button("Apply Object"))

            # Geometry/appearance is created here but inserted after Position below, so
            # the common object workflow reads Object → Measurement → Position → Geometry → DRC.
            self.geometry_appearance_section = self._inspector_section(
                "Geometry and Appearance", expanded=True
            )
            self.geometry_appearance_form = QFormLayout()
            self.geometry_appearance_section.setContentLayout(self.geometry_appearance_form)
            self.geometry_colour_button = QPushButton()
            self.geometry_colour_button.setObjectName("geometryColourButton")
            self.geometry_colour_button.setMinimumWidth(110)
            self.geometry_colour_button.setToolTip("Choose this object's display colour.")
            self._style_geometry_colour_button(
                self.geometry_colour_button, str(properties["colour"])
            )
            self.geometry_colour_button.clicked.connect(self._choose_geometry_colour)
            self.geometry_opacity_spin = _number_spin(
                properties["opacity"], decimals=2, minimum=0.02, maximum=1.0
            )
            self.geometry_opacity_spin.setSingleStep(0.05)

            # DRC is an independent disclosure group. Its nested controls are revealed
            # only when the parent option is active.
            self.drc_section = self._inspector_section("DRC", expanded=False)
            drc_form = QFormLayout()
            self.drc_section.setContentLayout(drc_form)
            self.geometry_drc_enabled_check = QCheckBox("Enable DRC")
            self.geometry_drc_enabled_check.setToolTip(
                "Include this geometry in design-rule checks. Measurement objects default "
                "off; Exclusion zones default on. The setting remains independently toggleable."
            )
            self.geometry_drc_enabled_check.setChecked(
                bool(properties.get("drc_enabled", False))
            )
            self.geometry_drc_setting_explicit = bool(
                properties.get("drc_enabled_explicit", False)
            )
            drc_form.addRow(self.geometry_drc_enabled_check)
            self.geometry_clearance_override_check = QCheckBox("DRC Global Override")
            clearance_override = properties.get("drc_clearance_override_mm")
            self.geometry_clearance_override_check.setChecked(clearance_override is not None)
            drc_form.addRow(self.geometry_clearance_override_check)
            self.geometry_clearance_spin = _number_spin(
                0.0 if clearance_override is None else clearance_override,
                decimals=3, minimum=0.0, maximum=1e6, suffix=" mm",
            )
            self.geometry_clearance_spin.setToolTip(
                "Object-level DRC clearance. Named regions inherit this value unless "
                "they provide their own override; when this override is off, the scene-wide "
                "DRC default is used."
            )
            drc_form.addRow("Zone override", self.geometry_clearance_spin)
            self.geometry_clearance_spin_label = drc_form.labelForField(
                self.geometry_clearance_spin
            )
            drc_form.addRow(self._inline_apply_button("Apply DRC"))

            # Role-specific measurement settings are nested under the selected role and
            # disappear completely when the object is not a Measurement object.
            if properties["measurement_capable"]:
                measurement = self.adapter.measurement_properties(object_id)
                self.measurement_section = self._inspector_section(
                    "Measurement", expanded=False
                )
                self.measurement_box = self.measurement_section
                measurement_form = QFormLayout()
                self.measurement_section.setContentLayout(measurement_form)
                self.measurement_quality_combo = QComboBox()
                self.measurement_quality_combo.addItem("Preview — 7 samples/axis", "preview")
                self.measurement_quality_combo.addItem("Standard — 15 samples/axis", "standard")
                self.measurement_quality_combo.addItem("Fine — 21 samples/axis", "fine")
                self.measurement_quality_combo.addItem("User defined points", "user_defined")
                sampling_choice = (
                    "user_defined"
                    if measurement.get("sampling_mode") == "user_defined"
                    else measurement["quality"]
                )
                self.measurement_quality_combo.setCurrentIndex(
                    max(0, self.measurement_quality_combo.findData(sampling_choice))
                )
                measurement_form.addRow("Sampling", self.measurement_quality_combo)
                self.measurement_defined_points_edit = QPlainTextEdit()
                self.measurement_defined_points_edit.setMaximumHeight(105)
                self.measurement_defined_points_edit.setPlaceholderText(
                    "[x0, y0, z0], [x1, y1, z1]  (local coordinates in mm)"
                )
                self.measurement_defined_points_edit.setPlainText(
                    self._format_defined_measurement_points(
                        measurement.get("defined_points_mm", [])
                    )
                )
                measurement_form.addRow(
                    "Defined points (local mm)", self.measurement_defined_points_edit
                )
                self.measurement_defined_points_label = measurement_form.labelForField(
                    self.measurement_defined_points_edit
                )
                self.measurement_target_spin = _number_spin(
                    measurement["target_uT"], decimals=4, minimum=0.0, maximum=1e12, suffix=" µT"
                )
                measurement_form.addRow("Target |B|", self.measurement_target_spin)
                self.measurement_tolerance_spin = _number_spin(
                    measurement["tolerance_pct"], decimals=2, minimum=0.0, maximum=100.0, suffix=" %"
                )
                self.measurement_tolerance_spin.setSingleStep(0.5)
                measurement_form.addRow("Tolerance", self.measurement_tolerance_spin)
                self.measurement_note = QLabel()
                self.measurement_note.setWordWrap(True)
                self.measurement_note.setObjectName("hint")
                measurement_form.addRow(self.measurement_note)
                measurement_form.addRow(self._inline_apply_button("Apply Measurement"))
                self.measurement_analyze_button = QPushButton("Apply && analyze volume")
                self.measurement_analyze_button.setObjectName("primaryButton")
                self.measurement_analyze_button.clicked.connect(
                    self.analyze_selected_measurement
                )
                measurement_form.addRow(self.measurement_analyze_button)
                self.inspector_layout.addWidget(self.measurement_section)
                self.measurement_quality_combo.currentIndexChanged.connect(
                    self._update_measurement_sampling_controls
                )
                self._update_measurement_sampling_controls()
            elif properties.get("measurement_eligibility_reason"):
                eligibility_note = QLabel(properties["measurement_eligibility_reason"])
                eligibility_note.setWordWrap(True)
                eligibility_note.setObjectName("hint")
                self.inspector_layout.addWidget(eligibility_note)

            self._deferred_surface_regions = properties.get("surface_regions")
            self._deferred_mesh_health = properties.get("mesh_health")

            self.geometry_role_combo.currentIndexChanged.connect(
                self._geometry_role_drc_default_changed
            )
            self.geometry_role_combo.currentIndexChanged.connect(
                self._update_measurement_controls
            )
            self.geometry_drc_enabled_check.toggled.connect(
                self._geometry_drc_enabled_toggled
            )
            self.geometry_clearance_override_check.toggled.connect(
                self._update_geometry_drc_controls
            )
            self._update_measurement_controls()
            self._update_geometry_drc_controls()
        else:
            identity_form.addRow(self._inline_apply_button("Apply Object"))

        transform = self.adapter.get_transform(object_id)
        self.axis_path_editable = obj["type"] == "Sensor" and transform["path_length"] > 1
        self.transform_editable = transform["path_length"] == 1
        self.position_spins = []
        self.rotation_spins = []
        if self.axis_path_editable:
            transform_box = self._inspector_section("Position", expanded=True)
            transform_form = QFormLayout()
            transform_box.setContentLayout(transform_form)
            path = transform["path"]
            self.axis_start_spins = []
            self.axis_end_spins = []
            for axis, value in zip("XYZ", path[0], strict=True):
                spin = _number_spin(value * 1000.0, decimals=3, suffix=" mm")
                self.axis_start_spins.append(spin)
                transform_form.addRow(f"Start {axis}", spin)
            for axis, value in zip("XYZ", path[-1], strict=True):
                spin = _number_spin(value * 1000.0, decimals=3, suffix=" mm")
                self.axis_end_spins.append(spin)
                transform_form.addRow(f"End {axis}", spin)
            self.axis_samples_spin = QSpinBox()
            self.axis_samples_spin.setRange(2, 2001)
            self.axis_samples_spin.setValue(transform["path_length"])
            self.axis_samples_spin.setSuffix(" points")
            self.axis_samples_spin.setMinimumWidth(140)
            transform_form.addRow("Samples", self.axis_samples_spin)
            path_note = QLabel(
                "The sensor samples evenly along this straight line. Its Sensor Plot uses all of these points."
            )
            path_note.setWordWrap(True)
            path_note.setObjectName("hint")
            transform_form.addRow(path_note)
        else:
            transform_box = self._inspector_section("Position", expanded=True)
            transform_form = QFormLayout()
            transform_box.setContentLayout(transform_form)
            for axis, value in zip("XYZ", transform["position"], strict=True):
                spin = _number_spin(value * 1000.0, decimals=3, suffix=" mm")
                spin.setEnabled(self.transform_editable)
                self.position_spins.append(spin)
                transform_form.addRow(f"{axis} position", spin)
            for axis, value in zip("XYZ", transform["euler"], strict=True):
                spin = _number_spin(
                    value,
                    decimals=3,
                    minimum=-360000,
                    maximum=360000,
                    suffix="°",
                )
                spin.setEnabled(self.transform_editable)
                self.rotation_spins.append(spin)
                transform_form.addRow(f"{axis} rotation", spin)
            if not self.transform_editable:
                path_note = QLabel(
                    f"This object follows a {transform['path_length']}-point path. "
                    "Only straight sensor paths are editable in this version."
                )
                path_note.setWordWrap(True)
                path_note.setObjectName("hint")
                transform_form.addRow(path_note)
        transform_form.addRow(self._inline_apply_button("Apply Position"))
        self.inspector_layout.addWidget(transform_box)
        for deferred_section in deferred_after_position:
            self.inspector_layout.addWidget(deferred_section)

        is_coil = self.adapter.is_coil(object_id)
        coil_properties = (
            self.adapter.coil_physical_properties(object_id) if is_coil else None
        )
        is_racetrack = bool(
            coil_properties and coil_properties.get("coil_shape") == "racetrack"
        )

        # Coil geometry is presented with the required magnetic model below.
        # Racetrack generated vertices remain an implementation detail.
        params = self.adapter.get_params(object_id)
        hidden_params = {"pixel", "magnetization"}
        if is_coil:
            hidden_params.add("current")
        if is_racetrack:
            hidden_params.add("vertices")
        editable_params = [p for p in params if p["name"] not in hidden_params]
        if obj.get("workbench_geometry"):
            physics = self.geometry_appearance_section
            physics_form = self.geometry_appearance_form
            for param in editable_params:
                editor, factor = self._parameter_editor(param)
                unit = self._display_unit(param.get("unit", ""))
                label = param["name"].replace("_", " ").title()
                if unit:
                    label += f" ({unit})"
                physics_form.addRow(label, editor)
                self.parameter_editors[param["name"]] = {
                    "widget": editor, "factor": factor, "kind": param["kind"],
                }
                if param.get("doc"):
                    editor.setToolTip(param["doc"])

            physics_form.addRow("Colour", self.geometry_colour_button)
            physics_form.addRow("Opacity", self.geometry_opacity_spin)

            geometry_properties = geometry_properties_for_inspector or {}
            source_dimensions = geometry_properties.get("source_dimensions_mm")
            if source_dimensions is not None:
                self.mesh_source_dimensions_mm = list(map(float, source_dimensions))
                selectable = (
                    Qt.TextInteractionFlag.TextSelectableByMouse
                    | Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
                original = QLabel()
                original.setTextInteractionFlags(selectable)
                current = QLabel()
                current.setTextInteractionFlags(selectable)
                self.mesh_dimension_labels = {"original": original, "current": current}
                physics_form.addRow("Imported dimensions", original)
                physics_form.addRow("Scaled dimensions", current)
                dimension_note = QLabel(
                    "Dimensions use the mesh's local X × Y × Z axes. Independent scale is "
                    "applied before object rotation."
                )
                dimension_note.setWordWrap(True)
                dimension_note.setObjectName("hint")
                physics_form.addRow(dimension_note)
                for name in ("scale_x", "scale_y", "scale_z"):
                    editor_info = self.parameter_editors.get(name)
                    if editor_info is not None:
                        editor_info["widget"].valueChanged.connect(
                            self._update_mesh_dimensions_preview
                        )
                self._update_mesh_dimensions_preview()

            source = geometry_properties.get("source")
            if source:
                source_label = QLabel(
                    f"{source} • {int(geometry_properties.get('vertices', 0)):,} vertices • "
                    f"{int(geometry_properties.get('faces', 0)):,} triangles"
                )
                source_label.setWordWrap(True)
                source_label.setObjectName("hint")
                physics_form.addRow(
                    "Built-in" if geometry_properties.get("builtin_template") else "Imported",
                    source_label,
                )
            if geometry_properties.get("upenn_gbm_case"):
                physics_form.addRow("UPENN-GBM case", QLabel(str(geometry_properties["upenn_gbm_case"])))
                region_value = str(geometry_properties.get("upenn_gbm_region", "")).replace("_", " ").title()
                physics_form.addRow("Tumor region", QLabel(region_value))
                volume = geometry_properties.get("upenn_gbm_segmented_volume_ml")
                if volume is not None:
                    physics_form.addRow("Segmented volume", QLabel(f"{float(volume):,.4g} mL"))
            well_plate = geometry_properties.get("well_plate")
            if isinstance(well_plate, dict):
                wells = int(well_plate.get("wells", 0) or 0)
                rows = int(well_plate.get("rows", 0) or 0)
                columns = int(well_plate.get("columns", 0) or 0)
                pitch_mm = float(well_plate.get("pitch_mm", 0.0) or 0.0)
                well_diameter_mm = float(well_plate.get("well_diameter_mm", 0.0) or 0.0)
                physics_form.addRow("Plate format", QLabel(f"{wells}-well ({rows} × {columns})"))
                physics_form.addRow("Well pitch", QLabel(f"{pitch_mm:g} mm centre-to-centre"))
                physics_form.addRow("Well opening", QLabel(f"{well_diameter_mm:g} mm representative diameter"))

            physics_form.addRow(self._inline_apply_button("Apply Geometry"))
            self.inspector_layout.addWidget(physics)
            self.inspector_layout.addWidget(self.drc_section)

            surface_regions = getattr(self, "_deferred_surface_regions", None)
            if isinstance(surface_regions, dict):
                self._add_surface_regions_inspector(object_id, surface_regions)
            mesh_health = getattr(self, "_deferred_mesh_health", None)
            if isinstance(mesh_health, dict):
                self._add_mesh_health_inspector(mesh_health)

        elif editable_params and not is_coil:
            physics = self._inspector_section("Physical parameters", expanded=True)
            physics_form = QFormLayout()
            physics.setContentLayout(physics_form)
            for param in editable_params:
                editor, factor = self._parameter_editor(param)
                unit = self._display_unit(param.get("unit", ""))
                label = param["name"].replace("_", " ").title()
                if is_coil and param["name"] == "diameter":
                    label = "Mean diameter (2 × radius)"
                if unit:
                    label += f" ({unit})"
                physics_form.addRow(label, editor)
                self.parameter_editors[param["name"]] = {
                    "widget": editor, "factor": factor, "kind": param["kind"],
                }
                if param.get("doc"):
                    editor.setToolTip(param["doc"])
            physics_form.addRow(self._inline_apply_button("Apply Parameters"))
            self.inspector_layout.addWidget(physics)

        if is_coil:
            self._add_coil_inspector(
                object_id,
                properties=coil_properties,
                magnetic_params=editable_params,
            )

        self.inspector_layout.addStretch(1)
        self._install_inspector_wheel_filter()

    def calculate_selected_magnetometer_probe(self) -> None:
        """Evaluate a selected physical probe without changing the scene document."""
        object_id = self.selected_id
        if not object_id or not self.adapter.is_magnetometer_probe(object_id):
            return
        self.statusBar().showMessage("Calculating magnetometer probe reading…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.adapter.magnetometer_probe_reading(object_id, field="B")
        except Exception as error:  # noqa: BLE001 - calculation/UI boundary
            QMessageBox.warning(self, "Magnetometer probe reading", str(error))
            self.statusBar().showMessage("Probe calculation failed", 4000)
            return
        finally:
            QApplication.restoreOverrideCursor()

        for channel in result["channels"]:
            label = self.probe_reading_labels.get(str(channel["name"]))
            if label is None:
                continue
            reading_uT = float(channel["reading"]) * 1.0e6
            label.setText(f"{reading_uT:.9g} µT")
            point = channel["world_position_mm"]
            world = np.asarray(channel["world_vector"], dtype=float) * 1.0e6
            label.setToolTip(
                "World point: "
                + ", ".join(f"{float(value):.6g}" for value in point)
                + " mm\nComplete world B: "
                + ", ".join(f"{float(value):.9g}" for value in world)
                + " µT"
            )
        if self.probe_magnitude_label is not None:
            self.probe_magnitude_label.setText(
                f"{float(result['reported_magnitude']) * 1.0e6:.9g} µT"
            )
        self.statusBar().showMessage("Magnetometer probe reading calculated", 4000)

    @staticmethod
    def _style_colour_button(
        button: QPushButton, colour: str, *, property_name: str
    ) -> None:
        value = str(colour).strip().lower()
        button.setProperty(property_name, value)
        button.setText(value.upper())
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: {value};"
            "color: white; border: 1px solid #475569; padding: 3px 6px;"
            "font-weight: 600; }"
        )

    @staticmethod
    def _style_surface_region_colour_button(button: QPushButton, colour: str) -> None:
        MainWindow._style_colour_button(
            button, colour, property_name="surface_region_colour"
        )

    @staticmethod
    def _style_geometry_colour_button(button: QPushButton, colour: str) -> None:
        MainWindow._style_colour_button(
            button, colour, property_name="geometry_colour"
        )

    def _choose_geometry_colour(self) -> None:
        button = self.geometry_colour_button
        if not isinstance(button, QPushButton):
            return
        current = str(button.property("geometry_colour") or "#64748b")
        colour = QColorDialog.getColor(
            QColor(current),
            self,
            "Object colour",
            QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if colour.isValid():
            self._style_geometry_colour_button(button, colour.name())

    def _choose_surface_region_colour(self, region_key: str) -> None:
        editor = self.surface_region_editors.get(str(region_key))
        if not editor:
            return
        button = editor.get("colour_button")
        if not isinstance(button, QPushButton):
            return
        current = str(button.property("surface_region_colour") or "#64748b")
        colour = QColorDialog.getColor(
            QColor(current),
            self,
            "Surface region colour",
            QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if colour.isValid():
            self._style_surface_region_colour_button(button, colour.name())

    def _set_all_surface_region_visibility(self, visible: bool) -> None:
        state = Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
        tree = self.surface_region_tree
        if tree is None:
            return
        for editor in self.surface_region_editors.values():
            item = editor.get("item")
            if isinstance(item, QTreeWidgetItem):
                item.setCheckState(3, state)

    def _add_mesh_health_inspector(self, mesh_health: dict[str, Any]) -> None:
        section = self._inspector_section("Imported mesh health", expanded=False)
        health_form = QFormLayout()
        section.setContentLayout(health_form)
        topology = (
            "Watertight manifold"
            if mesh_health.get("watertight") and mesh_health.get("manifold")
            else "Open or non-manifold"
        )
        health_form.addRow("Topology", QLabel(topology))
        health_form.addRow(
            "Edges",
            QLabel(
                f"{int(mesh_health.get('boundary_edges', 0)):,} open • "
                f"{int(mesh_health.get('non_manifold_edges', 0)):,} non-manifold"
            ),
        )
        health_form.addRow(
            "Faces",
            QLabel(
                f"{int(mesh_health.get('degenerate_or_invalid_faces', 0)):,} "
                f"degenerate/invalid • {int(mesh_health.get('duplicate_faces', 0)):,} duplicate"
            ),
        )
        health_form.addRow(
            "Normal joins",
            QLabel(f"{int(mesh_health.get('inconsistent_edges', 0)):,} inconsistent"),
        )
        verdict = QLabel(
            "Inside/outside DRC enabled"
            if mesh_health.get("containment_reliable")
            else "Surface clearance only"
        )
        verdict.setProperty(
            "statusTone",
            "success" if mesh_health.get("containment_reliable") else "warning",
        )
        health_form.addRow("Capability", verdict)
        health_note = QLabel(str(mesh_health.get("summary", "")))
        health_note.setWordWrap(True)
        health_note.setObjectName("hint")
        health_form.addRow(health_note)
        self.inspector_layout.addWidget(section)

    def _add_surface_regions_inspector(
        self, object_id: str, region_map: dict[str, Any]
    ) -> None:
        """Edit display, placement, and DRC semantics on a region-aware mesh."""
        regions = region_map.get("regions", {})
        if not isinstance(regions, dict) or not regions:
            return
        box = self._inspector_section("Surface regions", expanded=False)
        layout = QVBoxLayout()
        box.setContentLayout(layout)
        validation = region_map.get("validation", {})
        summary = QLabel(
            f"{len(regions):,} named regions • "
            f"{int(validation.get('unique_assigned_faces', 0)):,} / "
            f"{int(region_map.get('topology', {}).get('face_count', 0)):,} triangles mapped"
        )
        summary.setWordWrap(True)
        summary.setObjectName("hint")
        layout.addWidget(summary)

        tree = QTreeWidget()
        tree.setObjectName("surfaceRegionTree")
        tree.setHeaderLabels(
            ["Region", "Triangles", "Active", "Show", "Place", "DRC", "Clearance", "Colour"]
        )
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tree.setMinimumHeight(190)
        tree.setMaximumHeight(270)
        tree.header().setStretchLastSection(False)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 8):
            tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        active_key = (
            self.surface_region_highlight[1]
            if self.surface_region_highlight is not None
            and self.surface_region_highlight[0] == object_id
            else None
        )
        active_item: QTreeWidgetItem | None = None
        for key, region in regions.items():
            if not isinstance(region, dict):
                continue
            key = str(key)
            settings = region.get("settings", {})
            if not isinstance(settings, dict):
                settings = {}
            item = QTreeWidgetItem(
                [
                    str(region.get("display_name") or key),
                    f"{int(region.get('face_count', 0)):,}",
                    "", "", "", "", "", "",
                ]
            )
            item.setData(0, USER_ROLE_ID, key)
            item.setCheckState(
                2, Qt.CheckState.Checked if bool(settings.get("active", False)) else Qt.CheckState.Unchecked
            )
            item.setCheckState(
                3, Qt.CheckState.Checked if bool(settings.get("visible", False)) else Qt.CheckState.Unchecked
            )
            item.setCheckState(
                4, Qt.CheckState.Checked if bool(settings.get("placement", False)) else Qt.CheckState.Unchecked
            )
            item.setCheckState(
                5, Qt.CheckState.Checked if bool(settings.get("drc_exclusion", False)) else Qt.CheckState.Unchecked
            )
            item.setToolTip(2, "Active is the master semantic switch. Hidden regions can remain active.")
            item.setToolTip(3, "Show or hide this coloured region overlay in the 3D scene.")
            item.setToolTip(4, "Allow this region to act as a future surface-placement provider.")
            item.setToolTip(5, "Apply region-aware DRC clearance to this surface patch.")
            tree.addTopLevelItem(item)

            clearance_widget = QWidget(tree)
            clearance_layout = QHBoxLayout(clearance_widget)
            clearance_layout.setContentsMargins(2, 0, 2, 0)
            clearance_layout.setSpacing(4)
            override_check = QCheckBox("Override", clearance_widget)
            clearance_value = settings.get("clearance_override_mm")
            override_check.setChecked(clearance_value is not None)
            override_check.setToolTip(
                "Override the object's inherited DRC clearance on this region. "
                "When off, the object override (if any) or scene default is used."
            )
            clearance_spin = CompactDoubleSpinBox(clearance_widget)
            clearance_spin.setRange(0.0, 1e6)
            clearance_spin.setDecimals(3)
            clearance_spin.setSuffix(" mm")
            clearance_spin.setValue(0.0 if clearance_value is None else float(clearance_value))
            clearance_spin.setEnabled(clearance_value is not None)
            clearance_spin.setMinimumWidth(92)
            clearance_spin.setToolTip("Required coil clearance from this region surface.")
            override_check.toggled.connect(clearance_spin.setEnabled)
            clearance_layout.addWidget(override_check)
            clearance_layout.addWidget(clearance_spin)
            tree.setItemWidget(item, 6, clearance_widget)

            colour_button = QPushButton(tree)
            colour_button.setMinimumWidth(82)
            colour_button.setToolTip("Choose the colour used for this region overlay.")
            self._style_surface_region_colour_button(
                colour_button, str(settings.get("colour", "#64748b"))
            )
            colour_button.clicked.connect(
                lambda _checked=False, region_key=key: self._choose_surface_region_colour(region_key)
            )
            tree.setItemWidget(item, 7, colour_button)
            self.surface_region_editors[key] = {
                "item": item,
                "clearance_override_check": override_check,
                "clearance_spin": clearance_spin,
                "colour_button": colour_button,
            }
            if key == active_key:
                active_item = item
        if active_item is not None:
            with QSignalBlocker(tree):
                tree.setCurrentItem(active_item)
                active_item.setSelected(True)
        tree.itemSelectionChanged.connect(
            lambda object_id=object_id: self._surface_region_selection_changed(object_id)
        )
        layout.addWidget(tree)
        self.surface_region_tree = tree

        controls = QHBoxLayout()
        show_all = QPushButton("Show all")
        show_all.clicked.connect(lambda: self._set_all_surface_region_visibility(True))
        controls.addWidget(show_all)
        hide_all = QPushButton("Hide all")
        hide_all.clicked.connect(lambda: self._set_all_surface_region_visibility(False))
        controls.addWidget(hide_all)
        clear_button = QPushButton("Clear preview")
        clear_button.setObjectName("clearSurfaceRegionHighlightButton")
        clear_button.setEnabled(active_item is not None)
        clear_button.clicked.connect(self._clear_surface_region_highlight)
        controls.addWidget(clear_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.surface_region_clear_button = clear_button

        status_bits: list[str] = []
        if validation.get("mesh_file_sha256_verified"):
            status_bits.append("source mesh identity verified")
        if validation.get("topology_counts_verified"):
            status_bits.append("topology counts verified")
        if validation.get("workbench_geometry_sha256_verified"):
            status_bits.append("triangle ordering verified")
        if validation.get("complete_partition"):
            status_bits.append("complete non-overlapping partition")
        note = QLabel(
            "Select a row for a temporary preview. Active is the master semantic switch; Show is "
            "display-only. Placement and DRC exclusion are independent, so a scalp region can support "
            "future surface attachment while still participating in physical clearance. Region DRC "
            "inherits the object/scene clearance unless Override is enabled. The full watertight host "
            "surface remains the penetration/containment shell whenever regional DRC is active. "
            "Use Apply changes to save edits as one undoable step."
            + ("\nValidation: " + " • ".join(status_bits) if status_bits else "")
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        layout.addWidget(note)
        layout.addWidget(self._inline_apply_button())
        self.inspector_layout.addWidget(box)

    def _surface_region_selection_changed(self, object_id: str) -> None:
        tree = self.surface_region_tree
        if tree is None:
            return
        items = tree.selectedItems()
        if not items:
            self.surface_region_highlight = None
        else:
            key = items[0].data(0, USER_ROLE_ID)
            self.surface_region_highlight = (str(object_id), str(key))
        if self.surface_region_clear_button is not None:
            self.surface_region_clear_button.setEnabled(
                self.surface_region_highlight is not None
            )
        self.refresh_plot(rebuild_base=False)

    def _clear_surface_region_highlight(self) -> None:
        tree = self.surface_region_tree
        if tree is not None:
            with QSignalBlocker(tree):
                tree.clearSelection()
                tree.setCurrentItem(None)
        self.surface_region_highlight = None
        if self.surface_region_clear_button is not None:
            self.surface_region_clear_button.setEnabled(False)
        self.refresh_plot(rebuild_base=False)

    def _install_inspector_wheel_filter(self) -> None:
        """Protect every numeric/dropdown editor currently in the inspector."""
        for widget in self.inspector_container.findChildren(QWidget):
            if isinstance(widget, (QDoubleSpinBox, QSpinBox, QComboBox)):
                widget.installEventFilter(self._inspector_wheel_filter)

    def _inline_apply_button(self, text: str = "Apply") -> QPushButton:
        # Keep every inspector section action compact and visually consistent.
        # The section heading already supplies the context for what is applied.
        button = QPushButton("Apply")
        button.setObjectName("inlineApplyButton")
        button.setToolTip("Apply every pending inspector change and keep this scroll position.")
        button.setMinimumHeight(30)
        button.setMaximumWidth(150)
        button.clicked.connect(self._apply_inspector_inline)
        return button

    def _apply_inspector_inline(self) -> None:
        scroll_bar = self.inspector_scroll.verticalScrollBar()
        scroll_position = scroll_bar.value()
        if self.apply_inspector():
            QTimer.singleShot(
                0,
                lambda position=scroll_position: scroll_bar.setValue(
                    min(position, scroll_bar.maximum())
                ),
            )

    def _update_measurement_controls(self, *_args) -> None:
        if self.measurement_box is None or self.geometry_role_combo is None:
            return
        enabled = self.geometry_role_combo.currentData() == "measurement"
        self.measurement_box.setEnabled(enabled)
        self.measurement_box.setVisible(enabled)

    @staticmethod
    def _format_defined_measurement_points(points: Any) -> str:
        try:
            rows = [list(map(float, point)) for point in points]
        except (TypeError, ValueError):
            return ""
        return ",\n".join(json.dumps(row, separators=(", ", ": ")) for row in rows)

    @staticmethod
    def _parse_defined_measurement_points(text: str) -> list[list[float]]:
        value = str(text).strip()
        if not value:
            return []
        candidates = [value]
        trimmed = value.rstrip().rstrip(",")
        candidates.append(f"[{trimmed}]")
        lines = [line.strip().rstrip(",") for line in value.splitlines() if line.strip()]
        if len(lines) > 1:
            candidates.append("[" + ",".join(lines) + "]")
        parsed = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
                break
            except (SyntaxError, ValueError) as error:
                last_error = error
        if parsed is None:
            raise ValueError(
                "Could not read the user-defined points. Use [x, y, z], [x, y, z], … "
                "with local coordinates in millimetres."
            ) from last_error
        if isinstance(parsed, (tuple, list)) and len(parsed) == 3 and all(
            isinstance(value, (int, float)) for value in parsed
        ):
            parsed = [parsed]
        if not isinstance(parsed, (tuple, list)):
            raise ValueError(
                "User-defined points must be a list of [X, Y, Z] coordinates."
            )
        result: list[list[float]] = []
        for index, point in enumerate(parsed, start=1):
            if not isinstance(point, (tuple, list)) or len(point) != 3:
                raise ValueError(
                    f"User-defined sample point {index} must contain exactly X, Y, and Z."
                )
            try:
                coordinates = [float(value) for value in point]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"User-defined sample point {index} contains a non-numeric coordinate."
                ) from error
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(
                    f"User-defined sample point {index} contains a non-finite coordinate."
                )
            result.append(coordinates)
        return result

    def _update_measurement_sampling_controls(self, *_args) -> None:
        if self.measurement_quality_combo is None:
            return
        user_defined = self.measurement_quality_combo.currentData() == "user_defined"
        if self.measurement_defined_points_edit is not None:
            self.measurement_defined_points_edit.setVisible(user_defined)
        if self.measurement_defined_points_label is not None:
            self.measurement_defined_points_label.setVisible(user_defined)
        if self.measurement_note is not None:
            if user_defined:
                self.measurement_note.setText(
                    "Each [X, Y, Z] row is an equal-weight sample point in the object's local "
                    "coordinate frame, in millimetres. Object position and rotation are applied "
                    "before field calculation; points are used exactly as entered rather than "
                    "clipped to the object. Apply & analyze also saves pending edits."
                )
            else:
                self.measurement_note.setText(
                    "Samples are equal-volume voxel centres inside the object's local volume. "
                    "Apply & analyze also saves any pending size, transform, role, or target edits."
                )

    def _geometry_has_surface_regions(self) -> bool:
        if self.selected_id is None:
            return False
        try:
            return isinstance(
                self.adapter.surface_region_properties(self.selected_id), dict
            )
        except (StudioOperationError, KeyError):
            return False

    def _geometry_has_active_region_drc(self) -> bool:
        if self.selected_id is None:
            return False
        try:
            region_map = self.adapter.surface_region_properties(self.selected_id)
        except (StudioOperationError, KeyError):
            region_map = None
        if not isinstance(region_map, dict):
            return False
        for region in region_map.get("regions", {}).values():
            settings = region.get("settings", {}) if isinstance(region, dict) else {}
            if (
                isinstance(settings, dict)
                and bool(settings.get("active"))
                and bool(settings.get("drc_exclusion"))
            ):
                return True
        return False

    def _geometry_role_drc_default_changed(self, *_args) -> None:
        if self.geometry_role_combo is None or self.geometry_drc_enabled_check is None:
            return
        if not self.geometry_drc_setting_explicit:
            default_enabled = (
                self.geometry_role_combo.currentData() == "exclusion"
                or self._geometry_has_active_region_drc()
            )
            self._geometry_drc_syncing = True
            try:
                self.geometry_drc_enabled_check.setChecked(default_enabled)
            finally:
                self._geometry_drc_syncing = False
        self._update_geometry_drc_controls()

    def _geometry_drc_enabled_toggled(self, *_args) -> None:
        if not self._geometry_drc_syncing:
            self.geometry_drc_setting_explicit = True
        self._update_geometry_drc_controls()

    def _update_geometry_drc_controls(self, *_args) -> None:
        if (
            self.geometry_role_combo is None
            or self.geometry_drc_enabled_check is None
            or self.geometry_clearance_override_check is None
            or self.geometry_clearance_spin is None
        ):
            return
        role = self.geometry_role_combo.currentData()
        has_surface_regions = self._geometry_has_surface_regions()
        drc_available = role in {"exclusion", "measurement"} or has_surface_regions
        self.geometry_drc_enabled_check.setEnabled(drc_available)
        drc_enabled = drc_available and self.geometry_drc_enabled_check.isChecked()
        self.geometry_clearance_override_check.setEnabled(drc_enabled)
        self.geometry_clearance_override_check.setVisible(drc_enabled)
        override_enabled = drc_enabled and self.geometry_clearance_override_check.isChecked()
        self.geometry_clearance_spin.setEnabled(override_enabled)
        self.geometry_clearance_spin.setVisible(override_enabled)
        if self.geometry_clearance_spin_label is not None:
            self.geometry_clearance_spin_label.setVisible(override_enabled)

    def _add_coil_inspector(
        self,
        object_id: str,
        *,
        properties: dict[str, Any] | None = None,
        magnetic_params: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the common coil inspector: magnetic model first, optional physical data after."""
        properties = properties or self.adapter.coil_physical_properties(object_id)
        magnetic_params = magnetic_params or []
        self.coil_object_id = object_id
        coil_shape = str(properties.get("coil_shape", "generic"))

        # --- Required magnetic model -------------------------------------------------
        magnetics = self._inspector_section("Magnetics", expanded=True)
        magnetic_layout = QVBoxLayout()
        magnetic_layout.setSpacing(8)
        magnetics.setContentLayout(magnetic_layout)
        magnetic_form = QFormLayout()
        magnetic_form.setContentsMargins(0, 0, 0, 0)
        magnetic_layout.addLayout(magnetic_form)

        enabled = QCheckBox()
        enabled.setChecked(bool(properties.get("enabled", True)))
        enabled.setToolTip(
            "When unchecked, the coil keeps its configured magnetic and construction values "
            "but contributes zero current to field calculations."
        )
        magnetic_form.addRow("Enable", enabled)
        self.coil_widgets["enabled"] = enabled

        # Native/current-path geometry belongs to the magnetic model because it is
        # required to define the source. Generated racetrack vertices stay hidden.
        if coil_shape == "racetrack":
            end_diameter = _number_spin(
                properties.get("racetrack_end_diameter_mm", 100.0),
                decimals=3,
                minimum=0.001,
                maximum=1e6,
                suffix=" mm",
            )
            end_diameter.setToolTip(
                "Diameter of each semicircular end on the magnetic centreline path."
            )
            magnetic_form.addRow("End diameter", end_diameter)
            self.coil_widgets["racetrack_end_diameter_mm"] = end_diameter

            straight_length = _number_spin(
                properties.get("racetrack_straight_length_mm", 100.0),
                decimals=3,
                minimum=0.0,
                maximum=1e6,
                suffix=" mm",
            )
            straight_length.setToolTip(
                "Length of each straight section between tangent points."
            )
            magnetic_form.addRow("Straight length", straight_length)
            self.coil_widgets["racetrack_straight_length_mm"] = straight_length

            arc_segments = QSpinBox()
            arc_segments.setRange(8, 256)
            arc_segments.setValue(int(properties.get("racetrack_arc_segments", 32)))
            arc_segments.setToolTip(
                "Polyline resolution used for each semicircular end."
            )
            magnetic_form.addRow("Segments / semicircle", arc_segments)
            self.coil_widgets["racetrack_arc_segments"] = arc_segments
        else:
            for param in magnetic_params:
                editor, factor = self._parameter_editor(param)
                name = str(param["name"])
                unit = self._display_unit(param.get("unit", ""))
                if name == "diameter":
                    label = "Mean diameter"
                elif name == "vertices":
                    label = "Current path vertices"
                else:
                    label = name.replace("_", " ").title()
                if unit:
                    label += f" ({unit})"
                magnetic_form.addRow(label, editor)
                self.parameter_editors[name] = {
                    "widget": editor,
                    "factor": factor,
                    "kind": param["kind"],
                }
                if param.get("doc"):
                    editor.setToolTip(param["doc"])

        turns = QSpinBox()
        turns.setRange(1, 1_000_000)
        turns.setValue(int(properties["turns"]))
        turns.setGroupSeparatorShown(True)
        turns.setMinimumWidth(100)
        magnetic_form.addRow("Turns", turns)
        self.coil_widgets["turns"] = turns

        drive = _number_spin(
            properties["drive_current_a"] * 1000.0,
            decimals=4,
            minimum=-100000.0,
            maximum=100000.0,
            suffix=" mA",
        )
        mode = QComboBox()
        mode.addItem("PK", "peak_sine")
        mode.addItem("RMS/DC", "rms_dc")
        mode.setToolTip("Interpret the entered current as sine-wave peak or as RMS/DC.")
        mode.setCurrentIndex(max(0, mode.findData(properties["current_mode"])))
        current_row = QWidget()
        current_row_layout = QHBoxLayout(current_row)
        current_row_layout.setContentsMargins(0, 0, 0, 0)
        current_row_layout.setSpacing(6)
        current_row_layout.addWidget(drive, 1)
        current_row_layout.addWidget(mode, 0)
        magnetic_form.addRow("Current", current_row)
        self.coil_widgets["drive_current_a"] = drive
        self.coil_widgets["current_mode"] = mode

        field_scale = _number_spin(
            properties["field_scale_factor"],
            decimals=6,
            minimum=0.000001,
            maximum=1_000_000.0,
            suffix=" ×",
        )
        field_scale.setSingleStep(0.05)
        field_scale.setToolTip(
            "Dimensionless calibration multiplier applied to this coil's field source. "
            "Use 1.0 for the ordinary air-core calculation."
        )
        fit_measurements = QPushButton("Data")
        fit_measurements.setObjectName("fitCoilFromMeasurementsButton")
        fit_measurements.setMaximumWidth(125)
        fit_measurements.setToolTip(
            "Open measurement calibration with this coil preselected."
        )
        fit_measurements.clicked.connect(
            lambda _checked=False, coil_id=object_id: self.open_measurement_calibration([coil_id])
        )
        correction_row = QWidget()
        correction_layout = QHBoxLayout(correction_row)
        correction_layout.setContentsMargins(0, 0, 0, 0)
        correction_layout.setSpacing(6)
        correction_layout.addWidget(field_scale, 1)
        correction_layout.addWidget(fit_measurements, 0)
        magnetic_form.addRow("Correction", correction_row)
        self.coil_widgets["field_scale_factor"] = field_scale

        magnetic_note = QLabel(
            "Magnetics defines the source used by the field solver. Construction and "
            "assembly below are optional physical/electrical descriptions and do not "
            "replace the magnetic current path."
        )
        magnetic_note.setWordWrap(True)
        magnetic_note.setObjectName("hint")
        magnetic_layout.addWidget(magnetic_note)
        magnetic_layout.addWidget(
            self._inline_apply_button("Apply Magnetics"),
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        self.inspector_layout.addWidget(magnetics)

        # --- Optional wire/construction description ---------------------------------
        construction = self._inspector_section("Coil Construction", expanded=False)
        construction_layout = QVBoxLayout()
        construction_layout.setSpacing(8)
        construction.setContentLayout(construction_layout)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        construction_layout.addLayout(form)

        awg = QComboBox()
        for gauge in range(4, 51):
            awg.addItem(f"{gauge} AWG", gauge)
        awg.setCurrentIndex(max(0, awg.findData(int(properties["awg"]))))
        form.addRow("Magnet wire", awg)
        self.coil_widgets["awg"] = awg

        wire_resistance = _number_spin(
            properties["resistance_20_ohm_per_km"],
            decimals=4,
            minimum=0.0001,
            maximum=1e9,
            suffix=" Ω/km",
        )
        wire_resistance.setSingleStep(0.1)
        wire_resistance.setToolTip(
            "DC resistance per kilometre at 20 °C. Changing AWG loads the nominal "
            "annealed-copper value; edit this field for measured/manufacturer wire."
        )
        form.addRow("Resistance @ 20 °C", wire_resistance)
        self.coil_widgets["resistance_20_ohm_per_km"] = wire_resistance

        leads = _number_spin(
            properties["lead_length_m"] * 1000.0,
            decimals=2,
            minimum=0.0,
            maximum=1e9,
            suffix=" mm",
        )
        form.addRow("Lead allowance", leads)
        self.coil_widgets["lead_length_m"] = leads

        enamel = _number_spin(
            properties["enamel_radial_mm"],
            decimals=4,
            minimum=0.0,
            maximum=2.0,
            suffix=" mm/side",
        )
        enamel.setSingleStep(0.005)
        form.addRow("Enamel build", enamel)
        self.coil_widgets["enamel_radial_mm"] = enamel

        packing = _number_spin(
            properties["packing_factor"],
            decimals=3,
            minimum=0.1,
            maximum=0.95,
        )
        packing.setSingleStep(0.01)
        form.addRow("Packing factor", packing)
        self.coil_widgets["packing_factor"] = packing

        temperature = _number_spin(
            properties["temperature_c"],
            decimals=1,
            minimum=-100.0,
            maximum=300.0,
            suffix=" °C",
        )
        form.addRow("Winding temperature", temperature)
        self.coil_widgets["temperature_c"] = temperature

        construction_note = QLabel(
            "Wire data drives resistance, copper mass, and automatic bundle/build estimates. "
            "It is optional metadata for magnetic calculations."
        )
        construction_note.setWordWrap(True)
        construction_note.setObjectName("hint")
        construction_layout.addWidget(construction_note)
        construction_layout.addWidget(
            self._inline_apply_button("Apply Coil Construction"),
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        self.inspector_layout.addWidget(construction)

        # --- Optional physical assembly ----------------------------------------------
        assembly = self._inspector_section("Assembly", expanded=False)
        assembly_layout = QVBoxLayout()
        assembly_layout.setSpacing(8)
        assembly.setContentLayout(assembly_layout)
        assembly_enable_form = QFormLayout()
        assembly_enable_form.setContentsMargins(0, 0, 0, 0)
        assembly_layout.addLayout(assembly_enable_form)

        enable_assembly = QCheckBox()
        enable_assembly.setChecked(bool(properties["physical_geometry_enabled"]))
        enable_assembly.setToolTip(
            "Enable construction-only winding, bobbin and flange geometry. This envelope "
            "is used by display and DRC but does not change the magnetic source."
        )
        assembly_enable_form.addRow("Enable physical assembly", enable_assembly)
        self.coil_envelope_group = enable_assembly
        self.coil_widgets["physical_geometry_enabled"] = enable_assembly

        assembly_details = QWidget()
        envelope_form = QFormLayout(assembly_details)
        envelope_form.setContentsMargins(0, 0, 0, 0)
        assembly_layout.addWidget(assembly_details)

        radial_build_mode = QComboBox()
        radial_build_mode.addItem("Automatic from winding", "auto")
        radial_build_mode.addItem("Manual override", "manual")
        radial_build_mode.setCurrentIndex(
            max(0, radial_build_mode.findData(properties["winding_radial_build_mode"]))
        )
        envelope_form.addRow("Radial build", radial_build_mode)
        self.coil_widgets["winding_radial_build_mode"] = radial_build_mode

        displayed_radial_build = properties["winding_radial_build_mm"]
        if properties["winding_radial_build_mode"] == "auto":
            displayed_radial_build = properties["calculations"]["auto_winding_radial_build_mm"]
        radial_build = _number_spin(
            displayed_radial_build,
            decimals=3,
            minimum=0.0,
            maximum=1e6,
            suffix=" mm",
        )
        radial_build.setEnabled(properties["winding_radial_build_mode"] == "manual")
        envelope_form.addRow("Calculated / manual build", radial_build)
        self.coil_widgets["winding_radial_build_mm"] = radial_build

        axial_width = _number_spin(
            properties["winding_axial_width_mm"],
            decimals=3,
            minimum=0.0,
            maximum=1e6,
            suffix=" mm",
        )
        envelope_form.addRow("Winding axial width", axial_width)
        self.coil_widgets["winding_axial_width_mm"] = axial_width

        bobbin_wall = _number_spin(
            properties["bobbin_wall_thickness_mm"],
            decimals=3,
            minimum=0.0,
            maximum=1e6,
            suffix=" mm",
        )
        envelope_form.addRow("Bobbin wall inward", bobbin_wall)
        self.coil_widgets["bobbin_wall_thickness_mm"] = bobbin_wall

        flange_height = _number_spin(
            properties["flange_height_mm"],
            decimals=3,
            minimum=0.0,
            maximum=1e6,
            suffix=" mm",
        )
        envelope_form.addRow("Flange height", flange_height)
        self.coil_widgets["flange_height_mm"] = flange_height

        flange_thickness = _number_spin(
            properties["flange_thickness_mm"],
            decimals=3,
            minimum=0.0,
            maximum=1e6,
            suffix=" mm each",
        )
        envelope_form.addRow("Flange thickness", flange_thickness)
        self.coil_widgets["flange_thickness_mm"] = flange_thickness

        additional_mass = _number_spin(
            properties["additional_mass_g"],
            decimals=2,
            minimum=0.0,
            maximum=1e9,
            suffix=" g",
        )
        envelope_form.addRow("Former / hardware", additional_mass)
        self.coil_widgets["additional_mass_g"] = additional_mass

        assembly_note = QLabel(
            "The physical assembly is electrically inert. Automatic build uses turns, "
            "finished wire diameter, packing factor, and axial winding width."
        )
        assembly_note.setWordWrap(True)
        assembly_note.setObjectName("hint")
        envelope_form.addRow(assembly_note)
        assembly_details.setVisible(enable_assembly.isChecked())
        enable_assembly.toggled.connect(assembly_details.setVisible)
        assembly_layout.addWidget(
            self._inline_apply_button("Apply Assembly"),
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        self.inspector_layout.addWidget(assembly)

        # --- Read-only estimates ------------------------------------------------------
        estimates = self._inspector_section("Estimates", expanded=False)
        estimate_form = QFormLayout()
        estimates.setContentLayout(estimate_form)
        estimate_rows = [
            ("turns", "Turns"),
            ("wire", "Wire diameter"),
            ("length", "Copper length"),
            ("mass", "Copper mass"),
            ("total_mass", "Total coil mass"),
            ("resistance", "Resistance"),
            ("ampere_turns", "Magnetic drive"),
            ("effective_drive", "Corrected magnetic drive"),
            ("voltage", "Resistive voltage"),
            ("loss", "Copper loss"),
            ("current_density", "Current density"),
            ("bundle", "Est. square bundle"),
        ]
        object_type = self.adapter.get_object(object_id)["type"]
        if object_type == "current.Circle":
            estimate_rows.insert(0, ("radius", "Mean radius (reference)"))
            estimate_rows.extend(
                [
                    ("bobbin_surface", "Bobbin winding-surface diameter"),
                    ("radial_build", "Effective winding radial build"),
                    ("bobbin_opening", "Bobbin central opening"),
                    ("winding_od", "Finished winding outside diameter"),
                    ("flange_od", "Flange outside diameter"),
                    ("assembly_bbox", "Assembly bounding box (local X × Y × Z)"),
                    ("assembly_status", "Assembly geometry check"),
                ]
            )
        elif object_type == "current.Polyline":
            estimate_rows.extend(
                [
                    ("radial_build", "Effective winding radial build"),
                    ("assembly_bbox", "Assembly bounding box (local X × Y × Z)"),
                    ("assembly_status", "Assembly geometry check"),
                ]
            )
        selectable = (
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        for key, label in estimate_rows:
            name = QLabel(label)
            name.setTextInteractionFlags(selectable)
            value = QLabel()
            value.setWordWrap(True)
            value.setTextInteractionFlags(selectable)
            self.coil_estimate_row_names[key] = label
            self.coil_estimate_labels[key] = value
            estimate_form.addRow(name, value)
        estimate_note = QLabel(
            "Length uses the present magnetic path plus lead allowance. Voltage is the "
            "resistive component only; an AC supply may also require inductive headroom."
        )
        estimate_note.setWordWrap(True)
        estimate_note.setObjectName("hint")
        estimate_note.setTextInteractionFlags(selectable)
        estimate_form.addRow(estimate_note)
        copy_button = QPushButton("Copy estimates")
        copy_button.setObjectName("copyCoilEstimatesButton")
        copy_button.clicked.connect(self._copy_coil_estimates)
        estimate_form.addRow(copy_button)
        self.coil_estimate_copy_button = copy_button
        self.inspector_layout.addWidget(estimates)

        for widget in (
            drive,
            turns,
            field_scale,
            wire_resistance,
            leads,
            enamel,
            packing,
            temperature,
            additional_mass,
        ):
            widget.valueChanged.connect(self._update_coil_estimates_preview)
        for name in (
            "racetrack_end_diameter_mm",
            "racetrack_straight_length_mm",
            "racetrack_arc_segments",
            "winding_radial_build_mm",
            "winding_axial_width_mm",
            "bobbin_wall_thickness_mm",
            "flange_height_mm",
            "flange_thickness_mm",
        ):
            widget = self.coil_widgets.get(name)
            if widget is not None:
                widget.valueChanged.connect(self._update_coil_estimates_preview)
        if self.coil_envelope_group is not None:
            self.coil_envelope_group.toggled.connect(self._update_coil_estimates_preview)
        radial_build_mode = self.coil_widgets.get("winding_radial_build_mode")
        if radial_build_mode is not None:
            radial_build_mode.currentIndexChanged.connect(self._coil_radial_build_mode_changed)
        awg.currentIndexChanged.connect(self._coil_awg_changed)
        mode.currentIndexChanged.connect(self._update_coil_estimates_preview)
        self._update_coil_estimates_preview()

    def _coil_radial_build_mode_changed(self, *_args) -> None:
        """Toggle the manual editor while keeping the live derived value visible."""
        mode = self.coil_widgets.get("winding_radial_build_mode")
        radial_build = self.coil_widgets.get("winding_radial_build_mm")
        if mode is None or radial_build is None:
            return
        radial_build.setEnabled(mode.currentData() == "manual")
        self._update_coil_estimates_preview()

    def _coil_awg_changed(self, *_args) -> None:
        """Load the selected gauge's nominal resistance without locking the field."""
        awg = self.coil_widgets.get("awg")
        wire_resistance = self.coil_widgets.get("resistance_20_ohm_per_km")
        if awg is None or wire_resistance is None or awg.currentData() is None:
            return
        blocker = QSignalBlocker(wire_resistance)
        wire_resistance.setValue(awg_resistance_20_ohm_per_km(int(awg.currentData())))
        del blocker
        self._update_coil_estimates_preview()

    def _coil_input_values(self) -> dict[str, Any]:
        if not self.coil_widgets:
            return {}
        values = {
            "enabled": self.coil_widgets["enabled"].isChecked(),
            "drive_current_a": self.coil_widgets["drive_current_a"].value() / 1000.0,
            "turns": self.coil_widgets["turns"].value(),
            "field_scale_factor": self.coil_widgets["field_scale_factor"].value(),
            "awg": self.coil_widgets["awg"].currentData(),
            "resistance_20_ohm_per_km": self.coil_widgets[
                "resistance_20_ohm_per_km"
            ].value(),
            "lead_length_m": self.coil_widgets["lead_length_m"].value() / 1000.0,
            "enamel_radial_mm": self.coil_widgets["enamel_radial_mm"].value(),
            "packing_factor": self.coil_widgets["packing_factor"].value(),
            "temperature_c": self.coil_widgets["temperature_c"].value(),
            "additional_mass_g": self.coil_widgets["additional_mass_g"].value(),
            "current_mode": self.coil_widgets["current_mode"].currentData(),
        }
        if self.coil_object_id:
            try:
                values["coil_shape"] = self.adapter.coil_physical_properties(self.coil_object_id).get("coil_shape", "generic")
            except Exception:
                values["coil_shape"] = "generic"
        envelope = self.coil_widgets.get("physical_geometry_enabled")
        values["physical_geometry_enabled"] = bool(
            envelope.isChecked() if envelope is not None else False
        )
        radial_build_mode = self.coil_widgets.get("winding_radial_build_mode")
        values["winding_radial_build_mode"] = (
            radial_build_mode.currentData()
            if radial_build_mode is not None
            else "auto"
        )
        for name in (
            "racetrack_end_diameter_mm",
            "racetrack_straight_length_mm",
            "racetrack_arc_segments",
            "winding_radial_build_mm",
            "winding_axial_width_mm",
            "bobbin_wall_thickness_mm",
            "flange_height_mm",
            "flange_thickness_mm",
        ):
            widget = self.coil_widgets.get(name)
            if widget is not None:
                values[name] = widget.value()
        return values

    def _update_mesh_dimensions_preview(self, *_args) -> None:
        if not self.mesh_dimension_labels or self.mesh_source_dimensions_mm is None:
            return
        scales: list[float] = []
        for name in ("scale_x", "scale_y", "scale_z"):
            info = self.parameter_editors.get(name)
            if info is None:
                return
            scales.append(float(info["widget"].value()))

        def dimensions_text(values: list[float]) -> str:
            return " × ".join(f"{value:,.6g}" for value in values) + " mm"

        current = [
            dimension * scale
            for dimension, scale in zip(
                self.mesh_source_dimensions_mm, scales, strict=True
            )
        ]
        self.mesh_dimension_labels["original"].setText(
            dimensions_text(self.mesh_source_dimensions_mm)
        )
        self.mesh_dimension_labels["current"].setText(dimensions_text(current))

    @staticmethod
    def _compact_value(value: float, unit: str) -> str:
        return f"{value:,.5g} {unit}".strip()

    def _copy_coil_estimates(self) -> None:
        """Copy the complete currently displayed estimate block as plain text."""
        if not self.coil_object_id or not self.coil_estimate_labels:
            return
        try:
            coil_name = self.adapter.get_object(self.coil_object_id)["label"]
        except KeyError:
            coil_name = self.coil_object_id
        lines = [f"Calculated winding estimates — {coil_name}"]
        for key, label in self.coil_estimate_row_names.items():
            value = self.coil_estimate_labels[key].text()
            lines.append(f"{label}: {value}")
        QApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage("Winding estimates copied", 2500)

    def _update_coil_estimates_preview(self, *_args) -> None:
        if not self.coil_object_id or not self.coil_estimate_labels:
            return
        try:
            properties = self.adapter.coil_physical_properties(
                self.coil_object_id, self._coil_input_values()
            )
            calc = properties["calculations"]
            radial_build_widget = self.coil_widgets.get("winding_radial_build_mm")
            if (
                radial_build_widget is not None
                and properties["winding_radial_build_mode"] == "auto"
            ):
                blocker = QSignalBlocker(radial_build_widget)
                radial_build_widget.setValue(
                    calc["auto_winding_radial_build_mm"]
                )
                del blocker
            mode_word = "peak" if properties["current_mode"] == "peak_sine" else "RMS/DC"
            values = {
                "turns": f"{calc['turns']:,}",
                "wire": (
                    f"{calc['bare_diameter_mm']:.4g} mm bare / "
                    f"{calc['outer_diameter_mm']:.4g} mm finished"
                ),
                "length": self._compact_value(calc["wire_length_m"], "m"),
                "mass": self._compact_value(calc["copper_mass_g"], "g"),
                "total_mass": self._compact_value(calc["total_mass_g"], "g"),
                "resistance": (
                    f"{calc['resistance_20_ohm']:.5g} Ω @ 20 °C / "
                    f"{calc['resistance_operating_ohm']:.5g} Ω operating"
                ),
                "ampere_turns": self._compact_value(calc["ampere_turns"], "A·turn"),
                "effective_drive": self._compact_value(
                    calc["effective_ampere_turns"], "effective A·turn"
                ),
                "voltage": f"{calc['resistive_voltage_v']:.5g} V {mode_word}",
                "loss": self._compact_value(calc["copper_loss_w"], "W"),
                "current_density": self._compact_value(
                    calc["current_density_a_mm2"], "A/mm² RMS"
                ),
                "bundle": self._compact_value(calc["estimated_bundle_side_mm"], "mm × mm"),
            }
            if "radius" in self.coil_estimate_labels:
                values["radius"] = self._compact_value(
                    calc["mean_turn_length_m"] * 1000.0 / (2.0 * math.pi),
                    "mm",
                )
            assembly = calc.get("assembly")
            if "assembly_bbox" in self.coil_estimate_labels:
                if not assembly:
                    for key in (
                        "bobbin_surface",
                        "radial_build",
                        "bobbin_opening",
                        "winding_od",
                        "flange_od",
                        "assembly_bbox",
                    ):
                        if key in self.coil_estimate_labels:
                            values[key] = "Not defined (optional)"
                    values["assembly_status"] = "Magnetic calculations only"
                elif not assembly.get("supported", False):
                    message = "; ".join(assembly.get("warnings", [])) or "Unsupported"
                    for key in (
                        "bobbin_surface",
                        "radial_build",
                        "bobbin_opening",
                        "winding_od",
                        "flange_od",
                        "assembly_bbox",
                    ):
                        if key in self.coil_estimate_labels:
                            values[key] = "—"
                    values["assembly_status"] = message
                else:
                    bbox = assembly["bounding_box_mm"]
                    warnings = list(assembly.get("errors", [])) + list(
                        assembly.get("warnings", [])
                    )
                    values["radial_build"] = (
                        self._compact_value(
                            assembly["winding_radial_build_mm"], "mm"
                        )
                        + (
                            " (automatic)"
                            if assembly["winding_radial_build_mode"] == "auto"
                            else " (manual)"
                        )
                    )
                    values["assembly_bbox"] = (
                        f"{bbox[0]:,.5g} × {bbox[1]:,.5g} × {bbox[2]:,.5g} mm"
                    )
                    values["assembly_status"] = (
                        "Ready for visual + DRC"
                        if not warnings
                        else "Warning: " + "; ".join(warnings)
                    )
                    if assembly.get("shape") == "annular_cylinder":
                        values.update(
                            {
                                "bobbin_surface": self._compact_value(
                                    assembly["winding_surface_diameter_mm"], "mm"
                                ),
                                "bobbin_opening": self._compact_value(
                                    assembly["bobbin_opening_diameter_mm"], "mm"
                                ),
                                "winding_od": self._compact_value(
                                    assembly["winding_outer_diameter_mm"], "mm"
                                ),
                                "flange_od": self._compact_value(
                                    assembly["flange_outer_diameter_mm"], "mm"
                                ),
                            }
                        )
            for key, label in self.coil_estimate_labels.items():
                label.setText(values[key])
        except Exception as error:  # noqa: BLE001 - live preview boundary
            for label in self.coil_estimate_labels.values():
                label.setText(str(error))

    @staticmethod
    def _display_unit(unit: str) -> str:
        return {"m": "mm", "A": "mA"}.get(unit, unit)

    def _parameter_editor(self, param: dict[str, Any]):
        unit = param.get("unit", "")
        factor = 1000.0 if unit in {"m", "A"} else 1.0
        value = scale_nested(param["value"], factor)
        if param["kind"] == "scalar":
            suffix = f" {self._display_unit(unit)}" if unit else ""
            return _number_spin(
                value,
                decimals=6,
                minimum=float(param.get("minimum", -1e12)),
                maximum=float(param.get("maximum", 1e12)),
                suffix=suffix,
            ), factor
        text = json.dumps(value, separators=(", ", ": "))
        if param["kind"] == "matrix":
            editor = QPlainTextEdit(text)
            editor.setMaximumHeight(110)
        else:
            editor = QLineEdit(text)
        return editor, factor

    def apply_inspector(self) -> bool:
        if not self.selected_id:
            return False
        operations: list[dict[str, Any]] = []
        label = self.label_edit.text().strip()
        if not label:
            QMessageBox.warning(self, "Object name", "Please enter a name.")
            return False
        operations.append(
            {"method": "apply_edit", "params": {"object_id": self.selected_id, "path": "label", "value": label}}
        )

        if self.geometry_role_combo is not None:
            operations.append(
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": self.selected_id,
                        "role": self.geometry_role_combo.currentData(),
                        "colour": (
                            str(self.geometry_colour_button.property("geometry_colour") or "#64748b")
                            if isinstance(self.geometry_colour_button, QPushButton)
                            else "#64748b"
                        ),
                        "opacity": self.geometry_opacity_spin.value(),
                    },
                }
            )
            operations.append(
                {
                    "method": "set_drc_clearance",
                    "params": {
                        "object_id": self.selected_id,
                        "enabled": (
                            self.geometry_drc_enabled_check.isChecked()
                            if self.geometry_drc_enabled_check is not None
                            and self.geometry_drc_enabled_check.isEnabled()
                            else False
                        ),
                        "clearance_override_mm": (
                            self.geometry_clearance_spin.value()
                            if self.geometry_clearance_override_check is not None
                            and self.geometry_clearance_override_check.isChecked()
                            else None
                        ),
                    },
                }
            )

        if self.surface_region_editors:
            region_settings: dict[str, dict[str, Any]] = {}
            for key, editor in self.surface_region_editors.items():
                item = editor.get("item")
                colour_button = editor.get("colour_button")
                override_check = editor.get("clearance_override_check")
                clearance_spin = editor.get("clearance_spin")
                if not isinstance(item, QTreeWidgetItem):
                    continue
                colour = (
                    str(colour_button.property("surface_region_colour") or "#64748b")
                    if isinstance(colour_button, QPushButton)
                    else "#64748b"
                )
                clearance_override_mm = None
                if (
                    isinstance(override_check, QCheckBox)
                    and override_check.isChecked()
                    and isinstance(clearance_spin, QDoubleSpinBox)
                ):
                    clearance_override_mm = clearance_spin.value()
                region_settings[str(key)] = {
                    "active": item.checkState(2) == Qt.CheckState.Checked,
                    "visible": item.checkState(3) == Qt.CheckState.Checked,
                    "placement": item.checkState(4) == Qt.CheckState.Checked,
                    "drc_exclusion": item.checkState(5) == Qt.CheckState.Checked,
                    "clearance_override_mm": clearance_override_mm,
                    "colour": colour,
                }
            operations.append(
                {
                    "method": "set_surface_region_settings",
                    "params": {
                        "object_id": self.selected_id,
                        "settings": region_settings,
                    },
                }
            )

        try:
            if self.measurement_quality_combo is not None:
                sampling_choice = str(self.measurement_quality_combo.currentData())
                sampling_mode = (
                    "user_defined" if sampling_choice == "user_defined" else "generated"
                )
                quality = (
                    self.adapter.measurement_properties(self.selected_id)["quality"]
                    if sampling_mode == "user_defined"
                    else sampling_choice
                )
                defined_points_mm = self._parse_defined_measurement_points(
                    self.measurement_defined_points_edit.toPlainText()
                    if self.measurement_defined_points_edit is not None
                    else ""
                )
                operations.append(
                    {
                        "method": "set_measurement_settings",
                        "params": {
                            "object_id": self.selected_id,
                            "sampling_mode": sampling_mode,
                            "quality": quality,
                            "defined_points_mm": defined_points_mm,
                            "target_uT": self.measurement_target_spin.value(),
                            "tolerance_pct": self.measurement_tolerance_spin.value(),
                        },
                    }
                )

            for name, info in self.parameter_editors.items():
                widget = info["widget"]
                if info["kind"] == "scalar":
                    display_value = widget.value()
                else:
                    text = widget.toPlainText() if isinstance(widget, QPlainTextEdit) else widget.text()
                    display_value = json.loads(text)
                value = scale_nested(display_value, 1.0 / info["factor"])
                operations.append(
                    {"method": "set_param", "params": {"object_id": self.selected_id, "name": name, "value": value}}
                )

            if self.coil_object_id == self.selected_id:
                operations.append(
                    {
                        "method": "set_coil_physical",
                        "params": {
                            "object_id": self.selected_id,
                            **self._coil_input_values(),
                        },
                    }
                )

            if self.axis_path_editable:
                start = [spin.value() / 1000.0 for spin in self.axis_start_spins]
                end = [spin.value() / 1000.0 for spin in self.axis_end_spins]
                samples = self.axis_samples_spin.value()
                path = [
                    [
                        start[axis] + (end[axis] - start[axis]) * index / (samples - 1)
                        for axis in range(3)
                    ]
                    for index in range(samples)
                ]
                operations.append(
                    {
                        "method": "set_transform",
                        "params": {"object_id": self.selected_id, "position": path},
                    }
                )
            elif self.transform_editable:
                position = [spin.value() / 1000.0 for spin in self.position_spins]
                euler = [spin.value() for spin in self.rotation_spins]
                new_rotation = Rotation.from_euler("xyz", euler, degrees=True)
                rotvec = new_rotation.as_rotvec(degrees=True).tolist()
                operations.append(
                    {
                        "method": "set_transform",
                        "params": {"object_id": self.selected_id, "position": position, "orientation": rotvec},
                    }
                )
            self.adapter.apply_operations(operations)
        except (ValueError, json.JSONDecodeError, StudioOperationError) as error:
            QMessageBox.warning(self, "Could not apply changes", str(error))
            return False
        self.refresh_scene(select_id=self.selected_id)
        return True

    # --- object operations --------------------------------------------
    def _editable_selected_ids(self) -> list[str]:
        editable: list[str] = []
        for object_id in self.selected_ids:
            try:
                if not self.adapter.get_object(object_id).get("derived"):
                    editable.append(object_id)
            except KeyError:
                continue
        return editable

    def _selected_group(self) -> str | None:
        parent = None
        if self.selected_id and len(self.selected_ids) == 1:
            try:
                selected = self.adapter.get_object(self.selected_id)
                if selected["type"] == "Collection" and not selected.get("derived"):
                    parent = self.selected_id
            except KeyError:
                pass
        return parent

    def group_selected(self) -> None:
        object_ids = self._editable_selected_ids()
        if len(object_ids) < 2:
            self.statusBar().showMessage("Select at least two objects to group.", 3000)
            return
        try:
            group_id = self.adapter.group_objects(object_ids)
            self.refresh_scene(select_id=group_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Group objects", str(error))

    def add_empty_group(self) -> None:
        """Create a small organizational container at the selected hierarchy level."""
        parent = self._selected_group()
        try:
            group_id = self.adapter.add_template("collection", parent=parent)
            self.refresh_scene(select_id=group_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "New group", str(error))

    def ungroup_selected(self) -> None:
        if self.selected_id is None or len(self.selected_ids) != 1:
            return
        try:
            children = self.adapter.ungroup(self.selected_id)
            self.selected_id = None
            self.selected_ids = []
            self.refresh_scene(select_ids=children)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Ungroup", str(error))

    def add_object(self, template: str) -> None:
        if template == "import_mesh":
            self.import_mesh()
            return
        parent = self._selected_group()
        if template == "magnetometer_probe":
            dialog = ProbeLibraryDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            try:
                object_id = self.adapter.add_magnetometer_probe(
                    dialog.probe_definition(), parent=parent
                )
                self.plot.request_home_on_next_figure()
                self.refresh_scene(select_id=object_id)
            except Exception as error:  # noqa: BLE001
                QMessageBox.warning(self, "Magnetometer probe", str(error))
            return
        if template == "background_field":
            dialog = BackgroundFieldDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            try:
                definition = dialog.field_definition()
                object_id = self.adapter.add_background_field(parent=parent, **definition)
                self.refresh_scene(select_id=object_id)
            except Exception as error:  # noqa: BLE001
                QMessageBox.warning(self, "Background field", str(error))
            return
        try:
            object_id = self.adapter.add_template(template, parent=parent)
            self.plot.request_home_on_next_figure()
            self.refresh_scene(select_id=object_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Add object", str(error))

    def apply_background_field_name(self) -> None:
        if not self.selected_id:
            return
        label = self.label_edit.text().strip()
        if not label:
            QMessageBox.warning(self, "Object name", "Please enter a name.")
            return
        try:
            operations = [
                {"method": "apply_edit", "params": {"object_id": self.selected_id, "path": "label", "value": label}}
            ]
            position = list(
                self.adapter.get_transform(self.selected_id).get(
                    "position", [0.0, 0.0, 0.0]
                )
            )
            if hasattr(self, "background_position_spins"):
                position = [spin.value() / 1000.0 for spin in self.background_position_spins]
                operations.append(
                    {
                        "method": "set_transform",
                        "params": {"object_id": self.selected_id, "position": position},
                    }
                )
            self.adapter.apply_operations(operations)
            selected = self.selected_id
            self.refresh_scene(select_id=selected)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Background field", str(error))

    def edit_background_field(self) -> None:
        if not self.selected_id or not self.adapter.is_background_field(self.selected_id):
            return
        object_id = self.selected_id
        dialog = BackgroundFieldDialog(
            self, initial=self.adapter.background_field_properties(object_id)
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.adapter.update_background_field(object_id, **dialog.field_definition())
            self.refresh_scene(select_id=object_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Background field", str(error))

    def import_mesh(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import object",
            "",
            "Mesh files (*.stl *.obj);;STL files (*.stl);;OBJ files (*.obj);;All files (*)",
        )
        if not filename:
            return
        options = MeshImportOptionsDialog(filename, self)
        if options.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            object_id = self.adapter.import_mesh(
                filename,
                unit_scale=options.unit_scale(),
                parent=self._selected_group(),
                center=options.should_center(),
            )
            self.plot.request_home_on_next_figure()
            self.refresh_scene(select_id=object_id)
            region_map = self.adapter.surface_region_properties(object_id)
            if region_map:
                count = len(region_map.get("regions", {}))
                mapped = int(region_map.get("validation", {}).get("unique_assigned_faces", 0))
                total = int(region_map.get("topology", {}).get("face_count", 0))
                centered = " and centered" if options.should_center() else ""
                self.statusBar().showMessage(
                    f"Imported{centered} {Path(filename).name} with {count} named surface "
                    f"region(s) ({mapped:,}/{total:,} triangles mapped)",
                    6500,
                )
            elif options.should_center():
                self.statusBar().showMessage(
                    f"Imported and centered {Path(filename).name}", 4500
                )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Import geometry", str(error))

    def import_upenn_gbm_case(self) -> None:
        dialog = UpennGbmImportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result = dialog.import_result()
            object_ids = self.adapter.add_upenn_gbm_case(
                result,
                role=dialog.requested_role(),
                parent=self._selected_group(),
            )
            self.plot.request_home_on_next_figure()
            self.refresh_scene(
                select_id=object_ids[-1] if object_ids else None,
                select_ids=object_ids,
            )
            case_id = str(result.get("case_id", "UPENN-GBM case"))
            noun = "region" if len(object_ids) == 1 else "regions"
            self.statusBar().showMessage(
                f"Added {case_id}: {len(object_ids)} tumor {noun} in the canonical head frame",
                6500,
            )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "UPENN-GBM import", str(error))

    def reparent_object(self, object_id: str, parent_id: str | None) -> None:
        try:
            self.adapter.move_object(object_id, parent_id)
            self.refresh_scene(select_id=object_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Move object", str(error))
            self.refresh_scene(select_id=object_id, rebuild_plot=False)

    def _reconcile_scene_tree_after_drag(self) -> None:
        """Rebuild rows from the scene model after any completed/cancelled drag."""
        self.refresh_scene(
            select_ids=list(self.selected_ids),
            refresh_plot=False,
            rebuild_plot=False,
        )

    def duplicate_selected(self) -> None:
        if not self.selected_id:
            return
        try:
            object_id = self.adapter.duplicate(self.selected_id)
            # Cloning changes the 3D trace topology even when the new object is
            # geometrically coincident with its source. Plotly.react can reuse
            # stale scatter3d/WebGL state across that index change; on a planar
            # scene this can blank every coincident centreline even though the
            # render-extent helper is still present. Build the changed trace set
            # on a fresh plot surface, but preserve the current camera/ranges.
            self.plot.request_clean_rebuild_on_next_figure()
            # Structural scene changes should not reuse Plotly's previously
            # resolved axis ranges verbatim.  Recompute them from the new trace
            # payload while retaining the user's camera direction.  This is
            # especially important for coincident planar clones, where some
            # Plotly/WebGL builds report a collapsed live axis even though the
            # preceding frame still looked valid.
            self.plot.request_fit_bounds_on_next_figure()
            self.refresh_scene(select_id=object_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Clone object", str(error))

    def delete_selected(self) -> None:
        object_ids = self._editable_selected_ids()
        if not object_ids:
            return
        if len(object_ids) == 1:
            obj = self.adapter.get_object(object_ids[0])
            prompt = f"Delete “{obj['label']}”?"
            if obj["type"] == "Collection":
                prompt += " Its contents will also be removed."
            title = "Delete object"
        else:
            group_count = 0
            for object_id in object_ids:
                try:
                    group_count += int(self.adapter.get_object(object_id)["type"] == "Collection")
                except KeyError:
                    pass
            prompt = f"Delete {len(object_ids)} selected objects?"
            if group_count:
                prompt += " Contents of selected groups will also be removed."
            title = "Delete objects"
        answer = QMessageBox.question(self, title, prompt)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.adapter.remove_many(object_ids)
            self.selected_id = None
            self.selected_ids = []
            self.plot.request_home_on_next_figure()
            self.refresh_scene()
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, title, str(error))

    def set_selected_coils_enabled(self, enabled: bool) -> None:
        coil_ids = [
            object_id
            for object_id in self._editable_selected_ids()
            if self.adapter.is_coil(object_id)
        ]
        if not coil_ids:
            return
        selected_ids = list(self.selected_ids)
        primary_id = self.selected_id
        try:
            self.adapter.set_coils_enabled(coil_ids, bool(enabled))
            self.plot.request_home_on_next_figure()
            self.refresh_scene(
                select_id=primary_id,
                select_ids=selected_ids,
            )
            state = "Enabled" if enabled else "Disabled"
            noun = "coil" if len(coil_ids) == 1 else "coils"
            self.statusBar().showMessage(f"{state} {len(coil_ids)} {noun} magnetically", 3500)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Coil enable", str(error))

    def toggle_visibility(self) -> None:
        if not self.selected_id:
            return
        obj = self.adapter.get_object(self.selected_id)
        try:
            self.adapter.set_visible(self.selected_id, not obj.get("visible", True))
            self.plot.request_home_on_next_figure()
            self.refresh_scene(select_id=self.selected_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Visibility", str(error))

    def undo(self) -> None:
        self._commit_scene_notes_session()
        try:
            self.adapter.undo()
            self._sync_scene_notes_dialog()
            self.plot.request_home_on_next_figure()
            self.refresh_scene(select_id=self.selected_id)
        except StudioOperationError as error:
            self.statusBar().showMessage(str(error), 3000)

    def redo(self) -> None:
        self._commit_scene_notes_session()
        try:
            self.adapter.redo()
            self._sync_scene_notes_dialog()
            self.plot.request_home_on_next_figure()
            self.refresh_scene(select_id=self.selected_id)
        except StudioOperationError as error:
            self.statusBar().showMessage(str(error), 3000)

    # --- files ---------------------------------------------------------
    def _scene_filter(self) -> str:
        return f"Magpylib scene (*{SCENE_SUFFIX});;JSON files (*.json);;All files (*)"

    def _recent_files(self) -> list[str]:
        stored = self.settings.value(self.RECENT_FILES_KEY, [])
        if stored is None:
            return []
        if isinstance(stored, str):
            stored = [stored]
        result: list[str] = []
        seen_keys: set[str] = set()
        for raw_path in stored:
            path = str(raw_path).strip()
            if not path:
                continue
            key = self._scene_path_key(path)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            result.append(path)
        return result[: self.MAX_RECENT_FILES]

    @staticmethod
    def _scene_path_display(path: str | Path) -> str:
        """Return an absolute path while preserving platform display casing."""
        return str(Path(path).expanduser().resolve())

    @classmethod
    def _scene_path_key(cls, path: str | Path) -> str:
        """Return the comparison key used for recent-file identity.

        ``normcase`` intentionally folds case on Windows, where path identity is
        normally case-insensitive, but it must not be used as the display value:
        doing so turns ``C:\\Users`` into ``c:\\users`` in the Recent menu.
        """
        return os.path.normcase(cls._scene_path_display(path))

    def _record_recent_file(self, path: str | Path) -> None:
        display_path = self._scene_path_display(path)
        key = self._scene_path_key(display_path)
        files = [
            existing
            for existing in self._recent_files()
            if self._scene_path_key(existing) != key
        ]
        files.insert(0, display_path)
        self.settings.setValue(self.RECENT_FILES_KEY, files[: self.MAX_RECENT_FILES])
        self.settings.sync()
        self._refresh_recent_menu()

    def _remove_recent_file(self, path: str | Path) -> None:
        key = self._scene_path_key(path)
        files = [
            existing
            for existing in self._recent_files()
            if self._scene_path_key(existing) != key
        ]
        self.settings.setValue(self.RECENT_FILES_KEY, files)
        self.settings.sync()
        self._refresh_recent_menu()

    def clear_recent_files(self) -> None:
        self.settings.remove(self.RECENT_FILES_KEY)
        self.settings.sync()
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        files = self._recent_files()
        if not files:
            empty = self.recent_menu.addAction("(No recent files)")
            empty.setEnabled(False)
            return
        for index, filename in enumerate(files, start=1):
            path = Path(filename)
            parent = str(path.parent)
            label = f"&{index}  {path.name} — {parent}"
            action = self.recent_menu.addAction(label)
            action.setData(filename)
            action.setToolTip(filename)
            action.triggered.connect(
                lambda checked=False, recent_path=filename: self.open_recent_scene(
                    recent_path
                )
            )
        self.recent_menu.addSeparator()
        clear_action = self.recent_menu.addAction("Clear Recent Files")
        clear_action.triggered.connect(self.clear_recent_files)

    def new_default_scene(self) -> None:
        adapter = StudioAdapter()
        adapter.new_default()
        self._apply_saved_preferences_to_adapter(adapter)
        self._append_or_reuse_default_document(
            self._make_document(
                adapter, selected_id="coil", disposable_default=True
            )
        )

    def new_empty_scene(self) -> None:
        adapter = StudioAdapter()
        adapter.new_empty()
        self._apply_saved_preferences_to_adapter(adapter)
        self._append_or_reuse_default_document(self._make_document(adapter))

    # --- scene view tools ---------------------------------------------
    def home_scene_view(self) -> None:
        """Fit all visible geometry and restore a useful isometric camera."""
        if not self.plot.home_view():
            self.statusBar().showMessage("The 3D view is still loading.", 2500)

    def export_scene_image(self) -> None:
        suggested = f"{self.current_path.stem if self.current_path else 'field-workbench-scene'}.png"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export current scene view",
            suggested,
            "PNG image (*.png)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".png"):
            filename += ".png"
        self._pending_export_path = Path(filename)
        width = max(900, self.plot.width())
        height = max(650, self.plot.height())
        if not self.plot.request_png(width=width, height=height, scale=2.0):
            self._pending_export_path = None
            QMessageBox.warning(self, "Export image", "The 3D view is not ready yet.")

    def _scene_image_ready(self, data_url: str) -> None:
        path = self._pending_export_path
        self._pending_export_path = None
        if path is None:
            return
        try:
            marker = "base64,"
            if marker not in data_url:
                raise ValueError("The renderer returned an unsupported image format.")
            encoded = data_url.split(marker, 1)[1]
            path.write_bytes(base64.b64decode(encoded, validate=True))
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Export image", f"Could not save the scene image:\n{error}")
            return
        self.statusBar().showMessage(f"Exported {path.name}", 3500)

    def _scene_image_failed(self, detail: str) -> None:
        self._pending_export_path = None
        QMessageBox.warning(self, "Export image", f"Could not render the scene image:\n{detail}")

    def import_objects_from_scene(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import objects from saved scene",
            "",
            self._scene_filter(),
        )
        if not filename:
            return
        try:
            catalog = self.adapter.scene_object_catalog_from_file(filename)
        except Exception as error:  # noqa: BLE001 - external scene boundary
            QMessageBox.warning(self, "Import objects from scene", str(error))
            return

        dialog = SceneObjectImportDialog(catalog, Path(filename).name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected_ids = dialog.checked_ids()
        if not selected_ids:
            return
        try:
            imported_ids = self.adapter.import_objects_from_file(filename, selected_ids)
        except Exception as error:  # noqa: BLE001 - external scene boundary
            QMessageBox.warning(self, "Import objects from scene", str(error))
            return

        imported_set = set(imported_ids)
        top_level_ids = []
        for object_id in imported_ids:
            try:
                parent_id = str(self.adapter.get_object(object_id).get("parent") or "")
            except KeyError:
                continue
            if parent_id not in imported_set:
                top_level_ids.append(object_id)
        self.plot.request_home_on_next_figure()
        self.refresh_scene(
            select_id=top_level_ids[-1] if top_level_ids else None,
            select_ids=top_level_ids,
        )
        noun = "object" if len(imported_ids) == 1 else "objects"
        self.statusBar().showMessage(
            f"Imported {len(imported_ids)} {noun} from {Path(filename).name}",
            4500,
        )

    def open_scene(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open Magpylib scene", "", self._scene_filter())
        if not filename:
            return
        self._load_scene_file(filename)

    def open_recent_scene(self, filename: str) -> None:
        path = Path(filename)
        if not path.is_file():
            self._remove_recent_file(path)
            QMessageBox.warning(
                self,
                "Recent scene unavailable",
                f"This file no longer exists and was removed from Open Recent:\n{path}",
            )
            return
        self._load_scene_file(path)

    def _load_scene_file(
        self,
        filename: str | Path,
        *,
        replace_initial: bool = False,
    ) -> bool:
        path = Path(filename).expanduser().resolve()
        existing = self._document_for_path(path)
        if existing is not None:
            self._activate_document_id(existing.document_id)
            self.statusBar().showMessage(f"{path.name} is already open", 3500)
            return True
        adapter = StudioAdapter()
        try:
            adapter.load_file(path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open scene", str(error))
            return False
        self._apply_saved_preferences_to_adapter(adapter)
        document = self._make_document(adapter, path=path)
        self._record_recent_file(path)
        if replace_initial and len(self._documents) == 1:
            self._commit_scene_notes_session()
            self._close_document_windows(self.active_document.document_id)
            self._documents[0] = document
            self._active_document_index = 0
            self._rebuild_document_tabs()
            self._sync_scene_notes_dialog()
            self.plot.request_home_on_next_figure()
            self.refresh_scene()
        else:
            self._append_or_reuse_default_document(document)
        return True

    def save_scene(self) -> bool:
        return self._save_document(self.active_document)

    def _save_document(self, document: SceneDocument) -> bool:
        if document.path is None:
            return self._save_document_as(document)
        if document.document_id == self.active_document.document_id:
            self._commit_scene_notes_session()
        try:
            document.adapter.save_file(document.path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save scene", str(error))
            return False
        document.saved_signature = document.adapter.signature()
        self._record_recent_file(document.path)
        self._update_document_tab(document)
        if document.document_id == self.active_document.document_id:
            self._update_title()
        self.statusBar().showMessage(f"Saved {document.path.name}", 3500)
        return True

    def save_scene_as(self) -> bool:
        return self._save_document_as(self.active_document)

    def _save_document_as(self, document: SceneDocument) -> bool:
        suggested = document.path.name if document.path else f"untitled{SCENE_SUFFIX}"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Magpylib scene", suggested, self._scene_filter())
        if not filename:
            return False
        if not filename.lower().endswith(".json"):
            filename += SCENE_SUFFIX
        target = Path(filename).expanduser().resolve()
        existing = self._document_for_path(target)
        if existing is not None and existing.document_id != document.document_id:
            QMessageBox.warning(
                self,
                "Scene already open",
                "That file is already open in another tab. Close that tab or choose a different filename.",
            )
            return False
        previous_path = document.path
        document.path = target
        if self._save_document(document):
            return True
        document.path = previous_path
        self._update_document_tab(document)
        return False

    def export_scene(self) -> None:
        dialog = SceneExportDialog(
            self.adapter, self, preview_base_figure=self._base_figure
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        object_ids = dialog.selected_object_ids()
        if not object_ids:
            return
        translator = dialog.translator()
        suffix = translator_suffix(translator)
        document = self.active_document
        if document.path is not None:
            base_name = document.path.name
            if base_name.lower().endswith(SCENE_SUFFIX):
                base_name = base_name[: -len(SCENE_SUFFIX)]
            else:
                base_name = Path(base_name).stem
        else:
            base_name = document.display_name()
        suggested = f"{base_name}{suffix}"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export scene",
            suggested,
            translator_filter(translator),
        )
        if not filename:
            return
        if not filename.lower().endswith(suffix.lower()):
            filename += suffix
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path = write_export(
                self.adapter,
                object_ids,
                translator,
                filename,
                include_construction=dialog.include_construction(),
                apply_field_scale=dialog.apply_field_scale(),
                getdp_use_physical_construction=dialog.getdp_use_physical_construction(),
                getdp_mesh_preset=dialog.getdp_mesh_preset(),
                getdp_source_elements=dialog.getdp_source_elements(),
                getdp_near_radius_divisions=dialog.getdp_near_radius_divisions(),
                getdp_far_radius_divisions=dialog.getdp_far_radius_divisions(),
                femm_reference_slice_plane=dialog.femm_reference_slice_plane(),
                femm_reference_half=dialog.femm_reference_half(),
            )
        except Exception as error:  # noqa: BLE001 - target translator/file boundary
            QMessageBox.critical(self, "Could not export scene", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Exported {len(object_ids)} object{'s' if len(object_ids) != 1 else ''} to {path.name}",
            4500,
        )

    def is_dirty(self) -> bool:
        return self.active_document.is_dirty()

    def maybe_save(self) -> bool:
        return self.maybe_save_document(self.active_document)

    def maybe_save_document(self, document: SceneDocument) -> bool:
        if not document.is_dirty():
            return True
        answer = QMessageBox.warning(
            self,
            "Unsaved changes",
            f"Save changes to {document.display_name()}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save_document(document)
        return answer == QMessageBox.StandardButton.Discard

    def _update_title(self) -> None:
        document = self.active_document
        self._update_document_tab(document)
        self.setWindowTitle(f"{self._document_tab_text(document)} — {APP_NAME}")

    # --- analysis/help -------------------------------------------------
    def _select_drc_objects(self, object_ids: list[str]) -> None:
        available = [value for value in object_ids if value in self._tree_items]
        if not available:
            return
        self.refresh_scene(
            select_id=available[0],
            select_ids=available,
            rebuild_plot=False,
        )

    def _drc_settings_changed(self) -> None:
        self._update_title()
        self._update_actions()

    def _select_drc_objects_for_document(
        self,
        document_id: str,
        object_ids: list[str],
    ) -> None:
        self._activate_document_id(
            document_id,
            lambda: self._select_drc_objects(object_ids),
        )

    def _drc_settings_changed_for_document(self, document_id: str) -> None:
        document = self._document_by_id(document_id)
        if document is None:
            return
        self._update_document_tab(document)
        if document_id == self.active_document.document_id:
            self._drc_settings_changed()

    def _drc_dialog_finished(self, document_id: str, dialog_id: int) -> None:
        document = self._document_by_id(document_id)
        if (
            document is not None
            and document.drc_dialog is not None
            and id(document.drc_dialog) == dialog_id
        ):
            document.drc_dialog = None

    def open_drc(self) -> None:
        document = self.active_document
        if document.drc_dialog is None:
            dialog = DrcResultsDialog(
                document.adapter,
                self,
                on_select=lambda object_ids, document_id=document.document_id: (
                    self._select_drc_objects_for_document(document_id, object_ids)
                ),
                on_settings_changed=lambda document_id=document.document_id: (
                    self._drc_settings_changed_for_document(document_id)
                ),
            )
            dialog.setWindowTitle(
                f"Design Rule Check — {document.display_name()}"
            )
            document.drc_dialog = dialog
            dialog_id = id(dialog)
            dialog.finished.connect(
                lambda _result, document_id=document.document_id, key=dialog_id: (
                    self._drc_dialog_finished(document_id, key)
                )
            )
        document.drc_dialog.show()
        document.drc_dialog.raise_()
        document.drc_dialog.activateWindow()

    @staticmethod
    def _optimizer_finalist_tab_label(candidate: dict) -> str:
        candidate_id = str(candidate.get("id", "")).strip()
        rank = ""
        if "_" in candidate_id:
            tail = candidate_id.rsplit("_", 1)[-1]
            if tail.isdigit():
                rank = str(int(tail))
        if candidate_id.startswith("candidate_"):
            base = f"Optimizer solution {rank}" if rank else "Optimizer solution"
        else:
            base = f"Optimizer finalist {rank}" if rank else "Optimizer finalist"
        setting = str(candidate.get("setting_summary", "")).strip()
        if setting and len(setting) <= 42:
            return f"{base} — {setting}"
        return base

    def _optimizer_candidate_opened_new_tab(
        self, candidate: dict, source_document: dict
    ) -> None:
        try:
            source_signature = StudioAdapter(source_document).signature()
            candidate_adapter = self.adapter.optimizer_candidate_adapter(
                candidate, source_document=source_document
            )
            self._apply_saved_preferences_to_adapter(candidate_adapter)
        except Exception as error:  # noqa: BLE001 - optimizer tab boundary
            QMessageBox.warning(self, "Open optimizer finalist", str(error))
            return

        document = self._make_document(candidate_adapter)
        # A finalist is a new unsaved derivative of the source scene.  Keep it
        # visibly dirty so closing it cannot silently discard an optimization result.
        document.saved_signature = source_signature
        document.display_label = self._optimizer_finalist_tab_label(candidate)
        document.disposable_default = False
        self._append_document(document)
        self.statusBar().showMessage(
            f"Opened {document.display_name()} as a new scene tab", 5000
        )

    def _optimizer_candidate_applied(self, candidate: dict, replace: bool) -> None:
        """Backward-compatible active-document apply path."""
        self._optimizer_candidate_applied_to_document(
            self.active_document.document_id,
            self.adapter.signature(),
            candidate,
            replace,
        )

    def _optimizer_candidate_applied_to_document(
        self,
        document_id: str,
        source_signature: str,
        candidate: dict,
        replace: bool,
    ) -> None:
        """Apply a finalist to the scene that originally launched its optimizer.

        Optimizer windows are independent and may keep running while the user
        edits or switches to other tabs.  A finalist is therefore bound to its
        source document rather than whichever tab happens to be active when the
        button is clicked.  If that source changed after launch, make the stale
        relationship explicit before touching it.
        """
        document = self._document_by_id(document_id)
        if document is None:
            QMessageBox.warning(
                self,
                "Apply optimizer finalist",
                "The source scene for this optimizer is no longer open. Open the finalist in a new tab instead.",
            )
            return
        if document.adapter.signature() != source_signature:
            answer = QMessageBox.question(
                self,
                "Source scene changed",
                f"{document.display_name()} changed after this optimization started.\n\n"
                "The finalist was calculated from the frozen launch-time scene. Apply its optimized properties "
                "to the current version of the source scene anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            document.adapter.apply_optimizer_candidate(
                candidate, replace_circular_coils=replace
            )
        except Exception as error:  # noqa: BLE001 - optimizer apply boundary
            QMessageBox.warning(self, "Apply optimizer finalist", str(error))
            return
        document.analysis_overlay = None
        document.analysis_overlay_owner = None
        self._update_document_tab(document)
        if document.document_id == self.active_document.document_id:
            self.plot.request_home_on_next_figure()
            self.refresh_scene()
        if candidate.get("mode") == "tune_existing":
            self.statusBar().showMessage(
                f"Applied optimizer tuning result to {document.display_name()} "
                f"({len(candidate.get('coil_ids', []))} coil(s))", 6000
            )
        else:
            action = "Replaced circular coils with" if replace else "Added"
            self.statusBar().showMessage(
                f"{action} {len(candidate.get('coils', []))}-coil optimizer finalist in {document.display_name()}",
                6000,
            )

    def _optimizer_candidate_appended_to_document(
        self,
        document_id: str,
        source_signature: str,
        candidate: dict,
    ) -> None:
        document = self._document_by_id(document_id)
        if document is None:
            QMessageBox.warning(
                self,
                "Append optimizer solution",
                "The source scene for this optimizer is no longer open. Open the solution in a new tab instead.",
            )
            return
        if document.adapter.signature() != source_signature:
            answer = QMessageBox.question(
                self,
                "Source scene changed",
                f"{document.display_name()} changed after this optimization started.\n\n"
                "Append this frozen optimizer solution to the current version of the source scene anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            clone_ids = document.adapter.append_optimizer_candidate(candidate)
        except Exception as error:  # noqa: BLE001 - optimizer append boundary
            QMessageBox.warning(self, "Append optimizer solution", str(error))
            return
        self._update_document_tab(document)
        if document.document_id == self.active_document.document_id:
            self.plot.request_home_on_next_figure()
            self.refresh_scene()
        self.statusBar().showMessage(
            f"Appended {len(clone_ids)} changed optimizer coil"
            f"{'s' if len(clone_ids) != 1 else ''} to {document.display_name()}",
            6000,
        )

    def _optimizer_snapshot_saved_to_document(
        self,
        document_id: str,
        result: dict,
        name: str,
        field_scene: dict,
    ) -> None:
        document = self._document_by_id(document_id)
        if document is None:
            QMessageBox.warning(
                self,
                "Save optimizer snapshot",
                "The optimizer source scene is no longer open, so the snapshot cannot be saved there.",
            )
            return
        try:
            snapshot_id = document.adapter.create_imported_snapshot(
                result,
                name,
                field_scene=field_scene,
                origin_label="Optimizer result",
            )
        except Exception as error:  # noqa: BLE001 - optimizer snapshot boundary
            QMessageBox.warning(self, "Save optimizer snapshot", str(error))
            return
        self._snapshot_saved(document.document_id, snapshot_id)

    def open_optimizer(self) -> None:
        suggested_target = None
        if self.selected_id and self.adapter.supports_measurement(self.selected_id):
            try:
                mesh = self.adapter._mesh(self.selected_id)
                if mesh.get("role") == "measurement":
                    suggested_target = self.selected_id
            except KeyError:
                pass
        if not self.adapter.optimizer_targets():
            QMessageBox.information(
                self,
                "Optimizer target",
                "Add or import an eligible closed volume, set its role to Measurement volume, "
                "and enter the target field and tolerance first. Boxes, cylinders, spheres, and healthy "
                "closed STL/OBJ meshes are supported.",
            )
            return
        source_document = self.active_document
        source_document_id = source_document.document_id
        source_signature = source_document.adapter.signature()
        dialog = OptimizerDialog(
            source_document.adapter,
            self,
            suggested_target_id=suggested_target,
            on_apply=lambda candidate, replace, doc_id=source_document_id, signature=source_signature: (
                self._optimizer_candidate_applied_to_document(
                    doc_id, signature, candidate, replace
                )
            ),
            on_append=lambda candidate, doc_id=source_document_id, signature=source_signature: (
                self._optimizer_candidate_appended_to_document(
                    doc_id, signature, candidate
                )
            ),
            on_open_new_tab=self._optimizer_candidate_opened_new_tab,
            on_save_snapshot=lambda result, name, field_scene, doc_id=source_document_id: (
                self._optimizer_snapshot_saved_to_document(
                    doc_id, result, name, field_scene
                )
            ),
            preferences=self.settings,
        )
        dialog.setWindowTitle(f"{dialog.windowTitle()} — {source_document.display_name()}")
        self._show_independent_dialog(dialog, document_id=source_document_id)

    def _show_independent_dialog(
        self,
        dialog: QDialog,
        *,
        document_id: str | None = None,
    ) -> QDialog:
        """Show and retain a normal top-level window until the user closes it.

        A non-modal QDialog that keeps the main window as its native parent is
        still treated as a transient child by many window managers, which can
        keep it stacked above the Workbench.  Detaching it makes it a regular
        task window: it can sit behind, beside, or minimized independently.
        The explicit Python reference below replaces Qt parent ownership.
        """
        dialog.setParent(None, Qt.WindowType.Window)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        # Request the native title-bar controls after all other flag changes so
        # Win32 cannot normalize away Close/Maximize during a later flag update.
        enable_standard_window_controls(dialog)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)
        key = id(dialog)
        self._independent_windows[key] = dialog
        self._independent_window_documents[key] = (
            document_id or self.active_document.document_id
        )
        dialog.destroyed.connect(
            lambda _object=None, window_key=key: self._independent_window_destroyed(
                window_key
            )
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _independent_window_destroyed(self, window_key: int) -> None:
        self._independent_windows.pop(window_key, None)
        self._independent_window_documents.pop(window_key, None)

    def _set_owned_analysis_overlay(self, owner: object, overlay: dict[str, Any]) -> None:
        self._set_document_analysis_overlay(
            self.active_document.document_id,
            owner,
            overlay,
        )

    def _set_document_analysis_overlay(
        self,
        document_id: str,
        owner: object,
        overlay: dict[str, Any],
    ) -> None:
        document = self._document_by_id(document_id)
        if document is None:
            return
        document.analysis_overlay_owner = owner
        document.analysis_overlay = overlay
        if document_id == self.active_document.document_id:
            self.refresh_plot(rebuild_base=False)

    def _clear_owned_analysis_overlay(self, owner: object) -> None:
        self._clear_document_analysis_overlay(self.active_document.document_id, owner)

    def _clear_document_analysis_overlay(
        self,
        document_id: str,
        owner: object,
    ) -> None:
        document = self._document_by_id(document_id)
        if document is None or document.analysis_overlay_owner is not owner:
            return
        document.analysis_overlay_owner = None
        document.analysis_overlay = None
        if document_id == self.active_document.document_id:
            self.refresh_plot(rebuild_base=False)

    def _open_measurement_result_window(
        self,
        result: dict,
        *,
        allow_snapshot: bool,
        document_id: str | None = None,
    ) -> MeasurementResultsDialog:
        document = self._document_by_id(document_id or self.active_document.document_id)
        if document is None:
            raise RuntimeError("The source scene is no longer open.")
        overlay_owner = object()
        dialog = MeasurementResultsDialog(
            document.adapter,
            result,
            self,
            on_preview=lambda overlay, owner=overlay_owner, source=document.document_id: (
                self._set_document_analysis_overlay(source, owner, overlay)
            ),
            on_snapshot_saved=(
                lambda snapshot_id, source=document.document_id: self._snapshot_saved(
                    source, snapshot_id
                )
            )
            if allow_snapshot
            else None,
            allow_snapshot=allow_snapshot,
        )
        dialog.setWindowTitle(f"{dialog.windowTitle()} — {document.display_name()}")
        dialog.finished.connect(
            lambda _code, owner=overlay_owner, source=document.document_id: (
                self._clear_document_analysis_overlay(source, owner)
            )
        )
        self._show_independent_dialog(dialog, document_id=document.document_id)
        return dialog

    def _open_snapshot_result_window(
        self,
        result: dict,
        *,
        document_id: str | None = None,
    ) -> None:
        self._open_measurement_result_window(
            result,
            allow_snapshot=False,
            document_id=document_id,
        )

    def _open_snapshot_comparison_window(
        self,
        snapshot_ids: list[str],
        *,
        document_id: str | None = None,
    ) -> None:
        document = self._document_by_id(document_id or self.active_document.document_id)
        if document is None:
            return
        dialog = SnapshotComparisonDialog(
            document.adapter,
            snapshot_ids,
            parent=self,
        )
        dialog.setWindowTitle(f"{dialog.windowTitle()} — {document.display_name()}")
        self._show_independent_dialog(
            dialog,
            document_id=document.document_id,
        )

    def _selected_measurement_volume_ids(self) -> list[str]:
        """Return selected, editable objects already assigned the Measurement role."""
        volume_ids: list[str] = []
        for object_id in self.selected_ids:
            try:
                obj = self.adapter.get_object(object_id)
            except KeyError:
                continue
            if obj.get("derived") or obj.get("role") != "measurement":
                continue
            if self.adapter.supports_measurement(object_id):
                volume_ids.append(object_id)
        return volume_ids

    def analyze_selected_measurement(self) -> None:
        # A single object can have pending inspector edits (including a newly
        # chosen Measurement role).  Commit those before collecting the final
        # selection.  Multi-selection has no editable inspector, so every object
        # is already represented by its stored scene state.
        if (
            len(self.selected_ids) == 1
            and self.selected_id is not None
            and self.adapter.supports_measurement(self.selected_id)
            and self.geometry_role_combo is not None
            and self.geometry_role_combo.currentData() == "measurement"
        ):
            if not self.apply_inspector():
                return

        object_ids = self._selected_measurement_volume_ids()
        if not object_ids:
            QMessageBox.information(
                self,
                "Analyze volume",
                "Please select one or more Measurement volumes.",
            )
            return

        results: list[dict[str, Any]] = []
        errors: list[str] = []
        total = len(object_ids)

        for index, object_id in enumerate(object_ids, start=1):
            try:
                label = str(self.adapter.get_object(object_id).get("label", object_id))
            except KeyError:
                label = object_id

            progress = MeasurementAnalysisProgressDialog(self)
            if total > 1:
                progress.setWindowTitle(
                    f"Analyzing measurement volume {index} of {total}"
                )
                progress.setLabelText(
                    f"Preparing {label}…\nVolume {index} of {total}"
                )
                QApplication.processEvents()

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage(
                f"Sampling measurement volume {index} of {total}: {label}…"
                if total > 1
                else f"Sampling measurement volume: {label}…"
            )
            try:
                result = self.adapter.analyze_measurement(
                    object_id,
                    progress_callback=progress.update_from_solver,
                )
            except FieldCalculationCancelled:
                elapsed = progress.elapsed_seconds()
                completed = len(results)
                self.statusBar().showMessage(
                    (
                        f"Measurement-volume analysis cancelled after {elapsed:.1f} s "
                        f"({completed} of {total} completed)"
                        if total > 1
                        else f"Measurement-volume analysis cancelled after {elapsed:.1f} s"
                    ),
                    6000,
                )
                return
            except Exception as error:  # noqa: BLE001 - GUI analysis boundary
                errors.append(f"{label}: {error}")
            else:
                results.append(result)
                self._open_measurement_result_window(result, allow_snapshot=True)
            finally:
                progress.finish()
                QApplication.restoreOverrideCursor()

        if errors:
            detail = "\n".join(f"• {message}" for message in errors)
            QMessageBox.warning(
                self,
                "Analyze volume",
                f"Some selected volumes could not be analyzed:\n\n{detail}",
            )

        if not results:
            self.statusBar().showMessage("No measurement volumes were analyzed.", 6000)
            return

        if total == 1 and len(results) == 1:
            result = results[0]
            stats = result["stats"]
            self.statusBar().showMessage(
                f"{result['label']}: mean |B| {stats['mean_uT']:.6g} µT • "
                f"{stats['in_band_pct']:.3g}% in target band",
                6000,
            )
        else:
            self.statusBar().showMessage(
                f"Analyzed {len(results)} of {total} selected Measurement volumes; "
                f"opened {len(results)} result window"
                f"{'s' if len(results) != 1 else ''}.",
                7000,
            )

    def _snapshot_saved(
        self,
        document_id: str,
        snapshot_id: str | None = None,
    ) -> None:
        if snapshot_id is None:
            snapshot_id = document_id
            document_id = self.active_document.document_id
        document = self._document_by_id(document_id)
        if document is None:
            return
        snapshot = next(
            (
                item
                for item in document.adapter.list_snapshots()
                if str(item.get("id")) == str(snapshot_id)
            ),
            None,
        )
        self._update_document_tab(document)
        if document_id == self.active_document.document_id:
            self._update_actions()
            self._update_title()
        if document.snapshot_manager_dialog is not None:
            document.snapshot_manager_dialog.refresh(select_id=snapshot_id)
        name = snapshot.get("name", snapshot_id) if snapshot else snapshot_id
        self.statusBar().showMessage(
            f"Saved field snapshot in {document.display_name()}: {name}",
            4500,
        )

    def _snapshots_changed(self, document_id: str | None = None) -> None:
        document = self._document_by_id(document_id or self.active_document.document_id)
        if document is None:
            return
        self._update_document_tab(document)
        if document.document_id == self.active_document.document_id:
            self._update_actions()
            self._update_title()

    def _snapshot_manager_destroyed(
        self,
        document_id: str,
        manager_id: int,
    ) -> None:
        document = self._document_by_id(document_id)
        if (
            document is not None
            and document.snapshot_manager_dialog is not None
            and id(document.snapshot_manager_dialog) == manager_id
        ):
            document.snapshot_manager_dialog = None

    def open_snapshots(self) -> None:
        document = self.active_document
        dialog = document.snapshot_manager_dialog
        if dialog is not None:
            dialog.refresh_if_changed()
            if dialog.isMinimized():
                dialog.showNormal()
            else:
                dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return

        dialog = SnapshotManagerDialog(
            document.adapter,
            self,
            on_changed=lambda document_id=document.document_id: self._snapshots_changed(
                document_id
            ),
            on_open_result=lambda result, document_id=document.document_id: (
                self._open_snapshot_result_window(
                    result,
                    document_id=document_id,
                )
            ),
            on_open_comparison=lambda snapshot_ids, document_id=document.document_id: (
                self._open_snapshot_comparison_window(
                    snapshot_ids,
                    document_id=document_id,
                )
            ),
        )
        dialog.setWindowTitle(f"Field snapshots — {document.display_name()}")
        document.snapshot_manager_dialog = dialog
        manager_id = id(dialog)
        dialog.destroyed.connect(
            lambda _object=None,
            document_id=document.document_id,
            key=manager_id: self._snapshot_manager_destroyed(document_id, key)
        )
        self._show_independent_dialog(dialog, document_id=document.document_id)

    def _scene_notes_text_changed(self) -> None:
        dialog = self.notes_dialog
        if dialog is None:
            return
        if self._scene_notes_edit_baseline is None:
            self._scene_notes_edit_baseline = self.adapter.get_scene_notes()
        self.adapter.set_scene_notes(dialog.notes(), record_history=False)
        self._update_title()

    def _commit_scene_notes_session(self) -> None:
        baseline = self._scene_notes_edit_baseline
        if baseline is None:
            return
        self.adapter.commit_scene_notes_edit(baseline)
        self._scene_notes_edit_baseline = None
        self._update_actions()
        self._update_title()

    def _sync_scene_notes_dialog(self) -> None:
        """Reload an existing Notes window after scene replacement or undo/redo."""
        self._scene_notes_edit_baseline = None
        if self.notes_dialog is not None:
            self.notes_dialog.set_notes(self.adapter.get_scene_notes())
            self.notes_dialog.setWindowTitle(
                f"Scene notes — {self.active_document.display_name()}"
            )

    def open_notes(self) -> None:
        dialog = self.notes_dialog
        if dialog is None:
            dialog = SceneNotesDialog(self.adapter.get_scene_notes(), self)
            dialog.setWindowTitle(
                f"Scene notes — {self.active_document.display_name()}"
            )
            self.notes_dialog = dialog
            dialog.editor.textChanged.connect(self._scene_notes_text_changed)
            dialog.apply_requested.connect(self._commit_scene_notes_session)
            dialog.finished.connect(
                lambda _result: self._commit_scene_notes_session()
            )
        elif not dialog.isVisible():
            dialog.set_notes(self.adapter.get_scene_notes())
            dialog.setWindowTitle(
                f"Scene notes — {self.active_document.display_name()}"
            )
            self._scene_notes_edit_baseline = None

        if dialog.isMinimized():
            dialog.showNormal()
        else:
            dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.editor.setFocus()

    def open_field_map(self) -> None:
        source_document = self.active_document
        dialog = FieldMapDialog(source_document.adapter)
        dialog.setWindowTitle(
            f"{dialog.windowTitle()} — {source_document.display_name()} (snapshot)"
        )
        # Start the embedded selector from the user's current main-scene camera
        # when WebEngine has a live view state. The callback is asynchronous;
        # PlotView safely defers it until the dialog's initial scene is rendered.
        self.plot.capture_view_state(dialog.plot.apply_view_state)
        show_application_window(dialog)

    def open_field_volume_map(self) -> None:
        source_document = self.active_document
        dialog = FieldVolumeMapDialog(source_document.adapter)
        dialog.setWindowTitle(
            f"{dialog.windowTitle()} — {source_document.display_name()} (snapshot)"
        )
        self.plot.capture_view_state(dialog.plot.apply_view_state)
        show_application_window(dialog)

    def _selected_sri24_brain_id(self) -> str | None:
        if self.selected_id is None or len(self.selected_ids) != 1:
            return None
        try:
            selected = self.adapter.get_object(self.selected_id)
        except KeyError:
            return None
        if selected.get("builtin_template") != "sri24_brain":
            return None
        return self.selected_id

    def _sri24_brain_ids(self) -> list[str]:
        return [
            str(item["id"])
            for item in self.adapter.list_objects()
            if item.get("builtin_template") == "sri24_brain" and item.get("id")
        ]

    def _brain_view_sri24_brain_id(self) -> str | None:
        """Resolve the one specific SRI24 brain Brain View should use.

        An explicitly selected SRI24 brain always wins. If the scene contains
        exactly one SRI24 brain, use it automatically even when some other
        scene object (or nothing) is selected. Multiple brains deliberately
        remain ambiguous until the user selects one.
        """

        selected = self._selected_sri24_brain_id()
        if selected is not None:
            return selected
        brain_ids = self._sri24_brain_ids()
        return brain_ids[0] if len(brain_ids) == 1 else None

    def open_brain_view(self) -> None:
        brain_object_id = self._brain_view_sri24_brain_id()
        if brain_object_id is None:
            sri24_count = len(self._sri24_brain_ids())
            message = (
                "Please select the SRI24 brain to use for Brain View."
                if sri24_count > 1
                else "Brain View requires an SRI24 brain in the scene."
            )
            QMessageBox.information(
                self,
                "Brain view",
                message,
            )
            return
        source_document = self.active_document
        try:
            dialog = BrainViewDialog(source_document.adapter, brain_object_id)
        except Exception as error:  # noqa: BLE001 - bundled atlas/UI boundary
            QMessageBox.warning(self, "Brain view", str(error))
            return
        dialog.setWindowTitle(
            f"{dialog.windowTitle()} — {source_document.display_name()} (snapshot)"
        )
        self.plot.capture_view_state(dialog.plot.apply_view_state)
        show_application_window(dialog)

    def open_sensor_plot(self) -> None:
        document = self.active_document
        dialog = SensorPlotDialog(document.adapter, self)
        dialog.setWindowTitle(f"{dialog.windowTitle()} — {document.display_name()}")
        self._show_independent_dialog(
            dialog,
            document_id=document.document_id,
        )

    def open_measurement_calibration(
        self,
        initial_coil_ids: list[str] | None = None,
    ) -> None:
        selected_before = self.selected_id
        dialog = MeasurementCalibrationDialog(
            self.adapter,
            self,
            initial_coil_ids=initial_coil_ids,
        )
        dialog.setWindowTitle(
            f"{dialog.windowTitle()} — {self.active_document.display_name()}"
        )
        dialog.exec()
        # Saving datasets and applying a calibration are scene edits. Refresh the
        # tree/inspector even when the dialog was only used to save raw data so
        # dirty-state, undo state, and any newly applied correction are visible.
        self.refresh_scene(select_id=selected_before)

    def open_preferences(self) -> None:
        dialog = QDialog(self)
        enable_standard_window_controls(dialog)
        dialog.setWindowTitle("Preferences")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        theme_selector = QComboBox()
        theme_selector.setObjectName("themeSelector")
        for label, theme_id in THEME_CHOICES:
            theme_selector.addItem(label, theme_id)
        theme_selector.setCurrentIndex(
            max(0, theme_selector.findData(current_theme()))
        )
        theme_selector.setToolTip(
            "Applies immediately to the main Workbench and every open detached window."
        )

        def apply_selected_theme(_index: int) -> None:
            selected_theme = theme_selector.currentData()
            if selected_theme is not None:
                self.set_theme(str(selected_theme))

        theme_selector.currentIndexChanged.connect(apply_selected_theme)
        form.addRow("Theme", theme_selector)

        model = QComboBox()
        for value in ("auto", "exact", "bundled", "current_sheet", "centreline"):
            model.addItem(COIL_MODELING_LABELS[value], value)
        current = self.adapter.coil_modeling_method
        model.setCurrentIndex(max(0, model.findData(current)))
        form.addRow("Coil field model", model)

        memory_budget = QSpinBox()
        memory_budget.setRange(
            MIN_SOLVER_MEMORY_BUDGET_MB, MAX_SOLVER_MEMORY_BUDGET_MB
        )
        memory_budget.setSingleStep(64)
        memory_budget.setSuffix(" MB")
        memory_budget.setValue(int(self.adapter.solver_memory_budget_mb))
        memory_budget.setToolTip(
            "Target maximum temporary memory for one detailed Magpylib kernel call. "
            "Lower values use smaller point/source batches and are slower but safer."
        )
        form.addRow("Solver temporary-memory budget", memory_budget)

        optimizer_worker_row = QWidget()
        optimizer_worker_layout = QHBoxLayout(optimizer_worker_row)
        optimizer_worker_layout.setContentsMargins(0, 0, 0, 0)
        optimizer_workers = QSpinBox()
        optimizer_workers.setRange(1, available_optimizer_workers())
        optimizer_workers.setValue(int(self.adapter.optimizer_worker_count))
        optimizer_workers.setSuffix(" workers")
        optimizer_workers.setToolTip(
            "Maximum concurrent workers used by optimizer candidate batches and queued "
            "2D/3D render jobs. A value of 1 keeps both workloads serial. "
            "Counts may exceed the machine's reported logical CPU count (oversubscription); "
            "use Benchmark to find the fastest practical value. Each process owns a private "
            "scene adapter, so memory use can rise with this value."
        )
        optimizer_worker_layout.addWidget(optimizer_workers)
        benchmark_workers = QPushButton("Benchmark…")
        benchmark_workers.setToolTip(
            "Measure packaged-scene optimizer throughput using the field model and memory "
            "budget currently selected in this Preferences window."
        )
        optimizer_worker_layout.addWidget(benchmark_workers)
        optimizer_worker_layout.addStretch(1)
        form.addRow("Worker concurrency", optimizer_worker_row)

        def run_optimizer_worker_benchmark() -> None:
            benchmark = OptimizerWorkerBenchmarkDialog(
                coil_modeling_method=str(model.currentData()),
                solver_memory_budget_mb=memory_budget.value(),
                current_workers=optimizer_workers.value(),
                parent=dialog,
            )
            if (
                benchmark.exec() == QDialog.DialogCode.Accepted
                and benchmark.recommended_workers is not None
            ):
                optimizer_workers.setValue(benchmark.recommended_workers)

        benchmark_workers.clicked.connect(run_optimizer_worker_benchmark)

        diagnostics = QCheckBox("Record solver diagnostics to a temporary JSONL log")
        diagnostics.setChecked(bool(self.adapter.solver_diagnostics_enabled))
        form.addRow("Diagnostics", diagnostics)

        diagnostics_path = QLabel(
            self.adapter.solver_diagnostics_path()
            or "A log path will be created when diagnostics are enabled."
        )
        diagnostics_path.setWordWrap(True)
        diagnostics_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        diagnostics_path.setObjectName("hint")
        form.addRow("Current log", diagnostics_path)

        solver_note = QLabel(
            "Solver: Auto balances speed and accuracy; memory and workers cap resource use."
        )
        solver_note.setWordWrap(True)
        solver_note.setObjectName("hint")
        form.addRow(solver_note)

        fluxline_group = QGroupBox("Fluxline cleanup")
        fluxline_form = QFormLayout(fluxline_group)

        minimum_field_row = QWidget()
        minimum_field_layout = QHBoxLayout(minimum_field_row)
        minimum_field_layout.setContentsMargins(0, 0, 0, 0)
        minimum_field_enabled = QCheckBox("Enabled")
        minimum_field_enabled.setChecked(
            bool(self.adapter.fluxline_minimum_field_cutoff_enabled)
        )
        minimum_field_value = CompactDoubleSpinBox()
        minimum_field_value.setRange(
            MIN_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
            MAX_FLUXLINE_MIN_FIELD_CUTOFF_PERCENT,
        )
        minimum_field_value.setDecimals(6)
        minimum_field_value.setSingleStep(0.01)
        minimum_field_value.setSuffix(" %")
        minimum_field_value.setValue(
            float(self.adapter.fluxline_minimum_field_cutoff_percent)
        )
        minimum_field_value.setToolTip(
            "Stops a traced fluxline when the local field falls below this percentage "
            "of the map's 99th-percentile field magnitude. This avoids unstable "
            "directions around near-zero/null regions and applies equally to B and H maps."
        )
        minimum_field_enabled.toggled.connect(minimum_field_value.setEnabled)
        minimum_field_value.setEnabled(minimum_field_enabled.isChecked())
        minimum_field_layout.addWidget(minimum_field_enabled)
        minimum_field_layout.addWidget(minimum_field_value)
        minimum_field_layout.addStretch(1)
        fluxline_form.addRow("Minimum field cutoff", minimum_field_row)

        conductor_row = QWidget()
        conductor_layout = QHBoxLayout(conductor_row)
        conductor_layout.setContentsMargins(0, 0, 0, 0)
        conductor_enabled = QCheckBox("Enabled")
        conductor_enabled.setChecked(bool(self.adapter.fluxline_conductor_cutoff_enabled))
        conductor_value = CompactDoubleSpinBox()
        conductor_value.setRange(
            MIN_FLUXLINE_CONDUCTOR_CUTOFF_MM,
            MAX_FLUXLINE_CONDUCTOR_CUTOFF_MM,
        )
        conductor_value.setDecimals(2)
        conductor_value.setSingleStep(0.5)
        conductor_value.setSuffix(" mm")
        conductor_value.setValue(float(self.adapter.fluxline_conductor_cutoff_mm))
        conductor_value.setToolTip(
            "Stops fluxlines before they enter this distance from an energized coil winding. "
            "When physical winding dimensions are available, the distance is measured from "
            "the winding-pack envelope; otherwise it is measured from the coil centreline."
        )
        conductor_enabled.toggled.connect(conductor_value.setEnabled)
        conductor_value.setEnabled(conductor_enabled.isChecked())
        conductor_layout.addWidget(conductor_enabled)
        conductor_layout.addWidget(conductor_value)
        conductor_layout.addStretch(1)
        fluxline_form.addRow("Conductor distance cutoff", conductor_row)

        fluxline_note = QLabel(
            "Cleanup stops unstable fluxlines in weak fields or near energized conductors."
        )
        fluxline_note.setWordWrap(True)
        fluxline_note.setObjectName("hint")
        fluxline_form.addRow(fluxline_note)

        layout.addLayout(form)
        layout.addWidget(fluxline_group)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_method = str(model.currentData())
            for document in self._documents:
                document.adapter.set_coil_modeling_method(selected_method)
                document.adapter.set_solver_memory_budget_mb(memory_budget.value())
                document.adapter.set_optimizer_worker_count(optimizer_workers.value())
                document.adapter.set_solver_diagnostics_enabled(diagnostics.isChecked())
                document.adapter.set_fluxline_minimum_field_cutoff(
                    minimum_field_enabled.isChecked(), minimum_field_value.value()
                )
                document.adapter.set_fluxline_conductor_cutoff(
                    conductor_enabled.isChecked(), conductor_value.value()
                )
            self.settings.setValue("coil_modeling/method", selected_method)
            self.settings.setValue("solver/memoryBudgetMb", memory_budget.value())
            self.settings.setValue("optimizer/workerCount", optimizer_workers.value())
            configure_field_render_worker_limit(optimizer_workers.value())
            self.settings.setValue("solver/diagnosticsEnabled", diagnostics.isChecked())
            self.settings.setValue(
                "fluxlines/minimumFieldCutoffEnabled",
                minimum_field_enabled.isChecked(),
            )
            self.settings.setValue(
                "fluxlines/minimumFieldCutoffPercent", minimum_field_value.value()
            )
            self.settings.setValue(
                "fluxlines/conductorDistanceCutoffEnabled", conductor_enabled.isChecked()
            )
            self.settings.setValue(
                "fluxlines/conductorDistanceCutoffMm", conductor_value.value()
            )
            self.settings.sync()
            self.analysis_overlay = None
            self._update_status()
            status = (
                f"Coil model: {model.currentText()} • solver budget: "
                f"{memory_budget.value()} MB • worker limit: {optimizer_workers.value()} worker"
                f"{'s' if optimizer_workers.value() != 1 else ''}"
            )
            if diagnostics.isChecked():
                status += " • diagnostics enabled"
            self.statusBar().showMessage(status, 5000)

    def open_help_topics(self) -> None:
        HelpTopicsDialog(self).exec()

    def open_ai_help(self) -> None:
        dialog = AiHelpDialog(self.adapter, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.pasted_into_new_scene and dialog.new_scene_adapter is not None:
            adapter = dialog.new_scene_adapter
            self._apply_saved_preferences_to_adapter(adapter)
            imported_set = set(dialog.imported_ids)
            top_level_ids: list[str] = []
            for object_id in dialog.imported_ids:
                try:
                    parent_id = str(adapter.get_object(object_id).get("parent") or "")
                except KeyError:
                    continue
                if parent_id not in imported_set:
                    top_level_ids.append(object_id)
            selected_id = (
                top_level_ids[-1]
                if top_level_ids
                else (dialog.imported_ids[-1] if dialog.imported_ids else None)
            )
            document = self._make_document(adapter, selected_id=selected_id)
            # This is a newly generated, unsaved result. Keep it visibly dirty so
            # closing the tab cannot silently discard the AI-produced scene.
            document.saved_signature = ""
            document.selected_ids = list(top_level_ids or dialog.imported_ids)
            self._append_document(document)
            noun = "object" if len(dialog.imported_ids) == 1 else "objects"
            self.statusBar().showMessage(
                f"Pasted {len(dialog.imported_ids)} AI Scene Recipe {noun} into a new scene tab.",
                4500,
            )
            return

        if not dialog.imported_ids and not dialog.replaced_scene:
            return
        imported_set = set(dialog.imported_ids)
        top_level_ids: list[str] = []
        for object_id in dialog.imported_ids:
            try:
                parent_id = str(self.adapter.get_object(object_id).get("parent") or "")
            except KeyError:
                continue
            if parent_id not in imported_set:
                top_level_ids.append(object_id)
        self.plot.request_home_on_next_figure()
        if dialog.imported_ids:
            self.refresh_scene(
                select_id=top_level_ids[-1] if top_level_ids else dialog.imported_ids[-1],
                select_ids=top_level_ids or dialog.imported_ids,
            )
        else:
            self.refresh_scene()
        noun = "object" if len(dialog.imported_ids) == 1 else "objects"
        self.statusBar().showMessage(
            f"Modified current scene with {len(dialog.imported_ids)} AI Scene Recipe {noun}.",
            4500,
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Field Workbench",
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {__version__}</p>"
            "<p>Designed, specified, tested, and maintained by John Carscallen.<br>"
            "Substantial portions of the source code were generated with "
            "assistance from ChatGPT by OpenAI.</p>"
            "<p>Free software licensed under GNU GPL version 3 or later "
            "(GPL-3.0-or-later).</p>",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if self._tab_switch_in_progress:
            self.statusBar().showMessage(
                "Wait for the current scene switch to finish before exiting.",
                2500,
            )
            event.ignore()
            return
        self._commit_scene_notes_session()
        running = next(
            (
                document
                for document in self._documents
                if document.drc_dialog is not None
                and document.drc_dialog.thread is not None
            ),
            None,
        )
        if running is not None:
            QMessageBox.information(
                self,
                "Design check running",
                (
                    f"Wait for the design check for {running.display_name()} "
                    "to finish before exiting."
                ),
            )
            event.ignore()
            return
        for document in self._documents:
            if not self.maybe_save_document(document):
                event.ignore()
                return
        for document in tuple(self._documents):
            self._close_document_windows(document.document_id)
        if self.notes_dialog is not None:
            self.notes_dialog.close()
        self.plot.cleanup()
        event.accept()
