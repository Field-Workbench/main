"""Main PySide6 window for the Field Workbench workflow prototype."""

from __future__ import annotations

import ast
import base64
import copy
import html
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QSignalBlocker, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QTabBar,
    QTabWidget,
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
from .brain_analysis import BrainRegionMetrics, BrainRegionNode
from .exposure_panel import ExposureRegionPanel, ExposureResultTab, REGION_KEY_ROLE
from .region_analysis import MeasurementRegions
from .exposures import exposure_coil_ids
from .waveform_dialog import WaveformPlaybackDialog
from .waveform import playback_frame_warning_text
from .help_topics import HelpTopicsDialog
from .preferences import apply_global_preferences_to_adapter, open_global_preferences
from .optimizer import (
    DEFAULT_OPTIMIZER_WORKER_COUNT,
    available_optimizer_workers,
)
from .plot_view import PlotView
from .waveform_gl_view import WaveformGLView
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
    GEOMETRY_ROLE_LABELS,
    SCENE_SUFFIX,
    FieldCalculationCancelled,
    StudioAdapter,
    StudioOperationError,
    awg_resistance_20_ohm_per_km,
    scale_nested,
)



if TYPE_CHECKING:
    from .project_context import ProjectContext

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
    """The single model document edited by one Model Editor instance."""

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


class _BreathingWordWrapLabel(QLabel):
    """Word-wrapped label with a few spare pixels above and below the glyphs.

    Some Qt/style/font combinations report a height-for-width that lands exactly
    on the painted text bounds. In a tightly capped sidebar that can shave the
    antialiased top/bottom pixels from the first and last wrapped lines.
    """

    _EXTRA_VERTICAL_PX = 6

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt virtual
        measured = int(super().heightForWidth(max(0, int(width))))
        if measured < 0:
            measured = int(super().sizeHint().height())
        return measured + self._EXTRA_VERTICAL_PX

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        hint = super().sizeHint()
        hint.setHeight(hint.height() + self._EXTRA_VERTICAL_PX)
        return hint

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        hint = super().minimumSizeHint()
        hint.setHeight(hint.height() + self._EXTRA_VERTICAL_PX)
        return hint


class _CurrentPageStackedWidget(QWidget):
    """Page switcher whose layout measures only the visible source form.

    QStackedLayout can still reserve the tallest page's height-for-width even
    when QStackedWidget.sizeHint is overridden. A normal layout excludes hidden
    pages from both minimum-size and wrapped-label height calculations.
    """

    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: list[QWidget] = []
        self._current_index = -1
        self._page_layout = QVBoxLayout(self)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        self._page_layout.setSpacing(0)
        self._page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def addWidget(self, page: QWidget) -> int:  # noqa: N802 - Qt-style helper
        index = len(self._pages)
        self._pages.append(page)
        self._page_layout.addWidget(page)
        page.hide()
        if self._current_index < 0:
            self.setCurrentIndex(index)
        return index

    def currentWidget(self) -> QWidget | None:  # noqa: N802
        return self._pages[self._current_index] if self._current_index >= 0 else None

    def currentIndex(self) -> int:  # noqa: N802
        return self._current_index

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt virtual
        current = self.currentWidget()
        layout = current.layout() if isinstance(current, QWidget) else None
        return bool(layout is not None and layout.hasHeightForWidth())

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt virtual
        """Measure only the visible form at the sidebar's actual width.

        Saved-exposure hints and summaries are word wrapped.  A plain container
        otherwise reports only its unconstrained size hint, which can leave the
        last wrapped line slightly clipped in a narrow Maps sidebar.
        """
        current = self.currentWidget()
        if not isinstance(current, QWidget):
            return super().heightForWidth(width)
        layout = current.layout()
        if layout is not None and layout.hasHeightForWidth():
            measured = int(layout.heightForWidth(max(0, int(width))))
            if measured >= 0:
                return measured
        return int(current.sizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        current = self.currentWidget()
        if isinstance(current, QWidget):
            return current.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        current = self.currentWidget()
        if isinstance(current, QWidget):
            return current.minimumSizeHint()
        return super().minimumSizeHint()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        if index == self._current_index or not 0 <= index < len(self._pages):
            return
        self._current_index = index
        for page_index, page in enumerate(self._pages):
            page.setVisible(page_index == index)
        self._page_layout.invalidate()
        self.updateGeometry()
        self.currentChanged.emit(index)


class _WorkspaceSharedTabs(QTabWidget):
    """QTabWidget whose visible tab bar lives outside the content splitter.

    Maps / Analysis needs the result tabs to span the entire viewer width while
    the measurement/region panel begins *below* that strip.  Keeping a real
    QTabWidget underneath preserves the existing map-page API (the individual
    renderers still call ``tabs.addTab(...)``), while this lightweight mirror
    presents the tabs in a separate QTabBar owned by the outer workspace.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace_tab_bar = QTabBar(parent)
        self.workspace_tab_bar.setObjectName("fieldWorkspaceExternalTabBar")
        self.workspace_tab_bar.setExpanding(False)
        self.workspace_tab_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        # The internal bar remains the source of QTabWidget bookkeeping only;
        # hiding it lets the pane start flush with the preview/right-panel row.
        super().tabBar().hide()
        self.workspace_tab_bar.currentChanged.connect(self._external_current_changed)
        self.workspace_tab_bar.tabCloseRequested.connect(self.tabCloseRequested.emit)
        self.currentChanged.connect(self._internal_current_changed)

    def _external_current_changed(self, index: int) -> None:
        if index >= 0 and index != self.currentIndex():
            self.setCurrentIndex(index)

    def _internal_current_changed(self, index: int) -> None:
        if self.workspace_tab_bar.currentIndex() == index:
            return
        blocker = QSignalBlocker(self.workspace_tab_bar)
        self.workspace_tab_bar.setCurrentIndex(index)
        del blocker

    def addTab(self, widget: QWidget, *args) -> int:  # noqa: N802 - Qt API
        index = super().addTab(widget, *args)
        icon = super().tabIcon(index)
        text = super().tabText(index)
        if icon.isNull():
            self.workspace_tab_bar.insertTab(index, text)
        else:
            self.workspace_tab_bar.insertTab(index, icon, text)
        tooltip = super().tabToolTip(index)
        if tooltip:
            self.workspace_tab_bar.setTabToolTip(index, tooltip)
        self._internal_current_changed(self.currentIndex())
        return index

    def insertTab(self, index: int, widget: QWidget, *args) -> int:  # noqa: N802 - Qt API
        inserted = super().insertTab(index, widget, *args)
        icon = super().tabIcon(inserted)
        text = super().tabText(inserted)
        if icon.isNull():
            self.workspace_tab_bar.insertTab(inserted, text)
        else:
            self.workspace_tab_bar.insertTab(inserted, icon, text)
        tooltip = super().tabToolTip(inserted)
        if tooltip:
            self.workspace_tab_bar.setTabToolTip(inserted, tooltip)
        self._internal_current_changed(self.currentIndex())
        return inserted

    def removeTab(self, index: int) -> None:  # noqa: N802 - Qt API
        super().removeTab(index)
        blocker = QSignalBlocker(self.workspace_tab_bar)
        if 0 <= int(index) < self.workspace_tab_bar.count():
            self.workspace_tab_bar.removeTab(int(index))
        del blocker
        self._internal_current_changed(self.currentIndex())

    def setTabText(self, index: int, text: str) -> None:  # noqa: N802 - Qt API
        super().setTabText(index, text)
        if 0 <= int(index) < self.workspace_tab_bar.count():
            self.workspace_tab_bar.setTabText(int(index), str(text))

    def setTabToolTip(self, index: int, tip: str) -> None:  # noqa: N802 - Qt API
        super().setTabToolTip(index, tip)
        if 0 <= int(index) < self.workspace_tab_bar.count():
            self.workspace_tab_bar.setTabToolTip(int(index), str(tip))

    def setTabsClosable(self, closable: bool) -> None:  # noqa: N802 - Qt API
        super().setTabsClosable(closable)
        self.workspace_tab_bar.setTabsClosable(bool(closable))


class UnifiedFieldWorkspaceDialog(QDialog):
    """One Maps workspace with switchable Static/Scripted and 2D/3D/Brain views.

    The workspace owns one persistent left-hand pane and one shared
    Selector/result tab strip.  Static and Scripted are presentation modes of
    the same frozen launch-time scene, just like 2D/3D/Brain are view modes.
    Each mode/view pair keeps its own established controls and renderer state
    while calculated result tabs remain side by side across every selector
    change. Scripted mode adds one compact source selector shared across its
    2D/3D/Brain pages; complete protocol authoring lives in Exposures.
    """

    VIEW_LABELS = {"2d": "2D", "3d": "3D", "brain": "Brain"}
    MODE_LABELS = {"static": "Static", "playback": "Scripted"}

    def __init__(
        self,
        *,
        mode: str = "static",
        scene_name: str,
        adapter: StudioAdapter,
        brain_object_id: str | None = None,
        project_adapter: StudioAdapter | None = None,
        on_project_resources_changed=None,
        on_open_snapshot_manager=None,
        parent=None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        normalized = str(mode).strip().lower()
        if normalized not in {"static", "playback"}:
            raise ValueError(f"Unsupported field workspace mode: {mode!r}")
        self.mode = normalized
        self.scene_name = str(scene_name)
        self.brain_object_id = brain_object_id
        # ``snapshot_adapter`` is the frozen launch-time scene used by every
        # map/result in this workspace. ``project_adapter`` is the live project
        # store used only for project resources created from Maps (currently
        # saved static measurement results). Keeping the two explicit prevents
        # a concurrently edited Model Editor from silently changing an open map.
        self.project_adapter = project_adapter or adapter
        self.on_project_resources_changed = on_project_resources_changed
        self.on_open_snapshot_manager = on_open_snapshot_manager
        # Freeze the source once for the whole Maps workspace. Every Static/
        # Scripted and 2D/3D/Brain page is derived from this same launch-time
        # snapshot, so changing selectors never silently picks up later edits.
        self.snapshot_adapter = adapter.detached_copy()
        # Saved maps are derived project resources and can be very large. They
        # must not be copied into every frozen solver scene or child render job.
        self.snapshot_adapter.saved_maps = []
        self._pages: dict[str, QDialog] = {}
        self._control_pages: dict[str, QWidget] = {}
        self._selector_pages: dict[str, QWidget] = {}
        self._viewer_layouts: dict[str, QVBoxLayout] = {}
        self._launch_view_state: dict[str, Any] | None = None
        self._handling_shared_tab_change = False
        self._closing = False
        self._result_windows: dict[int, QDialog] = {}
        self._static_measurement_results: dict[str, dict[str, Any]] = {}
        self._static_measurement_metrics: dict[str, BrainRegionMetrics] = {}
        self._measurement_panel_width = 460
        self._measurement_panel_min_width = 220
        # Responsive Maps/Analysis geometry.  Standalone map dialogs keep their
        # roomier historical defaults, while the unified workspace can shrink
        # enough for ordinary laptop screens without squeezing the viewer down
        # to a sliver between the controls and analysis panel.
        self._sidebar_width = 330
        self._sidebar_min_width = 240
        self._workspace_view_min_width = 540
        self._measurement_view_min_width = 320
        self._applying_workspace_splitter_state = False
        # One preferred panel width belongs to the whole Maps/Analysis window,
        # but visibility is remembered per visual view. Static 2D/3D owns the
        # Measurement-volumes snapshot panel; Brain owns its anatomical Brain
        # Areas panel. Scripted 2D/3D deliberately has no region/snapshot panel:
        # playback is a visualization path, not a place to create static
        # snapshots or cumulative exposure analyses.
        self._measurement_panel_visibility_by_view = {
            "2d": False,
            "3d": False,
            "brain": True,
        }
        self._measurement_panel_visible = self._measurement_panel_visibility_by_view["2d"]
        self._applying_measurement_panel_state = False
        self.playback_controller: WaveformPlaybackDialog | None = None
        self.scripted_source_host: QWidget | None = None
        self._scripted_source_wheel_filter: QObject | None = None
        self._scripted_traces_wheel_filter: QObject | None = None
        self.scripted_source_type: QComboBox | None = None
        self.scripted_waveform: QComboBox | None = None
        self.scripted_amplitude: QDoubleSpinBox | None = None
        self.scripted_frequency: QDoubleSpinBox | None = None
        self.scripted_duration: QDoubleSpinBox | None = None
        self.scripted_slew: QDoubleSpinBox | None = None
        self.scripted_exposure: QComboBox | None = None
        self.scripted_source_stack: _CurrentPageStackedWidget | None = None
        self.scripted_exposure_summary: QLabel | None = None
        # Result labels describe both how the field is viewed and which field
        # source produced it. Number independently within each view/source pair
        # (for example ``2D\Static - 1`` and ``2D\Sine - 1``) so the tab
        # strip remains readable when Static and several Scripted sources are
        # interleaved. The identity map freezes a label when the result is made.
        self._result_label_counters: dict[tuple[str, str], int] = {}
        self._result_label_map: dict[tuple[str, str, int], str] = {}

        self.setObjectName("mapsUnifiedFieldWorkspace")
        self.setWindowTitle(f"Maps / Analysis — {self.scene_name}")
        self.resize(1360, 820)
        # Keep a useful floor, but do not force a desktop-sized window.  At the
        # minimum width the left controls remain usable, the right panel can
        # shrink, and the central viewer still retains a few hundred pixels.
        self.setMinimumSize(820, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setObjectName("unifiedFieldWorkspaceSplitter")
        self.workspace_splitter.setChildrenCollapsible(False)
        root.addWidget(self.workspace_splitter, 1)

        self.sidebar = QWidget()
        self.sidebar.setObjectName("fieldWorkspaceSidebar")
        self.sidebar.setMinimumWidth(self._sidebar_min_width)
        self.sidebar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Static/Scripted is intentionally the first selector: it chooses what
        # kind of field result is being made. 2D/3D/Brain immediately below it
        # chooses only how that result is viewed.
        sidebar_layout.addWidget(self._make_mode_selector(), 0)
        sidebar_layout.addWidget(self._make_view_selector(), 0)

        self.control_stack = QStackedWidget()
        self.control_stack.setObjectName("fieldWorkspaceControlStack")
        self.control_stack.setMinimumWidth(0)
        self.control_stack.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        sidebar_layout.addWidget(self.control_stack, 1)

        self.view_stack = QStackedWidget()
        self.view_stack.setObjectName("fieldWorkspaceViewStack")
        self.view_stack.setMinimumWidth(0)
        self.view_stack.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )

        # One permanent Selector plus all calculated Static and Scripted result
        # tabs. The Selector itself is a stack of the six mode/view previews.
        self.shared_tabs = _WorkspaceSharedTabs(self)
        self.shared_tabs.setObjectName("fieldWorkspaceSharedTabs")
        self.shared_tabs.setTabsClosable(True)
        self.selector_stack = QStackedWidget()
        self.selector_stack.setObjectName("fieldWorkspaceSelectorStack")
        self._shared_selector_tab_index = self.shared_tabs.addTab(
            self.selector_stack, "Selector"
        )
        shared_tab_bar = self.shared_tabs.workspace_tab_bar
        shared_tab_bar.setTabButton(
            self._shared_selector_tab_index, QTabBar.ButtonPosition.LeftSide, None
        )
        shared_tab_bar.setTabButton(
            self._shared_selector_tab_index, QTabBar.ButtonPosition.RightSide, None
        )
        self.shared_tabs.tabCloseRequested.connect(self._close_shared_tab)
        self.shared_tabs.currentChanged.connect(self._shared_tab_changed)
        self.load_saved_map_button = QPushButton("Load saved map…")
        self.load_saved_map_button.setObjectName("mapsLoadSavedMapButton")
        self.load_saved_map_button.setToolTip(
            "Open a saved Static or Scripted map from this project without recalculating it"
        )
        self.load_saved_map_button.clicked.connect(self._choose_saved_map_to_load)
        self._refresh_saved_map_actions()

        # Static 2D/3D uses the shared Measurement-volumes panel. Brain has its
        # own embedded Brain Areas panel. Scripted 2D/3D intentionally leaves
        # the right side entirely to the viewer so a playback can never look as
        # though a paused frame is a saveable measurement snapshot.
        self.measurement_view_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.measurement_view_splitter.setObjectName("mapsMeasurementViewSplitter")
        self.measurement_view_splitter.setChildrenCollapsible(True)
        self.measurement_view_splitter.setHandleWidth(8)
        self.measurement_view_splitter.addWidget(self.shared_tabs)
        self.measurement_panel_host = self._make_workspace_measurement_panel()
        self.measurement_view_splitter.addWidget(self.measurement_panel_host)
        self.measurement_view_splitter.setCollapsible(0, False)
        self.measurement_view_splitter.setCollapsible(1, True)
        self.measurement_view_splitter.setStretchFactor(0, 1)
        self.measurement_view_splitter.setStretchFactor(1, 0)
        self.measurement_view_splitter.setSizes([780, self._measurement_panel_width])
        self._track_workspace_panel_splitter(self.measurement_view_splitter)
        self.measurement_view_splitter.handle(1).setToolTip(
            "Drag right to hide measurements; drag left to show them."
        )

        # Keep the viewer actions visually attached to the part of the split
        # view that they control. Save/Fullscreen are centered beneath the plot,
        # while the panel toggle sits by itself beneath the right-hand panel.
        self.measurement_shell = QWidget(self)
        self.measurement_shell.setObjectName("mapsMeasurementShell")
        measurement_shell_layout = QVBoxLayout(self.measurement_shell)
        measurement_shell_layout.setContentsMargins(0, 0, 0, 0)
        measurement_shell_layout.setSpacing(0)

        # Keep the shared tab strip outside the viewer/right-panel splitter so
        # result tabs can use the full workspace width.  The right-hand panel
        # now starts level with the preview itself rather than with the tabs.
        self.workspace_tab_strip = QWidget(self.measurement_shell)
        self.workspace_tab_strip.setObjectName("mapsWorkspaceTabStrip")
        workspace_tab_strip_layout = QHBoxLayout(self.workspace_tab_strip)
        workspace_tab_strip_layout.setContentsMargins(0, 0, 0, 0)
        workspace_tab_strip_layout.setSpacing(6)
        workspace_tab_strip_layout.addWidget(self.load_saved_map_button, 0)
        workspace_tab_strip_layout.addWidget(self.shared_tabs.workspace_tab_bar, 1)
        measurement_shell_layout.addWidget(self.workspace_tab_strip, 0)
        measurement_shell_layout.addWidget(self.measurement_view_splitter, 1)

        self.workspace_footer = QWidget(self.measurement_shell)
        self.workspace_footer.setObjectName("mapsWorkspaceFooter")
        workspace_footer_layout = QHBoxLayout(self.workspace_footer)
        workspace_footer_layout.setContentsMargins(0, 0, 0, 6)
        workspace_footer_layout.setSpacing(0)

        self.workspace_viewer_footer = QWidget(self.workspace_footer)
        self.workspace_viewer_footer.setObjectName("mapsViewerFooter")
        viewer_footer_layout = QHBoxLayout(self.workspace_viewer_footer)
        viewer_footer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_footer_layout.setSpacing(6)
        viewer_footer_layout.addStretch(1)
        self.workspace_save_map_button = QPushButton("Save map…")
        self.workspace_save_map_button.setObjectName("mapsSaveRenderedMapButton")
        self.workspace_save_map_button.setToolTip(
            "Save this already-calculated map with the project so it can be reopened without recalculation"
        )
        self.workspace_save_map_button.clicked.connect(self._save_current_map)
        viewer_footer_layout.addWidget(self.workspace_save_map_button)
        self.workspace_fullscreen_button = QPushButton("Fullscreen viewer")
        self.workspace_fullscreen_button.setObjectName("mapsFullscreenViewerButton")
        self.workspace_fullscreen_button.setToolTip(
            "Open the active map in a separate true-fullscreen viewer (Esc/F11 closes it)"
        )
        self.workspace_fullscreen_button.clicked.connect(self._toggle_workspace_fullscreen)
        viewer_footer_layout.addWidget(self.workspace_fullscreen_button)
        viewer_footer_layout.addStretch(1)
        workspace_footer_layout.addWidget(self.workspace_viewer_footer, 1)

        self.workspace_panel_footer = QWidget(self.workspace_footer)
        self.workspace_panel_footer.setObjectName("mapsPanelFooter")
        panel_footer_layout = QHBoxLayout(self.workspace_panel_footer)
        panel_footer_layout.setContentsMargins(0, 0, 0, 0)
        panel_footer_layout.addStretch(1)
        self.workspace_regions_button = QPushButton("Show measurements")
        self.workspace_regions_button.setObjectName("mapsShowHidePanelButton")
        self.workspace_regions_button.setToolTip(
            "Show the right-hand measurement panel for the active Maps view"
        )
        self.workspace_regions_button.clicked.connect(self._toggle_workspace_regions)
        panel_footer_layout.addWidget(self.workspace_regions_button)
        panel_footer_layout.addStretch(1)
        workspace_footer_layout.addWidget(self.workspace_panel_footer, 0)
        measurement_shell_layout.addWidget(self.workspace_footer, 0)

        self.workspace_splitter.addWidget(self.sidebar)
        self.workspace_splitter.addWidget(self.view_stack)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([self._sidebar_width, 1030])
        self.workspace_splitter.splitterMoved.connect(self._workspace_splitter_moved)

        self._sync_mode_selector(self.mode)
        self._switch_view("2d")
        if self.mode == "playback":
            self._initialize_playback_protocol()

    def _measurement_volume_rows(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        try:
            objects = self.snapshot_adapter.list_objects()
        except Exception:
            objects = []
        for item in objects if isinstance(objects, list) else []:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("id", "")).strip()
            if not object_id or str(item.get("role", "")) != "measurement":
                continue
            try:
                supported = self.snapshot_adapter.supports_measurement(object_id)
            except Exception:
                supported = False
            if supported:
                rows.append((object_id, str(item.get("label") or object_id)))
        return rows

    def _make_workspace_measurement_panel(self) -> QWidget:
        """Build the persistent Selector/Static Measurement-volumes panel.

        Scripted and Brain result tabs already carry their own region panel.
        This shared panel fills the gap for Selector and ordinary Static map
        tabs so the unified viewer never loses its measurement interface.
        """

        host = QWidget(self)
        host.setObjectName("mapsMeasurementPanelHost")
        host.setMinimumWidth(self._measurement_panel_min_width)
        host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        rows = self._measurement_volume_rows()
        nodes = tuple(
            BrainRegionNode(object_id, label, (index + 1,), ())
            for index, (object_id, label) in enumerate(rows)
        )
        hierarchy = MeasurementRegions(nodes)
        panel = ExposureRegionPanel(hierarchy, measurement=True)
        panel.setObjectName("mapsMeasurementRegionPanel")
        panel.metrics_note.setText(
            "Measurement volumes sample the magnetic field inside defined regions. "
            "Analyze a volume to see its field statistics, then save the result as a snapshot "
            "for later review or comparison."
        )
        panel.frame_note.setText("No snapshot analyzed yet")
        panel.mode_controls.hide()
        panel.selection_note.setText(
            "Select a Measurement volume, then analyze it to populate statistics or save a snapshot."
        )
        panel.table.itemSelectionChanged.connect(self._measurement_panel_selection_changed)
        layout.addWidget(panel, 1)
        self.static_measurement_panel = panel

        actions = QWidget(host)
        actions.setObjectName("mapsMeasurementActions")
        action_layout = QGridLayout(actions)
        action_layout.setContentsMargins(8, 0, 8, 8)
        action_layout.setHorizontalSpacing(6)
        action_layout.setVerticalSpacing(6)
        action_layout.setColumnStretch(0, 1)
        action_layout.setColumnStretch(1, 1)

        self.measurement_analyze_button = QPushButton("Analyze selected…")
        self.measurement_analyze_button.setObjectName("mapsMeasurementAnalyzeButton")
        self.measurement_analyze_button.setToolTip(
            "Analyze the selected Measurement volume and calculate its field statistics"
        )
        self.measurement_analyze_button.clicked.connect(
            self._analyze_selected_workspace_measurement
        )
        action_layout.addWidget(self.measurement_analyze_button, 0, 0)

        self.measurement_analyze_all_button = QPushButton("Analyze all")
        self.measurement_analyze_all_button.setObjectName("mapsMeasurementAnalyzeAllButton")
        self.measurement_analyze_all_button.setToolTip(
            "Calculate field statistics for every Measurement volume in this scene"
        )
        self.measurement_analyze_all_button.clicked.connect(
            self._analyze_all_workspace_measurements
        )
        action_layout.addWidget(self.measurement_analyze_all_button, 0, 1)

        self.measurement_view_result_button = QPushButton("View / save snapshot…")
        self.measurement_view_result_button.setObjectName("mapsMeasurementViewResultButton")
        self.measurement_view_result_button.setToolTip(
            "Open the selected analyzed result; the result window can save its exact vector sample grid as a project snapshot"
        )
        self.measurement_view_result_button.clicked.connect(
            self._open_selected_workspace_measurement_result
        )
        action_layout.addWidget(self.measurement_view_result_button, 1, 0)

        self.measurement_saved_results_button = QPushButton("Saved results…")
        self.measurement_saved_results_button.setObjectName("mapsMeasurementSavedResultsButton")
        self.measurement_saved_results_button.setToolTip(
            "Manage saved Static maps, Scripted maps, and quantitative snapshots for this project"
        )
        self.measurement_saved_results_button.clicked.connect(
            self._open_snapshot_manager_from_maps
        )
        action_layout.addWidget(self.measurement_saved_results_button, 1, 1)
        layout.addWidget(actions, 0)

        if rows:
            first = panel.table.topLevelItem(0)
            if first is not None:
                panel.table.setCurrentItem(first)
        self._measurement_panel_selection_changed()
        return host

    def _selected_workspace_measurement_id(self) -> str:
        panel = getattr(self, "static_measurement_panel", None)
        if not isinstance(panel, ExposureRegionPanel):
            return ""
        item = panel.table.currentItem()
        if item is None:
            selected = panel.table.selectedItems()
            item = selected[0] if selected else None
        return str(item.data(0, REGION_KEY_ROLE) or "") if item is not None else ""

    def _measurement_panel_selection_changed(self) -> None:
        object_id = self._selected_workspace_measurement_id()
        has_selection = bool(object_id)
        rows = self._measurement_volume_rows()
        analyze_button = getattr(self, "measurement_analyze_button", None)
        if isinstance(analyze_button, QPushButton):
            analyze_button.setEnabled(has_selection)
        analyze_all = getattr(self, "measurement_analyze_all_button", None)
        if isinstance(analyze_all, QPushButton):
            analyze_all.setEnabled(bool(rows))
        view_button = getattr(self, "measurement_view_result_button", None)
        if isinstance(view_button, QPushButton):
            view_button.setEnabled(
                has_selection and object_id in self._static_measurement_results
            )

    @staticmethod
    def _measurement_region_metric(result: dict[str, Any]) -> BrainRegionMetrics:
        magnitudes = np.asarray(result.get("magnitudes_uT", []), dtype=float).reshape(-1)
        vectors = np.asarray(result.get("values_t", []), dtype=float).reshape(-1, 3)
        finite = np.isfinite(magnitudes)
        if len(vectors) == len(magnitudes):
            finite &= np.isfinite(vectors).all(axis=1)
        magnitudes = magnitudes[finite]
        vectors = vectors[finite] if len(vectors) == len(finite) else np.empty((0, 3))
        if not len(magnitudes):
            return BrainRegionMetrics(
                *(math.nan,) * 7, sample_count=0, voxel_count=0
            )

        mean_uT = float(np.mean(magnitudes))
        rms_uT = float(math.sqrt(np.mean(magnitudes * magnitudes)))
        p5_uT, p95_uT = (float(value) for value in np.percentile(magnitudes, [5, 95]))
        peak_uT = float(np.max(magnitudes))
        if mean_uT > 1e-30:
            uniformity = max(
                0.0, min(100.0, 100.0 * (1.0 - (p95_uT - p5_uT) / mean_uT))
            )
        else:
            uniformity = 100.0 if p95_uT <= 1e-30 else 0.0

        directional_consistency = 100.0
        if len(vectors) == len(magnitudes):
            magnitudes_t = np.linalg.norm(vectors, axis=1)
            directional = magnitudes_t > 1e-30
            if np.any(directional):
                units = vectors[directional] / magnitudes_t[directional, None]
                directional_consistency = max(
                    0.0, min(100.0, float(np.linalg.norm(np.mean(units, axis=0)) * 100.0))
                )

        stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
        volume_cm3 = float(
            result.get("volume_cm3", stats.get("volume_cm3", math.nan))
        )
        count = int(len(magnitudes))
        return BrainRegionMetrics(
            rms_uT=rms_uT,
            mean_uT=mean_uT,
            p95_uT=p95_uT,
            peak_uT=peak_uT,
            uniformity_pct=uniformity,
            directional_consistency_pct=directional_consistency,
            volume_cm3=volume_cm3,
            sample_count=count,
            voxel_count=count,
        )

    def _update_workspace_measurement_metrics(self, object_id: str, result: dict[str, Any]) -> None:
        self._static_measurement_results[str(object_id)] = result
        self._static_measurement_metrics[str(object_id)] = self._measurement_region_metric(result)
        panel = getattr(self, "static_measurement_panel", None)
        if isinstance(panel, ExposureRegionPanel):
            panel.set_static_metrics(self._static_measurement_metrics)
            total = len(self._measurement_volume_rows())
            analyzed = len(self._static_measurement_metrics)
            panel.frame_note.setText(
                f"{analyzed} of {total} analyzed"
                if total
                else "No measurement volumes"
            )
        self._measurement_panel_selection_changed()

    def _analyze_workspace_measurement(
        self, object_id: str, *, open_result: bool
    ) -> dict[str, Any] | None:
        object_id = str(object_id or "").strip()
        if not object_id:
            return None
        try:
            label = str(self.snapshot_adapter.get_object(object_id).get("label", object_id))
        except Exception:
            label = object_id
        progress = MeasurementAnalysisProgressDialog(self)
        progress.setWindowTitle("Analyzing measurement volume")
        progress.setLabelText(f"Preparing {label}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.snapshot_adapter.analyze_measurement(
                object_id, progress_callback=progress.update_from_solver
            )
        except FieldCalculationCancelled:
            return None
        except Exception as error:  # noqa: BLE001 - field-analysis boundary
            QMessageBox.warning(self, "Measurement snapshot", str(error))
            return None
        finally:
            progress.finish()
            QApplication.restoreOverrideCursor()

        self._update_workspace_measurement_metrics(object_id, result)
        if open_result:
            self._show_workspace_measurement_result(result)
        return result

    def _show_workspace_measurement_result(self, result: dict[str, Any]) -> None:
        dialog = MeasurementResultsDialog(
            self.snapshot_adapter,
            result,
            self,
            on_snapshot_saved=self._static_snapshot_saved,
            allow_snapshot=True,
            snapshot_store_adapter=self.project_adapter,
        )
        dialog.setWindowTitle(f"{dialog.windowTitle()} — {self.scene_name}")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)
        self._track_result_window(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _analyze_selected_workspace_measurement(self, *_args) -> None:
        object_id = self._selected_workspace_measurement_id()
        if not object_id:
            QMessageBox.information(
                self,
                "Measurement snapshot",
                "Select a Measurement volume in the right-hand panel first.",
            )
            return
        self._analyze_workspace_measurement(object_id, open_result=True)

    def _analyze_all_workspace_measurements(self, *_args) -> None:
        rows = self._measurement_volume_rows()
        if not rows:
            QMessageBox.information(
                self,
                "Measurement snapshot",
                "This scene does not contain a Measurement volume to analyze.",
            )
            return
        completed = 0
        for object_id, _label in rows:
            if self._analyze_workspace_measurement(object_id, open_result=True) is not None:
                completed += 1
        page = self._pages.get(self.current_page_key)
        status = getattr(page, "calculation_status", None) if page is not None else None
        if isinstance(status, QStatusBar):
            status.showMessage(
                f"Analyzed {completed} of {len(rows)} Measurement volumes",
                6000,
            )

    def _open_selected_workspace_measurement_result(self, *_args) -> None:
        object_id = self._selected_workspace_measurement_id()
        result = self._static_measurement_results.get(object_id)
        if isinstance(result, dict):
            self._show_workspace_measurement_result(result)

    # Compatibility entry point retained for older tests/callers; the Static
    # snapshot workflow now lives in the persistent right-hand panel.
    def _analyze_static_snapshot_volume(self, *_args) -> None:
        self._analyze_selected_workspace_measurement()

    def _track_result_window(self, dialog: QDialog) -> None:
        key = id(dialog)
        self._result_windows[key] = dialog
        dialog.destroyed.connect(
            lambda _object=None, window_key=key: self._result_windows.pop(
                window_key, None
            )
        )

    def _project_resources_changed(self) -> None:
        callback = self.on_project_resources_changed
        if callable(callback):
            callback()

    def _static_snapshot_saved(self, snapshot_id: str) -> None:
        self._project_resources_changed()
        page = self._pages.get(self.current_page_key)
        status = getattr(page, "calculation_status", None) if page is not None else None
        if isinstance(status, QStatusBar):
            status.showMessage(f"Saved result {snapshot_id} to project", 4500)

    def _refresh_saved_map_actions(self) -> None:
        """Refresh Save/Load affordances for the active rendered result."""
        try:
            has_saved_maps = bool(self.project_adapter.list_saved_maps())
        except Exception:
            has_saved_maps = False
        load_button = getattr(self, "load_saved_map_button", None)
        if isinstance(load_button, QPushButton):
            load_button.setEnabled(has_saved_maps)

        index = self.shared_tabs.currentIndex() if hasattr(self, "shared_tabs") else -1
        widget = self.shared_tabs.widget(index) if index >= 0 else None
        viewer = getattr(widget, "viewer", widget)
        can_save = (
            index >= 0
            and index != getattr(self, "_shared_selector_tab_index", -1)
            and isinstance(viewer, (PlotView, WaveformGLView))
        )
        button = getattr(self, "workspace_save_map_button", None)
        if isinstance(button, QPushButton):
            button.setVisible(can_save)
            button.setEnabled(can_save)

    def _install_saved_map_button(self, page: QDialog) -> None:
        """Route one view's save-map affordance through the shared Maps footer."""
        button = getattr(self, "workspace_save_map_button", None)
        if isinstance(button, QPushButton):
            page.save_map_button = button
            page._saved_map_actions_changed = self._refresh_saved_map_actions

    def _toggle_workspace_fullscreen(self, *_args) -> None:
        """Open the active selector/result using its established fullscreen action."""
        page = self._pages.get(self.current_page_key)
        toggle = getattr(page, "_toggle_current_plot_fullscreen", None) if page is not None else None
        if callable(toggle):
            toggle()

    def _save_current_map(self, *_args) -> None:
        index = self.shared_tabs.currentIndex()
        if index < 0 or index == self._shared_selector_tab_index:
            return
        widget = self.shared_tabs.widget(index)
        viewer = getattr(widget, "viewer", widget)
        owner_mode, owner_view = self._result_tab_owner(widget)
        if owner_mode not in {"static", "playback"} or owner_view not in self.VIEW_LABELS:
            QMessageBox.information(self, "Save map", "Select a rendered map tab first.")
            return
        try:
            if isinstance(viewer, PlotView):
                renderer = "plotly"
                content = viewer.export_figure()
            elif isinstance(viewer, WaveformGLView):
                renderer = "webgl"
                content = viewer.export_render_payload()
            else:
                raise StudioOperationError("The selected tab is not a rendered map.")
        except Exception as error:  # noqa: BLE001 - viewer serialization boundary
            QMessageBox.warning(self, "Save map", str(error))
            return

        default_name = self.shared_tabs.tabText(index).replace("\\", " - ").strip() or "Saved map"
        name, accepted = QInputDialog.getText(
            self,
            "Save calculated map",
            "Map name",
            text=default_name,
        )
        if not accepted:
            return
        if not str(name).strip():
            QMessageBox.warning(self, "Save map", "Map name cannot be blank.")
            return
        aux = copy.deepcopy(getattr(widget, "_saved_map_aux", {}))
        try:
            source_signature = hashlib.sha256(
                self.snapshot_adapter.signature().encode("utf-8")
            ).hexdigest()
            map_id = self.project_adapter.save_map_resource(
                name=str(name),
                kind="static" if owner_mode == "static" else "scripted",
                view=owner_view,
                renderer=renderer,
                content=content,
                aux=aux,
                source_signature=source_signature,
                tab_label=self.shared_tabs.tabText(index),
                tooltip=self.shared_tabs.tabToolTip(index),
            )
        except Exception as error:  # noqa: BLE001 - project resource boundary
            QMessageBox.warning(self, "Save map", str(error))
            return
        self._project_resources_changed()
        self._refresh_saved_map_actions()
        page = self._pages.get(self.current_page_key)
        status = getattr(page, "calculation_status", None) if page is not None else None
        if isinstance(status, QStatusBar):
            status.showMessage(f"Saved calculated map {map_id} to project", 4500)

    def _choose_saved_map_to_load(self, *_args) -> None:
        try:
            rows = self.project_adapter.list_saved_maps()
        except Exception as error:  # noqa: BLE001 - project resource boundary
            QMessageBox.warning(self, "Load saved map", str(error))
            return
        if not rows:
            QMessageBox.information(
                self, "Load saved map", "This project does not contain any saved maps yet."
            )
            self._refresh_saved_map_actions()
            return
        displays = [
            f"{row.get('name', row.get('id', 'Saved map'))}  ·  "
            f"{str(row.get('kind', 'static')).title()} / "
            f"{self.VIEW_LABELS.get(str(row.get('view', '2d')).lower(), str(row.get('view', '2d')))}"
            for row in rows
        ]
        selected, accepted = QInputDialog.getItem(
            self, "Load saved map", "Saved map", displays, 0, False
        )
        if not accepted:
            return
        try:
            map_id = str(rows[displays.index(selected)].get("id", ""))
        except (ValueError, IndexError):
            return
        self.load_saved_map(map_id)

    def load_saved_map(self, map_id: str) -> bool:
        """Open one cached map resource as a normal interactive result tab."""
        try:
            record = self.project_adapter.saved_map(map_id)
        except Exception as error:  # noqa: BLE001 - project resource boundary
            QMessageBox.warning(self, "Load saved map", str(error))
            return False

        view_key = str(record.get("view", "2d")).strip().lower()
        kind = str(record.get("kind", "static")).strip().lower()
        mode = "playback" if kind == "scripted" else "static"
        if view_key not in self.VIEW_LABELS:
            QMessageBox.warning(self, "Load saved map", "The saved map uses an unsupported view.")
            return False
        self._switch_mode(mode, view_key=view_key, select_selector=False)
        page = self._pages.get(self._page_key(mode, view_key))
        if page is None:
            QMessageBox.warning(
                self,
                "Load saved map",
                "The saved map view is unavailable for this project.",
            )
            return False

        content = record.get("content")
        renderer = str(record.get("renderer", "")).strip().lower()
        try:
            if renderer == "plotly":
                if not isinstance(content, dict):
                    raise StudioOperationError("The saved Plotly map payload is malformed.")
                viewer = PlotView(
                    include_3d_return_axes=True,
                    synchronize_legend_visibility=True,
                    image_filename=f"saved-map-{map_id}",
                )
                viewer.set_figure(content)
            elif renderer == "webgl":
                if not isinstance(content, dict):
                    raise StudioOperationError("The saved WebGL map payload is malformed.")
                viewer = WaveformGLView(content)
            else:
                raise StudioOperationError("The saved map uses an unsupported renderer.")

            connector = getattr(page, "_connect_result_camera_viewer", None)
            if callable(connector):
                connector(viewer)
            tab_widget: QWidget = viewer
            wrapper = getattr(page, "_wrap_render_result_view", None)
            if renderer == "webgl" and callable(wrapper):
                aux = record.get("aux") if isinstance(record.get("aux"), dict) else {}
                job: dict[str, Any] = {}
                static_metrics = aux.get("brain_static_metrics")
                frame_metrics = aux.get("brain_frame_metrics")
                if isinstance(static_metrics, dict) and static_metrics:
                    job["brain_static_metrics"] = static_metrics
                if isinstance(frame_metrics, dict) and frame_metrics:
                    job["brain_frame_metrics"] = frame_metrics
                wrapped = wrapper(viewer, job, {"payload": content})
                if isinstance(wrapped, QWidget):
                    tab_widget = wrapped
                    tab_widget._saved_map_aux = copy.deepcopy(aux)
            tagger = getattr(page, "_tag_workspace_result_widget", None)
            if callable(tagger):
                tagger(viewer)
                tagger(tab_widget)
            tab_widget._saved_map_resource_id = str(map_id)
            label = f"{self.VIEW_LABELS[view_key]}\\{record.get('name', 'Saved map')}"
            index = self.shared_tabs.addTab(tab_widget, label)
            tooltip = str(record.get("tooltip") or "").strip()
            self.shared_tabs.setTabToolTip(
                index, (tooltip + " — " if tooltip else "") + "loaded saved map"
            )
            self.shared_tabs.setCurrentIndex(index)
            refresh = getattr(viewer, "refresh_viewport", None)
            if callable(refresh):
                QTimer.singleShot(0, refresh)
            status = getattr(page, "calculation_status", None)
            if isinstance(status, QStatusBar):
                status.showMessage(
                    f"Loaded saved map '{record.get('name', map_id)}' without recalculation",
                    5000,
                )
            self._refresh_saved_map_actions()
            return True
        except Exception as error:  # noqa: BLE001 - saved-view reconstruction boundary
            QMessageBox.warning(self, "Load saved map", str(error))
            return False

    def _open_snapshot_manager_from_maps(self, *_args) -> None:
        callback = self.on_open_snapshot_manager
        if callable(callback):
            callback()
            return
        dialog = SnapshotManagerDialog(
            self.project_adapter,
            self,
            on_changed=self._project_resources_changed,
            on_open_map=lambda map_id, _kind: self.load_saved_map(map_id),
        )
        dialog.setWindowTitle(f"Saved results — {self.scene_name}")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._track_result_window(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @staticmethod
    def _page_key(mode: str, view_key: str) -> str:
        return f"{mode}:{view_key}"

    @property
    def current_page_key(self) -> str:
        current = self.view_stack.currentWidget()
        for key, page in self._pages.items():
            if page is current:
                return key
        return self._page_key(self.mode, "2d")

    @property
    def current_view_key(self) -> str:
        key = self.current_page_key
        return key.split(":", 1)[1] if ":" in key else "2d"

    @staticmethod
    def _collapse_workspace_sections(owner: object) -> None:
        """Start unified Static/Scripted sidebars with every disclosure closed.

        The established 2D/3D/Brain and waveform editors remember their own
        disclosure state when used as standalone windows.  In the unified
        workspaces we deliberately want a tidy category overview on every new
        launch instead.  Marking only the disclosure headers as non-persistent
        keeps all of the actual map/protocol parameter persistence intact.
        Once a workspace is open, sections remain wherever the user leaves them
        while switching between 2D, 3D, and Brain.
        """
        for value in vars(owner).values():
            try:
                header = getattr(value, "header", None)
            except RuntimeError:
                continue
            if not isinstance(header, QAbstractButton) or not header.isCheckable():
                continue
            header.setProperty("persistUiState", False)
            header.setChecked(False)

    def _make_mode_selector(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("fieldWorkspaceModeSelector")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 2)
        layout.setSpacing(5)

        label = QLabel("Mode")
        label.setObjectName("sectionHeading")
        layout.addWidget(label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.mode_buttons: dict[str, QPushButton] = {}
        for key in ("static", "playback"):
            button = QPushButton(self.MODE_LABELS[key])
            button.setObjectName(f"fieldWorkspaceMode_{key}")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setToolTip(
                "Inspect one fixed-current field state"
                if key == "static"
                else "Inspect a time-varying scripted exposure"
            )
            button.clicked.connect(
                lambda _checked=False, mode_key=key: self._switch_mode(mode_key)
            )
            self.mode_buttons[key] = button
            row.addWidget(button, 1)
        layout.addLayout(row)
        return panel

    def _sync_mode_selector(self, active_mode: str) -> None:
        for key, button in self.mode_buttons.items():
            button.setChecked(key == active_mode)

    def _make_view_selector(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("fieldWorkspaceViewSelector")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(5)

        label = QLabel("View")
        label.setObjectName("sectionHeading")
        layout.addWidget(label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.view_buttons: dict[str, QPushButton] = {}
        for key in ("2d", "3d", "brain"):
            button = QPushButton(self.VIEW_LABELS[key])
            button.setObjectName(f"fieldWorkspaceView_{key}")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if key == "brain" and self.brain_object_id is None:
                button.setEnabled(False)
                button.setToolTip("Brain view requires an SRI24 brain in the scene.")
            else:
                button.setToolTip(f"Show the {self.VIEW_LABELS[key]} view in Maps")
                button.clicked.connect(
                    lambda _checked=False, view_key=key: self._switch_view(view_key)
                )
            self.view_buttons[key] = button
            row.addWidget(button, 1)
        layout.addLayout(row)
        return panel

    def _playback_map_settings(self, key: str, page: QDialog) -> dict[str, Any]:
        getter = getattr(page, "_waveform_map_settings", None)
        settings = dict(getter()) if callable(getter) else {}
        if key == "brain":
            # Brain playback needs regional frame metrics for the shared exposure
            # panel and Analysis workspace, just as the former modal path did.
            settings["precompute_brain_region_frames"] = True
        return settings

    @staticmethod
    def _scripted_spin(
        value: float, minimum: float, maximum: float, decimals: int, suffix: str
    ) -> QDoubleSpinBox:
        spin = CompactDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setValue(float(value))
        spin.setSuffix(str(suffix))
        spin.setKeyboardTracking(False)
        return spin

    def _initialize_playback_protocol(self) -> None:
        """Create the hidden renderer controller and compact Scripted source UI.

        Maps no longer owns a full exposure editor. Simple generated waveforms are
        configured here as quick one-off map sources; project waveforms, voltage/RL
        drive models, and sequencing live in saved Exposures. Output/observer trace
        choices are viewer-local and live at the bottom of Scripted mode, below
        Camera. The established WaveformPlaybackDialog remains hidden as the
        numerical request builder so this UI change does not fork the playback engine.
        """
        if self.mode != "playback" or self.playback_controller is not None:
            return
        page = self._pages.get(self.current_page_key)
        if page is None:
            return
        key = self.current_view_key
        controller = WaveformPlaybackDialog(
            self.snapshot_adapter,
            map_kind=key,
            map_settings=self._playback_map_settings(key, page),
            primary_action="playback",
            defer_payload_build=True,
            parent=self,
        )
        controller.setWindowFlags(Qt.WindowType.Widget)
        controller.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        controller.hide()
        self.playback_controller = controller

        # Restore familiar generated-waveform defaults and viewer-local trace
        # choices without exposing the old Drive/Waveform/Sequencer authoring
        # sections in Maps. The Traces disclosure itself is persistent too; its
        # constructor default remains collapsed when there is no saved state.
        state_filter = getattr(controller, "_persistent_window_state_filter", None)
        restore = getattr(state_filter, "_restore", None)
        if callable(restore):
            restore()

        host = QWidget(self)
        host.setObjectName("scriptedSourceHost")
        host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(2)

        section = CollapsibleSection("Scripted source", expanded=False)
        section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        section.content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Maximum)
        source_layout = QVBoxLayout()
        source_layout.setContentsMargins(0, 0, 0, 4)
        source_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        source_layout.setSpacing(6)

        source_form = QFormLayout()
        source_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        source_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.scripted_source_type = QComboBox()
        self.scripted_source_type.addItem("Built-in waveform", "builtin")
        self.scripted_source_type.addItem("Saved exposure", "exposure")
        source_form.addRow("Source", self.scripted_source_type)
        source_layout.addLayout(source_form)

        self.scripted_source_stack = _CurrentPageStackedWidget()
        source_stack_policy = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        source_stack_policy.setHeightForWidth(True)
        self.scripted_source_stack.setSizePolicy(source_stack_policy)

        built_in = QWidget()
        built_in.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        built_form = QFormLayout(built_in)
        built_form.setContentsMargins(0, 0, 0, 0)
        built_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        built_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        built_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.scripted_waveform = QComboBox()
        self.scripted_waveform.addItem("Sine", "sine")
        self.scripted_waveform.addItem("Square", "square")
        self.scripted_waveform.addItem("Triangle", "triangle")
        try:
            restored_mode = str(controller._waveform_mode()).strip().lower()
        except Exception:
            restored_mode = "sine"
        restored_index = self.scripted_waveform.findData(restored_mode)
        if restored_index >= 0:
            self.scripted_waveform.setCurrentIndex(restored_index)

        starter_amplitude = 10.0
        try:
            starter_amplitude = max(0.0, float(controller._starter_amplitude_value()))
        except Exception:
            pass
        self.scripted_amplitude = self._scripted_spin(starter_amplitude, 0.0, 1.0e9, 6, " mA")
        self.scripted_frequency = self._scripted_spin(
            float(controller.waveform_frequency.value()), 1.0e-6, 1.0e6, 6, " Hz"
        )
        self.scripted_duration = self._scripted_spin(
            float(controller.waveform_duration.value()), 1.0e-6, 1.0e9, 6, " s"
        )
        # 1 mA/ms == 1 A/s, so the displayed value equals the SI slew number.
        try:
            slew_value = float(controller._slew_rate_per_second())
        except Exception:
            slew_value = 100.0
        self.scripted_slew = self._scripted_spin(slew_value, 0.0, 1.0e12, 6, " mA/ms")
        built_form.addRow("Waveform", self.scripted_waveform)
        built_form.addRow("Amplitude", self.scripted_amplitude)
        built_form.addRow("Frequency", self.scripted_frequency)
        built_form.addRow("Duration", self.scripted_duration)
        built_form.addRow("Slew limit", self.scripted_slew)
        built_hint = QLabel(
            "Quick waveforms use the active coils selected in Scene options. "
            "Reusable/custom/imported signals are created in Waveform Editor; timed sequencing and voltage/RL source-to-load wiring are created in Exposure Editor."
        )
        built_hint.setObjectName("hint")
        built_hint.setWordWrap(True)
        built_form.addRow(built_hint)
        self.scripted_source_stack.addWidget(built_in)

        exposure_panel = QWidget()
        exposure_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        exposure_form = QFormLayout(exposure_panel)
        exposure_form.setContentsMargins(0, 0, 0, 0)
        exposure_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        exposure_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        exposure_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.scripted_exposure = QComboBox()
        summaries = self.snapshot_adapter.list_exposures()
        for item in summaries:
            exposure_id = str(item.get("id", ""))
            if exposure_id:
                self.scripted_exposure.addItem(str(item.get("name", "Exposure")), exposure_id)
        exposure_form.addRow("Exposure", self.scripted_exposure)
        self.scripted_exposure_duration = self._scripted_spin(2.0, 1e-6, 1e9, 6, " s")
        self.scripted_exposure_duration.setToolTip("Preview from time zero, limited to the physical exposure duration.")
        exposure_form.addRow("Duration", self.scripted_exposure_duration)

        self.scripted_exposure_summary = _BreathingWordWrapLabel()
        self.scripted_exposure_summary.setContentsMargins(0, 4, 0, 0)
        self.scripted_exposure_summary.setObjectName("hint")
        exposure_form.addRow(self.scripted_exposure_summary)
        if self.scripted_exposure.count() == 0:
            self.scripted_exposure.setEnabled(False)
            self.scripted_exposure_summary.setText(
                "No saved exposures are stored in this Maps snapshot. Create one in Exposure Editor, then reopen Maps."
            )
        self.scripted_source_stack.addWidget(exposure_panel)
        source_layout.addWidget(self.scripted_source_stack)
        section.setContentLayout(source_layout)
        host_layout.addWidget(section)
        self.scripted_source_section = section
        self.scripted_source_host = host

        self.scripted_source_type.currentIndexChanged.connect(self._scripted_source_changed)
        self.scripted_exposure.currentIndexChanged.connect(self._update_scripted_exposure_summary)
        self._scripted_source_changed()
        self._update_scripted_exposure_summary()
        self._attach_playback_protocol(key, page)

    def _scripted_source_changed(self, *_args) -> None:
        if self.scripted_source_type is None or self.scripted_source_stack is None:
            return
        kind = str(self.scripted_source_type.currentData() or "builtin")
        self.scripted_source_stack.setCurrentIndex(1 if kind == "exposure" else 0)
        # Do not keep the inactive page's preferred height. In particular, the
        # Saved exposure page should collapse to its short selector/summary rather
        # than reserving the built-in waveform editor's empty vertical space.
        self.scripted_source_stack.updateGeometry()
        current = self.scripted_source_stack.currentWidget()
        if isinstance(current, QWidget):
            current.updateGeometry()
        if self.scripted_source_section is not None:
            self.scripted_source_section.updateGeometry()
        if self.scripted_source_host is not None:
            self.scripted_source_host.updateGeometry()

    def _update_scripted_exposure_summary(self, *_args) -> None:
        label = self.scripted_exposure_summary
        combo = self.scripted_exposure
        if label is None or combo is None or combo.count() == 0:
            return
        exposure_id = str(combo.currentData() or "")
        try:
            item = next(
                summary for summary in self.snapshot_adapter.list_exposures()
                if str(summary.get("id", "")) == exposure_id
            )
        except StopIteration:
            label.setText("The selected exposure is no longer available in this Maps snapshot.")
            return
        self.scripted_exposure_duration.setMaximum(max(1e-6, float(item.get("duration_s", 1.0))))
        missing_text = ""
        try:
            exposure = self.snapshot_adapter.exposure(exposure_id)
            protocol = exposure.get("protocol", {}) if isinstance(exposure, dict) else {}
            referenced = set(exposure_coil_ids(protocol))
            available = {
                str(choice.get("id"))
                for choice in getattr(self.playback_controller, "_drive_choices", [])
                if str(choice.get("id", ""))
            }
            missing = sorted(value for value in referenced if value and value not in available)
            if missing:
                labels = protocol.get("coil_labels", {}) if isinstance(protocol.get("coil_labels"), dict) else {}
                names = [str(labels.get(value) or value) for value in missing]
                missing_text = " Missing coil reference(s): " + ", ".join(names) + "."
        except Exception:
            pass
        sequence_text = (
            f"{int(item.get('step_count', 0))} step(s), "
            f"{int(item.get('drive_group_count', 0))} drive group(s)"
        )
        label.setText(
            f"{item.get('waveform', 'Waveform')} • {sequence_text} • "
            f"{int(item.get('coil_count', 0))} coil(s) • "
            f"Physical duration {float(item.get('duration_s', 0.0)):.6g} s.{missing_text}"
        )

    def _attach_playback_protocol(self, key: str, page: QDialog) -> None:
        """Place Scripted source before Camera and viewer-local Traces after it."""
        if self.mode != "playback" or self.scripted_source_host is None:
            return
        layout = getattr(page, "control_root", None)
        if not isinstance(layout, QVBoxLayout):
            return
        camera_section = getattr(page, "camera_section", None)
        camera_index = layout.indexOf(camera_section) if isinstance(camera_section, QWidget) else -1
        if camera_index >= 0:
            layout.insertWidget(camera_index, self.scripted_source_host)
        else:
            layout.insertWidget(max(0, layout.count() - 1), self.scripted_source_host)

        previous_filter = self._scripted_source_wheel_filter
        current_filter = getattr(page, "_control_wheel_filter", None)
        for editor in self.scripted_source_host.findChildren(QWidget):
            if not isinstance(editor, (QDoubleSpinBox, QSpinBox, QComboBox)):
                continue
            if previous_filter is not None and previous_filter is not current_filter:
                editor.removeEventFilter(previous_filter)
            if current_filter is not None and current_filter is not previous_filter:
                editor.installEventFilter(current_filter)
        self._scripted_source_wheel_filter = current_filter

        if self.playback_controller is not None:
            self.playback_controller.set_playback_context(
                map_kind=key,
                map_settings=self._playback_map_settings(key, page),
            )
            traces = self.playback_controller.traces_section
            # Recalculate after inserting Scripted source: Camera shifted down by
            # one slot. Traces is always the final scrolling section, immediately
            # after Camera and before the layout stretch.
            camera_index = layout.indexOf(camera_section) if isinstance(camera_section, QWidget) else -1
            if camera_index >= 0:
                layout.insertWidget(camera_index + 1, traces)
            else:
                layout.insertWidget(max(0, layout.count() - 1), traces)

            previous_trace_filter = self._scripted_traces_wheel_filter
            for editor in traces.findChildren(QWidget):
                if not isinstance(editor, (QDoubleSpinBox, QSpinBox, QComboBox)):
                    continue
                if previous_trace_filter is not None and previous_trace_filter is not current_filter:
                    editor.removeEventFilter(previous_trace_filter)
                if current_filter is not None and current_filter is not previous_trace_filter:
                    editor.installEventFilter(current_filter)
            self._scripted_traces_wheel_filter = current_filter

    def _apply_scripted_source_to_controller(self, key: str, page: QDialog) -> str:
        """Load the selected quick waveform or saved Exposure into the renderer controller."""
        controller = self.playback_controller
        if controller is None or self.scripted_source_type is None:
            return "Scripted"
        controller.set_playback_context(
            map_kind=key,
            map_settings=self._playback_map_settings(key, page),
        )
        source_kind = str(self.scripted_source_type.currentData() or "builtin")
        if source_kind == "exposure":
            if self.scripted_exposure is None or self.scripted_exposure.count() == 0:
                raise StudioOperationError("This model does not contain a saved exposure to play.")
            exposure_id = str(self.scripted_exposure.currentData() or "")
            exposure = self.snapshot_adapter.exposure(exposure_id)
            protocol = exposure.get("protocol", {}) if isinstance(exposure, dict) else {}
            referenced = set(exposure_coil_ids(protocol))
            available = {
                str(choice.get("id"))
                for choice in getattr(controller, "_drive_choices", [])
                if str(choice.get("id", ""))
            }
            missing = sorted(value for value in referenced if value and value not in available)
            if missing:
                labels = protocol.get("coil_labels", {}) if isinstance(protocol.get("coil_labels"), dict) else {}
                shown = [str(labels.get(value) or value) for value in missing]
                raise StudioOperationError(
                    "This exposure references coil(s) that are not in this Maps snapshot: "
                    + ", ".join(shown)
                    + ". Edit/remap the exposure or reopen Maps with the matching model."
                )
            sequence = protocol.get("sequence", {}) if isinstance(protocol, dict) else {}
            steps = sequence.get("steps", []) if isinstance(sequence, dict) else []
            if not steps:
                raise StudioOperationError("This exposure does not contain a drive step.")
            # Resolve once at the source/load boundary so missing waveform/load
            # data fails before the render job is queued.  The playback worker
            # then samples the same canonical exposure directly into per-coil
            # current timelines; nothing is projected back through the retired
            # single-source playback protocol.
            for step_index in range(len(steps)):
                self.snapshot_adapter.resolve_exposure_step_electrical(protocol, step_index)
            controller.set_runtime_exposure_protocol(
                protocol,
                metadata={
                    "exposure_id": exposure_id,
                    "exposure_name": str(exposure.get("name") or "Exposure"),
                    "description": str(exposure.get("description") or ""),
                    "preview_duration_s": float(self.scripted_exposure_duration.value()),
                },
            )
            return str(exposure.get("name") or "Exposure")

        if None in (
            self.scripted_waveform, self.scripted_amplitude, self.scripted_frequency,
            self.scripted_duration, self.scripted_slew,
        ):
            raise StudioOperationError("The built-in scripted source controls are unavailable.")
        settings = self._playback_map_settings(key, page)
        active_ids = [str(value) for value in settings.get("drive_coil_ids", []) if str(value)]
        available = {str(choice.get("id")): choice for choice in getattr(controller, "_drive_choices", [])}
        selected_ids = [value for value in active_ids if value in available]
        mode = str(self.scripted_waveform.currentData() or "sine")
        controller.configure_quick_current_source(
            mode=mode,
            amplitude_ma=float(self.scripted_amplitude.value()),
            frequency_hz=float(self.scripted_frequency.value()),
            duration_s=float(self.scripted_duration.value()),
            slew_ma_per_ms=float(self.scripted_slew.value()),
            active_coil_ids=selected_ids,
        )
        return str(self.scripted_waveform.currentText() or mode.title())

    def _create_shared_playback(self, *_args) -> None:
        """Build the current view from the selected quick waveform or saved Exposure."""
        if self.mode != "playback" or self.playback_controller is None:
            return
        key = self.current_view_key
        page = self._pages.get(self._page_key(self.mode, key))
        if page is None:
            return
        try:
            scene_objects = self.snapshot_adapter.list_objects()
        except Exception:  # noqa: BLE001 - let the normal render boundary report snapshot errors
            scene_objects = None
        if scene_objects == []:
            status = getattr(page, "calculation_status", None)
            if isinstance(status, QStatusBar):
                status.showMessage("Scene is empty — nothing to render.")
            if self.playback_controller is not None:
                self.playback_controller.status.setText(
                    "Scene is empty — nothing to render."
                )
            return
        request_camera = getattr(page, "_request_action_camera", None)
        if not callable(request_camera):
            return

        def launch(initial_camera: dict[str, Any]) -> None:
            if self._closing or self.playback_controller is None:
                return
            timeline_getter = getattr(page, "_camera_timeline_for_action", None)
            options_getter = getattr(page, "_camera_playback_options_for_action", None)
            camera_timeline = (
                timeline_getter(initial_camera) if callable(timeline_getter) else None
            )
            playback_fps, playback_speed = (
                options_getter() if callable(options_getter) else (60.0, 1.0)
            )
            controller = self.playback_controller
            try:
                self._apply_scripted_source_to_controller(key, page)
            except Exception as error:  # noqa: BLE001 - saved exposure/script boundary
                QMessageBox.warning(self, "Unable to create scripted map", str(error))
                return
            controller.set_playback_context(
                map_kind=key,
                map_settings=controller.map_settings,
                camera_timeline=camera_timeline,
                initial_camera=initial_camera,
                playback_fps=playback_fps,
                playback_speed=playback_speed,
            )
            prepared = controller.prepare_playback_request()
            if prepared is None:
                return
            request, coil_ids = prepared
            warning_text = playback_frame_warning_text(
                int(request.get("frame_count", 0) or 0),
                float(request.get("playback_fps", 60.0) or 60.0),
            )
            if warning_text is not None:
                answer = QMessageBox.question(
                    self,
                    "Large scripted playback",
                    warning_text,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    controller.status.setText("Playback cancelled before calculation")
                    return
            controller._set_result_label(coil_ids)
            result_label = controller.result_label
            controller.status.setText(
                f"{self.VIEW_LABELS.get(key, key)} playback queued — preparing field cache…"
            )

            page._waveform_tab_counter += 1
            label_getter = getattr(page, "_workspace_result_tab_label", None)
            job_label = (
                label_getter(
                    page._waveform_tab_counter,
                    f"Waveform {page._waveform_tab_counter}",
                )
                if callable(label_getter)
                else f"{self.VIEW_LABELS.get(key, key)}-{page._waveform_tab_counter}"
            )
            if key == "brain":
                start = getattr(page, "_start_brain_render_job", None)
                tooltip = (
                    result_label
                    + " — LPBA40 field map and checked-region spatial RMS traces"
                )
            else:
                start = getattr(page, "_start_render_job", None)
                tooltip = result_label + " — interactive playback"
            if callable(start):
                start(
                    render_kind="waveform",
                    request=request,
                    job_label=job_label,
                    tooltip=tooltip,
                )

        request_camera(launch)

    def _adopt_page_controls(self, page_key: str, page: QDialog) -> None:
        """Move one established viewer's control column into the shared sidebar.

        Standalone map dialogs deliberately reserve ~300 px for their controls.
        Inside the unified Maps window that inherited minimum prevented the
        outer splitter from responding to a smaller window, so neutralize the
        standalone floor here only.  The scroll area then wraps/scrolls its
        contents instead of forcing the whole Maps window wider.
        """
        controls = getattr(page, "control_column", None)
        if not isinstance(controls, QWidget):
            return
        controls.setMinimumWidth(0)
        controls.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        control_scroll = getattr(page, "control_scroll", None)
        if isinstance(control_scroll, QScrollArea):
            control_scroll.setMinimumWidth(0)
            control_scroll.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
            )
        control_panel = getattr(page, "control_panel", None)
        if isinstance(control_panel, QWidget):
            control_panel.setMinimumWidth(0)
            control_panel.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
        self.control_stack.addWidget(controls)
        self._control_pages[page_key] = controls

        inner_splitter = getattr(page, "workspace_splitter", None)
        if isinstance(inner_splitter, QSplitter):
            inner_splitter.setChildrenCollapsible(False)

    @staticmethod
    def _result_tab_owner(widget: QWidget | None) -> tuple[str, str]:
        if not isinstance(widget, QWidget):
            return "", ""
        try:
            mode = str(widget.property("fieldWorkspaceOwnerMode") or "")
            view = str(widget.property("fieldWorkspaceOwnerView") or "")
        except RuntimeError:
            return "", ""
        return mode, view

    def _scripted_result_descriptor(self) -> str:
        """Return the quick waveform or saved Exposure name for result tabs."""
        if self.scripted_source_type is None:
            return "Scripted"
        source_kind = str(self.scripted_source_type.currentData() or "builtin")
        if source_kind == "exposure" and self.scripted_exposure is not None:
            text = str(self.scripted_exposure.currentText()).strip()
            return text or "Exposure"
        if self.scripted_waveform is not None:
            text = str(self.scripted_waveform.currentText()).strip()
            if text:
                return text
        return "Scripted"

    def _workspace_result_label(
        self,
        mode: str,
        view_key: str,
        counter: int,
        fallback: str,
    ) -> str:
        """Allocate ``View\\Source - N`` labels for the shared Maps tab strip."""

        normalized_mode = str(mode).strip().lower()
        normalized_view = str(view_key).strip().lower()
        identity = (normalized_mode, normalized_view, int(counter))
        existing = self._result_label_map.get(identity)
        if existing:
            return existing
        view_label = self.VIEW_LABELS.get(normalized_view)
        if not view_label:
            return str(fallback)
        descriptor = (
            "Static"
            if normalized_mode == "static"
            else self._scripted_result_descriptor()
            if normalized_mode == "playback"
            else self.MODE_LABELS.get(normalized_mode, normalized_mode.title())
        )
        descriptor = " ".join(str(descriptor).split()) or "Result"
        sequence_key = (normalized_view, descriptor.casefold())
        sequence = self._result_label_counters.get(sequence_key, 0) + 1
        self._result_label_counters[sequence_key] = sequence
        label = f"{view_label}\\{descriptor} - {sequence}"
        self._result_label_map[identity] = label
        return label

    def _adopt_page_tabs(
        self,
        page_key: str,
        page_mode: str,
        view_key: str,
        page: QDialog,
    ) -> None:
        """Give one mode/view page the Maps-wide Selector/result tab system."""

        old_tabs = getattr(page, "tabs", None)
        if not isinstance(old_tabs, QTabWidget):
            return
        selector_index = int(getattr(page, "_selector_tab_index", 0))
        selector_widget = old_tabs.widget(selector_index)
        viewer_panel = old_tabs.parentWidget()
        viewer_layout = viewer_panel.layout() if isinstance(viewer_panel, QWidget) else None
        if not isinstance(viewer_layout, QVBoxLayout) or not isinstance(selector_widget, QWidget):
            return

        blocker = QSignalBlocker(old_tabs)
        old_tabs.removeTab(selector_index)
        del blocker
        viewer_layout.removeWidget(old_tabs)
        old_tabs.hide()
        old_tabs.deleteLater()

        selector_widget.setProperty("fieldWorkspaceSelectorMode", page_mode)
        selector_widget.setProperty("fieldWorkspaceSelectorView", view_key)
        self.selector_stack.addWidget(selector_widget)
        self._selector_pages[page_key] = selector_widget
        self._viewer_layouts[page_key] = viewer_layout

        page.tabs = self.shared_tabs
        page._selector_tab_index = self._shared_selector_tab_index
        page._unified_workspace_mode_key = page_mode
        page._unified_workspace_view_key = view_key
        page._unified_workspace_result_prefix = self.VIEW_LABELS[view_key]
        page._unified_workspace_result_label_provider = (
            lambda counter, fallback, m=page_mode, v=view_key:
            self._workspace_result_label(m, v, counter, fallback)
        )

    def _attach_shared_tabs(self, page_key: str) -> None:
        """Place the shared tab/measurement shell above the active renderer."""

        layout = self._viewer_layouts.get(page_key)
        if not isinstance(layout, QVBoxLayout):
            return
        host = self.measurement_shell
        target_parent = layout.parentWidget()
        current_parent = host.parentWidget()
        if current_parent is not target_parent:
            if isinstance(current_parent, QWidget):
                current_layout = current_parent.layout()
                if current_layout is not None:
                    current_layout.removeWidget(host)
            if isinstance(target_parent, QWidget):
                host.setParent(target_parent)
        if layout.indexOf(host) != 0:
            layout.insertWidget(0, host, 1)

    def _active_tab_has_embedded_region_panel(self) -> bool:
        # Only Brain owns an embedded region panel in the unified workspace.
        # Scripted 2D/3D intentionally has no Measurement-volumes panel; those
        # volumes are a Static snapshot workflow.
        if self.current_view_key != "brain":
            return False
        widget = self.shared_tabs.currentWidget()
        if isinstance(widget, ExposureResultTab):
            return True
        return (
            self.shared_tabs.currentIndex() == self._shared_selector_tab_index
        )

    def _track_workspace_panel_splitter(self, splitter: QSplitter | None) -> None:
        """Let any right-panel splitter update the Maps-wide panel geometry."""

        if not isinstance(splitter, QSplitter):
            return
        if bool(splitter.property("mapsPanelStateTracked")):
            return
        splitter.setProperty("mapsPanelStateTracked", True)
        splitter.splitterMoved.connect(
            lambda _position, _index, tracked=splitter: self._workspace_panel_splitter_moved(tracked)
        )

    def _workspace_panel_splitter_moved(self, splitter: QSplitter) -> None:
        if self._applying_measurement_panel_state:
            return
        sizes = splitter.sizes()
        if len(sizes) < 2:
            return
        right = int(sizes[1])
        if right <= 0:
            self._measurement_panel_visible = False
        else:
            self._measurement_panel_visible = True
            self._measurement_panel_width = max(self._measurement_panel_min_width, right)
        self._measurement_panel_visibility_by_view[self.current_view_key] = (
            self._measurement_panel_visible
        )
        self._sync_workspace_regions_button()
        QTimer.singleShot(0, self._sync_workspace_footer_geometry)

    def _workspace_splitter_moved(self, _position: int, _index: int) -> None:
        """Remember a deliberate sidebar width without fighting responsive clamps."""
        if self._applying_workspace_splitter_state:
            return
        sizes = self.workspace_splitter.sizes()
        if len(sizes) >= 2 and int(sizes[0]) > 0:
            self._sidebar_width = max(self._sidebar_min_width, int(sizes[0]))

    def _apply_workspace_splitter_state(self) -> None:
        """Clamp sidebar/panel widths when the Maps window becomes narrow."""
        splitter = getattr(self, "workspace_splitter", None)
        if not isinstance(splitter, QSplitter):
            return
        sizes = splitter.sizes()
        total = max(1, int(splitter.width()), int(sum(sizes)) if sizes else 0)
        max_sidebar = max(self._sidebar_min_width, total - self._workspace_view_min_width)
        sidebar = min(max(self._sidebar_min_width, self._sidebar_width), max_sidebar)
        self._applying_workspace_splitter_state = True
        try:
            splitter.setSizes([sidebar, max(1, total - sidebar)])
        finally:
            self._applying_workspace_splitter_state = False

        active_panel = self._active_workspace_panel_splitter()
        if isinstance(active_panel, QSplitter):
            self._set_panel_splitter_state(
                active_panel, visible=self._measurement_panel_visible
            )
        QTimer.singleShot(0, self._sync_workspace_footer_geometry)

    def _set_panel_splitter_state(
        self,
        splitter: QSplitter | None,
        *,
        visible: bool,
    ) -> None:
        if not isinstance(splitter, QSplitter):
            return
        self._track_workspace_panel_splitter(splitter)
        sizes = splitter.sizes()
        total = max(
            1,
            int(splitter.width()),
            int(sum(sizes)) if sizes else 0,
        )
        self._applying_measurement_panel_state = True
        try:
            if visible:
                # Preserve the user's preferred panel width while there is room,
                # but shrink the panel before sacrificing the centre viewer.
                # The unified window minimum is chosen so this still leaves at
                # least the normal panel minimum at its narrowest supported size.
                max_right = max(
                    self._measurement_panel_min_width,
                    total - self._measurement_view_min_width,
                )
                right = min(
                    max(self._measurement_panel_min_width, self._measurement_panel_width),
                    max_right,
                )
                splitter.setSizes([max(1, total - right), right])
            else:
                splitter.setSizes([total, 0])
        finally:
            self._applying_measurement_panel_state = False

    def _active_workspace_panel_splitter(self) -> QSplitter | None:
        # Footer geometry updates are queued from splitter moves.  During
        # window teardown one of those zero-delay callbacks can run after Qt
        # has already deleted the shared tab widget, leaving only the Python
        # wrapper alive.  Treat that as "no active panel" rather than letting
        # libshiboken raise from the queued cleanup callback.
        try:
            widget = self.shared_tabs.currentWidget()
            current_index = self.shared_tabs.currentIndex()
        except RuntimeError:
            return None
        if self.current_view_key == "brain" and isinstance(widget, ExposureResultTab):
            return widget.splitter
        if (
            current_index == self._shared_selector_tab_index
            and self.current_view_key == "brain"
        ):
            page = self._pages.get(self.current_page_key)
            splitter = getattr(page, "selector_splitter", None) if page is not None else None
            if isinstance(splitter, QSplitter):
                return splitter
        if self.mode == "static" and self.current_view_key in {"2d", "3d"}:
            return self.measurement_view_splitter
        return None

    def _sync_workspace_footer_geometry(self) -> None:
        """Align the footer's right zone with whichever region panel is active."""
        try:
            footer = getattr(self, "workspace_footer", None)
            panel_footer = getattr(self, "workspace_panel_footer", None)
            panel_button = getattr(self, "workspace_regions_button", None)
            if not isinstance(footer, QWidget) or not isinstance(panel_footer, QWidget):
                return
            if not isinstance(panel_button, QPushButton) or not panel_button.isVisible():
                panel_footer.setVisible(False)
                return
            panel_footer.setVisible(True)
            splitter = self._active_workspace_panel_splitter()
            sizes = splitter.sizes() if isinstance(splitter, QSplitter) else []
            right = int(sizes[1]) if len(sizes) >= 2 else 0
            if right > 0:
                target = right
            else:
                # Keep the reveal action reachable after the panel has been fully
                # collapsed without reserving the old panel's entire width.
                target = (panel_button.sizeHint().width() + 16) if isinstance(panel_button, QPushButton) else 150
            available = max(1, footer.width() - 1)
            panel_footer.setFixedWidth(max(1, min(target, available)))
        except RuntimeError:
            # A queued geometry sync is harmless once the Maps workspace has
            # begun deleting its Qt children.
            return

    def _sync_workspace_regions_button(self) -> None:
        """Keep the shared panel toggle explicit about the action it will take."""
        button = getattr(self, "workspace_regions_button", None)
        if not isinstance(button, QPushButton):
            return
        embedded_brain = self._active_tab_has_embedded_region_panel()
        measurement_available = (
            self.mode == "static" and self.current_view_key in {"2d", "3d"}
        )
        brain_available = self.current_view_key == "brain" and embedded_brain
        button.setVisible(measurement_available or brain_available)
        if not (measurement_available or brain_available):
            return
        noun = "brain areas" if brain_available else "measurements"
        if self._measurement_panel_visible:
            button.setText(f"Hide {noun}")
            button.setToolTip(
                f"Hide the right-hand {noun} panel for the active Maps view"
            )
        else:
            button.setText(f"Show {noun}")
            button.setToolTip(
                f"Show the right-hand {noun} panel for the active Maps view"
            )

    def _sync_workspace_measurement_panel(self) -> None:
        self._sync_workspace_regions_button()
        # Measurement volumes are Static-only. Brain uses its own embedded
        # anatomical panel in both Static and Scripted modes. During a Brain
        # render-progress tab there is deliberately no fallback Measurement
        # panel: otherwise the progress screen looks like a paused playback can
        # be analyzed/saved as a static snapshot.
        embedded = self._active_tab_has_embedded_region_panel()
        show_outer_measurements = (
            self.mode == "static"
            and self.current_view_key in {"2d", "3d"}
            and not embedded
        )
        outer_host = getattr(self, "measurement_panel_host", None)
        if isinstance(outer_host, QWidget):
            # Hiding the QSplitter child removes its handle as well. A zero-size
            # visible child is still used for Static 2D/3D so users can drag the
            # divider back open after deliberately collapsing Measurements.
            outer_host.setVisible(show_outer_measurements)
        self._set_panel_splitter_state(
            self.measurement_view_splitter,
            visible=self._measurement_panel_visible and show_outer_measurements,
        )
        if embedded:
            self._set_panel_splitter_state(
                self._active_workspace_panel_splitter(),
                visible=self._measurement_panel_visible,
            )
        # Individual map pages still own this compatibility button for their
        # standalone dialogs; inside the unified workspace the one shared
        # footer control is the only panel toggle shown.
        page = self._pages.get(self.current_page_key)
        region_button = getattr(page, "regions_button", None) if page is not None else None
        if isinstance(region_button, QPushButton):
            region_button.setVisible(False)
        QTimer.singleShot(0, self._sync_workspace_footer_geometry)

    def _toggle_workspace_regions(self, *_args) -> None:
        self._measurement_panel_visible = not self._measurement_panel_visible
        self._measurement_panel_visibility_by_view[self.current_view_key] = (
            self._measurement_panel_visible
        )
        self._sync_workspace_measurement_panel()

    def _close_shared_tab(self, index: int) -> None:
        """Route a Maps result-tab close request to the mode/view that owns it."""

        if index == self._shared_selector_tab_index:
            return
        widget = self.shared_tabs.widget(index)
        owner_mode, owner_view = self._result_tab_owner(widget)
        page = self._pages.get(self._page_key(owner_mode, owner_view))
        closer = getattr(page, "_close_map_tab", None) if page is not None else None
        if callable(closer):
            closer(index)
            return
        self.shared_tabs.removeTab(index)
        if isinstance(widget, QWidget):
            widget.deleteLater()

    def _shared_tab_changed(self, index: int) -> None:
        """Restore the mode/view that owns a selected result tab."""

        if self._handling_shared_tab_change or index < 0:
            return
        self._handling_shared_tab_change = True
        try:
            if index != self._shared_selector_tab_index:
                owner_mode, owner_view = self._result_tab_owner(
                    self.shared_tabs.widget(index)
                )
                owner_key = self._page_key(owner_mode, owner_view)
                if owner_key in self._pages and owner_key != self.current_page_key:
                    self._switch_mode(
                        owner_mode,
                        view_key=owner_view,
                        select_selector=False,
                    )
            page = self._pages.get(self.current_page_key)
            handler = getattr(page, "_render_tab_changed", None) if page is not None else None
            if callable(handler):
                handler(index)
            self._sync_workspace_measurement_panel()
            self._refresh_saved_map_actions()
        finally:
            self._handling_shared_tab_change = False

    def _create_page(
        self,
        view_key: str,
        *,
        mode: str | None = None,
    ) -> QDialog | None:
        page_mode = str(mode or self.mode).strip().lower()
        if page_mode not in {"static", "playback"}:
            return None
        page_key = self._page_key(page_mode, view_key)
        if page_key in self._pages:
            return self._pages[page_key]

        if view_key == "2d":
            page = FieldMapDialog(
                self.snapshot_adapter,
                self,
                workspace_mode=page_mode,
            )
        elif view_key == "3d":
            page = FieldVolumeMapDialog(
                self.snapshot_adapter,
                self,
                workspace_mode=page_mode,
            )
        elif view_key == "brain":
            if self.brain_object_id is None:
                return None
            try:
                page = BrainViewDialog(
                    self.snapshot_adapter,
                    self.brain_object_id,
                    self,
                    workspace_mode=page_mode,
                )
            except Exception as error:  # noqa: BLE001 - bundled atlas/UI boundary
                QMessageBox.warning(self, "Brain view", str(error))
                return None
        else:
            raise ValueError(f"Unknown field view: {view_key!r}")

        page.setWindowFlags(Qt.WindowType.Widget)
        page.setWindowModality(Qt.WindowModality.NonModal)
        page.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        page.setObjectName(f"fieldWorkspacePage_{page_mode}_{view_key}")

        self._collapse_workspace_sections(page)
        self._adopt_page_controls(page_key, page)
        self._adopt_page_tabs(page_key, page_mode, view_key, page)
        self._install_saved_map_button(page)

        region_button = getattr(page, "regions_button", None)
        if isinstance(region_button, QPushButton):
            try:
                region_button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
            region_button.setText("Show measurements")
            region_button.setToolTip(
                "Show the right-hand measurement panel for the active Maps view"
            )
            region_button.clicked.connect(self._toggle_workspace_regions)
            region_button.setVisible(False)

        fullscreen_button = getattr(page, "fullscreen_button", None)
        if isinstance(fullscreen_button, QPushButton):
            fullscreen_button.setVisible(False)

        exit_button = getattr(page, "exit_button", None)
        if isinstance(exit_button, QPushButton):
            try:
                exit_button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
            exit_button.setText("Close Maps")
            exit_button.setToolTip("Close this Maps workspace")
            exit_button.clicked.connect(self.close)

        if page_mode == "playback":
            waveform_button = getattr(page, "waveform_button", None)
            if isinstance(waveform_button, QPushButton):
                try:
                    waveform_button.clicked.disconnect()
                except (RuntimeError, TypeError):
                    pass
                waveform_button.setText("Create playback")
                waveform_button.setToolTip(
                    "Create playback in the active view using the selected built-in waveform or saved exposure"
                )
                waveform_button.clicked.connect(self._create_shared_playback)

        self._pages[page_key] = page
        self.view_stack.addWidget(page)
        if self._launch_view_state is not None:
            plot = getattr(page, "plot", None)
            apply_state = getattr(plot, "apply_view_state", None)
            if callable(apply_state):
                apply_state(dict(self._launch_view_state))
        return page

    def _sync_view_selector(self, active_key: str) -> None:
        for key, button in self.view_buttons.items():
            button.setChecked(key == active_key)

    def _switch_mode(
        self,
        mode: str,
        *,
        view_key: str | None = None,
        select_selector: bool = True,
    ) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"static", "playback"}:
            return
        desired_view = view_key or self.current_view_key or "2d"
        self.mode = normalized
        self._sync_mode_selector(normalized)
        self._switch_view(desired_view, select_selector=select_selector)
        if normalized == "playback" and self.playback_controller is None:
            self._initialize_playback_protocol()
            page = self._pages.get(self.current_page_key)
            if page is not None and self.scripted_source_host is not None:
                self._attach_playback_protocol(desired_view, page)

    def _switch_view(self, key: str, *, select_selector: bool = True) -> None:
        page = self._create_page(key, mode=self.mode)
        if page is None:
            self._sync_view_selector(self.current_view_key)
            return
        page_key = self._page_key(self.mode, key)
        self.view_stack.setCurrentWidget(page)
        self._measurement_panel_visible = bool(
            self._measurement_panel_visibility_by_view.get(key, key == "brain")
        )
        controls = self._control_pages.get(page_key)
        if controls is not None:
            self.control_stack.setCurrentWidget(controls)
        selector = self._selector_pages.get(page_key)
        if selector is not None:
            self.selector_stack.setCurrentWidget(selector)
        self._attach_shared_tabs(page_key)
        self._sync_mode_selector(self.mode)
        self._sync_view_selector(key)
        if self.mode == "playback":
            if self.playback_controller is None:
                self._initialize_playback_protocol()
            if self.scripted_source_host is not None:
                self._attach_playback_protocol(key, page)

        # Any explicit Static/Scripted or 2D/3D/Brain selector change returns to
        # the shared Selector. Selecting an existing result uses
        # select_selector=False and restores that result's selectors instead.
        if select_selector:
            if self.shared_tabs.currentIndex() != self._shared_selector_tab_index:
                self.shared_tabs.setCurrentIndex(self._shared_selector_tab_index)
            else:
                handler = getattr(page, "_render_tab_changed", None)
                if callable(handler):
                    handler(self._shared_selector_tab_index)
            plot = getattr(page, "plot", None)
            refresh = getattr(plot, "refresh_viewport", None)
            if callable(refresh):
                QTimer.singleShot(0, refresh)
        self._sync_workspace_measurement_panel()
        self._refresh_saved_map_actions()

    def apply_launch_view_state(self, state: dict[str, Any] | None) -> None:
        """Apply the main-scene camera state to present/future selector previews."""
        if not isinstance(state, dict):
            return
        self._launch_view_state = dict(state)
        for page in self._pages.values():
            plot = getattr(page, "plot", None)
            apply_state = getattr(plot, "apply_view_state", None)
            if callable(apply_state):
                apply_state(dict(state))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        # QSplitter keeps non-stretching panes close to their previous sizes.
        # Re-clamp them after a window resize so the centre viewer, not an old
        # sidebar/right-panel preference, remains the useful part of a small
        # Maps window.  Queue this until Qt has applied the new layout geometry.
        QTimer.singleShot(0, self._apply_workspace_splitter_state)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._closing:
            super().closeEvent(event)
            return
        self._closing = True
        for dialog in tuple(self._result_windows.values()):
            try:
                dialog.close()
            except RuntimeError:
                pass
        self._result_windows.clear()
        if self.playback_controller is not None:
            # Keep the compact Built-in waveform controls synchronized with the
            # hidden controller when that source is active. A selected saved
            # Exposure must never overwrite those quick-waveform defaults.
            source_kind = (
                str(self.scripted_source_type.currentData() or "builtin")
                if self.scripted_source_type is not None
                else "builtin"
            )
            if source_kind == "builtin":
                page = self._pages.get(self._page_key("playback", self.current_view_key))
                if page is None:
                    page = next(
                        (
                            candidate
                            for key, candidate in self._pages.items()
                            if key.startswith("playback:")
                        ),
                        None,
                    )
                if page is not None:
                    view_key = str(page.objectName()).rsplit("_", 1)[-1]
                    if view_key not in self.VIEW_LABELS:
                        view_key = "2d"
                    try:
                        self._apply_scripted_source_to_controller(view_key, page)
                    except Exception:
                        pass

            # Trace choices and the Traces disclosure are viewer-local settings,
            # independent of whether the active source is Built-in or a saved
            # Exposure. Save them on every Maps close. Previously an Exposure
            # source skipped this save path, so those choices appeared to reset.
            state_filter = getattr(
                self.playback_controller, "_persistent_window_state_filter", None
            )
            save = getattr(state_filter, "_save", None)
            if callable(save):
                save()
        # Close shared result tabs through their owning view first. This keeps
        # cancellation and renderer cleanup exactly where the established 2D/3D/
        # Brain code expects it, while the common Selector tab remains permanent
        # for the lifetime of the workspace. Suppress cross-view selection churn
        # while tabs disappear during shutdown.
        self._handling_shared_tab_change = True
        while self.shared_tabs.count() > 1:
            self._close_shared_tab(self.shared_tabs.count() - 1)

        # Embedded dialogs still own their selector renderer callbacks, timers,
        # and background-job controllers. Close them while the unified top-level
        # native surface is alive, then explicitly release each selector plot:
        # the selector widgets were re-parented into the shared Selector stack,
        # so they are no longer encountered by the per-dialog tab cleanup.
        for page in tuple(self._pages.values()):
            try:
                page.close()
            except RuntimeError:
                pass
            plot = getattr(page, "plot", None)
            cleanup = getattr(plot, "cleanup", None)
            if callable(cleanup):
                cleanup()
        super().closeEvent(event)



class MainWindow(QMainWindow):
    """The Field Workbench model/scene editor.

    The editor owns model-building state for exactly one project. Project
    lifetime, project tabs, saving, and launches of the other suite applications
    belong to Project Manager.
    """

    RECENT_FILES_KEY = "files/recentScenes"
    MAX_RECENT_FILES = 10

    project_state_changed = Signal()
    suite_new_project_adapter_requested = Signal(object)

    def __init__(
        self,
        adapter: StudioAdapter | None = None,
        settings: QSettings | None = None,
        *,
        suite_managed: bool = False,
        initial_path: str | Path | None = None,
        project_context: "ProjectContext | None" = None,
    ):
        super().__init__()
        install_persistent_window_settings(self)
        self.settings = settings if settings is not None else QSettings()
        self._suite_managed = bool(suite_managed)
        self.project_context = project_context
        self._suite_project_closing = False
        self._untitled_counter = 1
        resolved_initial_path = (
            None
            if initial_path is None
            else Path(initial_path).expanduser().resolve()
        )
        initial_adapter = adapter or StudioAdapter()
        self._document = SceneDocument(
            adapter=initial_adapter,
            untitled_number=self._untitled_counter,
            document_id=(
                project_context.project_id
                if self._suite_managed and project_context is not None
                else uuid4().hex
            ),
            path=resolved_initial_path,
            saved_signature=initial_adapter.signature(),
        )
        # Application-wide solver/resource preferences are owned by the suite
        # Project Manager.  Apply them to this editor's live adapter on creation;
        # viewer-specific fluxline cleanup is configured inside Maps instead.
        apply_global_preferences_to_adapter(self.adapter, settings=self.settings)
        configure_field_render_worker_limit(self.adapter.optimizer_worker_count)
        self.saved_signature = self.adapter.signature()
        self._tree_items: dict[str, QTreeWidgetItem] = {}
        self._pending_export_path: Path | None = None
        self._independent_windows: dict[int, QDialog] = {}
        self._independent_window_documents: dict[int, str] = {}
        if self.project_context is not None:
            self.project_context.register_window("model_editor", self)
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
        if self._suite_managed:
            self.exit_action.setText("Close Model Editor")
        self._apply_theme()
        self.refresh_scene(select_id="coil")

    @property
    def active_document(self) -> SceneDocument:
        return self._document

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
        # Inspector editors represent values from the active scene, not reusable
        # application UI preferences.  Opt the whole dynamic form out of the
        # persistent-control system so startup can never restore a name/setting
        # from whichever object happened to be selected when FWB last closed.
        self.inspector_container.setProperty("persistUiState", False)
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
        self.measurement_action = QAction("Volume analysis", self)
        self.measurement_action.setToolTip(
            "Analyze one or more selected Measurement volumes and review spatial field statistics"
        )
        self.measurement_action.triggered.connect(self.analyze_selected_measurement)
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

        history_actions = [("undo", self.undo_action), ("redo", self.redo_action)]
        if not self._suite_managed:
            history_actions = [("open", self.open_action), ("save", self.save_action), *history_actions]
        for key, action in history_actions:
            _icon_button(key, action)

        analysis_shortcuts = (
            ("sensor", "Sensor plot", self.sensor_plot_action),
            ("volume", "Volume analysis", self.measurement_action),
            ("optimize", "Optimize", self.optimizer_action),
            ("check", "DRC", self.drc_action),
            ("notes", "Notes", self.notes_action),
        )
        for key, label, action in analysis_shortcuts:
            _analysis_button(key, label, action)

        toolbar.addWidget(toolbar_widget)
        self.main_toolbar_widget = toolbar_widget


        self.file_menu = self.menuBar().addMenu("&File")
        if not self._suite_managed:
            self.file_menu.addAction(self.new_action)
            self.file_menu.addSeparator()
            self.file_menu.addAction(self.open_action)
            self.recent_menu = self.file_menu.addMenu("Open Recent")
            self._refresh_recent_menu()
        self.import_objects_action = self.file_menu.addAction("Import objects from scene…")
        self.import_objects_action.setToolTip(
            "Choose objects from another saved .magpy.json project and copy them into this model"
        )
        self.import_objects_action.triggered.connect(self.import_objects_from_scene)
        self.file_menu.addSeparator()
        if not self._suite_managed:
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
        self.exit_action = self.file_menu.addAction(
            "Close Model Editor" if self._suite_managed else "Exit", self.close
        )

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        self.group_selected_action = edit_menu.addAction("Group selected objects")
        self.group_selected_action.setShortcut(QKeySequence("Ctrl+G"))
        self.group_selected_action.triggered.connect(self.group_selected)
        edit_menu.addAction("Clone selected", self.duplicate_selected, QKeySequence("Ctrl+D"))
        edit_menu.addAction("Delete selected", self.delete_selected, QKeySequence.StandardKey.Delete)
        if not self._suite_managed:
            edit_menu.addSeparator()
            preferences_action = edit_menu.addAction("Preferences…")
            preferences_action.triggered.connect(self.open_preferences)

        self.analysis_menu = self.menuBar().addMenu("&Analyze")
        self.analysis_menu.addAction(self.sensor_plot_action)
        self.analysis_menu.addAction(self.measurement_action)
        self.analysis_menu.addSeparator()
        self.analysis_menu.addAction(self.optimizer_action)
        self.analysis_menu.addAction(self.drc_action)
        self.analysis_menu.addSeparator()
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

    # --- project model document ---------------------------------------
    def _document_by_id(self, document_id: str) -> SceneDocument | None:
        document = self._document
        return document if document.document_id == str(document_id) else None

    def _document_for_path(self, path: str | Path) -> SceneDocument | None:
        document = self._document
        if document.path is None:
            return None
        return document if self._scene_path_key(document.path) == self._scene_path_key(path) else None

    @staticmethod
    def _document_title_text(document: SceneDocument) -> str:
        marker = " *" if document.is_dirty() else ""
        return f"{document.display_name()}{marker}"

    def _document_state_changed(self, document: SceneDocument) -> None:
        if document.document_id == self._document.document_id:
            self.project_state_changed.emit()

    def _activate_document_id(self, document_id: str, on_complete=None) -> None:
        if self._document.document_id == str(document_id) and on_complete is not None:
            on_complete()

    def _apply_saved_preferences_to_adapter(self, adapter: StudioAdapter) -> None:
        """Apply Project-Manager-owned application preferences to this project adapter."""
        apply_global_preferences_to_adapter(adapter, settings=self.settings)

    def _make_document(
        self,
        adapter: StudioAdapter,
        *,
        path: str | Path | None = None,
        selected_id: str | None = None,
        untitled_number: int | None = None,
    ) -> SceneDocument:
        resolved_path = None if path is None else Path(path).expanduser().resolve()
        if untitled_number is None:
            self._untitled_counter += 1
            untitled_number = self._untitled_counter
        return SceneDocument(
            adapter=adapter,
            untitled_number=untitled_number,
            path=resolved_path,
            saved_signature=adapter.signature(),
            selected_id=selected_id,
            selected_ids=[selected_id] if selected_id is not None else [],
        )

    def _replace_document(self, document: SceneDocument) -> None:
        old_document = self._document
        self._commit_scene_notes_session()
        self._close_document_windows(old_document.document_id)
        self._document = document
        self._sync_scene_notes_dialog()
        self.plot.exit_fullscreen()
        self.plot.request_home_on_next_figure()
        self.refresh_scene(select_id=document.selected_id)

    def close_active_document(self) -> None:
        self.close()

    def _close_document_windows(self, document_id: str) -> None:
        document = self._document_by_id(document_id)
        if document is not None:
            if document.drc_dialog is not None:
                document.drc_dialog.close()
            document.drc_dialog = None
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
        """Mirror a Plotly object/empty-space click into the scene tree selection."""
        object_id = str(object_id or "")
        if not object_id:
            selected = []
        elif object_id not in self._tree_items:
            return
        elif additive:
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
            self.tree.setCurrentItem(self._tree_items[primary] if primary is not None else None)
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
        self.coil_electrical_summary_label: QLabel | None = None
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

        correction_mode = QComboBox()
        correction_mode.addItem("Custom", "custom")
        correction_mode.addItem("Calibration", "calibration")
        correction_mode.setToolTip(
            "Use a manually entered correction factor or a Calibration / Composite fit managed in Dataset Manager."
        )
        correction_mode.setCurrentIndex(
            max(0, correction_mode.findData(properties.get("correction_mode", "custom")))
        )

        field_scale = _number_spin(
            properties["field_scale_factor"],
            decimals=6,
            minimum=0.000001,
            maximum=1_000_000.0,
            suffix=" ×",
        )
        field_scale.setSingleStep(0.05)
        field_scale.setToolTip(
            "Resolved dimensionless calibration multiplier applied to this coil's field source. "
            "In Custom mode this value is editable; Calibration mode derives it from Dataset Manager."
        )
        self._coil_custom_scale_value = float(
            properties.get("custom_field_scale_factor", properties["field_scale_factor"])
        )

        correction_calibration = QComboBox()
        correction_calibration.setObjectName("coilCorrectionCalibrationCombo")
        correction_calibration.setToolTip("Calibration / Composite fit supplying this coil's correction factor.")

        def populate_correction_source() -> None:
            previous_block = correction_calibration.blockSignals(True)
            try:
                correction_calibration.clear()
                linked_id = str(properties.get("correction_calibration_id", ""))
                if correction_mode.currentData() == "calibration":
                    for calibration in self.adapter.list_measurement_calibrations():
                        calibration_id = str(calibration.get("id", ""))
                        if not calibration_id:
                            continue
                        try:
                            factor = self.adapter.measurement_calibration_fit_factor(calibration_id)
                        except Exception:
                            continue
                        label = str(calibration.get("label") or calibration_id)
                        status = self.adapter.measurement_calibration_status(calibration_id)
                        if status != "Ready" and calibration_id != linked_id:
                            continue
                        suffix = "" if status == "Ready" else f" · {status}"
                        correction_calibration.addItem(f"{label}  ({factor:.7g} ×{suffix})", calibration_id)
                if linked_id:
                    index = correction_calibration.findData(linked_id)
                    if index >= 0:
                        correction_calibration.setCurrentIndex(index)
            finally:
                correction_calibration.blockSignals(previous_block)

        populate_correction_source()

        # Keep the correction controls on separate form rows.  Calibration names
        # and resolved factors can both be fairly long, and packing all three
        # controls into one horizontal row made the useful text unreadable in the
        # normal inspector width.
        magnetic_form.addRow("Correction", correction_mode)
        magnetic_form.addRow("Source", correction_calibration)
        correction_source_label = magnetic_form.labelForField(correction_calibration)
        magnetic_form.addRow("Factor", field_scale)
        self.coil_widgets["correction_mode"] = correction_mode
        self.coil_widgets["correction_calibration_id"] = correction_calibration
        self.coil_widgets["field_scale_factor"] = field_scale

        def update_correction_widgets(*_args) -> None:
            linked_mode = correction_mode.currentData() == "calibration"
            correction_calibration.setVisible(linked_mode)
            if correction_source_label is not None:
                correction_source_label.setVisible(linked_mode)
            correction_calibration.setEnabled(linked_mode and correction_calibration.count() > 0)
            if linked_mode:
                source_id = correction_calibration.currentData()
                if source_id:
                    try:
                        factor = self.adapter.measurement_calibration_fit_factor(str(source_id))
                    except Exception:
                        factor = properties["field_scale_factor"]
                    blocker = QSignalBlocker(field_scale)
                    field_scale.setValue(float(factor))
                    del blocker
                field_scale.setReadOnly(True)
                field_scale.setToolTip(
                    "Resolved correction from Dataset Manager. Edit/recalculate the linked calibration there, or choose Custom to change it manually."
                )
            else:
                blocker = QSignalBlocker(field_scale)
                field_scale.setValue(float(getattr(self, "_coil_custom_scale_value", 1.0)))
                del blocker
                field_scale.setReadOnly(False)
                field_scale.setToolTip(
                    "Custom dimensionless calibration multiplier applied to this coil's field source. Use 1.0 for no correction."
                )
            self._update_coil_estimates_preview()

        def remember_custom_scale(value: float) -> None:
            if correction_mode.currentData() == "custom":
                self._coil_custom_scale_value = float(value)

        def correction_mode_changed(*_args) -> None:
            populate_correction_source()
            update_correction_widgets()

        correction_mode.currentIndexChanged.connect(correction_mode_changed)
        correction_calibration.currentIndexChanged.connect(update_correction_widgets)
        field_scale.valueChanged.connect(remember_custom_scale)
        update_correction_widgets()

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
        form.addRow("Wire resistance @ 20 °C", wire_resistance)
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

        # --- Electrical load ---------------------------------------------------------
        electrical = self._inspector_section("Electrical Load", expanded=False)
        electrical_layout = QVBoxLayout()
        electrical_layout.setSpacing(8)
        electrical.setContentLayout(electrical_layout)
        electrical_form = QFormLayout()
        electrical_form.setContentsMargins(0, 0, 0, 0)
        electrical_layout.addLayout(electrical_form)

        resistance_mode = QComboBox()
        resistance_mode.addItem("Calculated from winding", "estimated")
        resistance_mode.addItem("Override", "override")
        resistance_mode.setCurrentIndex(
            max(0, resistance_mode.findData(properties.get("resistance_mode", "estimated")))
        )
        resistance_mode.setToolTip(
            "Use the operating winding resistance calculated from wire length, gauge and temperature, "
            "or enter a measured/known coil resistance."
        )
        electrical_form.addRow("Resistance", resistance_mode)
        self.coil_widgets["resistance_mode"] = resistance_mode

        resistance_override = _number_spin(
            properties.get("resistance_override_ohm", 0.0),
            decimals=6,
            minimum=0.0,
            maximum=1e9,
            suffix=" Ω",
        )
        resistance_override.setToolTip(
            "Measured or otherwise known total coil resistance used instead of the winding estimate."
        )
        electrical_form.addRow("Resistance override", resistance_override)
        self.coil_widgets["resistance_override_ohm"] = resistance_override

        inductance_mode = QComboBox()
        inductance_mode.addItem("Estimated from geometry", "estimated")
        inductance_mode.addItem("Override", "override")
        inductance_mode.setCurrentIndex(
            max(0, inductance_mode.findData(properties.get("inductance_mode", "estimated")))
        )
        inductance_mode.setToolTip(
            "Use Field Workbench's approximate air-core self-inductance estimate or enter a measured/known value."
        )
        electrical_form.addRow("Inductance", inductance_mode)
        self.coil_widgets["inductance_mode"] = inductance_mode

        inductance_override = _number_spin(
            float(properties.get("inductance_override_h", 0.001)) * 1000.0,
            decimals=6,
            minimum=0.000001,
            maximum=1e12,
            suffix=" mH",
        )
        inductance_override.setToolTip(
            "Measured or otherwise known self-inductance used instead of the geometry estimate."
        )
        electrical_form.addRow("Inductance override", inductance_override)
        self.coil_widgets["inductance_override_h"] = inductance_override

        electrical_summary = QLabel()
        electrical_summary.setWordWrap(True)
        electrical_summary.setObjectName("hint")
        electrical_layout.addWidget(electrical_summary)
        self.coil_electrical_summary_label = electrical_summary

        def update_electrical_override_state(*_args) -> None:
            resistance_override.setEnabled(resistance_mode.currentData() == "override")
            inductance_override.setEnabled(inductance_mode.currentData() == "override")
            self._update_coil_estimates_preview()

        resistance_mode.currentIndexChanged.connect(update_electrical_override_state)
        inductance_mode.currentIndexChanged.connect(update_electrical_override_state)
        resistance_override.valueChanged.connect(self._update_coil_estimates_preview)
        inductance_override.valueChanged.connect(self._update_coil_estimates_preview)
        update_electrical_override_state()

        electrical_note = QLabel(
            "These are the coil/load values exposures will use. Resistance defaults to the winding calculation; "
            "inductance is an approximate air-core self-inductance. Use overrides for measured or manufacturer values. "
            "Mutual inductance between coils is not included here."
        )
        electrical_note.setWordWrap(True)
        electrical_note.setObjectName("hint")
        electrical_layout.addWidget(electrical_note)
        electrical_layout.addWidget(
            self._inline_apply_button("Apply Electrical Load"),
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        self.inspector_layout.addWidget(electrical)

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
            ("inductance", "Self-inductance"),
            ("ampere_turns", "Magnetic drive"),
            ("effective_drive", "Corrected magnetic drive"),
            ("voltage", "Resistive voltage"),
            ("loss", "Resistive loss"),
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
            "Length uses the present magnetic path plus lead allowance. Resistance and self-inductance "
            "show the resolved load values above; voltage is the resistive component only and an AC supply "
            "may also require inductive headroom."
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
            "resistance_mode": self.coil_widgets["resistance_mode"].currentData(),
            "resistance_override_ohm": self.coil_widgets[
                "resistance_override_ohm"
            ].value(),
            "inductance_mode": self.coil_widgets["inductance_mode"].currentData(),
            "inductance_override_h": self.coil_widgets[
                "inductance_override_h"
            ].value()
            / 1000.0,
        }
        correction_mode = self.coil_widgets.get("correction_mode")
        correction_calibration = self.coil_widgets.get("correction_calibration_id")
        mode_value = correction_mode.currentData() if correction_mode is not None else "custom"
        values["correction_mode"] = mode_value
        values["correction_calibration_id"] = (
            str(correction_calibration.currentData() or "")
            if mode_value == "calibration" and correction_calibration is not None
            else ""
        )
        values["custom_field_scale_factor"] = float(
            getattr(self, "_coil_custom_scale_value", values["field_scale_factor"])
        )
        if mode_value == "custom":
            values["field_scale_factor"] = values["custom_field_scale_factor"]
        elif values["correction_calibration_id"]:
            try:
                values["field_scale_factor"] = self.adapter.measurement_calibration_fit_factor(
                    values["correction_calibration_id"]
                )
            except Exception:
                pass

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
                    f"{calc['resistance_ohm']:.5g} Ω "
                    + (
                        f"(override; winding estimate {calc['resistance_estimated_ohm']:.5g} Ω)"
                        if properties["resistance_mode"] == "override"
                        else f"(winding estimate @ {properties['temperature_c']:.4g} °C)"
                    )
                ),
                "inductance": (
                    f"{calc['inductance_h'] * 1000.0:.5g} mH "
                    + (
                        f"(override; geometry estimate {calc['inductance_estimated_h'] * 1000.0:.5g} mH)"
                        if properties["inductance_mode"] == "override"
                        else "(air-core geometry estimate)"
                    )
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
            if self.coil_electrical_summary_label is not None:
                self.coil_electrical_summary_label.setText(
                    "Resolved load: "
                    f"{calc['resistance_ohm']:.6g} Ω · "
                    f"{calc['inductance_h'] * 1000.0:.6g} mH"
                )
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

    def new_empty_scene(self) -> None:
        adapter = StudioAdapter()
        adapter.new_empty()
        self._apply_saved_preferences_to_adapter(adapter)
        if self._suite_managed:
            self.suite_new_project_adapter_requested.emit(adapter)
            return
        if not self.maybe_save_document(self.active_document):
            return
        self._replace_document(self._make_document(adapter))

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
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Field Workbench project" if self._suite_managed else "Open Magpylib scene",
            "",
            self._scene_filter(),
        )
        if not filename:
            return
        if self._suite_managed:
            return
        self._load_scene_file(filename)

    def open_recent_scene(self, filename: str) -> None:
        if self._suite_managed:
            return
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
        if not replace_initial and not self.maybe_save_document(self.active_document):
            return False
        self._record_recent_file(path)
        self._replace_document(document)
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
        self._document_state_changed(document)
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
        self._document_state_changed(document)
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
        if self._suite_managed and self.project_context is not None:
            document_text = self.project_context.display_name()
            if self.project_context.is_dirty():
                document_text += " *"
            self.setWindowTitle(f"{document_text} — Model Editor — {APP_NAME}")
        else:
            document_text = self._document_title_text(document)
            self.setWindowTitle(f"{document_text} — {APP_NAME}")
        self.project_state_changed.emit()

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
        self._document_state_changed(document)
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

    def _optimizer_candidate_opened_new_project(
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

        # Optimizer derivatives are first-class unsaved projects. Project Manager
        # owns their lifetime rather than Model Editor creating an internal tab.
        label = self._optimizer_finalist_tab_label(candidate)
        candidate_adapter.suggested_project_name = label
        self.suite_new_project_adapter_requested.emit(candidate_adapter)
        self.statusBar().showMessage(f"Opened {label} as a new project", 5000)

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
                "The source scene for this optimizer is no longer open. Open the finalist as a new project instead.",
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
        self._document_state_changed(document)
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
                "The source scene for this optimizer is no longer open. Open the solution as a new project instead.",
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
        self._document_state_changed(document)
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
                "Save optimizer result",
                "The optimizer source scene is no longer open, so the result cannot be saved.",
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
            QMessageBox.warning(self, "Save optimizer result", str(error))
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
            on_open_project=self._optimizer_candidate_opened_new_project,
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
        if self.project_context is not None:
            self.project_context.register_window("model_tool", dialog)
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
                "Volume analysis",
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
                "Volume analysis",
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
        self._document_state_changed(document)
        if document_id == self.active_document.document_id:
            self._update_actions()
            self._update_title()
        name = snapshot.get("name", snapshot_id) if snapshot else snapshot_id
        self.statusBar().showMessage(
            f"Saved result in {document.display_name()}: {name}",
            4500,
        )

    def _snapshots_changed(self, document_id: str | None = None) -> None:
        document = self._document_by_id(document_id or self.active_document.document_id)
        if document is None:
            return
        self._document_state_changed(document)
        if document.document_id == self.active_document.document_id:
            self._update_actions()
            self._update_title()

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

    def open_sensor_plot(self) -> None:
        document = self.active_document
        dialog = SensorPlotDialog(document.adapter, self)
        dialog.setWindowTitle(f"{dialog.windowTitle()} — {document.display_name()}")
        self._show_independent_dialog(
            dialog,
            document_id=document.document_id,
        )

    def _measurement_datasets_changed(self, document_id: str | None = None) -> None:
        document = self._document_by_id(document_id or self.active_document.document_id)
        if document is None:
            return
        self._document_state_changed(document)
        if document.document_id == self.active_document.document_id:
            selected_before = self.selected_id
            self.refresh_scene(select_id=selected_before)
            self._update_actions()
            self._update_title()

    def external_measurement_datasets_changed(self) -> None:
        """Refresh Model Editor state after Dataset Manager edits shared project data.

        Suite applications operate directly on the same adapter.  Dataset fitting
        and coil-link changes can alter resolved correction factors, so an already
        open Model Editor needs to refresh its object/property presentation without
        becoming the owner of the Dataset Manager again.
        """
        document = self.active_document
        self._measurement_datasets_changed(document.document_id)

    def external_suite_resources_changed(self) -> None:
        """Refresh shared-project chrome after sibling suite applications commit.

        Waveform/Exposure edits do not require a scene rebuild, but they do change
        the shared adapter history and project dirty state. Keep Model Editor's
        Undo/Redo actions and suite-managed title synchronized without emitting
        ``project_state_changed`` back into Project Manager and creating a loop.
        """
        self._update_actions()
        if self._suite_managed and self.project_context is not None:
            document_text = self.project_context.display_name()
            if self.project_context.is_dirty():
                document_text += " *"
            self.setWindowTitle(f"{document_text} — Model Editor — {APP_NAME}")

    def open_preferences(self) -> None:
        """Compatibility entry point for standalone Model Editor launches.

        Suite-managed editors intentionally expose global Preferences only from
        Project Manager so every open project shares one authoritative setting.
        """
        resolved = open_global_preferences(
            self,
            self.settings,
            [self.adapter],
            dialog_factory=QDialog,
        )
        if resolved is None:
            return
        self.analysis_overlay = None
        self._update_status()
        status = (
            f"Coil model: {COIL_MODELING_LABELS.get(resolved.coil_modeling_method, resolved.coil_modeling_method)} "
            f"• solver budget: {resolved.solver_memory_budget_mb} MB "
            f"• worker limit: {resolved.worker_count} worker"
            f"{'s' if resolved.worker_count != 1 else ''}"
        )
        if resolved.solver_diagnostics_enabled:
            status += " • diagnostics enabled"
        self.statusBar().showMessage(status, 5000)

    def open_help_topics(self) -> None:
        HelpTopicsDialog(self).exec()

    def open_ai_help(self) -> None:
        dialog = AiHelpDialog(self.adapter, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.pasted_into_new_project and dialog.new_project_adapter is not None:
            adapter = dialog.new_project_adapter
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
            noun = "object" if len(dialog.imported_ids) == 1 else "objects"
            if self._suite_managed:
                adapter.suggested_project_name = "AI Scene Recipe"
                self.suite_new_project_adapter_requested.emit(adapter)
                self.statusBar().showMessage(
                    f"Pasted {len(dialog.imported_ids)} AI Scene Recipe {noun} into a new project.",
                    4500,
                )
            else:
                document = self._make_document(adapter, selected_id=selected_id)
                document.saved_signature = ""
                document.selected_ids = list(top_level_ids or dialog.imported_ids)
                self._replace_document(document)
                self.statusBar().showMessage(
                    f"Pasted {len(dialog.imported_ids)} AI Scene Recipe {noun} into the model editor.",
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

    def close_suite_project(self) -> bool:
        """Close a suite-owned project and all of its application windows.

        Ordinary title-bar/File→Close operations in suite mode close only the
        Model Editor surface; the project and sibling application windows stay
        alive under the Project Manager.  The manager calls this method when the
        project tab itself is actually being closed.
        """

        self._suite_project_closing = True
        accepted = bool(self.close())
        if not accepted:
            self._suite_project_closing = False
        return accepted

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if self._suite_managed and not self._suite_project_closing:
            # Project lifetime belongs to Project Manager. Closing Model Editor
            # only hides this project view.
            self.hide()
            event.ignore()
            return
        self._commit_scene_notes_session()
        document = self.active_document
        if document.drc_dialog is not None and document.drc_dialog.thread is not None:
            QMessageBox.information(
                self,
                "Design check running",
                f"Wait for the design check for {document.display_name()} to finish before closing it.",
            )
            event.ignore()
            return
        if not self._suite_managed and not self.maybe_save_document(document):
            event.ignore()
            return
        self._close_document_windows(document.document_id)
        if self.notes_dialog is not None:
            self.notes_dialog.close()
        self.plot.cleanup()
        event.accept()
