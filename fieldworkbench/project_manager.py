"""Field Workbench suite project manager.

Each ``.magpy.json`` file is one project container with a shared project-scoped
context and resource registry. Project Manager owns project lifetime and save
state; Model Editor, Map Creator, Exposure Analysis, Exposure Editor, Waveform
Editor, Dataset Manager, and Saved Results are project applications that operate on
that context.
"""

from __future__ import annotations

import os
from datetime import datetime
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QPointF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QFontMetricsF,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .main_window import MainWindow, UnifiedFieldWorkspaceDialog
from .dialogs import SnapshotManagerDialog
from .exposure_library import ExposureEditorWindow
from .measurement_library import DatasetManagerWindow
from .preferences import apply_global_preferences_to_adapter, open_global_preferences
from .project_context import (
    ProjectContext,
    RESOURCE_CALIBRATION,
    RESOURCE_DATASET,
    RESOURCE_EXPOSURE,
    RESOURCE_MODEL_OBJECT,
    RESOURCE_SAVED_MAP,
    RESOURCE_EXPOSURE_ANALYSIS,
    RESOURCE_SNAPSHOT,
    RESOURCE_WAVEFORM,
)
from .studio_adapter import APP_NAME, SCENE_SUFFIX, StudioAdapter
from .theme import apply_workbench_theme, load_theme_preference, themed_icon
from .window_utils import install_persistent_window_settings
from .waveform_editor.ui.main_window import WaveformEditorWindow


_PROJECT_ICON_DIRECTORY = Path(__file__).with_name("assets") / "icons"
_PROJECT_APPLICATION_ICONS = {
    "Model Editor": "model_editor.svg",
    "Map Creator": "map_creator.svg",
    "Exposure Analysis": "exposure_analysis.svg",
    "Saved Results": "saved_results.svg",
    "Exposure Editor": "exposure_editor.svg",
    "Waveform Editor": "waveform_editor.svg",
    "Dataset Manager": "dataset_manager.svg",
}

_PROJECT_BANNER_FILENAME = "project_banner.svg"
_PROJECT_BANNER_ASPECT = 2750.0 / 128.0
_PROJECT_BANNER_RIGHT_INSET = 22


class OutlinedStateLabel(QLabel):
    """Label that gives its active green state a crisp black text outline."""

    _ACTIVE_FILL = QColor("#75ff00")
    _OUTLINE = QColor("#000000")
    _OUTLINE_WIDTH = 0.3

    def __init__(
        self,
        text: str = "",
        *,
        active_property: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._outline_active_property = str(active_property)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if (
            not bool(self.property(self._outline_active_property))
            or not self.text()
            or self.hasSelectedText()
        ):
            super().paintEvent(event)
            return

        # These Project Manager labels are intentionally single-line.  Building
        # the glyphs as a painter path gives the active #75ff00 text a native
        # vector stroke/fill treatment without maintaining SVG copies of labels.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setClipRect(event.rect())

        label_text = self.text()
        font = self.font()
        metrics = QFontMetricsF(font)
        rect = self.contentsRect()
        outline_pad = self._OUTLINE_WIDTH
        available_left = float(rect.left()) + outline_pad
        available_right = float(rect.right()) - outline_pad
        advance = metrics.horizontalAdvance(label_text)

        alignment = self.alignment()
        if alignment & Qt.AlignmentFlag.AlignHCenter:
            x = float(rect.center().x()) - (advance / 2.0)
        elif alignment & Qt.AlignmentFlag.AlignRight:
            x = available_right - advance
        else:
            x = available_left

        baseline = float(rect.center().y()) + (metrics.ascent() - metrics.descent()) / 2.0
        path = QPainterPath()
        path.addText(QPointF(x, baseline), font, label_text)

        pen = QPen(self._OUTLINE)
        pen.setWidthF(self._OUTLINE_WIDTH)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(self._ACTIVE_FILL)
        painter.drawPath(path)


class UnderlineStateLabel(QLabel):
    """Theme-coloured clickable label with a green populated-state underline."""

    clicked = Signal()

    _UNDERLINE = QColor("#77ff00")
    _UNDERLINE_WIDTH = 2.5
    _UNDERLINE_GAP = 2.0

    def __init__(
        self,
        text: str = "",
        *,
        active_property: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._underline_active_property = str(active_property)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        # Let QLabel/QSS render the text normally so its foreground still follows
        # the active light/dark theme. The populated-state accent is deliberately
        # limited to a separate green line beneath the visible label text.
        super().paintEvent(event)
        if not bool(self.property(self._underline_active_property)) or not self.text():
            return

        rect = self.contentsRect()
        metrics = QFontMetricsF(self.font())
        text_width = min(float(rect.width()), metrics.horizontalAdvance(self.text()))
        if text_width <= 0.0:
            return

        alignment = self.alignment()
        if alignment & Qt.AlignmentFlag.AlignHCenter:
            start_x = float(rect.center().x()) - (text_width / 2.0)
        elif alignment & Qt.AlignmentFlag.AlignRight:
            start_x = float(rect.right()) - text_width
        else:
            start_x = float(rect.left())

        if alignment & Qt.AlignmentFlag.AlignTop:
            text_top = float(rect.top())
        elif alignment & Qt.AlignmentFlag.AlignBottom:
            text_top = float(rect.bottom()) - metrics.height()
        else:
            text_top = float(rect.center().y()) - (metrics.height() / 2.0)

        underline_y = min(
            float(rect.bottom()) - (self._UNDERLINE_WIDTH / 2.0),
            text_top + metrics.height() + self._UNDERLINE_GAP,
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipRect(event.rect())
        pen = QPen(self._UNDERLINE)
        pen.setWidthF(self._UNDERLINE_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(start_x, underline_y), QPointF(start_x + text_width, underline_y))


class ResponsiveAppIconButton(QPushButton):
    """Square icon-only launcher that grows with its dashboard row."""

    _MIN_SIDE = 42
    _MAX_SIDE = 160

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing_width = False
        self.setMinimumSize(self._MIN_SIDE, self._MIN_SIDE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(68, 68)

    def minimumSizeHint(self) -> QSize:
        return QSize(self._MIN_SIDE, self._MIN_SIDE)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        target_side = max(self._MIN_SIDE, min(self._MAX_SIDE, self.height()))
        if not self._syncing_width and self.width() != target_side:
            self._syncing_width = True
            self.setFixedWidth(target_side)
            self._syncing_width = False
        icon_side = max(24, min(target_side, self.height()) - 14)
        self.setIconSize(QSize(icon_side, icon_side))


@dataclass
class ProjectRecord:
    """UI holder for one project-scoped suite context."""

    context: ProjectContext
    model_window: MainWindow | None = None
    waveform_window: WaveformEditorWindow | None = None
    exposure_window: ExposureEditorWindow | None = None
    dataset_window: DatasetManagerWindow | None = None
    maps_window: UnifiedFieldWorkspaceDialog | None = None
    snapshot_window: SnapshotManagerDialog | None = None
    startup_placeholder: bool = False

    @property
    def adapter(self) -> StudioAdapter:
        return self.context.adapter

    @property
    def untitled_number(self) -> int:
        return self.context.untitled_number

    @property
    def path(self) -> Path | None:
        return self.context.path

    @path.setter
    def path(self, value: Path | None) -> None:
        self.context.path = value

    @property
    def saved_signature(self) -> str:
        return self.context.saved_signature

    @saved_signature.setter
    def saved_signature(self, value: str) -> None:
        self.context.saved_signature = str(value)

    @property
    def project_id(self) -> str:
        return self.context.project_id

    def display_name(self) -> str:
        return self.context.display_name()

    def is_dirty(self) -> bool:
        return self.context.is_dirty()


class ProjectPage(QWidget):
    """Project summary dashboard and application launcher for one project tab."""

    def __init__(
        self,
        manager: "ProjectManagerWindow",
        record: ProjectRecord,
    ) -> None:
        super().__init__(manager)
        self.setObjectName("projectManagerProjectPage")
        self.manager = manager
        self.record = record
        self._last_dashboard_signature: tuple | None = None
        self._exposure_analysis_windows: dict[str, QWidget] = {}
        self._build_ui()
        self.refresh(force=True)

    # --- UI ---------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # Project identity/status is its own recessed band below the tab strip.
        # Keeping this separate from the resource/application area makes the
        # project tabs, project metadata, and working panes read as three clear
        # layers of the suite shell.
        info_frame = QFrame()
        info_frame.setObjectName("projectManagerInfoFrame")
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_frame.setFrameShadow(QFrame.Shadow.Sunken)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 9, 10, 9)
        info_layout.setSpacing(7)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.title_label = OutlinedStateLabel(active_property="namedProject")
        self.title_label.setObjectName("projectManagerProjectTitle")
        title_font = self.title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 3)
        self.title_label.setFont(title_font)
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label = QLabel()
        self.path_label.setObjectName("projectManagerProjectPath")
        path_font = self.path_label.font()
        path_font.setPointSize(max(8, path_font.pointSize() - 1))
        self.path_label.setFont(path_font)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.path_label)
        header.addLayout(title_box, 1)

        self.save_button = QPushButton("Save project")
        self.save_button.clicked.connect(self.manager.save_current_project)
        header.addWidget(self.save_button)
        info_layout.addLayout(header)

        metadata_row = QHBoxLayout()
        metadata_row.setContentsMargins(0, 0, 0, 0)
        metadata_row.setSpacing(8)

        self.project_status_label = QLabel()
        self.project_status_label.setObjectName("projectManagerProjectStatus")
        self.project_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        metadata_row.addWidget(self.project_status_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.project_metadata_label = QLabel()
        self.project_metadata_label.setObjectName("projectManagerProjectMetadata")
        self.project_metadata_label.setWordWrap(True)
        self.project_metadata_label.setTextFormat(Qt.TextFormat.RichText)
        self.project_metadata_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        metadata_row.addWidget(self.project_metadata_label, 1)
        info_layout.addLayout(metadata_row)
        outer.addWidget(info_frame)
        self.info_frame = info_frame

        # Use one shared six-row grid for both project components and their
        # applications.  This is intentionally not two adjacent layouts: a
        # single grid makes every application row inherit the exact height of
        # the project component row it belongs to, with no divider strip or
        # independent vertical spacing to drift out of alignment.
        dashboard_scroll = QScrollArea()
        dashboard_scroll.setObjectName("projectManagerDashboardScroll")
        dashboard_scroll.setWidgetResizable(True)
        dashboard_scroll.setFrameShape(QFrame.Shape.NoFrame)
        dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dashboard_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        dashboard_widget = QWidget()
        dashboard_widget.setObjectName("projectManagerDashboardWidget")
        self.dashboard_grid = QGridLayout(dashboard_widget)
        self.dashboard_grid.setContentsMargins(10, 10, 10, 10)
        self.dashboard_grid.setHorizontalSpacing(0)
        self.dashboard_grid.setVerticalSpacing(8)
        self.dashboard_grid.setColumnStretch(0, 5)
        self.dashboard_grid.setColumnMinimumWidth(1, 42)
        self.dashboard_grid.setColumnStretch(2, 4)
        for row in range(1, 8):
            self.dashboard_grid.setRowMinimumHeight(row, 50)
            self.dashboard_grid.setRowStretch(row, 1)

        dashboard_scroll.setWidget(dashboard_widget)
        outer.addWidget(dashboard_scroll, 1)
        self.dashboard_scroll = dashboard_scroll
        self.dashboard_widget = dashboard_widget

        project_heading = QLabel("PROJECT COMPONENTS")
        project_heading.setObjectName("projectManagerSectionTitle")
        # The component heading deliberately spans the connector column too;
        # arrows belong to the component-to-application relationship, not to a
        # visually separate middle pane.
        self.dashboard_grid.addWidget(project_heading, 0, 0, 1, 2)

        apps_heading = QLabel("APPLICATIONS")
        apps_heading.setObjectName("projectManagerSectionTitle")
        self.dashboard_grid.addWidget(apps_heading, 0, 2)

        self.project_summary_labels: dict[str, QLabel] = {}
        self.project_component_title_labels: dict[str, QLabel] = {}
        self.project_summary_frames: dict[str, QFrame] = {}
        self.project_connector_labels: list[QLabel] = []
        self.app_cards: list[QFrame] = []

        # Each logical project area is one sunken frame across all three
        # dashboard columns.  Maps / Analysis, Exposures, and Data / Results
        # each span two application rows inside that same outer frame, so their paired
        # launchers no longer introduce a border through the middle of the
        # grouping.
        self._add_dashboard_group(
            1,
            1,
            "Model",
            "model",
            ((
                "Model Editor",
                "Build and edit scene geometry, coils, sensors, volumes, and optimization.",
                self.open_model_editor,
                QStyle.StandardPixmap.SP_ComputerIcon,
            ),),
        )
        self._add_dashboard_group(
            2,
            2,
            "Maps / Analysis",
            "maps",
            (
                (
                    "Map Creator",
                    "Create Static or Scripted 2D, 3D, and Brain maps.",
                    self.open_maps,
                    QStyle.StandardPixmap.SP_FileDialogDetailedView,
                ),
                (
                    "Exposure Analysis",
                    "Analyze field exposure over time and across spatial regions.",
                    self.open_exposure_analysis,
                    QStyle.StandardPixmap.SP_FileDialogDetailedView,
                ),
            ),
        )
        self._add_dashboard_group(
            4,
            2,
            "Exposures",
            "exposures",
            (
                (
                    "Exposure Editor",
                    "Build and manage exposure protocols.",
                    self.open_exposures,
                    QStyle.StandardPixmap.SP_MediaPlay,
                ),
                (
                    "Waveform Editor",
                    "Create or import project waveforms.",
                    self.open_waveforms,
                    QStyle.StandardPixmap.SP_MediaVolume,
                ),
            ),
        )
        self._add_dashboard_group(
            6,
            2,
            "Data / Results",
            "data_results",
            (
                (
                    "Dataset Manager",
                    "Manage measurement datasets and calibrations.",
                    self.open_datasets,
                    QStyle.StandardPixmap.SP_FileDialogListView,
                ),
                (
                    "Saved Results",
                    "Review saved maps, analyses, and snapshots.",
                    self.open_snapshots,
                    QStyle.StandardPixmap.SP_DialogSaveButton,
                ),
            ),
        )

    def _add_dashboard_group(
        self,
        row: int,
        row_span: int,
        title: str,
        key: str,
        applications: tuple[tuple[str, str, Callable[[], None], QStyle.StandardPixmap], ...],
    ) -> QFrame:
        if len(applications) != row_span:
            raise ValueError("dashboard group application count must match row span")

        group = QFrame()
        group.setObjectName("projectManagerDashboardGroupFrame")
        group.setFrameShape(QFrame.Shape.StyledPanel)
        group.setFrameShadow(QFrame.Shadow.Sunken)
        group.setMinimumHeight(50 * row_span)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        group_grid = QGridLayout(group)
        group_grid.setContentsMargins(0, 0, 0, 0)
        group_grid.setHorizontalSpacing(0)
        group_grid.setVerticalSpacing(0)
        group_grid.setColumnStretch(0, 5)
        group_grid.setColumnMinimumWidth(1, 42)
        group_grid.setColumnStretch(2, 4)
        for inner_row in range(row_span):
            group_grid.setRowStretch(inner_row, 1)

        # Project component content occupies the full height of the logical
        # group.  It intentionally has no independent frame: the group frame is
        # the sole sunken boundary shared by component, arrows, and applications.
        resource_area = QFrame()
        resource_area.setObjectName("projectManagerResourceArea")
        resource_area.setFrameShape(QFrame.Shape.NoFrame)
        resource_layout = QHBoxLayout(resource_area)
        resource_layout.setContentsMargins(16, 10, 16, 10)
        resource_layout.setSpacing(22)

        title_label = UnderlineStateLabel(title, active_property="populated")
        title_label.setObjectName("projectManagerResourceTitle")
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(190)
        title_label.setMaximumWidth(190)
        title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        primary_application_name = applications[0][0]
        primary_application_callback = applications[0][2]
        title_label.setAccessibleName(f"Open {primary_application_name}")
        title_label.setToolTip(f"Open {primary_application_name}")
        title_label.clicked.connect(primary_application_callback)
        resource_layout.addWidget(title_label, 0)

        info_label = QLabel()
        info_label.setObjectName("projectManagerResourceInfo")
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        # Keep the summary as a compact block centred in the space between the
        # category title and connector arrow.  On narrow windows the stretches
        # collapse and the label can still wrap; on wide windows it no longer
        # hugs the title column simply because extra width is available.
        info_label.setMinimumWidth(150)
        info_label.setMaximumWidth(320)
        info_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        resource_layout.addStretch(1)
        resource_layout.addWidget(info_label, 0)
        resource_layout.addStretch(1)

        group_grid.addWidget(resource_area, 0, 0, row_span, 1)

        connector_prefix = title
        connector_targets = [app[0] for app in applications]
        for inner_row, target in enumerate(connector_targets):
            accessible_name = f"{connector_prefix} → {target}"
            arrow = QLabel("→")
            arrow.setObjectName("projectManagerConnectorArrow")
            arrow.setAccessibleName(accessible_name)
            arrow.setToolTip(accessible_name)
            arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
            arrow.setFixedWidth(42)
            arrow.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            group_grid.addWidget(arrow, inner_row, 1)
            self.project_connector_labels.append(arrow)

        for inner_row, (button_text, description, callback, icon_kind) in enumerate(applications):
            app_area = QFrame()
            app_area.setObjectName("projectManagerAppArea")
            app_area.setFrameShape(QFrame.Shape.NoFrame)
            app_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            app_layout = QHBoxLayout(app_area)
            app_layout.setContentsMargins(8, 4, 8, 4)
            app_layout.setSpacing(12)

            button = ResponsiveAppIconButton()
            button.setObjectName("projectManagerAppButton")
            button.setAccessibleName(button_text)
            button.setToolTip(f"Open {button_text}")
            button.setProperty("applicationName", button_text)
            button.setIcon(self._application_launcher_icon(button_text, icon_kind))
            button.clicked.connect(callback)
            app_layout.addWidget(button, 0)

            text_column = QVBoxLayout()
            text_column.setContentsMargins(0, 0, 0, 0)
            text_column.setSpacing(2)

            name_label = QLabel(button_text)
            name_label.setObjectName("projectManagerAppName")
            name_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)

            detail = QLabel(description)
            detail.setObjectName("projectManagerAppDescription")
            detail.setWordWrap(True)
            detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            text_column.addStretch(1)
            text_column.addWidget(name_label)
            text_column.addWidget(detail)
            text_column.addStretch(1)
            app_layout.addLayout(text_column, 1)

            group_grid.addWidget(app_area, inner_row, 2)
            self.app_cards.append(app_area)

        self.project_summary_labels[key] = info_label
        self.project_component_title_labels[key] = title_label
        self.project_summary_frames[key] = group
        self.dashboard_grid.addWidget(group, row, 0, row_span, 3)
        return group

    def _application_launcher_icon(
        self,
        application_name: str,
        fallback_kind: QStyle.StandardPixmap,
    ) -> QIcon:
        """Return a bundled project launcher icon with a themed fallback."""

        filename = _PROJECT_APPLICATION_ICONS.get(application_name)
        if filename:
            custom_icon = QIcon(str(_PROJECT_ICON_DIRECTORY / filename))
            if not custom_icon.isNull():
                return custom_icon
        return themed_icon(
            self.style().standardIcon(fallback_kind),
            load_theme_preference(self.manager.settings),
        )

    # --- project dashboard -----------------------------------------
    def _resource_signature(self) -> tuple:
        """Return the already-refreshed registry signature without polling the model."""
        try:
            return self.record.context.resources.signature(refresh=False)
        except Exception:
            return ("unavailable",)

    @staticmethod
    def _last_saved_text(path: Path | None) -> str:
        if path is None:
            return "Not yet saved"
        try:
            stamp = datetime.fromtimestamp(path.stat().st_mtime)
        except (OSError, ValueError, OverflowError):
            return "Unavailable"
        date_text = stamp.strftime("%b %d, %Y").replace(" 0", " ")
        time_text = stamp.strftime("%I:%M %p").lstrip("0")
        return f"{date_text} · {time_text}"

    def _update_project_metadata(self, dirty: bool, resources: tuple) -> None:
        path = self.record.path
        if path is None:
            status = "Unsaved project"
        elif dirty:
            status = "Unsaved changes"
        else:
            status = "Saved"

        counts = {
            RESOURCE_MODEL_OBJECT: 0,
            RESOURCE_SAVED_MAP: 0,
            RESOURCE_EXPOSURE_ANALYSIS: 0,
            RESOURCE_SNAPSHOT: 0,
            RESOURCE_EXPOSURE: 0,
            RESOURCE_WAVEFORM: 0,
            RESOURCE_DATASET: 0,
            RESOURCE_CALIBRATION: 0,
        }
        for resource in resources:
            if resource.kind in counts:
                counts[resource.kind] += 1

        saved_results = (
            counts[RESOURCE_SAVED_MAP]
            + counts[RESOURCE_EXPOSURE_ANALYSIS]
            + counts[RESOURCE_SNAPSHOT]
        )
        exposure_items = counts[RESOURCE_EXPOSURE] + counts[RESOURCE_WAVEFORM]
        data_items = counts[RESOURCE_DATASET] + counts[RESOURCE_CALIBRATION]
        contents = (
            f"{counts[RESOURCE_MODEL_OBJECT]} objects · "
            f"{saved_results} saved results · "
            f"{exposure_items} exposure items · "
            f"{data_items} data items"
        )
        status_is_dirty = status == "Unsaved changes"
        self.project_status_label.setText(f"Status: {status}")
        if bool(self.project_status_label.property("dirty")) != status_is_dirty:
            self.project_status_label.setProperty("dirty", status_is_dirty)
            style = self.project_status_label.style()
            style.unpolish(self.project_status_label)
            style.polish(self.project_status_label)
            self.project_status_label.update()

        self.project_metadata_label.setText(
            f"<b>Last saved:</b> {escape(self._last_saved_text(path))} &nbsp; • &nbsp; "
            f"<b>Contents:</b> {contents}"
        )

    @staticmethod
    def _set_boolean_style_property(widget: QWidget, name: str, value: bool) -> None:
        """Set a boolean QSS state property and repolish only when it changes."""
        value = bool(value)
        if bool(widget.property(name)) == value:
            return
        widget.setProperty(name, value)
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _update_title_colour_state(self) -> None:
        """Keep saved/named project titles green; Untitled follows the active theme."""
        self._set_boolean_style_property(
            self.title_label,
            "namedProject",
            self.record.path is not None,
        )

    def _set_component_populated(self, key: str, populated: bool) -> None:
        label = self.project_component_title_labels.get(key)
        if label is None:
            return
        self._set_boolean_style_property(label, "populated", populated)

    def _set_summary_text(self, key: str, lines: list[str]) -> None:
        label = self.project_summary_labels.get(key)
        if label is None:
            return
        cleaned = [line.strip() for line in lines if line and line.strip()]
        label.setText("<br>".join(cleaned))

    def refresh(self, *, force: bool = False) -> None:
        # Project Manager is event-driven: suite applications call this when
        # project state actually changes.  Do not poll/serialize the adapter just
        # to keep an idle dashboard current; that steals CPU from interactive
        # Model Editor rendering for no user-visible benefit.
        dirty = self.record.is_dirty()
        self.title_label.setText(self.record.display_name() + (" *" if dirty else ""))
        self._update_title_colour_state()
        self.path_label.setText(
            str(self.record.path)
            if self.record.path is not None
            else "Unsaved project · save to create a .magpy.json project file"
        )
        self.save_button.setText("Save project" if self.record.path is not None else "Save project as…")
        self.manager._update_tab_for_page(self, dirty=dirty)

        registry = self.record.context.resources
        if force:
            resources = registry.refresh(live_snapshot_status=True)
        else:
            resources = registry.all()
            signature = self._resource_signature()
            if signature == self._last_dashboard_signature:
                self._update_project_metadata(dirty, tuple(resources))
                return
        self._last_dashboard_signature = self._resource_signature()
        self._update_project_metadata(dirty, tuple(resources))
        objects = tuple(item for item in resources if item.kind == RESOURCE_MODEL_OBJECT)
        snapshots = tuple(item for item in resources if item.kind == RESOURCE_SNAPSHOT)
        saved_maps = tuple(item for item in resources if item.kind == RESOURCE_SAVED_MAP)
        exposure_analyses = tuple(
            item for item in resources if item.kind == RESOURCE_EXPOSURE_ANALYSIS
        )
        exposures = tuple(item for item in resources if item.kind == RESOURCE_EXPOSURE)
        datasets = tuple(item for item in resources if item.kind == RESOURCE_DATASET)
        calibrations = tuple(item for item in resources if item.kind == RESOURCE_CALIBRATION)
        waveform_resources = tuple(item for item in resources if item.kind == RESOURCE_WAVEFORM)

        model_lines = [
            f"<b>{len(objects)} object{'s' if len(objects) != 1 else ''}</b>",
            "Model is empty." if not objects else "Scene model",
        ]
        self._set_summary_text("model", model_lines)
        self._set_component_populated("model", bool(objects))

        saved_result_count = len(saved_maps) + len(exposure_analyses) + len(snapshots)
        map_lines = [
            f"<b>{len(saved_maps)} map{'s' if len(saved_maps) != 1 else ''}</b>",
            f"<b>{len(exposure_analyses)} exposure analys{'es' if len(exposure_analyses) != 1 else 'is'}</b>",
        ]
        self._set_summary_text("maps", map_lines)
        self._set_component_populated("maps", bool(saved_maps or exposure_analyses))

        exposure_lines = [
            f"<b>{len(exposures)} exposure{'s' if len(exposures) != 1 else ''}</b>",
            f"<b>{len(waveform_resources)} waveform{'s' if len(waveform_resources) != 1 else ''}</b>",
        ]
        self._set_summary_text("exposures", exposure_lines)
        self._set_component_populated(
            "exposures", bool(exposures or waveform_resources)
        )

        data_result_lines = [
            (
                f"<b>{len(datasets)} dataset{'s' if len(datasets) != 1 else ''}</b> · "
                f"<b>{len(calibrations)} calibration{'s' if len(calibrations) != 1 else ''}</b>"
            ),
            f"<b>{saved_result_count} saved result{'s' if saved_result_count != 1 else ''}</b>",
        ]
        self._set_summary_text("data_results", data_result_lines)
        self._set_component_populated(
            "data_results", bool(datasets or calibrations or saved_result_count)
        )

    # --- existing applications ------------------------------------
    def ensure_model_window(self) -> MainWindow:
        window = self.record.model_window
        if window is None:
            window = MainWindow(
                adapter=self.record.adapter,
                settings=self.manager.settings,
                suite_managed=True,
                initial_path=self.record.path,
                project_context=self.record.context,
            )
            window.suite_new_project_adapter_requested.connect(self.manager.add_project)
            window.project_state_changed.connect(self._model_project_state_changed)
            self.record.model_window = window
        return window

    def _model_project_state_changed(self) -> None:
        """Refresh suite resource views after model/undo/redo changes."""
        self.refresh(force=True)
        self._notify_saved_results_project_changed()
        exposure_window = self.record.exposure_window
        if exposure_window is not None:
            try:
                exposure_window.project_resources_changed()
            except RuntimeError:
                pass
        waveform_window = self.record.waveform_window
        if waveform_window is not None:
            try:
                waveform_window.refresh_project_resources()
            except RuntimeError:
                pass

    def sync_from_model_window(self) -> None:
        """Compatibility hook; event-driven dashboard refresh owns registry updates."""
        return

    def open_model_editor(self) -> None:
        window = self.ensure_model_window()
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        window.raise_()
        window.activateWindow()

    def _maps_brain_object_id(self) -> str | None:
        """Resolve the SRI24 brain for a project-owned Maps launch.

        A project-manager launch has no Model-Editor selection to disambiguate
        multiple brains.  A unique SRI24 brain is therefore selected
        automatically; multiple candidates leave Brain view disabled until a
        future chooser is added rather than guessing which anatomy is intended.
        """

        try:
            rows = self.record.adapter.list_objects()
        except Exception:
            rows = []
        brain_ids = [
            str(item.get("id"))
            for item in rows if isinstance(item, dict)
            and item.get("builtin_template") == "sri24_brain"
            and item.get("id")
        ]
        return brain_ids[0] if len(brain_ids) == 1 else None

    def _maps_destroyed(self, window_id: int) -> None:
        window = self.record.maps_window
        if window is not None and id(window) == window_id:
            self.record.maps_window = None

    def open_maps(self, mode: str = "static", *, saved_map_id: str | None = None) -> None:
        normalized = str(mode or "static").strip().lower()
        if normalized not in {"static", "playback"}:
            normalized = "static"
        window = self.record.maps_window
        if window is None:
            window = UnifiedFieldWorkspaceDialog(
                mode=normalized,
                scene_name=self.record.display_name(),
                adapter=self.record.adapter,
                brain_object_id=self._maps_brain_object_id(),
                project_adapter=self.record.adapter,
                on_project_resources_changed=self._snapshots_changed,
                on_open_snapshot_manager=lambda: self.open_snapshots(),
            )
            window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.record.context.register_window("maps_analysis", window)
            self.record.maps_window = window
            window_id = id(window)
            window.destroyed.connect(
                lambda _object=None, key=window_id: self._maps_destroyed(key)
            )

            # If Model Editor is already open, preserve the convenient launch
            # camera hand-off. Maps itself no longer depends on Model Editor.
            model = self.record.model_window
            if model is not None:
                try:
                    model.plot.capture_view_state(window.apply_launch_view_state)
                except (AttributeError, RuntimeError):
                    pass
        elif normalized != window.mode:
            window._switch_mode(normalized)

        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        window.raise_()
        window.activateWindow()
        if saved_map_id:
            QTimer.singleShot(0, lambda map_id=saved_map_id, target=window: target.load_saved_map(map_id))

    def _exposure_resources_changed(self) -> None:
        """Refresh project surfaces that summarize saved exposure resources."""
        self.refresh(force=True)
        self._notify_saved_results_project_changed()
        window = self.record.exposure_window
        if window is not None:
            try:
                window.refresh_project_resources()
            except RuntimeError:
                pass
        model = self.record.model_window
        if model is not None:
            try:
                model.external_suite_resources_changed()
            except RuntimeError:
                pass

    def _waveform_resources_changed(self) -> None:
        """Propagate waveform catalogue changes to open exposure tooling.

        Exposures reference stable waveform IDs, so edits/renames do not rewrite
        protocols.  The live Exposure Editor still needs to refresh names and
        diagnostics when a waveform changes in the sibling application.
        """
        self.refresh(force=True)
        self._notify_saved_results_project_changed()
        window = self.record.exposure_window
        if window is not None:
            try:
                window.waveform_resources_changed()
            except RuntimeError:
                pass
        model = self.record.model_window
        if model is not None:
            try:
                model.external_suite_resources_changed()
            except RuntimeError:
                pass

    def open_exposures(self, exposure_id: str | None = None) -> None:
        window = self.record.exposure_window
        if window is None:
            window = ExposureEditorWindow(self.record.context)
            self.record.context.register_window("exposure_editor", window)
            window.projectResourcesChanged.connect(self._exposure_resources_changed)
            window.openWaveformRequested.connect(self.open_waveforms)
            self.record.exposure_window = window
        window.refresh_project_resources(exposure_id)
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        window.raise_()
        window.activateWindow()
        if exposure_id:
            window.open_exposure(exposure_id)

    def _exposure_analysis_cache_changed(self, _analysis_id: str | None = None) -> None:
        """Refresh dashboard/Saved Results after first-class Exposure Analysis work."""
        self.refresh(force=True)
        self._notify_saved_results_project_changed(immediate=True)

    def _exposure_analysis_window_destroyed(self, exposure_id: str, window_id: int) -> None:
        window = self._exposure_analysis_windows.get(str(exposure_id))
        if window is not None and id(window) == window_id:
            self._exposure_analysis_windows.pop(str(exposure_id), None)

    def open_exposure_analysis(self, exposure_id: str | None = None) -> None:
        """Open Exposure Analysis as a first-class suite application.

        The analysis workspace itself is scoped to one saved exposure.  Project
        Manager therefore opens the only exposure directly, or asks which saved
        exposure to use when the project contains several.  Exposure Editor keeps
        its existing contextual Analyze exposure shortcut.
        """
        summaries = list(self.record.adapter.list_exposures())
        if not summaries:
            QMessageBox.information(
                self,
                "Exposure Analysis",
                "This project has no saved exposures yet. Create and save an exposure in Exposure Editor first.",
            )
            return

        selected_id = str(exposure_id or "").strip()
        if selected_id:
            summary = next((row for row in summaries if str(row.get("id", "")) == selected_id), None)
            if summary is None:
                QMessageBox.warning(self, "Exposure Analysis", "The selected exposure no longer exists.")
                return
        elif len(summaries) == 1:
            summary = summaries[0]
            selected_id = str(summary.get("id", ""))
        else:
            labels: list[str] = []
            rows_by_label: dict[str, dict] = {}
            for index, row in enumerate(summaries, start=1):
                name = str(row.get("name") or f"Exposure {index}")
                duration = row.get("duration_s")
                try:
                    duration_text = f"{float(duration):g} s"
                except (TypeError, ValueError):
                    duration_text = "duration unavailable"
                label = f"{name} — {duration_text}"
                # Names are normally unique, but keep the chooser deterministic
                # even for hand-edited project files that contain duplicates.
                if label in rows_by_label:
                    label = f"{label} [{index}]"
                labels.append(label)
                rows_by_label[label] = row
            chosen, accepted = QInputDialog.getItem(
                self,
                "Exposure Analysis",
                "Exposure to analyze:",
                labels,
                0,
                False,
            )
            if not accepted or not chosen:
                return
            summary = rows_by_label[str(chosen)]
            selected_id = str(summary.get("id", ""))

        existing = self._exposure_analysis_windows.get(selected_id)
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
                self._exposure_analysis_windows.pop(selected_id, None)

        try:
            exposure = self.record.adapter.exposure(selected_id)
        except Exception as error:  # noqa: BLE001 - project resource boundary
            QMessageBox.warning(self, "Exposure Analysis", str(error))
            return

        from .exposure_workspace import ExposureAnalysisWorkspace

        workspace = ExposureAnalysisWorkspace(
            self.record.adapter,
            exposure,
            project_label=self.record.display_name(),
            on_cache_changed=self._exposure_analysis_cache_changed,
        )
        self._exposure_analysis_windows[selected_id] = workspace
        window_id = id(workspace)
        workspace.destroyed.connect(
            lambda _object=None, exposure_key=selected_id, key=window_id: self._exposure_analysis_window_destroyed(exposure_key, key)
        )
        self.record.context.register_window("exposure_analysis", workspace)
        workspace.show()
        workspace.raise_()
        workspace.activateWindow()

    def _dataset_resources_changed(self) -> None:
        self.refresh(force=True)
        self._notify_saved_results_project_changed()
        window = self.record.model_window
        if window is not None:
            try:
                window.external_measurement_datasets_changed()
            except RuntimeError:
                pass

    def open_datasets(
        self,
        dataset_id: str | None = None,
        *,
        calibration_id: str | None = None,
    ) -> None:
        window = self.record.dataset_window
        if window is None:
            window = DatasetManagerWindow(self.record.context)
            self.record.context.register_window("dataset_manager", window)
            window.projectResourcesChanged.connect(self._dataset_resources_changed)
            self.record.dataset_window = window
        window.refresh_project_resources(dataset_id, calibration_id=calibration_id)
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        window.raise_()
        window.activateWindow()
        if dataset_id:
            window.open_dataset(dataset_id)
        elif calibration_id:
            window.open_calibration(calibration_id)

    def _notify_saved_results_project_changed(self, *, immediate: bool = False) -> None:
        """Refresh Saved Results from real project events, never idle polling.

        Resource edits can arrive in bursts and remain debounced.  A saved-result
        catalogue mutation (snapshot/map create/delete) is already a discrete user
        action, so callers may request an immediate signature refresh; this keeps
        the open manager visibly live without reintroducing periodic polling.
        """
        window = self.record.snapshot_window
        if window is not None:
            try:
                if immediate:
                    window.refresh_if_changed()
                else:
                    window.notify_project_changed()
            except RuntimeError:
                pass

    def _snapshots_changed(self) -> None:
        self.refresh(force=True)
        model = self.record.model_window
        if model is not None:
            try:
                model._snapshots_changed(model.active_document.document_id)
            except (AttributeError, RuntimeError):
                pass
        self._notify_saved_results_project_changed(immediate=True)

    def _snapshot_window_destroyed(self, window_id: int) -> None:
        window = self.record.snapshot_window
        if window is not None and id(window) == window_id:
            self.record.snapshot_window = None

    def open_snapshots(
        self,
        snapshot_id: str | None = None,
        *,
        view_result: bool = False,
        section: str | None = None,
    ) -> None:
        window = self.record.snapshot_window
        if window is None:
            window = SnapshotManagerDialog(
                self.record.adapter,
                None,
                on_changed=self._snapshots_changed,
                on_open_map=lambda map_id, kind: self.open_maps(
                    "playback" if kind == "scripted" else "static",
                    saved_map_id=map_id,
                ),
            )
            window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            window.setWindowTitle(f"Saved results — {self.record.display_name()}")
            self.record.context.register_window("snapshot_manager", window)
            self.record.snapshot_window = window
            window_id = id(window)
            window.destroyed.connect(
                lambda _object=None, key=window_id: self._snapshot_window_destroyed(key)
            )
        if snapshot_id:
            window.refresh(select_id=snapshot_id)
            window.show_section("snapshots")
        else:
            window.refresh_if_changed()
            if section:
                window.show_section(section)
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        window.raise_()
        window.activateWindow()
        if snapshot_id and view_result:
            window.view_snapshot(snapshot_id)

    def open_waveforms(self, waveform_id: str | None = None) -> None:
        window = self.record.waveform_window
        if window is None:
            window = WaveformEditorWindow(self.record.context)
            self.record.context.register_window("waveform_editor", window)
            window.projectResourcesChanged.connect(self._waveform_resources_changed)
            self.record.waveform_window = window
        if waveform_id:
            # Cross-suite requests are guarded by Waveform Editor so an active
            # working copy cannot be replaced by a catalogue refresh.
            window.request_open_resource(waveform_id)
        else:
            window.refresh_project_resources()
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        window.raise_()
        window.activateWindow()


class ProjectManagerWindow(QMainWindow):
    """Suite-level project manager with independently open project tabs."""

    RECENT_FILES_KEY = MainWindow.RECENT_FILES_KEY
    MAX_RECENT_FILES = MainWindow.MAX_RECENT_FILES
    project_closed = Signal(str)

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        install_persistent_window_settings(self)
        self.settings = settings if settings is not None else QSettings()
        self._untitled_counter = 0
        self._closing = False
        self.resize(1180, 760)
        self.setMinimumSize(900, 820)
        self.setWindowTitle(f"{APP_NAME} — Project Manager")
        self._build_ui()
        self._build_actions()
        apply_workbench_theme(self, load_theme_preference(self.settings))


    # --- construction ------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        # Keep the complete project-tab workspace on one slightly recessed
        # chrome surface.  The themed object selectors paint this the same
        # family as the menu/toolbar while the asymmetric border gives it an
        # inset/sunken edge instead of the old near-black document background.
        tab_well = QFrame()
        tab_well.setObjectName("projectManagerTabWell")
        tab_well.setFrameShape(QFrame.Shape.StyledPanel)
        tab_well.setFrameShadow(QFrame.Shadow.Plain)
        tab_well_layout = QVBoxLayout(tab_well)
        tab_well_layout.setContentsMargins(1, 1, 1, 1)
        tab_well_layout.setSpacing(0)
        central_layout.addWidget(tab_well, 1)
        self.tab_well = tab_well

        self.tabs = QTabWidget()
        self.tabs.setObjectName("projectManagerTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setObjectName("projectManagerTabBar")
        self.tabs.tabBar().setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tabs.currentChanged.connect(self._current_tab_changed)
        self.tabs.tabCloseRequested.connect(self.close_project_tab)
        tab_well_layout.addWidget(self.tabs, 1)

        plus = QToolButton()
        plus.setObjectName("projectManagerNewProjectButton")
        plus.setText("+")
        plus.setToolTip("New empty project")
        plus.setAccessibleName("New project")
        plus.setAutoRaise(True)
        plus.setFixedWidth(30)
        plus.clicked.connect(lambda: self.new_project_from_kind("empty"))
        self.tabs.setCornerWidget(plus, Qt.Corner.TopLeftCorner)
        self.new_project_button = plus

        self.statusBar().setObjectName("projectManagerStatusBar")
        self.statusBar().showMessage("Ready")

    def _project_toolbar_icon(
        self,
        filename: str,
        fallback_kind: QStyle.StandardPixmap,
    ) -> QIcon:
        """Return a bundled Project Manager toolbar icon with themed fallback."""

        custom_icon = QIcon(str(_PROJECT_ICON_DIRECTORY / filename))
        if not custom_icon.isNull():
            return custom_icon
        return themed_icon(
            self.style().standardIcon(fallback_kind),
            load_theme_preference(self.settings),
        )

    def _refresh_toolbar_icons(self) -> None:
        if hasattr(self, "new_action"):
            self.new_action.setIcon(
                self._project_toolbar_icon(
                    "project_new.svg",
                    QStyle.StandardPixmap.SP_FileIcon,
                )
            )
        if hasattr(self, "open_action"):
            self.open_action.setIcon(
                self._project_toolbar_icon(
                    "project_open.svg",
                    QStyle.StandardPixmap.SP_DialogOpenButton,
                )
            )
        if hasattr(self, "save_action"):
            self.save_action.setIcon(
                self._project_toolbar_icon(
                    "project_save.svg",
                    QStyle.StandardPixmap.SP_DialogSaveButton,
                )
            )

    def _position_project_banner(self) -> None:
        """Align the toolbar wordmark with the dashboard's right content edge."""

        if not hasattr(self, "project_banner_label") or not hasattr(self, "project_toolbar"):
            return
        toolbar = self.project_toolbar
        label = self.project_banner_label
        x = max(0, toolbar.width() - label.width() - _PROJECT_BANNER_RIGHT_INSET)
        y = max(0, (toolbar.height() - label.height()) // 2)
        label.move(x, y)
        label.raise_()

    def _refresh_project_banner(self, icon_size: QSize | None = None) -> None:
        """Render the Project Manager wordmark at the toolbar icon height."""

        if not hasattr(self, "project_banner_label"):
            return
        if icon_size is None or not icon_size.isValid():
            icon_size = self.project_toolbar.iconSize()
        banner_height = max(18, int(icon_size.height()))
        banner_width = max(1, int(round(banner_height * _PROJECT_BANNER_ASPECT)))
        target_size = QSize(banner_width, banner_height)
        banner_icon = QIcon(str(_PROJECT_ICON_DIRECTORY / _PROJECT_BANNER_FILENAME))
        self.project_banner_label.setFixedSize(target_size)
        self.project_banner_label.setPixmap(banner_icon.pixmap(target_size))
        self._position_project_banner()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._position_project_banner()

    def apply_workbench_theme(self, _theme: str) -> None:
        """Refresh locally rendered suite icons after a theme switch."""
        self._refresh_toolbar_icons()
        self._refresh_project_banner()

    def _build_actions(self) -> None:
        toolbar = QToolBar("Project")
        toolbar.setObjectName("projectManagerToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        self.project_toolbar = toolbar

        self.new_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon),
            "New project",
            self,
        )
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(lambda: self.new_project_from_kind("empty"))
        toolbar.addAction(self.new_action)

        self.open_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "Open project",
            self,
        )
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_project_dialog)
        toolbar.addAction(self.open_action)

        self.save_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save project",
            self,
        )
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_current_project)
        self._refresh_toolbar_icons()
        toolbar.addAction(self.save_action)

        # The Field Workbench wordmark is an overlay child rather than a normal
        # toolbar item.  This keeps the artwork pinned to the far-right edge
        # without letting New/Open/Save consume or shift its layout space.
        self.project_banner_label = QLabel(toolbar)
        self.project_banner_label.setObjectName("projectManagerToolbarBanner")
        self.project_banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.project_banner_label.setToolTip("Field Workbench")
        self.project_banner_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.project_banner_label.show()
        toolbar.iconSizeChanged.connect(self._refresh_project_banner)
        self._refresh_project_banner(toolbar.iconSize())

        # Application launchers live in the project dashboard.  Keep these
        # actions for the Project menu without duplicating a partial launcher
        # set in the top toolbar.
        self.model_action = QAction("Model Editor", self)
        self.model_action.triggered.connect(lambda: self._with_current_page(lambda page: page.open_model_editor()))
        self.maps_action = QAction("Map Creator", self)
        self.maps_action.triggered.connect(lambda: self._with_current_page(lambda page: page.open_maps()))
        self.exposure_analysis_action = QAction("Exposure Analysis", self)
        self.exposure_analysis_action.triggered.connect(
            lambda: self._with_current_page(lambda page: page.open_exposure_analysis())
        )

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.new_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open Recent")
        self._refresh_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        save_as = file_menu.addAction("Save project as…")
        save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as.triggered.connect(self.save_current_project_as)
        file_menu.addSeparator()
        close_project = file_menu.addAction("Close project")
        close_project.setShortcut(QKeySequence.StandardKey.Close)
        close_project.triggered.connect(self.close_current_project)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        preferences_action = edit_menu.addAction("Preferences…")
        preferences_action.triggered.connect(self.open_preferences)

        project_menu = self.menuBar().addMenu("&Project")
        project_menu.addAction(self.model_action)
        project_menu.addAction(self.maps_action)
        project_menu.addAction(self.exposure_analysis_action)
        exposure_action = project_menu.addAction("Exposure Editor")
        exposure_action.triggered.connect(lambda: self._with_current_page(lambda page: page.open_exposures()))
        waveform_action = project_menu.addAction("Waveform Editor")
        waveform_action.triggered.connect(lambda: self._with_current_page(lambda page: page.open_waveforms()))
        dataset_action = project_menu.addAction("Dataset Manager")
        dataset_action.triggered.connect(lambda: self._with_current_page(lambda page: page.open_datasets()))
        snapshot_action = project_menu.addAction("Saved Results")
        snapshot_action.triggered.connect(lambda: self._with_current_page(lambda page: page.open_snapshots()))
        project_menu.addSeparator()
        refresh_action = project_menu.addAction("Refresh project dashboard")
        refresh_action.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh_action.triggered.connect(lambda: self.refresh_projects(force=True))

        help_menu = self.menuBar().addMenu("&Help")
        help_topics = help_menu.addAction("Help Topics…")
        help_topics.setShortcut(QKeySequence.StandardKey.HelpContents)
        help_topics.triggered.connect(self.open_help_topics)
        help_menu.addSeparator()
        about = help_menu.addAction("About Field Workbench…")
        about.triggered.connect(self.show_about)

    def _live_preference_adapters(self) -> list[StudioAdapter]:
        """Return every live adapter that should follow suite-global preferences.

        Open Maps windows use frozen scene copies for geometry, but solver/model
        policy is still application-wide.  Include those private adapters so an
        already-open workspace picks up a Preferences change before its next job.
        """
        adapters: list[StudioAdapter] = []
        seen: set[int] = set()

        def add(candidate) -> None:
            if not isinstance(candidate, StudioAdapter) or id(candidate) in seen:
                return
            seen.add(id(candidate))
            adapters.append(candidate)

        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            if not isinstance(page, ProjectPage):
                continue
            add(page.record.adapter)
            for window in page.record.context.windows():
                add(getattr(window, "adapter", None))
                add(getattr(window, "snapshot_adapter", None))
                for child_window in getattr(window, "_independent_windows", {}).values():
                    add(getattr(child_window, "adapter", None))
                    add(getattr(child_window, "snapshot_adapter", None))
                for viewer_page in getattr(window, "_pages", {}).values():
                    add(getattr(viewer_page, "adapter", None))
                controller = getattr(window, "playback_controller", None)
                add(getattr(controller, "adapter", None))
        return adapters

    def open_preferences(self) -> None:
        """Edit application-wide settings from the suite's authoritative shell."""
        resolved = open_global_preferences(
            self, self.settings, self._live_preference_adapters()
        )
        if resolved is None:
            return
        self.statusBar().showMessage(
            f"Preferences updated • solver budget {resolved.solver_memory_budget_mb} MB "
            f"• worker limit {resolved.worker_count}",
            5000,
        )

    # --- project tabs -------------------------------------------------
    def current_page(self) -> ProjectPage | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, ProjectPage) else None

    def current_context(self) -> ProjectContext | None:
        """Return the active project independently of any application window."""
        page = self.current_page()
        return page.record.context if page is not None else None

    def _with_current_page(self, callback: Callable[[ProjectPage], None]) -> None:
        page = self.current_page()
        if page is None:
            self.statusBar().showMessage("Open or create a project first.", 3000)
            return
        callback(page)

    def _new_record(
        self,
        adapter: StudioAdapter,
        path: Path | None = None,
        *,
        startup_placeholder: bool = False,
    ) -> ProjectRecord:
        if path is None:
            self._untitled_counter += 1
            untitled_number = self._untitled_counter
        else:
            untitled_number = max(1, self._untitled_counter)
        context = ProjectContext(
            adapter,
            path=path,
            untitled_number=untitled_number,
            saved_signature=adapter.signature(),
            display_label=str(getattr(adapter, "suggested_project_name", "") or "").strip(),
        )
        return ProjectRecord(
            context=context,
            startup_placeholder=bool(startup_placeholder),
        )

    def add_project(
        self,
        adapter: StudioAdapter,
        path: str | Path | None = None,
        *,
        startup_placeholder: bool = False,
        replace_page: ProjectPage | None = None,
    ) -> ProjectPage:
        # Project files do not own solver/resource policy.  Every project joins
        # the same application-wide preferences as soon as it enters the suite.
        apply_global_preferences_to_adapter(adapter, settings=self.settings)
        resolved = None if path is None else Path(path).expanduser().resolve()
        if resolved is not None:
            existing = self._page_for_path(resolved)
            if existing is not None:
                self.tabs.setCurrentWidget(existing)
                return existing
        record = self._new_record(
            adapter,
            resolved,
            startup_placeholder=startup_placeholder,
        )
        page = ProjectPage(self, record)
        replace_index = self.tabs.indexOf(replace_page) if replace_page is not None else -1
        if replace_index >= 0:
            self.tabs.removeTab(replace_index)
            replace_page.deleteLater()
            index = self.tabs.insertTab(replace_index, page, record.display_name())
        else:
            index = self.tabs.addTab(page, record.display_name())
        self.tabs.setTabToolTip(index, str(resolved or "Unsaved project"))
        self.tabs.setCurrentIndex(index)
        if resolved is not None:
            self._record_recent_file(resolved)
        self._update_actions()
        return page

    def new_project_from_kind(self, kind: str = "empty") -> None:
        # New suite projects intentionally start with no scene objects. Existing
        # project files remain available through Open. ``kind`` remains a small
        # public API convenience, but all new projects intentionally begin empty.
        adapter = StudioAdapter()
        adapter.new_empty()
        self.add_project(adapter)
        self.statusBar().showMessage("Created a new empty project tab.", 2500)

    def add_startup_empty_project(self) -> None:
        """Create the initial blank project shown by the suite shell."""
        adapter = StudioAdapter()
        adapter.new_empty()
        self.add_project(adapter, startup_placeholder=True)

    def _replaceable_startup_page(self) -> ProjectPage | None:
        """Return the untouched startup placeholder that Open may reuse.

        Deliberately-created New Project tabs are never candidates.  The startup
        tab also stops being disposable as soon as its project data changes or a
        project application window is opened, which avoids silently replacing a
        user's active workspace or private editor draft.
        """
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            if not isinstance(page, ProjectPage):
                continue
            record = page.record
            if not record.startup_placeholder or record.path is not None:
                continue
            if record.is_dirty() or record.context.windows():
                continue
            if any(
                window is not None
                for window in (
                    record.model_window,
                    record.waveform_window,
                    record.exposure_window,
                    record.dataset_window,
                    record.maps_window,
                    record.snapshot_window,
                )
            ):
                continue
            return page
        return None

    def open_project_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Field Workbench project",
            "",
            self._project_filter(),
        )
        if filename:
            self.open_project_path(filename)

    def open_project_path(self, filename: str) -> None:
        path = Path(filename).expanduser().resolve()
        existing = self._page_for_path(path)
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            self.statusBar().showMessage(f"{path.name} is already open.", 2500)
            return
        if not path.is_file():
            self._remove_recent_file(path)
            QMessageBox.warning(self, "Project unavailable", f"This project file does not exist:\n{path}")
            return
        adapter = StudioAdapter()
        try:
            adapter.load_file(path)
        except Exception as error:  # noqa: BLE001 - external project boundary
            QMessageBox.critical(self, "Could not open project", str(error))
            return
        placeholder = self._replaceable_startup_page()
        self.add_project(adapter, path, replace_page=placeholder)
        self.statusBar().showMessage(f"Opened {path.name}.", 3000)

    def _page_for_path(self, path: str | Path) -> ProjectPage | None:
        target = os.path.normcase(str(Path(path).expanduser().resolve()))
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            if not isinstance(page, ProjectPage) or page.record.path is None:
                continue
            if os.path.normcase(str(page.record.path.resolve())) == target:
                return page
        return None

    def _update_tab_for_page(
        self,
        page: ProjectPage,
        *,
        dirty: bool | None = None,
    ) -> None:
        index = self.tabs.indexOf(page)
        if index < 0:
            return
        if dirty is None:
            dirty = page.record.is_dirty()
        label = page.record.display_name() + (" *" if dirty else "")
        if self.tabs.tabText(index) != label:
            self.tabs.setTabText(index, label)
        self.tabs.setTabToolTip(index, str(page.record.path or "Unsaved project"))
        if page is self.current_page():
            self.setWindowTitle(f"{label} — {APP_NAME} Project Manager")

    def _current_tab_changed(self, _index: int) -> None:
        page = self.current_page()
        if page is not None:
            page.refresh(force=True)
        else:
            self.setWindowTitle(f"{APP_NAME} — Project Manager")
        self._update_actions()

    def refresh_projects(self, *, force: bool = False) -> None:
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            if isinstance(page, ProjectPage):
                page.refresh(force=force)
        self._update_actions()

    def close_current_project(self) -> None:
        index = self.tabs.currentIndex()
        if index >= 0:
            self.close_project_tab(index)

    def close_project_tab(self, index: int) -> bool:
        page = self.tabs.widget(index)
        if not isinstance(page, ProjectPage):
            return False
        if not self._prepare_page_close(page):
            return False
        self.tabs.removeTab(index)
        self.project_closed.emit(page.record.context.project_id)
        page.deleteLater()
        self._update_actions()
        if self.tabs.count() == 0 and not self._closing:
            self.statusBar().showMessage("No project open. Create or open a project to continue.")
        return True

    @staticmethod
    def _prepare_project_context_windows(page: ProjectPage) -> bool:
        """Give suite windows a chance to commit/cancel private working copies.

        Some editors intentionally keep drafts outside the project adapter until
        the user presses Save.  Those drafts must be resolved *before* checking
        the project dirty signature, otherwise closing a clean project could
        silently discard an unsaved waveform working copy.
        """
        model_window = page.record.model_window
        exposure_window = page.record.exposure_window

        # Exposure edits live in a modal private draft that must be explicitly
        # Saved/Cancelled by the user. Check that non-mutating blocker before a
        # sibling editor (notably Waveform Editor) is allowed to commit a draft.
        if exposure_window is not None:
            prepare = getattr(exposure_window, "prepare_project_close", None)
            if callable(prepare):
                try:
                    if not bool(prepare()):
                        return False
                except RuntimeError:
                    pass

        for window in page.record.context.windows():
            if window is model_window or window is exposure_window:
                continue
            prepare = getattr(window, "prepare_project_close", None)
            if not callable(prepare):
                continue
            try:
                if not bool(prepare()):
                    return False
            except RuntimeError:
                continue
        return True

    @staticmethod
    def _close_project_context_windows(page: ProjectPage) -> None:
        """Close project-scoped suite windows other than Model Editor."""
        model_window = page.record.model_window
        for window in page.record.context.windows():
            if window is model_window:
                continue
            try:
                window.close()
            except RuntimeError:
                pass
        page.record.waveform_window = None
        page.record.exposure_window = None
        page.record.dataset_window = None
        page.record.maps_window = None
        page.record.snapshot_window = None

    def _prepare_page_close(self, page: ProjectPage) -> bool:
        """Let Project Manager own the save/discard decision for one project."""
        if not self._prepare_project_context_windows(page):
            return False
        # A child editor may just have committed its private draft into the
        # adapter, so determine project dirtiness only after that preparation.
        if page.record.is_dirty():
            answer = QMessageBox.warning(
                self,
                "Unsaved project changes",
                f"Save changes to {page.record.display_name()}?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save and not self._save_page(
                page, save_as=page.record.path is None
            ):
                return False

        window = page.record.model_window
        if window is not None:
            if not bool(window.close_suite_project()):
                return False
            page.record.model_window = None
        self._close_project_context_windows(page)
        return True

    # --- save ---------------------------------------------------------
    def save_current_project(self) -> bool:
        page = self.current_page()
        if page is None:
            return False
        return self._save_page(page, save_as=page.record.path is None)

    def save_current_project_as(self) -> bool:
        page = self.current_page()
        if page is None:
            return False
        return self._save_page(page, save_as=True)

    def _save_page(self, page: ProjectPage, *, save_as: bool) -> bool:
        page.sync_from_model_window()
        target = page.record.path
        if save_as or target is None:
            suggested = target.name if target is not None else f"untitled{SCENE_SUFFIX}"
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Field Workbench project",
                suggested,
                self._project_filter(),
            )
            if not filename:
                return False
            if not filename.lower().endswith(".json"):
                filename += SCENE_SUFFIX
            target = Path(filename).expanduser().resolve()
            existing = self._page_for_path(target)
            if existing is not None and existing is not page:
                QMessageBox.warning(
                    self,
                    "Project already open",
                    "That file is already open in another project tab.",
                )
                return False
        assert target is not None
        try:
            self.save_project_to_path(page, target)
        except Exception as error:  # noqa: BLE001 - filesystem boundary
            QMessageBox.critical(self, "Could not save project", str(error))
            return False
        return True

    def save_project_to_path(
        self, page: ProjectPage, target: str | Path, *, overwrite: bool = True
    ) -> None:
        """Save an explicit project without dialogs, sharing the UI save baseline.

        Errors propagate to the calling UI or automation boundary. Private editor
        drafts retain their existing Save/Cancel ownership in the corresponding app.
        """
        target = Path(target).expanduser().resolve()
        existing = self._page_for_path(target)
        if existing is not None and existing is not page:
            raise ValueError("That file is already open in another project tab.")
        page.record.adapter.save_file(target, overwrite=overwrite)
        page.record.context.mark_saved(target)
        window = page.record.model_window
        if window is not None:
            try:
                # The adapter is shared, but project path/save baseline remain
                # authoritative in ProjectContext. Model Editor only refreshes
                # its presentation after the manager saves.
                window._update_title()
            except (AttributeError, RuntimeError):
                pass
        self._record_recent_file(target)
        page.refresh(force=True)
        self.statusBar().showMessage(f"Saved {target.name}.", 3000)

    # --- recent projects ---------------------------------------------
    @staticmethod
    def _path_key(path: str | Path) -> str:
        try:
            return os.path.normcase(str(Path(path).expanduser().resolve()))
        except OSError:
            return os.path.normcase(str(path))

    def _recent_files(self) -> list[str]:
        stored = self.settings.value(self.RECENT_FILES_KEY, [])
        if stored is None:
            return []
        if isinstance(stored, str):
            stored = [stored]
        result: list[str] = []
        seen: set[str] = set()
        for value in stored:
            text = str(value).strip()
            if not text:
                continue
            key = self._path_key(text)
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result[: self.MAX_RECENT_FILES]

    def _record_recent_file(self, path: str | Path) -> None:
        display = str(Path(path).expanduser().resolve())
        key = self._path_key(display)
        files = [value for value in self._recent_files() if self._path_key(value) != key]
        files.insert(0, display)
        self.settings.setValue(self.RECENT_FILES_KEY, files[: self.MAX_RECENT_FILES])
        self.settings.sync()
        self._refresh_recent_menu()

    def _remove_recent_file(self, path: str | Path) -> None:
        key = self._path_key(path)
        files = [value for value in self._recent_files() if self._path_key(value) != key]
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
            empty = self.recent_menu.addAction("(No recent projects)")
            empty.setEnabled(False)
            return
        for index, filename in enumerate(files, start=1):
            path = Path(filename)
            action = self.recent_menu.addAction(f"&{index}  {path.name} — {path.parent}")
            action.setToolTip(filename)
            action.triggered.connect(lambda _checked=False, value=filename: self.open_project_path(value))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear Recent Projects", self.clear_recent_files)

    # --- misc ---------------------------------------------------------
    def _project_filter(self) -> str:
        return f"Field Workbench project (*{SCENE_SUFFIX});;JSON files (*.json);;All files (*)"

    def _update_actions(self) -> None:
        enabled = self.current_page() is not None
        self.save_action.setEnabled(enabled)
        self.model_action.setEnabled(enabled)
        self.maps_action.setEnabled(enabled)
        self.exposure_analysis_action.setEnabled(enabled)

    def open_help_topics(self) -> None:
        # Help is suite-level and does not require a project or Model Editor.
        from .help_topics import HelpTopicsDialog
        from .window_utils import show_application_window

        show_application_window(HelpTopicsDialog())

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Field Workbench",
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {__version__}</p>"
            "<p>Project Manager owns project lifetime and save state. Model Editor, Map Creator, Exposure Analysis, "
            "Exposure Editor, Waveform Editor, Dataset Manager, and Saved Results operate directly on the shared project context.</p>"
            "<p>Free software licensed under GNU GPL version 3 or later (GPL-3.0-or-later).</p>",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        self._closing = True
        # Close from right to left so indexes remain stable. Project Manager
        # performs each save/discard decision before closing project windows.
        for index in range(self.tabs.count() - 1, -1, -1):
            page = self.tabs.widget(index)
            if not isinstance(page, ProjectPage):
                continue
            if not self._prepare_page_close(page):
                self._closing = False
                event.ignore()
                return
            self.tabs.removeTab(index)
            self.project_closed.emit(page.record.context.project_id)
            page.deleteLater()
        event.accept()
