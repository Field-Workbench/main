"""Interactive LPBA40 brain-region field and waveform workspace."""

from __future__ import annotations

import base64
import copy
import math
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, Signal, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette, QPen, QPolygon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QDialog,
)

from .brain_analysis import (
    BrainAnalysisError,
    BrainRegionMetrics,
    BrainRegionNode,
    BrainSampleSet,
    downsample_brain_samples,
    lpba40_region_hierarchy,
    lpba40_region_metrics,
    sample_lpba40_voxels,
)
from .brain_mesh import (
    BrainMeshError,
    LPBA40MeshSet,
    LPBA40_MESH_CACHE_STRIDE,
    load_lpba40_region_meshes,
)
from .dialogs import (
    _BackgroundFieldRenderMixin,
    CollapsibleSection,
    EditorWheelScrollFilter,
    FieldVolumeMapDialog,
    _CameraTimelineMixin,
    _PlaybackCoilSelectionMixin,
    _spin,
)
from .plot_view import PlotView
from .field_grid import field_volume_grid_spec
from .sri24_atlas import SRI24Parcellation, load_sri24_parcellation
from .studio_adapter import (
    FIELD_HEAT_COLOURSCALE,
    StudioAdapter,
    StudioOperationError,
    _field_colour_at_fraction,
)
from .video_export import default_video_camera
from .waveform_dialog import WaveformPlaybackDialog, remembered_waveform_timing
from .waveform_gl_view import WaveformGLView
from .waveform_video_dialog import WaveformVideoExportDialog
from .window_utils import (
    CompactDoubleSpinBox,
    capture_window_default_settings,
    centre_window_on_parent,
    enable_standard_window_controls,
    reset_window_to_default_settings,
)


STATIC_SAMPLES_PER_LPBA40_LABEL = 900
MAXIMUM_DISPLAY_ATLAS_POINTS = 14000
LPBA40_VISUAL_MESH_STRIDE = LPBA40_MESH_CACHE_STRIDE
BRAIN_FIELD_RENDER_MODE = "volume"
BRAIN_FIELD_RENDER_RESOLUTION = 33
BRAIN_FIELD_RENDER_SLICE_COUNT = 9
LPBA40_REGION_OPACITY = 0.34
LPBA40_SELECTED_REGION_OPACITY = 0.92
LPBA40_SELECTED_REGION_COLOUR = "#facc15"
LPBA40_DIVISION_COLOURS: dict[str, str] = {
    "frontal": "#4e79a7",
    "parietal": "#59a14f",
    "occipital": "#76b7b2",
    "temporal": "#b07aa1",
    "insula": "#edc948",
    "limbic": "#f28e2b",
    "deep_gray": "#e15759",
    "hindbrain": "#9c755f",
}
REGION_KEY_ROLE = int(Qt.ItemDataRole.UserRole) + 1
REGION_LABEL_IDS_ROLE = int(Qt.ItemDataRole.UserRole) + 2
REGION_SIDE_ROLE = int(Qt.ItemDataRole.UserRole) + 3
BRAIN_FILTER_DEFAULT_HIGHLIGHT = "#facc15"


def _brain_filter_action_widgets(cell: QWidget | None) -> tuple[QComboBox | None, QPushButton | None]:
    if cell is None:
        return None, None
    action = cell.findChild(QComboBox, "brainFilterActionCombo")
    colour = cell.findChild(QPushButton, "brainFilterColourButton")
    return action, colour


def _set_brain_filter_colour_button(button: QPushButton, colour: str) -> None:
    parsed = QColor(str(colour))
    if not parsed.isValid():
        parsed = QColor(BRAIN_FILTER_DEFAULT_HIGHLIGHT)
    value = parsed.name(QColor.NameFormat.HexRgb)
    button.setProperty("brainFilterHighlightColour", value)
    button.setToolTip(f"Highlight colour: {value} — click to change")
    button.setStyleSheet(
        "QPushButton#brainFilterColourButton {"
        f"background-color: {value}; border: 1px solid rgba(127,127,127,0.75);"
        "min-width: 24px; max-width: 24px; min-height: 20px; max-height: 20px; padding: 0px;"
        "}"
    )


def _brain_filter_highlight_brushes(colour: str) -> tuple[QBrush, QBrush]:
    background = QColor(str(colour))
    if not background.isValid():
        background = QColor(BRAIN_FILTER_DEFAULT_HIGHLIGHT)
    # WCAG-style perceived luminance is sufficient for choosing readable row text.
    luminance = (
        0.2126 * background.redF()
        + 0.7152 * background.greenF()
        + 0.0722 * background.blueF()
    )
    foreground = QColor("#111827" if luminance > 0.56 else "#f8fafc")
    return QBrush(background), QBrush(foreground)


def _brain_item_matches_filter_rule(
    item: QTreeWidgetItem,
    rule: tuple[int | str, str, float | str, str, str],
) -> bool:
    metric, operator, threshold, _action, _colour = rule
    if metric == "side":
        side = str(item.data(0, REGION_SIDE_ROLE) or "").casefold()
        return operator == "=" and side == str(threshold).casefold()
    raw_value = item.data(int(metric), Qt.ItemDataRole.UserRole)
    if not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
        return False
    metric_value = float(raw_value)
    numeric_threshold = float(threshold)
    if operator == ">":
        return metric_value > numeric_threshold
    if operator == "<":
        return metric_value < numeric_threshold
    return False


def _clear_brain_item_filter_highlight(item: QTreeWidgetItem, columns: int) -> None:
    for column in range(columns):
        item.setData(column, Qt.ItemDataRole.BackgroundRole, None)
        item.setData(column, Qt.ItemDataRole.ForegroundRole, None)


def _apply_brain_item_filter_highlight(
    item: QTreeWidgetItem, colour: str, columns: int
) -> None:
    background, foreground = _brain_filter_highlight_brushes(colour)
    for column in range(columns):
        item.setBackground(column, background)
        item.setForeground(column, foreground)


class _TableEditorWheelScrollFilter(QObject):
    """Route wheel gestures over embedded table editors to the table scrollbar.

    QComboBox and spin-box cell widgets otherwise consume wheel events before
    QTableWidget can scroll.  Brain Areas filter tables use the wheel strictly
    for navigation while the pointer is over an editor; values still change by
    clicking/opening the control, typing, or using the step buttons.
    """

    def __init__(self, table: QTableWidget):
        super().__init__(table)
        self._table = table

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if event.type() != QEvent.Type.Wheel:
            return super().eventFilter(watched, event)

        try:
            scroll_bar = self._table.verticalScrollBar()
        except RuntimeError:
            event.accept()
            return True

        pixel_delta = event.pixelDelta().y()
        if pixel_delta:
            distance = pixel_delta
        else:
            wheel_steps = event.angleDelta().y() / 120.0
            distance = wheel_steps * max(36, scroll_bar.singleStep() * 3)
        scroll_bar.setValue(scroll_bar.value() - round(distance))
        event.accept()
        return True


class _EditorWheelBlockFilter(QObject):
    """Make embedded Brain filter editors behave like deliberate controls.

    ``QTableWidget.setCellWidget()`` installs real child widgets over the item
    view.  Those widgets have their own focus machinery, independent of the
    table's ``NoFocus``/``NoSelection`` settings.  In particular, a spin box's
    internal ``QLineEdit`` can receive a ``FocusIn`` without the table ever
    seeing a click.  Earlier fixes only disabled hover/current-index behaviour
    on the table itself, so they could not prevent that child-widget focus jump.

    Arm keyboard focus only after an actual mouse press inside the same cell
    widget.  Any unsolicited FocusIn (including focus-following pointer quirks
    from a platform/style) is rejected.  Wheel gestures are still routed to the
    filter table rather than editing a value accidentally.
    """

    def __init__(self, select_row, table: QTableWidget):
        super().__init__(table)
        self._select_row = select_row
        self._table = table
        self._focus_armed_root_id: int | None = None
        self._focus_arm_generation = 0

    def _editor_root(self, watched):
        """Return the direct cell widget containing *watched*, if available."""
        if not isinstance(watched, QWidget):
            return None
        try:
            viewport = self._table.viewport()
        except RuntimeError:
            return None
        current = watched
        while isinstance(current, QWidget) and current is not viewport:
            try:
                parent = current.parentWidget()
            except RuntimeError:
                return None
            if parent is viewport:
                return current
            current = parent
        return watched

    def _arm_focus_for(self, root: QWidget | None) -> None:
        self._focus_arm_generation += 1
        generation = self._focus_arm_generation
        self._focus_armed_root_id = id(root) if root is not None else None

        # FocusIn normally follows the press synchronously.  Keep a short grace
        # period for platform styles that defer it until release/event-loop idle,
        # then disarm so later pointer movement cannot inherit click permission.
        def disarm() -> None:
            if generation == self._focus_arm_generation:
                self._focus_armed_root_id = None

        QTimer.singleShot(250, disarm)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress:
            root = self._editor_root(watched)
            self._arm_focus_for(root)
            row = watched.property("brainFilterRow")
            if isinstance(row, int) and row >= 0:
                self._select_row(row)
        elif event_type == QEvent.Type.FocusIn:
            root = self._editor_root(watched)
            if root is None or id(root) != self._focus_armed_root_id:
                # This editor was not explicitly clicked.  Do not let hover,
                # item-view current-index changes, or platform focus-following
                # behaviour place a text caret in it.
                try:
                    watched.clearFocus()
                except RuntimeError:
                    pass
                event.accept()
                return True
            # One explicit press authorises focus for this cell.  Once acquired,
            # leave the editor focused normally until the user clicks elsewhere.
            self._focus_armed_root_id = None
            self._focus_arm_generation += 1
        elif event_type == QEvent.Type.Wheel:
            # Embedded combo/spin widgets normally consume wheel events before
            # QTableWidget sees them. Move the table scrollbar directly rather
            # than re-sending the QWheelEvent with editor-local coordinates.
            try:
                scroll_bar = self._table.verticalScrollBar()
            except RuntimeError:
                event.accept()
                return True
            pixel_delta = event.pixelDelta().y()
            if pixel_delta:
                distance = pixel_delta
            else:
                wheel_steps = event.angleDelta().y() / 120.0
                distance = wheel_steps * max(36, scroll_bar.singleStep() * 3)
            scroll_bar.setValue(scroll_bar.value() - round(distance))
            event.accept()
            return True
        return super().eventFilter(watched, event)



class _NumericTreeItem(QTreeWidgetItem):
    """Tree row that compares numeric user data for metric columns."""

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        left = self.data(column, Qt.ItemDataRole.UserRole)
        right = other.data(column, Qt.ItemDataRole.UserRole)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if math.isfinite(float(left)) and math.isfinite(float(right)):
                return float(left) < float(right)
        return self.text(column).casefold() < other.text(column).casefold()


class _BrainRegionTreeWidget(QTreeWidget):
    """QTreeWidget that restores classic parent/child connector lines.

    Qt's Fusion style intentionally paints only disclosure arrows for
    ``PE_IndicatorBranch``.  That makes the LPBA40 hierarchy harder to scan
    once several lobes and hemispheres are expanded, because indentation is
    the only remaining ancestry cue.  Draw the conventional elbow/vertical
    guides behind the native branch indicator while leaving Qt responsible
    for the expand/collapse arrow itself.
    """

    def _has_visible_later_sibling(self, index) -> bool:
        model = self.model()
        parent = index.parent()
        for row in range(index.row() + 1, model.rowCount(parent)):
            if not self.isRowHidden(row, parent):
                return True
        return False

    def drawBranches(self, painter: QPainter, rect, index) -> None:  # noqa: N802 - Qt API name
        indent = max(1, int(self.indentation()))
        row_mid_y = rect.center().y()
        inner_left = rect.right() + 1 - indent
        # Fusion places the disclosure glyph slightly left of the nominal
        # indentation-cell centre. Keep the guides just left of centre; this
        # final one-pixel adjustment places the dotted spine through the
        # visual centre of the disclosure triangles.
        connector_left_shift = max(0, min(1, indent // 10))
        branch_x = inner_left + indent // 2 - connector_left_shift

        painter.save()
        line_colour = self.palette().color(QPalette.ColorRole.Dark)
        line_colour.setAlpha(165)
        pen = QPen(line_colour)
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        painter.setPen(pen)

        # The current item's innermost branch. Top-level rows have no parent,
        # so they keep only Qt's native disclosure arrow rather than acquiring
        # a line that appears to enter from nowhere above the root.
        if index.parent().isValid():
            if self._has_visible_later_sibling(index):
                painter.drawLine(branch_x, rect.top(), branch_x, rect.bottom())
            else:
                painter.drawLine(branch_x, rect.top(), branch_x, row_mid_y)

            has_children = self.model().rowCount(index) > 0
            arrow_clearance = max(3, min(6, indent // 4)) if has_children else 0
            painter.drawLine(
                branch_x + arrow_clearance,
                row_mid_y,
                rect.right(),
                row_mid_y,
            )

        # Each ancestor that still has a later visible sibling contributes a
        # continuous vertical guide through this descendant row. This is what
        # produces the familiar long spine beside an expanded lobe/subtree.
        ancestor = index.parent()
        ancestor_x = branch_x - indent
        while ancestor.isValid():
            if self._has_visible_later_sibling(ancestor):
                painter.drawLine(ancestor_x, rect.top(), ancestor_x, rect.bottom())
            ancestor = ancestor.parent()
            ancestor_x -= indent

        painter.restore()

        # Draw our own disclosure triangle instead of Fusion's unusually tiny
        # native glyph.  Keep it centred on the same x-coordinate as the
        # dotted branch spine so the connector/arrow alignment remains exact.
        # Hit testing and expand/collapse behaviour still come from QTreeView;
        # this only changes the painted glyph.
        if self.model().rowCount(index) > 0:
            painter.save()
            arrow_colour = self.palette().color(QPalette.ColorRole.Text)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(arrow_colour))
            if self.isExpanded(index):
                arrow = QPolygon([
                    QPoint(branch_x - 4, row_mid_y - 2),
                    QPoint(branch_x + 4, row_mid_y - 2),
                    QPoint(branch_x, row_mid_y + 4),
                ])
            else:
                arrow = QPolygon([
                    QPoint(branch_x - 2, row_mid_y - 4),
                    QPoint(branch_x - 2, row_mid_y + 4),
                    QPoint(branch_x + 4, row_mid_y),
                ])
            painter.drawPolygon(arrow)
            painter.restore()


class _ToggleSelectedTreeRowFilter(QObject):
    """Let a second click on the current tree row clear its selection safely.

    QTreeWidget's normal single-selection mode keeps the current row selected
    when it is clicked again. Brain View uses selection as a visual highlight,
    so a repeated left click is more useful as an explicit unhighlight action.

    Keep the viewport wrapper captured at construction time and never call
    ``tree.viewport()`` from the event filter. During Qt teardown the tree's C++
    object can already be gone while one last viewport event is unwinding; the
    former lookup raised a libshiboken RuntimeError when Brain View closed.
    """

    def __init__(self, tree: QTreeWidget):
        super().__init__(tree)
        self._tree: QTreeWidget | None = tree
        self._viewport = tree.viewport()
        tree.destroyed.connect(self._tree_destroyed)

    def _tree_destroyed(self, *_args) -> None:
        self._tree = None
        self._viewport = None

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        tree = self._tree
        if tree is None or watched is not self._viewport:
            return False
        try:
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                try:
                    point = event.position().toPoint()
                except AttributeError:
                    point = event.pos()
                item = tree.itemAt(point)
                if (
                    item is not None
                    and item is tree.currentItem()
                    and item.isSelected()
                ):
                    # Clear the current item first so the synchronous
                    # itemSelectionChanged signal observes an actually empty
                    # selection/current state instead of momentarily treating
                    # the Whole brain fallback as selected.
                    tree.setCurrentItem(None)
                    tree.clearSelection()
                    event.accept()
                    return True
        except RuntimeError:
            # A final event can race QObject destruction on PySide6. There is
            # nothing useful to handle once the underlying C++ item view is gone.
            self._tree = None
            self._viewport = None
            return False
        return False


def _collapse_brain_region_tree_to_major(
    tree: QTreeWidget, hierarchy: BrainRegionNode
) -> None:
    """Show the Whole brain root plus the eight major groups, collapsed below."""

    tree.collapseAll()
    root_item = tree.topLevelItem(0)
    if root_item is not None:
        root_item.setExpanded(True)


def _expand_brain_region_tree(tree: QTreeWidget) -> None:
    """Expand the complete visible Brain Areas hierarchy."""

    tree.expandAll()


def _expand_brain_region_tree_major(tree: QTreeWidget) -> None:
    """Expand the root and major groups, leaving deeper structures collapsed."""

    tree.collapseAll()
    root_item = tree.topLevelItem(0)
    if root_item is None:
        return
    root_item.setExpanded(True)
    for index in range(root_item.childCount()):
        root_item.child(index).setExpanded(True)


class _BrainResultRegionPanel(QWidget):
    """Per-result Brain Areas table with independent filters/sort/selection."""

    region_selected = Signal(object)
    FRAME_HEADER_LABELS = (
        "Structure", "RMS |B|", "Mean |B|", "P95 |B|", "Peak |B|",
        "Uniformity", "Direction", "Volume",
    )
    SUMMARY_HEADER_LABELS = (
        "Structure", "Playback RMS |B|", "Mean |B|", "Max P95 |B|",
        "Absolute peak |B|", "B-time", "B²-time", "Volume",
    )
    FRAME_UNITS = ("", "µT", "µT", "µT", "µT", "%", "%", "cm³")
    SUMMARY_UNITS = ("", "µT", "µT", "µT", "µT", "µT·s", "µT²·s", "cm³")
    HEADER_LABELS = FRAME_HEADER_LABELS

    def __init__(self, hierarchy: BrainRegionNode, parent=None):
        super().__init__(parent)
        self.hierarchy = hierarchy
        self._items_by_key: dict[str, QTreeWidgetItem] = {}
        self._frame_values: np.ndarray | None = None
        self._frame_sample_indices: np.ndarray | None = None
        self._source_frame_count = 0
        self._frame_keys: list[str] = []
        self._sample_counts: list[int] = []
        self._voxel_counts: list[int] = []
        self._pending_frame: tuple[int, float] | None = None
        self._frame_update_scheduled = False
        self._current_frame_index = 0
        self._current_frame_time_s = 0.0
        self._summary_values: np.ndarray | None = None
        self._summary_peak_times_s: np.ndarray | None = None
        self._playback_metadata: dict[str, Any] = {}
        self._display_mode = "frame"
        self._loading_filter_rules = False
        self._filter_rules_by_mode: dict[
            str, list[tuple[int | str, str, float | str, str, str]]
        ] = {"frame": [], "summary": []}
        self._table_state_by_mode: dict[str, dict[str, Any]] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        heading_row = QHBoxLayout()
        heading = QLabel("Brain areas")
        heading.setObjectName("dialogHeading")
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        self.frame_note = QLabel("Static field")
        self.frame_note.setObjectName("hint")
        heading_row.addWidget(self.frame_note)
        root.addLayout(heading_row)

        self.metrics_note = QLabel(
            "Rows summarize volume-weighted atlas samples for this result tab. "
            "Filters and sorting belong to this tab only."
        )
        self.metrics_note.setObjectName("hint")
        self.metrics_note.setWordWrap(True)
        root.addWidget(self.metrics_note)

        self.region_sections_splitter = QSplitter(Qt.Orientation.Vertical)
        self.region_sections_splitter.setChildrenCollapsible(False)
        self.region_sections_splitter.setHandleWidth(6)

        filter_section = QWidget()
        filter_section_layout = QVBoxLayout(filter_section)
        filter_section_layout.setContentsMargins(0, 0, 0, 10)
        filter_section_layout.setSpacing(6)
        filter_heading = QLabel("Filters")
        filter_heading.setObjectName("sectionHeading")
        filter_section_layout.addWidget(filter_heading)
        self.filter_table = QTableWidget(0, 5)
        self.filter_table.setObjectName("brainRegionFilterTable")
        self.filter_table.setHorizontalHeaderLabels(["#", "Metric", "Condition", "Value", "Action"])
        self.filter_table.verticalHeader().setVisible(False)
        # Match the Selector tab's deliberately non-greedy filter grid. Persistent
        # editor widgets must not drive QTableWidget current-row/focus state just
        # because the pointer crosses them. Rule selection is owned explicitly:
        # Add selects the new rule and a deliberate click selects a different one.
        self._selected_filter_row = -1
        self.filter_table.setAlternatingRowColors(True)
        self.filter_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.filter_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.filter_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.filter_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.filter_table.setMouseTracking(False)
        self.filter_table.setAttribute(Qt.WidgetAttribute.WA_Hover, False)
        self.filter_table.viewport().setMouseTracking(False)
        self.filter_table.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, False)
        fh = self.filter_table.horizontalHeader()
        fh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        fh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        fh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        fh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        fh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.filter_table.setColumnWidth(0, 38)
        self._filter_wheel_filter = _EditorWheelBlockFilter(
            self._set_filter_selected_row, self.filter_table
        )
        self.filter_table.setMinimumHeight(78)
        filter_section_layout.addWidget(self.filter_table, 1)

        actions = QHBoxLayout()
        self.add_filter_button = QPushButton("Add")
        self.remove_filter_button = QPushButton("Remove")
        self.remove_filter_button.setEnabled(False)
        self.add_filter_button.clicked.connect(self._add_filter)
        self.remove_filter_button.clicked.connect(self._remove_filter)
        self.filter_table.cellClicked.connect(self._filter_cell_clicked)
        actions.addWidget(self.add_filter_button)
        actions.addWidget(self.remove_filter_button)
        actions.addStretch(1)
        filter_section_layout.addLayout(actions)
        self.region_sections_splitter.addWidget(filter_section)

        table_section = QWidget()
        table_section_layout = QVBoxLayout(table_section)
        table_section_layout.setContentsMargins(0, 0, 0, 0)
        table_section_layout.setSpacing(6)

        self.mode_controls = QWidget()
        mode_layout = QHBoxLayout(self.mode_controls)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(12)
        mode_label = QLabel("Values")
        mode_label.setObjectName("sectionHeading")
        self.current_frame_radio = QRadioButton("Current frame")
        self.playback_summary_radio = QRadioButton("Playback summary")
        self.current_frame_radio.setChecked(True)
        self.playback_summary_radio.setEnabled(False)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.current_frame_radio)
        mode_layout.addWidget(self.playback_summary_radio)
        mode_layout.addStretch(1)
        self.mode_controls.setVisible(False)
        table_section_layout.addWidget(self.mode_controls)

        self.playback_statistics_widget = QWidget()
        playback_statistics = QGridLayout(self.playback_statistics_widget)
        playback_statistics.setContentsMargins(8, 6, 8, 6)
        playback_statistics.setHorizontalSpacing(10)
        playback_statistics.setVerticalSpacing(3)
        self.playback_statistic_values: dict[str, QLabel] = {}
        for row, (key, label_text) in enumerate(
            (
                ("exposure", "Exposure time"),
                ("waveform", "Waveform"),
                ("sequencer", "Sequencer"),
                ("sampling", "Sampling"),
            )
        ):
            label = QLabel(label_text)
            label.setObjectName("sectionHeading")
            value = QLabel("—")
            value.setObjectName("hint")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            playback_statistics.addWidget(label, row, 0, Qt.AlignmentFlag.AlignTop)
            playback_statistics.addWidget(value, row, 1)
            self.playback_statistic_values[key] = value
        playback_statistics.setColumnStretch(1, 1)
        self.playback_statistics_widget.setVisible(False)
        table_section_layout.addWidget(self.playback_statistics_widget)

        tree_actions = QHBoxLayout()
        self.collapse_tree_button = QPushButton("Collapse")
        self.expand_tree_button = QPushButton("Expand")
        self.expand_major_tree_button = QPushButton("Expand major")
        self.collapse_tree_button.setToolTip(
            "Collapse the Brain Areas hierarchy to the eight major groups."
        )
        self.expand_tree_button.setToolTip("Expand the complete Brain Areas hierarchy.")
        self.expand_major_tree_button.setToolTip(
            "Expand each major Brain Area one level while leaving deeper structures collapsed."
        )
        tree_actions.addWidget(self.collapse_tree_button)
        tree_actions.addWidget(self.expand_tree_button)
        tree_actions.addWidget(self.expand_major_tree_button)
        tree_actions.addStretch(1)

        self.table = _BrainRegionTreeWidget()
        self.table.setObjectName("brainResultRegionTable")
        self.table.setColumnCount(8)
        self.table.setHeaderLabels(list(self.FRAME_HEADER_LABELS))
        header = self.table.header()
        header.setObjectName("brainRegionHeader")
        header.setMinimumHeight(30)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)
        header.setStyleSheet(
            "QHeaderView#brainRegionHeader::up-arrow, "
            "QHeaderView#brainRegionHeader::down-arrow { image:none; width:0px; height:0px; }"
        )
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(58)
        for column, width in enumerate((220, 92, 92, 92, 92, 104, 96, 92)):
            self.table.setColumnWidth(column, width)
        self.column_menu = QMenu(self.table)
        self.column_actions = []
        for column, label in enumerate(self.FRAME_HEADER_LABELS):
            action = self.column_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda checked, selected_column=column: self.table.setColumnHidden(
                    selected_column, not checked
                )
            )
            self.column_actions.append(action)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_column_menu)
        header.sortIndicatorChanged.connect(self._update_sort_indicator)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self._populate()
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._update_sort_indicator(0, Qt.SortOrder.AscendingOrder)
        self._row_toggle_filter = _ToggleSelectedTreeRowFilter(self.table)
        self.table.viewport().installEventFilter(self._row_toggle_filter)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.collapse_tree_button.clicked.connect(
            lambda: _collapse_brain_region_tree_to_major(self.table, self.hierarchy)
        )
        self.expand_tree_button.clicked.connect(lambda: _expand_brain_region_tree(self.table))
        self.expand_major_tree_button.clicked.connect(
            lambda: _expand_brain_region_tree_major(self.table)
        )
        self.current_frame_radio.toggled.connect(
            lambda checked: self._set_display_mode("frame") if checked else None
        )
        self.playback_summary_radio.toggled.connect(
            lambda checked: self._set_display_mode("summary") if checked else None
        )
        table_section_layout.addWidget(self.table, 1)
        table_section_layout.addLayout(tree_actions)

        self.selection_note = QLabel("Select a row to highlight that region in this result viewer.")
        self.selection_note.setObjectName("hint")
        self.selection_note.setWordWrap(True)
        table_section_layout.addWidget(self.selection_note)
        self.region_sections_splitter.addWidget(table_section)
        self.region_sections_splitter.setStretchFactor(0, 0)
        self.region_sections_splitter.setStretchFactor(1, 1)
        self.region_sections_splitter.setSizes([190, 520])
        root.addWidget(self.region_sections_splitter, 1)
        self._apply_mode_headers()
        self._table_state_by_mode["frame"] = self._capture_table_state()
        self._table_state_by_mode["summary"] = {
            "sort_column": 0,
            "sort_order": Qt.SortOrder.AscendingOrder,
            "hidden": [False] * self.table.columnCount(),
            "widths": [220, 126, 98, 102, 116, 100, 108, 92],
        }

    @staticmethod
    def _metric_text(value: float, suffix: str) -> str:
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):,.4g} {suffix}".rstrip()

    def _active_header_labels(self) -> tuple[str, ...]:
        return (
            self.SUMMARY_HEADER_LABELS
            if self._display_mode == "summary"
            else self.FRAME_HEADER_LABELS
        )

    def _active_units(self) -> tuple[str, ...]:
        return (
            self.SUMMARY_UNITS
            if self._display_mode == "summary"
            else self.FRAME_UNITS
        )

    def _filter_metric_choices(self) -> tuple[tuple[str, int | str], ...]:
        if self._display_mode == "summary":
            return (
                ("Playback RMS |B|", 1),
                ("Mean |B|", 2),
                ("Max P95 |B|", 3),
                ("Absolute peak |B|", 4),
                ("B-time", 5),
                ("B²-time", 6),
                ("Volume", 7),
                ("Side", "side"),
            )
        return (
            ("RMS |B|", 1),
            ("Mean |B|", 2),
            ("P95 |B|", 3),
            ("Peak |B|", 4),
            ("Uniformity", 5),
            ("Direction", 6),
            ("Volume", 7),
            ("Side", "side"),
        )

    def _capture_table_state(self) -> dict[str, Any]:
        header = self.table.header()
        return {
            "sort_column": int(max(0, header.sortIndicatorSection())),
            "sort_order": header.sortIndicatorOrder(),
            "hidden": [
                bool(self.table.isColumnHidden(column))
                for column in range(self.table.columnCount())
            ],
            "widths": [
                int(self.table.columnWidth(column))
                for column in range(self.table.columnCount())
            ],
        }

    def _restore_table_state(self, state: dict[str, Any]) -> None:
        hidden = list(state.get("hidden", []))
        widths = list(state.get("widths", []))
        for column in range(self.table.columnCount()):
            self.table.setColumnHidden(
                column, bool(hidden[column]) if column < len(hidden) else False
            )
            if column < len(widths):
                self.table.setColumnWidth(column, max(35, int(widths[column])))
        column = max(
            0,
            min(
                self.table.columnCount() - 1,
                int(state.get("sort_column", 0)),
            ),
        )
        order = state.get("sort_order", Qt.SortOrder.AscendingOrder)
        if order not in (Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder):
            order = Qt.SortOrder.AscendingOrder
        self.table.sortByColumn(column, order)
        self._update_sort_indicator(column, order)

    def _apply_mode_headers(self) -> None:
        labels = self._active_header_labels()
        tooltips = (
            (
                "LPBA40 structure",
                "Space-time RMS of field magnitude over the physical playback",
                "Time-weighted mean of the regional mean field magnitude",
                "Highest rendered-frame spatial 95th percentile",
                "Maximum field magnitude among retained atlas samples and regional frames",
                "Integral of instantaneous spatial RMS magnitude over physical time",
                "Integral of squared instantaneous spatial RMS magnitude over physical time",
                "Atlas volume after the selected brain object's scale",
            )
            if self._display_mode == "summary"
            else (
                "LPBA40 structure",
                "Spatial RMS of field magnitude in microtesla",
                "Volume-weighted mean field magnitude",
                "Volume-weighted 95th percentile field magnitude",
                "Maximum field magnitude among retained atlas analysis samples",
                "P5–P95 uniformity score; higher is more uniform",
                "Consistency of field-vector direction; higher is more aligned",
                "Atlas volume after the selected brain object's scale",
            )
        )
        header_item = self.table.headerItem()
        if header_item is not None:
            for column, (label, tooltip) in enumerate(
                zip(labels, tooltips, strict=True)
            ):
                header_item.setText(column, label)
                header_item.setToolTip(column, tooltip)
        for column, action in enumerate(self.column_actions):
            action.setText(labels[column])
        header = self.table.header()
        self._update_sort_indicator(
            max(0, header.sortIndicatorSection()), header.sortIndicatorOrder()
        )

    def _set_display_mode(self, mode: str) -> None:
        requested = "summary" if str(mode) == "summary" else "frame"
        if requested == "summary" and self._summary_values is None:
            return
        if requested == self._display_mode:
            return
        self._filter_rules_by_mode[self._display_mode] = copy.deepcopy(
            self._filter_rules()
        )
        self._table_state_by_mode[self._display_mode] = self._capture_table_state()
        self._display_mode = requested
        self._apply_mode_headers()
        self._restore_table_state(self._table_state_by_mode.get(requested, {}))
        self.set_filter_rules(
            copy.deepcopy(self._filter_rules_by_mode.get(requested, []))
        )
        summary = requested == "summary"
        self.playback_statistics_widget.setVisible(summary)
        if summary:
            self.metrics_note.setText(
                "Rows summarize the complete physical playback. B-time and B²-time "
                "are comparative magnetic-field exposure indices, not a biological dose. "
                "Summary filters and sorting are independent from Current frame."
            )
            self._show_summary_values()
        else:
            self.metrics_note.setText(
                "Rows summarize volume-weighted atlas samples for the current frame. "
                "Filters and sorting belong to this mode and result tab only."
            )
            self._show_frame_values(
                self._current_frame_index, self._current_frame_time_s
            )

    @staticmethod
    def _run_number(value: Any, *, decimals: int = 6) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if not math.isfinite(number):
            return "—"
        rounded = round(number)
        if math.isclose(number, rounded, rel_tol=1e-10, abs_tol=1e-10):
            return f"{int(rounded):,}"
        return f"{number:,.{decimals}g}"

    def _update_playback_statistics(self) -> None:
        metadata = self._playback_metadata
        duration = self._run_number(metadata.get("exposure_time_s"))
        self.playback_statistic_values["exposure"].setText(
            f"{duration} s of physical waveform time"
        )

        mode = str(metadata.get("waveform_mode", "custom")).strip().lower()
        mode_label = {
            "sine": "Sine",
            "square": "Square",
            "triangle": "Triangle",
            "custom": "Custom trace",
        }.get(mode, mode.replace("_", " ").title() or "Custom trace")
        cycles = metadata.get("waveform_cycle_count")
        frequency = metadata.get("waveform_frequency_hz")
        if cycles is None or frequency is None:
            waveform_text = (
                f"{mode_label} • {int(metadata.get('waveform_pass_count', 1) or 1):,} "
                "playback pass"
            )
        else:
            waveform_text = (
                f"{mode_label} • {self._run_number(frequency)} Hz • "
                f"{self._run_number(cycles)} cycles"
            )
        self.playback_statistic_values["waveform"].setText(waveform_text)

        if not bool(metadata.get("sequencer_enabled", False)):
            sequencer_text = "Off"
        else:
            step_count = int(metadata.get("sequencer_step_count", 0) or 0)
            activations = int(metadata.get("sequencer_step_activations", 0) or 0)
            if bool(metadata.get("sequencer_loop_enabled", False)):
                loops = self._run_number(metadata.get("sequencer_equivalent_loops", 0.0))
                sequencer_text = (
                    f"{step_count:,} configured steps • {loops} loops • "
                    f"{activations:,} step activations"
                )
            else:
                sequencer_text = (
                    f"{step_count:,} configured steps • single pass • "
                    f"{activations:,} step activations"
                )
        self.playback_statistic_values["sequencer"].setText(sequencer_text)

        frame_count = int(metadata.get("frame_count", 0) or 0)
        analysis_count = int(metadata.get("analysis_sample_count", 0) or 0)
        regional_count = int(metadata.get("regional_frame_sample_count", 0) or 0)
        atlas_count = int(metadata.get("atlas_sample_count", 0) or 0)
        sampling_text = (
            f"{frame_count:,} rendered frames • {analysis_count:,} time samples • "
            f"{atlas_count:,} atlas samples"
        )
        if regional_count and regional_count != frame_count:
            sampling_text += f" • {regional_count:,} retained regional frames"
        self.playback_statistic_values["sampling"].setText(sampling_text)

    def _playback_statistic_lines(self) -> list[str]:
        return [
            f"Exposure time: {self.playback_statistic_values['exposure'].text()}",
            f"Waveform: {self.playback_statistic_values['waveform'].text()}",
            f"Sequencer: {self.playback_statistic_values['sequencer'].text()}",
            f"Sampling: {self.playback_statistic_values['sampling'].text()}",
        ]

    def _populate(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.clear()
        self._items_by_key.clear()

        def add_node(node: BrainRegionNode, parent: QTreeWidgetItem | None) -> None:
            item = _NumericTreeItem([node.name, "—", "—", "—", "—", "—", "—", "—"])
            item.setData(0, Qt.ItemDataRole.UserRole, node.name.casefold())
            item.setData(0, REGION_KEY_ROLE, node.key)
            item.setData(0, REGION_LABEL_IDS_ROLE, list(node.label_ids))
            side = "left" if node.key.endswith(".left") else "right" if node.key.endswith(".right") else ""
            item.setData(0, REGION_SIDE_ROLE, side)
            self._items_by_key[node.key] = item
            if parent is None:
                self.table.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for child in node.children:
                add_node(child, item)

        add_node(self.hierarchy, None)
        root_item = self._items_by_key.get(self.hierarchy.key)
        if root_item is not None:
            root_item.setExpanded(True)
            for index in range(root_item.childCount()):
                root_item.child(index).setExpanded(True)
        self.table.setSortingEnabled(True)

    def _update_sort_indicator(self, column: int, order: Qt.SortOrder) -> None:
        header_item = self.table.headerItem()
        if header_item is None:
            return
        labels = self._active_header_labels()
        for index, label in enumerate(labels):
            header_item.setText(index, label)
        if 0 <= int(column) < len(labels):
            arrow = "▲" if order == Qt.SortOrder.AscendingOrder else "▼"
            header_item.setText(int(column), f"{labels[int(column)]}  {arrow}")

    def _show_column_menu(self, position) -> None:
        for column, action in enumerate(self.column_actions):
            action.blockSignals(True)
            action.setChecked(not self.table.isColumnHidden(column))
            action.blockSignals(False)
        header = self.table.header()
        self.column_menu.exec(header.mapToGlobal(position))

    def _resort(self) -> None:
        header = self.table.header()
        self.table.sortItems(max(0, header.sortIndicatorSection()), header.sortIndicatorOrder())

    def _filter_row_for_widget(self, widget: QWidget) -> int:
        for row in range(self.filter_table.rowCount()):
            for column in range(self.filter_table.columnCount()):
                if self.filter_table.cellWidget(row, column) is widget:
                    return row
        return -1

    def _prepare_filter_editor(self, editor: QWidget) -> None:
        """Keep result-tab filter editors deliberate, like the Selector tab."""
        row = self._filter_row_for_widget(editor)
        for target in (editor, *editor.findChildren(QWidget)):
            target.installEventFilter(self._filter_wheel_filter)
            target.setProperty("brainFilterRow", row)
            target.setMouseTracking(False)
            target.setAttribute(Qt.WidgetAttribute.WA_Hover, False)

        # Only the public cell editor gets ClickFocus. Do not turn spin-box
        # implementation children into independent focus targets.
        if isinstance(editor, (QComboBox, QDoubleSpinBox, QSpinBox, QPushButton)):
            editor.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        else:
            editor.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        editor.setProperty("brainFilterSelected", row == self._selected_filter_row)

    def _refresh_filter_selection_visual(self) -> None:
        selected_row = self._selected_filter_row
        palette = self.filter_table.palette()
        for row in range(self.filter_table.rowCount()):
            selected = row == selected_row
            number_item = self.filter_table.item(row, 0)
            if number_item is not None:
                number_item.setData(
                    Qt.ItemDataRole.BackgroundRole,
                    palette.brush(QPalette.ColorRole.Highlight) if selected else None,
                )
                number_item.setData(
                    Qt.ItemDataRole.ForegroundRole,
                    palette.brush(QPalette.ColorRole.HighlightedText) if selected else None,
                )
            for column in (1, 2, 3, 4):
                editor = self.filter_table.cellWidget(row, column)
                if editor is None:
                    continue
                editor.setProperty("brainFilterSelected", selected)
                editor.style().unpolish(editor)
                editor.style().polish(editor)
                editor.update()

    def _set_filter_selected_row(self, row: int) -> None:
        self._selected_filter_row = (
            int(row) if 0 <= int(row) < self.filter_table.rowCount() else -1
        )
        self.filter_table.clearSelection()
        self.filter_table.setCurrentCell(-1, -1)
        self._refresh_filter_selection_visual()
        self.remove_filter_button.setEnabled(self._selected_filter_row >= 0)

    def _filter_cell_clicked(self, row: int, column: int) -> None:
        if int(column) == 0:
            self._set_filter_selected_row(int(row))

    def _renumber_filter_rows(self) -> None:
        for row in range(self.filter_table.rowCount()):
            item = self.filter_table.item(row, 0)
            if item is None:
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip("Click here to select this filter")
                self.filter_table.setItem(row, 0, item)
            item.setText(str(row + 1))
            for column in (1, 2, 3, 4):
                editor = self.filter_table.cellWidget(row, column)
                if editor is None:
                    continue
                for target in (editor, *editor.findChildren(QWidget)):
                    target.setProperty("brainFilterRow", row)
        self._refresh_filter_selection_visual()

    def _add_filter(self, *_args) -> None:
        row = self.filter_table.rowCount()
        self.filter_table.insertRow(row)
        number = QTableWidgetItem(str(row + 1))
        number.setFlags(Qt.ItemFlag.ItemIsEnabled)
        number.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        number.setToolTip("Click here to select this filter")
        self.filter_table.setItem(row, 0, number)
        metric = QComboBox()
        for label, data in self._filter_metric_choices():
            metric.addItem(label, data)
        condition = QComboBox()
        value = CompactDoubleSpinBox()
        value.setKeyboardTracking(False)
        action_cell = QWidget()
        action_layout = QHBoxLayout(action_cell)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        action = QComboBox()
        action.setObjectName("brainFilterActionCombo")
        action.addItem("Filter", "filter")
        action.addItem("Highlight", "highlight")
        colour = QPushButton()
        colour.setObjectName("brainFilterColourButton")
        colour.setFixedWidth(26)
        _set_brain_filter_colour_button(colour, BRAIN_FILTER_DEFAULT_HIGHLIGHT)
        colour.setVisible(False)
        action_layout.addWidget(action, 1)
        action_layout.addWidget(colour, 0)
        self.filter_table.setCellWidget(row, 1, metric)
        self.filter_table.setCellWidget(row, 2, condition)
        self.filter_table.setCellWidget(row, 3, value)
        self.filter_table.setCellWidget(row, 4, action_cell)
        for editor in (metric, condition, value, action_cell):
            self._prepare_filter_editor(editor)
        metric.currentIndexChanged.connect(lambda _index, selected=row: self._configure_filter_row(selected))
        condition.currentIndexChanged.connect(self._apply_filters)
        value.valueChanged.connect(self._apply_filters)
        action.currentIndexChanged.connect(
            lambda _index, selected=row: self._filter_action_changed(selected)
        )
        colour.clicked.connect(lambda _checked=False, button=colour: self._pick_filter_colour(button))
        self._configure_filter_row(row)
        self._renumber_filter_rows()
        self._set_filter_selected_row(row)
        self._apply_filters()

    def _filter_action_changed(self, row: int) -> None:
        cell = self.filter_table.cellWidget(row, 4)
        action, colour = _brain_filter_action_widgets(cell)
        if action is None or colour is None:
            return
        colour.setVisible(str(action.currentData()) == "highlight")
        self._apply_filters()

    def _pick_filter_colour(self, button: QPushButton) -> None:
        current = QColor(str(button.property("brainFilterHighlightColour") or BRAIN_FILTER_DEFAULT_HIGHLIGHT))
        chosen = QColorDialog.getColor(current, self, "Choose Brain Areas highlight colour")
        if not chosen.isValid():
            return
        _set_brain_filter_colour_button(button, chosen.name(QColor.NameFormat.HexRgb))
        self._apply_filters()

    def _configure_filter_row(self, row: int) -> None:
        metric = self.filter_table.cellWidget(row, 1)
        condition = self.filter_table.cellWidget(row, 2)
        if not isinstance(metric, QComboBox) or not isinstance(condition, QComboBox):
            return
        key = metric.currentData()
        condition.blockSignals(True)
        condition.clear()
        if key == "side":
            condition.addItem("=", "=")
            value = self.filter_table.cellWidget(row, 3)
            if not isinstance(value, QComboBox):
                value = QComboBox()
                value.addItem("Left", "left")
                value.addItem("Right", "right")
                self.filter_table.setCellWidget(row, 3, value)
                self._prepare_filter_editor(value)
                value.currentIndexChanged.connect(self._apply_filters)
        else:
            condition.addItem(">", ">")
            condition.addItem("<", "<")
            value = self.filter_table.cellWidget(row, 3)
            if not isinstance(value, QDoubleSpinBox):
                value = CompactDoubleSpinBox()
                value.setKeyboardTracking(False)
                self.filter_table.setCellWidget(row, 3, value)
                self._prepare_filter_editor(value)
                value.valueChanged.connect(self._apply_filters)
            column = int(key)
            value.setSuffix("")
            value.setDecimals(8)
            if isinstance(value, CompactDoubleSpinBox):
                value.setMinimumDisplayDecimals(2)
            line_edit = value.lineEdit()
            if line_edit is not None:
                line_edit.setMaxLength(48)
            unit = self._active_units()[column]
            if unit == "%":
                value.setRange(0.0, 100.0)
            elif unit == "cm³":
                value.setRange(0.0, 1e12)
            else:
                value.setRange(0.0, 1e30)
            value.setSuffix(f" {unit}" if unit else "")
        condition.blockSignals(False)
        self._apply_filters()

    def _remove_filter(self, *_args) -> None:
        row = self._selected_filter_row
        if row < 0 or row >= self.filter_table.rowCount():
            return
        self.filter_table.removeRow(row)
        self._selected_filter_row = -1
        self._renumber_filter_rows()
        self._set_filter_selected_row(-1)
        self._apply_filters()

    def set_filter_rules(
        self,
        rules: list[tuple[int | str, str, float | str, str, str]],
    ) -> None:
        """Replace this result tab's filters with an independent rule snapshot.

        The Selector owns the source rules.  New rendered tabs receive a copy
        at creation time and are then free to edit that copy without changing
        the Selector or any sibling result tab.
        """

        self._loading_filter_rules = True
        self.filter_table.setRowCount(0)
        self._selected_filter_row = -1
        self.remove_filter_button.setEnabled(False)
        for raw_rule in rules:
            if len(raw_rule) < 3:
                continue
            metric_key = raw_rule[0]
            condition_key = str(raw_rule[1])
            rule_value = raw_rule[2]
            action_key = str(raw_rule[3] if len(raw_rule) > 3 else "filter")
            colour_value = str(
                raw_rule[4] if len(raw_rule) > 4 else BRAIN_FILTER_DEFAULT_HIGHLIGHT
            )

            self._add_filter()
            row = self.filter_table.rowCount() - 1
            metric = self.filter_table.cellWidget(row, 1)
            if not isinstance(metric, QComboBox):
                continue
            metric_index = metric.findData(metric_key)
            if metric_index >= 0:
                metric.setCurrentIndex(metric_index)

            condition = self.filter_table.cellWidget(row, 2)
            if isinstance(condition, QComboBox):
                condition_index = condition.findData(condition_key)
                if condition_index >= 0:
                    condition.setCurrentIndex(condition_index)

            value = self.filter_table.cellWidget(row, 3)
            if metric_key == "side" and isinstance(value, QComboBox):
                value_index = value.findData(str(rule_value))
                if value_index >= 0:
                    value.setCurrentIndex(value_index)
            elif isinstance(value, QDoubleSpinBox):
                try:
                    value.setValue(float(rule_value))
                except (TypeError, ValueError):
                    pass

            action, colour = _brain_filter_action_widgets(
                self.filter_table.cellWidget(row, 4)
            )
            if action is not None:
                action_index = action.findData(action_key)
                if action_index >= 0:
                    action.setCurrentIndex(action_index)
            if colour is not None:
                _set_brain_filter_colour_button(colour, colour_value)
                colour.setVisible(action_key == "highlight")

        self._loading_filter_rules = False
        self._set_filter_selected_row(-1)
        self._filter_rules_by_mode[self._display_mode] = copy.deepcopy(
            self._filter_rules()
        )
        self._apply_filters()

    def _filter_rules(self) -> list[tuple[int | str, str, float | str, str, str]]:
        rules: list[tuple[int | str, str, float | str, str, str]] = []
        for row in range(self.filter_table.rowCount()):
            metric = self.filter_table.cellWidget(row, 1)
            condition = self.filter_table.cellWidget(row, 2)
            value = self.filter_table.cellWidget(row, 3)
            action, colour_button = _brain_filter_action_widgets(
                self.filter_table.cellWidget(row, 4)
            )
            if (
                not isinstance(metric, QComboBox)
                or not isinstance(condition, QComboBox)
                or action is None
            ):
                continue
            action_key = str(action.currentData() or "filter")
            colour = str(
                colour_button.property("brainFilterHighlightColour")
                if colour_button is not None
                else BRAIN_FILTER_DEFAULT_HIGHLIGHT
            )
            key = metric.currentData()
            if key == "side" and isinstance(value, QComboBox):
                rules.append(("side", "=", str(value.currentData()), action_key, colour))
            elif isinstance(value, QDoubleSpinBox):
                rules.append(
                    (int(key), str(condition.currentData()), float(value.value()), action_key, colour)
                )
        return rules

    def _apply_filters(self, *_args) -> None:
        # Persistent cell widgets can briefly make QTableWidget assign a current
        # index while focus changes. Explicit rule selection is separate state,
        # so discard that transient index on every result-tab filter refresh.
        self.filter_table.setCurrentCell(-1, -1)
        rules = self._filter_rules()
        if not self._loading_filter_rules:
            self._filter_rules_by_mode[self._display_mode] = copy.deepcopy(rules)
        filter_rules = [rule for rule in rules if rule[3] == "filter"]
        highlight_rules = [rule for rule in rules if rule[3] == "highlight"]
        root_item = self._items_by_key.get(self.hierarchy.key)
        if root_item is None:
            return

        def filter_excludes(item: QTreeWidgetItem) -> bool:
            # Filter actions are exclusion rules: rows satisfying all active
            # Filter conditions are removed.  Multiple Filter rules therefore
            # retain the long-standing AND relationship while matching the UI
            # wording: e.g. RMS < 1 µT removes rows below 1 µT.
            return bool(filter_rules) and all(
                _brain_item_matches_filter_rule(item, rule) for rule in filter_rules
            )

        def apply(item: QTreeWidgetItem) -> bool:
            child_visible = False
            for index in range(item.childCount()):
                child_visible = apply(item.child(index)) or child_visible
            # A matching parent must remain visible when it is needed as the
            # hierarchy path to a non-matching visible child.
            visible = not filter_excludes(item) or child_visible
            item.setHidden(not visible)
            return visible

        apply(root_item)

        # Highlight actions do not participate in row visibility. Rules are
        # evaluated top-to-bottom; when several match, the later rule wins.
        for item in self._items_by_key.values():
            _clear_brain_item_filter_highlight(item, self.table.columnCount())
            selected_colour = ""
            for rule in highlight_rules:
                if _brain_item_matches_filter_rule(item, rule):
                    selected_colour = rule[4]
            if selected_colour:
                _apply_brain_item_filter_highlight(
                    item, selected_colour, self.table.columnCount()
                )

        current = self.table.currentItem()
        if current is not None and current.isHidden():
            self.table.clearSelection()

    def set_static_metrics(self, metrics: dict[str, BrainRegionMetrics]) -> None:
        self.frame_note.setText("Static field")
        self.table.setSortingEnabled(False)
        for key, metric in metrics.items():
            item = self._items_by_key.get(key)
            if item is None:
                continue
            values = (
                metric.rms_uT, metric.mean_uT, metric.p95_uT, metric.peak_uT,
                metric.uniformity_pct, metric.directional_consistency_pct, metric.volume_cm3,
            )
            self._set_item_values(
                item,
                values,
                metric.sample_count,
                metric.voxel_count,
                units=self.FRAME_UNITS[1:],
            )
        self.table.setSortingEnabled(True)
        self._resort()
        self._apply_filters()

    def set_frame_metric_data(self, data: dict[str, Any]) -> None:
        self._frame_keys = [str(value) for value in data.get("keys", [])]
        shape = tuple(int(value) for value in data.get("shape", []))
        raw = base64.b64decode(str(data.get("values_b64", "")))
        values = np.frombuffer(raw, dtype=np.float32)
        if len(shape) != 3 or int(np.prod(shape)) != values.size:
            raise BrainAnalysisError("Animated Brain Areas metric payload has an invalid shape.")
        self._frame_values = values.reshape(shape)
        self._source_frame_count = max(int(data.get("source_frame_count", shape[0])), int(shape[0]))
        encoded_indices = str(data.get("sampled_frame_indices_b64", ""))
        if encoded_indices:
            indices = np.frombuffer(base64.b64decode(encoded_indices), dtype=np.int32)
            self._frame_sample_indices = indices if len(indices) == shape[0] else None
        else:
            self._frame_sample_indices = None
        self._sample_counts = [int(value) for value in data.get("sample_counts", [])]
        self._voxel_counts = [int(value) for value in data.get("voxel_counts", [])]
        summary = data.get("playback_summary")
        if isinstance(summary, dict):
            summary_shape = tuple(int(value) for value in summary.get("shape", []))
            summary_raw = base64.b64decode(str(summary.get("values_b64", "")))
            summary_values = np.frombuffer(summary_raw, dtype=np.float64)
            if (
                summary_shape != (len(self._frame_keys), 7)
                or int(np.prod(summary_shape)) != summary_values.size
            ):
                raise BrainAnalysisError(
                    "Animated Brain Areas playback summary has an invalid shape."
                )
            peak_raw = base64.b64decode(
                str(summary.get("peak_times_s_b64", ""))
            )
            peak_times = np.frombuffer(peak_raw, dtype=np.float64)
            if peak_times.shape != (len(self._frame_keys),):
                raise BrainAnalysisError(
                    "Animated Brain Areas playback peak times are invalid."
                )
            self._summary_values = summary_values.reshape(summary_shape)
            self._summary_peak_times_s = peak_times
            self._playback_metadata = copy.deepcopy(
                summary.get("playback")
                if isinstance(summary.get("playback"), dict)
                else {}
            )
            self._update_playback_statistics()
            self.playback_summary_radio.setEnabled(True)
        else:
            self._summary_values = None
            self._summary_peak_times_s = None
            self._playback_metadata = {}
            self.playback_summary_radio.setEnabled(False)
        self.mode_controls.setVisible(True)
        self.request_frame(0, 0.0)

    def video_frame_metric_payload(self) -> dict[str, Any] | None:
        """Re-encode the cached animated LPBA40 metrics for video export."""

        if self._frame_values is None:
            return None
        values = np.ascontiguousarray(self._frame_values, dtype=np.float32)
        payload = {
            "keys": list(self._frame_keys),
            "shape": [int(value) for value in values.shape],
            "values_b64": base64.b64encode(values.tobytes()).decode("ascii"),
            "source_frame_count": int(self._source_frame_count or len(values)),
            "sampled_frame_indices_b64": (
                base64.b64encode(np.ascontiguousarray(self._frame_sample_indices, dtype=np.int32).tobytes()).decode("ascii")
                if self._frame_sample_indices is not None else ""
            ),
            "sample_counts": list(self._sample_counts),
            "voxel_counts": list(self._voxel_counts),
        }
        if self._summary_values is not None and self._summary_peak_times_s is not None:
            summary_values = np.ascontiguousarray(
                self._summary_values, dtype=np.float64
            )
            peak_times = np.ascontiguousarray(
                self._summary_peak_times_s, dtype=np.float64
            )
            payload["playback_summary"] = {
                "shape": [int(value) for value in summary_values.shape],
                "values_b64": base64.b64encode(summary_values.tobytes()).decode("ascii"),
                "peak_times_s_b64": base64.b64encode(peak_times.tobytes()).decode("ascii"),
                "playback": copy.deepcopy(self._playback_metadata),
            }
        return payload

    def request_frame(self, frame_index: int, time_s: float) -> None:
        self._pending_frame = (int(frame_index), float(time_s))
        if self._frame_update_scheduled:
            return
        self._frame_update_scheduled = True
        QTimer.singleShot(0, self._apply_pending_frame)

    def _apply_pending_frame(self) -> None:
        self._frame_update_scheduled = False
        pending = self._pending_frame
        self._pending_frame = None
        if pending is None or self._frame_values is None or not len(self._frame_values):
            return
        frame_index, time_s = pending
        self._current_frame_index = int(frame_index)
        self._current_frame_time_s = float(time_s)
        if self._display_mode != "frame":
            return
        self._show_frame_values(frame_index, time_s)

    def _show_frame_values(self, frame_index: int, time_s: float) -> None:
        if self._frame_values is None or not len(self._frame_values):
            return
        source_frame_index = max(0, min(max(0, self._source_frame_count - 1), frame_index))
        cache_index = source_frame_index
        if self._frame_sample_indices is not None and len(self._frame_sample_indices):
            insertion = int(np.searchsorted(self._frame_sample_indices, source_frame_index, side="left"))
            insertion = min(insertion, len(self._frame_sample_indices) - 1)
            if insertion > 0 and abs(int(self._frame_sample_indices[insertion - 1]) - source_frame_index) <= abs(int(self._frame_sample_indices[insertion]) - source_frame_index):
                insertion -= 1
            cache_index = insertion
        elif self._source_frame_count > 1 and len(self._frame_values) > 1:
            cache_index = int(round(source_frame_index * (len(self._frame_values) - 1) / (self._source_frame_count - 1)))
        cache_index = max(0, min(len(self._frame_values) - 1, cache_index))
        self.frame_note.setText(f"Frame {source_frame_index + 1:,}/{max(1, self._source_frame_count):,} • {time_s:.4g} s")
        self.table.setSortingEnabled(False)
        frame = self._frame_values[cache_index]
        for row_index, key in enumerate(self._frame_keys):
            item = self._items_by_key.get(key)
            if item is None or row_index >= len(frame):
                continue
            sample_count = self._sample_counts[row_index] if row_index < len(self._sample_counts) else 0
            voxel_count = self._voxel_counts[row_index] if row_index < len(self._voxel_counts) else 0
            self._set_item_values(
                item,
                frame[row_index],
                sample_count,
                voxel_count,
                units=self.FRAME_UNITS[1:],
            )
        self.table.setSortingEnabled(True)
        self._resort()
        self._apply_filters()

    def _show_summary_values(self) -> None:
        if self._summary_values is None:
            return
        duration = self._run_number(self._playback_metadata.get("exposure_time_s"))
        self.frame_note.setText(f"Full playback • {duration} s")
        self.table.setSortingEnabled(False)
        for row_index, key in enumerate(self._frame_keys):
            item = self._items_by_key.get(key)
            if item is None or row_index >= len(self._summary_values):
                continue
            sample_count = (
                self._sample_counts[row_index]
                if row_index < len(self._sample_counts)
                else 0
            )
            voxel_count = (
                self._voxel_counts[row_index]
                if row_index < len(self._voxel_counts)
                else 0
            )
            peak_time = (
                float(self._summary_peak_times_s[row_index])
                if self._summary_peak_times_s is not None
                and row_index < len(self._summary_peak_times_s)
                else None
            )
            self._set_item_values(
                item,
                self._summary_values[row_index],
                sample_count,
                voxel_count,
                units=self.SUMMARY_UNITS[1:],
                peak_time_s=peak_time,
            )
        self.table.setSortingEnabled(True)
        self._resort()
        self._apply_filters()

    def _set_item_values(
        self,
        item: QTreeWidgetItem,
        values,
        sample_count: int,
        voxel_count: int,
        *,
        units: tuple[str, ...],
        peak_time_s: float | None = None,
    ) -> None:
        for column, (value, suffix) in enumerate(
            zip(values, units, strict=True), start=1
        ):
            number = float(value)
            item.setText(column, self._metric_text(number, suffix))
            item.setData(column, Qt.ItemDataRole.UserRole, number)
        item.setToolTip(0, f"{voxel_count:,} atlas voxels; {sample_count:,} retained analysis samples")
        if peak_time_s is not None and math.isfinite(float(peak_time_s)):
            item.setToolTip(
                4,
                "Maximum retained atlas-sample field magnitude occurred at "
                f"{float(peak_time_s):.6g} s of physical playback time.",
            )
        else:
            item.setToolTip(4, "")

    def video_table_snapshot(self) -> dict[str, Any]:
        snapshot = _brain_region_video_table_snapshot(
            self.hierarchy,
            self.table,
            self._filter_rules(),
            self._active_header_labels(),
        )
        snapshot.update(
            {
                "mode": self._display_mode,
                "units": list(self._active_units()),
                "title": (
                    "Brain areas — Playback summary"
                    if self._display_mode == "summary"
                    else "Brain areas"
                ),
            }
        )
        if self._display_mode == "summary" and self._summary_values is not None:
            snapshot["summary_values"] = np.asarray(
                self._summary_values, dtype=float
            ).tolist()
            snapshot["summary_lines"] = self._playback_statistic_lines()
        return snapshot

    def _selection_changed(self) -> None:
        selected = self.table.selectedItems()
        item = selected[0] if selected else None
        if item is None:
            self.selection_note.setText("No region highlighted in this result viewer.")
            self.region_selected.emit(())
            return
        label_ids = tuple(int(value) for value in (item.data(0, REGION_LABEL_IDS_ROLE) or []))
        self.selection_note.setText(
            f"Highlighted: {item.text(0)} • {len(label_ids)} LPBA40 region"
            f"{'s' if len(label_ids) != 1 else ''}."
        )
        self.region_selected.emit(label_ids)


def _brain_region_video_table_snapshot(
    hierarchy: BrainRegionNode,
    tree: QTreeWidget,
    rules: list[tuple[int | str, str, float | str, str, str]],
    header_labels: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Freeze the active Brain Areas view for exported-video table rendering."""

    rows: list[dict[str, Any]] = []

    def add_node(node: BrainRegionNode, parent_key: str | None, depth: int) -> None:
        side = (
            "left" if node.key.endswith(".left")
            else "right" if node.key.endswith(".right")
            else ""
        )
        rows.append(
            {
                "key": str(node.key),
                "name": str(node.name),
                "parent_key": parent_key,
                "depth": int(depth),
                "side": side,
            }
        )
        for child in node.children:
            add_node(child, str(node.key), depth + 1)

    add_node(hierarchy, None, 0)

    expanded_keys: list[str] = []

    def collect_expansion(item: QTreeWidgetItem) -> None:
        key = str(item.data(0, REGION_KEY_ROLE) or "")
        if key and item.isExpanded():
            expanded_keys.append(key)
        for index in range(item.childCount()):
            collect_expansion(item.child(index))

    for index in range(tree.topLevelItemCount()):
        collect_expansion(tree.topLevelItem(index))

    header = tree.header()
    visual_columns = [
        int(header.logicalIndex(index))
        for index in range(header.count())
        if not tree.isColumnHidden(int(header.logicalIndex(index)))
    ]
    filters = [
        {
            "metric": metric,
            "operator": str(operator),
            "value": value,
        }
        for metric, operator, value, action, _colour in rules
        if action == "filter"
    ]
    highlights = [
        {
            "metric": metric,
            "operator": str(operator),
            "value": value,
            "colour": str(colour),
        }
        for metric, operator, value, action, colour in rules
        if action == "highlight"
    ]
    return {
        "enabled": True,
        "title": "Brain areas",
        "mode": "frame",
        "units": ["", "µT", "µT", "µT", "µT", "%", "%", "cm³"],
        "headers": [str(value) for value in header_labels],
        "rows": rows,
        "filters": filters,
        "highlights": highlights,
        "sort_column": int(max(0, header.sortIndicatorSection())),
        "sort_order": (
            "ascending"
            if header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
            else "descending"
        ),
        "expanded_keys": expanded_keys,
        "visible_columns": visual_columns,
        "column_widths": [int(tree.columnWidth(index)) for index in range(tree.columnCount())],
    }


class _BrainResultTab(QWidget):
    """One rendered Brain View result plus its independent Brain Areas panel."""

    def __init__(self, viewer: WaveformGLView, panel: _BrainResultRegionPanel, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.region_panel = panel
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(viewer)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([820, 560])
        layout.addWidget(splitter, 1)
        self.splitter = splitter


def _percentage_spin(value: float) -> QDoubleSpinBox:
    editor = QDoubleSpinBox()
    editor.setRange(0.0, 100.0)
    editor.setDecimals(1)
    editor.setSingleStep(5.0)
    editor.setSuffix(" %")
    editor.setValue(float(value))
    return editor


class BrainViewDialog(
    _BackgroundFieldRenderMixin,
    _CameraTimelineMixin,
    _PlaybackCoilSelectionMixin,
    QDialog,
):
    """Analyze a detached scene on volume-weighted LPBA40 atlas samples."""

    def __init__(self, adapter: StudioAdapter, brain_object_id: str, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter.detached_copy()
        self.brain_object_id = str(brain_object_id)
        try:
            brain_object = self.adapter.get_object(self.brain_object_id)
        except KeyError as error:
            raise BrainAnalysisError(
                "The selected SRI24 brain is no longer present in the scene snapshot."
            ) from error
        if brain_object.get("builtin_template") != "sri24_brain":
            raise BrainAnalysisError(
                "Brain View requires a selected SRI24 brain volume."
            )
        self.brain_object_label = str(brain_object.get("label", "SRI24 brain"))
        self.parcellation: SRI24Parcellation = load_sri24_parcellation("lpba40")
        self.region_hierarchy = lpba40_region_hierarchy(self.parcellation)
        self._division_key_by_label: dict[int, str] = {
            int(label_id): division.key
            for division in self.region_hierarchy.children
            for label_id in division.label_ids
        }
        mesh = self.adapter._mesh(self.brain_object_id)
        position_m, rotation = self.adapter._mesh_world_transform(mesh)
        scale_xyz = self.adapter._mesh_scale_xyz(mesh)
        try:
            self.region_meshes: LPBA40MeshSet = load_lpba40_region_meshes(
                self.parcellation,
                position_m=position_m,
                rotation=rotation,
                scale_xyz=scale_xyz,
                visual_stride=LPBA40_VISUAL_MESH_STRIDE,
            )
        except BrainMeshError as error:
            raise BrainAnalysisError(str(error)) from error
        self.analysis_samples = sample_lpba40_voxels(
            self.parcellation,
            position_m=position_m,
            rotation=rotation,
            scale_xyz=scale_xyz,
            maximum_samples_per_label=STATIC_SAMPLES_PER_LPBA40_LABEL,
        )
        self.display_samples = downsample_brain_samples(
            self.analysis_samples,
            maximum_points=MAXIMUM_DISPLAY_ATLAS_POINTS,
        )
        # Brain View map controls are Workbench scene coordinates rather than
        # atlas-image coordinates.  The canonical SRI24 brain is centred higher
        # than the scene origin, so default the field-map cube to Z = +50 mm
        # while keeping the same 300 mm cube width used by the other 3D maps.
        self._brain_centre_mm = [0.0, 0.0, 50.0]
        self._brain_span_mm = 300.0
        self._field_vectors_t: np.ndarray | None = None
        self._region_metrics: dict[str, BrainRegionMetrics] = {}
        self._items_by_key: dict[str, QTreeWidgetItem] = {}
        self._selector_scene_figure: dict[str, Any] | None = None
        self._selector_figure_initialized = False
        self._region_trace_index: dict[int, int] = {}
        self._scene_trace_indices: dict[str, list[int]] = {}
        self._scene_legend_trace_indices: dict[str, list[int]] = {}
        self._scene_trace_opacity_scale: dict[int, float] = {}
        self._map_volume_trace_index: int | None = None
        self._selector_grid_visible_layout: dict[str, Any] = {}
        self._last_selected_region_labels: set[int] = set()
        self._map_tab_counter = 0
        self._waveform_tab_counter = 0
        self._render_job_serial = 0
        self._render_jobs: dict[int, dict[str, Any]] = {}
        self._render_document_snapshot: dict[str, Any] | None = None
        self._active_render_view: WaveformGLView | None = None
        self.setWindowTitle("Brain View")
        self.resize(1680, 900)

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter = splitter
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        control_column = QWidget()
        control_column.setMinimumWidth(310)
        control_column_layout = QVBoxLayout(control_column)
        control_column_layout.setContentsMargins(0, 0, 0, 0)
        control_column_layout.setSpacing(0)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        control_panel = QWidget()
        control_panel.setMinimumWidth(0)
        control_panel.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        control_root = QVBoxLayout(control_panel)
        control_root.setContentsMargins(8, 8, 8, 8)
        control_scroll.setWidget(control_panel)
        control_column_layout.addWidget(control_scroll, 1)

        self.map_volume_section = CollapsibleSection("Map volume", expanded=False)
        map_volume_layout = QGridLayout()
        self.coordinates: list[QDoubleSpinBox] = []
        for row, (axis, value) in enumerate(zip("XYZ", self._brain_centre_mm)):
            spin = _spin(float(value), -10000, 10000, 2, " mm")
            spin.setToolTip(
                f"Centre {axis} coordinate of the field-map cube in Workbench coordinates"
            )
            self.coordinates.append(spin)
            map_volume_layout.addWidget(QLabel(f"Centre {axis}"), row, 0)
            map_volume_layout.addWidget(spin, row, 1)
        self.span = _spin(float(self._brain_span_mm), 1, 100000, 1, " mm")
        self.span.setToolTip(
            "Width of the cubic field volume used by Static analysis, Animated analysis, and video export"
        )
        map_volume_layout.addWidget(QLabel("Cube width"), 3, 0)
        map_volume_layout.addWidget(self.span, 3, 1)
        map_volume_layout.setColumnStretch(1, 1)
        self.map_volume_section.setContentLayout(map_volume_layout)
        control_root.addWidget(self.map_volume_section)

        self._init_scene_options()
        control_root.addWidget(self.scene_options_section)

        self.logarithmic = QCheckBox("Log scale")
        self.logarithmic.setToolTip(
            "Use a log₁₀ colour scale for atlas sample |B| values."
        )
        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(True)
        self.intensity_opacity_enabled = QCheckBox("Intensity-linked opacity")
        self.intensity_opacity_enabled.setChecked(True)
        self.volume_opacity_lower = _percentage_spin(18.0)
        self.volume_opacity_upper = _percentage_spin(72.0)
        self.volume_opacity_lower_level = _percentage_spin(0.0)
        self.volume_opacity_upper_level = _percentage_spin(100.0)
        self.colour_minimum_enabled = QCheckBox("Lower limit")
        self.colour_minimum = QDoubleSpinBox()
        self.colour_minimum.setRange(0.0, 1e12)
        self.colour_minimum.setDecimals(3)
        self.colour_minimum.setSuffix(" µT")
        self.colour_minimum.setEnabled(False)
        self.colour_maximum_enabled = QCheckBox("Upper limit")
        self.colour_maximum = QDoubleSpinBox()
        self.colour_maximum.setRange(0.0, 1e12)
        self.colour_maximum.setDecimals(3)
        self.colour_maximum.setSuffix(" µT")
        self.colour_maximum.setValue(1000.0)
        self.colour_maximum.setEnabled(False)
        for editor in (
            self.volume_opacity_lower,
            self.volume_opacity_upper,
            self.volume_opacity_lower_level,
            self.volume_opacity_upper_level,
            self.colour_minimum,
            self.colour_maximum,
        ):
            # QAbstractSpinBox sizes itself for the widest representable text
            # (not the current value). With the 1e12 colour-limit range that
            # can make a numeric editor demand ~30 characters and squeeze its
            # label off the narrow control panel. Let the grid own horizontal
            # allocation instead; the editor remains fully usable/editable.
            editor.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
            )
        display_layout = QGridLayout()
        # Keep the main labels/editors roomy, but treat the small "at level"
        # text as a compact suffix between the two numeric editors.  Giving that
        # column stretch made it consume a quarter of this narrow panel.
        display_layout.setColumnStretch(0, 1)
        display_layout.setColumnStretch(1, 1)
        display_layout.setColumnStretch(2, 0)
        display_layout.setColumnStretch(3, 1)
        display_layout.addWidget(self.logarithmic, 0, 0, 1, 4)
        display_layout.addWidget(self.show_grid, 1, 0, 1, 4)
        display_layout.addWidget(self.intensity_opacity_enabled, 2, 0, 1, 4)
        self.volume_opacity_lower_label = QLabel("Lower opacity")
        self.volume_opacity_upper_label = QLabel("Upper opacity")
        self.volume_opacity_lower_level_label = QLabel("at level")
        self.volume_opacity_upper_level_label = QLabel("at level")
        for label in (
            self.volume_opacity_lower_level_label,
            self.volume_opacity_upper_level_label,
        ):
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        display_layout.addWidget(self.volume_opacity_lower_label, 3, 0)
        display_layout.addWidget(self.volume_opacity_lower, 3, 1)
        display_layout.addWidget(self.volume_opacity_lower_level_label, 3, 2)
        display_layout.addWidget(self.volume_opacity_lower_level, 3, 3)
        display_layout.addWidget(self.volume_opacity_upper_label, 4, 0)
        display_layout.addWidget(self.volume_opacity_upper, 4, 1)
        display_layout.addWidget(self.volume_opacity_upper_level_label, 4, 2)
        display_layout.addWidget(self.volume_opacity_upper_level, 4, 3)
        colour_title = QLabel("Colour scale")
        colour_title.setStyleSheet("font-weight: 600;")
        display_layout.addWidget(colour_title, 5, 0, 1, 4)
        self.colour_limits_hint = QLabel(
            "Leave limits unchecked for the automatic |B| range"
        )
        self.colour_limits_hint.setObjectName("hint")
        self.colour_limits_hint.setWordWrap(True)
        display_layout.addWidget(self.colour_limits_hint, 6, 0, 1, 4)
        display_layout.addWidget(self.colour_minimum_enabled, 7, 0, 1, 2)
        display_layout.addWidget(self.colour_minimum, 7, 2, 1, 2)
        display_layout.addWidget(self.colour_maximum_enabled, 8, 0, 1, 2)
        display_layout.addWidget(self.colour_maximum, 8, 2, 1, 2)
        self.display_options_section = CollapsibleSection(
            "Display Options", expanded=True
        )
        self.display_options_section.setContentLayout(display_layout)
        control_root.addWidget(self.display_options_section)

        self._init_camera_timeline_controls(view_mode="3d")
        control_root.addWidget(self.camera_section)
        control_root.addStretch(1)

        self.static_analysis_button = QPushButton("Static analysis")
        self.static_analysis_button.setObjectName("primaryButton")
        self.static_analysis_button.setToolTip(
            "Evaluate the current scene field at volume-weighted LPBA40 samples"
        )
        self.static_analysis_button.clicked.connect(self.calculate_static_analysis)
        self.animated_analysis_button = QPushButton("Animated analysis")
        self.animated_analysis_button.setToolTip(
            "Configure animated field playback and optional brain-region spatial RMS traces"
        )
        self.animated_analysis_button.clicked.connect(self.open_animated_analysis)
        self.reset_defaults_button = QPushButton("Reset defaults")
        self.reset_defaults_button.clicked.connect(self._reset_defaults)
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        # Compatibility aliases mirror the ordinary 3D workspace's public names.
        self.calculate_button = self.static_analysis_button
        self.waveform_button = self.animated_analysis_button

        action_panel = QWidget()
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(8, 6, 8, 8)
        action_layout.setSpacing(6)
        action_layout.addWidget(self.camera_timeline_edit_widget)
        action_layout.addWidget(self.static_analysis_button)
        action_layout.addWidget(self.animated_analysis_button)
        action_layout.addWidget(self.reset_defaults_button)
        action_layout.addWidget(self.exit_button)
        control_column_layout.addWidget(action_panel, 0)
        splitter.addWidget(control_column)

        self._control_wheel_filter = EditorWheelScrollFilter(control_scroll)
        for editor_type in (QDoubleSpinBox, QSpinBox, QComboBox):
            for editor in control_panel.findChildren(editor_type):
                editor.installEventFilter(self._control_wheel_filter)

        viewer_panel = QWidget()
        viewer_root = QVBoxLayout(viewer_panel)
        viewer_root.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_view_tab)
        self.plot = PlotView(
            include_3d_return_axes=True,
            synchronize_legend_visibility=True,
            image_filename="brain-view-selector",
        )
        self.plot.visibility_requested.connect(
            self._preview_object_visibility_requested
        )
        self.plot.selection_requested.connect(
            self._selector_object_selection_requested
        )
        self.plot.live_3d_view_changed.connect(self._selector_camera_changed)
        self._selector_tab_index = -1
        self.tabs.currentChanged.connect(self._render_tab_changed)
        viewer_root.addWidget(self.tabs, 1)
        viewer_actions = QHBoxLayout()
        viewer_actions.setContentsMargins(8, 0, 8, 6)
        viewer_actions.addStretch(1)
        self.export_video_button = QPushButton("Export video")
        self.export_video_button.setToolTip(
            "Export the selected rendered brain animation without recalculating its field"
        )
        self.export_video_button.clicked.connect(self._export_current_animation_video)
        self.export_video_button.setVisible(False)
        viewer_actions.addWidget(self.export_video_button)
        self.fullscreen_button = QPushButton("Fullscreen viewer")
        self.fullscreen_button.clicked.connect(self._toggle_current_plot_fullscreen)
        viewer_actions.addWidget(self.fullscreen_button)
        viewer_root.addLayout(viewer_actions)
        self.calculation_status = QStatusBar()
        self.calculation_status.setObjectName("brainViewStatusBar")
        self.calculation_status.setSizeGripEnabled(False)
        self.calculation_status.showMessage(
            f"Ready — {len(self.region_meshes.regions)} LPBA40 shells • "
            f"{self.analysis_samples.point_count:,} weighted analysis samples"
        )
        viewer_root.addWidget(self.calculation_status)
        splitter.addWidget(viewer_panel)

        region_panel = QWidget()
        region_panel.setMinimumWidth(500)
        region_root = QVBoxLayout(region_panel)
        region_root.setContentsMargins(8, 8, 8, 8)
        heading = QLabel("Brain areas")
        heading.setObjectName("dialogHeading")
        region_root.addWidget(heading)
        method_note = QLabel(
            "Rows summarize volume-weighted atlas samples. Uniformity = "
            "100 × [1 − (P95 − P5) / mean], clamped to 0–100%; higher is more uniform."
        )
        method_note.setObjectName("hint")
        method_note.setWordWrap(True)
        region_root.addWidget(method_note)
        self.region_sections_splitter = QSplitter(Qt.Orientation.Vertical)
        self.region_sections_splitter.setChildrenCollapsible(False)
        self.region_sections_splitter.setHandleWidth(6)
        filter_section = QWidget()
        filter_section_layout = QVBoxLayout(filter_section)
        filter_section_layout.setContentsMargins(0, 0, 0, 10)
        filter_section_layout.setSpacing(6)
        filter_heading_row = QHBoxLayout()
        filter_heading = QLabel("Filters")
        filter_heading.setObjectName("sectionHeading")
        filter_heading_row.addWidget(filter_heading)
        filter_heading_row.addStretch(1)
        filter_section_layout.addLayout(filter_heading_row)

        self.region_filter_table = QTableWidget(0, 5)
        self.region_filter_table.setObjectName("brainRegionFilterTable")
        self.region_filter_table.setHorizontalHeaderLabels(
            ["#", "Metric", "Condition", "Value", "Action"]
        )
        # Persistent editor widgets inside a QTableWidget participate in the
        # view's current-index/selection machinery even when we do not want
        # them to. Do not use Qt item-view selection for filter rules at all.
        # Brain View owns explicit rule selection instead: Add selects the new
        # rule and clicking the # cell selects a different rule.
        self._selected_region_filter_row = -1
        self.region_filter_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.region_filter_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.region_filter_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.region_filter_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        # This editor grid has no hover affordance. Pointer movement must not
        # change either explicit rule selection or editor focus.
        self.region_filter_table.setMouseTracking(False)
        self.region_filter_table.setAttribute(Qt.WidgetAttribute.WA_Hover, False)
        self.region_filter_table.viewport().setMouseTracking(False)
        self.region_filter_table.viewport().setAttribute(
            Qt.WidgetAttribute.WA_Hover, False
        )
        self.region_filter_table.setAlternatingRowColors(True)
        self.region_filter_table.verticalHeader().setVisible(False)
        filter_header = self.region_filter_table.horizontalHeader()
        filter_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        filter_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        filter_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        filter_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        filter_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.region_filter_table.setColumnWidth(0, 38)
        self._region_filter_wheel_filter = _EditorWheelBlockFilter(
            self._set_region_filter_selected_row, self.region_filter_table
        )
        self.region_filter_table.setMinimumHeight(78)
        filter_section_layout.addWidget(self.region_filter_table, 1)

        filter_actions = QHBoxLayout()
        self.region_filter_add_button = QPushButton("Add")
        self.region_filter_remove_button = QPushButton("Remove")
        self.region_filter_remove_button.setEnabled(False)
        self.region_filter_add_button.clicked.connect(self._add_region_filter)
        self.region_filter_remove_button.clicked.connect(self._remove_region_filter)
        self.region_filter_table.cellClicked.connect(self._region_filter_cell_clicked)
        filter_actions.addWidget(self.region_filter_add_button)
        filter_actions.addWidget(self.region_filter_remove_button)
        filter_actions.addStretch(1)
        filter_section_layout.addLayout(filter_actions)
        self.region_sections_splitter.addWidget(filter_section)

        table_section = QWidget()
        table_section_layout = QVBoxLayout(table_section)
        table_section_layout.setContentsMargins(0, 0, 0, 0)
        table_section_layout.setSpacing(6)
        tree_actions = QHBoxLayout()
        self.region_collapse_button = QPushButton("Collapse")
        self.region_expand_button = QPushButton("Expand")
        self.region_expand_major_button = QPushButton("Expand major")
        self.region_collapse_button.setToolTip(
            "Collapse the Brain Areas hierarchy to the eight major groups."
        )
        self.region_expand_button.setToolTip("Expand the complete Brain Areas hierarchy.")
        self.region_expand_major_button.setToolTip(
            "Expand each major Brain Area one level while leaving deeper structures collapsed."
        )
        tree_actions.addWidget(self.region_collapse_button)
        tree_actions.addWidget(self.region_expand_button)
        tree_actions.addWidget(self.region_expand_major_button)
        tree_actions.addStretch(1)
        self.region_table = _BrainRegionTreeWidget()
        self.region_table.setObjectName("brainRegionTable")
        self.region_table.setColumnCount(8)
        self._region_header_labels = (
            "Structure",
            "RMS |B|",
            "Mean |B|",
            "P95 |B|",
            "Peak |B|",
            "Uniformity",
            "Direction",
            "Volume",
        )
        self.region_table.setHeaderLabels(list(self._region_header_labels))
        header = self.region_table.header()
        header.setSectionsClickable(True)
        # Use one explicit text arrow for sorting. Some Linux Qt styles still
        # paint their native sort glyph even when setSortIndicatorShown(False),
        # which produced a second, smaller arrow beside ours. Suppress both native
        # arrow subcontrols for this header and keep only the larger label glyph.
        header.setObjectName("brainRegionHeader")
        # Give the horizontal header enough vertical room for the bold label
        # text and our explicit sort-arrow glyph. Some Linux Qt styles report a
        # compact header size hint that clips the tops/bottoms of these labels.
        header.setMinimumHeight(30)
        header.setSortIndicatorShown(False)
        header.setStyleSheet(
            "QHeaderView#brainRegionHeader::up-arrow, "
            "QHeaderView#brainRegionHeader::down-arrow { "
            "image: none; width: 0px; height: 0px; }"
        )
        header.sortIndicatorChanged.connect(self._update_region_sort_indicator)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(58)
        for column, width in enumerate((220, 92, 92, 92, 92, 104, 96, 92)):
            self.region_table.setColumnWidth(column, width)
        self.region_column_menu = QMenu(self.region_table)
        self.region_column_actions = []
        for column, label in enumerate(self._region_header_labels):
            action = self.region_column_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda checked, selected_column=column: self.region_table.setColumnHidden(
                    selected_column, not checked
                )
            )
            self.region_column_actions.append(action)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_region_column_menu)
        header_item = self.region_table.headerItem()
        header_item.setToolTip(1, "Spatial RMS of field magnitude in microtesla")
        header_item.setToolTip(2, "Volume-weighted mean field magnitude")
        header_item.setToolTip(3, "Volume-weighted 95th percentile field magnitude")
        header_item.setToolTip(4, "Maximum field magnitude among retained atlas analysis samples")
        header_item.setToolTip(5, "P5–P95 uniformity score; higher is more uniform")
        header_item.setToolTip(6, "Consistency of field-vector direction; higher is more aligned")
        header_item.setToolTip(7, "Atlas volume after the selected brain object's scale")
        self.region_table.setAlternatingRowColors(True)
        self.region_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.region_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.region_table.setSortingEnabled(True)
        self.region_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._update_region_sort_indicator(0, Qt.SortOrder.AscendingOrder)
        self._populate_region_table()
        self._selector_row_toggle_filter = _ToggleSelectedTreeRowFilter(self.region_table)
        self.region_table.viewport().installEventFilter(self._selector_row_toggle_filter)
        self.region_table.itemSelectionChanged.connect(self._region_selection_changed)
        self.region_collapse_button.clicked.connect(
            lambda: _collapse_brain_region_tree_to_major(self.region_table, self.region_hierarchy)
        )
        self.region_expand_button.clicked.connect(
            lambda: _expand_brain_region_tree(self.region_table)
        )
        self.region_expand_major_button.clicked.connect(
            lambda: _expand_brain_region_tree_major(self.region_table)
        )
        table_section_layout.addWidget(self.region_table, 1)
        table_section_layout.addLayout(tree_actions)
        self.region_selection_note = QLabel(
            "Select any lobe, structure, or hemisphere to highlight its LPBA40 shell."
        )
        self.region_selection_note.setObjectName("hint")
        self.region_selection_note.setWordWrap(True)
        table_section_layout.addWidget(self.region_selection_note)
        self.region_sections_splitter.addWidget(table_section)
        self.region_sections_splitter.setStretchFactor(0, 0)
        self.region_sections_splitter.setStretchFactor(1, 1)
        self.region_sections_splitter.setSizes([190, 520])
        region_root.addWidget(self.region_sections_splitter, 1)

        # The Selector owns its own Brain Areas panel. Rendered result tabs get
        # independent panels so metrics, filters, sorting, and selection stay
        # with the result that produced them.
        selector_page = QWidget()
        selector_page_layout = QHBoxLayout(selector_page)
        selector_page_layout.setContentsMargins(0, 0, 0, 0)
        self.selector_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.selector_splitter.setChildrenCollapsible(False)
        self.selector_splitter.addWidget(self.plot)
        self.selector_splitter.addWidget(region_panel)
        self.selector_splitter.setStretchFactor(0, 1)
        self.selector_splitter.setStretchFactor(1, 0)
        self.selector_splitter.setSizes([790, 560])
        selector_page_layout.addWidget(self.selector_splitter, 1)
        self._selector_page = selector_page
        self._selector_tab_index = self.tabs.addTab(selector_page, "Selector")
        tab_bar = self.tabs.tabBar()
        tab_bar.setTabButton(
            self._selector_tab_index, QTabBar.ButtonPosition.LeftSide, None
        )
        tab_bar.setTabButton(
            self._selector_tab_index, QTabBar.ButtonPosition.RightSide, None
        )

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1350])

        for checkbox in (
            self.logarithmic,
            self.intensity_opacity_enabled,
        ):
            checkbox.toggled.connect(self._display_options_changed)
        self.show_grid.toggled.connect(self._selector_grid_changed)
        for editor in (
            self.volume_opacity_lower,
            self.volume_opacity_upper,
            self.volume_opacity_lower_level,
            self.volume_opacity_upper_level,
            self.colour_minimum,
            self.colour_maximum,
        ):
            editor.valueChanged.connect(self._display_options_changed)
        self.colour_minimum_enabled.toggled.connect(
            self._colour_limit_availability_changed
        )
        self.colour_maximum_enabled.toggled.connect(
            self._colour_limit_availability_changed
        )
        self.intensity_opacity_enabled.toggled.connect(
            self._update_opacity_controls
        )
        self.volume_opacity_lower_level.valueChanged.connect(
            self._lower_opacity_level_changed
        )
        self.volume_opacity_upper_level.valueChanged.connect(
            self._upper_opacity_level_changed
        )
        self._update_opacity_controls()
        for spin in self.coordinates:
            spin.valueChanged.connect(self._map_volume_changed)
        self.span.valueChanged.connect(self._map_volume_changed)
        # Do not build the selector figure in the constructor. Window-level UI
        # persistence is restored on the first Show event, and those restored
        # values can change grid/scene/map-volume options. Rendering here would
        # therefore create the default full scene first and immediately mutate
        # or redraw it as the persisted Brain View controls are restored, causing
        # a visible flash. The first real figure is built in showEvent(), after
        # the persistent-state event filter has applied the current controls.
        self.finished.connect(lambda _code: self._cleanup_plots())
        capture_window_default_settings(self)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        # enable_standard_window_controls() installs the shared persistent-state
        # event filter on this dialog. Qt runs object event filters before this
        # handler, so by the time we get here the saved Brain View controls have
        # already been restored. Build the selector exactly once from that final
        # state rather than rendering constructor defaults and then correcting it.
        super().showEvent(event)
        if not self._selector_figure_initialized:
            self._render_selector(structural_update=True)

    def _show_region_column_menu(self, position) -> None:
        for column, action in enumerate(self.region_column_actions):
            action.blockSignals(True)
            action.setChecked(not self.region_table.isColumnHidden(column))
            action.blockSignals(False)
        header = self.region_table.header()
        self.region_column_menu.exec(header.mapToGlobal(position))

    # --- map volume ------------------------------------------------------
    def _map_volume_centre_mm(self) -> list[float]:
        return [float(spin.value()) for spin in self.coordinates]

    def _map_volume_span_mm(self) -> float:
        return float(self.span.value())

    def _map_volume_trace(self) -> dict[str, Any]:
        grid = field_volume_grid_spec(
            self._map_volume_centre_mm(),
            self._map_volume_span_mm(),
            enabled=True,
        )
        points: list[list[float | None]] = []
        for first, second in grid["box_segments_mm"]:
            points.extend([first, second, [None, None, None]])
        values = np.asarray(points, dtype=object)
        return {
            "type": "scatter3d",
            "mode": "lines",
            "x": values[:, 0].tolist(),
            "y": values[:, 1].tolist(),
            "z": values[:, 2].tolist(),
            "line": {"color": "rgba(71,85,105,0.70)", "width": 2},
            "hoverinfo": "skip",
            "name": "Map volume",
            "showlegend": False,
            "meta": {"field_workbench_brain_map_volume": True},
        }

    def _map_volume_changed(self, *_args) -> None:
        if not getattr(self, "_selector_figure_initialized", False):
            return
        index = self._map_volume_trace_index
        if index is None:
            return
        trace = self._map_volume_trace()
        self.plot.apply_incremental_update(
            trace_updates={
                index: {
                    "x": trace["x"],
                    "y": trace["y"],
                    "z": trace["z"],
                }
            }
        )

    def _camera_default(self) -> dict[str, Any]:
        return default_video_camera(
            {
                "centre_mm": self._map_volume_centre_mm(),
                "span_mm": self._map_volume_span_mm(),
            }
        )

    def _timeline_duration_s(self) -> float:
        duration_s, _old_multiplier = remembered_waveform_timing("brain")
        return max(1.0e-6, float(duration_s))

    # --- scene options ---------------------------------------------------
    def _init_scene_options(self) -> None:
        self.include_scene = QCheckBox("Show scene geometry")
        self.include_scene.setChecked(True)
        self.include_scene.setToolTip(
            "Show coils, magnets, sensors, and other scene geometry behind the LPBA40 region shells"
        )
        self._scene_selected_object_ids: set[str] = set()
        self._scene_known_object_ids: set[str] = set()
        self._scene_object_opacity_overrides: dict[str, float] = {}
        self._scene_selection_initialized = False
        self.scene_objects_menu = QMenu(self)
        self.scene_objects_menu.aboutToShow.connect(self._refresh_scene_objects_menu)
        self.scene_objects_button = QPushButton("Objects…")
        self.scene_objects_button.setMenu(self.scene_objects_menu)
        self.scene_objects_button.setToolTip(
            "Choose viewer-only scene visibility and opacity; the source scene is unchanged"
        )
        self.include_scene.toggled.connect(self.scene_objects_button.setEnabled)
        self.include_scene.toggled.connect(self._preview_scene_options_changed)
        self._refresh_scene_objects_menu()
        self._init_playback_coil_selector()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.include_scene)
        layout.addWidget(self.scene_objects_button)
        active_label = QLabel("Active coils")
        active_label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(active_label)
        layout.addWidget(self.playback_coils_button)
        self.scene_options_section = CollapsibleSection(
            "Scene options", expanded=False
        )
        self.scene_options_section.setContentLayout(layout)

    def _brain_view_scene_object_choices(self) -> list[dict[str, Any]]:
        """Return scene objects other than the selected SRI24 shell.

        Brain View reconstructs the visible brain from LPBA40 component shells;
        drawing the original monolithic SRI24 STL underneath would obscure the
        internal/selectable region surfaces.
        """

        return [
            choice
            for choice in self.adapter.field_volume_map_scene_objects()
            if str(choice.get("id", "")) != self.brain_object_id
        ]

    def _refresh_scene_objects_menu(self) -> None:
        choices = self._brain_view_scene_object_choices()
        current_ids = {
            str(choice.get("id", "")) for choice in choices if choice.get("id")
        }
        if not self._scene_selection_initialized:
            self._scene_selected_object_ids = {
                str(choice["id"])
                for choice in choices
                if choice.get("id") and bool(choice.get("visible", True))
            }
            self._scene_selection_initialized = True
        else:
            self._scene_selected_object_ids.intersection_update(current_ids)
        self._scene_known_object_ids = current_ids
        self._scene_object_opacity_overrides = {
            object_id: float(value)
            for object_id, value in self._scene_object_opacity_overrides.items()
            if object_id in current_ids
        }
        self.scene_objects_menu.clear()
        select_all = self.scene_objects_menu.addAction("Select all")
        select_all.triggered.connect(self._select_all_scene_objects)
        clear_all = self.scene_objects_menu.addAction("Clear all")
        clear_all.triggered.connect(self._clear_all_scene_objects)
        self.scene_objects_menu.addSeparator()
        if not choices:
            empty = self.scene_objects_menu.addAction("No scene objects")
            empty.setEnabled(False)
        else:
            header = QWidget(self.scene_objects_menu)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(8, 0, 8, 0)
            object_heading = QLabel("Object")
            object_heading.setStyleSheet("font-weight: 600;")
            opacity_heading = QLabel("Opacity")
            opacity_heading.setStyleSheet("font-weight: 600;")
            header_layout.addWidget(object_heading, 1)
            header_layout.addWidget(opacity_heading)
            header_action = QWidgetAction(self.scene_objects_menu)
            header_action.setDefaultWidget(header)
            self.scene_objects_menu.addAction(header_action)
        for choice in choices:
            object_id = str(choice["id"])
            label = str(choice.get("label") or object_id)
            if not bool(choice.get("visible", True)):
                label += " (hidden)"
            row = QWidget(self.scene_objects_menu)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 1, 8, 1)
            row_layout.setSpacing(8)
            selected = QCheckBox(label)
            selected.setChecked(object_id in self._scene_selected_object_ids)
            selected.toggled.connect(
                lambda checked, selected_id=object_id: self._set_scene_object_selected(
                    selected_id, checked
                )
            )
            opacity = QSpinBox()
            opacity.setRange(0, 100)
            opacity.setSuffix(" %")
            inherited = max(0.0, min(1.0, float(choice.get("opacity", 1.0))))
            opacity.setValue(
                round(
                    self._scene_object_opacity_overrides.get(
                        object_id, inherited
                    )
                    * 100.0
                )
            )
            opacity.setFixedWidth(78)
            opacity.valueChanged.connect(
                lambda value, selected_id=object_id: self._set_scene_object_opacity(
                    selected_id, value / 100.0
                )
            )
            row_layout.addWidget(selected, 1)
            row_layout.addWidget(opacity)
            row_action = QWidgetAction(self.scene_objects_menu)
            row_action.setDefaultWidget(row)
            self.scene_objects_menu.addAction(row_action)
        self._update_scene_objects_button_text()

    def _update_scene_objects_button_text(self) -> None:
        total = len(self._scene_known_object_ids)
        selected = len(
            self._scene_selected_object_ids & self._scene_known_object_ids
        )
        self.scene_objects_button.setText(
            f"Objects {selected}/{total}…" if total else "Objects 0…"
        )

    def _set_scene_object_selected(self, object_id: str, checked: bool) -> None:
        if checked:
            self._scene_selected_object_ids.add(str(object_id))
        else:
            self._scene_selected_object_ids.discard(str(object_id))
        self._update_scene_objects_button_text()
        self._preview_scene_options_changed()

    def _set_scene_object_opacity(self, object_id: str, opacity: float) -> None:
        self._scene_object_opacity_overrides[str(object_id)] = max(
            0.0, min(1.0, float(opacity))
        )
        self._preview_scene_options_changed()

    def _select_all_scene_objects(self, *_args) -> None:
        self._scene_selected_object_ids = set(self._scene_known_object_ids)
        self._refresh_scene_objects_menu()
        self._preview_scene_options_changed()

    def _clear_all_scene_objects(self, *_args) -> None:
        self._scene_selected_object_ids.clear()
        self._refresh_scene_objects_menu()
        self._preview_scene_options_changed()

    def selected_scene_object_ids(self) -> list[str]:
        self._refresh_scene_objects_menu()
        return sorted(
            self._scene_selected_object_ids & self._scene_known_object_ids
        )

    def scene_object_opacity_values(self) -> dict[str, float]:
        choices = {
            str(choice.get("id", "")): max(
                0.0, min(1.0, float(choice.get("opacity", 1.0)))
            )
            for choice in self._brain_view_scene_object_choices()
            if choice.get("id")
        }
        return {
            object_id: float(
                self._scene_object_opacity_overrides.get(object_id, inherited)
            )
            for object_id, inherited in sorted(choices.items())
        }

    def _preview_scene_options_changed(self, *_args) -> None:
        if not self._selector_figure_initialized:
            return
        self._update_selector_scene_objects()

    def _preview_object_visibility_requested(
        self, object_id: str, visible: bool
    ) -> None:
        if str(object_id) in self._scene_known_object_ids:
            self._set_scene_object_selected(str(object_id), bool(visible))

    def _preview_scene_base_figure(self) -> dict[str, Any]:
        if self._selector_scene_figure is not None:
            return self._selector_scene_figure
        choices = self._brain_view_scene_object_choices()
        enabled = self.include_scene.isChecked()
        selected = (
            self._scene_selected_object_ids & self._scene_known_object_ids
            if enabled
            else set()
        )
        self._selector_scene_figure = self.adapter.field_map_incremental_preview_scene(
            choices,
            selected,
            object_opacities=self.scene_object_opacity_values(),
        )
        self._mark_selector_context_noninteractive(self._selector_scene_figure)
        return self._selector_scene_figure

    @staticmethod
    def _mark_selector_context_noninteractive(figure: dict[str, Any]) -> None:
        """Keep ordinary scene models visible but out of Brain View picking.

        Plotly's GL3D picking pass normally includes every rendered mesh, so an
        enclosing bust/scalp surface can win the depth test before an LPBA40
        shell behind it.  ``PlotView`` understands this Workbench-only meta flag
        and removes marked traces from the WebGL pick buffer while leaving their
        normal render pass untouched.  ``hoverinfo=skip`` is retained as a safe
        public-API fallback and suppresses context-model hover labels even if a
        Plotly build changes its private GL3D object layout.
        """

        data = figure.get("data", []) if isinstance(figure, dict) else []
        for trace in data if isinstance(data, list) else []:
            if not isinstance(trace, dict):
                continue
            meta = trace.get("meta")
            if not isinstance(meta, dict) or not meta.get(
                "field_workbench_object_id"
            ):
                continue
            meta["field_workbench_ignore_picking"] = True
            trace["hoverinfo"] = "skip"
            trace.pop("hovertemplate", None)

    # --- region table ----------------------------------------------------
    def _populate_region_table(self) -> None:
        self.region_table.setSortingEnabled(False)
        self.region_table.clear()
        self._items_by_key.clear()

        def add_node(node: BrainRegionNode, parent: QTreeWidgetItem | None) -> None:
            item = _NumericTreeItem([node.name, "—", "—", "—", "—", "—", "—", "—"])
            item.setData(0, Qt.ItemDataRole.UserRole, node.name.casefold())
            item.setData(0, REGION_KEY_ROLE, node.key)
            item.setData(0, REGION_LABEL_IDS_ROLE, list(node.label_ids))
            side = (
                "left"
                if node.key.endswith(".left")
                else "right"
                if node.key.endswith(".right")
                else ""
            )
            item.setData(0, REGION_SIDE_ROLE, side)
            item.setToolTip(0, f"LPBA40 label ids: {', '.join(map(str, node.label_ids))}")
            self._items_by_key[node.key] = item
            if parent is None:
                self.region_table.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for child in node.children:
                add_node(child, item)

        add_node(self.region_hierarchy, None)
        root_item = self._items_by_key[self.region_hierarchy.key]
        root_item.setExpanded(True)
        for index in range(root_item.childCount()):
            root_item.child(index).setExpanded(True)
        self.region_table.setSortingEnabled(True)
        self._resort_region_table()
        self._apply_region_filters()

    @staticmethod
    def _metric_text(value: float, suffix: str) -> str:
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):,.4g} {suffix}".rstrip()

    def _update_region_metrics(self) -> None:
        self.region_table.setSortingEnabled(False)
        for key, metrics in self._region_metrics.items():
            item = self._items_by_key.get(key)
            if item is None:
                continue
            values = (
                metrics.rms_uT,
                metrics.mean_uT,
                metrics.p95_uT,
                metrics.peak_uT,
                metrics.uniformity_pct,
                metrics.directional_consistency_pct,
                metrics.volume_cm3,
            )
            suffixes = ("µT", "µT", "µT", "µT", "%", "%", "cm³")
            for column, (value, suffix) in enumerate(
                zip(values, suffixes, strict=True), start=1
            ):
                item.setText(column, self._metric_text(value, suffix))
                item.setData(column, Qt.ItemDataRole.UserRole, float(value))
            item.setToolTip(
                0,
                f"LPBA40 label ids: {', '.join(map(str, self._node_label_ids(key)))}"
                + f"\n{metrics.voxel_count:,} atlas voxels; "
                f"{metrics.sample_count:,} retained analysis samples",
            )
        self.region_table.setSortingEnabled(True)
        self._resort_region_table()
        self._apply_region_filters()

    def _update_region_sort_indicator(
        self, column: int, order: Qt.SortOrder
    ) -> None:
        """Show a sort arrow whose visual direction matches the displayed order."""
        header_item = self.region_table.headerItem()
        if header_item is None:
            return
        for index, label in enumerate(self._region_header_labels):
            header_item.setText(index, label)
        if 0 <= int(column) < len(self._region_header_labels):
            # Conventional sort semantics: up means ascending (low -> high /
            # A -> Z), down means descending (high -> low / Z -> A).
            arrow = "▲" if order == Qt.SortOrder.AscendingOrder else "▼"
            header_item.setText(int(column), f"{self._region_header_labels[int(column)]}  {arrow}")

    def _resort_region_table(self) -> None:
        header = self.region_table.header()
        column = header.sortIndicatorSection()
        if column < 0:
            column = 0
        self.region_table.sortItems(column, header.sortIndicatorOrder())

    @staticmethod
    def _configure_region_filter_value_editor(
        editor: QDoubleSpinBox, metric_column: int
    ) -> None:
        # Keep generous input precision/length while displaying only useful
        # trailing zeroes. Brain-filter values retain at least two decimal
        # places, but users can type substantially more precision directly.
        editor.setDecimals(8)
        if isinstance(editor, CompactDoubleSpinBox):
            editor.setMinimumDisplayDecimals(2)
        line_edit = editor.lineEdit()
        if line_edit is not None:
            line_edit.setMaxLength(48)
        if metric_column in (5, 6):
            editor.setRange(0.0, 100.0)
            editor.setSingleStep(5.0)
            editor.setSuffix(" %")
        elif metric_column == 7:
            editor.setRange(0.0, 1.0e9)
            editor.setSingleStep(1.0)
            editor.setSuffix(" cm³")
        else:
            editor.setRange(0.0, 1.0e12)
            editor.setSingleStep(10.0)
            editor.setSuffix(" µT")

    def _prepare_region_filter_editor(self, editor: QWidget) -> None:
        """Keep filter editors deliberate: explicit click focus, no hover/wheel."""
        row = self._region_filter_row_for_widget(editor)
        targets = (editor, *editor.findChildren(QWidget))
        for target in targets:
            target.installEventFilter(self._region_filter_wheel_filter)
            target.setProperty("brainFilterRow", row)
            target.setMouseTracking(False)
            target.setAttribute(Qt.WidgetAttribute.WA_Hover, False)

        # Only configure the public cell widget.  Do *not* rewrite focus policy
        # on implementation children such as QDoubleSpinBox's private QLineEdit
        # and step buttons.  Doing that made those internals independent focus
        # targets and defeated the table-level NoFocus policy.
        if isinstance(editor, (QComboBox, QDoubleSpinBox, QSpinBox, QPushButton)):
            editor.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        else:
            editor.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        editor.setProperty(
            "brainFilterSelected", row == self._selected_region_filter_row
        )

    @staticmethod
    def _replace_combo_items(
        combo: QComboBox, choices: tuple[tuple[str, str], ...], preferred: str = ""
    ) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            for label, data in choices:
                combo.addItem(label, data)
            index = combo.findData(preferred) if preferred else -1
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _configure_region_filter_row(self, row: int) -> None:
        metric = self.region_filter_table.cellWidget(row, 1)
        condition = self.region_filter_table.cellWidget(row, 2)
        if not isinstance(metric, QComboBox) or not isinstance(condition, QComboBox):
            return
        metric_key = metric.currentData()
        previous_operator = str(condition.currentData() or "")
        if metric_key == "side":
            self._replace_combo_items(condition, (("=", "="),), "=")
            value = self.region_filter_table.cellWidget(row, 3)
            if not isinstance(value, QComboBox) or not bool(
                value.property("brainSideFilterValue")
            ):
                side = QComboBox()
                side.setProperty("brainSideFilterValue", True)
                side.addItem("Left", "left")
                side.addItem("Right", "right")
                self.region_filter_table.setCellWidget(row, 3, side)
                self._prepare_region_filter_editor(side)
                side.currentIndexChanged.connect(
                    lambda _index, widget=side: self._region_filter_widget_changed(widget)
                )
            return

        metric_column = int(metric_key)
        self._replace_combo_items(
            condition, ((">", ">"), ("<", "<")), previous_operator
        )
        value = self.region_filter_table.cellWidget(row, 3)
        if not isinstance(value, QDoubleSpinBox):
            value = CompactDoubleSpinBox()
            value.setKeyboardTracking(False)
            self.region_filter_table.setCellWidget(row, 3, value)
            self._prepare_region_filter_editor(value)
            value.valueChanged.connect(
                lambda _value, widget=value: self._region_filter_widget_changed(widget)
            )
        current_value = value.value()
        self._configure_region_filter_value_editor(value, metric_column)
        value.setValue(current_value)

    def _region_filter_row_for_widget(self, widget: QWidget) -> int:
        for row in range(self.region_filter_table.rowCount()):
            for column in range(self.region_filter_table.columnCount()):
                if self.region_filter_table.cellWidget(row, column) is widget:
                    return row
        return -1

    def _refresh_region_filter_selection_visual(self) -> None:
        selected_row = self._selected_region_filter_row
        palette = self.region_filter_table.palette()
        for row in range(self.region_filter_table.rowCount()):
            selected = row == selected_row
            number_item = self.region_filter_table.item(row, 0)
            if number_item is not None:
                number_item.setData(
                    Qt.ItemDataRole.BackgroundRole,
                    palette.brush(QPalette.ColorRole.Highlight) if selected else None,
                )
                number_item.setData(
                    Qt.ItemDataRole.ForegroundRole,
                    palette.brush(QPalette.ColorRole.HighlightedText) if selected else None,
                )
            for column in (1, 2, 3, 4):
                editor = self.region_filter_table.cellWidget(row, column)
                if editor is None:
                    continue
                editor.setProperty("brainFilterSelected", selected)
                editor.style().unpolish(editor)
                editor.style().polish(editor)
                editor.update()

    def _set_region_filter_selected_row(self, row: int) -> None:
        self._selected_region_filter_row = (
            int(row) if 0 <= int(row) < self.region_filter_table.rowCount() else -1
        )
        # QTableWidget's own current index is intentionally never the source of
        # truth. Embedded combo/spin widgets can otherwise make the view move
        # its current cell as focus changes.
        self.region_filter_table.clearSelection()
        self.region_filter_table.setCurrentCell(-1, -1)
        self._refresh_region_filter_selection_visual()
        self._region_filter_selection_changed()

    def _region_filter_cell_clicked(self, row: int, column: int) -> None:
        if int(column) == 0:
            self._set_region_filter_selected_row(int(row))

    def _renumber_region_filter_rows(self) -> None:
        for row in range(self.region_filter_table.rowCount()):
            item = self.region_filter_table.item(row, 0)
            if item is None:
                item = QTableWidgetItem()
                # The # cell is a click target, not a Qt-selected item.
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip("Click here to select this filter")
                self.region_filter_table.setItem(row, 0, item)
            item.setText(str(row + 1))
            for column in (1, 2, 3, 4):
                editor = self.region_filter_table.cellWidget(row, column)
                if editor is None:
                    continue
                for target in (editor, *editor.findChildren(QWidget)):
                    target.setProperty("brainFilterRow", row)
        self._refresh_region_filter_selection_visual()

    def _add_region_filter(self, *_args) -> None:
        row = self.region_filter_table.rowCount()
        self.region_filter_table.insertRow(row)

        metric = QComboBox()
        for label, column in (
            ("RMS |B|", 1),
            ("Mean |B|", 2),
            ("P95 |B|", 3),
            ("Peak |B|", 4),
            ("Uniformity", 5),
            ("Direction", 6),
            ("Volume", 7),
            ("Side", "side"),
        ):
            metric.addItem(label, column)
        condition = QComboBox()
        condition.addItem(">", ">")
        condition.addItem("<", "<")
        value = CompactDoubleSpinBox()
        value.setKeyboardTracking(False)
        self._configure_region_filter_value_editor(value, 1)
        action_cell = QWidget()
        action_layout = QHBoxLayout(action_cell)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        action = QComboBox()
        action.setObjectName("brainFilterActionCombo")
        action.addItem("Filter", "filter")
        action.addItem("Highlight", "highlight")
        colour = QPushButton()
        colour.setObjectName("brainFilterColourButton")
        colour.setFixedWidth(26)
        _set_brain_filter_colour_button(colour, BRAIN_FILTER_DEFAULT_HIGHLIGHT)
        colour.setVisible(False)
        action_layout.addWidget(action, 1)
        action_layout.addWidget(colour, 0)

        self.region_filter_table.setCellWidget(row, 1, metric)
        self.region_filter_table.setCellWidget(row, 2, condition)
        self.region_filter_table.setCellWidget(row, 3, value)
        self.region_filter_table.setCellWidget(row, 4, action_cell)
        for editor in (metric, condition, value, action_cell):
            self._prepare_region_filter_editor(editor)
        self._renumber_region_filter_rows()
        metric.currentIndexChanged.connect(
            lambda _index, widget=metric: self._region_filter_metric_changed(widget)
        )
        condition.currentIndexChanged.connect(
            lambda _index, widget=condition: self._region_filter_widget_changed(widget)
        )
        value.valueChanged.connect(
            lambda _value, widget=value: self._region_filter_widget_changed(widget)
        )
        action.currentIndexChanged.connect(
            lambda _index, widget=action: self._region_filter_action_changed(widget)
        )
        colour.clicked.connect(
            lambda _checked=False, button=colour: self._pick_region_filter_colour(button)
        )

        # Adding a rule selects it. Afterwards, deliberate clicks on either the
        # # cell or one of that row's child editors may select another rule.
        self._set_region_filter_selected_row(row)
        self._apply_region_filters()

    def _region_filter_action_changed(self, action: QComboBox) -> None:
        row_value = action.property("brainFilterRow")
        row = int(row_value) if isinstance(row_value, int) else -1
        if row < 0:
            row = self._region_filter_row_for_widget(action.parentWidget())
        if row < 0:
            return
        cell = self.region_filter_table.cellWidget(row, 4)
        current_action, colour = _brain_filter_action_widgets(cell)
        if current_action is None or colour is None:
            return
        colour.setVisible(str(current_action.currentData()) == "highlight")
        self.region_filter_table.setCurrentCell(-1, -1)
        self._refresh_region_filter_selection_visual()
        self._apply_region_filters()

    def _pick_region_filter_colour(self, button: QPushButton) -> None:
        current = QColor(
            str(button.property("brainFilterHighlightColour") or BRAIN_FILTER_DEFAULT_HIGHLIGHT)
        )
        chosen = QColorDialog.getColor(current, self, "Choose Brain Areas highlight colour")
        if not chosen.isValid():
            return
        _set_brain_filter_colour_button(button, chosen.name(QColor.NameFormat.HexRgb))
        self._apply_region_filters()

    def _remove_region_filter(self, *_args) -> None:
        row = self._selected_region_filter_row
        if row < 0 or row >= self.region_filter_table.rowCount():
            return
        self.region_filter_table.removeRow(row)
        self._selected_region_filter_row = -1
        self._renumber_region_filter_rows()
        self._set_region_filter_selected_row(-1)
        self._apply_region_filters()

    def _region_filter_selection_changed(self) -> None:
        self.region_filter_remove_button.setEnabled(
            0 <= self._selected_region_filter_row < self.region_filter_table.rowCount()
        )

    def _region_filter_metric_changed(self, metric: QComboBox) -> None:
        row = self._region_filter_row_for_widget(metric)
        if row < 0:
            return
        self._configure_region_filter_row(row)
        self.region_filter_table.setCurrentCell(-1, -1)
        self._refresh_region_filter_selection_visual()
        self._apply_region_filters()

    def _region_filter_widget_changed(self, widget: QWidget) -> None:
        # Child-widget mouse presses explicitly select their rule via the event
        # filter. Value changes themselves do not move selection. Discard any
        # transient QTableWidget current index so it never becomes state.
        self.region_filter_table.setCurrentCell(-1, -1)
        self._apply_region_filters()

    def _region_filter_rules(self) -> list[tuple[int | str, str, float | str, str, str]]:
        rules: list[tuple[int | str, str, float | str, str, str]] = []
        for row in range(self.region_filter_table.rowCount()):
            metric = self.region_filter_table.cellWidget(row, 1)
            condition = self.region_filter_table.cellWidget(row, 2)
            value = self.region_filter_table.cellWidget(row, 3)
            action, colour_button = _brain_filter_action_widgets(
                self.region_filter_table.cellWidget(row, 4)
            )
            if (
                not isinstance(metric, QComboBox)
                or not isinstance(condition, QComboBox)
                or action is None
            ):
                continue
            metric_key = metric.currentData()
            action_key = str(action.currentData() or "filter")
            colour = str(
                colour_button.property("brainFilterHighlightColour")
                if colour_button is not None
                else BRAIN_FILTER_DEFAULT_HIGHLIGHT
            )
            if metric_key == "side":
                if not isinstance(value, QComboBox):
                    continue
                rules.append(("side", "=", str(value.currentData()), action_key, colour))
                continue
            if not isinstance(value, QDoubleSpinBox):
                continue
            rules.append(
                (
                    int(metric_key),
                    str(condition.currentData()),
                    float(value.value()),
                    action_key,
                    colour,
                )
            )
        return rules

    @staticmethod
    def _region_item_matches_rules(
        item: QTreeWidgetItem,
        rules: list[tuple[int | str, str, float | str, str, str]],
    ) -> bool:
        return all(_brain_item_matches_filter_rule(item, rule) for rule in rules)

    def _apply_region_filters(self) -> None:
        rules = self._region_filter_rules()
        filter_rules = [rule for rule in rules if rule[3] == "filter"]
        highlight_rules = [rule for rule in rules if rule[3] == "highlight"]
        root = self._items_by_key.get(self.region_hierarchy.key)
        if root is None:
            return

        # Before a static solve exists, numeric filter rules retain the previous
        # Brain View behaviour: they do not empty the selector table merely
        # because there are no numeric metrics yet. Side-only filters can still
        # be applied immediately.
        effective_filter_rules = filter_rules
        effective_highlight_rules = highlight_rules
        if not self._region_metrics:
            effective_filter_rules = [rule for rule in filter_rules if rule[0] == "side"]
            effective_highlight_rules = [rule for rule in highlight_rules if rule[0] == "side"]

        def apply_item(item: QTreeWidgetItem) -> bool:
            child_visible = False
            for child_index in range(item.childCount()):
                child_visible = apply_item(item.child(child_index)) or child_visible
            excluded = bool(effective_filter_rules) and self._region_item_matches_rules(
                item, effective_filter_rules
            )
            # Keep an excluded parent as a structural ancestor when one of its
            # descendants survives the filter. Leaf/terminal matches disappear.
            visible = not excluded or child_visible
            item.setHidden(not visible)
            return visible

        apply_item(root)

        # Highlight rules affect appearance only. A row that matches more than
        # one Highlight rule receives the colour from the last matching rule.
        for item in self._items_by_key.values():
            _clear_brain_item_filter_highlight(item, self.region_table.columnCount())
            selected_colour = ""
            for rule in effective_highlight_rules:
                if _brain_item_matches_filter_rule(item, rule):
                    selected_colour = rule[4]
            if selected_colour:
                _apply_brain_item_filter_highlight(
                    item, selected_colour, self.region_table.columnCount()
                )

        current = self.region_table.currentItem()
        if current is not None and current.isHidden():
            self.region_table.clearSelection()

    @staticmethod
    def _trace_region_tree_payload(node: BrainRegionNode) -> dict[str, Any]:
        """Serialize the LPBA40 hierarchy for waveform trace selection."""

        return {
            "key": str(node.key),
            "name": str(node.name),
            "label_ids": [int(value) for value in node.label_ids],
            "children": [
                BrainViewDialog._trace_region_tree_payload(child)
                for child in node.children
            ],
        }

    def _selected_region(self) -> tuple[str, str, tuple[int, ...]]:
        selected = self.region_table.selectedItems()
        item = selected[0] if selected else None
        if item is None:
            # No selected row means no visual highlight. Whole brain remains a
            # normal selectable row rather than serving as an implicit fallback.
            return "", "", ()
        key = str(item.data(0, REGION_KEY_ROLE) or "")
        values = item.data(0, REGION_LABEL_IDS_ROLE)
        label_ids = tuple(int(value) for value in (values or []))
        return key, item.text(0), label_ids

    def _region_selection_changed(self) -> None:
        if not self.region_table.selectedItems():
            self.region_selection_note.setText("No LPBA40 region highlighted.")
            self._render_selector()
            return
        _key, name, label_ids = self._selected_region()
        self.region_selection_note.setText(
            f"Highlighted: {name} • {len(label_ids)} LPBA40 shell"
            f"{'s' if len(label_ids) != 1 else ''}."
        )
        self._render_selector()

    # --- selector rendering ---------------------------------------------
    def _display_options_changed(self, *_args) -> None:
        # These controls describe the next static/animated result. The selector
        # itself contains categorical LPBA40 shells, so changing field colour or
        # opacity settings requires no Plotly work at all.
        return

    def _selector_grid_changed(self, *_args) -> None:
        if not self._selector_figure_initialized:
            return
        self.plot.apply_incremental_update(
            layout_updates=self._selector_grid_layout_updates(
                self.show_grid.isChecked()
            )
        )

    def _colour_limit_availability_changed(self, *_args) -> None:
        self.colour_minimum.setEnabled(self.colour_minimum_enabled.isChecked())
        self.colour_maximum.setEnabled(self.colour_maximum_enabled.isChecked())

    def _update_opacity_controls(self, *_args) -> None:
        enabled = self.intensity_opacity_enabled.isChecked()
        for widget in (
            self.volume_opacity_lower_label,
            self.volume_opacity_upper_label,
            self.volume_opacity_lower_level_label,
            self.volume_opacity_upper_level_label,
            self.volume_opacity_lower,
            self.volume_opacity_upper,
            self.volume_opacity_lower_level,
            self.volume_opacity_upper_level,
        ):
            widget.setEnabled(enabled)

    def _lower_opacity_level_changed(self, value: float) -> None:
        if float(value) > self.volume_opacity_upper_level.value():
            self.volume_opacity_upper_level.setValue(float(value))

    def _upper_opacity_level_changed(self, value: float) -> None:
        if float(value) < self.volume_opacity_lower_level.value():
            self.volume_opacity_lower_level.setValue(float(value))

    def _resolved_display_values(
        self, values_uT: np.ndarray
    ) -> tuple[np.ndarray, float, float, str]:
        values = np.asarray(values_uT, dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise BrainAnalysisError("Brain field display contains no finite values.")
        manual_min = (
            self.colour_minimum.value()
            if self.colour_minimum_enabled.isChecked()
            else None
        )
        manual_max = (
            self.colour_maximum.value()
            if self.colour_maximum_enabled.isChecked()
            else None
        )
        if self.logarithmic.isChecked():
            positive = finite[finite > 0.0]
            floor = (
                max(float(np.min(positive)) * 1e-6, 1e-300)
                if len(positive)
                else 1e-300
            )
            displayed = np.log10(np.maximum(values, floor))
            automatic_min = float(np.min(displayed[np.isfinite(displayed)]))
            automatic_max = float(np.max(displayed[np.isfinite(displayed)]))
            if manual_min is not None and manual_min <= 0.0:
                raise BrainAnalysisError(
                    "A logarithmic lower colour limit must be greater than zero."
                )
            if manual_max is not None and manual_max <= 0.0:
                raise BrainAnalysisError(
                    "A logarithmic upper colour limit must be greater than zero."
                )
            lower = automatic_min if manual_min is None else math.log10(manual_min)
            upper = automatic_max if manual_max is None else math.log10(manual_max)
            title = "log₁₀ |B| (µT)"
        else:
            displayed = values
            automatic_min = float(np.min(finite))
            automatic_max = float(np.max(finite))
            lower = automatic_min if manual_min is None else float(manual_min)
            upper = automatic_max if manual_max is None else float(manual_max)
            title = "|B| (µT)"
        if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-15):
            pad = max(abs(lower) * 0.01, 1e-12)
            lower -= pad
            upper += pad
        if not lower < upper:
            raise BrainAnalysisError(
                "The selected brain colour limits do not form a valid range."
            )
        return displayed, float(lower), float(upper), title

    def _field_point_colours(
        self, displayed: np.ndarray, lower: float, upper: float
    ) -> list[str]:
        fractions = np.clip((displayed - lower) / (upper - lower), 0.0, 1.0)
        opacity_enabled = self.intensity_opacity_enabled.isChecked()
        opacity_low = self.volume_opacity_lower.value() / 100.0
        opacity_high = self.volume_opacity_upper.value() / 100.0
        level_low = self.volume_opacity_lower_level.value() / 100.0
        level_high = self.volume_opacity_upper_level.value() / 100.0
        colours: list[str] = []
        for fraction in fractions:
            value = float(fraction)
            red, green, blue = _field_colour_at_fraction(
                FIELD_HEAT_COLOURSCALE, value
            )
            if not opacity_enabled:
                alpha = 1.0
            elif value <= level_low:
                alpha = opacity_low
            elif value >= level_high or level_high <= level_low:
                alpha = opacity_high
            else:
                alpha = opacity_low + (opacity_high - opacity_low) * (
                    (value - level_low) / (level_high - level_low)
                )
            colours.append(f"rgba({red},{green},{blue},{alpha:.4f})")
        return colours

    def _selector_object_selection_requested(
        self, object_id: str, _additive: bool
    ) -> None:
        """Mirror a clicked LPBA40 shell into the Brain Areas tree."""

        prefix = "brain-region:"
        value = str(object_id)
        if not value.startswith(prefix):
            return
        try:
            label_id = int(value[len(prefix) :])
        except ValueError:
            return
        candidates = [
            item
            for item in self._items_by_key.values()
            if self._node_label_ids(str(item.data(0, REGION_KEY_ROLE) or ""))
            == (label_id,)
        ]
        if not candidates:
            return
        # Bilateral structures have a one-label Left/Right leaf; unpaired
        # cerebellum/brainstem rows are already leaves. Prefer a true leaf in
        # either case so a click selects the most specific table row.
        item = next((candidate for candidate in candidates if candidate.childCount() == 0), candidates[0])
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self.region_table.setCurrentItem(item)
        item.setSelected(True)
        self.region_table.scrollToItem(
            item, QAbstractItemView.ScrollHint.PositionAtCenter
        )

    def _region_shell_trace(
        self, label_id: int, *, selected: bool
    ) -> dict[str, Any]:
        mesh = self.region_meshes.regions[int(label_id)]
        vertices = mesh.vertices_mm
        triangles = mesh.triangles
        division_key = self._division_key_by_label.get(int(label_id), "")
        colour = (
            LPBA40_SELECTED_REGION_COLOUR
            if selected
            else LPBA40_DIVISION_COLOURS.get(division_key, "#9ca3af")
        )
        opacity = (
            LPBA40_SELECTED_REGION_OPACITY if selected else LPBA40_REGION_OPACITY
        )
        definition = self.parcellation.labels[int(label_id)]
        display_name = definition.name.replace("_", " ")
        return {
            "type": "mesh3d",
            "x": vertices[:, 0].tolist(),
            "y": vertices[:, 1].tolist(),
            "z": vertices[:, 2].tolist(),
            "i": triangles[:, 0].tolist(),
            "j": triangles[:, 1].tolist(),
            "k": triangles[:, 2].tolist(),
            "color": colour,
            "opacity": opacity,
            "flatshading": False,
            "lighting": {
                "ambient": 0.58,
                "diffuse": 0.72,
                "specular": 0.16,
                "roughness": 0.78,
                "fresnel": 0.08,
            },
            "name": display_name,
            "hovertemplate": display_name + "<extra></extra>",
            "showlegend": False,
            "uid": f"brain-view-region-{int(label_id)}",
            "meta": {
                "field_workbench_object_id": f"brain-region:{int(label_id)}",
                "brain_region_label_id": int(label_id),
            },
        }

    def _region_shell_style(self, label_id: int, *, selected: bool) -> dict[str, Any]:
        division_key = self._division_key_by_label.get(int(label_id), "")
        return {
            "color": (
                LPBA40_SELECTED_REGION_COLOUR
                if selected
                else LPBA40_DIVISION_COLOURS.get(division_key, "#9ca3af")
            ),
            "opacity": (
                LPBA40_SELECTED_REGION_OPACITY
                if selected
                else LPBA40_REGION_OPACITY
            ),
        }

    def _selector_title(
        self, selected_name: str, selected_label_ids: set[int]
    ) -> str:
        title = (
            "LPBA40 brain-region field analysis"
            if self._field_vectors_t is not None
            else "LPBA40 atlas selector"
        )
        if selected_label_ids:
            title += f" — {selected_name}"
        return title

    def _capture_selector_grid_visible_layout(
        self, figure: dict[str, Any]
    ) -> None:
        scene = figure.get("layout", {}).get("scene", {})
        if not isinstance(scene, dict):
            scene = {}
        properties = (
            "title",
            "visible",
            "showgrid",
            "zeroline",
            "showbackground",
            "showline",
            "showticklabels",
            "ticks",
            "showspikes",
            "backgroundcolor",
            "gridcolor",
            "zerolinecolor",
            "showaxeslabels",
            "linecolor",
            "tickfont",
        )
        restored: dict[str, Any] = {}
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = scene.get(axis_name)
            axis = axis if isinstance(axis, dict) else {}
            for property_name in properties:
                restored[f"scene.{axis_name}.{property_name}"] = copy.deepcopy(
                    axis.get(property_name)
                )
        restored["scene.annotations"] = copy.deepcopy(scene.get("annotations"))
        self._selector_grid_visible_layout = restored

    def _selector_grid_layout_updates(self, show_grid: bool) -> dict[str, Any]:
        if show_grid:
            return dict(self._selector_grid_visible_layout)
        hidden: dict[str, Any] = {"scene.annotations": []}
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            prefix = f"scene.{axis_name}."
            hidden.update(
                {
                    prefix + "title": {
                        "text": "",
                        "font": {"color": "rgba(0,0,0,0)"},
                    },
                    prefix + "visible": False,
                    prefix + "showgrid": False,
                    prefix + "zeroline": False,
                    prefix + "showbackground": False,
                    prefix + "showline": False,
                    prefix + "showticklabels": False,
                    prefix + "ticks": "",
                    prefix + "showspikes": False,
                    prefix + "backgroundcolor": "rgba(0,0,0,0)",
                    prefix + "gridcolor": "rgba(0,0,0,0)",
                    prefix + "zerolinecolor": "rgba(0,0,0,0)",
                    prefix + "showaxeslabels": False,
                    prefix + "linecolor": "rgba(0,0,0,0)",
                    prefix + "tickfont": {"color": "rgba(0,0,0,0)"},
                }
            )
        return hidden

    def _capture_selector_trace_indices(self, figure: dict[str, Any]) -> None:
        self._region_trace_index.clear()
        self._scene_trace_indices.clear()
        self._scene_legend_trace_indices.clear()
        self._scene_trace_opacity_scale.clear()
        self._map_volume_trace_index = None
        for index, trace in enumerate(figure.get("data", [])):
            if not isinstance(trace, dict):
                continue
            metadata = trace.get("meta")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("field_workbench_brain_map_volume"):
                self._map_volume_trace_index = index
                continue
            label_id = metadata.get("brain_region_label_id")
            if isinstance(label_id, int):
                self._region_trace_index[int(label_id)] = index
                continue
            object_id = str(metadata.get("field_workbench_object_id", ""))
            if object_id not in self._scene_known_object_ids:
                continue
            if metadata.get("field_workbench_map_preview_legend"):
                self._scene_legend_trace_indices.setdefault(object_id, []).append(index)
                continue
            self._scene_trace_indices.setdefault(object_id, []).append(index)
            try:
                scale = float(
                    metadata.get("field_workbench_preview_opacity_scale", 1.0)
                )
            except (TypeError, ValueError):
                scale = 1.0
            self._scene_trace_opacity_scale[index] = (
                scale if math.isfinite(scale) and scale >= 0.0 else 1.0
            )

    def _update_selector_scene_objects(self) -> None:
        selected = (
            self._scene_selected_object_ids & self._scene_known_object_ids
            if self.include_scene.isChecked()
            else set()
        )
        opacity_values = self.scene_object_opacity_values()
        trace_updates: dict[int, dict[str, object]] = {}
        for object_id, indices in self._scene_trace_indices.items():
            requested_opacity = float(opacity_values.get(object_id, 1.0))
            visible = object_id in selected
            for index in indices:
                scale = self._scene_trace_opacity_scale.get(index, 1.0)
                trace_updates[index] = {
                    "visible": visible,
                    "opacity": max(0.0, min(1.0, requested_opacity * scale)),
                }
        for object_id, indices in self._scene_legend_trace_indices.items():
            visible: bool | str = True if object_id in selected else "legendonly"
            for index in indices:
                trace_updates[index] = {"visible": visible}
        self.plot.apply_incremental_update(trace_updates=trace_updates)

    def _update_selector_highlight(self) -> None:
        _key, selected_name, selected_ids = self._selected_region()
        selected_set = {int(value) for value in selected_ids}
        changed_labels = self._last_selected_region_labels ^ selected_set
        trace_updates = {
            self._region_trace_index[label_id]: self._region_shell_style(
                label_id, selected=label_id in selected_set
            )
            for label_id in changed_labels
            if label_id in self._region_trace_index
        }
        self._last_selected_region_labels = set(selected_set)
        self.plot.apply_incremental_update(
            trace_updates=trace_updates,
            layout_updates={
                "title.text": self._selector_title(selected_name, selected_set)
            },
        )

    def _render_selector(self, *, structural_update: bool = False) -> None:
        if not hasattr(self, "plot"):
            return
        try:
            if self._selector_figure_initialized and not structural_update:
                self._update_selector_highlight()
                return
            if self._selector_figure_initialized:
                self._selector_scene_figure = None
            figure = copy.deepcopy(self._preview_scene_base_figure())
            data = figure.setdefault("data", [])
            data.append(self._map_volume_trace())
            _key, selected_name, selected_ids = self._selected_region()
            selected_set = set(int(value) for value in selected_ids)
            for label_id in sorted(self.region_meshes.regions):
                data.append(
                    self._region_shell_trace(
                        label_id, selected=label_id in selected_set
                    )
                )

            layout = figure.setdefault("layout", {})
            layout["title"] = {
                "text": self._selector_title(selected_name, selected_set),
                "x": 0.5,
            }
            layout["uirevision"] = "field-workbench-brain-view"
            layout["margin"] = {"l": 8, "r": 24, "t": 48, "b": 8}
            scene = layout.setdefault("scene", {})
            scene["aspectmode"] = "data"
            self._capture_selector_grid_visible_layout(figure)
            figure = FieldVolumeMapDialog._apply_selector_grid_visibility(
                figure, show_grid=self.show_grid.isChecked()
            )
            self._capture_selector_trace_indices(figure)
            self._last_selected_region_labels = set(selected_set)
            self._selector_figure_initialized = True
            self.plot.request_clean_rebuild_on_next_figure()
            self.plot.set_figure(figure, coalesce=True)
        except (BrainAnalysisError, BrainMeshError, StudioOperationError, ValueError) as error:
            self.calculation_status.showMessage(str(error))

    # --- analysis and waveform ------------------------------------------
    def calculate_static_analysis(self) -> None:
        if not self.static_analysis_button.isEnabled():
            return
        self._request_action_camera(self._calculate_static_webgl)

    def _static_map_settings(self) -> dict[str, Any]:
        """Snapshot Brain View display settings for one fixed-current map.

        Brain View now renders the calculated field with the shared 3D WebGL
        field renderer rather than showing only atlas points. The authoritative
        Brain Areas statistics still come from the full volume-weighted LPBA40
        analysis sample set, solved alongside the displayed field in one worker
        pass so no second scene solve is required.
        """

        settings = self._waveform_map_settings()
        # Trace selection belongs only to Animated analysis / Export video.
        # Static analysis needs the full LPBA40 analysis samples plus the
        # smaller coloured overlay sample set, but no waveform trace controls.
        for key in (
            "trace_region_tree",
            "trace_regions",
            "scroll_observation_traces",
            "trace_points_mm",
            "trace_point_label_ids",
            "trace_point_weights_mm3",
            "trace_source_voxel_counts",
            "trace_voxel_volume_mm3",
        ):
            settings.pop(key, None)
        settings.update(
            {
                "analysis_points_mm": self.analysis_samples.points_mm.tolist(),
                "_return_field_vectors": True,
            }
        )
        return settings

    def _start_brain_render_job(
        self,
        *,
        render_kind: str,
        request: dict[str, Any],
        job_label: str,
        tooltip: str,
    ) -> None:
        """Start a render while freezing the Selector's Brain Areas filters."""

        selector_filter_rules = copy.deepcopy(self._region_filter_rules())
        previous_serial = int(self._render_job_serial)
        self._start_render_job(
            render_kind=render_kind,
            request=request,
            job_label=job_label,
            tooltip=tooltip,
        )
        if int(self._render_job_serial) == previous_serial:
            return
        job = self._render_jobs.get(int(self._render_job_serial))
        if isinstance(job, dict):
            job["initial_brain_region_filter_rules"] = selector_filter_rules

    def _calculate_static_webgl(self, initial_camera: dict[str, Any]) -> None:
        self._map_tab_counter += 1
        job_label = f"Analysis {self._map_tab_counter}"
        settings = self._static_map_settings()
        self._start_brain_render_job(
            render_kind="brain_static",
            request={
                "map_settings": settings,
                "active_coil_ids": self.selected_active_coil_ids(),
                "initial_camera": copy.deepcopy(initial_camera),
            },
            job_label=job_label,
            tooltip=(
                f"LPBA40 static field • {self.analysis_samples.point_count:,} "
                "volume-weighted samples • WebGL"
            ),
        )

    def _render_job_result_loaded(
        self,
        _job_id: int,
        job: dict[str, Any],
        isolated_result: dict[str, Any],
    ) -> None:
        """Attach Brain Areas data to the result that produced it."""

        render_kind = str(job.get("render_kind", ""))
        if render_kind == "brain_static":
            raw_vectors = isolated_result.pop("field_vectors_t", None)
            if raw_vectors is None:
                raise StudioOperationError(
                    "The brain static render returned no field vectors for regional statistics."
                )
            vectors = np.asarray(raw_vectors, dtype=float).reshape(-1, 3)
            if vectors.shape != (self.analysis_samples.point_count, 3):
                raise StudioOperationError(
                    "The brain static render returned an unexpected vector shape."
                )
            job["brain_static_metrics"] = lpba40_region_metrics(
                vectors, self.analysis_samples, self.region_hierarchy
            )
            return

        if render_kind == "waveform":
            payload = isolated_result.get("payload")
            if isinstance(payload, dict):
                frame_metrics = payload.pop("brain_region_frame_metrics", None)
                if isinstance(frame_metrics, dict):
                    job["brain_frame_metrics"] = frame_metrics

    def _wrap_render_result_view(
        self,
        viewer: WaveformGLView,
        job: dict[str, Any],
        _isolated_result: dict[str, Any],
    ) -> QWidget:
        """Give every rendered Brain View tab its own Brain Areas panel."""

        panel = _BrainResultRegionPanel(self.region_hierarchy)
        static_metrics = job.get("brain_static_metrics")
        if isinstance(static_metrics, dict):
            panel.set_static_metrics(static_metrics)
        frame_metrics = job.get("brain_frame_metrics")
        if isinstance(frame_metrics, dict):
            panel.set_frame_metric_data(frame_metrics)
            viewer.frameChanged.connect(panel.request_frame)
        initial_filter_rules = job.get("initial_brain_region_filter_rules")
        if isinstance(initial_filter_rules, list):
            panel.set_filter_rules(copy.deepcopy(initial_filter_rules))
        panel.region_selected.connect(viewer.set_brain_region_highlight)
        return _BrainResultTab(viewer, panel)

    def _waveform_map_settings(self) -> dict[str, Any]:
        _key, selected_name, selected_ids = self._selected_region()
        default_trace_regions = [
            {
                "key": str(self.region_hierarchy.key),
                "name": str(self.region_hierarchy.name),
                "label_ids": [int(value) for value in self.region_hierarchy.label_ids],
            }
        ]
        return {
            "atlas_name": "LPBA40",
            # Brain View displays the shared 3D field map while keeping atlas
            # samples available privately for region-waveform traces.
            # Full analysis samples drive region traces and per-frame Brain Areas
            # metrics; a smaller display subset remains the coloured point overlay.
            "trace_points_mm": self.analysis_samples.points_mm.tolist(),
            "trace_point_label_ids": self.analysis_samples.label_ids.tolist(),
            "trace_point_weights_mm3": self.analysis_samples.weights_mm3.tolist(),
            "trace_source_voxel_counts": {
                str(key): int(value)
                for key, value in self.analysis_samples.source_voxel_counts.items()
            },
            "trace_voxel_volume_mm3": float(self.analysis_samples.voxel_volume_mm3),
            "overlay_points_mm": self.display_samples.points_mm.tolist(),
            "overlay_point_label_ids": self.display_samples.label_ids.tolist(),
            # Selection controls only the visual highlight in the Selector.
            # Brain-region waveform traces are configured later inside the
            # Animated playback dialog's Traces section.
            "selected_region_name": selected_name,
            "selected_region_label_ids": list(selected_ids),
            "highlight_region_label_ids": list(selected_ids),
            "highlight_selected_region": bool(self.region_table.selectedItems()),
            "trace_region_tree": self._trace_region_tree_payload(self.region_hierarchy),
            "trace_regions": default_trace_regions,
            "scroll_observation_traces": False,
            "centre_mm": self._map_volume_centre_mm(),
            "span_mm": self._map_volume_span_mm(),
            "slice_axis": "z",
            "slice_count": BRAIN_FIELD_RENDER_SLICE_COUNT,
            "resolution": BRAIN_FIELD_RENDER_RESOLUTION,
            "field": "B",
            "component": "magnitude",
            "view_type": BRAIN_FIELD_RENDER_MODE,
            "waveform_render_mode": BRAIN_FIELD_RENDER_MODE,
            "point_size": 4.0,
            "logarithmic": self.logarithmic.isChecked(),
            "show_grid": self.show_grid.isChecked(),
            "include_scene": self.include_scene.isChecked(),
            "scene_object_ids": self.selected_scene_object_ids(),
            "scene_object_opacities": self.scene_object_opacity_values(),
            "drive_coil_ids": self.selected_active_coil_ids(),
            "intensity_opacity_enabled": self.intensity_opacity_enabled.isChecked(),
            "volume_opacity_lower": self.volume_opacity_lower.value() / 100.0,
            "volume_opacity_upper": self.volume_opacity_upper.value() / 100.0,
            "volume_opacity_lower_level": self.volume_opacity_lower_level.value() / 100.0,
            "volume_opacity_upper_level": self.volume_opacity_upper_level.value() / 100.0,
            "colour_minimum": (
                self.colour_minimum.value()
                if self.colour_minimum_enabled.isChecked()
                else None
            ),
            "colour_maximum": (
                self.colour_maximum.value()
                if self.colour_maximum_enabled.isChecked()
                else None
            ),
        }

    def _video_region_table_snapshot(self) -> dict[str, Any]:
        current = self.tabs.currentWidget()
        if isinstance(current, _BrainResultTab):
            return current.region_panel.video_table_snapshot()
        return _brain_region_video_table_snapshot(
            self.region_hierarchy,
            self.region_table,
            self._region_filter_rules(),
            self._region_header_labels,
        )

    def _open_waveform_action(self, primary_action: str) -> None:
        def launch(initial_camera: dict[str, Any]) -> None:
            camera_timeline = self._camera_timeline_for_action(initial_camera)
            target_fps, playback_speed = self._camera_playback_options_for_action()
            map_settings = self._waveform_map_settings()
            map_settings["precompute_brain_region_frames"] = primary_action in {"playback", "export"}
            if primary_action == "export":
                map_settings["brain_region_video_table"] = self._video_region_table_snapshot()
            dialog = WaveformPlaybackDialog(
                self.adapter,
                map_kind="brain",
                map_settings=map_settings,
                camera_timeline=camera_timeline,
                initial_camera=initial_camera,
                playback_fps=target_fps,
                playback_speed=playback_speed,
                primary_action=primary_action,
                defer_payload_build=primary_action == "playback",
                parent=self,
            )
            try:
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                if primary_action == "export" or dialog.result_gpu_payload is None:
                    if (
                        primary_action == "playback"
                        and isinstance(dialog.result_gpu_request, dict)
                    ):
                        self._waveform_tab_counter += 1
                        job_label = f"Waveform {self._waveform_tab_counter}"
                        self._start_brain_render_job(
                            render_kind="waveform",
                            request=dialog.result_gpu_request,
                            job_label=job_label,
                            tooltip=(
                                dialog.result_label
                                + " — LPBA40 field map and checked-region spatial RMS traces"
                            ),
                        )
                        return
                    self.calculation_status.showMessage("Brain waveform video export finished")
                    return
                self._waveform_tab_counter += 1
                viewer = WaveformGLView(dialog.result_gpu_payload)
                index = self.tabs.addTab(
                    viewer, f"Waveform {self._waveform_tab_counter}"
                )
                self.tabs.setTabToolTip(
                    index,
                    dialog.result_label
                    + " — LPBA40 field map and checked-region spatial RMS traces",
                )
                self.tabs.setCurrentIndex(index)
                viewer.refresh_viewport()
                self.calculation_status.showMessage("Animated brain analysis ready")
            finally:
                dialog.result_figure = None
                dialog.result_gpu_payload = None
                dialog.result_gpu_request = None
                dialog.deleteLater()

        self._request_action_camera(launch)

    def open_animated_analysis(self) -> None:
        self._open_waveform_action("playback")

    def _current_video_export_tab(self) -> _BrainResultTab | None:
        widget = self.tabs.currentWidget()
        if (
            isinstance(widget, _BrainResultTab)
            and widget.viewer.can_export_video()
        ):
            return widget
        return None

    def _update_video_export_button(self) -> None:
        self.export_video_button.setVisible(self._current_video_export_tab() is not None)

    def _export_current_animation_video(self, *_args) -> None:
        """Run the original exporter with the selected Brain result frozen.

        The export dialog itself is the pre-layout-change implementation.  The
        selected result supplies its cached payload/table state, then its WebGL
        surface is fully retired so the original exporter is again the only GPU
        renderer alive.  The result viewer is recreated afterward.
        """
        result_tab = self._current_video_export_tab()
        if result_tab is None:
            return
        view = result_tab.viewer
        view.stop_playback()

        def launch(camera: dict[str, Any] | None) -> None:
            try:
                payload = view.export_payload()
                if isinstance(camera, dict):
                    payload["initial_camera"] = copy.deepcopy(camera)
                payload["brain_region_video_table"] = (
                    result_tab.region_panel.video_table_snapshot()
                )
                frame_metrics = result_tab.region_panel.video_frame_metric_payload()
                if isinstance(frame_metrics, dict):
                    payload["brain_region_frame_metrics"] = frame_metrics
            except Exception as error:  # noqa: BLE001 - rendered-tab payload boundary
                QMessageBox.warning(self, "Unable to export video", str(error))
                return

            splitter_sizes = result_tab.splitter.sizes()
            placeholder = QLabel("Video export in progress…")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            result_tab.splitter.replaceWidget(0, placeholder)

            # The original exporter was stable with one WebGL surface.  Tear the
            # rendered Brain viewer down completely before creating that export
            # preview; do not attempt to share or reconnect to its Chromium page.
            view.cleanup()
            view.deleteLater()

            def run_original_exporter() -> None:
                dialog = WaveformVideoExportDialog(payload, parent=self)
                centre_window_on_parent(dialog, self)
                try:
                    if (
                        dialog.exec() == QDialog.DialogCode.Accepted
                        and dialog.output_path is not None
                    ):
                        self.calculation_status.showMessage(
                            f"Video exported to {dialog.output_path}"
                        )
                finally:
                    dialog.deleteLater()
                    restored = WaveformGLView(
                        payload,
                        initial_camera=(camera if isinstance(camera, dict) else None),
                    )
                    result_tab.splitter.replaceWidget(0, restored)
                    result_tab.viewer = restored
                    restored.frameChanged.connect(result_tab.region_panel.request_frame)
                    result_tab.region_panel.region_selected.connect(
                        restored.set_brain_region_highlight
                    )
                    result_tab.region_panel._selection_changed()
                    if splitter_sizes:
                        result_tab.splitter.setSizes(splitter_sizes)
                    restored.refresh_viewport()
                    placeholder.deleteLater()
                    self._update_video_export_button()

            QTimer.singleShot(0, run_original_exporter)

        view.request_camera(launch)

    def open_video_export(self) -> None:
        """Compatibility entry point: export only the selected rendered animation."""
        self._export_current_animation_video()

    # --- window lifecycle ------------------------------------------------
    def _persistent_ui_state(self) -> dict[str, Any]:
        return self._camera_persistent_state()

    def _restore_persistent_ui_state(self, state: dict[str, Any]) -> None:
        self._restore_camera_persistent_state(state)

    def _reset_defaults(self, *_args) -> None:
        if not reset_window_to_default_settings(self):
            return
        self._fixed_camera_user_edited = False
        self.region_filter_table.setRowCount(0)
        self._selected_region_filter_row = -1
        self._region_filter_selection_changed()
        self._apply_region_filters()
        self._update_opacity_controls()
        self._render_selector()
        self.calculation_status.showMessage("Brain View settings reset to defaults")

    def _view_for_tab_widget(self, widget: QWidget | None):
        if widget is getattr(self, "_selector_page", None):
            return self.plot
        if isinstance(widget, _BrainResultTab):
            return widget.viewer
        if isinstance(widget, (PlotView, WaveformGLView)):
            return widget
        return None

    def _current_plot(self):
        return self._view_for_tab_widget(self.tabs.currentWidget()) or self.plot

    def _toggle_current_plot_fullscreen(self, *_args) -> None:
        toggle = getattr(self._current_plot(), "toggle_fullscreen", None)
        if callable(toggle):
            toggle()

    def _close_view_tab(self, index: int) -> None:
        if index == self._selector_tab_index:
            return
        widget = self.tabs.widget(index)
        for job in getattr(self, "_render_jobs", {}).values():
            if job.get("page") is not widget:
                continue
            job["discard"] = True
            worker = job.get("worker")
            request_cancellation = getattr(worker, "request_cancellation", None)
            if callable(request_cancellation):
                request_cancellation()
            break
        self.tabs.removeTab(index)
        view = self._view_for_tab_widget(widget)
        if view is self._active_render_view:
            self._active_render_view = None
        if isinstance(view, (PlotView, WaveformGLView)):
            view.cleanup()
        if widget is not None:
            widget.deleteLater()
        self._render_tab_changed(self.tabs.currentIndex())

    # The shared background-render controller uses the 2D/3D workspace name
    # for its terminal-page Close action. Brain View keeps its own public tab
    # helper but exposes the same behavior to the mixin.
    def _close_map_tab(self, index: int) -> None:
        self._close_view_tab(index)

    def _render_tab_changed(self, current_index: int) -> None:
        current = self.tabs.widget(current_index)
        resolved = self._view_for_tab_widget(current)
        current_view = resolved if isinstance(resolved, WaveformGLView) else None
        for index in range(self.tabs.count()):
            view = self._view_for_tab_widget(self.tabs.widget(index))
            if not isinstance(view, WaveformGLView):
                continue
            if view is current_view:
                view.set_backgrounded(False)
            else:
                view.set_backgrounded(True, discard=True)
        self._active_render_view = current_view
        self._update_video_export_button()

    def _cleanup_plots(self) -> None:
        if getattr(self, "_plots_cleanup_started", False):
            return
        self._plots_cleanup_started = True
        self._cancel_all_render_jobs()
        seen: set[int] = set()
        for index in range(self.tabs.count()):
            view = self._view_for_tab_widget(self.tabs.widget(index))
            if isinstance(view, (PlotView, WaveformGLView)) and id(view) not in seen:
                seen.add(id(view))
                view.cleanup()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._cleanup_plots()
        super().closeEvent(event)
