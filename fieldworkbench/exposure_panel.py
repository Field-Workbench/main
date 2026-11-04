"""Shared regional field and playback panels for Brain, 2D, and 3D views."""

from __future__ import annotations

import base64
import copy
import math
import textwrap
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .waveform_gl_view import WaveformGLView

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, Signal, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette, QPen, QPolygon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QColorDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .spatial_analysis import (
    AnalysisRegionNode, RegionMetrics, RegionalAnalysisError,
)
from .window_utils import CompactDoubleSpinBox

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


class _WrappedToolTipHeaderView(QHeaderView):
    """Header view that keeps long metric help readable on narrow columns.

    Some Linux Qt styles constrain the native header-item tooltip to the hovered
    section, which can leave only the tail of a long sentence visible.  Show the
    same tooltip explicitly with sensible line wrapping so it is sized as a normal
    popup instead of being clipped to a statistics column.
    """

    def viewportEvent(self, event):  # noqa: N802 - Qt API name
        if event.type() == QEvent.Type.ToolTip:
            section = self.logicalIndexAt(event.pos())
            if section >= 0:
                tooltip = self.model().headerData(
                    section, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole
                )
                tooltip = str(tooltip or "").strip()
                if tooltip:
                    wrapped = "\n".join(textwrap.wrap(tooltip, width=52))
                    QToolTip.showText(event.globalPos(), wrapped, self)
                    event.accept()
                    return True
        return super().viewportEvent(event)


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
    tree: QTreeWidget, hierarchy: AnalysisRegionNode
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


class ExposureRegionPanel(QWidget):
    """Shared per-result region table with independent filters/sort/selection."""

    region_selected = Signal(object)
    FRAME_HEADER_LABELS = (
        "Structure", "RMS |B|", "Mean |B|", "P95 |B|", "Peak |B|",
        "Uniformity", "Direction", "Volume",
    )
    SUMMARY_HEADER_LABELS = (
        "Structure", "Playback RMS |B|", "Mean |B|", "Max P95 |B|",
        "Absolute peak |B|", "B-time", "B²-time", "Volume",
    )
    EXPOSURE_SUMMARY_HEADER_LABELS = (
        "Region", "Mean RMS |B|", "Exposure RMS |B|", "Peak RMS |B|",
        "Peak time", "B-time", "B²-time", "",
    )
    FRAME_UNITS = ("", "µT", "µT", "µT", "µT", "%", "%", "cm³")
    SUMMARY_UNITS = ("", "µT", "µT", "µT", "µT", "µT·s", "µT²·s", "cm³")
    EXPOSURE_SUMMARY_UNITS = ("", "µT", "µT", "µT", "s", "µT·s", "µT²·s", "")
    HEADER_LABELS = FRAME_HEADER_LABELS

    def __init__(
        self, hierarchy: AnalysisRegionNode, parent=None, *, measurement=False, region_mode=None
    ):
        super().__init__(parent)
        self.hierarchy = hierarchy
        if region_mode is None:
            region_mode = "measurement" if measurement else "brain"
        self.region_mode = str(region_mode).strip().lower() or "brain"
        if self.region_mode not in {"brain", "measurement", "surface"}:
            self.region_mode = "brain"
        self.measurement = self.region_mode == "measurement"
        self.surface_regions = self.region_mode == "surface"
        self.non_brain = self.region_mode != "brain"
        if self.measurement:
            self.FRAME_HEADER_LABELS = ("Object", *self.FRAME_HEADER_LABELS[1:])
            self.SUMMARY_HEADER_LABELS = ("Object", *self.SUMMARY_HEADER_LABELS[1:])
            self.EXPOSURE_SUMMARY_HEADER_LABELS = (
                "Object", *self.EXPOSURE_SUMMARY_HEADER_LABELS[1:]
            )
        elif self.surface_regions:
            self.FRAME_HEADER_LABELS = ("Region", *self.FRAME_HEADER_LABELS[1:])
            self.SUMMARY_HEADER_LABELS = ("Region", *self.SUMMARY_HEADER_LABELS[1:])
            self.EXPOSURE_SUMMARY_HEADER_LABELS = (
                "Region", *self.EXPOSURE_SUMMARY_HEADER_LABELS[1:7], "Volume"
            )
            self.EXPOSURE_SUMMARY_UNITS = (
                *self.EXPOSURE_SUMMARY_UNITS[:7], "cm³"
            )
        self.analysis_data = None
        self._items_by_key: dict[str, QTreeWidgetItem] = {}
        self._nodes_by_key: dict[str, AnalysisRegionNode] = {}
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
        ] = {"frame": [], "summary": [], "exposure_summary": []}
        self._table_state_by_mode: dict[str, dict[str, Any]] = {}
        self._exposure_summary_filters_visible = False

        # Keep the entire region form scrollable rather than letting a short
        # result host compress splitter children below their controls' geometry.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.region_scroll = QScrollArea(self)
        self.region_scroll.setObjectName("regionPanelScroll")
        self.region_scroll.setWidgetResizable(True)
        self.region_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.region_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.region_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content = QWidget()
        content.setObjectName("regionPanelScrollContent")
        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.region_scroll.setWidget(content)
        outer.addWidget(self.region_scroll)
        heading = QLabel(
            "Measurement volumes" if self.measurement
            else "Surface regions" if self.surface_regions
            else "Brain areas"
        )
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        self.frame_note = QLabel("Static field")
        self.frame_note.setObjectName("hint")
        self.frame_note.setWordWrap(True)
        root.addWidget(self.frame_note)

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

        self.filter_section = QWidget()
        filter_section = self.filter_section
        filter_section_layout = QVBoxLayout(filter_section)
        filter_section_layout.setContentsMargins(0, 0, 0, 10)
        filter_section_layout.setSpacing(6)
        filter_section_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.filter_heading = QLabel("Filters")
        self.filter_heading.setObjectName("sectionHeading")
        filter_section_layout.addWidget(self.filter_heading)
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
        self.filter_table.setMinimumHeight(120)
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

        self.table_section = QWidget()
        table_section = self.table_section
        table_section_layout = QVBoxLayout(table_section)
        table_section_layout.setContentsMargins(0, 0, 0, 0)
        table_section_layout.setSpacing(6)
        table_section_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

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
        self.toggle_filters_button = QPushButton("Show/hide filters")
        self.toggle_filters_button.setVisible(False)
        self.toggle_filters_button.clicked.connect(self._toggle_exposure_summary_filters)
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
        tree_actions.addWidget(self.toggle_filters_button)
        tree_actions.addStretch(1)

        self.table = _BrainRegionTreeWidget()
        self.table.setObjectName("brainResultRegionTable")
        self.table.setMinimumHeight(180)
        self.table.setColumnCount(8)
        self.table.setHeader(_WrappedToolTipHeaderView(Qt.Orientation.Horizontal, self.table))
        self.table.setHeaderLabels(list(self.FRAME_HEADER_LABELS))
        header = self.table.header()
        header.setObjectName("brainRegionHeader")
        header.setMinimumHeight(30)
        header.setSectionsClickable(True)
        header.setSectionsMovable(True)
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
        if self.measurement:
            self.expand_major_tree_button.hide()
            if not any(node.children for node in self.hierarchy.children):
                self.collapse_tree_button.hide()
                self.expand_tree_button.hide()
            self.metrics_note.setText(self._measurement_note())
        elif self.surface_regions:
            self.metrics_note.setText(self._surface_region_note())
        self._apply_mode_headers()
        self._table_state_by_mode["frame"] = self._capture_table_state()
        self._table_state_by_mode["summary"] = {
            "sort_column": 0,
            "sort_order": Qt.SortOrder.AscendingOrder,
            "hidden": [False] * self.table.columnCount(),
            "widths": [220, 126, 98, 102, 116, 100, 108, 92],
        }
        self._table_state_by_mode["exposure_summary"] = {
            "sort_column": 0,
            "sort_order": Qt.SortOrder.AscendingOrder,
            "hidden": [False, False, False, False, False, False, False, True],
            "widths": [250, 128, 142, 128, 112, 124, 132, 72],
        }

    def _measurement_note(self):
        if not self.hierarchy.children:
            return "No measurement volumes in this playback's scene. Assign the Measurement role before creating a new playback."
        return ("Whole-object samples from this playback's frozen scene. Generated samples are volume-weighted; "
                "rows marked [samples] use equal-weight user points. Overlapping objects are reported independently."
                + (" Expand an object to see its defined points (up to 2,000 per object)."
                   if any(node.children for node in self.hierarchy.children) else ""))

    def _surface_region_note(self):
        return (
            "Rows represent inward near-surface volumes derived from authored mesh regions. "
            "Retained field samples are volume-weighted to represent the denser integration shell."
        )

    @staticmethod
    def _metric_text(value: float, suffix: str) -> str:
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):,.4g} {suffix}".rstrip()

    def _active_header_labels(self) -> tuple[str, ...]:
        if self._display_mode == "exposure_summary":
            return self.EXPOSURE_SUMMARY_HEADER_LABELS
        return (
            self.SUMMARY_HEADER_LABELS
            if self._display_mode == "summary"
            else self.FRAME_HEADER_LABELS
        )

    def _active_units(self) -> tuple[str, ...]:
        if self._display_mode == "exposure_summary":
            return self.EXPOSURE_SUMMARY_UNITS
        return (
            self.SUMMARY_UNITS
            if self._display_mode == "summary"
            else self.FRAME_UNITS
        )

    def _filter_metric_choices(self) -> tuple[tuple[str, int | str], ...]:
        if self._display_mode == "exposure_summary":
            return (
                ("Mean RMS |B|", 1),
                ("Exposure RMS |B|", 2),
                ("Peak RMS |B|", 3),
                ("Peak time", 4),
                ("B-time", 5),
                ("B²-time", 6),
                *(() if self.non_brain else (("Side", "side"),)),
            )
        if self._display_mode == "summary":
            return (
                ("Playback RMS |B|", 1),
                ("Mean |B|", 2),
                ("Max P95 |B|", 3),
                ("Absolute peak |B|", 4),
                ("B-time", 5),
                ("B²-time", 6),
                ("Volume", 7),
                *(() if self.non_brain else (("Side", "side"),)),
            )
        return (
            ("RMS |B|", 1),
            ("Mean |B|", 2),
            ("P95 |B|", 3),
            ("Peak |B|", 4),
            ("Uniformity", 5),
            ("Direction", 6),
            ("Volume", 7),
            *(() if self.non_brain else (("Side", "side"),)),
        )

    def _capture_table_state(self) -> dict[str, Any]:
        header = self.table.header()
        return {
            "sort_column": int(max(0, header.sortIndicatorSection())),
            "sort_order": header.sortIndicatorOrder(),
            "column_order": [header.logicalIndex(index) for index in range(header.count())],
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
        header = self.table.header()
        for visual_index, logical_index in enumerate(state.get("column_order", range(header.count()))):
            current_index = header.visualIndex(logical_index)
            if current_index != visual_index:
                header.moveSection(current_index, visual_index)
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
        if self._display_mode == "exposure_summary":
            tooltips = (
                (
                    "Prepared Measurement object" if self.measurement
                    else "Prepared inward surface region" if self.surface_regions
                    else "Prepared LPBA40 region"
                ),
                "Time-weighted mean of the regional spatial-RMS field magnitude",
                "RMS over physical time of the regional spatial-RMS field magnitude",
                "Highest regional spatial-RMS field magnitude during the exposure",
                "Physical exposure time at which the regional spatial-RMS peak occurs",
                "Integral of regional spatial-RMS magnitude over physical time",
                "Integral of squared regional spatial-RMS magnitude over physical time",
                "Unused in Exposure Summary",
            )
        elif self._display_mode == "summary":
            tooltips = (
                "LPBA40 structure",
                "Space-time RMS of field magnitude over the physical playback",
                "Time-weighted mean of the regional mean field magnitude",
                "Highest rendered-frame spatial 95th percentile",
                "Maximum field magnitude among retained samples and regional frames",
                "Integral of instantaneous spatial RMS magnitude over physical time",
                "Integral of squared instantaneous spatial RMS magnitude over physical time",
                "Atlas volume after the selected brain object's scale",
            )
        else:
            tooltips = (
                "LPBA40 structure",
                "Spatial RMS of field magnitude in microtesla",
                "Volume-weighted mean field magnitude",
                "Volume-weighted 95th percentile field magnitude",
                "Maximum field magnitude among retained analysis samples",
                "P5–P95 uniformity score; higher is more uniform",
                "Consistency of field-vector direction; higher is more aligned",
                "Atlas volume after the selected brain object's scale",
            )
        if self.measurement and self._display_mode != "exposure_summary":
            tooltips = list(tooltips)
            tooltips[0] = "Measurement object from this playback's frozen scene"
            tooltips[7] = "Geometric object volume; custom point sets do not imply uniform volume coverage"
            if self._display_mode == "frame":
                tooltips[2] = "Weighted mean magnitude; equal weights for rows marked [samples]"
                tooltips[3] = "Weighted spatial 95th percentile; equal weights for rows marked [samples]"
        elif self.surface_regions and self._display_mode != "exposure_summary":
            tooltips = list(tooltips)
            tooltips[0] = "Authored surface region represented by an inward sampling volume"
            tooltips[7] = "Integrated inward shell volume represented by this region"
            if self._display_mode == "frame":
                tooltips[2] = "Volume-weighted mean magnitude across retained shell samples"
                tooltips[3] = "Volume-weighted spatial 95th percentile across retained shell samples"
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

    def configure_exposure_summary(self) -> None:
        """Use this shared regional panel for authoritative Exposure Summary results."""
        if self._display_mode != "exposure_summary":
            self._filter_rules_by_mode[self._display_mode] = copy.deepcopy(
                self._filter_rules()
            )
            self._table_state_by_mode[self._display_mode] = self._capture_table_state()
        self._display_mode = "exposure_summary"
        self.mode_controls.hide()
        self.playback_statistics_widget.hide()
        self._apply_mode_headers()
        self._restore_table_state(self._table_state_by_mode["exposure_summary"])
        self.column_actions[7].setVisible(self.surface_regions)
        self.table.setColumnHidden(7, not self.surface_regions)
        # Exposure Summary already presents every prepared region together, so let the
        # regional statistics table fill the available width and keep filters available
        # on demand in a secondary panel beneath it.
        self.region_sections_splitter.setOrientation(Qt.Orientation.Vertical)
        self.region_sections_splitter.insertWidget(0, self.table_section)
        self.region_sections_splitter.insertWidget(1, self.filter_section)
        self.region_sections_splitter.setStretchFactor(0, 1)
        self.region_sections_splitter.setStretchFactor(1, 0)
        self.region_sections_splitter.setSizes([820, 0])
        self.filter_section.setMinimumWidth(0)
        self.table_section.setMinimumWidth(0)
        self.filter_heading.setText("Filters / highlights")
        self.filter_table.setMinimumHeight(160)
        self.filter_table.setMaximumHeight(280)
        self.toggle_filters_button.setVisible(True)
        self._set_exposure_summary_filters_visible(False)
        self.metrics_note.hide()
        self.selection_note.setText(
            "Select a row to make it the active region for the other Exposure Analysis tools."
        )
        self.set_filter_rules(
            copy.deepcopy(self._filter_rules_by_mode.get("exposure_summary", []))
        )

    def _set_exposure_summary_filters_visible(self, visible: bool) -> None:
        self._exposure_summary_filters_visible = bool(visible)
        self.filter_section.setVisible(self._exposure_summary_filters_visible)
        if self._display_mode == "exposure_summary":
            if self._exposure_summary_filters_visible:
                self.region_sections_splitter.setSizes([620, 220])
            else:
                self.region_sections_splitter.setSizes([820, 0])

    def _toggle_exposure_summary_filters(self) -> None:
        if self._display_mode != "exposure_summary":
            return
        self._set_exposure_summary_filters_visible(
            not self._exposure_summary_filters_visible
        )

    def set_exposure_summary_status(self, text: str) -> None:
        if self._display_mode != "exposure_summary":
            self.configure_exposure_summary()
        self.frame_note.setText(str(text))

    def set_exposure_summary_results(self, results: dict[str, Any]) -> None:
        """Populate authoritative all-region summary metrics without viewer-only statistics."""
        if self._display_mode != "exposure_summary":
            self.configure_exposure_summary()
        self.table.setSortingEnabled(False)
        result_keys = {str(key) for key in results}
        for key, item in self._items_by_key.items():
            if key in result_keys:
                continue
            for column in range(1, 7):
                item.setText(column, "—")
                item.setData(column, Qt.ItemDataRole.UserRole, None)

        duration = None
        samples = None
        max_interval = None
        for key, data in results.items():
            item = self._items_by_key.get(str(key))
            if item is None or not isinstance(data, dict):
                continue
            values = (
                data.get("mean_spatial_rms_uT"),
                data.get("playback_rms_uT"),
                data.get("peak_spatial_rms_uT"),
                data.get("peak_time_s"),
                data.get("b_time_uT_s"),
                data.get("b2_time_uT2_s"),
            )
            for column, (value, suffix) in enumerate(
                zip(values, self.EXPOSURE_SUMMARY_UNITS[1:7], strict=True), start=1
            ):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = float("nan")
                item.setText(column, self._metric_text(number, suffix))
                item.setData(
                    column,
                    Qt.ItemDataRole.UserRole,
                    number if math.isfinite(number) else None,
                )
            base_tooltip = (
                "Spatial-RMS statistics across this prepared Measurement sample set over the complete physical exposure."
                if self.measurement
                else "Spatial-RMS statistics across this inward near-surface volume over the complete physical exposure."
                if self.surface_regions
                else "Regional spatial-RMS statistics over the complete physical exposure."
            )
            if self.surface_regions:
                node = self._nodes_by_key.get(str(key))
                metadata = node.metadata if node is not None and isinstance(node.metadata, dict) else {}
                try:
                    volume = float(metadata.get("volume_cm3"))
                except (TypeError, ValueError):
                    volume = float("nan")
                if math.isfinite(volume) and volume > 0.0:
                    item.setText(7, self._metric_text(volume, "cm³"))
                    item.setData(7, Qt.ItemDataRole.UserRole, volume)
                extra = item.toolTip(0)
                if extra:
                    base_tooltip += "\n" + extra
            item.setToolTip(0, base_tooltip)
            duration = data.get("duration_s", duration)
            samples = data.get("source_sample_count", samples)
            max_interval = data.get("max_source_interval_s", max_interval)

        self.table.setSortingEnabled(True)
        self._resort()
        self._apply_filters()
        details = []
        try:
            if duration is not None and math.isfinite(float(duration)):
                details.append(f"Physical exposure: {float(duration):,.6g} s")
        except (TypeError, ValueError):
            pass
        try:
            if samples is not None and int(samples) > 0:
                details.append(f"source samples: {int(samples):,}")
        except (TypeError, ValueError):
            pass
        try:
            if max_interval is not None and math.isfinite(float(max_interval)):
                details.append(f"largest source interval: {float(max_interval):,.6g} s")
        except (TypeError, ValueError):
            pass
        self.frame_note.setText(" • ".join(details) if details else "Exposure summary calculated")

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
        self.metrics_note.show()
        self.playback_statistics_widget.setVisible(summary)
        if summary:
            scope = str(self._playback_metadata.get("statistics_scope", "full") or "full").strip().lower()
            scope_text = "preview duration" if scope == "preview" else "complete physical exposure"
            self.metrics_note.setText(
                f"Rows summarize the {scope_text}. B-time and B²-time "
                "are comparative magnetic-field exposure indices, not a biological dose. "
                "Summary filters and sorting are independent from Current frame."
            )
            self._show_summary_values()
        else:
            self.metrics_note.setText(
                "Rows summarize volume-weighted atlas samples for the current frame. "
                "Filters and sorting belong to this mode and result tab only."
            )
            if self.measurement:
                self.metrics_note.setText(self._measurement_note())
            elif self.surface_regions:
                self.metrics_note.setText(self._surface_region_note())
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
        if metadata.get("waveform_mode") == "exposure":
            preview = self._run_number(metadata.get("preview_duration_s", metadata.get("exposure_time_s")))
            statistics_scope = str(metadata.get("statistics_scope", "full") or "full").strip().lower()
            statistics_duration = self._run_number(
                metadata.get("statistics_duration_s", metadata.get("exposure_time_s"))
            )
            statistics_text = (
                f"preview stats {statistics_duration} s"
                if statistics_scope == "preview"
                else f"full-exposure stats {statistics_duration} s"
            )
            exposure_text = f"{duration} s physical · preview {preview} s · {statistics_text}"
        else:
            exposure_text = f"{duration} s of physical waveform time"
        self.playback_statistic_values["exposure"].setText(exposure_text)

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
        analysis_estimated = bool(metadata.get("analysis_sample_count_estimated", False))
        regional_count = int(metadata.get("regional_frame_sample_count", 0) or 0)
        atlas_count = int(metadata.get("atlas_sample_count", 0) or 0)
        analysis_text = f"~{analysis_count:,} time samples" if analysis_estimated else f"{analysis_count:,} time samples"
        sampling_text = (
            f"{frame_count:,} rendered frames • {analysis_text} • "
            f"{atlas_count:,} analysis samples"
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
        self._nodes_by_key.clear()

        def add_node(node: AnalysisRegionNode, parent: QTreeWidgetItem | None) -> None:
            item = _NumericTreeItem([node.name, "—", "—", "—", "—", "—", "—", "—"])
            item.setData(0, Qt.ItemDataRole.UserRole, node.name.casefold())
            item.setData(0, REGION_KEY_ROLE, node.key)
            item.setData(0, REGION_LABEL_IDS_ROLE, list(node.label_ids))
            side = "left" if node.key.endswith(".left") else "right" if node.key.endswith(".right") else ""
            item.setData(0, REGION_SIDE_ROLE, side)
            self._items_by_key[node.key] = item
            self._nodes_by_key[node.key] = node
            if self.surface_regions and isinstance(node.metadata, dict):
                details = []
                try:
                    details.append(f"Sampling depth: {float(node.metadata.get('sampling_depth_mm')):,.6g} mm inward")
                except (TypeError, ValueError):
                    pass
                try:
                    details.append(f"Sampled volume: {float(node.metadata.get('volume_cm3')):,.6g} cm³")
                except (TypeError, ValueError):
                    pass
                try:
                    details.append(f"Surface area: {float(node.metadata.get('surface_area_cm2')):,.6g} cm²")
                except (TypeError, ValueError):
                    pass
                try:
                    source_count = int(node.metadata.get("source_voxel_count", 0) or 0)
                except (TypeError, ValueError):
                    source_count = 0
                try:
                    retained_count = int(node.metadata.get("retained_sample_count", 0) or 0)
                except (TypeError, ValueError):
                    retained_count = 0
                if source_count:
                    details.append(f"Source voxels: {source_count:,}")
                if retained_count:
                    details.append(f"Retained field samples: {retained_count:,}")
                if details:
                    item.setToolTip(0, "\n".join(details))
            if parent is None:
                self.table.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for child in node.children:
                add_node(child, item)

        if self.hierarchy.label_ids:
            add_node(self.hierarchy, None)
        else:
            for node in self.hierarchy.children:
                add_node(node, None)
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

        for index in range(self.table.topLevelItemCount()):
            apply(self.table.topLevelItem(index))

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

    def set_static_metrics(self, metrics: dict[str, RegionMetrics]) -> None:
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
            raise RegionalAnalysisError("Animated Brain Areas metric payload has an invalid shape.")
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
                raise RegionalAnalysisError(
                    "Animated Brain Areas playback summary has an invalid shape."
                )
            peak_raw = base64.b64decode(
                str(summary.get("peak_times_s_b64", ""))
            )
            peak_times = np.frombuffer(peak_raw, dtype=np.float64)
            if peak_times.shape != (len(self._frame_keys),):
                raise RegionalAnalysisError(
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
        # New Scripted playback carries current-frame Brain Areas metrics only;
        # cumulative exposure summaries belong to the separate exposure-analysis
        # workflow. Keep the old summary UI available for legacy saved maps that
        # actually contain one, but do not show a disabled two-mode selector.
        self.mode_controls.setVisible(self._summary_values is not None)
        self.request_frame(0, 0.0)


    def playback_summary_for_region(self, key: str) -> dict[str, Any] | None:
        """Return one frozen playback-summary row for analysis pages.

        The returned values are the same authoritative values displayed by the
        Playback summary table; this avoids maintaining a second definition of
        the cumulative regional statistics in the Exposure Analysis workflow.
        """
        if self._summary_values is None:
            return None
        try:
            index = self._frame_keys.index(str(key))
        except ValueError:
            return None
        if index >= len(self._summary_values):
            return None
        values = np.asarray(self._summary_values[index], dtype=np.float64)
        if values.shape != (7,):
            return None
        peak_time = None
        if self._summary_peak_times_s is not None and index < len(self._summary_peak_times_s):
            candidate = float(self._summary_peak_times_s[index])
            peak_time = candidate if math.isfinite(candidate) else None
        return {
            "playback_rms_uT": float(values[0]),
            "mean_uT": float(values[1]),
            "max_p95_uT": float(values[2]),
            "absolute_peak_uT": float(values[3]),
            "b_time_uT_s": float(values[4]),
            "b2_time_uT2_s": float(values[5]),
            "volume_cm3": float(values[6]),
            "peak_time_s": peak_time,
            "sample_count": self._sample_counts[index] if index < len(self._sample_counts) else 0,
            "voxel_count": self._voxel_counts[index] if index < len(self._voxel_counts) else 0,
            "playback": copy.deepcopy(self._playback_metadata),
        }

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
        scope = str(self._playback_metadata.get("statistics_scope", "full") or "full").strip().lower()
        duration = self._run_number(
            self._playback_metadata.get(
                "statistics_duration_s", self._playback_metadata.get("exposure_time_s")
            )
        )
        self.frame_note.setText(
            f"{'Preview statistics' if scope == 'preview' else 'Full exposure'} • {duration} s"
        )
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
        if self.measurement:
            sample_tooltip = f"{sample_count:,} retained analysis samples"
        elif self.surface_regions:
            sample_tooltip = (
                f"{voxel_count:,} source shell voxels; "
                f"{sample_count:,} retained weighted field samples"
            )
        else:
            sample_tooltip = (
                f"{voxel_count:,} atlas voxels; {sample_count:,} retained analysis samples"
            )
        item.setToolTip(0, sample_tooltip)
        if peak_time_s is not None and math.isfinite(float(peak_time_s)):
            item.setToolTip(
                4,
                "Maximum retained-sample field magnitude occurred at "
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
                    (
                        "Measurement volumes — Playback summary" if self.measurement
                        else "Surface regions — Playback summary" if self.surface_regions
                        else "Brain areas — Playback summary"
                    )
                    if self._display_mode == "summary"
                    else (
                        "Measurement volumes" if self.measurement
                        else "Surface regions" if self.surface_regions
                        else "Brain areas"
                    )
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
            if self._display_mode == "exposure_summary":
                self.selection_note.setText(
                    "Select a row to make it the active region for the other Exposure Analysis tools."
                )
            else:
                self.selection_note.setText("No region highlighted in this result viewer.")
            self.region_selected.emit(())
            return
        label_ids = tuple(int(value) for value in (item.data(0, REGION_LABEL_IDS_ROLE) or []))
        if self._display_mode == "exposure_summary":
            self.selection_note.setText(
                f"Active region: {item.text(0)} — used by the other Exposure Analysis tools."
            )
        else:
            if self.measurement:
                self.selection_note.setText(
                    f"Highlighted samples: {item.text(0)}."
                )
            elif self.surface_regions:
                self.selection_note.setText(f"Selected surface region: {item.text(0)}.")
            else:
                self.selection_note.setText(
                    f"Highlighted: {item.text(0)} • {len(label_ids)} LPBA40 region"
                    f"{'s' if len(label_ids) != 1 else ''}."
                )
        self.region_selected.emit(label_ids)


def _brain_region_video_table_snapshot(
    hierarchy: AnalysisRegionNode,
    tree: QTreeWidget,
    rules: list[tuple[int | str, str, float | str, str, str]],
    header_labels: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Freeze the active Brain Areas view for exported-video table rendering."""

    rows: list[dict[str, Any]] = []

    def add_node(node: AnalysisRegionNode, parent_key: str | None, depth: int) -> None:
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

    if hierarchy.label_ids:
        add_node(hierarchy, None, 0)
    else:
        for node in hierarchy.children:
            add_node(node, None, 0)

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


class ExposureResultTab(QWidget):
    """One rendered result plus its independently collapsible region panel."""

    def __init__(self, viewer: WaveformGLView, panel: ExposureRegionPanel, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.region_panel = panel
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(8)
        splitter.addWidget(viewer)
        splitter.addWidget(panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        panel.setMinimumWidth(220)
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([820, 560])
        layout.addWidget(splitter, 1)
        self.splitter = splitter
        splitter.handle(1).setToolTip("Drag right to hide regions; drag left to show them.")

    def cleanup(self):
        self.viewer.cleanup()

    def toggle_regions(self):
        width = max(1, self.splitter.width())
        if self.splitter.sizes()[1] == 0:
            self.splitter.setSizes([max(1, width - 460), 460])
        else:
            self.splitter.setSizes([width, 0])

