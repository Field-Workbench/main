"""Shared upper-panel template for Exposure Analysis workspaces."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .analysis_region_preview import AnalysisRegionPreview


class ExposureAnalysisHeaderPanel(QWidget):
    """Reusable Exposure Analysis context header with a target preview.

    The expanded header owns page navigation/title, a short explanation, the
    common target/exposure identity card, caller-provided context rows, the
    Region Viewer launcher, and one compact whole-target preview.  Every user
    of this shared header also gets the same Show/Hide analysis-info control.
    Collapsing the header leaves the navigation/title/control row in place, hides
    the detailed information and preview, and pauses the hidden preview renderer.
    """

    def __init__(
        self,
        parent=None,
        *,
        title: str,
        explanation: str,
        show_targets_button: bool = False,
        show_home_button: bool = False,
        preview_object_name: str = "",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("exposureAnalysisHeaderPanel")
        self._analysis_info_visible = True
        self._expanded_parent_fraction: float | None = None
        self._expanded_internal_sizes = [680, 340]
        self._preview_requested_active = True
        self._preview_inactive_detail: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setObjectName("exposureAnalysisHeaderSplitter")
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)

        self.info_panel = QWidget(self.splitter)
        info = QVBoxLayout(self.info_panel)
        info.setContentsMargins(0, 0, 10, 0)
        info.setSpacing(5)

        self.navigation_widget = QWidget(self.info_panel)
        self.navigation_layout = QHBoxLayout(self.navigation_widget)
        self.navigation_layout.setContentsMargins(0, 0, 0, 0)
        self.navigation_layout.setSpacing(7)
        self.targets_button = QPushButton("← Analysis targets", self.navigation_widget)
        self.targets_button.setToolTip(
            "Return to the target chooser to switch between brain, surface-region, and Measurement analyses."
        )
        self.targets_button.setVisible(bool(show_targets_button))
        self.navigation_layout.addWidget(self.targets_button, 0)

        self.home_button = QPushButton("← Analysis home", self.navigation_widget)
        self.home_button.setToolTip("Return to the guided analysis overview.")
        self.home_button.setVisible(bool(show_home_button))
        self.navigation_layout.addWidget(self.home_button, 0)

        self.heading_label = QLabel(str(title), self.navigation_widget)
        self.heading_label.setObjectName("dialogHeading")
        self.heading_label.setWordWrap(True)
        self.navigation_layout.addWidget(self.heading_label, 1)

        self.analysis_info_button = QPushButton("Hide analysis info", self.navigation_widget)
        self.analysis_info_button.setObjectName("exposureAnalysisToggleInfoButton")
        self.analysis_info_button.setToolTip(
            "Collapse the shared analysis information and target preview to make more room for the current page."
        )
        self.analysis_info_button.clicked.connect(self.toggle_analysis_info)
        self.navigation_layout.addWidget(self.analysis_info_button, 0)

        self.region_viewer_button = QPushButton("Open region viewer…", self.navigation_widget)
        self.region_viewer_button.setObjectName("exposureAnalysisOpenRegionViewerButton")
        self.region_viewer_button.setToolTip(
            "Open a modeless 3D viewer for browsing and highlighting this target's regions."
        )
        self.region_viewer_button.setEnabled(False)
        self.navigation_layout.addWidget(self.region_viewer_button, 0)
        info.addWidget(self.navigation_widget)

        self.explanation_label = QLabel(str(explanation), self.info_panel)
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        info.addWidget(self.explanation_label)

        self.target_label = QLabel("Target: Choose a target", self.info_panel)
        self.target_label.setObjectName("dialogHeading")
        self.target_label.setWordWrap(True)
        self.target_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.addWidget(self.target_label)

        self.exposure_label = QLabel("Exposure: Exposure", self.info_panel)
        self.exposure_label.setObjectName("dialogHeading")
        self.exposure_label.setWordWrap(True)
        self.exposure_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.addWidget(self.exposure_label)

        self.stats_label = QLabel("", self.info_panel)
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.addWidget(self.stats_label)

        self.extra_widget = QWidget(self.info_panel)
        self.extra_layout = QVBoxLayout(self.extra_widget)
        self.extra_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_layout.setSpacing(5)
        info.addWidget(self.extra_widget)
        info.addStretch(1)

        self.preview_panel = QWidget(self.splitter)
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(10, 0, 0, 0)
        preview_layout.setSpacing(4)
        self.preview_heading = QLabel("Target preview", self.preview_panel)
        self.preview_heading.setObjectName("sectionHeading")
        preview_layout.addWidget(self.preview_heading)
        self.preview = AnalysisRegionPreview(
            self.preview_panel,
            show_region_viewer_button=False,
        )
        if preview_object_name:
            self.preview.setObjectName(str(preview_object_name))
        preview_em = max(1, int(self.fontMetrics().height()))
        self.preview.setMinimumWidth(max(240, preview_em * 14))
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_layout.addWidget(self.preview, 1)

        self.preview_note = QLabel("", self.preview_panel)
        self.preview_note.setObjectName("hint")
        self.preview_note.setWordWrap(True)
        self.preview_note.hide()
        preview_layout.addWidget(self.preview_note)

        self.splitter.addWidget(self.info_panel)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([680, 340])

    @property
    def analysis_info_visible(self) -> bool:
        return bool(self._analysis_info_visible)

    def _parent_vertical_splitter(self) -> QSplitter | None:
        parent = self.parentWidget()
        while parent is not None and parent is not self.window():
            if isinstance(parent, QSplitter) and parent.orientation() == Qt.Orientation.Vertical:
                return parent
            parent = parent.parentWidget()
        return None

    def _remember_parent_fraction(self) -> None:
        parent = self._parent_vertical_splitter()
        if parent is None:
            return
        index = parent.indexOf(self)
        sizes = parent.sizes()
        total = sum(max(0, int(value)) for value in sizes)
        if 0 <= index < len(sizes) and total > 0 and sizes[index] > 0:
            self._expanded_parent_fraction = float(sizes[index]) / float(total)

    def _collapsed_height(self) -> int:
        return max(34, int(self.navigation_widget.sizeHint().height()) + 2)

    def _resize_parent_for_state(self, expanded: bool) -> None:
        parent = self._parent_vertical_splitter()
        if parent is None:
            return
        index = parent.indexOf(self)
        sizes = parent.sizes()
        if index < 0 or index >= len(sizes) or len(sizes) < 2:
            return
        total = sum(max(0, int(value)) for value in sizes)
        if total <= 0:
            return
        if expanded:
            fraction = self._expanded_parent_fraction
            if fraction is None or not (0.05 <= fraction <= 0.95):
                fraction = 1.0 / 3.0
            wanted = max(self._collapsed_height(), int(round(total * fraction)))
        else:
            wanted = min(total, self._collapsed_height())
        delta = int(sizes[index]) - wanted
        sizes[index] = wanted
        sibling = 0 if index != 0 else 1
        if 0 <= sibling < len(sizes):
            sizes[sibling] = max(0, int(sizes[sibling]) + delta)
        parent.setSizes(sizes)

    def _apply_preview_activity(self) -> None:
        active = self._analysis_info_visible and self._preview_requested_active
        detail = self._preview_inactive_detail
        if not self._analysis_info_visible:
            detail = "Target preview is hidden while analysis info is collapsed."
        self.preview.set_rendering_active(active, inactive_detail=detail)

    def set_preview_rendering_active(
        self,
        active: bool,
        *,
        inactive_detail: str | None = None,
    ) -> None:
        """Remember the owner's preview state while also respecting header collapse."""
        self._preview_requested_active = bool(active)
        if inactive_detail is not None:
            self._preview_inactive_detail = str(inactive_detail)
        self._apply_preview_activity()

    def set_analysis_info_visible(self, visible: bool) -> None:
        """Expand or collapse the shared analysis information banner."""
        visible = bool(visible)
        if visible == self._analysis_info_visible:
            self.analysis_info_button.setText(
                "Hide analysis info" if visible else "Show analysis info"
            )
            return

        if not visible:
            self._remember_parent_fraction()
            current_sizes = self.splitter.sizes()
            if len(current_sizes) >= 2 and sum(current_sizes) > 0:
                self._expanded_internal_sizes = list(current_sizes[:2])

        self._analysis_info_visible = visible
        if visible:
            self.setMaximumHeight(16777215)
            self.analysis_info_button.setText("Hide analysis info")
            self.analysis_info_button.setToolTip(
                "Collapse the shared analysis information and target preview to make more room for the current page."
            )
            self.heading_label.show()
            self.explanation_label.show()
            self.target_label.show()
            self.exposure_label.show()
            self.stats_label.setVisible(bool(self.stats_label.text().strip()))
            self.extra_widget.show()
            self.region_viewer_button.show()
            self.preview_panel.show()
            self.splitter.setSizes(self._expanded_internal_sizes)
            self._resize_parent_for_state(True)
        else:
            self.analysis_info_button.setText("Show analysis info")
            self.analysis_info_button.setToolTip(
                "Restore the shared analysis information and target preview."
            )
            self.heading_label.show()
            self.explanation_label.hide()
            self.target_label.hide()
            self.exposure_label.hide()
            self.stats_label.hide()
            self.extra_widget.hide()
            self.region_viewer_button.show()
            self.preview_panel.hide()
            self.splitter.setSizes([1, 0])
            self.setMaximumHeight(self._collapsed_height())
            self._resize_parent_for_state(False)

        self._apply_preview_activity()
        self.updateGeometry()

    def toggle_analysis_info(self, *_args) -> None:
        self.set_analysis_info_visible(not self._analysis_info_visible)

    def set_page(self, title: str, explanation: str) -> None:
        self.heading_label.setText(str(title))
        self.explanation_label.setText(str(explanation))

    def set_identity(self, target: str, exposure: str) -> None:
        self.target_label.setText(f"Target: {target}")
        self.exposure_label.setText(f"Exposure: {exposure}")

    def set_stats_text(self, text: str) -> None:
        self.stats_label.setText(str(text or ""))
        self.stats_label.setVisible(
            self._analysis_info_visible and bool(str(text or "").strip())
        )

    def set_preview_note(self, text: str | None) -> None:
        value = str(text or "").strip()
        self.preview_note.setText(value)
        self.preview_note.setVisible(bool(value))

    def set_navigation(self, *, targets: bool, home: bool) -> None:
        self.targets_button.setVisible(bool(targets))
        self.home_button.setVisible(bool(home))
        if not self._analysis_info_visible:
            self.setMaximumHeight(self._collapsed_height())
            self._resize_parent_for_state(False)

    def cleanup(self) -> None:
        self.preview.cleanup()
