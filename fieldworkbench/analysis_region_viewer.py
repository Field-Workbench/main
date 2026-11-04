"""Modeless browser for Exposure Analysis region-set display geometry."""
from __future__ import annotations

from PySide6.QtCore import QTimer, Signal, Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis_region_preview import AnalysisRegionPreview
from .analysis_region_scene import (
    AnalysisRegionScene,
    AnalysisRegionSceneError,
    analysis_region_node_display_metrics,
    analysis_region_scene_identity,
)
from .scene_view_controls import SceneViewControls
from .spatial_analysis import AnalysisRegionNode
from .window_utils import configure_application_window


_REGION_KIND_NAMES = {
    "near_surface_volume_set": "Near-surface region set",
    "near_surface_volume": "Near-surface region",
    "measurement_volume": "Measurement volume",
    "brain_atlas": "Brain atlas region set",
    "brain_region": "Brain region",
    "volume": "Volume region",
}


class AnalysisRegionViewer(QDialog):
    """One reusable top-level region browser for an Exposure Analysis workspace.

    Closing the window only hides it.  Its owner keeps the instance alive so the
    splitter position, camera and WebEngine surface survive repeated opens during
    the workspace session.  :meth:`cleanup` is called explicitly when the owning
    Exposure Analysis workspace is closed.
    """

    viewer_hidden = Signal()

    def __init__(self, parent=None) -> None:
        # A native-parentless application window remains modeless and gets its own
        # normal desktop window/taskbar treatment.  The logical owner is responsible
        # for closing/cleaning it up with the Exposure Analysis workspace.
        super().__init__(None)
        configure_application_window(self)
        # Region Viewer is intentionally reusable during the owning Exposure
        # Analysis session; closing it hides rather than deletes the top-level.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle("Region viewer")
        self.resize(1040, 720)
        self.setMinimumSize(720, 480)

        self._scene: AnalysisRegionScene | None = None
        self._items_by_key: dict[str, QTreeWidgetItem] = {}
        self._metrics_by_key: dict[str, dict[str, object]] = {}
        self._scene_identity = ""
        self._cleaned_up = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        heading_row = QHBoxLayout()
        self.title_label = QLabel("Region viewer", self)
        self.title_label.setObjectName("dialogHeading")
        self.title_label.setWordWrap(True)
        heading_row.addWidget(self.title_label, 1)
        root.addLayout(heading_row)

        self.context_note = QLabel("", self)
        self.context_note.setObjectName("hint")
        self.context_note.setWordWrap(True)
        self.context_note.hide()
        root.addWidget(self.context_note)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setObjectName("analysisRegionViewerSplitter")
        self.splitter.setChildrenCollapsible(False)

        browser_panel = QFrame(self.splitter)
        browser_panel.setFrameShape(QFrame.Shape.StyledPanel)
        browser_layout = QVBoxLayout(browser_panel)
        browser_layout.setContentsMargins(8, 8, 8, 8)
        browser_layout.setSpacing(7)

        browser_heading = QLabel("Object / regions", browser_panel)
        browser_heading.setObjectName("sectionHeading")
        browser_layout.addWidget(browser_heading)

        self.region_tree = QTreeWidget(browser_panel)
        self.region_tree.setObjectName("analysisRegionViewerTree")
        self.region_tree.setColumnCount(1)
        self.region_tree.setHeaderLabels(["Object / regions"])
        self.region_tree.setUniformRowHeights(True)
        self.region_tree.currentItemChanged.connect(self._tree_selection_changed)
        browser_layout.addWidget(self.region_tree, 1)

        self.selection_title = QLabel("No region selected", browser_panel)
        self.selection_title.setObjectName("sectionHeading")
        self.selection_title.setWordWrap(True)
        browser_layout.addWidget(self.selection_title)

        self.selection_details = QLabel("", browser_panel)
        self.selection_details.setObjectName("hint")
        self.selection_details.setWordWrap(True)
        self.selection_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        browser_layout.addWidget(self.selection_details)

        viewer_panel = QWidget(self.splitter)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(4)
        self.preview = AnalysisRegionPreview(
            viewer_panel,
            triangle_budget=None,
            show_region_viewer_button=False,
            regional_display=True,
            display_modebar=True,
            modebar_profile="full_3d",
            native_image_save=True,
            allow_fullscreen=True,
        )
        self.preview.setObjectName("analysisRegionViewerPreview")
        self.preview.setMinimumSize(420, 360)
        self.scene_tools = SceneViewControls(self.preview.plot, viewer_panel)
        viewer_layout.addWidget(self.scene_tools)
        viewer_layout.addWidget(self.preview, 1)
        self.render_note = QLabel("", viewer_panel)
        self.render_note.setObjectName("hint")
        self.render_note.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        viewer_layout.addWidget(self.render_note)

        self.splitter.addWidget(browser_panel)
        self.splitter.addWidget(viewer_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setSizes([280, 760])
        root.addWidget(self.splitter, 1)

        self._set_controls_enabled(False)
        self.preview.show_empty("No region set", "Open a target from Exposure Analysis.")

    @property
    def scene(self) -> AnalysisRegionScene | None:
        return self._scene

    @property
    def target_id(self) -> str | None:
        return self._scene.target_id if self._scene is not None else None

    @property
    def selected_region_key(self) -> str | None:
        item = self.region_tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def set_scene(
        self,
        scene: AnalysisRegionScene,
        *,
        selected_region_key: str | None = None,
    ) -> None:
        if not isinstance(scene, AnalysisRegionScene):
            raise AnalysisRegionSceneError("AnalysisRegionViewer requires an AnalysisRegionScene.")
        self._scene = scene
        self._scene_identity = analysis_region_scene_identity(scene)
        self._metrics_by_key.clear()
        self.setWindowTitle(f"Region viewer — {scene.target_name}")
        self.title_label.setText(scene.target_name)
        self.preview.set_scene(scene)
        self._populate_tree(scene)
        key = str(selected_region_key or "").strip()
        if key not in self._items_by_key:
            key = scene.hierarchy.key
        self._set_controls_enabled(True)
        self.region_tree.setCurrentItem(self._items_by_key[key])
        plan = self.preview.render_plan
        if plan is not None:
            self.render_note.setText(
                f"Display mesh: {plan.rendered_triangle_count:,} of "
                f"{plan.source_triangle_count:,} triangles"
                if plan.rendered_triangle_count != plan.source_triangle_count
                else f"Display mesh: {plan.source_triangle_count:,} triangles"
            )

    def present(
        self,
        scene: AnalysisRegionScene,
        *,
        selected_region_key: str | None = None,
        context_note: str | None = None,
    ) -> None:
        """Show/reuse this viewer for ``scene`` without spawning another window.

        Reopening the same unchanged target preserves the live WebGL surface and
        camera.  A changed mesh/transform/region definition replaces the scene.
        """
        if not isinstance(scene, AnalysisRegionScene):
            raise AnalysisRegionSceneError("AnalysisRegionViewer requires an AnalysisRegionScene.")
        same_scene = self._scene is scene
        if not same_scene and self._scene is not None and self._scene.target_id == scene.target_id:
            identity = analysis_region_scene_identity(scene)
            same_scene = (
                identity == self._scene_identity
                and scene.target_name == self._scene.target_name
                and scene.base_colour == self._scene.base_colour
                and scene.base_opacity == self._scene.base_opacity
                and scene.label_colours == self._scene.label_colours
            )
        if same_scene:
            self.set_selected_region(selected_region_key)
        else:
            self.set_scene(scene, selected_region_key=selected_region_key)
        note = str(context_note or "").strip()
        self.context_note.setText(note)
        self.context_note.setVisible(bool(note))
        self.show()
        self.raise_()
        self.activateWindow()

    def set_selected_region(self, region_key: str | None) -> None:
        """Select/highlight one region without changing scene geometry or camera."""
        if self._scene is None:
            return
        key = str(region_key or "").strip() or self._scene.hierarchy.key
        item = self._items_by_key.get(key)
        if item is None:
            key = self._scene.hierarchy.key
            item = self._items_by_key.get(key)
        if item is None:
            return
        if self.region_tree.currentItem() is item:
            # The analysis may have reset to the same key after the preview was
            # refreshed.  Reapply the highlight without rebuilding geometry.
            self.preview.set_selected_region(key)
            return
        self.region_tree.setCurrentItem(item)
        self.region_tree.scrollToItem(item)

    def show_unavailable(self, target_name: str, detail: str) -> None:
        """Replace stale live geometry with an explicit unavailable state."""
        self._scene = None
        self._scene_identity = ""
        self._metrics_by_key.clear()
        self._items_by_key.clear()
        self.region_tree.clear()
        self.setWindowTitle(f"Region viewer — {str(target_name or 'target')}")
        self.title_label.setText(str(target_name or "Region viewer"))
        self.selection_title.setText("Live target unavailable")
        self.selection_details.setText(str(detail or "The target is no longer available in the current model."))
        self.context_note.setText("The saved numerical analysis remains readable; only the live display geometry is unavailable.")
        self.context_note.show()
        self.render_note.clear()
        self._set_controls_enabled(False)
        self.preview.show_empty("3D target unavailable", str(detail))

    def _populate_tree(self, scene: AnalysisRegionScene) -> None:
        self.region_tree.clear()
        self._items_by_key.clear()

        root_item = QTreeWidgetItem([scene.target_name])
        root_item.setData(0, Qt.ItemDataRole.UserRole, scene.hierarchy.key)
        root_item.setToolTip(0, scene.hierarchy.name)
        self.region_tree.addTopLevelItem(root_item)
        self._items_by_key[scene.hierarchy.key] = root_item

        def add_children(parent_item: QTreeWidgetItem, node: AnalysisRegionNode) -> None:
            for child in node.children:
                item = QTreeWidgetItem([str(child.name)])
                item.setData(0, Qt.ItemDataRole.UserRole, str(child.key))
                parent_item.addChild(item)
                self._items_by_key[str(child.key)] = item
                add_children(item, child)

        add_children(root_item, scene.hierarchy)
        root_item.setExpanded(True)
        for index in range(root_item.childCount()):
            root_item.child(index).setExpanded(True)
        self.region_tree.resizeColumnToContents(0)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.region_tree.setEnabled(bool(enabled))
        self.scene_tools.setEnabled(bool(enabled))

    def _tree_selection_changed(self, item, _previous) -> None:
        if self._scene is None or item is None:
            return
        key = str(item.data(0, Qt.ItemDataRole.UserRole) or self._scene.hierarchy.key)
        try:
            node = self._scene.node(key)
            self.preview.set_selected_region(key)
        except AnalysisRegionSceneError:
            return
        self.selection_title.setText(node.name if key != self._scene.hierarchy.key else self._scene.target_name)
        self.selection_details.setText(self._selection_details_text(key, node))

    def _selection_details_text(self, key: str, node: AnalysisRegionNode) -> str:
        if self._scene is None:
            return ""
        metrics = self._metrics_by_key.get(key)
        if metrics is None:
            metrics = analysis_region_node_display_metrics(self._scene, key)
            self._metrics_by_key[key] = metrics

        lines = []
        kind_name = _REGION_KIND_NAMES.get(str(node.region_kind), str(node.region_kind).replace("_", " ").title())
        lines.append(f"Type: {kind_name}")
        label_count = int(metrics.get("atomic_label_count", 0) or 0)
        if label_count > 1:
            lines.append(f"Atomic regions: {label_count:,}")
        lines.append(f"Display triangles: {int(metrics['triangle_count']):,}")
        lines.append(f"Display surface area: {float(metrics['surface_area_cm2']):,.6g} cm²")
        size_mm = metrics.get("size_mm", (0.0, 0.0, 0.0))
        lines.append(
            "Display bounds: " + " × ".join(f"{float(value):,.6g}" for value in size_mm) + " mm"
        )

        metadata = self._scene.metadata if isinstance(self._scene.metadata, dict) else {}
        if self._scene.target_kind == "surface_region_volume":
            try:
                lines.append(f"Sampling depth: {float(metadata.get('sampling_depth_mm')):,.6g} mm inward")
            except (TypeError, ValueError):
                pass
        atlas = str(metadata.get("atlas") or "").strip()
        if atlas:
            lines.append(f"Atlas: {atlas}")
        return "\n".join(lines)

    def apply_workbench_theme(self, theme: str) -> None:
        self.preview.apply_workbench_theme(theme)

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self.preview.cleanup()

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        # Keep the live renderer for reuse during this Exposure Analysis session.
        # The logical owner calls cleanup() when the whole workspace is closed.
        event.accept()
        super().closeEvent(event)
        if not self._cleaned_up:
            # Emit after Qt has actually hidden the top-level so the owner can
            # safely decide which embedded preview should reclaim WebGL.
            QTimer.singleShot(0, self.viewer_hidden.emit)
