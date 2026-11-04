"""Reusable compact 3D preview for Exposure Analysis region targets."""
from __future__ import annotations

import copy

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .analysis_region_render import (
    AnalysisRegionRenderPlan,
    analysis_region_trace_styles,
    build_analysis_region_render_plan,
)
from .analysis_region_scene import (
    AnalysisRegionScene,
    AnalysisRegionSceneError,
    analysis_region_scene_identity,
)
from .plot_view import PlotView


class AnalysisRegionPreview(QWidget):
    """Small whole-target 3D view shared by Exposure Analysis pages.

    The widget owns one PlotView for its lifetime. Compact instances display the
    canonical target-only figure supplied by the Model Editor renderer; the
    modeless Region Viewer opts into regional display and uses scalar Plotly
    restyles for highlighting rather than retransmitting geometry. The button
    only emits a viewer-open request.
    """

    region_viewer_requested = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        triangle_budget: int | None = None,
        show_region_viewer_button: bool = True,
        regional_display: bool = False,
        display_modebar: bool = True,
        modebar_profile: str | None = None,
        native_image_save: bool | None = None,
        allow_fullscreen: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self._triangle_budget = (
            None if triangle_budget is None else max(1, int(triangle_budget))
        )
        self._regional_display = bool(regional_display)
        self._scene: AnalysisRegionScene | None = None
        self._object_figure: dict | None = None
        self._render_plan: AnalysisRegionRenderPlan | None = None
        self._selected_region_key: str | None = None
        self._rendering_active = True
        self._inactive_detail = "Preview is temporarily disabled while this view is inactive."

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        resolved_modebar_profile = str(
            modebar_profile or ("full_3d" if self._regional_display else "rotation")
        )
        resolved_native_image_save = (
            self._regional_display if native_image_save is None else bool(native_image_save)
        )
        resolved_allow_fullscreen = (
            self._regional_display if allow_fullscreen is None else bool(allow_fullscreen)
        )
        self.plot = PlotView(
            self,
            allow_fullscreen=resolved_allow_fullscreen,
            display_modebar=bool(display_modebar),
            modebar_profile=resolved_modebar_profile,
            default_3d_dragmode="turntable",
            native_image_save=resolved_native_image_save,
            enable_selection_events=False,
            publish_live_view=False,
            plot_gl_pixel_ratio=1.0,
            image_filename=(
                "analysis-region-viewer"
                if self._regional_display
                else "analysis-region-preview"
            ),
        )
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot.setMinimumSize(180, 130)
        layout.addWidget(self.plot, 1)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        self.open_region_viewer_button = QPushButton("Open region viewer…")
        self.open_region_viewer_button.setToolTip(
            "Open a modeless 3D viewer for browsing and highlighting this target's regions."
        )
        self.open_region_viewer_button.clicked.connect(self._request_region_viewer)
        self.open_region_viewer_button.setVisible(bool(show_region_viewer_button))
        self.open_region_viewer_button.setEnabled(False)
        button_row.addWidget(self.open_region_viewer_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(button_row)

        self.show_empty("Select an analysis target", "A 3D preview will appear here.")

    @property
    def scene(self) -> AnalysisRegionScene | None:
        return self._scene

    @property
    def selected_region_key(self) -> str | None:
        return self._selected_region_key

    @property
    def render_plan(self) -> AnalysisRegionRenderPlan | None:
        return self._render_plan

    def set_region_viewer_button_visible(self, visible: bool) -> None:
        self.open_region_viewer_button.setVisible(bool(visible))

    def show_loading(self, detail: str = "Preparing target geometry…") -> None:
        self._scene = None
        self._object_figure = None
        self._render_plan = None
        self._selected_region_key = None
        self.open_region_viewer_button.setEnabled(False)
        self.plot.show_message("Preparing 3D preview", str(detail))

    def show_empty(self, heading: str = "No preview", detail: str = "") -> None:
        self._scene = None
        self._object_figure = None
        self._render_plan = None
        self._selected_region_key = None
        self.open_region_viewer_button.setEnabled(False)
        self.plot.show_message(str(heading), str(detail))

    def set_scene(
        self,
        scene: AnalysisRegionScene,
        *,
        selected_region_key: str | None = None,
        object_figure: dict | None = None,
    ) -> None:
        if not isinstance(scene, AnalysisRegionScene):
            raise AnalysisRegionSceneError("AnalysisRegionPreview requires an AnalysisRegionScene.")
        self._scene = scene
        self._object_figure = copy.deepcopy(object_figure) if isinstance(object_figure, dict) else None
        self._selected_region_key = str(selected_region_key) if selected_region_key else None
        self.open_region_viewer_button.setEnabled(True)
        if not self._rendering_active:
            self._render_plan = None
            self.plot.release_rendered_figure(
                "3D preview paused",
                self._inactive_detail,
            )
            return
        self._render_current_scene()

    @property
    def rendering_active(self) -> bool:
        return bool(self._rendering_active)

    def set_rendering_active(
        self,
        active: bool,
        *,
        inactive_detail: str | None = None,
    ) -> None:
        """Pause hidden 3D previews and restore them only when needed."""
        active = bool(active)
        if inactive_detail is not None:
            detail = str(inactive_detail).strip()
            if detail:
                self._inactive_detail = detail
        if active == self._rendering_active:
            return
        self._rendering_active = active
        if not active:
            self._render_plan = None
            self.plot.release_rendered_figure(
                "3D preview paused",
                self._inactive_detail,
            )
            return
        if self._scene is not None:
            self._render_current_scene()

    def _render_current_scene(self) -> None:
        if self._scene is None or not self._rendering_active:
            return
        if self._regional_display:
            plan = build_analysis_region_render_plan(
                self._scene,
                triangle_budget=self._triangle_budget,
                selected_region_key=self._selected_region_key,
            )
        else:
            if self._object_figure is not None:
                # Compact previews use the exact object trace produced by the
                # Model Editor scene renderer.  Exposure Analysis only supplies
                # the clean target-only layout; it does not reinterpret the mesh.
                triangle_count = int(len(self._scene.object_faces))
                plan = AnalysisRegionRenderPlan(
                    figure=copy.deepcopy(self._object_figure),
                    trace_index_by_label={},
                    rendered_triangle_count=triangle_count,
                    source_triangle_count=triangle_count,
                    scene_identity=analysis_region_scene_identity(self._scene),
                    triangle_budget=None,
                )
            else:
                raise AnalysisRegionSceneError(
                    "Compact analysis previews require the canonical Model Editor object figure."
                )
        self._render_plan = plan
        self.plot.request_home_on_next_figure()
        self.plot.set_figure(plan.figure)

    def set_selected_region(self, region_key: str | None) -> None:
        if self._scene is None:
            return
        key = str(region_key or "").strip() or None
        if key is not None:
            self._scene.node(key)  # validate before mutating display state
        self._selected_region_key = key
        if not self._regional_display:
            # Compact previews intentionally remain an unsegmented whole-object
            # context view. Region highlighting belongs to Region Viewer.
            return
        if not self._rendering_active or self._render_plan is None:
            return
        updates = analysis_region_trace_styles(
            self._scene,
            self._render_plan.trace_index_by_label,
            key,
        )
        self.plot.apply_incremental_update(trace_updates=updates)

    def apply_workbench_theme(self, theme: str) -> None:
        self.plot.apply_workbench_theme(theme)

    def _request_region_viewer(self) -> None:
        if self._scene is not None:
            self.region_viewer_requested.emit(self._scene.target_id)

    def cleanup(self) -> None:
        self.plot.cleanup()

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        self.cleanup()
        super().closeEvent(event)
