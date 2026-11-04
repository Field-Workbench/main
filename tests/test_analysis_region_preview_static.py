"""Source-contract checks for the reusable Exposure Analysis 3D preview."""
from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnalysisRegionPreviewSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preview = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_preview.py").read_text(
            encoding="utf-8"
        )
        cls.header = (PROJECT_ROOT / "fieldworkbench" / "exposure_analysis_header.py").read_text(
            encoding="utf-8"
        )
        cls.render = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_render.py").read_text(
            encoding="utf-8"
        )
        cls.plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(
            encoding="utf-8"
        )
        cls.workspace = (PROJECT_ROOT / "fieldworkbench" / "exposure_workspace.py").read_text(
            encoding="utf-8"
        )
        cls.actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(
            encoding="utf-8"
        )
        cls.adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(
            encoding="utf-8"
        )

    def test_preview_owns_one_plotview_and_only_emits_region_viewer_request(self):
        self.assertIn("class AnalysisRegionPreview(QWidget)", self.preview)
        self.assertIn("region_viewer_requested = Signal(str)", self.preview)
        self.assertIn('("full_3d" if self._regional_display else "rotation")', self.preview)
        self.assertIn("display_modebar=bool(display_modebar)", self.preview)
        self.assertIn("allow_fullscreen=resolved_allow_fullscreen", self.preview)
        self.assertIn("native_image_save=resolved_native_image_save", self.preview)
        self.assertIn("enable_selection_events=False", self.preview)
        self.assertIn("publish_live_view=False", self.preview)
        self.assertIn("plot_gl_pixel_ratio=1.0", self.preview)
        self.assertIn("Open region viewer…", self.preview)
        self.assertNotIn("QDialog", self.preview)

    def test_compact_preview_uses_scene_renderer_while_regional_mode_keeps_highlighting(self):
        self.assertIn("object_figure: dict | None = None", self.preview)
        self.assertIn("if self._object_figure is not None:", self.preview)
        self.assertNotIn("build_analysis_object_render_plan", self.preview)
        self.assertIn("Compact analysis previews require the canonical Model Editor object figure.", self.preview)
        self.assertIn("build_analysis_region_render_plan", self.preview)
        self.assertIn("regional_display: bool = False", self.preview)
        self.assertIn("if self._regional_display:", self.preview)
        self.assertIn("analysis_region_trace_styles", self.preview)
        self.assertIn("apply_incremental_update(trace_updates=updates)", self.preview)
        self.assertNotIn("DEFAULT_PREVIEW_TRIANGLE_BUDGET", self.render)
        self.assertNotIn("DEFAULT_REGION_VIEWER_TRIANGLE_BUDGET", self.render)
        self.assertNotIn("_weld_anatomical_object_surface", self.render)
        self.assertIn('"field_workbench_hide_3d_axes": True', self.render)


    def test_create_analysis_chooser_uses_shared_target_selection_wording_and_identity(self):
        self.assertIn('title="Exposure target selection"', self.workspace)
        self.assertIn(
            '"Choose a target for the analysis. After the base is prepared, various statistical "',
            self.workspace,
        )
        self.assertIn(
            '"calculations can be run on the target. Calculations are automatically cached so they "',
            self.workspace,
        )
        self.assertIn("self.target_title = self.analysis_header.target_label", self.workspace)
        self.assertIn("self.exposure_title = self.analysis_header.exposure_label", self.workspace)
        self.assertIn('self.target_label.setObjectName("dialogHeading")', self.header)
        self.assertIn('self.exposure_label.setObjectName("dialogHeading")', self.header)
        self.assertNotIn("Create exposure analysis", self.workspace)

    def test_chooser_header_avoids_duplicate_surface_depth_and_saved_analysis_summary(self):
        start = self.workspace.index("def _surface_region_details")
        end = self.workspace.index("\n    def _brain_details", start)
        helper = self.workspace[start:end]
        self.assertNotIn("Sampling depth:", helper)
        self.assertNotIn("surface_depth_note", self.workspace)
        self.assertNotIn("saved_analysis_status", self.workspace)
        self.assertIn("surface_depth_layout.addStretch(1)", self.workspace)

    def test_create_analysis_chooser_embeds_and_refreshes_target_preview(self):
        self.assertIn("from .exposure_analysis_header import ExposureAnalysisHeaderPanel", self.workspace)
        self.assertIn('title="Exposure target selection"', self.workspace)
        self.assertIn('preview_object_name="exposureAnalysisTargetPreview"', self.workspace)
        self.assertIn("self.target_preview = self.analysis_header.preview", self.workspace)
        self.assertIn("def _queue_target_preview(self, target_id):", self.workspace)
        self.assertIn("QTimer.singleShot(0, lambda: self._resolve_target_preview(target_id, serial))", self.workspace)
        self.assertIn("scene = self.adapter.analysis_region_scene(str(target_id))", self.workspace)
        self.assertIn("object_figure = self.adapter.analysis_target_preview_figure(str(target_id))", self.workspace)
        self.assertIn("self.target_preview.set_scene(scene, object_figure=object_figure)", self.workspace)
        self.assertIn("preview.cleanup()", self.workspace)
        self.assertIn('setObjectName("exposureAnalysisChooserVerticalSplitter")', self.workspace)
        self.assertIn("chooser_splitter.setSizes([235, 465])", self.workspace)
        self.assertIn("split.setSizes([340, 680])", self.workspace)
        self.assertIn('setObjectName("exposureAnalysisOpenRegionViewerButton")', self.header)
        self.assertIn("self.splitter.setSizes([680, 340])", self.header)

    def test_compact_preview_reuses_canonical_model_editor_mesh_trace_builder(self):
        self.assertIn("def _geometry_mesh_traces(", self.adapter)
        self.assertIn("traces.extend(self._geometry_mesh_traces(mesh, respect_visibility=True))", self.adapter)
        self.assertIn("def analysis_target_preview_figure(self, object_id: str)", self.adapter)
        self.assertIn("self._geometry_mesh_traces(mesh, respect_visibility=False)", self.adapter)
        preview_start = self.adapter.index("def analysis_target_preview_figure")
        preview_end = self.adapter.index("def surface_region_volume_set", preview_start)
        helper = self.adapter[preview_start:preview_end]
        self.assertIn('"field_workbench_hide_3d_axes": True', helper)
        self.assertIn('trace["showlegend"] = False', helper)
        self.assertNotIn("_weld_anatomical_object_surface", helper)
        self.assertIn("traces = self._geometry_mesh_traces(mesh, respect_visibility=False)", helper)
        mesh_helper_start = self.adapter.index("def _geometry_mesh_traces")
        mesh_helper_end = self.adapter.index("def _geometry_traces", mesh_helper_start)
        mesh_helper = self.adapter[mesh_helper_start:mesh_helper_end]
        self.assertIn('"flatshading": True', mesh_helper)
        self.assertIn('"color": mesh.get("colour", self._role_colour(role))', mesh_helper)
        self.assertIn('"opacity": float(mesh.get("opacity", 0.28))', mesh_helper)


    def test_shared_header_template_owns_common_identity_preview_and_navigation(self):
        self.assertIn("class ExposureAnalysisHeaderPanel(QWidget)", self.header)
        self.assertIn('self.target_label.setObjectName("dialogHeading")', self.header)
        self.assertIn('self.exposure_label.setObjectName("dialogHeading")', self.header)
        self.assertIn('self.preview = AnalysisRegionPreview(', self.header)
        self.assertIn('show_region_viewer_button=False', self.header)
        self.assertIn('self.targets_button = QPushButton("← Analysis targets"', self.header)
        self.assertIn('self.home_button = QPushButton("← Analysis home"', self.header)
        self.assertIn('self.analysis_info_button = QPushButton("Hide analysis info"', self.header)
        self.assertIn('self.analysis_info_button.setObjectName("exposureAnalysisToggleInfoButton")', self.header)
        self.assertIn('self.navigation_layout.addWidget(self.analysis_info_button, 0)', self.header)
        self.assertIn('self.navigation_layout.addWidget(self.region_viewer_button, 0)', self.header)
        self.assertNotIn('self.viewer_controls = QWidget', self.header)
        self.assertIn("def set_analysis_info_visible(self, visible: bool) -> None:", self.header)
        self.assertIn('self.analysis_info_button.setText("Show analysis info")', self.header)
        self.assertIn("self.preview_panel.hide()", self.header)
        self.assertIn("self.heading_label.show()", self.header)
        self.assertIn("self.region_viewer_button.show()", self.header)
        self.assertIn("self._apply_preview_activity()", self.header)
        self.assertIn("self.splitter.setStretchFactor(0, 2)", self.header)
        self.assertIn("self.splitter.setStretchFactor(1, 1)", self.header)
        self.assertIn("self.analysis_header = ExposureAnalysisHeaderPanel(", self.workspace)
        self.assertIn("panel = ExposureAnalysisHeaderPanel(", self.actions)

    def test_create_analysis_preview_is_visual_only_and_failure_tolerant(self):
        start = self.workspace.index("def _resolve_target_preview")
        end = self.workspace.index("\n    def _refresh_target_info", start)
        helper = self.workspace[start:end]
        self.assertIn("except Exception as error", helper)
        self.assertIn('"3D preview unavailable"', helper)
        self.assertNotIn("prepare_exposure_analysis_payload", helper)
        self.assertNotIn("surface_region_volume_set", helper)


    def test_prepared_analysis_header_reuses_one_persistent_target_preview(self):
        self.assertIn("from .exposure_analysis_header import ExposureAnalysisHeaderPanel", self.actions)
        self.assertIn('title=self.page_titles["home"]', self.actions)
        self.assertIn('preview_object_name="exposureAnalysisPreparedTargetPreview"', self.actions)
        self.assertIn("self.analysis_target_preview = panel.preview", self.actions)
        self.assertIn("self._queue_analysis_target_preview()", self.actions)
        self.assertIn("QTimer.singleShot(0, lambda: self._resolve_analysis_target_preview(target_id, serial))", self.actions)
        self.assertIn("scene = self.project_adapter.analysis_region_scene(str(target_id))", self.actions)
        self.assertIn("object_figure = self.project_adapter.analysis_target_preview_figure(str(target_id))", self.actions)
        self.assertIn("preview.set_scene(scene, object_figure=object_figure)", self.actions)
        self.assertIn('setObjectName("exposureAnalysisPreparedVerticalSplitter")', self.actions)
        self.assertIn("self.analysis_page_splitter.setSizes([235, 465])", self.actions)

    def test_prepared_header_hides_routine_live_geometry_note_but_keeps_stale_warning(self):
        start = self.actions.index("def _build_preparation_overview")
        end = self.actions.index("\n    def set_visual_active", start)
        helper = self.actions[start:end]
        self.assertNotIn("Live model geometry for this current preparation.", helper)
        self.assertNotIn("prepared numerical samples remain frozen separately", helper)
        self.assertIn("saved preparation is stale", helper)

    def test_prepared_analysis_preview_is_visual_only_failure_tolerant_and_cleaned_up(self):
        start = self.actions.index("def _resolve_analysis_target_preview")
        end = self.actions.index("\n    def _build_home_page", start)
        helper = self.actions[start:end]
        self.assertIn("except Exception as error", helper)
        self.assertIn('"3D target preview unavailable"', helper)
        self.assertNotIn("prepare_exposure_analysis_payload", helper)
        self.assertNotIn("surface_region_volume_set", helper)
        self.assertIn('preview = getattr(self, "analysis_target_preview", None)', self.actions)
        self.assertIn("preview.cleanup()", self.actions)

    def test_region_preview_releases_inactive_webgl_and_disables_picking(self):
        self.assertIn("def set_rendering_active(\n", self.preview)
        self.assertIn("inactive_detail: str | None = None", self.preview)
        self.assertIn("Preview is disabled while the Region Viewer is active.", self.workspace)
        self.assertIn("header.set_preview_rendering_active(", self.workspace)
        self.assertIn("panel.set_preview_rendering_active(", self.actions)
        self.assertIn("release_rendered_figure", self.preview)
        self.assertIn('"hoverinfo": "skip"', self.render)
        self.assertIn('"hovermode": False', self.render)
        self.assertIn('"field_workbench_ignore_picking": True', self.render)
        self.assertIn("def release_rendered_figure(", self.plot)
        self.assertIn('plot_config["plotGlPixelRatio"]', self.plot)

    def test_plotview_modebar_profiles_keep_compact_controls_native(self):
        self.assertIn("display_modebar: bool | None = None", self.plot)
        self.assertIn('plot_config["displayModeBar"] = bool(self._display_modebar)', self.plot)
        self.assertIn('"rotation": ("zoom3d", "pan3d", "hoverClosest3d", "toImage")', self.plot)
        self.assertIn('"full_3d": ()', self.plot)
        self.assertIn('modebar_buttons_to_remove.remove("resetCameraDefault3d")', self.plot)
        self.assertIn('default_3d_dragmode="turntable"', self.preview)


if __name__ == "__main__":
    unittest.main()
