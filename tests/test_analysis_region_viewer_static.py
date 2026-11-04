"""Source-contract checks for the modeless Exposure Analysis Region Viewer."""
from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnalysisRegionViewerSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viewer = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_viewer.py").read_text(encoding="utf-8")
        cls.preview = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_preview.py").read_text(encoding="utf-8")
        cls.plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(encoding="utf-8")
        cls.workspace = (PROJECT_ROOT / "fieldworkbench" / "exposure_workspace.py").read_text(encoding="utf-8")
        cls.actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        cls.region_render = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_render.py").read_text(encoding="utf-8")
        cls.brain_view = (PROJECT_ROOT / "fieldworkbench" / "brain_view.py").read_text(encoding="utf-8")
        cls.lpba_display = (PROJECT_ROOT / "fieldworkbench" / "lpba40_display.py").read_text(encoding="utf-8")

    def test_viewer_is_modeless_tree_plus_shared_large_preview(self):
        self.assertIn("class AnalysisRegionViewer(QDialog)", self.viewer)
        self.assertIn("Qt.WindowModality.NonModal", self.viewer)
        self.assertIn('QLabel("Object / regions"', self.viewer)
        self.assertIn("self.region_tree = QTreeWidget", self.viewer)
        self.assertIn("triangle_budget=None", self.viewer)
        self.assertNotIn("DEFAULT_REGION_VIEWER_TRIANGLE_BUDGET", self.viewer)
        self.assertIn("show_region_viewer_button=False", self.viewer)
        self.assertIn("regional_display=True", self.viewer)
        self.assertIn('modebar_profile="full_3d"', self.viewer)
        self.assertIn("self.scene_tools = SceneViewControls(self.preview.plot, viewer_panel)", self.viewer)
        self.assertNotIn('QPushButton("Fit object"', self.viewer)
        self.assertNotIn('QPushButton("Fit region"', self.viewer)

    def test_compact_preview_is_unsegmented_but_region_viewer_opts_into_regions(self):
        self.assertNotIn("build_analysis_object_render_plan", self.preview)
        self.assertIn("if self._regional_display:", self.preview)
        self.assertIn("regional_display=True", self.viewer)
        self.assertIn("field_workbench_hide_3d_axes", self.plot)
        self.assertIn("fieldWorkbenchApply3dAxisPolicy", self.plot)

    def test_tree_selection_highlights_shared_scene_regions_without_rebuilding_geometry(self):
        self.assertIn("self.preview.set_selected_region(key)", self.viewer)
        self.assertIn("analysis_region_node_display_metrics", self.viewer)
        self.assertNotIn("fit_region", self.viewer)
        self.assertNotIn("def fit_region(self, region_key: str | None = None) -> bool:", self.preview)
        self.assertNotIn("def fit_3d_traces(self, trace_indices) -> bool:", self.plot)

    def test_lpba40_region_viewer_reuses_brain_selector_trace_builder_without_face_sampling(self):
        self.assertIn("build_lpba40_region_trace", self.brain_view)
        self.assertIn("lpba40_region_style", self.brain_view)
        self.assertIn("build_lpba40_region_trace", self.region_render)
        self.assertIn("_build_lpba40_region_render_plan", self.region_render)
        brain_branch = self.region_render[
            self.region_render.index("def _build_lpba40_region_render_plan"):
            self.region_render.index("def _build_labelled_region_render_plan")
        ]
        self.assertNotIn("_even_face_sample", brain_branch)
        self.assertIn("LPBA40_DIVISION_COLOURS", self.lpba_display)
        self.assertIn("LPBA40_REGION_OPACITY = 0.34", self.lpba_display)
        self.assertIn("LPBA40_SELECTED_REGION_OPACITY = 1.0", self.lpba_display)


    def test_builtin_region_viewers_preserve_complete_topology_without_face_sampling(self):
        main_start = self.region_render.index("def build_analysis_region_render_plan")
        main_body = self.region_render[main_start:]
        self.assertIn('target_kind in {"surface_region_volume", "measurement_volume"}', main_body)
        self.assertIn("triangle_budget=None", main_body)
        self.assertNotIn("DEFAULT_REGION_VIEWER_TRIANGLE_BUDGET", self.region_render)
        self.assertNotIn("DEFAULT_PREVIEW_TRIANGLE_BUDGET", self.region_render)
        self.assertNotIn("_weld_anatomical_object_surface", self.region_render)

    def test_workspace_reuses_one_region_viewer_for_chooser_and_prepared_views(self):
        self.assertIn("from .analysis_region_viewer import AnalysisRegionViewer", self.workspace)
        self.assertIn("self._region_viewer: AnalysisRegionViewer | None = None", self.workspace)
        self.assertIn("self.open_region_viewer_button.clicked.connect(self._open_selected_region_viewer)", self.workspace)
        self.assertIn("def _open_selected_region_viewer(self, *_args):", self.workspace)
        self.assertIn("on_region_viewer_requested=self._open_region_viewer", self.workspace)
        self.assertIn("def _open_region_viewer(self, target_id, selected_region_key=None, context_note=None):", self.workspace)
        self.assertIn("if self._region_viewer is None:", self.workspace)
        self.assertIn("self._region_viewer.present(scene, selected_region_key=selected_region_key)", self.workspace)
        self.assertIn("on_region_selection_changed=self._sync_region_viewer_selection", self.workspace)

    def test_standalone_saved_analysis_owns_one_viewer_and_owner_cleanup_retires_webengine(self):
        self.assertIn("from .analysis_region_viewer import AnalysisRegionViewer", self.actions)
        self.assertIn("on_back_to_targets=None, on_region_viewer_requested=None", self.actions)
        self.assertIn("self._owned_region_viewer: AnalysisRegionViewer | None = None", self.actions)
        self.assertIn("self.analysis_target_preview.region_viewer_requested.connect(self._request_region_viewer)", self.actions)
        self.assertIn("self._owned_region_viewer.present(scene, selected_region_key=selected_key)", self.actions)
        self.assertIn("viewer.cleanup()", self.actions)
        self.assertIn("viewer.cleanup()", self.workspace)

    def test_closing_viewer_hides_for_session_reuse_instead_of_destroying_renderer(self):
        close_start = self.viewer.index("def closeEvent")
        close_body = self.viewer[close_start:]
        self.assertIn("event.accept()", close_body)
        self.assertIn("QTimer.singleShot(0, self.viewer_hidden.emit)", close_body)
        self.assertNotIn("self.cleanup()", close_body)
        self.assertIn("def _set_embedded_preview_activity", self.workspace)
        self.assertIn("viewer_hidden.connect(self._resume_visible_embedded_preview)", self.workspace)


if __name__ == "__main__":
    unittest.main()
