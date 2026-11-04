"""Phase 6 source-contract checks for Region Viewer synchronization and polish."""
from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnalysisRegionViewerPhase6SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viewer = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_viewer.py").read_text(encoding="utf-8")
        cls.workspace = (PROJECT_ROOT / "fieldworkbench" / "exposure_workspace.py").read_text(encoding="utf-8")
        cls.actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        cls.preview = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_preview.py").read_text(encoding="utf-8")
        cls.header = (PROJECT_ROOT / "fieldworkbench" / "exposure_analysis_header.py").read_text(encoding="utf-8")
        cls.help_text = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

    def test_analysis_selection_drives_preview_and_open_viewer_one_way(self):
        self.assertIn("def _sync_analysis_region_visuals(self):", self.actions)
        self.assertIn("preview.set_selected_region(selected_key)", self.actions)
        self.assertIn("viewer.set_selected_region(selected_key)", self.actions)
        self.assertIn("on_region_selection_changed=None", self.actions)
        self.assertIn("on_region_selection_changed=self._sync_region_viewer_selection", self.workspace)
        self.assertIn("def _sync_region_viewer_selection", self.workspace)
        self.assertIn("viewer.set_selected_region(selected_region_key)", self.workspace)
        # Region Viewer deliberately has no callback into Exposure Analysis selection.
        self.assertNotIn("analysis_region_selected = Signal", self.viewer)

    def test_home_returns_visuals_to_whole_object(self):
        self.assertIn('if self.current_page_key == "home":\n            return None', self.actions)
        start = self.actions.index("def show_home")
        body = self.actions[start:self.actions.index("def _cancel_running_analyses", start)]
        self.assertIn("self._sync_analysis_region_visuals()", body)

    def test_same_unchanged_target_reuses_camera_and_renderer(self):
        self.assertIn("analysis_region_scene_identity", self.viewer)
        self.assertIn("if same_scene:", self.viewer)
        self.assertIn("self.set_selected_region(selected_region_key)", self.viewer)
        self.assertIn("else:\n            self.set_scene(scene, selected_region_key=selected_region_key)", self.viewer)

    def test_explicit_open_refreshes_live_geometry_and_deleted_target_is_reported(self):
        self.assertIn("scene = self.adapter.analysis_region_scene(target_id)", self.workspace)
        self.assertIn("viewer.show_unavailable(label, str(error))", self.workspace)
        self.assertIn("scene = self.project_adapter.analysis_region_scene(target_id)", self.actions)
        self.assertIn("show_unavailable(\"Prepared target\", str(error))", self.actions)
        self.assertIn("def show_unavailable", self.viewer)
        self.assertIn("saved numerical analysis remains readable", self.viewer)

    def test_stale_preparation_distinguishes_live_display_from_frozen_results(self):
        self.assertIn("Live model geometry", self.actions)
        self.assertIn("saved preparation is stale", self.actions)
        self.assertIn("context_note", self.viewer)
        self.assertIn("live geometry may no longer exactly match", self.help_text)

    def test_theme_and_accessibility_hooks_remain_on_shared_preview_path(self):
        self.assertIn("def apply_workbench_theme(self, theme: str) -> None:", self.viewer)
        self.assertIn("self.preview.apply_workbench_theme(theme)", self.viewer)
        self.assertIn("def apply_workbench_theme(self, theme: str) -> None:", self.preview)
        self.assertIn("preview_em = max(1, int(self.fontMetrics().height()))", self.header)
        self.assertIn("ExposureAnalysisHeaderPanel", self.actions)
        self.assertIn("ExposureAnalysisHeaderPanel", self.workspace)


if __name__ == "__main__":
    unittest.main()
