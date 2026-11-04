"""Source contracts for shared 3-D viewer controls."""
from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SceneViewControlsSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(
            encoding="utf-8"
        )
        cls.controls = (PROJECT_ROOT / "fieldworkbench" / "scene_view_controls.py").read_text(
            encoding="utf-8"
        )
        cls.main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        cls.preview = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_preview.py").read_text(
            encoding="utf-8"
        )
        cls.region_viewer = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_viewer.py").read_text(
            encoding="utf-8"
        )

    def test_plotview_exposes_named_native_modebar_profiles(self):
        self.assertIn('"standard": ()', self.plot)
        self.assertIn('"rotation": ("zoom3d", "pan3d", "hoverClosest3d", "toImage")', self.plot)
        self.assertIn('"full_3d": ()', self.plot)
        self.assertIn('modebar_profile: str = "standard"', self.plot)
        self.assertIn('native_image_save: bool = True', self.plot)
        self.assertIn('self.MODEBAR_PROFILES[self._modebar_profile]', self.plot)
        self.assertIn('self._bridge is not None and self._native_image_save', self.plot)
        self.assertIn('if self._modebar_profile == "rotation":', self.plot)
        self.assertIn('modebar_buttons_to_remove.remove("resetCameraDefault3d")', self.plot)

    def test_fullscreen_clone_preserves_modebar_and_interaction_contract(self):
        self.assertIn('modebar_profile=self._modebar_profile', self.plot)
        self.assertIn('native_image_save=self._native_image_save', self.plot)
        self.assertIn('display_modebar=self._display_modebar', self.plot)
        self.assertIn('enable_selection_events=self._enable_selection_events', self.plot)
        self.assertIn('publish_live_view=self._publish_live_view', self.plot)
        self.assertIn('plot_gl_pixel_ratio=self._plot_gl_pixel_ratio', self.plot)
        self.assertIn('default_3d_dragmode=self._default_3d_dragmode', self.plot)

    def test_scene_view_command_bar_is_reusable_and_model_editor_uses_it(self):
        self.assertIn('class SceneViewControls(QFrame)', self.controls)
        for text in ('Export image…', 'Home', 'Fullscreen', 'Front', 'Right', 'Top'):
            self.assertIn(text, self.controls)
        self.assertIn('self.plot.set_camera_preset(view_name)', self.controls)
        self.assertIn('self.plot.toggle_fullscreen', self.controls)
        self.assertIn('from .scene_view_controls import SceneViewControls', self.main)
        self.assertIn('self.scene_tools = SceneViewControls(', self.main)
        self.assertIn('export_callback=self.export_scene_image', self.main)
        self.assertIn('home_callback=self.home_scene_view', self.main)
        self.assertIn('self.view_preset_buttons = self.scene_tools.view_preset_buttons', self.main)

    def test_exposure_analysis_applies_native_profiles_and_shared_region_toolbar(self):
        self.assertIn('("full_3d" if self._regional_display else "rotation")', self.preview)
        self.assertIn('native_image_save=resolved_native_image_save', self.preview)
        self.assertIn('allow_fullscreen=resolved_allow_fullscreen', self.preview)
        self.assertIn('default_3d_dragmode="turntable"', self.preview)
        self.assertIn('from .scene_view_controls import SceneViewControls', self.region_viewer)
        self.assertIn('modebar_profile="full_3d"', self.region_viewer)
        self.assertIn('self.scene_tools = SceneViewControls(self.preview.plot, viewer_panel)', self.region_viewer)
        self.assertNotIn('QPushButton("Fit object"', self.region_viewer)
        self.assertNotIn('QPushButton("Fit region"', self.region_viewer)
        self.assertNotIn('itemDoubleClicked.connect', self.region_viewer)

    def test_shared_scene_view_export_has_plotview_default(self):
        self.assertIn('def save_png_with_dialog(self) -> None:', self.plot)
        self.assertIn('self.export_image_button.clicked.connect(self.plot.save_png_with_dialog)', self.controls)


if __name__ == "__main__":
    unittest.main()
