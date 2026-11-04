"""Dependency-light source-contract checks for Phase 6 surface-exposure UI."""
from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SurfaceExposureUiSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        cls.workspace = (PROJECT_ROOT / "fieldworkbench" / "exposure_workspace.py").read_text(
            encoding="utf-8"
        )
        cls.actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(
            encoding="utf-8"
        )
        cls.header = (PROJECT_ROOT / "fieldworkbench" / "exposure_analysis_header.py").read_text(
            encoding="utf-8"
        )
        cls.help = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(
            encoding="utf-8"
        )

    def test_model_inspector_exposes_surface_sampling_depth(self):
        self.assertIn('setObjectName("surfaceRegionAnalysisDepthSpin")', self.main_window)
        self.assertIn(
            'setRange(MINIMUM_SURFACE_DEPTH_MM, MAXIMUM_SURFACE_DEPTH_MM)',
            self.main_window,
        )
        self.assertIn('"method": "set_surface_region_analysis_settings"', self.main_window)
        self.assertIn("geometric field-sampling layer", self.main_window)

    def test_analysis_chooser_groups_anatomy_and_edits_depth(self):
        self.assertIn('"Region sets"', self.workspace)
        self.assertIn('"Measurement volumes"', self.workspace)
        self.assertIn('setObjectName("exposureSurfaceSamplingDepthSpin")', self.workspace)
        self.assertIn(
            'setRange(MINIMUM_SURFACE_DEPTH_MM, MAXIMUM_SURFACE_DEPTH_MM)',
            self.workspace,
        )
        self.assertIn('def _apply_surface_depth', self.workspace)
        self.assertIn('"method": "set_surface_region_analysis_settings"', self.workspace)

    def test_open_analysis_can_return_to_target_chooser(self):
        self.assertIn("on_back_to_targets=self._show_objects", self.workspace)
        # The navigation button is owned by the shared Exposure Analysis header;
        # the prepared-analysis view still controls its busy-state enablement.
        self.assertIn('self.targets_button = QPushButton("← Analysis targets"', self.header)
        self.assertIn("self.targets_button.setEnabled(not busy)", self.actions)

    def test_surface_region_browser_exposes_preparation_geometry_metadata(self):
        self.assertIn('"Sampled volume:', self.actions)
        self.assertIn('"Surface area:', self.actions)
        self.assertIn('"Source voxels:', self.actions)
        self.assertIn('"Retained field samples:', self.actions)
        self.assertIn('cm³ sampled volume', self.actions)

    def test_help_distinguishes_geometric_sampling_from_tissue_or_dose(self):
        self.assertIn("geometric field-sampling setting", self.help)
        self.assertIn("not an assertion about", self.help)
        self.assertIn("← Analysis targets", self.help)


if __name__ == "__main__":
    unittest.main()
