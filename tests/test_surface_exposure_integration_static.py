"""Dependency-light source-contract checks for Phase 4 surface Exposure Analysis."""
from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SurfaceExposureIntegrationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = (PROJECT_ROOT / "fieldworkbench" / "exposure_analysis.py").read_text(
            encoding="utf-8"
        )
        cls.adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(
            encoding="utf-8"
        )
        cls.workspace = (PROJECT_ROOT / "fieldworkbench" / "exposure_workspace.py").read_text(
            encoding="utf-8"
        )
        cls.actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(
            encoding="utf-8"
        )

    def test_surface_provider_is_a_first_class_exposure_target(self):
        self.assertIn('"kind": "surface_region_volume"', self.analysis)
        self.assertIn("adapter.surface_region_volume_set", self.analysis)
        self.assertIn("resolve_exposure_analysis_region_set", self.analysis)
        self.assertIn("surface_region_volume", self.workspace)

    def test_surface_geometry_identity_is_provider_native_and_depth_scoped(self):
        self.assertIn('"provider": "surface_region_volume"', self.adapter)
        self.assertIn('"sampling_depth_mm": float(properties["depth_mm"])', self.adapter)
        self.assertIn('region_source.get("geometry_cache_identity")', self.adapter)
        self.assertIn("surface_region_volume_cache_identity", self.adapter)

    def test_saved_payload_carries_generic_region_hierarchy(self):
        self.assertIn('"region_hierarchy": analysis_region_hierarchy_payload(hierarchy)', self.analysis)
        self.assertIn("analysis_region_hierarchy_from_payload", self.actions)
        self.assertIn('self._surface_target = self._target_kind == "surface_region_volume"', self.actions)

    def test_surface_shell_preparation_is_progress_and_cancel_aware(self):
        self.assertIn("progress_callback=progress_callback", self.analysis)
        self.assertIn("cancel_requested=cancel_requested", self.analysis)
        self.assertIn("ExposureAnalysisCancelled", self.analysis)


if __name__ == "__main__":
    unittest.main()
