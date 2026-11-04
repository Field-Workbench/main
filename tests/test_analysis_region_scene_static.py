"""Source-contract checks for the Exposure Analysis region-scene provider layer."""
from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnalysisRegionSceneSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = (PROJECT_ROOT / "fieldworkbench" / "analysis_region_scene.py").read_text(
            encoding="utf-8"
        )
        cls.adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(
            encoding="utf-8"
        )
        cls.analysis = (PROJECT_ROOT / "fieldworkbench" / "exposure_analysis.py").read_text(
            encoding="utf-8"
        )
        cls.help = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(
            encoding="utf-8"
        )

    def test_scene_contract_is_renderer_agnostic_and_adapter_exposes_it(self):
        self.assertIn("class AnalysisRegionScene", self.scene)
        self.assertIn("face_label_ids", self.scene)
        self.assertIn("object_vertices_m", self.scene)
        self.assertIn("object_faces", self.scene)
        self.assertIn("def face_mask_for_node", self.scene)
        self.assertIn("def analysis_region_scene_identity", self.scene)
        self.assertIn("def resolve_analysis_region_scene", self.scene)
        self.assertIn("def analysis_region_scene(self, object_id: str)", self.adapter)
        self.assertNotIn("from PySide", self.scene)
        self.assertNotIn("import plotly", self.scene.lower())

    def test_scene_providers_share_existing_analysis_hierarchy_identities(self):
        self.assertIn('region_set_key="sri24.lpba40"', self.scene)
        self.assertIn('key="surface_regions"', self.scene)
        self.assertIn('region_set_key=f"surface_regions:{target_id}:inward_surface_volume"', self.scene)
        self.assertIn('region_set_key=f"measurement:{target_id}"', self.scene)
        self.assertIn("lpba40_region_hierarchy", self.scene)
        self.assertIn("geometry.face_region_labels", self.scene)
        self.assertIn("adapter._mesh_world_vertices_faces(mesh)", self.scene)

    def test_surface_regions_are_role_independent_in_analysis_and_help(self):
        start = self.analysis.index("def _is_surface_region_analysis_target")
        end = self.analysis.index("\ndef exposure_analysis_targets", start)
        helper = self.analysis[start:end]
        self.assertNotIn('role", "") == "measurement"', helper)
        self.assertIn("adapter.supports_surface_region_volume", helper)
        self.assertIn("regardless of its scene role", self.help)
        self.assertIn("does not replace Face/Scalp/etc.", self.help)


if __name__ == "__main__":
    unittest.main()
