"""Scene Recipe parser regressions for the current JSON-only AI contract."""

from __future__ import annotations

import json
import unittest

from fieldworkbench.scene_recipe import SceneRecipeError, parse_scene_recipe


class SceneRecipeParserTests(unittest.TestCase):
    @staticmethod
    def _minimal_recipe() -> dict:
        return {
            "format": "fieldworkbench-scene-recipe",
            "version": 1,
            "objects": [],
        }

    def test_accepts_plain_json_document(self):
        parsed = parse_scene_recipe(json.dumps(self._minimal_recipe()))
        self.assertEqual(parsed["recipe"]["format"], "fieldworkbench-scene-recipe")
        self.assertEqual(parsed["recipe"]["version"], 1)
        self.assertEqual(parsed["recipe"]["objects"], [])

    def test_rejects_markdown_fences(self):
        text = "```json\n" + json.dumps(self._minimal_recipe()) + "\n```"
        with self.assertRaisesRegex(SceneRecipeError, "JSON only"):
            parse_scene_recipe(text)

    def test_rejects_trailing_reference_or_sources_text(self):
        text = json.dumps(self._minimal_recipe()) + "\n\nSources:\n- https://example.invalid/"
        with self.assertRaisesRegex(SceneRecipeError, "Extra data"):
            parse_scene_recipe(text)

    def test_coil_electrical_load_modes_are_normalized(self):
        recipe = self._minimal_recipe()
        recipe["objects"] = [
            {
                "key": "coil_1",
                "type": "circular_coil",
                "name": "Test coil",
                "diameter_mm": 100.0,
                "resistance_mode": "override",
                "resistance_override_ohm": 3.25,
                "inductance_mode": "override",
                "inductance_override_mH": 12.0,
            }
        ]
        parsed = parse_scene_recipe(json.dumps(recipe))["recipe"]["objects"][0]
        self.assertEqual(parsed["resistance_mode"], "override")
        self.assertEqual(parsed["resistance_override_ohm"], 3.25)
        self.assertEqual(parsed["inductance_mode"], "override")
        self.assertEqual(parsed["inductance_override_mH"], 12.0)

    def test_coil_electrical_load_modes_reject_unknown_values(self):
        recipe = self._minimal_recipe()
        recipe["objects"] = [
            {
                "key": "coil_1",
                "type": "circular_coil",
                "name": "Test coil",
                "diameter_mm": 100.0,
                "inductance_mode": "magic",
            }
        ]
        with self.assertRaisesRegex(SceneRecipeError, "estimated.*override"):
            parse_scene_recipe(json.dumps(recipe))


if __name__ == "__main__":
    unittest.main()
