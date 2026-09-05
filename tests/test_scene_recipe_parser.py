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


if __name__ == "__main__":
    unittest.main()
