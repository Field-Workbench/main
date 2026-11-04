"""Dependency-light tests for the shared Exposure Analysis region renderer."""
from __future__ import annotations

import unittest

import numpy as np

from fieldworkbench.analysis_region_render import (
    SELECTED_REGION_COLOUR,
    analysis_region_trace_styles,
    build_analysis_region_render_plan,
)
from fieldworkbench.analysis_region_scene import AnalysisRegionScene
from fieldworkbench.lpba40_display import (
    LPBA40_DIVISION_COLOURS,
    LPBA40_REGION_OPACITY,
    LPBA40_SELECTED_REGION_COLOUR,
    LPBA40_SELECTED_REGION_OPACITY,
)
from fieldworkbench.spatial_analysis import AnalysisRegionNode


def _scene(face_count_per_label: int = 12) -> AnalysisRegionScene:
    # Build independent triangles so compacting/remapping is easy to verify.
    vertices = []
    faces = []
    labels = []
    for label_id in (1, 2, 3):
        for index in range(face_count_per_label):
            base = len(vertices)
            x = float(label_id * 10 + index)
            vertices.extend(
                [
                    [x, 0.0, 0.0],
                    [x + 0.4, 0.0, 0.0],
                    [x, 0.4, 0.0],
                ]
            )
            faces.append([base, base + 1, base + 2])
            labels.append(label_id)
    hierarchy = AnalysisRegionNode(
        "all",
        "All regions",
        (1, 2, 3),
        (
            AnalysisRegionNode("one", "One", (1,)),
            AnalysisRegionNode("pair", "Pair", (2, 3)),
        ),
    )
    return AnalysisRegionScene(
        target_id="example",
        target_name="Example",
        target_kind="test",
        region_set_key="test.regions",
        hierarchy=hierarchy,
        vertices_m=np.asarray(vertices, dtype=float) / 1000.0,
        faces=np.asarray(faces, dtype=int),
        face_label_ids=np.asarray(labels, dtype=int),
        label_colours={1: "#112233", 2: "#445566", 3: "#778899"},
        base_opacity=0.65,
    )


def _brain_scene(face_count_per_label: int = 7) -> AnalysisRegionScene:
    base = _scene(face_count_per_label)
    hierarchy = AnalysisRegionNode(
        "whole_brain",
        "Whole brain",
        (1, 2, 3),
        (
            AnalysisRegionNode(
                "frontal",
                "Frontal lobe",
                (1, 2),
                (
                    AnalysisRegionNode("one", "One", (1,)),
                    AnalysisRegionNode("two", "Two", (2,)),
                ),
            ),
            AnalysisRegionNode(
                "parietal",
                "Parietal lobe",
                (3,),
                (AnalysisRegionNode("three", "Three", (3,)),),
            ),
        ),
    )
    return AnalysisRegionScene(
        target_id="brain",
        target_name="SRI24 brain",
        target_kind="brain_lpba40",
        region_set_key="sri24.lpba40",
        hierarchy=hierarchy,
        vertices_m=base.vertices_m,
        faces=base.faces,
        face_label_ids=base.face_label_ids,
        label_colours=base.label_colours,
        base_opacity=0.72,
    )


class AnalysisRegionRenderTests(unittest.TestCase):
    def test_preview_budget_preserves_every_atomic_region_deterministically(self):
        scene = _scene(20)
        first = build_analysis_region_render_plan(scene, triangle_budget=9)
        second = build_analysis_region_render_plan(scene, triangle_budget=9)
        self.assertEqual(first.rendered_triangle_count, 9)
        self.assertEqual(set(first.trace_index_by_label), {1, 2, 3})
        self.assertEqual(first.figure, second.figure)
        self.assertEqual(first.scene_identity, second.scene_identity)

    def test_unlimited_plan_keeps_all_source_triangles(self):
        scene = _scene(4)
        plan = build_analysis_region_render_plan(scene, triangle_budget=None)
        self.assertEqual(plan.source_triangle_count, 12)
        self.assertEqual(plan.rendered_triangle_count, 12)
        self.assertEqual(len(plan.figure["data"]), 3)
        self.assertEqual(plan.figure["layout"]["scene"]["dragmode"], "orbit")
        self.assertFalse(plan.figure["layout"]["scene"]["xaxis"]["visible"])

    def test_group_highlight_uses_shared_hierarchy_labels(self):
        scene = _scene(3)
        plan = build_analysis_region_render_plan(scene, triangle_budget=None)
        updates = analysis_region_trace_styles(scene, plan.trace_index_by_label, "pair")
        first = updates[plan.trace_index_by_label[1]]
        second = updates[plan.trace_index_by_label[2]]
        third = updates[plan.trace_index_by_label[3]]
        self.assertLess(first["opacity"], second["opacity"])
        self.assertEqual(second["color"], SELECTED_REGION_COLOUR)
        self.assertEqual(third["color"], SELECTED_REGION_COLOUR)

    def test_root_selection_is_whole_object_not_special_highlight(self):
        scene = _scene(2)
        plan = build_analysis_region_render_plan(scene, selected_region_key="all")
        colours = [trace["color"] for trace in plan.figure["data"]]
        self.assertEqual(colours, ["#112233", "#445566", "#778899"])

    def test_surface_region_viewer_is_fully_opaque_until_a_region_is_highlighted(self):
        base = _scene(2)
        scene = AnalysisRegionScene(
            target_id="bust", target_name="Bust", target_kind="surface_region_volume",
            region_set_key="surface", hierarchy=base.hierarchy,
            vertices_m=base.vertices_m, faces=base.faces, face_label_ids=base.face_label_ids,
            label_colours=base.label_colours, base_opacity=0.35,
        )
        plan = build_analysis_region_render_plan(scene, triangle_budget=None)
        self.assertTrue(all(float(trace["opacity"]) == 1.0 for trace in plan.figure["data"]))

        updates = analysis_region_trace_styles(scene, plan.trace_index_by_label, "one")
        selected = updates[plan.trace_index_by_label[1]]
        other = updates[plan.trace_index_by_label[2]]
        self.assertEqual(float(selected["opacity"]), 0.96)
        self.assertLess(float(other["opacity"]), 1.0)

    def test_lpba40_region_viewer_uses_full_shells_even_with_display_budget(self):
        scene = _brain_scene(11)
        plan = build_analysis_region_render_plan(scene, triangle_budget=5)
        self.assertEqual(plan.source_triangle_count, 33)
        self.assertEqual(plan.rendered_triangle_count, 33)
        self.assertIsNone(plan.triangle_budget)
        self.assertEqual(len(plan.figure["data"]), 3)

    def test_lpba40_region_viewer_matches_brain_selector_division_styling(self):
        scene = _brain_scene(2)
        plan = build_analysis_region_render_plan(scene)
        first = plan.figure["data"][plan.trace_index_by_label[1]]
        second = plan.figure["data"][plan.trace_index_by_label[2]]
        third = plan.figure["data"][plan.trace_index_by_label[3]]
        self.assertEqual(first["color"], LPBA40_DIVISION_COLOURS["frontal"])
        self.assertEqual(second["color"], LPBA40_DIVISION_COLOURS["frontal"])
        self.assertEqual(third["color"], LPBA40_DIVISION_COLOURS["parietal"])
        self.assertEqual(float(first["opacity"]), LPBA40_REGION_OPACITY)
        self.assertFalse(first["flatshading"])
        self.assertEqual(first["lighting"]["ambient"], 0.58)
        self.assertEqual(first["lighting"]["diffuse"], 0.72)

    def test_lpba40_selection_highlights_without_fading_other_shells(self):
        scene = _brain_scene(2)
        plan = build_analysis_region_render_plan(scene)
        updates = analysis_region_trace_styles(scene, plan.trace_index_by_label, "one")
        selected = updates[plan.trace_index_by_label[1]]
        other_same_division = updates[plan.trace_index_by_label[2]]
        other_division = updates[plan.trace_index_by_label[3]]
        self.assertEqual(selected["color"], LPBA40_SELECTED_REGION_COLOUR)
        self.assertEqual(float(selected["opacity"]), LPBA40_SELECTED_REGION_OPACITY)
        self.assertEqual(other_same_division["color"], LPBA40_DIVISION_COLOURS["frontal"])
        self.assertEqual(float(other_same_division["opacity"]), LPBA40_REGION_OPACITY)
        self.assertEqual(other_division["color"], LPBA40_DIVISION_COLOURS["parietal"])
        self.assertEqual(float(other_division["opacity"]), LPBA40_REGION_OPACITY)

    def test_explicit_budget_remains_available_only_for_generic_custom_scenes(self):
        scene = _scene(12_000)
        plan = build_analysis_region_render_plan(scene, triangle_budget=20_000)
        self.assertEqual(plan.source_triangle_count, 36_000)
        self.assertEqual(plan.rendered_triangle_count, 20_000)

    def test_surface_region_builtin_ignores_face_budget_and_keeps_complete_mesh(self):
        base = _scene(20)
        scene = AnalysisRegionScene(
            target_id="bust", target_name="Bust", target_kind="surface_region_volume",
            region_set_key="surface", hierarchy=base.hierarchy,
            vertices_m=base.vertices_m, faces=base.faces,
            face_label_ids=base.face_label_ids, label_colours=base.label_colours,
        )
        plan = build_analysis_region_render_plan(scene, triangle_budget=5)
        self.assertEqual(plan.source_triangle_count, 60)
        self.assertEqual(plan.rendered_triangle_count, 60)
        self.assertIsNone(plan.triangle_budget)

    def test_measurement_builtin_ignores_face_budget_and_keeps_complete_mesh(self):
        base = _scene(20)
        scene = AnalysisRegionScene(
            target_id="measurement", target_name="Measurement", target_kind="measurement_volume",
            region_set_key="measurement", hierarchy=base.hierarchy,
            vertices_m=base.vertices_m, faces=base.faces,
            face_label_ids=base.face_label_ids, label_colours=base.label_colours,
        )
        plan = build_analysis_region_render_plan(scene, triangle_budget=5)
        self.assertEqual(plan.source_triangle_count, 60)
        self.assertEqual(plan.rendered_triangle_count, 60)
        self.assertIsNone(plan.triangle_budget)


if __name__ == "__main__":
    unittest.main()
