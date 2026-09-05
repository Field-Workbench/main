from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import io
import json
import math
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import plotly.io as pio
from scipy.spatial.transform import Rotation

from fieldworkbench.anatomy_frames import (
    SRI24_HEAD_FIDUCIALS_RAS_MM,
    sri24_ras_mm_to_head_mm,
)
from fieldworkbench.scene_recipe import scene_recipe_guide
from fieldworkbench.probe_models import (
    ProbeDefinitionError,
    builtin_probe_definitions,
    validate_probe_definition,
)
from fieldworkbench.portable_scene import (
    PORTABLE_SCENE_FORMAT,
    TRANSLATOR_COMSOL,
    TRANSLATOR_FEMM,
    TRANSLATOR_GETDP,
    TRANSLATOR_OPENSCAD,
    TRANSLATOR_PORTABLE,
    build_portable_scene,
    _clip_femm_reference_segment_positive_r,
    _femm_reference_clean_segments,
    _femm_reference_cycle_edge_indices,
    _femm_reference_sanity_open_segments,
    femm_selection_status,
    getdp_selection_status,
    render_comsol_java,
    render_femm,
    render_openscad,
    write_export,
)
from fieldworkbench.getdp_export import (
    GETDP_MESH_PRESETS,
    GETDP_PROJECT_FORMAT,
    getdp_export_plan,
    render_getdp_project,
)
from fieldworkbench.studio_adapter import (
    FIELD_HEAT_COLOURSCALE,
    FIELD_SIGNED_COLOURSCALE,
    field_heat_colourscale_with_opacity,
    field_signed_colourscale_with_opacity,
    WORKBENCH_METADATA_VERSION,
    FIELD_PROGRESS_POINT_CHUNK,
    FieldCalculationCancelled,
    StudioAdapter,
    StudioOperationError,
    awg_diameter_mm,
    awg_resistance_20_ohm_per_km,
    default_scene,
    figure_in_mm,
)
from workflow_fixture import compact_workflow_scene


FINGERPRINT_POINTS = [
    [0.0, 0.0, 0.0],
    [0.01, 0.0, 0.0],
    [0.0, 0.01, 0.0],
    [0.0, 0.0, 0.025],
]


class CoreWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.adapter = StudioAdapter(compact_workflow_scene())

    def test_scene_document_validation_reports_event_records_in_objects(self):
        malformed = {
            "version": 1,
            "objects": [
                {
                    "id": "e1",
                    "op": "position",
                    "target": "coil",
                    "value": [0.0, 0.0, 0.0],
                }
            ],
        }
        with self.assertRaisesRegex(
            StudioOperationError,
            r"objects\[0\].*missing required 'type'.*events.*objects",
        ):
            StudioAdapter(malformed)

    def test_maxwell_fixture_uses_objects_and_events_in_native_scene_slots(self):
        fixture_path = Path(__file__).with_name("fixtures") / "maxwell_demo_reference.magpy.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["objects"]), 3)
        self.assertTrue(all(item.get("type") == "current.Circle" for item in payload["objects"]))
        self.assertEqual(len(payload["events"]), 8)
        self.assertEqual(sum(event.get("op") == "create" for event in payload["events"]), 3)
        self.assertEqual(sum(event.get("op") == "position" for event in payload["events"]), 2)
        self.assertEqual(sum(event.get("op") == "orientation" for event in payload["events"]), 3)

    def test_detached_copy_rebuilds_model_and_preserves_launch_preferences(self):
        self.adapter.set_coil_modeling_method("bundled")
        self.adapter.set_solver_memory_budget_mb(320)
        self.adapter.set_optimizer_worker_count(1)
        self.adapter.set_fluxline_minimum_field_cutoff(False, 1.25)
        self.adapter.set_fluxline_conductor_cutoff(False, 3.5)
        source_document = self.adapter.document()

        detached = self.adapter.detached_copy()

        self.assertIsNot(detached, self.adapter)
        self.assertEqual(detached.document(), source_document)
        self.assertEqual(detached.coil_modeling_method, "bundled")
        self.assertEqual(detached.solver_memory_budget_mb, 320)
        self.assertEqual(detached.optimizer_worker_count, 1)
        self.assertFalse(detached.fluxline_minimum_field_cutoff_enabled)
        self.assertEqual(detached.fluxline_minimum_field_cutoff_percent, 1.25)
        self.assertFalse(detached.fluxline_conductor_cutoff_enabled)
        self.assertEqual(detached.fluxline_conductor_cutoff_mm, 3.5)

        self.adapter.new_empty()
        self.assertEqual(detached.document(), source_document)
        detached.set_visible("coil_b", False)
        self.assertEqual(self.adapter.list_objects(), [])
        self.assertNotEqual(detached.document(), source_document)


    def test_portable_scene_normalizes_coils_before_translation(self):
        scene = build_portable_scene(self.adapter, ["coil_a", "coil_b", "axis_sensor"])
        self.assertEqual(scene["format"], PORTABLE_SCENE_FORMAT)
        self.assertEqual(scene["version"], 1)
        self.assertEqual(scene["units"]["length"], "mm")
        by_id = {item["id"]: item for item in scene["objects"]}
        self.assertEqual(by_id["coil_a"]["kind"], "coil")
        self.assertEqual(by_id["coil_a"]["geometry"]["shape"], "circle")
        self.assertAlmostEqual(by_id["coil_a"]["geometry"]["mean_diameter_mm"], 200.0)
        self.assertAlmostEqual(by_id["coil_a"]["transform"]["position_mm"][2], -50.0)
        self.assertEqual(len(by_id["coil_a"]["transform"]["matrix4x4_mm"]), 4)
        self.assertEqual(by_id["axis_sensor"]["kind"], "sensor")
        self.assertEqual(len(by_id["axis_sensor"]["sample_path_mm"]), 61)

        scad = render_openscad(scene)
        self.assertIn("Field Workbench OpenSCAD export", scad)
        self.assertIn("multmatrix(m=", scad)
        self.assertIn("turns=1", scad)
        self.assertIn("thin centreline reference only", scad)
        self.assertNotIn("Axis sensor 1", scad)

    def test_femm_axisymmetric_translator_emits_magnetic_coil_builder(self):
        self.adapter.coil_specs["coil_a"]["turns"] = 320
        self.adapter.coil_specs["coil_a"]["field_scale_factor"] = 1.25
        scene = build_portable_scene(self.adapter, ["coil_a", "coil_b"])

        lua = render_femm(scene, model_stem="fixture", apply_field_scale=False)
        self.assertIn('mi_probdef(0, "millimeters", "axi"', lua)
        self.assertIn('mi_addcircprop("FW_coil_1", 0.07, 1)', lua)
        self.assertIn('mi_setblockprop("Copper"', lua)
        self.assertIn(', 1, 320)', lua)
        self.assertIn('mi_makeABC(7,', lua)
        self.assertIn('mi_attachdefault()', lua)
        self.assertLess(lua.index('mi_makeABC('), lua.index('mi_attachdefault()'))
        self.assertIn('mi_saveas("fixture.fem")', lua)
        self.assertIn("Empirical Field correction is NOT applied", lua)

        corrected = render_femm(scene, apply_field_scale=True)
        self.assertIn('mi_addcircprop("FW_coil_1", 0.0875, 1)', corrected)

    def test_femm_translator_normalizes_arbitrary_coaxial_pair_to_z(self):
        operations = [
            {
                "method": "set_transform",
                "params": {
                    "object_id": "coil_a",
                    "position": [0.012, -0.04, 0.033],
                    "orientation": [90.0, 0.0, 0.0],
                },
            },
            {
                "method": "set_transform",
                "params": {
                    "object_id": "coil_b",
                    "position": [0.012, 0.04, 0.033],
                    "orientation": [90.0, 0.0, 0.0],
                },
            },
        ]
        self.adapter.apply_operations(operations)

        valid, status = femm_selection_status(self.adapter, ["coil_a", "coil_b"])
        self.assertTrue(valid, status)
        self.assertIn("automatically map", status)
        self.assertIn("canonical Z", status)

        scene = build_portable_scene(self.adapter, ["coil_a", "coil_b"])
        lua = render_femm(scene, model_stem="rotated_pair")
        self.assertIn("automatically rigidly normalizes", lua)
        self.assertIn("Source common-axis direction (world)", lua)
        self.assertIn("Source common-axis origin (world mm): [12, 0, 33]", lua)
        self.assertIn("normalized FEMM z=40", lua)
        self.assertIn("normalized FEMM z=-40", lua)
        self.assertIn('mi_saveas("rotated_pair.fem")', lua)

    def test_femm_translator_rejects_noncoaxial_selected_pair(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_b",
                        "position": [0.001, 0.0, 0.05],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        valid, status = femm_selection_status(self.adapter, ["coil_a", "coil_b"])
        self.assertFalse(valid)
        self.assertIn("coaxial", status.lower())
        scene = build_portable_scene(self.adapter, ["coil_a", "coil_b"])
        with self.assertRaisesRegex(StudioOperationError, "coaxial"):
            render_femm(scene)

    def test_femm_translator_adds_passive_reference_slice_outlines(self):
        guide = self.adapter.add_template("guide_box")
        selected = ["coil_a", "coil_b", guide]

        valid, status = femm_selection_status(
            self.adapter,
            selected,
            reference_slice_plane="xz",
        )
        self.assertTrue(valid, status)
        self.assertIn("1 selected passive object", status)
        self.assertIn("XZ exact source plane", status)

        scene = build_portable_scene(self.adapter, selected)
        lua = render_femm(scene, reference_slice_plane="xz")
        self.assertIn("Reference slice request: XZ; using XZ (exact source plane).", lua)
        self.assertIn("Reference slice offset in source-world plane coordinates: 0 mm (automatically locked to common axis)", lua)
        self.assertIn("Reference half: positive", lua)
        self.assertIn("Reference outline: Guide box", lua)
        self.assertIn("FEMM-safe contour cleanup", lua)
        self.assertIn("Reference opening: gap=0.25 mm; r=0 clearance=0.25 mm", lua)
        self.assertIn("residual cycles=0", lua)
        self.assertNotIn("mi_selectsegment(", lua)
        self.assertNotIn("mi_selectrectangle(", lua)
        self.assertNotIn("mi_setsegmentprop(", lua)
        self.assertLess(
            lua.index("mi_makeABC("),
            lua.index("-- Reference-only scene cross-sections (added after physics setup)."),
        )
        self.assertIn("No FEMM selection/grouping commands are used for guides", lua)
        self.assertIn("main Air label is also attached as FEMM's default block label", lua)
        self.assertLess(
            lua.index("mi_attachdefault()"),
            lua.index("-- Reference-only scene cross-sections (added after physics setup)."),
        )

        transverse_valid, transverse_status = femm_selection_status(
            self.adapter,
            selected,
            reference_slice_plane="xy",
        )
        self.assertTrue(transverse_valid, transverse_status)
        self.assertIn("XY exact source plane", transverse_status)
        self.assertIn("does not contain the full coil-axis direction", transverse_status)
        transverse_lua = render_femm(
            scene,
            reference_slice_plane="xy",
            reference_half="negative",
        )
        self.assertIn("Reference slice request: XY; using XY (exact source plane).", transverse_lua)
        self.assertIn("Reference slice offset in source-world plane coordinates: 0 mm (automatically locked to common axis)", transverse_lua)
        self.assertIn("Reference half: negative", transverse_lua)

    def test_femm_reference_cleanup_collapses_duplicate_tessellation_without_closing_loop(self):
        dirty_segments = []
        corners = [
            np.asarray([0.0, 0.0]),
            np.asarray([10.0, 0.0]),
            np.asarray([10.0, 10.0]),
            np.asarray([0.0, 10.0]),
        ]
        for first, second in zip(corners, corners[1:] + corners[:1], strict=True):
            for index in range(100):
                start = first + (second - first) * (index / 100.0)
                end = first + (second - first) * ((index + 1) / 100.0)
                dirty_segments.extend([(start, end), (end, start)])

        cleaned, stats = _femm_reference_clean_segments(dirty_segments)
        self.assertEqual(stats["raw_segment_count"], 800)
        self.assertEqual(stats["chain_count"], 1)
        self.assertLess(len(cleaned), 20)
        self.assertTrue(
            all(np.linalg.norm(second - first) >= 0.005 for first, second in cleaned)
        )
        self.assertEqual(_femm_reference_cycle_edge_indices(cleaned), [])
        # The 40 mm square perimeter retains a real 0.25 mm opening after
        # simplification rather than merely dropping one tiny STL triangle edge.
        retained_length = sum(float(np.linalg.norm(second - first)) for first, second in cleaned)
        self.assertAlmostEqual(retained_length, 39.75, places=6)

    def test_femm_reference_guides_stay_clear_of_axis_and_final_sanity_breaks_cycles(self):
        clipped = _clip_femm_reference_segment_positive_r(
            np.asarray([-2.0, 1.0]),
            np.asarray([2.0, 1.0]),
            min_r_mm=0.25,
        )
        self.assertIsNotNone(clipped)
        first, second = clipped
        self.assertAlmostEqual(float(first[0]), 0.25, places=9)
        self.assertAlmostEqual(float(second[0]), 2.0, places=9)

        triangle = [
            (np.asarray([1.0, 0.0]), np.asarray([5.0, 0.0])),
            (np.asarray([5.0, 0.0]), np.asarray([3.0, 4.0])),
            (np.asarray([3.0, 4.0]), np.asarray([1.0, 0.0])),
        ]
        opened, stats = _femm_reference_sanity_open_segments(triangle)
        self.assertEqual(stats["sanity_break_count"], 1)
        self.assertEqual(stats["residual_cycle_count"], 0)
        self.assertEqual(_femm_reference_cycle_edge_indices(opened), [])

    def test_femm_reference_slice_auto_offset_tracks_rotated_common_axis(self):
        guide = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.012, -0.04, 0.033],
                        "orientation": [90.0, 0.0, 0.0],
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_b",
                        "position": [0.012, 0.04, 0.033],
                        "orientation": [90.0, 0.0, 0.0],
                    },
                },
            ]
        )
        selected = ["coil_a", "coil_b", guide]
        valid, status = femm_selection_status(
            self.adapter, selected, reference_slice_plane="yz", reference_half="negative"
        )
        self.assertTrue(valid, status)
        self.assertIn("automatically locked to the common axis at 12 mm", status)
        scene = build_portable_scene(self.adapter, selected)
        lua = render_femm(
            scene, reference_slice_plane="yz", reference_half="negative"
        )
        self.assertIn(
            "Reference slice offset in source-world plane coordinates: 12 mm (automatically locked to common axis)",
            lua,
        )
        self.assertIn("Reference half: negative", lua)

    def test_femm_reference_slice_choice_does_not_block_coil_only_export(self):
        scene = build_portable_scene(self.adapter, ["coil_a", "coil_b"])
        valid, status = femm_selection_status(
            self.adapter,
            ["coil_a", "coil_b"],
            reference_slice_plane="xy",
        )
        self.assertTrue(valid, status)
        self.assertIn("No passive reference geometry", status)

        lua = render_femm(scene, reference_slice_plane="xy")
        self.assertIn("unused because no passive reference geometry was selected", lua)
        self.assertNotIn("Reference outline:", lua)

    def test_comsol_translator_emits_3d_edge_current_model(self):
        self.adapter.coil_specs["coil_a"]["turns"] = 10
        self.adapter.coil_specs["coil_a"]["field_scale_factor"] = 2.0
        scene = build_portable_scene(self.adapter, ["coil_a", "coil_b"])

        java = render_comsol_java(scene, apply_field_scale=False)
        self.assertIn('physics().create("mf", "InductionCurrents", "geom1")', java)
        self.assertIn('"BezierPolygon"', java)
        self.assertIn('"EdgeCurrent", 1', java)
        self.assertIn('model.param().set("Ieff1", "0.7[A]"', java)
        self.assertIn('selection().named("geom1_coil1_edg")', java)
        self.assertIn('study("std1").create("stat", "Stationary")', java)
        self.assertIn("empirical field scale is intentionally not applied", java)

        corrected = render_comsol_java(scene, apply_field_scale=True)
        self.assertIn('model.param().set("Ieff1", "1.4[A]"', corrected)

    def test_getdp_selection_accepts_multi_source_coils_and_sensor_samples(self):
        valid, status = getdp_selection_status(self.adapter, ["coil_a"])
        self.assertTrue(valid, status)
        self.assertIn("five generated verification points", status)

        valid, status = getdp_selection_status(
            self.adapter, ["coil_a", "axis_sensor"]
        )
        self.assertTrue(valid, status)
        self.assertIn("61 selected sensor sample points", status)

        valid, status = getdp_selection_status(
            self.adapter, ["coil_a", "coil_b"]
        )
        self.assertTrue(valid, status)
        self.assertIn("2 3D circular sources", status)

        square_id = self.adapter.add_template("polyline")
        valid, status = getdp_selection_status(self.adapter, [square_id])
        self.assertTrue(valid, status)
        self.assertIn("Square/rectangular", status)

        arbitrary_polyline = self.adapter.add_template("polyline")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": arbitrary_polyline,
                        "name": "vertices",
                        "value": [
                            [-0.05, -0.05, 0.0],
                            [0.06, -0.03, 0.0],
                            [0.04, 0.05, 0.0],
                            [-0.05, 0.04, 0.0],
                            [-0.05, -0.05, 0.0],
                        ],
                    },
                }
            ]
        )
        valid, status = getdp_selection_status(self.adapter, [arbitrary_polyline])
        self.assertFalse(valid)
        self.assertIn("not arbitrary Polyline coils", status)

    def test_getdp_translator_emits_runnable_single_loop_project(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "turns": 25,
                        "drive_current_a": 0.08,
                    },
                }
            ]
        )
        scene = build_portable_scene(self.adapter, ["coil_a"])

        files = render_getdp_project(
            scene,
            project_stem="single_loop.getdp.zip",
        )
        self.assertEqual(
            set(files),
            {
                "single_loop.geo",
                "single_loop.pro",
                "run.sh",
                "run.bat",
                "README.txt",
                "samples.csv",
                "scene.json",
                "project.json",
            },
        )

        geo = files["single_loop.geo"]
        self.assertIn('SetFactory("OpenCASCADE")', geo)
        self.assertIn("Mesh.MshFileVersion = 2.2", geo)
        self.assertIn("Mesh.Binary = 0", geo)
        self.assertIn("coil_1_volume[] = BooleanDifference", geo)
        self.assertIn("Coherence;", geo)
        self.assertIn("air_volume() -= excluded_volumes();", geo)
        self.assertIn("Mesh.CharacteristicLengthExtendFromBoundary = 0", geo)
        self.assertIn("Field[1] = Box", geo)
        self.assertIn('Physical Volume("Air", 1000)', geo)
        self.assertIn('Physical Volume("AirInf", 1001)', geo)
        self.assertIn('Physical Volume("Coil_1", 1101)', geo)
        self.assertIn('Physical Surface("OuterBoundary", 2000)', geo)

        problem = files["single_loop.pro"]
        self.assertIn("Name Hcurl_a_Gauge; Type Form1", problem)
        self.assertIn("AirInf = Region[1001]", problem)
        self.assertIn("Jacobian VolSphShell", problem)
        self.assertIn("EntityType EdgesOfTreeIn", problem)
        self.assertIn("Name Magnetostatics_a_3D; Type FemEquation", problem)
        self.assertIn("Effective signed ampere-turns in this export: 2 A.turn", problem)
        self.assertIn("SourceCurrentDensity[Coil_1]", problem)
        self.assertIn("Smoothing, Format Table", problem)
        self.assertIn("In DomainWithSourceCurrentDensity", problem)
        self.assertIn("1.e6 * {d a}", problem)
        self.assertIn("Name ExportFields", problem)
        self.assertEqual(problem.count("OnPoint"), 5)

        rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
        self.assertEqual(len(rows), 5)
        centre = rows[2]
        self.assertEqual(centre["label"], "axis +0 R")
        self.assertAlmostEqual(float(centre["reference_Bx_uT"]), 0.0)
        self.assertAlmostEqual(float(centre["reference_By_uT"]), 0.0)
        self.assertAlmostEqual(float(centre["reference_Bz_uT"]), 4.0 * math.pi)

        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["format"], GETDP_PROJECT_FORMAT)
        self.assertEqual(manifest["compatibility"]["gmsh_minimum"], "4.8.4")
        self.assertEqual(manifest["compatibility"]["getdp_minimum"], "3.2.0")
        self.assertEqual(manifest["compatibility"]["mesh_format"], "2.2 ASCII")
        self.assertIn('gmsh "single_loop.geo" -3 -format msh2', files["run.sh"])
        self.assertIn('getdp "single_loop.pro"', files["run.bat"])

    def test_getdp_standard_mesh_preset_uses_automatic_infinite_shell(self):
        scene = build_portable_scene(self.adapter, ["coil_a"])
        plan = getdp_export_plan(scene)
        self.assertEqual(plan["mesh_preset"], "standard")
        self.assertEqual(plan["source_elements"], 1.8)
        self.assertEqual(plan["near_radius_divisions"], 14.0)
        self.assertEqual(plan["far_radius_divisions"], 8.0)
        self.assertEqual(plan["open_boundary"], "VolSphShell")
        self.assertGreater(plan["shell_outer_radius_mm"], plan["inner_air_radius_mm"])

        files = render_getdp_project(scene, project_stem="standard_mesh")
        manifest = json.loads(files["project.json"])
        mesh = manifest["model"]["mesh"]
        self.assertEqual(mesh["preset"], "standard")
        self.assertEqual(mesh["controls"]["source_elements"], 1.8)
        self.assertEqual(mesh["controls"]["near_radius_divisions"], 14.0)
        self.assertEqual(mesh["controls"]["far_radius_divisions"], 8.0)
        self.assertEqual(mesh["open_boundary"]["getdp_jacobian"], "VolSphShell")
        self.assertTrue(mesh["open_boundary"]["automatic"])
        self.assertIn("// Mesh preset: standard", files["standard_mesh.geo"])
        self.assertIn("automatic spherical infinite-element shell", files["README.txt"])

    def test_getdp_mesh_presets_refine_locally_without_moving_open_boundary(self):
        scene = build_portable_scene(self.adapter, ["coil_a"])
        plans = {
            key: getdp_export_plan(scene, mesh_preset=key)
            for key in ("fast", "standard", "validation")
        }
        self.assertEqual(
            [GETDP_MESH_PRESETS[key]["label"] for key in plans],
            ["Fast", "Standard", "Validation"],
        )
        self.assertGreater(plans["fast"]["source_mesh_mm"], plans["standard"]["source_mesh_mm"])
        self.assertGreater(plans["standard"]["source_mesh_mm"], plans["validation"]["source_mesh_mm"])
        self.assertGreaterEqual(plans["fast"]["near_mesh_mm"], plans["standard"]["near_mesh_mm"])
        self.assertGreaterEqual(plans["standard"]["near_mesh_mm"], plans["validation"]["near_mesh_mm"])
        # Boundary placement is physics-driven and identical for every preset.
        for key in ("standard", "validation"):
            self.assertAlmostEqual(plans[key]["inner_air_radius_mm"], plans["fast"]["inner_air_radius_mm"])
            self.assertAlmostEqual(plans[key]["shell_outer_radius_mm"], plans["fast"]["shell_outer_radius_mm"])
        self.assertEqual(GETDP_MESH_PRESETS["validation"]["source_elements"], 3.0)
        self.assertEqual(GETDP_MESH_PRESETS["validation"]["near_radius_divisions"], 32.0)
        self.assertEqual(GETDP_MESH_PRESETS["validation"]["far_radius_divisions"], 12.0)
        self.assertFalse(plans["fast"]["source_grading_enabled"])
        self.assertFalse(plans["standard"]["source_grading_enabled"])
        self.assertTrue(plans["validation"]["source_grading_enabled"])
        self.assertEqual(plans["fast"]["gmsh_mesher"], "Delaunay")
        self.assertEqual(plans["standard"]["gmsh_mesher"], "Delaunay")
        self.assertEqual(plans["validation"]["gmsh_mesher"], "HXT")
        self.assertEqual(plans["validation"]["gmsh_thread_policy"], "system")
        self.assertGreater(
            plans["validation"]["source_grading_air_min_mm"],
            plans["validation"]["source_mesh_mm"],
        )
        self.assertLessEqual(
            plans["validation"]["source_grading_air_min_mm"],
            plans["validation"]["near_mesh_mm"],
        )
        self.assertAlmostEqual(
            plans["validation"]["source_grading_start_mm"],
            plans["validation"]["source_grading_air_min_mm"],
        )
        self.assertGreater(
            plans["validation"]["source_grading_end_mm"],
            plans["validation"]["source_grading_start_mm"],
        )

        validation_geo = render_getdp_project(
            scene, project_stem="validation_grading", mesh_preset="validation"
        )["validation_grading.geo"]
        self.assertIn("Field[2] = Distance", validation_geo)
        self.assertIn("Field[2].SurfacesList = {coil_surfaces()}", validation_geo)
        self.assertIn("Field[3] = Threshold", validation_geo)
        self.assertIn("source_grade_air_min =", validation_geo)
        self.assertIn("Field[3].SizeMin = source_grade_air_min", validation_geo)
        self.assertNotIn("Field[3].SizeMin = lc_source", validation_geo)
        self.assertIn("Field[3].StopAtDistMax = 1", validation_geo)
        self.assertIn("Field[4].FieldsList = {1, 3}", validation_geo)
        self.assertIn("Background Field = 4", validation_geo)
        self.assertIn("General.NumThreads = 0", validation_geo)
        self.assertIn("Mesh.MaxNumThreads3D = 0", validation_geo)
        self.assertIn("Mesh.Algorithm3D = 10", validation_geo)
        validation_files = render_getdp_project(
            scene, project_stem="validation_hxt", mesh_preset="validation"
        )
        self.assertIn(
            'gmsh "validation_hxt.geo" -3 -nt 0 -format msh2',
            validation_files["run.sh"],
        )
        self.assertIn(
            'gmsh "validation_hxt.geo" -3 -nt 0 -format msh2',
            validation_files["run.bat"],
        )

        standard_files = render_getdp_project(
            scene, project_stem="standard_no_grading", mesh_preset="standard"
        )
        standard_geo = standard_files["standard_no_grading.geo"]
        self.assertNotIn("Field[2] = Distance", standard_geo)
        self.assertIn("Mesh.Algorithm3D = 1", standard_geo)
        self.assertNotIn("General.NumThreads = 0", standard_geo)
        self.assertNotIn(" -nt 0 ", standard_files["run.sh"])
        self.assertIn("Background Field = 1", standard_geo)

    def test_getdp_can_force_numerical_source_even_when_physical_pack_exists(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "turns": 25,
                        "drive_current_a": 0.08,
                        "physical_geometry_enabled": True,
                        "winding_axial_width_mm": 12.0,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 4.0,
                    },
                }
            ]
        )
        scene = build_portable_scene(self.adapter, ["coil_a"])
        physical = getdp_export_plan(scene, use_physical_construction=True)
        numerical = getdp_export_plan(scene, use_physical_construction=False)
        self.assertEqual(physical["source_geometry"], ["saved physical winding pack"])
        self.assertEqual(numerical["source_geometry"], ["forced numerical winding pack"])
        self.assertEqual(physical["physical_source_count"], 1)
        self.assertEqual(numerical["numerical_source_count"], 1)

        files = render_getdp_project(
            scene,
            project_stem="forced_numerical",
            use_physical_construction=False,
        )
        manifest = json.loads(files["project.json"])
        self.assertFalse(manifest["model"]["use_physical_construction"])
        self.assertEqual(manifest["model"]["source_geometry"], ["forced numerical winding pack"])

        broken_scene = copy.deepcopy(scene)
        broken_scene["objects"][0]["construction"] = {
            "enabled": True,
            "supported": False,
            "valid": False,
            "errors": ["fixture construction failure"],
            "components": [],
        }
        with self.assertRaisesRegex(StudioOperationError, "invalid physical construction"):
            render_getdp_project(broken_scene, project_stem="broken_physical")
        bypass = render_getdp_project(
            broken_scene,
            project_stem="broken_bypass",
            use_physical_construction=False,
        )
        bypass_manifest = json.loads(bypass["project.json"])
        self.assertEqual(
            bypass_manifest["model"]["source_geometry"],
            ["forced numerical winding pack"],
        )

    def test_getdp_translator_emits_two_coil_helmholtz_project(self):
        scene = build_portable_scene(self.adapter, ["coil_a", "coil_b"])
        files = render_getdp_project(scene, project_stem="helmholtz_pair")

        geo = files["helmholtz_pair.geo"]
        self.assertIn("coil_1_volume[] = BooleanDifference", geo)
        self.assertIn("coil_2_volume[] = BooleanDifference", geo)
        self.assertIn('Physical Volume("Coil_1", 1101)', geo)
        self.assertIn('Physical Volume("Coil_2", 1102)', geo)
        self.assertIn("coil_volumes() = {coil_1_region(0), coil_2_region(0)}", geo)

        problem = files["helmholtz_pair.pro"]
        self.assertIn("Coil_1 = Region[1101]", problem)
        self.assertIn("Coil_2 = Region[1102]", problem)
        self.assertIn("SourceCurrentDensity[Coil_1]", problem)
        self.assertIn("SourceCurrentDensity[Coil_2]", problem)
        self.assertEqual(problem.count("Smoothing, Format Table"), 5)

        rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
        self.assertEqual(len(rows), 5)
        centre = rows[2]
        self.assertEqual(centre["label"], "pair axis +0 Rmax")
        self.assertAlmostEqual(float(centre["reference_Bx_uT"]), 0.0)
        self.assertAlmostEqual(float(centre["reference_By_uT"]), 0.0)
        self.assertAlmostEqual(
            float(centre["reference_Bz_uT"]),
            0.6294233998181445,
            places=9,
        )

        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["version"], 10)
        self.assertEqual(manifest["model"]["coil_count"], 2)
        self.assertEqual(manifest["model"]["coil_shapes"], ["circle", "circle"])
        self.assertEqual(manifest["model"]["sample_postprocessing"], "nodal_smoothing")

    def test_getdp_translator_emits_three_coil_maxwell_demo_project(self):
        fixture_path = Path(__file__).with_name("fixtures") / "maxwell_demo_reference.magpy.json"
        adapter = StudioAdapter(json.loads(fixture_path.read_text(encoding="utf-8")))
        coil_ids = ["central_ring", "maxwell_outer_pos", "maxwell_outer_neg"]

        valid, status = getdp_selection_status(adapter, coil_ids)
        self.assertTrue(valid, status)
        self.assertIn("3 3D circular sources", status)

        scene = build_portable_scene(adapter, coil_ids)
        files = render_getdp_project(scene, project_stem="maxwell_demo")

        geo = files["maxwell_demo.geo"]
        self.assertIn("Stranded-current source count: 3", geo)
        self.assertIn('Physical Volume("Coil_1", 1101)', geo)
        self.assertIn('Physical Volume("Coil_2", 1102)', geo)
        self.assertIn('Physical Volume("Coil_3", 1103)', geo)
        self.assertIn("coil_volumes() = {coil_1_region(0), coil_2_region(0), coil_3_region(0)}", geo)

        problem = files["maxwell_demo.pro"]
        self.assertIn("Coil_3 = Region[1103]", problem)
        self.assertIn("SourceCurrentDensity[Coil_3]", problem)
        self.assertIn("DomainWithSourceCurrentDensity = Region[{Coil_1, Coil_2, Coil_3}]", problem)
        self.assertEqual(problem.count("Smoothing, Format Table"), 5)

        rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[2]["label"], "system axis +0 Rmax")
        self.assertAlmostEqual(float(rows[0]["x_m"]), -0.1, places=12)
        self.assertAlmostEqual(float(rows[4]["x_m"]), 0.1, places=12)
        self.assertAlmostEqual(float(rows[2]["reference_Bx_uT"]), 200.03219758521539, places=9)
        self.assertAlmostEqual(float(rows[2]["reference_By_uT"]), 0.0, places=12)
        self.assertAlmostEqual(float(rows[2]["reference_Bz_uT"]), 0.0, places=12)

        sample_points = np.asarray(
            [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in rows]
        )
        csv_reference = np.asarray(
            [[float(row["reference_Bx_uT"]), float(row["reference_By_uT"]), float(row["reference_Bz_uT"])] for row in rows]
        )
        adapter.set_coil_modeling_method("centreline")
        workbench_reference = 1e6 * np.asarray(adapter.field_fingerprint(sample_points.tolist()), dtype=float)
        np.testing.assert_allclose(csv_reference, workbench_reference, rtol=2e-10, atol=2e-9)

        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["version"], 10)
        self.assertEqual(manifest["model"]["coil_count"], 3)
        self.assertEqual(manifest["model"]["coil_shapes"], ["circle", "circle", "circle"])
        self.assertEqual(
            manifest["model"]["source_geometry"],
            ["saved physical winding pack", "saved physical winding pack", "saved physical winding pack"],
        )

    def test_getdp_translator_emits_continuous_racetrack_winding_project(self):
        self.adapter.new_empty()
        racetrack_id = self.adapter.add_template("racetrack")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": racetrack_id,
                        "turns": 222,
                        "drive_current_a": 0.0275,
                        "physical_geometry_enabled": True,
                        "winding_axial_width_mm": 12.0,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 4.0,
                        "bobbin_wall_thickness_mm": 1.5,
                        "flange_height_mm": 5.0,
                        "flange_thickness_mm": 1.0,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": racetrack_id,
                        "position": [0.012, -0.008, 0.021],
                        "orientation": [90.0, 0.0, 0.0],
                    },
                },
            ]
        )

        valid, status = getdp_selection_status(self.adapter, [racetrack_id])
        self.assertTrue(valid, status)
        self.assertIn("3D racetrack source", status)

        scene = build_portable_scene(self.adapter, [racetrack_id])
        self.assertEqual(scene["objects"][0]["geometry"]["shape"], "racetrack")
        files = render_getdp_project(scene, project_stem="racetrack")

        geo = files["racetrack.geo"]
        self.assertIn("exact capsule band", geo)
        self.assertIn("Box(101)", geo)
        self.assertIn("outer_1[] = BooleanUnion", geo)
        self.assertIn("inner_1[] = BooleanUnion", geo)
        self.assertIn("coil_1_volume[] = BooleanDifference", geo)
        self.assertIn("Rotate {{1, 0, 0}", geo)
        self.assertIn("Translate {0.012, -0.008, 0.021}", geo)

        problem = files["racetrack.pro"]
        self.assertIn("Shape: racetrack", problem)
        self.assertIn("Coil1Q[] = Max[", problem)
        self.assertIn("Coil1DU[] = Coil1U[] - Coil1Q[]", problem)
        self.assertIn("Sqrt[Coil1DU[]^2 + Coil1V[]^2]", problem)
        self.assertIn("SourceCurrentDensity[Coil_1]", problem)
        self.assertIn("Effective signed ampere-turns in this export: 6.105 A.turn", problem)

        rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
        self.assertEqual(len(rows), 5)
        centre = rows[2]
        self.assertEqual(centre["reference_kind"], "FWB segmented-racetrack centreline Biot-Savart")
        reference = np.asarray(
            [
                float(centre["reference_Bx_uT"]),
                float(centre["reference_By_uT"]),
                float(centre["reference_Bz_uT"]),
            ]
        )
        self.assertGreater(float(np.linalg.norm(reference)), 1.0)

        sample_points = np.asarray(
            [
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
                for row in rows
            ]
        )
        csv_reference = np.asarray(
            [
                [
                    float(row["reference_Bx_uT"]),
                    float(row["reference_By_uT"]),
                    float(row["reference_Bz_uT"]),
                ]
                for row in rows
            ]
        )
        self.adapter.set_coil_modeling_method("centreline")
        workbench_reference = 1e6 * np.asarray(
            self.adapter.field_fingerprint(sample_points.tolist()), dtype=float
        )
        np.testing.assert_allclose(
            csv_reference,
            workbench_reference,
            rtol=2e-8,
            atol=2e-9,
        )

        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["version"], 10)
        self.assertEqual(manifest["model"]["coil_shapes"], ["racetrack"])
        self.assertEqual(
            manifest["model"]["source_geometry"],
            ["saved physical racetrack winding pack"],
        )

    def test_getdp_translator_emits_square_winding_project(self):
        self.adapter.new_empty()
        square_id = self.adapter.add_template("polyline")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": square_id,
                        "turns": 25,
                        "drive_current_a": 0.08,
                        "physical_geometry_enabled": True,
                        "winding_axial_width_mm": 6.0,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 3.0,
                    },
                }
            ]
        )

        valid, status = getdp_selection_status(self.adapter, [square_id])
        self.assertTrue(valid, status)
        self.assertIn("3D Square/rectangular source", status)

        scene = build_portable_scene(self.adapter, [square_id])
        self.assertEqual(scene["objects"][0]["geometry"]["shape"], "square")
        self.assertAlmostEqual(scene["objects"][0]["geometry"]["length_mm"], 100.0)
        self.assertAlmostEqual(scene["objects"][0]["geometry"]["width_mm"], 100.0)
        files = render_getdp_project(scene, project_stem="square")

        geo = files["square.geo"]
        self.assertIn("exact mitered path band", geo)
        self.assertIn("Box(7001)", geo)
        self.assertIn("Box(7002)", geo)
        self.assertIn("coil_1_volume[] = BooleanDifference", geo)

        problem = files["square.pro"]
        self.assertIn("Shape: square", problem)
        self.assertIn("Coil1DX[] = Fabs[Coil1U[]] - 0.05", problem)
        self.assertIn("Coil1DY[] = Fabs[Coil1V[]] - 0.05", problem)
        self.assertIn("Coil1DX[] >= Coil1DY[]", problem)
        self.assertIn("Sign[Coil1U[]]", problem)
        self.assertIn("Sign[Coil1V[]]", problem)
        self.assertIn("Effective signed ampere-turns in this export: 2 A.turn", problem)

        rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
        self.assertEqual(len(rows), 5)
        centre = rows[2]
        self.assertEqual(centre["reference_kind"], "FWB Square/rectangular centreline Biot-Savart")
        self.assertAlmostEqual(float(centre["reference_Bx_uT"]), 0.0, places=12)
        self.assertAlmostEqual(float(centre["reference_By_uT"]), 0.0, places=12)
        self.assertAlmostEqual(float(centre["reference_Bz_uT"]), 22.62741699796952, places=9)

        sample_points = np.asarray(
            [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in rows]
        )
        csv_reference = np.asarray(
            [[float(row["reference_Bx_uT"]), float(row["reference_By_uT"]), float(row["reference_Bz_uT"])] for row in rows]
        )
        workbench_reference = 1e6 * np.asarray(
            self.adapter.field_fingerprint(sample_points.tolist()), dtype=float
        )
        np.testing.assert_allclose(csv_reference, workbench_reference, rtol=2e-10, atol=2e-9)

        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["version"], 10)
        self.assertEqual(manifest["model"]["coil_shapes"], ["square"])
        self.assertEqual(
            manifest["model"]["source_geometry"],
            ["saved physical Square winding pack"],
        )

    def test_getdp_translator_emits_two_square_pair_fixture(self):
        fixture_path = Path(__file__).with_name("fixtures") / "square_pair_reference.magpy.json"
        adapter = StudioAdapter(json.loads(fixture_path.read_text(encoding="utf-8")))
        coil_ids = ["lower_reference_square", "upper_reference_square"]
        valid, status = getdp_selection_status(adapter, coil_ids)
        self.assertTrue(valid, status)
        self.assertIn("2 3D Square/rectangular sources", status)

        scene = build_portable_scene(adapter, coil_ids)
        files = render_getdp_project(scene, project_stem="square_pair")
        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["version"], 10)
        self.assertEqual(manifest["model"]["coil_shapes"], ["square", "square"])
        self.assertIn("Box(7001)", files["square_pair.geo"])
        self.assertIn("Box(7021)", files["square_pair.geo"])
        self.assertIn("SourceCurrentDensity[Coil_2]", files["square_pair.pro"])

        rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
        self.assertEqual(len(rows), 5)
        centre = rows[2]
        self.assertEqual(centre["label"], "pair axis +0 scale")
        self.assertAlmostEqual(float(centre["reference_Bx_uT"]), 0.0, places=12)
        self.assertAlmostEqual(float(centre["reference_By_uT"]), 0.0, places=12)
        self.assertAlmostEqual(float(centre["reference_Bz_uT"]), 34.13333333333334, places=9)

        sample_points = np.asarray(
            [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in rows]
        )
        csv_reference = np.asarray(
            [[float(row["reference_Bx_uT"]), float(row["reference_By_uT"]), float(row["reference_Bz_uT"])] for row in rows]
        )
        workbench_reference = 1e6 * np.asarray(
            adapter.field_fingerprint(sample_points.tolist()), dtype=float
        )
        np.testing.assert_allclose(csv_reference, workbench_reference, rtol=2e-10, atol=2e-9)

    def test_getdp_translator_accepts_mixed_circle_square_and_racetrack(self):
        square_id = self.adapter.add_template("polyline")
        racetrack_id = self.adapter.add_template("racetrack")
        coil_ids = ["coil_a", square_id, racetrack_id]
        valid, status = getdp_selection_status(self.adapter, coil_ids)
        self.assertTrue(valid, status)
        self.assertIn("mixed circular/Square/racetrack", status)

        scene = build_portable_scene(self.adapter, coil_ids)
        files = render_getdp_project(scene, project_stem="mixed_three_families")
        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["model"]["coil_shapes"], ["circle", "square", "racetrack"])
        self.assertIn('Physical Volume("Coil_3", 1103)', files["mixed_three_families.geo"])

    def test_getdp_translator_accepts_mixed_circle_and_racetrack_pair(self):
        racetrack_id = self.adapter.add_template("racetrack")
        valid, status = getdp_selection_status(self.adapter, ["coil_a", racetrack_id])
        self.assertTrue(valid, status)
        self.assertIn("mixed circular/racetrack", status)

        scene = build_portable_scene(self.adapter, ["coil_a", racetrack_id])
        files = render_getdp_project(scene, project_stem="mixed_pair")
        manifest = json.loads(files["project.json"])
        self.assertEqual(manifest["model"]["coil_shapes"], ["circle", "racetrack"])
        self.assertIn('Physical Volume("Coil_1", 1101)', files["mixed_pair.geo"])
        self.assertIn('Physical Volume("Coil_2", 1102)', files["mixed_pair.geo"])

    def test_getdp_project_preserves_rotation_sign_and_optional_field_scale(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "turns": 25,
                        "drive_current_a": -0.08,
                        "field_scale_factor": 1.5,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.03, -0.02, 0.01],
                        "orientation": [0.0, 60.0, 0.0],
                    },
                }
            ]
        )
        scene = build_portable_scene(self.adapter, ["coil_a"])
        files = render_getdp_project(
            scene,
            project_stem="rotated",
            apply_field_scale=True,
        )

        problem = files["rotated.pro"]
        self.assertIn("Effective signed ampere-turns in this export: -3 A.turn", problem)
        self.assertRegex(problem, r"\n\s+-[0-9.]+ \* Vector\[")
        self.assertNotIn(" - -", problem)
        manifest = json.loads(files["project.json"])
        self.assertTrue(manifest["model"]["apply_field_scale"])

        rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
        centre = rows[2]
        normal = np.asarray([math.sqrt(3.0) / 2.0, 0.0, 0.5])
        expected_ut = 1e6 * (4e-7 * math.pi) * -3.0 / (2.0 * 0.1) * normal
        np.testing.assert_allclose(
            [
                float(centre["reference_Bx_uT"]),
                float(centre["reference_By_uT"]),
                float(centre["reference_Bz_uT"]),
            ],
            expected_ut,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_getdp_writer_packages_sources_scripts_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.getdp.zip"
            result = write_export(
                self.adapter,
                ["coil_a"],
                TRANSLATOR_GETDP,
                path,
            )
            self.assertEqual(result, path)
            import zipfile

            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "fixture.geo",
                        "fixture.pro",
                        "run.sh",
                        "run.bat",
                        "README.txt",
                        "samples.csv",
                        "scene.json",
                        "project.json",
                    },
                )
                self.assertIn(
                    "GetDP 3-D magnetostatic project",
                    archive.read("fixture.pro").decode("utf-8"),
                )
                self.assertTrue((archive.getinfo("run.sh").external_attr >> 16) & 0o111)

    def test_portable_package_and_openscad_writer_use_same_normalized_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            package_path = Path(directory) / "fixture.fwpkg"
            scad_path = Path(directory) / "fixture.scad"
            write_export(
                self.adapter,
                ["coil_a", "axis_sensor"],
                TRANSLATOR_PORTABLE,
                package_path,
            )
            write_export(
                self.adapter,
                ["coil_a"],
                TRANSLATOR_OPENSCAD,
                scad_path,
            )
            import zipfile
            with zipfile.ZipFile(package_path) as archive:
                self.assertEqual(set(archive.namelist()), {"manifest.json", "scene.json"})
                portable = json.loads(archive.read("scene.json"))
            self.assertEqual(portable["format"], PORTABLE_SCENE_FORMAT)
            self.assertIn("coil_a", {item["id"] for item in portable["objects"]})
            self.assertIn("axis_sensor", {item["id"] for item in portable["objects"]})
            self.assertIn("Lower coil", scad_path.read_text(encoding="utf-8"))

    def test_portable_scene_preserves_builtin_well_plate_samples(self):
        plate_id = self.adapter.add_template("well_plate_96")
        scene = build_portable_scene(self.adapter, [plate_id])
        record = next(item for item in scene["objects"] if item["id"] == plate_id)
        self.assertEqual(record["kind"], "geometry")
        self.assertEqual(record["builtin_template"], "well_plate_96")
        self.assertEqual(record["well_plate"]["wells"], 96)
        self.assertEqual(len(record["well_plate"]["sample_points_mm"]), 96)
        self.assertEqual(record["measurement"]["sampling_mode"], "user_defined")
        self.assertEqual(len(record["measurement"]["defined_points_mm"]), 96)

    def test_optimizer_run_report_includes_plan_top_three_and_selected_detail(self):
        def candidate(index: int, spread: float):
            return {
                "id": f"tuning_{index}",
                "mode": "tune_existing",
                "setting_summary": f"diameter={200 + index} mm; spacing={100 + index} mm",
                "coil_labels": ["Coil 1", "Coil 2"],
                "currents_a": {"coil_1": 0.075 + index * 0.001, "coil_2": 0.075 + index * 0.001},
                "metrics": {
                    "mean_uT": 200.0 + index,
                    "min_uT": 198.0,
                    "max_uT": 202.0,
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": spread,
                    "directional_consistency_pct": 99.99,
                    "rms_target_error_uT": 0.5 + index,
                    "max_voltage_v": 2.0,
                    "total_power_w": 0.25,
                    "minimum_clearance_mm": 20.0 + index,
                },
            }

        finalists = [candidate(1, 0.10), candidate(2, 0.12), candidate(3, 0.15), candidate(4, 0.18)]
        result = {
            "settings": {"optimization_plan": "OPTIMIZATION PLAN\nTARGET\nUse Guide sphere 1."},
            "finalists": finalists,
            "generated_candidate_count": 120,
            "feasible_candidate_count": 116,
            "rejected_candidate_count": 4,
            "distinct_finalist_count": 4,
            "finalist_filter": {
                "primary_tied_count": 4,
                "nondominated_count": 4,
                "near_duplicate_removed": 0,
            },
            "parameter_count": 3,
            "final_sample_count": 791,
            "targets": [{"id": "target"}],
            "elapsed_seconds": 74.8,
            "search_strategy": {
                "outer_search": "active_set_screening_reduced_search",
                "current_mode": "inner_solve",
            },
            "search_reduction": {
                "enabled": True,
                "initial_structural_count": 5,
                "active_structural_count": 2,
                "active_focus_coordinate_count": 2,
                "active_raw_structural_count": 3,
                "screening_evaluations": 41,
                "active_labels": ["Circular diameter Group A", "Pair spacing Group A"],
                "variables": [
                    {"id": "diameter", "label": "Circular diameter Group A", "status": "active", "reason": "one-at-a-time screening found a material improvement", "trend": "best screened value was near the upper bound"},
                    {"id": "dx", "label": "ΔX Position Group A", "status": "frozen", "reason": "all screened departures were no better than the source design", "trend": ""},
                ],
                "collective_modes": [
                    {
                        "id": "collective:common:position_x",
                        "label": "Common ΔX Position",
                        "mode": "common",
                        "members": ["dx_a", "dx_b"],
                        "status": "active",
                        "reason": "collective move improved materially",
                        "best_probe_value": -12.5,
                    }
                ],
                "interactions": [
                    {"variables": ["diameter", "spacing"], "labels": ["Circular diameter Group A", "Pair spacing Group A"], "reason": "combined bound probes improved materially beyond either one-at-a-time screen"}
                ],
                "challenge_rounds": [{"round": 1, "reactivated": []}],
                "finalist_normalization": {
                    "enabled": True,
                    "source_preserved_replacements": 3,
                    "evaluations": 2,
                    "meaningful_frozen_ids": [],
                    "meaningful_frozen_labels": [],
                },
            },
        }
        with mock.patch(
            "fieldworkbench.studio_adapter.reported_cpu_description",
            return_value="Intel(R) Example CPU @ 3.40GHz",
        ):
            report = StudioAdapter.optimizer_run_report(
                result, selected_candidate=finalists[1], top_n=3
            )
        self.assertRegex(
            report,
            r"^FIELD WORKBENCH \[version [^\]]+\] OPTIMIZATION REPORT\n",
        )
        self.assertIn("System: Intel(R) Example CPU @ 3.40GHz", report)
        self.assertIn("OPTIMIZATION PLAN", report)
        self.assertIn("Use Guide sphere 1.", report)
        self.assertIn("Authoritative settings tested: 120", report)
        self.assertIn("Distinct finalists: 4", report)
        self.assertIn("Finalist curation:", report)
        self.assertIn("SEARCH REDUCTION", report)
        self.assertIn("focused search used 2 internal active coordinates representing 3 raw freedoms", report)
        self.assertIn("Collective assembly modes detected", report)
        self.assertIn("Common ΔX Position", report)
        self.assertIn("Circular diameter Group A + Pair spacing Group A", report)
        self.assertIn("restored frozen freedoms to their source-scene values in 3 contenders", report)
        self.assertIn("TOP 3 FINALISTS", report)
        self.assertIn("1. diameter=201 mm; spacing=101 mm", report)
        self.assertIn("3. diameter=203 mm; spacing=103 mm", report)
        self.assertNotIn("4. diameter=204 mm; spacing=104 mm", report)
        self.assertIn("SELECTED FINALIST DETAILS", report)
        self.assertIn("Field Workbench tuning result — tuning_2", report)

    def test_optimizer_run_report_includes_registry_fidelity_audit(self):
        def evaluation(fidelity, samples, coverage, vector, magnitude, angle, mean, current_ma):
            rank = {"scout": 10, "fine_scout": 20, "authoritative": 40}[fidelity]
            return {
                "stage": fidelity,
                "fidelity": fidelity,
                "fidelity_rank": rank,
                "model": "Centreline" if rank < 40 else "Segmented winding",
                "sample_count": samples,
                "feasible": True,
                "metrics": {
                    "snapshot_match_coverage_pct": coverage,
                    "snapshot_vector_rms_uT": vector,
                    "snapshot_magnitude_rms_uT": magnitude,
                    "snapshot_mean_angle_deg": angle,
                    "mean_uT": mean,
                },
                "currents_a": {"coil_a": current_ma / 1000.0, "coil_b": current_ma / 1000.0},
            }

        result = {
            "settings": {"optimization_plan": "OPTIMIZATION PLAN", "search_effort": "fast"},
            "reference_mode": "snapshot",
            "objective": "snapshot_vector_match",
            "variables": [
                {"id": "spacing", "label": "Pair spacing", "unit": "mm"},
            ],
            "finalists": [],
            "generated_candidate_count": 0,
            "feasible_candidate_count": 0,
            "rejected_candidate_count": 0,
            "distinct_finalist_count": 0,
            "parameter_count": 1,
            "final_sample_count": 1791,
            "targets": [{"id": "target"}],
            "elapsed_seconds": 1.0,
            "candidate_registry": {
                "candidate_count": 3,
                "views": {
                    "source_baseline": ["candidate_000001"],
                    "drc_boundary": ["candidate_000001", "candidate_000003"],
                    "finalists": ["candidate_000003"],
                },
                "candidates": [
                    {
                        "id": "candidate_000001",
                        "setting_summary": "Pair spacing=171 mm",
                        "structural_values": {"spacing": 171.0},
                        "tags": ["source_baseline", "drc_boundary"],
                        "drc": {"status": "clear", "minimum_clearance_mm": 1.25},
                        "evaluations": [
                            evaluation("fine_scout", 80, 93.75, 0.252, 0.18, 1.2, 6.526, 12.4),
                            evaluation("authoritative", 1791, 80.0, 0.50, 0.32, 2.0, 6.49, 12.3),
                        ],
                    },
                    {
                        "id": "candidate_000002",
                        "setting_summary": "Pair spacing=150 mm",
                        "structural_values": {"spacing": 150.0},
                        "tags": [],
                        "drc": {"status": "blocked", "minimum_clearance_mm": -4.0},
                        "evaluations": [
                            evaluation("fine_scout", 80, 98.75, 0.10, 0.08, 0.4, 6.50, 11.8),
                        ],
                    },
                    {
                        "id": "candidate_000003",
                        "setting_summary": "Pair spacing=185 mm",
                        "structural_values": {"spacing": 185.0},
                        "tags": ["drc_boundary"],
                        "drc": {"status": "clear", "minimum_clearance_mm": 0.10},
                        "evaluations": [
                            evaluation("scout", 80, 95.0, 0.20, 0.15, 0.8, 6.51, 13.0),
                            evaluation("authoritative", 1791, 92.0, 0.30, 0.22, 1.0, 6.48, 13.1),
                        ],
                    },
                ],
            },
        }
        report = StudioAdapter.optimizer_run_report(result)
        self.assertIn("FIDELITY AUDIT", report)
        self.assertIn("candidate_000001 — Source baseline; Authoritative-tested", report)
        self.assertIn("candidate_000002 — Best overall scout", report)
        self.assertIn("candidate_000003 — Best DRC-boundary solution; Authoritative-tested finalist", report)
        self.assertIn("Scout [fine scout; Centreline; 80 samples]", report)
        self.assertIn("Authoritative [authoritative; Segmented winding; 1791 samples]", report)
        self.assertIn("Scout → authoritative Δ (authoritative minus scout): match -13.75 percentage points", report)
        self.assertIn("vector RMS +0.248 µT", report)

    def test_optimizer_run_report_describes_discrete_1d_resolution_and_effective_search(self):
        result = {
            "settings": {
                "optimization_plan": "OPTIMIZATION PLAN\nTARGET\nUse three plates.",
                "search_effort": "fast",
            },
            "variables": [
                {
                    "id": "current_a", "kind": "current", "label": "Current Group A",
                    "unit": "mA", "min": 23.1, "max": 231.0, "base": 85.0,
                    "step": 0.1, "step_origin": 85.0, "integer": False,
                },
                {
                    "id": "spacing_a", "kind": "pair_spacing", "label": "Pair spacing Group A",
                    "unit": "mm", "min": 73.5, "max": 136.5, "base": 89.0,
                    "step": 0.5, "step_origin": 89.0, "integer": False,
                },
            ],
            "finalists": [],
            "generated_candidate_count": 81,
            "feasible_candidate_count": 81,
            "rejected_candidate_count": 0,
            "distinct_finalist_count": 0,
            "parameter_count": 2,
            "final_sample_count": 288,
            "targets": [{"id": "middle"}, {"id": "lower"}, {"id": "upper"}],
            "elapsed_seconds": 142.0,
            "search_strategy": {
                "outer_search": "adaptive_1d_convergence",
                "outer_parameter_count": 1,
                "current_mode": "inner_solve",
                "current_parameter_count": 1,
            },
            "convergence": {
                "converged": True,
                "reason": "One-dimensional search completed its configured full-range sampling pass on the requested 0.5 mm increment grid.",
            },
        }
        with mock.patch(
            "fieldworkbench.studio_adapter.reported_cpu_description",
            return_value="Intel(R) Example CPU",
        ):
            report = StudioAdapter.optimizer_run_report(result)
        self.assertIn(
            "Search resolution: Current Group A 0.1 mA; Pair spacing Group A 0.5 mm",
            report,
        )
        self.assertIn(
            "Search strategy: 1D increment-grid search on Pair spacing Group A (0.5 mm); "
            "inner current solve on 0.1 mA grid",
            report,
        )
        self.assertIn(
            "Effective search: Legacy 1D structural route with inner-solved current; retained only when universal "
            "cheap-first mapping is bypassed",
            report,
        )
        self.assertNotIn("configured parameter tolerance", report)

    def test_optimizer_failed_run_report_keeps_plan_and_rejection_diagnostics(self):
        result = {
            "settings": {"optimization_plan": "OPTIMIZATION PLAN\nTARGET\nUse Guide sphere 1."},
            "finalists": [],
            "generated_candidate_count": 64,
            "feasible_candidate_count": 0,
            "rejected_candidate_count": 64,
            "distinct_finalist_count": 0,
            "parameter_count": 11,
            "final_sample_count": 1791,
            "targets": [{"id": "target"}],
            "elapsed_seconds": 612.0,
            "search_strategy": {
                "outer_search": "adaptive_global_multistart_coordinate_convergence",
                "current_mode": "inner_solve",
            },
            "convergence": {
                "converged": False,
                "reason": "No feasible candidate was found.",
            },
            "failure_diagnostics": {
                "source_candidate": {
                    "tested": True,
                    "feasible": False,
                    "error": "DRC intersecting: Coil 1 / Coil 2.",
                },
                "rejection_reasons": [
                    {"reason": "DRC intersecting: Coil 1 / Coil 2.", "count": 51},
                    {"reason": "Electrical limit exceeded: resistive voltage per coil.", "count": 13},
                ],
                "rejection_examples": [
                    {
                        "setting_summary": "Turns Group A=448 turns; Pair spacing Group A=56 mm",
                        "error": "DRC intersecting: Coil 1 / Coil 2.",
                    }
                ],
            },
        }
        report = StudioAdapter.optimizer_run_report(result, top_n=3)
        self.assertIn("OPTIMIZATION PLAN", report)
        self.assertIn("Feasible / rejected at authoritative fidelity: 0 / 64", report)
        self.assertIn("REJECTION DIAGNOSTICS", report)
        self.assertIn("Source-scene starting point: rejected", report)
        self.assertIn("51 × DRC intersecting", report)
        self.assertIn("Example rejected settings", report)
        self.assertIn("TOP 0 FINALISTS", report)
        self.assertIn("No feasible finalists were returned.", report)
        self.assertNotIn("SELECTED FINALIST DETAILS", report)


    def test_optimizer_refinement_score_keeps_coverage_primary_and_uses_uniformity_on_plateau(self):
        perfect_flat = {
            "in_band_pct": 100.0,
            "uniformity_spread_pct": 0.02,
            "rms_target_error_uT": 0.05,
            "directional_consistency_pct": 100.0,
        }
        perfect_less_flat = {
            "in_band_pct": 100.0,
            "uniformity_spread_pct": 0.08,
            "rms_target_error_uT": 0.01,
            "directional_consistency_pct": 100.0,
        }
        one_sample_missing = {
            "in_band_pct": 100.0 - 100.0 / 1791.0,
            "uniformity_spread_pct": 0.001,
            "rms_target_error_uT": 0.001,
            "directional_consistency_pct": 100.0,
        }
        score = StudioAdapter._optimizer_tuning_refinement_score
        self.assertLess(score(perfect_flat, "coverage", 1791), score(perfect_less_flat, "coverage", 1791))
        self.assertLess(score(perfect_less_flat, "coverage", 1791), score(one_sample_missing, "coverage", 1791))

    def test_optimizer_turns_current_margin_requires_complete_current_groups(self):
        turns = {
            "id": "turns_lower", "kind": "turns", "coil_ids": ["lower"],
            "min": 192.0, "max": 448.0,
        }
        independent_current = [{
            "id": "current_lower", "kind": "current", "coil_ids": ["lower"],
            "min": 50.0, "max": 300.0,
        }]
        margin = StudioAdapter._optimizer_tuning_turns_current_margin(
            turns, independent_current, {"current_lower": 75.0}
        )
        self.assertAlmostEqual(margin, 0.1)

        linked_current = [{
            "id": "current_pair", "kind": "current", "coil_ids": ["lower", "upper"],
            "min": 50.0, "max": 300.0,
        }]
        self.assertIsNone(
            StudioAdapter._optimizer_tuning_turns_current_margin(
                turns, linked_current, {"current_pair": 75.0}
            )
        )

    def test_optimizer_primary_saturation_requires_every_coverage_sample(self):
        sample_count = 1791
        perfect = {"in_band_pct": 100.0}
        one_missing = {"in_band_pct": 100.0 - 100.0 / sample_count}
        self.assertTrue(
            StudioAdapter._optimizer_tuning_primary_saturated(
                perfect, "coverage", sample_count
            )
        )
        self.assertFalse(
            StudioAdapter._optimizer_tuning_primary_saturated(
                one_missing, "coverage", sample_count
            )
        )
        self.assertTrue(
            StudioAdapter._optimizer_tuning_primary_saturated(
                {"snapshot_match_coverage_pct": 100.0},
                "snapshot_coverage",
                sample_count,
            )
        )
        self.assertFalse(
            StudioAdapter._optimizer_tuning_primary_saturated(
                {"uniformity_spread_pct": 0.0}, "uniformity", sample_count
            )
        )

    def test_optimizer_distinct_saturated_seeds_keep_alternate_structural_basins(self):
        variables = [
            {"id": "diameter", "min": 100.0, "max": 200.0},
            {"id": "x", "min": -25.0, "max": 25.0},
            {"id": "rz", "min": -30.0, "max": 30.0},
        ]

        def candidate(diameter, x, rz, spread):
            return {
                "variable_values": {
                    "diameter": float(diameter),
                    "x": float(x),
                    "rz": float(rz),
                },
                "metrics": {
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": float(spread),
                    "rms_target_error_uT": float(spread),
                    "directional_consistency_pct": 100.0,
                },
            }

        cloud = [
            candidate(188.5, -25.0, 30.0, 0.70),
            candidate(187.0, -24.0, 29.0, 0.71),
            candidate(183.7, -8.0, 11.0, 0.32),
            candidate(182.8, -7.0, 10.0, 0.33),
            candidate(160.0, 10.0, -10.0, 0.50),
            candidate(140.0, 20.0, -25.0, 0.90),
        ]
        seeds = StudioAdapter._optimizer_tuning_distinct_saturated_seeds(
            cloud, "coverage", variables, 1791, limit=4
        )
        self.assertEqual(len(seeds), 4)
        self.assertAlmostEqual(seeds[0]["metrics"]["uniformity_spread_pct"], 0.32)
        rows = [item["variable_values"] for item in seeds]
        self.assertTrue(any(abs(item["x"] + 25.0) < 1e-9 for item in rows))
        self.assertTrue(any(abs(item["x"] - 10.0) < 1e-9 for item in rows))

    def test_zero_current_detailed_coil_descriptor_short_circuits(self):
        descriptor = {
            "kind": "circle",
            "current": 0.0,
            "diameter": 0.18,
            "position": np.zeros(3, dtype=float),
            "rotation": np.eye(3, dtype=float),
            "direction": 1.0,
            "path": np.empty((0, 3), dtype=float),
        }
        dimensions = {
            "turns": 320,
            "outer_wire_m": 0.00037,
            "axial_width_m": 0.022,
            "radial_build_m": 0.0021,
        }
        for requested in ("exact", "bundled", "current_sheet"):
            compiled, actual, error = self.adapter._compile_coil_descriptor(
                descriptor, dimensions, requested
            )
            self.assertEqual(compiled, [])
            self.assertEqual(actual, requested)
            self.assertIsNone(error)

    def test_optimizer_scout_spatial_sample_keeps_extrema_and_spreads_points(self):
        x, y, z = np.meshgrid(
            np.linspace(-1.0, 1.0, 9),
            np.linspace(-2.0, 2.0, 7),
            np.linspace(-3.0, 3.0, 5),
            indexing="ij",
        )
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        indices = StudioAdapter._optimizer_tuning_spatial_sample_indices(points, 24)
        self.assertEqual(len(indices), 24)
        self.assertEqual(len(set(map(int, indices))), 24)
        sampled = points[indices]
        for axis in range(3):
            self.assertAlmostEqual(float(np.min(sampled[:, axis])), float(np.min(points[:, axis])))
            self.assertAlmostEqual(float(np.max(sampled[:, axis])), float(np.max(points[:, axis])))

    def test_optimizer_sobol_map_samples_are_deterministic_and_well_spread(self):
        first = StudioAdapter._optimizer_tuning_sobol_unit_samples(257, 5, 12345)
        second = StudioAdapter._optimizer_tuning_sobol_unit_samples(257, 5, 12345)
        self.assertEqual(first.shape, (257, 5))
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(first < 1.0))
        # Every dimension should cover most of the unit interval rather than
        # clustering into a small random patch.
        self.assertTrue(np.all(np.min(first, axis=0) < 0.02))
        self.assertTrue(np.all(np.max(first, axis=0) > 0.98))

    def test_optimizer_design_map_keeps_distinct_quality_landmarks(self):
        variables = [
            {"id": "x", "min": -25.0, "max": 25.0},
            {"id": "diameter", "min": 100.0, "max": 190.0},
        ]

        def candidate(x, diameter, spread):
            return {
                "variable_values": {"x": float(x), "diameter": float(diameter)},
                "metrics": {
                    "in_band_pct": 100.0,
                    "rms_target_error_uT": float(spread),
                    "uniformity_spread_pct": float(spread),
                    "directional_consistency_pct": 100.0,
                },
            }

        cloud = [
            candidate(-24.0, 188.0, 0.30),
            candidate(-23.5, 187.5, 0.31),
            candidate(8.0, 184.0, 0.34),
            candidate(9.0, 183.0, 0.35),
            candidate(21.0, 130.0, 0.50),
            candidate(-2.0, 150.0, 0.45),
        ]
        seeds = StudioAdapter._optimizer_tuning_distinct_map_seeds(
            cloud, "coverage", variables, 256, limit=4
        )
        self.assertEqual(len(seeds), 4)
        self.assertAlmostEqual(seeds[0]["metrics"]["uniformity_spread_pct"], 0.30)
        xs = [item["variable_values"]["x"] for item in seeds]
        self.assertTrue(any(value > 5.0 for value in xs))
        self.assertTrue(any(value < -20.0 for value in xs))

    def test_optimizer_report_describes_plateau_and_final_bound_verification(self):
        result = {
            "settings": {},
            "finalists": [],
            "generated_candidate_count": 40,
            "feasible_candidate_count": 39,
            "rejected_candidate_count": 1,
            "distinct_finalist_count": 0,
            "parameter_count": 7,
            "final_sample_count": 1791,
            "targets": [{"id": "target"}],
            "elapsed_seconds": 12.0,
            "variables": [
                {"id": "position_z", "kind": "position_z", "label": "ΔZ Position Group A", "unit": "mm", "step": 0.5},
                {"id": "assembly_rz", "kind": "assembly_orientation_z", "label": "ΔRZ Assembly rotation Group A", "unit": "°", "step": 10.0},
            ],
            "search_strategy": {
                "outer_search": "multifidelity_design_map_active_set_reduced_search",
                "current_mode": "inner_solve",
                "worker_serial_retry_count": 2,
                "worker_serial_recovery_count": 2,
                "worker_serial_retry_details": [
                    {
                        "reason": "coordinate_mismatch",
                        "candidate_index": 211,
                        "registry_candidate_id": "candidate_000211",
                        "expected_raw": {"position_z": 0.5, "assembly_rz": 270.0},
                        "returned_raw": {"position_z": 0.0, "assembly_rz": -90.0},
                        "mismatches": [
                            {
                                "variable_id": "position_z",
                                "label": "ΔZ Position Group A",
                                "unit": "mm",
                                "expected_raw": 0.5,
                                "returned_raw": 0.0,
                                "expected_canonical": 0.5,
                                "returned_canonical": 0.0,
                            }
                        ],
                    },
                    {
                        "reason": "worker_error",
                        "candidate_index": 140,
                        "registry_candidate_id": "candidate_000140",
                        "expected_raw": {"position_z": 0.0, "assembly_rz": -90.0},
                        "error": "int() argument must be a string, a bytes-like object or a real number, not 'dict'",
                    },
                ],
            },
            "convergence": {
                "converged": True,
                "reason": "plateau stabilized",
                "primary_goal_saturated": True,
                "saturation_shortcut_used": True,
            },
            "saturation_basin_verification": {
                "performed": True,
                "complete": True,
                "available_saturated_count": 17,
                "seed_count": 5,
                "polished_seed_count": 5,
                "global_probe_count": 36,
                "improved": True,
            },
            "multi_fidelity": {
                "enabled": True,
                "scout_sample_count": 256,
                "full_sample_count": 1791,
                "scout_evaluations": 3072,
                "scout_feasible": 3072,
                "scout_map_base_evaluations": 2200,
                "scout_map_canonical_evaluations": 120,
                "scout_map_relation_evaluations": 480,
                "scout_map_symmetry_evaluations": 600,
                "scout_map_global_evaluations": 1000,
                "scout_map_adaptive_evaluations": 854,
                "scout_map_fine_recheck_evaluations": 18,
                "scout_map_basin_count": 18,
                "scout_map_saturated_count": 713,
                "scout_map_relation_guided": True,
                "scout_map_relation_pattern": "two_circular_coil_pair",
                "scout_map_relation_labels": [
                    "target-centred pair midpoint",
                    "spacing/radius sweep including s=R",
                ],
                "scout_field_stamp_enabled": True,
                "scout_field_stamp_coil_count": 2,
                "scout_field_stamp_resolution": 56,
                "scout_field_stamp_build_points": 479232,
                "scout_field_stamp_validation_rms_error_pct": 0.08,
                "scout_field_stamp_coarse_resolution": 40,
                "scout_field_stamp_coarse_build_points": 128000,
                "scout_field_stamp_coarse_validation_rms_error_pct": 0.21,
                "scout_field_stamp_fine_enabled": True,
                "scout_field_stamp_fine_resolution": 56,
                "scout_field_stamp_fine_build_points": 351232,
                "scout_field_stamp_fine_validation_rms_error_pct": 0.08,
                "scout_field_stamp_fine_recheck_evaluations": 18,
                "scout_field_stamp_sampled_basis_count": 21000,
                "scout_field_stamp_exact_fallback_count": 0,
                "scout_coordinate_count": 14,
                "scout_raw_structural_count": 16,
                "scout_collective_coordinate_labels": ["Common Circular diameter", "Differential ΔX Position"],
                "physical_seed_count": 18,
                "drc_probe_count": 91,
                "drc_probe_request_count": 146,
                "drc_probe_cache_hit_count": 55,
                "drc_clear_seed_count": 11,
                "drc_intersecting_seed_count": 7,
                "drc_boundary_refinement_count": 49,
                "drc_boundary_seed_count": 6,
                "drc_discarded_seed_count": 1,
                "physical_seed_feasible_count": 5,
                "physical_seed_repaired_count": 6,
                "physical_refinement_seed_count": 3,
                "physical_refinement_evaluations": 72,
                "fast_map_guided_exact_cleanup": True,
                "fast_exact_global_challenge_evaluations": 64,
                "promoted_variable_labels": ["Differential ΔX Position", "Common Circular diameter"],
            },
            "boundary_verification": {
                "performed": True,
                "endpoint_count": 14,
                "feasible_endpoint_count": 14,
                "improved": False,
                "best_endpoint_label": "Common Circular diameter",
                "best_endpoint_edge": "maximum",
                "best_endpoint_value": 188.5,
                "best_endpoint_unit": "mm",
                "best_endpoint_improved": False,
                "conditional_repolish_performed": True,
                "conditional_repolish_count": 1,
                "conditional_repolish_feasible_count": 1,
                "conditional_repolish_improved": False,
                "conditional_repolish_best_label": "Common Circular diameter",
                "conditional_repolish_best_edge": "maximum",
                "conditional_repolish_best_value": 188.5,
                "conditional_repolish_best_unit": "mm",
                "conditional_repolish_best_improved": False,
            },
        }
        report = StudioAdapter.optimizer_run_report(result)
        self.assertIn("Primary-goal saturation", report)
        self.assertIn("Worker resilience: 2 unexpected worker-integrity candidate(s) retried serially; 2 recovered", report)
        self.assertIn("Worker retry diagnostics (first up to 8):", report)
        self.assertIn("candidate_000211: coordinate mismatch", report)
        self.assertIn("Expected: ΔZ Position Group A=0.5 mm; ΔRZ Assembly rotation Group A=270 °", report)
        self.assertIn("Returned: ΔZ Position Group A=0 mm; ΔRZ Assembly rotation Group A=-90 °", report)
        self.assertIn("Mismatch: ΔZ Position Group A: expected 0.5 mm, returned 0 mm (canonical 0.5 vs 0)", report)
        self.assertIn("candidate_000140: worker error", report)
        self.assertIn("Worker error: int() argument must be a string", report)
        self.assertIn("Optimizer 2.2 cheap-first pipeline", report)
        self.assertIn("256/1791 target samples", report)
        self.assertIn("3072 simplified evaluations", report)
        self.assertIn("2200 pre-adaptive atlas evaluations", report)
        self.assertIn("18 fine-cache landmark rechecks", report)
        self.assertIn("two-level 40³ → 56³", report)
        self.assertIn("source-pose RMS checks 0.21% → 0.08%", report)
        self.assertIn("18 basin landmark recheck(s) at the fine level", report)
        self.assertIn("Stages A-D cheap atlas", report)
        self.assertIn("480 known-relation probes", report)
        self.assertIn("1000 global Sobol skeptic probes", report)
        self.assertIn("seeds only, not constraints", report)
        self.assertIn(
            "DRC accounting v2: 146 optimizer DRC request(s) = "
            "55 geometry-cache hit(s) + 91 unique geometry evaluation(s).",
            report,
        )
        self.assertIn("DRC feasibility map: 91 unique geometry-only setting(s) evaluated", report)
        self.assertIn("49 boundary-refinement request(s)", report)
        self.assertIn("Map-promoted freedoms", report)
        self.assertIn("Fast map-first cleanup", report)
        self.assertIn("only 64 additional global skeptic settings", report)
        self.assertIn("Perfect-coverage basin challenge", report)
        self.assertIn("polished 5 of 5 structurally distinct perfect-coverage seeds", report)
        self.assertIn("Final bound verification: checked 14", report)
        self.assertIn("Common Circular diameter maximum=188.5 mm", report)
        self.assertIn("did not beat the incumbent", report)
        self.assertIn("Conditional bound re-polish: pinned 1 promising endpoint", report)
        self.assertIn("still did not beat the incumbent after coupled re-polishing", report)
        self.assertNotIn("Accounting warning", report)

    def test_optimizer_report_explains_drc_rescue_search(self):
        result = {
            "settings": {},
            "finalists": [],
            "generated_candidate_count": 0,
            "feasible_candidate_count": 0,
            "rejected_candidate_count": 0,
            "distinct_finalist_count": 0,
            "parameter_count": 2,
            "final_sample_count": 80,
            "targets": [{"id": "target"}],
            "elapsed_seconds": 2.0,
            "search_strategy": {"outer_search": "cheap_first_map_finalist_validation"},
            "convergence": {"converged": False, "reason": "No finalist."},
            "multi_fidelity": {
                "enabled": True,
                "scout_sample_count": 80,
                "full_sample_count": 1791,
                "scout_coordinate_count": 2,
                "physical_seed_count": 8,
                "drc_probe_count": 26,
                "drc_clear_seed_count": 0,
                "drc_intersecting_seed_count": 8,
                "drc_boundary_refinement_count": 0,
                "drc_boundary_seed_count": 0,
                "drc_discarded_seed_count": 8,
                "drc_rescue_attempted": True,
                "drc_rescue_probe_count": 18,
                "drc_rescue_clear_count": 0,
                "drc_rescue_retained_count": 0,
                "drc_rescue_best_clearance_mm": -1.25,
                "drc_rescue_sobol_planned_count": 6,
                "drc_rescue_axis_planned_count": 9,
            },
            "failure_diagnostics": {
                "source_candidate": {
                    "tested": True,
                    "feasible": False,
                    "error": "DRC intersecting: Coil / Anatomy.",
                },
                "rejection_reasons": [],
                "rejection_examples": [],
            },
        }
        report = StudioAdapter.optimizer_run_report(result)
        self.assertIn("DRC rescue search", report)
        self.assertIn("18 additional geometry-only setting(s)", report)
        self.assertIn("no DRC-clear anchor was found", report)
        self.assertIn("Best signed clearance reached -1.25 mm", report)
        self.assertIn("protected 6 coupled Sobol probe(s)", report)
        self.assertIn("18-probe geometry-only rescue search", report)
        self.assertNotIn("no rejection reason was recorded", report.lower())

    def test_optimizer_inherited_fixed_drc_baseline_ignores_only_unaffected_source_blockers(self):
        fixed_row = {
            "status": "intersecting",
            "object_a_id": "coil_l3",
            "object_a_label": "L3",
            "object_b_id": "bust",
            "object_b_label": "Generic bust 1 / Physical shell",
            "object_a_region": None,
            "object_b_region": None,
            "signed_clearance_mm": -0.477605,
            "required_clearance_mm": 0.0,
            "rule": "Physical shell collision",
        }
        adjustable_row = {
            **fixed_row,
            "object_a_id": "coil_r1",
            "object_a_label": "R1",
            "signed_clearance_mm": -0.2,
        }
        drc = {"results": [fixed_row, adjustable_row]}
        baseline = StudioAdapter._optimizer_tuning_fixed_drc_baseline(
            drc, {"coil_r1", "coil_l1"}
        )
        self.assertEqual(len(baseline), 1)
        self.assertIn(
            StudioAdapter._optimizer_tuning_drc_result_key(fixed_row), baseline
        )
        reasons, clearance = StudioAdapter._optimizer_tuning_drc_summary(drc, baseline)
        self.assertEqual(reasons, ["DRC intersecting: R1 / Generic bust 1 / Physical shell."])
        self.assertAlmostEqual(clearance, -0.2)

        fixed_only_reasons, fixed_only_clearance = StudioAdapter._optimizer_tuning_drc_summary(
            {"results": [fixed_row]}, baseline
        )
        self.assertEqual(fixed_only_reasons, [])
        self.assertIsNone(fixed_only_clearance)

    def test_optimizer_inherited_fixed_drc_baseline_rejects_unexpected_worsening(self):
        baseline_row = {
            "status": "intersecting",
            "object_a_id": "coil_l3",
            "object_a_label": "L3",
            "object_b_id": "bust",
            "object_b_label": "Generic bust 1 / Physical shell",
            "object_a_region": None,
            "object_b_region": None,
            "signed_clearance_mm": -0.477605,
            "required_clearance_mm": 0.0,
            "rule": "Physical shell collision",
        }
        key = StudioAdapter._optimizer_tuning_drc_result_key(baseline_row)
        worsened = {**baseline_row, "signed_clearance_mm": -0.6}
        reasons, clearance = StudioAdapter._optimizer_tuning_drc_summary(
            {"results": [worsened]}, {key: baseline_row}
        )
        self.assertEqual(
            reasons,
            ["DRC intersecting: L3 / Generic bust 1 / Physical shell."],
        )
        self.assertAlmostEqual(clearance, -0.6)

    def test_optimizer_report_lists_inherited_fixed_drc_baseline(self):
        result = {
            "settings": {},
            "finalists": [],
            "generated_candidate_count": 0,
            "feasible_candidate_count": 0,
            "rejected_candidate_count": 0,
            "distinct_finalist_count": 0,
            "parameter_count": 2,
            "final_sample_count": 80,
            "targets": [{"id": "target"}],
            "elapsed_seconds": 1.0,
            "search_strategy": {"outer_search": "cheap_first_map_finalist_validation"},
            "convergence": {"converged": False, "reason": "No finalist."},
            "multi_fidelity": {
                "enabled": True,
                "scout_sample_count": 80,
                "full_sample_count": 1791,
                "physical_seed_count": 0,
                "drc_probe_count": 0,
                "inherited_fixed_drc_count": 1,
                "inherited_fixed_drc_violations": [{
                    "status": "intersecting",
                    "object_a_label": "L3",
                    "object_b_label": "Generic bust 1 / Physical shell",
                    "signed_clearance_mm": -0.477605,
                }],
            },
            "failure_diagnostics": {
                "source_candidate": {
                    "tested": True,
                    "feasible": True,
                    "inherited_fixed_drc_count": 1,
                },
                "rejection_reasons": [],
                "rejection_examples": [],
            },
        }
        report = StudioAdapter.optimizer_run_report(result)
        self.assertIn("Inherited fixed DRC baseline", report)
        self.assertIn("L3 / Generic bust 1 / Physical shell", report)
        self.assertIn("-0.477605 mm", report)
        self.assertIn("feasible for the participating structural search", report)

    def test_optimizer_drc_rescue_keeps_a_clear_anchor_in_fast_final_vet_pool(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn('progress_phase = "Reality map • locating physical DRC boundaries"', source)
        self.assertIn('work_detail="finding a DRC-clear rescue anchor"', source)
        self.assertIn('"fast": 18', source)
        self.assertIn('(\"rescue\", np.asarray(row, dtype=float))', source)
        self.assertIn("insert_at = min(3, len(drc_mapped_rows))", source)
        self.assertIn('"drc_rescue_retained_count": int(drc_rescue_retained_count)', source)
        self.assertIn('{"fast": 6, "balanced": 10, "full": 18}', source)
        self.assertLess(
            source.index("sobol_quota = min("),
            source.index("# Source-relative endpoints answer"),
        )
        self.assertIn('process_settings["_optimizer_inherited_fixed_drc"]', source)

    def test_optimizer_design_map_rng_uses_random_seed_after_basin_seed_rename(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn("seed=int(random_seed)", source)
        self.assertIn("int(random_seed) ^ 0x20B01", source)
        self.assertIn("int(random_seed) ^ (0xAD470 + local_index)", source)
        self.assertNotIn("seed=int(seed)", source)
        self.assertNotIn("outer_dimensions, seed ^ 0x20B01", source)
        self.assertNotIn("outer_dimensions, seed ^ (0xAD470 + local_index)", source)

    def test_optimizer_circular_stamp_rotation_canonicalization_ignores_axial_spin(self):
        identity = Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix()
        spun = Rotation.from_euler("xyz", [0.0, 0.0, 47.0], degrees=True).as_matrix()
        canonical_identity = StudioAdapter._optimizer_tuning_axisymmetric_rotation_matrix(identity)
        canonical_spun = StudioAdapter._optimizer_tuning_axisymmetric_rotation_matrix(spun)
        np.testing.assert_allclose(canonical_identity, canonical_spun, atol=1e-12)

        tilted = Rotation.from_euler("xyz", [12.0, -7.0, 31.0], degrees=True).as_matrix()
        tilted_spun = Rotation.from_rotvec(
            tilted[:, 2] * np.deg2rad(63.0)
        ).as_matrix() @ tilted
        canonical_tilted = StudioAdapter._optimizer_tuning_axisymmetric_rotation_matrix(tilted)
        canonical_tilted_spun = StudioAdapter._optimizer_tuning_axisymmetric_rotation_matrix(tilted_spun)
        np.testing.assert_allclose(canonical_tilted, canonical_tilted_spun, atol=1e-12)

    def test_optimizer_circular_orientation_distance_ignores_axial_spin(self):
        variables = [
            {
                "id": "rz", "kind": "orientation_z", "coil_ids": ["coil_a"],
                "min": -30.0, "max": 30.0,
                "_axisymmetric_circle_orientation": True,
            },
            {
                "id": "rx", "kind": "orientation_x", "coil_ids": ["coil_a"],
                "min": -30.0, "max": 30.0,
                "_axisymmetric_circle_orientation": True,
            },
        ]
        baseline = {
            "variable_values": {"rz": 0.0, "rx": 0.0},
            "transforms": {"coil_a": {"euler_deg": [0.0, 0.0, 0.0]}},
        }
        spun = {
            "variable_values": {"rz": 25.0, "rx": 0.0},
            "transforms": {"coil_a": {"euler_deg": [0.0, 0.0, 25.0]}},
        }
        tilted = {
            "variable_values": {"rz": 0.0, "rx": 10.0},
            "transforms": {"coil_a": {"euler_deg": [10.0, 0.0, 0.0]}},
        }
        self.assertAlmostEqual(
            StudioAdapter._optimizer_tuning_structural_distance(baseline, spun, variables),
            0.0,
            places=12,
        )
        self.assertGreater(
            StudioAdapter._optimizer_tuning_structural_distance(baseline, tilted, variables),
            0.1,
        )

    def test_optimizer_circular_orientation_screen_marks_source_axial_spin_null(self):
        variable = {
            "id": "rz", "kind": "orientation_z", "coil_ids": ["coil_a"],
            "min": -30.0, "max": 30.0,
            "_axisymmetric_circle_orientation": True,
        }
        baseline = {"transforms": {"coil_a": {"euler_deg": [0.0, 0.0, 0.0]}}}
        spin_probes = [
            {"transforms": {"coil_a": {"euler_deg": [0.0, 0.0, angle]}}}
            for angle in (-30.0, -15.0, 15.0, 30.0)
        ]
        tilt_probe = {"transforms": {"coil_a": {"euler_deg": [8.0, 0.0, 0.0]}}}
        self.assertTrue(
            StudioAdapter._optimizer_tuning_orientation_screen_is_axisymmetric_null(
                variable, baseline, spin_probes
            )
        )
        self.assertFalse(
            StudioAdapter._optimizer_tuning_orientation_screen_is_axisymmetric_null(
                variable, baseline, spin_probes + [tilt_probe]
            )
        )

    def test_optimizer_active_set_finalist_row_restores_frozen_source_values(self):
        variables = [
            {"id": "turns", "kind": "turns", "min": 192.0, "max": 448.0, "integer": True},
            {"id": "diameter", "kind": "circular_diameter", "min": 112.0, "max": 208.0},
            {"id": "spacing", "kind": "pair_spacing", "min": 56.0, "max": 104.0},
            {"id": "ry", "kind": "orientation_y", "min": -30.0, "max": 30.0},
        ]
        source = np.asarray([320.0, 160.0, 80.0, 0.0])
        candidate_values = {
            "turns": 384.0,
            "diameter": 208.0,
            "spacing": 103.57,
            "ry": 27.0,
        }
        row = StudioAdapter._optimizer_tuning_source_preserved_row(
            variables, source, {"diameter", "spacing"}, candidate_values
        )
        np.testing.assert_allclose(row, [320.0, 208.0, 103.57, 0.0])

    def test_optimizer_variable_increments_snap_to_source_anchored_lattice(self):
        position = {
            "id": "dx", "kind": "position_x", "min": -25.0, "max": 25.0,
            "base": 0.0, "step": 0.5, "step_origin": 0.0, "integer": False,
        }
        self.assertEqual(
            StudioAdapter._optimizer_tuning_normalize_variable_value(position, 1.24),
            1.0,
        )
        self.assertEqual(
            StudioAdapter._optimizer_tuning_normalize_variable_value(position, 1.26),
            1.5,
        )

        diameter = {
            "id": "diameter", "kind": "circular_diameter", "min": 101.5, "max": 188.5,
            "base": 145.0, "step": 0.5, "step_origin": 145.0, "integer": False,
        }
        self.assertEqual(
            StudioAdapter._optimizer_tuning_normalize_variable_value(diameter, 174.58),
            174.5,
        )

        turns = {
            "id": "turns", "kind": "turns", "min": 192.0, "max": 448.0,
            "base": 320.0, "step": 5.0, "step_origin": 320.0, "integer": True,
        }
        self.assertEqual(
            StudioAdapter._optimizer_tuning_normalize_variable_value(turns, 333.2),
            335.0,
        )

        # Backward compatibility: a non-integer variable from an older saved or
        # programmatic job remains continuous when it has no explicit increment.
        continuous = {"min": -1.0, "max": 1.0, "integer": False}
        self.assertAlmostEqual(
            StudioAdapter._optimizer_tuning_normalize_variable_value(continuous, 0.123456),
            0.123456,
        )

    def test_optimizer_collective_modes_expand_common_and_differential_independent_pair(self):
        variables = [
            {
                "id": "dx_lower", "kind": "position_x", "group_kind": "position", "group": "I:lower",
                "coil_ids": ["lower"], "min": -25.0, "max": 25.0, "base": 0.0,
                "label": "ΔX Position Independent lower", "unit": "mm", "integer": False,
            },
            {
                "id": "dx_upper", "kind": "position_x", "group_kind": "position", "group": "I:upper",
                "coil_ids": ["upper"], "min": -25.0, "max": 25.0, "base": 0.0,
                "label": "ΔX Position Independent upper", "unit": "mm", "integer": False,
            },
            {
                "id": "diameter_lower", "kind": "circular_diameter", "group_kind": "geometry", "group": "I:lower",
                "coil_ids": ["lower"], "min": 101.5, "max": 188.5, "base": 145.0,
                "label": "Circular diameter Independent lower", "unit": "mm", "integer": False,
            },
            {
                "id": "diameter_upper", "kind": "circular_diameter", "group_kind": "geometry", "group": "I:upper",
                "coil_ids": ["upper"], "min": 101.5, "max": 188.5, "base": 145.0,
                "label": "Circular diameter Independent upper", "unit": "mm", "integer": False,
            },
        ]
        modes = StudioAdapter._optimizer_tuning_collective_modes(variables)
        lookup = {(item["mode"], item["kind"]): item for item in modes}
        self.assertIn(("common", "position_x"), lookup)
        self.assertIn(("differential", "position_x"), lookup)
        self.assertIn(("common", "circular_diameter"), lookup)

        source = np.asarray([0.0, 0.0, 145.0, 145.0])
        common_x = StudioAdapter._optimizer_tuning_collective_row(
            variables, source, lookup[("common", "position_x")], -12.5
        )
        differential_x = StudioAdapter._optimizer_tuning_collective_row(
            variables, source, lookup[("differential", "position_x")], 5.0
        )
        common_diameter = StudioAdapter._optimizer_tuning_collective_row(
            variables, source, lookup[("common", "circular_diameter")], 188.5
        )
        np.testing.assert_allclose(common_x, [-12.5, -12.5, 145.0, 145.0])
        np.testing.assert_allclose(differential_x, [5.0, -5.0, 145.0, 145.0])
        np.testing.assert_allclose(common_diameter, [0.0, 0.0, 188.5, 188.5])

    def test_optimizer_differential_collective_variable_moves_pair_oppositely(self):
        candidate = StudioAdapter(compact_workflow_scene())
        variable = {
            "id": "collective:differential:position_x:test",
            "kind": "position_x",
            "group_kind": "collective",
            "group": "DIFFERENTIAL",
            "coil_ids": ["coil_a", "coil_b"],
            "min": -25.0,
            "max": 25.0,
            "base": 0.0,
            "label": "Differential ΔX Position",
            "unit": "mm",
            "integer": False,
            "_collective_coefficients": {"coil_a": 1.0, "coil_b": -1.0},
        }
        source_currents = {"coil_a": 0.07, "coil_b": 0.07}
        self.adapter._optimizer_tuning_apply_variables(
            candidate, [variable], [10.0], source_currents
        )
        lower = np.asarray(candidate.get_transform("coil_a")["position"], dtype=float)
        upper = np.asarray(candidate.get_transform("coil_b")["position"], dtype=float)
        np.testing.assert_allclose(lower, [0.01, 0.0, -0.05], atol=1e-12)
        np.testing.assert_allclose(upper, [-0.01, 0.0, 0.05], atol=1e-12)

    def test_optimizer_finalist_curation_removes_dominated_and_near_duplicate_results(self):
        variables = [
            {"id": "diameter", "kind": "circular_diameter", "min": 100.0, "max": 220.0},
            {"id": "spacing", "kind": "pair_spacing", "min": 50.0, "max": 110.0},
        ]

        def candidate(diameter, spacing, spread, rms, direction=100.0):
            return {
                "variable_values": {"diameter": diameter, "spacing": spacing, "current": 72.0},
                "metrics": {
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": spread,
                    "rms_target_error_uT": rms,
                    "directional_consistency_pct": direction,
                },
            }

        best = candidate(208.0, 104.0, 0.096, 0.056)
        dominated_nearby = candidate(208.0, 102.9, 0.104, 0.070)
        dominated_farther = candidate(196.0, 98.0, 0.125, 0.090)
        real_tradeoff = candidate(170.0, 85.0, 0.080, 0.40)
        curated, metadata = StudioAdapter._optimizer_tuning_curate_finalists(
            [best, dominated_nearby, dominated_farther, real_tradeoff],
            "coverage",
            variables,
            8,
            1791,
        )
        self.assertEqual(curated, [best, real_tradeoff])
        self.assertEqual(metadata["primary_tied_count"], 4)
        self.assertEqual(metadata["nondominated_count"], 2)

        capped, capped_metadata = StudioAdapter._optimizer_tuning_curate_finalists(
            [best, real_tradeoff],
            "coverage",
            variables,
            1,
            1791,
        )
        self.assertEqual(capped, [best])
        self.assertTrue(capped_metadata["limit_reached"])
        self.assertEqual(capped_metadata["finalist_limit"], 1)

    def test_optimizer_finalist_curation_collapses_field_equivalent_turn_current_families(self):
        variables = [
            {"id": "turns_lower", "kind": "turns", "label": "Turns lower", "min": 192.0, "max": 448.0, "base": 320.0, "unit": "turns", "integer": True},
            {"id": "turns_upper", "kind": "turns", "label": "Turns upper", "min": 192.0, "max": 448.0, "base": 320.0, "unit": "turns", "integer": True},
            {"id": "diameter", "kind": "circular_diameter", "label": "Diameter", "min": 100.0, "max": 220.0, "base": 145.0, "unit": "mm"},
            {"id": "current_lower", "kind": "current", "label": "Current lower", "min": 15.0, "max": 150.0, "base": 60.0, "unit": "mA"},
            {"id": "current_upper", "kind": "current", "label": "Current upper", "min": 15.0, "max": 150.0, "base": 60.0, "unit": "mA"},
        ]
        structural = [item for item in variables if item["kind"] != "current"]

        def candidate(turns_upper, current_upper, field_delta_uT=0.0):
            vectors = np.tile(np.asarray([[0.0, 0.0, (200.0 + field_delta_uT) * 1.0e-6]]), (8, 1))
            return {
                "mode": "tune_existing",
                "variables": copy.deepcopy(variables),
                "setting_summary": f"Turns upper={turns_upper} turns; Current upper={current_upper} mA; Diameter=188.5 mm",
                "variable_values": {
                    "turns_lower": 410.0,
                    "turns_upper": float(turns_upper),
                    "diameter": 188.5,
                    "current_lower": 51.0,
                    "current_upper": float(current_upper),
                },
                "currents_a": {"lower": 0.051, "upper": float(current_upper) / 1000.0},
                "field_vectors_t": vectors,
                "metrics": {
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": 0.13,
                    "rms_target_error_uT": 0.097,
                    "directional_consistency_pct": 99.99999,
                    "mean_uT": 200.0 + field_delta_uT,
                    "max_voltage_v": 2.6,
                    "total_power_w": 0.33,
                    "minimum_clearance_mm": 69.0,
                },
            }

        first = candidate(277, 75.6)
        equivalent = candidate(252, 83.1, field_delta_uT=0.0002)
        source_nearest = candidate(320, 65.5, field_delta_uT=0.0001)
        curated, metadata = StudioAdapter._optimizer_tuning_curate_finalists(
            [first, equivalent, source_nearest], "coverage", structural, 6, 1791
        )
        self.assertEqual(len(curated), 1)
        self.assertEqual(metadata["field_equivalent_removed"], 2)
        self.assertEqual(metadata["field_equivalent_family_count"], 1)
        self.assertGreaterEqual(metadata["field_equivalent_source_preferred_replacements"], 1)
        self.assertEqual(curated[0]["variable_values"]["turns_upper"], 320.0)
        self.assertEqual(curated[0]["field_equivalent_family_size"], 3)
        self.assertEqual(len(curated[0]["field_equivalent_variants"]), 2)
        variant_text = " ".join(item["setting_summary"] for item in curated[0]["field_equivalent_variants"])
        self.assertIn("277", variant_text)
        self.assertIn("252", variant_text)

    def test_optimizer_equivalent_result_simplicity_prefers_source_then_symmetry_then_axis(self):
        variables = [
            {
                "id": "x_a", "kind": "position_x", "coil_ids": ["a"],
                "min": -10.0, "max": 10.0, "base": 0.0,
            },
            {
                "id": "x_b", "kind": "position_x", "coil_ids": ["b"],
                "min": -10.0, "max": 10.0, "base": 0.0,
            },
        ]
        symmetric = {
            "variables": variables,
            "variable_values": {"x_a": 5.0, "x_b": 5.0},
            "transforms": {
                "a": {"euler_deg": [0.0, 0.0, 0.0]},
                "b": {"euler_deg": [0.0, 0.0, 0.0]},
            },
        }
        antisymmetric = {
            "variables": variables,
            "variable_values": {"x_a": 5.0, "x_b": -5.0},
            "transforms": copy.deepcopy(symmetric["transforms"]),
        }
        self.assertLess(
            StudioAdapter._optimizer_tuning_source_preference_key(symmetric),
            StudioAdapter._optimizer_tuning_source_preference_key(antisymmetric),
        )

        aligned = copy.deepcopy(symmetric)
        tilted = copy.deepcopy(symmetric)
        tilted["transforms"]["a"]["euler_deg"] = [15.0, 0.0, 0.0]
        self.assertLess(
            StudioAdapter._optimizer_tuning_source_preference_key(aligned),
            StudioAdapter._optimizer_tuning_source_preference_key(tilted),
        )

    def test_optimizer_finalist_curation_collapses_only_strictly_tied_structural_simplifications(self):
        variables = [
            {
                "id": "x_a", "kind": "position_x", "coil_ids": ["a"],
                "min": -25.0, "max": 25.0, "base": 0.0, "step": 0.5, "step_origin": 0.0,
            },
            {
                "id": "x_b", "kind": "position_x", "coil_ids": ["b"],
                "min": -25.0, "max": 25.0, "base": 0.0, "step": 0.5, "step_origin": 0.0,
            },
        ]

        def candidate(x_a, x_b, field_uT, spread, rms=0.10):
            return {
                "variables": copy.deepcopy(variables),
                "variable_values": {"x_a": float(x_a), "x_b": float(x_b)},
                "setting_summary": f"x_a={x_a}; x_b={x_b}",
                "transforms": {
                    "a": {"euler_deg": [0.0, 0.0, 0.0]},
                    "b": {"euler_deg": [0.0, 0.0, 0.0]},
                },
                "field_vectors_t": np.tile([[0.0, 0.0, float(field_uT) * 1.0e-6]], (16, 1)),
                "metrics": {
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": float(spread),
                    "rms_target_error_uT": float(rms),
                    "directional_consistency_pct": 100.0,
                },
            }

        # A true numerical tie may simplify to the cleaner common placement.
        asymmetric_tie = candidate(-20.0, -4.0, 200.0, 0.140, 0.10)
        symmetric_tie = candidate(-12.0, -12.0, 200.0, 0.140, 0.10)
        curated, metadata = StudioAdapter._optimizer_tuning_curate_finalists(
            [asymmetric_tie, symmetric_tie], "coverage", variables, 6, 1791
        )
        self.assertEqual(len(curated), 1)
        self.assertEqual(curated[0]["variable_values"], {"x_a": -12.0, "x_b": -12.0})
        self.assertEqual(metadata["canonical_equivalent_removed"], 1)
        self.assertEqual(metadata["canonical_simplicity_replacements"], 1)

        # A measurable magnetic improvement must survive even if the worse answer is simpler.
        better_asymmetric = candidate(-20.0, -4.0, 200.0, 0.120, 0.08)
        simpler_but_worse = candidate(-12.0, -12.0, 200.0005, 0.140, 0.10)
        curated, metadata = StudioAdapter._optimizer_tuning_curate_finalists(
            [better_asymmetric, simpler_but_worse], "coverage", variables, 6, 1791
        )
        self.assertEqual(curated, [better_asymmetric])
        self.assertEqual(metadata["canonical_equivalent_removed"], 0)

    def test_optimizer_turn_current_family_does_not_use_cache_scale_tolerance_for_finalists(self):
        structural = [
            {
                "id": "turns_upper", "kind": "turns", "label": "Turns upper",
                "unit": "turns", "integer": True, "min": 192.0, "max": 448.0,
                "base": 320.0, "coil_ids": ["upper"],
            }
        ]
        variables = structural + [
            {
                "id": "current_upper", "kind": "current", "label": "Current upper",
                "unit": "mA", "integer": False, "min": 15.0, "max": 150.0,
                "base": 60.0, "coil_ids": ["upper"],
            }
        ]

        def candidate(turns, current, field_uT):
            return {
                "variables": copy.deepcopy(variables),
                "variable_values": {
                    "turns_upper": float(turns),
                    "current_upper": float(current),
                },
                "setting_summary": f"turns={turns}; current={current}",
                "currents_a": {"upper": float(current) / 1000.0},
                "field_vectors_t": np.tile([[0.0, 0.0, float(field_uT) * 1.0e-6]], (12, 1)),
                "metrics": {
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": 0.13,
                    "rms_target_error_uT": abs(float(field_uT) - 200.0),
                    "directional_consistency_pct": 100.0,
                    "mean_uT": float(field_uT),
                    "max_voltage_v": 2.6,
                    "total_power_w": 0.33,
                    "minimum_clearance_mm": 69.0,
                },
            }

        altered = candidate(388, 53.9, 200.00)
        source_turns = candidate(320, 65.5, 200.10)
        self.assertFalse(
            StudioAdapter._optimizer_tuning_field_equivalent_turn_current_family(
                altered, source_turns, structural
            )
        )
        curated, metadata = StudioAdapter._optimizer_tuning_curate_finalists(
            [altered, source_turns], "coverage", structural, 6, 1791
        )
        self.assertEqual(len(curated), 1)
        self.assertEqual(curated[0]["variable_values"]["turns_upper"], 388.0)
        self.assertEqual(metadata["field_equivalent_removed"], 0)

    def test_optimizer_cheap_first_policy_applies_to_all_structural_dimensions(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn("cheap_first_policy = bool(", source)
        self.assertIn("outer_dimensions >= 1", source)
        self.assertIn('design_map_mode = bool(raw_settings.get("_optimizer_design_map", False)) and outer_dimensions >= 2', source)
        self.assertIn('if outer_dimensions == 2 and not design_map_mode:', source)
        self.assertIn('scout_budget_default = {"fast": 96, "balanced": 128, "full": 192}[search_effort]', source)
        self.assertIn('scout_budget_default = {"fast": 320, "balanced": 640, "full": 1280}[search_effort]', source)
        self.assertIn('scout_settings["_optimizer_use_field_stamps"] = True', source)
        self.assertIn('scout_adapter.set_coil_modeling_method("centreline")', source)
        self.assertIn('fallback_settings["_optimizer_design_map"] = False', source)
        self.assertIn('fallback_settings["_optimizer_use_field_stamps"] = False', source)
        self.assertIn('fallback_adapter.set_coil_modeling_method("centreline")', source)
        self.assertIn('"source_full_field_deferred": False', source)
        self.assertIn('multi_fidelity["source_full_field_deferred"] = True', source)
        self.assertIn('progress_phase = "Finalist vet • authoritative full-model validation"', source)
        self.assertIn('"final_validation_only": bool(cheap_first_vetted)', source)
        self.assertIn('"cheap_first_completed": bool(cheap_first_vetted)', source)
        self.assertIn('and not cheap_first_vetted', source)
        self.assertIn('"cheap_first_map_finalist_validation" if cheap_first_vetted', source)
        self.assertNotIn('refine_limit = {"fast": 0, "balanced": 2, "full": 5}', source)

    def test_optimizer_effort_spends_extra_budget_on_cheap_map_and_small_finalist_vet(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn('default_cap = min(8, max(6, requested_finalists))', source)
        self.assertIn('default_cap = min(14, max(10, requested_finalists + 4))', source)
        self.assertIn('default_cap = min(24, max(14, 2 * requested_finalists + 2))', source)
        self.assertIn('scout_budget_default = min(1024, max(640, 96 * max(4, scout_coordinate_count)))', source)
        self.assertIn('scout_budget_default = min(4096, max(2560, 192 * max(4, scout_coordinate_count)))', source)
        self.assertIn('max(4096, 768 * max(4, scout_coordinate_count))', source)
        self.assertIn('"fast": min(3, max(2, finalist_count))', source)
        self.assertIn('"balanced": min(5, max(3, finalist_count))', source)
        self.assertIn('"full": min(8, max(5, finalist_count + 2))', source)
        self.assertIn('intermediate_model = "bundled"', source)
        self.assertIn('"fast": 96, "balanced": 160, "full": 256', source)
        self.assertIn('authoritative_initial_validation_limit', source)
        self.assertIn('authoritative_progressive_extra_count', source)
        self.assertIn('"physical_refinement_seed_count": 0', source)
        self.assertIn('"physical_refinement_evaluations": 0', source)
        self.assertIn('low_dimensional_vet_neighbour_count', source)
        self.assertIn('physical_only_probe_count', source)
        self.assertIn('No exploratory full-model sweep, active-set search', source)

    def test_optimizer_report_marks_fast_budget_completion_and_lists_family_realizations(self):
        variables = [
            {"id": "turns", "kind": "turns", "label": "Turns upper", "unit": "turns", "integer": True},
            {"id": "current", "kind": "current", "label": "Current upper", "unit": "mA", "integer": False},
        ]
        finalist = {
            "id": "tuning_1",
            "mode": "tune_existing",
            "setting_summary": "Turns upper=277 turns; Current upper=75.6 mA",
            "variables": variables,
            "coil_labels": ["lower", "upper"],
            "currents_a": {"lower": 0.051, "upper": 0.0756},
            "metrics": {
                "mean_uT": 200.0, "min_uT": 199.5, "max_uT": 200.2,
                "in_band_pct": 100.0, "uniformity_spread_pct": 0.13,
                "directional_consistency_pct": 100.0, "rms_target_error_uT": 0.1,
                "max_voltage_v": 2.6, "total_power_w": 0.33,
                "minimum_clearance_mm": 69.0,
            },
            "field_equivalent_family_size": 2,
            "field_equivalent_variants": [
                {
                    "setting_summary": "Turns upper=252 turns; Current upper=83.1 mA",
                    "variable_values": {"turns": 252.0, "current": 83.1},
                    "currents_a": {"lower": 0.051, "upper": 0.0831},
                    "metrics": {"max_voltage_v": 2.5, "total_power_w": 0.34, "minimum_clearance_mm": 69.0},
                }
            ],
        }
        result = {
            "settings": {"search_effort": "fast", "optimization_plan": "OPTIMIZATION PLAN"},
            "variables": variables,
            "finalists": [finalist],
            "generated_candidate_count": 320, "feasible_candidate_count": 314, "rejected_candidate_count": 6,
            "distinct_finalist_count": 1, "parameter_count": 2, "final_sample_count": 1791,
            "targets": [{"id": "target"}], "elapsed_seconds": 392.0,
            "finalist_filter": {
                "primary_tied_count": 229, "nondominated_count": 35,
                "field_equivalent_removed": 1, "field_equivalent_family_count": 1,
                "finalist_limit": 6, "limit_reached": False,
            },
            "search_strategy": {"outer_search": "adaptive_global_multistart_coordinate_convergence", "outer_parameter_count": 1, "current_mode": "inner_solve"},
            "convergence": {
                "converged": False, "fast_budget_completed": True,
                "reason": "Fast search completed at the configured exact-search budget.",
            },
        }
        report = StudioAdapter.optimizer_run_report(result, top_n=1)
        self.assertIn("collapsed 1 strictly field-equivalent turn/current variant across 1 magnetic family", report)
        self.assertIn("Field-equivalent turn/current family: 2 electrical realizations", report)
        self.assertIn("Turns upper=252 turns", report)
        self.assertIn("Convergence: Fast search completed at configured budget", report)
        detail = StudioAdapter.optimizer_candidate_report(finalist)
        self.assertIn("FIELD-EQUIVALENT TURN/CURRENT FAMILY (2 realizations)", detail)
        self.assertIn("Current upper=83.1 mA", detail)

    def test_optimizer_finalist_curation_keeps_better_helmholtz_boundary_solution(self):
        variables = [
            {"id": "diameter", "kind": "circular_diameter", "min": 112.0, "max": 208.0, "base": 160.0, "coil_ids": ["a", "b"]},
            {"id": "spacing", "kind": "pair_spacing", "min": 56.0, "max": 104.0, "base": 80.0, "coil_ids": ["a", "b"]},
        ]

        def candidate(diameter, spacing, vectors_uT, spread, rms):
            vectors = np.asarray([[0.0, 0.0, value * 1.0e-6] for value in vectors_uT], dtype=float)
            return {
                "variables": copy.deepcopy(variables),
                "variable_values": {"diameter": float(diameter), "spacing": float(spacing)},
                "field_vectors_t": vectors,
                "metrics": {
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": float(spread),
                    "rms_target_error_uT": float(rms),
                    "directional_consistency_pct": 100.0,
                },
            }

        larger_helmholtz = candidate(208.0, 104.0, [199.98, 200.00, 200.02], 0.020, 0.016)
        source_like = candidate(160.0, 80.0, [199.90, 200.00, 200.10], 0.100, 0.082)
        curated, metadata = StudioAdapter._optimizer_tuning_curate_finalists(
            [larger_helmholtz, source_like], "coverage", variables, 6, 1791
        )
        self.assertEqual(curated, [larger_helmholtz])
        self.assertEqual(metadata["canonical_equivalent_removed"], 0)
        self.assertEqual(metadata["canonical_simplicity_replacements"], 0)

    def test_optimizer_current_only_curation_returns_one_excitation_answer(self):
        candidates = [
            {
                "variable_values": {"current": value},
                "metrics": {
                    "in_band_pct": 100.0,
                    "uniformity_spread_pct": 0.1,
                    "rms_target_error_uT": rms,
                    "directional_consistency_pct": 100.0,
                },
            }
            for value, rms in [(70.0, 0.5), (72.0, 0.05), (74.0, 0.6)]
        ]
        candidates.sort(
            key=lambda item: StudioAdapter._optimizer_tuning_current_objective_key(
                item["metrics"], "coverage"
            )
        )
        curated, _ = StudioAdapter._optimizer_tuning_curate_finalists(
            candidates, "coverage", [], 8, 1791
        )
        self.assertEqual(len(curated), 1)
        self.assertEqual(curated[0]["variable_values"]["current"], 72.0)

    def test_drc_progress_callback_reports_start_completion_and_streamed_results(self):
        sphere_id = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": sphere_id,
                        "role": "exclusion",
                        "colour": "#f59e0b",
                        "opacity": 0.28,
                    },
                }
            ]
        )
        updates = []
        report = self.adapter.run_drc(progress_callback=updates.append)
        self.assertIsInstance(report, dict)
        self.assertGreaterEqual(len(updates), 2)
        self.assertEqual(updates[0]["percent"], 0)
        self.assertEqual(updates[-1]["percent"], 100)
        self.assertEqual(updates[-1]["message"], "Design-rule check complete")
        self.assertEqual(
            [update["percent"] for update in updates],
            sorted(update["percent"] for update in updates),
        )
        streamed = [update["result"] for update in updates if "result" in update]
        self.assertEqual(len(streamed), len(report["results"]))
        self.assertEqual(
            {
                (
                    result["status"],
                    result["object_a_id"],
                    result.get("object_b_id"),
                    result["rule"],
                )
                for result in streamed
            },
            {
                (
                    result["status"],
                    result["object_a_id"],
                    result.get("object_b_id"),
                    result["rule"],
                )
                for result in report["results"]
            },
        )

    def test_default_scene_renders_and_has_expected_centre_field(self):
        adapter = StudioAdapter(default_scene())
        objects = {item["id"]: item for item in adapter.list_objects()}
        self.assertEqual(
            set(objects), {"group", "coil", "coil_1", "sensor", "guide_sphere"}
        )
        self.assertEqual(objects["group"]["label"], "Helmholtz Coils")
        self.assertEqual(objects["coil"]["parent"], "group")
        self.assertEqual(objects["coil_1"]["parent"], "group")
        self.assertEqual(objects["guide_sphere"]["role"], "measurement")
        self.assertEqual(adapter.coil_specs["coil"]["turns"], 320)
        self.assertEqual(adapter.coil_specs["coil_1"]["turns"], 320)
        self.assertEqual(len(adapter.snapshots), 1)
        self.assertEqual(adapter.coil_modeling_method, "auto")
        adapter.set_coil_modeling_method("centreline")
        centre = adapter.field_at_mm([0, 0, 0])
        self.assertAlmostEqual(centre["magnitude"][0], 1.997969706279882e-4, places=14)
        figure = adapter.figure()
        self.assertTrue(figure["data"])

    def test_edit_undo_redo(self):
        original = self.adapter.field_at_mm([0, 0, 0])["magnitude"][0]
        self.adapter.apply_operations(
            [{"method": "set_param", "params": {"object_id": "coil_a", "name": "current", "value": 0.14}}]
        )
        edited = self.adapter.field_at_mm([0, 0, 0])["magnitude"][0]
        self.assertGreater(edited, original)
        self.adapter.undo()
        self.assertAlmostEqual(self.adapter.field_at_mm([0, 0, 0])["magnitude"][0], original, places=16)
        self.adapter.redo()
        self.assertAlmostEqual(self.adapter.field_at_mm([0, 0, 0])["magnitude"][0], edited, places=16)


    def test_scene_recipe_validation_and_atomic_import(self):
        adapter = StudioAdapter({"objects": []})
        recipe = {
            "format": "fieldworkbench-scene-recipe",
            "version": 1,
            "assistant_message": "I made a symmetric first-pass pair around the target.",
            "assumptions": ["Spacing was not specified, so I used 160 mm between coil centers."],
            "suggested_next_steps": ["Inspect an axis sensor through the target before changing current."],
            "objects": [
                {"key": "pair", "type": "group", "name": "AI pair"},
                {
                    "key": "left",
                    "type": "circular_coil",
                    "name": "AI left coil",
                    "parent": "pair",
                    "position_mm": [-80.0, 0.0, 10.0],
                    "rotation_deg": [0.0, 90.0, 0.0],
                    "diameter_mm": 160.0,
                    "turns": 320,
                    "drive_current_mA": 55.5,
                    "awg": 28,
                },
                {
                    "type": "sphere",
                    "name": "AI target",
                    "position_mm": [0.0, 20.0, 30.0],
                    "diameter_mm": 40.0,
                    "role": "measurement",
                    "colour": "#123456",
                    "opacity": 0.5,
                },
            ],
        }
        text = json.dumps(recipe)
        before_signature = adapter.signature()
        history_before = adapter.history()["undo"]

        validation_progress = []
        validation = adapter.validate_scene_recipe(
            text, progress_callback=validation_progress.append
        )
        self.assertEqual(validation["object_count"], 3)
        self.assertEqual(validation_progress[0]["percent"], 0)
        self.assertEqual(validation_progress[-1]["percent"], 100)
        self.assertTrue(any("Built" in update["message"] for update in validation_progress))
        self.assertEqual(validation["warnings"], [])
        self.assertIn("symmetric first-pass", validation["assistant_message"])
        self.assertEqual(len(validation["assumptions"]), 1)
        self.assertEqual(len(validation["suggested_next_steps"]), 1)
        self.assertEqual(adapter.signature(), before_signature)
        self.assertEqual(adapter.history()["undo"], history_before)

        import_progress = []
        imported = adapter.import_scene_recipe(
            text, progress_callback=import_progress.append
        )
        self.assertEqual(len(imported["object_ids"]), 3)
        self.assertIn("symmetric first-pass", imported["assistant_message"])
        self.assertEqual(import_progress[0]["percent"], 0)
        self.assertEqual(import_progress[-1]["percent"], 100)
        self.assertTrue(any("Built" in update["message"] for update in import_progress))
        self.assertEqual(adapter.history()["undo"], history_before + 1)
        group_id, coil_id, sphere_id = imported["object_ids"]
        self.assertEqual(adapter.get_object(group_id)["label"], "AI pair")
        self.assertEqual(adapter.get_object(coil_id)["parent"], group_id)
        self.assertEqual(adapter.get_object(coil_id)["label"], "AI left coil")
        self.assertEqual(adapter.get_object(sphere_id)["label"], "AI target")
        self.assertEqual(adapter.geometry_properties(sphere_id)["role"], "measurement")
        self.assertEqual(adapter.geometry_properties(sphere_id)["colour"], "#123456")
        transform = adapter.get_transform(coil_id)
        np.testing.assert_allclose(transform["position"], [-0.08, 0.0, 0.01], atol=1e-12)
        diameter = next(
            param["value"] for param in adapter.get_params(coil_id) if param["name"] == "diameter"
        )
        self.assertAlmostEqual(diameter, 0.16, places=12)
        physical = adapter.coil_physical_properties(coil_id)
        self.assertEqual(physical["turns"], 320)
        self.assertEqual(physical["awg"], 28)
        self.assertAlmostEqual(physical["drive_current_a"], 0.0555, places=12)

        adapter.undo()
        self.assertEqual(adapter.signature(), before_signature)

    def test_scene_recipe_context_exports_full_coil_physics_and_workbench_derived_values(self):
        adapter = StudioAdapter({"objects": []})
        coil_id = adapter.add_template("circle")
        adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {"object_id": coil_id, "name": "diameter", "value": 0.16},
                },
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": coil_id,
                        "turns": 222,
                        "drive_current_a": 0.0275,
                        "awg": 30,
                        "enabled": True,
                        "field_scale_factor": 1.037,
                        "current_mode": "peak_sine",
                        "resistance_20_ohm_per_km": 351.25,
                        "lead_length_m": 0.35,
                        "enamel_radial_mm": 0.031,
                        "packing_factor": 0.68,
                        "temperature_c": 45.0,
                        "additional_mass_g": 12.5,
                        "physical_geometry_enabled": True,
                        "winding_axial_width_mm": 62.0,
                        "winding_radial_build_mode": "auto",
                        "winding_radial_build_mm": 0.0,
                        "bobbin_wall_thickness_mm": 2.0,
                        "flange_height_mm": 4.0,
                        "flange_thickness_mm": 2.0,
                    },
                },
            ]
        )

        summary = adapter.scene_recipe_current_scene_summary()
        record = next(item for item in summary["objects"] if item["id"] == coil_id)
        coil = record["coil"]
        self.assertEqual(coil["turns"], 222)
        self.assertAlmostEqual(coil["drive_current_mA"], 27.5, places=9)
        self.assertEqual(coil["current_mode"], "peak_sine")
        self.assertAlmostEqual(coil["lead_length_mm"], 350.0, places=9)
        self.assertAlmostEqual(coil["resistance_20_ohm_per_km"], 351.25, places=9)
        self.assertAlmostEqual(coil["enamel_radial_mm"], 0.031, places=9)
        self.assertAlmostEqual(coil["packing_factor"], 0.68, places=9)
        self.assertAlmostEqual(coil["temperature_c"], 45.0, places=9)
        self.assertAlmostEqual(coil["additional_mass_g"], 12.5, places=9)
        self.assertEqual(
            coil["construction"],
            {
                "enabled": True,
                "axial_width_mm": 62.0,
                "radial_build_mode": "auto",
                "radial_build_mm": 0.0,
                "bobbin_wall_thickness_mm": 2.0,
                "flange_height_mm": 4.0,
                "flange_thickness_mm": 2.0,
            },
        )
        derived = coil["workbench_derived"]
        self.assertGreater(derived["wire_length_m"], 0.0)
        self.assertGreater(derived["resistance_operating_ohm"], 0.0)
        self.assertGreater(derived["resistive_voltage_v"], 0.0)
        self.assertGreater(derived["copper_loss_w"], 0.0)
        self.assertGreater(derived["copper_mass_g"], 0.0)
        self.assertGreater(derived["current_density_a_mm2"], 0.0)
        self.assertIn("assembly", derived)
        self.assertTrue(derived["assembly"]["valid"])

    def test_scene_recipe_tolerates_null_parent_and_identity_group_transform(self):
        adapter = StudioAdapter({"objects": []})
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "objects": [
                    {
                        "type": "group",
                        "key": "pair",
                        "name": "Pair",
                        "parent": None,
                        "position_mm": [0, 0, 0],
                        "rotation_deg": [0, 0, 0],
                    },
                    {
                        "type": "racetrack_coil",
                        "name": "A",
                        "parent": "pair",
                        "position_mm": [-52.5, 0, 0],
                        "rotation_deg": [90, 0, 0],
                        "end_diameter_mm": 100,
                        "straight_length_mm": 100,
                        "turns": 222,
                        "drive_current_mA": 0,
                        "awg": 18,
                        "enabled": False,
                        "construction": {"enabled": True, "axial_width_mm": 70},
                    },
                ],
            }
        )
        validation = adapter.validate_scene_recipe(recipe)
        self.assertEqual(validation["object_count"], 2)
        self.assertTrue(any("parent was null" in warning for warning in validation["warnings"]))
        self.assertTrue(any("ignored identity position_mm" in warning for warning in validation["warnings"]))
        self.assertTrue(any("ignored identity rotation_deg" in warning for warning in validation["warnings"]))

        result = adapter.import_scene_recipe(recipe)
        group_id, coil_id = result["object_ids"]
        self.assertIsNone(adapter.get_object(group_id).get("parent"))
        self.assertEqual(adapter.get_object(coil_id)["parent"], group_id)
        physical = adapter.coil_physical_properties(coil_id)
        self.assertTrue(physical["physical_geometry_enabled"])
        self.assertAlmostEqual(physical["winding_axial_width_mm"], 70.0, places=9)
        self.assertTrue(physical["calculations"]["assembly"]["valid"])

    def test_scene_recipe_zero_or_null_wire_resistance_uses_nominal_awg_with_warning(self):
        for supplied in (0, None):
            with self.subTest(supplied=supplied):
                adapter = StudioAdapter({"objects": []})
                recipe = json.dumps(
                    {
                        "format": "fieldworkbench-scene-recipe",
                        "version": 1,
                        "objects": [
                            {
                                "type": "circular_coil",
                                "name": "Nominal AWG fallback",
                                "diameter_mm": 160,
                                "turns": 100,
                                "drive_current_mA": 25,
                                "awg": 28,
                                "resistance_20_ohm_per_km": supplied,
                            }
                        ],
                    }
                )
                validation = adapter.validate_scene_recipe(recipe)
                self.assertEqual(validation["object_count"], 1)
                self.assertEqual(len(validation["warnings"]), 1)
                self.assertIn("using nominal", validation["warnings"][0])
                result = adapter.import_scene_recipe(recipe)
                physical = adapter.coil_physical_properties(result["object_ids"][0])
                self.assertAlmostEqual(
                    physical["resistance_20_ohm_per_km"],
                    awg_resistance_20_ohm_per_km(28),
                    places=9,
                )

    def test_scene_recipe_accepts_nested_coil_context_alias_and_flattens_it(self):
        adapter = StudioAdapter({"objects": []})
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "objects": [
                    {
                        "type": "racetrack_coil",
                        "name": "External AI nested coil",
                        "position_mm": [0.0, 17.5, 0.0],
                        "rotation_deg": [90.0, 0.0, 0.0],
                        "end_diameter_mm": 100.0,
                        "straight_length_mm": 100.0,
                        "coil": {
                            "turns": 222,
                            "drive_current_mA": 83.79,
                            "awg": 18,
                            "current_mode": "rms_dc",
                            "construction": {
                                "enabled": True,
                                "axial_width_mm": 62.0,
                                "radial_build_mode": "auto",
                                "radial_build_mm": 0.348,
                                "bobbin_wall_thickness_mm": 2.0,
                                "flange_height_mm": 2.0,
                                "flange_thickness_mm": 2.0,
                            },
                            "workbench_derived": {
                                "resistance_operating_ohm": 123.456
                            },
                        },
                    }
                ],
            }
        )
        validation = adapter.validate_scene_recipe(recipe)
        self.assertEqual(validation["object_count"], 1)
        self.assertTrue(any("flattened" in warning for warning in validation["warnings"]))
        self.assertTrue(any("workbench_derived" in warning for warning in validation["warnings"]))

        result = adapter.import_scene_recipe(recipe)
        physical = adapter.coil_physical_properties(result["object_ids"][0])
        self.assertEqual(physical["turns"], 222)
        self.assertEqual(physical["awg"], 18)
        self.assertAlmostEqual(physical["drive_current_a"], 0.08379, places=12)
        self.assertTrue(physical["physical_geometry_enabled"])
        self.assertAlmostEqual(physical["winding_axial_width_mm"], 62.0, places=9)

    def test_scene_recipe_rejects_conflicting_nested_and_flat_coil_values(self):
        adapter = StudioAdapter({"objects": []})
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "objects": [
                    {
                        "type": "circular_coil",
                        "diameter_mm": 160.0,
                        "turns": 100,
                        "coil": {"turns": 222},
                    }
                ],
            }
        )
        with self.assertRaisesRegex(StudioOperationError, "conflicting values"):
            adapter.validate_scene_recipe(recipe)

    def test_scene_recipe_clipboard_guide_stays_compact(self):
        for guidance_level in ("exploratory", "balanced", "strict"):
            with self.subTest(guidance_level=guidance_level):
                base = scene_recipe_guide(
                    import_mode="replace", guidance_level=guidance_level
                )
                self.assertLess(len(base), 11000)
                self.assertIn("SCENE INTERACTION / AUTHORITY", base)
                self.assertIn("frozen launch-time scene copies", base)
                self.assertIn("Manual/WMM2025 Background fields", base)
                self.assertIn("complete-scene replacement", base)

        context = {
            "objects": [
                {
                    "id": "coil-1",
                    "name": "Physical racetrack",
                    "recipe_type_hint": "racetrack_coil",
                    "position_mm": [0, 0, 0],
                    "rotation_deg": [0, 0, 0],
                    "coil": {
                        "turns": 222,
                        "drive_current_mA": 27.5,
                        "awg": 30,
                        "construction": {"enabled": True, "axial_width_mm": 62.0},
                        "workbench_derived": {
                            "wire_length_m": 114.3,
                            "resistance_operating_ohm": 39.4,
                            "resistive_voltage_v": 1.084,
                            "copper_loss_w": 0.0298,
                        },
                    },
                }
            ]
        }
        for guidance_level in ("exploratory", "balanced", "strict"):
            with self.subTest(scene_guidance_level=guidance_level):
                with_scene = scene_recipe_guide(
                    context,
                    import_mode="replace",
                    guidance_level=guidance_level,
                )
                self.assertLess(len(with_scene), 12000)

        with self.assertRaises(ValueError):
            scene_recipe_guide(guidance_level="paranoid-foreman")

    def test_scene_recipe_review_questions_are_nonblocking(self):
        adapter = StudioAdapter({"objects": []})
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "assistant_message": "A physically plausible first pass is buildable.",
                "review_questions": [
                    "Confirm whether the source spacing is centre-to-centre before fabrication."
                ],
                "objects": [
                    {
                        "type": "circular_coil",
                        "name": "Reviewable coil",
                        "diameter_mm": 120,
                    }
                ],
            }
        )
        validation = adapter.validate_scene_recipe(recipe)
        self.assertFalse(validation["requires_clarification"])
        self.assertEqual(len(validation["review_questions"]), 1)
        self.assertEqual(validation["object_count"], 1)
        result = adapter.import_scene_recipe(recipe)
        self.assertEqual(len(result["object_ids"]), 1)
        self.assertEqual(len(result["review_questions"]), 1)

    def test_scene_recipe_clarification_request_validates_but_cannot_import(self):
        adapter = StudioAdapter({"objects": []})
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "assistant_message": (
                    "I know the racetrack dimensions, but the source text does not uniquely "
                    "determine the pair orientation or separation axis."
                ),
                "requires_clarification": True,
                "clarifying_questions": [
                    "Are the two racetrack coil planes parallel?",
                    "Along which axis are their centres separated, and is the stated spacing centre-to-centre or face-to-face?",
                ],
                "objects": [],
            }
        )
        validation = adapter.validate_scene_recipe(recipe)
        self.assertTrue(validation["requires_clarification"])
        self.assertEqual(validation["object_count"], 0)
        self.assertEqual(len(validation["clarifying_questions"]), 2)
        with self.assertRaisesRegex(StudioOperationError, "requires clarification"):
            adapter.import_scene_recipe(recipe)

    def test_scene_recipe_clarification_request_rejects_speculative_objects(self):
        adapter = StudioAdapter({"objects": []})
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "requires_clarification": True,
                "clarifying_questions": ["Which plane should the racetrack coil occupy?"],
                "objects": [
                    {
                        "type": "racetrack_coil",
                        "end_diameter_mm": 100,
                        "straight_length_mm": 100,
                    }
                ],
            }
        )
        with self.assertRaisesRegex(StudioOperationError, "must be empty"):
            adapter.validate_scene_recipe(recipe)

    def test_scene_recipe_replace_round_trip_preserves_editable_coil_physics(self):
        adapter = StudioAdapter({"objects": []})
        coil_id = adapter.add_template("circle")
        adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {"object_id": coil_id, "name": "diameter", "value": 0.16},
                },
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": coil_id,
                        "turns": 222,
                        "drive_current_a": 0.0275,
                        "awg": 30,
                        "enabled": False,
                        "field_scale_factor": 1.037,
                        "current_mode": "peak_sine",
                        "resistance_20_ohm_per_km": 351.25,
                        "lead_length_m": 0.35,
                        "enamel_radial_mm": 0.031,
                        "packing_factor": 0.68,
                        "temperature_c": 45.0,
                        "additional_mass_g": 12.5,
                        "physical_geometry_enabled": True,
                        "winding_axial_width_mm": 62.0,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 1.8,
                        "bobbin_wall_thickness_mm": 2.0,
                        "flange_height_mm": 4.0,
                        "flange_thickness_mm": 2.0,
                    },
                },
            ]
        )
        before = adapter.coil_physical_properties(coil_id)
        context = adapter.scene_recipe_current_scene_summary()
        record = next(item for item in context["objects"] if item["id"] == coil_id)
        coil = record["coil"]
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "objects": [
                    {
                        "type": "circular_coil",
                        "name": "Identity replacement coil",
                        "position_mm": record["position_mm"],
                        "rotation_deg": record["rotation_deg"],
                        "diameter_mm": coil["diameter_mm"],
                        "turns": coil["turns"],
                        "drive_current_mA": coil["drive_current_mA"],
                        "awg": coil["awg"],
                        "enabled": coil["enabled"],
                        "field_scale_factor": coil["field_scale_factor"],
                        "current_mode": coil["current_mode"],
                        "lead_length_mm": coil["lead_length_mm"],
                        "resistance_20_ohm_per_km": coil["resistance_20_ohm_per_km"],
                        "enamel_radial_mm": coil["enamel_radial_mm"],
                        "packing_factor": coil["packing_factor"],
                        "temperature_c": coil["temperature_c"],
                        "additional_mass_g": coil["additional_mass_g"],
                        "construction": coil["construction"],
                    }
                ],
            }
        )

        validation = adapter.validate_scene_recipe(recipe, mode="replace")
        self.assertEqual(validation["object_count"], 1)
        result = adapter.import_scene_recipe(recipe, mode="replace")
        new_id = result["object_ids"][0]
        after = adapter.coil_physical_properties(new_id)
        for key in (
            "enabled",
            "turns",
            "drive_current_a",
            "field_scale_factor",
            "awg",
            "resistance_20_ohm_per_km",
            "lead_length_m",
            "enamel_radial_mm",
            "packing_factor",
            "temperature_c",
            "additional_mass_g",
            "current_mode",
            "physical_geometry_enabled",
            "winding_radial_build_mode",
            "winding_radial_build_mm",
            "winding_axial_width_mm",
            "bobbin_wall_thickness_mm",
            "flange_height_mm",
            "flange_thickness_mm",
        ):
            if isinstance(before[key], float):
                self.assertAlmostEqual(after[key], before[key], places=9, msg=key)
            else:
                self.assertEqual(after[key], before[key], msg=key)

    def test_scene_recipe_replace_is_atomic_and_preserves_scene_wide_context(self):
        adapter = StudioAdapter({"objects": []})
        old_id = adapter.add_template("guide_sphere")
        adapter.set_drc_settings(
            {
                "default_clearance_mm": 4.0,
                "coil_to_coil_clearance_mm": 2.0,
                "mesh_tolerance_mm": 0.1,
            }
        )
        adapter.set_scene_notes("Keep this note")
        before = adapter.document()
        history_before = adapter.history()["undo"]
        recipe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "objects": [
                    {
                        "type": "circular_coil",
                        "name": "Replacement coil",
                        "diameter_mm": 100,
                        "drive_current_mA": 20,
                    }
                ],
            }
        )

        validation = adapter.validate_scene_recipe(recipe, mode="replace")
        self.assertEqual(validation["mode"], "replace")
        self.assertEqual(validation["object_count"], 1)
        self.assertIsNotNone(adapter.get_object(old_id))

        result = adapter.import_scene_recipe(recipe, mode="replace")
        self.assertEqual(result["mode"], "replace")
        self.assertEqual(adapter.history()["undo"], history_before + 1)
        labels = [item.get("label") for item in adapter.list_objects() if not item.get("derived")]
        self.assertEqual(labels, ["Replacement coil"])
        self.assertEqual(adapter.get_scene_notes(), "Keep this note")
        self.assertEqual(adapter.get_drc_settings()["default_clearance_mm"], 4.0)

        adapter.undo()
        self.assertEqual(adapter.document(), before)

    def test_scene_recipe_context_can_include_fresh_drc_rules_and_results(self):
        adapter = StudioAdapter({"objects": []})
        bust_id = adapter.add_template("generic_bust")
        adapter.set_drc_settings(
            {
                "default_clearance_mm": 3.0,
                "coil_to_coil_clearance_mm": 1.0,
                "mesh_tolerance_mm": 0.1,
            }
        )
        progress_updates = []
        summary = adapter.scene_recipe_current_scene_summary(
            include_drc=True, progress_callback=progress_updates.append
        )
        self.assertIn("design_rules", summary)
        self.assertIn("design_check", summary)
        self.assertTrue(summary["design_check"]["fresh"])
        self.assertFalse(summary["design_check"]["reused_fresh_report"])
        self.assertIsNotNone(adapter.fresh_drc_report())
        self.assertEqual(summary["design_rules"]["scene"]["default_clearance_mm"], 3.0)
        surface = next(
            item for item in summary["design_rules"]["surface_objects"]
            if item["id"] == bust_id
        )
        regions = {item["name"]: item for item in surface["regions"]}
        self.assertTrue(regions["Scalp"]["placement_surface"])
        self.assertTrue(regions["Scalp"]["drc_exclusion"])
        self.assertTrue(regions["Face"]["drc_exclusion"])
        self.assertTrue(progress_updates)
        self.assertEqual(progress_updates[-1]["percent"], 94)
        self.assertTrue(any(update.get("phase") == "drc" for update in progress_updates))
        self.assertTrue(
            any("Checking" in str(update.get("message", "")) for update in progress_updates)
        )
        cached_progress = []
        with mock.patch.object(
            adapter,
            "run_drc",
            side_effect=AssertionError("fresh cached DRC should have been reused"),
        ):
            cached_summary = adapter.scene_recipe_current_scene_summary(
                include_drc=True, progress_callback=cached_progress.append
            )
        self.assertTrue(cached_summary["design_check"]["reused_fresh_report"])
        self.assertTrue(
            any("cached" in str(update.get("message", "")).lower() for update in cached_progress)
        )

        guide_progress = []
        guide = adapter.scene_recipe_help_text(
            include_current_scene=True,
            include_drc=True,
            import_mode="replace",
            progress_callback=guide_progress.append,
        )
        self.assertIsInstance(guide, str)
        self.assertTrue(guide)
        self.assertEqual(guide_progress[0]["percent"], 0)
        self.assertEqual(guide_progress[-1]["percent"], 98)
        self.assertTrue(any(update.get("phase") == "drc" for update in guide_progress))
        self.assertTrue(any(update.get("phase") == "scene_context" for update in guide_progress))

        adapter.add_template("guide_sphere")
        self.assertIsNone(adapter.fresh_drc_report())

    def test_scene_recipe_rejects_unknown_fields_and_uses_safe_missing_source_defaults(self):
        adapter = StudioAdapter({"objects": []})
        bad = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "objects": [{"type": "circular_coil", "python": "print('nope')"}],
            }
        )
        with self.assertRaisesRegex(StudioOperationError, "unsupported field"):
            adapter.validate_scene_recipe(bad)

        bad_advice = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "assumptions": "this should be an array",
                "objects": [],
            }
        )
        with self.assertRaisesRegex(StudioOperationError, "array of text strings"):
            adapter.validate_scene_recipe(bad_advice)

        safe = json.dumps(
            {
                "format": "fieldworkbench-scene-recipe",
                "version": 1,
                "objects": [
                    {"type": "circular_coil", "diameter_mm": 80},
                    {"type": "sphere_magnet", "diameter_mm": 10},
                ],
            }
        )
        result = adapter.validate_scene_recipe(safe)
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(len(result["warnings"]), 2)
        self.assertIn("0 mA", result["warnings"][0])
        self.assertIn("[0,0,0] T", result["warnings"][1])

    def test_import_selected_objects_from_scene_and_multi_delete_are_single_undo_steps(self):
        source = StudioAdapter({"objects": []})
        group_id = source.add_template("collection")
        coil_id = source.add_template("circle", parent=group_id)
        sphere_id = source.add_template("guide_sphere", parent=group_id)
        source.coil_specs[coil_id]["turns"] = 37
        source.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": coil_id,
                        "position": [0.012, -0.034, 0.056],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": sphere_id,
                        "role": "exclusion",
                        "colour": "#123456",
                        "opacity": 0.42,
                    },
                },
            ]
        )

        target = StudioAdapter({"objects": []})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.magpy.json"
            source.save_file(path)
            catalog = target.scene_object_catalog_from_file(path)
            self.assertEqual({item["id"] for item in catalog}, {group_id, coil_id, sphere_id})

            history_before = target.history()["undo"]
            imported = target.import_objects_from_file(path, [group_id])
            self.assertEqual(len(imported), 3)
            self.assertEqual(target.history()["undo"], history_before + 1)
            imported_objects = {item["id"]: item for item in target.list_objects()}
            imported_group = next(
                object_id for object_id in imported if imported_objects[object_id]["type"] == "Collection"
            )
            imported_coil = next(
                object_id for object_id in imported if imported_objects[object_id]["type"] == "current.Circle"
            )
            imported_sphere = next(
                object_id for object_id in imported if imported_objects[object_id].get("workbench_geometry")
            )
            self.assertEqual(imported_objects[imported_coil]["parent"], imported_group)
            self.assertEqual(imported_objects[imported_sphere]["parent"], imported_group)
            self.assertEqual(target.coil_specs[imported_coil]["turns"], 37)
            self.assertEqual(target.geometry_properties(imported_sphere)["role"], "exclusion")
            np.testing.assert_allclose(
                target.get_transform(imported_coil)["position"],
                [0.012, -0.034, 0.056],
                atol=1e-12,
            )

            remove_history = target.history()["undo"]
            removed = target.remove_many([imported_coil, imported_sphere])
            self.assertEqual(set(removed), {imported_coil, imported_sphere})
            self.assertEqual(target.history()["undo"], remove_history + 1)
            self.assertEqual(
                {item["id"] for item in target.list_objects()},
                {imported_group},
            )
            target.undo()
            self.assertEqual(
                {item["id"] for item in target.list_objects()},
                {imported_group, imported_coil, imported_sphere},
            )

            # Importing a child without its source group intentionally makes it
            # top-level while retaining its world placement.
            standalone = StudioAdapter({"objects": []})
            imported_child = standalone.import_objects_from_file(path, [coil_id])[0]
            self.assertIsNone(standalone.get_object(imported_child).get("parent"))
            np.testing.assert_allclose(
                standalone.get_transform(imported_child)["position"],
                [0.012, -0.034, 0.056],
                atol=1e-12,
            )

    def test_scene_notes_roundtrip_and_coalesced_undo(self):
        self.adapter.set_scene_notes("Initial build note")
        self.assertEqual(self.adapter.get_scene_notes(), "Initial build note")
        self.assertEqual(
            self.adapter.document()["field_workbench"]["notes"],
            "Initial build note",
        )

        original = self.adapter.get_scene_notes()
        self.adapter.set_scene_notes("Initial build note\nTurn count checked", record_history=False)
        self.adapter.set_scene_notes(
            "Initial build note\nTurn count checked\nMeasured spacing: 89 mm",
            record_history=False,
        )
        self.adapter.commit_scene_notes_edit(original)
        self.adapter.undo()
        self.assertEqual(self.adapter.get_scene_notes(), original)
        self.adapter.redo()
        self.assertIn("Measured spacing: 89 mm", self.adapter.get_scene_notes())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.magpy.json"
            self.adapter.save_file(path)
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(path)
            self.assertEqual(reopened.get_scene_notes(), self.adapter.get_scene_notes())

    def test_add_duplicate_hide_and_remove(self):
        new_id = self.adapter.add_template("circle")
        self.assertTrue(self.adapter.coil_physical_properties(new_id)["enabled"])
        self.assertEqual(
            self.adapter.coil_physical_properties(new_id)["current_mode"], "rms_dc"
        )
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties(new_id)["field_scale_factor"], 1.0
        )
        copy_id = self.adapter.duplicate(new_id)
        self.assertIn(copy_id, {item["id"] for item in self.adapter.list_objects()})
        self.adapter.set_visible(copy_id, False)
        self.assertFalse(self.adapter.get_object(copy_id)["visible"])
        self.adapter.remove(copy_id)
        self.assertNotIn(copy_id, {item["id"] for item in self.adapter.list_objects()})

    def test_clone_repairs_reused_studio_object_id_history(self):
        # Older Workbench builds could delete a Studio object and later reuse the
        # same id. magpylib-studio keeps both lifetimes in its event log; copying
        # the current object would then replay the old remove onto the clone and
        # fail on the following transform with "targets unknown object".
        adapter = StudioAdapter(
            {
                "version": 1,
                "objects": [
                    {
                        "id": "coil",
                        "type": "current.Circle",
                        "params": {"diameter": 0.1, "current": 0.07},
                        "style": {"label": "Circular coil 1"},
                    }
                ],
                "events": [
                    {
                        "id": "e1",
                        "op": "create",
                        "target": "coil",
                        "type": "current.Circle",
                        "params": {"diameter": 0.1, "current": 0.07},
                        "style": {"label": "Old circular coil"},
                    },
                    {"id": "e2", "op": "position", "target": "coil", "value": [0.1, 0, 0]},
                    {"id": "e3", "op": "remove", "target": "coil"},
                    {
                        "id": "e4",
                        "op": "create",
                        "target": "coil",
                        "type": "current.Circle",
                        "params": {"diameter": 0.1, "current": 0.07},
                        "style": {"label": "Circular coil 1"},
                    },
                    {"id": "e5", "op": "position", "target": "coil", "value": [-0.1, 0, 0]},
                    {"id": "e6", "op": "orientation", "target": "coil", "rotvec": [0, 90, 0]},
                ],
            }
        )

        clone_id = adapter.duplicate("coil")
        self.assertIn(clone_id, {item["id"] for item in adapter.list_objects()})
        np.testing.assert_allclose(
            adapter.get_transform(clone_id)["position"],
            adapter.get_transform("coil")["position"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            adapter.get_transform(clone_id)["orientation"],
            adapter.get_transform("coil")["orientation"],
            atol=1e-12,
        )
        current_events = adapter.document().get("events", [])
        self.assertEqual(
            sum(
                1
                for event in current_events
                if event.get("op") == "create" and event.get("target") == "coil"
            ),
            1,
        )

    def test_deleted_studio_ids_remain_reserved_for_new_objects(self):
        adapter = StudioAdapter({"objects": []})
        first_id = adapter.add_template("circle")
        adapter.remove(first_id)
        second_id = adapter.add_template("circle")
        self.assertNotEqual(second_id, first_id)
        clone_id = adapter.duplicate(second_id)
        self.assertNotEqual(clone_id, first_id)
        self.assertIn(clone_id, {item["id"] for item in adapter.list_objects()})

    def test_display_trace_normalization_suppresses_only_hidden_coincident_lines(self):
        visible = {
            "type": "scatter3d",
            "mode": "lines",
            "x": [-100.0, -100.0, -100.0],
            "y": [-50.0, 0.0, 50.0],
            "z": [0.0, 50.0, 0.0],
            "showlegend": True,
            "name": "Collection (2 sources)",
        }
        duplicate = copy.deepcopy(visible)
        duplicate["showlegend"] = False
        duplicate["name"] = "duplicate source"
        distinct = copy.deepcopy(visible)
        distinct["x"] = [100.0, 100.0, 100.0]
        distinct["showlegend"] = False

        normalized = StudioAdapter._suppress_coincident_hidden_legend_lines(
            [visible, duplicate, distinct]
        )
        self.assertEqual(len(normalized), 2)
        self.assertIs(normalized[0], visible)
        self.assertIs(normalized[1], distinct)

        # Two hidden click targets on the exact same coil path collapse to one,
        # but the translated target remains available once the clone is moved.
        proxy_a = copy.deepcopy(duplicate)
        proxy_b = copy.deepcopy(duplicate)
        proxy_moved = copy.deepcopy(distinct)
        proxies = StudioAdapter._suppress_coincident_hidden_legend_lines(
            [proxy_a, proxy_b, proxy_moved]
        )
        self.assertEqual(len(proxies), 2)
        self.assertEqual(proxies[0]["x"], [-100.0, -100.0, -100.0])
        self.assertEqual(proxies[1]["x"], [100.0, 100.0, 100.0])

    def test_builtin_well_plates_use_exact_centred_measurement_grids(self):
        expectations = {
            "well_plate_6": (6, 2, 3, 39.12),
            "well_plate_12": (12, 3, 4, 26.01),
            "well_plate_24": (24, 4, 6, 19.30),
            "well_plate_48": (48, 6, 8, 13.08),
            "well_plate_96": (96, 8, 12, 9.00),
        }
        for template_name, (wells, rows, columns, pitch_mm) in expectations.items():
            with self.subTest(template=template_name):
                plate_id = self.adapter.add_template(template_name)
                obj = self.adapter.get_object(plate_id)
                self.assertEqual(obj["builtin_template"], template_name)
                self.assertEqual(self.adapter.object_type_label(plate_id), f"{wells}-well plate")

                properties = self.adapter.geometry_properties(plate_id)
                self.assertEqual(properties["role"], "measurement")
                self.assertEqual(properties["well_plate"]["wells"], wells)
                self.assertEqual(properties["well_plate"]["rows"], rows)
                self.assertEqual(properties["well_plate"]["columns"], columns)
                self.assertAlmostEqual(properties["well_plate"]["pitch_mm"], pitch_mm)
                np.testing.assert_allclose(
                    np.asarray(self.adapter._mesh(plate_id)["mesh"]["dimension"]) * 1000.0,
                    [84.0, 126.0, 15.0],
                    atol=1e-12,
                )

                settings = self.adapter.measurement_properties(plate_id)
                self.assertEqual(settings["sampling_mode"], "user_defined")
                self.assertEqual(len(settings["defined_points_mm"]), wells)
                sample = self.adapter.measurement_sample_points(plate_id)
                self.assertEqual(sample["sampling_mode"], "user_defined")
                self.assertEqual(sample["quality"], "user_defined")
                self.assertEqual(len(sample["local_points_m"]), wells)
                np.testing.assert_allclose(
                    np.mean(sample["local_points_m"], axis=0),
                    [0.0, 0.0, 0.0],
                    atol=1e-12,
                )

        plate_id = self.adapter.add_template("well_plate_96")
        sample_before = self.adapter.measurement_sample_points(plate_id)["world_points_m"]
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": plate_id,
                        "position": [0.01, -0.02, 0.03],
                        "orientation": [0.0, 0.0, 90.0],
                    },
                }
            ]
        )
        sample_after = self.adapter.measurement_sample_points(plate_id)["world_points_m"]
        self.assertEqual(len(sample_before), len(sample_after))
        np.testing.assert_allclose(np.mean(sample_after, axis=0), [0.01, -0.02, 0.03], atol=1e-12)

        reopened = StudioAdapter(self.adapter.document())
        reopened_properties = reopened.geometry_properties(plate_id)
        self.assertEqual(reopened_properties["well_plate"]["wells"], 96)
        self.assertEqual(len(reopened.measurement_properties(plate_id)["defined_points_mm"]), 96)

    def test_builtin_well_plate_scene_overlay_marks_rims_and_sample_centres(self):
        plate_id = self.adapter.add_template("well_plate_24")
        traces = self.adapter._well_plate_overlay_traces(self.adapter._mesh(plate_id))
        self.assertEqual(len(traces), 2)
        rings, centres = traces
        self.assertTrue(rings["meta"]["field_workbench_well_rims"])
        self.assertTrue(centres["meta"]["field_workbench_well_centres"])
        self.assertEqual(len(centres["x"]), 24)
        self.assertEqual(centres["text"][0], "A1")
        self.assertEqual(centres["text"][-1], "D6")

    def test_builtin_generic_bust_has_canonical_pose_and_all_geometry_roles(self):
        bust_id = self.adapter.add_template("generic_bust")
        obj = self.adapter.get_object(bust_id)
        self.assertEqual(obj["builtin_template"], "generic_bust")
        self.assertEqual(self.adapter.object_type_label(bust_id), "Generic bust")

        transform = self.adapter.get_transform(bust_id)
        np.testing.assert_allclose(transform["position"], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(transform["euler"], [0.0, 0.0, 0.0], atol=1e-10)

        properties = self.adapter.geometry_properties(bust_id)
        self.assertEqual(properties["source"], "generic_bust_v1.stl")
        self.assertEqual(properties["builtin_template"], "generic_bust")
        self.assertEqual(properties["eligible_roles"], ["visual", "exclusion", "measurement"])
        self.assertTrue(properties["mesh_health"]["containment_reliable"])
        self.assertEqual(properties["vertices"], 19_218)
        self.assertEqual(properties["faces"], 38_432)

        for role in ("exclusion", "measurement", "visual"):
            self.adapter.apply_operations(
                [
                    {
                        "method": "set_geometry_style",
                        "params": {"object_id": bust_id, "role": role},
                    }
                ]
            )
            self.assertEqual(self.adapter.geometry_properties(bust_id)["role"], role)
            if role == "measurement":
                sample = self.adapter.measurement_sample_points(bust_id, quality="preview")
                self.assertGreater(len(sample["local_points_m"]), 0)
                self.assertEqual(sample["axis_count"], 7)

        reopened = StudioAdapter(self.adapter.document())
        self.assertEqual(reopened.get_object(bust_id)["builtin_template"], "generic_bust")
        np.testing.assert_allclose(
            reopened.get_transform(bust_id)["euler"], [0.0, 0.0, 0.0], atol=1e-10
        )

    def test_builtin_generic_bust_surface_regions_are_complete_and_highlightable(self):
        bust_id = self.adapter.add_template("generic_bust")
        region_map = self.adapter.surface_region_properties(bust_id)
        self.assertIsNotNone(region_map)
        self.assertEqual(
            {key: region["face_count"] for key, region in region_map["regions"].items()},
            {
                "scalp": 9_106,
                "face": 8_117,
                "ear_l": 1_103,
                "ear_r": 1_167,
                "neck_shoulders": 18_939,
            },
        )
        validation = region_map["validation"]
        self.assertTrue(validation["mesh_file_sha256_verified"])
        self.assertTrue(validation["topology_counts_verified"])
        self.assertTrue(validation["workbench_geometry_sha256_verified"])
        self.assertTrue(validation["complete_partition"])
        self.assertEqual(validation["overlap_count"], 0)
        self.assertEqual(validation["missing_face_count"], 0)

        properties = self.adapter.geometry_properties(bust_id)
        self.assertEqual(properties["surface_regions"]["source"], "generic_bust_v1.regions.json")
        self.assertEqual(
            properties["surface_regions"]["regions"]["scalp"]["settings"],
            {
                "colour": "#3b82f6",
                "active": True,
                "visible": False,
                "placement": True,
                "drc_exclusion": True,
                "clearance_override_mm": None,
            },
        )
        self.assertTrue(
            properties["surface_regions"]["regions"]["face"]["settings"]["drc_exclusion"]
        )
        self.assertFalse(
            properties["surface_regions"]["regions"]["face"]["settings"]["placement"]
        )

        figure = self.adapter.figure(
            selected_id=bust_id,
            surface_region_highlight=(bust_id, "face"),
        )
        overlays = [
            trace
            for trace in figure["data"]
            if isinstance(trace.get("meta"), dict)
            and trace["meta"].get("field_workbench_surface_region") == "face"
        ]
        self.assertEqual(len(overlays), 1)
        self.assertEqual(overlays[0]["name"], "Region: Face")
        self.assertEqual(len(overlays[0]["i"]), 8_117)
        self.assertEqual(overlays[0]["color"], "#ef4444")

        self.adapter.apply_operations(
            [
                {
                    "method": "set_surface_region_settings",
                    "params": {
                        "object_id": bust_id,
                        "settings": {
                            "scalp": {
                                "colour": "#112233",
                                "active": False,
                                "visible": True,
                                "placement": False,
                                "drc_exclusion": False,
                                "clearance_override_mm": 12.5,
                            }
                        },
                    },
                }
            ]
        )
        changed = self.adapter.surface_region_properties(bust_id)["regions"]["scalp"]["settings"]
        self.assertEqual(
            changed,
            {
                "colour": "#112233",
                "active": False,
                "visible": True,
                "placement": False,
                "drc_exclusion": False,
                "clearance_override_mm": 12.5,
            },
        )
        persistent = [
            trace
            for trace in self.adapter.figure()["data"]
            if isinstance(trace.get("meta"), dict)
            and trace["meta"].get("field_workbench_surface_region") == "scalp"
            and trace["meta"].get("field_workbench_surface_region_persistent")
        ]
        self.assertEqual(len(persistent), 1)
        self.assertEqual(persistent[0]["color"], "#112233")

        reopened = StudioAdapter(self.adapter.document())
        self.assertEqual(
            reopened.surface_region_properties(bust_id)["regions"]["scalp"]["face_count"],
            9_106,
        )
        self.assertEqual(
            reopened.surface_region_properties(bust_id)["regions"]["scalp"]["settings"],
            changed,
        )

    def test_imported_companion_surface_regions_support_arbitrary_material_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = Path(temp_dir) / "custom_head.obj"
            mesh_path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 10 0 0",
                        "v 0 10 0",
                        "v 0 0 10",
                        "f 1 3 2",
                        "f 1 2 4",
                        "f 2 3 4",
                        "f 3 1 4",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            sidecar = {
                "schema": "fieldworkbench.surface_regions",
                "schema_version": 1,
                "asset": "custom_head",
                "mesh_file": "custom_head.obj",
                "mesh_sha256": digest,
                "authoring": {"method": "Blender material-slot face assignments"},
                "topology": {
                    "vertex_count": 4,
                    "face_count": 4,
                    "all_triangles": True,
                    "ordered_geometry_sha256": "authoring-provenance-only",
                },
                "regions": {
                    "Keep 12 mm away": {
                        "display_name": "Keep 12 mm away",
                        "face_count": 2,
                        "face_indices": [0, 1],
                    },
                    "Électrode fenêtre β": {
                        "display_name": "Électrode fenêtre β",
                        "face_count": 1,
                        "face_indices": [2],
                    },
                },
            }
            mesh_path.with_suffix(".regions.json").write_text(
                json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
            )

            object_id = self.adapter.import_mesh(
                mesh_path, unit_scale=0.001, center=True
            )
            region_map = self.adapter.surface_region_properties(object_id)
            self.assertIsNotNone(region_map)
            self.assertEqual(
                set(region_map["regions"]),
                {"Keep 12 mm away", "Électrode fenêtre β"},
            )
            self.assertEqual(region_map["validation"]["unique_assigned_faces"], 3)
            self.assertEqual(region_map["validation"]["missing_face_count"], 1)
            self.assertFalse(region_map["validation"]["complete_partition"])
            self.assertTrue(region_map["validation"]["mesh_file_sha256_verified"])
            self.assertTrue(region_map["validation"]["workbench_geometry_sha256_verified"])
            for region in region_map["regions"].values():
                self.assertFalse(region["settings"]["active"])
                self.assertFalse(region["settings"]["visible"])
                self.assertFalse(region["settings"]["placement"])
                self.assertFalse(region["settings"]["drc_exclusion"])

            self.adapter.apply_operations(
                [
                    {
                        "method": "set_surface_region_settings",
                        "params": {
                            "object_id": object_id,
                            "settings": {
                                "Keep 12 mm away": {
                                    "active": True,
                                    "drc_exclusion": True,
                                    "clearance_override_mm": 12.0,
                                }
                            },
                        },
                    }
                ]
            )
            entries = self.adapter._drc_region_exclusion_entries(
                self.adapter._mesh(object_id)
            )
            self.assertEqual(entries[0]["drc_component"], "physical_shell")
            marked = next(
                entry
                for entry in entries
                if entry.get("surface_region_key") == "Keep 12 mm away"
            )
            self.assertEqual(marked["required_clearance_mm"], 12.0)
            self.assertEqual(len(marked["faces"]), 2)

            reopened = StudioAdapter(self.adapter.document())
            reopened_map = reopened.surface_region_properties(object_id)
            self.assertEqual(
                reopened_map["regions"]["Électrode fenêtre β"]["display_name"],
                "Électrode fenêtre β",
            )
            self.assertTrue(
                reopened_map["regions"]["Keep 12 mm away"]["settings"]["drc_exclusion"]
            )

    def test_imported_surface_region_sidecar_rejects_wrong_mesh_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = Path(temp_dir) / "marked.obj"
            mesh_path.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            sidecar = {
                "schema": "fieldworkbench.surface_regions",
                "schema_version": 1,
                "asset": "marked",
                "mesh_file": "marked.obj",
                "mesh_sha256": "0" * 64,
                "topology": {
                    "vertex_count": 3,
                    "face_count": 1,
                    "all_triangles": True,
                },
                "regions": {
                    "Pad": {
                        "display_name": "Pad",
                        "face_count": 1,
                        "face_indices": [0],
                    }
                },
            }
            mesh_path.with_suffix(".regions.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
            with self.assertRaisesRegex(StudioOperationError, "does not match"):
                self.adapter.import_mesh(mesh_path, unit_scale=1.0)

    def test_surface_region_legacy_roles_migrate_to_independent_capabilities(self):
        bust_id = self.adapter.add_template("generic_bust")
        mesh = self.adapter._mesh(bust_id)
        mesh["surface_region_settings"] = {
            "scalp": {
                "colour": "#112233",
                "active": True,
                "visible": False,
                "role": "placement",
            },
            "face": {
                "colour": "#445566",
                "active": True,
                "visible": False,
                "role": "exclusion",
            },
        }
        regions = self.adapter.surface_region_properties(bust_id)["regions"]
        self.assertTrue(regions["scalp"]["settings"]["placement"])
        self.assertFalse(regions["scalp"]["settings"]["drc_exclusion"])
        self.assertFalse(regions["face"]["settings"]["placement"])
        self.assertTrue(regions["face"]["settings"]["drc_exclusion"])
        self.assertNotIn("role", regions["scalp"]["settings"])

    def test_region_aware_drc_uses_shell_region_and_base_clearance_hierarchy(self):
        bust_id = self.adapter.add_template("generic_bust")
        self.adapter.set_drc_settings(
            {
                "default_clearance_mm": 3.0,
                "coil_to_coil_clearance_mm": 0.0,
                "mesh_tolerance_mm": 0.05,
            }
        )
        self.adapter.apply_operations(
            [
                {
                    "method": "set_surface_region_settings",
                    "params": {
                        "object_id": bust_id,
                        "settings": {
                            "face": {
                                "active": True,
                                "drc_exclusion": True,
                                "placement": False,
                                "clearance_override_mm": 15.0,
                            },
                            "ear_l": {
                                "active": False,
                                "drc_exclusion": True,
                                "placement": False,
                            },
                        },
                    },
                }
            ]
        )
        mesh = self.adapter._mesh(bust_id)
        entries = self.adapter._drc_region_exclusion_entries(mesh)
        components = [entry.get("drc_component") for entry in entries]
        self.assertEqual(components[0], "physical_shell")
        self.assertEqual(components.count("region"), 4)
        self.assertNotIn("base_surface", components)
        region_entries = {
            entry.get("surface_region_key"): entry
            for entry in entries
            if entry.get("drc_component") == "region"
        }
        self.assertEqual(region_entries["face"]["required_clearance_mm"], 15.0)
        self.assertTrue(region_entries["face"]["override"])
        self.assertEqual(region_entries["scalp"]["required_clearance_mm"], 3.0)
        self.assertFalse(region_entries["scalp"]["override"])
        self.assertNotIn("ear_l", region_entries)

        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": bust_id, "role": "exclusion"},
                },
                {
                    "method": "set_drc_clearance",
                    "params": {"object_id": bust_id, "clearance_override_mm": 4.0},
                },
            ]
        )
        entries = self.adapter._drc_region_exclusion_entries(self.adapter._mesh(bust_id))
        base = next(entry for entry in entries if entry.get("drc_component") == "base_surface")
        self.assertEqual(base["required_clearance_mm"], 4.0)
        self.assertTrue(base["override"])
        self.assertEqual(len(base["faces"]), 1_103)
        scalp = next(
            entry for entry in entries
            if entry.get("surface_region_key") == "scalp"
        )
        self.assertEqual(scalp["required_clearance_mm"], 4.0)
        face = next(
            entry for entry in entries
            if entry.get("surface_region_key") == "face"
        )
        self.assertEqual(face["required_clearance_mm"], 15.0)

    def test_builtin_sri24_brain_uses_rigid_head_registration(self):
        fiducials = np.asarray(
            [
                SRI24_HEAD_FIDUCIALS_RAS_MM["LPA"],
                SRI24_HEAD_FIDUCIALS_RAS_MM["NAS"],
                SRI24_HEAD_FIDUCIALS_RAS_MM["RPA"],
            ],
            dtype=float,
        )
        head = sri24_ras_mm_to_head_mm(fiducials)
        self.assertAlmostEqual(head[0, 1], 0.0, places=4)
        self.assertAlmostEqual(head[0, 2], 0.0, places=4)
        self.assertAlmostEqual(head[1, 0], 0.0, places=4)
        self.assertAlmostEqual(head[1, 2], 0.0, places=4)
        self.assertAlmostEqual(head[2, 1], 0.0, places=4)
        self.assertAlmostEqual(head[2, 2], 0.0, places=4)
        self.assertLess(head[0, 0], 0.0)
        self.assertGreater(head[1, 1], 0.0)
        self.assertGreater(head[2, 0], 0.0)

        brain_id = self.adapter.add_template("sri24_brain")
        obj = self.adapter.get_object(brain_id)
        self.assertEqual(obj["builtin_template"], "sri24_brain")
        self.assertEqual(self.adapter.object_type_label(brain_id), "SRI24 brain")

        transform = self.adapter.get_transform(brain_id)
        np.testing.assert_allclose(transform["position"], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(transform["euler"], [0.0, 0.0, 0.0], atol=1e-10)

        properties = self.adapter.geometry_properties(brain_id)
        self.assertEqual(properties["source"], "sri24_brain_gmwm_v1.stl")
        self.assertEqual(properties["builtin_template"], "sri24_brain")
        self.assertEqual(properties["coordinate_frame"], "NAS/LPA/RPA head frame")
        self.assertEqual(
            properties["source_coordinate_frame"], "SRI24 v2.0 RAS millimetres"
        )
        self.assertEqual(properties["eligible_roles"], ["visual", "exclusion", "measurement"])
        self.assertTrue(properties["mesh_health"]["containment_reliable"])
        self.assertEqual(properties["vertices"], 12_855)
        self.assertEqual(properties["faces"], 26_250)
        np.testing.assert_allclose(
            properties["source_dimensions_mm"],
            [132.3969, 171.1446, 126.0200],
            atol=0.02,
        )

        reopened = StudioAdapter(self.adapter.document())
        self.assertEqual(reopened.get_object(brain_id)["builtin_template"], "sri24_brain")

    def test_coil_enable_switch_zeroes_source_without_forgetting_drive(self):
        physical_before = self.adapter.coil_physical_properties("coil_a")
        self.assertTrue(physical_before["enabled"])
        configured_drive = physical_before["drive_current_a"]
        current_before = {
            item["name"]: item["value"] for item in self.adapter.get_params("coil_a")
        }["current"]
        self.assertNotEqual(current_before, 0.0)

        self.adapter.set_coils_enabled(["coil_a"], False)
        disabled = self.adapter.coil_physical_properties("coil_a")
        self.assertFalse(disabled["enabled"])
        self.assertAlmostEqual(disabled["drive_current_a"], configured_drive)
        current_disabled = {
            item["name"]: item["value"] for item in self.adapter.get_params("coil_a")
        }["current"]
        self.assertEqual(current_disabled, 0.0)

        reopened = StudioAdapter(self.adapter.document())
        reopened_disabled = reopened.coil_physical_properties("coil_a")
        self.assertFalse(reopened_disabled["enabled"])
        self.assertAlmostEqual(reopened_disabled["drive_current_a"], configured_drive)
        self.assertEqual(
            {item["name"]: item["value"] for item in reopened.get_params("coil_a")}["current"],
            0.0,
        )

        reopened.set_coils_enabled(["coil_a"], True)
        reenabled = reopened.coil_physical_properties("coil_a")
        self.assertTrue(reenabled["enabled"])
        self.assertAlmostEqual(reenabled["drive_current_a"], configured_drive)
        self.assertAlmostEqual(
            {item["name"]: item["value"] for item in reopened.get_params("coil_a")}["current"],
            current_before,
        )

    def test_coil_construction_drives_field_and_calculates_copper(self):
        original = self.adapter.field_at_mm([0, 0, 0])["magnitude"][0]
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "drive_current_a": 0.07,
                        "turns": 100,
                        "field_scale_factor": 1.0,
                        "awg": 30,
                        "resistance_20_ohm_per_km": 351.25,
                        "lead_length_m": 0.2,
                        "enamel_radial_mm": 0.025,
                        "packing_factor": 0.75,
                        "temperature_c": 60.0,
                        "additional_mass_g": 25.0,
                        "current_mode": "peak_sine",
                    },
                }
            ]
        )
        engine_params = {
            item["name"]: item["value"] for item in self.adapter.get_params("coil_a")
        }
        self.assertAlmostEqual(engine_params["current"], 7.0)
        self.assertGreater(self.adapter.field_at_mm([0, 0, 0])["magnitude"][0], original)

        physical = self.adapter.coil_physical_properties("coil_a")
        calc = physical["calculations"]
        self.assertEqual(physical["turns"], 100)
        self.assertEqual(calc["turns"], 100)
        self.assertAlmostEqual(calc["ampere_turns"], 7.0)
        self.assertAlmostEqual(calc["effective_ampere_turns"], 7.0)
        self.assertAlmostEqual(calc["mean_turn_length_m"], math.pi * 0.2)
        self.assertAlmostEqual(calc["wire_length_m"], math.pi * 0.2 * 100 + 0.2)
        self.assertAlmostEqual(calc["wire_resistance_20_ohm_per_km"], 351.25)
        self.assertAlmostEqual(
            calc["resistance_20_ohm"], calc["wire_length_m"] * 351.25 / 1000.0
        )
        self.assertGreater(calc["copper_mass_g"], 0.0)
        self.assertAlmostEqual(calc["total_mass_g"], calc["copper_mass_g"] + 25.0)
        self.assertGreater(calc["resistance_operating_ohm"], calc["resistance_20_ohm"])
        self.assertAlmostEqual(
            calc["copper_loss_w"],
            (0.07 / math.sqrt(2.0)) ** 2 * calc["resistance_operating_ohm"],
        )

    def test_circular_coil_size_is_mean_diameter_not_radius(self):
        coil_id = self.adapter.add_template("circle")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": coil_id,
                        "name": "diameter",
                        "value": 0.08,
                    },
                },
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": coil_id,
                        "turns": 320,
                        "resistance_20_ohm_per_km": 217.0,
                        "lead_length_m": 0.2,
                    },
                },
            ]
        )
        calc = self.adapter.coil_physical_properties(coil_id)["calculations"]
        expected_length = math.pi * 0.08 * 320 + 0.2
        self.assertAlmostEqual(calc["mean_turn_length_m"], math.pi * 0.08)
        self.assertAlmostEqual(calc["wire_length_m"], expected_length)
        self.assertAlmostEqual(
            calc["resistance_20_ohm"], expected_length * 217.0 / 1000.0
        )

    def test_optional_circular_coil_assembly_derives_bobbin_and_bounding_box(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": "coil_a",
                        "name": "diameter",
                        "value": 0.210,
                    },
                },
            ]
        )
        field_before = self.adapter.field_fingerprint(FINGERPRINT_POINTS)
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "physical_geometry_enabled": True,
                        "winding_radial_build_mm": 6.4,
                        "winding_axial_width_mm": 6.4,
                        "bobbin_wall_thickness_mm": 2.0,
                        "flange_height_mm": 8.0,
                        "flange_thickness_mm": 2.0,
                    },
                },
            ]
        )
        physical = self.adapter.coil_physical_properties("coil_a")
        assembly = physical["calculations"]["assembly"]
        self.assertTrue(physical["physical_geometry_enabled"])
        self.assertEqual(physical["winding_radial_build_mode"], "manual")
        self.assertTrue(assembly["supported"])
        self.assertEqual(assembly["shape"], "annular_cylinder")
        self.assertAlmostEqual(assembly["winding_surface_diameter_mm"], 203.6)
        self.assertAlmostEqual(assembly["bobbin_opening_diameter_mm"], 199.6)
        self.assertAlmostEqual(assembly["winding_outer_diameter_mm"], 216.4)
        self.assertAlmostEqual(assembly["flange_outer_diameter_mm"], 219.6)
        np.testing.assert_allclose(assembly["bounding_box_mm"], [219.6, 219.6, 10.4])
        self.assertEqual(assembly["warnings"], [])

        # Construction geometry is metadata only and must never change B.
        np.testing.assert_array_equal(
            self.adapter.field_fingerprint(FINGERPRINT_POINTS), field_before
        )
        clone_id = self.adapter.duplicate("coil_a")
        clone = self.adapter.coil_physical_properties(clone_id)
        self.assertTrue(clone["physical_geometry_enabled"])
        np.testing.assert_allclose(
            clone["calculations"]["assembly"]["bounding_box_mm"],
            [219.6, 219.6, 10.4],
        )

    def test_auto_radial_build_uses_wire_turns_packing_and_axial_width(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "turns": 320,
                        "awg": 28,
                        "enamel_radial_mm": 0.03,
                        "packing_factor": 0.72,
                    },
                }
            ]
        )
        field_before_envelope = self.adapter.field_fingerprint(FINGERPRINT_POINTS)
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "physical_geometry_enabled": True,
                        "winding_radial_build_mode": "auto",
                        "winding_axial_width_mm": 8.0,
                        "bobbin_wall_thickness_mm": 2.0,
                        "flange_height_mm": 12.0,
                        "flange_thickness_mm": 2.0,
                    },
                }
            ]
        )
        physical = self.adapter.coil_physical_properties("coil_a")
        calc = physical["calculations"]
        assembly = calc["assembly"]
        finished_diameter = awg_diameter_mm(28) + 2.0 * 0.03
        finished_wire_area = math.pi * (finished_diameter / 2.0) ** 2
        expected_window_area = 320 * finished_wire_area / 0.72
        expected_radial_build = expected_window_area / 8.0

        self.assertEqual(physical["winding_radial_build_mode"], "auto")
        self.assertAlmostEqual(
            calc["estimated_winding_cross_section_mm2"], expected_window_area
        )
        self.assertAlmostEqual(
            calc["auto_winding_radial_build_mm"], expected_radial_build
        )
        self.assertAlmostEqual(
            assembly["winding_radial_build_mm"], expected_radial_build
        )
        self.assertEqual(assembly["winding_radial_build_mode"], "auto")
        self.assertAlmostEqual(assembly["estimated_capacity_turns"], 320.0)
        self.assertNotIn(
            "The entered winding cross-section is smaller than the estimated wire bundle.",
            assembly["warnings"],
        )

        doubled = self.adapter.coil_physical_properties(
            "coil_a", {"turns": 640}
        )["calculations"]["assembly"]
        self.assertAlmostEqual(
            doubled["winding_radial_build_mm"], 2.0 * expected_radial_build
        )
        wider = self.adapter.coil_physical_properties(
            "coil_a", {"winding_axial_width_mm": 16.0}
        )["calculations"]["assembly"]
        self.assertAlmostEqual(
            wider["winding_radial_build_mm"], 0.5 * expected_radial_build
        )

        # Like every assembly-envelope input, automatic build remains
        # construction metadata and must not alter the field source.
        np.testing.assert_array_equal(
            self.adapter.field_fingerprint(FINGERPRINT_POINTS),
            field_before_envelope,
        )
        clone_id = self.adapter.duplicate("coil_a")
        clone = self.adapter.coil_physical_properties(clone_id)
        self.assertEqual(clone["winding_radial_build_mode"], "auto")
        self.assertAlmostEqual(
            clone["calculations"]["assembly"]["winding_radial_build_mm"],
            expected_radial_build,
        )
        reopened = StudioAdapter(self.adapter.document())
        reopened_clone = reopened.coil_physical_properties(clone_id)
        self.assertEqual(reopened_clone["winding_radial_build_mode"], "auto")
        self.assertAlmostEqual(
            reopened_clone["calculations"]["assembly"]["winding_radial_build_mm"],
            expected_radial_build,
        )

    def test_saved_radial_build_without_mode_migrates_to_manual_override(self):
        scene = compact_workflow_scene()
        old_spec = scene["field_workbench"]["coil_specs"]["coil_a"]
        old_spec.pop("winding_radial_build_mode", None)
        old_spec.update(
            {
                "physical_geometry_enabled": True,
                "winding_radial_build_mm": 6.4,
                "winding_axial_width_mm": 6.4,
                "flange_height_mm": 8.0,
                "flange_thickness_mm": 2.0,
            }
        )
        migrated = StudioAdapter(scene).coil_physical_properties("coil_a")
        self.assertEqual(migrated["winding_radial_build_mode"], "manual")
        self.assertAlmostEqual(
            migrated["calculations"]["assembly"]["winding_radial_build_mm"],
            6.4,
        )

    def test_geometry_drc_master_switch_allows_measurement_and_disables_exclusion(self):
        box_id = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box_id, "role": "measurement"},
                }
            ]
        )
        properties = self.adapter.geometry_properties(box_id)
        self.assertFalse(properties["drc_enabled"])
        self.assertFalse(properties["drc_enabled_explicit"])
        self.assertFalse(
            any(
                box_id in {item.get("object_a_id"), item.get("object_b_id")}
                for item in self.adapter.run_drc()["results"]
            )
        )

        self.adapter.apply_operations(
            [
                {
                    "method": "set_drc_clearance",
                    "params": {
                        "object_id": box_id,
                        "enabled": True,
                        "clearance_override_mm": 2.5,
                    },
                }
            ]
        )
        properties = self.adapter.geometry_properties(box_id)
        self.assertTrue(properties["drc_enabled"])
        self.assertTrue(properties["drc_enabled_explicit"])
        self.assertAlmostEqual(properties["drc_clearance_override_mm"], 2.5)
        self.assertTrue(
            any(
                box_id in {item.get("object_a_id"), item.get("object_b_id")}
                for item in self.adapter.run_drc()["results"]
            )
        )

        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box_id, "role": "exclusion"},
                },
                {
                    "method": "set_drc_clearance",
                    "params": {
                        "object_id": box_id,
                        "enabled": False,
                        "clearance_override_mm": 2.5,
                    },
                },
            ]
        )
        properties = self.adapter.geometry_properties(box_id)
        self.assertFalse(properties["drc_enabled"])
        self.assertAlmostEqual(properties["drc_clearance_override_mm"], 2.5)
        self.assertFalse(
            any(
                box_id in {item.get("object_a_id"), item.get("object_b_id")}
                for item in self.adapter.run_drc()["results"]
            )
        )

        reopened = StudioAdapter(self.adapter.document())
        reopened_properties = reopened.geometry_properties(box_id)
        self.assertFalse(reopened_properties["drc_enabled"])
        self.assertTrue(reopened_properties["drc_enabled_explicit"])
        self.assertAlmostEqual(reopened_properties["drc_clearance_override_mm"], 2.5)

    def _enable_test_assembly(self, object_id: str) -> None:
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": object_id,
                        "physical_geometry_enabled": True,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 10.0,
                        "winding_axial_width_mm": 10.0,
                        "bobbin_wall_thickness_mm": 2.0,
                        "flange_height_mm": 15.0,
                        "flange_thickness_mm": 2.0,
                    },
                }
            ]
        )

    def _drc_pair(self, report, first_id: str, second_id: str):
        for result in report["results"]:
            if {result["object_a_id"], result["object_b_id"]} == {
                first_id,
                second_id,
            }:
                return result
        self.fail(f"No DRC result for {first_id} and {second_id}")

    def _add_triangle_cube(
        self,
        object_id: str,
        *,
        size_m: float,
        position=(0.0, 0.0, 0.0),
        open_surface: bool = False,
        flip_first_face: bool = False,
        extra_degenerate_face: bool = False,
    ) -> str:
        vertices, faces = self.adapter._mesh_local_vertices_faces(
            {"mesh": {"kind": "box", "dimension": [size_m] * 3}}
        )
        faces = faces.copy()
        if open_surface:
            faces = faces[:-2]
        if flip_first_face:
            faces[0] = faces[0][::-1]
        if extra_degenerate_face:
            faces = np.vstack([faces, [0, 0, 1]])
        self.adapter.geometry.append(
            {
                "id": object_id,
                "type": "workbench.Mesh",
                "label": object_id.replace("_", " ").title(),
                "visible": True,
                "parent": None,
                "position": list(position),
                "euler": [0.0, 0.0, 0.0],
                "scale_xyz": [1.0, 1.0, 1.0],
                "role": "exclusion",
                "colour": "#dc2626",
                "opacity": 0.3,
                "source": f"{object_id}.obj",
                "mesh": {
                    "kind": "triangles",
                    "vertices": vertices.tolist(),
                    "faces": faces.tolist(),
                },
            }
        )
        return object_id

    def test_physical_assembly_renders_winding_barrel_and_flange_solids(self):
        field_before = self.adapter.field_fingerprint(FINGERPRINT_POINTS)
        self._enable_test_assembly("coil_a")
        figure = self.adapter.figure(selected_id="coil_a")
        components = [
            trace.get("meta", {}).get("field_workbench_construction_component")
            for trace in figure["data"]
            if trace.get("meta", {}).get("field_workbench_object_id") == "coil_a"
            and trace.get("meta", {}).get("field_workbench_construction_component")
        ]
        self.assertEqual(
            components,
            ["winding pack", "bobbin barrel", "lower flange", "upper flange"],
        )
        assembly_traces = [
            trace
            for trace in figure["data"]
            if trace.get("meta", {}).get("field_workbench_construction_component")
        ]
        self.assertTrue(all(trace["type"] == "mesh3d" for trace in assembly_traces))
        self.assertGreaterEqual(
            sum(
                trace.get("name") == "Selected assembly envelope"
                for trace in figure["data"]
            ),
            4,
        )
        np.testing.assert_array_equal(
            self.adapter.field_fingerprint(FINGERPRINT_POINTS), field_before
        )

    def test_planar_coil_bobbins_share_visual_and_drc_geometry(self):
        for template in ("racetrack", "polyline"):
            coil_id = self.adapter.add_template(template)
            field_before = self.adapter.field_fingerprint(FINGERPRINT_POINTS)
            self.adapter.apply_operations(
                [
                    {
                        "method": "set_coil_physical",
                        "params": {
                            "object_id": coil_id,
                            "physical_geometry_enabled": True,
                            "winding_radial_build_mode": "manual",
                            "winding_radial_build_mm": 10.0,
                            "winding_axial_width_mm": 10.0,
                            "bobbin_wall_thickness_mm": 2.0,
                            "flange_height_mm": 15.0,
                            "flange_thickness_mm": 2.0,
                        },
                    }
                ]
            )
            physical = self.adapter.coil_physical_properties(coil_id)
            assembly = physical["calculations"]["assembly"]
            self.assertTrue(assembly["supported"])
            self.assertTrue(assembly["valid"])
            self.assertEqual(assembly["shape"], "planar_path_band")

            figure = self.adapter.figure(selected_id=coil_id)
            components = [
                trace.get("meta", {}).get("field_workbench_construction_component")
                for trace in figure["data"]
                if trace.get("meta", {}).get("field_workbench_object_id") == coil_id
                and trace.get("meta", {}).get(
                    "field_workbench_construction_component"
                )
            ]
            self.assertEqual(
                components,
                ["winding pack", "bobbin barrel", "lower flange", "upper flange"],
            )
            self.assertGreaterEqual(
                sum(
                    trace.get("name") == "Selected assembly envelope"
                    for trace in figure["data"]
                ),
                4,
            )
            np.testing.assert_array_equal(
                self.adapter.field_fingerprint(FINGERPRINT_POINTS), field_before
            )

            exclusion_id = self.adapter.add_template("guide_box")
            self.adapter.apply_operations(
                [
                    {
                        "method": "set_param",
                        "params": {
                            "object_id": exclusion_id,
                            "name": "dimension",
                            "value": [0.006, 0.006, 0.006],
                        },
                    },
                    {
                        "method": "set_transform",
                        "params": {
                            "object_id": exclusion_id,
                            "position": [0.0, 0.05, 0.0],
                            "orientation": [0.0, 0.0, 0.0],
                        },
                    },
                    {
                        "method": "set_geometry_style",
                        "params": {
                            "object_id": exclusion_id,
                            "role": "exclusion",
                        },
                    },
                ]
            )
            result = self._drc_pair(
                self.adapter.run_drc(), coil_id, exclusion_id
            )
            self.assertEqual(result["status"], "intersecting")
            self.assertTrue(result["approximate"])

            self.adapter.remove(exclusion_id)
            self.adapter.remove(coil_id)

    def test_drc_uses_annular_geometry_not_overlapping_world_boxes(self):
        self.adapter.remove("coil_b")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 45.0, 0.0],
                    },
                }
            ]
        )
        self._enable_test_assembly("coil_a")
        box_id = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": box_id,
                        "name": "dimension",
                        "value": [0.02, 0.02, 0.02],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box_id, "role": "exclusion"},
                },
            ]
        )
        report = self.adapter.run_drc()
        result = self._drc_pair(report, "coil_a", box_id)
        self.assertEqual(result["status"], "clear")
        self.assertFalse(result["conservative_lower_bound"])
        self.assertGreater(result["signed_clearance_mm"], 70.0)

        # Inverse containment is also a collision even when no box corner or
        # the box centre happens to lie in the annular material.
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": box_id,
                        "name": "dimension",
                        "value": [0.3, 0.3, 0.3],
                    },
                }
            ]
        )
        containing = self._drc_pair(self.adapter.run_drc(), "coil_a", box_id)
        self.assertEqual(containing["status"], "intersecting")

    def test_drc_hidden_sphere_override_detects_clearance_and_intersection(self):
        self.adapter.remove("coil_b")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        self._enable_test_assembly("coil_a")
        sphere_id = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": sphere_id,
                        "name": "diameter",
                        "value": 0.01,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": sphere_id,
                        "position": [0.116, 0.0, 0.006],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": sphere_id, "role": "exclusion"},
                },
                {
                    "method": "set_drc_clearance",
                    "params": {
                        "object_id": sphere_id,
                        "clearance_override_mm": 2.0,
                    },
                },
            ]
        )
        self.adapter.set_visible(sphere_id, False)
        report = self.adapter.run_drc()
        result = self._drc_pair(report, "coil_a", sphere_id)
        self.assertEqual(result["status"], "below_clearance")
        self.assertAlmostEqual(result["signed_clearance_mm"], 1.0, places=5)
        self.assertEqual(result["required_clearance_mm"], 2.0)
        self.assertIn("override", result["rule"])
        self.assertEqual(
            StudioAdapter(self.adapter.document()).geometry_properties(sphere_id)[
                "drc_clearance_override_mm"
            ],
            2.0,
        )

        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": sphere_id,
                        "position": [0.109, 0.0, 0.006],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        intersecting = self._drc_pair(
            self.adapter.run_drc(), "coil_a", sphere_id
        )
        self.assertEqual(intersecting["status"], "intersecting")
        self.assertLess(intersecting["signed_clearance_mm"], 0.0)

    def test_optimizer_prepared_drc_matches_fresh_drc_and_uses_transient_edits(self):
        self._enable_test_assembly("coil_a")
        self._enable_test_assembly("coil_b")
        sphere_id = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": sphere_id,
                        "name": "diameter",
                        "value": 0.02,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": sphere_id,
                        "position": [0.118, 0.0, -0.05],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": sphere_id, "role": "exclusion"},
                },
                {
                    "method": "set_drc_clearance",
                    "params": {
                        "object_id": sphere_id,
                        "clearance_override_mm": 1.5,
                    },
                },
            ]
        )
        source_document = self.adapter.document()
        variables = [
            {
                "id": "position_x:A",
                "kind": "position_x",
                "coil_ids": ["coil_a"],
                "min": -25.0,
                "max": 25.0,
                "base": 0.0,
                "label": "ΔX Position Group A",
                "unit": "mm",
            }
        ]
        context = StudioAdapter._optimizer_tuning_prepare_drc_context(
            source_document,
            coil_modeling_method=self.adapter.coil_modeling_method,
            solver_memory_budget_mb=self.adapter.solver_memory_budget_mb,
            dynamic_coil_ids={"coil_a"},
            structural_variables=variables,
        )
        self.assertTrue(context["source_report"]["prepared_optimizer_drc"])
        self.assertEqual(context["dynamic_coil_ids"], {"coil_a"})
        self.assertGreaterEqual(len(context["static_results"]), 1)
        self.assertIn("coil_a", context["rigid_coil_templates"])

        def compact_rows(report):
            compact = {}
            for item in report["results"]:
                key = (
                    item.get("object_a_id"),
                    item.get("object_b_id"),
                    item.get("object_a_region"),
                    item.get("object_b_region"),
                    item.get("rule"),
                )
                clearance = item.get("signed_clearance_mm")
                compact[key] = (
                    item.get("status"),
                    None if clearance is None else round(float(clearance), 9),
                    round(float(item.get("required_clearance_mm", 0.0)), 9),
                )
            return compact

        self.assertEqual(
            compact_rows(self.adapter.run_drc()),
            compact_rows(context["source_report"]),
        )

        source_currents = {
            object_id: float(self.adapter.coil_specs[object_id]["drive_current_a"])
            for object_id in ("coil_a", "coil_b")
        }
        candidate_value = [12.5]

        fresh = StudioAdapter(copy.deepcopy(source_document))
        self.adapter._optimizer_tuning_apply_variables(
            fresh, variables, candidate_value, source_currents
        )
        expected = fresh.run_drc()

        prepared = context["adapter"]
        with mock.patch.object(
            prepared,
            "document",
            side_effect=AssertionError(
                "Prepared optimizer DRC transient edits must not materialize documents."
            ),
        ), mock.patch.object(
            prepared,
            "_drc_coil_entry",
            side_effect=AssertionError(
                "Rigid prepared DRC must transform the cached local coil template instead of rebuilding construction."
            ),
        ):
            prepared._apply_optimizer_transient_operations(context["reset_operations"])
            self.adapter._optimizer_tuning_apply_variables(
                prepared, variables, candidate_value, source_currents
            )
            actual = StudioAdapter._optimizer_tuning_prepared_drc_report(context)

        self.assertEqual(compact_rows(expected), compact_rows(actual))
        self.assertEqual(prepared.history(), {"undo": 0, "redo": 0})
        self.assertEqual(context["rigid_transform_count"], 1)
        self.assertEqual(context["construction_rebuild_count"], 0)

    def test_optimizer_prepared_mesh_broadphase_preserves_surface_distance(self):
        solid = {
            "solid_kind": "annular",
            "inner_radius_m": 0.0,
            "outer_radius_m": 0.02,
            "half_thickness_m": 0.005,
            "centre_z_m": 0.0,
            "centre": np.zeros(3, dtype=float),
            "rotation": np.eye(3, dtype=float),
        }
        vertices = np.asarray(
            [
                [0.03, -0.002, -0.002],
                [0.03, 0.002, -0.002],
                [0.03, 0.0, 0.002],
                [1.0, -0.01, -0.01],
                [1.0, 0.01, -0.01],
                [1.0, 0.0, 0.01],
            ],
            dtype=float,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int)
        mesh = {"vertices": vertices, "faces": faces}
        expected = StudioAdapter._solid_mesh_surface_distance(
            solid, mesh, tolerance_m=0.001
        )
        triangles = vertices[faces]
        stats = {"calls": 0, "faces_total": 0, "faces_retained": 0}
        prepared_mesh = {
            **mesh,
            "_optimizer_face_aabb_min": np.min(triangles, axis=1),
            "_optimizer_face_aabb_max": np.max(triangles, axis=1),
            "_optimizer_broadphase_stats": stats,
        }
        actual = StudioAdapter._solid_mesh_surface_distance(
            solid, prepared_mesh, tolerance_m=0.001
        )
        self.assertAlmostEqual(float(expected[0]), float(actual[0]), places=12)
        self.assertEqual(expected[1:], actual[1:])
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["faces_total"], 2)
        self.assertEqual(stats["faces_retained"], 1)

    def test_imported_mesh_health_reports_topology_and_containment_confidence(self):
        closed_id = self._add_triangle_cube("closed_scan", size_m=0.04)
        closed = self.adapter.mesh_health(closed_id)
        self.assertTrue(closed["watertight"])
        self.assertTrue(closed["manifold"])
        self.assertTrue(closed["consistently_oriented"])
        self.assertTrue(closed["containment_reliable"])
        self.assertEqual(closed["boundary_edges"], 0)
        self.assertEqual(closed["non_manifold_edges"], 0)
        self.assertEqual(closed["degenerate_or_invalid_faces"], 0)
        self.assertAlmostEqual(abs(closed["signed_volume_cm3"]), 64.0)

        open_id = self._add_triangle_cube(
            "open_scan", size_m=0.04, open_surface=True
        )
        opened = self.adapter.mesh_health(open_id)
        self.assertFalse(opened["watertight"])
        self.assertFalse(opened["containment_reliable"])
        self.assertEqual(opened["boundary_edges"], 4)
        self.assertIn("Surface contact and clearance", opened["summary"])

        flipped_id = self._add_triangle_cube(
            "flipped_scan", size_m=0.04, flip_first_face=True
        )
        flipped = self.adapter.mesh_health(flipped_id)
        self.assertTrue(flipped["watertight"])
        self.assertFalse(flipped["consistently_oriented"])
        self.assertFalse(flipped["containment_reliable"])
        self.assertEqual(flipped["inconsistent_edges"], 3)

        non_manifold_id = self._add_triangle_cube(
            "non_manifold_scan", size_m=0.04
        )
        self.adapter._mesh(non_manifold_id)["mesh"]["faces"].append([0, 1, 6])
        non_manifold = self.adapter.mesh_health(non_manifold_id)
        self.assertGreaterEqual(non_manifold["non_manifold_edges"], 1)
        self.assertFalse(non_manifold["containment_reliable"])

        duplicate_id = self._add_triangle_cube("duplicate_scan", size_m=0.04)
        duplicate_mesh = self.adapter._mesh(duplicate_id)["mesh"]
        duplicate_mesh["faces"].append(list(duplicate_mesh["faces"][0]))
        duplicate = self.adapter.mesh_health(duplicate_id)
        self.assertEqual(duplicate["duplicate_faces"], 1)
        self.assertFalse(duplicate["containment_reliable"])

        damaged_id = self._add_triangle_cube(
            "damaged_scan", size_m=0.04, extra_degenerate_face=True
        )
        damaged = self.adapter.mesh_health(damaged_id)
        self.assertEqual(damaged["degenerate_or_invalid_faces"], 1)
        self.assertFalse(damaged["containment_reliable"])

        unusable_id = "unusable_scan"
        self.adapter.geometry.append(
            {
                "id": unusable_id,
                "label": "Unusable scan",
                "visible": True,
                "position": [0.0, 0.0, 0.0],
                "euler": [0.0, 0.0, 0.0],
                "scale_xyz": [1.0, 1.0, 1.0],
                "role": "exclusion",
                "mesh": {
                    "kind": "triangles",
                    "vertices": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
                    "faces": [[0, 1, 2]],
                },
            }
        )
        unusable_result = next(
            item
            for item in self.adapter.run_drc()["results"]
            if item["object_a_id"] == unusable_id
        )
        self.assertEqual(unusable_result["status"], "invalid")
        self.assertEqual(
            unusable_result["mesh_health"]["degenerate_or_invalid_faces"], 1
        )

    def test_imported_obj_exclusion_enters_mesh_health_and_drc_pipeline(self):
        self.adapter.remove("coil_b")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        self._enable_test_assembly("coil_a")
        vertices, faces = self.adapter._mesh_local_vertices_faces(
            {"mesh": {"kind": "box", "dimension": [0.04, 0.04, 0.04]}}
        )
        lines = [
            "v " + " ".join(f"{coordinate * 1000.0:.9g}" for coordinate in vertex)
            for vertex in vertices
        ]
        lines.extend(
            "f " + " ".join(str(int(index) + 1) for index in face)
            for face in faces
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "closed head exclusion.obj"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            mesh_id = self.adapter.import_mesh(path, unit_scale=0.001)
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": mesh_id, "role": "exclusion"},
                }
            ]
        )
        self.assertTrue(
            self.adapter.geometry_properties(mesh_id)["mesh_health"][
                "containment_reliable"
            ]
        )
        report = self.adapter.run_drc()
        self.assertEqual(report["imported_mesh_count"], 1)
        result = self._drc_pair(report, "coil_a", mesh_id)
        self.assertEqual(result["status"], "clear")
        self.assertEqual(result["rule"], "Coil to imported mesh")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_drc_clearance",
                    "params": {
                        "object_id": mesh_id,
                        "clearance_override_mm": 70.0,
                    },
                }
            ]
        )
        constrained = self._drc_pair(self.adapter.run_drc(), "coil_a", mesh_id)
        self.assertEqual(constrained["status"], "below_clearance")
        self.assertEqual(constrained["required_clearance_mm"], 70.0)
        self.assertIn("zone override", constrained["rule"])

    def test_geometry_capabilities_unify_roles_and_explain_broken_mesh_limits(self):
        box_id = self.adapter.add_template("guide_box")
        sphere_id = self.adapter.add_template("guide_sphere")
        closed_id = self._add_triangle_cube("closed_provider", size_m=0.04)
        open_id = self._add_triangle_cube(
            "open_provider", size_m=0.04, open_surface=True
        )
        inward_id = self._add_triangle_cube("inward_provider", size_m=0.04)
        inward_mesh = self.adapter._mesh(inward_id)
        inward_mesh["mesh"]["faces"] = [
            list(reversed(face)) for face in inward_mesh["mesh"]["faces"]
        ]

        for object_id in (box_id, sphere_id, closed_id, inward_id):
            capabilities = self.adapter.geometry_capabilities(object_id)
            self.assertTrue(capabilities["measurement"])
            self.assertTrue(capabilities["snapshot"])
            self.assertTrue(capabilities["optimizer_target"])
            self.assertTrue(capabilities["optimizer_placement_surface"])
            self.assertIn("measurement", capabilities["eligible_roles"])

        open_capabilities = self.adapter.geometry_capabilities(open_id)
        self.assertFalse(open_capabilities["measurement"])
        self.assertFalse(open_capabilities["snapshot"])
        self.assertFalse(open_capabilities["optimizer_target"])
        self.assertTrue(open_capabilities["optimizer_placement_surface"])
        self.assertNotIn("measurement", open_capabilities["eligible_roles"])
        self.assertIn("watertight", open_capabilities["measurement_reason"])
        self.assertEqual(self.adapter.mesh_health(inward_id)["orientation"], "inward")
        self.assertEqual(
            len(self.adapter.measurement_sample_points(inward_id, quality="preview")[
                "local_points_m"
            ]),
            7**3,
        )
        with self.assertRaisesRegex(Exception, "not available"):
            self.adapter.apply_operations(
                [
                    {
                        "method": "set_geometry_style",
                        "params": {"object_id": open_id, "role": "measurement"},
                    }
                ]
            )

        surfaces = {item["id"]: item for item in self.adapter.optimizer_anatomy_meshes()}
        self.assertEqual(surfaces[box_id]["kind"], "box")
        self.assertEqual(surfaces[sphere_id]["kind"], "sphere")
        self.assertTrue(surfaces[closed_id]["containment_reliable"])
        self.assertFalse(surfaces[open_id]["containment_reliable"])
        for object_id in (box_id, sphere_id, closed_id, open_id, inward_id):
            sites = self.adapter._optimizer_surface_sites(
                object_id, np.zeros(3, dtype=float)
            )
            self.assertGreater(len(sites), 0)
            self.assertTrue(
                np.isfinite([site["position_m"] for site in sites]).all()
            )
            self.assertTrue(np.isfinite([site["normal"] for site in sites]).all())

    def test_closed_mesh_measurement_snapshot_and_optimizer_target_share_provider(self):
        mesh_id = self._add_triangle_cube(
            "tumour_mask", size_m=0.04, position=(0.01, -0.02, 0.03)
        )
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": mesh_id, "role": "measurement"},
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": mesh_id,
                        "quality": "preview",
                        "target_uT": 25.0,
                        "tolerance_pct": 7.5,
                    },
                },
            ]
        )
        sample = self.adapter.measurement_sample_points(mesh_id)
        self.assertEqual(sample["shape"], "triangles")
        self.assertEqual(sample["shape_label"], "Closed mesh")
        self.assertEqual(len(sample["local_points_m"]), 7**3)
        self.assertAlmostEqual(sample["volume_cm3"], 64.0)
        np.testing.assert_allclose(sample["centre_m"], [0.01, -0.02, 0.03])

        result = self.adapter.analyze_measurement(mesh_id)
        snapshot_id = self.adapter.create_snapshot(result, "Tumour mesh field")
        self.assertEqual(self.adapter.snapshot_status(snapshot_id), "current")
        self.assertEqual(
            self.adapter.snapshot_result(snapshot_id)["shape_label"], "Closed mesh"
        )
        np.testing.assert_array_equal(
            self.adapter.snapshot_result(snapshot_id)["local_points_m"],
            result["local_points_m"],
        )

        target = next(
            item for item in self.adapter.optimizer_targets() if item["id"] == mesh_id
        )
        self.assertEqual(target["shape"], "triangles")
        self.assertEqual(target["shape_label"], "Closed mesh")
        self.assertEqual(target["dimensions_mm"], [40.0, 40.0, 40.0])
        self.assertEqual(target["position_mm"], [10.0, -20.0, 30.0])
        self.assertIsNone(target["diameter_mm"])
        self.assertIn("64 cm³", target["summary"])

        settings = self.adapter.optimizer_default_settings(mesh_id)
        settings.update(
            {
                "coil_count_optimize": False,
                "coil_count_fixed": 2,
                "placement_optimize": False,
                "diameter_optimize": False,
                "diameter_fixed_mm": 200.0,
                "turns_optimize": False,
                "turns_fixed": 100,
                "awg_optimize": False,
                "awg_fixed": 30,
                "axial_width_optimize": False,
                "axial_width_fixed_mm": 8.0,
                "minimum_in_band_pct": 0.0,
                "max_current_a": 1.0,
                "max_voltage_v": 1000.0,
                "max_power_per_coil_w": 1000.0,
                "max_total_power_w": 2000.0,
                "max_total_mass_g": 5000.0,
                "population": 8,
                "generations": 1,
                "finalist_count": 1,
                "seed": 4321,
                "coarse_quality": "preview",
                "final_quality": "preview",
                "finite_pack_order": 1,
            }
        )
        optimizer_result = self.adapter.run_optimizer(settings)
        self.assertEqual(optimizer_result["target_shape"], "triangles")
        self.assertGreaterEqual(len(optimizer_result["finalists"]), 1)

        # If a later edit opens the target surface, its frozen snapshot remains
        # viewable but becomes stale; the still-present object is not "missing".
        self.adapter._mesh(mesh_id)["mesh"]["faces"] = self.adapter._mesh(mesh_id)[
            "mesh"
        ]["faces"][:-2]
        self.assertFalse(self.adapter.supports_measurement(mesh_id))
        self.assertEqual(self.adapter.snapshot_status(snapshot_id), "stale")
        self.assertNotIn(mesh_id, {item["id"] for item in self.adapter.optimizer_targets()})

    def test_cylinder_object_supports_editing_measurement_and_optimizer_target(self):
        cylinder_id = self.adapter.add_template("guide_cylinder")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": cylinder_id,
                        "name": "diameter",
                        "value": 0.08,
                    },
                },
                {
                    "method": "set_param",
                    "params": {
                        "object_id": cylinder_id,
                        "name": "height",
                        "value": 0.12,
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": cylinder_id, "role": "measurement"},
                },
            ]
        )

        params = {item["name"]: item["value"] for item in self.adapter.get_params(cylinder_id)}
        self.assertAlmostEqual(params["diameter"], 0.08)
        self.assertAlmostEqual(params["height"], 0.12)
        capabilities = self.adapter.geometry_capabilities(cylinder_id)
        self.assertTrue(capabilities["measurement"])
        self.assertTrue(capabilities["exclusion"])
        self.assertTrue(capabilities["optimizer_placement_surface"])

        info = self.adapter.measurement_geometry_info(cylinder_id)
        self.assertEqual(info["shape_label"], "Cylinder")
        self.assertEqual(info["dimensions_mm"], [80.0, 80.0, 120.0])
        self.assertAlmostEqual(info["volume_cm3"], math.pi * 40.0**2 * 120.0 / 1000.0)
        samples = self.adapter.measurement_sample_points(cylinder_id, quality="preview")
        local = np.asarray(samples["local_points_m"], dtype=float)
        self.assertGreater(len(local), 0)
        self.assertTrue(np.all(np.linalg.norm(local[:, :2], axis=1) <= 0.04 + 1e-12))
        self.assertTrue(np.all(np.abs(local[:, 2]) <= 0.06 + 1e-12))

        target = next(
            item for item in self.adapter.optimizer_targets() if item["id"] == cylinder_id
        )
        self.assertEqual(target["shape"], "cylinder")
        self.assertEqual(target["diameter_mm"], 80.0)
        self.assertEqual(target["height_mm"], 120.0)
        self.assertIn("Cylinder", target["summary"])

        reopened = StudioAdapter(self.adapter.document())
        self.assertEqual(reopened.get_object(cylinder_id)["geometry_kind"], "cylinder")
        self.assertTrue(reopened.figure()["data"])

    def test_box_measurement_volume_is_also_an_optimizer_target(self):
        box_id = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": box_id,
                        "name": "dimension",
                        "value": [0.08, 0.06, 0.04],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box_id, "role": "measurement"},
                },
            ]
        )
        target = next(
            item for item in self.adapter.optimizer_targets() if item["id"] == box_id
        )
        self.assertEqual(target["shape"], "box")
        self.assertEqual(target["dimensions_mm"], [80.0, 60.0, 40.0])
        self.assertAlmostEqual(target["volume_cm3"], 192.0)
        self.assertIn("80 × 60 × 40 mm", target["summary"])

    def test_drc_closed_imported_mesh_checks_hole_and_inverse_containment(self):
        self.adapter.remove("coil_b")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        self._enable_test_assembly("coil_a")
        field_before = self.adapter.field_fingerprint(FINGERPRINT_POINTS)
        scan_id = self._add_triangle_cube("head_scan", size_m=0.04)
        self.adapter.set_visible(scan_id, False)

        report = self.adapter.run_drc()
        clear = self._drc_pair(report, "coil_a", scan_id)
        self.assertEqual(clear["status"], "clear")
        self.assertEqual(clear["verdict_basis"], "closed_mesh")
        self.assertGreater(clear["signed_clearance_mm"], 60.0)
        self.assertTrue(clear["mesh_health"]["containment_reliable"])
        health_row = next(
            item
            for item in report["results"]
            if item["object_a_id"] == scan_id and item["rule"] == "Imported mesh health"
        )
        self.assertEqual(health_row["status"], "mesh_ok")

        # A closed exclusion shell surrounding the complete ring is an
        # intersection even though the two triangle/assembly surfaces do not cross.
        vertices, faces = self.adapter._mesh_local_vertices_faces(
            {"mesh": {"kind": "box", "dimension": [0.3, 0.3, 0.3]}}
        )
        scan = self.adapter._mesh(scan_id)
        scan["mesh"]["vertices"] = vertices.tolist()
        scan["mesh"]["faces"] = faces.tolist()
        containing = self._drc_pair(self.adapter.run_drc(), "coil_a", scan_id)
        self.assertEqual(containing["status"], "intersecting")
        self.assertEqual(containing["verdict_basis"], "closed_mesh_containment")
        self.assertLess(containing["signed_clearance_mm"], 0.0)
        np.testing.assert_array_equal(
            self.adapter.field_fingerprint(FINGERPRINT_POINTS), field_before
        )

    def test_drc_open_imported_mesh_is_surface_only_and_still_finds_contact(self):
        self.adapter.remove("coil_b")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        self._enable_test_assembly("coil_a")
        scan_id = self._add_triangle_cube(
            "open_head_scan", size_m=0.02, open_surface=True
        )
        report = self.adapter.run_drc()
        surface_only = self._drc_pair(report, "coil_a", scan_id)
        self.assertEqual(surface_only["status"], "clearance_only")
        self.assertEqual(surface_only["verdict_basis"], "surface_only")
        self.assertFalse(surface_only["mesh_health"]["containment_reliable"])
        warning = next(
            item
            for item in report["results"]
            if item["object_a_id"] == scan_id and item["rule"] == "Imported mesh health"
        )
        self.assertEqual(warning["status"], "mesh_warning")

        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": scan_id,
                        "position": [0.1, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        self.adapter.set_visible(scan_id, False)
        contact = self._drc_pair(self.adapter.run_drc(), "coil_a", scan_id)
        self.assertEqual(contact["status"], "intersecting")
        self.assertEqual(contact["verdict_basis"], "surface_intersection")
        self.assertLess(contact["signed_clearance_mm"], 0.0)

    def test_drc_rotated_coil_intersects_imported_mesh_at_transformed_assembly(self):
        self.adapter.remove("coil_b")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.012, -0.008, 0.02],
                        "orientation": [21.0, 37.0, -14.0],
                    },
                }
            ]
        )
        self._enable_test_assembly("coil_a")
        entry = self.adapter._drc_coil_entry("coil_a")
        solid = entry["solids"][0]
        radius = 0.5 * (solid["inner_radius_m"] + solid["outer_radius_m"])
        contact_point = solid["centre"] + solid["rotation"] @ np.asarray(
            [radius, 0.0, 0.0]
        )
        scan_id = self._add_triangle_cube(
            "rotated_contact", size_m=0.008, position=contact_point
        )
        result = self._drc_pair(self.adapter.run_drc(), "coil_a", scan_id)
        self.assertEqual(result["status"], "intersecting")
        self.assertEqual(result["verdict_basis"], "surface_intersection")

    def test_drc_coil_spacing_settings_roundtrip_and_staleness(self):
        for coil_id in ("coil_a", "coil_b"):
            self._enable_test_assembly(coil_id)
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_b",
                        "position": [0.0, 0.0, 0.005],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
            ]
        )
        self.adapter.set_drc_settings(
            {
                "default_clearance_mm": 3.0,
                "coil_to_coil_clearance_mm": 4.0,
                "mesh_tolerance_mm": 0.25,
            }
        )
        report = self.adapter.run_drc()
        result = self._drc_pair(report, "coil_a", "coil_b")
        self.assertEqual(result["status"], "intersecting")
        self.assertFalse(self.adapter.drc_report_is_stale(report))

        reopened = StudioAdapter(self.adapter.document())
        self.assertEqual(
            reopened.get_drc_settings(),
            {
                "default_clearance_mm": 3.0,
                "coil_to_coil_clearance_mm": 4.0,
                "mesh_tolerance_mm": 0.25,
            },
        )
        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_b",
                        "position": [0.0, 0.0, 0.02],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        self.assertTrue(self.adapter.drc_report_is_stale(report))
        clear = self._drc_pair(self.adapter.run_drc(), "coil_a", "coil_b")
        self.assertEqual(clear["status"], "clear")
        self.assertGreaterEqual(clear["signed_clearance_mm"], 6.0 - 1e-6)

    def test_legacy_coils_gain_defaults_without_changing_field(self):
        legacy = compact_workflow_scene()
        legacy.pop("field_workbench", None)
        adapter = StudioAdapter(legacy)
        physical = adapter.coil_physical_properties("coil_a")
        self.assertEqual(physical["turns"], 1)
        self.assertAlmostEqual(physical["drive_current_a"], 0.07)
        self.assertAlmostEqual(physical["field_scale_factor"], 1.0)
        self.assertFalse(physical["physical_geometry_enabled"])
        self.assertIsNone(physical["calculations"]["assembly"])
        engine_current = {
            item["name"]: item["value"] for item in adapter.get_params("coil_a")
        }["current"]
        self.assertAlmostEqual(engine_current, 0.07)

    def test_old_coil_spec_derives_wire_resistance_from_its_saved_awg(self):
        scene = compact_workflow_scene()
        scene["field_workbench"]["coil_specs"]["coil_a"].pop(
            "resistance_20_ohm_per_km"
        )
        scene["field_workbench"]["coil_specs"]["coil_a"]["awg"] = 28
        adapter = StudioAdapter(scene)
        physical = adapter.coil_physical_properties("coil_a")
        self.assertAlmostEqual(
            physical["resistance_20_ohm_per_km"],
            awg_resistance_20_ohm_per_km(28),
        )

    def test_pre_factor_coil_scene_migrates_without_changing_field(self):
        scene = compact_workflow_scene()
        spec = scene["field_workbench"]["coil_specs"]["coil_a"]
        spec.pop("field_scale_factor")
        spec["turns"] = 100
        spec["drive_current_a"] = 0.07
        scene["objects"][0]["params"]["current"] = 7.0

        adapter = StudioAdapter(scene)
        physical = adapter.coil_physical_properties("coil_a")
        engine_current = {
            item["name"]: item["value"] for item in adapter.get_params("coil_a")
        }["current"]
        self.assertAlmostEqual(physical["field_scale_factor"], 1.0)
        self.assertAlmostEqual(physical["drive_current_a"], 0.07)
        self.assertEqual(physical["turns"], 100)
        self.assertAlmostEqual(engine_current, 7.0)

    def test_custom_wire_resistance_is_undoable(self):
        nominal = self.adapter.coil_physical_properties("coil_a")[
            "resistance_20_ohm_per_km"
        ]
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "resistance_20_ohm_per_km": 412.5,
                    },
                }
            ]
        )
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_a")[
                "resistance_20_ohm_per_km"
            ],
            412.5,
        )
        self.adapter.undo()
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_a")[
                "resistance_20_ohm_per_km"
            ],
            nominal,
        )
        self.adapter.redo()
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_a")[
                "resistance_20_ohm_per_km"
            ],
            412.5,
        )

    def test_cloned_coil_preserves_physical_spec_and_roundtrip(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "turns": 484,
                        "drive_current_a": 0.07,
                        "field_scale_factor": 2.75,
                        "awg": 30,
                        "resistance_20_ohm_per_km": 347.75,
                    },
                }
            ]
        )
        clone_id = self.adapter.duplicate("coil_a")
        clone = self.adapter.coil_physical_properties(clone_id)
        self.assertEqual(clone["turns"], 484)
        self.assertAlmostEqual(clone["field_scale_factor"], 2.75)
        self.assertEqual(clone["awg"], 30)
        self.assertAlmostEqual(clone["resistance_20_ohm_per_km"], 347.75)
        self.assertAlmostEqual(clone["drive_current_a"], 0.07)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical-coil.magpy.json"
            self.adapter.save_file(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["field_workbench"]["version"], WORKBENCH_METADATA_VERSION)
            self.assertIn(clone_id, saved["field_workbench"]["coil_specs"])
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(path)
            reopened_clone = reopened.coil_physical_properties(clone_id)
            self.assertEqual(reopened_clone["turns"], 484)
            self.assertAlmostEqual(reopened_clone["field_scale_factor"], 2.75)
            self.assertAlmostEqual(reopened_clone["drive_current_a"], 0.07)
            self.assertAlmostEqual(
                reopened_clone["resistance_20_ohm_per_km"], 347.75
            )

    def test_save_reopen_preserves_document_and_field(self):
        before = self.adapter.field_fingerprint(FINGERPRINT_POINTS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.magpy.json"
            self.adapter.save_file(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["version"], 1)
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(path)
            after = reopened.field_fingerprint(FINGERPRINT_POINTS)
            np.testing.assert_array_equal(after, before)

    def test_fluxline_conductor_distance_uses_winding_pack_when_available(self):
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "physical_geometry_enabled": True,
                        "turns": 100,
                        "awg": 30,
                        "winding_axial_width_mm": 20.0,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 10.0,
                    },
                }
            ]
        )
        regions = self.adapter._fluxline_conductor_regions()
        # coil_a has R=100 mm and z=-50 mm. A point on the centreline is inside
        # the winding pack, while 8 mm radially outside the mean path is 3 mm
        # beyond a 5 mm half-build.
        on_pack = self.adapter._fluxline_conductor_distance_mm(
            np.asarray([100.0, 0.0, -50.0]), regions
        )
        outside = self.adapter._fluxline_conductor_distance_mm(
            np.asarray([108.0, 0.0, -50.0]), regions
        )
        self.assertAlmostEqual(on_pack, 0.0, places=9)
        self.assertAlmostEqual(outside, 3.0, places=6)

    def test_2d_streamline_cutoffs_can_reject_seeds(self):
        axis = np.linspace(-10.0, 10.0, 11)
        vectors = np.zeros((11, 11, 2), dtype=float)
        vectors[:, :, 0] = 1.0
        unblocked = self.adapter._streamline_paths_2d(
            axis, axis, vectors, density=5, minimum_field_cutoff_percent=0.0
        )
        blocked = self.adapter._streamline_paths_2d(
            axis,
            axis,
            vectors,
            density=5,
            minimum_field_cutoff_percent=0.0,
            blocked=lambda _point: True,
        )
        self.assertTrue(unblocked)
        self.assertEqual(blocked, [])

    def test_active_coil_scope_is_independent_of_scene_enabled_state(self):
        self.adapter.set_coils_enabled(["coil_b"], False)
        point = [[0.0, 0.0, 0.0]]

        scene_default = np.asarray(
            self.adapter.get_field(points=point, field="B")["values"], dtype=float
        )
        only_a = np.asarray(
            self.adapter.get_field(
                points=point, field="B", active_coil_ids=["coil_a"]
            )["values"],
            dtype=float,
        )
        only_b = np.asarray(
            self.adapter.get_field(
                points=point, field="B", active_coil_ids=["coil_b"]
            )["values"],
            dtype=float,
        )
        both = np.asarray(
            self.adapter.get_field(
                points=point, field="B", active_coil_ids=["coil_a", "coil_b"]
            )["values"],
            dtype=float,
        )

        np.testing.assert_allclose(scene_default, only_a, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(only_a, only_b, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(both, only_a + only_b, rtol=1e-12, atol=1e-15)
        self.assertFalse(self.adapter.coil_enabled("coil_b"))

    def test_planar_field_map(self):
        figure = self.adapter.field_map(
            plane="xz",
            offset_mm=0,
            span_mm=300,
            resolution=20,
            field="B",
            component="magnitude",
        )
        self.assertEqual(figure["data"][0]["type"], "heatmap")
        self.assertEqual(len(figure["data"][0]["z"]), 20)
        trace = figure["data"][0]
        self.assertEqual(trace["colorscale"], FIELD_HEAT_COLOURSCALE)
        values = np.asarray(trace["z"], dtype=float)
        self.assertAlmostEqual(float(trace["zmin"]), float(np.min(values)))
        self.assertAlmostEqual(float(trace["zmax"]), float(np.max(values)))
        self.assertNotIn("zmid", trace)
        self.assertEqual(trace["colorbar"]["title"]["text"], "|B| (µT)")
        self.assertIn("µT", trace["hovertemplate"])
        self.assertEqual(figure["layout"]["dragmode"], "pan")

    def test_planar_field_map_distinguishes_axis_magnitude_from_signed_component(self):
        signed = self.adapter.field_map(
            plane="xy", offset_mm=0.0, span_mm=160.0, resolution=14,
            field="B", component="y",
        )
        magnitude = self.adapter.field_map(
            plane="xy", offset_mm=0.0, span_mm=160.0, resolution=14,
            field="B", component="abs_y",
        )
        signed_trace = signed["data"][0]
        magnitude_trace = magnitude["data"][0]
        signed_values = np.asarray(signed_trace["z"], dtype=float)
        magnitude_values = np.asarray(magnitude_trace["z"], dtype=float)
        self.assertTrue(np.allclose(magnitude_values, np.abs(signed_values)))
        self.assertEqual(signed_trace["colorscale"], FIELD_SIGNED_COLOURSCALE)
        self.assertAlmostEqual(float(signed_trace["zmin"]), -float(signed_trace["zmax"]), places=12)
        self.assertEqual(float(signed_trace["zmid"]), 0.0)
        self.assertEqual(signed_trace["colorbar"]["title"]["text"], "By (µT)")
        self.assertEqual(magnitude_trace["colorscale"], FIELD_HEAT_COLOURSCALE)
        self.assertGreaterEqual(float(magnitude_trace["zmin"]), 0.0)
        self.assertEqual(magnitude_trace["colorbar"]["title"]["text"], "|By| (µT)")

        logged = self.adapter.field_map(
            plane="xy", offset_mm=0.0, span_mm=160.0, resolution=14,
            field="B", component="abs_y", logarithmic=True,
        )
        self.assertIn("log₁₀ |By|", logged["data"][0]["colorbar"]["title"]["text"] )
        with self.assertRaises(StudioOperationError):
            self.adapter.field_map(
                plane="xy", offset_mm=0.0, span_mm=160.0, resolution=14,
                field="B", component="y", logarithmic=True,
            )

    def test_planar_field_map_supports_contour_overlays(self):
        figure = self.adapter.field_map(
            plane="xy",
            offset_mm=0,
            span_mm=200,
            resolution=18,
            field="B",
            component="magnitude",
            show_contours=True,
            contour_count=7,
            contour_labels=True,
        )
        contour_traces = [
            trace
            for trace in figure["data"]
            if trace.get("meta", {}).get("field_workbench_field_map_contours")
        ]
        self.assertEqual(len(contour_traces), 1)
        contour = contour_traces[0]
        self.assertEqual(contour["type"], "contour")
        self.assertEqual(contour["contours"]["coloring"], "none")
        self.assertTrue(contour["contours"]["showlabels"])
        self.assertGreater(float(contour["contours"]["size"]), 0.0)
        self.assertFalse(contour["showscale"])

    def test_planar_field_map_can_hide_coordinate_grid(self):
        visible = self.adapter.field_map(
            plane="xz", offset_mm=12.0, span_mm=160.0, resolution=12,
            field="B", component="magnitude", show_grid=True,
        )
        hidden = self.adapter.field_map(
            plane="xz", offset_mm=12.0, span_mm=160.0, resolution=12,
            field="B", component="magnitude", show_grid=False,
        )
        self.assertTrue(visible["layout"]["xaxis"]["showgrid"])
        self.assertTrue(visible["layout"]["yaxis"]["showgrid"])
        self.assertFalse(hidden["layout"]["xaxis"]["showgrid"])
        self.assertFalse(hidden["layout"]["yaxis"]["showgrid"])
        self.assertEqual(visible["layout"]["xaxis"]["tickvals"], [-80.0, -40.0, 0.0, 40.0, 80.0])
        self.assertEqual(visible["layout"]["yaxis"]["tickvals"], [-80.0, -40.0, 0.0, 40.0, 80.0])

    def test_planar_field_map_rejects_invalid_contour_count(self):
        with self.assertRaises(StudioOperationError):
            self.adapter.field_map(
                plane="xy",
                offset_mm=0,
                span_mm=200,
                resolution=16,
                field="B",
                component="magnitude",
                contour_count=0,
            )

    def test_planar_field_map_overlays_visible_object_cross_sections(self):
        guide = self.adapter.add_template("guide_box")
        figure = self.adapter.field_map(
            plane="xy",
            offset_mm=0.0,
            span_mm=300.0,
            resolution=12,
            field="B",
            component="magnitude",
            show_object_outlines=True,
        )
        sections = [
            trace
            for trace in figure["data"]
            if trace.get("meta", {}).get("field_workbench_field_map_section")
        ]
        self.assertEqual(len(sections), 1)
        self.assertEqual(
            sections[0].get("meta", {}).get("field_workbench_object_id"), guide
        )
        x_values = np.asarray([value for value in sections[0]["x"] if value is not None])
        y_values = np.asarray([value for value in sections[0]["y"] if value is not None])
        self.assertAlmostEqual(float(np.min(x_values)), -50.0, places=6)
        self.assertAlmostEqual(float(np.max(x_values)), 50.0, places=6)
        self.assertAlmostEqual(float(np.min(y_values)), -50.0, places=6)
        self.assertAlmostEqual(float(np.max(y_values)), 50.0, places=6)

        hidden = self.adapter.field_map(
            plane="xy",
            offset_mm=0.0,
            span_mm=300.0,
            resolution=12,
            field="B",
            component="magnitude",
            show_object_outlines=False,
        )
        self.assertFalse(
            any(
                trace.get("meta", {}).get("field_workbench_field_map_section")
                for trace in hidden["data"]
            )
        )

        filtered = self.adapter.field_map(
            plane="xy",
            offset_mm=0.0,
            span_mm=300.0,
            resolution=12,
            field="B",
            component="magnitude",
            show_object_outlines=True,
            outline_object_ids=[],
        )
        self.assertFalse(
            any(
                trace.get("meta", {}).get("field_workbench_field_map_section")
                for trace in filtered["data"]
            )
        )

        choices = self.adapter.field_map_outline_objects()
        self.assertIn(guide, {choice["id"] for choice in choices})

    def test_field_map_progress_reports_determinate_sample_counts(self):
        updates = []
        resolution = 48
        self.adapter.set_coil_modeling_method("centreline")
        self.adapter.field_map(
            plane="xy",
            offset_mm=0.0,
            span_mm=120.0,
            resolution=resolution,
            field="B",
            component="magnitude",
            progress_callback=updates.append,
        )

        total = resolution * resolution
        self.assertGreater(total, FIELD_PROGRESS_POINT_CHUNK)
        self.assertGreaterEqual(len(updates), 3)
        self.assertEqual(updates[0]["completed"], 0)
        self.assertEqual(updates[0]["total"], total)
        self.assertEqual(updates[-1]["completed"], total)
        self.assertEqual(updates[-1]["total"], total)
        self.assertTrue(
            all(
                first["completed"] <= second["completed"]
                for first, second in zip(updates, updates[1:])
            )
        )
        self.assertTrue(all(update["unit"] == "field samples" for update in updates))

    def test_field_backend_propagates_user_cancellation_without_wrapping(self):
        def cancel_on_first_update(_update):
            raise FieldCalculationCancelled("cancelled by test")

        with self.assertRaises(FieldCalculationCancelled):
            self.adapter.get_field(
                points=[[0.0, 0.0, 0.0]],
                field="B",
                progress_callback=cancel_on_first_update,
            )

    def test_planar_field_map_accepts_optional_manual_colour_limits(self):
        figure = self.adapter.field_map(
            plane="xy",
            offset_mm=0,
            span_mm=200,
            resolution=16,
            field="B",
            component="magnitude",
            colour_minimum=0.0,
            colour_maximum=2000.0,
        )
        trace = figure["data"][0]
        self.assertEqual(float(trace["zmin"]), 0.0)
        self.assertEqual(float(trace["zmax"]), 2000.0)

        upper_only = self.adapter.field_map(
            plane="xy",
            offset_mm=0,
            span_mm=200,
            resolution=16,
            field="B",
            component="magnitude",
            colour_maximum=2000.0,
        )
        upper_trace = upper_only["data"][0]
        self.assertEqual(float(upper_trace["zmax"]), 2000.0)
        self.assertLess(float(upper_trace["zmin"]), float(upper_trace["zmax"]))

        with self.assertRaises(StudioOperationError):
            self.adapter.field_map(
                plane="xy",
                offset_mm=0,
                span_mm=200,
                resolution=16,
                field="B",
                component="magnitude",
                colour_minimum=2000.0,
                colour_maximum=1000.0,
            )

    def test_layered_3d_field_map_uses_shared_scale_and_slice_slider(self):
        figure = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=240.0,
            slice_axis="z",
            slice_count=5,
            resolution=12,
            field="B",
            component="magnitude",
            include_scene=False,
        )
        surfaces = [trace for trace in figure["data"] if trace.get("type") == "surface"]
        self.assertEqual(len(surfaces), 5)
        self.assertTrue(all(trace.get("coloraxis") == "coloraxis" for trace in surfaces))
        self.assertEqual(len(surfaces[0]["surfacecolor"]), 12)
        self.assertEqual(len(surfaces[0]["surfacecolor"][0]), 12)
        self.assertEqual(figure["layout"]["scene"]["aspectmode"], "cube")
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = figure["layout"]["scene"][axis_name]
            self.assertTrue(axis["visible"])
            self.assertTrue(axis["showgrid"])
            self.assertTrue(axis["zeroline"])
            self.assertTrue(axis["showbackground"])
            self.assertTrue(axis["showline"])
            self.assertTrue(axis["showticklabels"])
            self.assertEqual(axis["tickmode"], "array")
            self.assertEqual(len(axis["tickvals"]), 5)
            self.assertEqual(len(axis["ticktext"]), 5)
        self.assertEqual(
            [step["label"] for step in figure["layout"]["sliders"][0]["steps"]],
            ["1", "3", "5"],
        )
        final_visibility = figure["layout"]["sliders"][0]["steps"][-1]["args"][0][
            "visible"
        ]
        self.assertEqual(sum(bool(value) for value in final_visibility[-5:]), 5)
        self.assertIn("µT", figure["layout"]["coloraxis"]["colorbar"]["title"]["text"])
        self.assertEqual(
            figure["layout"]["coloraxis"]["colorscale"],
            field_heat_colourscale_with_opacity(0.15, 0.40),
        )
        self.assertTrue(all(float(trace.get("opacity", 0.0)) == 1.0 for trace in surfaces))

        gridless = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=80.0,
            slice_axis="z",
            slice_count=1,
            resolution=6,
            field="B",
            component="magnitude",
            include_scene=False,
            show_grid=False,
        )
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = gridless["layout"]["scene"][axis_name]
            self.assertFalse(axis["visible"])
            self.assertFalse(axis["showgrid"])
            self.assertFalse(axis["zeroline"])
            self.assertFalse(axis["showbackground"])
            self.assertFalse(axis["showline"])
            self.assertFalse(axis["showticklabels"])
        self.assertFalse(
            any(trace.get("name") == "Map volume" for trace in gridless["data"])
        )

        signed = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="x",
            slice_count=3,
            resolution=10,
            field="B",
            component="z",
            include_scene=False,
        )
        colour = signed["layout"]["coloraxis"]
        signed_values = np.concatenate(
            [
                np.asarray(trace["surfacecolor"], dtype=float).reshape(-1)
                for trace in signed["data"]
                if trace.get("type") == "surface"
            ]
        )
        signed_extent = float(np.max(np.abs(signed_values)))
        self.assertAlmostEqual(float(colour["cmin"]), -signed_extent)
        self.assertAlmostEqual(float(colour["cmax"]), signed_extent)
        self.assertAlmostEqual(float(colour["cmid"]), 0.0)
        self.assertEqual(
            colour["colorscale"], field_signed_colourscale_with_opacity(0.15, 0.40)
        )

    def test_slice_3d_field_map_uses_intensity_linked_opacity(self):
        figure = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="z",
            slice_count=3,
            resolution=8,
            field="B",
            component="magnitude",
            include_scene=False,
            volume_opacity_lower=0.04,
            volume_opacity_upper=0.32,
        )
        surfaces = [trace for trace in figure["data"] if trace.get("type") == "surface"]
        self.assertEqual(len(surfaces), 3)
        self.assertEqual(
            figure["layout"]["coloraxis"]["colorscale"],
            field_heat_colourscale_with_opacity(0.04, 0.32),
        )
        self.assertTrue(all(float(trace["opacity"]) == 1.0 for trace in surfaces))

        opaque = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="z",
            slice_count=3,
            resolution=8,
            field="B",
            component="magnitude",
            include_scene=False,
            intensity_opacity_enabled=False,
            volume_opacity_lower=0.04,
            volume_opacity_upper=0.32,
        )
        self.assertEqual(opaque["layout"]["coloraxis"]["colorscale"], FIELD_HEAT_COLOURSCALE)

    def test_3d_field_opacity_levels_clamp_and_interpolate_at_configured_scale_percentages(self):
        slices = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="z",
            slice_count=3,
            resolution=8,
            field="B",
            component="magnitude",
            include_scene=False,
            volume_opacity_lower=0.0,
            volume_opacity_upper=0.5,
            volume_opacity_lower_level=0.20,
            volume_opacity_upper_level=0.80,
        )
        self.assertEqual(
            slices["layout"]["coloraxis"]["colorscale"],
            field_heat_colourscale_with_opacity(0.0, 0.5, 0.20, 0.80),
        )
        scale = slices["layout"]["coloraxis"]["colorscale"]
        at_lower = next(colour for stop, colour in scale if abs(float(stop) - 0.20) < 1e-12)
        at_upper = next(colour for stop, colour in scale if abs(float(stop) - 0.80) < 1e-12)
        self.assertTrue(str(at_lower).endswith(",0)"))
        self.assertTrue(str(at_upper).endswith(",0.5)"))

        volume = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="z",
            slice_count=3,
            resolution=8,
            field="B",
            component="magnitude",
            view_type="volume",
            include_scene=False,
            volume_opacity_lower=0.0,
            volume_opacity_upper=0.5,
            volume_opacity_lower_level=0.20,
            volume_opacity_upper_level=0.80,
        )
        trace = next(item for item in volume["data"] if item.get("type") == "volume")
        self.assertEqual(
            trace["opacityscale"],
            [[0.0, 0.0], [0.20, 0.0], [0.80, 0.5], [1.0, 0.5]],
        )

    def test_3d_signed_component_uses_zero_centred_scale_and_absolute_strength_opacity(self):
        figure = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0], span_mm=120.0, slice_axis="z",
            slice_count=3, resolution=8, field="B", component="y",
            include_scene=False, volume_opacity_lower=0.15, volume_opacity_upper=0.40,
        )
        axis = figure["layout"]["coloraxis"]
        self.assertAlmostEqual(float(axis["cmin"]), -float(axis["cmax"]), places=12)
        self.assertEqual(float(axis["cmid"]), 0.0)
        self.assertEqual(
            axis["colorscale"],
            field_signed_colourscale_with_opacity(0.15, 0.40),
        )

        volume = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0], span_mm=120.0, slice_axis="z",
            slice_count=3, resolution=8, field="B", component="y",
            view_type="volume", include_scene=False,
            volume_opacity_lower=0.15, volume_opacity_upper=0.40,
        )
        trace = next(item for item in volume["data"] if item.get("type") == "volume")
        self.assertAlmostEqual(float(trace["isomin"]), -float(trace["isomax"]), places=12)
        self.assertEqual(trace["colorscale"], FIELD_SIGNED_COLOURSCALE)
        self.assertEqual(trace["opacityscale"], [[0.0, 0.40], [0.5, 0.15], [1.0, 0.40]])

    def test_full_volume_3d_field_map_uses_intensity_linked_opacity(self):
        figure = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="z",
            slice_count=5,
            resolution=8,
            field="B",
            component="magnitude",
            view_type="volume",
            include_scene=False,
            intensity_opacity_enabled=True,
            volume_opacity_lower=0.02,
            volume_opacity_upper=0.18,
        )

        volumes = [trace for trace in figure["data"] if trace.get("type") == "volume"]
        self.assertEqual(len(volumes), 1)
        volume = volumes[0]
        self.assertEqual(len(volume["value"]), 8 ** 3)
        self.assertEqual(volume["opacityscale"], [[0.0, 0.02], [1.0, 0.18]])
        self.assertLess(float(volume["isomin"]), float(volume["isomax"]))
        self.assertNotIn("sliders", figure["layout"])
        self.assertIn("Full Volume", figure["layout"]["title"]["text"])
        self.assertIn("µT", volume["colorbar"]["title"]["text"])

        opaque = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="z",
            slice_count=3,
            resolution=6,
            field="B",
            component="magnitude",
            view_type="volume",
            include_scene=False,
            intensity_opacity_enabled=False,
            volume_opacity_lower=0.25,
            volume_opacity_upper=0.50,
        )
        opaque_volume = next(trace for trace in opaque["data"] if trace.get("type") == "volume")
        self.assertEqual(opaque_volume["opacityscale"], [[0.0, 1.0], [1.0, 1.0]])

        inverted = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=120.0,
            slice_axis="z",
            slice_count=3,
            resolution=8,
            field="B",
            component="magnitude",
            view_type="volume",
            include_scene=False,
            volume_opacity_lower=0.5,
            volume_opacity_upper=0.1,
        )
        inverted_volume = next(
            trace for trace in inverted["data"] if trace.get("type") == "volume"
        )
        self.assertEqual(inverted_volume["opacityscale"], [[0.0, 0.5], [1.0, 0.1]])

    def test_3d_field_map_passes_selected_scene_objects_to_context_figure(self):
        guide = self.adapter.add_template("guide_box")
        choices = self.adapter.field_volume_map_scene_objects()
        self.assertIn(guide, {choice["id"] for choice in choices})
        with mock.patch.object(
            self.adapter,
            "_field_volume_scene_figure",
            return_value={"data": [], "layout": {}},
        ) as scene_figure:
            self.adapter.field_volume_map(
                centre_mm=[0.0, 0.0, 0.0],
                span_mm=120.0,
                slice_axis="z",
                slice_count=1,
                resolution=10,
                field="B",
                component="magnitude",
                include_scene=True,
                scene_object_ids=[guide],
                scene_object_opacities={guide: 0.4},
            )
        scene_figure.assert_called_once_with(
            [guide], scene_object_opacities={guide: 0.4}
        )

    def test_3d_field_map_scene_opacity_is_a_viewer_only_absolute_override(self):
        guide = self.adapter.add_template("guide_box")
        original_opacity = float(self.adapter._mesh(guide).get("opacity", 0.28))
        figure = self.adapter._field_volume_scene_figure(
            [guide], scene_object_opacities={guide: 0.5}
        )
        guide_traces = [
            trace
            for trace in figure.get("data", [])
            if isinstance(trace.get("meta"), dict)
            and trace["meta"].get("field_workbench_object_id") == guide
            and trace.get("type") == "mesh3d"
        ]
        self.assertTrue(guide_traces)
        self.assertAlmostEqual(guide_traces[0]["opacity"], 0.5)
        self.assertAlmostEqual(float(self.adapter._mesh(guide).get("opacity", 0.28)), original_opacity)

        with self.assertRaises(StudioOperationError):
            self.adapter._field_volume_scene_figure(
                [guide], scene_object_opacities={guide: 1.1}
            )

    def test_single_3d_field_map_slice_uses_the_central_plane(self):
        centre = [12.0, -8.0, 37.5]
        figure = self.adapter.field_volume_map(
            centre_mm=centre,
            span_mm=240.0,
            slice_axis="z",
            slice_count=1,
            resolution=10,
            field="B",
            component="magnitude",
            include_scene=False,
        )

        surfaces = [
            trace for trace in figure["data"] if trace.get("type") == "surface"
        ]
        self.assertEqual(len(surfaces), 1)
        self.assertTrue(
            np.allclose(np.asarray(surfaces[0]["z"], dtype=float), centre[2])
        )
        self.assertIn("Z = 37.5 mm", surfaces[0]["name"])
        self.assertEqual(
            [step["label"] for step in figure["layout"]["sliders"][0]["steps"]],
            ["1"],
        )

    def test_layered_3d_field_map_accepts_manual_limits_in_display_units(self):
        figure = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=160.0,
            slice_axis="z",
            slice_count=3,
            resolution=10,
            field="B",
            component="magnitude",
            include_scene=False,
            colour_minimum=50.0,
            colour_maximum=500.0,
        )
        colour = figure["layout"]["coloraxis"]
        self.assertEqual(float(colour["cmin"]), 50.0)
        self.assertEqual(float(colour["cmax"]), 500.0)

        logarithmic = self.adapter.field_volume_map(
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=160.0,
            slice_axis="z",
            slice_count=3,
            resolution=10,
            field="B",
            component="magnitude",
            include_scene=False,
            logarithmic=True,
            colour_minimum=10.0,
            colour_maximum=1000.0,
        )
        log_colour = logarithmic["layout"]["coloraxis"]
        self.assertEqual(float(log_colour["cmin"]), 1.0)
        self.assertEqual(float(log_colour["cmax"]), 3.0)

    def test_sensor_plot_can_limit_calculation_to_selected_sensors(self):
        point_id = self.adapter.add_template("sensor")

        all_sensors = self.adapter.sensor_plot("B")
        self.assertEqual(
            {trace["name"] for trace in all_sensors["data"]},
            {"Axis sensor 1", "Field sensor 1"},
        )

        point_only = self.adapter.sensor_plot("B", sensor_ids=[point_id])
        self.assertEqual(len(point_only["data"]), 1)
        self.assertEqual(point_only["data"][0]["name"], "Field sensor 1")
        self.assertEqual(point_only["data"][0]["x"], [0])

        axis_only = self.adapter.sensor_plot_export_data(
            "B", sensor_ids=["axis_sensor"]
        )
        self.assertTrue(axis_only["rows"])
        self.assertEqual(
            {row["series"] for row in axis_only["rows"]}, {"Axis sensor 1"}
        )

        with self.assertRaisesRegex(StudioOperationError, "Select at least one sensor"):
            self.adapter.sensor_plot("B", sensor_ids=[])
        with self.assertRaisesRegex(StudioOperationError, "no longer available"):
            self.adapter.sensor_plot("B", sensor_ids=["missing_sensor"])

    def test_lone_point_sensor_plot_uses_its_numeric_field_reading(self):
        self.adapter.remove("axis_sensor")
        sensor_id = self.adapter.add_template("sensor")
        expected = self.adapter.field_at_mm([0.0, 0.0, 0.0], field="B")

        magnitude = self.adapter.sensor_plot("B")
        self.assertEqual(len(magnitude["data"]), 1)
        self.assertEqual(magnitude["data"][0]["name"], "Field sensor 1")
        self.assertEqual(magnitude["data"][0]["x"], [0])
        self.assertGreater(float(magnitude["data"][0]["y"][0]), 0.0)
        self.assertAlmostEqual(
            float(magnitude["data"][0]["y"][0]),
            float(expected["magnitude"][0]),
            places=16,
        )
        self.assertEqual(magnitude["layout"]["yaxis"]["range"][0], 0.0)
        self.assertGreater(
            magnitude["layout"]["yaxis"]["range"][1],
            magnitude["data"][0]["y"][0],
        )
        self.assertIn("T", magnitude["layout"]["annotations"][0]["text"])

        z_component = self.adapter.sensor_plot("Bz")
        self.assertAlmostEqual(
            float(z_component["data"][0]["y"][0]),
            float(np.asarray(expected["values"], dtype=float).reshape(-1, 3)[0, 2]),
            places=16,
        )
        export = self.adapter.sensor_plot_export_data("B")
        self.assertEqual(len(export["rows"]), 1)
        self.assertAlmostEqual(
            float(export["rows"][0]["value"]),
            float(expected["magnitude"][0]),
            places=16,
        )
        self.assertEqual(sensor_id, "sensor")

    def test_passive_geometry_roles_hierarchy_and_roundtrip(self):
        before = self.adapter.field_fingerprint(FINGERPRINT_POINTS)
        group = self.adapter.add_template("collection")
        guide = self.adapter.add_template("guide_box")
        self.adapter.move_object(guide, group)
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": guide,
                        "role": "exclusion",
                        "colour": "#dc2626",
                        "opacity": 0.4,
                    },
                }
            ]
        )
        np.testing.assert_array_equal(self.adapter.field_fingerprint(FINGERPRINT_POINTS), before)
        obj = self.adapter.get_object(guide)
        self.assertEqual(obj["parent"], group)
        self.assertEqual(obj["role"], "exclusion")
        figure = self.adapter.figure(selected_id=guide)
        self.assertTrue(any("(exclusion)" in str(trace.get("name")) for trace in figure["data"]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.magpy.json"
            self.adapter.save_file(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("field_workbench", saved)
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(path)
            self.assertEqual(reopened.get_object(guide)["parent"], group)
            self.assertEqual(reopened.geometry_properties(guide)["role"], "exclusion")
            np.testing.assert_array_equal(reopened.field_fingerprint(FINGERPRINT_POINTS), before)

    def test_legacy_test_role_migrates_to_visual_without_changing_geometry(self):
        guide = self.adapter.add_template("guide_box")
        legacy_mesh = self.adapter._mesh(guide)
        legacy_mesh["role"] = "test"
        legacy_mesh["colour"] = "#16a34a"
        legacy_mesh["opacity"] = 0.42
        legacy_mesh["position"] = [0.01, -0.02, 0.03]
        legacy_document = self.adapter.document()

        reopened = StudioAdapter(legacy_document)
        properties = reopened.geometry_properties(guide)
        self.assertEqual(properties["role"], "visual")
        self.assertEqual(properties["colour"], "#16a34a")
        self.assertAlmostEqual(properties["opacity"], 0.42)
        self.assertNotIn("test", properties["eligible_roles"])
        self.assertEqual(
            reopened.get_transform(guide)["position"], [0.01, -0.02, 0.03]
        )

    def test_measurement_box_sampling_statistics_and_visuals(self):
        box = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": box,
                        "name": "dimension",
                        "value": [0.08, 0.06, 0.04],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": box,
                        "role": "measurement",
                        "colour": "#64748b",
                        "opacity": 0.28,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box,
                        "quality": "preview",
                        "target_uT": 0.6,
                        "tolerance_pct": 10.0,
                    },
                },
            ]
        )
        progress_updates = []
        result = self.adapter.analyze_measurement(
            box, progress_callback=progress_updates.append
        )
        stats = result["stats"]
        self.assertTrue(progress_updates)
        self.assertEqual(progress_updates[-1]["completed"], progress_updates[-1]["total"])
        self.assertIn("Volume analysis", progress_updates[-1]["stage"])
        self.assertEqual(result["shape"], "box")
        self.assertEqual(result["quality"], "preview")
        self.assertEqual(stats["samples_per_axis"], 7)
        self.assertEqual(stats["sample_count"], 7**3)
        self.assertEqual(stats["valid_sample_count"], 7**3)
        self.assertGreater(stats["mean_uT"], 0.0)
        self.assertGreaterEqual(stats["directional_consistency_pct"], 0.0)
        self.assertLessEqual(stats["directional_consistency_pct"], 100.0 + 1e-10)
        self.assertAlmostEqual(
            stats["below_target_pct"]
            + stats["in_band_pct"]
            + stats["above_target_pct"],
            100.0,
        )
        overlay = self.adapter.measurement_overlay(result, max_points=100)
        self.assertEqual(overlay["kind"], "measurement_samples")
        self.assertEqual(len(overlay["points_m"]), 100)
        figure = self.adapter.compose_figure(
            self.adapter.base_figure(), analysis_overlay=overlay
        )
        sample_trace = next(
            trace for trace in figure["data"] if str(trace.get("name", "")).startswith("Samples:")
        )
        self.assertEqual(len(sample_trace["x"]), 100)
        histogram = self.adapter.measurement_histogram_figure(result)
        self.assertEqual(histogram["data"][0]["type"], "histogram")

    def test_measurement_user_defined_points_use_local_frame_and_roundtrip(self):
        box = self.adapter.add_template("guide_box")
        defined_points_mm = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [-10.0, 5.0, 2.5],
        ]
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": box,
                        "role": "measurement",
                        "colour": "#64748b",
                        "opacity": 0.28,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": box,
                        "position": [0.1, -0.02, 0.03],
                        "orientation": [0.0, 0.0, 90.0],
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box,
                        "sampling_mode": "user_defined",
                        "defined_points_mm": defined_points_mm,
                        "target_uT": 200.0,
                        "tolerance_pct": 10.0,
                    },
                },
            ]
        )

        settings = self.adapter.measurement_properties(box)
        self.assertEqual(settings["sampling_mode"], "user_defined")
        self.assertEqual(settings["defined_points_mm"], defined_points_mm)

        sample = self.adapter.measurement_sample_points(box, quality="fine")
        self.assertEqual(sample["sampling_mode"], "user_defined")
        self.assertEqual(sample["quality"], "user_defined")
        self.assertEqual(sample["axis_count"], 0)
        self.assertEqual(sample["spacing_mm"], [])
        np.testing.assert_allclose(
            sample["local_points_m"], np.asarray(defined_points_mm) / 1000.0
        )
        np.testing.assert_allclose(
            sample["world_points_m"],
            [
                [0.1, -0.02, 0.03],
                [0.1, -0.01, 0.03],
                [0.095, -0.03, 0.0325],
            ],
            atol=1e-12,
        )

        result = self.adapter.analyze_measurement(box)
        self.assertEqual(result["stats"]["sample_count"], 3)
        self.assertEqual(result["stats"]["samples_per_axis"], 0)
        self.assertEqual(result["stats"]["sample_spacing_mm"], [])
        target = next(
            item for item in self.adapter.optimizer_targets() if item["id"] == box
        )
        self.assertEqual(target["sampling_mode"], "user_defined")
        self.assertEqual(target["defined_point_count"], 3)

        snapshot_id = self.adapter.create_snapshot(result, "Defined point field")
        snapshot = self.adapter.snapshot_result(snapshot_id)
        self.assertEqual(snapshot["sampling_mode"], "user_defined")
        np.testing.assert_allclose(snapshot["local_points_m"], result["local_points_m"])
        self.assertEqual(self.adapter.snapshot_status(snapshot_id), "current")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "defined-points.magpy.json"
            self.adapter.save_file(path)
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(path)
            reopened_settings = reopened.measurement_properties(box)
            self.assertEqual(reopened_settings["sampling_mode"], "user_defined")
            self.assertEqual(reopened_settings["defined_points_mm"], defined_points_mm)
            np.testing.assert_allclose(
                reopened.measurement_sample_points(box)["world_points_m"],
                sample["world_points_m"],
            )

    def test_real_measurement_dataset_roundtrips_and_group_calibration_is_absolute(self):
        samples = self.adapter.measurement_sensor_samples("axis_sensor")[:3]
        dataset_id = self.adapter.save_measurement_dataset(
            {
                "label": "Bench axial scan",
                "sensor_id": "axis_sensor",
                "sensor_label": "Axis sensor 1",
                "output": "B",
                "coil_ids": ["coil_a", "coil_b"],
                "samples": [
                    {**sample, "measured_uT": value}
                    for sample, value in zip(samples, [10.0, 20.0, 30.0], strict=True)
                ],
                "fit": {
                    "mode": "scale_only",
                    "factor": 1.25,
                    "sample_count": 3,
                    "rms_before_uT": 2.0,
                    "rms_after_uT": 0.1,
                    "max_abs_residual_uT": 0.2,
                    "r_squared": 0.99,
                    "coil_modeling_method": "auto",
                },
            }
        )
        self.assertEqual(dataset_id, "measurement_dataset")
        self.adapter.apply_measurement_calibration(["coil_a", "coil_b"], 1.25)
        self.adapter.mark_measurement_dataset_applied(
            dataset_id,
            coil_ids=["coil_a", "coil_b"],
            factor=1.25,
        )
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_a")["field_scale_factor"],
            1.25,
        )
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_b")["field_scale_factor"],
            1.25,
        )

        reopened = StudioAdapter(self.adapter.document())
        stored = reopened.measurement_dataset(dataset_id)
        self.assertEqual(stored["label"], "Bench axial scan")
        self.assertEqual(stored["coil_ids"], ["coil_a", "coil_b"])
        self.assertEqual(len(stored["samples"]), 3)
        self.assertEqual(stored["samples"][0]["measured_uT"], 10.0)
        self.assertEqual(np.asarray(stored["samples"][0]["rotation_matrix"]).shape, (3, 3))
        self.assertAlmostEqual(stored["applied"]["factor"], 1.25)

    def test_real_measurement_fit_recovers_scale_only_factor_from_saved_geometry(self):
        samples = self.adapter.measurement_sensor_samples("axis_sensor")[:3]
        synthetic_field = {
            "values": np.asarray(
                [
                    [1.0e-6, 0.0, 0.0],
                    [2.0e-6, 0.0, 0.0],
                    [4.0e-6, 0.0, 0.0],
                ],
                dtype=float,
            )
        }
        # Existing empirical factors must not be compounded by a re-fit. The
        # calibration clone resets selected coils to 1.000× before evaluating.
        self.adapter.apply_measurement_calibration(["coil_a", "coil_b"], 3.0)
        with mock.patch.object(
            StudioAdapter,
            "get_field",
            side_effect=[
                synthetic_field,
                {"values": np.zeros((3, 3), dtype=float)},
            ],
        ):
            result = self.adapter.fit_measurement_calibration(
                sensor_id="axis_sensor",
                coil_ids=["coil_a", "coil_b"],
                measured_uT=[1.25, 2.5, 5.0],
                output="B",
                samples=samples,
            )
        self.assertAlmostEqual(result["factor"], 1.25, places=12)
        self.assertAlmostEqual(result["rms_after_uT"], 0.0, places=12)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["predicted_uT"], [1.0, 2.0, 4.0])
        # Fitting itself is non-destructive; the live scene retains its old value
        # until Apply calibration is explicitly requested.
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_a")["field_scale_factor"],
            3.0,
        )
        self.adapter.apply_measurement_calibration(["coil_a", "coil_b"], result["factor"])
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_a")["field_scale_factor"],
            1.25,
        )
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties("coil_b")["field_scale_factor"],
            1.25,
        )

    def test_measurement_user_defined_points_require_xyz_rows(self):
        box = self.adapter.add_template("guide_box")
        with self.assertRaisesRegex(StudioOperationError, "at least one"):
            self.adapter.apply_operations(
                [
                    {
                        "method": "set_measurement_settings",
                        "params": {
                            "object_id": box,
                            "sampling_mode": "user_defined",
                            "defined_points_mm": [],
                        },
                    }
                ]
            )
        with self.assertRaisesRegex(StudioOperationError, r"list of \[X, Y, Z\]"):
            self.adapter.apply_operations(
                [
                    {
                        "method": "set_measurement_settings",
                        "params": {
                            "object_id": box,
                            "sampling_mode": "user_defined",
                            "defined_points_mm": [[0.0, 1.0]],
                        },
                    }
                ]
            )

    def test_measurement_sphere_uses_local_frame_and_roundtrips_settings(self):
        sphere = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {"object_id": sphere, "name": "diameter", "value": 0.08},
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": sphere,
                        "position": [0.01, -0.02, 0.03],
                        "orientation": [0.0, 0.0, 45.0],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": sphere,
                        "role": "measurement",
                        "colour": "#64748b",
                        "opacity": 0.3,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": sphere,
                        "quality": "standard",
                        "target_uT": 200.0,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        samples = self.adapter.measurement_sample_points(sphere)
        local = np.asarray(samples["local_points_m"])
        world = np.asarray(samples["world_points_m"])
        self.assertTrue(np.all(np.linalg.norm(local, axis=1) <= 0.04 + 1e-15))
        self.assertLess(len(local), 15**3)
        self.assertGreater(len(local), 0)
        self.assertFalse(np.allclose(local, world))
        self.assertAlmostEqual(samples["volume_cm3"], 4 / 3 * math.pi * 40**3 / 1000)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurement.magpy.json"
            self.adapter.save_file(path)
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(path)
            self.assertEqual(reopened.get_object(sphere)["role"], "measurement")
            self.assertEqual(reopened.measurement_properties(sphere)["quality"], "standard")
            self.assertAlmostEqual(reopened.measurement_properties(sphere)["target_uT"], 200.0)

    def test_field_snapshot_roundtrip_freshness_and_undo(self):
        box = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": box,
                        "role": "measurement",
                        "colour": "#7c3aed",
                        "opacity": 0.28,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box,
                        "quality": "preview",
                        "target_uT": 0.6,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        analyzed = self.adapter.analyze_measurement(box)
        snapshot_id = self.adapter.create_snapshot(analyzed, "Central box")
        self.assertEqual(self.adapter.snapshot_status(snapshot_id), "current")
        restored = self.adapter.snapshot_result(snapshot_id)
        np.testing.assert_array_equal(restored["local_points_m"], analyzed["local_points_m"])
        np.testing.assert_array_equal(restored["points_m"], analyzed["points_m"])
        np.testing.assert_array_equal(restored["values_t"], analyzed["values_t"])
        self.assertEqual(restored["stats"], analyzed["stats"])

        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": "coil_a",
                        "name": "current",
                        "value": 0.08,
                    },
                }
            ]
        )
        self.assertEqual(self.adapter.snapshot_status(snapshot_id), "stale")
        self.adapter.undo()
        self.assertEqual(self.adapter.snapshot_status(snapshot_id), "current")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.magpy.json"
            self.adapter.save_file(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["field_workbench"]["version"], WORKBENCH_METADATA_VERSION)
            self.assertEqual(len(saved["field_workbench"]["snapshots"]), 1)
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(path)
            self.assertEqual(reopened.snapshot_status(snapshot_id), "current")
            np.testing.assert_array_equal(
                reopened.snapshot_result(snapshot_id)["values_t"], analyzed["values_t"]
            )

        self.adapter.rename_snapshot(snapshot_id, "Renamed box")
        self.assertEqual(self.adapter.list_snapshots()[0]["name"], "Renamed box")
        second_snapshot_id = self.adapter.create_snapshot(analyzed, "Second box")
        self.adapter.delete_snapshots([snapshot_id, second_snapshot_id])
        self.assertEqual(self.adapter.list_snapshots(), [])
        self.adapter.undo()
        self.assertEqual(
            [item["name"] for item in self.adapter.list_snapshots()],
            ["Renamed box", "Second box"],
        )

    def test_snapshot_import_from_another_scene_is_copy_only_and_undoable(self):
        def add_measurement(adapter: StudioAdapter) -> tuple[str, dict]:
            box_id = adapter.add_template("guide_box")
            adapter.apply_operations(
                [
                    {
                        "method": "set_geometry_style",
                        "params": {
                            "object_id": box_id,
                            "role": "measurement",
                            "colour": "#7c3aed",
                            "opacity": 0.28,
                        },
                    },
                    {
                        "method": "set_measurement_settings",
                        "params": {
                            "object_id": box_id,
                            "quality": "preview",
                            "target_uT": 0.6,
                            "tolerance_pct": 5.0,
                        },
                    },
                ]
            )
            return box_id, adapter.analyze_measurement(box_id)

        source = StudioAdapter(compact_workflow_scene())
        _source_box, source_result = add_measurement(source)
        source_snapshot = source.create_snapshot(source_result, "Other model")

        _local_box, local_result = add_measurement(self.adapter)
        local_snapshot = self.adapter.create_snapshot(local_result, "This model")
        object_ids_before = [item["id"] for item in self.adapter.list_objects()]
        history_before = self.adapter.history()["undo"]

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "other-model.magpy.json"
            source.save_file(source_path)
            catalog = self.adapter.snapshot_catalog_from_file(source_path)
            self.assertEqual([item["id"] for item in catalog], [source_snapshot])
            self.assertTrue(catalog[0]["importable"])

            imported_ids = self.adapter.import_snapshots_from_file(
                source_path, [source_snapshot]
            )
            self.assertEqual(len(imported_ids), 1)
            imported_id = imported_ids[0]
            self.assertEqual(self.adapter.snapshot_status(imported_id), "imported")
            self.assertEqual(
                self.adapter.list_snapshots()[-1]["imported_from"]["file_name"],
                "other-model.magpy.json",
            )
            self.assertEqual(
                [item["id"] for item in self.adapter.list_objects()], object_ids_before
            )
            np.testing.assert_array_equal(
                self.adapter.snapshot_result(imported_id)["values_t"],
                source_result["values_t"],
            )
            comparison = self.adapter.compare_snapshots(
                local_snapshot, imported_id, mode="actual"
            )
            self.assertLess(comparison["stats"]["rms_vector_difference"], 1e-12)

            roundtrip_path = Path(directory) / "combined.magpy.json"
            self.adapter.save_file(roundtrip_path)
            reopened = StudioAdapter({"objects": []})
            reopened.load_file(roundtrip_path)
            self.assertEqual(reopened.snapshot_status(imported_id), "imported")

        self.assertEqual(self.adapter.history()["undo"], history_before + 1)
        self.adapter.undo()
        self.assertEqual(
            [item["id"] for item in self.adapter.list_snapshots()], [local_snapshot]
        )

    def test_snapshot_comparison_actual_and_shape_only(self):
        box = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": box,
                        "role": "measurement",
                        "colour": "#7c3aed",
                        "opacity": 0.28,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box,
                        "quality": "preview",
                        "target_uT": 0.6,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        result_a = self.adapter.analyze_measurement(box)
        snapshot_a = self.adapter.create_snapshot(result_a, "Field A")
        result_b = copy.deepcopy(result_a)
        result_b["values_t"] = np.asarray(result_a["values_t"]) * 2.0
        result_b["magnitudes_uT"] = np.asarray(result_a["magnitudes_uT"]) * 2.0
        snapshot_b = self.adapter.create_snapshot(result_b, "Field B")

        actual = self.adapter.compare_snapshots(snapshot_a, snapshot_b, mode="actual")
        self.assertEqual(actual["stats"]["sample_count"], len(result_a["values_t"]))
        self.assertAlmostEqual(actual["stats"]["raw_mean_ratio_b_over_a"], 2.0, places=12)
        self.assertGreater(actual["stats"]["rms_vector_difference"], 0.0)
        self.assertAlmostEqual(actual["stats"]["median_angular_difference_deg"], 0.0, places=9)

        normalized = self.adapter.compare_snapshots(
            snapshot_a, snapshot_b, mode="normalized"
        )
        self.assertLess(normalized["stats"]["rms_vector_difference"], 1e-10)
        self.assertLess(normalized["stats"]["rms_magnitude_difference"], 1e-10)
        self.assertEqual(len(self.adapter.comparison_slice_figure(actual)["data"]), 3)
        self.assertEqual(
            self.adapter.comparison_difference_figure(actual)["data"][0]["type"],
            "scatter3d",
        )
        self.assertEqual(
            self.adapter.comparison_direction_figure(actual)["data"][0]["type"],
            "scatter3d",
        )
        self.assertEqual(len(self.adapter.comparison_distribution_figure(actual)["data"]), 2)

    def test_multi_snapshot_benchmark_and_reference_matching(self):
        box = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box, "role": "measurement"},
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box,
                        "quality": "preview",
                        "target_uT": 0.6,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        result_a = self.adapter.analyze_measurement(box)
        snapshot_a = self.adapter.create_snapshot(result_a, "Reference")
        result_b = copy.deepcopy(result_a)
        result_b["values_t"] = np.asarray(result_a["values_t"]) * 1.05
        result_b["magnitudes_uT"] = np.asarray(result_a["magnitudes_uT"]) * 1.05
        snapshot_b = self.adapter.create_snapshot(result_b, "Candidate B")
        result_c = copy.deepcopy(result_a)
        result_c["values_t"] = np.asarray(result_a["values_t"]) * 0.90
        result_c["magnitudes_uT"] = np.asarray(result_a["magnitudes_uT"]) * 0.90
        snapshot_c = self.adapter.create_snapshot(result_c, "Candidate C")

        target = float(np.mean(result_a["magnitudes_uT"]))
        benchmark = self.adapter.benchmark_snapshots(
            [snapshot_a, snapshot_b, snapshot_c],
            target_uT=target,
            tolerance_pct=6.0,
        )
        self.assertEqual(len(benchmark["entries"]), 3)
        low = target * 0.94
        high = target * 1.06
        expected_reference_coverage = 100.0 * float(
            np.mean(
                (np.asarray(result_a["magnitudes_uT"]) >= low)
                & (np.asarray(result_a["magnitudes_uT"]) <= high)
            )
        )
        self.assertAlmostEqual(
            benchmark["entries"][0]["target_coverage_pct"],
            expected_reference_coverage,
        )

        matched = self.adapter.match_snapshot_exposures(
            snapshot_a,
            [snapshot_b, snapshot_c],
            magnitude_tolerance_pct=6.0,
            direction_tolerance_deg=1.0,
        )
        self.assertEqual(len(matched["candidates"]), 2)
        self.assertAlmostEqual(
            matched["reference_mean_uT"],
            float(np.mean(result_a["magnitudes_uT"])),
            places=12,
        )
        self.assertAlmostEqual(
            matched["candidates"][0]["stats"]["mean_intensity_bias_pct"],
            5.0,
            places=9,
        )
        self.assertAlmostEqual(
            matched["candidates"][0]["stats"]["match_coverage_pct"],
            100.0,
            places=9,
        )
        self.assertAlmostEqual(
            matched["candidates"][1]["stats"]["mean_intensity_bias_pct"],
            -10.0,
            places=9,
        )

    def test_scene_normalization_metadata_keeps_display_labels_and_group_path(self):
        scene = {
            "objects": [
                {
                    "id": "coil_group",
                    "type": "Collection",
                    "style": {"label": "Primary Pair"},
                    "children": [
                        {
                            "id": "coil_1",
                            "type": "current.Polyline",
                            "style": {"label": "Left Racetrack"},
                        },
                        {
                            "id": "coil_2",
                            "type": "current.Polyline",
                            "style": {"label": "Right Racetrack"},
                        },
                    ],
                }
            ]
        }
        metadata = self.adapter._scene_normalization_metadata(scene)
        self.assertEqual(metadata["coil_1"]["label"], "Left Racetrack")
        self.assertEqual(
            metadata["coil_1"]["group_path"],
            [{"id": "coil_group", "label": "Primary Pair"}],
        )
        self.assertEqual(metadata["coil_2"]["label"], "Right Racetrack")

    def test_current_normalization_reports_common_and_independent_currents(self):
        local_points = np.asarray(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]]
        )
        result = {
            "local_points_m": local_points,
            "world_points_m": local_points.copy(),
            "values_t": np.tile([0.0, 0.0, 3e-6], (3, 1)),
        }
        basis = np.asarray(
            [
                np.tile([0.0, 0.0, 1.0], (3, 1)),
                np.tile([0.0, 0.0, 2.0], (3, 1)),
            ]
        )
        decomposition = {
            "result": result,
            "basis_per_amp_uT": basis,
            "coil_ids": ["coil_1", "coil_2"],
            "coil_labels": ["Coil 1", "Coil 2"],
            "original_currents_a": np.asarray([1.0, 1.0]),
            "polarity_signs": np.asarray([1.0, 1.0]),
        }
        reference = np.tile([0.0, 0.0, 6.0], (3, 1))
        with mock.patch.object(
            self.adapter, "_snapshot_current_basis", return_value=decomposition
        ):
            common = self.adapter.normalize_snapshot_currents(
                "candidate",
                reference,
                selected_coil_ids=["coil_1", "coil_2"],
                current_mode="common",
            )
            independent = self.adapter.normalize_snapshot_currents(
                "candidate",
                reference,
                selected_coil_ids=["coil_1", "coil_2"],
                current_mode="independent",
            )
        np.testing.assert_allclose(common["fitted_currents_a"], [2.0, 2.0])
        np.testing.assert_allclose(common["vectors_uT"], reference)
        np.testing.assert_allclose(independent["vectors_uT"], reference, atol=1e-10)
        self.assertEqual(len(independent["fitted_currents_a"]), 2)

    def test_current_normalization_supports_shared_groups_independent_and_fixed_coils(self):
        local_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [0.002, 0.0, 0.0],
                [0.003, 0.0, 0.0],
            ]
        )
        stored_uT = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 4.0],
            ]
        )
        result = {
            "local_points_m": local_points,
            "world_points_m": local_points.copy(),
            "values_t": stored_uT * 1e-6,
        }
        basis = np.zeros((3, 4, 3), dtype=float)
        basis[0, 0, 2] = 1.0
        basis[1, 1, 2] = 1.0
        basis[2, 2, 2] = 2.0
        decomposition = {
            "result": result,
            "basis_per_amp_uT": basis,
            "coil_ids": ["coil_1", "coil_2", "coil_3"],
            "coil_labels": ["Coil 1", "Coil 2", "Coil 3"],
            "original_currents_a": np.asarray([1.0, 1.0, 1.0]),
            "polarity_signs": np.asarray([1.0, 1.0, 1.0]),
        }
        reference = np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 6.0],
                [0.0, 0.0, 4.0],
            ]
        )
        with mock.patch.object(
            self.adapter, "_snapshot_current_basis", return_value=decomposition
        ):
            grouped = self.adapter.normalize_snapshot_currents(
                "candidate",
                reference,
                coil_assignments={
                    "coil_1": "Primary pair",
                    "coil_2": "Primary pair",
                    "coil_3": "independent",
                    "coil_4": "fixed",
                },
            )
        np.testing.assert_allclose(grouped["vectors_uT"], reference, atol=1e-10)
        np.testing.assert_allclose(grouped["fitted_currents_a"], [2.0, 2.0, 3.0])
        self.assertEqual(grouped["current_mode"], "grouped")
        self.assertEqual(len(grouped["fit_groups"]), 2)
        self.assertEqual(grouped["fit_groups"][0]["coil_ids"], ["coil_1", "coil_2"])
        self.assertEqual(grouped["fit_groups"][1]["relationship"], "independent")

    def test_current_normalization_matches_magnitude_without_fitting_direction(self):
        local_points = np.asarray(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]]
        )
        result = {
            "local_points_m": local_points,
            "world_points_m": local_points.copy(),
            "values_t": np.tile([0.0, 0.0, 4e-6], (3, 1)),
        }
        decomposition = {
            "result": result,
            "basis_per_amp_uT": np.asarray(
                [np.tile([0.0, 0.0, 1.0], (3, 1))]
            ),
            "coil_ids": ["coil_1"],
            "coil_labels": ["Coil 1"],
            "original_currents_a": np.asarray([4.0]),
            "polarity_signs": np.asarray([1.0]),
        }
        # The reference points in the opposite direction.  Current normalization
        # should still match |B| and leave direction disagreement for the normal
        # comparison statistics to report.
        reference = np.tile([0.0, 0.0, -6.0], (3, 1))
        with mock.patch.object(
            self.adapter, "_snapshot_current_basis", return_value=decomposition
        ):
            normalized = self.adapter.normalize_snapshot_currents(
                "candidate",
                reference,
                selected_coil_ids=["coil_1"],
                current_mode="common",
            )
        np.testing.assert_allclose(normalized["fitted_currents_a"], [6.0], atol=1e-8)
        np.testing.assert_allclose(
            np.linalg.norm(normalized["vectors_uT"], axis=1),
            np.linalg.norm(reference, axis=1),
            atol=1e-8,
        )
        self.assertTrue(np.all(normalized["vectors_uT"][:, 2] > 0.0))
        self.assertEqual(normalized["fit_metric"], "pointwise_field_magnitude")

    def test_current_normalization_can_raise_a_captured_zero_current(self):
        local_points = np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        result = {
            "local_points_m": local_points,
            "world_points_m": local_points.copy(),
            "values_t": np.zeros((2, 3), dtype=float),
        }
        decomposition = {
            "result": result,
            "basis_per_amp_uT": np.asarray(
                [np.tile([0.0, 0.0, 2.0], (2, 1))]
            ),
            "coil_ids": ["coil_1"],
            "coil_labels": ["Coil 1"],
            "original_currents_a": np.asarray([0.0]),
            "polarity_signs": np.asarray([1.0]),
        }
        reference = np.tile([0.0, 0.0, 10.0], (2, 1))
        with mock.patch.object(
            self.adapter, "_snapshot_current_basis", return_value=decomposition
        ):
            normalized = self.adapter.normalize_snapshot_currents(
                "candidate",
                reference,
                selected_coil_ids=["coil_1"],
                current_mode="common",
            )
        np.testing.assert_allclose(normalized["fitted_currents_a"], [5.0], atol=1e-8)
        np.testing.assert_allclose(normalized["vectors_uT"], reference, atol=1e-8)

    def test_group_existing_mixed_objects_is_one_undoable_edit(self):
        guide = self.adapter.add_template("guide_box")
        history_before = self.adapter.history()["undo"]
        group = self.adapter.group_objects(["coil_a", "coil_b", guide])
        self.assertEqual(self.adapter.get_object(group)["type"], "Collection")
        self.assertEqual(self.adapter.get_object("coil_a")["parent"], group)
        self.assertEqual(self.adapter.get_object("coil_b")["parent"], group)
        self.assertEqual(self.adapter.get_object(guide)["parent"], group)
        self.assertEqual(self.adapter.history()["undo"], history_before + 1)
        self.adapter.undo()
        self.assertNotIn(group, {item["id"] for item in self.adapter.list_objects()})
        self.assertIsNone(self.adapter.get_object("coil_a").get("parent"))
        self.assertIsNone(self.adapter.get_object(guide).get("parent"))

    def test_ungroup_moves_immediate_children_and_is_one_undoable_edit(self):
        guide = self.adapter.add_template("guide_box")
        group = self.adapter.group_objects(["coil_a", "coil_b", guide])
        history_before = self.adapter.history()["undo"]
        children = self.adapter.ungroup(group)
        self.assertEqual(set(children), {"coil_a", "coil_b", guide})
        self.assertNotIn(group, {item["id"] for item in self.adapter.list_objects()})
        self.assertIsNone(self.adapter.get_object("coil_a").get("parent"))
        self.assertIsNone(self.adapter.get_object(guide).get("parent"))
        self.assertEqual(self.adapter.history()["undo"], history_before + 1)
        self.adapter.undo()
        self.assertEqual(self.adapter.get_object("coil_a")["parent"], group)
        self.assertEqual(self.adapter.get_object(guide)["parent"], group)

    def test_obj_mesh_import_uses_explicit_unit_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triangle.obj"
            path.write_text("v 0 0 0\nv 10 0 0\nv 0 20 0\nf 1 2 3\n", encoding="utf-8")
            mesh_id = self.adapter.import_mesh(path, unit_scale=0.001)
            mesh = self.adapter._mesh(mesh_id)
            self.assertEqual(mesh["mesh"]["vertices"][1], [0.01, 0.0, 0.0])
            self.assertEqual(mesh["mesh"]["vertices"][2], [0.0, 0.02, 0.0])
            self.assertEqual(mesh["mesh"]["faces"], [[0, 1, 2]])
            self.assertEqual(mesh["scale_xyz"], [1.0, 1.0, 1.0])

    def test_imported_mesh_reports_dimensions_and_scales_each_axis_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sized_tetrahedron.stl"
            path.write_text(
                """solid sized
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 100 0 0
    vertex 0 80 0
  endloop
endfacet
facet normal 0 -1 0
  outer loop
    vertex 0 0 0
    vertex 0 0 60
    vertex 100 0 0
  endloop
endfacet
facet normal 1 1 1
  outer loop
    vertex 100 0 0
    vertex 0 0 60
    vertex 0 80 0
  endloop
endfacet
facet normal -1 0 0
  outer loop
    vertex 0 0 0
    vertex 0 80 0
    vertex 0 0 60
  endloop
endfacet
endsolid sized
""",
                encoding="utf-8",
            )
            mesh_id = self.adapter.import_mesh(path, unit_scale=0.001)
            properties = self.adapter.geometry_properties(mesh_id)
            np.testing.assert_allclose(properties["source_dimensions_mm"], [100, 80, 60])
            np.testing.assert_allclose(properties["dimensions_mm"], [100, 80, 60])
            self.assertEqual(
                {item["name"]: item["value"] for item in self.adapter.get_params(mesh_id)},
                {"scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0},
            )

            self.adapter.apply_operations(
                [
                    {
                        "method": "set_param",
                        "params": {
                            "object_id": mesh_id,
                            "name": "scale_x",
                            "value": 2.0,
                        },
                    },
                    {
                        "method": "set_param",
                        "params": {
                            "object_id": mesh_id,
                            "name": "scale_y",
                            "value": 0.5,
                        },
                    },
                    {
                        "method": "set_param",
                        "params": {
                            "object_id": mesh_id,
                            "name": "scale_z",
                            "value": 1.5,
                        },
                    },
                ]
            )
            np.testing.assert_allclose(
                self.adapter.geometry_properties(mesh_id)["dimensions_mm"],
                [200, 40, 90],
            )
            trace = next(
                item
                for item in self.adapter.base_figure()["data"]
                if item.get("meta", {}).get("field_workbench_object_id") == mesh_id
                and not item.get("meta", {}).get("field_workbench_pick_proxy")
            )
            np.testing.assert_allclose(
                [max(trace[axis]) - min(trace[axis]) for axis in ("x", "y", "z")],
                [200, 40, 90],
            )
            self.adapter.undo()
            np.testing.assert_allclose(
                self.adapter.geometry_properties(mesh_id)["dimensions_mm"],
                [100, 80, 60],
            )
            self.adapter.redo()
            np.testing.assert_allclose(
                self.adapter.geometry_properties(mesh_id)["dimensions_mm"],
                [200, 40, 90],
            )

    def test_legacy_uniform_mesh_scale_migrates_to_three_axes(self):
        mesh_id = self.adapter.add_template("guide_box")
        mesh = self.adapter._mesh(mesh_id)
        mesh["mesh"] = {
            "kind": "triangles",
            "vertices": [[0, 0, 0], [0.1, 0, 0], [0, 0.08, 0.06]],
            "faces": [[0, 1, 2]],
        }
        mesh["scale"] = 2.0
        properties = self.adapter.geometry_properties(mesh_id)
        self.assertEqual(properties["scale_xyz"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(properties["dimensions_mm"], [200, 160, 120])

    def test_imported_stl_is_rendered_and_show_hide_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            # Deliberately exercise punctuation, spaces, brackets, a hash, and
            # Unicode.  Trace identities derived from imported objects must be
            # safe when Plotly inserts them into CSS selectors during a later
            # selection-only Plotly.react update.
            path = Path(directory) / "4D Box: Shell 5in (visual) #[α].stl"
            path.write_text(
                """solid four_d_box
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 100 0 0
    vertex 0 80 0
  endloop
endfacet
facet normal 0 -1 0
  outer loop
    vertex 0 0 0
    vertex 0 0 60
    vertex 100 0 0
  endloop
endfacet
facet normal 1 1 1
  outer loop
    vertex 100 0 0
    vertex 0 0 60
    vertex 0 80 0
  endloop
endfacet
facet normal -1 0 0
  outer loop
    vertex 0 0 0
    vertex 0 80 0
    vertex 0 0 60
  endloop
endfacet
endsolid four_d_box
""",
                encoding="utf-8",
            )
            mesh_id = self.adapter.import_mesh(path, unit_scale=0.001)
            self.assertEqual(len(self.adapter.list_objects()), 4)

            def rendered_mesh():
                matches = [
                    trace
                    for trace in self.adapter.base_figure()["data"]
                    if trace.get("meta", {}).get("field_workbench_object_id") == mesh_id
                    and not trace.get("meta", {}).get("field_workbench_pick_proxy")
                ]
                self.assertEqual(len(matches), 1)
                return matches[0]

            trace = rendered_mesh()
            self.assertEqual(trace["type"], "mesh3d")
            self.assertNotEqual(trace.get("visible"), False)
            self.assertGreater(trace["opacity"], 0.0)
            self.assertEqual(len(trace["i"]), 4)
            np.testing.assert_allclose(
                [
                    max(trace[axis]) - min(trace[axis])
                    for axis in ("x", "y", "z")
                ],
                [100.0, 80.0, 60.0],
            )
            # Check the same boundary used by PlotView.  A valid in-memory
            # trace is not sufficient if Plotly's serializer drops its mesh
            # topology or visibility before the browser receives it.
            browser_payload = json.loads(
                pio.to_json(
                    self.adapter.figure(selected_id=mesh_id),
                    validate=False,
                    pretty=False,
                    remove_uids=False,
                )
            )
            browser_meshes = [
                item
                for item in browser_payload["data"]
                if item.get("meta", {}).get("field_workbench_object_id") == mesh_id
                and item.get("type") == "mesh3d"
            ]
            self.assertEqual(len(browser_meshes), 1)
            self.assertGreater(len(browser_payload["data"]), len(browser_meshes))
            self.assertEqual(len(browser_meshes[0]["i"]), 4)
            self.assertGreater(browser_meshes[0]["opacity"], 0.0)
            self.assertNotEqual(browser_meshes[0].get("visible"), False)
            trace_uids = [item.get("uid") for item in browser_payload["data"]]
            self.assertTrue(all(trace_uids))
            self.assertEqual(len(trace_uids), len(set(trace_uids)))
            for trace_uid in trace_uids:
                self.assertRegex(trace_uid, r"\A[A-Za-z0-9_-]+\Z")

            self.adapter.set_visible(mesh_id, False)
            self.assertFalse(
                any(
                    item.get("meta", {}).get("field_workbench_object_id") == mesh_id
                    for item in self.adapter.base_figure()["data"]
                )
            )
            self.adapter.set_visible(mesh_id, True)
            restored = rendered_mesh()
            self.assertEqual(restored["i"], trace["i"])
            self.assertEqual(restored["j"], trace["j"])
            self.assertEqual(restored["k"], trace["k"])

    def test_imported_mesh_dimensions_do_not_change_with_scene_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rectangular_box.obj"
            vertices = [
                (-60, -40, -30), (60, -40, -30), (-60, 40, -30), (60, 40, -30),
                (-60, -40, 30), (60, -40, 30), (-60, 40, 30), (60, 40, 30),
            ]
            faces = [
                (1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6),
                (1, 5, 6), (1, 6, 2), (3, 4, 8), (3, 8, 7),
                (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4),
            ]
            path.write_text(
                "\n".join(
                    [
                        *(f"v {x} {y} {z}" for x, y, z in vertices),
                        *(f"f {i} {j} {k}" for i, j, k in faces),
                    ]
                ),
                encoding="utf-8",
            )
            self.adapter.new_empty()
            mesh_id = self.adapter.import_mesh(path, unit_scale=0.001)

            def displayed_spans_mm():
                trace = next(
                    item
                    for item in self.adapter.base_figure()["data"]
                    if item.get("meta", {}).get("field_workbench_object_id") == mesh_id
                )
                return [
                    max(trace[axis]) - min(trace[axis])
                    for axis in ("x", "y", "z")
                ]

            np.testing.assert_allclose(displayed_spans_mm(), [120.0, 80.0, 60.0])
            coil_id = self.adapter.add_template("circle")
            np.testing.assert_allclose(displayed_spans_mm(), [120.0, 80.0, 60.0])
            self.adapter.remove(coil_id)
            np.testing.assert_allclose(displayed_spans_mm(), [120.0, 80.0, 60.0])

    def test_passive_mesh_trace_identity_and_colour_survive_source_changes(self):
        self.adapter.new_empty()
        mesh_id = self.adapter.add_template("guide_box")

        def mesh_trace(figure):
            return next(
                trace
                for trace in figure["data"]
                if trace.get("meta", {}).get("field_workbench_object_id") == mesh_id
                and not trace.get("meta", {}).get("field_workbench_pick_proxy")
            )

        initial_figure = self.adapter.base_figure()
        initial_mesh = mesh_trace(initial_figure)
        initial_uid = initial_mesh["uid"]
        self.assertEqual(initial_mesh["color"], "#64748b")
        self.assertNotIn("facecolor", initial_mesh)

        # Magpylib source meshes may legitimately contain per-face pole or
        # direction colours. Their identities must never collide with a
        # passive workbench mesh when source traces are inserted or removed.
        coil_id = self.adapter.add_template("circle")
        with_source = self.adapter.base_figure()
        updated_mesh = mesh_trace(with_source)
        self.assertEqual(updated_mesh["uid"], initial_uid)
        self.assertEqual(updated_mesh["color"], "#64748b")
        self.assertNotIn("facecolor", updated_mesh)
        source_uids = {
            trace["uid"]
            for trace in with_source["data"]
            if trace["uid"].startswith("field-workbench-magpylib-")
        }
        self.assertNotIn(initial_uid, source_uids)

        self.adapter.remove(coil_id)
        restored_mesh = mesh_trace(self.adapter.base_figure())
        self.assertEqual(restored_mesh["uid"], initial_uid)
        self.assertEqual(restored_mesh["color"], "#64748b")

        complete = self.adapter.figure(selected_id=mesh_id)
        uids = [trace.get("uid") for trace in complete["data"]]
        self.assertTrue(all(uids))
        self.assertEqual(len(uids), len(set(uids)))
        for uid in uids:
            self.assertRegex(uid, r"\A[A-Za-z0-9_-]+\Z")

    def test_3d_slice_selector_overlay_marks_centre_and_projected_planes(self):
        figure = self.adapter.compose_figure(
            self.adapter.base_figure(),
            analysis_overlay={
                "kind": "slice_stack",
                "centre_mm": [10.0, -20.0, 30.0],
                "span_mm": 100.0,
                "slice_axis": "z",
                "slice_count": 5,
            },
        )
        centre = next(
            trace
            for trace in figure["data"]
            if str(trace.get("name", "")).startswith("3D map centre slice")
        )
        self.assertEqual(centre["type"], "mesh3d")
        self.assertTrue(all(abs(float(value) - 30.0) < 1e-9 for value in centre["z"]))
        self.assertAlmostEqual(min(float(value) for value in centre["x"]), -40.0)
        self.assertAlmostEqual(max(float(value) for value in centre["x"]), 60.0)
        self.assertAlmostEqual(min(float(value) for value in centre["y"]), -70.0)
        self.assertAlmostEqual(max(float(value) for value in centre["y"]), 30.0)

        projected = next(
            trace for trace in figure["data"] if trace.get("name") == "Projected slice planes"
        )
        self.assertEqual(projected["type"], "scatter3d")
        self.assertEqual(projected["mode"], "lines")
        self.assertEqual(projected["x"].count(None), 4)
        projected_z = sorted(
            {round(float(value), 9) for value in projected["z"] if value is not None}
        )
        self.assertEqual(projected_z, [-20.0, 5.0, 55.0, 80.0])

    def test_3d_figure_uses_compact_bottom_legend_and_full_scene_domain(self):
        figure = self.adapter.figure(selected_id="coil_a")
        layout = figure["layout"]

        self.assertEqual(layout["scene"]["domain"], {"x": [0.0, 1.0], "y": [0.0, 1.0]})
        self.assertEqual(layout["legend"]["orientation"], "h")
        self.assertEqual(layout["legend"]["xanchor"], "center")
        self.assertEqual(layout["legend"]["yanchor"], "top")
        self.assertEqual(layout["legend"]["x"], 0.5)
        self.assertLess(layout["legend"]["y"], 0.0)
        self.assertLessEqual(layout["margin"]["l"], 10)
        self.assertLessEqual(layout["margin"]["r"], 10)
        self.assertLessEqual(layout["margin"]["t"], 10)
        self.assertGreaterEqual(layout["margin"]["b"], 48)

    def test_figure_in_mm_respects_magpylib_display_unit(self):
        for unit, source, expected in (
            ("m", [-0.05, 0.05], [-50.0, 50.0]),
            ("cm", [-5.0, 5.0], [-50.0, 50.0]),
            ("mm", [-50.0, 50.0], [-50.0, 50.0]),
        ):
            with self.subTest(unit=unit):
                converted = figure_in_mm(
                    {
                        "data": [
                            {
                                "type": "scatter3d",
                                "x": source,
                                "y": [0.0, 0.0],
                                "z": [0.0, 0.0],
                            }
                        ],
                        "layout": {
                            "scene": {
                                "xaxis": {
                                    "title": {"text": f"x ({unit})"},
                                    "range": source,
                                },
                                "yaxis": {"title": {"text": f"y ({unit})"}},
                                "zaxis": {"title": {"text": f"z ({unit})"}},
                            }
                        },
                    }
                )
                self.assertEqual(converted["data"][0]["x"], expected)
                self.assertEqual(converted["layout"]["scene"]["xaxis"]["range"], expected)
                self.assertEqual(
                    converted["layout"]["scene"]["xaxis"]["title"]["text"],
                    "X (mm)",
                )

    def test_new_100_mm_coil_engine_geometry_and_display_scale(self):
        self.adapter.new_empty()
        coil_id = self.adapter.add_template("circle")
        figure = self.adapter.figure(selected_id=coil_id)

        params = {
            item["name"]: item["value"] for item in self.adapter.get_params(coil_id)
        }
        self.assertAlmostEqual(params["diameter"], 0.1, places=12)

        selected = next(
            trace
            for trace in figure["data"]
            if str(trace.get("name", "")).startswith("Selected:")
        )
        source_traces = [
            trace
            for trace in figure["data"]
            if not str(trace.get("name", "")).startswith("Selected:")
            and not trace.get("meta", {}).get("field_workbench_pick_proxy")
        ]

        def diameter_xy(traces):
            points = []
            for trace in traces:
                for axis in ("x", "y"):
                    values = np.asarray(trace.get(axis, []), dtype=float).reshape(-1)
                    points.extend(values[np.isfinite(values)].tolist())
            return max(points) - min(points)

        self.assertAlmostEqual(diameter_xy([selected]), 100.0, places=6)
        source_trace_diameters = [diameter_xy([trace]) for trace in source_traces]
        self.assertAlmostEqual(max(source_trace_diameters), 100.0, delta=0.1)
        # Magpylib's decorative current-direction arrow extends a few millimetres
        # beyond the mathematical line-current loop. Keep this bound broad enough
        # for that decoration while still catching the former 1000x unit error.
        source_display_diameter = diameter_xy(source_traces)
        self.assertGreater(source_display_diameter, 90.0)
        self.assertLess(source_display_diameter, 120.0)

    def test_circular_coil_backend_matches_independent_on_axis_formula(self):
        self.adapter.new_empty()
        coil_id = self.adapter.add_template("circle")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": coil_id,
                        "drive_current_a": 0.08,
                        "turns": 25,
                        "field_scale_factor": 1.0,
                    },
                }
            ]
        )

        radius_m = 0.05
        z_m = np.asarray([0.0, 0.025], dtype=float)
        ampere_turns = 0.08 * 25
        mu0 = 4.0 * math.pi * 1e-7
        expected_bz_t = (
            mu0
            * ampere_turns
            * radius_m**2
            / (2.0 * (radius_m**2 + z_m**2) ** 1.5)
        )
        values_t = np.vstack(
            [
                np.asarray(
                    self.adapter.field_at_mm([0.0, 0.0, float(z * 1000.0)])["values"],
                    dtype=float,
                ).reshape(-1, 3)[0]
                for z in z_m
            ]
        )
        uncorrected = self.adapter.coil_physical_properties(coil_id)
        np.testing.assert_allclose(values_t[:, :2], 0.0, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            values_t[:, 2], expected_bz_t, rtol=1e-9, atol=1e-14
        )

        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": coil_id,
                        "field_scale_factor": 2.5,
                    },
                }
            ]
        )
        corrected_values_t = np.vstack(
            [
                np.asarray(
                    self.adapter.field_at_mm([0.0, 0.0, float(z * 1000.0)])["values"],
                    dtype=float,
                ).reshape(-1, 3)[0]
                for z in z_m
            ]
        )
        np.testing.assert_allclose(
            corrected_values_t, values_t * 2.5, rtol=1e-12, atol=1e-14
        )
        physical = self.adapter.coil_physical_properties(coil_id)
        self.assertAlmostEqual(physical["calculations"]["ampere_turns"], 2.0)
        self.assertAlmostEqual(
            physical["calculations"]["effective_ampere_turns"], 5.0
        )
        for key in (
            "wire_length_m",
            "copper_mass_g",
            "resistance_operating_ohm",
            "resistive_voltage_v",
            "copper_loss_w",
            "current_density_a_mm2",
        ):
            self.assertAlmostEqual(
                physical["calculations"][key],
                uncorrected["calculations"][key],
            )

        self.adapter.undo()
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties(coil_id)["field_scale_factor"], 1.0
        )
        undone_values_t = np.vstack(
            [
                np.asarray(
                    self.adapter.field_at_mm([0.0, 0.0, float(z * 1000.0)])["values"],
                    dtype=float,
                ).reshape(-1, 3)[0]
                for z in z_m
            ]
        )
        np.testing.assert_allclose(
            undone_values_t,
            values_t,
            rtol=1e-12,
            atol=1e-14,
        )
        self.adapter.redo()
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties(coil_id)["field_scale_factor"], 2.5
        )

    def test_multi_selection_and_pick_metadata(self):
        figure = self.adapter.figure(selected_ids=["coil_a", "coil_b"])
        selected_names = {
            str(trace.get("name", ""))
            for trace in figure["data"]
            if str(trace.get("name", "")).startswith("Selected:")
        }
        self.assertIn("Selected: Lower coil", selected_names)
        self.assertIn("Selected: Upper coil", selected_names)
        pick_ids = {
            trace.get("meta", {}).get("field_workbench_object_id")
            for trace in figure["data"]
            if trace.get("meta", {}).get("field_workbench_pick_proxy")
        }
        self.assertIn("coil_a", pick_ids)
        self.assertIn("coil_b", pick_ids)

    def test_map_preview_scene_legend_is_the_object_checklist_visibility_model(self):
        choices = self.adapter.field_volume_map_scene_objects()
        known_ids = {str(choice["id"]) for choice in choices}
        source_visibility = {
            str(choice["id"]): bool(choice["visible"])
            for choice in choices
        }
        figure = self.adapter.field_map_preview_scene(choices, ["coil_a"])

        legend_traces = [
            trace
            for trace in figure["data"]
            if trace.get("meta", {}).get("field_workbench_map_preview_legend")
        ]
        self.assertEqual(
            {
                trace["meta"]["field_workbench_object_id"]
                for trace in legend_traces
            },
            known_ids,
        )
        visibility = {
            trace["meta"]["field_workbench_object_id"]: trace["visible"]
            for trace in legend_traces
        }
        self.assertIs(visibility["coil_a"], True)
        for object_id in known_ids - {"coil_a"}:
            self.assertEqual(visibility[object_id], "legendonly")
        self.assertTrue(all(trace["showlegend"] for trace in legend_traces))
        self.assertTrue(
            all(
                not trace.get("showlegend", False)
                for trace in figure["data"]
                if not trace.get("meta", {}).get(
                    "field_workbench_map_preview_legend"
                )
            )
        )
        self.assertFalse(
            any(
                str(trace.get("name", "")).startswith("Selected:")
                for trace in figure["data"]
            )
        )
        self.assertEqual(
            {
                str(choice["id"]): bool(choice["visible"])
                for choice in self.adapter.field_volume_map_scene_objects()
            },
            source_visibility,
        )

    def test_incremental_map_preview_loads_each_frozen_object_once(self):
        choices = self.adapter.field_volume_map_scene_objects()
        known_ids = {str(choice["id"]) for choice in choices}
        source_document = self.adapter.document()

        figure = self.adapter.field_map_incremental_preview_scene(
            choices,
            ["coil_a"],
            object_opacities={"coil_a": 0.4},
        )

        actual_by_id = {object_id: [] for object_id in known_ids}
        legends_by_id = {object_id: [] for object_id in known_ids}
        for trace in figure["data"]:
            metadata = trace.get("meta", {})
            object_id = str(metadata.get("field_workbench_object_id", ""))
            self.assertIn(object_id, known_ids)
            self.assertTrue(metadata.get("field_workbench_incremental_preview"))
            self.assertFalse(metadata.get("field_workbench_pick_proxy", False))
            if metadata.get("field_workbench_map_preview_legend"):
                legends_by_id[object_id].append(trace)
            else:
                actual_by_id[object_id].append(trace)
                self.assertIn("field_workbench_preview_opacity_scale", metadata)

        self.assertTrue(all(actual_by_id.values()))
        self.assertTrue(all(legends_by_id.values()))
        self.assertTrue(all(trace["visible"] is True for trace in actual_by_id["coil_a"]))
        self.assertTrue(
            all(
                trace["visible"] is False
                for object_id in known_ids - {"coil_a"}
                for trace in actual_by_id[object_id]
            )
        )
        self.assertTrue(all(trace["visible"] is True for trace in legends_by_id["coil_a"]))
        self.assertTrue(
            all(
                trace["visible"] == "legendonly"
                for object_id in known_ids - {"coil_a"}
                for trace in legends_by_id[object_id]
            )
        )
        self.assertEqual(self.adapter.document(), source_document)

    def test_spherical_measurement_volume_is_reusable_optimizer_target(self):
        target_id = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#16a34a",
                        "opacity": 0.25,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": target_id,
                        "quality": "standard",
                        "target_uT": 25.0,
                        "tolerance_pct": 7.5,
                    },
                },
                {
                    "method": "set_param",
                    "params": {
                        "object_id": target_id,
                        "name": "diameter",
                        "value": 0.04,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": target_id,
                        "position": [0.01, -0.02, 0.03],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
            ]
        )
        target = self.adapter.optimizer_targets()[0]
        self.assertEqual(target["id"], target_id)
        self.assertEqual(target["diameter_mm"], 40.0)
        self.assertEqual(target["position_mm"], [10.0, -20.0, 30.0])
        self.assertEqual(target["target_uT"], 25.0)
        self.assertEqual(target["tolerance_pct"], 7.5)
        defaults = self.adapter.optimizer_default_settings(target_id)
        self.assertEqual(defaults["target_object_id"], target_id)
        self.assertEqual(defaults["coil_count_max"], 4)

    def test_question_optimizer_pair_spacing_preserves_midpoint_and_exact_separation(self):
        candidate = StudioAdapter(self.adapter.document())
        before = [
            np.asarray(candidate.get_transform(object_id)["position"], dtype=float)
            for object_id in ("coil_a", "coil_b")
        ]
        midpoint_before = 0.5 * (before[0] + before[1])
        transforms = candidate._optimizer_tuning_apply_spacing(
            candidate, ["coil_a", "coil_b"], 80.0
        )
        after = [
            np.asarray(candidate.get_transform(object_id)["position"], dtype=float)
            for object_id in ("coil_a", "coil_b")
        ]
        self.assertTrue(np.allclose(0.5 * (after[0] + after[1]), midpoint_before))
        self.assertAlmostEqual(float(np.linalg.norm(after[1] - after[0])), 0.08, places=12)
        self.assertEqual(set(transforms), {"coil_a", "coil_b"})

        reasons, clearance = candidate._optimizer_tuning_drc_summary(
            {
                "results": [
                    {
                        "status": "below_clearance",
                        "object_a_label": "Lower coil",
                        "object_b_label": "Upper coil",
                        "rule": "Coil-to-coil clearance",
                        "signed_clearance_mm": 0.5,
                    }
                ]
            }
        )
        self.assertTrue(reasons)
        self.assertAlmostEqual(clearance, 0.5)

    def test_question_optimizer_vector_field_stamp_tracks_moved_rotated_scaled_circle(self):
        self.adapter.set_coil_modeling_method("centreline")
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.015, 0.0, 0.0],
                [0.0, -0.012, 0.008],
                [-0.01, 0.006, -0.004],
            ],
            dtype=float,
        )
        variables = [
            {
                "id": "diameter:A",
                "kind": "circular_diameter",
                "coil_ids": ["coil_a"],
                "min": 180.0,
                "max": 220.0,
                "base": 200.0,
            },
            {
                "id": "position_x:A",
                "kind": "position_x",
                "coil_ids": ["coil_a"],
                "min": -12.0,
                "max": 12.0,
                "base": 0.0,
            },
            {
                "id": "orientation_y:A",
                "kind": "orientation_y",
                "coil_ids": ["coil_a"],
                "min": -20.0,
                "max": 20.0,
                "base": 0.0,
            },
        ]
        stamps, diagnostics = self.adapter._optimizer_tuning_build_field_stamps(
            points, ["coil_a"], variables, resolution=32
        )
        self.assertTrue(diagnostics["enabled"], diagnostics.get("reason"))
        self.assertEqual(set(stamps), {"coil_a"})

        candidate = StudioAdapter(self.adapter.document())
        candidate.set_coil_modeling_method("centreline")
        candidate._optimizer_tuning_apply_path_dimension(
            candidate, ["coil_a"], field="circular_diameter", value_mm=220.0
        )
        candidate._optimizer_tuning_apply_translation(
            candidate, ["coil_a"], axis=0, offset_mm=9.0
        )
        candidate._optimizer_tuning_apply_orientation_delta(
            candidate, ["coil_a"], axis=1, delta_deg=13.0
        )

        record = stamps["coil_a"]
        position, rotation = candidate._coil_world_transform("coil_a")
        approximate = record["stamp"].sample_world(
            points,
            position_m=position,
            rotation_matrix=rotation,
            geometric_scale=0.22 / float(record["base_diameter_m"]),
        )

        direct = StudioAdapter(candidate.document())
        direct.set_coil_modeling_method("centreline")
        direct.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": "coil_a", "drive_current_a": 1.0},
                },
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": "coil_b", "drive_current_a": 0.0},
                },
            ]
        )
        exact = np.asarray(direct.get_field(points=points, field="B")["values"], dtype=float)
        np.testing.assert_allclose(approximate, exact, rtol=0.05, atol=1e-10)

    def test_question_optimizer_translation_only_cache_uses_exact_observer_shift_not_3d_stamp(self):
        self.adapter.set_coil_modeling_method("centreline")
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.012, 0.0, 0.004],
                [-0.008, 0.006, -0.003],
            ],
            dtype=float,
        )
        variables = [
            {
                "id": "position_z:A",
                "kind": "position_z",
                "coil_ids": ["coil_a"],
                "min": -20.0,
                "max": 20.0,
                "base": 0.0,
            }
        ]
        cache, diagnostics = self.adapter._optimizer_tuning_build_field_stamps(
            points, ["coil_a"], variables, resolution=32
        )
        self.assertTrue(diagnostics["enabled"], diagnostics.get("reason"))
        self.assertEqual(diagnostics.get("mode"), "direct_observer_translation")
        self.assertEqual(diagnostics.get("build_points"), 0)
        self.assertEqual(diagnostics.get("resolution"), 0)
        record = cache["coil_a"]
        self.assertEqual(record.get("mode"), "direct_translation")
        self.assertNotIn("stamp", record)

        candidate = StudioAdapter(self.adapter.document())
        candidate.set_coil_modeling_method("centreline")
        candidate._optimizer_tuning_apply_translation(
            candidate, ["coil_a"], axis=2, offset_mm=9.0
        )
        position, rotation = candidate._coil_world_transform("coil_a")
        local_points = (points - position.reshape(1, 3)) @ rotation
        shifted = np.asarray(
            self.adapter._evaluate_magpylib_sources(
                [record["local_source"]], local_points, "B"
            ),
            dtype=float,
        ) @ rotation.T

        direct = StudioAdapter(candidate.document())
        direct.set_coil_modeling_method("centreline")
        direct.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": "coil_a", "drive_current_a": 1.0},
                },
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": "coil_b", "drive_current_a": 0.0},
                },
            ]
        )
        exact = np.asarray(direct.get_field(points=points, field="B")["values"], dtype=float)
        np.testing.assert_allclose(shifted, exact, rtol=1e-8, atol=1e-12)

    def test_question_optimizer_translation_cache_is_source_cost_aware_for_segmented_polylines(self):
        source = inspect.getsource(StudioAdapter._optimizer_tuning_build_field_stamps)
        self.assertIn("translation_polyline_ids", source)
        self.assertIn("translation_expensive_polyline_ids", source)
        self.assertIn("requested_resolution = min(requested_resolution, 24)", source)
        self.assertIn("finely segmented Polyline/racetrack source", source)

        scene = compact_workflow_scene()
        scene["objects"][0] = {
            "id": "coil_a",
            "type": "current.Polyline",
            "params": {
                "vertices": [
                    [-0.05, -0.03, 0.0],
                    [-0.025, -0.03, 0.0],
                    [0.0, -0.03, 0.0],
                    [0.02500000000000001, -0.03, 0.0],
                    [0.05, -0.03, 0.0],
                    [0.05, -0.015, 0.0],
                    [0.05, 0.0, 0.0],
                    [0.05, 0.015, 0.0],
                    [0.05, 0.03, 0.0],
                    [0.025, 0.03, 0.0],
                    [0.0, 0.03, 0.0],
                    [-0.02500000000000001, 0.03, 0.0],
                    [-0.05, 0.03, 0.0],
                    [-0.05, 0.015, 0.0],
                    [-0.05, 0.0, 0.0],
                    [-0.05, -0.015, 0.0],
                    [-0.05, -0.03, 0.0],
                    [-0.05, -0.03, 0.0],
                ],
                "current": 0.07,
                "position": [0.0, 0.0, -0.05],
            },
            "style": {"label": "Segmented coil"},
        }
        polyline = StudioAdapter(scene)
        polyline.set_coil_modeling_method("centreline")
        points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.005]], dtype=float)
        variables = [{
            "id": "position_z:A", "kind": "position_z", "coil_ids": ["coil_a"],
            "min": -20.0, "max": 20.0, "base": 0.0,
        }]
        with mock.patch.object(
            polyline,
            "_evaluate_magpylib_sources",
            side_effect=lambda _sources, query_points, _field: np.zeros((len(query_points), 3), dtype=float),
        ):
            cache, diagnostics = polyline._optimizer_tuning_build_field_stamps(
                points, ["coil_a"], variables, resolution=32
            )
        self.assertTrue(diagnostics["enabled"], diagnostics.get("reason"))
        self.assertEqual(diagnostics.get("mode"), "grid_stamp")
        self.assertEqual(diagnostics.get("resolution"), 24)
        self.assertEqual(diagnostics.get("translation_expensive_polyline_count"), 1)
        self.assertEqual(diagnostics.get("build_points"), 24 ** 3)
        self.assertIn("stamp", cache["coil_a"])

    def test_question_optimizer_stamp_native_nd_map_bypasses_candidate_adapters(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn("def compute_stamp_native_nd", source)
        self.assertIn("stamp_native_nd = bool(", source)
        self.assertIn('"pair_spacing"', source)
        self.assertIn('"orientation_x"', source)
        self.assertIn('"circular_diameter"', source)
        self.assertIn("precomputed_basis=(fixed_vectors, basis)", source)
        self.assertIn("defer_electrical_metrics=True", source)
        self.assertIn("native_scene_construction_count", source)
        self.assertIn("if stamp_native_nd:", source)
        native_segment = source.split("def compute_stamp_native_nd", 1)[1].split("def commit_structure", 1)[0]
        self.assertNotIn("StudioAdapter(copy.deepcopy", native_segment)
        self.assertNotIn("compute_structure(clean", native_segment)

    def test_question_optimizer_authoritative_candidates_use_direct_final_field(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        build_segment = source.split("def build_candidate", 1)[1].split("if current_only:", 1)[0]
        self.assertIn('registry_default_fidelity == "authoritative"', build_segment)
        self.assertIn('direct_adapter.get_field(points=points, field="B")', build_segment)
        self.assertIn("stats = stats_for_vectors(canonical_vectors, registration_matrix_local)", build_segment)
        self.assertIn('"field_vectors_t": canonical_vectors', build_segment)

    def test_optimizer_registry_candidate_adapter_reuses_recorded_model(self):
        candidate = {
            "mode": "tune_existing",
            "transforms": {},
            "physical_updates": {},
            "parameter_updates": {},
            "currents_a": {},
            "registry_model": "centreline",
        }
        adapter = self.adapter.optimizer_candidate_adapter(
            candidate,
            source_document=self.adapter.document(),
        )
        self.assertEqual(adapter.coil_modeling_method, "centreline")

    def test_optimizer_field_stamp_rotation_envelope_caps_full_xyz_orbit_once(self):
        variables = [
            {
                "id": f"ar_{axis}",
                "kind": f"assembly_orientation_{axis}",
                "coil_ids": ["coil_a", "coil_b"],
                "min": -180.0,
                "max": 180.0,
                "base": 0.0,
            }
            for axis in ("x", "y", "z")
        ]
        allowance = self.adapter._optimizer_tuning_field_stamp_position_allowance_m(
            "coil_a", variables
        )
        # coil_a is 50 mm from the pair midpoint, so arbitrary rigid rotation can
        # move its centre at most to the opposite side of that 50 mm orbit: 100 mm.
        self.assertAlmostEqual(allowance, 0.1, places=12)

    def test_optimizer_field_stamp_pair_pivot_bound_ignores_euler_range_size(self):
        variables = [
            {
                "id": f"ar_{axis}",
                "kind": f"assembly_orientation_{axis}",
                "coil_ids": ["coil_a", "coil_b"],
                "min": -360.0,
                "max": 360.0,
                "base": 0.0,
            }
            for axis in ("x", "y", "z")
        ]
        variables.append({
            "id": "spacing", "kind": "pair_spacing",
            "coil_ids": ["coil_a", "coil_b"],
            "min": 80.0, "max": 200.0, "base": 100.0,
        })
        points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float)
        bound = self.adapter._optimizer_tuning_field_stamp_target_distance_bound_m(
            "coil_a", points, variables
        )
        # 10 mm target radius about the pair midpoint + 100 mm maximum half-spacing.
        self.assertAlmostEqual(bound, 0.11, places=12)

    def test_question_optimizer_scene_free_exact_centreline_fallback_bypasses_candidate_adapters(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn("numeric_native_sources", source)
        self.assertIn('"exact_numeric"', source)
        self.assertIn("local_query = local_points / geometric_scale", source)
        self.assertIn("native_exact_evaluation_count", source)
        self.assertIn("field_representation_label", source)
        native_segment = source.split("def stamp_native_unit_vectors", 1)[1].split(
            "def compute_stamp_native_nd", 1
        )[0]
        self.assertNotIn("StudioAdapter(copy.deepcopy", native_segment)

    def test_question_optimizer_progressive_intermediate_physical_promotion_precedes_full_vet(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn("Finalist promotion •", source)
        self.assertIn('intermediate_settings["_optimizer_sample_limit"]', source)
        self.assertIn('promotion_pool = _PersistentOptimizerProcessPool', source)
        self.assertIn('intermediate_ranking_decisive', source)
        self.assertIn('initial_validation_target', source)
        self.assertIn('1 if intermediate_decisive else 2', source)
        self.assertIn('progressive_extra_count', source)
        self.assertIn("equivalent_families", source)
        self.assertIn("representative_pairs", source)
        self.assertIn("successful_screen_keys", source)
        self.assertIn('"intermediate_equivalent_removed_count"', source)

    def test_question_optimizer_unsaturated_map_landmarks_survive_budget_exhaustion(self):
        source = inspect.getsource(StudioAdapter.run_optimizer_tuning)
        self.assertIn("Saturation is a stopping hint, not an", source)
        self.assertIn("coarse ranked landmarks were preserved", source)
        self.assertIn("unsaturated_landmarks_retained", source)
        self.assertIn('field_stamp_diagnostics.get("mode") == "direct_observer_translation"', source)
        self.assertIn("prefill_direct_translation_vectors(pending)", source)

    def test_question_optimizer_full_freedom_helpers_cover_translation_orientation_and_dimensions(self):
        candidate = StudioAdapter(self.adapter.document())
        original_a = np.asarray(candidate.get_transform("coil_a")["position"], dtype=float)
        original_b = np.asarray(candidate.get_transform("coil_b")["position"], dtype=float)

        moved = candidate._optimizer_tuning_apply_translation(
            candidate, ["coil_a", "coil_b"], axis=0, offset_mm=12.5
        )
        moved_a = np.asarray(candidate.get_transform("coil_a")["position"], dtype=float)
        moved_b = np.asarray(candidate.get_transform("coil_b")["position"], dtype=float)
        self.assertTrue(np.allclose(moved_a - original_a, [0.0125, 0.0, 0.0]))
        self.assertTrue(np.allclose(moved_b - original_b, [0.0125, 0.0, 0.0]))
        self.assertTrue(np.allclose(moved_b - moved_a, original_b - original_a))
        self.assertEqual(set(moved), {"coil_a", "coil_b"})

        rotated = candidate._optimizer_tuning_apply_orientation_delta(
            candidate, ["coil_a", "coil_b"], axis=1, delta_deg=15.0
        )
        self.assertEqual(set(rotated), {"coil_a", "coil_b"})
        self.assertTrue(np.allclose(candidate.get_transform("coil_a")["position"], moved_a))

        # Explicit assembly rotation is different: it rotates the coil centres
        # around the linked-group midpoint as well as rotating each coil plane.
        rigid = StudioAdapter(self.adapter.document())
        rigid_before = [
            np.asarray(rigid.get_transform(object_id)["position"], dtype=float)
            for object_id in ("coil_a", "coil_b")
        ]
        rigid_midpoint = 0.5 * (rigid_before[0] + rigid_before[1])
        rigid_axis_before = rigid_before[1] - rigid_before[0]
        rigid_result = rigid._optimizer_tuning_apply_assembly_orientation_delta(
            rigid, ["coil_a", "coil_b"], axis=2, delta_deg=90.0
        )
        rigid_after = [
            np.asarray(rigid.get_transform(object_id)["position"], dtype=float)
            for object_id in ("coil_a", "coil_b")
        ]
        self.assertEqual(set(rigid_result), {"coil_a", "coil_b"})
        self.assertTrue(np.allclose(0.5 * (rigid_after[0] + rigid_after[1]), rigid_midpoint))
        expected_axis = Rotation.from_euler("z", 90.0, degrees=True).apply(rigid_axis_before)
        self.assertTrue(np.allclose(rigid_after[1] - rigid_after[0], expected_axis, atol=1e-12))

        # Pair spacing now acts along that already-rotated centreline.
        rigid._optimizer_tuning_apply_spacing(rigid, ["coil_a", "coil_b"], 80.0)
        spaced = [
            np.asarray(rigid.get_transform(object_id)["position"], dtype=float)
            for object_id in ("coil_a", "coil_b")
        ]
        spaced_axis = spaced[1] - spaced[0]
        self.assertAlmostEqual(float(np.linalg.norm(spaced_axis)), 0.08, places=12)
        self.assertTrue(
            np.allclose(
                spaced_axis / np.linalg.norm(spaced_axis),
                expected_axis / np.linalg.norm(expected_axis),
                atol=1e-12,
            )
        )

        axial = candidate._optimizer_tuning_apply_physical_value(
            candidate, ["coil_a", "coil_b"], field="winding_axial_width_mm", value=22.0
        )
        self.assertEqual(axial["coil_a"]["winding_axial_width_mm"], 22.0)
        self.assertAlmostEqual(
            candidate.coil_physical_properties("coil_a")["winding_axial_width_mm"], 22.0
        )

        path = candidate._optimizer_tuning_apply_path_dimension(
            candidate, ["coil_a", "coil_b"], field="circular_diameter", value_mm=180.0
        )
        self.assertEqual(path["coil_a"]["name"], "diameter")
        diameter = next(
            item["value"] for item in candidate.get_params("coil_a") if item["name"] == "diameter"
        )
        self.assertAlmostEqual(float(diameter), 0.18)

    def test_question_optimizer_can_apply_linked_turn_count_on_private_candidate(self):
        candidate = StudioAdapter(self.adapter.document())
        source_signature = self.adapter.signature()
        updates = candidate._optimizer_tuning_apply_physical_value(
            candidate, ["coil_a", "coil_b"], field="turns", value=333
        )
        self.assertEqual(updates["coil_a"]["turns"], 333)
        self.assertEqual(updates["coil_b"]["turns"], 333)
        self.assertEqual(candidate.coil_physical_properties("coil_a")["turns"], 333)
        self.assertEqual(candidate.coil_physical_properties("coil_b")["turns"], 333)
        self.assertEqual(self.adapter.signature(), source_signature)

    def test_question_optimizer_tunes_existing_current_without_editing_source(self):
        target_id = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#64748b",
                        "opacity": 0.2,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": target_id,
                        "sampling_mode": "user_defined",
                        "defined_points_mm": [
                            [0.0, 0.0, 0.0],
                            [10.0, 0.0, 0.0],
                            [-10.0, 0.0, 0.0],
                        ],
                        "target_uT": 200.0,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        coils = self.adapter.optimizer_existing_coils()
        self.assertEqual([item["id"] for item in coils], ["coil_a", "coil_b"])
        signature = self.adapter.signature()
        result = self.adapter.run_optimizer_tuning(
            {
                "workflow_mode": "tuning",
                "tuning_mode": "current",
                "objective": "coverage",
                "coil_ids": ["coil_a", "coil_b"],
                "target_object_ids": [target_id],
                "target_uT": 200.0,
                "tolerance_pct": 5.0,
                "current_scale_min": 0.5,
                "current_scale_max": 1.5,
                "current_steps": 5,
                "use_drc": False,
                "enforce_electrical_limits": False,
                "finalist_count": 3,
            }
        )
        self.assertEqual(self.adapter.signature(), signature)
        self.assertEqual(result["run_mode"], "tuning")
        self.assertEqual(result["generated_candidate_count"], 1)
        self.assertTrue(result["convergence"]["converged"])
        self.assertEqual(result["search_strategy"]["outer_search"], "direct_current_solve")
        self.assertGreaterEqual(len(result["finalists"]), 1)
        best = result["finalists"][0]
        self.assertEqual(best["mode"], "tune_existing")
        self.assertEqual(set(best["currents_a"]), {"coil_a", "coil_b"})
        self.adapter.apply_optimizer_candidate(best)
        tuned_a = self.adapter.coil_physical_properties("coil_a")["drive_current_a"]
        tuned_b = self.adapter.coil_physical_properties("coil_b")["drive_current_a"]
        self.assertAlmostEqual(tuned_a, tuned_b, places=12)
        self.adapter.undo()
        self.assertEqual(self.adapter.signature(), signature)

    def test_optimizer_fixed_structural_value_map_preserves_requested_coordinate(self):
        target_id = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#64748b",
                        "opacity": 0.2,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": target_id,
                        "sampling_mode": "user_defined",
                        "defined_points_mm": [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                        "target_uT": 200.0,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        variables = [
            {
                "id": "current:A", "kind": "current", "group_kind": "current", "group": "A",
                "coil_ids": ["coil_a", "coil_b"], "min": 20.0, "max": 100.0, "base": 70.0,
                "label": "Current Group A", "unit": "mA", "integer": False,
                "current_link_mode": "same_magnitude",
            },
            {
                "id": "position_x:position:A", "kind": "position_x",
                "group_kind": "position", "group": "A",
                "coil_ids": ["coil_a", "coil_b"], "min": -5.0, "max": 5.0, "base": 0.0,
                "step": 0.5, "step_origin": 0.0,
                "label": "X translation Group A", "unit": "mm", "integer": False,
            },
            {
                "id": "position_z:position:A", "kind": "position_z",
                "group_kind": "position", "group": "A",
                "coil_ids": ["coil_a", "coil_b"], "min": -5.0, "max": 5.0, "base": 0.0,
                "step": 0.5, "step_origin": 0.0,
                "label": "Z translation Group A", "unit": "mm", "integer": False,
            },
        ]
        result = self.adapter.run_optimizer_tuning(
            {
                "workflow_mode": "tuning",
                "tuning_mode": "multi",
                "objective": "coverage",
                "coil_ids": ["coil_a", "coil_b"],
                "variables": variables,
                "target_object_ids": [target_id],
                "target_uT": 200.0,
                "tolerance_pct": 5.0,
                "worker_count": 1,
                "use_drc": False,
                "enforce_electrical_limits": False,
                "finalist_count": 1,
                "_fixed_structural_value_map": {
                    "position_x:position:A": 0.0,
                    "position_z:position:A": 0.5,
                },
                "_fixed_candidate_index": 211,
            }
        )
        self.assertEqual(len(result["finalists"]), 1)
        finalist = result["finalists"][0]
        self.assertEqual(finalist["search_index"], 211)
        self.assertAlmostEqual(
            float(finalist["variable_values"]["position_x:position:A"]), 0.0
        )
        self.assertAlmostEqual(
            float(finalist["variable_values"]["position_z:position:A"]), 0.5
        )
        # A fixed worker/targeted evaluation is an exact coordinate evaluation.
        # It must not fall through into the normal finalist canonical-simplification
        # pass, which is allowed to restore small source offsets or axis rotations.
        self.assertFalse(
            bool(result.get("multi_fidelity", {}).get("canonical_simplification_performed", False))
        )

    def test_optimizer_existing_coils_include_scene_group_ancestry(self):
        group_id = self.adapter.group_objects(["coil_a", "coil_b"])
        coils = {item["id"]: item for item in self.adapter.optimizer_existing_coils()}
        for coil_id in ("coil_a", "coil_b"):
            path = coils[coil_id].get("group_path", [])
            self.assertTrue(path)
            self.assertEqual(path[-1]["id"], group_id)

    def test_optimizer_snapshot_reference_uses_measurement_volume_centre_for_exposure_pivot(self):
        target_id = "offset_target"
        snapshot_id = "snapshot_offset"
        local_points = np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]],
            dtype=float,
        )
        source_rotation = Rotation.from_euler(
            "xyz", [7.0, -11.0, 23.0], degrees=True
        ).as_matrix()
        source_position = np.asarray([0.2, -0.1, 0.4], dtype=float)
        snapshot_world = local_points @ source_rotation.T + source_position
        snapshot_vectors_local = np.asarray(
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.3, 0.7, 0.1], [0.1, 0.2, 0.9]],
            dtype=float,
        ) * 1e-6
        snapshot_vectors_world = snapshot_vectors_local @ source_rotation.T

        target_rotation = Rotation.from_euler(
            "xyz", [-13.0, 5.0, 31.0], degrees=True
        ).as_matrix()
        target_origin = np.asarray([-0.3, 0.25, 0.12], dtype=float)
        local_volume_centre = np.asarray([0.018, -0.007, 0.011], dtype=float)
        expected_world_centre = local_volume_centre @ target_rotation.T + target_origin
        target_record = {
            "id": target_id,
            "label": "Offset target",
            "shape": "triangles",
            "dimensions_mm": [80.0, 60.0, 50.0],
        }
        snapshot = {
            "snapshot_name": "Offset reference",
            "snapshot_status": "current",
            "shape": "triangles",
            "dimensions_mm": [80.0, 60.0, 50.0],
            "local_points_m": local_points,
            "world_points_m": snapshot_world,
            "values_t": snapshot_vectors_world,
        }
        fake_mesh = {"id": target_id}
        with (
            mock.patch.object(self.adapter, "optimizer_targets", return_value=[target_record]),
            mock.patch.object(self.adapter, "snapshot_result", return_value=snapshot),
            mock.patch.object(self.adapter, "_mesh", return_value=fake_mesh),
            mock.patch.object(
                self.adapter, "_mesh_world_transform",
                return_value=(target_origin, target_rotation),
            ),
            mock.patch.object(
                self.adapter, "measurement_geometry_info",
                return_value={"local_centre_m": local_volume_centre},
            ),
        ):
            reference = self.adapter.optimizer_snapshot_reference(target_id, snapshot_id)

        np.testing.assert_allclose(reference["position"], target_origin, atol=1e-12)
        np.testing.assert_allclose(
            reference["registration_centre_local_m"], local_volume_centre, atol=1e-12
        )
        np.testing.assert_allclose(
            reference["registration_centre_world_m"], expected_world_centre, atol=1e-12
        )
        # The exposure pivot deliberately differs from the Measurement object's
        # transform origin when asymmetric geometry has an offset physical centre.
        self.assertFalse(
            np.allclose(reference["registration_centre_world_m"], target_origin, atol=1e-12)
        )

    def test_exposure_registration_rigidly_rotates_source_pose_about_target(self):
        target_position = np.asarray([0.1, -0.2, 0.3], dtype=float)
        target_rotation = Rotation.from_euler(
            "xyz", [10.0, 20.0, -15.0], degrees=True
        ).as_matrix()
        local_registration = Rotation.from_euler(
            "xyz", [0.0, 0.0, 90.0], degrees=True
        ).as_matrix()
        source_position = target_position + target_rotation @ np.asarray(
            [0.04, 0.0, 0.0], dtype=float
        )
        source_rotation = target_rotation.copy()

        registered_position, registered_rotation = (
            StudioAdapter._exposure_registered_pose(
                source_position,
                source_rotation,
                target_position_m=target_position,
                target_rotation=target_rotation,
                local_rotation=local_registration,
            )
        )
        expected_position = target_position + target_rotation @ np.asarray(
            [0.0, 0.04, 0.0], dtype=float
        )
        expected_world_rotation = (
            target_rotation @ local_registration @ target_rotation.T
        )
        np.testing.assert_allclose(registered_position, expected_position, atol=1e-12)
        np.testing.assert_allclose(
            registered_rotation, expected_world_rotation @ source_rotation, atol=1e-12
        )
        # A genuine exposure registration moves spatial structure around the target;
        # it is not merely a post-hoc rotation of B-vector arrows at fixed samples.
        self.assertFalse(np.allclose(registered_position, source_position, atol=1e-12))

    def test_vector_alignment_helper_still_uses_a_proper_rotation(self):
        reference = np.asarray(
            [[1.0, 0.0, 0.0], [0.2, 2.0, 0.1], [-0.3, 0.4, 1.5], [0.7, -0.2, 0.9]],
            dtype=float,
        )
        active = Rotation.from_euler("xyz", [35.0, -20.0, 70.0], degrees=True).as_matrix()
        candidate = reference @ active.T
        q, aligned = StudioAdapter._best_fit_vector_rotation(candidate, reference)
        self.assertAlmostEqual(float(np.linalg.det(q)), 1.0, places=10)
        np.testing.assert_allclose(aligned, reference, atol=1e-10)

    def test_question_optimizer_can_match_a_stored_snapshot_with_current_freedom(self):
        target_id = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#64748b",
                        "opacity": 0.2,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": target_id,
                        "sampling_mode": "user_defined",
                        "defined_points_mm": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                        "target_uT": 200.0,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        reference_result = self.adapter.analyze_measurement(target_id)
        snapshot_id = self.adapter.create_snapshot(reference_result, "Reference exposure")
        # Move away from the captured exposure before asking the optimizer to recover it.
        self.adapter.apply_operations(
            [
                {"method": "set_coil_physical", "params": {"object_id": "coil_a", "drive_current_a": 0.04}},
                {"method": "set_coil_physical", "params": {"object_id": "coil_b", "drive_current_a": 0.04}},
            ]
        )
        variables = [
            {
                "id": "current:A", "kind": "current", "group_kind": "current", "group": "A",
                "coil_ids": ["coil_a", "coil_b"], "min": 20.0, "max": 100.0, "base": 40.0,
                "label": "Current Group A", "unit": "mA", "integer": False,
                "current_link_mode": "same_magnitude",
            }
        ]
        result = self.adapter.run_optimizer_tuning(
            {
                "workflow_mode": "tuning",
                "tuning_mode": "current",
                "reference_mode": "snapshot",
                "reference_snapshot_id": snapshot_id,
                "objective": "snapshot_vector_match",
                "coil_ids": ["coil_a", "coil_b"],
                "variables": variables,
                "target_object_ids": [target_id],
                "snapshot_magnitude_tolerance_pct": 5.0,
                "snapshot_direction_tolerance_deg": 3.0,
                "use_drc": False,
                "enforce_electrical_limits": False,
                "finalist_count": 3,
            }
        )
        self.assertEqual(result["reference_mode"], "snapshot")
        self.assertEqual(result["reference_snapshot"]["id"], snapshot_id)
        self.assertEqual(result["generated_candidate_count"], 1)
        best = result["finalists"][0]
        self.assertLess(best["metrics"]["snapshot_vector_rms_uT"], 1e-3)
        self.assertGreater(best["metrics"]["snapshot_match_coverage_pct"], 99.9)
        self.assertAlmostEqual(best["currents_a"]["coil_a"], 0.07, places=4)
        self.assertAlmostEqual(best["currents_a"]["coil_b"], 0.07, places=4)

        # Exposure registration must seed against the *reduced* snapshot reference,
        # not the original full target.  This specifically guards the Fast-scout bug
        # where candidate vectors were downsampled but the Kabsch reference was not,
        # silently forcing every registration search to start from identity.
        exposure_result = self.adapter.run_optimizer_tuning(
            {
                "workflow_mode": "tuning",
                "tuning_mode": "current",
                "reference_mode": "snapshot",
                "reference_snapshot_id": snapshot_id,
                "snapshot_alignment_mode": "exposure",
                "objective": "snapshot_vector_match",
                "coil_ids": ["coil_a", "coil_b"],
                "variables": variables,
                "target_object_ids": [target_id],
                "snapshot_magnitude_tolerance_pct": 5.0,
                "snapshot_direction_tolerance_deg": 3.0,
                "_optimizer_sample_limit": 3,
                "use_drc": False,
                "enforce_electrical_limits": False,
                "finalist_count": 1,
            }
        )
        exposure_diag = exposure_result["search_strategy"]["exposure_registration"]
        self.assertGreaterEqual(int(exposure_diag.get("candidate_count", 0)), 1)
        self.assertGreaterEqual(int(exposure_diag.get("kabsch_seed_count", 0)), 1)

    def test_question_optimizer_accepts_simultaneous_current_turns_and_translation_variables(self):
        target_id = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#64748b",
                        "opacity": 0.2,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": target_id,
                        "sampling_mode": "user_defined",
                        "defined_points_mm": [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [-12.0, 0.0, 0.0]],
                        "target_uT": 200.0,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        signature = self.adapter.signature()
        variables = [
            {
                "id": "current:A", "kind": "current", "group_kind": "current", "group": "A",
                "coil_ids": ["coil_a", "coil_b"], "min": 40.0, "max": 100.0, "base": 70.0,
                "label": "Current Group A", "unit": "mA", "integer": False,
                "current_link_mode": "same_magnitude",
            },
            {
                "id": "turns:A", "kind": "turns", "group_kind": "geometry", "group": "A",
                "coil_ids": ["coil_a", "coil_b"], "min": 250.0, "max": 350.0, "base": 300.0,
                "label": "Turns Group A", "unit": "turns", "integer": True,
            },
            {
                "id": "position_x:A", "kind": "position_x", "group_kind": "position", "group": "A",
                "coil_ids": ["coil_a", "coil_b"], "min": -5.0, "max": 5.0, "base": 0.0,
                "label": "X translation Group A", "unit": "mm", "integer": False,
            },
        ]
        result = self.adapter.run_optimizer_tuning(
            {
                "workflow_mode": "tuning",
                "tuning_mode": "multi",
                "objective": "coverage",
                "coil_ids": ["coil_a", "coil_b"],
                "variables": variables,
                "target_object_ids": [target_id],
                "target_uT": 200.0,
                "tolerance_pct": 5.0,
                "evaluation_budget": 12,
                "worker_count": 2,
                "random_seed": 1234,
                "use_drc": False,
                "enforce_electrical_limits": False,
                "finalist_count": 3,
            }
        )
        self.assertEqual(self.adapter.signature(), signature)
        self.assertEqual(result["run_mode"], "tuning")
        self.assertEqual(result["parameter_count"], 3)
        self.assertEqual(result["search_strategy"]["current_mode"], "inner_solve")
        self.assertEqual(result["search_strategy"]["outer_parameter_count"], 2)
        self.assertEqual(result["search_strategy"]["worker_count"], 2)
        self.assertEqual(
            result["search_strategy"]["worker_backend"],
            "persistent_spawned_processes",
        )
        self.assertEqual(result["settings"]["worker_count"], 2)
        self.assertLessEqual(result["generated_candidate_count"], 12)
        self.assertGreaterEqual(len(result["finalists"]), 1)
        serial_settings = copy.deepcopy(result["settings"])
        serial_settings["worker_count"] = 1
        serial_result = self.adapter.run_optimizer_tuning(serial_settings)
        self.assertEqual(serial_result["search_strategy"]["worker_backend"], "serial")
        # Worker count is an execution policy, not a promise that a small,
        # progressively promoted search will consume identical tied basins.
        # Both backends must nevertheless preserve the requested simultaneous
        # variable contract and return independently valid settings.
        requested_ids = {item["id"] for item in variables}
        for backend_result in (result, serial_result):
            self.assertGreaterEqual(len(backend_result["finalists"]), 1)
            for finalist in backend_result["finalists"]:
                self.assertEqual(set(finalist["variable_values"]), requested_ids)
                for variable in variables:
                    value = float(finalist["variable_values"][variable["id"]])
                    self.assertGreaterEqual(value, float(variable["min"]))
                    self.assertLessEqual(value, float(variable["max"]))
        best = result["finalists"][0]
        self.assertEqual(best["tuning_mode"], "multi")
        self.assertEqual(set(best["variable_values"]), {item["id"] for item in variables})
        self.assertIn("Current Group A", best["setting_summary"])
        self.assertIn("Turns Group A", best["setting_summary"])
        self.adapter.apply_optimizer_candidate(best)
        self.assertNotEqual(self.adapter.signature(), signature)
        self.adapter.undo()
        self.assertEqual(self.adapter.signature(), signature)

    def test_optimizer_worker_benchmark_uses_persistent_process_workloads(self):
        outcome: dict[str, object] = {}

        def run_benchmark() -> None:
            try:
                outcome["result"] = self.adapter.benchmark_optimizer_workers(
                    max_workers=2
                )
            except BaseException as error:  # noqa: BLE001 - preserve thread failure
                outcome["error"] = error

        thread = threading.Thread(target=run_benchmark, daemon=True)
        thread.start()
        thread.join(timeout=30.0)
        self.assertFalse(thread.is_alive(), "Optimizer process benchmark did not finish")
        if "error" in outcome:
            raise outcome["error"]
        result = outcome["result"]
        self.assertEqual([row["workers"] for row in result["results"]], [1, 2])
        self.assertEqual(result["task_count"], 12)
        self.assertEqual(result["observer_count"], 315)
        self.assertIn(result["recommended_workers"], {1, 2})
        self.assertEqual(result["worker_backend"], "persistent_spawned_processes")

    def test_optimizer_coverage_tie_prefers_flatter_field_before_rms_centering(self):
        flatter = {
            "in_band_pct": 100.0,
            "uniformity_spread_pct": 0.05,
            "rms_target_error_uT": 0.8,
            "directional_consistency_pct": 99.9,
        }
        better_centered_but_less_flat = {
            "in_band_pct": 100.0,
            "uniformity_spread_pct": 0.25,
            "rms_target_error_uT": 0.1,
            "directional_consistency_pct": 100.0,
        }
        self.assertLess(
            StudioAdapter._optimizer_tuning_objective_key(flatter, "coverage"),
            StudioAdapter._optimizer_tuning_objective_key(better_centered_but_less_flat, "coverage"),
        )

    def test_optimizer_inner_current_coverage_tie_prefers_target_centering(self):
        near_band_edge_with_tiny_spread_noise = {
            "in_band_pct": 100.0,
            "uniformity_spread_pct": 0.096060799,
            "rms_target_error_uT": 9.688,
            "directional_consistency_pct": 100.0,
        }
        centered_with_tiny_spread_noise = {
            "in_band_pct": 100.0,
            "uniformity_spread_pct": 0.096060801,
            "rms_target_error_uT": 0.05,
            "directional_consistency_pct": 100.0,
        }
        # Outer geometry ranking still prefers the microscopically flatter field.
        self.assertLess(
            StudioAdapter._optimizer_tuning_objective_key(
                near_band_edge_with_tiny_spread_noise, "coverage"
            ),
            StudioAdapter._optimizer_tuning_objective_key(
                centered_with_tiny_spread_noise, "coverage"
            ),
        )
        # For one fixed geometry, current should instead center the target.
        self.assertLess(
            StudioAdapter._optimizer_tuning_current_objective_key(
                centered_with_tiny_spread_noise, "coverage"
            ),
            StudioAdapter._optimizer_tuning_current_objective_key(
                near_band_edge_with_tiny_spread_noise, "coverage"
            ),
        )

    def test_optimizer_candidate_application_is_undoable(self):
        physical = self.adapter.coil_physical_properties("coil_a")
        spec = {
            key: value
            for key, value in physical.items()
            if key != "calculations"
        }
        spec.update(
            {
                "physical_geometry_enabled": True,
                "winding_axial_width_mm": 8.0,
                "bobbin_wall_thickness_mm": 3.0,
                "flange_height_mm": 6.0,
                "flange_thickness_mm": 3.0,
                "turns": 120,
                "drive_current_a": 0.03,
            }
        )
        candidate = {
            "coils": [
                {
                    "index": 1,
                    "diameter_mm": 80.0,
                    "turns": 120,
                    "awg": int(spec["awg"]),
                    "axial_width_mm": 8.0,
                    "radial_build_mm": 2.0,
                    "position_m": [0.0, 0.0, 0.12],
                    "euler_deg": [0.0, 0.0, 0.0],
                    "drive_current_a": 0.03,
                    "spec": spec,
                }
            ]
        }
        before = self.adapter.signature()
        self.adapter.apply_optimizer_candidate(candidate, replace_circular_coils=True)
        circles = [
            item
            for item in self.adapter.list_objects()
            if item["type"] == "current.Circle" and not item.get("derived")
        ]
        self.assertEqual(len(circles), 1)
        self.assertEqual(circles[0]["label"], "Optimized coil 1")
        self.assertAlmostEqual(
            self.adapter.coil_physical_properties(circles[0]["id"])["drive_current_a"],
            0.03,
        )
        self.adapter.undo()
        self.assertEqual(self.adapter.signature(), before)

    def test_small_fixed_optimizer_run_is_seed_reproducible_and_non_destructive(self):
        target_id = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#16a34a",
                        "opacity": 0.25,
                    },
                },
                {
                    "method": "set_param",
                    "params": {
                        "object_id": target_id,
                        "name": "diameter",
                        "value": 0.02,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": target_id,
                        "quality": "preview",
                        "target_uT": 10.0,
                        "tolerance_pct": 50.0,
                    },
                },
            ]
        )
        settings = self.adapter.optimizer_default_settings(target_id)
        settings.update(
            {
                "coil_count_optimize": False,
                "coil_count_fixed": 2,
                "placement_optimize": False,
                "diameter_optimize": False,
                "diameter_fixed_mm": 200.0,
                "turns_optimize": False,
                "turns_fixed": 300,
                "awg_optimize": False,
                "awg_fixed": 30,
                "axial_width_optimize": False,
                "axial_width_fixed_mm": 8.0,
                "minimum_in_band_pct": 0.0,
                "max_current_a": 0.2,
                "max_voltage_v": 100.0,
                "max_power_per_coil_w": 100.0,
                "max_total_power_w": 200.0,
                "max_total_mass_g": 5000.0,
                "population": 8,
                "generations": 1,
                "finalist_count": 2,
                "seed": 1234,
                "finite_pack_order": 1,
            }
        )
        signature = self.adapter.signature()
        first = self.adapter.run_optimizer(settings)
        second = self.adapter.run_optimizer(settings)
        self.assertEqual(self.adapter.signature(), signature)
        self.assertGreaterEqual(len(first["finalists"]), 1)
        self.assertEqual(
            [item["genome_signature"] for item in first["finalists"]],
            [item["genome_signature"] for item in second["finalists"]],
        )
        self.assertAlmostEqual(
            first["finalists"][0]["metrics"]["rms_target_error_uT"],
            second["finalists"][0]["metrics"]["rms_target_error_uT"],
            places=12,
        )

    def test_numbered_labels_square_coil_and_axis_sensor(self):
        sensor_1 = self.adapter.add_template("sensor")
        sensor_2 = self.adapter.add_template("sensor")
        axis_2 = self.adapter.add_template("axis_sensor")
        square = self.adapter.add_template("polyline")
        self.assertEqual(self.adapter.get_object(sensor_1)["label"], "Field sensor 1")
        self.assertEqual(self.adapter.get_object(sensor_2)["label"], "Field sensor 2")
        self.assertEqual(self.adapter.get_object(axis_2)["label"], "Linear axis sensor 1")
        self.assertEqual(self.adapter.get_object(square)["label"], "Square coil 1")
        transform = self.adapter.get_transform(axis_2)
        self.assertEqual(transform["path_length"], 61)
        self.assertEqual(transform["path"][0], [0.0, 0.0, -0.15])
        self.assertEqual(transform["path"][-1], [0.0, 0.0, 0.15])

    def test_import_mesh_optional_centering_uses_bounding_box_centre(self):
        mesh_text = "\n".join(
            [
                "v 100 200 300",
                "v 140 200 300",
                "v 100 260 300",
                "v 100 200 380",
                "f 1 3 2",
                "f 1 2 4",
                "f 2 3 4",
                "f 3 1 4",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset mesh.obj"
            path.write_text(mesh_text, encoding="utf-8")
            uncentred_id = self.adapter.import_mesh(path, unit_scale=0.001)
            centred_id = self.adapter.import_mesh(
                path, unit_scale=0.001, center=True
            )

        uncentred = np.asarray(self.adapter._mesh(uncentred_id)["mesh"]["vertices"])
        centred_mesh = self.adapter._mesh(centred_id)
        centred = np.asarray(centred_mesh["mesh"]["vertices"])
        self.assertTrue(np.allclose(np.min(uncentred, axis=0), [0.1, 0.2, 0.3]))
        self.assertTrue(
            np.allclose(
                0.5 * (np.min(centred, axis=0) + np.max(centred, axis=0)),
                [0.0, 0.0, 0.0],
            )
        )
        self.assertTrue(centred_mesh["centered_on_import"])
        self.assertTrue(
            np.allclose(centred_mesh["source_center_m"], [0.12, 0.23, 0.34])
        )
        self.assertEqual(
            self.adapter.geometry_properties(uncentred_id)["dimensions_mm"],
            self.adapter.geometry_properties(centred_id)["dimensions_mm"],
        )

    def test_sensor_plot_routes_progress_through_selected_solver(self):
        updates = []
        figure = self.adapter.sensor_plot(
            "Bz",
            progress_callback=updates.append,
            progress_label="Sensor path test",
        )
        self.assertTrue(figure.get("data"))
        self.assertTrue(updates)
        self.assertEqual(updates[-1]["completed"], updates[-1]["total"])
        self.assertIn("Sensor path test", updates[-1]["stage"])

    def test_sensor_plot_csv_rows_preserve_all_plotted_series(self):
        figure = {
            "data": [
                {"name": "Axis sensor 1", "x": [0.0, 5.0], "y": [1e-6, 2e-6]},
                {"name": "Axis sensor 2", "x": [0.0, 5.0], "y": [3e-6, 4e-6]},
            ],
            "layout": {
                "xaxis": {"title": {"text": "Path distance (mm)"}},
                "yaxis": {"title": {"text": "Bz (µT)"}},
            },
        }
        rows, metadata = self.adapter.sensor_plot_export_rows(figure, "Bz")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["series"], "Axis sensor 1")
        self.assertEqual(rows[-1]["series"], "Axis sensor 2")
        self.assertEqual(rows[-1]["sample_index"], 1)
        self.assertEqual(rows[-1]["path_value"], 5.0)
        self.assertEqual(rows[-1]["value"], 4e-6)
        self.assertEqual(metadata["path_axis_label"], "Path distance (mm)")
        self.assertEqual(metadata["value_unit"], "µT")
        self.assertEqual(rows[-1]["value_axis_label"], "Bz (µT)")


    def _configure_solver_backend_coil(self, *, turns: int = 100) -> None:
        self.adapter.set_coils_enabled(["coil_b"], False)
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "turns": turns,
                        "drive_current_a": 0.035,
                        "field_scale_factor": 1.0,
                        "physical_geometry_enabled": True,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 8.0,
                        "winding_axial_width_mm": 18.0,
                        "bobbin_wall_thickness_mm": 1.0,
                        "flange_height_mm": 10.0,
                        "flange_thickness_mm": 1.0,
                    },
                }
            ]
        )

    def test_auto_stage_one_heuristics_cover_exact_sheet_bundled_and_centreline(self):
        self.adapter.set_coil_modeling_method("auto")
        self.assertEqual(self.adapter.coil_modeling_method, "auto")

        circle_descriptor = {
            "kind": "circle",
            "diameter": 0.21,
            "path": np.empty((0, 3)),
            "position": np.zeros(3),
            "rotation": np.eye(3),
        }
        compact_dimensions = {
            "turns": 484,
            "outer_wire_m": 0.0003,
            "axial_width_m": 0.0064,
            "radial_build_m": 0.0064,
        }
        distant_points = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.02, 0.0]])
        compact = self.adapter._auto_coil_decision(
            circle_descriptor, compact_dimensions, distant_points
        )
        self.assertEqual(compact["selected"], "centreline")

        sparse_dimensions = {**compact_dimensions, "turns": 8}
        sparse = self.adapter._auto_coil_decision(
            circle_descriptor, sparse_dimensions, distant_points
        )
        self.assertEqual(sparse["selected"], "exact")

        ordinary_dimensions = {
            "turns": 320,
            "outer_wire_m": 0.00035,
            "axial_width_m": 0.022,
            "radial_build_m": 0.0021,
        }
        grid_axis = np.linspace(-0.04, 0.04, 160)
        grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
        ordinary_points = np.column_stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.zeros(grid_x.size)]
        )
        ordinary = self.adapter._auto_coil_decision(
            {**circle_descriptor, "diameter": 0.16},
            ordinary_dimensions,
            ordinary_points,
        )
        self.assertEqual(ordinary["selected"], "bundled")
        optimizer_choice = self.adapter._auto_coil_decision(
            {**circle_descriptor, "diameter": 0.16},
            ordinary_dimensions,
            [[0.0, 0.0, 0.0]],
            purpose="optimizer",
        )
        self.assertEqual(optimizer_choice["selected"], "bundled")
        self.assertEqual(optimizer_choice["metrics"]["purpose"], "optimizer")

        radius = 0.05
        straight = 0.10
        right_angle = np.linspace(-math.pi / 2.0, math.pi / 2.0, 33)
        left_angle = np.linspace(math.pi / 2.0, 3.0 * math.pi / 2.0, 33)
        racetrack_path = np.vstack(
            [
                np.column_stack(
                    [
                        straight / 2.0 + radius * np.cos(right_angle),
                        radius * np.sin(right_angle),
                        np.zeros_like(right_angle),
                    ]
                ),
                np.column_stack(
                    [
                        -straight / 2.0 + radius * np.cos(left_angle),
                        radius * np.sin(left_angle),
                        np.zeros_like(left_angle),
                    ]
                ),
            ]
        )
        racetrack = self.adapter._auto_coil_decision(
            {
                "kind": "polyline",
                "path": racetrack_path,
                "position": np.zeros(3),
                "rotation": np.eye(3),
            },
            {
                "turns": 222,
                "outer_wire_m": 0.00028,
                "axial_width_m": 0.065,
                "radial_build_m": 0.00028,
            },
            ordinary_points,
        )
        self.assertEqual(racetrack["selected"], "current_sheet")
        self.assertGreater(racetrack["metrics"]["exact_primitives"], 10_000)

    def test_auto_report_records_per_coil_choice_and_reason(self):
        self._configure_solver_backend_coil(turns=8)
        self.adapter.set_coil_modeling_method("auto")
        result = self.adapter.get_field(points=[[0.0, 0.0, 0.0]], field="B")
        entry = next(
            item for item in result["coil_modeling"]["coils"] if item["id"] == "coil_a"
        )
        self.assertEqual(result["coil_modeling"]["requested"], "auto")
        self.assertEqual(entry["actual"], "exact")
        self.assertEqual(entry["auto"]["selected"], "exact")
        self.assertTrue(entry["auto"]["reason"])
        self.assertEqual(result["coil_modeling"]["auto_counts"]["exact"], 1)

    def test_auto_stage_two_error_metrics_and_cheapest_passing_selection(self):
        reference = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ]
        )
        identical = self.adapter._auto_validation_error(reference, reference.copy())
        self.assertEqual(identical["rms_relative"], 0.0)
        self.assertEqual(identical["p95_relative"], 0.0)

        heuristic = {
            "selected": "current_sheet",
            "metrics": {
                "turns": 222,
                "point_count": 25_600,
                "path_segments": 66,
                "bundled_interactions": 54_067_200,
                "current_sheet_interactions": 3_379_200,
                "exact_interactions": 375_091_200,
            },
        }
        validation = {
            "status": "complete",
            "candidates": {
                "centreline": {
                    "available": True,
                    "rms_relative": 0.05,
                    "p95_relative": 0.08,
                },
                "bundled": {
                    "available": True,
                    "rms_relative": 0.004,
                    "p95_relative": 0.009,
                },
                "current_sheet": {
                    "available": True,
                    "rms_relative": 0.003,
                    "p95_relative": 0.007,
                },
            },
        }
        selected, reason, tolerance_met = self.adapter._auto_stage_two_selection(
            heuristic, validation
        )
        self.assertEqual(selected, "current_sheet")
        self.assertTrue(tolerance_met)
        self.assertIn("within tolerance", reason)

    def test_auto_stage_two_uses_lowest_error_when_tolerance_is_unavailable(self):
        heuristic = {
            "selected": "current_sheet",
            "metrics": {
                "turns": 222,
                "point_count": 25_600,
                "path_segments": 66,
                "bundled_interactions": 54_067_200,
                "current_sheet_interactions": 3_379_200,
                "exact_interactions": 375_091_200,
            },
        }
        validation = {
            "status": "complete",
            "candidates": {
                "centreline": {
                    "available": True,
                    "rms_relative": 0.12,
                    "p95_relative": 0.20,
                },
                "bundled": {
                    "available": True,
                    "rms_relative": 0.03,
                    "p95_relative": 0.045,
                },
                "current_sheet": {
                    "available": True,
                    "rms_relative": 0.04,
                    "p95_relative": 0.06,
                },
            },
        }
        selected, reason, tolerance_met = self.adapter._auto_stage_two_selection(
            heuristic, validation
        )
        self.assertEqual(selected, "bundled")
        self.assertFalse(tolerance_met)
        self.assertIn("lowest-error", reason)

    def test_auto_stage_two_skips_tiny_observer_sets(self):
        descriptor = {
            "kind": "circle",
            "current": 1.0,
            "diameter": 0.2,
            "path": np.empty((0, 3)),
            "position": np.zeros(3),
            "rotation": np.eye(3),
        }
        dimensions = {
            "turns": 100,
            "outer_wire_m": 0.0003,
            "axial_width_m": 0.006,
            "radial_build_m": 0.006,
        }
        heuristic = self.adapter._auto_coil_decision(
            descriptor, dimensions, [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]
        )
        validation = self.adapter._auto_stage_two_validation(
            descriptor,
            dimensions,
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            heuristic,
        )
        self.assertEqual(validation["status"], "skipped")
        self.assertIn("at least", validation["reason"])

    def test_auto_stage_two_cache_key_reuses_same_region_at_new_resolution(self):
        angle = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
        descriptor = {
            "kind": "circle",
            "current": 1.0,
            "diameter": 0.2,
            "path": np.column_stack(
                [0.1 * np.cos(angle), 0.1 * np.sin(angle), np.zeros_like(angle)]
            ),
            "position": np.zeros(3),
            "rotation": np.eye(3),
        }
        dimensions = {
            "turns": 100,
            "outer_wire_m": 0.0003,
            "axial_width_m": 0.01,
            "radial_build_m": 0.005,
        }

        def grid(resolution):
            axis = np.linspace(-0.08, 0.08, resolution)
            grid_x, grid_y = np.meshgrid(axis, axis)
            return np.column_stack(
                [grid_x.reshape(-1), grid_y.reshape(-1), np.zeros(grid_x.size)]
            )

        low_resolution = self.adapter._auto_validation_cache_key(
            descriptor, dimensions, grid(31)
        )
        high_resolution = self.adapter._auto_validation_cache_key(
            descriptor, dimensions, grid(101)
        )
        self.assertEqual(low_resolution, high_resolution)

    def test_solver_memory_budget_and_source_partitioning(self):
        self.adapter.set_solver_memory_budget_mb(128)
        self.assertEqual(self.adapter.solver_memory_budget_mb, 128)
        with self.assertRaises(StudioOperationError):
            self.adapter.set_solver_memory_budget_mb(1)

        class FakePolyline:
            def __init__(self, segment_count):
                self.vertices = np.zeros((segment_count + 1, 3), dtype=float)

        sources = [FakePolyline(10) for _ in range(130)]
        self.assertEqual(self.adapter._magpylib_source_complexity(sources[0]), 10)
        batches, complexities = self.adapter._partition_magpylib_sources(
            sources, maximum_interactions=500
        )
        self.assertEqual([len(batch) for batch in batches], [50, 50, 30])
        self.assertEqual(complexities, [500, 500, 300])
        self.assertTrue(all(len(batch) <= 64 for batch in batches))

    def test_detailed_solver_progress_counts_scene_subtraction_and_model_passes(self):
        self._configure_solver_backend_coil(turns=8)
        self.adapter.set_coil_modeling_method("exact")
        updates = []
        points = np.zeros((5, 3), dtype=float)
        self.adapter.get_field(
            points=points,
            field="B",
            progress_callback=updates.append,
            progress_label="Detailed test",
        )

        self.assertEqual(updates[0]["total"], 15)
        self.assertEqual(updates[-1]["completed"], 15)
        stages = {update["stage"] for update in updates}
        self.assertTrue(any("scene field" in stage for stage in stages))
        self.assertTrue(any("subtracting centreline" in stage for stage in stages))
        self.assertTrue(any("Exact turns" in stage for stage in stages))

    def test_solver_backend_exact_bundled_sheet_and_centreline_share_one_entry_point(self):
        self._configure_solver_backend_coil(turns=100)
        point = [[0.024, 0.011, 0.018]]

        self.adapter.set_coil_modeling_method("centreline")
        centreline = np.asarray(self.adapter.get_field(points=point, field="B")["values"])
        self.assertEqual(
            self.adapter.coil_modeling_report()["coils"][0]["actual"], "centreline"
        )

        self.adapter.set_coil_modeling_method("exact")
        exact_result = self.adapter.get_field(points=point, field="B")
        exact = np.asarray(exact_result["values"])
        exact_entry = next(
            item for item in exact_result["coil_modeling"]["coils"] if item["id"] == "coil_a"
        )
        self.assertEqual(exact_entry["actual"], "exact")
        self.assertEqual(exact_entry["source_count"], 100)
        self.assertFalse(np.allclose(exact, centreline, rtol=1e-12, atol=1e-18))

        self.adapter.set_coil_modeling_method("bundled")
        bundled_result = self.adapter.get_field(points=point, field="B")
        bundled_entry = next(
            item for item in bundled_result["coil_modeling"]["coils"] if item["id"] == "coil_a"
        )
        self.assertEqual(bundled_entry["actual"], "bundled")
        self.assertEqual(bundled_entry["source_count"], 32)

        self.adapter.set_coil_modeling_method("current_sheet")
        sheet_result = self.adapter.get_field(points=point, field="B")
        sheet_entry = next(
            item for item in sheet_result["coil_modeling"]["coils"] if item["id"] == "coil_a"
        )
        self.assertEqual(sheet_entry["actual"], "current_sheet")
        self.assertGreaterEqual(sheet_entry["source_count"], 1)
        self.assertLessEqual(sheet_entry["source_count"], 8)
        self.assertTrue(np.isfinite(np.asarray(sheet_result["values"], dtype=float)).all())

    def test_racetrack_coil_uses_exact_and_current_sheet_backends(self):
        self.adapter.set_coils_enabled(["coil_a", "coil_b"], False)
        racetrack_id = self.adapter.add_template("racetrack")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": racetrack_id,
                        "turns": 24,
                        "drive_current_a": 0.05,
                        "physical_geometry_enabled": True,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 1.0,
                        "winding_axial_width_mm": 12.0,
                    },
                }
            ]
        )
        self.adapter.set_coil_modeling_method("exact")
        exact = self.adapter.get_field(points=[[0.0, 0.0, 0.03]], field="B")
        exact_entry = next(
            item for item in exact["coil_modeling"]["coils"] if item["id"] == racetrack_id
        )
        self.assertEqual(exact_entry["actual"], "exact")
        self.assertEqual(exact_entry["source_count"], 24)

        self.adapter.set_coil_modeling_method("current_sheet")
        sheet = self.adapter.get_field(points=[[0.0, 0.0, 0.03]], field="B")
        sheet_entry = next(
            item for item in sheet["coil_modeling"]["coils"] if item["id"] == racetrack_id
        )
        self.assertEqual(sheet_entry["actual"], "current_sheet")
        self.assertGreaterEqual(sheet_entry["source_count"], 1)
        self.assertTrue(np.isfinite(np.asarray(sheet["values"], dtype=float)).all())

    def test_solver_backend_falls_back_per_coil_when_physical_dimensions_are_missing(self):
        self.adapter.set_coil_modeling_method("centreline")
        reference = np.asarray(
            self.adapter.get_field(points=[[0.01, 0.0, 0.02]], field="B")["values"]
        )
        self.adapter.set_coil_modeling_method("exact")
        result = self.adapter.get_field(points=[[0.01, 0.0, 0.02]], field="B")
        np.testing.assert_allclose(np.asarray(result["values"]), reference, rtol=0.0, atol=0.0)
        self.assertTrue(
            all(item["actual"] == "centreline" for item in result["coil_modeling"]["coils"])
        )
        self.assertTrue(
            all("fallback_reason" in item for item in result["coil_modeling"]["coils"])
        )

    def test_solver_backend_applies_base_spec_to_derived_pattern_copies(self):
        self._configure_solver_backend_coil(turns=12)
        original_list_objects = self.adapter.session.list_objects
        original_get_params = self.adapter.session.get_params
        original_get_transform = self.adapter.session.get_transform
        original_get_field = self.adapter.session.get_field
        base_object = next(
            item for item in original_list_objects() if item["id"] == "coil_a"
        )
        derived_object = copy.deepcopy(base_object)
        derived_object.update(
            {
                "id": "coil_a_pattern_1",
                "label": "Coil A pattern 1",
                "derived": "coil_a",
            }
        )
        derived_transform = copy.deepcopy(original_get_transform("coil_a"))
        derived_transform["position"] = [0.075, 0.0, 0.0]

        def list_objects_with_copy():
            return original_list_objects() + [copy.deepcopy(derived_object)]

        def params_with_copy(object_id):
            if object_id == derived_object["id"]:
                return copy.deepcopy(original_get_params("coil_a"))
            return original_get_params(object_id)

        def transform_with_copy(object_id):
            if object_id == derived_object["id"]:
                return copy.deepcopy(derived_transform)
            return original_get_transform(object_id)

        def field_with_copy(*, points, field="B"):
            result = copy.deepcopy(original_get_field(points=points, field=field))
            point_array = np.asarray(points, dtype=float).reshape(-1, 3)
            copy_source = self.adapter._coil_centreline_source(derived_object["id"])
            copy_values = self.adapter._evaluate_magpylib_sources(
                [copy_source], point_array, field
            )
            values = np.asarray(result["values"], dtype=float).reshape(-1, 3)
            values = values + copy_values
            result["values"] = values.tolist()
            result["magnitude"] = np.linalg.norm(values, axis=1).tolist()
            return result

        with (
            mock.patch.object(
                self.adapter.session, "list_objects", side_effect=list_objects_with_copy
            ),
            mock.patch.object(
                self.adapter.session, "get_params", side_effect=params_with_copy
            ),
            mock.patch.object(
                self.adapter.session, "get_transform", side_effect=transform_with_copy
            ),
            mock.patch.object(
                self.adapter.session, "get_field", side_effect=field_with_copy
            ),
        ):
            self.adapter.set_coil_modeling_method("exact")
            result = self.adapter.get_field(points=[[0.01, 0.0, 0.02]], field="B")

        entry = next(
            item
            for item in result["coil_modeling"]["coils"]
            if item["id"] == derived_object["id"]
        )
        self.assertTrue(entry["derived"])
        self.assertEqual(entry["source_id"], "coil_a")
        self.assertEqual(entry["actual"], "exact")
        self.assertEqual(entry["source_count"], 12)

    def test_solver_backend_applies_source_spec_to_collection_pattern_descendants(self):
        self._configure_solver_backend_coil(turns=12)
        point = np.asarray([[0.01, 0.0, 0.02]], dtype=float)
        original_get_field = self.adapter.session.get_field
        inherited_copy = self.adapter.session._objs["coil_a"].copy()
        inherited_copy.move([0.075, 0.0, 0.0])

        def field_with_inherited(*, points, field="B"):
            result = copy.deepcopy(original_get_field(points=points, field=field))
            point_array = np.asarray(points, dtype=float).reshape(-1, 3)
            copy_values = self.adapter._evaluate_magpylib_sources(
                [inherited_copy], point_array, field
            )
            values = np.asarray(result["values"], dtype=float).reshape(-1, 3)
            values = values + copy_values
            result["values"] = values.tolist()
            result["magnitude"] = np.linalg.norm(values, axis=1).tolist()
            return result

        base_compiled, _, _ = self.adapter._modeled_coil_sources(
            "coil_a", "exact", spec_object_id="coil_a"
        )
        copy_compiled, _, _ = self.adapter._modeled_inherited_coil_sources(
            inherited_copy, "exact", spec_object_id="coil_a"
        )
        baseline = np.asarray(field_with_inherited(points=point, field="B")["values"])
        removed = self.adapter._evaluate_magpylib_sources(
            [self.adapter._coil_centreline_source("coil_a"), inherited_copy], point, "B"
        )
        added = self.adapter._evaluate_magpylib_sources(
            [*base_compiled, *copy_compiled], point, "B"
        )
        expected = baseline - removed + added

        with (
            mock.patch.object(
                self.adapter.session,
                "_inherited",
                {"coil_a": [inherited_copy]},
            ),
            mock.patch.object(
                self.adapter.session, "get_field", side_effect=field_with_inherited
            ),
        ):
            self.adapter.set_coil_modeling_method("exact")
            result = self.adapter.get_field(points=point, field="B")

        np.testing.assert_allclose(np.asarray(result["values"]), expected)
        entry = next(
            item
            for item in result["coil_modeling"]["coils"]
            if item.get("inherited")
        )
        self.assertTrue(entry["derived"])
        self.assertEqual(entry["source_id"], "coil_a")
        self.assertEqual(entry["actual"], "exact")
        self.assertEqual(entry["source_count"], 12)

    def test_background_field_is_uniform_persistent_and_visibility_controlled(self):
        adapter = StudioAdapter({"objects": []})
        object_id = adapter.add_background_field(
            label="Ambient test field",
            vector_uT=[50.0, -20.0, 10.0],
            mode="manual",
            source={"kind": "manual"},
        )
        self.assertTrue(adapter.is_background_field(object_id))
        document = adapter.document()
        saved = document["field_workbench"]["background_fields"][0]
        self.assertEqual(saved["label"], "Ambient test field 1")
        self.assertEqual(saved["vector_uT"], [50.0, -20.0, 10.0])
        self.assertEqual(saved["display_position_m"], [0.0, 0.0, 0.0])

        adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": object_id,
                        "position": [0.125, -0.050, 0.025],
                    },
                }
            ]
        )
        self.assertEqual(
            adapter.get_transform(object_id)["position"],
            [0.125, -0.05, 0.025],
        )
        self.assertEqual(
            adapter.background_field_properties(object_id)["display_position_m"],
            [0.125, -0.05, 0.025],
        )

        points = np.asarray([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
        zero_result = {"values": np.zeros((2, 3)).tolist(), "magnitude": [0.0, 0.0]}
        with mock.patch.object(adapter, "_evaluate_studio_field", return_value=zero_result):
            result = adapter.get_field(points=points, field="B")
        expected = np.asarray([[50.0e-6, -20.0e-6, 10.0e-6]] * 2)
        np.testing.assert_allclose(np.asarray(result["values"]), expected, rtol=0, atol=1e-15)
        self.assertEqual(result["coil_modeling"]["background_fields"][0]["id"], object_id)

        glyphs = adapter._background_field_traces()
        self.assertEqual(len(glyphs), 3)
        self.assertEqual(
            glyphs[0]["meta"]["field_workbench_object_id"], object_id
        )
        self.assertEqual(glyphs[0]["mode"], "lines")
        self.assertEqual(glyphs[1]["mode"], "markers")
        self.assertEqual(glyphs[2]["mode"], "lines")
        self.assertEqual(glyphs[1]["marker"]["size"], 4)
        self.assertEqual(glyphs[0]["x"][0], 0.125)
        self.assertEqual(glyphs[0]["y"][0], -0.05)
        self.assertEqual(glyphs[0]["z"][0], 0.025)
        self.assertEqual(len(glyphs[1]["x"]), 1)
        self.assertEqual(glyphs[1]["x"][0], 0.125)
        self.assertEqual(glyphs[1]["y"][0], -0.05)
        self.assertEqual(glyphs[1]["z"][0], 0.025)

        document = adapter.document()
        adapter.set_visible(object_id, False)
        with mock.patch.object(adapter, "_evaluate_studio_field", return_value=zero_result):
            hidden = adapter.get_field(points=points, field="B")
        np.testing.assert_allclose(np.asarray(hidden["values"]), 0.0, rtol=0, atol=1e-15)

        reopened = StudioAdapter(document)
        self.assertEqual(
            reopened.background_field_properties(object_id)["vector_uT"],
            [50.0, -20.0, 10.0],
        )
        self.assertEqual(
            reopened.get_transform(object_id)["position"],
            [0.125, -0.05, 0.025],
        )

    def test_background_field_h_conversion_uses_mu0(self):
        adapter = StudioAdapter({"objects": []})
        adapter.add_background_field(vector_uT=[1.0, 0.0, 0.0])
        zero_result = {"values": [[0.0, 0.0, 0.0]], "magnitude": [0.0]}
        with mock.patch.object(adapter, "_evaluate_studio_field", return_value=zero_result):
            result = adapter.get_field(points=[[0.0, 0.0, 0.0]], field="H")
        expected = 1.0e-6 / (4.0e-7 * math.pi)
        self.assertAlmostEqual(float(result["values"][0][0]), expected, places=12)

    def test_measurement_fingerprint_tracks_effective_background_field(self):
        measurement_id = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": measurement_id, "role": "measurement"},
                }
            ]
        )
        baseline = self.adapter.measurement_fingerprint(measurement_id)
        background_id = self.adapter.add_background_field(vector_uT=[25.0, 0.0, -5.0])
        active = self.adapter.measurement_fingerprint(measurement_id)
        self.assertNotEqual(active, baseline)

        self.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": background_id,
                        "position": [0.2, -0.1, 0.05],
                    },
                }
            ]
        )
        self.assertEqual(
            self.adapter.measurement_fingerprint(measurement_id), active
        )

        self.adapter.set_visible(background_id, False)
        hidden = self.adapter.measurement_fingerprint(measurement_id)
        self.assertEqual(hidden, baseline)

        self.adapter.set_visible(background_id, True)
        replay = self.adapter._snapshot_field_scene()
        replay_backgrounds = replay["field_workbench"]["background_fields"]
        self.assertEqual(len(replay_backgrounds), 1)
        self.assertEqual(replay_backgrounds[0]["vector_uT"], [25.0, 0.0, -5.0])
        self.assertIsNone(replay_backgrounds[0]["parent"])
        self.assertTrue(replay_backgrounds[0]["visible"])

    def test_maps_sensors_and_measurement_fingerprints_use_selected_solver(self):
        self._configure_solver_backend_coil(turns=40)
        self.adapter.set_coil_modeling_method("bundled")
        self.adapter.field_map(
            plane="xy",
            offset_mm=0.0,
            span_mm=100.0,
            resolution=8,
            field="B",
            component="magnitude",
        )
        self.assertEqual(self.adapter.coil_modeling_report()["requested"], "bundled")
        self.adapter.sensor_plot("Bz")
        self.assertEqual(self.adapter.coil_modeling_report()["requested"], "bundled")

        measurement_id = self.adapter.add_template("guide_sphere")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": measurement_id, "role": "measurement"},
                }
            ]
        )
        bundled_fingerprint = self.adapter.measurement_fingerprint(measurement_id)
        self.adapter.set_coil_modeling_method("centreline")
        centreline_fingerprint = self.adapter.measurement_fingerprint(measurement_id)
        self.assertNotEqual(bundled_fingerprint, centreline_fingerprint)

    def test_builtin_fgm3d_definition_preserves_manufacturer_reference_geometry(self):
        definition = builtin_probe_definitions()[0]
        self.assertEqual(definition["manufacturer"], "SENSYS GmbH")
        self.assertEqual(definition["model"], "FGM3D")
        self.assertEqual(definition["housing"]["dimensions_mm"], [26.0, 26.0, 149.0])
        self.assertEqual(definition["housing"]["center_mm"], [0.0, 0.0, 0.0])
        self.assertEqual(definition["housing"]["sticker_face"], "-Y")
        self.assertEqual(definition["housing"]["cable_face"], "-Z")
        channels = {item["name"]: item for item in definition["channels"]}
        self.assertEqual(channels["X"]["reference_from_edge_mm"], 54.5)
        self.assertEqual(channels["Y"]["reference_from_edge_mm"], 14.5)
        self.assertEqual(channels["Z"]["reference_from_edge_mm"], 34.5)
        self.assertEqual(channels["X"]["position_mm"], [0.0, 0.0, 20.0])
        self.assertEqual(channels["Y"]["position_mm"], [0.0, 0.0, 60.0])
        self.assertEqual(channels["Z"]["position_mm"], [0.0, 0.0, 40.0])
        for channel in channels.values():
            self.assertAlmostEqual(
                74.5 - channel["position_mm"][2],
                channel["reference_from_edge_mm"],
            )
        self.assertTrue(all(source["url"].startswith("https://") for source in definition["sources"]))
        invalid = copy.deepcopy(definition)
        invalid["housing"]["cable_face"] = "connector end"
        with self.assertRaises(ProbeDefinitionError):
            validate_probe_definition(invalid)

    def test_builtin_meda_fm300_definition_preserves_fvm400_reference_geometry(self):
        definition = next(
            item
            for item in builtin_probe_definitions()
            if item["id"] == "meda-fm300-fvm400-shared-probe-v1"
        )
        self.assertEqual(definition["manufacturer"], "MEDA, Inc.")
        self.assertEqual(definition["model"], "FM300 / FVM400")
        self.assertEqual(definition["housing"]["dimensions_mm"], [101.6, 25.4, 25.4])
        self.assertEqual(definition["housing"]["center_mm"], [0.0, 0.0, 0.0])
        self.assertEqual(definition["housing"]["sticker_face"], "-Z")
        self.assertEqual(definition["housing"]["cable_face"], "-X")

        channels = {item["name"]: item for item in definition["channels"]}
        self.assertEqual(channels["X"]["position_mm"], [11.684, 0.0, -0.254])
        self.assertEqual(channels["Y"]["position_mm"], [33.528, 0.0, -0.254])
        self.assertEqual(channels["Z"]["position_mm"], channels["Y"]["position_mm"])
        self.assertEqual(channels["X"]["sensitive_axis"], [1.0, 0.0, 0.0])
        self.assertEqual(channels["Y"]["sensitive_axis"], [0.0, 1.0, 0.0])
        self.assertEqual(channels["Z"]["sensitive_axis"], [0.0, 0.0, 1.0])
        for channel in channels.values():
            self.assertAlmostEqual(
                50.8 - channel["position_mm"][0],
                channel["reference_from_edge_mm"],
            )
            self.assertAlmostEqual(channel["position_mm"][2] + 12.7, 12.446)

        source_urls = {source["url"] for source in definition["sources"]}
        self.assertIn(
            "https://www.meda.com/index.php/products?name=FVM400",
            source_urls,
        )
        self.assertIn(
            "https://www.meda.com/pdf/FVM400%20Instruction%20Manual%20Rev%20A.pdf",
            source_urls,
        )
        self.assertIn("https://www.meda.com/index.php/pages/about", source_urls)

        adapter = StudioAdapter({"objects": []})
        probe_id = adapter.add_magnetometer_probe(definition)
        adapter.add_background_field(vector_uT=[10.0, 20.0, 30.0])
        reading = adapter.magnetometer_probe_reading(probe_id)
        np.testing.assert_allclose(
            np.asarray(reading["reported_vector"]) * 1e6,
            [10.0, 20.0, 30.0],
            atol=1e-10,
        )
        traces = adapter._magnetometer_probe_traces()
        cable_trace = next(
            trace
            for trace in traces
            if trace.get("meta", {}).get("field_workbench_probe_cable_end")
        )
        sticker_trace = next(
            trace
            for trace in traces
            if trace.get("meta", {}).get("field_workbench_probe_sticker")
        )
        self.assertAlmostEqual(float(np.mean(cable_trace["x"])), -0.05092)
        self.assertAlmostEqual(float(np.mean(sticker_trace["z"])), -0.01282)

    def test_magnetometer_probe_reads_separate_points_and_local_axes_and_round_trips(self):
        adapter = StudioAdapter({"objects": []})
        probe_id = adapter.add_magnetometer_probe(builtin_probe_definitions()[0])
        adapter.add_background_field(vector_uT=[10.0, 20.0, 30.0])

        geometry = {item["name"]: item for item in adapter.magnetometer_probe_sample_geometry(probe_id)}
        self.assertEqual(geometry["X"]["world_position_mm"], [0.0, 0.0, 20.0])
        self.assertEqual(geometry["Y"]["world_position_mm"], [0.0, 0.0, 60.0])
        self.assertEqual(geometry["Z"]["world_position_mm"], [0.0, 0.0, 40.0])
        reading = adapter.magnetometer_probe_reading(probe_id)
        np.testing.assert_allclose(
            np.asarray(reading["reported_vector"]) * 1e6,
            [10.0, 20.0, 30.0],
            atol=1e-10,
        )

        adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": probe_id,
                        "position": [0.1, -0.2, 0.3],
                        "orientation": [0.0, 0.0, 90.0],
                    },
                }
            ]
        )
        rotated = adapter.magnetometer_probe_reading(probe_id)
        np.testing.assert_allclose(
            np.asarray(rotated["reported_vector"]) * 1e6,
            [20.0, -10.0, 30.0],
            atol=1e-9,
        )

        document = adapter.document()
        self.assertEqual(document["field_workbench"]["version"], WORKBENCH_METADATA_VERSION)
        self.assertEqual(len(document["field_workbench"]["magnetometer_probes"]), 1)
        reopened = StudioAdapter(document)
        self.assertTrue(reopened.is_magnetometer_probe(probe_id))
        np.testing.assert_allclose(
            reopened.get_transform(probe_id)["position"], [0.1, -0.2, 0.3]
        )
        self.assertEqual(
            reopened.magnetometer_probe_properties(probe_id)["definition"]["id"],
            "sensys-fgm3d-standard-v3",
        )

    def test_magnetometer_probe_visuals_clone_group_visibility_and_portable_export(self):
        adapter = StudioAdapter({"objects": []})
        probe_id = adapter.add_magnetometer_probe(builtin_probe_definitions()[0])
        traces = adapter._magnetometer_probe_traces()
        self.assertEqual(
            [trace["type"] for trace in traces],
            [
                "mesh3d",
                "mesh3d",
                "mesh3d",
                "scatter3d",
                "scatter3d",
                "scatter3d",
                "scatter3d",
                "scatter3d",
                "scatter3d",
                "scatter3d",
            ],
        )
        cable_trace = next(
            trace
            for trace in traces
            if trace.get("meta", {}).get("field_workbench_probe_cable_end")
        )
        self.assertEqual(cable_trace["type"], "mesh3d")
        self.assertEqual(cable_trace["color"], "#dc2626")
        self.assertAlmostEqual(float(np.mean(cable_trace["x"])), 0.0)
        self.assertAlmostEqual(float(np.mean(cable_trace["y"])), 0.0)
        self.assertAlmostEqual(float(np.mean(cable_trace["z"])), -0.07462)
        sticker_traces = [
            trace
            for trace in traces
            if trace.get("meta", {}).get("field_workbench_probe_sticker")
        ]
        self.assertEqual(len(sticker_traces), 1)
        sticker_trace = sticker_traces[0]
        self.assertEqual(sticker_trace["type"], "mesh3d")
        self.assertTrue(sticker_trace["name"].endswith("sticker side"))
        self.assertAlmostEqual(float(np.mean(sticker_trace["x"])), 0.0)
        self.assertAlmostEqual(float(np.mean(sticker_trace["y"])), -0.01312)
        self.assertAlmostEqual(float(np.mean(sticker_trace["z"])), 0.0)
        point_trace = next(
            trace for trace in traces if trace.get("mode") == "markers+text"
        )
        self.assertEqual(point_trace["text"], ["X point", "Y point", "Z point"])
        arrowheads = [
            trace
            for trace in traces
            if trace.get("meta", {}).get("field_workbench_probe_arrowhead")
        ]
        self.assertEqual(len(arrowheads), 3)
        for arrowhead in arrowheads:
            channel = arrowhead["meta"]["field_workbench_probe_channel_axis"]
            shaft = next(
                trace
                for trace in traces
                if trace.get("meta", {}).get("field_workbench_probe_channel_axis") == channel
                and not trace.get("meta", {}).get("field_workbench_probe_arrowhead")
            )
            self.assertEqual(arrowhead["mode"], "lines")
            self.assertEqual(len(arrowhead["x"]), 12)
            self.assertEqual(
                [arrowhead[axis][0] for axis in ("x", "y", "z")],
                [shaft[axis][-1] for axis in ("x", "y", "z")],
            )

        clone_id = adapter.duplicate(probe_id)
        self.assertTrue(adapter.is_magnetometer_probe(clone_id))
        group_id = adapter.group_objects([probe_id, clone_id])
        self.assertEqual(adapter._selection_traces(group_id), [])
        adapter.set_visible(group_id, False)
        self.assertEqual(adapter._magnetometer_probe_traces(), [])
        adapter.set_visible(group_id, True)

        portable = build_portable_scene(adapter, [probe_id])
        record = portable["objects"][0]
        self.assertEqual(record["kind"], "magnetometer_probe")
        self.assertEqual(record["definition"]["model"], "FGM3D")
        self.assertEqual(len(record["channel_geometry"]), 3)


if __name__ == "__main__":
    unittest.main()
