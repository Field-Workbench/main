from __future__ import annotations

import base64
import math
import unittest

import numpy as np

from fieldworkbench.brain_analysis import sample_lpba40_voxels
from fieldworkbench.studio_adapter import StudioAdapter
from fieldworkbench.waveform import (
    build_2d_animation,
    build_2d_gpu_playback,
    build_3d_gpu_playback,
    build_3d_gpu_static,
    analysis_sample_count,
    current_at_times,
    dac_8bit_voltage_trace,
    dac_file_is_text,
    drive_at_times,
    drive_coil_catalog,
    field_basis_for_drive,
    field_bases_for_drives,
    lpba40_region_hierarchy,
    _brain_playback_run_metadata,
    _gpu_scene_payload,
    normalize_coil_sequence,
    sequenced_drive_trace,
    playback_frame_count,
    playback_real_time_multiplier,
    parse_dac_8bit_values,
    time_statistics,
    validate_animation_field_basis,
    validate_trace,
)
from workflow_fixture import compact_workflow_scene


class WaveformTests(unittest.TestCase):
    def test_brain_frame_metric_cache_hierarchy_helper_is_available(self):
        self.assertTrue(callable(lpba40_region_hierarchy))

    def test_brain_playback_metadata_counts_cycles_sequence_and_sampling(self):
        trace = validate_trace([0.0, 1.0], [0.0, 0.0], kind="current")
        metadata = _brain_playback_run_metadata(
            trace,
            frame_count=61,
            analysis_sample_count=361,
            summary_sample_count=61,
            atlas_sample_count=504,
            waveform_metadata={
                "mode": "sine",
                "frequency_hz": 3.0,
                "cycle_count": 3.0,
                "pass_count": 1,
            },
            sequence_steps=[
                {"duration_s": 0.1, "active_coil_ids": ["coil_a"]},
                {"duration_s": 0.15, "active_coil_ids": ["coil_b"]},
            ],
            sequence_loop=True,
        )
        self.assertAlmostEqual(metadata["exposure_time_s"], 1.0)
        self.assertEqual(metadata["waveform_cycle_count"], 3.0)
        self.assertEqual(metadata["sequencer_step_count"], 2)
        self.assertEqual(metadata["sequencer_full_loops"], 4)
        self.assertAlmostEqual(metadata["sequencer_equivalent_loops"], 4.0)
        self.assertEqual(metadata["sequencer_step_activations"], 8)
        self.assertEqual(metadata["frame_count"], 61)
        self.assertEqual(metadata["analysis_sample_count"], 361)
        self.assertEqual(metadata["regional_frame_sample_count"], 61)
        self.assertEqual(metadata["atlas_sample_count"], 504)

    def test_binary_dac_import_maps_codes_and_appends_zero_cycle_delay(self):
        codes = parse_dac_8bit_values(bytes([0, 128, 255]))
        trace = dac_8bit_voltage_trace(
            codes,
            point_delay_ms=3.0,
            cycle_delay_ms=3.0,
            lower_voltage_v=-5.0,
            upper_voltage_v=5.0,
        )
        np.testing.assert_allclose(trace.time_s, [0.0, 0.003, 0.006, 0.009, 0.012], atol=1e-15)
        self.assertAlmostEqual(trace.value[0], -5.0, places=12)
        self.assertAlmostEqual(trace.value[1], -5.0 + 128.0 * 10.0 / 255.0, places=12)
        self.assertAlmostEqual(trace.value[2], 5.0, places=12)
        self.assertEqual(trace.value[-2], 0.0)
        self.assertEqual(trace.value[-1], 0.0)

    def test_text_dac_import_accepts_decimal_and_hex_values(self):
        codes = parse_dac_8bit_values(b"0, 0x80, 255\n", text_hint=True)
        np.testing.assert_array_equal(codes, [0, 128, 255])

    def test_legacy_dac_extension_uses_text_parser(self):
        self.assertTrue(dac_file_is_text("CRING.DAC"))
        self.assertTrue(dac_file_is_text("waveform.txt"))
        self.assertFalse(dac_file_is_text("waveform.bin"))

    def test_dac_import_discards_matching_leading_point_count(self):
        text_codes = parse_dac_8bit_values(b"3, 0, 0x80, 255\n", text_hint=True)
        binary_codes = parse_dac_8bit_values(bytes([3, 0, 128, 255]))
        np.testing.assert_array_equal(text_codes, [0, 128, 255])
        np.testing.assert_array_equal(binary_codes, [0, 128, 255])

    def test_dac_import_keeps_nonmatching_first_sample(self):
        text_codes = parse_dac_8bit_values(b"2, 0, 0x80, 255\n", text_hint=True)
        binary_codes = parse_dac_8bit_values(bytes([2, 0, 128, 255]))
        np.testing.assert_array_equal(text_codes, [2, 0, 128, 255])
        np.testing.assert_array_equal(binary_codes, [2, 0, 128, 255])

    def test_current_trace_is_dac_sample_and_hold_by_default(self):
        trace = validate_trace([0.0, 1.0, 2.0], [0.0, 2.0, 0.0], kind="current")
        values = current_at_times(trace, [0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(values, [0.0, 2.0, 2.0, 0.0], rtol=0.0, atol=1e-12)

    def test_initial_drive_value_seeds_finite_slew(self):
        trace = validate_trace([0.0, 1.0, 2.0], [2.0, 2.0, 2.0], kind="current")
        values = drive_at_times(
            trace, [0.0, 0.25, 0.5, 1.0], slew_rate_per_s=4.0, initial_value=0.0
        )
        np.testing.assert_allclose(values, [0.0, 1.0, 2.0, 2.0], rtol=0.0, atol=1e-12)

    def test_dac_slew_rate_limits_each_command_transition(self):
        trace = validate_trace([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 0.0, 0.0], kind="current")
        values = drive_at_times(trace, [0.5, 1.0, 1.25, 1.5, 1.75, 2.25, 2.5], slew_rate_per_s=4.0)
        np.testing.assert_allclose(values, [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0], rtol=0.0, atol=1e-12)

    def test_constant_voltage_rl_solution_matches_closed_form(self):
        trace = validate_trace([0.0, 1.0], [1.0, 1.0], kind="voltage")
        values = current_at_times(
            trace,
            [0.0, 0.5, 1.0],
            resistance_ohm=1.0,
            inductance_h=1.0,
            initial_current_a=0.0,
        )
        self.assertAlmostEqual(values[-1], 1.0 - math.exp(-1.0), places=12)

    def test_slew_limited_voltage_pure_inductor_integrates_exactly(self):
        trace = validate_trace([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], kind="voltage")
        value = current_at_times(
            trace,
            [2.0],
            resistance_ohm=0.0,
            inductance_h=2.0,
            initial_current_a=0.0,
            slew_rate_per_s=2.0,
        )[0]
        self.assertAlmostEqual(value, 0.375, places=12)


    def test_looping_coil_sequencer_gates_shared_drive_trace_by_step(self):
        trace = validate_trace([0.0, 0.25, 0.5], [1.0, 1.0, 1.0], kind="current")
        steps = normalize_coil_sequence(
            [
                {"duration_s": 0.05, "active_coil_ids": ["a", "b"]},
                {"duration_s": 0.10, "active_coil_ids": ["c", "d"]},
            ],
            available_coil_ids=["a", "b", "c", "d"],
        )
        a_trace = sequenced_drive_trace(trace, coil_id="a", steps=steps, loop=True)
        c_trace = sequenced_drive_trace(trace, coil_id="c", steps=steps, loop=True)
        sample_times = [0.025, 0.05, 0.125, 0.15, 0.175]
        np.testing.assert_allclose(
            drive_at_times(a_trace, sample_times), [1.0, 0.0, 0.0, 1.0, 1.0], atol=1e-12
        )
        np.testing.assert_allclose(
            drive_at_times(c_trace, sample_times), [0.0, 1.0, 1.0, 0.0, 0.0], atol=1e-12
        )

    def test_nonlooping_coil_sequencer_turns_all_drives_off_after_last_step(self):
        trace = validate_trace([0.0, 1.0], [2.0, 2.0], kind="current")
        steps = normalize_coil_sequence(
            [{"duration_s": 0.1, "active_coil_ids": ["a"]}],
            available_coil_ids=["a"],
        )
        gated = sequenced_drive_trace(trace, coil_id="a", steps=steps, loop=False)
        np.testing.assert_allclose(
            drive_at_times(gated, [0.05, 0.1, 0.5]), [2.0, 0.0, 0.0], atol=1e-12
        )

    def test_time_statistics_are_time_weighted(self):
        stats = time_statistics([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(stats["mean"], 0.5, places=12)
        self.assertAlmostEqual(stats["rms"], math.sqrt(0.5), places=12)
        self.assertAlmostEqual(stats["pk_pk"], 1.0, places=12)


    def test_analysis_sample_multiplier_preserves_timeline_endpoints(self):
        self.assertEqual(analysis_sample_count(61, 6), 361)
        self.assertEqual(analysis_sample_count(121, 3), 361)

    def test_playback_fps_and_multiplier_choose_matching_frame_count(self):
        self.assertEqual(playback_frame_count(1.0, 60.0, 1.0), 61)
        self.assertEqual(playback_frame_count(1.0, 60.0, 0.5), 121)
        self.assertAlmostEqual(playback_real_time_multiplier(1.0, 121, 60.0), 0.5, places=12)


    def test_drive_coil_catalog_includes_scene_disabled_coils(self):
        scene = compact_workflow_scene()
        scene["field_workbench"]["coil_specs"]["coil_b"]["enabled"] = False
        adapter = StudioAdapter(scene)
        choices = drive_coil_catalog(adapter)
        by_id = {choice["id"]: choice for choice in choices}
        self.assertEqual(set(by_id), {"coil_a", "coil_b"})
        self.assertTrue(by_id["coil_a"]["enabled"])
        self.assertFalse(by_id["coil_b"]["enabled"])

    def test_disabled_scene_coil_can_be_driven_by_waveform_basis(self):
        scene = compact_workflow_scene()
        scene["field_workbench"]["coil_specs"]["coil_b"]["enabled"] = False
        adapter = StudioAdapter(scene)
        adapter.set_coil_modeling_method("centreline")
        points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float)
        fixed, bases = field_bases_for_drives(
            adapter,
            coil_ids=["coil_b"],
            zero_coil_ids=["coil_a", "coil_b"],
            points_m=points,
            field="B",
        )
        current = 0.031
        candidate = StudioAdapter(scene)
        candidate.set_coil_modeling_method("centreline")
        candidate.apply_operations(
            [
                {"method": "set_coil_physical", "params": {"object_id": "coil_a", "drive_current_a": 0.0}},
                {"method": "set_coil_physical", "params": {"object_id": "coil_b", "enabled": True, "drive_current_a": current}},
            ]
        )
        expected = np.asarray(candidate.get_field(points=points, field="B")["values"], dtype=float)
        np.testing.assert_allclose(fixed + current * bases[0], expected, rtol=2e-11, atol=1e-15)

    def test_field_basis_reconstructs_selected_coil_current(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float)
        fixed, basis = field_basis_for_drive(
            adapter,
            coil_id="coil_a",
            points_m=points,
            field="B",
        )
        current = 0.031
        candidate = StudioAdapter(adapter.document())
        candidate.set_coil_modeling_method("centreline")
        candidate.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": "coil_a", "drive_current_a": current},
                }
            ]
        )
        expected = np.asarray(candidate.get_field(points=points, field="B")["values"], dtype=float)
        np.testing.assert_allclose(fixed + current * basis, expected, rtol=2e-11, atol=1e-15)

    def test_controlled_but_unselected_coil_is_off_during_playback(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float)
        fixed, bases = field_bases_for_drives(
            adapter,
            coil_ids=["coil_a"],
            zero_coil_ids=["coil_a", "coil_b"],
            points_m=points,
            field="B",
        )
        current = 0.031
        candidate = StudioAdapter(adapter.document())
        candidate.set_coil_modeling_method("centreline")
        candidate.apply_operations(
            [
                {"method": "set_coil_physical", "params": {"object_id": "coil_a", "drive_current_a": current}},
                {"method": "set_coil_physical", "params": {"object_id": "coil_b", "drive_current_a": 0.0}},
            ]
        )
        expected = np.asarray(candidate.get_field(points=points, field="B")["values"], dtype=float)
        np.testing.assert_allclose(fixed + current * bases[0], expected, rtol=2e-11, atol=1e-15)

    def test_multi_coil_field_bases_reconstruct_independent_currents(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float)
        fixed, bases = field_bases_for_drives(
            adapter, coil_ids=["coil_a", "coil_b"], points_m=points, field="B"
        )
        self.assertEqual(bases.shape, (2, 2, 3))
        currents = [0.031, -0.017]
        candidate = StudioAdapter(adapter.document())
        candidate.set_coil_modeling_method("centreline")
        candidate.apply_operations(
            [
                {"method": "set_coil_physical", "params": {"object_id": "coil_a", "drive_current_a": currents[0]}},
                {"method": "set_coil_physical", "params": {"object_id": "coil_b", "drive_current_a": currents[1]}},
            ]
        )
        expected = np.asarray(candidate.get_field(points=points, field="B")["values"], dtype=float)
        reconstructed = fixed + currents[0] * bases[0] + currents[1] * bases[1]
        np.testing.assert_allclose(reconstructed, expected, rtol=2e-11, atol=1e-15)

    def test_animation_field_validation_reconstructs_direct_solver_cube(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        report = validate_animation_field_basis(
            adapter,
            coil_ids=["coil_a", "coil_b"],
            controlled_coil_ids=["coil_a", "coil_b"],
            centre_mm=[0.0, 0.0, 0.0],
            span_mm=30.0,
            resolution=3,
            field="B",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["point_count"], 27)
        self.assertGreaterEqual(len(report["states"]), 4)
        self.assertLess(report["max_relative_error_pct"], 1e-6)

    def test_small_2d_animation_contains_plotly_frames_with_fixed_colour_range(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.5, 1.0], [0.0, 0.02, 0.0], kind="current")
        figure = build_2d_animation(
            adapter,
            map_settings={
                "plane": "xy",
                "offset_mm": 0.0,
                "span_mm": 50.0,
                "resolution": 4,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "show_object_outlines": False,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=3,
            analysis_samples=9,
            playback_fps=25.0,
        )
        self.assertEqual(len(figure["frames"]), 3)
        self.assertEqual(figure["frames"][0]["traces"], [0])
        self.assertIn("zmin", figure["data"][0])
        self.assertIn("zmax", figure["data"][0])
        self.assertIn("updatemenus", figure["layout"])
        self.assertEqual(figure["layout"]["updatemenus"], [])
        self.assertIn("sliders", figure["layout"])
        playback_meta = figure["layout"]["meta"]["field_workbench_waveform"]
        self.assertEqual(playback_meta["playback_fps"], 25.0)
        self.assertAlmostEqual(playback_meta["real_time_multiplier"], 12.5, places=12)
        self.assertEqual(playback_meta["frame_count"], 3)
        self.assertEqual(playback_meta["update_mode"], "restyle")
        self.assertTrue(all(step["method"] == "skip" for step in figure["layout"]["sliders"][0]["steps"]))


    def test_small_2d_gpu_playback_uploads_basis_instead_of_heatmap_frames(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.5, 1.0], [0.0, 0.02, 0.0], kind="current")
        payload = build_2d_gpu_playback(
            adapter,
            map_settings={
                "plane": "xy",
                "offset_mm": 0.0,
                "span_mm": 50.0,
                "resolution": 4,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "show_grid": False,
                "show_contours": True,
                "contour_count": 7,
                "contour_labels": True,
                "show_object_outlines": False,
                "outline_object_ids": [],
                "colour_minimum": None,
                "colour_maximum": None,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=61,
            analysis_samples=9,
            playback_fps=60.0,
        )
        self.assertEqual(payload["renderer"], "field-workbench-webgl2-waveform-v4")
        self.assertEqual(payload["view_mode"], "2d")
        self.assertEqual(payload["plane"], "xy")
        self.assertEqual(payload["slice_count"], 1)
        self.assertEqual(payload["resolution"], 4)
        self.assertEqual(payload["frame_count"], 61)
        self.assertEqual(len(base64.b64decode(payload["frame_times_b64"])), 61 * 8)
        self.assertEqual(len(base64.b64decode(payload["frame_currents_b64"])), 61 * 4)
        self.assertIsInstance(payload["positions"], str)
        self.assertIsInstance(payload["fixed_vectors"], str)
        self.assertIsInstance(payload["basis_vectors"], str)
        self.assertNotIn("frames", payload)
        self.assertGreater(payload["colour_max"], payload["colour_min"])
        self.assertFalse(payload["show_grid"])
        self.assertFalse(payload["grid"]["enabled"])
        self.assertEqual(payload["grid"]["plane"], "xy")
        self.assertEqual(len(payload["grid"]["plane_segments_mm"]), 10)
        self.assertTrue(payload["show_contours"])
        self.assertEqual(payload["contour_count"], 7)
        self.assertTrue(payload["contour_labels"])

    def test_gpu_playback_supports_multiple_observation_sources_and_base_excitation(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.1, 0.2], [0.01, -0.02, 0.01], kind="current")
        payload = build_2d_gpu_playback(
            adapter,
            map_settings={
                "plane": "xy", "offset_mm": 0.0, "span_mm": 30.0, "resolution": 3,
                "field": "B", "component": "magnitude", "logarithmic": False,
                "show_object_outlines": False, "outline_object_ids": [],
                "colour_minimum": None, "colour_maximum": None,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=5,
            analysis_samples=9,
            playback_fps=60.0,
            observation=[
                {"kind": "sensor", "id": "axis_sensor", "label": "Sample A", "sample_index": 0, "component": "abs_z"},
                {"kind": "sensor", "id": "axis_sensor", "label": "Sample B", "sample_index": 1, "component": "abs_z"},
            ],
            include_excitation_trace=True,
        )
        self.assertEqual(len(payload["observations"]), 2)
        self.assertTrue(payload["observations"][0]["label"].startswith("Sample A — |Bz|"))
        self.assertTrue(payload["observations"][1]["label"].startswith("Sample B — |Bz|"))
        self.assertTrue(all(value >= 0.0 for item in payload["observations"] for value in item["values"]))
        self.assertEqual(payload["excitation"]["label"], "Base excitation current")
        self.assertEqual(payload["excitation"]["unit"], "mA")
        self.assertLess(min(payload["excitation"]["values"]), 0.0)
        self.assertGreater(max(payload["excitation"]["values"]), 0.0)

    def test_gpu_scene_payload_keeps_object_opacity_explicit_for_webgl(self):
        figure = {
            "data": [
                {
                    "type": "mesh3d",
                    "x": [0.0, 1.0, 0.0],
                    "y": [0.0, 0.0, 1.0],
                    "z": [0.0, 0.0, 0.0],
                    "i": [0], "j": [1], "k": [2],
                    "color": "rgba(100,150,200,0.5)",
                    "opacity": 0.28,
                }
            ]
        }
        payload = _gpu_scene_payload(figure)
        self.assertEqual(len(payload["meshes"]), 1)
        mesh = payload["meshes"][0]
        self.assertAlmostEqual(mesh["opacity"], 0.28)
        self.assertAlmostEqual(mesh["colour"][3], 0.5)

    def test_gpu_playback_distinguishes_signed_and_axis_magnitude_components(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.2], [0.01, 0.01], kind="current")
        common = {
            "plane": "xy", "offset_mm": 0.0, "span_mm": 30.0, "resolution": 3,
            "field": "B", "logarithmic": False, "show_object_outlines": False,
            "outline_object_ids": [], "colour_minimum": None, "colour_maximum": None,
        }
        signed = build_2d_gpu_playback(
            adapter, map_settings={**common, "component": "y"}, coil_id="coil_a",
            trace=trace, frame_count=5, analysis_samples=9, playback_fps=60.0,
        )
        magnitude = build_2d_gpu_playback(
            adapter, map_settings={**common, "component": "abs_y"}, coil_id="coil_a",
            trace=trace, frame_count=5, analysis_samples=9, playback_fps=60.0,
        )
        self.assertTrue(signed["signed_component"])
        self.assertAlmostEqual(signed["colour_min"], -signed["colour_max"], places=12)
        self.assertEqual(signed["component_label"], "By")
        self.assertFalse(magnitude["signed_component"])
        self.assertGreaterEqual(magnitude["colour_min"], 0.0)
        self.assertEqual(magnitude["component_label"], "|By|")

    def test_2d_gpu_playback_uploads_multiple_coil_bases_and_current_columns(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.2], [0.01, 0.01], kind="current")
        payload = build_2d_gpu_playback(
            adapter,
            map_settings={
                "plane": "xy", "offset_mm": 0.0, "span_mm": 30.0, "resolution": 3,
                "field": "B", "component": "magnitude", "logarithmic": False,
                "show_object_outlines": False, "outline_object_ids": [],
                "colour_minimum": None, "colour_maximum": None,
            },
            coil_ids=["coil_a", "coil_b"],
            trace=trace,
            frame_count=5,
            analysis_samples=9,
            playback_fps=60.0,
            sequence_steps=[
                {"duration_s": 0.05, "active_coil_ids": ["coil_a"]},
                {"duration_s": 0.05, "active_coil_ids": ["coil_b"]},
            ],
            sequence_loop=True,
        )
        self.assertEqual(payload["drive_coil_count"], 2)
        self.assertEqual(payload["drive_coil_ids"], ["coil_a", "coil_b"])
        self.assertEqual(len(payload["basis_vectors_b64"]), 2)
        self.assertEqual(len(base64.b64decode(payload["frame_currents_b64"])), 5 * 2 * 4)
        self.assertEqual(len(base64.b64decode(payload["free_frame_currents_b64"])), 5 * 2 * 4)
        self.assertTrue(payload["playback_sequence"]["loop"])
        self.assertAlmostEqual(payload["playback_sequence"]["cycle_s"], 0.1, places=12)
        self.assertEqual(
            [step["active_coil_indices"] for step in payload["playback_sequence"]["steps"]],
            [[0], [1]],
        )
        free_currents = np.frombuffer(
            base64.b64decode(payload["free_frame_currents_b64"]), dtype=np.float32
        ).reshape(5, 2)
        np.testing.assert_allclose(free_currents, 0.01, atol=1e-9)

    def test_small_3d_gpu_playback_uploads_basis_instead_of_plotly_frames(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.5, 1.0], [0.0, 0.02, 0.0], kind="current")
        progress_updates = []
        payload = build_3d_gpu_playback(
            adapter,
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 50.0,
                "slice_axis": "z",
                "slice_count": 3,
                "resolution": 4,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "include_scene": False,
                "scene_object_ids": [],
                "colour_minimum": None,
                "colour_maximum": None,
                "volume_opacity_lower": 0.03,
                "volume_opacity_upper": 0.27,
                "volume_opacity_lower_level": 0.20,
                "volume_opacity_upper_level": 0.80,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=61,
            analysis_samples=9,
            playback_fps=60.0,
            progress_callback=progress_updates.append,
        )
        self.assertEqual(payload["renderer"], "field-workbench-webgl2-waveform-v4")
        self.assertEqual(payload["slice_count"], 3)
        self.assertEqual(payload["resolution"], 4)
        self.assertEqual(payload["frame_count"], 61)
        self.assertEqual(len(base64.b64decode(payload["frame_times_b64"])), 61 * 8)
        self.assertEqual(len(base64.b64decode(payload["frame_currents_b64"])), 61 * 4)
        self.assertIsInstance(payload["positions"], str)
        self.assertIsInstance(payload["fixed_vectors"], str)
        self.assertIsInstance(payload["basis_vectors"], str)
        self.assertNotIn("frames", payload)
        self.assertGreater(payload["colour_max"], payload["colour_min"])
        self.assertTrue(payload["intensity_opacity_enabled"])
        self.assertAlmostEqual(payload["volume_opacity_lower"], 0.03)
        self.assertAlmostEqual(payload["volume_opacity_upper"], 0.27)
        self.assertAlmostEqual(payload["volume_opacity_lower_level"], 0.20)
        self.assertAlmostEqual(payload["volume_opacity_upper_level"], 0.80)
        self.assertTrue(payload["show_grid"])
        self.assertTrue(payload["grid"]["enabled"])
        self.assertEqual(
            payload["axis_labels"],
            ["X (LR, mm)", "Y (AP, mm)", "Z (SI, mm)"],
        )
        self.assertEqual(len(payload["grid"]["box_segments_mm"]), 12)
        # Even with scene geometry disabled, the shared map-volume frame remains.
        self.assertEqual(len(payload["scene"]["lines"]), 1)
        task_ids = [update.get("render_task") for update in progress_updates]
        self.assertEqual(task_ids[0], "grid")
        self.assertIn("field", task_ids)
        self.assertEqual(task_ids[-1], "assemble")
        self.assertTrue(progress_updates[-1]["render_task_complete"])



    def test_3d_gpu_point_playback_builds_independent_brain_region_traces(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.5, 1.0], [0.0, 0.02, 0.0], kind="current")
        payload = build_3d_gpu_playback(
            adapter,
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 20.0,
                "slice_axis": "z",
                "slice_count": 1,
                "resolution": 2,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "include_scene": False,
                "scene_object_ids": [],
                "colour_minimum": None,
                "colour_maximum": None,
                "view_type": "points",
                "waveform_render_mode": "points",
                "points_mm": [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                "point_label_ids": [1, 1, 2, 2],
                "point_weights_mm3": [1.0, 2.0, 1.0, 2.0],
                "highlight_selected_region": True,
                "highlight_region_label_ids": [2],
                "trace_regions": [
                    {"name": "Region one", "label_ids": [1]},
                    {"name": "Region two", "label_ids": [2]},
                ],
                "scroll_observation_traces": True,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=11,
            analysis_samples=9,
            playback_fps=60.0,
        )
        self.assertEqual(payload["render_mode"], "points")
        self.assertTrue(payload["scroll_observation_traces"])
        self.assertEqual(
            [entry["label"] for entry in payload["observations"][:2]],
            [
                "Region one — spatial RMS |B|",
                "Region two — spatial RMS |B|",
            ],
        )
        self.assertEqual(len(payload["observations"][0]["values"]), 9)
        self.assertEqual(len(payload["observations"][1]["values"]), 9)
        # Region selection/highlighting belongs to the rendered Brain Areas
        # result tab now; it is no longer baked into the playback payload.
        self.assertEqual(payload["scene"]["points"], [])

    def test_3d_brain_playback_payload_contains_summary_metrics_and_run_facts(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        samples = sample_lpba40_voxels(maximum_samples_per_label=9)
        trace = validate_trace([0.0, 0.5, 1.0], [0.0, 0.02, 0.0], kind="current")
        payload = build_3d_gpu_playback(
            adapter,
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 30.0,
                "slice_axis": "z",
                "slice_count": 1,
                "resolution": 2,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "include_scene": False,
                "scene_object_ids": [],
                "colour_minimum": None,
                "colour_maximum": None,
                "view_type": "slices",
                "waveform_render_mode": "slices",
                "trace_points_mm": samples.points_mm.tolist(),
                "trace_point_label_ids": samples.label_ids.tolist(),
                "trace_point_weights_mm3": samples.weights_mm3.tolist(),
                "trace_source_voxel_counts": {
                    str(key): int(value)
                    for key, value in samples.source_voxel_counts.items()
                },
                "trace_voxel_volume_mm3": samples.voxel_volume_mm3,
                "trace_regions": [],
                "precompute_brain_region_frames": True,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=5,
            analysis_samples=9,
            playback_fps=60.0,
            waveform_metadata={
                "mode": "sine",
                "frequency_hz": 2.0,
                "cycle_count": 2.0,
                "pass_count": 1,
            },
            sequence_steps=[
                {"duration_s": 0.25, "active_coil_ids": ["coil_a"]},
                {"duration_s": 0.25, "active_coil_ids": []},
            ],
            sequence_loop=True,
        )
        frame_payload = payload["brain_region_frame_metrics"]
        summary = frame_payload["playback_summary"]
        summary_values = np.frombuffer(
            base64.b64decode(summary["values_b64"]), dtype=np.float64
        ).reshape(summary["shape"])
        peak_times = np.frombuffer(
            base64.b64decode(summary["peak_times_s_b64"]), dtype=np.float64
        )
        self.assertEqual(summary_values.shape[1], 7)
        self.assertTrue(np.isfinite(summary_values).all())
        self.assertEqual(len(peak_times), summary_values.shape[0])
        whole_index = frame_payload["keys"].index("whole_brain")
        playback_rms = summary_values[whole_index, 0]
        b2_time = summary_values[whole_index, 5]
        self.assertAlmostEqual(b2_time, playback_rms**2, places=8)
        facts = summary["playback"]
        self.assertEqual(facts["waveform_cycle_count"], 2.0)
        self.assertEqual(facts["sequencer_step_count"], 2)
        self.assertEqual(facts["sequencer_full_loops"], 2)
        self.assertEqual(facts["sequencer_step_activations"], 4)
        self.assertEqual(facts["frame_count"], 5)
        self.assertEqual(facts["analysis_sample_count"], 9)
        self.assertEqual(facts["atlas_sample_count"], samples.point_count)

    def test_3d_gpu_volume_playback_uploads_complete_voxel_basis(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 1.0], [0.0, 0.02], kind="current")
        payload = build_3d_gpu_playback(
            adapter,
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 30.0,
                "slice_axis": "z",
                "slice_count": 3,
                "resolution": 4,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "include_scene": False,
                "scene_object_ids": [],
                "colour_minimum": None,
                "colour_maximum": None,
                "view_type": "volume",
                "intensity_opacity_enabled": True,
                "volume_opacity_lower": 0.0,
                "volume_opacity_upper": 0.2,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=61,
            analysis_samples=9,
            playback_fps=60.0,
        )
        self.assertEqual(payload["render_mode"], "volume")
        self.assertEqual(payload["resolution"], 4)
        self.assertEqual(payload["slice_positions_mm"], [])
        self.assertEqual(payload["positions"], "")
        self.assertEqual(payload["fixed_vectors"], "")
        self.assertEqual(payload["basis_vectors"], "")
        voxel_vector_bytes = 4 * 4 * 4 * 3 * 4
        self.assertEqual(len(base64.b64decode(payload["volume_fixed_vectors"])), voxel_vector_bytes)
        self.assertEqual(len(base64.b64decode(payload["volume_basis_vectors"])), voxel_vector_bytes)
        self.assertTrue(payload["intensity_opacity_enabled"])
        self.assertAlmostEqual(payload["volume_opacity_lower"], 0.0)
        self.assertAlmostEqual(payload["volume_opacity_upper"], 0.2)
        self.assertEqual(payload["volume_surface_count"], 12)

    def test_3d_gpu_volume_playback_keeps_brain_region_traces(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        trace = validate_trace([0.0, 0.5, 1.0], [0.0, 0.02, 0.0], kind="current")
        payload = build_3d_gpu_playback(
            adapter,
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 30.0,
                "slice_axis": "z",
                "slice_count": 3,
                "resolution": 4,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "include_scene": False,
                "scene_object_ids": [],
                "colour_minimum": None,
                "colour_maximum": None,
                "view_type": "volume",
                "waveform_render_mode": "volume",
                "trace_points_mm": [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                "trace_point_label_ids": [1, 1, 2, 2],
                "trace_point_weights_mm3": [1.0, 2.0, 1.0, 2.0],
                "trace_regions": [
                    {"name": "Region one", "label_ids": [1]},
                    {"name": "Region two", "label_ids": [2]},
                ],
                "scroll_observation_traces": True,
            },
            coil_id="coil_a",
            trace=trace,
            frame_count=11,
            analysis_samples=9,
            playback_fps=60.0,
        )
        self.assertEqual(payload["render_mode"], "volume")
        self.assertEqual(
            [entry["label"] for entry in payload["observations"][:2]],
            [
                "Region one — spatial RMS |B|",
                "Region two — spatial RMS |B|",
            ],
        )
        self.assertEqual(len(payload["observations"][0]["values"]), 9)
        self.assertEqual(len(payload["observations"][1]["values"]), 9)

    def test_3d_gpu_static_can_return_separate_brain_analysis_vectors(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.set_coil_modeling_method("centreline")
        payload = build_3d_gpu_static(
            adapter,
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 30.0,
                "slice_axis": "z",
                "slice_count": 3,
                "resolution": 4,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "include_scene": False,
                "scene_object_ids": [],
                "colour_minimum": None,
                "colour_maximum": None,
                "view_type": "volume",
                "analysis_points_mm": [
                    [-2.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                "_return_field_vectors": True,
            },
        )
        self.assertEqual(payload["render_mode"], "volume")
        self.assertIn("_field_vectors_t", payload)
        returned = np.asarray(payload["_field_vectors_t"], dtype=float)
        self.assertEqual(returned.shape, (2, 3))
        voxel_vector_bytes = 4 * 4 * 4 * 3 * 4
        self.assertEqual(len(base64.b64decode(payload["volume_fixed_vectors"])), voxel_vector_bytes)

    def test_frame_limit_allows_long_gpu_slow_motion_timelines(self):
        self.assertEqual(playback_frame_count(1.0, 60.0, 0.0001), 600001)



if __name__ == "__main__":
    unittest.main()
