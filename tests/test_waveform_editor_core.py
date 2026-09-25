"""Headless regression checks for the native FWB Waveform Editor core."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fieldworkbench.waveform_editor.core.dac import parse_dac_text, read_dac, write_dac
from fieldworkbench.waveform_editor.core.encoding import convert_points_bit_depth
from fieldworkbench.waveform_editor.core.native import (
    FreeFormPoint,
    GeneratedTrace,
    compose_generated_sequence,
    compose_generated_trace,
    generated_trace_from_recipe_segment,
    action_potential_trace,
    free_form_trace,
    linear_map_codes,
    quantize_to_unsigned_codes,
    sine_wave_trace,
    square_wave_trace,
)
from fieldworkbench.waveform_editor.core.rigol import (
    CsvSettings,
    RafSettings,
    build_raf,
    iter_csv_samples,
    plan_csv,
)


class NativeWaveformEditorCoreTests(unittest.TestCase):
    def test_dc_generator_creates_constant_physical_segment(self) -> None:
        from fieldworkbench.waveform_editor.core.native import dc_trace

        trace = dc_trace(duration_s=0.25, level=-3.5)
        self.assertEqual(trace.time_s, (0.0, 0.25))
        self.assertEqual(trace.values, (-3.5, -3.5))
        self.assertEqual(trace.settings["shape"], "dc")
    def test_signal_generators_use_physical_controls_not_point_counts(self) -> None:
        square = square_wave_trace(
            frequency_hz=20.0,
            duration_s=0.1,
            peak_to_peak=10.0,
            offset=2.0,
            duty_cycle_percent=25.0,
        )
        self.assertAlmostEqual(square.time_s[0], 0.0)
        self.assertAlmostEqual(square.time_s[-1], 0.1)
        self.assertEqual(square.settings["frequency_hz"], 20.0)
        self.assertEqual(square.settings["duty_cycle_percent"], 25.0)
        self.assertAlmostEqual(max(square.values), 7.0)
        self.assertAlmostEqual(min(square.values), -3.0)

        sine = sine_wave_trace(
            frequency_hz=20.0,
            duration_s=0.1,
            peak_to_peak=10.0,
            offset=2.0,
        )
        self.assertAlmostEqual(sine.time_s[-1], 0.1)
        self.assertAlmostEqual(sine.values[0], 2.0)
        self.assertAlmostEqual(max(sine.values), 7.0, places=5)
        self.assertAlmostEqual(min(sine.values), -3.0, places=5)

    def test_chunk_composition_preserves_independent_time_axes(self) -> None:
        existing_times = (0.0, 0.1, 0.2)
        existing_values = (0.0, 1.0, 2.0)
        generated = GeneratedTrace(
            (4.0, 4.003, 4.010),
            (9.0, 8.0, 7.0),
            {"shape": "sine", "duration_s": 0.010},
        )

        composed = compose_generated_trace(
            generated,
            append=True,
            existing_time_s=existing_times,
            existing_values=existing_values,
        )

        self.assertEqual(composed.append_boundary_s, 0.2)
        self.assertEqual(composed.time_s[:3], existing_times)
        self.assertEqual(composed.values[:3], existing_values)
        # A discontinuous join keeps the old endpoint and begins the new chunk
        # immediately after it rather than replacing that endpoint.
        self.assertGreater(composed.time_s[3], 0.2)
        self.assertEqual(composed.values[3], 9.0)
        self.assertAlmostEqual(composed.time_s[-2], 0.203)
        self.assertAlmostEqual(composed.time_s[-1], 0.210)
        self.assertEqual(composed.values[-2:], (8.0, 7.0))

    def test_chunk_composition_continuous_join_reuses_boundary(self) -> None:
        generated = GeneratedTrace(
            (0.0, 0.003, 0.010),
            (2.0, 3.0, 4.0),
            {"shape": "square", "duration_s": 0.010},
        )
        composed = compose_generated_trace(
            generated,
            append=True,
            existing_time_s=(0.0, 0.1, 0.2),
            existing_values=(0.0, 1.0, 2.0),
        )
        self.assertEqual(len(composed.time_s), 5)
        self.assertEqual(composed.values, (0.0, 1.0, 2.0, 3.0, 4.0))
        self.assertAlmostEqual(composed.time_s[-2], 0.203)
        self.assertAlmostEqual(composed.time_s[-1], 0.210)

    def test_replace_composition_normalizes_generated_time_origin(self) -> None:
        generated = GeneratedTrace(
            (12.0, 12.25, 12.5),
            (-1.0, 0.0, 1.0),
            {"shape": "free_form", "duration_s": 0.5},
        )
        composed = compose_generated_trace(generated, append=False)
        self.assertEqual(composed.time_s, (0.0, 0.25, 0.5))
        self.assertEqual(composed.values, generated.values)
        self.assertIsNone(composed.append_boundary_s)

    def test_action_potential_and_free_form_use_duration_not_output_points(self) -> None:
        action = action_potential_trace(
            duration_s=0.02,
            start_percent=10,
            peak_percent=30,
            trough_percent=60,
            recovery_percent=80,
            resting_level=1.2,
            peak_level=7.5,
            trough_level=-2.25,
        )
        self.assertAlmostEqual(action.time_s[-1], 0.02)
        self.assertAlmostEqual(action.values[0], 1.2)
        self.assertGreater(max(action.values), 7.0)
        self.assertLess(min(action.values), -2.0)

        free = free_form_trace([(0, -3.0), (50, 9.0), (100, 1.0)], duration_s=2.0)
        self.assertAlmostEqual(free.time_s[-1], 2.0)
        middle = min(range(len(free.time_s)), key=lambda index: abs(free.time_s[index] - 1.0))
        self.assertAlmostEqual(free.values[middle], 9.0, places=5)
        self.assertEqual(free.settings["curve"], "cubic_bezier_mirrored_tangents")
        self.assertEqual(
            [point["time_percent"] for point in free.settings["control_points"]],
            [0.0, 50.0, 100.0],
        )

    def test_free_form_bezier_handles_shape_curve_without_exposing_sampling(self) -> None:
        free = free_form_trace(
            [
                FreeFormPoint(0.0, 0.0, 0.2, 1.0),
                FreeFormPoint(1.0, 0.0, 0.2, -1.0),
            ],
            duration_s=1.0,
        )
        midpoint = min(range(len(free.time_s)), key=lambda index: abs(free.time_s[index] - 0.5))
        self.assertGreater(free.values[midpoint], 0.7)
        self.assertAlmostEqual(free.values[0], 0.0)
        self.assertAlmostEqual(free.values[-1], 0.0)
        self.assertEqual(free.settings["control_points"][0]["handle_time_percent"], 20.0)
        self.assertEqual(free.settings["control_points"][0]["handle_value"], 1.0)

    def test_dac_import_translation_and_export_quantization_are_boundary_operations(self) -> None:
        mapped = linear_map_codes(
            [0, 128, 255],
            code_min=0,
            code_max=255,
            value_at_min=-5.0,
            value_at_max=5.0,
        )
        self.assertAlmostEqual(float(mapped[0]), -5.0)
        self.assertAlmostEqual(float(mapped[-1]), 5.0)
        codes = quantize_to_unsigned_codes(
            mapped,
            bit_depth=8,
            value_at_zero=-5.0,
            value_at_full_scale=5.0,
        )
        np.testing.assert_array_equal(codes, np.asarray([0, 128, 255], dtype=np.uint8))

    def test_integer_width_conversion_remains_available_only_for_external_codecs(self) -> None:
        self.assertEqual(
            convert_points_bit_depth(
                [0, 128, 255], source_bit_depth=8, target_bit_depth=16
            ),
            [0, 32896, 65535],
        )

    def test_legacy_dac_parser_and_file_round_trip(self) -> None:
        parsed = parse_dac_text("3\n0\n128\n255\n")
        self.assertEqual(parsed.points, (0, 128, 255))
        self.assertTrue(parsed.header_removed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.dac"
            write_dac(path, parsed.points)
            self.assertEqual(read_dac(path).points, parsed.points)

    def test_rigol_export_boundary_paths_remain_available(self) -> None:
        build = build_raf(
            [0, 64, 128, 200, 255],
            file_name="test.RAF",
            settings=RafSettings(
                point_delay_ms=0.01,
                cycle_delay_ms=0.25,
                rest_value=128,
                bit_depth=8,
                slew_rate_v_per_us=1.0,
            ),
        )
        self.assertEqual(len(build.header), 56)
        self.assertGreater(len(build.payload), 0)

        settings = CsvSettings(
            point_delay_ms=1,
            cycle_delay_ms=0,
            rest_value=32768,
            bit_depth=16,
            target_points=3,
        )
        plan = plan_csv([0, 32768, 65535], settings)
        samples = list(iter_csv_samples([0, 32768, 65535], settings, plan))
        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(samples[0], -5.0)
        self.assertAlmostEqual(samples[1], 0.0)


if __name__ == "__main__":
    unittest.main()


class WaveformAuthoringRecipeTests(unittest.TestCase):
    def test_sampled_recipe_segment_is_lossless_and_reconstructable(self) -> None:
        from fieldworkbench.waveform_recipe import make_authoring_recipe, sampled_segment
        from fieldworkbench.waveform_editor.core.native import trace_from_recipe_segment

        segment = sampled_segment([4.0, 4.1, 4.25], [1.0, -2.0, 3.0], start_s=0.5, description="DAC")
        recipe = make_authoring_recipe([segment], quantity="voltage", unit="V")
        restored = trace_from_recipe_segment(recipe["segments"][0])
        np.testing.assert_allclose(restored.time_s, (0.0, 0.1, 0.25))
        self.assertEqual(restored.values, (1.0, -2.0, 3.0))

    def test_runtime_encoded_recipe_is_format_neutral_and_reconstructable(self) -> None:
        from fieldworkbench.waveform_formats import WaveformFormatProfile
        from fieldworkbench.waveform_recipe import encoded_segment, make_authoring_recipe, normalize_authoring_recipe
        from fieldworkbench.waveform_editor.core.native import trace_from_recipe_segment

        profile = WaveformFormatProfile(
            profile_id="tests.runtime-bytes",
            name="Runtime bytes",
            data_mode="binary",
            number_format="uint8",
            binary_record_bytes=1,
            time_mode="runtime_interval",
            sample_interval=1.0,
            sample_interval_unit="ms",
            conversion_mode="runtime_linear",
            runtime_raw_min=0.0,
            runtime_raw_max=255.0,
            output_quantity="current",
            output_unit="mA",
        )
        segment = encoded_segment(
            [0, 128, 255],
            start_s=0.0,
            sample_interval_s=0.002,
            profile=profile.to_dict(),
            runtime_value_at_min=-4.0,
            runtime_value_at_max=6.0,
            source_name="anything.bin",
        )
        canonical = normalize_authoring_recipe(
            make_authoring_recipe([segment], quantity="current", unit="mA")
        )["segments"][0]
        self.assertEqual(canonical["kind"], "encoded")
        self.assertEqual(canonical["raw_values"], [0.0, 128.0, 255.0])
        restored = trace_from_recipe_segment(canonical)
        np.testing.assert_allclose(restored.time_s, (0.0, 0.002, 0.004))
        np.testing.assert_allclose((restored.values[0], restored.values[2]), (-4.0, 6.0))

    def test_generated_recipe_segment_reopens_original_signal_generator_controls(self) -> None:
        from fieldworkbench.waveform_recipe import generated_segment_from_settings

        original = sine_wave_trace(
            frequency_hz=17.5,
            duration_s=0.4,
            peak_to_peak=6.0,
            offset=-1.25,
            phase_deg=37.0,
        )
        segment = generated_segment_from_settings(original.settings, start_s=0.0)
        rebuilt = generated_trace_from_recipe_segment(segment)

        self.assertEqual(rebuilt.settings["shape"], "sine")
        self.assertEqual(rebuilt.settings["frequency_hz"], 17.5)
        self.assertEqual(rebuilt.settings["duration_s"], 0.4)
        self.assertEqual(rebuilt.settings["peak_to_peak"], 6.0)
        self.assertEqual(rebuilt.settings["offset"], -1.25)
        self.assertEqual(rebuilt.settings["phase_deg"], 37.0)
        self.assertEqual(rebuilt.time_s, original.time_s)
        self.assertEqual(rebuilt.values, original.values)

    def test_free_form_recipe_reopens_bezier_geometry(self) -> None:
        from fieldworkbench.waveform_recipe import generated_segment_from_settings

        original = free_form_trace(
            [
                FreeFormPoint(0.0, -2.0, 0.12, 0.6),
                FreeFormPoint(0.35, 4.0, 0.08, -0.5),
                FreeFormPoint(1.0, 1.0, 0.15, 0.25),
            ],
            duration_s=0.65,
        )
        segment = generated_segment_from_settings(original.settings, start_s=0.0)
        rebuilt = generated_trace_from_recipe_segment(segment)

        self.assertEqual(rebuilt.settings["shape"], "free_form")
        self.assertEqual(rebuilt.settings["duration_s"], 0.65)
        self.assertEqual(rebuilt.settings["control_points"], original.settings["control_points"])
        self.assertEqual(rebuilt.values, original.values)

    def test_reopened_segment_can_change_duration_and_recompose_following_chunks(self) -> None:
        from fieldworkbench.waveform_recipe import generated_segment_from_settings

        first = square_wave_trace(
            frequency_hz=10.0,
            duration_s=0.2,
            peak_to_peak=4.0,
            offset=0.0,
            duty_cycle_percent=50.0,
        )
        second = sine_wave_trace(
            frequency_hz=20.0,
            duration_s=0.1,
            peak_to_peak=2.0,
            offset=0.0,
            phase_deg=0.0,
        )
        second_segment = generated_segment_from_settings(second.settings, start_s=0.2)
        reopened_second = generated_trace_from_recipe_segment(second_segment)
        edited_first = square_wave_trace(
            frequency_hz=10.0,
            duration_s=0.35,
            peak_to_peak=4.0,
            offset=0.0,
            duty_cycle_percent=25.0,
        )
        composed = compose_generated_sequence([edited_first, reopened_second])

        self.assertAlmostEqual(composed.time_s[0], 0.0)
        self.assertAlmostEqual(composed.time_s[-1], 0.45)
        self.assertEqual(composed.values[0], edited_first.values[0])

    def test_versioned_recipe_keeps_generator_intent_separate_from_samples(self) -> None:
        from fieldworkbench.waveform_recipe import (
            AUTHORING_RECIPE_FORMAT,
            AUTHORING_RECIPE_VERSION,
            generated_segment_from_settings,
            make_authoring_recipe,
            normalize_authoring_recipe,
            recipe_is_fully_generated,
        )

        trace = square_wave_trace(
            frequency_hz=20.0,
            duration_s=0.25,
            peak_to_peak=12.0,
            offset=1.0,
            duty_cycle_percent=37.5,
        )
        segment = generated_segment_from_settings(trace.settings, start_s=0.0)
        recipe = make_authoring_recipe([segment], quantity="current", unit="mA")
        canonical = normalize_authoring_recipe(recipe)

        self.assertEqual(canonical["format"], AUTHORING_RECIPE_FORMAT)
        self.assertEqual(canonical["version"], AUTHORING_RECIPE_VERSION)
        self.assertEqual(canonical["signal"], {"quantity": "current", "unit": "mA"})
        self.assertEqual(canonical["segments"][0]["generator"], "square")
        self.assertEqual(canonical["segments"][0]["parameters"]["frequency_hz"], 20.0)
        self.assertEqual(canonical["segments"][0]["parameters"]["duty_cycle_percent"], 37.5)
        self.assertNotIn("sample_count", str(canonical))
        self.assertNotIn("sample_interval", str(canonical))
        self.assertTrue(recipe_is_fully_generated(canonical))

    def test_free_form_recipe_preserves_bezier_geometry(self) -> None:
        from fieldworkbench.waveform_recipe import generated_segment_from_settings, make_authoring_recipe

        trace = free_form_trace(
            [
                FreeFormPoint(0.0, -2.0, 0.15, 1.25),
                FreeFormPoint(0.4, 5.0, 0.10, -0.75),
                FreeFormPoint(1.0, 1.0, 0.15, 0.5),
            ],
            duration_s=0.75,
        )
        recipe = make_authoring_recipe(
            [generated_segment_from_settings(trace.settings, start_s=0.0)],
            quantity="voltage",
            unit="V",
        )
        points = recipe["segments"][0]["parameters"]["control_points"]
        self.assertEqual([point["time_percent"] for point in points], [0.0, 40.0, 100.0])
        self.assertEqual(points[0]["handle_time_percent"], 15.0)
        self.assertEqual(points[0]["handle_value"], 1.25)
        self.assertEqual(recipe["segments"][0]["parameters"]["curve"], "cubic_bezier_mirrored_tangents")

    def test_direct_sample_edit_detaches_recipe_but_keeps_it_as_provenance(self) -> None:
        from fieldworkbench.waveform_recipe import (
            detach_authoring_recipe,
            generated_segment_from_settings,
            make_authoring_recipe,
            recipe_is_exact,
        )

        trace = sine_wave_trace(
            frequency_hz=30.0,
            duration_s=1.0,
            peak_to_peak=8.0,
            offset=0.0,
            phase_deg=45.0,
        )
        recipe = make_authoring_recipe(
            [generated_segment_from_settings(trace.settings, start_s=0.0)],
            quantity="current",
            unit="mA",
        )
        detached = detach_authoring_recipe(recipe, reason="manual sample edit")
        self.assertIsNotNone(detached)
        assert detached is not None
        self.assertEqual(detached["integrity"], "detached")
        self.assertEqual(detached["detached_reason"], "manual sample edit")
        self.assertEqual(detached["segments"][0]["parameters"]["phase_deg"], 45.0)
        self.assertFalse(recipe_is_exact(detached))

    def test_authoring_recipe_round_trips_through_native_signal_payload(self) -> None:
        from fieldworkbench.waveform_io import (
            WaveformDocument,
            WaveformSeries,
            waveform_document_from_payload,
            waveform_signal_to_payload,
        )
        from fieldworkbench.waveform_recipe import (
            AUTHORING_RECIPE_KEY,
            generated_segment_from_settings,
            make_authoring_recipe,
        )

        first = square_wave_trace(
            frequency_hz=10.0,
            duration_s=0.2,
            peak_to_peak=4.0,
            offset=0.0,
            duty_cycle_percent=50.0,
        )
        second = sine_wave_trace(
            frequency_hz=25.0,
            duration_s=0.1,
            peak_to_peak=2.0,
            offset=1.0,
            phase_deg=90.0,
        )
        recipe = make_authoring_recipe(
            [
                generated_segment_from_settings(first.settings, start_s=0.0),
                generated_segment_from_settings(second.settings, start_s=0.2),
            ],
            quantity="current",
            unit="mA",
        )
        document = WaveformDocument(
            [WaveformSeries("Recipe", [0.0, 0.2, 0.3], [2.0, 1.0, 2.0], quantity="current", unit="mA")],
            title="Recipe",
            metadata={AUTHORING_RECIPE_KEY: recipe},
        )
        payload = waveform_signal_to_payload(document)
        restored = waveform_document_from_payload(payload)
        restored_recipe = restored.metadata[AUTHORING_RECIPE_KEY]
        self.assertEqual(len(restored_recipe["segments"]), 2)
        self.assertEqual(restored_recipe["segments"][0]["generator"], "square")
        self.assertEqual(restored_recipe["segments"][1]["generator"], "sine")
        self.assertEqual(restored_recipe["segments"][1]["parameters"]["phase_deg"], 90.0)
