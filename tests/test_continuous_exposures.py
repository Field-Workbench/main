"""Exposure-clock regressions against explicit signals and analytic RL responses."""
import copy
import json
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fieldworkbench.exposures import normalize_exposure_protocol, ExposureError
from fieldworkbench.studio_adapter import StudioAdapter, StudioOperationError
from fieldworkbench.waveform import (
    _prepare_exposure_current_timeline, _brain_playback_run_metadata,
    _streaming_series_statistics, time_statistics,
    MAX_ANALYSIS_SAMPLES, MAX_RETAINED_ANALYSIS_SAMPLES,
    _cycle_analysis_times, _find_exposure_periodic_plan,
    _exposure_analysis_clock, _estimate_exposure_analysis_sample_count,
    _exposure_drive_trace_payloads,
)
from fieldworkbench.waveform import validate_trace
from fieldworkbench.brain_analysis import (
    BrainRegionNode, BrainSampleSet, lpba40_frame_region_metrics,
    lpba40_playback_region_metrics, lpba40_playback_region_metrics_streaming,
)
from fieldworkbench.region_analysis import (
    authoritative_analysis_packet_factory, waveform_basis_payload,
    analyse_region_exposure_summary, analyse_region_threshold_duration,
    analyse_region_waveform, analyse_region_spectrum,
    analyse_region_frequency_weighted,
)


def route(source, coil):
    return {'source_id': source, 'coil_ids': [coil], 'connection': 'series'}


def protocol(duration=2.0, repeat=True):
    return normalize_exposure_protocol({
        'duration_s': duration,
        'sources': [{'id': 'a', 'name': 'A', 'waveform_id': 'wave_a'}],
        'sequence': {'loop': repeat, 'steps': [
            {'duration_s': .3, 'drive_groups': [route('a', 'c1')]},
            {'duration_s': .3, 'drive_groups': [route('a', 'c2')]},
        ]},
    })


class ContinuousExposureTests(unittest.TestCase):
    def setUp(self):
        self.adapter = StudioAdapter({'objects': []})
        self.adapter.coil_electrical_load = Mock(return_value={'resistance_ohm': 1.0, 'inductance_h': 0.01})
        self.traces = {'wave_a': ([0, .25, .5, .75, 1], [0, 1, 2, 3, 0], {'source_kind': 'current', 'slew_rate_per_s': 0.0})}
        self.adapter._waveform_source_trace = lambda key: tuple(copy.deepcopy(x) for x in self.traces[key])

    def test_machine_precision_waveform_seam_does_not_become_analysis_clock(self):
        # A component join can produce adjacent representable floats around a
        # real waveform boundary.  That seam must remain an event, not become
        # the uniform clock for an hours-long exposure.
        seam = float(np.nextafter(2.0, math.inf))
        self.traces['wave_a'] = (
            [0.0, 0.0033, 2.0, seam, 2.8017],
            [0.0, 1.0, 1.0, -1.0, 0.0],
            {'source_kind': 'current', 'slew_rate_per_s': 0.0},
        )
        state = protocol(duration=1800.0)
        spacing, clocks = _exposure_analysis_clock(self.adapter, state)
        self.assertAlmostEqual(spacing, 0.0033, places=12)
        estimate = _estimate_exposure_analysis_sample_count(state, spacing, clocks)
        self.assertLess(estimate, 5_000_000)
        self.assertGreater(estimate, 500_000)

    def test_oversized_repeat_cycle_falls_back_before_building_time_grid(self):
        state = protocol(duration=30.0)
        _spacing, clocks = _exposure_analysis_clock(self.adapter, state)
        with patch('fieldworkbench.waveform._iter_exposure_analysis_time_chunks') as chunks:
            plan = _find_exposure_periodic_plan(
                self.adapter, state, state, 1e-9, clocks, ['c1', 'c2']
            )
        self.assertIsNone(plan)
        chunks.assert_not_called()

    def test_event_enriched_repeat_cycle_stops_at_sample_budget(self):
        def chunks(*args):
            yield np.linspace(0, .4, 4)
            yield np.linspace(.4, .8, 4)
            self.fail('Oversized cycle must stop consuming analysis chunks')
        with patch('fieldworkbench.waveform.MAX_PERIODIC_ANALYSIS_CYCLE_SAMPLES', 6), \
             patch('fieldworkbench.waveform._iter_exposure_analysis_time_chunks', chunks):
            self.assertIsNone(_cycle_analysis_times(protocol(), .5, [], 1.0))

    def test_drive_statistics_preserve_pulses_missing_from_retained_trace(self):
        self.traces['wave_a'] = (
            [0, .1, .11, 1], [0, 1, 0, 0],
            {'source_kind': 'current', 'slew_rate_per_s': 0.0},
        )
        for repeat in (True, False):
            with self.subTest(repeat=repeat), \
                 patch('fieldworkbench.waveform.MAX_RETAINED_ANALYSIS_SAMPLES', 5):
                timeline = _prepare_exposure_current_timeline(
                    self.adapter, protocol(duration=30.0, repeat=repeat),
                    coil_ids=['c1', 'c2'], start_s=0, stop_s=.2,
                    frame_count=3, analysis_samples=7,
                )
                self.assertTrue(timeline['analysis_trace_reduced'])
                trace = _exposure_drive_trace_payloads(timeline)[0]
                self.assertAlmostEqual(trace['time_s'][-1], .2)
                self.assertGreater(max(trace['values']), 900.0)
                self.assertGreater(trace['stats']['rms'], 0.0)
                self.assertAlmostEqual(trace['stats']['pk_pk'], 1000.0, places=8)

    def test_switches_and_schedule_wraps_keep_source_phase(self):
        times = [.1, .3, .4, .6, .7, .9, 1.1, 1.4]
        result = self.adapter.resolve_exposure_currents_at_times(protocol(), times)
        np.testing.assert_allclose(result['coil_currents_a']['c1'], [0, 0, 0, 2, 2, 0, 0, 1])
        np.testing.assert_allclose(result['coil_currents_a']['c2'], [0, 1, 1, 0, 0, 3, 0, 0])
        # A sparse, unsorted request must integrate the same preceding history.
        sparse = self.adapter.resolve_exposure_currents_at_times(protocol(), [1.4, .3])
        np.testing.assert_allclose(sparse['coil_currents_a']['c1'], [1, 0])
        np.testing.assert_allclose(sparse['coil_currents_a']['c2'], [0, 1])

    def test_streaming_shared_voltage_source_drives_parallel_series_groups(self):
        self.traces['wave_v'] = (
            [0.0, 1.0], [10.0, 10.0],
            {'source_kind': 'voltage', 'slew_rate_per_s': 0.0, 'output_resistance_ohm': 1.0},
        )
        self.adapter.coil_electrical_load = Mock(
            return_value={'resistance_ohm': 2.0, 'inductance_h': 0.1}
        )
        state = normalize_exposure_protocol({
            'duration_s': 1.0,
            'sources': [{'id': 'v', 'name': 'V', 'waveform_id': 'wave_v'}],
            'sequence': {'loop': False, 'steps': [{
                'duration_s': 1.0,
                'drive_groups': [
                    {'source_id': 'v', 'coil_ids': ['c1', 'c2'], 'connection': 'series'},
                    {'source_id': 'v', 'coil_ids': ['c3', 'c4'], 'connection': 'series'},
                    {'source_id': 'v', 'coil_ids': ['c5', 'c6'], 'connection': 'series'},
                ],
            }]},
        })
        result = self.adapter.resolve_exposure_currents_at_times(state, [0.5])
        expected_total = (10.0 / (1.0 + 4.0 / 3.0)) * (1.0 - math.exp(-35.0 * 0.5))
        expected_branch = expected_total / 3.0
        for first, second in (('c1', 'c2'), ('c3', 'c4'), ('c5', 'c6')):
            self.assertAlmostEqual(result['coil_currents_a'][first][0], expected_branch, places=8)
            self.assertAlmostEqual(result['coil_currents_a'][second][0], expected_branch, places=8)

    def test_once_per_step_source_fires_one_cycle_then_zero_and_restarts(self):
        state = normalize_exposure_protocol({
            'duration_s': 3.0,
            'sources': [{
                'id': 'a', 'name': 'A', 'waveform_id': 'wave_a',
                'playback_mode': 'once_per_step', 'phase_offset_s': 0.0,
            }],
            'sequence': {'loop': True, 'steps': [{
                'duration_s': 1.5, 'drive_groups': [route('a', 'c1')],
            }]},
        })
        times = [.1, .3, .8, 1.1, 1.4, 1.6, 1.8, 2.4, 2.6]
        result = self.adapter.resolve_exposure_currents_at_times(state, times)
        np.testing.assert_allclose(
            result['coil_currents_a']['c1'], [0, 1, 3, 0, 0, 0, 1, 3, 0]
        )

    def test_once_per_step_source_honours_phase_offset_for_single_shot(self):
        state = normalize_exposure_protocol({
            'duration_s': 1.5,
            'sources': [{
                'id': 'a', 'name': 'A', 'waveform_id': 'wave_a',
                'playback_mode': 'once_per_step', 'phase_offset_s': .5,
            }],
            'sequence': {'loop': False, 'steps': [{
                'duration_s': 1.5, 'drive_groups': [route('a', 'c1')],
            }]},
        })
        result = self.adapter.resolve_exposure_currents_at_times(
            state, [.1, .3, .6, .8, 1.0, 1.1]
        )
        np.testing.assert_allclose(
            result['coil_currents_a']['c1'], [2, 3, 0, 1, 0, 0]
        )

    def test_once_per_step_source_only_fires_in_steps_that_route_it(self):
        state = normalize_exposure_protocol({
            'duration_s': 2.4,
            'sources': [{
                'id': 'a', 'name': 'A', 'waveform_id': 'wave_a',
                'playback_mode': 'once_per_step', 'phase_offset_s': 0.0,
            }],
            'sequence': {'loop': False, 'steps': [
                {'duration_s': .6, 'drive_groups': [route('a', 'c1')]},
                {'duration_s': .6, 'drive_groups': []},
                {'duration_s': 1.2, 'drive_groups': [route('a', 'c1')]},
            ]},
        })
        result = self.adapter.resolve_exposure_currents_at_times(
            state, [.3, .8, 1.3, 1.5, 2.25]
        )
        np.testing.assert_allclose(
            result['coil_currents_a']['c1'], [1, 0, 0, 1, 0]
        )

    def test_two_sources_sharing_waveform_have_independent_phase(self):
        state = protocol()
        state['sources'].append({'id': 'b', 'name': 'B', 'waveform_id': 'wave_a', 'phase_offset_s': .5})
        state['sequence']['steps'] = [{'duration_s': 2, 'drive_groups': [route('a', 'c1'), route('b', 'c2')]}]
        result = self.adapter.resolve_exposure_currents_at_times(state, [.1, .4, 1.1])
        np.testing.assert_allclose(result['coil_currents_a']['c1'], [0, 1, 0])
        np.testing.assert_allclose(result['coil_currents_a']['c2'], [2, 3, 2])

    def test_nonrepeat_schedule_goes_idle_and_duration_bounds_requests(self):
        result = self.adapter.resolve_exposure_currents_at_times(protocol(repeat=False), [.4, .6, .8, 2])
        np.testing.assert_allclose(result['coil_currents_a']['c2'], [1, 0, 0, 0])
        with self.assertRaisesRegex(StudioOperationError, 'duration'):
            self.adapter.resolve_exposure_currents_at_times(protocol(), [2.01])

    def test_slew_runs_while_source_is_disconnected(self):
        self.traces['wave_a'] = ([0, .1, 1], [0, 1, 0], {'source_kind': 'current', 'slew_rate_per_s': 1.0})
        state = protocol()
        state['sequence']['steps'][0]['drive_groups'] = []
        result = self.adapter.resolve_exposure_currents_at_times(state, [.3, .4, .5])
        np.testing.assert_allclose(result['coil_currents_a']['c2'], [.2, .3, .4], atol=1e-10)


    def test_drive_group_load_overrides_replace_model_rl_for_resolution(self):
        self.traces["wave_a"] = (
            [0.0, 1.0], [10.0, 10.0],
            {"source_kind": "voltage", "slew_rate_per_s": 0.0, "output_resistance_ohm": 0.0},
        )
        self.adapter.coil_electrical_load = Mock(return_value={
            "resistance_ohm": 2.0,
            "inductance_h": 1.0,
            "resistance_mode": "estimated",
            "inductance_mode": "estimated",
            "resistance_estimated_ohm": 2.0,
            "inductance_estimated_h": 1.0,
        })
        resolved = self.adapter.resolve_exposure_drive_group({
            "id": "drive_a",
            "waveform_id": "wave_a",
            "coil_ids": ["c1"],
            "connection": "series",
            "load_overrides": {
                "c1": {"resistance_ohm": 4.0, "inductance_h": 2.0},
            },
        })
        load = resolved["loads"][0]
        self.assertEqual(load["resistance_mode"], "drive_override")
        self.assertEqual(load["inductance_mode"], "drive_override")
        self.assertAlmostEqual(load["resistance_ohm"], 4.0)
        self.assertAlmostEqual(load["inductance_h"], 2.0)
        expected = 2.5 * (1.0 - math.exp(-2.0))
        self.assertAlmostEqual(resolved["source_current_a"][-1], expected, places=10)


    def test_streamed_exposure_uses_drive_group_load_overrides(self):
        self.traces["wave_a"] = (
            [0.0, 1.0], [10.0, 10.0],
            {"source_kind": "voltage", "slew_rate_per_s": 0.0, "output_resistance_ohm": 0.0},
        )
        self.adapter.coil_electrical_load = Mock(return_value={
            "resistance_ohm": 2.0,
            "inductance_h": 1.0,
            "resistance_mode": "estimated",
            "inductance_mode": "estimated",
            "resistance_estimated_ohm": 2.0,
            "inductance_estimated_h": 1.0,
        })
        state = normalize_exposure_protocol({
            "duration_s": 1.0,
            "sources": [{"id": "a", "name": "A", "waveform_id": "wave_a"}],
            "sequence": {"loop": False, "steps": [{
                "duration_s": 1.0,
                "drive_groups": [{
                    "id": "drive_a", "source_id": "a",
                    "coil_ids": ["c1"], "connection": "series",
                    "load_overrides": {"c1": {"resistance_ohm": 4.0, "inductance_h": 2.0}},
                }],
            }]},
        })
        result = self.adapter.resolve_exposure_currents_at_times(state, [0.0, 1.0])
        expected = 2.5 * (1.0 - math.exp(-2.0))
        self.assertAlmostEqual(result["coil_currents_a"]["c1"][-1], expected, places=10)

    def test_voltage_rl_state_survives_routing_cycle(self):
        self.traces['wave_a'] = ([0, 1], [1, 1], {'source_kind': 'voltage', 'slew_rate_per_s': 0.0})
        self.adapter.coil_electrical_load.return_value['inductance_h'] = 1.0
        state = protocol()
        state['sequence']['steps'][1]['drive_groups'] = [route('a', 'c1')]
        times = np.array([0, .3, .6, 1.2, 1.9])
        result = self.adapter.resolve_exposure_currents_at_times(state, times)
        np.testing.assert_allclose(result['coil_currents_a']['c1'], 1 - np.exp(-times), atol=1e-10)

    def test_preview_trace_is_scoped_but_full_statistics_source_is_not(self):
        first = _prepare_exposure_current_timeline(self.adapter, protocol(), coil_ids=['c1', 'c2'], start_s=0, stop_s=.2, frame_count=3, analysis_samples=7)
        second = _prepare_exposure_current_timeline(self.adapter, protocol(), coil_ids=['c1', 'c2'], start_s=0, stop_s=.8, frame_count=19, analysis_samples=55)
        self.assertAlmostEqual(first['frame_times'][-1], .2)
        self.assertAlmostEqual(first['analysis_times'][-1], .2)
        self.assertAlmostEqual(second['analysis_times'][-1], .8)
        self.assertEqual(first['statistics_scope'], 'full')
        self.assertEqual(first['statistics_duration_s'], 2.0)
        self.assertEqual(first['analysis_sample_count_estimate'], second['analysis_sample_count_estimate'])
        frozen_first = first['frozen_analysis_source_factory']()
        frozen_second = second['frozen_analysis_source_factory']()
        self.assertEqual(frozen_first['protocol']['duration_s'], 2.0)
        self.assertEqual(frozen_second['protocol']['duration_s'], 2.0)

    def test_preview_statistics_scope_limits_authoritative_analysis(self):
        timeline = _prepare_exposure_current_timeline(
            self.adapter, protocol(), coil_ids=['c1', 'c2'], start_s=0, stop_s=.2,
            frame_count=3, analysis_samples=7, statistics_scope='preview',
        )
        self.assertEqual(timeline['statistics_scope'], 'preview')
        self.assertAlmostEqual(timeline['statistics_duration_s'], .2)
        packets = list(timeline['analysis_packet_factory']())
        self.assertTrue(packets)
        self.assertLessEqual(max(float(np.max(packet[0])) for packet in packets), .2 + 1e-12)
        frozen = timeline['frozen_analysis_source_factory']()
        self.assertAlmostEqual(frozen['protocol']['duration_s'], .2)

    def test_long_preview_scope_reuses_verified_physical_cycle(self):
        timeline = _prepare_exposure_current_timeline(
            self.adapter, protocol(duration=15.0), coil_ids=['c1', 'c2'],
            start_s=0, stop_s=12.0, frame_count=5, analysis_samples=7,
            statistics_scope='preview',
        )
        self.assertEqual(timeline['statistics_scope'], 'preview')
        self.assertAlmostEqual(timeline['statistics_duration_s'], 12.0)
        self.assertTrue(timeline['analysis_compression']['enabled'])
        self.assertGreaterEqual(timeline['analysis_compression']['repeat_count'], 2)
        frozen = timeline['frozen_analysis_source_factory']()
        self.assertEqual(frozen['mode'], 'verified_packets')
        self.assertAlmostEqual(frozen['protocol']['duration_s'], 12.0)
        self.assertTrue(any(int(packet.get('repeat_count', 1)) > 1 for packet in frozen['packets']))

    def test_playback_only_timeline_does_not_attach_authoritative_analysis(self):
        timeline = _prepare_exposure_current_timeline(
            self.adapter, protocol(), coil_ids=['c1', 'c2'], start_s=0, stop_s=.2,
            frame_count=3, analysis_samples=7, statistics_scope='full',
            include_authoritative_analysis=False,
        )
        self.assertFalse(timeline['authoritative_analysis_enabled'])
        self.assertEqual(timeline['statistics_scope'], 'preview')
        self.assertAlmostEqual(timeline['statistics_duration_s'], .2)
        self.assertIsNone(timeline['analysis_chunk_factory'])
        self.assertIsNone(timeline['analysis_packet_factory'])
        self.assertIsNone(timeline['analysis_source_packet_factory'])
        self.assertIsNone(timeline['frozen_analysis_source_factory'])
        self.assertFalse(timeline['analysis_compression']['enabled'])
        self.assertLessEqual(float(timeline['analysis_times'][-1]), .2 + 1e-12)


    def test_preview_endpoint_on_routing_boundary_uses_new_step(self):
        timeline = _prepare_exposure_current_timeline(
            self.adapter, protocol(), coil_ids=['c1', 'c2'],
            start_s=0, stop_s=.3, frame_count=4, analysis_samples=7,
        )
        np.testing.assert_allclose(timeline['frame_currents'][-1], [0.0, 1.0])

    def test_frozen_uncompressed_analysis_recipe_replays_without_live_project(self):
        state = protocol(duration=2.0, repeat=False)
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=.3,
            frame_count=5, analysis_samples=9,
        )
        frozen = copy.deepcopy(timeline['frozen_analysis_source_factory']())
        self.assertEqual(frozen['mode'], 'replay_recipe')
        # Round-trip through the same JSON boundary used by saved maps, then
        # mutate the live adapter so replay can only succeed from frozen data.
        frozen = json.loads(json.dumps(frozen))
        self.adapter._waveform_source_trace = Mock(side_effect=AssertionError('live waveform accessed'))
        self.adapter.coil_electrical_load = Mock(side_effect=AssertionError('live load accessed'))

        factory = authoritative_analysis_packet_factory({'authoritative_analysis': frozen})
        replayed = list(factory())

        # Build a clean reference adapter from the original fixture values.
        reference = StudioAdapter({'objects': []})
        reference.coil_electrical_load = Mock(return_value={'resistance_ohm': 1.0, 'inductance_h': 0.01})
        reference._waveform_source_trace = lambda key: tuple(copy.deepcopy(x) for x in self.traces[key])
        expected_timeline = _prepare_exposure_current_timeline(
            reference, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=.3,
            frame_count=5, analysis_samples=9,
        )
        expected = list(expected_timeline['analysis_packet_factory']())
        self.assertEqual(len(replayed), len(expected))
        for actual, wanted in zip(replayed, expected, strict=True):
            np.testing.assert_allclose(actual[0], wanted[0], atol=0, rtol=0)
            np.testing.assert_allclose(actual[1], wanted[1], atol=1e-12, rtol=1e-12)

    def test_saved_nonperiodic_regional_analysis_replays_without_display_trace(self):
        state = protocol(duration=2.0, repeat=False)
        with patch('fieldworkbench.waveform.MAX_RETAINED_ANALYSIS_SAMPLES', 17):
            timeline = _prepare_exposure_current_timeline(
                self.adapter, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=.3,
                frame_count=5, analysis_samples=9,
            )
        self.assertTrue(timeline['analysis_trace_reduced'])

        region = BrainRegionNode('r', 'Region', (1,))
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.array([1]), np.array([1000.]),
            np.array([0]), {1: 1}, {1: 1}, 1000.,
        )
        fixed = np.zeros((1, 3))
        bases = np.array([[[1e-6, 0.0, 0.0]], [[0.0, 1e-6, 0.0]]])
        payload = waveform_basis_payload(
            fixed, bases, samples, region, timeline, ['c1', 'c2']
        )
        self.assertEqual(payload['authoritative_analysis']['mode'], 'replay_recipe')

        # Cross the saved-map JSON boundary and deliberately destroy the retained
        # display trace. Every reduced-map numerical tool must still use only the
        # self-contained authoritative replay recipe.
        payload = json.loads(json.dumps(payload))
        payload['times_b64'] = 'not-used-by-authoritative-analysis'
        payload['currents_b64'] = 'not-used-by-authoritative-analysis'
        payload['shape'] = [0, 2]

        summary = analyse_region_exposure_summary(payload, 'r')
        threshold = analyse_region_threshold_duration(payload, 'r', 0.1)
        timing = analyse_region_waveform(payload, 'r', 50.0)
        spectrum = analyse_region_spectrum(payload, 'r', fft_samples=256)
        weighted = analyse_region_frequency_weighted(payload, 'r', fft_samples=256)

        self.assertEqual(summary['analysis_source'], 'authoritative_packets')
        self.assertGreater(summary['b2_time_uT2_s'], 0.0)
        self.assertGreaterEqual(threshold['excursion_count'], 1)
        self.assertGreater(timing['peak_spatial_rms_slew_uT_s'], 0.0)
        self.assertEqual(spectrum['analysis_mode'], 'segmented_full_duration')
        self.assertEqual(weighted['analysis_mode'], 'segmented_full_duration')
        from fieldworkbench.region_analysis import analyse_regions_frequency_weighted
        payload['regions']['second'] = dict(payload['regions']['r'], name='Second')
        batch = analyse_regions_frequency_weighted(payload, fft_samples=256)
        for result in batch.values():
            np.testing.assert_allclose(result['weighted_index'], weighted['weighted_index'])
            np.testing.assert_allclose(result['bin_rms_uT'], weighted['bin_rms_uT'])

        self.assertTrue(np.isfinite(spectrum['bin_rms_uT']).all())
        self.assertTrue(np.isfinite(weighted['weighted_index']).all())

    def test_frozen_analysis_survives_saved_map_codec(self):
        state = protocol(duration=2.0, repeat=False)
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=.3,
            frame_count=5, analysis_samples=9,
        )
        frozen = timeline['frozen_analysis_source_factory']()
        content = {
            'measurement_regions': {
                'waveform': {'authoritative_analysis': frozen},
            },
        }
        payload_b64, raw_bytes, compressed_bytes = StudioAdapter._saved_map_payload_blob(
            content, {}
        )
        self.assertGreater(raw_bytes, 0)
        self.assertGreater(compressed_bytes, 0)
        decoded = StudioAdapter._decode_saved_map_payload({'payload_b64': payload_b64})
        self.assertEqual(
            decoded['content']['measurement_regions']['waveform']['authoritative_analysis'],
            frozen,
        )

    def test_frozen_verified_analysis_replays_weighted_packets_directly(self):
        state = normalize_exposure_protocol({
            'duration_s': 30.0,
            'sources': [{'id': 'a', 'name': 'A', 'waveform_id': 'wave_a'}],
            'sequence': {'loop': True, 'steps': [
                {'duration_s': 1.0, 'drive_groups': [route('a', 'c1')]},
            ]},
        })
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1'], start_s=0, stop_s=1,
            frame_count=11, analysis_samples=101,
        )
        frozen = timeline['frozen_analysis_source_factory']()
        self.assertTrue(timeline['analysis_compression']['enabled'])
        self.assertEqual(frozen['mode'], 'verified_packets')
        self.assertLessEqual(len(frozen['packets']), 3)
        frozen = json.loads(json.dumps(frozen))
        factory = authoritative_analysis_packet_factory({'authoritative_analysis': frozen})
        replayed = list(factory())
        expected = list(timeline['analysis_packet_factory']())
        self.assertEqual(len(replayed), len(expected))
        for actual, wanted in zip(replayed, expected, strict=True):
            self.assertEqual(actual[2], wanted[2] if len(wanted) == 3 else 1)
            np.testing.assert_allclose(actual[0], wanted[0], atol=0, rtol=0)
            np.testing.assert_allclose(actual[1], wanted[1], atol=1e-12, rtol=1e-12)

    def test_streaming_resolver_matches_legacy_short_run(self):
        times = np.linspace(0, 2, 137)
        legacy = self.adapter.resolve_exposure_currents_at_times(protocol(), times)
        streamed = next(self.adapter.iter_exposure_currents_at_times(protocol(), [times]))
        for coil_id in ['c1', 'c2']:
            np.testing.assert_allclose(
                streamed['coil_currents_a'][coil_id],
                legacy['coil_currents_a'][coil_id],
                atol=1e-12, rtol=1e-12,
            )

    def test_long_exposure_uses_bounded_retained_trace_and_streamed_analysis(self):
        # A sparse command recipe with one very short source interval forces the
        # authoritative analysis clock above the former one-million-sample cap,
        # while keeping this regression inexpensive to resolve.
        self.traces['wave_a'] = (
            [0, .001, 1], [0, 1, 0],
            {'source_kind': 'current', 'slew_rate_per_s': 0.0},
        )
        duration = 1200.0
        state = normalize_exposure_protocol({
            'duration_s': duration,
            'sources': [{'id': 'a', 'name': 'A', 'waveform_id': 'wave_a'}],
            'sequence': {'loop': True, 'steps': [
                {'duration_s': duration, 'drive_groups': [route('a', 'c1')]},
            ]},
        })
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1'], start_s=0, stop_s=.2,
            frame_count=3, analysis_samples=7,
        )
        self.assertGreater(timeline['analysis_sample_count_estimate'], MAX_ANALYSIS_SAMPLES)
        self.assertLess(len(timeline['analysis_times']), MAX_RETAINED_ANALYSIS_SAMPLES)
        self.assertAlmostEqual(timeline['analysis_times'][-1], .2)
        self.assertTrue(timeline['analysis_trace_reduced'])
        self.assertTrue(callable(timeline.get('analysis_chunk_factory')))
        first_times, first_currents = next(timeline['analysis_chunk_factory']())
        self.assertLessEqual(len(first_times), 2 * 4096 + 32)
        self.assertEqual(first_currents.shape, (len(first_times), 1))
        self.assertAlmostEqual(first_times[0], 0.0)

    def test_nonperiodic_retained_trace_reports_chunked_timeline_progress(self):
        self.traces['wave_a'] = (
            [0, .001, 1], [0, 1, 0],
            {'source_kind': 'current', 'slew_rate_per_s': 0.0},
        )
        state = protocol(duration=3.0, repeat=False)
        updates = []
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=.2,
            frame_count=3, analysis_samples=7, progress_callback=updates.append,
        )
        expected = self.adapter.resolve_exposure_currents_at_times(
            state, timeline['analysis_times']
        )
        expected_matrix = np.column_stack([
            expected['coil_currents_a']['c1'], expected['coil_currents_a']['c2']
        ])
        np.testing.assert_allclose(timeline['analysis_currents'], expected_matrix)
        chunk_updates = [
            update for update in updates
            if update.get('render_task') == 'prepare_playback'
            and 'chunks' in str(update.get('detail', ''))
        ]
        self.assertGreaterEqual(len(chunk_updates), 1)
        self.assertIn('preview display trace', str(chunk_updates[-1].get('stage', '')).lower())
        self.assertLess(int(chunk_updates[-1]['completed']), int(chunk_updates[-1]['total']))

    def test_invalid_source_reference_rejected_and_shared_source_groups_allowed(self):
        state = protocol()
        state['sequence']['steps'][0]['drive_groups'][0]['source_id'] = 'missing'
        with self.assertRaises(ExposureError):
            normalize_exposure_protocol(state)
        state = protocol()
        state['sequence']['steps'][0]['drive_groups'].append(route('a', 'c3'))
        normalized = normalize_exposure_protocol(state)
        self.assertEqual(len(normalized['sequence']['steps'][0]['drive_groups']), 2)
        self.assertEqual(
            [group['source_id'] for group in normalized['sequence']['steps'][0]['drive_groups']],
            ['a', 'a'],
        )

    def test_full_region_summary_includes_peak_after_preview(self):
        region = BrainRegionNode('r', 'Region', (1,))
        samples = BrainSampleSet(np.zeros((1, 3)), np.array([1]), np.array([1000.]), np.array([0]), {1: 1}, {1: 1}, 1000.)
        fixed = np.zeros((1, 3))
        bases = np.array([[[1e-6, 0, 0]]])
        times = np.linspace(0, 2, 1025)
        currents = times[:, None]  # One microtesla per second.
        preview_times = np.array([0., .2])
        preview = lpba40_frame_region_metrics(fixed, bases, preview_times[:, None], samples, region)
        row = lpba40_playback_region_metrics(fixed, bases, currents, times, preview, preview_times, samples, region, full_timeline=True)['r']
        self.assertAlmostEqual(row.mean_uT, 1.0, places=6)
        self.assertAlmostEqual(row.absolute_peak_uT, 2.0, places=6)
        self.assertAlmostEqual(row.peak_time_s, 2.0)
        self.assertAlmostEqual(row.b_time_uT_s, 2.0, places=6)
        self.assertAlmostEqual(row.b2_time_uT2_s, 8/3, places=5)
        self.assertAlmostEqual(row.playback_rms_uT, math.sqrt(4/3), places=5)

    def test_streamed_scalar_statistics_match_array_statistics(self):
        times = np.linspace(0, 2, 1025)
        values = (0.5 + np.sin(2 * np.pi * 3 * times))[:, None]

        def chunks():
            for start in range(0, len(times), 113):
                stop = min(len(times), start + 113)
                yield times[start:stop], values[start:stop]

        streamed = _streaming_series_statistics(
            {'analysis_chunk_factory': chunks}, lambda currents: currents
        )[0]
        legacy = time_statistics(times, values[:, 0])
        for name in ('mean', 'rms', 'pk_pk'):
            self.assertAlmostEqual(streamed[name], legacy[name], places=12)

    def test_streamed_region_metrics_match_array_metrics(self):
        region = BrainRegionNode('r', 'Region', (1,))
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.array([1]), np.array([1000.]),
            np.array([0]), {1: 1}, {1: 1}, 1000.,
        )
        fixed = np.zeros((1, 3))
        bases = np.array([[[1e-6, 0, 0]]])
        times = np.linspace(0, 2, 1025)
        currents = times[:, None]
        preview_times = np.array([0., .2])
        preview = lpba40_frame_region_metrics(
            fixed, bases, preview_times[:, None], samples, region
        )
        legacy = lpba40_playback_region_metrics(
            fixed, bases, currents, times, preview, preview_times, samples, region,
            full_timeline=True,
        )['r']

        def chunks():
            for start in range(0, len(times), 127):
                stop = min(len(times), start + 127)
                yield times[start:stop], currents[start:stop]

        streamed = lpba40_playback_region_metrics_streaming(
            fixed, bases, chunks, samples, region, analysis_sample_count=len(times)
        )['r']
        for name in (
            'playback_rms_uT', 'mean_uT', 'max_p95_uT', 'absolute_peak_uT',
            'b_time_uT_s', 'b2_time_uT2_s', 'peak_time_s',
        ):
            self.assertAlmostEqual(getattr(streamed, name), getattr(legacy, name), places=10)


    def test_periodic_compression_handles_mismatched_source_and_routing_periods(self):
        state = protocol(duration=30.0)
        # Source period is 1.0 s while the routing cycle is 0.6 s. Their true
        # combined repeat period is 3.0 s; this used to expose floating-point
        # disagreements when absolute routing boundaries were compared.
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=1,
            frame_count=11, analysis_samples=101,
        )
        compression = timeline['analysis_compression']
        self.assertTrue(compression['enabled'])
        self.assertAlmostEqual(compression['period_s'], 3.0, places=12)
        self.assertGreaterEqual(compression['repeat_count'], 2)
        packets = list(timeline['analysis_packet_factory']())
        self.assertTrue(any(len(packet) == 3 and packet[2] > 1 for packet in packets))

    def test_compressed_scalar_statistics_match_explicit_stream(self):
        state = protocol(duration=12.0)
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=.5,
            frame_count=7, analysis_samples=101,
        )
        self.assertTrue(timeline['analysis_compression']['enabled'])
        compressed = _streaming_series_statistics(
            timeline, lambda currents: currents
        )
        explicit = _streaming_series_statistics(
            {'analysis_chunk_factory': timeline['analysis_chunk_factory']},
            lambda currents: currents,
        )
        self.assertEqual(len(compressed), len(explicit))
        for compressed_row, explicit_row in zip(compressed, explicit):
            for name in ('mean', 'rms', 'pk_pk'):
                self.assertAlmostEqual(
                    compressed_row[name], explicit_row[name], places=8
                )

    def test_compressed_region_metrics_match_explicit_stream(self):
        state = protocol(duration=12.0)
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1', 'c2'], start_s=0, stop_s=.5,
            frame_count=7, analysis_samples=101,
        )
        self.assertTrue(timeline['analysis_compression']['enabled'])
        region = BrainRegionNode('r', 'Region', (1,))
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.array([1]), np.array([1000.]),
            np.array([0]), {1: 1}, {1: 1}, 1000.,
        )
        fixed = np.zeros((1, 3))
        bases = np.array([[[1e-6, 0, 0]], [[0, 1e-6, 0]]])
        compressed = lpba40_playback_region_metrics_streaming(
            fixed, bases, timeline['analysis_packet_factory'], samples, region
        )['r']
        explicit = lpba40_playback_region_metrics_streaming(
            fixed, bases, timeline['analysis_chunk_factory'], samples, region
        )['r']
        for name in (
            'playback_rms_uT', 'mean_uT', 'max_p95_uT', 'absolute_peak_uT',
            'b_time_uT_s', 'b2_time_uT2_s', 'peak_time_s',
        ):
            self.assertAlmostEqual(
                getattr(compressed, name), getattr(explicit, name), places=7
            )

    def test_fast_rl_transient_settles_then_compresses(self):
        self.traces['wave_a'] = (
            [0, 1], [1, 1],
            {'source_kind': 'voltage', 'slew_rate_per_s': 0.0},
        )
        self.adapter.coil_electrical_load.return_value = {
            'resistance_ohm': 1.0, 'inductance_h': 0.05,
        }
        state = normalize_exposure_protocol({
            'duration_s': 30.0,
            'sources': [{'id': 'a', 'name': 'A', 'waveform_id': 'wave_a'}],
            'sequence': {'loop': True, 'steps': [
                {'duration_s': 1.0, 'drive_groups': [route('a', 'c1')]},
            ]},
        })
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1'], start_s=0, stop_s=1,
            frame_count=11, analysis_samples=101,
        )
        self.assertTrue(timeline['analysis_compression']['enabled'])
        self.assertGreater(timeline['analysis_compression']['stable_start_s'], 0.0)

    def test_slow_rl_transient_falls_back_to_streaming(self):
        self.traces['wave_a'] = (
            [0, 1], [1, 1],
            {'source_kind': 'voltage', 'slew_rate_per_s': 0.0},
        )
        self.adapter.coil_electrical_load.return_value = {
            'resistance_ohm': 1.0, 'inductance_h': 100.0,
        }
        state = normalize_exposure_protocol({
            'duration_s': 30.0,
            'sources': [{'id': 'a', 'name': 'A', 'waveform_id': 'wave_a'}],
            'sequence': {'loop': True, 'steps': [
                {'duration_s': 1.0, 'drive_groups': [route('a', 'c1')]},
            ]},
        })
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1'], start_s=0, stop_s=1,
            frame_count=11, analysis_samples=101,
        )
        self.assertFalse(timeline['analysis_compression']['enabled'])

    def test_twenty_hz_thirty_minute_exposure_compresses_logical_samples(self):
        source_times = np.linspace(0.0, 0.05, 257)
        self.traces['wave_a'] = (
            source_times.tolist(),
            np.sin(2 * np.pi * 20 * source_times).tolist(),
            {'source_kind': 'current', 'slew_rate_per_s': 0.0},
        )
        state = normalize_exposure_protocol({
            'duration_s': 1800.0,
            'sources': [{'id': 'a', 'name': 'A', 'waveform_id': 'wave_a'}],
            'sequence': {'loop': True, 'steps': [
                {'duration_s': 1.0, 'drive_groups': [route('a', 'c1')]},
            ]},
        })
        timeline = _prepare_exposure_current_timeline(
            self.adapter, state, coil_ids=['c1'], start_s=0, stop_s=1,
            frame_count=31, analysis_samples=101,
        )
        compression = timeline['analysis_compression']
        self.assertTrue(compression['enabled'])
        self.assertAlmostEqual(compression['period_s'], 1.0, places=12)
        self.assertEqual(compression['stable_start_s'], 1.0)
        self.assertGreater(compression['repeat_count'], 1700)
        self.assertGreater(timeline['analysis_sample_count_estimate'], 10_000_000)
        self.assertLess(len(timeline['analysis_times']), MAX_RETAINED_ANALYSIS_SAMPLES)
        self.assertAlmostEqual(timeline['analysis_times'][-1], 1.0)
        frozen = timeline['frozen_analysis_source_factory']()
        self.assertEqual(frozen['mode'], 'verified_packets')
        frozen_rows = sum(int(packet['shape'][0]) for packet in frozen['packets'])
        self.assertLess(frozen_rows, 100_000)
        self.assertLess(len(json.dumps(frozen).encode('utf-8')), 2_000_000)

    def test_metadata_reports_two_durations(self):
        trace = validate_trace([0, .2], [0, 0], kind='current')
        data = _brain_playback_run_metadata(trace, frame_count=3, analysis_sample_count=1025, summary_sample_count=3, atlas_sample_count=1,
            waveform_metadata={'mode': 'exposure', 'physical_duration_s': 2.0},
            sequence_steps=[{'duration_s': .5}], sequence_loop=True)
        self.assertEqual(data['exposure_time_s'], 2)
        self.assertEqual(data['preview_duration_s'], .2)
        self.assertEqual(data['statistics_scope'], 'full')
        self.assertEqual(data['statistics_duration_s'], 2.0)
        self.assertEqual(data['sequencer_full_loops'], 4)


if __name__ == '__main__':
    unittest.main()
