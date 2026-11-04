"""Playback region coverage, physical-time statistics and vector spectra."""
import base64
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from fieldworkbench.brain_analysis import BrainRegionNode, BrainSampleSet
from fieldworkbench.region_analysis import (
    analyse_region_exposure_summary, analyse_region_frequency_weighted, analyse_region_spectrum,
    analyse_region_threshold_duration, analyse_region_waveform,
    frequency_weighting_reference_uT, measurement_playback_payload, waveform_basis_payload,
)
from fieldworkbench.studio_adapter import StudioAdapter
from fieldworkbench.waveform import build_2d_gpu_playback, build_3d_gpu_playback, validate_trace
from workflow_fixture import compact_workflow_scene


def decode_frames(result):
    p = result["metrics"]
    return np.frombuffer(base64.b64decode(p["values_b64"]), dtype=np.float32).reshape(p["shape"])


def adapter_with_regions():
    adapter = StudioAdapter(compact_workflow_scene())
    adapter.set_coil_modeling_method("centreline")
    for _ in range(2):
        obj = adapter.add_template("guide_box")
        adapter.apply_operations([
            {"method": "set_geometry_style", "params": {"object_id": obj, "role": "measurement"}},
            {"method": "set_measurement_settings", "params": {"object_id": obj, "quality": "preview"}},
        ])
    adapter.add_template("well_plate_24")
    return adapter


class RegionAnalysisTests(unittest.TestCase):
    def test_reduced_trace_without_authoritative_source_fails_cleanly(self):
        times = np.linspace(0.0, 1800.0, 17)
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.ones(1, dtype=int), np.ones(1),
            np.arange(1), {1: 1}, {1: 1}, 1,
        )
        payload = waveform_basis_payload(
            np.zeros((1, 3)), np.array([[[1e-6, 0, 0]]]), samples,
            BrainRegionNode('r', 'R', (1,)),
            {'analysis_times': times, 'analysis_currents': np.zeros((17, 1)),
             'analysis_trace_reduced': True, 'analysis_spacing_s': .001,
             'analysis_compression': {'enabled': True, 'period_s': 1.0}},
            ['coil'],
        )
        payload = json.loads(json.dumps(payload))
        self.assertTrue(payload['analysis_trace_reduced'])
        self.assertEqual(payload['analysis_compression']['period_s'], 1.0)
        for analyse, args in (
            (analyse_region_spectrum, ()),
            (analyse_region_frequency_weighted, ()),
            (analyse_region_waveform, ()),
            (analyse_region_threshold_duration, (1.0,)),
            (analyse_region_exposure_summary, ()),
        ):
            with self.subTest(analyse=analyse.__name__), \
                 self.assertRaisesRegex(ValueError, 'authoritative analysis source'):
                analyse(payload, 'r', *args)

    def test_authoritative_analysis_source_survives_regional_payload_json(self):
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.ones(1, dtype=int), np.ones(1),
            np.arange(1), {1: 1}, {1: 1}, 1,
        )
        frozen = {
            'schema': 'fieldworkbench.authoritative-exposure-analysis',
            'version': 1,
            'kind': 'exposure',
            'mode': 'replay_recipe',
            'protocol': {'duration_s': 1.0},
            'drive_ids': ['coil'],
            'analysis_spacing_s': .001,
            'analysis_sample_count_estimate': 1001,
            'compression': {'enabled': False},
            'source_traces': {},
            'coil_loads': {},
        }
        payload = waveform_basis_payload(
            np.zeros((1, 3)), np.array([[[1e-6, 0, 0]]]), samples,
            BrainRegionNode('r', 'R', (1,)),
            {'analysis_times': np.array([0.0, .5, 1.0]),
             'analysis_currents': np.zeros((3, 1)),
             'frozen_analysis_source': frozen},
            ['coil'],
        )
        restored = json.loads(json.dumps(payload))
        self.assertEqual(restored['authoritative_analysis'], frozen)
        self.assertEqual(restored['statistics_scope'], 'full')
        self.assertEqual(restored['statistics_duration_s'], 1.0)

    def test_overlapping_volumes_independent_and_custom_points_explicit(self):
        adapter = adapter_with_regions()
        times = np.array([0., 0.2, 1.])
        currents = np.array([[0.], [1.], [2.]])
        timeline = {"frame_times": times, "frame_currents": currents,
                    "analysis_times": times, "analysis_currents": currents}
        # A uniform 1 µT/A field makes the numerical result independent of geometry.
        def uniform(_adapter, *, points_m, **kwargs):
            assert kwargs["field"] == "B"
            return np.zeros_like(points_m), np.tile([1e-6, 0, 0], (1, len(points_m), 1))
        with patch('fieldworkbench.waveform.field_bases_for_drives', side_effect=uniform) as solve:
            result = measurement_playback_payload(adapter, timeline, ['coil_a'], ['coil_a'])
        assert solve.call_count == 1
        assert len(result['objects']) == 3
        assert result['metrics']['shape'] == [3, 3, 7]  # No synthetic overlap total.
        assert result['objects'][2]['sampling_mode'] == 'user_defined'
        assert result['objects'][2]['name'].endswith('[samples]')
        frames = decode_frames(result)
        np.testing.assert_allclose(frames[:, :, 0], np.tile([0., 1., 2.], (3, 1)).T)
        summary = result['metrics']['playback_summary']
        values = np.frombuffer(base64.b64decode(summary['values_b64']),dtype=np.float64).reshape(summary['shape'])
        np.testing.assert_allclose(values[:, 4], 1.3)  # Uneven physical dt, trapezoidal integral.
        np.testing.assert_allclose(values[:, 5], 2.1)
        np.testing.assert_allclose(values[:, 0], np.sqrt(2.1))
        frozen = copy.deepcopy(result)
        adapter.add_template('well_plate_96')
        assert result == frozen


    def test_scripted_builders_do_not_attach_measurement_volume_snapshots(self):
        adapter = adapter_with_regions()
        common = dict(coil_id='coil_a', trace=validate_trace([0, .5, 1], [0, .02, 0], kind='current'),
                      frame_count=5, analysis_samples=9, playback_fps=30.)
        settings = dict(plane='xy',offset_mm=0.,span_mm=30.,resolution=2,field='H',component='magnitude',show_object_outlines=False)
        a = build_2d_gpu_playback(adapter,map_settings=settings,**common)
        settings.update(plane='yz',offset_mm=80.)
        b = build_2d_gpu_playback(adapter,map_settings=settings,**common)
        volume = dict(centre_mm=[30,0,0],span_mm=20,slice_axis='x',slice_count=2,resolution=2,field='B',
                      component='magnitude',include_scene=False,view_type='slices',waveform_render_mode='slices')
        c = build_3d_gpu_playback(adapter,map_settings=volume,**common)
        self.assertIsNone(a['measurement_regions'])
        self.assertIsNone(b['measurement_regions'])
        self.assertIsNone(c['measurement_regions'])


    def test_live_measurement_metrics_follow_the_same_frames_in_2d_and_3d(self):
        adapter = adapter_with_regions()
        common = dict(
            coil_id="coil_a", trace=validate_trace([0, .5, 1], [0, .02, 0], kind="current"),
            frame_count=5, analysis_samples=9, playback_fps=30.,
        )
        analysis = dict(resolution=2, component="magnitude", live_measurement_analysis=True)
        planar = build_2d_gpu_playback(
            adapter, **common, map_settings=dict(
                analysis, plane="xy", offset_mm=0., span_mm=30., field="H",
                show_object_outlines=False,
            ),
        )["measurement_regions"]
        volume = build_3d_gpu_playback(
            adapter, **common, map_settings=dict(
                analysis, centre_mm=[30, 0, 0], span_mm=20, slice_axis="x", slice_count=2,
                field="B", include_scene=False, view_type="slices", waveform_render_mode="slices",
            ),
        )["measurement_regions"]
        self.assertEqual(planar["metrics"]["shape"], [5, 27, 7])
        plate = planar["objects"][-1]
        self.assertEqual(plate["sample_count"], 24)
        self.assertEqual(len(plate["children"]), 24)
        self.assertEqual(plate["children"][0]["name"], "A1")
        highlights = {row["label_id"]: row["positions"] for row in planar["highlights"]}
        self.assertEqual(len(highlights[plate["label_id"]]), 24)
        first_well = plate["children"][0]
        self.assertEqual(highlights[first_well["label_id"]], [first_well["position_mm"]])
        self.assertNotIn("playback_summary", planar["metrics"])
        self.assertNotIn("waveform", planar)
        values = decode_frames(planar)
        np.testing.assert_allclose(values, decode_frames(volume))
        self.assertGreater(values[2, -1, 0], values[0, -1, 0])
        np.testing.assert_allclose(values[:, 2, 0], np.sqrt(np.mean(values[:, 3:, 0] ** 2, axis=1)))
        self.assertTrue(np.isnan(values[:, 3:, 6]).all())

    def test_vector_spectrum_retains_opposed_fields_and_physical_slew(self):
        t=np.linspace(0,1,1025)
        samples=BrainSampleSet(np.zeros((2,3)),np.ones(2,dtype=int),np.ones(2),np.arange(2),{1:2},{1:2},1)
        root=BrainRegionNode('r','Opposed samples',(1,))
        bases=np.array([[[1e-6,0,0],[-1e-6,0,0]]])
        payload=waveform_basis_payload(np.zeros((2,3)),bases,samples,root,
                 {'analysis_times':t,'analysis_currents':np.sin(2*np.pi*10*t)[:,None]},['coil'])
        result=analyse_region_waveform(payload,'r')
        spectrum=analyse_region_spectrum(payload,'r')
        peak=np.argmax(spectrum['bin_rms_uT'])
        assert spectrum['frequency_hz'][peak] == 10
        self.assertAlmostEqual(spectrum['bin_rms_uT'][peak], 1 / np.sqrt(2))
        self.assertAlmostEqual(result['peak_spatial_rms_slew_uT_s'], 20 * np.pi, delta=20 * np.pi * 0.001)
        self.assertAlmostEqual(spectrum['dc_rms_uT'], 0.0, delta=1e-12)
        assert result['drive_timing'][0]['pulses'] == 20



    def test_custom_fft_samples_window_and_detrending(self):
        t=np.linspace(0,1,1025)
        samples=BrainSampleSet(np.zeros((1,3)),np.ones(1,dtype=int),np.ones(1),np.arange(1),{1:1},{1:1},1)
        root=BrainRegionNode('r','R',(1,))
        bases=np.array([[[1e-6,0,0]]])
        payload=waveform_basis_payload(np.array([[2e-6,0,0]]),bases,samples,root,
                 {'analysis_times':t,'analysis_currents':(1+np.sin(2*np.pi*10*t))[:,None]},['coil'])
        result=analyse_region_spectrum(payload,'r',fft_samples=512,window='hann',detrend=True)
        assert result['fft_samples'] == 512
        assert result['fft_bins'] == 257
        assert result['max_fft_samples'] == 1024
        self.assertAlmostEqual(result['frequency_resolution_hz'], 1.0)
        self.assertAlmostEqual(result['nyquist_hz'], 256.0)
        assert result['window'] == 'hann'
        assert result['detrended'] is True
        self.assertAlmostEqual(result['dc_rms_uT'], 3.0)
        peak=np.argmax(result['bin_rms_uT'][1:])+1
        self.assertAlmostEqual(result['frequency_hz'][peak], 10.0)
        # Energy-normalized windows redistribute a coherent tone between bins while
        # preserving the combined one-sided RMS contribution.
        self.assertAlmostEqual(np.sqrt(np.sum(result['bin_rms_uT'][1:] ** 2)), 1 / np.sqrt(2), delta=(1 / np.sqrt(2)) * 1e-6)

        full=analyse_region_spectrum(payload,'r',window='rectangular',detrend=False)
        self.assertAlmostEqual(full['bin_rms_uT'][0], 3.0)
        with self.assertRaisesRegex(ValueError, 'between 2 and 1,024'):
            analyse_region_spectrum(payload, 'r', fft_samples=2048)


    def test_threshold_duration_and_spatial_rms_summary_use_physical_time(self):
        times = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        currents = np.array([[0.0], [2.0], [2.0], [0.0]], dtype=np.float64)
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.ones(1, dtype=int), np.ones(1),
            np.arange(1), {1: 1}, {1: 1}, 1
        )
        payload = waveform_basis_payload(
            np.zeros((1, 3)), np.array([[[1e-6, 0.0, 0.0]]]), samples,
            BrainRegionNode('r', 'R', (1,)),
            {'analysis_times': times, 'analysis_currents': currents}, ['coil']
        )
        threshold = analyse_region_threshold_duration(payload, 'r', 1.0)
        self.assertAlmostEqual(threshold['time_above_s'], 2.0)
        self.assertAlmostEqual(threshold['percent_above'], 200.0 / 3.0)
        self.assertAlmostEqual(threshold['longest_interval_s'], 2.0)
        self.assertEqual(threshold['excursion_count'], 1)
        self.assertAlmostEqual(threshold['intervals_s'][0][0], 0.5)
        self.assertAlmostEqual(threshold['intervals_s'][0][1], 2.5)

        summary = analyse_region_exposure_summary(payload, 'r')
        self.assertAlmostEqual(summary['duration_s'], 3.0)
        self.assertAlmostEqual(summary['b_time_uT_s'], 4.0)
        self.assertAlmostEqual(summary['b2_time_uT2_s'], 8.0)
        self.assertAlmostEqual(summary['playback_rms_uT'], np.sqrt(8.0 / 3.0))

    def test_frequency_weighted_single_tone_matches_reference_ratio(self):
        t = np.linspace(0.0, 1.0, 10001)
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.ones(1, dtype=int), np.ones(1),
            np.arange(1), {1: 1}, {1: 1}, 1
        )
        root = BrainRegionNode('r', 'R', (1,))
        bases = np.array([[[1e-6, 0, 0]]])
        # 100 µT RMS at 10 Hz with a 1 µT/A field basis.  ICNIRP 2010
        # general-public B reference level is 500 µT RMS at 10 Hz, so both
        # the weighted-peak and single-line spectral-sum indices are 0.2.
        currents = (100.0 * np.sqrt(2.0) * np.sin(2 * np.pi * 10 * t))[:, None]
        payload = waveform_basis_payload(
            np.zeros((1, 3)), bases, samples, root,
            {'analysis_times': t, 'analysis_currents': currents}, ['coil']
        )
        result = analyse_region_frequency_weighted(
            payload, 'r', profile='icnirp2010_public'
        )
        self.assertAlmostEqual(
            frequency_weighting_reference_uT(np.array([10.0]), 'icnirp2010_public')[0],
            500.0,
        )
        self.assertAlmostEqual(result['weighted_peak_index'], 0.2, delta=2e-6)
        self.assertAlmostEqual(result['spectral_sum_index'], 0.2, delta=2e-6)
        self.assertEqual(result['reference_rms_uT'], 200.0)

    def test_frequency_weighting_profiles_match_tabulated_breakpoints(self):
        public = frequency_weighting_reference_uT(
            np.array([1.0, 8.0, 25.0, 50.0, 400.0, 3000.0, 100000.0]),
            'icnirp2010_public',
        )
        np.testing.assert_allclose(
            public, [40000.0, 625.0, 200.0, 200.0, 200.0, 27.0, 27.0]
        )
        occupational = frequency_weighting_reference_uT(
            np.array([1.0, 8.0, 25.0, 300.0, 3000.0, 100000.0]),
            'icnirp2010_occupational',
        )
        np.testing.assert_allclose(
            occupational, [200000.0, 3125.0, 1000.0, 1000.0, 100.0, 100.0]
        )

    def test_dc_and_pulse_gaps(self):
        times=np.arange(7.)
        currents=np.array([0,0,1,1,0,0,0.])[:,None]
        samples=BrainSampleSet(np.zeros((1,3)),np.ones(1,dtype=int),np.ones(1),np.arange(1),{1:1},{1:1},1)
        payload=waveform_basis_payload(np.array([[0,2e-6,0]]),np.array([[[1e-6,0,0]]]),samples,
                 BrainRegionNode('r','R',(1,)),{'analysis_times':times,'analysis_currents':currents},['coil'])
        result=analyse_region_waveform(payload,'r',75)
        timing=result['drive_timing'][0]
        assert timing['pulses'] == 1
        assert timing['active_s'] == 1.5
        assert timing['inactive_s'] == 4.5
        self.assertAlmostEqual(result['peak_spatial_rms_slew_uT_s'], 1.0)


if __name__ == "__main__":
    unittest.main()
