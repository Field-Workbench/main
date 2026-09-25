"""Phase 2: independent numerical parity, shared processing and grouped persistence."""
import unittest
from unittest.mock import patch
import numpy as np
from fieldworkbench import region_analysis as ra
from fieldworkbench.studio_adapter import StudioAdapter
import test_frequency_weighted_batch as fixtures


class RegionalBatchTests(unittest.TestCase):
    def assert_result(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for field, value in expected.items():
            if isinstance(value, (str, list)):
                self.assertEqual(actual[field], value, field)
            else:
                np.testing.assert_allclose(actual[field], value, rtol=1e-11, atol=1e-12,
                                           err_msg=field)

    def test_summary_timing_parity_and_shared_source_passes(self):
        for mode in ('whole_record', 'verified_periodic_cycle', 'segmented_full_duration'):
            p = fixtures.BatchWeightingTests().payload(mode)
            # Include a nonzero static field and cross terms.
            p['regions']['left'].update(constant=4e-12, linear=[.1e-12, -.2e-12])
            for batch, single, passes in ((ra.analyse_regions_exposure_summary,
                                          ra.analyse_region_exposure_summary, 1),
                                         (ra.analyse_regions_waveform, ra.analyse_region_waveform, 2)):
                with self.subTest(mode=mode, function=batch.__name__):
                    expected = {k: single(p, k) for k in p['regions']}
                    with patch.object(ra, '_iter_time_domain_packets', wraps=ra._iter_time_domain_packets) as packets:
                        actual = batch(p)
                        self.assertEqual(packets.call_count, passes)
                    for key in expected:
                        self.assert_result(actual[key], expected[key])

    def test_packet_gaps_singletons_and_repeat_boundaries(self):
        from test_region_time_domain_streaming import _verified_payload, _encode
        p = _verified_payload([0., 2., 0.], repeat_count=4)
        p['regions']['other'] = dict(p['regions']['r'], quadratic=[[4e-12]])
        p['authoritative_analysis']['packets'].extend([
            dict(times_b64=_encode([3.]), currents_b64=_encode([[8.]]), shape=[1, 1], repeat_count=1),
            dict(times_b64=_encode([4., 4.5, 5.]), currents_b64=_encode([[0.], [1.], [0.]]),
                 shape=[3, 1], repeat_count=3)])
        for batch, single in ((ra.analyse_regions_exposure_summary, ra.analyse_region_exposure_summary),
                              (ra.analyse_regions_waveform, ra.analyse_region_waveform)):
            actual = batch(p)
            for key in p['regions']:
                self.assert_result(actual[key], single(p, key))

    def test_spectrum_parity_and_shared_ffts(self):
        for mode in ('whole_record', 'verified_periodic_cycle', 'segmented_full_duration'):
            p = fixtures.BatchWeightingTests().payload(mode)
            p['regions']['left'].update(constant=4e-12, linear=[.1e-12, -.2e-12])
            for window, detrend in (('rectangular', True), ('hann', False)):
                settings = dict(fft_samples=256, window=window, detrend=detrend)
                with self.subTest(mode=mode, window=window):
                    with patch.object(ra.np.fft, 'rfft', wraps=np.fft.rfft) as fft:
                        expected = {k: ra.analyse_region_spectrum(p, k, **settings) for k in p['regions']}
                        separate = fft.call_count
                    with patch.object(ra.np.fft, 'rfft', wraps=np.fft.rfft) as fft:
                        actual = ra.analyse_regions_spectrum(p, **settings)
                        self.assertEqual(fft.call_count * len(p['regions']), separate)
                    for key in expected:
                        self.assert_result(actual[key], expected[key])

    def test_grouped_cache_codec_and_catalogue(self):
        p = fixtures.BatchWeightingTests().payload('whole_record')
        cache = {('summary', '__all_regions__'): {'regions': ra.analyse_regions_exposure_summary(p)},
                 ('spectrum', '__all_regions__', 256, 'hann', True):
                    {'regions': ra.analyse_regions_spectrum(p, fft_samples=256, window='hann')}}
        payload_blob = StudioAdapter._exposure_analysis_value_blob(p)[0]
        cache_blob = StudioAdapter._exposure_analysis_cache_blob(cache)[0]
        _, restored = StudioAdapter._decode_exposure_analysis_payload(
            dict(payload_b64=payload_blob, cache_b64=cache_blob))
        self.assertEqual(set(restored), set(cache))
        for key in cache:
            for region in p['regions']:
                self.assert_result(restored[key]['regions'][region], cache[key]['regions'][region])
        summary = StudioAdapter._exposure_analysis_cache_summary(restored)
        self.assertEqual(summary['cached_result_count'], 2)
        self.assertEqual(summary['cached_region_count'], 4)
        self.assertEqual(summary['cached_tool_counts'], {'summary': 1, 'spectrum': 1})

    def test_invalid_region_rejects_whole_run(self):
        for batch in (ra.analyse_regions_exposure_summary, ra.analyse_regions_spectrum,
                      ra.analyse_regions_waveform):
            p = fixtures.BatchWeightingTests().payload('whole_record')
            p['regions']['right']['quadratic'] = [[float('nan')]]
            with self.assertRaises(ValueError):
                batch(p)
