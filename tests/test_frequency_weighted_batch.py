"""Batch weighting must match independent regional runs and share transforms."""
import unittest
from unittest.mock import patch
import numpy as np
from fieldworkbench import region_analysis as ra
from test_region_frequency_weighting_streaming import _base_payload, _packet_source, _encode


class BatchWeightingTests(unittest.TestCase):
    def payload(self, mode):
        times = np.linspace(0, 2, 2049)
        currents = np.column_stack((np.sin(2*np.pi*13*times) + .3*np.cos(2*np.pi*7*times),
                                    .7*np.sin(2*np.pi*19*times + .4)))
        p = _base_payload(2)
        p['drive_ids'] = ['a', 'b']
        p['regions'] = {}
        for name, basis in [('whole', [[1, .4], [.4, .8]]),
                            ('left', [[1, -.5], [-.5, 1]]),
                            ('right', [[.1, .1], [.1, 2]]),
                            ('zero', [[0, 0], [0, 0]])]:
            p['regions'][name] = dict(name=name, constant=0., linear=[0., 0.],
                                     quadratic=(np.array(basis)*1e-12).tolist())
        if mode == 'whole_record':
            p.update(times_b64=_encode(times), currents_b64=_encode(currents),
                     shape=list(currents.shape), analysis_trace_reduced=False)
        else:
            source = _packet_source(times, currents, 20 if mode == 'verified_periodic_cycle' else 2,
                                    repeat_count=10 if mode == 'verified_periodic_cycle' else 1)
            source['drive_ids'] = p['drive_ids']
            p['authoritative_analysis'] = source
        return p

    def test_all_modes_profiles_match_and_share_ffts(self):
        for mode in ('whole_record', 'verified_periodic_cycle', 'segmented_full_duration'):
            p = self.payload(mode)
            for profile in ra.FREQUENCY_WEIGHTING_PROFILES:
                with self.subTest(mode=mode, profile=profile):
                    with patch.object(ra.np.fft, 'rfft', wraps=np.fft.rfft) as fft:
                        expected = {k: ra.analyse_region_frequency_weighted(p, k, profile=profile,
                                                fft_samples=256) for k in p['regions']}
                        separate_count = fft.call_count
                    with patch.object(ra.np.fft, 'rfft', wraps=np.fft.rfft) as fft:
                        actual = ra.analyse_regions_frequency_weighted(p, profile=profile, fft_samples=256)
                        self.assertEqual(fft.call_count * len(p['regions']), separate_count)
                    for key in p['regions']:
                        self.assertEqual(actual[key]['analysis_mode'], mode)
                        self.assertEqual(set(actual[key]), set(expected[key]))
                        for field, value in expected[key].items():
                            if isinstance(value, (str, bool)):
                                self.assertEqual(actual[key][field], value)
                            else:
                                np.testing.assert_allclose(actual[key][field], value, rtol=1e-11, atol=1e-12,
                                                           err_msg=f'{key}: {field}')

    def test_segmented_batch_keeps_bounded_traces_and_full_peak_search(self):
        times = np.linspace(0, 100, 100001)
        currents = (np.sin(2*np.pi*10*times) * (1 + times/100))[:, None]
        p = _base_payload(100)
        p['regions']['second'] = dict(p['regions']['r'], name='Second', quadratic=[[4e-12]])
        p['authoritative_analysis'] = _packet_source(times, currents, 100)
        batch = ra.analyse_regions_frequency_weighted(p, fft_samples=1024)
        expected = ra.analyse_region_frequency_weighted(p, 'second', fft_samples=1024)
        self.assertTrue(batch['second']['weighted_trace_reduced'])
        self.assertLessEqual(len(batch['second']['time_s']), ra.MAX_FFT_SAMPLES)
        for field in ('time_s', 'weighted_index', 'weighted_peak_index', 'weighted_peak_time_s'):
            np.testing.assert_allclose(batch['second'][field], expected[field])

    def test_failure_returns_no_partial_batch(self):
        p = self.payload('whole_record')
        p['regions']['right']['quadratic'] = [[float('nan')]]
        with self.assertRaises(ValueError):
            ra.analyse_regions_frequency_weighted(p)
