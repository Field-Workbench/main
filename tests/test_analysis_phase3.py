"""Threshold batching and grouped weighting lifecycle regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu')
import time
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from fieldworkbench import region_analysis as ra, exposure_actions as ea
from fieldworkbench.studio_adapter import StudioAdapter
import test_frequency_weighted_batch as fixtures
from test_region_time_domain_streaming import _verified_payload, _encode


class ThresholdBatchTests(unittest.TestCase):
    def check(self, payload, threshold):
        expected = {k: ra.analyse_region_threshold_duration(payload, k, threshold)
                    for k in payload['regions']}
        with patch.object(ra, '_iter_time_domain_packets', wraps=ra._iter_time_domain_packets) as packets:
            actual = ra.analyse_regions_threshold_duration(payload, threshold)
            self.assertEqual(packets.call_count, 1)
        for key, expected_result in expected.items():
            self.assertEqual(set(actual[key]), set(expected_result))
            for field, value in expected_result.items():
                if isinstance(value, str):
                    self.assertEqual(actual[key][field], value)
                else:
                    np.testing.assert_allclose(actual[key][field], value, rtol=1e-12, atol=1e-12)
        return actual

    def test_modes_thresholds_static_fields_and_cross_terms(self):
        for mode in ('whole_record', 'verified_periodic_cycle', 'segmented_full_duration'):
            payload = fixtures.BatchWeightingTests().payload(mode)
            payload['regions']['left'].update(constant=4e-12, linear=[.1e-12, -.2e-12])
            for threshold in (0., .1, 1., 5.):
                with self.subTest(mode=mode, threshold=threshold):
                    self.check(payload, threshold)

    def test_large_repeats_gaps_and_singletons(self):
        p = _verified_payload([2., 0., 2.], repeat_count=100000)
        p['regions']['other'] = dict(p['regions']['r'], quadratic=[[4e-12]])
        actual = self.check(p, 1.)
        self.assertTrue(actual['r']['intervals_truncated'])
        self.assertLessEqual(len(actual['r']['intervals_s']), ra.MAX_RETAINED_INTERVALS)
        p['authoritative_analysis']['packets'].extend([
            dict(times_b64=_encode([50001.]), currents_b64=_encode([[8.]]), shape=[1, 1], repeat_count=1),
            dict(times_b64=_encode([50002., 50002.5, 50003.]),
                 currents_b64=_encode([[0.], [1.], [0.]]), shape=[3, 1], repeat_count=3)])
        self.check(p, 1.)

    def test_invalid_inputs_reject_entire_batch(self):
        p = fixtures.BatchWeightingTests().payload('whole_record')
        for threshold in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ra.analyse_regions_threshold_duration(p, threshold)
        p['regions']['right']['quadratic'] = [[float('nan')]]
        with self.assertRaises(ValueError):
            ra.analyse_regions_threshold_duration(p, 1.)

    def test_weighting_reuses_spectrum_without_changing_results(self):
        p = fixtures.BatchWeightingTests().payload('segmented_full_duration')
        spectra = ra.analyse_regions_spectrum(p, fft_samples=256)
        for profile in ra.FREQUENCY_WEIGHTING_PROFILES:
            expected = ra.analyse_regions_frequency_weighted(p, profile=profile, fft_samples=256)
            with patch.object(ra, 'analyse_regions_spectrum', side_effect=AssertionError('Unexpected FFT')):
                actual = ra.analyse_regions_frequency_weighted(p, profile=profile, fft_samples=256, spectra=spectra)
            for key in expected:
                for field, value in expected[key].items():
                    if isinstance(value, str):
                        self.assertEqual(actual[key][field], value)
                    else:
                        np.testing.assert_allclose(actual[key][field], value)


class GroupedLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.payload = fixtures.BatchWeightingTests().payload('whole_record')
        self.dialog = ea.ExposureAnalysisTargetView(self.payload)

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def wait(self, page):
        end = time.monotonic() + 15
        while page._analysis_busy and time.monotonic() < end:
            self.app.processEvents()
            time.sleep(.005)
        self.assertFalse(page._analysis_busy)
        self.app.processEvents()

    def test_threshold_settings_coexist_and_regions_browse(self):
        page = self.dialog.open_threshold()
        self.wait(page)
        for value in (1., 2.):
            page.threshold.setValue(value)
            self.wait(page)
            self.assertIsInstance(ea._analysis_cached_result(self.dialog.analysis_cache, page._request_key()), dict)
        self.assertEqual(sum(k[0] == 'threshold' for k in self.dialog.analysis_cache), 2)
        self.assertFalse(hasattr(page, 'recalculate_button'))
        with patch.object(ea, '_start_analysis_job') as start:
            for key in self.payload['regions']:
                page.region.setCurrentIndex(page.region.findData(key))
                for value in (1., 2.):
                    page.threshold.setValue(value)
                    self.assertIn(f'Threshold: {value:g} µT', page.summary.toPlainText())
            start.assert_not_called()

    def test_weighted_profiles_share_exact_spectrum_and_view_keeps_settings(self):
        page = self.dialog.open_frequency_weighted()
        profiles = list(ra.FREQUENCY_WEIGHTING_PROFILES)[:2]
        self.assertFalse(hasattr(page, 'run_profile'))
        with patch.object(ea, 'analyse_regions_spectrum', wraps=ra.analyse_regions_spectrum) as spectra:
            for profile in profiles:
                with patch.object(page, '_analysis_cache_changed_callback', wraps=page._analysis_cache_changed_callback) as notify:
                    page.view_profile(profile)
                    self.wait(page)
                    notify.assert_called_once()
                    page.result_window.close()
            self.assertEqual(spectra.call_count, 1)
            page.view_profile(profiles[0])
            original = page._analysis
            self.assertIn('Sampling: Auto', page.summary.toPlainText())
            page.custom_mode.setChecked(True)
            page.samples.setValue(256)
            # An open result retains its own settings while the setup changes.
            self.assertIs(page._analysis, original)
            self.assertIn('Sampling: Auto', page.summary.toPlainText())
            page.result_window.close()
            page.view_profile(profiles[0])
            self.wait(page)
            self.assertEqual(spectra.call_count, 2)
            self.assertIn('Sampling: Custom — 256', page.summary.toPlainText())
        cache = self.dialog.analysis_cache
        self.assertEqual(sum(k[0] == 'weighted' for k in cache), 3)
        self.assertEqual(sum(k[0] == 'spectrum' for k in cache), 2)
        blob = StudioAdapter._exposure_analysis_cache_blob(cache)[0]
        pb = StudioAdapter._exposure_analysis_value_blob(self.payload)[0]
        payload, restored = StudioAdapter._decode_exposure_analysis_payload(dict(payload_b64=pb, cache_b64=blob))
        self.assertEqual(StudioAdapter._exposure_analysis_cache_summary(restored)['cached_result_count'], 5)
        reopened = ea.FrequencyWeightedPage(payload, 'right', analysis_cache=restored)
        try:
            with patch.object(ea, '_start_analysis_job') as start:
                reopened.view_profile(profiles[1])
                self.assertEqual(reopened._analysis['profile'], profiles[1])
                self.assertIn('Sampling: Auto', reopened.summary.toPlainText())
                start.assert_not_called()
        finally:
            reopened.close()
            reopened.deleteLater()

    def test_weighting_uses_matching_spectrum_page_result(self):
        fft = self.dialog.open_fft()
        self.wait(fft)
        fft.custom_mode.setChecked(True)
        fft.samples.setValue(256)
        fft.window.setCurrentIndex(fft.window.findData('hann'))
        self.wait(fft)
        weighted = self.dialog.open_frequency_weighted()
        weighted.custom_mode.setChecked(True)
        weighted.samples.setValue(256)
        with patch.object(ea, 'analyse_regions_spectrum', wraps=ra.analyse_regions_spectrum) as spectra:
            weighted.view_profile(weighted.profile.currentData())
            self.wait(weighted)
            self.assertEqual(spectra.call_count, 1)  # Hann cannot substitute for rectangular.
            weighted.result_window.close()
        with patch.object(ea, 'analyse_regions_spectrum', side_effect=AssertionError('Should reuse')):
            weighted.view_profile(list(ra.FREQUENCY_WEIGHTING_PROFILES)[1])
            self.wait(weighted)
            self.assertNotEqual(weighted._analysis_progress_state, 'failed')

