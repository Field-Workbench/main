"""Exposure Analysis page navigation, automatic first-run calculation and cache reuse."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu')
import unittest
import time
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from fieldworkbench import exposure_actions as ea
from fieldworkbench.brain_analysis import lpba40_region_hierarchy
from fieldworkbench.region_analysis import encode


class WorkspaceNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.nodes = list(lpba40_region_hierarchy().walk())[:5]
        t = np.linspace(0, 1, 1001)
        c = np.sin(2*np.pi*20*t)[:, None]
        self.payload = dict(times_b64=encode(t, np.float64), currents_b64=encode(c, np.float64),
                            shape=list(c.shape), drive_ids=['coil'], statistics_duration_s=1,
                            regions={n.key: dict(name=n.name, constant=0., linear=[0.],
                                                quadratic=[[(i+1)*1e-12]]) for i,n in enumerate(self.nodes)})
        self.dialog = None

    def wait_for_run(self, page):
        # A completed regional configuration can schedule one follow-up run with
        # QTimer.singleShot when controls changed while the prior job was busy.
        # Require a few idle event-loop passes so the helper waits for the stable
        # terminal state rather than stopping in that hand-off gap.
        deadline = time.monotonic() + 15
        stable_idle_cycles = 0
        while time.monotonic() < deadline:
            self.app.processEvents()
            if page._analysis_busy:
                stable_idle_cycles = 0
            else:
                stable_idle_cycles += 1
                if stable_idle_cycles >= 3:
                    break
            time.sleep(.005)
        self.assertFalse(page._analysis_busy, "Analysis did not finish")
        self.app.processEvents()

    def tearDown(self):
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.deleteLater()
            self.app.processEvents()

    def test_open_pages_calculate_once_and_navigation_has_no_run_or_footer_buttons(self):
        d = self.dialog = ea.ExposureAnalysisTargetView(self.payload)
        d.show()
        self.app.processEvents()
        self.assertEqual(d.page_heading.text(), 'Exposure analysis home')
        self.assertTrue(d.region_tree.isHidden())
        self.assertTrue(d.selection_note.isHidden())
        self.assertFalse(hasattr(d, 'footer_back_button'))

        for key in ('summary', 'spectrum', 'fft', 'timing', 'threshold'):
            page = d._show_page(key)
            self.wait_for_run(page)
            self.assertTrue(page.region.isHidden())
            self.assertFalse(hasattr(page, 'recalculate_button'))
            self.assertIsInstance(ea._analysis_cached_result(d.analysis_cache, page._request_key()), dict)
            if key == 'summary':
                self.assertTrue(d.region_tree.isHidden())
            else:
                self.assertTrue(d.region_tree.isVisible())
            d.region_tree.setCurrentItem(d.region_tree._region_items[self.nodes[-1].key])
            self.assertEqual(page.region.currentData(), self.nodes[-1].key)
            d.back_button.click()
            self.assertEqual(d.current_page_key, 'home')
            self.assertTrue(d.region_tree.isHidden())
            d.region_tree.setCurrentItem(d.region_tree._region_items[self.nodes[0].key])

        weighted = d._show_page('weighted')
        self.assertTrue(weighted.region.isHidden())
        self.assertTrue(weighted.region_tree.isHidden())
        self.assertFalse(hasattr(weighted, 'run_profile'))
        self.assertTrue(all(len(actions) == 2 for actions in weighted._profile_actions.values()))

    def test_cached_browsing_and_setting_changes_use_configuration_cache(self):
        d = self.dialog = ea.ExposureAnalysisTargetView(self.payload)
        d.show()
        for key in ('summary', 'spectrum', 'timing', 'threshold', 'fft'):
            page = d._show_page(key)
            self.wait_for_run(page)
            self.assertIsInstance(ea._analysis_cached_result(d.analysis_cache, page._request_key()), dict)

        fft = d._pages['fft']
        fft.custom_mode.setChecked(True)
        self.wait_for_run(fft)
        fft.samples.setValue(128)
        self.wait_for_run(fft)
        self.assertEqual(fft._spectrum['fft_samples'], 128)

        timing = d._pages['timing']
        timing.threshold.setValue(3.)
        self.wait_for_run(timing)
        timing_key = timing._request_key()
        self.assertEqual(timing_key[2], 3.)
        self.assertIsInstance(ea._analysis_cached_result(d.analysis_cache, timing_key), dict)
        timing.threshold.setValue(1.)
        self.wait_for_run(timing)
        timing_key = timing._request_key()
        self.assertEqual(timing_key[2], 1.)
        self.assertIsInstance(ea._analysis_cached_result(d.analysis_cache, timing_key), dict)

        with patch.object(ea, '_start_analysis_job') as start, patch.object(ea, 'analyse_regions_spectrum') as calc:
            d.region_tree.setCurrentItem(d.region_tree._region_items[self.nodes[-1].key])
            self.assertEqual(fft._spectrum['fft_samples'], 128)
            summary = d._pages['summary']
            summary_item = summary.region_panel._items_by_key[self.nodes[-1].key]
            self.assertNotEqual(summary_item.text(3), '—')
            for key in ('spectrum', 'timing'):
                self.assertNotIn('Not calculated', d._pages[key].summary.toPlainText())
            self.assertNotIn('Not calculated', d._pages['threshold'].summary.toPlainText())
            d.region_tree.setCurrentItem(d.region_tree._region_items[self.nodes[0].key])
            self.assertEqual(fft._spectrum['fft_samples'], 128)
            d.open_spectrum()
            d.show_home()
            start.assert_not_called()
            calc.assert_not_called()
        self.assertGreaterEqual(
            int(d._analysis_home_cards['spectrum'].cache_status.text().split('•')[-1].split()[0])
            if '•' in d._analysis_home_cards['spectrum'].cache_status.text() else 1,
            2,
        )
        self.assertNotIn('second-moment', d._pages['summary'].region_panel.metrics_note.text())

    def test_background_open_locks_selection_and_failure_is_visible(self):
        self.payload.update(analysis_trace_reduced=True, authoritative_analysis={})
        d = self.dialog = ea.ExposureAnalysisTargetView(self.payload)
        weighted = d.open_frequency_weighted()
        # Keep the scheduler/status path intact while suppressing the actual
        # background thread. Patching _start_analysis_job itself bypasses the
        # single-worker queue introduced in 0.14.0.245 and therefore no longer
        # exercises the real busy/selection state.
        with patch.object(ea, '_launch_analysis_job') as launch:
            page = d.open_spectrum()
            launch.assert_called_once()
        self.assertFalse(d.region_tree.isEnabled())
        self.assertFalse(weighted.result_region_tree.isEnabled())
        d.show_home()
        card = d._analysis_home_cards['spectrum']
        self.assertEqual(card.cache_status.text(), 'Cache: Not calculated')
        self.assertEqual(card.activity_status.text(), 'Running…')
        page._analysis_failed('Example failure')
        self.assertTrue(d.region_tree.isEnabled())
        self.assertEqual(card.cache_status.text(), 'Cache: Not calculated')
        self.assertIn('Last run failed', card.activity_status.text())
        self.assertFalse(d.analysis_cache)

    def test_batch_saved_reopen_preserves_all_region_results(self):
        from fieldworkbench.studio_adapter import StudioAdapter
        d = self.dialog = ea.ExposureAnalysisTargetView(self.payload)
        page = d.open_exposure_summary()
        self.wait_for_run(page)
        config = ea._regional_config_key(page._request_key())
        previous = d.analysis_cache[config]
        self.assertEqual(set(previous['regions']), set(self.payload['regions']))
        self.assertEqual(len(ea._analysis_cache_result_keys(d.analysis_cache)), 1)

        payload_blob = StudioAdapter._exposure_analysis_value_blob(self.payload)[0]
        cache_blob = StudioAdapter._exposure_analysis_cache_blob(d.analysis_cache)[0]
        payload, cache = StudioAdapter._decode_exposure_analysis_payload(
            dict(payload_b64=payload_blob, cache_b64=cache_blob))
        with patch.object(ea, '_start_analysis_job') as start:
            reopened = ea.ExposureAnalysisTargetView(payload, analysis_cache=cache)
            try:
                for node in self.nodes:
                    reopened.region_tree.setCurrentItem(reopened.region_tree._region_items[node.key])
                    summary = reopened.open_exposure_summary()
                    item = summary.region_panel._items_by_key[node.key]
                    self.assertNotEqual(item.text(3), '—')
                    self.assertEqual(summary.region.currentData(), node.key)
                    reopened.show_home()
                    self.assertEqual(reopened._analysis_home_cards['summary'].cache_status.text(), 'Cache: Cached')
                start.assert_not_called()
            finally:
                reopened.close()
                reopened.deleteLater()

    def test_weighted_result_tree_keeps_workspace_selection_in_sync(self):
        d = self.dialog = ea.ExposureAnalysisTargetView(self.payload)
        page = d.open_frequency_weighted()
        page.result_region_tree.setCurrentItem(page.result_region_tree._region_items[self.nodes[-1].key])
        self.assertEqual(d._selected_key(), self.nodes[-1].key)
        self.assertEqual(d.region_tree.currentItem().data(0, Qt.ItemDataRole.UserRole), self.nodes[-1].key)
        self.assertFalse(d.analysis_cache)
