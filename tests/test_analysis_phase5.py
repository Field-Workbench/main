"""Final lifecycle gates: cancel races, independent saves and stale/deleted caches."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu')
import threading
import time
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from fieldworkbench import exposure_actions as ea, region_analysis as ra
from fieldworkbench.exposure_analysis import prepare_exposure_analysis_payload
from fieldworkbench.exposure_workspace import ExposureAnalysisWorkspace
import test_frequency_weighted_batch as fixtures
from test_analysis_phase4 import scene


class FinalLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump_until(self, predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.002)
        self.assertTrue(predicate(), 'Timed out waiting for analysis lifecycle')
        self.app.processEvents()

    def drain(self):
        self.pump_until(lambda: not ea._ACTIVE_ANALYSIS_JOBS)

    def test_worker_cancel_before_start_and_during_progress(self):
        event = threading.Event()
        event.set()
        calls = []
        worker = ea._AnalysisWorker(lambda p: calls.append('ran'), event)
        worker.cancelled.connect(lambda: calls.append('cancelled'))
        worker.run()
        self.assertEqual(calls, ['cancelled'])
        calls.clear()
        event.clear()
        def calculate(progress):
            event.set()
            progress({'message': 'checkpoint'})
            calls.append('completed')
        worker = ea._AnalysisWorker(calculate, event)
        worker.cancelled.connect(lambda: calls.append('cancelled'))
        worker.run()
        self.assertEqual(calls, ['cancelled'])

    def test_all_embedded_tools_cancel_keep_cache_and_can_calculate_again(self):
        for kind, function in (('summary', 'analyse_regions_exposure_summary'),
                               ('spectrum', 'analyse_regions_spectrum'),
                               ('timing', 'analyse_regions_waveform'),
                               ('threshold', 'analyse_regions_threshold_duration'),
                               ('weighted', 'analyse_regions_frequency_weighted')):
            with self.subTest(kind=kind):
                payload = fixtures.BatchWeightingTests().payload('whole_record')
                hub = ea.ExposureAnalysisTargetView(payload)
                hub.show()
                marker = {'unchanged': True}
                hub.analysis_cache[('other', 'unrelated')] = marker
                entered = threading.Event()
                pause = threading.Event()
                def slow(*args, progress_callback=None, **kwargs):
                    entered.set()
                    while True:
                        progress_callback({'message': 'Working', 'completed': 1, 'total': 10**12})
                        pause.wait(.002)
                with patch.object(ea, function, side_effect=slow):
                    if kind == 'weighted':
                        page = hub._show_page(kind)
                        page.view_profile(page.profile.currentData())
                    else:
                        page = hub._show_page(kind)
                    self.pump_until(entered.is_set)
                    hub.show_home()
                    self.assertFalse(hub.analysis_footer_status.isHidden())
                    progress_row = hub._analysis_progress_rows[id(page)]
                    self.assertFalse(progress_row.isHidden())
                    self.assertTrue(progress_row.cancel_button.isEnabled())
                    progress_row.cancel_button.click()
                    self.pump_until(lambda: not page._analysis_busy)
                self.assertEqual(page._analysis_progress_state, 'cancelled')
                self.assertEqual(list(hub.analysis_cache), [('other', 'unrelated')])
                self.assertIs(hub.analysis_cache[('other', 'unrelated')], marker)
                self.assertTrue(hub.region_tree.isEnabled())
                if kind == 'weighted':
                    page.view_profile(page.profile.currentData())
                else:
                    (getattr(page, 'refresh', None) or page.recalculate)()
                self.pump_until(lambda: not page._analysis_busy)
                self.assertIsInstance(ea._analysis_cached_result(hub.analysis_cache, page._request_key()), dict)
                hub.close()
                hub.deleteLater()
                self.drain()

    def test_single_worker_queue_pause_resume_and_cancel(self):
        payload = fixtures.BatchWeightingTests().payload('whole_record')
        hub = ea.ExposureAnalysisTargetView(payload)
        hub.show()
        entered = {'timing': threading.Event(), 'threshold': threading.Event()}
        counts = {'timing': 0, 'threshold': 0}

        def slow_timing(*args, progress_callback=None, **kwargs):
            entered['timing'].set()
            while True:
                counts['timing'] += 1
                progress_callback({
                    'message': 'Timing work', 'completed': counts['timing'], 'total': 10**9,
                })
                time.sleep(.001)

        def slow_threshold(*args, progress_callback=None, **kwargs):
            entered['threshold'].set()
            while True:
                counts['threshold'] += 1
                progress_callback({
                    'message': 'Threshold work', 'completed': counts['threshold'], 'total': 10**9,
                })
                time.sleep(.001)

        with patch.object(ea, 'analyse_regions_waveform', side_effect=slow_timing), \
             patch.object(ea, 'analyse_regions_threshold_duration', side_effect=slow_threshold):
            timing = hub._show_page('timing')
            hub.show_home()
            threshold = hub._show_page('threshold')
            hub.show_home()
            self.pump_until(entered['timing'].is_set)
            self.assertFalse(entered['threshold'].is_set())
            self.pump_until(lambda: len([
                row for row in hub._analysis_progress_rows.values() if not row.isHidden()
            ]) == 2)

            timing_row = hub._analysis_progress_rows[id(timing)]
            threshold_row = hub._analysis_progress_rows[id(threshold)]
            self.assertEqual(timing._analysis_progress_state, 'running')
            self.assertEqual(threshold._analysis_progress_state, 'queued')
            self.assertEqual(timing_row.pause_button.text(), 'Pause')
            self.assertFalse(threshold_row.pause_button.isVisible())
            self.assertTrue(threshold_row.cancel_button.isEnabled())

            timing_row.pause_button.click()
            self.pump_until(lambda: timing._analysis_progress_state == 'paused')
            self.assertEqual(timing_row.pause_button.text(), 'Run')
            timing_count = counts['timing']
            deadline = time.monotonic() + .08
            while time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.002)
            self.assertLessEqual(counts['timing'], timing_count + 1)
            self.assertFalse(entered['threshold'].is_set())

            timing_row.pause_button.click()
            self.pump_until(lambda: timing._analysis_progress_state == 'running')
            timing_row.cancel_button.click()
            self.pump_until(lambda: not timing._analysis_busy)
            self.pump_until(entered['threshold'].is_set)
            self.assertEqual(threshold._analysis_progress_state, 'running')

            threshold_row.cancel_button.click()
            self.pump_until(lambda: not threshold._analysis_busy)
            self.assertTrue(hub.analysis_footer_status.isHidden())

        hub.close()
        hub.deleteLater()
        self.drain()

    def test_calculate_all_queues_each_uncached_home_method(self):
        payload = fixtures.BatchWeightingTests().payload('whole_record')
        hub = ea.ExposureAnalysisTargetView(payload)
        hub.show()
        entered = threading.Event()

        def slow_summary(*args, progress_callback=None, **kwargs):
            entered.set()
            while True:
                progress_callback({'message': 'Summary work', 'completed': 1, 'total': 10**9})
                time.sleep(.002)

        with patch.object(ea, 'analyse_regions_exposure_summary', side_effect=slow_summary):
            hub.calculate_all_button.click()
            self.pump_until(entered.is_set)
            self.assertIs(hub._analysis_active_page, hub._pages['summary'])
            self.assertGreaterEqual(len(hub._analysis_job_queue), 3)
            self.assertEqual(hub._pages['spectrum']._analysis_progress_state, 'queued')
            self.assertEqual(hub._pages['timing']._analysis_progress_state, 'queued')
            self.assertEqual(hub._pages['threshold']._analysis_progress_state, 'queued')
            self.assertEqual(hub._pages['weighted']._analysis_progress_state, 'queued')
            hub.cancel_running_analyses()
            self.pump_until(lambda: not hub.is_busy())

        hub.close()
        hub.deleteLater()
        self.drain()

    def test_cancel_queued_completion_cannot_publish(self):
        hub = ea.ExposureAnalysisTargetView(fixtures.BatchWeightingTests().payload('whole_record'))
        page = hub.open_exposure_summary()
        page._analysis_request_key = page._request_key()
        page._set_analysis_busy(True)
        event = threading.Event()
        delivery = ea._AnalysisDelivery(page, event)
        result = ra.analyse_regions_exposure_summary(page.payload)
        event.set()
        delivery.completed(result)
        self.assertFalse(hub.analysis_cache)
        self.assertEqual(page._analysis_progress_state, 'cancelled')
        hub.close()
        hub.deleteLater()
        self.app.processEvents()

    def prepared(self):
        adapter, exposure, boxes = scene()
        payload = prepare_exposure_analysis_payload(adapter, exposure, target_id=boxes[0])
        analysis_id = adapter.save_exposure_analysis_resource(payload, analysis_cache={})
        return adapter, exposure, boxes, adapter.exposure_analysis(analysis_id)

    def complete(self, hub, kind):
        page = hub._show_page(kind)
        self.pump_until(lambda: not page._analysis_busy)
        self.assertIsInstance(ea._analysis_cached_result(hub.analysis_cache, page._request_key()), dict)
        return page

    def test_two_windows_merge_independent_updates_and_reopen(self):
        adapter, exposure, boxes, record = self.prepared()
        windows = [ea.ExposureAnalysisTargetView(record['payload'], adapter=adapter,
                   saved_record=record, analysis_cache=record['analysis_cache']) for _ in range(2)]
        try:
            self.complete(windows[0], 'summary')
            self.complete(windows[1], 'threshold')
            saved = adapter.exposure_analysis(record['id'])
            self.assertIn(('summary', '__all_regions__'), saved['analysis_cache'])
            self.assertIn(('threshold', '__all_regions__', 1.), saved['analysis_cache'])
            # Rerunning summary in the earlier window must not remove the later threshold.
            self.complete(windows[0], 'summary')
            restored = type(adapter)(adapter.document()).exposure_analysis(record['id'])
            self.assertEqual(set(restored['analysis_cache']), set(saved['analysis_cache']))
            self.assertEqual(restored['status'], 'current')
        finally:
            for hub in windows:
                hub.close()
                hub.deleteLater()
            self.app.processEvents()

    def test_failed_persistence_retains_memory_and_retry_saves(self):
        adapter, exposure, boxes, record = self.prepared()
        hub = ea.ExposureAnalysisTargetView(record['payload'], adapter=adapter, saved_record=record)
        try:
            with patch.object(adapter, 'update_exposure_analysis_cache', side_effect=ValueError('Write unavailable')), patch.object(ea.QMessageBox, 'warning'):
                self.complete(hub, 'summary')
            self.assertTrue(hub.analysis_cache)
            self.assertFalse(adapter.exposure_analysis(record['id'])['analysis_cache'])
            self.assertFalse(hub.cache_save_note.isHidden())
            card = hub._analysis_home_cards['summary']
            self.assertEqual(card.cache_status.text(), 'Cache: Cached')
            self.assertIn('Save failed', card.activity_status.text())
            self.assertFalse(card.retry_save_button.isHidden())
            card.retry_save_button.click()
            self.assertTrue(hub.cache_save_note.isHidden())
            self.assertTrue(card.retry_save_button.isHidden())
            self.assertEqual(card.activity_status.text(), '')
            self.assertTrue(adapter.exposure_analysis(record['id'])['analysis_cache'])
        finally:
            hub.close()
            hub.deleteLater()
            self.app.processEvents()

    def test_external_clear_stale_and_delete_are_rechecked_for_exact_saved_analysis(self):
        adapter, exposure, boxes, record = self.prepared()
        workspace = ExposureAnalysisWorkspace(adapter, exposure)
        try:
            workspace.target_tree.setCurrentItem(workspace._items[boxes[0]])
            workspace.open_analysis_button.click()
            first = workspace._active_view
            self.complete(first, 'summary')
            adapter.update_exposure_analysis_cache(record['id'], {})
            workspace._show_objects()
            refreshed = workspace._load_saved_analysis_view(record['id'])
            self.assertIsNot(refreshed, first)
            self.assertFalse(refreshed.analysis_cache)
            adapter._mesh(boxes[0])['position'][0] += .01
            self.assertIs(workspace._load_saved_analysis_view(record['id']), refreshed)
            self.assertEqual(adapter.exposure_analysis(record['id'])['status'], 'stale')
            workspace._refresh_target_info(boxes[0])
            self.assertEqual(workspace.saved_analysis_tree.topLevelItem(0).text(1), 'Stale')
            adapter._mesh(boxes[0])['position'][0] -= .01
            self.assertIs(workspace._load_saved_analysis_view(record['id']), refreshed)
            adapter.delete_exposure_analysis(record['id'])
            self.assertIsNone(workspace._load_saved_analysis_view(record['id']))
        finally:
            workspace.close()
            workspace.deleteLater()
            self.app.processEvents()

    def test_close_cancels_running_workspace_and_preserves_saved_result(self):
        adapter, exposure, boxes, record = self.prepared()
        workspace = ExposureAnalysisWorkspace(adapter, exposure)
        workspace.target_tree.setCurrentItem(workspace._items[boxes[0]])
        workspace.open_analysis_button.click()
        workspace.show()
        entered = threading.Event()
        pause = threading.Event()

        def slow(*args, progress_callback=None, **kwargs):
            entered.set()
            while True:
                progress_callback({'message': 'Working'})
                pause.wait(.002)

        destroyed = threading.Event()
        workspace.destroyed.connect(lambda *_args: destroyed.set())
        with patch.object(ea, 'analyse_regions_exposure_summary', side_effect=slow):
            page = workspace._active_view.open_exposure_summary()
            self.pump_until(entered.is_set)
            workspace.close()
            self.pump_until(destroyed.is_set)
            self.drain()
        self.assertFalse(adapter.exposure_analysis(record['id'])['analysis_cache'])

    def test_single_volume_weighting_is_background_and_cancelable(self):
        adapter, exposure, boxes, record = self.prepared()
        hub = ea.ExposureAnalysisTargetView(record['payload'])
        page = hub.open_frequency_weighted()
        with patch.object(ea, '_start_analysis_job') as start:
            page.view_profile(page.profile.currentData())
            start.assert_called_once()
        page._analysis_failed('Simulated stop')
        hub.close()
        hub.deleteLater()
        self.app.processEvents()

    def test_source_waveform_and_routing_changes_invalidate_saved_preparation(self):
        from fieldworkbench.waveform_io import WaveformDocument, WaveformSeries
        adapter, exposure, boxes, record = self.prepared()
        waveform_id = exposure['protocol']['sources'][0]['waveform_id']
        document = WaveformDocument([WaveformSeries('Changed', [0., .01, .02], [0., 2., 0.],
                                                     quantity='current', unit='A')])
        adapter.save_waveform_document(document, waveform_id=waveform_id)
        self.assertEqual(adapter.exposure_analysis(record['id'])['status'], 'stale')
        adapter, exposure, boxes, record = self.prepared()
        protocol = adapter.exposures[0]['protocol']
        protocol['sequence']['steps'][0]['duration_s'] *= 2
        self.assertEqual(adapter.exposure_analysis(record['id'])['status'], 'stale')

    def test_page_cleanup_cancels_worker_without_dialog_lifecycle(self):
        entered = threading.Event()
        pause = threading.Event()

        def slow(*args, progress_callback=None, **kwargs):
            entered.set()
            while True:
                progress_callback({'message': 'Working'})
                pause.wait(.002)

        with patch.object(ea, 'analyse_regions_exposure_summary', side_effect=slow):
            page = ea.ExposureSummaryPage(fixtures.BatchWeightingTests().payload('whole_record'))
            page.show()
            self.pump_until(entered.is_set)
            page.cleanup()
            page.close()
            self.pump_until(lambda: not page._analysis_busy and not page.isVisible())
        self.assertFalse(page.analysis_cache)
        page.deleteLater()
        self.drain()

    def test_preparation_cancel_wins_over_queued_completion(self):
        adapter, exposure, boxes, record = self.prepared()
        workspace = ExposureAnalysisWorkspace(adapter, exposure)
        workspace._preparation_cancel_requested = True
        with patch.object(adapter, 'save_exposure_analysis_resource') as save:
            workspace._preparation_completed(record['payload'])
            save.assert_not_called()
        self.assertIsNone(workspace._prepared_record)
        self.assertIn('cancelled', workspace.preparation_progress_label.text())
        workspace.close()
        workspace.deleteLater()
        self.app.processEvents()
