"""Shared panel interaction and export-state regression checks."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtQml import QJSEngine
from PySide6.QtWidgets import QApplication, QLabel, QDialog, QPushButton, QTabWidget, QVBoxLayout

from fieldworkbench.brain_analysis import BrainRegionNode
from fieldworkbench.exposure_actions import (
    ExposureActionsMixin,
    ExposureAnalysisTargetView,
    ExposureAnalysisWindow,
    FFTSpectrumPage,
    FrequencyWeightedPage,
    ThresholdDurationPage,
    SpectrumAnalysisPage,
)
from fieldworkbench.exposure_panel import ExposureRegionPanel, ExposureResultTab
from fieldworkbench.region_analysis import MeasurementRegions, encode
from fieldworkbench.waveform_gl_view import WaveformGLView


def metric_data():
    values = np.zeros((3, 2, 7), dtype=np.float32)
    values[:, :, 0] = [[1, 2], [2, 3], [3, 4]]
    values[:, :, 6] = [10, 20]
    return {
        "keys": ["a", "b"],
        "shape": list(values.shape),
        "values_b64": encode(values, np.float32),
        "source_frame_count": 3,
        "sample_counts": [2, 2],
        "voxel_counts": [2, 2],
        "playback_summary": {
            "shape": [2, 7],
            "values_b64": encode(values[-1], np.float64),
            "peak_times_s_b64": encode([1, 1], np.float64),
            "playback": {},
        },
    }


class Viewer(QLabel):
    frameChanged = Signal(int, float)

    def __init__(self, payload=None, **kwargs):
        super().__init__("Viewer")
        self.payload = payload or {}
        self.cleaned = False
        self.highlight = ()

    def cleanup(self):
        self.cleaned = True

    def can_export_video(self):
        return True

    def export_payload(self):
        return self.payload.copy()

    def stop_playback(self):
        pass

    def request_camera(self, callback):
        callback({"eye": {"x": 1, "y": 1, "z": 1}})

    def refresh_viewport(self):
        pass

    def set_brain_region_highlight(self, labels):
        self.highlight = labels


class Window(ExposureActionsMixin, QDialog):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

    def _render_tab_changed(self, index):
        self.refreshed = index


class ExposurePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        self.app.processEvents()

    def test_region_selection_refreshes_visible_view_without_waking_hidden_view(self):
        viewer = WaveformGLView({}, start_hibernated=True, _source_html_path="unused.html")
        try:
            with patch.object(viewer, "isVisible", return_value=True), \
                    patch.object(viewer, "refresh_viewport") as refresh:
                viewer.set_brain_region_highlight([182, 181, 181])
                self.assertEqual(viewer._brain_highlight_label_ids, (181, 182))
                refresh.assert_called_once_with()
                refresh.reset_mock()
                viewer.set_brain_region_highlight(())
                self.assertEqual(viewer._brain_highlight_label_ids, ())
                refresh.assert_called_once_with()
            with patch.object(viewer, "isVisible", return_value=False), \
                    patch.object(viewer, "refresh_viewport") as refresh:
                viewer.set_brain_region_highlight([181])
                refresh.assert_not_called()
                self.assertIsNone(viewer.web)
                self.assertEqual(viewer._requested_lifecycle, "discarded")
        finally:
            viewer.cleanup()
            viewer.deleteLater()

    def test_2d_selection_is_drawn_over_sensor_markers(self):
        # Execute the shipped render branch with drawing probes: the opaque
        # sensor marker used to overwrite the smaller yellow selection point.
        render = next(line for line in WaveformGLView._html({}).splitlines()
                      if line.startswith("function render(){"))
        engine = QJSEngine()
        result = engine.evaluate("""
            var pixel = '', is2d = true, isVolume = false, showField = true;
            var workbenchDark = false, fpsCount = 0, fpsStamp = 0, dirty = true;
            var performance = {now: function() { return 0; }};
            var gl = {clearColor: function(){}, clear: function(){},
                      enable: function(){}, disable: function(){}, blendFunc: function(){}};
            function resize() {} function mvp() { return []; }
            function drawField() {} function drawOverlayFieldPoints() {}
            function drawGrid() {} function drawGridOverlay() {}
            function drawScene() { pixel = 'sensor'; }
            function drawBrainHighlight() { pixel = 'selection'; }
        """ + render + ";render();pixel;")
        self.assertFalse(result.isError(), result.toString())
        self.assertEqual(result.toString(), "selection")

    def test_volume_selection_overlays_equal_depth_sensor_but_not_nearer_geometry(self):
        render = next(line for line in WaveformGLView._html({}).splitlines()
                      if line.startswith("function render(){"))
        for scene_depth, expected in ((0.5, "selection"), (0.2, "scene")):
            with self.subTest(scene_depth=scene_depth):
                engine = QJSEngine()
                result = engine.evaluate("""
                    var pixel = '', depth = 1, depthTest = false, depthMode = 'less';
                    var is2d = false, isVolume = true, showField = true;
                    var workbenchDark = false, fpsCount = 0, fpsStamp = 0, dirty = true;
                    var performance = {now: function() { return 0; }};
                    var gl = {DEPTH_TEST: 'depth', LESS: 'less', LEQUAL: 'lequal',
                        clearColor: function(){}, clear: function(){depth = 1;},
                        enable: function(v){if(v === 'depth') depthTest = true;},
                        disable: function(v){if(v === 'depth') depthTest = false;},
                        blendFunc: function(){}, depthFunc: function(v){depthMode = v;}};
                    function resize() {} function mvp() { return []; }
                    function drawVolume() {pixel = 'field';}
                    function drawOverlayFieldPoints() {} function drawGrid() {}
                    function drawGridOverlay() {}
                    function drawScene() {pixel = 'scene'; depth = SCENE_DEPTH;}
                    function drawBrainHighlight() {
                        if (!depthTest || (depthMode === 'lequal' ? 0.5 <= depth : 0.5 < depth))
                            pixel = 'selection';
                    }
                """.replace("SCENE_DEPTH", str(scene_depth)) + render +
                    ";render(); JSON.stringify([pixel,depthTest,depthMode]);")
                self.assertFalse(result.isError(), result.toString())
                self.assertEqual(result.toString(), f'["{expected}",true,"less"]')

    def wait_for_analysis(self, page, timeout=15.0):
        """Wait for a directly-instantiated analysis page to finish its async first run."""
        import time

        deadline = time.monotonic() + float(timeout)
        stable_idle_cycles = 0
        while time.monotonic() < deadline:
            self.app.processEvents()
            if getattr(page, "_analysis_busy", False):
                stable_idle_cycles = 0
            else:
                stable_idle_cycles += 1
                if stable_idle_cycles >= 3:
                    break
            time.sleep(0.005)
        self.assertFalse(getattr(page, "_analysis_busy", False), "Analysis did not finish")
        self.app.processEvents()

    def test_object_filters_modes_selection_and_collapse(self):
        root = MeasurementRegions(
            (
                BrainRegionNode("a", "A", (1,)),
                BrainRegionNode("b", "B [samples]", (2,)),
            )
        )
        panel = ExposureRegionPanel(root, measurement=True)
        panel.set_frame_metric_data(metric_data())
        viewer = QLabel("Viewer")
        tab = ExposureResultTab(viewer, panel)
        tab.resize(1250, 740)
        tab.show()
        self.app.processEvents()
        self.assertEqual(panel.table.topLevelItemCount(), 2)
        self.assertFalse(any(value == "side" for _, value in panel._filter_metric_choices()))
        panel.set_filter_rules([(1, "<", 1.5, "filter", "#facc15")])
        self.assertTrue(panel._items_by_key["a"].isHidden())
        self.assertFalse(panel._items_by_key["b"].isHidden())
        panel.playback_summary_radio.click()
        self.app.processEvents()
        self.assertEqual(panel.filter_table.rowCount(), 0)
        panel.current_frame_radio.click()
        self.app.processEvents()
        self.assertEqual(panel.filter_table.rowCount(), 1)
        selected = []
        panel.region_selected.connect(selected.append)
        panel.table.setCurrentItem(panel._items_by_key["b"])
        self.assertEqual(selected[-1], (2,))
        self.assertEqual(panel.selection_note.text(), "Highlighted samples: B [samples].")
        panel.request_frame(2, 1)
        self.app.processEvents()
        self.assertFalse(panel._items_by_key["a"].isHidden())
        snapshot = panel.video_table_snapshot()
        self.assertEqual(len(snapshot["rows"]), 2)
        self.assertTrue(all(row["parent_key"] is None for row in snapshot["rows"]))
        self.assertEqual(snapshot["title"], "Measurement volumes")
        tab.toggle_regions()
        self.app.processEvents()
        self.assertEqual(tab.splitter.sizes()[1], 0)
        tab.toggle_regions()
        self.app.processEvents()
        self.assertGreater(tab.splitter.sizes()[1], 0)
        tab.close()
        tab.deleteLater()
        self.app.processEvents()

    def test_waveform_analysis_exposes_fft_view_button(self):
        times = np.array([0.0, 0.5, 1.0])
        currents = np.array([[0.0], [1.0], [0.0]])
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "regions": {
                "a": {
                    "name": "A",
                    "constant": 0.0,
                    "linear": [0.0],
                    "quadratic": [[1e-12]],
                }
            },
        }
        dialog = SpectrumAnalysisPage(payload, "a")
        self.wait_for_analysis(dialog)
        self.assertEqual(dialog.fft_button.text(), "View FFT…")
        self.assertTrue(dialog.fft_button.isEnabled())
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()


    def test_exposure_analysis_target_view_exposes_guided_categories(self):
        times = np.array([0.0, 0.5, 1.0])
        currents = np.array([[0.0], [1.0], [0.0]])
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "exposure_analysis_context": {"physical_duration_s": 6.0},
            "regions": {"a": {"name": "A", "constant": 0.0, "linear": [0.0], "quadratic": [[1e-12]]}},
        }
        view = ExposureAnalysisTargetView(payload, "a")
        self.assertEqual(view.window_title, "Exposure analysis")
        self.assertEqual(view.region.currentData(), "a")
        self.assertEqual(view.preparation_state_label.text(), "Preparation: In memory")
        self.assertEqual(view.preparation_duration_label.text(), "Full exposure: 6 s")
        self.assertEqual(view.preparation_regions_label.text(), "Prepared regions: 1")
        self.assertEqual(view.preparation_samples_label.text(), "Sample points: —")
        self.assertEqual(set(view._analysis_home_cards), {"summary", "spectrum", "timing", "threshold", "weighted"})
        self.assertEqual(view._analysis_home_cards["spectrum"].open_button.text(), "Open spectrum / FFT…")
        self.assertEqual(view._analysis_home_cards["summary"].cache_status.text(), "Cache: Not calculated")
        self.assertTrue(view.region.isHidden())
        self.assertTrue(view.region_tree.isHidden())
        self.assertEqual(view.region_tree.currentItem().data(0, Qt.ItemDataRole.UserRole), "a")
        self.assertEqual(view.current_page_key, "home")
        self.assertTrue(view.back_button.isHidden())
        self.assertFalse(hasattr(view, "footer_back_button"))

        spectrum = view.open_spectrum()
        self.assertEqual(view.current_page_key, "spectrum")
        self.assertIs(view.stack.currentWidget(), spectrum)
        self.assertFalse(view.back_button.isHidden())
        self.assertEqual(view.back_button.text(), "← Analysis home")
        view.back_button.click()
        self.assertEqual(view.current_page_key, "home")
        self.assertIs(view.stack.currentWidget(), view.home_page)
        self.assertTrue(view.back_button.isHidden())

        fft_page = view.open_fft()
        self.assertEqual(view.current_page_key, "fft")
        self.assertIs(view.stack.currentWidget(), fft_page)
        self.assertIsInstance(fft_page, FFTSpectrumPage)
        view.show_home()
        self.assertEqual(view.current_page_key, "home")
        self.assertIs(view.stack.currentWidget(), view.home_page)
        self.assertTrue(view.back_button.isHidden())
        view.cleanup()
        view.close()
        view.deleteLater()
        self.app.processEvents()

    def test_exposure_analysis_window_uses_target_view_context(self):
        times = np.array([0.0, 0.5, 1.0])
        currents = np.array([[0.0], [1.0], [0.0]])
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "regions": {
                "a": {
                    "name": "A", "constant": 0.0, "linear": [0.0],
                    "quadratic": [[1e-12]],
                }
            },
            "exposure_analysis_context": {
                "exposure_name": "SAM360 LTP",
                "target_name": "SRI24 brain",
                "atlas": "LPBA40",
                "sample_count": 1234,
                "physical_duration_s": 120.0,
            },
        }
        dialog = ExposureAnalysisWindow(payload, "a")
        view = dialog.analysis_view
        self.assertIsNone(dialog.parentWidget())
        self.assertEqual(dialog.windowType(), Qt.WindowType.Window)
        self.assertFalse(dialog.isModal())
        self.assertEqual(dialog.windowModality(), Qt.WindowModality.NonModal)
        self.assertEqual(dialog.windowTitle(), "Exposure analysis — SAM360 LTP")
        self.assertEqual(view.page_heading.text(), "Exposure analysis home")
        self.assertEqual(view.preparation_duration_label.text(), "Full exposure: 120 s")
        self.assertEqual(view.preparation_target_label.text(), "Target: SRI24 brain")
        self.assertEqual(view.preparation_exposure_label.text(), "Exposure: SAM360 LTP")
        self.assertEqual(view.preparation_atlas_label.text(), "Atlas: LPBA40")
        self.assertEqual(view.preparation_samples_label.text(), "Sample points: 1,234")
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def test_exposure_analysis_home_reports_cache_state_for_selected_region(self):
        times = np.array([0.0, 0.5, 1.0])
        currents = np.array([[0.0], [1.0], [0.0]])
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "regions": {
                "a": {
                    "name": "A", "constant": 0.0, "linear": [0.0],
                    "quadratic": [[1e-12]],
                },
                "b": {
                    "name": "B", "constant": 0.0, "linear": [0.0],
                    "quadratic": [[1e-12]],
                },
            },
            "exposure_analysis_context": {
                "exposure_name": "SAM360 LTP",
                "target_name": "SRI24 brain",
                "atlas": "LPBA40",
                "sample_count": 1234,
                "physical_duration_s": 120.0,
                "preparation_processing_time_s": 19.25,
            },
        }
        both_regions = {"regions": {"a": {"value": 1}, "b": {"value": 2}}}
        cache = {
            ("summary", "__all_regions__"): both_regions,
            ("__cache_meta__", "summary", "__all_regions__"): {"processing_time_s": 2.0},
            ("spectrum", "__all_regions__", None, "rectangular", True): both_regions,
            ("spectrum", "__all_regions__", 4096, "hann", True): both_regions,
            ("timing", "__all_regions__", 0.25): both_regions,
            ("timing", "__all_regions__", 0.5): both_regions,
            ("threshold", "__all_regions__", 1.0): both_regions,
            ("threshold", "__all_regions__", 5.0): both_regions,
            ("threshold", "__all_regions__", 10.0): both_regions,
            ("weighted", "__all_regions__", "icnirp", None): both_regions,
            ("weighted", "__all_regions__", "eu", 32768): both_regions,
        }
        dialog = ExposureAnalysisTargetView(
            payload, "a", analysis_cache=cache,
            saved_record={"id": "analysis_1", "name": "SAM360 full", "status": "current"},
        )
        self.assertEqual(dialog.preparation_state_label.text(), "Preparation: Current")
        self.assertEqual(dialog.preparation_regions_label.text(), "Prepared regions: 2")
        self.assertEqual(dialog.preparation_samples_label.text(), "Sample points: 1,234")
        self.assertEqual(dialog.preparation_time_label.text(), "Preparation time: 19.2 s")
        def status(key):
            return dialog._analysis_home_cards[key].cache_status.text()
        self.assertEqual(status("summary"), "Cache: Cached")
        self.assertEqual(status("spectrum"), "Cache: Cached • 2 settings")
        self.assertEqual(status("timing"), "Cache: Cached • 2 settings")
        self.assertEqual(status("threshold"), "Cache: Cached • 3 settings")
        self.assertEqual(status("weighted"), "Cache: Cached • 2 settings")
        dialog.region_tree.setCurrentItem(dialog.region_tree._region_items["b"])
        self.app.processEvents()
        # Derived caches belong to the prepared object/configuration; changing
        # the viewed region does not make the same all-region run look missing.
        self.assertEqual(status("summary"), "Cache: Cached")
        self.assertEqual(status("spectrum"), "Cache: Cached • 2 settings")
        self.assertEqual(status("timing"), "Cache: Cached • 2 settings")
        self.assertEqual(status("threshold"), "Cache: Cached • 3 settings")
        self.assertEqual(status("weighted"), "Cache: Cached • 2 settings")
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def test_threshold_duration_page_reports_longest_excursion(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        currents = np.array([[0.0], [2.0], [2.0], [0.0]])
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "regions": {
                "a": {
                    "name": "A", "constant": 0.0, "linear": [0.0],
                    "quadratic": [[1e-12]],
                }
            },
        }
        dialog = ThresholdDurationPage(payload, "a")
        self.wait_for_analysis(dialog)
        self.assertIn("Time above threshold: 2 s", dialog.summary.toPlainText())
        self.assertIn("Longest uninterrupted interval: 2 s", dialog.summary.toPlainText())
        self.assertEqual(dialog.table.rowCount(), 1)
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def test_frequency_weighted_page_exposes_profiles_and_owned_result_window(self):
        times = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
        currents = (100.0 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 10.0 * times))[:, None]
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "regions": {"a": {"name": "A", "constant": 0.0, "linear": [0.0], "quadratic": [[1e-12]]}},
        }
        with patch("fieldworkbench.exposure_actions.analyse_regions_frequency_weighted") as calculate:
            dialog = FrequencyWeightedPage(payload, "a")
            dialog.custom_mode.setChecked(True)
            dialog.samples.setValue(500)
            dialog.auto_mode.setChecked(True)
            calculate.assert_not_called()
        self.assertFalse(dialog.result_window.isVisible())
        self.assertIs(dialog.result_window.parentWidget(), dialog)
        self.assertFalse(dialog.result_window.isModal())
        self.assertEqual(dialog.result_window.windowModality(), Qt.WindowModality.NonModal)
        self.assertFalse(hasattr(dialog, "run_profile"))
        profile = dialog.profile.currentData()
        row, open_button = dialog._profile_actions[profile]
        self.assertEqual(dialog.profile_table.item(row, 1).text(), "Not calculated")
        self.assertTrue(open_button.isEnabled())

        # An uncached row calculates first; the same action becomes Open once the
        # completed all-region configuration is cached.
        self.assertEqual(open_button.text(), "Calculate")
        open_button.click()
        self.assertFalse(dialog.result_window.isVisible())
        request_key = dialog._request_key(profile)
        cached = dialog.analysis_cache[("weighted", "__all_regions__", profile, request_key[3])]["regions"]["a"]
        self.assertEqual(cached["weighted_cache_format"], "compact-v1")
        self.assertEqual(np.asarray(cached["weighted_index"]).dtype, np.dtype(np.float32))
        self.assertLessEqual(len(cached["spectral_top_rows"]), 30)
        for bulky_key in (
            "weighted_spatial_rms_uT", "frequency_hz", "bin_rms_uT",
            "reference_rms_by_bin_uT", "spectral_ratio_by_bin",
        ):
            self.assertNotIn(bulky_key, cached)
        self.assertIn(("spectrum", "__all_regions__", request_key[3], "rectangular", True), dialog.analysis_cache)
        self.assertIn("Ready", dialog.profile_table.item(row, 1).text())
        self.assertEqual(open_button.text(), "Open")
        with patch("fieldworkbench.exposure_actions.analyse_regions_frequency_weighted") as calculate:
            open_button.click()
            calculate.assert_not_called()
        self.assertTrue(dialog.result_window.isVisible())
        self.assertIs(dialog._analysis, cached)

        # The compact weighted result is independently viewable after its shared
        # rectangular Spectrum/FFT cache is removed through Manage cache.
        dialog.result_window.close()
        spectrum_key = ("spectrum", "__all_regions__", request_key[3], "rectangular", True)
        dialog.analysis_cache.pop(spectrum_key, None)
        dialog.analysis_cache.pop(("__cache_meta__",) + spectrum_key, None)
        with patch("fieldworkbench.exposure_actions.analyse_regions_frequency_weighted") as calculate:
            open_button.click()
            calculate.assert_not_called()
        self.assertTrue(dialog.result_window.isVisible())
        self.assertGreater(dialog.table.rowCount(), 0)
        self.assertIs(dialog._analysis, cached)

        # A different sampling key does not pretend to be cached and exposes Calculate.
        dialog.result_window.close()
        with patch("fieldworkbench.exposure_actions.analyse_regions_frequency_weighted") as calculate:
            dialog.custom_mode.setChecked(True)
            self.assertEqual(dialog.profile_table.item(row, 1).text(), "Not calculated")
            self.assertEqual(open_button.text(), "Calculate")
            self.assertTrue(open_button.isEnabled())
            calculate.assert_not_called()

        # Cached Auto results remain openable while another reference is running.
        dialog.auto_mode.setChecked(True)
        other = list(dialog._profile_actions)[1]
        other_row, other_open = dialog._profile_actions[other]
        dialog.payload["analysis_trace_reduced"] = True
        dialog.payload["authoritative_analysis"] = {}
        with patch("fieldworkbench.exposure_actions._start_analysis_job") as start:
            other_open.click()
            start.assert_called_once()
            self.assertEqual(dialog.profile_table.item(other_row, 1).text(), "Running…")
            self.assertFalse(other_open.isEnabled())
            self.assertEqual(open_button.text(), "Open")
            self.assertTrue(open_button.isEnabled())
            open_button.click()
            self.assertTrue(dialog.result_window.isVisible())
            self.assertEqual(dialog._analysis_progress_state, "running")
            self.assertTrue(dialog._analysis_busy)
        dialog._analysis_failed("test background failure")
        self.assertTrue(other_open.isEnabled())
        self.assertEqual(other_open.text(), "Calculate")
        self.assertIn("test background failure", dialog.error_label.text())

        # Verify the cached Auto result browser layout.
        dialog.view_profile(profile)
        self.assertGreaterEqual(dialog.profile.count(), 5)
        self.assertIn("ICNIRP 2010", dialog.profile.currentText())
        self.assertIn("weighted-peak index", dialog.summary.toPlainText())
        self.assertGreater(dialog.table.rowCount(), 0)
        self.assertEqual(dialog.result_splitter.orientation(), Qt.Orientation.Horizontal)
        self.assertFalse(dialog.result_splitter.childrenCollapsible())
        self.assertIs(dialog.result_splitter.widget(0), dialog.result_text_panel)
        self.assertIs(dialog.result_splitter.widget(1), dialog.result_plot_panel)
        self.assertIs(dialog.summary.parentWidget(), dialog.result_text_panel)
        self.assertIs(dialog.table.parentWidget(), dialog.result_text_panel)
        self.assertIs(dialog.plot.parentWidget(), dialog.result_plot_panel)
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def test_frequency_weighted_reference_queue_deduplicates_and_advances(self):
        times = np.linspace(0.0, 1.0, 101, dtype=np.float64)
        currents = np.sin(2.0 * np.pi * 10.0 * times)[:, None]
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "analysis_trace_reduced": True,
            "authoritative_analysis": {},
            "regions": {
                "a": {
                    "name": "A", "constant": 0.0, "linear": [0.0],
                    "quadratic": [[1e-12]],
                }
            },
        }
        dialog = FrequencyWeightedPage(payload, "a")
        # Mimic ownership by the target workspace so follow-up submission waits
        # for the shared worker-release callback instead of a direct-page timer.
        dialog._analysis_job_finished_callback = lambda _page: None
        profiles = list(dialog._profile_actions)
        first, second = profiles[:2]
        first_button = dialog._profile_actions[first][1]
        second_row, second_button = dialog._profile_actions[second]

        with patch("fieldworkbench.exposure_actions._start_analysis_job") as start:
            first_button.click()
            self.assertEqual(start.call_count, 1)
            self.assertTrue(dialog._analysis_busy)
            self.assertEqual(first_button.text(), "Running")

            second_button.click()
            second_button.click()  # Disabled/deduplicated; must not enqueue twice.
            self.assertIn((second, None), dialog._weighted_pending_set)
            self.assertEqual(dialog.profile_table.item(second_row, 1).text(), "Queued…")
            self.assertEqual(start.call_count, 1)

            dialog.calculate_all_profiles_button.click()
            self.assertEqual(len(dialog._weighted_pending_set), len(profiles) - 1)
            self.assertFalse(dialog.auto_mode.isEnabled())
            self.assertFalse(dialog.custom_mode.isEnabled())
            self.assertFalse(dialog.samples.isEnabled())

            # Completing the active reference does not jump ahead of the
            # target-wide scheduler. Once the workspace releases this page, the
            # next reference is submitted and the remainder stay pending.
            dialog._analysis_completed({"a": {"weighted_peak_index": 0.25}})
            self.assertFalse(dialog._analysis_busy)
            self.assertTrue(dialog._weighted_followup_pending)
            self.assertEqual(start.call_count, 1)
            dialog._analysis_job_released()
            self.assertEqual(start.call_count, 2)
            self.assertTrue(dialog._analysis_busy)
            self.assertEqual(dialog._analysis_request_key[2], second)
            self.assertEqual(len(dialog._weighted_pending_set), len(profiles) - 2)

        dialog._analysis_cancel_requested()
        self.assertEqual(dialog._weighted_pending_requests, [])
        self.assertEqual(dialog._weighted_pending_set, set())
        self.assertFalse(dialog._weighted_followup_pending)
        dialog._set_analysis_busy(False)
        self.assertTrue(dialog.auto_mode.isEnabled())
        self.assertTrue(dialog.custom_mode.isEnabled())
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def test_weighted_batch_tree_navigation_and_single_cache_notification(self):
        from fieldworkbench.region_analysis import analyse_regions_frequency_weighted
        from fieldworkbench.brain_analysis import lpba40_region_hierarchy
        nodes = list(lpba40_region_hierarchy().walk())[:5]
        times = np.linspace(0, 1, 101)
        currents = np.sin(2*np.pi*10*times)[:, None]
        payload = dict(times_b64=encode(times, np.float64), currents_b64=encode(currents, np.float64),
                       shape=list(currents.shape), drive_ids=["coil"], regions={
                           node.key: dict(name=node.name, constant=0., linear=[0.],
                                          quadratic=[[(i+1)*1e-12]]) for i, node in enumerate(nodes)})
        with patch("fieldworkbench.exposure_actions._start_analysis_job") as start:
            dialog = FrequencyWeightedPage(payload, nodes[0].key)
            start.assert_not_called()
            self.assertTrue(dialog.region.isHidden())
            self.assertIsNotNone(dialog.region_tree._region_items[nodes[1].key].parent())
            profile = dialog.profile.currentData()
            dialog.view_profile(profile)
            start.assert_called_once()
        results = analyse_regions_frequency_weighted(payload, profile=profile)
        with patch.object(dialog, "_analysis_cache_changed_callback") as notify:
            dialog._analysis_completed(results)
            notify.assert_called_once()
        with patch("fieldworkbench.exposure_actions._start_analysis_job") as start:
            dialog.result_region_tree.setCurrentItem(dialog.result_region_tree._region_items[nodes[-1].key])
            self.assertEqual(dialog.region.currentData(), nodes[-1].key)
            cached_result = dialog.analysis_cache[("weighted", "__all_regions__", profile, None)]["regions"][nodes[-1].key]
            self.assertIs(dialog._analysis, cached_result)
            self.assertEqual(dialog._analysis["weighted_peak_index"], results[nodes[-1].key]["weighted_peak_index"])
            self.assertNotIn("spectral_ratio_by_bin", dialog._analysis)
            self.assertIs(dialog.region_tree.currentItem(), dialog.region_tree._region_items[nodes[-1].key])
            dialog.view_profile(profile)
            start.assert_not_called()
        self.assertEqual(sum(k[0] == "weighted" for k in dialog.analysis_cache), 1)
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def test_fft_plot_max_frequency_updates_immediately_in_auto_mode(self):
        times = np.linspace(0.0, 1.0, 101, dtype=np.float64)
        currents = np.sin(2.0 * np.pi * 10.0 * times)[:, None]
        payload = {
            "times_b64": encode(times, np.float64),
            "currents_b64": encode(currents, np.float64),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "regions": {
                "a": {
                    "name": "A",
                    "constant": 0.0,
                    "linear": [0.0],
                    "quadratic": [[1e-12]],
                }
            },
        }
        dialog = FFTSpectrumPage(payload, "a")
        self.wait_for_analysis(dialog)
        self.assertTrue(dialog.auto_mode.isChecked())
        self.assertTrue(dialog.max_frequency.isEnabled())
        self.assertFalse(hasattr(dialog, "recalculate_button"))
        self.assertEqual(dialog.fullscreen_button.text(), "Fullscreen viewer")
        self.assertIn("Esc/F11", dialog.fullscreen_button.toolTip())
        with patch.object(dialog.plot, "toggle_fullscreen") as toggle_fullscreen:
            dialog.fullscreen_button.click()
            self.app.processEvents()
        toggle_fullscreen.assert_called_once_with()
        with (
            patch.object(dialog.plot, "set_figure") as set_figure,
            patch.object(dialog.plot, "request_reset_2d_view_on_next_figure") as reset_view,
        ):
            dialog.max_frequency.setValue(20.0)
            self.app.processEvents()
        figure = set_figure.call_args.args[0]
        reset_view.assert_called_once_with()
        self.assertEqual(figure["layout"]["xaxis"]["range"], [0.0, 20.0])
        self.assertFalse(figure["layout"]["xaxis"]["autorange"])
        self.assertTrue(figure["layout"]["yaxis"]["autorange"])
        self.assertLessEqual(max(figure["data"][0]["x"]), 20.0)
        self.assertIn("max=20", figure["layout"]["uirevision"])

        first_revision = figure["layout"]["uirevision"]
        with (
            patch.object(dialog.plot, "set_figure") as set_figure,
            patch.object(dialog.plot, "request_reset_2d_view_on_next_figure") as reset_view,
        ):
            dialog.recalculate()
        reset_view.assert_called_once_with()
        second_revision = set_figure.call_args.args[0]["layout"]["uirevision"]
        self.assertNotEqual(first_revision, second_revision)
        dialog.close()
        dialog.deleteLater()
        self.app.processEvents()

    def _exercise_export_restore(self, hidden):
        window = Window()
        window.resize(1200, 740)
        payload = {
            "measurement_regions": {
                "objects": [
                    {"key": "a", "name": "A", "label_id": 1},
                    {"key": "b", "name": "B", "label_id": 2},
                ],
                "metrics": metric_data(),
                "waveform": None,
            }
        }
        old = Viewer(payload)
        tab = window._wrap_render_result_view(old, {}, {"payload": payload})
        window.tabs.addTab(tab, "Playback")
        window.show()
        self.app.processEvents()
        panel = tab.region_panel
        panel.set_filter_rules([(1, ">", 99, "filter", "#facc15")])
        panel.table.setCurrentItem(panel._items_by_key["a"])
        if hidden:
            tab.toggle_regions()
        captured = []

        class Export(QDialog):
            output_path = None

            def __init__(self, data, parent=None):
                super().__init__(parent)
                captured.append(data)

            def exec(self):
                return QDialog.DialogCode.Rejected

        with (
            patch("fieldworkbench.waveform_gl_view.WaveformGLView", Viewer),
            patch("fieldworkbench.waveform_video_dialog.WaveformVideoExportDialog", Export),
        ):
            window._export_region_video()
            self.app.processEvents()
        self.assertTrue(old.cleaned)
        self.assertIs(tab.region_panel, panel)
        self.assertIsNot(tab.viewer, old)
        self.assertEqual(panel.filter_table.rowCount(), 1)
        self.assertEqual(tab.viewer.highlight, (1,))
        self.assertEqual(tab.splitter.sizes()[1] == 0, hidden)
        self.assertNotEqual(bool(captured[0].get("brain_region_video_table")), hidden)
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_export_restores_same_panel_filters_selection_and_visibility_visible(self):
        self._exercise_export_restore(False)

    def test_export_restores_same_panel_filters_selection_and_visibility_hidden(self):
        self._exercise_export_restore(True)


if __name__ == "__main__":
    unittest.main()
