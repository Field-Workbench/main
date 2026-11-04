from __future__ import annotations

import ast
import base64
import inspect
import json
import os
import signal
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSettings, QSignalBlocker, Qt
from PySide6.QtGui import QColor, QKeySequence, QPalette, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabBar,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import fieldworkbench.main_window as main_window_module

from fieldworkbench.exposure_panel import ExposureRegionPanel, ExposureResultTab
from fieldworkbench.main_window import (
    UnifiedFieldWorkspaceDialog,
    MainWindow,
    MeshImportOptionsDialog,
    SceneTreeWidget,
    USER_ROLE_ID,
)
from fieldworkbench.brain_analysis import lpba40_region_hierarchy
from fieldworkbench.brain_view import BrainViewDialog, _BrainResultRegionPanel
from fieldworkbench.dialogs import (
    AiHelpDialog,
    BackgroundFieldDialog,
    CollapsibleSection,
    CurrentNormalizationDialog,
    SceneExportDialog,
    SceneObjectImportDialog,
    DrcResultsDialog,
    FieldMapDialog,
    FieldVolumeMapDialog,
    MeasurementAnalysisProgressDialog,
    MeasurementCalibrationDialog,
    MeasurementDatasetEditorDialog,
    MeasurementResultsDialog,
    OptimizerDialog,
    OptimizerWorkerBenchmarkDialog,
    CustomProbeDialog,
    ProbeLibraryDialog,
    SensorPathProgressDialog,
    SensorPlotDialog,
    SnapshotComparisonDialog,
    ExposureAnalysisCacheManagerDialog,
    SnapshotManagerDialog,
    UpennGbmImportDialog,
    _BackgroundFieldRenderMixin,
    _FieldRenderConcurrencyLimiter,
    _FieldRenderJobPage,
    _FieldRenderWorker,
    _field_render_process_exit_message,
    _finish_field_map_progress,
    _show_field_map_progress,
    _update_field_map_progress,
)
from fieldworkbench.help_topics import CATEGORY_ORDER, HELP_TOPICS, HelpTopicsDialog
from fieldworkbench.measurement_library import (
    DatasetManagerWindow,
    MeasurementCompositeCalibrationDialog,
    MeasurementDatasetManagerDialog,
)
from fieldworkbench.project_context import ProjectContext
from fieldworkbench.project_manager import ProjectManagerWindow
from fieldworkbench.plot_view import PlotView
from fieldworkbench.waveform_gl_view import WaveformGLView
from fieldworkbench.waveform_dialog import WaveformPlaybackDialog
from fieldworkbench.waveform_video_dialog import WaveformVideoExportDialog
from fieldworkbench.theme import (
    DARK_THEME_COLORS,
    DARK_WORKBENCH_STYLE_SHEET,
    LIGHT_THEME_COLORS,
    WORKBENCH_STYLE_SHEET,
    apply_workbench_theme,
    current_theme,
)
from fieldworkbench.window_utils import (
    CompactDoubleSpinBox,
    _PersistentWindowStateFilter,
    _control_state,
    _restore_control_state,
    application_owned_windows,
    enable_standard_window_controls,
    install_editor_wheel_guard,
)
from fieldworkbench.studio_adapter import (
    FieldCalculationCancelled,
    StudioAdapter,
    awg_resistance_20_ohm_per_km,
)
from workflow_fixture import compact_workflow_scene


def _source_for(obj) -> str:
    """Return Python source for regression assertions, with a file/AST fallback.

    ``inspect.getsource`` can fail for otherwise ordinary Python callables when
    test runners or import hooks do not keep a matching linecache entry.  These
    smoke tests care about the checked-in source, so fall back to the defining
    file and the object's qualified name instead of turning that environment
    detail into a UI-test failure.
    """

    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        filename = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source = Path(filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
        parts = [part for part in getattr(obj, "__qualname__", "").split(".") if part != "<locals>"]
        body = tree.body
        node = None
        for part in parts:
            node = next(
                (
                    item
                    for item in body
                    if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == part
                ),
                None,
            )
            if node is None:
                raise OSError(f"Could not locate source for {getattr(obj, '__qualname__', obj)!r}")
            body = getattr(node, "body", [])
        segment = ast.get_source_segment(source, node) if node is not None else None
        if segment is not None:
            return segment
        if node is None or not hasattr(node, "end_lineno"):
            raise OSError(f"Could not extract source for {getattr(obj, '__qualname__', obj)!r}")
        lines = source.splitlines(keepends=True)
        return "".join(lines[node.lineno - 1 : node.end_lineno])


class UiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        cls.application = QApplication.instance() or QApplication([])

    def tearDown(self):
        """Release WebEngine pages even when an assertion aborts a test early."""
        for widget in QApplication.topLevelWidgets():
            plot_views = widget.findChildren(PlotView)
            if isinstance(widget, PlotView):
                plot_views.append(widget)
            for plot_view in set(plot_views):
                plot_view.cleanup()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()

    @staticmethod
    def make_window(**kwargs) -> MainWindow:
        return MainWindow(adapter=StudioAdapter(compact_workflow_scene()), **kwargs)

    def test_window_wheel_guard_covers_combo_and_spin_editors(self):
        scroll = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        focus_anchor = QLineEdit()
        focus_anchor.setText("focus stays here")
        combo = QComboBox()
        combo.addItems(["First", "Second", "Third"])
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(50)
        # Force the wheel-focus policy this regression is meant to remove;
        # platform styles may otherwise choose different defaults.
        combo.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        spin.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        layout.addWidget(focus_anchor)
        layout.addWidget(combo)
        layout.addWidget(spin)
        layout.addStretch(1)
        content.setMinimumHeight(1200)
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.resize(300, 200)
        scroll.show()
        guard = install_editor_wheel_guard(scroll)
        self.application.processEvents()
        self.assertIs(guard, getattr(scroll, "_field_workbench_editor_wheel_guard"))
        self.assertEqual(combo.focusPolicy(), Qt.FocusPolicy.StrongFocus)
        self.assertEqual(spin.focusPolicy(), Qt.FocusPolicy.StrongFocus)

        focus_anchor.setFocus()
        self.application.processEvents()
        self.assertIs(QApplication.focusWidget(), focus_anchor)

        scroll_bar = scroll.verticalScrollBar()
        self.assertGreater(scroll_bar.maximum(), 0)

        def wheel_up(widget):
            event = QWheelEvent(
                QPointF(5.0, 5.0),
                QPointF(5.0, 5.0),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.ScrollUpdate,
                False,
            )
            QApplication.sendEvent(widget, event)

        scroll_bar.setValue(min(500, scroll_bar.maximum()))
        before_scroll = scroll_bar.value()
        before_combo = combo.currentIndex()
        wheel_up(combo)
        self.assertEqual(combo.currentIndex(), before_combo)
        self.assertLess(scroll_bar.value(), before_scroll)
        self.assertIs(QApplication.focusWidget(), focus_anchor)

        scroll_bar.setValue(min(500, scroll_bar.maximum()))
        before_scroll = scroll_bar.value()
        before_spin = spin.value()
        wheel_up(spin)
        self.assertEqual(spin.value(), before_spin)
        self.assertLess(scroll_bar.value(), before_scroll)
        self.assertIs(QApplication.focusWidget(), focus_anchor)

        standalone_host = QWidget()
        standalone = QSpinBox(standalone_host)
        standalone.setRange(0, 100)
        standalone.setValue(50)
        install_editor_wheel_guard(standalone_host)
        wheel_up(standalone)
        self.assertEqual(standalone.value(), 50)
        standalone_host.deleteLater()
        scroll.deleteLater()

    def test_window_local_wheel_guard_allows_model_and_map_teardown(self):
        window = self.make_window()
        install_editor_wheel_guard(window)
        dialog = FieldMapDialog(window.adapter, window)
        install_editor_wheel_guard(dialog)
        dialog.show()
        self.application.processEvents()
        dialog.close()
        dialog.deleteLater()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()

    def close_project_manager_for_test(self, manager: ProjectManagerWindow) -> None:
        """Close a test project manager without opening an interactive save prompt."""
        try:
            for index in range(manager.tabs.count()):
                page = manager.tabs.widget(index)
                record = getattr(page, "record", None)
                context = getattr(record, "context", None)
                if context is not None:
                    context.mark_saved()
            manager.close()
            manager.deleteLater()
        except RuntimeError:
            pass
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()

    def test_video_export_coordinate_group_stacks_label_above_xyz(self):
        parent = QWidget()
        parent_layout = QVBoxLayout(parent)
        editors = WaveformVideoExportDialog._coordinate_group(
            parent_layout, "Camera position"
        )

        group = parent_layout.itemAt(0).widget()
        self.assertIsNotNone(group)
        grid = group.layout()
        section_label = next(
            label
            for label in group.findChildren(QLabel)
            if label.text() == "Camera position"
        )
        self.assertEqual(
            grid.getItemPosition(grid.indexOf(section_label)), (0, 0, 1, 3)
        )
        for column, editor in enumerate(editors):
            self.assertEqual(grid.getItemPosition(grid.indexOf(editor)), (2, column, 1, 1))
            self.assertGreaterEqual(editor.minimumWidth(), 120)
        parent.deleteLater()

    def test_file_menu_exposes_portable_scene_export_dialog(self):
        window = self.make_window()
        self.assertEqual(window.export_scene_action.text(), "Export…")
        self.assertIn(window.export_scene_action, window.file_menu.actions())

        dialog = SceneExportDialog(window.adapter, window)
        self.assertGreaterEqual(dialog.width(), 1320)
        self.assertGreaterEqual(dialog.height(), 760)
        self.assertEqual(dialog.translator(), "openscad")
        checked = set(dialog.selected_object_ids())
        self.assertEqual(checked, {"coil_a", "coil_b"})
        sensor_item = dialog._items["axis_sensor"]
        self.assertFalse(sensor_item.flags() & Qt.ItemFlag.ItemIsEnabled)
        self.assertTrue(dialog.include_construction())
        self.assertTrue(dialog.femm_slice_row.isHidden())
        self.assertTrue(dialog.preview_tabs.isTabVisible(dialog.scene_preview_tab_index))
        self.assertFalse(dialog.preview_tabs.isTabVisible(dialog.slice_preview_tab_index))

        femm_index = dialog.translator_selector.findData("femm")
        dialog.translator_selector.setCurrentIndex(femm_index)
        self.assertTrue(dialog._records["coil_a"]["supported"])
        self.assertFalse(dialog._records["axis_sensor"]["supported"])
        self.assertTrue(dialog.include_construction_checkbox.isHidden())
        self.assertFalse(dialog.apply_field_scale_checkbox.isHidden())
        self.assertFalse(dialog.apply_field_scale())
        self.assertFalse(dialog.femm_slice_row.isHidden())
        self.assertEqual(dialog.femm_reference_slice_plane(), "auto")
        self.assertEqual(dialog.femm_reference_slice_offset_mm(), 0.0)
        self.assertEqual(dialog.femm_reference_half(), "positive")
        self.assertTrue(dialog.femm_slice_offset.isReadOnly())
        self.assertIn("common axis", dialog.femm_slice_offset.text())
        self.assertTrue(dialog.preview_tabs.isTabVisible(dialog.slice_preview_tab_index))
        self.assertFalse(dialog.selection_validation_hint.isHidden())
        self.assertIn("automatically map", dialog.selection_validation_hint.text())
        self.assertTrue(dialog.export_button.isEnabled())

        getdp_index = dialog.translator_selector.findData("getdp")
        dialog.translator_selector.setCurrentIndex(getdp_index)
        self.assertTrue(dialog._records["coil_a"]["supported"])
        self.assertTrue(dialog._records["axis_sensor"]["supported"])
        self.assertFalse(dialog.getdp_options_box.isHidden())
        self.assertTrue(dialog.getdp_use_physical_construction())
        self.assertEqual(dialog.getdp_mesh_preset(), "standard")
        self.assertAlmostEqual(dialog.getdp_source_elements(), 1.8)
        self.assertEqual(dialog.getdp_near_radius_divisions(), 14.0)
        self.assertEqual(dialog.getdp_far_radius_divisions(), 8.0)
        self.assertTrue(dialog.getdp_advanced_widget.isHidden())
        self.assertIn("Derived characteristic lengths", dialog.getdp_plan_hint.text())
        self.assertIn("Automatic air/shell radii", dialog.getdp_plan_hint.text())
        self.assertIn("2 3D circular sources", dialog.selection_validation_hint.text())
        self.assertTrue(dialog.export_button.isEnabled())
        validation_index = dialog.getdp_mesh_preset_selector.findData("validation")
        dialog.getdp_mesh_preset_selector.setCurrentIndex(validation_index)
        self.assertEqual(dialog.getdp_source_elements(), 3.0)
        self.assertEqual(dialog.getdp_near_radius_divisions(), 32.0)
        self.assertEqual(dialog.getdp_far_radius_divisions(), 12.0)
        dialog.getdp_advanced_toggle.setChecked(True)
        self.assertFalse(dialog.getdp_advanced_widget.isHidden())
        dialog.getdp_source_elements_spin.setValue(5.0)
        self.assertEqual(dialog.getdp_mesh_preset(), "custom")
        standard_index = dialog.getdp_mesh_preset_selector.findData("standard")
        dialog.getdp_mesh_preset_selector.setCurrentIndex(standard_index)
        dialog._items["coil_b"].setCheckState(0, Qt.CheckState.Unchecked)
        self.assertIn("five generated verification points", dialog.selection_validation_hint.text())
        self.assertTrue(dialog.export_button.isEnabled())
        dialog._items["axis_sensor"].setCheckState(0, Qt.CheckState.Checked)
        self.assertIn("61 selected sensor sample points", dialog.selection_validation_hint.text())
        self.assertFalse(dialog.apply_field_scale_checkbox.isHidden())

        comsol_index = dialog.translator_selector.findData("comsol")
        dialog.translator_selector.setCurrentIndex(comsol_index)
        self.assertTrue(dialog._records["coil_a"]["supported"])
        self.assertTrue(dialog._records["coil_b"]["supported"])
        self.assertTrue(dialog.apply_field_scale_checkbox.isEnabled())
        self.assertTrue(dialog.femm_slice_row.isHidden())

        portable_index = dialog.translator_selector.findData("portable")
        dialog.translator_selector.setCurrentIndex(portable_index)
        self.assertIn("axis_sensor", dialog._records)
        self.assertTrue(dialog._records["axis_sensor"]["supported"])
        self.assertFalse(dialog.include_construction_checkbox.isEnabled())
        self.assertTrue(dialog.apply_field_scale_checkbox.isHidden())

    def test_getdp_options_wheel_scrolls_without_changing_editor_values(self):
        window = self.make_window()
        dialog = SceneExportDialog(window.adapter, window)
        getdp_index = dialog.translator_selector.findData("getdp")
        dialog.translator_selector.setCurrentIndex(getdp_index)
        dialog.getdp_advanced_toggle.setChecked(True)

        scroll_bar = dialog.getdp_options_scroll.verticalScrollBar()
        scroll_bar.setRange(0, 1000)

        def wheel_up(widget):
            event = QWheelEvent(
                QPointF(5.0, 5.0),
                QPointF(5.0, 5.0),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.ScrollUpdate,
                False,
            )
            QApplication.sendEvent(widget, event)

        scroll_bar.setValue(500)
        preset_before = dialog.getdp_mesh_preset_selector.currentIndex()
        wheel_up(dialog.getdp_mesh_preset_selector)
        self.assertEqual(dialog.getdp_mesh_preset_selector.currentIndex(), preset_before)
        self.assertLess(scroll_bar.value(), 500)

        scroll_bar.setValue(500)
        source_before = dialog.getdp_source_elements_spin.value()
        wheel_up(dialog.getdp_source_elements_spin)
        self.assertEqual(dialog.getdp_source_elements_spin.value(), source_before)
        self.assertLess(scroll_bar.value(), 500)

        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def assert_standard_window_controls(self, dialog: QDialog) -> None:
        flags = dialog.windowFlags()
        # Qt/Win32 may consume CustomizeWindowHint while normalizing the native
        # title bar even though the requested minimize/maximize/close controls are
        # retained.  Assert the user-visible contract, not that transient helper
        # implementation flag.
        for hint in (
            Qt.WindowType.WindowMinimizeButtonHint,
            Qt.WindowType.WindowMaximizeButtonHint,
            Qt.WindowType.WindowCloseButtonHint,
        ):
            self.assertTrue(bool(flags & hint), f"missing window flag {hint}")
        self.assertFalse(bool(flags & Qt.WindowType.WindowContextHelpButtonHint))

    def test_standard_window_control_helper_requests_minimize_maximize_and_close(self):
        dialog = QDialog()
        enable_standard_window_controls(dialog)
        self.assert_standard_window_controls(dialog)
        dialog.deleteLater()

    def test_persistent_state_skips_deleted_dynamic_control_wrappers(self):
        dialog = QDialog()
        dialog.dynamic_value = CompactDoubleSpinBox(dialog)
        stale_wrapper = dialog.dynamic_value
        state_filter = _PersistentWindowStateFilter(dialog)

        stale_wrapper.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertIsNone(_control_state(stale_wrapper))

        settings = mock.Mock()
        with (
            mock.patch(
                "fieldworkbench.window_utils._application_uses_persistent_settings",
                return_value=True,
            ),
            mock.patch("fieldworkbench.window_utils.QSettings", return_value=settings),
        ):
            # Inspector rebuilds retain exactly this kind of dead Python wrapper.
            # Closing the top-level window must omit it instead of raising from
            # QObject.property() inside the Python event-filter override.
            state_filter._save()
        settings.sync.assert_called_once_with()
        dialog.deleteLater()

    def test_persistent_state_container_opt_out_protects_dynamic_document_editors(self):
        dialog = QDialog()
        dynamic_form = QWidget(dialog)
        dynamic_form.setProperty("persistUiState", False)
        name_editor = QLineEdit("Coil 1", dynamic_form)

        self.assertIsNone(_control_state(name_editor))
        _restore_control_state(
            name_editor,
            {"kind": "lineedit", "text": "Helmholtz Coils"},
        )
        self.assertEqual(name_editor.text(), "Coil 1")
        dialog.deleteLater()

    def test_persistent_state_saves_once_for_close_then_hide(self):
        dialog = QDialog()
        state_filter = _PersistentWindowStateFilter(dialog)
        state_filter._save = mock.Mock()

        state_filter.eventFilter(dialog, QEvent(QEvent.Type.Close))
        state_filter.eventFilter(dialog, QEvent(QEvent.Type.Hide))

        state_filter._save.assert_called_once_with()
        dialog.deleteLater()

    def test_upenn_gbm_import_dialog_defaults_to_tumor_core_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog = UpennGbmImportDialog(cache_directory=directory)
            self.assertEqual(dialog.cases.topLevelItemCount(), 147)
            self.assertEqual(dialog.selected_region_keys(), ["tumor_core"])
            self.assertEqual(dialog.role.currentData(), "measurement")
            self.assertEqual(dialog.cases.currentItem().text(0), "UPENN-GBM-00002")
            self.assertEqual(dialog.refresh_button.text(), "Refresh catalogue")
            self.assertEqual(
                [dialog.cases.headerItem().text(column) for column in range(6)],
                [
                    "Case",
                    "Approximate location",
                    "Tumor core (mL)",
                    "Whole tumor (mL)",
                    "Extent X × Y × Z (mm)",
                    "Status",
                ],
            )
            dialog.search.setText("sub-006")
            visible = [
                dialog.cases.topLevelItem(index).text(0)
                for index in range(dialog.cases.topLevelItemCount())
                if not dialog.cases.topLevelItem(index).isHidden()
            ]
            self.assertEqual(visible, ["UPENN-GBM-00006"])
            dialog.deleteLater()

    def test_plot_view_fullscreen_uses_true_fullscreen_renderer_in_modal_family(self):
        # Use a QDialog owner because the real 2D/3D map windows are launched via
        # exec(). The detached top-level viewer must remain a child of that modal
        # window family or Qt can paint it while refusing all activation/input.
        parent = QDialog()
        layout = QVBoxLayout(parent)
        plot = PlotView(parent)
        layout.addWidget(plot, 1)

        original_web = plot.web
        plot.enter_fullscreen()
        self.assertTrue(plot.is_fullscreen_view())
        self.assertIs(plot.parentWidget(), parent)
        self.assertEqual(layout.indexOf(plot), 0)
        self.assertFalse(plot.isEnabled())
        self.assertIsNotNone(plot._fullscreen_host)
        self.assertIs(plot._fullscreen_host.parentWidget(), parent)
        self.assertTrue(plot._fullscreen_host.isWindow())
        self.assertTrue(bool(plot._fullscreen_host.windowFlags() & Qt.WindowType.FramelessWindowHint))
        self.assertTrue(plot._fullscreen_host.isFullScreen())
        self.assertIsNotNone(plot._fullscreen_clone)
        self.assertIsNot(plot._fullscreen_clone, plot)
        if original_web is not None and plot._fullscreen_clone.web is not None:
            self.assertIsNot(plot._fullscreen_clone.web, original_web)

        plot.exit_fullscreen()
        self.assertFalse(plot.is_fullscreen_view())
        self.assertIs(plot.parentWidget(), parent)
        self.assertEqual(layout.indexOf(plot), 0)
        self.assertTrue(plot.isEnabled())

        # Deliver the queued post-show activation after teardown.  This used to
        # call a bound C++ host method after Shiboken had deleted the host.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()

        plot.cleanup()
        parent.deleteLater()

    def test_plot_view_cleanup_retires_embedded_webengine_surface(self):
        view = PlotView()
        web = view.web

        view.cleanup()
        self.assertTrue(view._cleaned_up)
        self.assertIsNone(view.web)
        self.assertFalse(view._pending_view_callbacks)
        if web is not None:
            self.assertEqual(view.indexOf(web), -1)

        # All normal close paths and the application-wide shutdown hook can
        # converge here, so cleanup must remain harmless when called twice.
        view.cleanup()
        view.deleteLater()

    def test_theme_draws_visible_unchecked_checkbox_indicator(self):
        self.assertIn("QCheckBox::indicator:unchecked", WORKBENCH_STYLE_SHEET)
        self.assertIn("border: 1px solid #64748b", WORKBENCH_STYLE_SHEET)

    def test_benchmark_overview_quality_border_scale(self):
        self.assertEqual(
            SnapshotComparisonDialog._benchmark_quality_colour(0.0, higher_is_better=True),
            "#dc2626",
        )
        self.assertEqual(
            SnapshotComparisonDialog._benchmark_quality_colour(50.0, higher_is_better=True),
            "#f59e0b",
        )
        self.assertEqual(
            SnapshotComparisonDialog._benchmark_quality_colour(100.0, higher_is_better=True),
            "#16a34a",
        )
        self.assertEqual(
            SnapshotComparisonDialog._benchmark_quality_colour(0.0, higher_is_better=False),
            "#16a34a",
        )
        self.assertEqual(
            SnapshotComparisonDialog._benchmark_quality_colour(100.0, higher_is_better=False),
            "#dc2626",
        )
        self.assertEqual(
            SnapshotComparisonDialog._benchmark_quality_colour(250.0, higher_is_better=False),
            "#dc2626",
        )

    def test_benchmark_overall_score_averages_three_quality_components(self):
        entry = {
            "target_coverage_pct": 79.1667,
            "uniformity_spread_pct": 12.7677,
            "directional_consistency_pct": 99.9201,
        }
        expected = (79.1667 + (100.0 - 12.7677) + 99.9201) / 3.0
        self.assertAlmostEqual(
            SnapshotComparisonDialog._benchmark_overall_score(entry),
            expected,
        )
        self.assertEqual(
            SnapshotComparisonDialog._benchmark_overall_score(
                {
                    "target_coverage_pct": 120.0,
                    "uniformity_spread_pct": 250.0,
                    "directional_consistency_pct": 110.0,
                }
            ),
            200.0 / 3.0,
        )

    def test_match_overall_score_averages_three_quality_components(self):
        stats = {
            "mean_intensity_bias_pct": -1.61987,
            "relative_rms_vector_difference_pct": 3.45188,
            "match_coverage_pct": 91.6667,
        }
        expected = ((100.0 - 1.61987) + (100.0 - 3.45188) + 91.6667) / 3.0
        self.assertAlmostEqual(
            SnapshotComparisonDialog._match_overall_score(stats),
            expected,
        )
        self.assertEqual(
            SnapshotComparisonDialog._match_overall_score(
                {
                    "mean_intensity_bias_pct": -250.0,
                    "relative_rms_vector_difference_pct": 150.0,
                    "match_coverage_pct": 120.0,
                }
            ),
            100.0 / 3.0,
        )

    def test_current_normalization_dialog_inherits_scene_groups_without_catch_all(self):
        adapter = mock.Mock()
        adapter.list_snapshots.return_value = [{"id": "candidate", "name": "Candidate"}]
        adapter.snapshot_normalization_coils.return_value = [
            {
                "id": "coil_a",
                "label": "FRILZ Left",
                "current_a": 0.1,
                "group_path": [{"id": "frilz", "label": "FRILZ Coils"}],
            },
            {
                "id": "coil_b",
                "label": "FRILZ Right",
                "current_a": 0.1,
                "group_path": [{"id": "frilz", "label": "FRILZ Coils"}],
            },
            {
                "id": "shim",
                "label": "Shim Coil",
                "current_a": 0.02,
                "group_path": [],
            },
            {
                "id": "solo_group",
                "label": "Solo Coil",
                "current_a": 0.03,
                "group_path": [{"id": "solo", "label": "One Coil Group"}],
            },
        ]
        dialog = CurrentNormalizationDialog(adapter, ["candidate"])

        self.assertEqual(len(dialog._groups["candidate"]), 1)
        inherited = next(iter(dialog._groups["candidate"].values()))
        self.assertEqual(inherited["name"], "FRILZ Coils")
        self.assertEqual(inherited["coil_ids"], ["coil_a", "coil_b"])
        self.assertNotIn(("candidate", "coil_a"), dialog._coil_combos)
        self.assertNotIn(("candidate", "coil_b"), dialog._coil_combos)
        self.assertIn(("candidate", "shim"), dialog._coil_combos)
        self.assertIn(("candidate", "solo_group"), dialog._coil_combos)
        self.assertEqual(dialog._coil_items[("candidate", "shim")].text(0), "Shim Coil")
        self.assertEqual(
            dialog._ungrouped_modes["candidate"]["shim"], "independent"
        )

    def test_current_normalization_dialog_does_not_auto_group_root_coils(self):
        adapter = mock.Mock()
        adapter.list_snapshots.return_value = [{"id": "candidate", "name": "Candidate"}]
        adapter.snapshot_normalization_coils.return_value = [
            {"id": "coil_a", "label": "Coil A", "current_a": 0.1, "group_path": []},
            {"id": "coil_b", "label": "Coil B", "current_a": 0.1, "group_path": []},
        ]
        dialog = CurrentNormalizationDialog(adapter, ["candidate"])
        self.assertEqual(dialog._groups["candidate"], {})
        self.assertIn(("candidate", "coil_a"), dialog._coil_combos)
        self.assertIn(("candidate", "coil_b"), dialog._coil_combos)

    def test_current_normalization_dialog_uses_visual_fit_groups(self):
        adapter = mock.Mock()
        adapter.list_snapshots.return_value = [{"id": "candidate", "name": "Candidate"}]
        adapter.snapshot_normalization_coils.return_value = [
            {"id": "coil_a", "label": "Coil A", "current_a": 0.1},
            {"id": "coil_b", "label": "Coil B", "current_a": 0.1},
            {"id": "coil_c", "label": "Coil C", "current_a": 0.05},
        ]
        dialog = CurrentNormalizationDialog(
            adapter,
            ["candidate"],
            selected_assignments={
                "candidate": {
                    "coil_a": "Primary pair",
                    "coil_b": "Primary pair",
                    "coil_c": "independent",
                }
            },
        )
        self.assertEqual(len(dialog._group_combos), 1)
        self.assertNotIn(("candidate", "coil_a"), dialog._coil_combos)
        self.assertNotIn(("candidate", "coil_b"), dialog._coil_combos)
        self.assertIn(("candidate", "coil_c"), dialog._coil_combos)
        assignments = dialog.selected_assignments()["candidate"]
        self.assertEqual(assignments["coil_a"], "Primary pair")
        self.assertEqual(assignments["coil_b"], "Primary pair")
        self.assertEqual(assignments["coil_c"], "independent")

        new_group = dialog._create_group(
            "candidate", ["coil_a", "coil_c"], name="Mixed pair"
        )
        dialog._rebuild_tree({("fit_group", "candidate", new_group)})
        regrouped = dialog.selected_assignments()["candidate"]
        self.assertEqual(regrouped["coil_a"], "Mixed pair")
        self.assertEqual(regrouped["coil_c"], "Mixed pair")
        dialog.deleteLater()

    def test_shared_light_theme_restores_workbench_light_appearance(self):
        # Simulate a dark desktop/application palette first. Light must restore
        # Field Workbench's established custom light palette and stylesheet.
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor("#202124"))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor("#111827"))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor("#f8fafc"))
        self.application.setPalette(dark_palette)

        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "theme.ini"), QSettings.Format.IniFormat
            )
            window = self.make_window(settings=settings)
            self.application.processEvents()
            palette = self.application.palette()
            self.assertEqual(
                palette.color(QPalette.ColorRole.Window), QColor(LIGHT_THEME_COLORS["window"])
            )
            self.assertEqual(
                palette.color(QPalette.ColorRole.Base), QColor(LIGHT_THEME_COLORS["panel"])
            )
            self.assertEqual(
                palette.color(QPalette.ColorRole.Text), QColor(LIGHT_THEME_COLORS["text"])
            )
            self.assertEqual(
                window.palette().color(QPalette.ColorRole.Window),
                palette.color(QPalette.ColorRole.Window),
            )
            self.assertTrue(self.application.styleSheet())
            self.assertIn("font-size: 10pt", self.application.styleSheet())

    def test_preferences_theme_choice_applies_and_persists_wavebuilder_dark_palette(self):
        self.assertEqual(DARK_THEME_COLORS["window"], "#20252b")
        self.assertEqual(DARK_THEME_COLORS["panel"], "#2a3037")
        self.assertEqual(DARK_THEME_COLORS["button"], "#303740")
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "appearance.ini"), QSettings.Format.IniFormat
            )
            window = self.make_window(settings=settings)
            captured: dict[str, object] = {}

            class PreferencesDialog(QDialog):
                def exec(self):
                    selector = self.findChild(QComboBox, "themeSelector")
                    captured["items"] = [
                        selector.itemText(index) for index in range(selector.count())
                    ]
                    selector.setCurrentIndex(selector.findData("dark"))
                    QApplication.processEvents()
                    captured["selector_base"] = selector.palette().color(
                        QPalette.ColorRole.Base
                    ).name()
                    captured["dialog_window"] = self.palette().color(
                        QPalette.ColorRole.Window
                    ).name()
                    return QDialog.DialogCode.Rejected

            try:
                with mock.patch.object(main_window_module, "QDialog", PreferencesDialog):
                    window.open_preferences()
                self.assertEqual(captured["items"], ["Light", "Dark"])
                self.assertEqual(captured["selector_base"], "#9ea4aa")
                # Dark dialogs intentionally use the explicit silver-grey QDialog
                # surface override rather than the application Window palette colour.
                self.assertEqual(captured["dialog_window"], "#2f353b")
                self.assertEqual(settings.value("appearance/theme"), "dark")
                self.assertEqual(current_theme(), "dark")
                palette = self.application.palette()
                self.assertEqual(
                    palette.color(QPalette.ColorRole.Window).name(), "#20252b"
                )
                self.assertEqual(
                    palette.color(QPalette.ColorRole.Base).name(), "#2a3037"
                )
                self.assertEqual(
                    palette.color(QPalette.ColorRole.Button).name(), "#303740"
                )
                self.assertEqual(
                    self.application.styleSheet(), DARK_WORKBENCH_STYLE_SHEET
                )
                self.assertIn(
                    "QDialog, QMessageBox, QProgressDialog",
                    DARK_WORKBENCH_STYLE_SHEET,
                )
                self.assertIn("background-color: #2f353b;", DARK_WORKBENCH_STYLE_SHEET)
                self.assertEqual(
                    window.tree.palette().color(QPalette.ColorRole.Base).name(),
                    "#9ea4aa",
                )
                open_icon = window.main_toolbar_buttons["open"].icon().pixmap(24, 24)
                icon_image = open_icon.toImage()
                opaque_colours = {
                    icon_image.pixelColor(x, y).name()
                    for y in range(icon_image.height())
                    for x in range(icon_image.width())
                    if icon_image.pixelColor(x, y).alpha() > 0
                }
                self.assertIn(DARK_THEME_COLORS["heading"], opaque_colours)
            finally:
                apply_workbench_theme(theme="light")

    def test_dark_plot_surface_matches_wavebuilder_high_contrast_canvas(self):
        view = PlotView()
        view._workbench_theme = "dark"
        source = json.dumps(
            {
                "data": [{"type": "scatter", "x": [0, 1], "y": [0, 1]}],
                "layout": {"title": {"text": "Theme check"}},
            }
        )
        themed = json.loads(view._themed_figure_json(source))
        self.assertEqual(themed["layout"]["paper_bgcolor"], "#aeb3b8")
        self.assertEqual(themed["layout"]["plot_bgcolor"], "#aeb3b8")
        self.assertEqual(themed["layout"]["font"]["color"], "#14181c")
        view.cleanup()


    def test_scene_object_import_dialog_group_checks_and_multi_delete_button(self):
        catalog = [
            {"id": "group", "label": "Imported group", "type": "Collection", "type_label": "Group", "parent": ""},
            {"id": "coil", "label": "Imported coil", "type": "current.Circle", "type_label": "Circular coil", "parent": "group"},
            {"id": "sensor", "label": "Imported sensor", "type": "Sensor", "type_label": "Sensor", "parent": ""},
        ]
        dialog = SceneObjectImportDialog(catalog, "source.magpy.json")
        self.assertFalse(dialog.import_button.isEnabled())
        group_item = dialog._items["group"]
        group_item.setCheckState(0, Qt.CheckState.Checked)
        self.application.processEvents()
        self.assertEqual(set(dialog.checked_ids()), {"group", "coil"})
        self.assertTrue(dialog.import_button.isEnabled())

        window = self.make_window()
        self.application.processEvents()
        with QSignalBlocker(window.tree):
            window.tree.clearSelection()
            window.tree.setCurrentItem(window._tree_items["coil_a"])
            window._tree_items["coil_a"].setSelected(True)
            window._tree_items["coil_b"].setSelected(True)
        window.selected_ids = ["coil_a", "coil_b"]
        window.selected_id = "coil_a"
        window.populate_inspector(window.selected_id)
        window._update_actions()
        self.assertTrue(window.delete_button.isEnabled())
        history_before = window.adapter.history()["undo"]
        with mock.patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window.delete_selected()
        self.assertEqual(
            {item["id"] for item in window.adapter.list_objects()},
            {"axis_sensor"},
        )
        self.assertEqual(window.adapter.history()["undo"], history_before + 1)

    def test_window_scene_tree_and_inspector(self):
        window = self.make_window()
        self.application.processEvents()
        self.assertFalse(hasattr(window, "document_tabs"))
        self.assertFalse(hasattr(window, "documents"))
        self.assertEqual(window.tree.topLevelItemCount(), 3)
        self.assertEqual(window.selected_id, "coil_a")
        self.assertEqual(window.selected_ids, ["coil_a"])
        self.assertEqual(
            list(window.palette_buttons),
            [
                "Coil",
                "Magnet",
                "Sensor",
                "Background",
                "Object",
                "Import object…",
                "Import UPENN-GBM case…",
            ],
        )
        self.assertIsNone(window.palette_buttons["Background"].menu())
        self.assertEqual(
            [action.text() for action in window.palette_buttons["Object"].menu().actions()],
            [
                "Box", "Cylinder", "Sphere",
                "6-well plate", "12-well plate", "24-well plate",
                "48-well plate", "96-well plate",
                "Generic bust", "SRI24 brain",
            ],
        )
        self.assertEqual(
            [action.text() for action in window.palette_buttons["Sensor"].menu().actions()],
            ["Point sensor", "Linear axis sensor", "Magnetometer probe…"],
        )
        self.assertEqual(window.new_group_button.text(), "+")
        self.assertEqual(window.new_group_button.toolTip(), "New empty group")
        self.assertFalse(hasattr(window, "refresh_button"))
        self.assertEqual(window.duplicate_button.text(), "Clone")
        self.assertEqual(window.new_action.text(), "New scene")
        self.assertEqual(window.import_objects_action.text(), "Import objects from scene…")
        self.assertFalse(hasattr(window, "new_default_action"))
        self.assertEqual(window.notes_action.text(), "Notes")
        self.assertEqual(window.drc_action.text(), "Check design")
        self.assertFalse(hasattr(window, "point_probe_action"))
        for retired_action in (
            "snapshots_action",
            "field_volume_map_action",
            "brain_view_action",
            "maps_action",
            "exposures_action",
            "static_workspace_action",
            "playback_workspace_action",
            "measurement_dataset_action",
            "new_scene_tab_button",
        ):
            self.assertFalse(hasattr(window, retired_action))
        self.assertEqual(
            list(window.main_toolbar_buttons),
            [
                "open",
                "save",
                "undo",
                "redo",
                "sensor",
                "volume",
                "optimize",
                "check",
                "notes",
            ],
        )
        self.assertEqual(
            [window.main_toolbar_buttons[key].text() for key in list(window.main_toolbar_buttons)[4:]],
            [
                "Sensor plot",
                "Volume analysis",
                "Optimize",
                "DRC",
                "Notes",
            ],
        )
        for key in ("open", "save", "undo", "redo"):
            self.assertEqual(window.main_toolbar_buttons[key].text(), "")
            self.assertEqual(window.main_toolbar_buttons[key].size().width(), 36)
            self.assertEqual(window.main_toolbar_buttons[key].size().height(), 36)
            self.assertEqual(window.main_toolbar_buttons[key].iconSize().width(), 24)
            self.assertEqual(window.main_toolbar_buttons[key].iconSize().height(), 24)
        self.assertFalse(hasattr(window, "main_toolbar_dividers"))
        toolbar_layout = window.main_toolbar_widget.layout()
        self.assertEqual(toolbar_layout.count(), 9)
        self.assertEqual(
            [toolbar_layout.stretch(i) for i in range(toolbar_layout.count())],
            [0] * 4 + [1] * 5,
        )
        for key in list(window.main_toolbar_buttons)[4:]:
            button = window.main_toolbar_buttons[key]
            self.assertEqual(button.minimumWidth(), 0)
            self.assertEqual(
                button.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Ignored
            )
        self.assertEqual(
            ["|" if action.isSeparator() else action.text() for action in window.analysis_menu.actions()],
            [
                "Sensor plot",
                "Volume analysis",
                "|",
                "Optimize design",
                "Check design",
                "|",
                "Notes",
            ],
        )
        self.assertFalse(hasattr(window, "direct_tools_menu"))
        self.assertIsNotNone(window.scene_tools)
        self.assertIsNotNone(window.fullscreen_view_button)
        self.assertEqual(
            list(window.view_preset_buttons),
            ["front", "right", "top"],
        )
        self.assertTrue(window.windowTitle().endswith("Field Workbench"))
        self.assertNotIn("current", window.parameter_editors)
        self.assertIn("enabled", window.coil_widgets)
        self.assertTrue(window.coil_widgets["enabled"].isChecked())
        self.assertIn("resistance_20_ohm_per_km", window.coil_widgets)
        self.assertIn("resistance_mode", window.coil_widgets)
        self.assertIn("resistance_override_ohm", window.coil_widgets)
        self.assertIn("inductance_mode", window.coil_widgets)
        self.assertIn("inductance_override_h", window.coil_widgets)
        self.assertIn("field_scale_factor", window.coil_widgets)
        self.assertIn("correction_mode", window.coil_widgets)
        self.assertIn("correction_calibration_id", window.coil_widgets)
        self.assertEqual(window.coil_widgets["correction_mode"].currentData(), "custom")
        self.assertIn("physical_geometry_enabled", window.coil_widgets)
        self.assertIn("winding_radial_build_mode", window.coil_widgets)
        self.assertAlmostEqual(window.coil_widgets["field_scale_factor"].value(), 1.0)
        self.assertFalse(window.coil_envelope_group.isChecked())
        self.assertEqual(
            window.coil_widgets["winding_radial_build_mode"].currentData(), "auto"
        )
        self.assertEqual(window.coil_widgets["current_mode"].currentData(), "rms_dc")
        self.assertEqual(window.coil_widgets["resistance_mode"].currentData(), "estimated")
        self.assertEqual(window.coil_widgets["inductance_mode"].currentData(), "estimated")
        self.assertFalse(window.coil_widgets["resistance_override_ohm"].isEnabled())
        self.assertFalse(window.coil_widgets["inductance_override_h"].isEnabled())

        inspector_labels = {label.text() for label in window.findChildren(QLabel)}
        self.assertIn("Mean diameter (mm)", inspector_labels)
        self.assertIn("Mean radius (reference)", inspector_labels)
        self.assertIn("Turns", inspector_labels)
        radius_label = window.coil_estimate_labels["radius"]
        self.assertTrue(
            radius_label.textInteractionFlags()
            & Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.assertTrue(
            radius_label.textInteractionFlags()
            & Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.assertIsNotNone(window.coil_estimate_copy_button)
        self.assertEqual(window.inspector_scroll.minimumWidth(), 350)
        self.assertEqual(window.inspector_scroll.maximumWidth(), 560)
        inline_apply_buttons = [
            button
            for button in window.findChildren(QPushButton)
            if button.objectName() == "inlineApplyButton"
        ]
        self.assertGreaterEqual(len(inline_apply_buttons), 4)
        window.coil_estimate_copy_button.click()
        copied = QApplication.clipboard().text()
        self.assertIn("Calculated winding estimates — Lower coil", copied)
        self.assertIn("Mean radius (reference): 100 mm", copied)
        self.assertIn("Turns: 1", copied)
        self.assertIn("Resistance:", copied)
        self.assertIn("Self-inductance:", copied)
        self.assertIn("Assembly bounding box", copied)
        awg = window.coil_widgets["awg"]
        wire_resistance = window.coil_widgets["resistance_20_ohm_per_km"]
        awg.setCurrentIndex(awg.findData(28))
        self.assertAlmostEqual(
            wire_resistance.value(), awg_resistance_20_ohm_per_km(28), places=3
        )
        wire_resistance.setValue(225.5)
        window.coil_widgets["field_scale_factor"].setValue(2.5)
        window.coil_widgets["drive_current_a"].setValue(100.0)
        window.coil_widgets["turns"].setValue(2)
        resistance_mode = window.coil_widgets["resistance_mode"]
        resistance_mode.setCurrentIndex(resistance_mode.findData("override"))
        window.coil_widgets["resistance_override_ohm"].setValue(12.5)
        inductance_mode = window.coil_widgets["inductance_mode"]
        inductance_mode.setCurrentIndex(inductance_mode.findData("override"))
        window.coil_widgets["inductance_override_h"].setValue(43.0)
        self.assertTrue(window.coil_widgets["resistance_override_ohm"].isEnabled())
        self.assertTrue(window.coil_widgets["inductance_override_h"].isEnabled())
        self.assertIn("12.5 Ω", window.coil_electrical_summary_label.text())
        self.assertIn("43 mH", window.coil_electrical_summary_label.text())
        window.coil_envelope_group.setChecked(True)
        window.coil_widgets["winding_axial_width_mm"].setValue(6.4)
        window.coil_widgets["bobbin_wall_thickness_mm"].setValue(2.0)
        window.coil_widgets["flange_height_mm"].setValue(8.0)
        window.coil_widgets["flange_thickness_mm"].setValue(2.0)
        radial_build = window.coil_widgets["winding_radial_build_mm"]
        self.assertFalse(radial_build.isEnabled())
        automatic_build = radial_build.value()
        self.assertGreater(automatic_build, 0.0)
        self.assertIn(
            "(automatic)", window.coil_estimate_labels["radial_build"].text()
        )
        window.coil_widgets["turns"].setValue(4)
        self.assertAlmostEqual(
            radial_build.value() / automatic_build, 2.0, delta=0.05
        )
        window.coil_widgets["turns"].setValue(2)

        build_mode = window.coil_widgets["winding_radial_build_mode"]
        build_mode.setCurrentIndex(build_mode.findData("manual"))
        self.assertTrue(radial_build.isEnabled())
        radial_build.setValue(6.4)
        self.assertIn("(manual)", window.coil_estimate_labels["radial_build"].text())
        self.assertIn("209.6 × 209.6 × 10.4 mm", window.coil_estimate_labels["assembly_bbox"].text())
        inline_apply_buttons[0].click()
        self.application.processEvents()
        stored = {item["name"]: item["value"] for item in window.adapter.get_params("coil_a")}
        self.assertAlmostEqual(stored["current"], 0.5)
        physical = window.adapter.coil_physical_properties("coil_a")
        self.assertEqual(physical["turns"], 2)
        self.assertAlmostEqual(physical["drive_current_a"], 0.1)
        self.assertAlmostEqual(physical["field_scale_factor"], 2.5)
        self.assertAlmostEqual(
            physical["calculations"]["effective_ampere_turns"], 0.5
        )
        self.assertEqual(physical["awg"], 28)
        self.assertAlmostEqual(physical["resistance_20_ohm_per_km"], 225.5)
        self.assertEqual(physical["resistance_mode"], "override")
        self.assertAlmostEqual(physical["resistance_override_ohm"], 12.5)
        self.assertEqual(physical["inductance_mode"], "override")
        self.assertAlmostEqual(physical["inductance_override_h"], 0.043)
        self.assertAlmostEqual(physical["calculations"]["resistance_ohm"], 12.5)
        self.assertAlmostEqual(physical["calculations"]["inductance_h"], 0.043)
        self.assertTrue(physical["physical_geometry_enabled"])
        self.assertEqual(physical["winding_radial_build_mode"], "manual")
        self.assertEqual(
            physical["calculations"]["assembly"]["bounding_box_mm"],
            [209.6, 209.6, 10.4],
        )

        axis_id = window.adapter.add_template("axis_sensor")
        window.refresh_scene(select_id=axis_id, refresh_plot=False)
        self.assertTrue(window.axis_path_editable)
        window.axis_start_spins[0].setValue(-20.0)
        window.axis_end_spins[0].setValue(20.0)
        window.axis_samples_spin.setValue(5)
        window.apply_inspector()
        path = window.adapter.get_transform(axis_id)
        self.assertEqual(path["path_length"], 5)
        self.assertEqual(path["path"][0], [-0.02, 0.0, -0.15])
        self.assertEqual(path["path"][-1], [0.02, 0.0, 0.15])
        window.plot.cleanup()
        window.deleteLater()

    def test_standalone_new_scene_replaces_the_single_model_document(self):
        window = MainWindow()
        window.plot.capture_view_state = lambda callback: callback(None)
        window.plot.apply_view_state = lambda _state: None
        original_id = window.active_document.document_id
        self.assertFalse(hasattr(window, "document_tabs"))
        self.assertFalse(hasattr(window, "documents"))

        window.new_empty_scene()

        self.assertNotEqual(window.active_document.document_id, original_id)
        self.assertEqual(window.adapter.list_objects(), [])
        self.assertIsNone(window.current_path)
        window.plot.cleanup()
        window.deleteLater()

    def test_standalone_open_replaces_the_single_model_document(self):
        window = MainWindow()
        window.plot.capture_view_state = lambda callback: callback(None)
        window.plot.apply_view_state = lambda _state: None
        original_id = window.active_document.document_id
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.magpy.json"
            StudioAdapter(compact_workflow_scene()).save_file(path)
            self.assertTrue(window._load_scene_file(path))
            self.assertNotEqual(window.active_document.document_id, original_id)
            self.assertEqual(window.current_path, path.resolve())
            opened_id = window.active_document.document_id
            self.assertTrue(window._load_scene_file(path))
            self.assertEqual(window.active_document.document_id, opened_id)
        window.plot.cleanup()
        window.deleteLater()

    def test_explicit_empty_model_is_replaced_not_appended(self):
        window = MainWindow(adapter=StudioAdapter())
        window.plot.capture_view_state = lambda callback: callback(None)
        window.plot.apply_view_state = lambda _state: None
        original_id = window.active_document.document_id
        window.new_empty_scene()
        self.assertNotEqual(window.active_document.document_id, original_id)
        self.assertFalse(hasattr(window, "document_tabs"))
        self.assertFalse(hasattr(window, "documents"))
        window.plot.cleanup()
        window.deleteLater()

    def test_scene_notes_window_updates_scene_and_coalesces_undo(self):
        window = self.make_window()
        original_history = window.adapter.history()["undo"]

        window.open_notes()
        self.assertIsNotNone(window.notes_dialog)
        self.assertTrue(window.notes_dialog.isVisible())
        window.notes_dialog.editor.setPlainText("222 turns\nSpacing trial: 89 mm")
        self.assertEqual(
            window.adapter.get_scene_notes(),
            "222 turns\nSpacing trial: 89 mm",
        )
        self.assertTrue(window.is_dirty())

        window._commit_scene_notes_session()
        self.assertEqual(window.adapter.history()["undo"], original_history + 1)
        window.undo()
        self.assertEqual(window.adapter.get_scene_notes(), "")
        self.assertEqual(window.notes_dialog.notes(), "")

    def test_coil_enable_switch_updates_source_and_scene_tree(self):
        window = self.make_window()
        self.application.processEvents()
        self.assertTrue(window.adapter.coil_enabled("coil_a"))

        window.coil_widgets["enabled"].setChecked(False)
        self.assertTrue(window.apply_inspector())
        self.application.processEvents()
        self.assertFalse(window.adapter.coil_enabled("coil_a"))
        self.assertEqual(
            {item["name"]: item["value"] for item in window.adapter.get_params("coil_a")}["current"],
            0.0,
        )
        self.assertIn("disabled", window._tree_items["coil_a"].text(1).lower())
        self.assertFalse(window.coil_widgets["enabled"].isChecked())

        window.set_selected_coils_enabled(True)
        self.application.processEvents()
        self.assertTrue(window.adapter.coil_enabled("coil_a"))
        self.assertNotEqual(
            {item["name"]: item["value"] for item in window.adapter.get_params("coil_a")}["current"],
            0.0,
        )
        self.assertNotIn("disabled", window._tree_items["coil_a"].text(1).lower())
        self.assertTrue(window.coil_widgets["enabled"].isChecked())
        window.plot.cleanup()
        window.deleteLater()

    def test_builtin_well_plate_menu_and_inspector_use_exact_well_samples(self):
        window = self.make_window()
        plate_id = window.adapter.add_template("well_plate_96")
        window.refresh_scene(select_id=plate_id)
        self.application.processEvents()

        self.assertEqual(window.adapter.object_type_label(plate_id), "96-well plate")
        properties = window.adapter.geometry_properties(plate_id)
        self.assertEqual(properties["role"], "measurement")
        self.assertEqual(properties["well_plate"]["rows"], 8)
        self.assertEqual(properties["well_plate"]["columns"], 12)
        self.assertEqual(properties["well_plate"]["pitch_mm"], 9.0)
        self.assertEqual(window.measurement_quality_combo.currentData(), "user_defined")
        defined_text = window.measurement_defined_points_edit.toPlainText()
        self.assertIn("-31.5", defined_text)
        self.assertIn("49.5", defined_text)

    def test_generic_bust_inspector_lists_and_highlights_surface_regions(self):
        window = self.make_window()
        bust_id = window.adapter.add_template("generic_bust")
        window.refresh_scene(select_id=bust_id, refresh_plot=False)
        self.assertIsNotNone(window.surface_region_tree)
        tree = window.surface_region_tree
        self.assertEqual(tree.topLevelItemCount(), 5)
        rows = {
            tree.topLevelItem(index).text(0): tree.topLevelItem(index).text(1)
            for index in range(tree.topLevelItemCount())
        }
        self.assertEqual(
            rows,
            {
                "Scalp": "9,106",
                "Face": "8,117",
                "Ear L": "1,103",
                "Ear R": "1,167",
                "Neck/Shoulders": "18,939",
            },
        )
        self.assertEqual(set(window.surface_region_editors), {"scalp", "face", "ear_l", "ear_r", "neck_shoulders"})
        scalp_editor = window.surface_region_editors["scalp"]
        self.assertEqual(scalp_editor["item"].checkState(2), Qt.CheckState.Checked)
        self.assertEqual(scalp_editor["item"].checkState(3), Qt.CheckState.Unchecked)
        self.assertEqual(scalp_editor["item"].checkState(4), Qt.CheckState.Checked)
        self.assertEqual(scalp_editor["item"].checkState(5), Qt.CheckState.Checked)
        self.assertFalse(scalp_editor["clearance_override_check"].isChecked())
        self.assertEqual(
            scalp_editor["colour_button"].property("surface_region_colour"), "#3b82f6"
        )

        window.refresh_plot = mock.Mock()
        face_item = next(
            tree.topLevelItem(index)
            for index in range(tree.topLevelItemCount())
            if tree.topLevelItem(index).text(0) == "Face"
        )
        tree.setCurrentItem(face_item)
        face_item.setSelected(True)
        self.application.processEvents()
        self.assertEqual(window.surface_region_highlight, (bust_id, "face"))
        self.assertTrue(window.surface_region_clear_button.isEnabled())
        window.refresh_plot.assert_called_with(rebuild_base=False)

        window.surface_region_clear_button.click()
        self.assertIsNone(window.surface_region_highlight)
        self.assertFalse(window.surface_region_clear_button.isEnabled())

        face_editor = window.surface_region_editors["face"]
        face_editor["item"].setCheckState(3, Qt.CheckState.Checked)
        face_editor["item"].setCheckState(2, Qt.CheckState.Unchecked)
        face_editor["item"].setCheckState(4, Qt.CheckState.Unchecked)
        face_editor["item"].setCheckState(5, Qt.CheckState.Checked)
        face_editor["clearance_override_check"].setChecked(True)
        face_editor["clearance_spin"].setValue(12.5)
        window._style_surface_region_colour_button(face_editor["colour_button"], "#123456")
        self.assertTrue(window.apply_inspector())
        face_settings = window.adapter.surface_region_properties(bust_id)["regions"]["face"]["settings"]
        self.assertEqual(
            face_settings,
            {
                "colour": "#123456",
                "active": False,
                "visible": True,
                "placement": False,
                "drc_exclusion": True,
                "clearance_override_mm": 12.5,
            },
        )
        window.plot.cleanup()
        window.deleteLater()

    def test_inspector_disclosure_state_survives_object_selection_changes(self):
        window = self.make_window()
        window.refresh_scene(select_id="coil_a", refresh_plot=False)

        def section_named(title: str) -> CollapsibleSection:
            return next(
                section
                for section in window.inspector_container.findChildren(CollapsibleSection)
                if section.header.text() == title
            )

        magnetics = section_named("Magnetics")
        construction = section_named("Coil Construction")
        self.assertTrue(magnetics.header.isChecked())
        self.assertFalse(construction.header.isChecked())

        # Arrange the inspector away from its defaults.  Selection changes rebuild
        # these widgets, so the new instances must inherit the user's session state.
        magnetics.header.setChecked(False)
        construction.header.setChecked(True)
        self.application.processEvents()

        window.refresh_scene(select_id="axis_sensor", refresh_plot=False)
        window.refresh_scene(select_id="coil_b", refresh_plot=False)
        self.application.processEvents()

        self.assertFalse(section_named("Magnetics").header.isChecked())
        self.assertTrue(section_named("Coil Construction").header.isChecked())
        self.assertTrue(section_named("Magnetics").content.isHidden())
        self.assertFalse(section_named("Coil Construction").content.isHidden())
        window.plot.cleanup()
        window.deleteLater()

    def test_geometry_colour_uses_visual_picker_and_persists(self):
        window = self.make_window()
        object_id = window.adapter.add_template("guide_box")
        window.refresh_scene(select_id=object_id, refresh_plot=False)

        button = window.geometry_colour_button
        self.assertIsInstance(button, QPushButton)
        self.assertEqual(button.objectName(), "geometryColourButton")
        original = window.adapter.geometry_properties(object_id)["colour"]
        self.assertEqual(button.property("geometry_colour"), original)
        self.assertEqual(button.text(), original.upper())
        geometry_box = next(
            section
            for section in window.inspector_container.findChildren(CollapsibleSection)
            if section.header.text() == "Geometry and Appearance"
        )
        geometry_inline_apply = [
            child
            for child in geometry_box.findChildren(QPushButton)
            if child.objectName() == "inlineApplyButton"
        ]
        self.assertEqual(len(geometry_inline_apply), 1)

        with mock.patch.object(
            main_window_module.QColorDialog, "getColor", return_value=QColor("#123456")
        ) as picker:
            button.click()
        picker.assert_called_once()
        self.assertEqual(button.property("geometry_colour"), "#123456")
        self.assertEqual(button.text(), "#123456")

        self.assertTrue(window.apply_inspector())
        self.assertEqual(
            window.adapter.geometry_properties(object_id)["colour"], "#123456"
        )
        window.plot.cleanup()
        window.deleteLater()

    def test_imported_mesh_inspector_shows_mm_dimensions_and_xyz_scales(self):
        window = self.make_window()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh dimensions.obj"
            path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 100 0 0",
                        "v 0 80 0",
                        "v 0 0 60",
                        "f 1 3 2",
                        "f 1 2 4",
                        "f 2 3 4",
                        "f 3 1 4",
                    ]
                ),
                encoding="utf-8",
            )
            mesh_id = window.adapter.import_mesh(path, unit_scale=0.001)
            window.refresh_scene(select_id=mesh_id, refresh_plot=False)
            self.assertEqual(
                set(window.parameter_editors),
                {"scale_x", "scale_y", "scale_z"},
            )
            self.assertEqual(
                window.mesh_dimension_labels["original"].text(),
                "100 × 80 × 60 mm",
            )
            window.parameter_editors["scale_x"]["widget"].setValue(2.0)
            window.parameter_editors["scale_y"]["widget"].setValue(0.5)
            window.parameter_editors["scale_z"]["widget"].setValue(1.5)
            self.assertEqual(
                window.mesh_dimension_labels["current"].text(),
                "200 × 40 × 90 mm",
            )
            self.assertTrue(window.apply_inspector())
            self.assertEqual(
                window.adapter.geometry_properties(mesh_id)["dimensions_mm"],
                [200.0, 40.0, 90.0],
            )
            inspector_labels = {label.text() for label in window.findChildren(QLabel)}
            self.assertIn("Watertight manifold", inspector_labels)
            self.assertIn("Inside/outside DRC enabled", inspector_labels)
        window.plot.cleanup()
        window.deleteLater()

    def test_inspector_wheel_scrolls_without_changing_editor_values(self):
        window = self.make_window()
        scroll_bar = window.inspector_scroll.verticalScrollBar()
        scroll_bar.setRange(0, 1000)
        scroll_bar.setValue(500)
        turns = window.coil_widgets["turns"]
        turns_before = turns.value()

        event = QWheelEvent(
            QPointF(5.0, 5.0),
            QPointF(5.0, 5.0),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(turns, event)

        self.assertEqual(turns.value(), turns_before)
        self.assertLess(scroll_bar.value(), 500)
        window.plot.cleanup()
        window.deleteLater()

    def test_new_scene_is_blank_and_plus_creates_an_empty_group(self):
        window = self.make_window()
        window.plot.capture_view_state = lambda callback: callback(None)
        window.plot.apply_view_state = lambda _state: None
        window.new_empty_scene()
        self.assertEqual(window.tree.topLevelItemCount(), 0)
        self.assertEqual(window.adapter.list_objects(), [])

        window.new_group_button.click()
        self.assertEqual(window.tree.topLevelItemCount(), 1)
        group_id = window.selected_id
        self.assertIsNotNone(group_id)
        self.assertEqual(window.adapter.get_object(group_id)["type"], "Collection")
        self.assertTrue(window.undo_action.isEnabled())
        window.plot.cleanup()
        window.deleteLater()

    def test_tree_and_viewport_multi_selection_can_group_existing_objects(self):
        window = self.make_window()
        window._plot_selection_requested("coil_a", False)
        window._plot_selection_requested("coil_b", True)
        self.assertEqual(set(window.selected_ids), {"coil_a", "coil_b"})
        self.assertTrue(window.group_selected_action.isEnabled())
        # Additive viewport selection also toggles an already selected object.
        window._plot_selection_requested("coil_b", True)
        self.assertEqual(window.selected_ids, ["coil_a"])
        window._plot_selection_requested("coil_b", True)
        self.assertEqual(set(window.selected_ids), {"coil_a", "coil_b"})
        window.group_selected()
        group_id = window.selected_id
        self.assertEqual(window.adapter.get_object(group_id)["type"], "Collection")
        self.assertEqual(window.adapter.get_object("coil_a")["parent"], group_id)
        self.assertEqual(window.adapter.get_object("coil_b")["parent"], group_id)
        window.undo()
        self.assertIsNone(window.adapter.get_object("coil_a").get("parent"))
        # An empty-space viewport click clears both the logical and tree selection.
        window._plot_selection_requested("", False)
        self.assertEqual(window.selected_ids, [])
        self.assertIsNone(window.selected_id)
        self.assertEqual(window.tree.selectedItems(), [])
        window.plot.cleanup()
        window.deleteLater()

    def test_scene_tree_drop_requests_never_mutate_qt_rows_directly(self):
        tree = SceneTreeWidget()
        source = QTreeWidgetItem(["Coil", "Coil"])
        source.setData(0, USER_ROLE_ID, "coil")
        group = QTreeWidgetItem(["Group", "Group"])
        group.setData(0, USER_ROLE_ID, "group")
        ordinary = QTreeWidgetItem(["Sensor", "Sensor"])
        ordinary.setData(0, USER_ROLE_ID, "sensor")
        tree.addTopLevelItems([source, group, ordinary])

        valid, parent_id = tree._drop_destination(
            source,
            group,
            QAbstractItemView.DropIndicatorPosition.OnItem,
        )
        self.assertTrue(valid)
        self.assertEqual(parent_id, "group")

        valid, parent_id = tree._drop_destination(
            source,
            ordinary,
            QAbstractItemView.DropIndicatorPosition.OnItem,
        )
        self.assertFalse(valid)
        self.assertIsNone(parent_id)

        valid, parent_id = tree._drop_destination(
            source,
            None,
            QAbstractItemView.DropIndicatorPosition.OnViewport,
        )
        self.assertTrue(valid)
        self.assertIsNone(parent_id)

        drop_source = _source_for(SceneTreeWidget.dropEvent)
        self.assertIn("event.ignore()", drop_source)
        self.assertNotIn("acceptProposedAction", drop_source)
        tree.deleteLater()

    def test_scene_tree_reconciles_from_adapter_after_a_drag(self):
        window = self.make_window()
        window.tree.takeTopLevelItem(0)
        self.assertEqual(window.tree.topLevelItemCount(), 2)
        window.tree.drag_finished.emit()
        self.assertEqual(window.tree.topLevelItemCount(), 3)
        self.assertEqual(
            {window.tree.topLevelItem(index).data(0, USER_ROLE_ID) for index in range(3)},
            {"coil_a", "coil_b", "axis_sensor"},
        )
        window.plot.cleanup()
        window.deleteLater()

    def test_waveform_dialog_quick_source_is_current_only_and_exposure_free(self):
        adapter = StudioAdapter(compact_workflow_scene())
        dialog = WaveformPlaybackDialog(
            adapter,
            map_kind="2d",
            map_settings={
                "plane": "xy", "offset_mm": 0.0, "span_mm": 50.0, "resolution": 4,
                "field": "B", "component": "magnitude", "logarithmic": False,
                "show_object_outlines": False, "outline_object_ids": [],
                "drive_coil_ids": ["coil_a", "coil_b"],
                "colour_minimum": None, "colour_maximum": None,
            },
            playback_fps=48.0,
            playback_speed=2.0,
        )
        self.assertTrue(dialog.waveform_sine.isChecked())
        self.assertEqual(dialog.slew_rate.value(), 100.0)
        self.assertAlmostEqual(dialog._slew_rate_per_second(), 100.0)
        for retired in (
            "drive_kind", "time_unit", "value_unit", "initial_drive",
            "use_model_resistance", "manual_resistance", "inductance",
            "advanced_sequencer", "sequence_panel", "sequence_table", "sequence_loop",
        ):
            self.assertFalse(hasattr(dialog, retired), retired)
        self.assertFalse(hasattr(dialog, "wave_points_section"))
        self.assertFalse(hasattr(dialog, "waveform_custom"))
        self.assertFalse(dialog.generated_waveform_controls.isHidden())
        self.assertTrue(dialog.drive_state_trace.isEnabled())
        self.assertEqual(dialog.base_excitation_trace.text(), "Drive trace")
        self.assertEqual(dialog.drive_state_trace.text(), "Sequencer trace")
        self.assertEqual(dialog.sensor_trace.text(), "Sensor trace")
        self.assertFalse(dialog.sensor_trace.isChecked())
        self.assertTrue(dialog.sensor_trace_table.isHidden())
        self.assertGreaterEqual(dialog.sensor_trace_table.rowCount(), 1)
        sensor_row = next(
            row
            for row in range(dialog.sensor_trace_table.rowCount())
            if (dialog._sensor_trace_choice(row) or {}).get("kind") == "sensor"
        )
        dialog.sensor_trace.setChecked(True)
        self.assertFalse(dialog.sensor_trace_table.isHidden())
        dialog.sensor_trace_table.item(sensor_row, 0).setCheckState(Qt.CheckState.Checked)
        signal_combo = dialog.sensor_trace_table.cellWidget(sensor_row, 3)
        self.assertEqual(signal_combo.itemText(0), "|B|")
        bx_index = signal_combo.findData("x")
        self.assertGreaterEqual(bx_index, 0)
        signal_combo.setCurrentIndex(bx_index)
        sample_spin = dialog.sensor_trace_table.cellWidget(sensor_row, 2)
        if isinstance(sample_spin, QSpinBox) and sample_spin.maximum() > 1:
            sample_spin.setValue(2)
        trace_settings = dialog._observation_settings()
        self.assertEqual(len(trace_settings), 1)
        self.assertEqual(trace_settings[0]["component"], "x")
        if isinstance(sample_spin, QSpinBox):
            self.assertEqual(trace_settings[0]["sample_index"], sample_spin.value() - 1)
        self.assertEqual(dialog.export_video_button.text(), "Export video…")
        self.assertTrue(dialog.export_video_button.isHidden())
        self.assertEqual(dialog._base_selected_drive_ids(), ["coil_a", "coil_b"])
        self.assertEqual(dialog.playback_fps_value, 48.0)
        self.assertEqual(dialog.playback_speed_value, 2.0)
        self.assertFalse(hasattr(dialog, "render_options_section"))
        frames, samples = dialog._resolved_sample_counts(1.0)
        self.assertEqual(frames, 25)
        self.assertEqual(samples, 145)
        dialog.close()

    def test_map_animation_coil_selector_lists_disabled_scene_coils_but_defaults_them_off(self):
        scene = compact_workflow_scene()
        scene["field_workbench"]["coil_specs"]["coil_b"]["enabled"] = False
        adapter = StudioAdapter(scene)
        map_dialog = FieldVolumeMapDialog(adapter)
        self.assertEqual(map_dialog._playback_known_coil_ids, {"coil_a", "coil_b"})
        self.assertEqual(map_dialog.selected_playback_coil_ids(), ["coil_a"])
        map_dialog._set_playback_coil_selected("coil_b", True)
        self.assertEqual(map_dialog.selected_playback_coil_ids(), ["coil_a", "coil_b"])

        dialog = WaveformPlaybackDialog(
            adapter,
            map_kind="3d",
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0], "span_mm": 50.0,
                "slice_axis": "z", "slice_count": 3, "resolution": 4,
                "field": "B", "component": "magnitude", "logarithmic": False,
                "view_type": "slices", "show_scene": False, "scene_object_ids": [],
                "drive_coil_ids": ["coil_a"],
                "colour_minimum": None, "colour_maximum": None,
            },
        )
        self.assertEqual(dialog._base_selected_drive_ids(), ["coil_a"])
        self.assertTrue(dialog.export_video_button.isHidden())
        self.assertFalse(hasattr(dialog, "advanced_sequencer"))
        self.assertFalse(hasattr(dialog, "sequence_table"))
        dialog.close()
        map_dialog.close()

    def test_3d_waveform_playback_can_defer_payload_build_to_render_tab(self):
        adapter = StudioAdapter(compact_workflow_scene())
        dialog = WaveformPlaybackDialog(
            adapter,
            map_kind="3d",
            map_settings={
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 50.0,
                "slice_axis": "z",
                "slice_count": 3,
                "resolution": 4,
                "field": "B",
                "component": "magnitude",
                "logarithmic": False,
                "view_type": "slices",
                "waveform_render_mode": "slices",
                "include_scene": False,
                "scene_object_ids": [],
                "drive_coil_ids": ["coil_a"],
                "colour_minimum": None,
                "colour_maximum": None,
            },
            defer_payload_build=True,
        )
        with mock.patch(
            "fieldworkbench.waveform_dialog.build_prepared_gpu_payload"
        ) as build_payload:
            dialog._create_playback()

        build_payload.assert_not_called()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertIsNone(dialog.result_gpu_payload)
        self.assertIsInstance(dialog.result_gpu_request, dict)
        self.assertEqual(dialog.result_gpu_request["map_kind"], "3d")
        self.assertEqual(dialog.result_gpu_request["coil_ids"], ["coil_a"])
        self.assertEqual(
            dialog.result_gpu_request["waveform_metadata"]["mode"], "sine"
        )
        self.assertAlmostEqual(
            dialog.result_gpu_request["waveform_metadata"]["cycle_count"],
            dialog.waveform_frequency.value() * dialog.waveform_duration.value(),
        )
        dialog.deleteLater()

    def test_field_map_defaults_to_xy_and_waits_for_calculate(self):
        window = self.make_window()
        guide = window.adapter.add_template("guide_box")
        dialog = FieldMapDialog(window.adapter, window)
        self.assert_standard_window_controls(dialog)
        self.assertEqual(dialog.fullscreen_button.text(), "Fullscreen viewer")
        self.assertEqual(dialog.export_video_button.text(), "Export video")
        self.assertTrue(dialog.export_video_button.isHidden())
        self.assertIs(
            dialog.export_video_button.parentWidget(), dialog.fullscreen_button.parentWidget()
        )
        self.assertEqual(dialog.exit_button.text(), "Exit")
        self.assertNotEqual(
            dialog.exit_button.parentWidget(), dialog.fullscreen_button.parentWidget()
        )
        self.assertEqual(dialog.plane.currentText(), "xy")
        self.assertTrue(dialog.plot._include_3d_return_axes)
        self.assertTrue(dialog.plot._synchronize_legend_visibility)
        self.assertFalse(dialog.colour_minimum_enabled.isChecked())
        self.assertFalse(dialog.colour_maximum_enabled.isChecked())
        self.assertFalse(dialog.show_contours.isChecked())
        self.assertTrue(dialog.show_grid.isChecked())
        self.assertFalse(dialog.contour_count.isEnabled())
        self.assertFalse(dialog.contour_labels.isEnabled())
        self.assertFalse(dialog.colour_minimum.isEnabled())
        self.assertFalse(dialog.colour_maximum.isEnabled())
        self.assertEqual(dialog.colour_minimum.suffix(), " µT")
        self.assertTrue(dialog.object_outlines.isChecked())
        self.assertEqual(dialog.selected_object_outline_ids(), [guide])
        self.assertEqual(dialog.object_outline_objects_button.text(), "Objects 1/1…")
        self.assertEqual(dialog.calculation_status.objectName(), "fieldMapStatusBar")
        self.assertEqual(dialog.tabs.count(), 1)
        self.assertEqual(dialog.tabs.tabText(0), "Selector")
        self.assertIs(dialog.tabs.widget(0), dialog.plot)
        self.assertIsNone(
            dialog.tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.LeftSide)
        )
        self.assertIsNone(
            dialog.tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.RightSide)
        )
        self.assertFalse(hasattr(dialog, "reset_button"))
        self.assertEqual(dialog.calculate_button.text(), "Static map")
        self.assertEqual(dialog.waveform_button.text(), "Animated map")
        self.assertFalse(dialog.map_plane_section.header.isChecked())
        self.assertFalse(dialog.sample_config_section.header.isChecked())
        self.assertTrue(dialog.field_view_options_section.header.isChecked())
        self.assertFalse(dialog.scene_options_section.header.isChecked())
        self.assertTrue(dialog.display_options_section.header.isChecked())
        self.assertIn("position the plane", dialog.calculation_status.currentMessage())

        with mock.patch.object(dialog, "_start_render_job") as start_render:
            dialog.calculate_button.click()
            self.assertTrue(dialog.calculate_button.isEnabled())
            start_render.assert_called_once()
            call = start_render.call_args.kwargs
            self.assertEqual(call["render_kind"], "2d_static")
            self.assertEqual(call["job_label"], "Map 1")
            settings = call["request"]["map_settings"]
            self.assertEqual(settings["plane"], "xy")
            self.assertEqual(settings["component"], "magnitude")
            self.assertEqual(
                call["request"]["active_coil_ids"],
                dialog.selected_active_coil_ids(),
            )
            self.assertIsNone(settings["colour_minimum"])
            self.assertIsNone(settings["colour_maximum"])
            self.assertFalse(settings["show_contours"])
            self.assertTrue(settings["show_grid"])
            self.assertEqual(settings["contour_count"], 10)
            self.assertFalse(settings["contour_labels"])
            self.assertTrue(settings["show_object_outlines"])
            self.assertEqual(settings["outline_object_ids"], [guide])

            dialog.span.setValue(240.0)
            dialog.calculate_button.click()
            self.assertEqual(start_render.call_count, 2)
            second = start_render.call_args.kwargs
            self.assertEqual(second["job_label"], "Map 2")
            self.assertEqual(second["request"]["map_settings"]["span_mm"], 240.0)

            dialog.show_grid.setChecked(False)
            dialog.show_contours.setChecked(True)
            dialog.contour_count.setValue(23)
            dialog.contour_labels.setChecked(True)
            waveform_settings = dialog._waveform_map_settings()
            self.assertFalse(waveform_settings["show_grid"])
            self.assertTrue(waveform_settings["show_contours"])
            self.assertEqual(waveform_settings["contour_count"], 23)
            self.assertTrue(waveform_settings["contour_labels"])

            dialog._set_object_outline_selected(guide, False)
            dialog.calculate_button.click()
            self.assertEqual(start_render.call_count, 3)
            third = start_render.call_args.kwargs
            self.assertEqual(third["request"]["map_settings"]["outline_object_ids"], [])
            self.assertEqual(dialog.object_outline_objects_button.text(), "Objects 0/1…")

        dialog.object_outlines.setChecked(False)
        self.assertFalse(dialog.object_outline_objects_button.isEnabled())
        dialog.plot.cleanup()
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_2d_render_jobs_open_immediate_tabs_and_use_task_lists(self):
        window = self.make_window()
        dialog = FieldMapDialog(window.adapter, window)
        dialog.display_mode.setCurrentIndex(0)  # Heatmap only keeps this smoke test light.
        dialog.resolution.setValue(10)

        first_settings = dialog._static_render_map_settings()
        second_settings = dict(first_settings, offset_mm=12.0)
        dialog._start_render_job(
            render_kind="2d_static",
            request={
                "map_settings": first_settings,
                "active_coil_ids": dialog.selected_active_coil_ids(),
            },
            job_label="Map 1",
            tooltip="First 2D background map",
        )
        dialog._start_render_job(
            render_kind="2d_static",
            request={
                "map_settings": second_settings,
                "active_coil_ids": dialog.selected_active_coil_ids(),
            },
            job_label="Map 2",
            tooltip="Second 2D background map",
        )

        self.assertEqual(dialog.tabs.count(), 3)
        self.assertIsInstance(dialog.tabs.widget(1), _FieldRenderJobPage)
        self.assertIsInstance(dialog.tabs.widget(2), _FieldRenderJobPage)
        self.assertEqual(dialog._active_render_job_count(), 2)
        first_page = dialog.tabs.widget(1)
        self.assertIn("Plotly", first_page._task_by_id["assemble"]["label"])
        self.assertIn("Plotly", first_page._task_by_id["viewer"]["label"])

        deadline = time.monotonic() + 30.0
        while dialog._render_jobs and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.application.processEvents()

        self.assertFalse(dialog._render_jobs)
        self.assertIsInstance(dialog.tabs.widget(1), PlotView)
        self.assertIsInstance(dialog.tabs.widget(2), PlotView)
        dialog._cleanup_plots()
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_map_scene_checklists_snapshot_model_then_remain_local_until_reopen(self):
        window = self.make_window()
        visible_guide = window.adapter.add_template("guide_box")
        hidden_guide = window.adapter.add_template("guide_box")
        window.adapter.set_visible(hidden_guide, False)
        window.adapter.set_coils_enabled(["coil_b"], False)

        map_2d = FieldMapDialog(window.adapter, window)
        self.assertEqual(map_2d.selected_active_coil_ids(), ["coil_a"])
        initial_2d_objects = set(map_2d.selected_object_outline_ids())
        self.assertIn(visible_guide, initial_2d_objects)
        self.assertNotIn(hidden_guide, initial_2d_objects)
        state_2d = map_2d._persistent_ui_state()
        self.assertIn("camera_timeline", state_2d)
        self.assertNotIn("active_coil_ids", state_2d)
        self.assertNotIn("outline_object_ids", state_2d)

        map_2d._set_playback_coil_selected("coil_a", False)
        map_2d._set_playback_coil_selected("coil_b", True)
        map_2d._set_object_outline_selected(visible_guide, False)
        map_2d._set_object_outline_selected(hidden_guide, True)

        # Scene changes while the viewer is open must not rewrite its local choices.
        window.adapter.set_coils_enabled(["coil_a"], False)
        window.adapter.set_coils_enabled(["coil_b"], True)
        window.adapter.set_visible(visible_guide, False)
        window.adapter.set_visible(hidden_guide, True)
        self.assertEqual(map_2d.selected_active_coil_ids(), ["coil_b"])
        changed_2d_objects = set(map_2d.selected_object_outline_ids())
        self.assertNotIn(visible_guide, changed_2d_objects)
        self.assertIn(hidden_guide, changed_2d_objects)

        map_2d.plot.cleanup()
        map_2d.deleteLater()
        reopened_2d = FieldMapDialog(window.adapter, window)
        self.assertEqual(reopened_2d.selected_active_coil_ids(), ["coil_b"])
        reopened_2d_objects = set(reopened_2d.selected_object_outline_ids())
        self.assertNotIn(visible_guide, reopened_2d_objects)
        self.assertIn(hidden_guide, reopened_2d_objects)

        map_3d = FieldVolumeMapDialog(window.adapter, window)
        self.assertEqual(map_3d.selected_active_coil_ids(), ["coil_b"])
        initial_3d_objects = set(map_3d.selected_scene_object_ids())
        self.assertNotIn(visible_guide, initial_3d_objects)
        self.assertIn(hidden_guide, initial_3d_objects)
        state_3d = map_3d._persistent_ui_state()
        self.assertIn("camera_timeline", state_3d)
        self.assertNotIn("active_coil_ids", state_3d)
        self.assertNotIn("scene_object_ids", state_3d)
        initial_opacities = map_3d.scene_object_opacity_values()
        self.assertAlmostEqual(initial_opacities[visible_guide], 0.28)
        self.assertAlmostEqual(initial_opacities[hidden_guide], 0.28)

        # An untouched viewer keeps its launch-time model opacity even when the
        # source scene changes.
        window.adapter._mesh(visible_guide)["opacity"] = 0.42
        inherited_opacities = map_3d.scene_object_opacity_values()
        self.assertAlmostEqual(inherited_opacities[visible_guide], 0.28)

        map_3d._set_playback_coil_selected("coil_a", True)
        map_3d._set_playback_coil_selected("coil_b", False)
        map_3d._set_scene_object_selected(visible_guide, True)
        map_3d._set_scene_object_selected(hidden_guide, False)
        map_3d._set_scene_object_opacity(visible_guide, 0.35)
        map_3d._set_scene_object_opacity(hidden_guide, 0.60)
        window.adapter._mesh(visible_guide)["opacity"] = 0.66

        window.adapter.set_coils_enabled(["coil_a"], True)
        window.adapter.set_coils_enabled(["coil_b"], False)
        window.adapter.set_visible(visible_guide, True)
        window.adapter.set_visible(hidden_guide, False)
        self.assertEqual(map_3d.selected_active_coil_ids(), ["coil_a"])
        changed_3d_objects = set(map_3d.selected_scene_object_ids())
        self.assertIn(visible_guide, changed_3d_objects)
        self.assertNotIn(hidden_guide, changed_3d_objects)
        changed_opacities = map_3d.scene_object_opacity_values()
        self.assertAlmostEqual(changed_opacities[visible_guide], 0.35)
        self.assertAlmostEqual(changed_opacities[hidden_guide], 0.60)

        map_3d.plot.cleanup()
        map_3d.deleteLater()
        reopened_3d = FieldVolumeMapDialog(window.adapter, window)
        self.assertEqual(reopened_3d.selected_active_coil_ids(), ["coil_a"])
        reopened_3d_objects = set(reopened_3d.selected_scene_object_ids())
        self.assertIn(visible_guide, reopened_3d_objects)
        self.assertNotIn(hidden_guide, reopened_3d_objects)
        reopened_opacities = reopened_3d.scene_object_opacity_values()
        self.assertAlmostEqual(reopened_opacities[visible_guide], 0.66)
        self.assertAlmostEqual(reopened_opacities[hidden_guide], 0.28)

        reopened_2d.plot.cleanup()
        reopened_2d.deleteLater()
        reopened_3d.plot.cleanup()
        reopened_3d.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_unified_maps_workspace_shells(self):
        window = self.make_window()
        maps = UnifiedFieldWorkspaceDialog(
            mode="static",
            scene_name="Demo",
            adapter=window.adapter,
            brain_object_id=None,
            parent=window,
        )
        self.assertEqual(maps.windowTitle(), "Maps / Analysis — Demo")
        self.assertEqual((maps.minimumWidth(), maps.minimumHeight()), (820, 560))
        self.assertEqual(maps.sidebar.minimumWidth(), 240)
        self.assertEqual(maps.measurement_panel_host.minimumWidth(), 220)
        self.assertEqual(maps.current_view_key, "2d")
        self.assertEqual(maps.current_page_key, "static:2d")
        self.assertTrue(maps.mode_buttons["static"].isChecked())
        self.assertFalse(maps.mode_buttons["playback"].isChecked())
        self.assertTrue(maps.view_buttons["2d"].isChecked())
        self.assertFalse(maps.view_buttons["brain"].isEnabled())
        self.assertIn("static:2d", maps._pages)
        static_2d = maps._pages["static:2d"]
        self.assertEqual(static_2d.control_scroll.minimumWidth(), 0)
        self.assertFalse(static_2d.calculate_button.isHidden())
        self.assertTrue(static_2d.waveform_button.isHidden())
        self.assertTrue(static_2d.camera_section.isHidden())
        self.assertIs(maps.control_stack.currentWidget(), static_2d.control_column)
        self.assertIs(static_2d.tabs, maps.shared_tabs)
        self.assertEqual(maps.shared_tabs.count(), 1)
        self.assertEqual(maps.shared_tabs.tabText(0), "Selector")
        self.assertIs(maps.shared_tabs.widget(0), maps.selector_stack)
        self.assertIs(
            maps.selector_stack.currentWidget(),
            maps._selector_pages["static:2d"],
        )
        self.assertIs(maps.measurement_view_splitter.widget(0), maps.shared_tabs)
        self.assertIs(maps.measurement_view_splitter.widget(1), maps.measurement_panel_host)
        self.assertIs(maps.measurement_view_splitter.parentWidget(), maps.measurement_shell)
        self.assertIs(maps.workspace_tab_strip.parentWidget(), maps.measurement_shell)
        self.assertIs(
            maps.shared_tabs.workspace_tab_bar.parentWidget(),
            maps.workspace_tab_strip,
        )
        self.assertEqual(maps.shared_tabs.workspace_tab_bar.tabText(0), "Selector")
        self.assertEqual(maps.load_saved_map_button.text(), "Load saved map…")
        self.assertEqual(maps.workspace_save_map_button.text(), "Save map…")
        self.assertEqual(maps.workspace_fullscreen_button.text(), "Fullscreen viewer")
        self.assertEqual(maps.workspace_regions_button.text(), "Show measurements")
        self.assertFalse(maps._measurement_panel_visible)
        self.assertTrue(static_2d.fullscreen_button.isHidden())
        self.assertTrue(static_2d.regions_button.isHidden())
        self.assertIsInstance(maps.static_measurement_panel, ExposureRegionPanel)
        self.assertFalse(hasattr(maps, "static_snapshot_host"))
        self.assertTrue(maps.measurement_analyze_button.isEnabled() or not maps._measurement_volume_rows())
        self.assertEqual(static_2d.regions_button.text(), "Show measurements")

        # Visibility has per-view defaults/preferences: ordinary 2D/3D begin
        # closed, while Brain begins open. Width remains a shared Maps-window
        # preference and a user's choice for one view survives mode switches.
        maps._measurement_panel_width = 333
        maps.workspace_regions_button.click()
        self.assertTrue(maps._measurement_panel_visible)
        self.assertTrue(maps._measurement_panel_visibility_by_view["2d"])
        self.assertEqual(maps.workspace_regions_button.text(), "Hide measurements")

        # Results retain both the Static/Scripted owner and visual owner so one
        # shared tab strip can restore the correct selector pair.
        retained_static = QWidget()
        retained_static.setProperty("fieldWorkspaceOwnerMode", "static")
        retained_static.setProperty("fieldWorkspaceOwnerView", "2d")
        static_result_index = maps.shared_tabs.addTab(retained_static, "2D\\Static - 1")
        maps.shared_tabs.setCurrentIndex(static_result_index)

        maps.view_buttons["3d"].click()
        self.assertEqual(maps.current_page_key, "static:3d")
        self.assertFalse(maps._measurement_panel_visible)
        self.assertFalse(maps._measurement_panel_visibility_by_view["3d"])
        self.assertEqual(maps.workspace_regions_button.text(), "Show measurements")
        self.assertEqual(maps._measurement_panel_width, 333)
        self.assertEqual(maps.shared_tabs.currentIndex(), 0)
        static_3d = maps._pages["static:3d"]
        self.assertIs(maps.control_stack.currentWidget(), static_3d.control_column)
        self.assertIs(
            maps.selector_stack.currentWidget(),
            maps._selector_pages["static:3d"],
        )

        # Mode switching stays inside Maps, preserves the selected visual mode,
        # and exposes the compact Scripted-source selector.
        maps.mode_buttons["playback"].click()
        self.assertEqual(maps.mode, "playback")
        # Scripted results have their own open-by-default live-analysis
        # preference; this does not expose the Static snapshot panel.
        self.assertTrue(maps._measurement_panel_visible)
        self.assertTrue(maps._measurement_panel_visibility_by_view["playback:3d"])
        self.assertFalse(maps._measurement_panel_visibility_by_view["3d"])
        self.assertTrue(maps.workspace_regions_button.isHidden())
        self.assertTrue(maps.measurement_panel_host.isHidden())
        self.assertEqual(maps._measurement_panel_width, 333)
        self.assertEqual(maps.current_page_key, "playback:3d")
        self.assertTrue(maps.mode_buttons["playback"].isChecked())
        playback_3d = maps._pages["playback:3d"]
        self.assertTrue(playback_3d.calculate_button.isHidden())
        self.assertFalse(playback_3d.waveform_button.isHidden())
        self.assertEqual(playback_3d.waveform_button.text(), "Create playback")
        self.assertIsNotNone(maps.playback_controller)
        self.assertIsNotNone(maps.scripted_source_host)
        self.assertIsNotNone(maps.scripted_source_type)
        self.assertIsNotNone(maps.scripted_waveform)
        self.assertIs(
            maps.scripted_source_host.parentWidget(),
            playback_3d.control_panel,
        )
        self.assertEqual(maps.scripted_source_type.currentData(), "builtin")
        self.assertEqual(maps.scripted_waveform.currentData(), "sine")

        # Per-view labels advance across Static and Scripted instead of creating
        # duplicate names when the mode/source changes.
        self.assertEqual(maps._workspace_result_label("static", "2d", 1, "Map 1"), "2D\\Static - 1")
        self.assertEqual(maps._workspace_result_label("playback", "2d", 1, "Waveform 1"), "2D\\Sine - 1")
        maps.scripted_waveform.setCurrentIndex(maps.scripted_waveform.findData("square"))
        self.assertEqual(maps._workspace_result_label("playback", "2d", 2, "Waveform 2"), "2D\\Square - 1")
        maps.scripted_waveform.setCurrentIndex(maps.scripted_waveform.findData("sine"))
        self.assertEqual(maps._workspace_result_label("playback", "2d", 3, "Waveform 3"), "2D\\Sine - 2")

        retained_scripted = QWidget()
        retained_scripted.setProperty("fieldWorkspaceOwnerMode", "playback")
        retained_scripted.setProperty("fieldWorkspaceOwnerView", "3d")
        scripted_result_index = maps.shared_tabs.addTab(retained_scripted, "3D\\Sine - 1")
        maps.shared_tabs.setCurrentIndex(scripted_result_index)
        # Ordinary Scripted maps still have no region panel or toggle.
        self.assertTrue(maps.workspace_regions_button.isHidden())
        self.assertTrue(maps.measurement_panel_host.isHidden())

        # A live-analysis result owns an embedded panel, not the outer Static
        # snapshot host. Its toggle must not change either Static preference.
        measurement_viewer = QWidget()
        measurement_viewer.cleanup = mock.Mock()
        measurement_result = ExposureResultTab(measurement_viewer, QWidget())
        measurement_result.setProperty("fieldWorkspaceOwnerMode", "playback")
        measurement_result.setProperty("fieldWorkspaceOwnerView", "3d")
        measurement_index = maps.shared_tabs.addTab(measurement_result, "3D live analysis")
        maps.shared_tabs.setCurrentIndex(measurement_index)
        self.assertFalse(maps.workspace_regions_button.isHidden())
        self.assertEqual(maps.workspace_regions_button.text(), "Hide measurements")
        self.assertTrue(maps.measurement_panel_host.isHidden())
        self.assertGreater(measurement_result.splitter.sizes()[1], 0)
        maps.workspace_regions_button.click()
        self.assertFalse(maps._measurement_panel_visibility_by_view["playback:3d"])
        self.assertEqual(measurement_result.splitter.sizes()[1], 0)
        self.assertFalse(maps._measurement_panel_visibility_by_view["3d"])
        self.assertTrue(maps._measurement_panel_visibility_by_view["2d"])
        maps.mode_buttons["static"].click()
        self.assertFalse(maps._measurement_panel_visible)
        self.assertFalse(maps.measurement_panel_host.isHidden())
        maps.shared_tabs.setCurrentIndex(measurement_index)
        self.assertEqual(maps.mode, "playback")
        self.assertFalse(maps._measurement_panel_visible)
        self.assertEqual(maps.workspace_regions_button.text(), "Show measurements")
        self.assertEqual(measurement_result.splitter.sizes()[1], 0)
        maps.workspace_regions_button.click()
        self.assertTrue(maps._measurement_panel_visibility_by_view["playback:3d"])
        self.assertGreater(measurement_result.splitter.sizes()[1], 0)
        maps.shared_tabs.setCurrentIndex(static_result_index)
        self.assertTrue(maps._measurement_panel_visible)
        self.assertEqual(maps.workspace_regions_button.text(), "Hide measurements")
        self.assertEqual(maps.mode, "static")
        self.assertEqual(maps.current_view_key, "2d")
        self.assertEqual(maps.shared_tabs.currentIndex(), static_result_index)

        for page in (static_2d, static_3d, playback_3d):
            for value in vars(page).values():
                header = getattr(value, "header", None)
                if isinstance(header, QAbstractButton) and header.isCheckable():
                    self.assertFalse(header.isChecked())
                    self.assertFalse(header.property("persistUiState"))

        self.assertFalse(maps.scripted_source_section.header.isChecked())
        self.assertFalse(maps.scripted_source_section.header.property("persistUiState"))

        for page in maps._pages.values():
            cleanup = getattr(page, "_cleanup_plots", None)
            if callable(cleanup):
                cleanup()
        maps.deleteLater()

    def test_project_manager_open_replaces_untouched_startup_placeholder(self):
        manager = ProjectManagerWindow()
        manager.add_startup_empty_project()
        startup_page = manager.current_page()
        self.assertIsNotNone(startup_page)
        self.assertTrue(startup_page.record.startup_placeholder)
        self.assertEqual(manager.tabs.count(), 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opened.magpy.json"
            source = StudioAdapter(compact_workflow_scene())
            source.save_file(path)

            manager.open_project_path(str(path))
            self.application.processEvents()

            self.assertEqual(manager.tabs.count(), 1)
            opened_page = manager.current_page()
            self.assertIsNotNone(opened_page)
            self.assertIsNot(opened_page, startup_page)
            self.assertEqual(opened_page.record.path, path.resolve())
            self.assertFalse(opened_page.record.startup_placeholder)

        self.close_project_manager_for_test(manager)

    def test_project_manager_open_preserves_changed_or_deliberate_empty_tabs(self):
        manager = ProjectManagerWindow()
        manager.add_startup_empty_project()
        startup_page = manager.current_page()
        self.assertIsNotNone(startup_page)
        startup_page.record.adapter.add_template("guide_box")
        self.assertTrue(startup_page.record.is_dirty())

        # A user-created blank project is deliberate and is never treated as the
        # disposable launch placeholder either.
        manager.new_project_from_kind("empty")
        deliberate_page = manager.current_page()
        self.assertIsNotNone(deliberate_page)
        self.assertFalse(deliberate_page.record.startup_placeholder)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opened.magpy.json"
            source = StudioAdapter(compact_workflow_scene())
            source.save_file(path)

            manager.open_project_path(str(path))
            self.application.processEvents()

            self.assertEqual(manager.tabs.count(), 3)
            self.assertGreaterEqual(manager.tabs.indexOf(startup_page), 0)
            self.assertGreaterEqual(manager.tabs.indexOf(deliberate_page), 0)
            self.assertEqual(manager.current_page().record.path, path.resolve())

        self.close_project_manager_for_test(manager)

    def test_project_manager_dashboard_summaries_align_with_application_rows(self):
        manager = ProjectManagerWindow()
        page = manager.add_project(StudioAdapter(compact_workflow_scene()))

        expected_positions = {
            "model": (1, 0, 1, 3),
            "maps": (2, 0, 2, 3),
            "exposures": (4, 0, 2, 3),
            "data_results": (6, 0, 2, 3),
        }
        for key, expected in expected_positions.items():
            frame = page.project_summary_frames[key]
            index = page.dashboard_grid.indexOf(frame)
            self.assertGreaterEqual(index, 0)
            self.assertEqual(page.dashboard_grid.getItemPosition(index), expected)
            self.assertEqual(frame.objectName(), "projectManagerDashboardGroupFrame")

        app_buttons = []
        for card in page.app_cards:
            self.assertEqual(card.objectName(), "projectManagerAppArea")
            self.assertEqual(card.frameShape(), QFrame.Shape.NoFrame)
            button = card.findChild(QPushButton, "projectManagerAppButton")
            name_label = card.findChild(QLabel, "projectManagerAppName")
            self.assertIsNotNone(button)
            self.assertIsNotNone(name_label)
            self.assertEqual(button.text(), "")
            self.assertEqual(button.property("applicationName"), name_label.text())
            layout = card.layout()
            self.assertIs(layout.itemAt(0).widget(), button)
            app_buttons.append(str(button.property("applicationName")))
        self.assertEqual(
            app_buttons,
            [
                "Model Editor",
                "Map Creator",
                "Exposure Analysis",
                "Exposure Editor",
                "Waveform Editor",
                "Dataset Manager",
                "Saved Results",
            ],
        )

        # Paired applications are children of the same outer group frame, so
        # Maps / Analysis and Exposures each have exactly one sunken boundary.
        self.assertIs(page.app_cards[1].parentWidget(), page.project_summary_frames["maps"])
        self.assertIs(page.app_cards[2].parentWidget(), page.project_summary_frames["maps"])
        self.assertIs(page.app_cards[3].parentWidget(), page.project_summary_frames["exposures"])
        self.assertIs(page.app_cards[4].parentWidget(), page.project_summary_frames["exposures"])
        self.assertIs(page.app_cards[5].parentWidget(), page.project_summary_frames["data_results"])
        self.assertIs(page.app_cards[6].parentWidget(), page.project_summary_frames["data_results"])

        self.assertIn("object", page.project_summary_labels["model"].text())
        self.assertIn("map", page.project_summary_labels["maps"].text())
        self.assertIn("exposure analys", page.project_summary_labels["maps"].text())
        self.assertIn("exposure", page.project_summary_labels["exposures"].text())
        self.assertIn("dataset", page.project_summary_labels["data_results"].text())
        self.assertIn("saved result", page.project_summary_labels["data_results"].text())
        self.assertEqual(len(page.project_connector_labels), 7)
        self.assertEqual(
            [label.accessibleName() for label in page.project_connector_labels],
            [
                "Model → Model Editor",
                "Maps / Analysis → Map Creator",
                "Maps / Analysis → Exposure Analysis",
                "Exposures → Exposure Editor",
                "Exposures → Waveform Editor",
                "Data / Results → Dataset Manager",
                "Data / Results → Saved Results",
            ],
        )
        self.assertEqual(
            {key: label.toolTip() for key, label in page.project_component_title_labels.items()},
            {
                "model": "Open Model Editor",
                "maps": "Open Map Creator",
                "exposures": "Open Exposure Editor",
                "data_results": "Open Dataset Manager",
            },
        )
        self.assertEqual(
            {key: label.accessibleName() for key, label in page.project_component_title_labels.items()},
            {
                "model": "Open Model Editor",
                "maps": "Open Map Creator",
                "exposures": "Open Exposure Editor",
                "data_results": "Open Dataset Manager",
            },
        )
        self.assertIn("Status:", page.project_status_label.text())
        self.assertFalse(bool(page.project_status_label.property("dirty")))
        self.assertIn("Contents:", page.project_metadata_label.text())
        self.close_project_manager_for_test(manager)


    def test_project_manager_component_labels_open_primary_applications(self):
        manager = ProjectManagerWindow()
        page = manager.add_project(StudioAdapter(compact_workflow_scene()))

        page.project_component_title_labels["model"].clicked.emit()
        self.application.processEvents()
        self.assertIsNotNone(page.record.model_window)
        page.record.model_window.close()

        page.project_component_title_labels["maps"].clicked.emit()
        self.application.processEvents()
        self.assertIsNotNone(page.record.maps_window)
        page.record.maps_window.close()

        page.project_component_title_labels["exposures"].clicked.emit()
        self.application.processEvents()
        self.assertIsNotNone(page.record.exposure_window)
        page.record.exposure_window.close()

        page.project_component_title_labels["data_results"].clicked.emit()
        self.application.processEvents()
        self.assertIsNotNone(page.record.dataset_window)
        page.record.dataset_window.close()

        self.close_project_manager_for_test(manager)

    def test_project_manager_reuses_one_project_owned_maps_workspace(self):
        manager = ProjectManagerWindow()
        page = manager.add_project(StudioAdapter(compact_workflow_scene()))
        source_document = page.record.adapter.document()

        page.open_maps("static")
        self.application.processEvents()
        maps = page.record.maps_window
        self.assertIsInstance(maps, UnifiedFieldWorkspaceDialog)
        self.assertEqual(maps.mode, "static")
        self.assertEqual(maps.snapshot_adapter.document(), source_document)
        self.assertIs(maps.project_adapter, page.record.adapter)

        page.open_maps("playback")
        self.application.processEvents()
        self.assertIs(page.record.maps_window, maps)
        self.assertEqual(maps.mode, "playback")
        self.assertIsNone(page.record.model_window)

        maps.close()
        self.close_project_manager_for_test(manager)

    def test_map_preview_legend_updates_each_dialogs_local_object_checklist(self):
        window = self.make_window()
        guide = window.adapter.add_template("guide_box")

        map_2d = FieldMapDialog(window.adapter, window)
        self.assertIn(guide, map_2d.selected_object_outline_ids())
        map_2d._preview_object_visibility_requested(guide, False)
        map_2d._selector_preview_timer.stop()
        self.assertNotIn(guide, map_2d.selected_object_outline_ids())
        self.assertTrue(map_2d._selector_scene_dirty)
        map_2d._render_preview_plane()
        two_d_legend = {
            trace["meta"]["field_workbench_object_id"]: trace["visible"]
            for trace in map_2d._selector_scene_figure["data"]
            if trace.get("meta", {}).get("field_workbench_map_preview_legend")
        }
        self.assertEqual(two_d_legend[guide], "legendonly")
        map_2d._preview_object_visibility_requested(guide, True)
        map_2d._selector_preview_timer.stop()
        self.assertIn(guide, map_2d.selected_object_outline_ids())

        map_3d = FieldVolumeMapDialog(window.adapter, window)
        self.assertIn(guide, map_3d.selected_scene_object_ids())
        map_3d._preview_object_visibility_requested(guide, False)
        map_3d._selector_preview_timer.stop()
        self.assertNotIn(guide, map_3d.selected_scene_object_ids())
        map_3d._render_preview_slice_volume()
        three_d_legend = {
            trace["meta"]["field_workbench_object_id"]: trace["visible"]
            for trace in map_3d._selector_scene_figure["data"]
            if trace.get("meta", {}).get("field_workbench_map_preview_legend")
        }
        self.assertEqual(three_d_legend[guide], "legendonly")
        map_3d._set_scene_object_opacity(guide, 0.37)
        map_3d._selector_preview_timer.stop()
        self.assertTrue(map_3d._selector_scene_dirty)

        self.assertFalse(hasattr(MainWindow, "open_field_map"))
        self.assertFalse(hasattr(MainWindow, "open_field_volume_map"))

        map_2d.plot.cleanup()
        map_2d.deleteLater()
        map_3d.plot.cleanup()
        map_3d.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_waveform_trace_csv_uses_one_common_time_column(self) -> None:
        traces = [
            {"label": "Sensor A — |B| (µT)", "unit": "µT", "time_s": [0.0, 0.5, 1.0], "values": [1.0, 2.0, 3.0]},
            {"label": "Drive voltage", "unit": "V", "time_s": [0.0, 0.5, 1.0], "values": [-5.0, 0.0, 5.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.csv"
            WaveformGLView._write_trace_csv(path, traces)
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows[0], "Time (s),Sensor A — |B| (µT),Drive voltage (V)")
        self.assertEqual(rows[1], "0,1,-5")
        self.assertEqual(rows[-1], "1,3,5")

    def test_3d_field_map_defaults_to_full_volume_and_waits_for_calculate(self):
        window = self.make_window()
        dialog = FieldVolumeMapDialog(window.adapter, window)
        self.assert_standard_window_controls(dialog)
        self.assertEqual(dialog.fullscreen_button.text(), "Fullscreen viewer")
        self.assertEqual(dialog.export_video_button.text(), "Export video")
        self.assertTrue(dialog.export_video_button.isHidden())
        self.assertIs(
            dialog.export_video_button.parentWidget(), dialog.fullscreen_button.parentWidget()
        )
        self.assertEqual(dialog.exit_button.text(), "Exit")
        self.assertNotEqual(
            dialog.exit_button.parentWidget(), dialog.fullscreen_button.parentWidget()
        )
        self.assertEqual(dialog.windowTitle(), "3D Field Map")
        self.assertEqual(dialog.slice_axis.currentData(), "z")
        self.assertEqual(dialog.slice_count.value(), 9)
        self.assertEqual(dialog.resolution.value(), 35)
        self.assertTrue(dialog.include_scene.isChecked())
        self.assertTrue(dialog.plot._include_3d_return_axes)
        self.assertTrue(dialog.plot._synchronize_legend_visibility)
        self.assertFalse(dialog.colour_minimum_enabled.isChecked())
        self.assertFalse(dialog.colour_maximum_enabled.isChecked())
        self.assertEqual(dialog.colour_minimum.suffix(), " µT")
        self.assertEqual(
            dialog.calculation_status.objectName(), "fieldVolumeMapStatusBar"
        )
        self.assertEqual(dialog.tabs.count(), 1)
        self.assertEqual(dialog.tabs.tabText(0), "Selector")
        self.assertIs(dialog.tabs.widget(0), dialog.plot)
        self.assertIsNone(
            dialog.tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.LeftSide)
        )
        self.assertIsNone(
            dialog.tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.RightSide)
        )
        self.assertFalse(hasattr(dialog, "reset_button"))
        self.assertEqual(dialog.calculate_button.text(), "Static map")
        self.assertEqual(dialog.waveform_button.text(), "Animated map")
        self.assertIn("map volume", dialog.calculation_status.currentMessage())
        self.assertEqual(dialog.view_type.currentData(), "volume")
        self.assertTrue(dialog.slice_options_section.isHidden())
        self.assertTrue(dialog.intensity_opacity_enabled.isChecked())
        self.assertAlmostEqual(dialog.volume_opacity_lower.value(), 15.0)
        self.assertAlmostEqual(dialog.volume_opacity_upper.value(), 40.0)
        self.assertAlmostEqual(dialog.volume_opacity_lower_level.value(), 0.0)
        self.assertAlmostEqual(dialog.volume_opacity_upper_level.value(), 100.0)
        self.assertTrue(dialog.volume_opacity_lower.isEnabled())
        self.assertTrue(dialog.volume_opacity_upper.isEnabled())
        self.assertTrue(dialog.volume_opacity_lower_level.isEnabled())
        self.assertTrue(dialog.volume_opacity_upper_level.isEnabled())
        self.assertFalse(dialog.map_volume_section.header.isChecked())
        self.assertFalse(dialog.sample_config_section.header.isChecked())
        self.assertTrue(dialog.slice_options_section.header.isChecked())
        self.assertFalse(dialog.scene_options_section.header.isChecked())
        self.assertTrue(dialog.display_options_section.header.isChecked())

        with mock.patch.object(dialog, "_start_render_job") as start_render:
            dialog._calculate_static_webgl(dialog._camera_default())

        start_render.assert_called_once()
        call = start_render.call_args.kwargs
        self.assertEqual(call["render_kind"], "static")
        self.assertEqual(call["job_label"], "Map 1")
        settings = call["request"]["map_settings"]
        self.assertEqual(settings["slice_axis"], "z")
        self.assertEqual(settings["view_type"], "volume")
        self.assertTrue(settings["intensity_opacity_enabled"])
        self.assertIn("scene_object_opacities", settings)
        self.assertEqual(
            settings["scene_object_opacities"],
            dialog.scene_object_opacity_values(),
        )
        self.assertAlmostEqual(settings["volume_opacity_lower"], 0.15)
        self.assertAlmostEqual(settings["volume_opacity_upper"], 0.40)
        self.assertAlmostEqual(settings["volume_opacity_lower_level"], 0.0)
        self.assertAlmostEqual(settings["volume_opacity_upper_level"], 1.0)
        self.assertEqual(settings["component"], "magnitude")
        self.assertIsNone(settings["colour_minimum"])
        self.assertIsNone(settings["colour_maximum"])
        self.assertTrue(dialog.calculate_button.isEnabled())
        self.assertEqual(dialog.tabs.count(), 1)
        dialog.plot.cleanup()
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_animated_brain_result_switches_between_frame_and_playback_summary(self):
        hierarchy = lpba40_region_hierarchy()
        keys = [node.key for node in hierarchy.walk()]
        row_count = len(keys)
        frame_values = np.zeros((3, row_count, 7), dtype=np.float32)
        for frame_index in range(3):
            frame_values[frame_index, :, 0:4] = np.asarray(
                [1.0, 2.0, 3.0, 4.0], dtype=np.float32
            ) + frame_index
            frame_values[frame_index, :, 4] = 85.0
            frame_values[frame_index, :, 5] = 95.0
            frame_values[frame_index, :, 6] = 12.5
        summary_values = np.tile(
            np.asarray([5.0, 4.0, 6.0, 8.0, 10.0, 20.0, 12.5]),
            (row_count, 1),
        )
        peak_times = np.full(row_count, 0.75, dtype=np.float64)

        def encoded(values, dtype):
            array = np.ascontiguousarray(values, dtype=dtype)
            return base64.b64encode(array.tobytes()).decode("ascii")

        payload = {
            "keys": keys,
            "shape": list(frame_values.shape),
            "values_b64": encoded(frame_values, np.float32),
            "source_frame_count": 3,
            "sampled_frame_indices_b64": encoded([0, 1, 2], np.int32),
            "sample_counts": [504] * row_count,
            "voxel_counts": [1000] * row_count,
            "playback_summary": {
                "shape": list(summary_values.shape),
                "values_b64": encoded(summary_values, np.float64),
                "peak_times_s_b64": encoded(peak_times, np.float64),
                "playback": {
                    "exposure_time_s": 1.0,
                    "waveform_mode": "sine",
                    "waveform_frequency_hz": 3.0,
                    "waveform_cycle_count": 3.0,
                    "waveform_pass_count": 1,
                    "sequencer_enabled": True,
                    "sequencer_step_count": 2,
                    "sequencer_loop_enabled": True,
                    "sequencer_equivalent_loops": 4.0,
                    "sequencer_step_activations": 8,
                    "frame_count": 3,
                    "analysis_sample_count": 13,
                    "regional_frame_sample_count": 3,
                    "atlas_sample_count": 504,
                },
            },
        }
        panel = _BrainResultRegionPanel(hierarchy)
        panel.set_frame_metric_data(payload)
        panel.set_filter_rules([(1, ">", 99.0, "filter", "#facc15")])
        self.application.processEvents()

        self.assertFalse(panel.mode_controls.isHidden())
        self.assertTrue(panel.current_frame_radio.isChecked())
        self.assertTrue(panel.playback_summary_radio.isEnabled())
        self.assertTrue(panel.table.headerItem().text(1).startswith("RMS |B|"))
        self.assertEqual(panel.filter_table.cellWidget(0, 1).currentText(), "RMS |B|")

        panel.playback_summary_radio.click()
        self.application.processEvents()
        self.assertEqual(panel._display_mode, "summary")
        self.assertFalse(panel.playback_statistics_widget.isHidden())
        self.assertTrue(
            panel.table.headerItem().text(1).startswith("Playback RMS |B|")
        )
        root = panel._items_by_key["whole_brain"]
        self.assertAlmostEqual(root.data(1, Qt.ItemDataRole.UserRole), 5.0)
        self.assertIn("µT·s", root.text(5))
        self.assertIn("1 s of physical waveform time", panel.playback_statistic_values["exposure"].text())
        self.assertIn("3 cycles", panel.playback_statistic_values["waveform"].text())
        self.assertIn("4 loops", panel.playback_statistic_values["sequencer"].text())
        self.assertIn("13 time samples", panel.playback_statistic_values["sampling"].text())
        self.assertEqual(panel.filter_table.rowCount(), 0)

        panel._add_filter()
        self.assertEqual(
            panel.filter_table.cellWidget(0, 1).currentText(), "Playback RMS |B|"
        )
        panel.current_frame_radio.click()
        self.application.processEvents()
        self.assertEqual(panel._display_mode, "frame")
        self.assertEqual(panel.filter_table.rowCount(), 1)
        self.assertEqual(panel.filter_table.cellWidget(0, 1).currentText(), "RMS |B|")
        panel.playback_summary_radio.click()
        self.application.processEvents()
        snapshot = panel.video_table_snapshot()
        self.assertEqual(snapshot["mode"], "summary")
        self.assertEqual(snapshot["units"][5], "µT·s")
        self.assertEqual(len(snapshot["summary_values"]), row_count)
        self.assertTrue(any(line.startswith("Exposure time:") for line in snapshot["summary_lines"]))

        panel.request_frame(2, 1.0)
        self.application.processEvents()
        self.assertAlmostEqual(root.data(1, Qt.ItemDataRole.UserRole), 5.0)
        panel.current_frame_radio.click()
        self.application.processEvents()
        self.assertAlmostEqual(root.data(1, Qt.ItemDataRole.UserRole), 3.0)
        panel.deleteLater()

    def test_brain_view_exposes_lpba40_region_controls(self):
        window = self.make_window()
        brain_id = window.adapter.add_template("sri24_brain")
        window.refresh_scene(refresh_plot=False)
        source_document = window.adapter.document()
        dialog = BrainViewDialog(window.adapter, brain_id, window)
        # Brain View deliberately defers the expensive selector figure until the
        # first showEvent(), after shared window-state persistence has restored the
        # saved controls. Exercise that real lifecycle instead of depending on the
        # old constructor-time render.
        self.assertFalse(dialog._selector_figure_initialized)
        self.assertEqual(len(dialog._region_trace_index), 0)
        dialog.show()
        self.application.processEvents()
        self.assertTrue(dialog._selector_figure_initialized)

        self.assert_standard_window_controls(dialog)
        self.assertIsNot(dialog.adapter, window.adapter)
        self.assertEqual(dialog.adapter.document(), source_document)
        self.assertEqual(dialog.windowTitle(), "Brain View")
        self.assertFalse(hasattr(dialog, "atlas_selection"))
        self.assertFalse(hasattr(dialog, "atlas_alignment_note"))
        self.assertEqual(len(dialog.coordinates), 3)
        np.testing.assert_allclose(
            [spin.value() for spin in dialog.coordinates],
            [0.0, 0.0, 50.0],
            atol=0.011,
        )
        self.assertAlmostEqual(dialog.span.value(), 300.0, places=1)
        self.assertEqual(dialog.map_volume_section.header.text(), "Map options")
        self.assertFalse(hasattr(dialog, "display_mode"))
        self.assertFalse(hasattr(dialog, "view_type"))
        self.assertEqual(dialog.static_analysis_button.text(), "Static analysis")
        self.assertEqual(dialog.animated_analysis_button.text(), "Animated analysis")
        self.assertEqual(dialog.export_video_button.text(), "Export video")
        self.assertTrue(dialog.export_video_button.isHidden())
        self.assertIs(
            dialog.export_video_button.parentWidget(), dialog.fullscreen_button.parentWidget()
        )
        self.assertEqual(dialog.reset_defaults_button.text(), "Reset defaults")
        self.assertEqual(dialog.exit_button.text(), "Exit")
        self.assertEqual(dialog.region_table.columnCount(), 8)
        header = dialog.region_table.header()
        self.assertEqual(
            header.contextMenuPolicy(), Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.assertTrue(
            all(
                header.sectionResizeMode(column) == QHeaderView.ResizeMode.Interactive
                for column in range(dialog.region_table.columnCount())
            )
        )
        # Sorting is enabled, and Qt may report the native sort indicator as shown
        # after setSortingEnabled(True). The native arrow subcontrols are suppressed
        # by the header stylesheet; Brain View renders its one explicit text glyph.
        self.assertEqual(header.objectName(), "brainRegionHeader")
        self.assertGreaterEqual(header.minimumHeight(), 30)
        self.assertIn("::up-arrow", header.styleSheet())
        self.assertIn("::down-arrow", header.styleSheet())
        self.assertTrue(dialog.region_table.headerItem().text(0).endswith("▲"))
        dialog.region_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
        dialog._update_region_sort_indicator(1, Qt.SortOrder.DescendingOrder)
        self.assertTrue(dialog.region_table.headerItem().text(1).endswith("▼"))
        dialog.region_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        dialog._update_region_sort_indicator(0, Qt.SortOrder.AscendingOrder)
        self.assertEqual(len(dialog.region_column_actions), 8)
        self.assertTrue(all(action.isCheckable() for action in dialog.region_column_actions))
        dialog.region_column_actions[2].setChecked(False)
        self.assertTrue(dialog.region_table.isColumnHidden(2))
        dialog.region_column_actions[2].setChecked(True)
        self.assertFalse(dialog.region_table.isColumnHidden(2))
        self.assertEqual(dialog.region_table.topLevelItemCount(), 1)
        root = dialog.region_table.topLevelItem(0)
        self.assertEqual(root.text(0), "Whole brain")
        self.assertEqual(root.childCount(), 8)
        # Whole brain is now an ordinary selectable/highlightable row rather than
        # an implicit fallback when nothing is selected.
        self.assertTrue(root.flags() & Qt.ItemFlag.ItemIsSelectable)
        self.assertFalse(hasattr(dialog, "region_trace_select_major_button"))
        self.assertFalse(hasattr(dialog, "region_trace_clear_button"))
        self.assertFalse(hasattr(dialog, "region_filter_note"))
        temporal = dialog.region_table.findItems(
            "Temporal lobe",
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive,
            0,
        )[0]
        self.assertEqual(temporal.childCount(), 5)
        self.assertTrue(all(temporal.child(index).childCount() == 2 for index in range(5)))
        self.assertFalse(hasattr(dialog, "region_sort"))
        self.assertEqual(dialog.region_filter_table.columnCount(), 5)
        self.assertEqual(dialog.region_filter_table.rowCount(), 0)
        # Do not assert QTableView/viewport mouse-tracking flags here. Qt's
        # stylesheet engine may enable viewport tracking internally when the
        # application QSS contains :hover selectors, even though Brain View does
        # not use hover to select rules or focus editors. The behavioral
        # invariants are asserted below: NoFocus/NoSelection on the table,
        # ClickFocus on editors, programmatic changes cannot move explicit rule
        # selection, and only a deliberate child-widget press may select a row.
        self.assertEqual(
            dialog.region_filter_table.focusPolicy(), Qt.FocusPolicy.NoFocus
        )
        self.assertEqual(
            dialog.region_filter_table.editTriggers(),
            QAbstractItemView.EditTrigger.NoEditTriggers,
        )
        self.assertEqual(
            dialog.region_filter_table.selectionMode(),
            QAbstractItemView.SelectionMode.NoSelection,
        )
        dialog._add_region_filter()
        self.assertEqual(dialog.region_filter_table.rowCount(), 1)
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, 0)
        self.assertTrue(dialog.region_filter_remove_button.isEnabled())
        row_number = dialog.region_filter_table.item(0, 0)
        metric = dialog.region_filter_table.cellWidget(0, 1)
        condition = dialog.region_filter_table.cellWidget(0, 2)
        threshold = dialog.region_filter_table.cellWidget(0, 3)
        action_cell = dialog.region_filter_table.cellWidget(0, 4)
        action = action_cell.findChild(QComboBox, "brainFilterActionCombo")
        colour = action_cell.findChild(QPushButton, "brainFilterColourButton")
        self.assertEqual(row_number.text(), "1")
        self.assertFalse(row_number.flags() & Qt.ItemFlag.ItemIsSelectable)
        self.assertIsInstance(metric, QComboBox)
        self.assertIsInstance(condition, QComboBox)
        self.assertEqual(metric.currentText(), "RMS |B|")
        self.assertEqual(condition.currentText(), ">")
        self.assertEqual(threshold.suffix(), " µT")
        self.assertIsInstance(action, QComboBox)
        self.assertEqual(action.currentText(), "Filter")
        self.assertEqual(action.currentData(), "filter")
        self.assertIsInstance(colour, QPushButton)
        self.assertTrue(colour.isHidden())
        # Do not assert WA_Hover on editor widgets either. Qt's stylesheet
        # engine may enable that implementation attribute while polishing
        # :hover rules. Focus and explicit-selection behavior below are the
        # stable user-facing contract.
        self.assertEqual(metric.focusPolicy(), Qt.FocusPolicy.ClickFocus)
        self.assertEqual(condition.focusPolicy(), Qt.FocusPolicy.ClickFocus)
        self.assertEqual(threshold.focusPolicy(), Qt.FocusPolicy.ClickFocus)

        uniformity_index = metric.findText("Uniformity")
        metric.setCurrentIndex(uniformity_index)
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, 0)
        self.assertTrue(dialog.region_filter_remove_button.isEnabled())
        self.assertEqual(threshold.suffix(), " %")
        self.assertAlmostEqual(threshold.maximum(), 100.0)

        metric.setCurrentIndex(metric.findText("Peak |B|"))
        self.assertEqual(metric.currentData(), 4)
        self.assertEqual(threshold.suffix(), " µT")

        metric.setCurrentIndex(metric.findText("RMS |B|"))
        threshold.setValue(50.0)
        for item in dialog._items_by_key.values():
            item.setData(1, Qt.ItemDataRole.UserRole, 10.0)
        matching_leaf = dialog._items_by_key["superior_temporal_gyrus.left"]
        matching_right = dialog._items_by_key["superior_temporal_gyrus.right"]
        matching_leaf.setData(1, Qt.ItemDataRole.UserRole, 100.0)
        matching_right.setData(1, Qt.ItemDataRole.UserRole, 100.0)
        dialog._region_metrics = {"filter-test": object()}
        dialog._apply_region_filters()
        # Filter rules are exclusionary: rows matching RMS |B| > 50 µT are
        # removed, while non-matching rows remain. Structural ancestors stay
        # visible when needed to preserve the hierarchy path.
        self.assertTrue(matching_leaf.isHidden())
        self.assertTrue(matching_right.isHidden())
        self.assertFalse(matching_leaf.parent().isHidden())
        self.assertFalse(temporal.isHidden())
        self.assertFalse(dialog._items_by_key["middle_temporal_gyrus"].isHidden())

        # Switching the same rule to Highlight stops filtering and colours only
        # matching rows. The swatch becomes available beside the Action combo.
        action.setCurrentIndex(action.findData("highlight"))
        self.assertFalse(colour.isHidden())
        self.assertFalse(dialog._items_by_key["middle_temporal_gyrus"].isHidden())
        self.assertFalse(matching_leaf.isHidden())
        self.assertFalse(matching_right.isHidden())
        highlight_brush = matching_leaf.background(0).color()
        self.assertTrue(highlight_brush.isValid())
        self.assertNotEqual(highlight_brush.alpha(), 0)
        action.setCurrentIndex(action.findData("filter"))
        self.assertTrue(colour.isHidden())
        self.assertTrue(matching_leaf.isHidden())
        self.assertTrue(matching_right.isHidden())
        self.assertFalse(dialog._items_by_key["middle_temporal_gyrus"].isHidden())

        dialog._add_region_filter()
        self.assertEqual(dialog.region_filter_table.rowCount(), 2)
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, 1)
        side_number = dialog.region_filter_table.item(1, 0)
        side_metric = dialog.region_filter_table.cellWidget(1, 1)
        side_metric.setCurrentIndex(side_metric.findText("Side"))
        side_condition = dialog.region_filter_table.cellWidget(1, 2)
        side_value = dialog.region_filter_table.cellWidget(1, 3)
        self.assertEqual(side_condition.currentText(), "=")
        self.assertIsInstance(side_value, QComboBox)
        self.assertEqual(side_value.currentText(), "Left")
        self.assertEqual(side_value.focusPolicy(), Qt.FocusPolicy.ClickFocus)

        # Selection is explicit state owned by Brain View, not QTableWidget's
        # current/selection model. Programmatic value changes do not steal it,
        # while an actual child-widget mouse press deliberately selects its row.
        dialog._region_filter_cell_clicked(0, 0)
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, 0)
        side_value.setCurrentIndex(side_value.findData("right"))
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, 0)
        dialog._region_filter_wheel_filter.eventFilter(
            side_value, QEvent(QEvent.Type.MouseButtonPress)
        )
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, 1)
        side_value.setCurrentIndex(side_value.findData("left"))
        self.assertEqual(dialog._selected_region_filter_row, 1)
        # Multiple Filter rules combine with AND for exclusion. The high-field
        # left leaf matches both RMS > 50 and Side = Left, so it is removed;
        # the high-field right leaf matches only the RMS rule and remains.
        self.assertTrue(matching_leaf.isHidden())
        self.assertFalse(matching_right.isHidden())
        self.assertFalse(matching_leaf.parent().isHidden())
        self.assertFalse(temporal.isHidden())

        dialog._region_filter_cell_clicked(1, 0)
        self.assertEqual(dialog._selected_region_filter_row, 1)
        dialog._remove_region_filter()
        self.assertEqual(dialog.region_filter_table.rowCount(), 1)
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, -1)

        dialog._region_filter_cell_clicked(0, 0)
        dialog._remove_region_filter()
        self.assertEqual(dialog.region_filter_table.rowCount(), 0)
        self.assertEqual(dialog.region_filter_table.currentRow(), -1)
        self.assertEqual(dialog._selected_region_filter_row, -1)
        self.assertTrue(all(not item.isHidden() for item in dialog._items_by_key.values()))
        dialog._region_metrics = {}
        self.assertGreater(dialog.analysis_samples.point_count, 0)
        self.assertLessEqual(dialog.display_samples.point_count, 14_000)
        self.assertEqual(len(dialog._region_trace_index), 56)
        self.assertTrue(dialog._scene_trace_indices)
        self.assertTrue(dialog._scene_legend_trace_indices)
        self.assertEqual(window.adapter.document(), source_document)

        middle_temporal = dialog._items_by_key["middle_temporal_gyrus"]
        with mock.patch.object(
            dialog.plot, "apply_incremental_update"
        ) as incremental_update, mock.patch.object(
            dialog.plot, "set_figure"
        ) as full_figure_update:
            dialog.region_table.setCurrentItem(middle_temporal)
            self.application.processEvents()

        full_figure_update.assert_not_called()
        incremental_update.assert_called_once()
        highlight_call = incremental_update.call_args.kwargs
        self.assertEqual(len(highlight_call["trace_updates"]), 2)
        self.assertTrue(
            all(
                set(update) == {"color", "opacity"}
                for update in highlight_call["trace_updates"].values()
            )
        )
        self.assertIn("Middle temporal gyrus", highlight_call["layout_updates"]["title.text"])

        with mock.patch.object(
            dialog.plot, "apply_incremental_update"
        ) as incremental_update, mock.patch.object(
            dialog.plot, "set_figure"
        ) as full_figure_update:
            dialog.show_grid.setChecked(False)
        full_figure_update.assert_not_called()
        incremental_update.assert_called_once()
        grid_call = incremental_update.call_args.kwargs
        self.assertFalse(grid_call["layout_updates"]["scene.xaxis.visible"])
        self.assertFalse(grid_call["layout_updates"]["scene.yaxis.visible"])
        self.assertFalse(grid_call["layout_updates"]["scene.zaxis.visible"])
        dialog.show_grid.setChecked(True)

        context_object_id = next(iter(dialog._scene_trace_indices))
        with mock.patch.object(
            dialog.plot, "apply_incremental_update"
        ) as incremental_update, mock.patch.object(
            dialog.plot, "set_figure"
        ) as full_figure_update:
            dialog._set_scene_object_opacity(context_object_id, 0.25)
            dialog._set_scene_object_selected(context_object_id, False)
        full_figure_update.assert_not_called()
        self.assertGreaterEqual(incremental_update.call_count, 2)
        for call in incremental_update.call_args_list:
            for update in call.kwargs["trace_updates"].values():
                self.assertTrue(set(update) <= {"visible", "opacity"})

        settings = dialog._waveform_map_settings()
        self.assertEqual(settings["atlas_name"], "LPBA40")
        self.assertEqual(settings["view_type"], "volume")
        self.assertEqual(settings["waveform_render_mode"], "volume")
        self.assertEqual(settings["selected_region_name"], "Middle temporal gyrus")
        self.assertTrue(settings["highlight_selected_region"])
        self.assertEqual(len(settings["selected_region_label_ids"]), 2)
        self.assertEqual(settings["highlight_region_label_ids"], settings["selected_region_label_ids"])
        self.assertEqual(len(settings["trace_regions"]), 1)
        self.assertEqual(settings["trace_regions"][0]["key"], "whole_brain")
        self.assertFalse(settings["scroll_observation_traces"])
        self.assertEqual(settings["trace_region_tree"]["key"], "whole_brain")
        self.assertEqual(len(settings["trace_region_tree"]["children"]), 8)
        self.assertEqual(len(settings["trace_points_mm"]), dialog.analysis_samples.point_count)
        self.assertEqual(len(settings["trace_point_weights_mm3"]), dialog.analysis_samples.point_count)
        self.assertEqual(len(settings["overlay_points_mm"]), dialog.display_samples.point_count)
        self.assertEqual(len(settings["overlay_point_label_ids"]), dialog.display_samples.point_count)

        dialog.close()
        window.plot.cleanup()
        window.deleteLater()

    def test_3d_render_jobs_open_immediate_tabs_and_keep_frozen_settings(self):
        window = self.make_window()
        dialog = FieldVolumeMapDialog(window.adapter, window)
        base_settings = dialog._waveform_map_settings()
        base_settings.update(
            {
                "resolution": 2,
                "slice_count": 1,
                "view_type": "slices",
                "waveform_render_mode": "slices",
                "display_mode": "slices",
                "include_scene": False,
                "scene_object_ids": [],
                "show_grid": False,
            }
        )
        first_request = {
            "map_settings": dict(base_settings, centre_mm=[0.0, 0.0, 0.0]),
            "active_coil_ids": ["coil_a"],
            "initial_camera": dialog._camera_default(),
        }
        second_request = {
            "map_settings": dict(base_settings, centre_mm=[15.0, 0.0, 0.0]),
            "active_coil_ids": ["coil_a"],
            "initial_camera": dialog._camera_default(),
        }

        dialog._start_render_job(
            render_kind="static",
            request=first_request,
            job_label="Map 1",
            tooltip="First background map",
        )
        first_request["map_settings"]["centre_mm"][0] = 999.0
        dialog._start_render_job(
            render_kind="static",
            request=second_request,
            job_label="Map 2",
            tooltip="Second background map",
        )

        self.assertEqual(dialog.tabs.count(), 3)
        self.assertIsInstance(dialog.tabs.widget(1), _FieldRenderJobPage)
        self.assertIsInstance(dialog.tabs.widget(2), _FieldRenderJobPage)
        self.assertEqual(dialog._active_render_job_count(), 2)
        self.assertTrue(dialog.calculate_button.isEnabled())
        self.assertTrue(dialog.waveform_button.isEnabled())

        deadline = time.monotonic() + 30.0
        while dialog._render_jobs and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.application.processEvents()

        self.assertFalse(dialog._render_jobs)
        self.assertIsInstance(dialog.tabs.widget(1), WaveformGLView)
        self.assertIsInstance(dialog.tabs.widget(2), WaveformGLView)
        self.assertEqual(dialog.tabs.widget(1)._payload["centre_mm"], [0.0, 0.0, 0.0])
        self.assertEqual(dialog.tabs.widget(2)._payload["centre_mm"], [15.0, 0.0, 0.0])
        self.assertNotIn("fixed_vectors", dialog.tabs.widget(1)._payload)
        self.assertNotIn("volume_fixed_vectors", dialog.tabs.widget(2)._payload)
        dialog._cleanup_plots()
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_3d_render_job_page_uses_weighted_task_plan_without_rewinding(self):
        page = _FieldRenderJobPage(
            "Map 1",
            render_kind="static",
            request={"map_settings": {"display_mode": "both"}},
        )
        task_ids = [task["id"] for task in page._task_plan]
        self.assertEqual(
            task_ids,
            ["queue", "launch", "grid", "field", "flux", "assemble", "viewer"],
        )
        self.assertAlmostEqual(
            sum(task["progress_weight"] for task in page._task_plan),
            1.0,
        )
        self.assertEqual(page.task_list.topLevelItemCount(), len(task_ids))
        self.assertEqual(
            page.task_list.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Expanding,
        )
        self.assertEqual(
            page.task_list.sizePolicy().verticalPolicy(),
            QSizePolicy.Policy.Expanding,
        )
        self.assertGreater(page.task_list.maximumHeight(), 10_000)
        self.assertEqual(page.progress_bar.maximum(), 1000)
        self.assertEqual(page._task_by_id["queue"]["status"], "active")

        page.update_progress(
            {
                "render_task": "launch",
                "stage": "Starting 3D render worker…",
                "completed": 0,
                "total": 1,
                "unit": "render job",
            }
        )
        self.assertEqual(page._task_by_id["queue"]["status"], "complete")
        self.assertEqual(page._task_by_id["launch"]["status"], "active")

        page.update_progress(
            {
                "render_task": "grid",
                "render_task_complete": True,
                "stage": "3D sample grid ready",
                "completed": 1,
                "total": 1,
                "unit": "render steps",
            }
        )
        page.update_progress(
            {
                "render_task": "field",
                "stage": "3D volume grid: evaluating scene field",
                "completed": 25,
                "total": 100,
                "unit": "field samples",
            }
        )
        first = page.progress_bar.value()
        self.assertGreater(first, 0)
        self.assertLess(first, 1000)
        self.assertIn("25/100 samples", page._task_items["field"].text(2))

        page.update_progress(
            {
                "render_task": "field",
                "stage": "3D volume grid: revised solver pass",
                "completed": 10,
                "total": 100,
                "unit": "field samples",
            }
        )
        self.assertGreaterEqual(page.progress_bar.value(), first)

        page.update_progress(
            {
                "render_task": "flux",
                "stage": "3D fluxline grid: preparing field samples",
                "completed": 0,
                "total": 50,
                "unit": "field samples",
            }
        )
        self.assertEqual(page._task_by_id["field"]["status"], "complete")
        page.update_progress(
            {
                "render_task": "assemble",
                "render_task_complete": True,
                "stage": "WebGL render package ready",
                "completed": 3,
                "total": 3,
                "unit": "render steps",
            }
        )
        page.set_rendering()
        self.assertEqual(page._task_by_id["viewer"]["status"], "active")
        self.assertLess(page.progress_bar.value(), 1000)
        page.set_ready()
        self.assertEqual(page.progress_bar.value(), 1000)
        self.assertTrue(
            all(task["status"] == "complete" for task in page._task_plan)
        )

        waveform_page = _FieldRenderJobPage(
            "Waveform 1", render_kind="waveform", request={}
        )
        self.assertNotIn("flux", waveform_page._task_by_id)
        self.assertIn("coil bases", waveform_page._task_by_id["field"]["label"])
        self.assertIn("prepare_playback", waveform_page._task_by_id)

        measured_waveform_page = _FieldRenderJobPage(
            "Measured waveform",
            render_kind="waveform",
            request={"map_kind": "3d", "measurement_volume_count": 1},
        )
        measured_ids = [task["id"] for task in measured_waveform_page._task_plan]
        self.assertEqual(
            measured_ids,
            [
                "queue", "launch", "grid", "field", "prepare_playback",
                "assemble", "viewer",
            ],
        )
        self.assertAlmostEqual(
            sum(task["progress_weight"] for task in measured_waveform_page._task_plan),
            1.0,
        )
        self.assertNotIn("measurement_basis", measured_waveform_page._task_by_id)
        self.assertNotIn("measurement_frames", measured_waveform_page._task_by_id)
        self.assertNotIn("measurement_summary", measured_waveform_page._task_by_id)

        measured_2d_waveform_page = _FieldRenderJobPage(
            "Measured 2D waveform",
            render_kind="waveform",
            request={"map_kind": "2d", "measurement_volume_count": 1},
        )
        measured_2d_ids = [task["id"] for task in measured_2d_waveform_page._task_plan]
        self.assertEqual(
            measured_2d_ids,
            [
                "queue", "launch", "grid", "field", "prepare_playback",
                "assemble", "viewer",
            ],
        )
        self.assertAlmostEqual(
            sum(task["progress_weight"] for task in measured_2d_waveform_page._task_plan),
            1.0,
        )

        two_d_page = _FieldRenderJobPage(
            "Map 2",
            render_kind="2d_static",
            request={"map_settings": {"display_mode": "both"}},
        )
        self.assertEqual(
            [task["id"] for task in two_d_page._task_plan],
            ["queue", "launch", "grid", "field", "flux", "assemble", "viewer"],
        )
        self.assertIn("heatmap", two_d_page._task_by_id["field"]["label"].lower())
        self.assertIn("Plotly", two_d_page._task_by_id["assemble"]["label"])
        self.assertIn("Plotly", two_d_page._task_by_id["viewer"]["label"])
        self.assertAlmostEqual(
            sum(task["progress_weight"] for task in two_d_page._task_plan),
            1.0,
        )
        page.deleteLater()
        waveform_page.deleteLater()
        measured_waveform_page.deleteLater()
        two_d_page.deleteLater()

    def test_3d_render_concurrency_limiter_honours_worker_preference(self):
        limiter = _FieldRenderConcurrencyLimiter(1)
        release_jobs = threading.Event()
        state_lock = threading.Lock()
        entered: list[int] = []
        active = 0
        maximum_active = 0

        def run_job(index: int) -> None:
            nonlocal active, maximum_active
            limiter.acquire(threading.Event())
            with state_lock:
                entered.append(index)
                active += 1
                maximum_active = max(maximum_active, active)
            release_jobs.wait(timeout=5.0)
            with state_lock:
                active -= 1
            limiter.release()

        threads = [
            threading.Thread(target=run_job, args=(index,), daemon=True)
            for index in range(3)
        ]
        try:
            threads[0].start()
            deadline = time.monotonic() + 2.0
            while limiter.snapshot()[:2] != (1, 0) and time.monotonic() < deadline:
                time.sleep(0.01)
            threads[1].start()
            deadline = time.monotonic() + 2.0
            while limiter.snapshot()[:2] != (1, 1) and time.monotonic() < deadline:
                time.sleep(0.01)
            threads[2].start()
            deadline = time.monotonic() + 2.0
            while limiter.snapshot()[:2] != (1, 2) and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual(limiter.snapshot(), (1, 2, 1))
            limiter.configure(2)
            deadline = time.monotonic() + 2.0
            while limiter.snapshot()[:2] != (2, 1) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(limiter.snapshot(), (2, 1, 2))
        finally:
            release_jobs.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(timeout=5.0)

        for thread in threads:
            self.assertFalse(thread.is_alive())
        self.assertEqual(entered, [0, 1, 2])
        self.assertEqual(maximum_active, 2)
        self.assertEqual(limiter.snapshot(), (0, 0, 2))

    def test_3d_render_process_exit_messages_decode_native_faults(self):
        posix = _field_render_process_exit_message(
            -int(signal.SIGSEGV),
            last_stage="Map 1: evaluating detailed coil model",
            crash_log_path="/tmp/render/native-crash.log",
        )
        self.assertIn("SIGSEGV (segmentation fault)", posix)
        self.assertIn("Field Workbench stayed open", posix)
        self.assertIn("Last reported stage", posix)
        self.assertIn("native-crash.log", posix)

        windows = _field_render_process_exit_message(-1073741819)
        self.assertIn("Windows access violation (0xC0000005)", windows)

    def test_3d_field_map_full_volume_view_reuses_main_options_for_static_and_waveform(self):
        window = self.make_window()
        dialog = FieldVolumeMapDialog(window.adapter, window)
        self.assertEqual(dialog.view_type.currentData(), "volume")
        self.assertTrue(dialog.slice_options_section.isHidden())
        dialog.view_type.setCurrentIndex(dialog.view_type.findData("slices"))
        self.application.processEvents()
        self.assertFalse(dialog.slice_options_section.isHidden())
        dialog.view_type.setCurrentIndex(dialog.view_type.findData("volume"))
        self.application.processEvents()
        self.assertTrue(dialog.slice_options_section.isHidden())
        self.assertEqual(dialog.display_mode.itemText(0), "Heatmap")
        self.assertEqual(dialog.display_mode.itemText(2), "Heatmap + fluxlines")
        dialog.volume_opacity_lower.setValue(2.0)
        dialog.volume_opacity_upper.setValue(18.0)
        dialog.volume_opacity_lower_level.setValue(20.0)
        dialog.volume_opacity_upper_level.setValue(80.0)

        waveform_settings = dialog._waveform_map_settings()
        self.assertEqual(waveform_settings["view_type"], "volume")
        self.assertEqual(waveform_settings["waveform_render_mode"], "volume")
        self.assertAlmostEqual(waveform_settings["volume_opacity_lower"], 0.02)
        self.assertAlmostEqual(waveform_settings["volume_opacity_upper"], 0.18)
        self.assertAlmostEqual(waveform_settings["volume_opacity_lower_level"], 0.20)
        self.assertAlmostEqual(waveform_settings["volume_opacity_upper_level"], 0.80)
        self.assertTrue(waveform_settings["intensity_opacity_enabled"])
        self.assertIn("scene_object_opacities", waveform_settings)

        with mock.patch.object(dialog, "_start_render_job") as start_render:
            dialog._calculate_static_webgl(dialog._camera_default())
        settings = start_render.call_args.kwargs["request"]["map_settings"]
        self.assertEqual(settings["view_type"], "volume")
        self.assertAlmostEqual(settings["volume_opacity_lower"], 0.02)
        self.assertAlmostEqual(settings["volume_opacity_upper"], 0.18)
        self.assertAlmostEqual(settings["volume_opacity_lower_level"], 0.20)
        self.assertAlmostEqual(settings["volume_opacity_upper_level"], 0.80)
        self.assertTrue(settings["intensity_opacity_enabled"])

        dialog.intensity_opacity_enabled.setChecked(False)
        self.application.processEvents()
        self.assertFalse(dialog.volume_opacity_lower.isEnabled())
        self.assertFalse(dialog.volume_opacity_upper.isEnabled())
        self.assertFalse(dialog.volume_opacity_lower_level.isEnabled())
        self.assertFalse(dialog.volume_opacity_upper_level.isEnabled())
        self.assertFalse(dialog._waveform_map_settings()["intensity_opacity_enabled"])

        dialog.plot.cleanup()
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_field_map_progress_is_explicitly_centered_and_close_requests_cancel(self):
        parent = QDialog()
        parent.resize(800, 600)
        parent.move(100, 100)
        parent.show()
        self.application.processEvents()

        progress = _show_field_map_progress(parent, "Calculating test map…")
        self.assertIn(
            "_centre_field_map_progress(progress, parent)",
            _source_for(_show_field_map_progress),
        )
        self.assertFalse(progress.cancellation_requested())

        progress.close()
        self.application.processEvents()
        self.assertTrue(progress.cancellation_requested())
        self.assertTrue(progress.isVisible())
        self.assertIn("Cancelling", progress.windowTitle())
        with self.assertRaises(FieldCalculationCancelled):
            _update_field_map_progress(
                progress,
                {"completed": 1, "total": 10, "stage": "Test batch"},
            )

        _finish_field_map_progress(progress)
        parent.deleteLater()

    def test_measurement_analysis_progress_matches_map_progress_and_cancels(self):
        parent = QDialog()
        parent.resize(800, 600)
        parent.move(120, 120)
        parent.show()
        self.application.processEvents()

        progress = MeasurementAnalysisProgressDialog(parent)
        self.assertEqual(progress.objectName(), "measurementAnalysisProgress")
        self.assertEqual(
            progress.windowModality(), Qt.WindowModality.WindowModal
        )
        progress.update_from_solver(
            {
                "completed": 25,
                "total": 100,
                "stage": "Volume analysis — test object",
            }
        )
        bar = progress.findChild(QProgressBar)
        self.assertIsNotNone(bar)
        self.assertIn("25 / 100", bar.text())

        progress.close()
        self.application.processEvents()
        self.assertTrue(progress.cancellation_requested())
        self.assertTrue(progress.isVisible())
        self.assertIn("Cancelling volume analysis", progress.windowTitle())
        with self.assertRaises(FieldCalculationCancelled):
            progress.update_from_solver(
                {"completed": 26, "total": 100, "stage": "Next batch"}
            )
        progress.finish()
        parent.deleteLater()

    def test_measurement_dataset_manager_lists_scene_data_and_custom_coil_status(self):
        adapter = StudioAdapter(compact_workflow_scene())
        dataset_id = adapter.save_measurement_dataset(
            {
                "id": "bench_a",
                "label": "FM300 ch1",
                "sensor_id": "axis_sensor",
                "sensor_label": "Axis sensor",
                "output": "B",
                "coil_ids": ["coil_a"],
                "samples": [],
                "fit": {"factor": 0.973},
            }
        )
        self.assertEqual(dataset_id, "bench_a")
        dialog = MeasurementDatasetManagerDialog(adapter)
        self.assertEqual(dialog.dataset_tree.topLevelItemCount(), 1)
        self.assertEqual(dialog.dataset_tree.topLevelItem(0).text(0), "FM300 ch1")
        self.assertEqual(dialog.dataset_tree.columnCount(), 5)
        self.assertEqual(
            [dialog.dataset_tree.headerItem().text(index) for index in range(5)],
            ["Name", "Sensor", "Quantity", "Samples", "Updated (UTC)"],
        )
        self.assertEqual(dialog.link_tree.topLevelItemCount(), 2)
        mode_combo = dialog.link_tree.itemWidget(dialog.link_tree.topLevelItem(0), 1)
        dataset_combo = dialog.link_tree.itemWidget(dialog.link_tree.topLevelItem(0), 2)
        self.assertIsInstance(mode_combo, QComboBox)
        self.assertIsInstance(dataset_combo, QComboBox)
        self.assertEqual(mode_combo.currentData(), "custom")
        self.assertFalse(dataset_combo.isEnabled())
        dialog.refresh()
        mode_combo = dialog.link_tree.itemWidget(dialog.link_tree.topLevelItem(0), 1)
        source_combo = dialog.link_tree.itemWidget(dialog.link_tree.topLevelItem(0), 2)
        self.assertEqual(mode_combo.currentData(), "custom")
        self.assertFalse(source_combo.isEnabled())
        self.assertNotEqual(source_combo.currentData(), dataset_id)
        self.assertEqual(dialog.link_tree.topLevelItem(0).text(3), "1 ×")
        self.assertTrue(dialog.new_button.isEnabled())
        self.assertFalse(dialog.edit_button.isEnabled())
        dialog.dataset_tree.setCurrentItem(dialog.dataset_tree.topLevelItem(0))
        dialog.dataset_tree.topLevelItem(0).setSelected(True)
        self.assertTrue(dialog.edit_button.isEnabled())
        dialog.close()

    def test_dataset_manager_can_apply_ready_calibration_outside_fit_coil_group(self):
        adapter = StudioAdapter(compact_workflow_scene())
        calibration_id = adapter.save_measurement_calibration(
            {
                "id": "channel_1_fit",
                "label": "Channel 1 fit",
                "coil_ids": ["coil_a"],
                "sources": [],
                "weighting": "points",
                "fit": {"factor": 1.2345},
                "source_versions": {},
                "model_signature": adapter._measurement_calibration_model_signature(["coil_a"]),
            }
        )
        self.assertEqual(adapter.measurement_calibration_status(calibration_id), "Ready")

        dialog = MeasurementDatasetManagerDialog(adapter)
        second_row = dialog.link_tree.topLevelItem(1)
        self.assertEqual(second_row.data(0, Qt.ItemDataRole.UserRole), "coil_b")
        mode_combo = dialog.link_tree.itemWidget(second_row, 1)
        self.assertIsInstance(mode_combo, QComboBox)

        with mock.patch.object(QMessageBox, "warning") as warning:
            mode_combo.setCurrentIndex(mode_combo.findData("calibration"))
            self.application.processEvents()
            warning.assert_not_called()

        state = adapter.coil_correction_state("coil_b")
        self.assertEqual(state["mode"], "calibration")
        self.assertEqual(state["calibration_id"], calibration_id)
        self.assertAlmostEqual(state["resolved_factor"], 1.2345)
        dialog.close()

    def test_composite_calibration_editor_selects_raw_dataset_portions_and_target_coils(self):
        adapter = StudioAdapter(compact_workflow_scene())
        samples = adapter.measurement_sensor_samples("axis_sensor")[:2]
        dataset_a = adapter.save_measurement_dataset(
            {
                "id": "scan_a",
                "label": "Scan A",
                "sensor_id": "axis_sensor",
                "sensor_label": "Axis sensor",
                "output": "B",
                "coil_ids": ["coil_a"],
                "samples": [
                    {**samples[0], "measured_uT": 1.0},
                    {**samples[1], "measured_uT": 2.0},
                ],
            }
        )
        adapter.save_measurement_dataset(
            {
                "id": "scan_b",
                "label": "Scan B",
                "sensor_id": "axis_sensor",
                "sensor_label": "Axis sensor",
                "output": "B",
                "coil_ids": ["coil_b"],
                "samples": [{**samples[0], "measured_uT": 3.0}],
            }
        )
        manager = MeasurementDatasetManagerDialog(adapter)
        self.assertEqual(manager.dataset_tree.topLevelItemCount(), 2)
        self.assertEqual(manager.calibration_tree.topLevelItemCount(), 0)
        self.assertEqual(manager.link_tree.topLevelItemCount(), 2)

        dialog = MeasurementCompositeCalibrationDialog(
            adapter, initial_dataset_ids=[dataset_a]
        )
        self.assertEqual(dialog.source_tree.topLevelItemCount(), 2)
        self.assertEqual(dialog.coil_tree.topLevelItemCount(), 2)
        selected = dialog._selected_sources()
        self.assertEqual(selected, [{"dataset_id": dataset_a, "sample_indices": []}])
        self.assertEqual(dialog.weighting_combo.count(), 2)
        self.assertEqual(dialog.weighting_combo.itemData(0), "points")
        self.assertEqual(dialog.weighting_combo.itemData(1), "sources")
        dialog.close()
        manager.close()

    def test_dataset_manager_suite_window_uses_project_context_without_model_editor(self):
        adapter = StudioAdapter(compact_workflow_scene())
        adapter.save_measurement_dataset(
            {
                "id": "bench_suite",
                "label": "Suite bench",
                "sensor_id": "axis_sensor",
                "sensor_label": "Axis sensor",
                "output": "B",
                "coil_ids": ["coil_a"],
                "samples": [],
                "fit": {"factor": 1.02},
            }
        )
        context = ProjectContext(adapter)
        window = DatasetManagerWindow(context)
        self.assertIs(window.project_context, context)
        self.assertIn("Dataset Manager", window.windowTitle())
        self.assertEqual(window.manager.dataset_tree.topLevelItemCount(), 1)
        self.assertEqual(window.manager.dataset_tree.topLevelItem(0).text(0), "Suite bench")
        window.close()

    def test_measurement_dataset_editor_is_raw_only_and_maps_sensor_rows(self):
        adapter = StudioAdapter(compact_workflow_scene())
        dialog = MeasurementDatasetEditorDialog(
            adapter,
            initial_coil_ids=["coil_a"],
        )
        self.assertEqual(dialog.table.rowCount(), 61)
        self.assertEqual(dialog.sensor_selector.currentData(), "axis_sensor")
        self.assertEqual(dialog.output_selector.currentData(), "B")
        self.assertEqual(dialog.table.columnCount(), 6)
        self.assertEqual(
            [dialog.table.horizontalHeaderItem(index).text() for index in range(6)],
            ["#", "X (mm)", "Y (mm)", "Z (mm)", "Path (mm)", "Measured (µT)"],
        )
        self.assertIsNone(dialog.findChild(QTreeWidget, "measurementCalibrationCoilTree"))
        self.assertIsNone(dialog.findChild(QPushButton, "fitMeasurementCalibrationButton"))
        self.assertIsNone(dialog.findChild(QPushButton, "applyMeasurementCalibrationButton"))
        self.assertIsNone(dialog.findChild(QComboBox, "measurementDatasetSelector"))
        self.assertIsNone(dialog.findChild(QPushButton, "deleteMeasurementDatasetButton"))
        self.assertEqual(dialog.windowTitle(), "New measurement dataset")
        self.assertTrue(dialog.add_row_button.isEnabled())
        self.assertTrue(dialog.remove_rows_button.isEnabled())
        self.assertEqual(dialog.section_splitter.objectName(), "measurementDatasetSectionSplitter")
        self.assertEqual(dialog.section_splitter.orientation(), Qt.Orientation.Vertical)
        self.assertEqual(dialog.section_splitter.count(), 2)
        self.assertFalse(dialog.section_splitter.childrenCollapsible())
        self.assertIs(dialog.section_splitter.widget(1).findChild(type(dialog.table), "measurementCalibrationTable"), dialog.table)
        self.assertTrue(
            dialog.table.item(0, dialog.COL_X).flags() & Qt.ItemFlag.ItemIsEditable
        )
        self.assertEqual(
            MeasurementCalibrationDialog._parse_measurement_text("1.2\n2.3\n3.4\n"),
            [1.2, 2.3, 3.4],
        )
        self.assertEqual(
            MeasurementCalibrationDialog._parse_measurement_text(
                "x_mm,y_mm,z_mm,B_uT\n0,0,0,5.8\n0,0,10,6.1\n"
            ),
            [5.8, 6.1],
        )
        horizontal = "5.892\t6.474\t8.579\t13.649\t27.371\t72.06\t102.499"
        self.assertEqual(
            MeasurementCalibrationDialog._parse_measurement_text(
                horizontal, spreadsheet_cells=True
            ),
            [5.892, 6.474, 8.579, 13.649, 27.371, 72.06, 102.499],
        )
        initial_rows = dialog.table.rowCount()
        dialog.table.setCurrentCell(initial_rows - 1, dialog.COL_MEASURED)
        dialog.add_measurement_row()
        self.assertEqual(dialog.table.rowCount(), initial_rows + 1)
        dialog.table.item(initial_rows, dialog.COL_Z).setText("123")
        samples = dialog._samples_from_table()
        self.assertAlmostEqual(samples[initial_rows]["position_m"][2], 0.123, places=12)
        dialog.table.clearSelection()
        dialog._paste_values(list(float(value) for value in range(initial_rows + 4)))
        self.assertEqual(dialog.table.rowCount(), initial_rows + 4)
        self.assertEqual(dialog.table.item(0, dialog.COL_MEASURED).text(), "0")
        self.assertEqual(
            dialog.table.item(initial_rows + 3, dialog.COL_MEASURED).text(),
            str(initial_rows + 3),
        )
        payload = dialog._dataset_payload()
        self.assertEqual(payload["coil_ids"], [])
        self.assertIsNone(payload["fit"])
        self.assertTrue(all(sample.get("model_uT") is None for sample in payload["samples"]))
        dialog.deleteLater()

    def test_sensor_path_progress_matches_shared_progress_and_cancels(self):
        parent = QDialog()
        parent.resize(800, 600)
        parent.move(140, 140)
        parent.show()
        self.application.processEvents()

        progress = SensorPathProgressDialog(parent)
        self.assertEqual(progress.objectName(), "sensorPathCalculationProgress")
        self.assertEqual(progress.windowModality(), Qt.WindowModality.WindowModal)
        progress.update_from_solver(
            {
                "completed": 12,
                "total": 48,
                "stage": "Sensor path: evaluating scene field",
            }
        )
        bar = progress.findChild(QProgressBar)
        self.assertIsNotNone(bar)
        self.assertIn("12 / 48", bar.text())

        progress.close()
        self.application.processEvents()
        self.assertTrue(progress.cancellation_requested())
        self.assertTrue(progress.isVisible())
        self.assertIn("Cancelling sensor path", progress.windowTitle())
        with self.assertRaises(FieldCalculationCancelled):
            progress.update_from_solver(
                {"completed": 13, "total": 48, "stage": "Next batch"}
            )
        progress.finish()
        parent.deleteLater()

    def test_sensor_plot_waits_for_calculate_reports_solver_and_is_independent(self):
        window = self.make_window()
        calls = []

        def fake_sensor_export(output, **kwargs):
            calls.append((output, kwargs))
            callback = kwargs.get("progress_callback")
            if callback is not None:
                callback(
                    {
                        "completed": 1,
                        "total": 1,
                        "stage": "Sensor path: field sampling complete",
                    }
                )
            return {
                "figure": {"data": [], "layout": {}},
                "rows": [{"series": "test"}],
                "metadata": {"output": output},
            }

        window.adapter.sensor_plot_export_data = fake_sensor_export
        window.open_sensor_plot()
        self.application.processEvents()
        dialog = next(
            item
            for item in window._independent_windows.values()
            if isinstance(item, SensorPlotDialog)
        )
        self.assertFalse(dialog.isModal())
        self.assertIsNone(dialog.parentWidget())
        self.assertFalse(bool(dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))
        self.assertEqual(calls, [])
        self.assertEqual(dialog.calculation_status.objectName(), "sensorPathStatusBar")
        self.assertIn("Ready", dialog.calculation_status.currentMessage())

        dialog.calculate_button.click()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "B")
        self.assertIn("progress_callback", calls[0][1])
        self.assertIn("Solver:", dialog.calculation_status.currentMessage())
        self.assertIn("Total elapsed:", dialog.calculation_status.currentMessage())
        self.assertTrue(dialog.export_csv_button.isEnabled())

        dialog.close()
        self.application.processEvents()
        window.plot.cleanup()
        window.deleteLater()

    def test_camera_presets_refit_data_and_use_readable_perspective(self):
        view = PlotView()
        if view.web is None:
            view.cleanup()
            self.skipTest("Qt WebEngine is unavailable in this test environment.")
        scripts = []
        view._run_plot_script = lambda body: scripts.append(body) or True
        self.assertTrue(view._home_on_next_figure)
        self.assertTrue(view.home_view())
        self.assertIn("fieldWorkbenchHome", scripts[-1])
        self.assertTrue(view.set_camera_preset("front"))
        self.assertIn("fieldWorkbenchFitScene", scripts[-1])
        self.assertIn('"type": "perspective"', scripts[-1])
        self.assertIn('"y"', scripts[-1])
        self.assertIn("resetCameraDefault3d", view.MODEBAR_BUTTONS_TO_REMOVE)
        self.assertIn("resetCameraLastSave3d", view.MODEBAR_BUTTONS_TO_REMOVE)
        persistent_page = (view._temporary_directory / "plot-view.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("fieldWorkbenchCurrentView", persistent_page)
        self.assertIn("fieldWorkbenchLayoutWithView", persistent_page)
        self.assertIn("'scene.aspectmode': 'data'", persistent_page)
        self.assertNotIn("'scene.aspectratio': aspectratio", persistent_page)
        self.assertIn("0.5 * maxSpan", persistent_page)
        self.assertIn("fittedHalf", persistent_page)
        self.assertIn("fieldWorkbenchRenderExtentTrace", persistent_page)
        self.assertIn("field_workbench_render_extent", persistent_page)
        self.assertIn("uid: 'field-workbench-render-extent'", persistent_page)
        self.assertNotIn("uid: 'field-workbench:render-extent'", persistent_page)
        self.assertIn("marker: {size: 0.1, opacity: 0.001", persistent_page)
        self.assertIn("delete scene.aspectratio", persistent_page)
        update_source = _source_for(PlotView._push_pending_figure)
        self.assertIn("fieldWorkbenchCurrentView", update_source)
        self.assertIn("figure.layout = window.fieldWorkbenchLayoutWithView", update_source)
        self.assertIn("figure.data = window.fieldWorkbenchRenderExtentTrace", update_source)
        self.assertIn("modeBarButtonsToAdd", update_source)
        self.assertIn("Return axes", update_source)
        self.assertIn("firstGroup.insertBefore(homeButton, firstGroup.firstChild)", update_source)
        self.assertIn("click: plot => window.fieldWorkbenchHome(plot)", update_source)
        self.assertIn("if (includeReturnAxes)", update_source)
        self.assertNotIn("await Plotly.relayout(plot, savedView)", update_source)
        self.assertLess(
            update_source.index("figure.layout = window.fieldWorkbenchLayoutWithView"),
            update_source.index("Plotly.react"),
        )
        incremental_source = _source_for(PlotView.apply_incremental_update)
        push_incremental_source = _source_for(PlotView._push_incremental_state)
        push_incremental_delta_source = _source_for(PlotView._push_incremental_delta)
        trace_group_source = _source_for(PlotView._incremental_trace_groups)
        self.assertIn("_incremental_trace_state", incremental_source)
        self.assertIn("_incremental_layout_state", incremental_source)
        self.assertIn("delta_trace_updates", incremental_source)
        self.assertIn("_push_incremental_delta", incremental_source)
        self.assertIn("_push_incremental_delta", push_incremental_source)
        self.assertIn("property *shape*", trace_group_source)
        self.assertIn("Plotly.restyle", push_incremental_delta_source)
        self.assertIn("Plotly.update", push_incremental_delta_source)
        self.assertIn("Plotly.relayout", push_incremental_delta_source)
        self.assertIn("fieldWorkbenchRenderQueue", push_incremental_delta_source)
        self.assertNotIn("fieldWorkbenchLatestIncrementalUpdate", push_incremental_delta_source)

        groups = PlotView._incremental_trace_groups(
            {
                3: {"color": "#111111", "opacity": 0.3},
                4: {"color": "#222222", "opacity": 0.4},
                5: {"visible": False},
            }
        )
        self.assertEqual(len(groups), 2)
        style_group = next(
            group for group in groups if set(group["updates"]) == {"color", "opacity"}
        )
        self.assertEqual(style_group["indices"], [3, 4])
        self.assertEqual(style_group["updates"]["color"], ["#111111", "#222222"])
        self.assertEqual(style_group["updates"]["opacity"], [0.3, 0.4])
        view.cleanup()
        view.deleteLater()

    def test_plotly_camera_uses_native_save_as_and_custom_filename(self):
        view = PlotView(image_filename="2D field map")
        self.assertEqual(view._image_filename, "2D field map")

        update_source = _source_for(PlotView._push_pending_figure)
        self.assertIn('modebar_buttons_to_remove.append("toImage")', update_source)
        self.assertIn("Save plot as PNG…", update_source)
        self.assertIn("requestImageSave", update_source)
        self.assertIn("camera-retro", update_source)

        with tempfile.TemporaryDirectory() as directory:
            destination_without_suffix = Path(directory) / "my custom field plot"
            with mock.patch(
                "fieldworkbench.plot_view.QFileDialog.getSaveFileName",
                return_value=(str(destination_without_suffix), "PNG image (*.png)"),
            ), mock.patch.object(view, "request_png", return_value=True) as request_png:
                view._modebar_image_save_requested(640, 480)

            destination = destination_without_suffix.with_suffix(".png")
            self.assertEqual(view._pending_modebar_export_path, destination)
            request_png.assert_called_once_with(width=640, height=480, scale=2.0)

            view._bridge_image_ready("data:image/png;base64,aGVsbG8=")
            self.assertEqual(destination.read_bytes(), b"hello")
            self.assertIsNone(view._pending_modebar_export_path)

        view.cleanup()
        view.deleteLater()

    def test_plot_payload_preserves_stable_trace_uids(self):
        set_figure_source = _source_for(PlotView.set_figure)
        self.assertIn("remove_uids=False", set_figure_source)

        view = PlotView()
        if view.web is None:
            view.cleanup()
            self.skipTest("Qt WebEngine is unavailable in this test environment.")
        view._ready = False
        view.set_figure(
            {
                "data": [
                    {
                        "type": "mesh3d",
                        "x": [0, 1, 0],
                        "y": [0, 0, 1],
                        "z": [0, 0, 0],
                        "i": [0],
                        "j": [1],
                        "k": [2],
                        "uid": "field-workbench-scene-0123456789abcdef",
                    }
                ],
                "layout": {"scene": {}},
            }
        )
        payload = json.loads(view._pending_figure_json)
        self.assertEqual(
            payload["data"][0]["uid"],
            "field-workbench-scene-0123456789abcdef",
        )
        view.cleanup()
        view.deleteLater()

    def test_field_map_side_panels_are_width_responsive(self):
        two_d_source = _source_for(FieldMapDialog.__init__)
        three_d_source = _source_for(FieldVolumeMapDialog.__init__)

        for source in (two_d_source, three_d_source):
            self.assertIn("QSizePolicy.Policy.Ignored", source)
            self.assertIn("QFormLayout.RowWrapPolicy.WrapLongRows", source)
            self.assertIn("setChildrenCollapsible(False)", source)
            self.assertIn('CollapsibleSection("Sampling options", expanded=False)', source)
            self.assertIn('CollapsibleSection("Scene options", expanded=False)', source)
            self.assertIn('CollapsibleSection("Heatmap options", expanded=True)', source)
            self.assertIn('scene_options_layout.addWidget(self.show_grid)', source)
            self.assertIn('QPushButton("Static map")', source)
            self.assertIn('QPushButton("Animated map")', source)

        self.assertNotIn("setMaximumWidth(430)", two_d_source)
        self.assertNotIn("setMaximumWidth(450)", three_d_source)
        self.assertIn('CollapsibleSection("Map options", expanded=False)', two_d_source)
        self.assertIn('field_view_row_layout.addRow("Field view", self.display_mode)', two_d_source)
        self.assertLess(two_d_source.index("control_root.addWidget(field_view_row)"), two_d_source.index("control_root.addWidget(self.map_plane_section)"))
        self.assertLess(
            two_d_source.index("control_root.addWidget(self.sample_config_section)"),
            two_d_source.index("control_root.addWidget(self.scene_options_section)"),
        )
        self.assertLess(
            two_d_source.index("control_root.addWidget(self.scene_options_section)"),
            two_d_source.index("control_root.addWidget(self.field_view_options_section)"),
        )
        self.assertLess(
            two_d_source.index("control_root.addWidget(self.display_options_section)"),
            two_d_source.index("control_root.addWidget(self.fluxline_options_section)"),
        )
        self.assertIn('"Contour lines", expanded=True', two_d_source)
        self.assertIn('QCheckBox("Log scale")', two_d_source)
        self.assertIn("field_view_options_layout.setColumnStretch(1, 1)", two_d_source)
        self.assertIn("display_options_layout.setColumnStretch(1, 1)", two_d_source)
        self.assertIn('CollapsibleSection("Map options", expanded=False)', three_d_source)
        self.assertIn('self.view_type.addItem("Full Volume", "volume")', three_d_source)
        self.assertIn('region_layout.addWidget(QLabel("Render mode"), 4, 0)', three_d_source)
        self.assertIn('self.view_type.setCurrentIndex(self.view_type.findData("volume"))', three_d_source)
        self.assertIn('self.display_mode.addItem("Heatmap", "slices")', three_d_source)
        self.assertIn('self.display_mode.addItem("Heatmap + fluxlines", "both")', three_d_source)
        self.assertIn('field_view_row_layout.addRow("Field view", self.display_mode)', three_d_source)
        self.assertLess(three_d_source.index("control_root.addWidget(field_view_row)"), three_d_source.index("control_root.addWidget(self.map_volume_section)"))
        self.assertLess(
            three_d_source.index("control_root.addWidget(self.display_options_section)"),
            three_d_source.index("control_root.addWidget(self.fluxline_options_section)"),
        )
        self.assertNotIn('field_display_row_layout.addRow("Field display", self.display_mode)', three_d_source)
        for source in (two_d_source, three_d_source):
            sample_block = source.split('self.sample_config_section = CollapsibleSection("Sampling options", expanded=False)', 1)[1].split('self.sample_config_section.setContentLayout(sample_config_layout)', 1)[0]
            self.assertNotIn('addRow("Flux density", self.flux_density)', sample_block)
        fluxline_source = _source_for(_BackgroundFieldRenderMixin._create_fluxline_options_section)
        self.assertIn('form.addRow("Flux density", flux_density)', fluxline_source)
        self.assertIn('CollapsibleSection("Slice options", expanded=True)', three_d_source)
        self.assertNotIn("self.view_options_stack = QStackedWidget()", three_d_source)
        self.assertIn("region_layout.setColumnStretch(1, 1)", three_d_source)
        self.assertIn('QCheckBox("Intensity-linked opacity")', three_d_source)
        self.assertIn('QCheckBox("Log scale")', three_d_source)
        self.assertIn('QCheckBox("Show grid")', three_d_source)
        self.assertIn('self.show_grid.setChecked(True)', three_d_source)
        for source in (two_d_source, three_d_source):
            self.assertIn("EditorWheelScrollFilter(control_scroll)", source)
            self.assertIn("editor.installEventFilter(self._control_wheel_filter)", source)

    def test_field_map_selector_previews_debounce_before_composing_and_coalesce_render(self):
        two_d_request_source = _source_for(FieldMapDialog.preview_plane)
        two_d_render_source = _source_for(FieldMapDialog._render_preview_plane)
        three_d_request_source = _source_for(FieldVolumeMapDialog.preview_slice_volume)
        three_d_render_source = _source_for(FieldVolumeMapDialog._render_preview_slice_volume)
        two_d_init_source = _source_for(FieldMapDialog.__init__)
        three_d_init_source = _source_for(FieldVolumeMapDialog.__init__)
        set_figure_source = _source_for(PlotView.set_figure)
        update_source = _source_for(PlotView._push_pending_figure)

        self.assertIn("_selector_preview_timer.start()", two_d_request_source)
        self.assertNotIn("compose_figure", two_d_request_source)
        self.assertIn("_selector_preview_timer.start()", three_d_request_source)
        self.assertNotIn("compose_figure", three_d_request_source)
        self.assertIn("setSingleShot(True)", two_d_init_source)
        self.assertIn("SELECTOR_PREVIEW_DEBOUNCE_MS", two_d_init_source)
        self.assertIn("setSingleShot(True)", three_d_init_source)
        self.assertIn("SELECTOR_PREVIEW_DEBOUNCE_MS", three_d_init_source)
        self.assertIn("compose_figure", two_d_render_source)
        self.assertIn(
            "field_map_preview_scene",
            _source_for(FieldMapDialog._preview_scene_base_figure),
        )
        self.assertIn("include_surface_regions=False", two_d_render_source)
        self.assertIn("request_clean_rebuild_on_next_figure", two_d_render_source)
        self.assertIn("coalesce=True", two_d_render_source)
        self.assertIn("compose_figure", three_d_render_source)
        self.assertIn(
            "field_map_preview_scene",
            _source_for(FieldVolumeMapDialog._preview_scene_base_figure),
        )
        self.assertIn("include_surface_regions=False", three_d_render_source)
        self.assertIn("request_clean_rebuild_on_next_figure", three_d_render_source)
        self.assertIn("coalesce=True", three_d_render_source)
        self.assertIn("coalesce: bool = False", set_figure_source)
        self.assertIn("_pending_figure_coalesce = bool(coalesce)", set_figure_source)
        self.assertIn("fieldWorkbenchLatestCoalescedRenderId", update_source)
        self.assertIn("if (!renderStillCurrent()) return null", update_source)
        self.assertIn("if (!plot || !renderStillCurrent()) return plot", update_source)

    def test_plot_view_reports_nonmodal_render_activity_inside_the_view(self):
        page_source = _source_for(PlotView._load_persistent_page)
        update_source = _source_for(PlotView._push_pending_figure)

        self.assertIn('Preparing the viewer…', page_source)
        self.assertIn('id="loading-track"', page_source)
        self.assertIn('field-workbench-loading-sweep', page_source)
        self.assertIn('window.fieldWorkbenchBeginLoading', page_source)
        self.assertIn('window.fieldWorkbenchEndLoading', page_source)
        self.assertNotIn('id="loading-elapsed"', page_source)
        self.assertNotIn('id="loading-stage"', page_source)
        self.assertNotIn('fieldWorkbenchSetLoadingStage', page_source)
        self.assertNotIn('Queued for renderer', update_source)
        self.assertNotIn('Building scene', update_source)
        self.assertNotIn('Finalizing view controls', update_source)
        self.assertIn('window.fieldWorkbenchBeginLoading(renderRequestId)', update_source)
        self.assertIn('window.fieldWorkbenchEndLoading(renderRequestId)', update_source)
        self.assertNotIn('showRenderActivity', update_source)
        self.assertNotIn('QProgressDialog', page_source)
        self.assertNotIn('QProgressDialog', update_source)

    def test_structural_plot_update_uses_atomic_clean_rebuild(self):
        update_source = _source_for(PlotView._push_pending_figure)
        self.assertIn("Plotly.purge(plot)", update_source)
        self.assertIn("Plotly.newPlot", update_source)
        self.assertIn("Plotly.react", update_source)
        self.assertIn("const replacement = document.createElement('div')", update_source)
        self.assertIn("replacement.style.visibility = 'hidden'", update_source)
        self.assertIn("replacement.id = 'plot'", update_source)
        self.assertIn("window.fieldWorkbenchRenderQueue", update_source)
        self.assertIn(".then(renderFigure)", update_source)
        self.assertLess(
            update_source.index("Plotly.newPlot"),
            update_source.index("Plotly.purge(plot)"),
        )
        self.assertLess(
            update_source.index("replacement.id = 'plot'"),
            update_source.index("Plotly.purge(plot)"),
        )

        page_view = PlotView()
        if page_view.web is None:
            page_view.cleanup()
            self.skipTest("Qt WebEngine is unavailable in this test environment.")
        persistent_page = (page_view._temporary_directory / "plot-view.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("window.fieldWorkbenchRenderQueue = Promise.resolve()", persistent_page)
        self.assertIn('class="plot-surface"', persistent_page)
        page_view.cleanup()
        page_view.deleteLater()

        view = PlotView()
        view._home_on_next_figure = False
        view._clean_rebuild_on_next_figure = False
        view.request_home_on_next_figure()
        self.assertTrue(view._home_on_next_figure)
        self.assertTrue(view._clean_rebuild_on_next_figure)
        view._home_on_next_figure = False
        view.request_fit_bounds_on_next_figure()
        self.assertTrue(view._fit_bounds_on_next_figure)
        update_source = _source_for(PlotView._push_pending_figure)
        self.assertIn("applyBoundsFit", update_source)
        self.assertIn("currentView['scene.camera']", update_source)
        view.cleanup()
        view.deleteLater()

    def test_background_field_add_flow_creates_scene_field_without_geometry_refit(self):
        window = self.make_window()
        home_requests = []
        window.plot.request_home_on_next_figure = lambda: home_requests.append(True)
        dialog = mock.Mock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted
        dialog.field_definition.return_value = {
            "mode": "manual",
            "vector_uT": [12.5, -3.0, 41.25],
            "source": {"kind": "manual"},
        }
        with mock.patch.object(
            main_window_module, "BackgroundFieldDialog", return_value=dialog
        ):
            window.palette_buttons["Background"].click()

        self.assertEqual(home_requests, [])
        self.assertTrue(window.adapter.is_background_field(window.selected_id))
        properties = window.adapter.background_field_properties(window.selected_id)
        self.assertEqual(properties["vector_uT"], [12.5, -3.0, 41.25])
        self.assertEqual(window._tree_items[window.selected_id].text(1), "Background field")
        section_titles = [
            section.header.text()
            for section in window.inspector_container.findChildren(CollapsibleSection)
        ]
        self.assertIn("Uniform static field", section_titles)
        self.assertIn("Position", section_titles)
        window.plot.cleanup()
        window.deleteLater()

    def test_background_field_wmm_location_defaults_to_sudbury(self):
        dialog = BackgroundFieldDialog()
        self.assertAlmostEqual(dialog.latitude.value(), 46.4939, places=4)
        self.assertAlmostEqual(dialog.longitude.value(), -80.9954, places=4)
        dialog.close()
        dialog.deleteLater()

    def test_structural_add_requests_a_complete_scene_refit(self):
        window = self.make_window()
        requests = []
        window.plot.request_home_on_next_figure = lambda: requests.append(True)
        window.add_object("circle")
        self.assertEqual(requests, [True])
        self.assertEqual(window.adapter.get_object(window.selected_id)["type"], "current.Circle")
        window.plot.cleanup()
        window.deleteLater()

    def test_transform_apply_requests_bounds_refit_without_home_reset(self):
        window = self.make_window()
        requests = []
        home_requests = []
        window.plot.request_fit_bounds_on_next_figure = lambda: requests.append(True)
        window.plot.request_home_on_next_figure = lambda: home_requests.append(True)

        window.position_spins[1].setValue(-140.0)
        self.assertTrue(window.apply_inspector())
        self.assertEqual(requests, [True])
        self.assertEqual(home_requests, [])
        self.assertAlmostEqual(
            window.adapter.get_transform(window.selected_id)["position"][1],
            -0.14,
        )

        requests.clear()
        self.assertTrue(window.apply_inspector())
        self.assertEqual(requests, [])
        window.plot.cleanup()
        window.deleteLater()

    def test_geometry_parameter_apply_refits_changed_scene_bounds_without_home_reset(self):
        window = self.make_window()
        requests = []
        home_requests = []
        window.plot.request_fit_bounds_on_next_figure = lambda: requests.append(True)
        window.plot.request_home_on_next_figure = lambda: home_requests.append(True)

        diameter = window.parameter_editors["diameter"]["widget"]
        diameter.setValue(diameter.value() * 2.0)
        self.assertTrue(window.apply_inspector())
        self.assertEqual(requests, [True])
        self.assertEqual(home_requests, [])

        # Re-applying identical geometry should not disturb a manually chosen zoom.
        requests.clear()
        self.assertTrue(window.apply_inspector())
        self.assertEqual(requests, [])

        window.plot.cleanup()
        window.deleteLater()

    def test_coil_construction_wraps_hint_outside_the_form(self):
        window = self.make_window()
        self.application.processEvents()

        construction = next(
            section
            for section in window.inspector_container.findChildren(CollapsibleSection)
            if section.header.text() == "Coil Construction"
        )
        self.assertIsInstance(construction.content.layout(), QVBoxLayout)
        outer_layout = construction.content.layout()
        self.assertIsInstance(outer_layout.itemAt(0).layout(), QFormLayout)

        hint = next(
            label
            for label in construction.findChildren(QLabel)
            if label.text().startswith("Wire data drives resistance")
        )
        apply_button = next(
            button
            for button in construction.findChildren(QPushButton)
            if button.objectName() == "inlineApplyButton"
        )
        self.assertGreater(outer_layout.indexOf(hint), 0)
        self.assertGreater(outer_layout.indexOf(apply_button), outer_layout.indexOf(hint))
        self.assertTrue(hint.wordWrap())

        window.plot.cleanup()
        window.deleteLater()

    def test_circular_coil_inspector_separates_magnetics_from_optional_physical_sections(self):
        window = self.make_window()
        self.application.processEvents()

        sections = {
            section.header.text(): section
            for section in window.inspector_container.findChildren(CollapsibleSection)
        }
        for title in ("Position", "Magnetics", "Coil Construction", "Assembly", "Estimates"):
            self.assertIn(title, sections)
        self.assertFalse(sections["Magnetics"].content.isHidden())
        self.assertTrue(sections["Coil Construction"].content.isHidden())
        self.assertTrue(sections["Assembly"].content.isHidden())
        self.assertTrue(sections["Estimates"].content.isHidden())
        self.assertIn("diameter", window.parameter_editors)
        self.assertIn("turns", window.coil_widgets)
        self.assertIn("drive_current_a", window.coil_widgets)
        self.assertIn("current_mode", window.coil_widgets)
        self.assertIn("field_scale_factor", window.coil_widgets)
        self.assertIsInstance(window.coil_envelope_group, QCheckBox)

        window.plot.cleanup()
        window.deleteLater()

    def test_racetrack_inspector_uses_parametric_geometry_instead_of_vertices(self):
        window = self.make_window()
        window.add_object("racetrack")
        self.application.processEvents()

        self.assertEqual(
            window.adapter.get_object(window.selected_id)["type"], "current.Polyline"
        )
        self.assertEqual(
            window._tree_items[window.selected_id].text(1), "Racetrack coil"
        )
        self.assertNotIn("vertices", window.parameter_editors)
        self.assertIn("racetrack_end_diameter_mm", window.coil_widgets)
        self.assertIn("racetrack_straight_length_mm", window.coil_widgets)
        self.assertIn("racetrack_arc_segments", window.coil_widgets)

        sections = {
            section.header.text(): section
            for section in window.inspector_container.findChildren(CollapsibleSection)
        }
        self.assertIn("Magnetics", sections)
        self.assertIn("Coil Construction", sections)
        self.assertIn("Assembly", sections)
        self.assertIn("Estimates", sections)
        self.assertNotIn("Racetrack geometry", sections)
        self.assertNotIn("Physical parameters", sections)
        self.assertLess(
            window.inspector_layout.indexOf(sections["Magnetics"]),
            window.inspector_layout.indexOf(sections["Coil Construction"]),
        )
        self.assertLess(
            window.inspector_layout.indexOf(sections["Coil Construction"]),
            window.inspector_layout.indexOf(sections["Assembly"]),
        )
        self.assertLess(
            window.inspector_layout.indexOf(sections["Assembly"]),
            window.inspector_layout.indexOf(sections["Estimates"]),
        )

        window.coil_widgets["racetrack_end_diameter_mm"].setValue(120.0)
        window.coil_widgets["racetrack_straight_length_mm"].setValue(80.0)
        window.coil_widgets["racetrack_arc_segments"].setValue(24)
        self.assertTrue(window.apply_inspector())

        properties = window.adapter.coil_physical_properties(window.selected_id)
        self.assertAlmostEqual(properties["racetrack_end_diameter_mm"], 120.0)
        self.assertAlmostEqual(properties["racetrack_straight_length_mm"], 80.0)
        self.assertEqual(properties["racetrack_arc_segments"], 24)
        vertices = window.adapter._studio_param_value(
            window.selected_id, "vertices", []
        )
        self.assertEqual(len(vertices), 51)
        self.assertAlmostEqual(
            properties["calculations"]["mean_turn_length_m"],
            (2.0 * 80.0 + 3.141592653589793 * 120.0) / 1000.0,
        )

        window.plot.cleanup()
        window.deleteLater()

    def test_square_coil_exposes_shared_visual_and_drc_assembly(self):
        window = self.make_window()
        window.add_object("polyline")
        self.application.processEvents()

        self.assertEqual(
            window.adapter.get_object(window.selected_id)["type"],
            "current.Polyline",
        )
        self.assertIsNotNone(window.coil_envelope_group)
        self.assertIn("winding_radial_build_mm", window.coil_widgets)
        self.assertIn("assembly_bbox", window.coil_estimate_labels)
        section_titles = [
            section.header.text()
            for section in window.inspector_container.findChildren(CollapsibleSection)
        ]
        self.assertIn("Assembly", section_titles)

        window.coil_envelope_group.setChecked(True)
        mode = window.coil_widgets["winding_radial_build_mode"]
        mode.setCurrentIndex(mode.findData("manual"))
        window.coil_widgets["winding_radial_build_mm"].setValue(6.0)
        window.coil_widgets["winding_axial_width_mm"].setValue(8.0)
        window.coil_widgets["bobbin_wall_thickness_mm"].setValue(2.0)
        window.coil_widgets["flange_height_mm"].setValue(8.0)
        window.coil_widgets["flange_thickness_mm"].setValue(2.0)
        self.assertTrue(window.apply_inspector())

        assembly = window.adapter.coil_physical_properties(
            window.selected_id
        )["calculations"]["assembly"]
        self.assertTrue(assembly["valid"])
        self.assertEqual(assembly["shape"], "planar_path_band")
        self.assertIn("Ready for visual + DRC", window.coil_estimate_labels["assembly_status"].text())

        window.plot.cleanup()
        window.deleteLater()

    def test_analyze_volume_requires_measurement_selection_and_opens_all_selected(self):
        window = self.make_window()

        # The command remains available with no selection so it can give a
        # useful instruction instead of silently disabling itself.
        window.selected_ids = []
        window.selected_id = None
        window.populate_inspector(None)
        window._update_actions()
        self.assertTrue(window.measurement_action.isEnabled())

        messages = []
        original_message_box = main_window_module.QMessageBox

        class RecordingMessageBox:
            @staticmethod
            def information(_parent, _title, message, *args, **kwargs):
                messages.append(message)

        main_window_module.QMessageBox = RecordingMessageBox
        try:
            window.analyze_selected_measurement()
        finally:
            main_window_module.QMessageBox = original_message_box
        self.assertEqual(messages, ["Please select one or more Measurement volumes."])

        volume_ids = []
        for index in range(2):
            object_id = window.adapter.add_template("guide_box")
            window.adapter.apply_operations(
                [
                    {
                        "method": "set_geometry_style",
                        "params": {"object_id": object_id, "role": "measurement"},
                    }
                ]
            )
            window.adapter.get_object(object_id)["label"] = f"Test volume {index + 1}"
            volume_ids.append(object_id)

        window.selected_ids = list(volume_ids)
        window.selected_id = volume_ids[-1]
        window.populate_inspector(window.selected_id)
        window._update_actions()

        analyzed = []
        opened = []

        def fake_analyze(object_id, *, progress_callback=None):
            analyzed.append(object_id)
            if progress_callback is not None:
                progress_callback(
                    {
                        "completed": 1,
                        "total": 1,
                        "stage": f"Volume analysis — {object_id}",
                    }
                )
            return {
                "object_id": object_id,
                "label": window.adapter.get_object(object_id)["label"],
                "stats": {"mean_uT": 10.0, "in_band_pct": 100.0},
            }

        window.adapter.analyze_measurement = fake_analyze
        window._open_measurement_result_window = (
            lambda result, *, allow_snapshot: opened.append(
                (result["object_id"], allow_snapshot)
            )
        )

        window.analyze_selected_measurement()
        self.assertEqual(analyzed, volume_ids)
        self.assertEqual(opened, [(volume_ids[0], True), (volume_ids[1], True)])
        self.assertIn("Analyzed 2 of 2", window.statusBar().currentMessage())

        window.plot.cleanup()
        window.deleteLater()

    def test_existing_box_can_become_a_measurement_volume(self):
        window = self.make_window()
        box_id = window.adapter.add_template("guide_box")
        window.refresh_scene(select_id=box_id, refresh_plot=False)
        self.assertIsNotNone(window.measurement_box)
        self.assertFalse(window.measurement_box.isEnabled())
        measurement_index = window.geometry_role_combo.findData("measurement")
        self.assertGreaterEqual(measurement_index, 0)
        window.geometry_role_combo.setCurrentIndex(measurement_index)
        self.assertTrue(window.measurement_box.isEnabled())
        window.measurement_quality_combo.setCurrentIndex(
            window.measurement_quality_combo.findData("preview")
        )
        window.measurement_target_spin.setValue(123.0)
        window.measurement_tolerance_spin.setValue(7.5)
        self.assertTrue(window.apply_inspector())
        self.assertEqual(window.adapter.get_object(box_id)["role"], "measurement")
        settings = window.adapter.measurement_properties(box_id)
        self.assertEqual(settings["quality"], "preview")
        self.assertAlmostEqual(settings["target_uT"], 123.0)
        self.assertAlmostEqual(settings["tolerance_pct"], 7.5)
        window.plot.cleanup()
        window.deleteLater()

    def test_measurement_user_defined_points_editor_applies_local_coordinates(self):
        window = self.make_window()
        box_id = window.adapter.add_template("guide_box")
        window.refresh_scene(select_id=box_id, refresh_plot=False)
        self.assertTrue(window.measurement_defined_points_edit.isHidden())

        measurement_index = window.geometry_role_combo.findData("measurement")
        window.geometry_role_combo.setCurrentIndex(measurement_index)
        defined_index = window.measurement_quality_combo.findData("user_defined")
        self.assertGreaterEqual(defined_index, 0)
        window.measurement_quality_combo.setCurrentIndex(defined_index)
        self.assertFalse(window.measurement_defined_points_edit.isHidden())
        window.measurement_defined_points_edit.setPlainText(
            "[0, 0, 0],\n[9, 0, 0],\n[18, 0, 0]"
        )
        self.assertTrue(window.apply_inspector())

        settings = window.adapter.measurement_properties(box_id)
        self.assertEqual(settings["sampling_mode"], "user_defined")
        self.assertEqual(
            settings["defined_points_mm"],
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [18.0, 0.0, 0.0]],
        )
        sample = window.adapter.measurement_sample_points(box_id)
        self.assertEqual(sample["quality"], "user_defined")
        self.assertEqual(len(sample["local_points_m"]), 3)

        window.plot.cleanup()
        window.deleteLater()

    def test_measurement_defined_point_text_parser_accepts_common_forms(self):
        expected = [[0.0, 0.0, 0.0], [9.0, 1.0, -2.0]]
        self.assertEqual(
            MainWindow._parse_defined_measurement_points(
                "[0, 0, 0], [9, 1, -2]"
            ),
            expected,
        )
        self.assertEqual(
            MainWindow._parse_defined_measurement_points(
                "[[0, 0, 0], [9, 1, -2]]"
            ),
            expected,
        )
        self.assertEqual(
            MainWindow._parse_defined_measurement_points("[0, 0, 0]"),
            [[0.0, 0.0, 0.0]],
        )
        self.assertEqual(
            MainWindow._parse_defined_measurement_points(
                "[0, 0, 0]\n[9, 1, -2]"
            ),
            expected,
        )

    def test_closed_and_broken_meshes_show_consistent_role_eligibility(self):
        window = self.make_window()
        vertices, faces = window.adapter._mesh_local_vertices_faces(
            {"mesh": {"kind": "box", "dimension": [0.04, 0.04, 0.04]}}
        )
        for object_id, selected_faces in (
            ("closed_target", faces),
            ("open_surface", faces[:-2]),
        ):
            window.adapter.geometry.append(
                {
                    "id": object_id,
                    "type": "workbench.Mesh",
                    "label": object_id.replace("_", " ").title(),
                    "visible": True,
                    "parent": None,
                    "position": [0.0, 0.0, 0.0],
                    "euler": [0.0, 0.0, 0.0],
                    "scale_xyz": [1.0, 1.0, 1.0],
                    "role": "visual",
                    "colour": "#64748b",
                    "opacity": 0.3,
                    "source": f"{object_id}.obj",
                    "mesh": {
                        "kind": "triangles",
                        "vertices": vertices.tolist(),
                        "faces": selected_faces.tolist(),
                    },
                }
            )

        window.refresh_scene(select_id="closed_target", refresh_plot=False)
        self.assertGreaterEqual(window.geometry_role_combo.findData("measurement"), 0)
        self.assertEqual(window.geometry_role_combo.findData("test"), -1)
        self.assertIsNotNone(window.measurement_box)
        self.assertIn(
            "optimizer targeting",
            window.adapter.geometry_capabilities("closed_target")["measurement_reason"],
        )

        window.refresh_scene(select_id="open_surface", refresh_plot=False)
        self.assertEqual(window.geometry_role_combo.findData("measurement"), -1)
        self.assertIsNone(window.measurement_box)
        self.assertTrue(
            window.adapter.geometry_capabilities("open_surface")[
                "optimizer_placement_surface"
            ]
        )
        window.plot.cleanup()
        window.deleteLater()

    def test_drc_dialog_streams_completed_rows_during_progress(self):
        window = self.make_window()
        dialog = DrcResultsDialog(window.adapter, window)
        streamed_result = {
            "status": "below_clearance",
            "object_a_id": "coil_a",
            "object_a_label": "Lower coil",
            "object_b_id": "guide",
            "object_b_label": "Guide",
            "signed_clearance_mm": 1.25,
            "required_clearance_mm": 3.0,
            "rule": "Centreline precheck — Region clearance (inherited)",
            "reason": "Streaming test result.",
            "conservative_lower_bound": False,
            "approximate": False,
        }
        dialog._progress(
            {
                "phase": "drc",
                "percent": 42,
                "message": "Checked Lower coil against Guide…",
                "result": streamed_result,
            }
        )
        self.assertEqual(dialog.progress_bar.value(), 42)
        self.assertEqual(dialog.results.topLevelItemCount(), 1)
        item = dialog.results.topLevelItem(0)
        self.assertEqual(item.text(0), "Below clearance")
        self.assertEqual(item.text(1), "Lower coil")
        self.assertEqual(item.text(2), "Guide")
        self.assertIn("Running", dialog.summary.text())
        self.assertIn("1 problem", dialog.summary.text())
        dialog.close()
        window.plot.cleanup()
        window.deleteLater()

    def test_geometry_drc_checkbox_defaults_by_role_and_gates_clearance(self):
        window = self.make_window()
        measurement_id = window.adapter.add_template("guide_box")
        window.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": measurement_id, "role": "measurement"},
                }
            ]
        )
        window.refresh_scene(select_id=measurement_id, refresh_plot=False)

        self.assertTrue(window.geometry_drc_enabled_check.isEnabled())
        self.assertFalse(window.geometry_drc_enabled_check.isChecked())
        self.assertFalse(window.geometry_clearance_override_check.isEnabled())
        self.assertFalse(window.geometry_clearance_spin.isEnabled())

        window.geometry_drc_enabled_check.setChecked(True)
        self.assertTrue(window.geometry_clearance_override_check.isEnabled())
        window.geometry_clearance_override_check.setChecked(True)
        window.geometry_clearance_spin.setValue(6.25)
        self.assertTrue(window.geometry_clearance_spin.isEnabled())
        self.assertTrue(window.apply_inspector())

        properties = window.adapter.geometry_properties(measurement_id)
        self.assertTrue(properties["drc_enabled"])
        self.assertAlmostEqual(properties["drc_clearance_override_mm"], 6.25)

        window.geometry_drc_enabled_check.setChecked(False)
        self.assertFalse(window.geometry_clearance_override_check.isEnabled())
        self.assertFalse(window.geometry_clearance_spin.isEnabled())
        self.assertTrue(window.apply_inspector())
        properties = window.adapter.geometry_properties(measurement_id)
        self.assertFalse(properties["drc_enabled"])
        self.assertAlmostEqual(properties["drc_clearance_override_mm"], 6.25)

        exclusion_id = window.adapter.add_template("guide_box")
        window.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": exclusion_id, "role": "exclusion"},
                }
            ]
        )
        window.refresh_scene(select_id=exclusion_id, refresh_plot=False)
        self.assertTrue(window.geometry_drc_enabled_check.isEnabled())
        self.assertTrue(window.geometry_drc_enabled_check.isChecked())
        window.geometry_drc_enabled_check.setChecked(False)
        self.assertTrue(window.apply_inspector())
        self.assertFalse(window.adapter.geometry_properties(exclusion_id)["drc_enabled"])

        window.plot.cleanup()
        window.deleteLater()

    def test_drc_override_results_selection_and_stale_state(self):
        window = self.make_window()
        window.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": "coil_a",
                        "physical_geometry_enabled": True,
                        "winding_radial_build_mode": "manual",
                        "winding_radial_build_mm": 10.0,
                        "winding_axial_width_mm": 10.0,
                        "bobbin_wall_thickness_mm": 2.0,
                        "flange_height_mm": 15.0,
                        "flange_thickness_mm": 2.0,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": "coil_a",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
            ]
        )
        box_id = window.adapter.add_template("guide_box")
        window.refresh_scene(select_id=box_id, refresh_plot=False)
        self.assertIsInstance(window.geometry_drc_enabled_check, QCheckBox)
        self.assertIsInstance(window.geometry_clearance_override_check, QCheckBox)
        self.assertFalse(window.geometry_drc_enabled_check.isEnabled())
        self.assertFalse(window.geometry_clearance_override_check.isEnabled())
        window.geometry_role_combo.setCurrentIndex(
            window.geometry_role_combo.findData("exclusion")
        )
        self.assertTrue(window.geometry_drc_enabled_check.isEnabled())
        self.assertTrue(window.geometry_drc_enabled_check.isChecked())
        self.assertTrue(window.geometry_clearance_override_check.isEnabled())
        window.geometry_clearance_override_check.setChecked(True)
        self.assertTrue(window.geometry_clearance_spin.isEnabled())
        window.geometry_clearance_spin.setValue(3.5)
        self.assertTrue(window.apply_inspector())
        self.assertAlmostEqual(
            window.adapter.geometry_properties(box_id)[
                "drc_clearance_override_mm"
            ],
            3.5,
        )

        window.open_drc()
        self.application.processEvents()
        self.assertIsInstance(window.drc_dialog, DrcResultsDialog)
        self.assertIsNone(window.drc_dialog.report)
        self.assertEqual(window.drc_dialog.progress_bar.value(), 0)
        self.assertIn("press Check design", window.drc_dialog.progress_bar.format())
        self.assertAlmostEqual(window.drc_dialog.mesh_tolerance.value(), 0.1)
        window.drc_dialog.mesh_tolerance.setValue(0.25)
        window.drc_dialog.run_checks()
        self.assertIsNotNone(window.drc_dialog.thread)
        deadline = time.monotonic() + 15.0
        while window.drc_dialog.thread is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.assertIsNone(window.drc_dialog.thread)
        self.assertIsNotNone(window.drc_dialog.report)
        self.assertEqual(window.drc_dialog.progress_bar.value(), 100)
        self.assertIn("Design-rule check complete", window.drc_dialog.progress_bar.format())
        self.assertIn("elapsed", window.drc_dialog.progress_bar.format())
        self.assertIsNotNone(window.adapter.fresh_drc_report())
        self.assertAlmostEqual(
            window.adapter.get_drc_settings()["mesh_tolerance_mm"], 0.25
        )
        pair_item = None
        for index in range(window.drc_dialog.results.topLevelItemCount()):
            item = window.drc_dialog.results.topLevelItem(index)
            result = item.data(0, Qt.ItemDataRole.UserRole)
            if {result.get("object_a_id"), result.get("object_b_id")} == {
                "coil_a",
                box_id,
            }:
                pair_item = item
                break
        self.assertIsNotNone(pair_item)
        window.drc_dialog.results.setCurrentItem(pair_item)
        self.application.processEvents()
        self.assertEqual(set(window.selected_ids), {"coil_a", box_id})

        window.adapter.apply_operations(
            [
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": box_id,
                        "position": [0.01, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                }
            ]
        )
        window.refresh_scene(select_id=box_id, refresh_plot=False)
        self.assertIn("Results stale", window.drc_dialog.summary.text())
        window.drc_dialog.close()
        window.plot.cleanup()
        window.deleteLater()

    def test_exposure_analysis_cache_manager_selectively_removes_derived_results(self):
        window = self.make_window()
        payload = {
            "times_b64": "AAAA",
            "currents_b64": "BBBB",
            "shape": [2, 1],
            "drive_ids": ["coil_a"],
            "regions": {
                "whole": {
                    "name": "Whole brain",
                    "constant": 0.0,
                    "linear": [0.0],
                    "quadratic": [[1.0]],
                },
            },
            "exposure_analysis_context": {
                "exposure_id": "missing_exp",
                "exposure_name": "SAM360",
                "target_id": "missing_brain",
                "target_name": "SRI24 brain",
                "target_kind": "brain_lpba40",
                "atlas": "LPBA40",
                "sample_count": 100,
                "physical_duration_s": 10.0,
                "dependency_signature": "cache-manager-test",
            },
        }
        summary_key = ("summary", "__all_regions__")
        weighted_key = ("weighted", "__all_regions__", "icnirp2010_public", None)
        cache = {
            summary_key: {"regions": {"whole": {"duration_s": 10.0}}},
            ("__cache_meta__",) + summary_key: {
                "cached_at": "2026-09-25T20:00:00Z",
                "processing_time_s": 1.2,
            },
            weighted_key: {"regions": {"whole": {"profile_label": "ICNIRP public"}}},
            ("__cache_meta__",) + weighted_key: {
                "cached_at": "2026-09-25T20:01:00Z",
                "processing_time_s": 65.0,
            },
        }
        analysis_id = window.adapter.save_exposure_analysis_resource(
            payload, name="SAM360 cache", analysis_cache=cache
        )
        preparation_blob = window.adapter.exposure_analyses[0]["payload_b64"]
        changed = []
        dialog = ExposureAnalysisCacheManagerDialog(
            window.adapter, analysis_id, window, on_changed=lambda: changed.append(True)
        )
        self.assertEqual(dialog.cache_tree.topLevelItemCount(), 2)
        self.assertIn("Preparation is kept", dialog.preparation_label.text())
        parents = {
            dialog.cache_tree.topLevelItem(index).text(0):
            dialog.cache_tree.topLevelItem(index)
            for index in range(dialog.cache_tree.topLevelItemCount())
        }
        self.assertIn("Exposure Summary", parents)
        self.assertIn("Frequency weighted", parents)
        self.assertEqual(parents["Exposure Summary"].childCount(), 1)
        self.assertIn("ICNIRP 2010", parents["Frequency weighted"].child(0).text(1))
        self.assertFalse(dialog.remove_selected_button.isEnabled())
        parents["Frequency weighted"].setCheckState(0, Qt.CheckState.Checked)
        self.application.processEvents()
        self.assertEqual(
            parents["Frequency weighted"].child(0).checkState(0),
            Qt.CheckState.Checked,
        )
        self.assertTrue(dialog.remove_selected_button.isEnabled())
        dialog.clear_selection()
        self.assertFalse(dialog.remove_selected_button.isEnabled())

        first = parents["Exposure Summary"].child(0)
        first_key = first.data(0, Qt.ItemDataRole.UserRole)
        first.setCheckState(0, Qt.CheckState.Checked)
        self.application.processEvents()
        self.assertTrue(dialog.remove_selected_button.isEnabled())
        before_cache_bytes = int(dialog._record["cache_compressed_bytes"])
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ):
            dialog.remove_selected()
        self.assertEqual(changed, [True])
        after_cache_bytes = int(dialog._record["cache_compressed_bytes"])
        self.assertLess(after_cache_bytes, before_cache_bytes)
        self.assertFalse(dialog.reclaimed_label.isHidden())
        self.assertIn("Last removal reclaimed", dialog.reclaimed_label.text())
        self.assertIn("derived cache", dialog.reclaimed_label.text())
        self.assertEqual(
            dialog._last_reclaimed_storage, (before_cache_bytes, after_cache_bytes)
        )
        self.assertEqual(dialog.cache_tree.topLevelItemCount(), 1)
        self.assertEqual(dialog.cache_tree.topLevelItem(0).text(0), "Frequency weighted")
        remaining = window.adapter.exposure_analysis(analysis_id)["analysis_cache"]
        self.assertNotIn(first_key, remaining)
        self.assertNotIn(("__cache_meta__",) + first_key, remaining)
        self.assertEqual(window.adapter.exposure_analyses[0]["payload_b64"], preparation_blob)

    def test_snapshot_manager_lists_a_saved_result(self):
        window = self.make_window()
        box_id = window.adapter.add_template("guide_box")
        window.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": box_id,
                        "role": "measurement",
                        "colour": "#7c3aed",
                        "opacity": 0.28,
                    },
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box_id,
                        "quality": "preview",
                        "target_uT": 0.6,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        result = window.adapter.analyze_measurement(box_id)
        snapshot_id = window.adapter.create_snapshot(result, "Test result")
        second_snapshot_id = window.adapter.create_snapshot(result, "Test result 2")
        dialog = SnapshotManagerDialog(window.adapter, window)
        self.assertIn("QLabel#resultValue", QApplication.instance().styleSheet())
        self.assertIn("QMainWindow, QDialog { background: #f5f7fa; }", QApplication.instance().styleSheet())
        self.assertEqual(dialog.refresh_button.text(), "Refresh")
        self.assertTrue(dialog._deferred_refresh_timer.isSingleShot())
        self.assertFalse(dialog._deferred_refresh_timer.isActive())
        self.assertEqual(dialog.import_button.text(), "Import from scene…")
        self.assertEqual(dialog.tree.topLevelItemCount(), 2)
        self.assertEqual(dialog.result_tabs.tabText(0), "Static maps (0)")
        self.assertEqual(dialog.result_tabs.tabText(1), "Scripted maps (0)")
        self.assertEqual(dialog.result_tabs.tabText(2), "Exposure analyses (0)")
        self.assertEqual(dialog.result_tabs.tabText(3), "Snapshots (2)")
        dialog._refresh_exposure_analysis_tree(
            [{
                "id": "analysis_test",
                "name": "SAM360 — SRI24 brain",
                "exposure_name": "SAM360",
                "target_name": "SRI24 brain",
                "status": "current",
                "updated_at": "2026-09-21T20:00:00Z",
                "cached_result_count": 4,
                "cached_region_count": 2,
                "cached_tool_counts": {"summary": 1, "spectrum": 2, "threshold": 1},
                "preparation_processing_time_s": 20.5,
                "sample_count": 50400,
                "physical_duration_s": 2400.0,
                "payload_bytes": 2_000_000,
                "compressed_bytes": 500_000,
            }],
            preserve_selection=False,
            select_analysis_id="analysis_test",
        )
        analysis_item = dialog.exposure_analysis_tree.topLevelItem(0)
        self.assertEqual(analysis_item.text(3), "Current")
        self.assertEqual(analysis_item.text(4), "3 tools • 4 configs")
        self.assertIn("Summary", dialog.exposure_analysis_details.text())
        self.assertIn("50,400 LPBA40 samples", dialog.exposure_analysis_details.text())
        self.assertEqual(dialog.exposure_analysis_manage_cache_button.text(), "Manage cache…")
        self.assertTrue(dialog.exposure_analysis_manage_cache_button.isEnabled())
        self.assertEqual(dialog.exposure_analysis_delete_button.text(), "Delete cache")
        self.assertEqual(dialog.tree.topLevelItem(0).text(0), "Test result")
        self.assertEqual(dialog.tree.topLevelItem(0).text(2), "Current scene")
        self.assertEqual(dialog.tree.topLevelItem(0).text(3), "Current")
        self.assertEqual(dialog.selected_snapshot_id(), snapshot_id)
        self.assertFalse(dialog.compare_button.isEnabled())
        self.assertEqual(dialog.view_button.objectName(), "primaryButton")
        self.assertEqual(dialog.compare_button.objectName(), "accentButton")
        dialog.tree.topLevelItem(1).setSelected(True)
        self.application.processEvents()
        self.assertEqual(
            set(dialog.selected_snapshot_ids()),
            {snapshot_id, second_snapshot_id},
        )
        self.assertTrue(dialog.view_button.isEnabled())
        self.assertTrue(dialog.compare_button.isEnabled())
        self.assertFalse(dialog.rename_button.isEnabled())
        self.assertTrue(dialog.delete_button.isEnabled())
        dialog.compare_selected()
        self.application.processEvents()
        comparison_dialog = next(
            child
            for child in dialog._result_windows.values()
            if isinstance(child, SnapshotComparisonDialog)
        )
        self.assertFalse(comparison_dialog.isModal())
        self.assertIsNone(comparison_dialog.parentWidget())
        self.assertFalse(
            bool(comparison_dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        )
        self.assert_standard_window_controls(comparison_dialog)
        self.assertTrue(dialog.isEnabled())

        dialog.tree.clearSelection()
        first_item = dialog.tree.topLevelItem(0)
        first_item.setSelected(True)
        dialog.tree.setCurrentItem(first_item)
        dialog.view_selected()
        self.application.processEvents()
        result_dialog = next(
            child
            for child in dialog._result_windows.values()
            if isinstance(child, MeasurementResultsDialog)
        )
        self.assertFalse(result_dialog.isModal())
        self.assertIsNone(result_dialog.parentWidget())
        self.assertFalse(
            bool(result_dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        )
        self.assert_standard_window_controls(result_dialog)
        self.assertGreaterEqual(result_dialog.object_preview.minimumHeight(), 190)
        comparison_copy_buttons = [
            button
            for button in comparison_dialog.findChildren(QPushButton)
            if button.objectName() == "copyStatisticsSectionButton"
        ]
        self.assertGreaterEqual(len(comparison_copy_buttons), 2)
        comparison_copy_buttons[-1].click()
        self.assertIn("Target coverage", QApplication.clipboard().text())
        comparison_dialog.close()
        result_dialog.close()
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_snapshot_manager_views_and_deletes_arbitrary_selection(self):
        window = self.make_window()
        box_id = window.adapter.add_template("guide_box")
        window.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box_id, "role": "measurement"},
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box_id,
                        "quality": "preview",
                        "target_uT": 0.6,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        result = window.adapter.analyze_measurement(box_id)
        snapshot_ids = [
            window.adapter.create_snapshot(result, f"Batch {index}")
            for index in range(1, 4)
        ]
        opened: list[str] = []
        changed: list[bool] = []
        dialog = SnapshotManagerDialog(
            window.adapter,
            window,
            on_open_result=lambda item: opened.append(str(item["snapshot_id"])),
            on_changed=lambda: changed.append(True),
        )
        dialog.tree.clearSelection()
        for index in range(dialog.tree.topLevelItemCount()):
            dialog.tree.topLevelItem(index).setSelected(True)
        self.application.processEvents()

        self.assertEqual(set(dialog.selected_snapshot_ids()), set(snapshot_ids))
        self.assertTrue(dialog.view_button.isEnabled())
        self.assertTrue(dialog.delete_button.isEnabled())
        self.assertTrue(dialog.compare_button.isEnabled())
        dialog.view_selected()
        self.assertEqual(set(opened), set(snapshot_ids))

        with mock.patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            dialog.delete_selected()
        self.assertEqual(window.adapter.list_snapshots(), [])
        self.assertEqual(changed, [True])
        self.assertEqual(dialog.tree.topLevelItemCount(), 0)
        self.assertEqual(dialog.result_tabs.tabText(3), "Snapshots (0)")
        window.adapter.undo()
        self.assertEqual(
            {str(item["id"]) for item in window.adapter.list_snapshots()},
            set(snapshot_ids),
        )

        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_project_snapshot_manager_is_project_owned_and_non_modal(self):
        adapter = StudioAdapter(compact_workflow_scene())
        box_id = adapter.add_template("guide_box")
        adapter.apply_operations(
            [
                {"method": "set_geometry_style", "params": {"object_id": box_id, "role": "measurement"}},
                {"method": "set_measurement_settings", "params": {"object_id": box_id, "quality": "preview", "target_uT": 0.6, "tolerance_pct": 5.0}},
            ]
        )
        result = adapter.analyze_measurement(box_id)
        adapter.create_snapshot(result, "Result A")

        manager = ProjectManagerWindow()
        page = manager.add_project(adapter)
        page.open_snapshots(section="snapshots")
        self.application.processEvents()
        saved = page.record.snapshot_window
        self.assertIsInstance(saved, SnapshotManagerDialog)
        self.assertFalse(saved.isModal())
        self.assertIsNone(saved.parentWidget())
        self.assertEqual(saved.tree.topLevelItemCount(), 1)
        self.assertIsNone(page.record.model_window)

        saved.close()
        self.close_project_manager_for_test(manager)

    def test_project_snapshot_manager_is_single_instance_and_refreshes_live(self):
        adapter = StudioAdapter(compact_workflow_scene())
        box_id = adapter.add_template("guide_box")
        adapter.apply_operations(
            [
                {"method": "set_geometry_style", "params": {"object_id": box_id, "role": "measurement"}},
                {"method": "set_measurement_settings", "params": {"object_id": box_id, "quality": "preview", "target_uT": 0.6, "tolerance_pct": 5.0}},
            ]
        )
        result = adapter.analyze_measurement(box_id)
        first_id = adapter.create_snapshot(result, "Result A")

        manager = ProjectManagerWindow()
        page = manager.add_project(adapter)
        page.open_snapshots(section="snapshots")
        self.application.processEvents()
        saved = page.record.snapshot_window
        self.assertIsInstance(saved, SnapshotManagerDialog)
        self.assertEqual(saved.tree.topLevelItemCount(), 1)

        page.open_snapshots(section="snapshots")
        self.application.processEvents()
        self.assertIs(page.record.snapshot_window, saved)

        second_id = adapter.create_snapshot(result, "Result B")
        page._snapshots_changed()
        self.application.processEvents()
        self.assertEqual(saved.tree.topLevelItemCount(), 2)
        self.assertIn(first_id, [str(saved.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)) for i in range(saved.tree.topLevelItemCount())])
        self.assertIn(second_id, [str(saved.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)) for i in range(saved.tree.topLevelItemCount())])

        saved.close()
        self.close_project_manager_for_test(manager)

    def test_optimizer_wizard_exposes_target_goal_freedom_limits_review_flow(self):
        window = self.make_window()
        target_id = window.adapter.add_template("guide_sphere")
        window.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#16a34a",
                        "opacity": 0.25,
                    },
                }
            ]
        )
        background_id = window.adapter.add_background_field(
            label="Ambient test field",
            vector_uT=[12.0, -3.0, 4.0],
        )
        dialog = OptimizerDialog(window.adapter, window, suggested_target_id=target_id)
        self.assertEqual(dialog.pages.count(), 8)
        self.assertEqual(dialog.pages.currentIndex(), dialog.PAGE_START)
        selected_targets = dialog._checked_tree_ids(dialog.target_tree)
        self.assertEqual(selected_targets, [target_id])
        self.assertTrue(dialog.use_drc.isChecked())
        self.assertEqual(dialog.search_effort.value(), 1)
        self.assertEqual(dialog._wizard_settings()["search_effort"], "balanced")
        self.assertEqual(dialog._wizard_settings()["finalist_count"], 6)
        self.assertIs(dialog.pages.widget(dialog.PAGE_RESULTS), dialog.results_page)
        self.assertNotIsInstance(dialog.results_page, QScrollArea)
        self.assertGreaterEqual(dialog.result_tree.minimumHeight(), 96)
        self.assertGreater(dialog.result_tree.maximumHeight(), 1000)
        self.assertEqual(dialog.result_splitter.orientation(), Qt.Orientation.Vertical)
        self.assertFalse(dialog.result_splitter.childrenCollapsible())
        self.assertFalse(hasattr(dialog, "candidate_details"))
        self.assertEqual(dialog.search_scope, "")
        self.assertEqual(dialog._choice_tree_values(dialog.simple_choices), set())
        self.assertEqual(dialog._choice_tree_values(dialog.complex_choices), set())
        self.assertEqual(window.main_toolbar_buttons["optimize"].text(), "Optimize")
        self.assertEqual(
            window.main_toolbar_buttons["optimize"].accessibleName(), "Optimize design"
        )
        self.assertEqual(dialog.goal_reference_mode.currentData(), "target")
        self.assertIn("samples inside the target band", dialog.goal_objective.currentText())
        self.assertTrue(dialog.goal_objective_explanation.text())
        dialog.goal_reference_mode.setCurrentIndex(dialog.goal_reference_mode.findData("snapshot"))
        self.assertTrue(dialog.goal_target_box.isHidden())
        self.assertFalse(dialog.goal_snapshot_box.isHidden())
        self.assertEqual(dialog.goal_objective.currentData(), "snapshot_vector_match")
        self.assertIn("full field vector", dialog.goal_objective.currentText())
        self.assertEqual(dialog.goal_snapshot_alignment.currentData(), "fixed")
        self.assertFalse(dialog.goal_snapshot_alignment_row.isHidden())
        exposure_index = dialog.goal_snapshot_alignment.findData("exposure")
        self.assertGreaterEqual(exposure_index, 0)
        dialog.goal_snapshot_alignment.setCurrentIndex(exposure_index)
        self.assertEqual(dialog._wizard_settings()["snapshot_alignment_mode"], "exposure")
        dialog.goal_snapshot_alignment.setCurrentIndex(dialog.goal_snapshot_alignment.findData("fixed"))
        dialog.goal_reference_mode.setCurrentIndex(dialog.goal_reference_mode.findData("target"))
        self.assertTrue(dialog.goal_snapshot_alignment_row.isHidden())

        # Simple and Complex routes reveal a separate parameter-family setup page.
        dialog._choose_scope("simple")
        self.assertEqual(dialog.pages.currentIndex(), dialog.PAGE_PARAMETERS)
        self.assertFalse(dialog.simple_setup_box.isHidden())
        self.assertTrue(dialog.complex_setup_box.isHidden())
        simple_current = dialog.simple_choices.topLevelItem(0)
        simple_current.setCheckState(0, Qt.CheckState.Checked)
        dialog._next_page()
        self.assertEqual(dialog.pages.currentIndex(), dialog.PAGE_TARGET)
        self.assertEqual(dialog.search_scope, "simple")
        self.assertEqual(dialog.requested_freedoms, {"current"})
        self.assertTrue(dialog.current_active.isChecked())
        self.assertTrue(dialog.current_active.isEnabled())
        self.assertFalse(dialog.position_active.isChecked())
        # Guided setup preselects a family but Freedom is the final authority.
        dialog.current_active.setChecked(False)
        self.assertNotIn("current", dialog._active_freedoms())
        dialog.current_active.setChecked(True)

        # Full mode skips the family-selection page, exposes every section, and
        # intentionally starts with no participating coils or freedoms preselected.
        dialog._choose_scope("full")
        self.assertEqual(dialog.pages.currentIndex(), dialog.PAGE_TARGET)
        self.assertEqual(
            set(dialog.dimension_controls),
            {
                "axial_winding_length",
                "circular_diameter",
                "racetrack_end_diameter",
                "racetrack_straight_length",
                "square_length",
                "square_width",
            },
        )
        self.assertEqual(set(dialog.position_controls), {"x", "y", "z", "pair_spacing"})
        self.assertEqual(set(dialog.orientation_controls), {"x", "y", "z"})
        self.assertEqual(set(dialog.coil_orientation_controls), {"x", "y", "z"})
        self.assertAlmostEqual(dialog.current_increment_mA.value(), 0.1)
        self.assertEqual(dialog.turns_increment.value(), 1)
        self.assertAlmostEqual(dialog.dimension_controls["circular_diameter"]["step"].value(), 0.5)
        self.assertAlmostEqual(dialog.position_controls["x"]["step"].value(), 0.5)
        self.assertAlmostEqual(dialog.orientation_controls["y"]["step"].value(), 1.0)
        self.assertAlmostEqual(dialog.coil_orientation_controls["z"]["step"].value(), 1.0)
        self.assertEqual(dialog._active_freedoms(), set())
        self.assertEqual(dialog._checked_tree_ids(dialog.freedom_coils), [])
        self.assertFalse(hasattr(dialog, "synthesis_box"))
        self.assertFalse(hasattr(dialog, "allow_synthesis"))
        self.assertFalse(any(
            row["check"].isChecked()
            for controls in (
                dialog.dimension_controls,
                dialog.position_controls,
                dialog.orientation_controls,
                dialog.coil_orientation_controls,
            )
            for row in controls.values()
        ))

        dialog._choose_scope("complex")
        self.assertEqual(dialog.pages.currentIndex(), dialog.PAGE_PARAMETERS)
        self.assertTrue(dialog.simple_setup_box.isHidden())
        self.assertFalse(dialog.complex_setup_box.isHidden())
        for index in range(dialog.complex_choices.topLevelItemCount()):
            item = dialog.complex_choices.topLevelItem(index)
            key = str(item.data(0, Qt.ItemDataRole.UserRole))
            item.setCheckState(
                0,
                Qt.CheckState.Checked if key in {"current", "position"} else Qt.CheckState.Unchecked,
            )
        dialog._next_page()
        self.assertEqual(dialog.pages.currentIndex(), dialog.PAGE_TARGET)
        self.assertEqual(dialog.search_scope, "complex")
        self.assertEqual(dialog.requested_freedoms, {"current", "position"})
        self.assertTrue(dialog.current_active.isChecked())
        self.assertTrue(dialog.position_active.isChecked())
        self.assertTrue(dialog.current_active.isEnabled())
        self.assertTrue(dialog.position_active.isEnabled())
        self.assertFalse(dialog.turns_active.isChecked())

        dialog.pages.setCurrentIndex(dialog.PAGE_REVIEW)
        dialog._refresh_review()
        review = dialog.review_text.toPlainText()
        self.assertIn("TARGET", review)
        self.assertIn("GOAL", review)
        self.assertIn("FREEDOM", review)
        self.assertIn("LIMITS", review)
        self.assertIn("RELATIONSHIPS", review)
        self.assertIn("BACKGROUND FIELDS", review)
        self.assertIn("Ambient test field", review)
        self.assertIn("B = (12, -3, 4) µT", review)
        self.assertIn("manual vector; active", review)
        self.assertIn("Search convergence: automatic", review)
        self.assertIn("Search effort: Balanced", review)
        self.assertIn("increment 0.1 mA", review)
        self.assertIn("concurrent worker", review)
        self.assertEqual(
            dialog._wizard_settings()["worker_count"],
            window.adapter.optimizer_worker_count,
        )
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_freedom_numeric_values_persist_but_selections_do_not(self):
        window = self.make_window()
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "optimizer_freedom.ini"), QSettings.Format.IniFormat
            )
            dialog = OptimizerDialog(window.adapter, window, preferences=settings)
            defaults = dict(dialog._freedom_default_values)

            dialog.current_min_mA.setValue(12.3)
            dialog.current_max_mA.setValue(456.7)
            dialog.current_increment_mA.setValue(0.2)
            dialog.position_controls["x"]["min"].setValue(-42.5)
            dialog.position_controls["x"]["max"].setValue(37.5)
            dialog.position_controls["pair_spacing"]["step"].setValue(2.5)
            dialog.orientation_controls["z"]["min"].setValue(-180.0)
            dialog.orientation_controls["z"]["max"].setValue(180.0)
            dialog.orientation_controls["z"]["step"].setValue(10.0)
            dialog.dimension_controls["circular_diameter"]["check"].setChecked(True)
            dialog.position_controls["x"]["check"].setChecked(True)
            dialog.current_active.setChecked(False)
            dialog._save_freedom_input_history()

            refreshed = OptimizerDialog(window.adapter, window, preferences=settings)
            self.assertAlmostEqual(refreshed.current_min_mA.value(), 12.3)
            self.assertAlmostEqual(refreshed.current_max_mA.value(), 456.7)
            self.assertAlmostEqual(refreshed.current_increment_mA.value(), 0.2)
            self.assertAlmostEqual(refreshed.position_controls["x"]["min"].value(), -42.5)
            self.assertAlmostEqual(refreshed.position_controls["x"]["max"].value(), 37.5)
            self.assertAlmostEqual(refreshed.position_controls["pair_spacing"]["step"].value(), 2.5)
            self.assertAlmostEqual(refreshed.orientation_controls["z"]["min"].value(), -180.0)
            self.assertAlmostEqual(refreshed.orientation_controls["z"]["max"].value(), 180.0)
            self.assertAlmostEqual(refreshed.orientation_controls["z"]["step"].value(), 10.0)

            # Only numeric entries are remembered. Freedom/row checkboxes never
            # inherit the prior run; before a new Simple/Complex/Full route is chosen,
            # no freedom family is silently preselected.
            self.assertFalse(refreshed.dimension_controls["circular_diameter"]["check"].isChecked())
            self.assertFalse(refreshed.position_controls["x"]["check"].isChecked())
            self.assertFalse(refreshed.current_active.isChecked())

            refreshed.pages.setCurrentIndex(refreshed.PAGE_FREEDOM)
            self.assertFalse(refreshed.freedom_defaults_button.isHidden())
            refreshed.freedom_defaults_button.click()
            self.assertAlmostEqual(
                refreshed.current_min_mA.value(), defaults["current/min"]
            )
            self.assertAlmostEqual(
                refreshed.position_controls["x"]["max"].value(),
                defaults["position/x/max"],
            )
            self.assertAlmostEqual(
                refreshed.orientation_controls["z"]["step"].value(),
                defaults["assembly_orientation/z/increment"],
            )

            dialog.deleteLater()
            refreshed.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_guided_dimension_rows_follow_participating_coil_shapes(self):
        window = self.make_window()
        dialog = OptimizerDialog(window.adapter, window)

        # Compact workflow scene contains only circular coils. Guided modes should
        # hide dimensions that cannot apply to the participating coils.
        dialog._choose_scope("simple")
        for index in range(dialog.simple_choices.topLevelItemCount()):
            item = dialog.simple_choices.topLevelItem(index)
            key = str(item.data(0, Qt.ItemDataRole.UserRole))
            item.setCheckState(
                0,
                Qt.CheckState.Checked if key == "dimensions" else Qt.CheckState.Unchecked,
            )
        dialog._next_page()
        self.assertEqual(dialog.requested_freedoms, {"dimensions"})
        self.assertFalse(dialog.dimension_controls["axial_winding_length"]["check"].isHidden())
        self.assertFalse(dialog.dimension_controls["circular_diameter"]["check"].isHidden())
        for key in (
            "racetrack_end_diameter",
            "racetrack_straight_length",
            "square_length",
            "square_width",
        ):
            self.assertTrue(dialog.dimension_controls[key]["check"].isHidden())

        # Complex uses the same shape-aware filtering. Full deliberately exposes
        # the complete inventory regardless of current participating coil shapes.
        dialog._choose_scope("complex")
        for index in range(dialog.complex_choices.topLevelItemCount()):
            item = dialog.complex_choices.topLevelItem(index)
            key = str(item.data(0, Qt.ItemDataRole.UserRole))
            item.setCheckState(
                0,
                Qt.CheckState.Checked if key == "dimensions" else Qt.CheckState.Unchecked,
            )
        dialog._next_page()
        self.assertTrue(dialog.dimension_controls["racetrack_end_diameter"]["check"].isHidden())
        self.assertTrue(dialog.dimension_controls["square_length"]["check"].isHidden())

        dialog._choose_scope("full")
        for row in dialog.dimension_controls.values():
            self.assertFalse(row["check"].isHidden())

        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_full_mode_compiles_many_linked_and_independent_variables(self):
        window = self.make_window()
        dialog = OptimizerDialog(window.adapter, window)
        dialog._choose_scope("full")

        # Participate with both fixture coils. Give each its own current variable,
        # while structural properties remain linked as Group A.
        ids = []
        for index in range(min(2, dialog.freedom_coils.topLevelItemCount())):
            item = dialog.freedom_coils.topLevelItem(index)
            item.setCheckState(0, Qt.CheckState.Checked)
            object_id = str(item.data(0, Qt.ItemDataRole.UserRole))
            ids.append(object_id)
            current_combo = dialog.freedom_link_widgets["current"][object_id]
            current_combo.setCurrentIndex(current_combo.findData("__independent__"))
            for kind in ("position", "orientation", "geometry"):
                combo = dialog.freedom_link_widgets[kind][object_id]
                combo.setCurrentIndex(combo.findData("A"))

        dialog.current_active.setChecked(True)
        dialog.turns_active.setChecked(True)
        dialog.dimension_active.setChecked(True)
        dialog.position_active.setChecked(True)
        dialog.orientation_active.setChecked(True)
        dialog.assembly_orientation_active.setChecked(True)
        dialog.coil_orientation_active.setChecked(True)
        dialog.dimension_controls["circular_diameter"]["check"].setChecked(True)
        dialog.position_controls["x"]["check"].setChecked(True)
        dialog.position_controls["z"]["check"].setChecked(True)
        dialog.orientation_controls["y"]["check"].setChecked(True)
        dialog.coil_orientation_controls["z"]["check"].setChecked(True)

        variables = dialog._optimizer_variable_specs()
        self.assertEqual(len([item for item in variables if item["kind"] == "current"]), 2)
        self.assertEqual(len([item for item in variables if item["kind"] == "turns"]), 1)
        self.assertEqual(len([item for item in variables if item["kind"] == "circular_diameter"]), 1)
        self.assertEqual(len([item for item in variables if item["kind"].startswith("position_")]), 2)
        self.assertEqual(len([item for item in variables if item["kind"] == "assembly_orientation_y"]), 1)
        self.assertEqual(len([item for item in variables if item["kind"] == "coil_orientation_z"]), 1)
        by_kind = {}
        for item in variables:
            by_kind.setdefault(item["kind"], []).append(item)
        self.assertTrue(all(abs(item["step"] - 0.1) < 1e-12 for item in by_kind["current"]))
        self.assertEqual(by_kind["turns"][0]["step"], 1.0)
        self.assertEqual(by_kind["circular_diameter"][0]["step"], 0.5)
        self.assertEqual(by_kind["position_x"][0]["step"], 0.5)
        self.assertEqual(by_kind["assembly_orientation_y"][0]["step"], 1.0)
        self.assertEqual(by_kind["coil_orientation_z"][0]["step"], 1.0)
        self.assertIsNone(dialog._validate_page(dialog.PAGE_FREEDOM))

        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_relationship_preset_buttons_link_all_unlink_and_scene_groups(self):
        window = self.make_window()
        group_id = window.adapter.group_objects(["coil_a", "coil_b"])
        dialog = OptimizerDialog(window.adapter, window)
        dialog._choose_scope("complex")

        # Guided setup defaults to mirroring the source scene group.  Both compact
        # fixture coils now share the same immediate Collection, so they should
        # share Group A in every relationship column.
        for kind in ("current", "position", "orientation", "geometry"):
            self.assertEqual(dialog.freedom_link_widgets[kind]["coil_a"].currentData(), "A")
            self.assertEqual(dialog.freedom_link_widgets[kind]["coil_b"].currentData(), "A")

        dialog.unlink_all_button.click()
        for kind in ("current", "position", "orientation", "geometry"):
            self.assertEqual(
                dialog.freedom_link_widgets[kind]["coil_a"].currentData(),
                "__independent__",
            )
            self.assertEqual(
                dialog.freedom_link_widgets[kind]["coil_b"].currentData(),
                "__independent__",
            )

        dialog.link_all_button.click()
        for kind in ("current", "position", "orientation", "geometry"):
            self.assertEqual(dialog.freedom_link_widgets[kind]["coil_a"].currentData(), "A")
            self.assertEqual(dialog.freedom_link_widgets[kind]["coil_b"].currentData(), "A")

        # Breaking the scene group and restoring Link by group leaves the two
        # root-level coils independently controllable.
        window.adapter.ungroup(group_id)
        refreshed = OptimizerDialog(window.adapter, window)
        refreshed._choose_scope("complex")
        for kind in ("current", "position", "orientation", "geometry"):
            self.assertEqual(
                refreshed.freedom_link_widgets[kind]["coil_a"].currentData(),
                "__independent__",
            )
            self.assertEqual(
                refreshed.freedom_link_widgets[kind]["coil_b"].currentData(),
                "__independent__",
            )

        dialog.deleteLater()
        refreshed.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_results_header_shows_effort_and_target_sampling(self):
        window = self.make_window()
        target_id = window.adapter.add_template("guide_sphere")
        window.adapter.apply_operations([{
            "method": "set_geometry_style",
            "params": {
                "object_id": target_id,
                "role": "measurement",
                "colour": "#64748b",
                "opacity": 0.2,
            },
        }])
        dialog = OptimizerDialog(window.adapter, window)
        target = dialog._target_records[target_id]
        dialog._refresh_results_context(
            settings={
                "search_effort": "fast",
                "target_object_ids": [target_id],
                "reference_mode": "target",
                "target_uT": 200.0,
                "tolerance_pct": 5.0,
            },
            progress={
                "target_sample_count": 548,
                "scout_sample_count": 80,
                "target_samples": [
                    {
                        "id": target_id,
                        "label": target.get("label", target_id),
                        "sampling_mode": "generated",
                        "sample_count": 548,
                    }
                ],
            },
        )
        self.assertEqual(dialog.results_heading.text(), "RESULTS — Fast search")
        self.assertIn(str(target.get("label", target_id)), dialog.results_target_info.text())
        self.assertIn("548 full samples", dialog.results_target_info.text())
        self.assertIn("80 scout samples", dialog.results_target_info.text())
        self.assertIn("Standard generated sampling", dialog.results_target_info.text())
        self.assertIn("200 µT ± 5%", dialog.results_target_info.text())
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_progress_shows_phase_counts_best_and_elapsed_time(self):
        window = self.make_window()
        dialog = OptimizerDialog(window.adapter, window)
        self.assertIs(dialog.result_views.currentWidget(), dialog.sweep_plot)
        self.assertFalse(dialog.result_views.isTabVisible(0))
        self.assertFalse(dialog.result_views.isTabVisible(1))
        self.assertTrue(dialog.result_views.isTabVisible(2))
        self.assertTrue(dialog.result_summary.isHidden())
        self.assertTrue(dialog.result_tree.isHidden())
        self.assertTrue(dialog.copy_button.isHidden())
        self.assertTrue(dialog.compare_button.isHidden())
        self.assertTrue(dialog.open_project_button.isHidden())
        self.assertTrue(dialog.apply_button.isHidden())
        self.assertTrue(dialog.progress_status.isHidden())
        self.assertTrue(dialog.progress_best.isHidden())
        self.assertGreaterEqual(dialog.result_views.minimumHeight(), 500)
        dialog._optimizer_run_started_at = time.monotonic() - 1.25
        dialog._progress(
            {
                "phase_label": "Refining diameter maximum boundary vs spacing",
                "candidate": 37,
                "feasible": 35,
                "rejected": 2,
                "solutions_discovered": 1846,
                "solutions_active": 24,
                "solutions_retired": 1773,
                "solutions_blocked": 49,
                "solutions_authoritative": 0,
                "best_setting": "Circular diameter Group A=300 mm; Pair spacing Group A=150 mm; Current Group A=104 mA",
                "best_coverage_pct": 100.0,
                "best_uniformity_spread_pct": 0.0123,
                "best_mean_uT": 200.0,
                "best_rms_target_error_uT": 0.02,
                "best_directional_consistency_pct": 99.999,
            }
        )
        status = dialog.progress_status.text()
        best = dialog.progress_best.text()
        self.assertIn("Refining diameter maximum boundary", status)
        self.assertIn("1,846 solutions discovered", status)
        self.assertIn("24 active", status)
        self.assertIn("1,773 retired", status)
        self.assertIn("49 blocked", status)
        self.assertIn("0 authoritative", status)
        self.assertIn("Best so far:", best)
        self.assertIn("coverage 100", best)
        self.assertIn("uniformity", best)
        self.assertIn("mean 200", best)
        trace_text = dialog.sweep_plot.message.toPlainText()
        self.assertIn("Search plan", trace_text)
        self.assertIn("Execution tree", trace_text)
        self.assertIn("Refining diameter maximum boundary", trace_text)
        self.assertIn("1,846 solutions discovered", trace_text)
        self.assertIn("49 blocked", trace_text)
        self.assertIn("Best so far", trace_text)
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_search_trace_shows_exposure_registration_accounting(self):
        window = self.make_window()
        dialog = OptimizerDialog(window.adapter, window)
        dialog._initialize_optimizer_trace_plan(
            {
                "search_effort": "fast",
                "snapshot_alignment_mode": "exposure",
                "variables": [
                    {"kind": "position_x", "min": -10.0, "max": 10.0},
                    {"kind": "orientation_z", "min": -90.0, "max": 90.0},
                    {"kind": "current", "min": 1.0, "max": 100.0},
                ],
            }
        )
        self.assertIn(
            "inner exposure registration",
            str(dialog._optimizer_trace_nodes["map"]["label"]).lower(),
        )
        self.assertEqual(
            dialog._optimizer_trace_leaf_for_phase(
                "Centreline map • Design-space map • Fast funnel • target placement"
            ),
            "map.align",
        )
        self.assertEqual(
            dialog._optimizer_trace_leaf_for_phase(
                "Centreline map • Design-space map • Fast funnel • final coordinate polish"
            ),
            "map.polish",
        )
        dialog._optimizer_run_started_at = time.monotonic() - 3.0
        dialog._progress(
            {
                "phase_label": "Centreline map — global Sobol skeptic",
                "candidate": 12,
                "feasible": 12,
                "rejected": 0,
                "solutions_discovered": 12,
                "solutions_active": 8,
                "solutions_retired": 4,
                "solutions_blocked": 0,
                "solutions_authoritative": 0,
                "best_setting": "Pair spacing Group A=198 mm",
                "best_snapshot_match_coverage_pct": 53.2105,
                "best_snapshot_vector_rms_uT": 0.4143919,
                "best_snapshot_magnitude_rms_uT": 0.321951,
                "best_snapshot_mean_angle_deg": 1.925329,
                "best_snapshot_alignment_rx_deg": -5.939461,
                "best_snapshot_alignment_ry_deg": 15.80432,
                "best_snapshot_alignment_rz_deg": -20.07977,
                "exposure_registration_enabled": True,
                "exposure_registration_candidates": 12,
                "exposure_registration_evaluations": 244,
                "exposure_registration_basis_builds": 232,
                "exposure_registration_cache_hits": 18,
                "exposure_registration_kabsch_seeds": 12,
                "exposure_registration_inherited_seeds": 8,
                "exposure_registration_coarse_candidates": 4,
                "exposure_registration_standard_candidates": 6,
                "exposure_registration_refined_candidates": 2,
                "exposure_registration_seconds": 2.5,
            }
        )
        trace_text = dialog.sweep_plot.message.toPlainText()
        self.assertIn("Exposure registration", trace_text)
        self.assertIn("244 pose evaluations", trace_text)
        self.assertIn("232 extra field-basis builds", trace_text)
        self.assertIn("18 registration-cache hits", trace_text)
        self.assertIn("12 Kabsch / 8 inherited", trace_text)
        self.assertIn("effort 4 coarse / 6 standard / 2 refined", trace_text)
        self.assertIn("magnitude RMS", trace_text)
        self.assertIn("angle", trace_text)
        self.assertIn("RX -5.939", trace_text)
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_progress_bar_tracks_search_plan_deterministically_and_never_rewinds(self):
        window = self.make_window()
        dialog = OptimizerDialog(window.adapter, window)
        dialog._initialize_optimizer_trace_plan(
            {
                "search_effort": "fast",
                "variables": [
                    {"kind": "position_x", "min": -10.0, "max": 10.0},
                    {"kind": "position_y", "min": -10.0, "max": 10.0},
                    {"kind": "orientation_z", "min": -5.0, "max": 5.0},
                    {"kind": "current", "min": 0.01, "max": 0.1},
                ],
            }
        )
        dialog._optimizer_run_started_at = time.monotonic() - 0.25
        dialog._progress(
            {
                "phase_label": "Centreline map — known-relation atlas",
                "candidate": 10,
                "feasible": 10,
                "rejected": 0,
                "trace_completed": 10,
                "trace_total": 100,
            }
        )
        self.assertEqual(dialog.progress_bar.minimum(), 0)
        self.assertEqual(dialog.progress_bar.maximum(), 1000)
        first = dialog.progress_bar.value()
        self.assertGreater(first, 0)
        self.assertLess(first, 1000)

        dialog._progress(
            {
                "phase_label": "Centreline map — known-relation atlas",
                "candidate": 50,
                "feasible": 50,
                "rejected": 0,
                "trace_completed": 50,
                "trace_total": 100,
            }
        )
        second = dialog.progress_bar.value()
        self.assertGreater(second, first)

        # Adaptive work may later reveal a larger denominator.  The underlying
        # raw fraction may shrink, but the visible plan progress must not rewind.
        dialog._progress(
            {
                "phase_label": "Centreline map — known-relation atlas",
                "candidate": 50,
                "feasible": 50,
                "rejected": 0,
                "trace_completed": 50,
                "trace_total": 200,
            }
        )
        self.assertGreaterEqual(dialog.progress_bar.value(), second)

        # Completing the genuinely cheap map must not imply that a Fast run is
        # almost finished before expensive DRC boundary work starts.
        weighted = OptimizerDialog(window.adapter, window)
        weighted._initialize_optimizer_trace_plan(
            {
                "search_effort": "fast",
                "variables": [
                    {"kind": "position_x", "min": -25.0, "max": 25.0},
                    {"kind": "position_y", "min": -25.0, "max": 25.0},
                    {"kind": "position_z", "min": -25.0, "max": 25.0},
                    {"kind": "assembly_orientation_x", "min": -360.0, "max": 360.0},
                    {"kind": "assembly_orientation_y", "min": -360.0, "max": 360.0},
                    {"kind": "assembly_orientation_z", "min": -360.0, "max": 360.0},
                    {"kind": "pair_spacing", "min": 120.0, "max": 220.0},
                    {"kind": "current", "min": 0.005, "max": 0.1},
                ],
            }
        )
        weighted._optimizer_trace_nodes["prepare"]["status"] = "complete"
        weighted._optimizer_trace_nodes["map"]["status"] = "complete"
        weighted._optimizer_trace_nodes["drc"]["status"] = "active"
        drc = weighted._optimizer_trace_nodes["drc.map"]
        drc["status"] = "active"
        drc["count"] = 0
        drc["total"] = 140
        weighted._optimizer_trace_active_leaf = "drc.map"
        weighted._refresh_optimizer_determinate_progress_bar()
        self.assertLess(weighted.progress_bar.value(), 500)
        weighted.deleteLater()

        # A high-dimensional Balanced run keeps DRC and authoritative validation
        # as independent top-level stages. Completing reconnaissance/DRC must not
        # make the global bar claim completion while the authoritative validation
        # ceiling is still being consumed.
        balanced = OptimizerDialog(window.adapter, window)
        balanced._initialize_optimizer_trace_plan(
            {
                "search_effort": "balanced",
                "max_evaluations": 384,
                "variables": [
                    {"kind": "position_x", "min": -25.0, "max": 25.0},
                    {"kind": "position_y", "min": -25.0, "max": 25.0},
                    {"kind": "position_z", "min": -25.0, "max": 25.0},
                    {"kind": "orientation_x", "min": -30.0, "max": 30.0},
                    {"kind": "orientation_y", "min": -30.0, "max": 30.0},
                    {"kind": "orientation_z", "min": -30.0, "max": 30.0},
                    {"kind": "diameter", "min": 100.0, "max": 200.0},
                    {"kind": "current", "min": 0.01, "max": 0.15},
                ],
            }
        )
        for stage_id in ("prepare", "map", "drc"):
            balanced._optimizer_trace_nodes[stage_id]["status"] = "complete"
        authority = balanced._optimizer_trace_nodes["authority"]
        authority["status"] = "active"
        balanced._optimizer_trace_nodes["authority.promote"]["status"] = "complete"

        validation = balanced._optimizer_trace_nodes["authority.validate"]
        validation["status"] = "active"
        validation_total = int(validation["total"] or 0)
        self.assertGreater(validation_total, 1)
        validation["count"] = validation_total - 1
        balanced._optimizer_trace_active_leaf = "authority.validate"
        balanced._optimizer_progress_fraction = 0.0
        balanced._refresh_optimizer_determinate_progress_bar()
        self.assertLess(balanced.progress_bar.value(), 1000)
        self.assertGreater(balanced.progress_bar.value(), 850)
        self.assertEqual(
            balanced._optimizer_trace_count_text(validation),
            f"{validation_total - 1}/{validation_total}",
        )

        # Adaptive/planned denominators stay visible after completion, but the
        # observed work becomes the final denominator.  This preserves the readable
        # 12/12, 14/14, 48/48 style without retaining stale plans such as 12/4.
        validation["status"] = "complete"
        self.assertEqual(
            balanced._optimizer_trace_count_text(validation),
            f"{validation_total - 1}/{validation_total - 1}",
        )
        validation["status"] = "active"

        # A provisional active denominator must never display below work already
        # observed when repeated/adaptive sub-passes reuse the same trace row.
        validation["count"] = validation_total + 3
        validation["total"] = validation_total
        self.assertEqual(
            balanced._optimizer_trace_count_text(validation),
            f"{validation_total + 3}/{validation_total + 3}",
        )
        validation["count"] = validation_total - 1
        validation["total"] = validation_total

        # Even if the progressive validation ceiling is reached before the worker
        # emits completion, the terminal 1% remains reserved for that signal.
        validation["count"] = validation_total
        balanced._refresh_optimizer_determinate_progress_bar()
        self.assertLess(balanced.progress_bar.value(), 1000)
        balanced._refresh_optimizer_determinate_progress_bar(force_complete=True)
        self.assertEqual(balanced.progress_bar.value(), 1000)
        balanced.deleteLater()

        dialog._refresh_optimizer_determinate_progress_bar(force_complete=True)
        self.assertEqual(dialog.progress_bar.value(), 1000)
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_worker_benchmark_dialog_reports_and_accepts_recommendation(self):
        dialog = OptimizerWorkerBenchmarkDialog(
            coil_modeling_method="centreline",
            solver_memory_budget_mb=256,
            current_workers=1,
        )
        self.assertEqual(dialog.max_workers_spin.minimum(), 1)
        self.assertEqual(dialog.max_workers_spin.maximum(), 32)
        dialog.max_workers_spin.setValue(23)
        self.assertEqual(dialog.max_workers_spin.value(), 23)
        self.assertTrue(dialog.start_button.isEnabled())
        self.assertFalse(dialog.cancel_button.isEnabled())
        dialog._update_progress(
            {
                "completed": 1,
                "total": 2,
                "result": {
                    "workers": 1,
                    "elapsed_seconds": 1.25,
                    "tasks_per_second": 9.6,
                    "speedup": 1.0,
                },
            }
        )
        self.assertEqual(dialog.results.topLevelItemCount(), 1)
        self.assertEqual(dialog.results.topLevelItem(0).text(0), "1")
        dialog._completed(
            {
                "recommended_workers": 1,
                "task_count": 12,
                "observer_count": 315,
            }
        )
        self.assertEqual(dialog.recommended_workers, 1)
        self.assertTrue(dialog.use_button.isEnabled())
        self.assertIn("Recommended: 1 worker", dialog.status.text())
        dialog.deleteLater()

    def test_optimizer_wizard_wheel_scrolls_without_changing_editor_values(self):
        window = self.make_window()
        dialog = OptimizerDialog(window.adapter, window)
        scroll_bar = dialog.goal_scroll.verticalScrollBar()
        scroll_bar.setRange(0, 1000)

        def wheel_up(widget):
            event = QWheelEvent(
                QPointF(5.0, 5.0),
                QPointF(5.0, 5.0),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.ScrollUpdate,
                False,
            )
            QApplication.sendEvent(widget, event)

        scroll_bar.setValue(500)
        target_before = dialog.goal_target_uT.value()
        wheel_up(dialog.goal_target_uT)
        self.assertEqual(dialog.goal_target_uT.value(), target_before)
        self.assertLess(scroll_bar.value(), 500)

        scroll_bar.setValue(500)
        objective_before = dialog.goal_objective.currentIndex()
        wheel_up(dialog.goal_objective)
        self.assertEqual(dialog.goal_objective.currentIndex(), objective_before)
        self.assertLess(scroll_bar.value(), 500)
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_opens_as_independent_non_modal_window(self):
        window = self.make_window()
        target_id = window.adapter.add_template("guide_sphere")
        window.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {
                        "object_id": target_id,
                        "role": "measurement",
                        "colour": "#16a34a",
                        "opacity": 0.25,
                    },
                }
            ]
        )
        source_id = window.active_document.document_id
        window.open_optimizer()
        self.application.processEvents()
        dialogs = [
            item for item in window._independent_windows.values()
            if isinstance(item, OptimizerDialog)
        ]
        self.assertEqual(len(dialogs), 1)
        dialog = dialogs[0]
        self.assertFalse(dialog.isModal())
        self.assertEqual(dialog.windowModality(), Qt.WindowModality.NonModal)
        self.assert_standard_window_controls(dialog)
        self.assertTrue(dialog.isSizeGripEnabled())
        self.assertEqual(
            dialog.pages.sizePolicy().verticalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        self.assertIn(source_id, window._independent_window_documents.values())
        self.assertIn("Untitled", dialog.windowTitle())
        dialog.close()
        self.application.processEvents()
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_finalist_is_emitted_as_a_new_project(self):
        window = self.make_window()
        source_document = window.adapter.document()
        original_document = window.active_document
        original_position = window.adapter.get_transform("coil_a")["position"]
        emitted = []
        window.suite_new_project_adapter_requested.connect(emitted.append)
        candidate = {
            "id": "tuning_2",
            "mode": "tune_existing",
            "setting_summary": "coil A ΔZ = +10 mm",
            "transforms": {
                "coil_a": {
                    "position_m": [0.0, 0.0, -0.04],
                    "euler_deg": [0.0, 0.0, 0.0],
                }
            },
            "currents_a": {"coil_a": 0.07, "coil_b": 0.07},
        }

        window._optimizer_candidate_opened_new_project(candidate, source_document)
        self.application.processEvents()

        self.assertEqual(len(emitted), 1)
        self.assertIs(window.active_document, original_document)
        self.assertEqual(window.adapter.get_transform("coil_a")["position"], original_position)
        candidate_adapter = emitted[0]
        self.assertIsNot(candidate_adapter, window.adapter)
        self.assertIn("Optimizer finalist 2", candidate_adapter.suggested_project_name)
        self.assertAlmostEqual(candidate_adapter.get_transform("coil_a")["position"][2], -0.04)
        self.assertFalse(hasattr(window, "document_tabs"))
        window.plot.cleanup()
        window.deleteLater()

    def test_optimizer_wizard_results_report_real_currents_and_metrics(self):
        window = self.make_window()
        dialog = OptimizerDialog(window.adapter, window)
        candidate = {
            "id": "tuning_1",
            "mode": "tune_existing",
            "tuning_mode": "spacing",
            "display_parameter_value": 92.4,
            "parameter_unit": "mm",
            "currents_a": {"coil_a": 0.08547, "coil_b": 0.08547},
            "metrics": {
                "in_band_pct": 100.0,
                "uniformity_spread_pct": 5.93,
                "directional_consistency_pct": 99.962,
                "mean_uT": 200.94,
                "rms_target_error_uT": 2.1,
                "minimum_clearance_mm": 22.4,
            },
        }
        dialog.result_tree.blockSignals(True)
        dialog._completed(
            {
                "run_mode": "tuning",
                "settings": {
                    "tuning_mode": "spacing",
                    "fit_current_to_target": True,
                    "spacing_steps": 41,
                    "current_steps": 121,
                },
                "finalists": [candidate],
                "generated_candidate_count": 4961,
                "rejected_candidate_count": 0,
                "final_sample_count": 288,
                "targets": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                "elapsed_seconds": 1.0,
            }
        )
        dialog.result_tree.blockSignals(False)
        self.assertTrue(dialog.result_views.isTabVisible(0))
        self.assertTrue(dialog.result_views.isTabVisible(1))
        self.assertTrue(dialog.result_views.isTabVisible(2))
        self.assertFalse(dialog.result_summary.isHidden())
        self.assertFalse(dialog.result_tree.isHidden())
        self.assertTrue(dialog.dialog_heading.isHidden())
        self.assertTrue(dialog.dialog_intro.isHidden())
        self.assertTrue(dialog.step_label.isHidden())
        self.assertTrue(dialog.next_button.isHidden())
        self.assertFalse(dialog.copy_button.isHidden())
        self.assertFalse(dialog.open_project_button.isHidden())
        self.assertEqual(dialog.result_views.minimumHeight(), 220)
        self.assertGreaterEqual(dialog.result_tree.height(), 96)
        self.assertGreater(dialog.result_tree.maximumHeight(), 1000)
        self.assertIs(dialog.result_views.currentWidget(), dialog.candidate_plot)
        self.assertEqual(dialog.result_views.tabText(0), "Comparison")
        self.assertEqual(dialog.result_views.tabText(1), "Full statistics")
        self.assertEqual(dialog.result_views.tabText(2), "Search history")
        self.assertEqual(dialog.result_tree.headerItem().text(2), "Resulting currents")
        self.assertEqual(dialog.open_project_button.text(), "Open as project")
        self.assertEqual(dialog.apply_button.text(), "Apply to source")
        self.assertEqual(dialog.append_button.text(), "Append to source")
        self.assertEqual(dialog.save_snapshot_button.text(), "Save result")
        self.assertEqual(dialog.compare_button.text(), "Compare selected…")
        self.assertEqual(dialog.result_tree.topLevelItem(0).text(1), "92.4 mm")
        self.assertIn("85.47 mA", dialog.result_tree.topLevelItem(0).text(2))
        self.assertIn("4961 tested settings across 1 free parameter", dialog.result_summary.text())
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_mesh_import_options_offer_non_destructive_centering_choice(self):
        dialog = MeshImportOptionsDialog("offset head scan.stl")
        self.assertAlmostEqual(dialog.unit_scale(), 0.001)
        self.assertFalse(dialog.should_center())
        self.assertEqual(dialog.center_mesh.objectName(), "centerImportedMeshCheckBox")
        dialog.center_mesh.setChecked(True)
        self.assertTrue(dialog.should_center())
        dialog.deleteLater()

    def test_coil_modeling_preference_is_applied_when_window_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "solver.ini"), QSettings.Format.IniFormat
            )
            settings.setValue("coil_modeling/method", "bundled")
            window = self.make_window(settings=settings)
            self.assertEqual(window.adapter.coil_modeling_method, "bundled")
            window.plot.cleanup()
            window.deleteLater()

            settings.setValue("coil_modeling/method", "not-a-model")
            fallback = self.make_window(settings=settings)
            self.assertEqual(fallback.adapter.coil_modeling_method, "auto")
            fallback.plot.cleanup()
            fallback.deleteLater()

    def test_optimizer_worker_preference_is_applied_when_window_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "optimizer.ini"), QSettings.Format.IniFormat
            )
            settings.setValue("optimizer/workerCount", 1)
            window = self.make_window(settings=settings)
            self.assertEqual(window.adapter.optimizer_worker_count, 1)
            window.plot.cleanup()
            window.deleteLater()

    def test_fluxline_cleanup_is_viewer_local_and_only_shown_for_fluxlines(self):
        window = self.make_window()
        dialog = FieldMapDialog(window.adapter, workspace_mode="static")
        self.assertTrue(dialog.fluxline_options_section.isVisibleTo(dialog.control_panel))

        heatmap_index = dialog.display_mode.findData("heatmap")
        dialog.display_mode.setCurrentIndex(heatmap_index)
        self.assertFalse(dialog.fluxline_options_section.isVisibleTo(dialog.control_panel))

        flux_index = dialog.display_mode.findData("flux")
        dialog.display_mode.setCurrentIndex(flux_index)
        self.assertTrue(dialog.fluxline_options_section.isVisibleTo(dialog.control_panel))
        dialog.fluxline_minimum_field_enabled.setChecked(False)
        dialog.fluxline_minimum_field_value.setValue(0.125)
        dialog.fluxline_conductor_cutoff_enabled.setChecked(True)
        dialog.fluxline_conductor_cutoff_value.setValue(3.5)
        self.assertFalse(dialog.adapter.fluxline_minimum_field_cutoff_enabled)
        self.assertAlmostEqual(dialog.adapter.fluxline_minimum_field_cutoff_percent, 0.125)
        self.assertTrue(dialog.adapter.fluxline_conductor_cutoff_enabled)
        self.assertAlmostEqual(dialog.adapter.fluxline_conductor_cutoff_mm, 3.5)

        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_open_recent_persists_and_deduplicates_scene_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "recent.ini"), QSettings.Format.IniFormat
            )
            first = self.make_window(settings=settings)
            scene_a = Path(directory) / "alpha.magpy.json"
            scene_b = Path(directory) / "beta.magpy.json"
            first.adapter.save_file(scene_a)
            first.adapter.save_file(scene_b)
            first._record_recent_file(scene_a)
            first._record_recent_file(scene_b)
            first._record_recent_file(scene_a)
            self.assertEqual(
                first._recent_files(),
                [str(scene_a.resolve()), str(scene_b.resolve())],
            )

            second = self.make_window(settings=settings)
            recent_actions = [
                action for action in second.recent_menu.actions() if action.data()
            ]
            self.assertEqual(len(recent_actions), 2)
            self.assertEqual(recent_actions[0].data(), str(scene_a.resolve()))
            self.assertIn("Clear Recent Files", [a.text() for a in second.recent_menu.actions()])
            second.clear_recent_files()
            self.assertEqual(second._recent_files(), [])
            first.plot.cleanup()
            second.plot.cleanup()
            first.deleteLater()
            second.deleteLater()

    def test_recent_file_identity_can_fold_case_without_lowercasing_display(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "recent_case.ini"), QSettings.Format.IniFormat
            )
            window = self.make_window(settings=settings)
            scene = Path(directory) / "MixedCase.magpy.json"
            expected_display = str(scene.resolve())
            with mock.patch.object(
                main_window_module.os.path,
                "normcase",
                side_effect=lambda value: str(value).casefold(),
            ):
                window._record_recent_file(scene)
                self.assertEqual(window._recent_files(), [expected_display])
                window._record_recent_file(expected_display.swapcase())
                self.assertEqual(len(window._recent_files()), 1)
            window.plot.cleanup()
            window.deleteLater()

    def test_sensor_plot_lists_scene_sensors_and_has_an_empty_state(self):
        window = self.make_window()
        sensor_plot = SensorPlotDialog(window.adapter, window)

        checks = sensor_plot.findChildren(QCheckBox, "sensorPlotSensorCheckbox")
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].property("sensorId"), "axis_sensor")
        self.assertTrue(checks[0].isChecked())
        self.assertTrue(sensor_plot.calculate_button.isEnabled())

        checks[0].setChecked(False)
        self.assertFalse(sensor_plot.calculate_button.isEnabled())
        checks[0].setChecked(True)
        self.assertTrue(sensor_plot.calculate_button.isEnabled())

        window.adapter.remove("axis_sensor")
        sensor_plot._refresh_sensor_choices()
        self.assertFalse(sensor_plot.calculate_button.isEnabled())
        self.assertEqual(sensor_plot.sensor_checks, {})
        self.assertIn(
            "Add a sensor to the scene to create a sensor plot",
            sensor_plot.plot.message.toPlainText(),
        )

        point_id = window.adapter.add_template("sensor")
        sensor_plot._refresh_sensor_choices()
        self.assertIn(point_id, sensor_plot.sensor_checks)
        self.assertTrue(sensor_plot.sensor_checks[point_id].isChecked())
        self.assertTrue(sensor_plot.calculate_button.isEnabled())

        sensor_plot.plot.cleanup()
        sensor_plot.deleteLater()
        window.plot.cleanup()
        window.deleteLater()

    def test_analysis_probe_and_sensor_results_have_copy_export_controls(self):
        window = self.make_window()
        box_id = window.adapter.add_template("guide_box")
        window.adapter.apply_operations(
            [
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box_id, "role": "measurement"},
                },
                {
                    "method": "set_measurement_settings",
                    "params": {
                        "object_id": box_id,
                        "quality": "preview",
                        "target_uT": 0.6,
                        "tolerance_pct": 5.0,
                    },
                },
            ]
        )
        result = window.adapter.analyze_measurement(box_id)
        measurement = MeasurementResultsDialog(window.adapter, result, window)
        self.assertEqual(measurement.save_snapshot_button.objectName(), "primaryButton")
        selectable = Qt.TextInteractionFlag.TextSelectableByMouse
        self.assertTrue(all(label.textInteractionFlags() & selectable for label in measurement.findChildren(QLabel)))
        section_buttons = [
            button
            for button in measurement.findChildren(QPushButton)
            if button.objectName() == "copyStatisticsSectionButton"
        ]
        self.assertEqual(len(section_buttons), 3)
        section_buttons[0].click()
        self.assertIn("Field magnitude |B|", QApplication.clipboard().text())
        self.assertIn("Mean:", QApplication.clipboard().text())

        sensor_plot = SensorPlotDialog(window.adapter, window)
        self.assertEqual(sensor_plot.calculate_button.objectName(), "primaryButton")
        self.assertEqual(
            sensor_plot.calculate_button.property("workbenchAction"),
            "calculateSensorPathButton",
        )
        self.assertEqual(sensor_plot.export_csv_button.text(), "Export CSV…")
        self.assertEqual(sensor_plot.export_csv_button.objectName(), "exportSensorPathCsvButton")
        self.assertTrue(
            any(
                label.objectName() == "dialogHeading"
                and label.text() == "Field along sensor paths"
                for label in sensor_plot.findChildren(QLabel)
            )
        )

        sensor_plot.plot.cleanup()
        measurement.histogram.cleanup()
        sensor_plot.deleteLater()
        measurement.deleteLater()
        window.plot.cleanup()
        window.deleteLater()


    def test_ai_help_dialog_copies_guide_validates_and_imports_recipe(self):
        adapter = StudioAdapter({"objects": []})
        dialog = AiHelpDialog(adapter)
        self.assert_standard_window_controls(dialog)
        self.assertTrue(dialog.include_scene.isChecked())
        self.assertTrue(dialog.include_drc.isChecked())
        self.assertFalse(hasattr(dialog, "import_mode"))
        self.assertEqual(dialog.guidance_level.count(), 3)
        self.assertEqual(dialog.guidance_level.currentData(), "balanced")
        self.assertEqual(dialog.guidance_level.itemData(0), "exploratory")
        self.assertEqual(dialog.guidance_level.itemData(2), "strict")
        self.assertFalse(dialog.modify_current_button.isEnabled())
        self.assertFalse(dialog.paste_new_project_button.isEnabled())
        self.assertEqual(dialog.modify_current_button.text(), "Modify current scene")
        self.assertEqual(dialog.paste_new_project_button.text(), "Paste into new project")
        self.assertEqual(dialog.copy_guide_button.text(), "Copy scene + AI hints")
        self.assertEqual(dialog.validate_button.text(), "Validate AI response")
        self.assertEqual(dialog.recipe_edit.minimumHeight(), 80)
        self.assertEqual(dialog.recipe_edit.maximumHeight(), 140)
        self.assertEqual(dialog.validation_output.minimumHeight(), 120)
        self.assertEqual(
            dialog.layout().sizeConstraint(),
            QLayout.SizeConstraint.SetMinimumSize,
        )
        group_titles = {group.title() for group in dialog.findChildren(QGroupBox)}
        self.assertIn("1 — Copy scene + AI hints", group_titles)
        self.assertIn("2 — Paste the AI response", group_titles)
        self.assertIn("3 — Review the AI interpretation and choose destination", group_titles)
        self.assertEqual(dialog.activity_progress.value(), 0)
        self.assertEqual(dialog.activity_progress.format(), "Ready")

        copy_progress_values = []
        dialog.activity_progress.valueChanged.connect(copy_progress_values.append)
        dialog.copy_format_guide()
        copied = QApplication.clipboard().text()
        self.assertTrue(copied.strip())
        self.assertEqual(dialog.activity_progress.value(), 100)
        self.assertIn("Format guide copied", dialog.activity_progress.format())
        self.assertIn("elapsed", dialog.activity_progress.format())
        self.assertTrue(any(20 <= value < 99 for value in copy_progress_values))
        self.assertIn(99, copy_progress_values)
        dialog.recipe_edit.setPlainText(
            json.dumps(
                {
                    "format": "fieldworkbench-scene-recipe",
                    "version": 1,
                    "assistant_message": "I made a small first-pass chat coil.",
                    "assumptions": ["The requested diameter was not specified."],
                    "suggested_next_steps": ["Add an axis sensor to inspect the field gradient."],
                    "objects": [
                        {
                            "type": "circular_coil",
                            "name": "Chat coil",
                            "diameter_mm": 120,
                            "drive_current_mA": 10,
                        }
                    ],
                }
            )
        )
        self.assertTrue(dialog.validate_recipe())
        self.assertTrue(dialog.modify_current_button.isEnabled())
        self.assertTrue(dialog.paste_new_project_button.isEnabled())
        self.assertEqual(dialog.activity_progress.value(), 100)
        self.assertIn("Validation complete", dialog.activity_progress.format())
        self.assertIn("elapsed", dialog.activity_progress.format())
        validation_text = dialog.validation_output.toPlainText()
        self.assertIn("AI interpretation", validation_text)
        self.assertIn("small first-pass chat coil", validation_text)
        self.assertIn("Assumptions", validation_text)
        self.assertIn("Suggested next steps", validation_text)
        self.assertIn("1 object(s) recognized", validation_text)
        with mock.patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            dialog.modify_current_scene()
        self.assertIn("Current scene modified", dialog.activity_progress.format())
        self.assertIn("elapsed", dialog.activity_progress.format())
        self.assertEqual(len(dialog.imported_ids), 1)
        self.assertTrue(dialog.replaced_scene)
        self.assertEqual(adapter.get_object(dialog.imported_ids[0])["label"], "Chat coil")
        dialog.deleteLater()

    def test_ai_help_dialog_can_build_validated_recipe_in_new_project(self):
        source = StudioAdapter({"objects": []})
        source.add_template("guide_box")
        source_count = len(source.list_objects())
        dialog = AiHelpDialog(source)
        dialog.recipe_edit.setPlainText(
            json.dumps(
                {
                    "format": "fieldworkbench-scene-recipe",
                    "version": 1,
                    "objects": [
                        {
                            "type": "circular_coil",
                            "name": "New-project coil",
                            "diameter_mm": 140,
                            "drive_current_mA": 12,
                        }
                    ],
                }
            )
        )
        self.assertTrue(dialog.validate_recipe())
        dialog.paste_into_new_project()
        self.assertTrue(dialog.pasted_into_new_project)
        self.assertIsNotNone(dialog.new_project_adapter)
        self.assertEqual(len(source.list_objects()), source_count)
        self.assertEqual(len(dialog.imported_ids), 1)
        self.assertEqual(
            dialog.new_project_adapter.get_object(dialog.imported_ids[0])["label"],
            "New-project coil",
        )
        dialog.deleteLater()

    def test_ai_help_dialog_review_questions_are_visible_but_nonblocking(self):
        adapter = StudioAdapter({"objects": []})
        dialog = AiHelpDialog(adapter)
        dialog.recipe_edit.setPlainText(
            json.dumps(
                {
                    "format": "fieldworkbench-scene-recipe",
                    "version": 1,
                    "review_questions": [
                        "Confirm the final centre-to-centre spacing before fabrication."
                    ],
                    "objects": [
                        {
                            "type": "circular_coil",
                            "name": "Reviewable coil",
                            "diameter_mm": 120,
                        }
                    ],
                }
            )
        )
        self.assertTrue(dialog.validate_recipe())
        self.assertTrue(dialog.modify_current_button.isEnabled())
        self.assertTrue(dialog.paste_new_project_button.isEnabled())
        validation_text = dialog.validation_output.toPlainText()
        self.assertIn("Worth confirming", validation_text)
        self.assertIn("Confirm the final centre-to-centre spacing", validation_text)
        dialog.deleteLater()

    def test_ai_help_dialog_surfaces_blocking_clarification_and_disables_import(self):
        adapter = StudioAdapter({"objects": []})
        dialog = AiHelpDialog(adapter)
        dialog.recipe_edit.setPlainText(
            json.dumps(
                {
                    "format": "fieldworkbench-scene-recipe",
                    "version": 1,
                    "assistant_message": "The source gives coil dimensions but not a unique assembly orientation.",
                    "requires_clarification": True,
                    "clarifying_questions": [
                        "Are the two racetrack coil planes parallel?",
                        "Which axis separates their centres?",
                    ],
                    "objects": [],
                }
            )
        )
        self.assertTrue(dialog.validate_recipe())
        self.assertFalse(dialog.modify_current_button.isEnabled())
        self.assertFalse(dialog.paste_new_project_button.isEnabled())
        validation_text = dialog.validation_output.toPlainText()
        self.assertIn("CLARIFICATION REQUIRED", validation_text)
        self.assertIn("Are the two racetrack coil planes parallel?", validation_text)
        self.assertIn("No scene changes will be imported", validation_text)
        self.assertIn("Clarification required", dialog.activity_progress.format())
        dialog.deleteLater()

    def test_probe_library_exposes_bundled_sources_and_custom_creator(self):
        dialog = ProbeLibraryDialog()
        self.assertEqual(dialog.probe_list.topLevelItemCount(), 2)
        first = dialog.probe_list.topLevelItem(0)
        self.assertEqual(first.text(0), "SENSYS GmbH")
        self.assertEqual(first.text(1), "FGM3D")
        details = dialog.details.toPlainText()
        self.assertIn("26 × 26 × 149 mm", details)
        self.assertIn("Geometric centre", details)
        self.assertIn("Sticker/label face: -Y", details)
        self.assertIn("Cable connection face: -Z", details)
        self.assertIn("14.5", details)
        self.assertIn("54.5", details)

        second = dialog.probe_list.topLevelItem(1)
        self.assertEqual(second.text(0), "MEDA, Inc.")
        self.assertEqual(second.text(1), "FM300 / FVM400")
        dialog.probe_list.setCurrentItem(second)
        meda_details = dialog.details.toPlainText()
        self.assertIn("101.6 × 25.4 × 25.4 mm", meda_details)
        self.assertIn("Geometric centre", meda_details)
        self.assertIn("Sticker/label face: -Z", meda_details)
        self.assertIn("Cable connection face: -X", meda_details)
        self.assertIn("11.684, 0, -0.254", meda_details)
        self.assertIn("33.528, 0, -0.254", meda_details)
        self.assertIn("MEDA FVM400 product page", meda_details)
        self.assertIn("MEDA FVM400 Instruction Manual", meda_details)
        self.assertIn("MEDA company history", meda_details)
        self.assertTrue(dialog.findChild(QPushButton, "createCustomProbeButton"))
        dialog.close()
        dialog.deleteLater()

        custom = CustomProbeDialog()
        self.assertEqual(custom.channel_table.rowCount(), 3)
        self.assertEqual(custom.channel_table.item(0, 0).text(), "X")
        custom.sticker_face.setCurrentIndex(custom.sticker_face.findData("-X"))
        custom.sticker_label.setText("LABEL SIDE")
        custom.cable_face.setCurrentIndex(custom.cable_face.findData("+Z"))
        custom._accept_definition()
        definition = custom.probe_definition()
        self.assertEqual(
            [channel["name"] for channel in definition["channels"]],
            ["X", "Y", "Z"],
        )
        self.assertEqual(definition["housing"]["sticker_face"], "-X")
        self.assertEqual(definition["housing"]["sticker_label"], "LABEL SIDE")
        self.assertEqual(definition["housing"]["cable_face"], "+Z")
        custom.close()
        custom.deleteLater()

    def test_magnetometer_probe_add_flow_and_inspector_reading(self):
        window = self.make_window()
        dialog = mock.Mock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted
        from fieldworkbench.probe_models import builtin_probe_definitions

        dialog.probe_definition.return_value = builtin_probe_definitions()[0]
        with mock.patch.object(main_window_module, "ProbeLibraryDialog", return_value=dialog):
            window.add_object("magnetometer_probe")

        probe_id = window.selected_id
        self.assertTrue(window.adapter.is_magnetometer_probe(probe_id))
        self.assertEqual(window._tree_items[probe_id].text(1), "Magnetometer probe")
        section_titles = [
            section.header.text()
            for section in window.inspector_container.findChildren(CollapsibleSection)
        ]
        self.assertIn("Probe model", section_titles)
        self.assertIn("Simulated probe reading", section_titles)
        inspector_text = "\n".join(
            label.text() for label in window.inspector_container.findChildren(QLabel)
        )
        self.assertIn("-Y — STICKER SIDE", inspector_text)
        self.assertIn("-Z — red end face", inspector_text)
        calculate = window.inspector_container.findChild(
            QPushButton, "calculateMagnetometerProbeButton"
        )
        self.assertIsNotNone(calculate)
        calculate.click()
        self.application.processEvents()
        self.assertTrue(
            all("µT" in label.text() for label in window.probe_reading_labels.values())
        )
        self.assertIn("µT", window.probe_magnitude_label.text())
        window.plot.cleanup()
        window.deleteLater()

    def test_help_menu_topics_and_statistics_reference(self):
        window = self.make_window()
        self.assertEqual(
            [
                "|" if action.isSeparator() else action.text()
                for action in window.help_menu.actions()
            ],
            ["Help Topics…", "AI Help…", "|", "About Field Workbench…"],
        )
        self.assertEqual(
            window.help_topics_action.shortcut(),
            QKeySequence(QKeySequence.StandardKey.HelpContents),
        )

        dialog = HelpTopicsDialog(window)
        self.assert_standard_window_controls(dialog)
        self.assertGreaterEqual(dialog.topic_count, 30)
        self.assertIn("field_map_2d", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["field_map_2d"][0], "Field views")
        self.assertIn("regular <b>XY, XZ, or YZ plane</b>", HELP_TOPICS["field_map_2d"][2])
        self.assertIn("field_map_3d", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["field_map_3d"][0], "Field views")
        self.assertIn("brain_view", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["brain_view"][0], "Field views")
        self.assertIn("Filter</b> is an exclusion rule", HELP_TOPICS["brain_view"][2])
        self.assertIn("lpba40_model", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["lpba40_model"][0], "Brain model and atlas")
        self.assertIn("56 non-background anatomical", HELP_TOPICS["lpba40_model"][2])
        self.assertIn("playback_overview", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["playback_overview"][0], "Playback")
        self.assertIn("saved_exposures", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["saved_exposures"][0], "Exposures")
        self.assertIn("exposure_analysis", HELP_TOPICS)
        self.assertIn("Exposures", CATEGORY_ORDER)
        self.assertEqual(HELP_TOPICS["exposure_analysis"][0], "Maps / Analysis")
        self.assertIn("Create new analysis", HELP_TOPICS["exposure_analysis"][2])
        self.assertIn("Analyze exposure…", HELP_TOPICS["exposure_analysis"][2])
        self.assertIn("spectrum_analysis", HELP_TOPICS)
        self.assertIn("View FFT…", HELP_TOPICS["spectrum_analysis"][2])
        self.assertIn("timing_analysis", HELP_TOPICS)
        self.assertIn("threshold_duration", HELP_TOPICS)
        self.assertIn("frequency_weighted", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["frequency_weighted"][0], "Exposures")
        self.assertIn("weighted-peak", HELP_TOPICS["frequency_weighted"][2])
        self.assertIn("playback_statistics", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["playback_statistics"][0], "Statistics and exposure metrics")
        self.assertEqual(HELP_TOPICS["playback_statistics"][1], "Exposure statistics and research basis")
        self.assertIn("Exposure RMS |B|", HELP_TOPICS["playback_statistics"][2])
        self.assertIn("B²-time", HELP_TOPICS["playback_statistics"][2])
        self.assertIn("descriptive magnetic-field metrics", HELP_TOPICS["playback_statistics"][2])
        self.assertNotIn("retained regional-frame cache", HELP_TOPICS["playback_statistics"][2])
        self.assertIn("ai_scene_recipe", HELP_TOPICS)
        self.assertIn("Scene Recipe v1", HELP_TOPICS["ai_scene_recipe"][2])
        self.assertIn("head_frame", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["head_frame"][0], "Coordinates and anatomy")
        self.assertIn("helix", HELP_TOPICS["head_frame"][2])
        self.assertIn("generic_bust", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["generic_bust"][0], "Coordinates and anatomy")
        self.assertIn("NAS/LPA/RPA", HELP_TOPICS["generic_bust"][2])
        self.assertIn("Human Base Meshes v1.4.1", HELP_TOPICS["generic_bust"][2])
        self.assertIn("Authored surface regions", HELP_TOPICS["generic_bust"][2])
        self.assertIn("generic_bust_v1.regions.json", HELP_TOPICS["generic_bust"][2])
        self.assertIn("custom_surface_regions", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["custom_surface_regions"][0], "Design tools")
        self.assertIn("arbitrary", HELP_TOPICS["custom_surface_regions"][2])
        self.assertIn("Copy Blender region exporter", HELP_TOPICS["custom_surface_regions"][2])
        self.assertIn("sri24_brain", HELP_TOPICS)
        self.assertEqual(HELP_TOPICS["sri24_brain"][0], "Brain model and atlas")
        self.assertIn("SRI24 v2.0", HELP_TOPICS["sri24_brain"][2])
        self.assertIn("CC BY-SA 3.0", HELP_TOPICS["sri24_brain"][2])
        self.assertIn("sri24_registration", HELP_TOPICS)
        self.assertIn("119.89204464", HELP_TOPICS["sri24_registration"][2])
        self.assertIn("sri24_fiducials_4.jpg", HELP_TOPICS["sri24_registration"][2])
        self.assertIn("upenn_catalogue", HELP_TOPICS)
        self.assertIn("upenn_regions", HELP_TOPICS)
        self.assertIn("glossary", HELP_TOPICS)
        dialog.select_topic("measurement_stats")
        self.application.processEvents()
        contents = dialog.browser.toPlainText()
        self.assertIn("Coefficient of variation", contents)
        self.assertIn("100 × (P95 − P5) ÷ |mean|", contents)
        # Search by the stable public topic title rather than pinning a phrase
        # from the optimizer help body, which is expected to evolve with the UI.
        optimizer_title = HELP_TOPICS["optimizer"][1]
        self.assertTrue(dialog.filter_indicator.isHidden())
        dialog.search_edit.setText(optimizer_title)
        self.application.processEvents()
        self.assertFalse(dialog.filter_indicator.isHidden())
        self.assertEqual(dialog.filter_indicator.text(), "Filtered")
        visible_titles = [
            item.text(0)
            for item in dialog._topic_items.values()
            if not item.isHidden()
        ]
        self.assertIn(optimizer_title, visible_titles)
        dialog.search_edit.clear()
        self.application.processEvents()
        self.assertTrue(dialog.filter_indicator.isHidden())
        dialog.deleteLater()
        window.plot.cleanup()
        window.deleteLater()



if __name__ == "__main__":
    unittest.main()
