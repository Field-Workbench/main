"""Static checks for release-build configuration."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_windows_build_collects_only_required_plot_stack_and_uses_splash(self) -> None:
        script = (PROJECT_ROOT / "build_windows.bat").read_text(encoding="utf-8")
        self.assertIn("--windowed", script)
        self.assertIn("--hidden-import=magpylib._src.display.backend_plotly", script)
        self.assertIn("--hidden-import=pygeomag.wmm.wmm_2025", script)
        self.assertIn("--copy-metadata=pygeomag", script)
        self.assertIn("--collect-data=plotly", script)
        self.assertIn("--collect-data=matplotlib", script)
        self.assertIn("--collect-submodules=plotly.graph_objs", script)
        self.assertNotIn("--collect-submodules=magpylib._src.display", script)
        self.assertNotIn("--collect-all=plotly", script)
        self.assertNotIn("--exclude-module=matplotlib", script)
        self.assertNotIn("--exclude-module=PIL", script)
        self.assertIn("--exclude-module=pandas", script)
        self.assertIn('--splash "fieldworkbench\\assets\\splash.png"', script)
        self.assertIn(
            '--add-data "fieldworkbench\\assets\\generic_bust_v1.stl;fieldworkbench\\assets"',
            script,
        )
        self.assertIn(
            '--add-data "fieldworkbench\\assets\\generic_bust_v1.regions.json;fieldworkbench\\assets"',
            script,
        )
        self.assertIn(
            '--add-data "fieldworkbench\\assets\\sri24_brain_gmwm_v1.stl;fieldworkbench\\assets"',
            script,
        )
        self.assertIn(
            '--add-data "fieldworkbench\\assets\\sri24_labels;fieldworkbench\\assets\\sri24_labels"',
            script,
        )
        self.assertIn(
            '--add-data "fieldworkbench\\assets\\help;fieldworkbench\\assets\\help"',
            script,
        )
        self.assertIn(
            '--add-binary "fieldworkbench\\assets\\ffmpeg\\windows-x86_64\\ffmpeg.exe;fieldworkbench\\assets\\ffmpeg\\windows-x86_64"',
            script,
        )
        self.assertIn(
            "dist\\FieldWorkbench\\_internal\\fieldworkbench\\assets\\ffmpeg\\windows-x86_64\\ffmpeg.exe",
            script,
        )
        self.assertIn("licenses\\FFmpeg-n8.1.2-x264", script)
        self.assertIn("collect_release_licenses.py", script)
        self.assertIn("audit_windows_build.py", script)
        self.assertIn("set FWB_MAX_BUILD_MB=800", script)

        splash = PROJECT_ROOT / "fieldworkbench" / "assets" / "splash.png"
        self.assertTrue(splash.is_file())
        self.assertEqual(splash.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        bust = PROJECT_ROOT / "fieldworkbench" / "assets" / "generic_bust_v1.stl"
        self.assertTrue(bust.is_file())
        self.assertGreater(bust.stat().st_size, 84)
        bust_regions = PROJECT_ROOT / "fieldworkbench" / "assets" / "generic_bust_v1.regions.json"
        self.assertTrue(bust_regions.is_file())
        self.assertGreater(bust_regions.stat().st_size, 1_000)
        self.assertIn("fieldworkbench.surface_regions", bust_regions.read_text(encoding="utf-8"))
        brain = PROJECT_ROOT / "fieldworkbench" / "assets" / "sri24_brain_gmwm_v1.stl"
        self.assertTrue(brain.is_file())
        self.assertGreater(brain.stat().st_size, 84)
        atlas_dir = PROJECT_ROOT / "fieldworkbench" / "assets" / "sri24_labels"
        for name in (
            "lpba40.nii.gz",
            "lpba40_meshes.npz",
            "tzo116plus.nii.gz",
            "suptent.nii.gz",
            "LPBA40-labels.txt",
            "SRI24-tzo116plus.txt",
            "LICENSE",
            "manifest.json",
        ):
            self.assertTrue((atlas_dir / name).is_file(), name)
        for name in ("lpba40.nii.gz", "tzo116plus.nii.gz", "suptent.nii.gz"):
            self.assertEqual((atlas_dir / name).read_bytes()[:2], b"\x1f\x8b")
        linux_script = (PROJECT_ROOT / "build_linux.sh").read_text(encoding="utf-8")
        self.assertIn(
            "--add-data fieldworkbench/assets/sri24_labels:fieldworkbench/assets/sri24_labels",
            linux_script,
        )
        help_source = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(
            encoding="utf-8"
        )
        exporter = PROJECT_ROOT / "fieldworkbench" / "assets" / "help" / "blender_surface_regions_export.py"
        self.assertTrue(exporter.is_file())
        exporter_source = exporter.read_text(encoding="utf-8")
        self.assertIn("fieldworkbench.surface_regions", exporter_source)
        self.assertIn("material-slot face assignments", exporter_source)
        self.assertIn("mesh_sha256", exporter_source)
        self.assertIn("custom_surface_regions", help_source)
        self.assertIn("Copy Blender region exporter", help_source)

        for number in range(1, 5):
            name = f"sri24_fiducials_{number}.jpg"
            image = PROJECT_ROOT / "fieldworkbench" / "assets" / "help" / name
            self.assertTrue(image.is_file(), name)
            self.assertGreater(image.stat().st_size, 10_000)
            self.assertEqual(image.read_bytes()[:3], b"\xff\xd8\xff")
            self.assertIn(name, help_source)

    def test_static_and_waveform_3d_views_share_complete_grid_contract(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        grid = (PROJECT_ROOT / "fieldworkbench" / "field_grid.py").read_text(encoding="utf-8")
        self.assertIn('"show_grid": self.show_grid.isChecked()', dialogs)
        self.assertIn("field_volume_grid_spec", adapter)
        self.assertIn("field_volume_grid_spec", waveform)
        self.assertIn('"grid": grid_spec', waveform)
        self.assertIn('scene_payload.setdefault("lines", []).append(', waveform)
        self.assertIn("FIELD_VOLUME_AXIS_LABELS", grid)
        self.assertIn("function drawGrid(M)", gpu)
        self.assertIn("function drawGridOverlay(M)", gpu)
        self.assertIn("tick_values_mm", gpu)

    def test_brain_view_is_bundled_with_lpba40_analysis_and_field_map_playback(self) -> None:
        brain_analysis = (
            PROJECT_ROOT / "fieldworkbench" / "brain_analysis.py"
        ).read_text(encoding="utf-8")
        brain_view = (PROJECT_ROOT / "fieldworkbench" / "brain_view.py").read_text(
            encoding="utf-8"
        )
        main_window = (
            PROJECT_ROOT / "fieldworkbench" / "main_window.py"
        ).read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(
            encoding="utf-8"
        )
        render_process = (
            PROJECT_ROOT / "fieldworkbench" / "render_process.py"
        ).read_text(encoding="utf-8")
        gpu = (
            PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py"
        ).read_text(encoding="utf-8")
        plot_view = (
            PROJECT_ROOT / "fieldworkbench" / "plot_view.py"
        ).read_text(encoding="utf-8")

        self.assertIn('("3d_view", "3D View", self.field_volume_map_action)', main_window)
        self.assertIn('("brain_view", "Brain view", self.brain_view_action)', main_window)
        self.assertLess(
            main_window.index('("3d_view", "3D View", self.field_volume_map_action)'),
            main_window.index('("brain_view", "Brain view", self.brain_view_action)'),
        )
        self.assertNotIn('self.atlas_selection', brain_view)
        self.assertNotIn('self.atlas_alignment_note', brain_view)
        self.assertIn('CollapsibleSection("Map volume", expanded=False)', brain_view)
        self.assertIn('self.coordinates: list[QDoubleSpinBox]', brain_view)
        self.assertIn('self._brain_centre_mm = [0.0, 0.0, 50.0]', brain_view)
        self.assertIn('self._brain_span_mm = 300.0', brain_view)
        self.assertIn('def showEvent(self, event)', brain_view)
        self.assertIn('if not self._selector_figure_initialized:', brain_view)
        self.assertIn('self._render_selector(structural_update=True)', brain_view)
        self.assertIn('Do not build the selector figure in the constructor', brain_view)
        self.assertIn('self.span = _spin(float(self._brain_span_mm)', brain_view)
        self.assertIn('"centre_mm": self._map_volume_centre_mm()', brain_view)
        self.assertIn('"span_mm": self._map_volume_span_mm()', brain_view)
        self.assertIn('"field_workbench_brain_map_volume": True', brain_view)
        brain_mesh = (PROJECT_ROOT / "fieldworkbench" / "brain_mesh.py").read_text(encoding="utf-8")
        mesh_builder = (
            PROJECT_ROOT / "tools" / "build_lpba40_mesh_cache.py"
        ).read_text(encoding="utf-8")
        studio_adapter = (
            PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn("load_lpba40_region_meshes", brain_mesh)
        self.assertIn("lpba40_meshes.npz", brain_mesh)
        self.assertNotIn("vtkmodules", brain_mesh)
        self.assertIn("vtkSurfaceNets3D", mesh_builder)
        self.assertIn("LPBA40_MESH_CACHE_STRIDE", mesh_builder)
        self.assertIn("self.adapter = adapter.detached_copy()", brain_view)
        self.assertIn("field_map_incremental_preview_scene", studio_adapter)
        self.assertIn("self._region_trace_index", brain_view)
        self.assertIn("self.plot.apply_incremental_update", brain_view)
        self.assertIn("Plotly.restyle", plot_view)
        self.assertIn("Plotly.update", plot_view)
        self.assertIn("Plotly.relayout", plot_view)
        self.assertIn("delta_trace_updates", plot_view)
        self.assertIn("_incremental_trace_groups", plot_view)
        self.assertNotIn("fieldWorkbenchLatestIncrementalUpdate", plot_view)
        self.assertIn('"type": "mesh3d"', brain_view)
        self.assertIn('f"brain-region:{int(label_id)}"', brain_view)
        self.assertIn("self.plot.selection_requested.connect", brain_view)
        self.assertIn("_brain_view_scene_object_choices", brain_view)
        self.assertIn("_mark_selector_context_noninteractive", brain_view)
        self.assertIn('meta["field_workbench_ignore_picking"] = True', brain_view)
        self.assertIn('trace["hoverinfo"] = "skip"', brain_view)
        self.assertIn("fieldWorkbenchApplyPickingPolicy", plot_view)
        self.assertIn("object.__fieldWorkbenchOriginalDrawPick", plot_view)
        self.assertIn("object.drawPick = null", plot_view)
        self.assertIn("self._brain_view_sri24_brain_id()", main_window)
        self.assertIn("BrainViewDialog(source_document.adapter, brain_object_id)", main_window)
        self.assertIn("Please select the SRI24 brain to use for Brain View", main_window)
        self.assertIn("self.brain_view_action.setEnabled(True)", main_window)
        self.assertNotIn("def _ensure_sri24_brain", brain_view)
        self.assertIn("QHeaderView.ResizeMode.Interactive", brain_view)
        self.assertIn('header.setObjectName("brainRegionHeader")', brain_view)
        self.assertIn("header.setMinimumHeight(30)", brain_view)
        self.assertIn("display_layout.setColumnStretch(2, 0)", brain_view)
        self.assertIn("label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)", brain_view)
        self.assertIn("QHeaderView#brainRegionHeader::up-arrow", brain_view)
        self.assertIn("QHeaderView#brainRegionHeader::down-arrow", brain_view)
        self.assertIn('arrow = "▲" if order == Qt.SortOrder.AscendingOrder else "▼"', brain_view)
        self.assertIn("Qt.ContextMenuPolicy.CustomContextMenu", brain_view)
        self.assertIn("self.region_column_menu", brain_view)
        self.assertIn("self.region_filter_table", brain_view)
        self.assertIn('["#", "Metric", "Condition", "Value", "Action"]', brain_view)
        self.assertIn("_EditorWheelBlockFilter", brain_view)
        self.assertIn("scroll_bar.setValue(scroll_bar.value() - round(distance))", brain_view)
        self.assertIn("self._selected_filter_row", brain_view)
        self.assertIn("self._filter_wheel_filter = _EditorWheelBlockFilter", brain_view)
        self.assertIn("Click here to select this filter", brain_view)
        self.assertIn("QAbstractItemView.SelectionMode.NoSelection", brain_view)
        self.assertIn("self._selected_region_filter_row", brain_view)
        self.assertIn('"Peak |B|"', brain_view)
        self.assertIn('action.addItem("Filter", "filter")', brain_view)
        self.assertIn('action.addItem("Highlight", "highlight")', brain_view)
        self.assertIn('QColorDialog.getColor', brain_view)
        self.assertIn('BRAIN_FILTER_DEFAULT_HIGHLIGHT', brain_view)
        self.assertIn('"highlights": highlights', brain_view)
        self.assertIn('brainRowHighlightColour', gpu)
        self.assertIn("metrics.peak_uT", brain_view)
        self.assertIn('watched.property("brainFilterRow")', brain_view)
        self.assertIn("Qt.FocusPolicy.ClickFocus", brain_view)
        self.assertNotIn("region_filter_table.itemSelectionChanged.connect", brain_view)
        self.assertIn('QPushButton("Add")', brain_view)
        self.assertIn('QPushButton("Remove")', brain_view)
        self.assertNotIn('self.region_trace_select_major_button', brain_view)
        self.assertNotIn('self.region_trace_clear_button', brain_view)
        self.assertIn("def _trace_region_tree_payload", brain_view)
        self.assertIn('"trace_region_tree": self._trace_region_tree_payload', brain_view)
        self.assertIn('"trace_regions": default_trace_regions', brain_view)
        self.assertIn('"highlight_region_label_ids": list(selected_ids)', brain_view)
        self.assertIn("_BackgroundFieldRenderMixin", brain_view)
        self.assertIn('render_kind="brain_static"', brain_view)
        self.assertIn('defer_payload_build=primary_action == "playback"', brain_view)
        self.assertIn('render_kind="waveform"', brain_view)
        self.assertIn('"_return_field_vectors": True', brain_view)
        self.assertIn('render_kind = str(job.get("render_kind", ""))', brain_view)
        self.assertIn('job["brain_static_metrics"]', brain_view)
        self.assertIn('job["brain_frame_metrics"]', brain_view)
        self.assertIn('class _BrainResultRegionPanel', brain_view)
        self.assertIn('QRadioButton("Current frame")', brain_view)
        self.assertIn('QRadioButton("Playback summary")', brain_view)
        self.assertIn('"B-time", "B²-time"', brain_view)
        self.assertIn('"playback_summary"', waveform)
        self.assertIn('lpba40_playback_region_metrics', waveform)
        self.assertGreaterEqual(
            brain_view.count('QSplitter(Qt.Orientation.Vertical)'), 2
        )
        self.assertIn('QPushButton("Collapse")', brain_view)
        self.assertIn('QPushButton("Expand")', brain_view)
        self.assertIn('_collapse_brain_region_tree_to_major', brain_view)
        self.assertIn('_expand_brain_region_tree', brain_view)
        self.assertIn('self.column_menu = QMenu(self.table)', brain_view)
        self.assertIn('header.customContextMenuRequested.connect(self._show_column_menu)', brain_view)
        self.assertIn('class _BrainResultTab', brain_view)
        self.assertIn('viewer.frameChanged.connect(panel.request_frame)', brain_view)
        self.assertIn('panel.region_selected.connect(viewer.set_brain_region_highlight)', brain_view)
        self.assertIn('def set_filter_rules(', brain_view)
        self.assertIn('def _start_brain_render_job(', brain_view)
        self.assertIn('selector_filter_rules = copy.deepcopy(self._region_filter_rules())', brain_view)
        self.assertIn('job["initial_brain_region_filter_rules"] = selector_filter_rules', brain_view)
        self.assertIn('panel.set_filter_rules(copy.deepcopy(initial_filter_rules))', brain_view)
        self.assertIn('class _ToggleSelectedTreeRowFilter', brain_view)
        self.assertIn('self.table.viewport().installEventFilter(self._row_toggle_filter)', brain_view)
        self.assertIn('self.region_table.viewport().installEventFilter(self._selector_row_toggle_filter)', brain_view)
        self.assertIn('tree.setCurrentItem(None)', brain_view)
        theme = (PROJECT_ROOT / "fieldworkbench" / "theme.py").read_text(encoding="utf-8")
        self.assertIn("QWidget {", theme)
        self.assertIn("background-color: transparent;", theme)
        self.assertIn("QTabWidget::pane {", theme)
        dark_source = theme.split('\nWORKBENCH_STYLE_SHEET = """', 1)[1]
        self.assertNotIn("brainRegionPanel", dark_source)
        self.assertNotIn("brainWorkspacePage", brain_view)
        self.assertNotIn("setAutoFillBackground(True)", brain_view)
        self.assertIn('return "", "", ()', brain_view)
        self.assertIn('tree.destroyed.connect(self._tree_destroyed)', brain_view)
        self.assertNotIn('watched is self._tree.viewport()', brain_view)
        self.assertNotIn("self.region_filter_note =", brain_view)
        self.assertNotIn("self.region_sort =", brain_view)
        self.assertNotIn("self.display_mode", brain_view)
        for label in (
            "Static analysis",
            "Animated analysis",
            "Export video",
            "Reset defaults",
            "Exit",
        ):
            self.assertIn(f'QPushButton("{label}")', brain_view)
        self.assertIn("lpba40_region_hierarchy", brain_analysis)
        self.assertIn("weights_mm3", brain_analysis)
        self.assertIn("weighted_rms_coefficients", brain_analysis)
        self.assertIn('"waveform_render_mode": BRAIN_FIELD_RENDER_MODE', brain_view)
        self.assertIn('"highlight_selected_region"', brain_view)
        self.assertIn('lpba40_frame_region_metrics', waveform)
        self.assertIn('"brain_region_frame_metrics": brain_region_frame_metrics_payload', waveform)
        self.assertIn('"brain_region_video_table": copy.deepcopy(map_settings.get("brain_region_video_table"))', waveform)
        self.assertIn('def _brain_region_video_table_snapshot', brain_view)
        self.assertIn('def video_frame_metric_payload', brain_view)
        self.assertIn('result_tab.region_panel.video_table_snapshot()', brain_view)
        self.assertIn('result_tab.region_panel.video_frame_metric_payload()', brain_view)
        self.assertIn('viewer_actions.addWidget(self.export_video_button)', brain_view)
        self.assertNotIn('action_layout.addWidget(self.export_video_button)', brain_view)
        self.assertIn('map_settings["brain_region_video_table"] = self._video_region_table_snapshot()', brain_view)
        self.assertIn('primary_action in {"playback", "export"}', brain_view)
        video_dialog = (
            PROJECT_ROOT / "fieldworkbench" / "waveform_video_dialog.py"
        ).read_text(encoding="utf-8")
        self.assertIn('format_form.addRow("Brain areas", self.brain_table_summary)', video_dialog)
        self.assertIn('drawBrainVideoTable', gpu)
        self.assertIn('brainVisibleRows', gpu)
        self.assertIn('brainTableMode', gpu)
        self.assertIn('brainSummaryValues', gpu)
        self.assertIn('brain_region_video_table', gpu)
        self.assertIn('map_settings.get("trace_regions")', waveform)
        waveform_dialog = (
            PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py"
        ).read_text(encoding="utf-8")
        self.assertIn('QLabel("Brain region traces")', waveform_dialog)
        self.assertIn('QPushButton("Select major")', waveform_dialog)
        self.assertIn('QPushButton("Clear")', waveform_dialog)
        self.assertIn('Qt.ItemFlag.ItemIsUserCheckable', waveform_dialog)
        self.assertIn('settings["trace_regions"] = trace_regions', waveform_dialog)
        self.assertIn('settings.pop("trace_region_tree", None)', waveform_dialog)
        self.assertIn('"scroll_observation_traces"', waveform)
        self.assertIn('render_mode not in {"slices", "volume", "points"}', waveform)
        self.assertIn('if render_mode == "points":', waveform)
        self.assertIn('"point_count": int(len(map_points_mm)) if render_mode == "points" else 0', waveform)
        self.assertIn('str(render_kind) in {"static", "brain_static"}', render_process)
        self.assertIn("scrollObservationTraces", gpu)
        self.assertIn("obsWrap.style.overflowY", gpu)
        self.assertIn("gl.POINTS", gpu)
        self.assertIn("uPointSize", gpu)
        self.assertIn("frameChanged = Signal(int, float)", gpu)
        self.assertIn("fieldWorkbenchSetBrainHighlight", gpu)
        self.assertIn("overlay_point_label_ids_b64", waveform)

    def test_2d_static_and_waveform_views_share_grid_and_contour_controls(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        grid = (PROJECT_ROOT / "fieldworkbench" / "field_grid.py").read_text(encoding="utf-8")
        self.assertIn('self.show_grid = QCheckBox("Show grid")', dialogs)
        self.assertIn('"show_grid": self.show_grid.isChecked()', dialogs)
        self.assertIn('"show_contours": heatmap_visible and self.show_contours.isChecked()', dialogs)
        self.assertIn("field_plane_grid_spec", adapter)
        self.assertIn("field_plane_grid_spec", waveform)
        self.assertIn('"show_contours": bool(map_settings.get("show_contours", False))', waveform)
        self.assertIn("uShowContours", gpu)
        self.assertIn("plane_segments_mm", gpu)
        self.assertIn("drawContourLabels", gpu)
        self.assertIn("def field_plane_grid_spec", grid)

    def test_render_extent_scaffold_is_persistent_and_volumetric(self) -> None:
        plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(encoding="utf-8")
        self.assertIn("const helper = {{x: [], y: [], z: []}}", plot)
        self.assertIn("for (const sx of [-1, 1])", plot)
        self.assertIn("for (const sy of [-1, 1])", plot)
        self.assertIn("for (const sz of [-1, 1])", plot)
        self.assertIn("x: helper.x", plot)
        self.assertIn("y: helper.y", plot)
        self.assertIn("z: helper.z", plot)
        self.assertNotIn("if (!needsExtent) return source", plot)
        self.assertIn("marker: {{size: 0.1, opacity: 0.001", plot)
        self.assertIn("The scaffold follows the current geometry rather than anchoring to the", plot)

    def test_main_scene_uses_native_data_aspect_for_explicit_ranges(self) -> None:
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(encoding="utf-8")
        self.assertIn('scene["aspectmode"] = "data"', adapter)
        self.assertIn('scene.pop("aspectratio", None)', adapter)
        self.assertIn("'scene.aspectmode': 'data'", plot)
        self.assertIn("delete scene.aspectratio", plot)
        self.assertIn("Math.max(0.5, 0.5 * maxSpan)", plot)
        self.assertNotIn('figure["data"] = self._suppress_coincident_hidden_legend_lines', adapter)
        self.assertNotIn('pick_proxies = self._suppress_coincident_hidden_legend_lines', adapter)

    def test_studio_clone_guards_against_reused_historical_ids(self) -> None:
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        self.assertIn("_compact_reused_studio_event_lifetimes", adapter)
        self.assertIn("Once an id has existed there, keep it reserved", adapter)
        self.assertIn("self._replace_document(compacted, clear_history=False)", adapter)

    def test_main_scene_rebuild_auto_refits_when_visible_3d_bounds_change(self) -> None:
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn("def _figure_3d_bounds(", main)
        self.assertIn("def _figure_3d_bounds_changed(", main)
        start = main.index("    def refresh_plot(self, *, rebuild_base: bool = False) -> None:")
        end = main.index("    def _update_status(self) -> None:", start)
        refresh_source = main[start:end]
        self.assertIn("previous_bounds = _figure_3d_bounds(self._base_figure)", refresh_source)
        self.assertIn("rebuilt_base = self.adapter.base_figure()", refresh_source)
        self.assertIn("rebuilt_bounds = _figure_3d_bounds(rebuilt_base)", refresh_source)
        self.assertIn("_figure_3d_bounds_changed", refresh_source)
        self.assertIn("self.plot.request_fit_bounds_on_next_figure()", refresh_source)
        self.assertLess(
            refresh_source.index("request_fit_bounds_on_next_figure"),
            refresh_source.index("self.plot.set_figure"),
        )
        apply_start = main.index("    def apply_inspector(self) -> bool:")
        apply_end = main.index("    # --- object operations", apply_start)
        self.assertNotIn(
            "request_fit_bounds_on_next_figure",
            main[apply_start:apply_end],
        )

    def test_clone_requests_clean_plot_rebuild_for_structural_trace_change(self) -> None:
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        start = main.index("    def duplicate_selected(self) -> None:")
        end = main.index("    def delete_selected(self) -> None:", start)
        duplicate_source = main[start:end]
        self.assertIn("self.adapter.duplicate(self.selected_id)", duplicate_source)
        self.assertIn("self.plot.request_clean_rebuild_on_next_figure()", duplicate_source)
        self.assertIn("self.plot.request_fit_bounds_on_next_figure()", duplicate_source)
        self.assertLess(
            duplicate_source.index("request_clean_rebuild_on_next_figure"),
            duplicate_source.index("self.refresh_scene(select_id=object_id)"),
        )

    def test_release_licensing_files_are_present(self) -> None:
        for name in ("LICENSE.txt", "PROJECT_NOTICE.txt", "THIRD_PARTY_NOTICES.txt"):
            self.assertTrue((PROJECT_ROOT / name).is_file(), name)

        notice = (PROJECT_ROOT / "PROJECT_NOTICE.txt").read_text(encoding="utf-8")
        self.assertIn("any later version", " ".join(notice.split()))
        self.assertIn("GPL-3.0-or-later", notice)

        third_party = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
        self.assertIn("Human Base Meshes v1.4.1", third_party)
        self.assertIn("CC0 1.0 Universal", third_party)
        self.assertIn("pyGeoMag 1.1.0", third_party)
        self.assertIn("SurfaceNets3D 9.6.2", third_party)
        self.assertIn("VTK (BSD-3-Clause)", third_party)
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        mesh_requirements = (
            PROJECT_ROOT / "requirements-mesh-build.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("pygeomag==1.1.0", requirements)
        self.assertNotIn("vtk", requirements.lower())
        self.assertIn("vtk==9.6.2", mesh_requirements)
        self.assertNotIn("nibabel", requirements.lower())
        self.assertIn("generic_bust_v1.stl", third_party)
        self.assertIn("sri24_brain_gmwm_v1.stl", third_party)
        self.assertIn("sri24_labels_nifti.zip", third_party)
        self.assertIn(
            "ed47fde305baefb687a38e0daa8527896480e4e0159a0a0f56c8cbe09fc97e1f",
            third_party,
        )
        self.assertIn("tzo116plus", third_party)
        self.assertIn("LPBA40", third_party)
        self.assertIn("CC BY-SA 3.0", third_party)
        self.assertIn("10.1002/hbm.20906", third_party)
        self.assertIn("sri24_fiducials_1.jpg", third_party)
        self.assertIn("Bundled Windows video encoder", third_party)
        self.assertIn("FFmpeg n8.1.2", third_party)
        self.assertIn("x264", third_party)
        self.assertIn("zlib 1.3.2", third_party)
        ffmpeg_asset = (
            PROJECT_ROOT / "fieldworkbench" / "assets" / "ffmpeg" / "windows-x86_64" / "ffmpeg.exe"
        )
        self.assertTrue(ffmpeg_asset.is_file())
        self.assertEqual(ffmpeg_asset.stat().st_size, 4_056_576)
        for name in ("BUILD_INFO.txt", "GPL-2.0.txt", "GPL-3.0.txt", "ZLIB_LICENSE.txt"):
            self.assertTrue((PROJECT_ROOT / "third_party" / "ffmpeg" / name).is_file(), name)
        self.assertIn("UPENN-GBM tumor segmentations", third_party)
        self.assertIn("10.7937/TCIA.709X-DN49", third_party)
        self.assertIn("CC BY 4.0", third_party)

        self.assertTrue((PROJECT_ROOT / "requirements-build.txt").is_file())
        self.assertTrue((PROJECT_ROOT / "requirements-mesh-build.txt").is_file())
        self.assertTrue((PROJECT_ROOT / "tools" / "build_source_release.py").is_file())

    def test_source_release_builder_keeps_parent_directory(self) -> None:
        source = (PROJECT_ROOT / "tools" / "build_source_release.py").read_text(encoding="utf-8")
        self.assertIn('archive.write(path, Path("field_workbench") / relative)', source)

    def test_light_dark_theme_contract_uses_wavebuilder_palette_and_persists(self) -> None:
        theme_source = (PROJECT_ROOT / "fieldworkbench" / "theme.py").read_text(
            encoding="utf-8"
        )
        theme_module = ast.parse(theme_source)

        def literal_assignment(name: str):
            for node in theme_module.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in node.targets
                ):
                    return ast.literal_eval(node.value)
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == name
                ):
                    return ast.literal_eval(node.value)
            self.fail(f"Missing theme assignment: {name}")

        self.assertEqual(
            literal_assignment("THEME_CHOICES"),
            (("Light", "light"), ("Dark", "dark")),
        )
        dark = literal_assignment("DARK_THEME_COLORS")
        self.assertEqual(dark["window"], "#20252b")
        self.assertEqual(dark["panel"], "#2a3037")
        self.assertEqual(dark["panel_alternate"], "#252b31")
        self.assertEqual(dark["button"], "#303740")
        self.assertEqual(dark["border"], "#48535e")
        plots = literal_assignment("PLOT_THEME_COLORS")
        self.assertEqual(plots["dark"]["figure"], "#aeb3b8")
        self.assertEqual(plots["dark"]["text"], "#14181c")
        self.assertIn('"appearance/theme"', theme_source)
        self.assertIn("for existing_widget in application.allWidgets()", theme_source)

        app = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(
            encoding="utf-8"
        )
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("apply_workbench_theme(theme=load_theme_preference())", app)
        self.assertIn('theme_selector.setObjectName("themeSelector")', main)
        self.assertIn("save_theme_preference(theme, self.settings)", main)
        self.assertIn("def apply_workbench_theme(self, theme: str)", plot)
        self.assertIn("layout[\"paper_bgcolor\"] = colours[\"figure\"]", plot)
        self.assertIn("window.fieldWorkbenchSetTheme=theme=>", gpu)

    def test_dark_theme_repolishes_existing_controls_and_tints_toolbar_icons(self) -> None:
        theme = (PROJECT_ROOT / "fieldworkbench" / "theme.py").read_text(
            encoding="utf-8"
        )
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('application.setStyleSheet("")', theme)
        self.assertIn("existing_widget.setPalette(palette)", theme)
        self.assertIn("style.unpolish(existing_widget)", theme)
        self.assertIn("style.polish(existing_widget)", theme)
        self.assertIn("def themed_icon", theme)
        self.assertIn("CompositionMode_SourceIn", theme)
        self.assertIn('DARK_THEME_COLORS["heading"]', theme)
        self.assertIn('DARK_THEME_COLORS["disabled"]', theme)
        self.assertIn("def apply_workbench_theme(self, theme: str)", main)
        self.assertIn("button.setIcon(action.icon())", main)

    def test_light_theme_uses_established_workbench_qss_and_palette(self) -> None:
        theme = (PROJECT_ROOT / "fieldworkbench" / "theme.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("LIGHT_WORKBENCH_STYLE_SHEET", theme)
        self.assertIn("build_workbench_light_palette", theme)
        self.assertIn("application.setStyleSheet(workbench_style_sheet(selected))", theme)
        self.assertNotIn("application.style().standardPalette()", theme)

    def test_dark_theme_translation_is_global_and_does_not_preserve_white_backgrounds(self) -> None:
        theme = (PROJECT_ROOT / "fieldworkbench" / "theme.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("import re", theme)
        self.assertIn(r'r"(?<![-\w])color\s*:\s*#ffffff\s*;"', theme)
        self.assertIn('("#ffffff", "#2a3037")', theme)
        self.assertIn("QWidget {", theme)
        self.assertIn("background-color: transparent;", theme)
        # The old substring replacement matched the trailing `color` inside
        # `background-color`, leaving white Light surfaces white in Dark mode.
        self.assertNotIn(
            'WORKBENCH_STYLE_SHEET.replace(\n        "color: #ffffff;"', theme
        )

    def test_dark_surfaces_use_global_widget_class_rules_not_window_specific_patches(self) -> None:
        theme = (PROJECT_ROOT / "fieldworkbench" / "theme.py").read_text(
            encoding="utf-8"
        )
        sources = "\n".join(
            (PROJECT_ROOT / "fieldworkbench" / filename).read_text(encoding="utf-8")
            for filename in ("main_window.py", "dialogs.py", "help_topics.py", "brain_view.py")
        )

        self.assertIn("/* Global dark cards/groups", theme)
        self.assertIn("QGroupBox {\n    background: #252b31;", theme)
        self.assertIn("QGroupBox::title {\n    background: #20252b;", theme)
        self.assertIn("QDialog, QMessageBox, QProgressDialog {\n    background-color: #2f353b;", theme)
        self.assertIn("QTabWidget::pane {", theme)
        self.assertIn("QTabBar::tab:selected {", theme)
        self.assertIn("QTreeView, QTreeWidget, QListView, QListWidget {", theme)
        self.assertIn("QTableView, QTableWidget {", theme)
        self.assertIn("background-color: #9ea4aa;", theme)
        self.assertIn("QTableCornerButton::section", theme)

        obsolete_theme_hooks = (
            "addObjectPanel",
            "addObjectPalette",
            "inspectorObjectBox",
            "documentTabToolbar",
            "sceneDocumentTabs",
            "sceneObjectTree",
            "fieldRenderTaskList",
            "helpTopicsTree",
            "snapshotList",
            "measurementResultsTabs",
            "measurementResultsTabsBar",
            "measurementOverviewPage",
            "fieldMapTabs",
            "fieldMapTabsBar",
            "brainWorkspacePage",
            "brainRegionPanel",
            "brainSelectorRegionPanel",
            "brainResultRegionPanel",
            "brainSelectorTab",
            "brainResultTab",
            "brainViewTabs",
        )
        dark_source = theme.split('\nWORKBENCH_STYLE_SHEET = """', 1)[1]
        for hook in obsolete_theme_hooks:
            self.assertNotIn(hook, dark_source)
            self.assertNotIn(f'setObjectName("{hook}")', sources)

    def test_preferences_explanations_are_short_and_local(self) -> None:
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("Solver: Auto balances speed and accuracy", main)
        self.assertIn("Cleanup stops unstable fluxlines", main)
        self.assertIn("fluxline_form.addRow(fluxline_note)", main)
        self.assertNotIn("Auto first screens each coil", main)

    def test_map_action_buttons_are_outside_scrolling_control_panels(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("control_column_layout.addWidget(control_scroll, 1)"), 2)
        self.assertEqual(source.count("control_column_layout.addWidget(action_panel, 0)"), 2)
        self.assertEqual(source.count("action_layout.addWidget(self.camera_timeline_edit_widget)"), 2)
        self.assertEqual(source.count("splitter.addWidget(control_column)"), 2)
        for button in (
            "self.calculate_button",
            "self.waveform_button",
            "self.export_video_button",
            "self.reset_defaults_button",
            "self.exit_button",
        ):
            self.assertNotIn(f"control_root.addWidget({button})", source)
        self.assertEqual(source.count('self.exit_button = QPushButton("Exit")'), 2)
        self.assertEqual(source.count("action_layout.addWidget(self.exit_button)"), 2)
        self.assertEqual(
            source.count("viewer_actions_layout.addWidget(self.fullscreen_button)"),
            2,
        )
        self.assertEqual(
            source.count("viewer_actions_layout.addWidget(self.export_video_button)"),
            2,
        )
        self.assertNotIn("action_layout.addWidget(self.export_video_button)", source)
        self.assertGreaterEqual(source.count("view.export_payload()"), 2)

        module = ast.parse(source)
        classes = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.ClassDef)
        }
        for class_name in ("FieldMapDialog", "FieldVolumeMapDialog"):
            constructor = next(
                node
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            )
            constructor_source = ast.unparse(constructor)
            self.assertNotIn("QDialogButtonBox.StandardButton.Close", constructor_source)
            self.assertIn("self.exit_button.clicked.connect(self.close)", constructor_source)

    def test_map_windows_are_non_modal_application_owned_model_snapshots(self) -> None:
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(
            encoding="utf-8"
        )
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(
            encoding="utf-8"
        )
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        window_utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("def detached_copy(self)", adapter)
        self.assertIn("StudioAdapter(self.document())", adapter)
        self.assertEqual(dialogs.count("self.adapter = adapter.detached_copy()"), 2)
        self.assertIn("def show_application_window", window_utils)
        self.assertIn("def application_owned_windows", window_utils)
        self.assertIn("setParent(None, Qt.WindowType.Window)", window_utils)
        self.assertIn("Qt.WindowModality.NonModal", window_utils)
        self.assertIn("WA_DeleteOnClose", window_utils)
        application_flags = window_utils.split("_APPLICATION_WINDOW_FLAGS = (", 1)[1].split(
            ")", 1
        )[0]
        self.assertIn("Qt.WindowType.WindowCloseButtonHint", application_flags)
        self.assertNotIn("Qt.WindowType.CustomizeWindowHint", application_flags)
        self.assertIn("window.setWindowFlags(_APPLICATION_WINDOW_FLAGS)", window_utils)

        module = ast.parse(main_window)
        main_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
        )
        methods = {
            node.name: ast.unparse(node)
            for node in main_class.body
            if isinstance(node, ast.FunctionDef)
        }
        for method_name in ("open_field_map", "open_field_volume_map"):
            source = methods[method_name]
            self.assertIn("show_application_window(dialog)", source)
            self.assertNotIn("dialog.exec()", source)
            self.assertNotIn("_show_independent_dialog", source)
            self.assertNotIn("_independent_windows", source)
            self.assertIn("(snapshot)", source)

    def test_application_closes_splash_then_raises_main_window(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("multiprocessing.freeze_support()", source)
        self.assertIn("prime_field_render_process_context()", source)
        self.assertLess(
            source.index("prime_field_render_process_context()"),
            source.index("from PySide6.QtCore import"),
        )
        self.assertIn('if __name__ == "__mp_main__":', source)
        self.assertIn("import pyi_splash", source)
        self.assertIn("_close_packaged_splash()", source)
        self.assertIn("_schedule_post_splash_raise(window)", source)
        self.assertIn("QTimer.singleShot(100", source)
        self.assertIn("QTimer.singleShot(350", source)
        self.assertIn("SetForegroundWindow(hwnd)", source)
        self.assertIn("window.raise_()", source)
        self.assertIn("handle.requestActivate()", source)

    def test_calculation_progress_dialogs_use_shared_centering_helper(self) -> None:
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(encoding="utf-8")
        self.assertIn("def centre_window_on_parent", utils)
        self.assertIn("centre_window_on_parent(progress, self)", waveform)
        self.assertIn("centre_window_on_parent(progress, parent)", dialogs)

    def test_numeric_editors_use_compact_text_and_camera_ratio_stepping(self) -> None:
        utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(encoding="utf-8")
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        timeline = (PROJECT_ROOT / "fieldworkbench" / "camera_timeline.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("class CompactDoubleSpinBox(QDoubleSpinBox)", utils)
        self.assertIn("class PlaybackSpeedSpinBox(CompactDoubleSpinBox)", utils)
        self.assertIn("compact_fixed_decimal_text", utils)
        self.assertIn("stepped_playback_speed", utils)
        self.assertEqual(dialogs.count("spinbox_type=PlaybackSpeedSpinBox"), 2)
        self.assertIn("PLAYBACK_SPEED_MINIMUM", timeline)
        self.assertIn("Math.max(1e-6,Number(point.playback_speed)||1)", gpu)
        for filename in (
            "dialogs.py",
            "main_window.py",
            "waveform_dialog.py",
            "waveform_video_dialog.py",
        ):
            source = (PROJECT_ROOT / "fieldworkbench" / filename).read_text(encoding="utf-8")
            self.assertNotIn("= QDoubleSpinBox(", source, filename)

    def test_waveform_gpu_view_has_independent_looping_sequence_and_reset_control(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn('id="loop" type="checkbox"', source)
        self.assertIn("function runningPlaybackElapsed", source)
        self.assertIn("function sequenceStepAt", source)
        self.assertIn("freeFrameCurrents", source)
        self.assertIn("↺ Reset", source)
        self.assertIn("if(running)stopRun();else resetPlayback();", source)
        self.assertNotIn('id="opacity-low"', source)
        self.assertNotIn('id="opacity-high"', source)

    def test_map_and_waveform_dialogs_offer_reset_defaults(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(dialogs.count('QPushButton("Reset defaults")'), 2)
        self.assertIn('"Reset defaults", QDialogButtonBox.ButtonRole.ActionRole', waveform)
        self.assertIn("capture_window_default_settings(self)", dialogs)
        self.assertIn("capture_window_default_settings(self)", waveform)
        self.assertIn("def reset_window_to_default_settings", utils)
        self.assertIn('primary_viewer = getattr(window, "plot", None)', utils)
        self.assertIn('request_home = getattr(primary_viewer, "request_home_on_next_figure", None)', utils)
        self.assertIn('state_filter = getattr(window, "_persistent_window_state_filter", None)', utils)

    def test_waveform_fullscreen_stops_embedded_playback_first(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("def stop_playback(self)", source)
        self.assertIn("window.fieldWorkbenchWaveformStop=stopRun", source)
        self.assertIn("self.stop_playback()", source)

    def test_waveform_dialog_exposes_multi_source_signed_absolute_and_excitation_traces(self) -> None:
        dialog_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        waveform_source = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        gpu_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn('QCheckBox("Base excitation trace")', dialog_source)
        self.assertIn('QCheckBox("Sequencer trace")', dialog_source)
        self.assertIn('include_sequencer_trace=bool(', waveform_source)
        self.assertIn('QListWidget()', dialog_source)
        self.assertIn('QLabel("Add a sensor or probe to collect traces")', dialog_source)
        for component in ("magnitude", "abs_x", "abs_y", "abs_z", "x", "y", "z"):
            self.assertIn(f'"{component}"', dialog_source)
        self.assertIn("def _observation_geometries", waveform_source)
        self.assertIn('"observations": observation_payloads', waveform_source)
        self.assertIn('"excitation": excitation_payload', waveform_source)
        self.assertIn('"sequencer": sequencer_payload', waveform_source)
        self.assertIn("Array.isArray(P.observations)", gpu_source)
        self.assertIn("const sequencerTrace=P.sequencer", gpu_source)

    def test_waveform_manual_resistance_survives_restore_and_reaches_all_playback_paths(self) -> None:
        dialog_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        dialogs_source = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        brain_source = (PROJECT_ROOT / "fieldworkbench" / "brain_view.py").read_text(encoding="utf-8")
        self.assertIn(
            "if self.use_model_resistance.isChecked():\n"
            "                self.manual_resistance.setValue(float(sum(resistances) / len(resistances)))",
            dialog_source,
        )
        self.assertIn("resistance_ohm = float(self.manual_resistance.value())", dialog_source)
        self.assertIn('"electrical_by_coil": copy.deepcopy(electrical_by_coil)', dialog_source)
        self.assertIn('map_kind="2d"', dialogs_source)
        self.assertIn('map_kind="3d"', dialogs_source)
        self.assertIn('map_kind="brain"', brain_source)

    def test_waveform_dialog_uses_collapsible_waveform_sections_and_custom_wave_points(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        for title, expanded in (
            ("Waveform", "True"),
            ("Drive", "False"),
            ("Coil sequencer", "False"),
            ("Traces", "False"),
        ):
            self.assertIn(f'_CollapsibleSection("{title}", expanded={expanded})', source)
        self.assertIn('_CollapsibleSection("Wave points", expanded=True)', source)
        for choice in ("Sine", "Square", "Triangle", "Custom"):
            self.assertIn(f'QCheckBox("{choice}")', source)
        self.assertIn('self.wave_points_section.setVisible(custom)', source)
        self.assertIn('self.generated_waveform_controls.setVisible(not custom)', source)
        self.assertIn('generated_form.addRow("Amplitude", self.waveform_amplitude)', source)
        self.assertIn('generated_form.addRow("Frequency", self.waveform_frequency)', source)
        self.assertIn('generated_form.addRow("Duration", self.waveform_duration)', source)
        self.assertNotIn("Enter an instantaneous current trace directly", source)
        self.assertNotIn("Playback stays locked to wall-clock time", source)
        self.assertNotIn("stale display frames", source)
        self.assertNotIn('QGroupBox("Trace points")', source)
        self.assertNotIn('QGroupBox("Drive trace")', source)
        self.assertNotIn('_CollapsibleSection("Render options"', source)
        self.assertNotIn('Sample multiplier', source)
        self.assertIn('def _automatic_sample_multiplier', source)
        self.assertIn('playback_fps: float = 60.0', source)
        self.assertIn('playback_speed: float = 1.0', source)
        self.assertIn('self.slew_rate = _double_spin(100.0, 0.0, 1e12, 6)', source)
        self.assertLess(
            source.index('# Drive ------------------------------------------------------------'),
            source.index('# Waveform ---------------------------------------------------------'),
        )

    def test_waveform_gpu_view_keeps_playback_speed_without_misleading_base_summary(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertNotIn('id="base-speed"', source)
        self.assertNotIn('id="effective-speed"', source)
        self.assertNotIn("× real time", source)
        self.assertNotIn("× effective", source)
        self.assertIn("configuredRealTimeMultiplier*livePlaybackSpeed*timelinePlaybackSpeedAt", source)
        self.assertIn("perf.textContent='Target '+fmt(targetFps)+' FPS • GPU '", source)

    def test_waveform_gpu_sequencer_lane_uses_live_continuous_sequence_clock(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("function sequencerLaneSamples(cursorT,sequenceT)", source)
        self.assertIn("function sequenceNextBoundary(elapsed)", source)
        self.assertIn("sequenceT+(traceT-cursorT)", source)
        self.assertIn("maxSegments=4096", source)
        self.assertNotIn("for(const traceT of ts)", source)
        self.assertIn('browser_payload["sequencer"]', source)
        self.assertIn('"time_s": [times[0], times[-1]]', source)
        self.assertIn("drawSequencerLane(ctx,obsBase.W", source)
        self.assertIn("function sequencerLaneHeight()", source)
        self.assertIn("function formatSequencerDuration(durationS)", source)
        self.assertIn("'Step '+step+(duration?' - '+duration:'')", source)
        self.assertIn("activeLists", source)
        self.assertNotIn("drawSequencerLane(x,W,t0,t1,padL,padR,laneTop,sequenceHeight)", source)

    def test_3d_selector_preview_respects_show_grid(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        self.assertIn("self.show_grid.toggled.connect(self.preview_slice_volume)", source)
        self.assertIn("def _apply_selector_grid_visibility", source)
        self.assertIn("overlay = self._slice_volume_overlay()", source)
        self.assertIn('"kind": "volume_box"', source)
        self.assertIn("analysis_overlay=overlay", source)
        self.assertIn('"showticklabels": False', source)
        self.assertIn('"annotations": []', source)

    def test_waveform_gpu_view_exports_selected_traces_and_uses_explicit_scene_alpha(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        self.assertIn('id="export-traces"', source)
        self.assertIn("Export traces…", source)
        self.assertIn("QWebChannel", source)
        self.assertIn("def export_traces(self)", source)
        self.assertIn("def _write_trace_csv", source)
        self.assertIn('["Time (s)"', source)
        self.assertIn('["Step number", "Active coils"]', source)
        self.assertIn("drawSequencerLane", source)
        self.assertIn("sceneAlpha(item)", source)
        self.assertIn('"opacity": opacity', waveform)
        # Transparent scene geometry must not be camera-sorted or write depth.
        # Both behaviours can make nested translucent anatomy pop in/out as the
        # camera crosses the sort boundary between objects.
        self.assertIn("gl.depthMask(false);", source)
        self.assertIn("for(const m of sceneMeshes){const a=sceneAlpha(m);if(a>.001&&a<.999)drawMeshItem(m);}", source)
        self.assertNotIn("transparent=sceneMeshes.filter", source)
        self.assertNotIn("meshCentre", source)

    def test_map_colour_limit_checkboxes_keep_full_text_width(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(
            dialogs.count("self.colour_minimum_enabled.sizeHint().width() + 8"), 2
        )
        self.assertGreaterEqual(
            dialogs.count("self.colour_maximum_enabled.sizeHint().width() + 8"), 2
        )

    def test_map_selectors_use_their_local_object_checklists_as_visibility_model(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(
            encoding="utf-8"
        )
        plot_view = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("selector_selected_ids", dialogs)
        self.assertNotIn("selector_selected_ids", main_window)
        self.assertGreaterEqual(dialogs.count("synchronize_legend_visibility=True"), 2)
        self.assertGreaterEqual(dialogs.count("field_map_preview_scene("), 2)
        self.assertGreaterEqual(dialogs.count("include_surface_regions=False"), 2)
        self.assertIn("field_workbench_map_preview_legend", adapter)
        self.assertIn("plotly_legendclick", plot_view)
        self.assertIn("requestVisibility", plot_view)
        self.assertIn("visibility_requested = Signal(str, bool)", plot_view)

    def test_3d_opacity_endpoints_are_wired_into_static_and_gpu_renderers(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("volume_opacity_lower_level", dialogs)
        self.assertIn("volume_opacity_upper_level", dialogs)
        self.assertIn('QLabel("at level")', dialogs)
        self.assertIn("_field_opacity_scale", adapter)
        self.assertIn("volume_opacity_lower_level", waveform)
        self.assertIn("uOpacityLowLevel", gpu)
        self.assertIn("uOpacityHighLevel", gpu)
        self.assertIn("opacityForStrength", gpu)
        # Full-volume playback must honour the same visible scalar window as
        # Plotly's static Volume trace (isomin/isomax). Values outside the
        # displayed range must not clamp to an endpoint colour. The realtime
        # volume should also select a dominant iso-surface-style level instead
        # of accumulating opacity through every voxel along the ray, otherwise
        # low/medium field samples hide the high-field red/orange core.
        self.assertIn("if(s<uMin||s>uMax)continue;", gpu)
        self.assertIn("uSurfaceCount", gpu)
        self.assertIn("bestStrength", gpu)
        self.assertIn("surfaceQ", gpu)
        self.assertNotIn("acc.rgb+=w*c", gpu)
        self.assertNotIn("uOpacityCorrection", gpu)

    def test_waveform_trace_invert_is_display_only(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("function observationInverted()", source)
        self.assertIn("vs=(t.values||[]).map(Number)", source)
        self.assertIn("inverted?f:1-f", source)
        self.assertIn("fmt(inverted?fieldRange[0]:fieldRange[1])", source)
        self.assertIn("fmt(inverted?fieldRange[1]:fieldRange[0])", source)
        self.assertNotIn("Number(value)*sign", source)
        self.assertNotIn("t.values=t.values.map", source)

    def test_main_inspector_uses_common_coil_section_hierarchy(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn('self._inspector_section("Magnetics", expanded=True)', source)
        self.assertIn('self._inspector_section("Coil Construction", expanded=False)', source)
        self.assertIn('self._inspector_section("Assembly", expanded=False)', source)
        self.assertIn('self._inspector_section("Estimates", expanded=False)', source)
        self.assertIn('button = QPushButton("Apply")', source)
        self.assertIn('fit_measurements = QPushButton("Data")', source)
        self.assertNotIn('QPushButton("Apply all changes")', source)
        self.assertIn('self._inspector_section_states: dict[str, bool] = {}', source)
        self.assertIn('section.header.toggled.connect(remember)', source)

    def test_main_inspector_uses_collapsible_geometry_workflow(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn('self._inspector_section("Position", expanded=True)', source)
        self.assertIn('"Geometry and Appearance", expanded=True', source)
        self.assertIn('self._inspector_section("DRC", expanded=False)', source)
        self.assertIn('self._inspector_section(\n                    "Measurement", expanded=False', source)
        self.assertIn('QCheckBox("DRC Global Override")', source)
        self.assertIn('self.measurement_box.setVisible(enabled)', source)
        self.assertIn('self.geometry_clearance_override_check.setVisible(drc_enabled)', source)



    def test_2d_and_3d_camera_controls_share_scripted_authoring_workflow(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        plot_view = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(dialogs.count('QPushButton("Set viewer")'), 2)
        self.assertIn('self.fixed_camera_widget.setVisible(False)', dialogs)
        self.assertIn('self.fixed_camera_widget.setVisible(enabled)', dialogs)
        self.assertIn('def _capture_2d_view_camera', dialogs)
        self.assertIn('def _plotly_view_from_2d_camera', dialogs)
        self.assertIn('live_2d_view_changed.connect', dialogs)
        self.assertIn('Timeline step active — move the 2D map viewer, then click Accept', dialogs)
        self.assertIn('def request_2d_view', plot_view)
        self.assertIn('live_2d_view_changed = Signal(dict)', plot_view)
        self.assertIn("view['xaxis.range'] || view['yaxis.range']", plot_view)
        self.assertIn('if(nativeBridge&&nativeBridge.cameraChanged)', gpu)

    def test_camera_timeline_is_shared_by_main_map_animation_and_video_export(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        timeline = (PROJECT_ROOT / "fieldworkbench" / "camera_timeline.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        video = (PROJECT_ROOT / "fieldworkbench" / "waveform_video_dialog.py").read_text(encoding="utf-8")
        plot_view = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(encoding="utf-8")
        waveform_dialog = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")

        self.assertIn('CollapsibleSection("Camera", expanded=False)', dialogs)
        self.assertIn('QRadioButton("Interactive (fixed export)")', dialogs)
        self.assertIn('QRadioButton("Scripted")', dialogs)
        self.assertIn('QLabel("Camera position")', dialogs)
        self.assertIn('QLabel("Look-at target (orientation)")', dialogs)
        self.assertNotIn('QLabel("Vertical field of view")', dialogs)
        self.assertIn('QPushButton("Set viewer")', dialogs)
        self.assertNotIn('QPushButton("Get position")', dialogs)
        self.assertIn('self.plot.live_3d_view_changed.connect(self._selector_camera_changed)', dialogs)
        self.assertIn('reportLiveView', plot_view)
        self.assertIn('plotly_relayouting', plot_view)
        self.assertIn('QLabel("Target FPS")', dialogs)
        self.assertIn('QLabel("Playback speed")', dialogs)
        self.assertIn('QCheckBox("Loop waveform/sequencer to fill timeline")', dialogs)
        self.assertIn('self.camera_script_loop.setVisible(enabled)', dialogs)
        self.assertIn('QTableWidget(0, 4)', dialogs)
        self.assertIn('["Step", "Time", "Speed", "Camera / view"]', dialogs)
        for label in ("Add", "Edit", "Remove", "Accept"):
            self.assertIn(f'QPushButton("{label}")', dialogs)
        self.assertIn('QLabel("Step playback speed")', dialogs)
        self.assertIn('self.camera_playback_speed.setVisible(not enabled)', dialogs)
        self.assertIn('request(received)', dialogs)
        self.assertIn('Timeline step active — move the Selector camera, then click Accept', dialogs)
        self.assertIn('self.camera_timeline_table.cellClicked.connect(self._camera_timeline_row_clicked)', dialogs)
        self.assertIn('def _camera_timeline_row_clicked', dialogs)
        self.assertIn('original_timeline=original_timeline', dialogs)
        self.assertIn('New timeline step cancelled', dialogs)
        self.assertIn('def video_camera_to_plotly_view', timeline)
        self.assertIn('def _plotly_view_updates', timeline)
        self.assertIn('deliverViewState', plot_view)
        self.assertIn('_bridge_view_state_ready', plot_view)
        self.assertNotIn('Configure timeline…', dialogs)
        self.assertNotIn("camera_timeline_dialog", dialogs)
        self.assertGreaterEqual(dialogs.count('QPushButton("Static map")'), 2)
        self.assertGreaterEqual(dialogs.count('QPushButton("Animated map")'), 2)
        self.assertGreaterEqual(dialogs.count('plot.refresh_viewport()'), 2)
        self.assertIn('def refresh_viewport(self)', gpu)
        self.assertIn('def can_export_video(self)', gpu)
        self.assertIn('def export_payload(self)', gpu)
        self.assertIn('The rendered animation payload', gpu)
        self.assertGreaterEqual(dialogs.count('QPushButton("Export video")'), 2)
        self.assertGreaterEqual(dialogs.count('QPushButton("Reset defaults")'), 2)
        self.assertIn('camera_timeline=camera_timeline', dialogs)
        self.assertIn('playback_fps=target_fps', dialogs)
        self.assertIn('playback_speed=playback_speed', dialogs)
        self.assertIn('primary_action=primary_action', dialogs)
        self.assertIn('def plotly_view_to_video_camera', timeline)
        self.assertIn('def projected_video_duration_s', timeline)
        self.assertIn('def scripted_timeline_duration_s', timeline)
        self.assertIn('def script_time_for_waveform_duration_s', timeline)
        self.assertIn('final camera pose simply holds', timeline)
        self.assertIn('def waveform_elapsed_for_script_time', timeline)
        self.assertIn('def physical_time_for_video_elapsed', timeline)
        self.assertIn('def _hermite_value', timeline)
        self.assertIn('def _active_camera_points', timeline)
        self.assertNotIn('def _smoothstep', timeline)
        self.assertIn('function timelineHermite(', gpu)
        self.assertIn('function timelineCameraAt(timeS)', gpu)
        self.assertNotIn('smoothTimelineFraction', gpu)
        self.assertIn('function timelinePlaybackSpeedAt(timeS)', gpu)
        self.assertIn('waveformAdvanceForScriptInterval', gpu)
        self.assertIn('camera_at_physical_time', video)
        self.assertIn('physical_time_for_video_elapsed', video)
        self.assertIn('def _source_frame_for_export_frame', video)
        self.assertIn('timeline_elapsed_s=self._timeline_time_for_export_frame', video)
        self.assertIn('sequence_elapsed_s=self._physical_time_for_export_frame', video)
        self.assertIn('self.camera_options_section.setVisible(False)', video)
        self.assertNotIn('_CollapsibleSection("Render options"', waveform_dialog)
        self.assertIn('def _automatic_sample_multiplier', waveform_dialog)
        self.assertIn(
            'result["waveform_duration_s"] = float(request["waveform_duration_s"])',
            waveform,
        )
        self.assertIn('self.payload.get("waveform_duration_s"', video)
        self.assertIn('<span>Video timeline</span>', gpu)
        self.assertIn('videoTimelineDuration=timelineSourceDuration/configuredRealTimeMultiplier', gpu)
        self.assertIn("formatPlaybackClock(videoT)+' / '+formatPlaybackClock(videoTimelineDuration)", gpu)
        self.assertIn("timeSlider.step=.01", gpu)
        self.assertNotIn('<span>Physical time</span>', gpu)
        self.assertIn('const stimulusLoop=!!(scriptedTimeline&&cameraTimeline&&cameraTimeline.loop_waveform);', gpu)
        self.assertIn('if(loop.checked){elapsed=((elapsed%endDuration)+endDuration)%endDuration;', gpu)
        self.assertNotIn('loop.checked=!!cameraTimeline.loop_waveform', gpu)
        self.assertIn('Repeat the entire video timeline when it reaches the end', gpu)
        self.assertIn('def build_3d_gpu_static', waveform)
        self.assertIn('const staticView=P.static_view===true;', gpu)
        self.assertIn('const showField=P.show_field!==false;', gpu)


    def test_3d_static_maps_use_shared_webgl_renderer(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        render_process = (PROJECT_ROOT / "fieldworkbench" / "render_process.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("def build_3d_gpu_static", waveform)
        self.assertIn("payload = build_3d_gpu_static(", render_process)
        self.assertIn("def _add_webgl_map_tab", dialogs)
        self.assertIn('"static_view": True', waveform)
        self.assertIn('"show_field": display_mode != "flux"', waveform)
        self.assertIn("adapter._field_fluxline_traces_3d(", waveform)
        self.assertIn("const staticView=P.static_view===true;", gpu)
        self.assertIn("if(showField)drawVolume(M)", gpu)
        self.assertIn("if(showField)drawField(M)", gpu)

    def test_2d_and_3d_maps_use_isolated_background_render_tabs(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        main_window = (
            PROJECT_ROOT / "fieldworkbench" / "main_window.py"
        ).read_text(encoding="utf-8")
        waveform_dialog = (
            PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py"
        ).read_text(encoding="utf-8")
        gpu_view = (
            PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py"
        ).read_text(encoding="utf-8")
        render_process = (
            PROJECT_ROOT / "fieldworkbench" / "render_process.py"
        ).read_text(encoding="utf-8")
        process_ipc = (
            PROJECT_ROOT / "fieldworkbench" / "process_ipc.py"
        ).read_text(encoding="utf-8")
        waveform = (
            PROJECT_ROOT / "fieldworkbench" / "waveform.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(dialogs)
        classes = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.ClassDef)
        }
        volume_method_nodes = {
            node.name: node
            for node in classes["FieldVolumeMapDialog"].body
            if isinstance(node, ast.FunctionDef)
        }
        volume_methods = set(volume_method_nodes)
        map_methods = {
            node.name
            for node in classes["FieldMapDialog"].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_FieldRenderJobPage", classes)
        self.assertIn("_FieldRenderWorker", classes)
        self.assertIn("_FieldRenderConcurrencyLimiter", classes)
        self.assertIn("_FieldRenderSignalRelay", classes)
        self.assertIn("_BackgroundFieldRenderMixin", classes)
        relay_bases = {
            ast.unparse(base) for base in classes["_FieldRenderSignalRelay"].bases
        }
        self.assertIn("QObject", relay_bases)
        background_methods = {
            node.name
            for node in classes["_BackgroundFieldRenderMixin"].body
            if isinstance(node, ast.FunctionDef)
        }
        for method in (
            "_start_render_job",
            "_render_job_progress",
            "_render_job_completed",
            "_render_job_failed",
            "_render_job_cancelled",
            "_cancel_all_render_jobs",
        ):
            self.assertIn(method, background_methods)
            self.assertNotIn(method, volume_methods)
            self.assertNotIn(method, map_methods)
        self.assertIn("_render_tab_changed", volume_methods)
        self.assertIn("_render_tab_changed", map_methods)
        map_bases = {ast.unparse(base) for base in classes["FieldMapDialog"].bases}
        volume_bases = {ast.unparse(base) for base in classes["FieldVolumeMapDialog"].bases}
        self.assertIn("_BackgroundFieldRenderMixin", map_bases)
        self.assertIn("_BackgroundFieldRenderMixin", volume_bases)
        self.assertIn("self._render_document_snapshot = self.adapter.document()", dialogs)
        self.assertIn("self.document = document", dialogs)
        worker_source = ast.unparse(classes["_FieldRenderWorker"])
        self.assertNotIn("StudioAdapter(", worker_source)
        self.assertIn("field_render_process_context()", worker_source)
        self.assertNotIn("context.Event()", worker_source)
        self.assertGreaterEqual(worker_source.count("context.Pipe(duplex=False)"), 2)
        self.assertIn("self._process_cancel_connection", worker_source)
        self.assertIn("connection.send_bytes", worker_source)
        self.assertIn("run_field_render_process", worker_source)
        self.assertIn("process.exitcode", worker_source)
        self.assertIn("process.terminate()", worker_source)
        self.assertIn("native-crash.log", worker_source)
        self.assertIn("'result_path': str(result_path)", worker_source)
        self.assertNotIn("pickle.load", worker_source)
        self.assertIn("isolated_result = _load_field_render_result(result)", dialogs)
        limiter_source = ast.unparse(classes["_FieldRenderConcurrencyLimiter"])
        self.assertIn("threading.Condition()", limiter_source)
        self.assertIn("self._waiting.append(token)", limiter_source)
        self.assertIn("self._active < self._max_workers", limiter_source)
        self.assertIn("self._active += 1", limiter_source)
        self.assertIn("self._active -= 1", limiter_source)
        self.assertIn("self._condition.notify_all()", limiter_source)
        self.assertNotIn("_FIELD_RENDER_SOLVER_LOCK", dialogs)
        self.assertIn("_FIELD_RENDER_CONCURRENCY.acquire", worker_source)
        self.assertIn("_FIELD_RENDER_CONCURRENCY.release()", worker_source)
        self.assertIn("Queued — waiting for a render slot", dialogs)
        self.assertIn('"worker_count": normalize_optimizer_worker_count(', dialogs)
        self.assertIn(
            'configure_field_render_worker_limit(preferences["worker_count"])',
            dialogs,
        )
        self.assertIn("with _FIELD_RENDER_PROCESS_START_LOCK:", worker_source)
        self.assertIn("Worker concurrency", main_window)
        self.assertIn(
            "configure_field_render_worker_limit(optimizer_workers.value())",
            main_window,
        )
        self.assertIn("thread.started.connect(worker.run)", dialogs)
        self.assertIn("relay = self._render_signal_relay()", dialogs)
        self.assertIn(
            "worker.completed.connect(relay.completed, Qt.ConnectionType.QueuedConnection)",
            dialogs,
        )
        self.assertIn(
            "worker.progress.connect(relay.progress, Qt.ConnectionType.QueuedConnection)",
            dialogs,
        )
        self.assertNotIn("worker.completed.connect(self._render_job_completed)", dialogs)
        self.assertIn(
            "QTimer.singleShot(_FIELD_RENDER_BROWSER_RELEASE_DELAY_MS, thread.start)",
            dialogs,
        )
        self.assertIn("application.aboutToQuit.connect", dialogs)
        self.assertIn("thread.wait()", dialogs)
        self.assertIn("Qt.ConnectionType.DirectConnection", dialogs)
        self.assertIn("job[\"discard\"] = True", dialogs)
        self.assertIn("worker.request_cancellation()", dialogs)
        self.assertIn(
            'defer_payload_build=primary_action == "playback"', dialogs
        )
        self.assertIn("def _prepare_gpu_request", waveform_dialog)
        self.assertNotIn("def build_prepared_gpu_payload", waveform_dialog)
        self.assertIn("def build_prepared_gpu_payload", waveform)
        self.assertIn("self.result_gpu_request = request", waveform_dialog)
        self.assertIn("def run_field_render_process", render_process)
        self.assertIn('str(render_kind) == "2d_static"', render_process)
        self.assertIn("adapter.field_map(", render_process)
        self.assertIn("PipeCancellationSignal(cancel_connection)", render_process)
        self.assertIn("cancel_event.close()", render_process)
        self.assertIn("faulthandler.enable", render_process)
        self.assertIn("pickle.dump", render_process)
        self.assertNotIn("PySide6", render_process)
        self.assertIn('multiprocessing.get_context("forkserver")', process_ipc)
        self.assertIn('multiprocessing.get_context("spawn")', process_ipc)
        self.assertIn("class PipeCancellationSignal", process_ipc)
        self.assertNotIn("multiprocessing.Event", process_ipc)
        self.assertNotIn("Semaphore", process_ipc)
        self.assertIn("def set_backgrounded", gpu_view)
        self.assertIn("QWebEnginePage.LifecycleState.Frozen", gpu_view)
        self.assertNotIn("QWebEnginePage.LifecycleState.Discarded", gpu_view)
        self.assertIn("QWebEnginePage.LifecycleState.Active", gpu_view)
        self.assertIn("def _discard_web_view", gpu_view)
        self.assertIn("start_hibernated=not was_current", dialogs)
        self.assertIn("widget.set_backgrounded(True, discard=True)", dialogs)
        self.assertNotIn("_warm_render_view", dialogs)
        self.assertGreaterEqual(dialogs.count("dialog.result_gpu_request = None"), 2)
        self.assertGreaterEqual(dialogs.count("dialog.deleteLater()"), 2)
        self.assertIn("def _native_payload", gpu_view)
        self.assertIn("_source_html_path=self._html_path", gpu_view)
        self.assertIn("document.addEventListener('freeze'", gpu_view)
        map_method_nodes = {
            node.name: node
            for node in classes["FieldMapDialog"].body
            if isinstance(node, ast.FunctionDef)
        }
        two_d_static_source = ast.unparse(map_method_nodes["calculate"])
        self.assertNotIn("_show_field_map_progress", two_d_static_source)
        self.assertNotIn("setOverrideCursor", two_d_static_source)
        self.assertIn("_start_render_job", two_d_static_source)
        self.assertIn("2d_static", two_d_static_source)
        self.assertGreaterEqual(dialogs.count("defer_payload_build=primary_action == \"playback\""), 2)

        static_source = ast.unparse(volume_method_nodes["_calculate_static_webgl"])
        self.assertNotIn("_show_field_map_progress", static_source)
        self.assertNotIn("setOverrideCursor", static_source)
        self.assertIn("_start_render_job", static_source)

    def test_waveform_gpu_view_recovers_stalled_loads_and_disposes_webgl(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("_GPU_PAGE_STALL_TIMEOUT_MS = 30_000", gpu)
        self.assertIn("_GPU_PAGE_AUTOMATIC_RETRIES = 1", gpu)
        self.assertIn("def _gpu_page_load_stalled", gpu)
        self.assertIn("progress_value > self._last_load_progress", gpu)
        self.assertIn("def _render_process_terminated", gpu)
        self.assertIn("renderProcessTerminated.connect", gpu)
        self.assertIn("GPU viewer stalled — restarting renderer", gpu)
        self.assertIn("Switch tabs and return to retry this result", gpu)
        self.assertNotIn("QWebEnginePage.LifecycleState.Discarded", gpu)
        self.assertIn("window.fieldWorkbenchDispose", gpu)
        self.assertIn("gl.deleteTexture", gpu)
        self.assertIn("gl.deleteBuffer", gpu)
        self.assertIn("gl.deleteProgram", gpu)
        self.assertIn("application.aboutToQuit.connect(_cleanup_waveform_gl_views_for_shutdown)", gpu)
        module = ast.parse(dialogs)
        classes = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.ClassDef)
        }
        for class_name in ("FieldMapDialog", "FieldVolumeMapDialog"):
            methods = {
                node.name
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef)
            }
            self.assertIn("closeEvent", methods)

    def test_application_shutdown_retires_plotly_webengine_surfaces(self) -> None:
        app = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(
            encoding="utf-8"
        )
        utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("_ACTIVE_PLOT_VIEWS", plot)
        self.assertIn(
            "application.aboutToQuit.connect(_cleanup_plot_views_for_shutdown)",
            plot,
        )
        self.assertIn("window.fieldWorkbenchDispose", plot)
        self.assertIn("Plotly.purge(plot)", plot)
        self.assertIn("self.removeWidget(web)", plot)
        self.assertIn("web.deleteLater()", plot)
        self.assertIn("def _cleanup_browser_surfaces_for_shutdown", app)
        self.assertIn(
            "QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)",
            app,
        )
        self.assertIn("_shiboken_is_valid", utils)
        self.assertIn("if not _qt_object_is_alive(widget)", utils)

    def test_3d_map_exposes_numerical_animation_field_validation(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        waveform = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        self.assertNotIn('QPushButton("Validate animation field")', dialogs)
        self.assertIn("def validate_animation_field_basis", waveform)
        self.assertIn("BASIS RECONSTRUCTION VS FRESH DIRECT SOLVES", dialogs)
        self.assertIn("This test deliberately does NOT validate volume transparency", dialogs)

    def test_3d_map_exposes_gpu_full_volume_sampling_validation(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertNotIn('QPushButton("Validate GPU volume")', dialogs)
        self.assertIn("def validate_gpu_volume", gpu)
        self.assertIn("gpuValidationReady", gpu)
        self.assertIn("gl.beginTransformFeedback(gl.POINTS)", gpu)
        self.assertIn("SOURCE LATTICE VS ACTUAL WEBGL TEXTURE SAMPLING", dialogs)
        self.assertIn("vec3 latticeTc(vec3 u)", gpu)
        self.assertIn("*(dims-vec3(1.0))+vec3(0.5))/dims", gpu)
        self.assertIn("+vec3(0.5))/float(r)", gpu)

        # Guard the exact 0.13.0.179 regression: the button lives on the 3D
        # FieldVolumeMapDialog, so both Qt-side handlers must live there too.
        module = ast.parse(dialogs)
        classes = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.ClassDef)
        }
        volume_methods = {
            node.name
            for node in classes["FieldVolumeMapDialog"].body
            if isinstance(node, ast.FunctionDef)
        }
        map_methods = {
            node.name
            for node in classes["FieldMapDialog"].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("validate_gpu_volume", volume_methods)
        self.assertIn("_show_gpu_volume_validation_report", volume_methods)
        self.assertNotIn("validate_gpu_volume", map_methods)

    def test_point_probe_is_retired_and_axis_sensor_uses_linear_name(self) -> None:
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        self.assertNotIn("point_probe_action", main_window)
        self.assertNotIn("FieldProbeDialog", main_window)
        self.assertNotIn("FieldProbeDialog", dialogs)
        self.assertNotIn('if kind == "point":', adapter)
        self.assertNotIn("Point Probe", help_topics)
        self.assertIn('(\"Linear axis sensor\", \"axis_sensor\")', main_window)
        self.assertIn('\"label\": \"Linear axis sensor\"', adapter)
        self.assertIn('type_label = \"Linear axis sensor\" if path_length > 1', main_window)


    def test_measurement_results_include_shared_cross_sections_and_limitations_help(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        self.assertIn('self.tabs.addTab(self.cross_sections, "Cross sections")', dialogs)
        self.assertIn("def measurement_cross_section_figure", adapter)
        self.assertIn('"Central local-coordinate cross sections — |B|"', adapter)
        self.assertIn('("XY", 2, 0, 1', adapter)
        self.assertIn('("XZ", 1, 0, 2', adapter)
        self.assertIn('("YZ", 0, 1, 2', adapter)
        self.assertIn('"coloraxis": "coloraxis"', adapter)
        self.assertIn('"Limitations and model boundaries"', help_topics)
        self.assertIn("no built-in maximum frequency", help_topics)
        self.assertIn("does <b>not</b> solve ferromagnetic cores", help_topics)
        self.assertIn("actual evaluated points nearest the measurement", help_topics)
        self.assertIn('"Brain View and statistics"', help_topics)
        self.assertIn("Filter</b> is an exclusion rule", help_topics)
        self.assertIn("Playback RMS = sqrt(B²-time / exposure time)", help_topics)



if __name__ == "__main__":
    unittest.main()
