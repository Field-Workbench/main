"""Static checks for release-build configuration."""

from __future__ import annotations

import ast
import builtins
import symtable
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_python_modules_have_no_unresolved_global_references(self) -> None:
        implicit_module_globals = {
            "__cached__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__spec__",
        }
        builtin_names = set(dir(builtins))
        problems: list[str] = []

        for source_path in sorted((PROJECT_ROOT / "fieldworkbench").rglob("*.py")):
            source = source_path.read_text(encoding="utf-8")
            table = symtable.symtable(source, str(source_path), "exec")
            module_definitions = {
                symbol.get_name()
                for symbol in table.get_symbols()
                if symbol.is_imported()
                or symbol.is_assigned()
                or symbol.is_namespace()
                or symbol.is_parameter()
            }
            referenced_globals: set[str] = set()

            def collect_referenced_globals(scope) -> None:
                referenced_globals.update(
                    symbol.get_name()
                    for symbol in scope.get_symbols()
                    if symbol.is_referenced() and symbol.is_global()
                )
                for child in scope.get_children():
                    collect_referenced_globals(child)

            collect_referenced_globals(table)
            missing = sorted(
                referenced_globals
                - module_definitions
                - builtin_names
                - implicit_module_globals
            )
            if missing:
                relative = source_path.relative_to(PROJECT_ROOT)
                problems.append(f"{relative}: {', '.join(missing)}")

        self.assertEqual([], problems, "Unresolved global references:\n" + "\n".join(problems))

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
            '--add-data "fieldworkbench\\assets\\icons;fieldworkbench\\assets\\icons"',
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
        self.assertIn(
            "--add-data fieldworkbench/assets/icons:fieldworkbench/assets/icons",
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
        # Region widgets and result actions are shared with 2D/3D playback.
        brain_view += (PROJECT_ROOT / "fieldworkbench" / "exposure_panel.py").read_text(encoding="utf-8")
        brain_view += (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
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

        self.assertNotIn('self.maps_action = QAction("Maps", self)', main_window)
        self.assertNotIn('self.snapshots_action', main_window)
        self.assertNotIn('self.atlas_selection', brain_view)
        self.assertNotIn('self.atlas_alignment_note', brain_view)
        self.assertIn('CollapsibleSection("Map options", expanded=False)', brain_view)
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
        lpba40_display = (
            PROJECT_ROOT / "fieldworkbench" / "lpba40_display.py"
        ).read_text(encoding="utf-8")
        self.assertIn("build_lpba40_region_trace", brain_view)
        self.assertIn('"type": "mesh3d"', lpba40_display)
        self.assertIn('f"brain-region:{int(label_id)}"', brain_view)
        self.assertIn("self.plot.selection_requested.connect", brain_view)
        self.assertIn("_brain_view_scene_object_choices", brain_view)
        self.assertIn("_mark_selector_context_noninteractive", brain_view)
        self.assertIn('meta["field_workbench_ignore_picking"] = True', brain_view)
        self.assertIn('trace["hoverinfo"] = "skip"', brain_view)
        self.assertIn("fieldWorkbenchApplyPickingPolicy", plot_view)
        self.assertIn("object.__fieldWorkbenchOriginalDrawPick", plot_view)
        self.assertIn("object.drawPick = null", plot_view)
        self.assertIn("__fieldWorkbenchSelectionGestureBound", plot_view)
        self.assertIn("movementThresholdPx = 5", plot_view)
        self.assertIn("__fieldWorkbenchQueueSelection", plot_view)
        self.assertIn("additive: additiveModifier(event)", plot_view)
        self.assertIn("effectiveAdditive = effectiveAdditive || Boolean(gesture.additive)", plot_view)
        self.assertIn("window.fieldWorkbenchBridge.requestSelection('', false)", plot_view)
        self.assertIn("selectionSurface: selectionSurfaceClick(event)", plot_view)
        self.assertIn("window.setTimeout(dispatchPendingSelection, 30)", plot_view)
        self.assertIn("page = BrainViewDialog(", main_window)
        self.assertIn("workspace_mode=page_mode", main_window)
        self.assertNotIn("self.brain_view_action", main_window)
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
        self.assertIn('class ExposureRegionPanel', brain_view)
        self.assertIn('QRadioButton("Current frame")', brain_view)
        self.assertIn('QRadioButton("Playback summary")', brain_view)
        self.assertIn('"B-time", "B²-time"', brain_view)
        self.assertNotIn('"playback_summary": {', waveform)
        self.assertNotIn('lpba40_playback_region_metrics(', waveform)
        self.assertNotIn('lpba40_playback_region_metrics_streaming(', waveform)
        self.assertGreaterEqual(
            brain_view.count('QSplitter(Qt.Orientation.Vertical)'), 2
        )
        self.assertIn('QPushButton("Collapse")', brain_view)
        self.assertIn('QPushButton("Expand")', brain_view)
        self.assertIn('_collapse_brain_region_tree_to_major', brain_view)
        self.assertIn('_expand_brain_region_tree', brain_view)
        self.assertIn('self.column_menu = QMenu(self.table)', brain_view)
        self.assertIn('header.customContextMenuRequested.connect(self._show_column_menu)', brain_view)
        self.assertIn('class ExposureResultTab', brain_view)
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
        self.assertIn('font-size: 18pt;', theme)
        self.assertIn('QLabel#projectManagerProjectTitle', theme)
        self.assertIn('font-size: 16pt;', theme)
        project_manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(encoding="utf-8")
        self.assertIn('def _update_title_colour_state(self) -> None:', project_manager)
        self.assertIn('"namedProject",', project_manager)
        self.assertIn('self.record.path is not None', project_manager)
        self.assertNotIn('def emphasize_loaded_title(self) -> None:', project_manager)
        self.assertNotIn('page.emphasize_loaded_title()', project_manager)
        self.assertIn('QLabel#projectManagerProjectTitle[namedProject="true"]', theme)
        self.assertIn('class OutlinedStateLabel(QLabel):', project_manager)
        self.assertIn('self.title_label = OutlinedStateLabel(active_property="namedProject")', project_manager)
        self.assertIn('title_label = UnderlineStateLabel(title, active_property="populated")', project_manager)
        self.assertIn('_ACTIVE_FILL = QColor("#75ff00")', project_manager)
        self.assertIn('_OUTLINE_WIDTH = 0.3', project_manager)
        self.assertIn('_OUTLINE = QColor("#000000")', project_manager)
        self.assertIn('path.addText(QPointF(x, baseline), font, label_text)', project_manager)
        self.assertIn('painter.drawPath(path)', project_manager)
        self.assertIn('class UnderlineStateLabel(QLabel):', project_manager)
        self.assertIn('clicked = Signal()', project_manager)
        self.assertIn('self.setCursor(Qt.CursorShape.PointingHandCursor)', project_manager)
        self.assertIn('title_label.clicked.connect(primary_application_callback)', project_manager)
        self.assertIn('_UNDERLINE = QColor("#77ff00")', project_manager)
        self.assertIn('_UNDERLINE_WIDTH = 2.5', project_manager)
        self.assertIn('pen.setCapStyle(Qt.PenCapStyle.RoundCap)', project_manager)
        self.assertIn('painter.drawLine(QPointF(start_x, underline_y), QPointF(start_x + text_width, underline_y))', project_manager)
        self.assertNotIn('class StretchHighlightStateLabel(QLabel):', project_manager)
        self.assertNotIn('QSvgRenderer', project_manager)
        self.assertNotIn('QLabel#projectManagerResourceTitle[populated="true"]', theme)
        self.assertGreaterEqual(theme.count('color: #75ff00;'), 2)
        self.assertIn('font-size: 11pt;', theme)
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
        self.assertIn('tab.region_panel.video_table_snapshot()', brain_view)
        self.assertIn('tab.region_panel.video_frame_metric_payload()', brain_view)
        self.assertIn('viewer_actions.addWidget(self.export_video_button)', brain_view)
        self.assertNotIn('action_layout.addWidget(self.export_video_button)', brain_view)
        self.assertIn('map_settings["brain_region_video_table"] = self._video_region_table_snapshot()', brain_view)
        self.assertIn('primary_action in {"playback", "export"}', brain_view)
        video_dialog = (
            PROJECT_ROOT / "fieldworkbench" / "waveform_video_dialog.py"
        ).read_text(encoding="utf-8")
        self.assertIn('format_form.addRow("Regions", self.brain_table_summary)', video_dialog)
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
        self.assertIn("function drawTracePanels(", gpu)
        self.assertIn("const panes=[view,...tracePanels.map(item=>item.pane)]", gpu)
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
        self.assertIn('"show_contours": self.show_contours.isChecked()', dialogs)
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
        cleaner = (PROJECT_ROOT / "tools" / "clean_obsolete_source_files.py").read_text(encoding="utf-8")
        self.assertIn('archive.write(path, Path("field_workbench") / relative)', source)
        self.assertIn('EXCLUDED_NAMES = {"RELEASE_REVIEW.md"}', source)
        self.assertIn('Path("RELEASE_REVIEW.md")', cleaner)
        self.assertFalse((PROJECT_ROOT / "RELEASE_REVIEW.md").exists())

    def test_release_facing_docs_match_current_suite_and_exposure_workflow(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        testing = (PROJECT_ROOT / "TESTING.md").read_text(encoding="utf-8")
        building = (PROJECT_ROOT / "BUILDING_WINDOWS.md").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        for app_name in (
            "Model Editor", "Map Creator", "Exposure Analysis", "Saved Results",
            "Exposure Editor", "Waveform Editor", "Dataset Manager",
        ):
            self.assertIn(app_name, readme)
        self.assertNotIn("[windows zip file]()", readme)
        self.assertIn("Calculate all uncached analyses", testing)
        self.assertIn("single-worker queue", testing)
        self.assertNotIn("Exposure analysis hub", testing)
        self.assertNotIn("**Recalculate**", testing)
        self.assertIn("taskbar-visible", building)
        self.assertIn("single-worker queue", building)
        self.assertIn("Show/hide filters</b>", help_topics)
        self.assertNotIn("highlighting grid above the table", help_topics)

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
        preferences = (PROJECT_ROOT / "fieldworkbench" / "preferences.py").read_text(
            encoding="utf-8"
        )
        plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(
            encoding="utf-8"
        )
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("apply_workbench_theme(theme=load_theme_preference())", app)
        self.assertIn('theme_selector.setObjectName("themeSelector")', preferences)
        self.assertIn("save_theme_preference(selected, settings)", preferences)
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

    def test_suite_phase2_uses_shared_project_context_and_resource_registry(self) -> None:
        project_context = (PROJECT_ROOT / "fieldworkbench" / "project_context.py").read_text(
            encoding="utf-8"
        )
        project_manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(
            encoding="utf-8"
        )
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("class ProjectContext", project_context)
        self.assertIn("class ProjectResourceRegistry", project_context)
        self.assertIn('return f"{str(kind).strip()}:{str(native_id).strip()}"', project_context)
        self.assertIn("context = ProjectContext(", project_manager)
        self.assertIn("self.record.context.resources.signature(refresh=False)", project_manager)
        self.assertNotIn("self.record.context.resources.signature(refresh=True)", project_manager)
        self.assertIn("project_context=self.record.context", project_manager)
        self.assertIn('register_window("maps_analysis", window)', project_manager)
        self.assertIn('register_window("exposure_editor", window)', project_manager)
        self.assertIn('register_window("dataset_manager", window)', project_manager)
        self.assertIn('register_window("snapshot_manager", window)', project_manager)

    def test_phase6_maps_and_saved_results_are_project_owned(self) -> None:
        project_manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(
            encoding="utf-8"
        )
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(
            encoding="utf-8"
        )
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("maps_window: UnifiedFieldWorkspaceDialog | None", project_manager)
        self.assertIn("snapshot_window: SnapshotManagerDialog | None", project_manager)
        self.assertIn("window = UnifiedFieldWorkspaceDialog(", project_manager)
        self.assertIn('self.record.context.register_window("maps_analysis", window)', project_manager)
        self.assertIn("window = SnapshotManagerDialog(", project_manager)
        self.assertIn('self.record.context.register_window("snapshot_manager", window)', project_manager)
        self.assertIn("window.notify_project_changed()", project_manager)
        self.assertIn("self._notify_saved_results_project_changed(immediate=True)", project_manager)
        snapshot_manager_source = dialogs.split("class SnapshotManagerDialog", 1)[1].split("class SceneNotesDialog", 1)[0]
        self.assertNotIn("setInterval(2000)", snapshot_manager_source)
        self.assertNotIn("timeout.connect(self.refresh_if_changed)", snapshot_manager_source)
        self.assertIn("self._deferred_refresh_timer.setSingleShot(True)", snapshot_manager_source)
        self.assertNotIn("self.ensure_model_window().open_maps_workspace()", project_manager)
        self.assertNotIn("self.ensure_model_window().open_snapshots()", project_manager)
        self.assertNotIn("suite_maps_analysis_requested", main)
        self.assertNotIn("suite_snapshot_manager_requested", main)
        self.assertNotIn('CollapsibleSection("Measurement snapshots", expanded=False)', main)
        self.assertIn('self.measurement_view_splitter = QSplitter(Qt.Orientation.Horizontal)', main)
        self.assertIn('self._measurement_panel_min_width = 220', main)
        self.assertIn('host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)', main)
        self.assertIn('self.measurement_panel_host = self._make_workspace_measurement_panel()', main)
        self.assertIn('ExposureRegionPanel(hierarchy, measurement=True)', main)
        self.assertIn('QPushButton("Analyze selected…")', main)
        self.assertIn('QPushButton("Analyze all")', main)
        self.assertIn('QPushButton("View / save snapshot…")', main)
        self.assertIn('QGridLayout(actions)', main)
        self.assertIn('region_button.setText("Show measurements")', main)
        self.assertIn('self.workspace_viewer_footer = QWidget(self.workspace_footer)', main)
        self.assertIn('self.workspace_panel_footer = QWidget(self.workspace_footer)', main)
        self.assertIn('self.workspace_fullscreen_button = QPushButton("Fullscreen viewer")', main)
        self.assertIn('self.workspace_regions_button = QPushButton("Show measurements")', main)
        self.assertIn('def _sync_workspace_footer_geometry(', main)
        self.assertIn('self._measurement_panel_visibility_by_view = {', main)
        self.assertIn('"2d": False', main)
        self.assertIn('"3d": False', main)
        self.assertIn('"brain": True', main)
        self.assertIn('def _sync_workspace_regions_button(', main)
        self.assertIn('self.setMinimumSize(820, 560)', main)
        self.assertIn('self._sidebar_min_width = 240', main)
        self.assertIn('self._workspace_view_min_width = 540', main)
        self.assertIn('self._measurement_view_min_width = 320', main)
        self.assertIn('def _apply_workspace_splitter_state(', main)
        self.assertIn('control_scroll.setMinimumWidth(0)', main)
        self.assertIn('QFormLayout.RowWrapPolicy.WrapLongRows', main)
        self.assertIn('def _workspace_panel_splitter_moved(', main)
        self.assertIn('def _active_workspace_panel_splitter(', main)
        self.assertIn('CollapsibleSection("Heatmap options", expanded=True)', dialogs)
        self.assertIn('scene_options_layout.addWidget(self.show_grid)', dialogs)
        self.assertIn('self.display_options_section.setVisible(heatmap_visible)', dialogs)
        self.assertIn('self.display_options_section.setVisible(field_visible)', dialogs)
        self.assertIn('field_view_row_layout.addRow("Field view", self.display_mode)', dialogs)
        self.assertGreaterEqual(dialogs.count('if self.workspace_mode != "playback":'), 2)
        self.assertGreaterEqual(dialogs.count('self.display_mode.setEnabled(False)'), 2)
        self.assertIn('"slices" if self.workspace_mode == "playback" else self.display_mode.currentData()', dialogs)
        self.assertIn('region_layout.addWidget(QLabel("Render mode"), 4, 0)', dialogs)
        self.assertIn("snapshot_store_adapter=self.project_adapter", main)
        self.assertIn("snapshot_store_adapter: StudioAdapter | None = None", dialogs)
        self.assertIn("def create_snapshot_from_frozen_analysis(", adapter)

    def test_saved_maps_are_project_resources_and_reopen_without_solver(self) -> None:
        project_manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(
            encoding="utf-8"
        )
        project_context = (PROJECT_ROOT / "fieldworkbench" / "project_context.py").read_text(
            encoding="utf-8"
        )
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(
            encoding="utf-8"
        )
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(
            encoding="utf-8"
        )
        plot = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(
            encoding="utf-8"
        )
        gpu = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('RESOURCE_SAVED_MAP = "saved_map"', project_context)
        self.assertIn('metadata["saved_maps"] = copy.deepcopy(self.saved_maps)', adapter)
        self.assertIn("def save_map_resource(", adapter)
        self.assertIn("def saved_map(self, map_id: str)", adapter)
        self.assertIn("def rename_saved_map(self, map_id: str, name: str)", adapter)
        self.assertIn("def delete_saved_maps(self, map_ids: Iterable[str])", adapter)
        self.assertIn('self.result_tabs.addTab(self.static_map_page, "Static maps (0)")', dialogs)
        self.assertIn('self.result_tabs.addTab(self.scripted_map_page, "Scripted maps (0)")', dialogs)
        self.assertIn('self.result_tabs.addTab(self.snapshot_page, "Snapshots (0)")', dialogs)
        self.assertIn('def _update_result_tab_labels(', dialogs)
        self.assertIn('f"Static maps ({static_count})"', dialogs)
        self.assertIn('f"Scripted maps ({scripted_count})"', dialogs)
        self.assertIn('f"Snapshots ({len(snapshots)})"', dialogs)
        self.assertIn("on_open_map: Callable[[str, str], None] | None = None", dialogs)
        self.assertIn('QPushButton("Save map…")', main)
        self.assertIn('QPushButton("Load saved map…")', main)
        self.assertIn("def load_saved_map(self, map_id: str) -> bool", main)
        self.assertIn("viewer.export_figure()", main)
        self.assertIn("viewer.export_render_payload()", main)
        self.assertIn("def export_figure(self) -> dict", plot)
        self.assertIn("def export_render_payload(self) -> dict[str, Any]", gpu)
        self.assertIn('saved_result_count = len(saved_maps) + len(exposure_analyses) + len(snapshots)', project_manager)
        self.assertIn('saved result', project_manager)
        self.assertIn("saved_map_id=map_id", project_manager)
        self.assertIn('"Saved Results"', project_manager)
        self.assertIn('self._add_dashboard_group(\n            2,\n            2,\n            "Maps / Analysis",', project_manager)
        self.assertIn('self._add_dashboard_group(\n            4,\n            2,\n            "Exposures",', project_manager)

    def test_project_manager_dashboard_aligns_project_summaries_with_applications(self) -> None:
        project_manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("QTreeWidget", project_manager)
        self.assertIn('self._add_dashboard_group(', project_manager)
        self.assertIn('"Maps / Analysis",\n            "maps",', project_manager)
        self.assertIn('"Exposures",\n            "exposures",', project_manager)
        self.assertIn('"Data / Results",\n            "data_results",', project_manager)
        self.assertIn('"Exposure Analysis",\n                    "Analyze field exposure over time and across spatial regions."', project_manager)
        self.assertIn('"Saved Results",\n                    "Review saved maps, analyses, and snapshots."', project_manager)
        exposure_index = project_manager.index('"Exposure Editor"')
        waveform_index = project_manager.index('"Waveform Editor"', exposure_index)
        dataset_index = project_manager.index('"Dataset Manager"', waveform_index)
        self.assertLess(exposure_index, waveform_index)
        self.assertLess(waveform_index, dataset_index)
        self.assertIn('dashboard_scroll.setWidget(dashboard_widget)', project_manager)
        self.assertIn('self.dashboard_grid = QGridLayout(dashboard_widget)', project_manager)
        self.assertIn('self.dashboard_grid.setColumnMinimumWidth(1, 42)', project_manager)
        self.assertIn('project_heading = QLabel("PROJECT COMPONENTS")', project_manager)
        self.assertIn('self.dashboard_grid.addWidget(project_heading, 0, 0, 1, 2)', project_manager)
        self.assertIn('self.dashboard_grid.setRowMinimumHeight(row, 50)', project_manager)
        self.assertNotIn('self.project_grid =', project_manager)
        self.assertNotIn('self.app_grid =', project_manager)
        self.assertIn('group.setObjectName("projectManagerDashboardGroupFrame")', project_manager)
        self.assertIn('self.dashboard_grid.addWidget(group, row, 0, row_span, 3)', project_manager)
        self.assertIn('group_grid.setVerticalSpacing(0)', project_manager)
        self.assertIn('app_area.setObjectName("projectManagerAppArea")', project_manager)
        self.assertIn('app_area.setFrameShape(QFrame.Shape.NoFrame)', project_manager)
        self.assertIn('class ResponsiveAppIconButton(QPushButton):', project_manager)
        self.assertIn('button = ResponsiveAppIconButton()', project_manager)
        self.assertIn('app_layout.addWidget(button, 0)', project_manager)
        self.assertIn('name_label.setObjectName("projectManagerAppName")', project_manager)
        self.assertIn('title_label.setWordWrap(True)', project_manager)
        self.assertIn('title_label.setMinimumWidth(190)', project_manager)
        self.assertIn('self.project_component_title_labels: dict[str, QLabel] = {}', project_manager)
        self.assertIn('self.project_component_title_labels[key] = title_label', project_manager)
        self.assertNotIn('project_component_highlight', project_manager)
        for obsolete in ('project_component_highlight.svg', 'project_component_highlight_left.svg', 'project_component_highlight_center.svg', 'project_component_highlight_right.svg'):
            self.assertFalse((PROJECT_ROOT / 'fieldworkbench' / 'assets' / 'icons' / obsolete).exists())
        self.assertIn('self._set_component_populated("model", bool(objects))', project_manager)
        self.assertIn('self._set_component_populated("maps", bool(saved_maps or exposure_analyses))', project_manager)
        self.assertIn('"exposures", bool(exposures or waveform_resources)', project_manager)
        self.assertIn('"data_results", bool(datasets or calibrations or saved_result_count)', project_manager)
        self.assertIn('button.setProperty("applicationName", button_text)', project_manager)
        self.assertIn('self.setMinimumSize(900, 820)', project_manager)
        self.assertIn('info_label.setMaximumWidth(320)', project_manager)
        self.assertIn('resource_layout.addStretch(1)\n        resource_layout.addWidget(info_label, 0)\n        resource_layout.addStretch(1)', project_manager)
        self.assertIn('arrow.setObjectName("projectManagerConnectorArrow")', project_manager)
        self.assertIn('self.project_metadata_label.setObjectName("projectManagerProjectMetadata")', project_manager)
        self.assertIn('self.project_status_label.setObjectName("projectManagerProjectStatus")', project_manager)
        self.assertIn('self.project_status_label.setProperty("dirty", status_is_dirty)', project_manager)
        self.assertIn('f"<b>{len(exposures)} exposure{', project_manager)
        self.assertIn('f"<b>{len(waveform_resources)} waveform{', project_manager)
        self.assertIn('f"<b>{len(datasets)} dataset{', project_manager)
        self.assertIn('f"<b>{len(calibrations)} calibration{', project_manager)
        self.assertNotIn('</b>  ·  \"\n            f\"<b>{len(waveform_resources)}', project_manager)
        self.assertNotIn('</b>  ·  \"\n            f\"<b>{len(calibrations)}', project_manager)
        self.assertIn('<b>Last saved:</b>', project_manager)
        self.assertNotIn('projectManagerPhaseNote', project_manager)
        self.assertNotIn('toolbar.addAction(self.model_action)', project_manager)
        self.assertNotIn('toolbar.addAction(self.maps_action)', project_manager)
        self.assertEqual(project_manager.count('toolbar.addAction('), 3)
        self.assertIn('project_menu.addAction(self.model_action)', project_manager)
        self.assertIn('project_menu.addAction(self.maps_action)', project_manager)
        self.assertIn('project_menu.addAction(self.exposure_analysis_action)', project_manager)
        self.assertIn('def open_exposure_analysis(self, exposure_id: str | None = None) -> None:', project_manager)
        self.assertIn('QInputDialog.getItem(', project_manager)
        self.assertIn('self.record.context.register_window("exposure_analysis", workspace)', project_manager)
        self.assertIn('"Data / Results",', project_manager)
        self.assertNotIn('self._refresh_timer = QTimer(self)', project_manager)
        self.assertNotIn('timeout.connect(self.refresh_projects)', project_manager)
        self.assertIn('project_state_changed.connect(self._model_project_state_changed)', project_manager)
        self.assertNotIn('projectManagerAppCategory', project_manager)
        self.assertNotIn('category_label = QLabel', project_manager)
        self.assertNotIn('projectManagerResourceGlyphPanel', project_manager)
        self.assertIn('_PROJECT_APPLICATION_ICONS = {', project_manager)
        expected_custom_icons = {
            "Model Editor": "model_editor.svg",
            "Map Creator": "map_creator.svg",
            "Exposure Analysis": "exposure_analysis.svg",
            "Saved Results": "saved_results.svg",
            "Exposure Editor": "exposure_editor.svg",
            "Waveform Editor": "waveform_editor.svg",
            "Dataset Manager": "dataset_manager.svg",
        }
        icon_map_block = project_manager.split('_PROJECT_APPLICATION_ICONS = {', 1)[1].split('}', 1)[0]
        for application_name, filename in expected_custom_icons.items():
            self.assertIn(f'"{application_name}": "{filename}"', icon_map_block)
            icon_path = PROJECT_ROOT / "fieldworkbench" / "assets" / "icons" / filename
            self.assertTrue(icon_path.is_file())
            icon_text = icon_path.read_text(encoding="utf-8")
            self.assertIn('viewBox="0 0 128 128"', icon_text)
            self.assertNotIn('<image', icon_text)
            self.assertNotIn('data:image/', icon_text)
        toolbar_icons = {
            "project_open.svg",
            "project_save.svg",
        }
        for filename in toolbar_icons:
            icon_path = PROJECT_ROOT / "fieldworkbench" / "assets" / "icons" / filename
            self.assertTrue(icon_path.is_file())
            icon_text = icon_path.read_text(encoding="utf-8")
            self.assertIn('viewBox="0 0 32 32"', icon_text)
            self.assertNotIn('<image', icon_text)
            self.assertNotIn('data:image/', icon_text)
        self.assertIn('"project_open.svg"', project_manager)
        self.assertIn('"project_save.svg"', project_manager)
        banner_path = PROJECT_ROOT / "fieldworkbench" / "assets" / "icons" / "project_banner.svg"
        self.assertTrue(banner_path.is_file())
        banner_text = banner_path.read_text(encoding="utf-8")
        self.assertIn('viewBox="0 0 2750 128"', banner_text)
        self.assertNotIn('<image', banner_text)
        self.assertNotIn('data:image/', banner_text)
        self.assertNotIn('<text', banner_text)
        self.assertNotIn('shape-inside', banner_text)
        self.assertIn('_PROJECT_BANNER_FILENAME = "project_banner.svg"', project_manager)
        self.assertIn('self.project_banner_label.setObjectName("projectManagerToolbarBanner")', project_manager)
        self.assertIn('toolbar.iconSizeChanged.connect(self._refresh_project_banner)', project_manager)
        self.assertIn('banner_height = max(18, int(icon_size.height()))', project_manager)
        self.assertIn('banner_width = max(1, int(round(banner_height * _PROJECT_BANNER_ASPECT)))', project_manager)
        self.assertNotIn('banner_spacer', project_manager)
        self.assertNotIn('toolbar.addWidget(self.project_banner_label)', project_manager)
        self.assertIn('def _position_project_banner(self) -> None:', project_manager)
        self.assertIn('_PROJECT_BANNER_RIGHT_INSET = 22', project_manager)
        self.assertIn('x = max(0, toolbar.width() - label.width() - _PROJECT_BANNER_RIGHT_INSET)', project_manager)
        self.assertIn('self.project_banner_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)', project_manager)
        self.assertIn('self._position_project_banner()', project_manager)
        self.assertIn('def _project_toolbar_icon(', project_manager)
        self.assertNotIn('def _save_toolbar_icon(', project_manager)
        self.assertIn('button.setIcon(self._application_launcher_icon(button_text, icon_kind))', project_manager)
        self.assertIn('self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)', project_manager)
        self.assertIn('target_side = max(self._MIN_SIDE, min(self._MAX_SIDE, self.height()))', project_manager)
        self.assertIn('self.setFixedWidth(target_side)', project_manager)
        self.assertIn('app_layout.addWidget(button, 0)', project_manager)
        self.assertNotIn('app_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)', project_manager)
        self.assertNotIn('projectManagerResourceGlyph', project_manager)
        theme = (PROJECT_ROOT / "fieldworkbench" / "theme.py").read_text(encoding="utf-8")
        self.assertIn('QFrame#projectManagerDashboardGroupFrame {\n    background-color: #ffffff;', theme)
        self.assertIn('QFrame#projectManagerDashboardGroupFrame {\n    background-color: #2c333a;', theme)
        self.assertIn('QFrame#projectManagerResourceArea,\nQFrame#projectManagerAppArea {', theme)
        self.assertIn('QLabel#projectManagerConnectorArrow', theme)
        self.assertIn('background: transparent;', theme)
        self.assertIn('QLabel#projectManagerProjectStatus[dirty="true"]', theme)
        self.assertIn('QLabel#projectManagerProjectMetadata', theme)
        self.assertIn('QLabel#projectManagerAppName', theme)
        self.assertIn('QPushButton#projectManagerAppButton', theme)

    def test_suite_owns_global_preferences_and_maps_own_fluxline_cleanup(self) -> None:
        project_manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(
            encoding="utf-8"
        )
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(
            encoding="utf-8"
        )
        preferences = (PROJECT_ROOT / "fieldworkbench" / "preferences.py").read_text(
            encoding="utf-8"
        )
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('edit_menu.addAction("Preferences…")', project_manager)
        self.assertIn("if not self._suite_managed:", main)
        # Standalone Model Editor smoke tests replace its QDialog with a
        # non-blocking double.  Preserve that injection through the shared
        # Preferences helper so offscreen tests cannot enter a real modal loop.
        self.assertIn("dialog_factory=QDialog", main)
        self.assertIn("dialog_factory: Callable[[QWidget], QDialog] | None = None", preferences)
        self.assertIn("make_dialog = dialog_factory or QDialog", preferences)
        self.assertNotIn("fluxlines/minimumFieldCutoff", preferences)
        self.assertNotIn("fluxlines/conductorDistanceCutoff", preferences)
        self.assertIn('CollapsibleSection("Fluxline options", expanded=False)', dialogs)
        self.assertIn('section.setVisible(mode in {"flux", "both"})', dialogs)

    def test_preferences_explanations_are_short_and_local(self) -> None:
        preferences = (PROJECT_ROOT / "fieldworkbench" / "preferences.py").read_text(
            encoding="utf-8"
        )
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("These settings are application-wide", preferences)
        self.assertIn("Cleanup stops unstable fluxlines", dialogs)
        self.assertIn('CollapsibleSection("Fluxline options", expanded=False)', dialogs)
        self.assertNotIn("Auto first screens each coil", preferences)

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
        shared_actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        self.assertIn("view.export_payload()", shared_actions)
        self.assertNotIn('QPushButton("Analysis…")', shared_actions)
        self.assertIn("class ExposureAnalysisTargetView(QWidget)", shared_actions)
        self.assertIn("class ExposureAnalysisWindow(QDialog)", shared_actions)
        self.assertNotIn("PlaybackAnalysisDialog", shared_actions)
        self.assertNotIn('"Preview statistics"', shared_actions)
        self.assertIn('self.preparation_duration_label = QLabel(', shared_actions)
        self.assertIn('f"Full exposure: {self._format_exposure_duration(self.payload)}"', shared_actions)
        self.assertIn('self.workspace_splitter.addWidget(self.region_tree)', shared_actions)
        self.assertIn("ThresholdDurationPage", shared_actions)
        self.assertIn("analyse_regions_frequency_weighted", shared_actions)
        self.assertEqual(source.count("self._export_region_video()"), 2)

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

    def test_maps_are_project_owned_non_modal_application_windows(self) -> None:
        manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(encoding="utf-8")
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        window_utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(encoding="utf-8")

        self.assertIn('def open_maps(self, mode: str = "static"', manager)
        self.assertIn("window = UnifiedFieldWorkspaceDialog(", manager)
        self.assertIn('register_window("maps_analysis", window)', manager)
        self.assertIn("window.show()", manager)
        self.assertNotIn("def open_field_map(", main_window)
        self.assertNotIn("def open_field_volume_map(", main_window)
        self.assertNotIn("def open_brain_view(", main_window)
        self.assertIn("def _maps_brain_object_id(self) -> str | None:", manager)
        self.assertNotIn("model._brain_view_sri24_brain_id()", manager)
        self.assertIn('item.get("builtin_template") == "sri24_brain"', manager)
        self.assertIn("def configure_application_window", window_utils)
        self.assertIn("def show_application_window", window_utils)
        self.assertIn("def application_owned_windows", window_utils)
        self.assertIn("setParent(None, Qt.WindowType.Window)", window_utils)
        self.assertIn("Qt.WindowModality.NonModal", window_utils)

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

    def test_dynamic_scene_inspector_is_excluded_from_persistent_ui_values(self) -> None:
        main = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(encoding="utf-8")
        self.assertIn('self.inspector_container.setProperty("persistUiState", False)', main)
        self.assertIn("def _ui_state_persistence_disabled(widget: QWidget)", utils)
        self.assertIn("current = current.parentWidget()", utils)

    def test_unittest_suite_has_no_accidental_pytest_dependency_or_stale_video_bridge_check(self) -> None:
        exposure = (PROJECT_ROOT / "tests" / "test_exposure_panel.py").read_text(encoding="utf-8")
        region = (PROJECT_ROOT / "tests" / "test_region_analysis.py").read_text(encoding="utf-8")
        video = (PROJECT_ROOT / "tests" / "test_video_export.py").read_text(encoding="utf-8")
        self.assertNotIn("import pytest", exposure)
        self.assertNotIn("import pytest", region)
        self.assertIn("unittest.TestCase", exposure)
        self.assertIn("unittest.TestCase", region)
        self.assertIn("test_rendered_tab_export_bridge_uses_shared_single_renderer_path", video)
        self.assertNotIn("test_rendered_tab_bridge_keeps_original_exporter_renderer", video)

    def test_waveform_fullscreen_stops_embedded_playback_first(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn("def stop_playback(self)", source)
        self.assertIn("window.fieldWorkbenchWaveformStop=stopRun", source)
        self.assertIn("self.stop_playback()", source)

    def test_waveform_dialog_exposes_multi_source_signed_absolute_and_excitation_traces(self) -> None:
        dialog_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        waveform_source = (PROJECT_ROOT / "fieldworkbench" / "waveform.py").read_text(encoding="utf-8")
        gpu_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        self.assertIn('QCheckBox("Drive trace")', dialog_source)
        self.assertIn('QCheckBox("Sequencer trace")', dialog_source)
        self.assertIn('self.drive_state_trace.setEnabled(bool(self._base_drive_ids))', dialog_source)
        self.assertIn('include_drive_state_trace = bool(self.drive_state_trace.isChecked())', dialog_source)
        self.assertIn('"exposure_protocol": exposure_protocol', dialog_source)
        self.assertIn('def set_runtime_exposure_protocol', dialog_source)
        self.assertIn('def _active_coil_trace_payload(', waveform_source)
        self.assertIn('def _exposure_drive_trace_payloads(', waveform_source)
        self.assertIn('"excitations": drive_trace_payloads', waveform_source)
        self.assertIn('elif drive_ids:\n            sequencer_payload = _active_coil_trace_payload(', waveform_source)
        self.assertIn('include_sequencer_trace=bool(', waveform_source)
        self.assertIn('QCheckBox("Sensor trace")', dialog_source)
        self.assertIn('QTableWidget(len(self._observation_choices), 4)', dialog_source)
        self.assertIn('["Trace", "Sensor / probe", "Sample", "Signal"]', dialog_source)
        self.assertIn('self.sensor_trace_table.setVisible(table_visible)', dialog_source)
        for component in ("magnitude", "abs_x", "abs_y", "abs_z", "x", "y", "z"):
            self.assertIn(f'"{component}"', dialog_source)
        self.assertIn("def _observation_geometries", waveform_source)
        self.assertIn('"observations": observation_payloads', waveform_source)
        self.assertIn('"excitation": excitation_payload', waveform_source)
        self.assertIn('"sequencer": sequencer_payload', waveform_source)
        self.assertIn("Array.isArray(P.observations)", gpu_source)
        self.assertIn("const sequencerTrace=P.sequencer", gpu_source)

    def test_quick_waveform_playback_has_no_exposure_electrical_or_sequencer_authoring(self) -> None:
        dialog_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        main_source = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        for retired in (
            "use_model_resistance", "manual_resistance", "self.inductance",
            "Advanced sequencer", "sequence_table", "scripted_control_state",
            "restore_scripted_control_state",
        ):
            self.assertNotIn(retired, dialog_source)
        self.assertIn("def configure_quick_current_source(", dialog_source)
        self.assertIn("controller.configure_quick_current_source(", main_source)
        self.assertIn('"electrical_by_coil": copy.deepcopy(electrical_by_coil)', dialog_source)
        self.assertIn('"exposure_protocol": exposure_protocol', dialog_source)

    def test_waveform_dialog_uses_quick_current_generator_and_runtime_exposures(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        for title, expanded in (
            ("Waveform", "True"),
            ("Quick current source", "False"),
            ("Traces", "False"),
        ):
            self.assertIn(f'_CollapsibleSection("{title}", expanded={expanded})', source)
        for choice in ("Sine", "Square", "Triangle"):
            self.assertIn(f'QCheckBox("{choice}")', source)
        self.assertNotIn('QCheckBox("Custom")', source)
        self.assertNotIn('_CollapsibleSection("Wave points"', source)
        self.assertNotIn('wave_points_section', source)
        self.assertNotIn('WaveformImportDialog', source)
        self.assertNotIn('DACImportDialog', source)
        self.assertNotIn('def _load_csv', source)
        self.assertNotIn('def _import_dac', source)
        self.assertNotIn('def set_runtime_waveform_trace(', source)
        self.assertNotIn('def clear_runtime_waveform_trace(', source)
        self.assertNotIn('self._runtime_trace', source)
        self.assertIn('def set_runtime_exposure_protocol(', source)
        self.assertIn('generated_form.addRow("Amplitude", self.waveform_amplitude)', source)
        self.assertIn('generated_form.addRow("Frequency", self.waveform_frequency)', source)
        self.assertIn('generated_form.addRow("Duration", self.waveform_duration)', source)
        self.assertNotIn('Sample multiplier', source)
        self.assertIn('def _automatic_sample_multiplier', source)
        self.assertIn('playback_fps: float = 60.0', source)
        self.assertIn('playback_speed: float = 1.0', source)
        self.assertIn('self.slew_rate = _double_spin(100.0, 0.0, 1e12, 6, " mA/ms")', source)
        self.assertLess(
            source.index('# Quick current source ---------------------------------------------'),
            source.index('# Waveform ---------------------------------------------------------'),
        )

    def test_waveform_import_uses_explicit_profiles_and_general_format_builder(self) -> None:
        editor_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_editor" / "ui" / "main_window.py").read_text(encoding="utf-8")
        import_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_import_dialog.py").read_text(encoding="utf-8")
        format_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_formats.py").read_text(encoding="utf-8")
        io_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_io.py").read_text(encoding="utf-8")
        self.assertIn('WaveformImportDialog', editor_source)
        self.assertIn('Import waveform…', editor_source)
        self.assertNotIn('WaveformImportDialog', (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8"))
        self.assertIn('class WaveformFormatBuilderDialog(QDialog)', import_source)
        self.assertIn('class WaveformImportDialog(QDialog)', import_source)
        self.assertIn('QPushButton("New format…")', import_source)
        self.assertIn('class TabularWaveformImportDialog(QDialog)', import_source)
        self.assertIn('Manual input…', import_source)
        self.assertIn('Waveforms to import', import_source)
        self.assertIn('data_mode: str = "text"', format_source)
        self.assertIn('binary_record_bytes: int = 0', format_source)
        self.assertIn('count_mode: str = "eof"', format_source)
        self.assertIn('output = raw * scale + offset', format_source)
        self.assertNotIn('def detect_reader(', io_source)
        self.assertNotIn('class WaveformProbe', io_source)
        self.assertIn('class _FormatBuilderWheelScrollFilter(QObject)', import_source)
        self.assertIn('for child in body.findChildren(QWidget):', import_source)
        self.assertIn('child.installEventFilter(self._wheel_scroll_filter)', import_source)

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
        self.assertIn("fullHeader=step>0?'Step '+step+(duration?' - '+duration:''):'Off'", source)
        self.assertIn("compactHeader=step>0?'S'+step+(compactDuration?' '+compactDuration:''):'Off'", source)
        self.assertIn("x.font='600 8.5px Arial'", source)
        self.assertIn("x.font='8.5px Arial'", source)
        self.assertIn("active_coil_display_labels", source)
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

    def test_3d_heatmap_colour_limits_use_the_2d_two_column_layout_contract(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        three_d = dialogs.split("class FieldVolumeMapDialog", 1)[1]
        self.assertIn("opacity_options_layout = QGridLayout()", three_d)
        self.assertIn("colour_limits_layout = QGridLayout()", three_d)
        self.assertIn("colour_limits_layout.setColumnStretch(1, 1)", three_d)
        self.assertIn(
            "colour_limits_layout.addWidget(self.colour_minimum_enabled, 0, 0)", three_d
        )
        self.assertIn(
            "colour_limits_layout.addWidget(self.colour_minimum, 0, 1)", three_d
        )
        self.assertIn(
            "colour_limits_layout.addWidget(self.colour_maximum_enabled, 1, 0)", three_d
        )
        self.assertIn(
            "colour_limits_layout.addWidget(self.colour_maximum, 1, 1)", three_d
        )
        self.assertNotIn(
            "display_options_layout.addWidget(self.colour_minimum_enabled", three_d
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
        self.assertNotIn('fit_measurements = QPushButton("Data")', source)
        self.assertNotIn('QPushButton("Apply all changes")', source)
        self.assertIn('self._inspector_section_states: dict[str, bool] = {}', source)
        self.assertIn('section.header.toggled.connect(remember)', source)
        self.assertIn('magnetic_form.addRow("Correction", correction_mode)', source)
        self.assertIn('magnetic_form.addRow("Source", correction_calibration)', source)
        self.assertIn('magnetic_form.addRow("Factor", field_scale)', source)
        self.assertNotIn('correction_layout.addWidget(correction_mode, 0)', source)

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
        self.assertIn('QCheckBox("Loop drive playback to fill timeline")', dialogs)
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
        self.assertIn('"show_field": display_mode != "flux" and not no_field_sources', waveform)
        self.assertIn('"field_sources_present": not no_field_sources', waveform)
        self.assertIn("allow_no_field_sources=True", render_process)
        self.assertIn("allow_no_field_sources=True", waveform)
        self.assertIn('"Scene is empty — nothing to render."', dialogs)
        self.assertIn('render_kind in {"static", "2d_static", "brain_static", "waveform"}', dialogs)
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn("scene_objects = self.snapshot_adapter.list_objects()", main_window)
        self.assertIn('status.showMessage("Scene is empty — nothing to render.")', main_window)
        self.assertIn("adapter._field_fluxline_traces_3d(", waveform)
        self.assertIn("const staticView=P.static_view===true;", gpu)
        self.assertIn("if(showField)drawVolume(M)", gpu)
        self.assertIn("if(showField)drawField(M)", gpu)

    def test_2d_and_3d_maps_use_isolated_background_render_tabs(self) -> None:
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        main_window = (
            PROJECT_ROOT / "fieldworkbench" / "main_window.py"
        ).read_text(encoding="utf-8")
        preferences = (
            PROJECT_ROOT / "fieldworkbench" / "preferences.py"
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
        self.assertIn("Worker concurrency", preferences)
        self.assertIn(
            "configure_field_render_worker_limit(resolved.worker_count)",
            preferences,
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
        preferences = (PROJECT_ROOT / "fieldworkbench" / "preferences.py").read_text(encoding="utf-8")
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


    def test_task_first_toolbar_and_workspace_shells_are_bundled(self) -> None:
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        brain_view = (PROJECT_ROOT / "fieldworkbench" / "brain_view.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        for snippet in (
            '("sensor", "Sensor plot", self.sensor_plot_action)',
            '("volume", "Volume analysis", self.measurement_action)',
            '("optimize", "Optimize", self.optimizer_action)',
            '("check", "DRC", self.drc_action)',
            '("notes", "Notes", self.notes_action)',
        ):
            self.assertIn(snippet, main_window)
        measurement_library = (PROJECT_ROOT / "fieldworkbench" / "measurement_library.py").read_text(encoding="utf-8")
        self.assertIn("class MeasurementDatasetManagerDialog", measurement_library)
        self.assertIn("MeasurementDatasetEditorDialog", measurement_library)
        self.assertIn('QPushButton("Add dataset…")', measurement_library)
        self.assertIn("class MeasurementDatasetEditorDialog(MeasurementCalibrationDialog)", dialogs)
        self.assertNotIn('QGroupBox("Fit source coils")', dialogs)
        self.assertNotIn('fitMeasurementCalibrationButton', dialogs)
        self.assertIn('QTableWidget(0, 6)', dialogs)
        self.assertIn('["Name", "Sensor", "Quantity", "Samples", "Updated (UTC)"]', measurement_library)
        self.assertNotIn('["Name", "Sensor", "Quantity", "Samples", "Own fit", "Fit coils", "Updated (UTC)"]', measurement_library)
        self.assertNotIn("fit_layout.addWidget(self.apply_button)", dialogs)
        self.assertIn('self.link_tree.setHeaderLabels(["Coil", "Correction mode", "Calibration / source", "Resolved factor"])', measurement_library)
        self.assertNotIn('suite_dataset_manager_requested', main_window)
        self.assertNotIn('suite_exposure_editor_requested', main_window)
        self.assertNotIn('suite_maps_analysis_requested', main_window)
        self.assertNotIn('suite_snapshot_manager_requested', main_window)
        self.assertIn("class UnifiedFieldWorkspaceDialog", main_window)
        self.assertIn("QStackedWidget", main_window)
        self.assertIn("fieldWorkspaceModeSelector", main_window)
        self.assertIn("fieldWorkspaceViewSelector", main_window)
        self.assertIn("fieldWorkspaceSidebar", main_window)
        self.assertIn("fieldWorkspaceControlStack", main_window)
        self.assertNotIn("fieldWorkspaceSharedControls", main_window)
        self.assertIn("fieldWorkspaceSharedTabs", main_window)
        self.assertIn("fieldWorkspaceExternalTabBar", main_window)
        self.assertIn("mapsWorkspaceTabStrip", main_window)
        self.assertIn("workspace_tab_strip_layout.addWidget(self.load_saved_map_button, 0)", main_window)
        self.assertIn("workspace_tab_strip_layout.addWidget(self.shared_tabs.workspace_tab_bar, 1)", main_window)
        self.assertIn("fieldWorkspaceSelectorStack", main_window)
        self.assertIn("mapsMeasurementViewSplitter", main_window)
        self.assertIn("mapsMeasurementRegionPanel", main_window)
        self.assertIn("def _sync_workspace_measurement_panel", main_window)
        self.assertIn("def _adopt_page_tabs", main_window)
        self.assertIn("def _shared_tab_changed", main_window)
        self.assertIn("def _close_shared_tab", main_window)
        self.assertIn('page._unified_workspace_result_prefix = self.VIEW_LABELS[view_key]', main_window)
        self.assertIn('def _workspace_result_label(', main_window)
        self.assertIn("scriptedSourceHost", main_window)
        self.assertIn('CollapsibleSection("Scripted source", expanded=False)', main_window)
        self.assertIn("def heightForWidth(self, width: int)", main_window)
        self.assertIn("layout.heightForWidth(max(0, int(width)))", main_window)
        self.assertIn("source_stack_policy.setHeightForWidth(True)", main_window)
        self.assertIn("class _BreathingWordWrapLabel(QLabel)", main_window)
        self.assertIn("return measured + self._EXTRA_VERTICAL_PX", main_window)
        self.assertIn("self.scripted_exposure_summary = _BreathingWordWrapLabel()", main_window)
        self.assertNotIn('section.header.setProperty("persistUiState", False)', main_window)
        self.assertNotIn('controller.traces_section.header.setProperty("persistUiState", False)', main_window)
        self.assertNotIn("Trace choices are configured below Camera", main_window)
        self.assertIn('self.scripted_source_type.addItem("Built-in waveform", "builtin")', main_window)
        self.assertIn('self.scripted_source_type.addItem("Saved exposure", "exposure")', main_window)
        self.assertNotIn('self.scripted_statistics_preview = QRadioButton("Preview")', main_window)
        self.assertNotIn('self.scripted_statistics_full = QRadioButton("Full exposure")', main_window)
        self.assertNotIn('"statistics_scope": statistics_scope', main_window)
        self.assertIn('"preview_duration_s": float(self.scripted_exposure_duration.value())', main_window)
        self.assertIn('self.mode == "static" and self.current_view_key in {"2d", "3d"}', main_window)
        self.assertIn('self.scripted_waveform.addItem("Sine", "sine")', main_window)
        self.assertIn('self.scripted_waveform.addItem("Square", "square")', main_window)
        self.assertIn('self.scripted_waveform.addItem("Triangle", "triangle")', main_window)
        self.assertIn("self.snapshot_adapter.list_exposures()", main_window)
        self.assertNotIn("controller.drive_section", main_window)
        self.assertNotIn("controller.waveform_section", main_window)
        self.assertNotIn("controller.wave_points_section", main_window)
        self.assertNotIn("controller.coil_sequencer_section", main_window)
        self.assertIn("controller.traces_section", main_window)
        self.assertIn("layout.insertWidget(camera_index + 1, traces)", main_window)
        self.assertIn("controller.hide()", main_window)
        self.assertIn("Trace choices and the Traces disclosure are viewer-local settings", main_window)
        self.assertIn("if callable(save):\n                save()", main_window)
        self.assertIn("def _collapse_workspace_sections", main_window)
        self.assertIn('header.setProperty("persistUiState", False)', main_window)
        self.assertIn("header.setChecked(False)", main_window)
        self.assertNotIn("self._collapse_workspace_sections(controller)", main_window)
        self.assertIn("self._collapse_workspace_sections(page)", main_window)
        self.assertIn("camera_index = layout.indexOf(camera_section)", main_window)
        self.assertNotIn('heading = QLabel("Playback protocol")', main_window)
        self.assertIn('waveform_button.setText("Create playback")', main_window)
        self.assertIn("def set_playback_context", (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8"))
        self.assertIn("self.control_root = control_root", dialogs)
        self.assertIn("def _workspace_result_tab_label", dialogs)
        self.assertIn('widget.setProperty("fieldWorkspaceOwnerView", owner)', dialogs)
        self.assertIn('widget.setProperty("fieldWorkspaceOwnerMode", mode)', dialogs)
        self.assertIn("self._switch_view(\"2d\")", main_window)
        self.assertIn("page.setWindowFlags(Qt.WindowType.Widget)", main_window)
        self.assertIn("self.control_stack.addWidget(controls)", main_window)
        self.assertNotIn("control_layout.insertWidget(0, self._make_view_selector(), 0)", main_window)
        self.assertNotIn("class FieldModeWorkspaceDialog", main_window)
        self.assertIn("self.control_column_layout = control_column_layout", dialogs)
        self.assertIn("self.control_column_layout = control_column_layout", brain_view)
        self.assertNotIn("class ResultsWorkspaceDialog", main_window)
        self.assertNotIn("results_workspace_action", main_window)
        self.assertNotIn("open_results_workspace", main_window)
        self.assertNotIn('("gallery", "Saved results"', main_window)
        self.assertNotIn('self.direct_tools_menu = self.analysis_menu.addMenu("Direct views")', main_window)
        self.assertIn('workspace_mode: str | None = None', dialogs)
        self.assertIn('self.waveform_button.setVisible(False)', dialogs)
        self.assertIn('self.calculate_button.setVisible(False)', dialogs)
        self.assertIn('workspace_mode: str | None = None', brain_view)
        self.assertIn('self.animated_analysis_button.setVisible(False)', brain_view)
        self.assertIn('self.static_analysis_button.setVisible(False)', brain_view)
        self.assertIn('Sensor plot | Volume analysis | Optimize | DRC | Notes', help_topics)
        exposures = (PROJECT_ROOT / "fieldworkbench" / "exposures.py").read_text(encoding="utf-8")
        exposure_ui = (PROJECT_ROOT / "fieldworkbench" / "exposure_library.py").read_text(encoding="utf-8")
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        self.assertIn('EXPOSURE_FILE_SUFFIX = ".fwbexposure.json"', exposures)
        self.assertIn('class ExposureLibraryDialog', exposure_ui)
        exposure_analysis = (PROJECT_ROOT / "fieldworkbench" / "exposure_analysis.py").read_text(encoding="utf-8")
        self.assertIn('self.analyze_button = QPushButton("Analyze exposure…")', exposure_ui)
        self.assertIn('ExposureAnalysisWorkspace', exposure_ui)
        workspace = (PROJECT_ROOT / 'fieldworkbench' / 'exposure_workspace.py').read_text(encoding='utf-8')
        self.assertNotIn('ExposureAnalysisSetupDialog', workspace)
        self.assertIn('ExposureAnalysisTargetView', workspace)
        self.assertNotIn('setWindowFlags(Qt.WindowType.Widget)', workspace)
        self.assertNotIn('_TargetAnalysisHub', workspace)
        self.assertIn('self.create_button = QPushButton("Create new analysis"', workspace)
        self.assertNotIn('self.objects_button = QPushButton("← Back to objects")', workspace)
        self.assertNotIn('Preview duration', workspace)
        self.assertNotIn('self.preview_scope', exposure_analysis)
        self.assertNotIn('self.preview_duration = QDoubleSpinBox()', exposure_analysis)
        self.assertNotIn('class ExposureAnalysisSetupDialog', exposure_analysis)
        self.assertNotIn('PySide6', exposure_analysis)
        self.assertIn('class _ExposureAnalysisBuildWorker(QObject)', workspace)
        self.assertIn('configure_application_window(self)', workspace)
        self.assertIn('super().__init__(None)', workspace)
        self.assertIn('_ACTIVE_EXPOSURE_PREPARATION_JOBS', workspace)
        self.assertNotIn('event.ignore()', workspace)
        self.assertNotIn('workspace.exec()', exposure_ui)
        self.assertIn('workspace.show()', exposure_ui)
        self.assertIn('register_window("exposure_analysis", workspace)', exposure_ui)
        self.assertIn('self.preparation_page = self._build_preparation_page()', workspace)
        self.assertIn('self._start_preparation(target_id)', workspace)
        self.assertIn('A cached analysis already exists, are you sure you want to create a new analysis?', workspace)
        self.assertIn('QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No', workspace)
        self.assertIn('exposureWorkspaceSavedAnalysisTree', workspace)
        self.assertIn('self.open_analysis_button = QPushButton("Open selected analysis"', workspace)
        self.assertIn('self.adapter.save_exposure_analysis_resource(result, analysis_cache={})', workspace)
        self.assertNotIn('_load_current_target_view', workspace)
        self.assertIn('def prepare_exposure_analysis_payload(', exposure_analysis)
        self.assertIn('include_authoritative_analysis=True', exposure_analysis)
        self.assertNotIn('def _reusable_saved_region_basis(', exposure_analysis)
        self.assertNotIn('waveform_regions_payload(', exposure_analysis)
        self.assertNotIn('basis_source', exposure_analysis)
        region_analysis = (PROJECT_ROOT / "fieldworkbench" / "region_analysis.py").read_text(encoding="utf-8")
        self.assertIn('def _hierarchy_weighted_rms_moments(', region_analysis)
        self.assertIn('def waveform_regions_payload(', region_analysis)
        spatial_analysis = (PROJECT_ROOT / "fieldworkbench" / "spatial_analysis.py").read_text(encoding="utf-8")
        self.assertIn('class AnalysisRegionSet:', spatial_analysis)
        self.assertIn('class AnalysisSampleSet:', spatial_analysis)
        self.assertIn('def frame_region_metrics(', spatial_analysis)
        self.assertIn('analysis_engine": "regional_analysis_v2', adapter)
        self.assertIn('def _region_spatial_rms_values_direct(', region_analysis)
        shared_actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        project_context = (PROJECT_ROOT / "fieldworkbench" / "project_context.py").read_text(encoding="utf-8")
        saved_results = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        self.assertIn('dependency_signature', exposure_analysis)
        self.assertIn('def save_exposure_analysis_resource(', adapter)
        self.assertIn('def exposure_analysis(self, analysis_id: str)', adapter)
        self.assertIn('def exposure_analysis_dependency_signature(', adapter)
        self.assertIn('EXPOSURE_ANALYSIS_DEPENDENCY_VERSION = 5', adapter)
        self.assertIn('def exposure_analysis_dependency_components(', adapter)
        self.assertIn('def _exposure_analysis_protocol_dependency(', adapter)
        self.assertIn('def _exposure_analysis_source_scene_dependency(', adapter)
        self.assertIn('def _exposure_analysis_coil_dependency(', adapter)
        self.assertIn('def _exposure_analysis_target_dependency(', adapter)
        self.assertNotIn('"scene": copy.deepcopy(self.session.to_dict())', adapter)
        self.assertNotIn('"meshes": copy.deepcopy(self.geometry)', adapter)
        self.assertIn('metadata["exposure_analyses"] = copy.deepcopy(self.exposure_analyses)', adapter)
        self.assertIn('RESOURCE_EXPOSURE_ANALYSIS = "exposure_analysis"', project_context)
        self.assertIn('self.result_tabs.addTab(self.exposure_analysis_page, "Exposure analyses (0)")', saved_results)
        self.assertIn('savedExposureAnalysisTree', saved_results)
        self.assertIn('def _show_managed_window(self, window: QDialog) -> None:', saved_results)
        self.assertIn('configure_application_window(window)', saved_results)
        self.assertNotIn('window.setParent(self, Qt.WindowType.Window)', saved_results)
        self.assertIn('self._show_managed_window(dialog)', saved_results)
        self.assertIn('"Cached work"', saved_results)
        self.assertIn('savedExposureAnalysisDetailsGroup', saved_results)
        self.assertIn('class ExposureAnalysisCacheManagerDialog(QDialog):', saved_results)
        self.assertIn('Manage cache…', saved_results)
        self.assertIn('Remove all derived results', saved_results)
        self.assertIn('Last removal reclaimed', saved_results)
        self.assertIn('individual rows do not have independent reclaimable sizes', saved_results)
        self.assertIn('Delete cache', saved_results)
        self.assertIn('def delete_exposure_analysis_cached_results(', adapter)
        self.assertIn('def _exposure_analysis_cache_summary(', adapter)
        self.assertIn('"cached_tool_counts"', adapter)
        self.assertIn('EXPOSURE_ANALYSIS_FORMAT_VERSION = 5', adapter)
        self.assertIn('EXPOSURE_ANALYSIS_BLOB_CODEC = "zlib-json-binary-v1"', adapter)
        self.assertIn('_EXPOSURE_ANALYSIS_BLOB_MAGIC = b"FWBEA5\\0"', adapter)
        self.assertIn('def _exposure_analysis_value_blob(', adapter)
        self.assertIn('def _exposure_analysis_cache_blob(', adapter)
        self.assertIn('def update_exposure_analysis_cache(', adapter)
        self.assertIn('def cache_analysis_region_set(', adapter)
        self.assertIn('def cached_analysis_region_set(', adapter)
        self.assertIn('def list_regional_geometry_caches(', adapter)
        self.assertIn('def delete_regional_geometry_caches(', adapter)
        surface_volume_method = adapter.split('    def surface_region_volume_set(', 1)[1].split('\n    def ', 1)[0]
        self.assertLess(
            surface_volume_method.index('self.cached_analysis_region_set('),
            surface_volume_method.index('build_surface_region_volume_set('),
        )
        self.assertIn('def exposure_analysis_cache_identities(', adapter)
        self.assertIn('"preparation_identity"', adapter)
        self.assertIn('"exposure_data_identity"', adapter)
        self.assertIn('"geometry_cache_identity"', adapter)
        self.assertIn('"preparation_compressed_bytes"', adapter)
        self.assertIn('"cache_compressed_bytes"', adapter)
        self.assertNotIn('Save analysis…', shared_actions)
        self.assertNotIn('def find_current_exposure_analysis(', adapter)
        self.assertNotIn('statistics_scope', workspace)
        self.assertNotIn('statistics_scope', saved_results)
        self.assertNotIn('statistics_duration_s', workspace)
        self.assertNotIn('def update_exposure_analysis_resource(', adapter)
        self.assertNotIn('def upsert_exposure_analysis_resource(', adapter)
        self.assertIn('self.project_adapter.update_exposure_analysis_cache(', shared_actions)
        self.assertIn('on_cache_changed=', exposure_ui)
        self.assertIn('_store_analysis_cache_result(', shared_actions)
        self.assertIn('class ExposureEditorDialog', exposure_ui)
        self.assertIn('class ExposureEditorWindow(QMainWindow)', exposure_ui)
        self.assertIn('class _DriveGroupDialog(QDialog)', exposure_ui)
        self.assertIn('self.source_table.setHeaderLabels(["Source", "Waveform", "Phase (s)", "Mode"])', exposure_ui)
        self.assertIn('mode.addItem("Continuous", "continuous")', exposure_ui)
        self.assertIn('mode.addItem("Once per step", "once_per_step")', exposure_ui)
        self.assertIn('self.duration_unit.addItem("min", 60.0)', exposure_ui)
        self.assertIn('self.duration_unit.addItem("hour", 3600.0)', exposure_ui)
        self.assertIn('self.duplicate_step_button = QPushButton("Duplicate")', exposure_ui)
        self.assertIn('def _duplicate_selected(self, *_args) -> None:', exposure_ui)
        self.assertIn('def _duplicate_step(self, *_args) -> None:', exposure_ui)
        self.assertIn('def _duplicate_drive_group(self, *_args) -> None:', exposure_ui)
        self.assertNotIn('def _source_dialog(', exposure_ui)
        self.assertNotIn('"Edit source…"', exposure_ui)
        self.assertIn('class _ExposureOneShotCursor(_ExposureSourceCursor)', adapter)
        self.assertIn('self.connection.addItem("Series", "series")', exposure_ui)
        self.assertIn('self.connection.addItem("Parallel", "parallel")', exposure_ui)
        self.assertIn('def resolve_exposure_drive_group', adapter)
        self.assertIn('def resolve_exposure_step_electrical', adapter)
        self.assertIn('def resolve_exposure_currents_at_times', adapter)
        self.assertIn('def _expanded_exposure_source_trace', adapter)
        self.assertNotIn('def resolve_exposure_waveform_trace', adapter)
        self.assertNotIn('_sequence_from_legacy', exposure_ui)
        self.assertNotIn('_legacy_projection', exposure_ui)
        self.assertIn('def waveform_exposure_references', adapter)
        project_manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(encoding="utf-8")
        self.assertIn('ExposureEditorWindow(self.record.context)', project_manager)
        self.assertNotIn('def open_exposures(self) -> None:\n        self.ensure_model_window().open_exposures()', project_manager)
        self.assertIn('metadata["exposures"]', adapter)
        self.assertIn('def list_exposures', adapter)
        self.assertIn('def save_exposure', adapter)
        self.assertIn('def import_exposure_file', adapter)
        waveform_dialog = (PROJECT_ROOT / "fieldworkbench" / "waveform_dialog.py").read_text(encoding="utf-8")
        self.assertNotIn('def scripted_control_state', waveform_dialog)
        self.assertNotIn('def restore_scripted_control_state', waveform_dialog)
        self.assertIn('def configure_quick_current_source', waveform_dialog)
        self.assertIn('def set_runtime_exposure_protocol', waveform_dialog)
        self.assertIn('"exposure_protocol": exposure_protocol', waveform_dialog)
        self.assertNotIn('def exposure_protocol_state', waveform_dialog)
        self.assertIn('controller.set_runtime_exposure_protocol(', main_window)
        self.assertNotIn('Phase 4.5 will place those resolved per-coil currents', main_window)
        self.assertIn('"saved_exposures"', help_topics)


    def test_measurement_and_exposure_editors_are_wheel_safe_and_roomier(self) -> None:
        window_utils = (PROJECT_ROOT / "fieldworkbench" / "window_utils.py").read_text(encoding="utf-8")
        measurement_library = (PROJECT_ROOT / "fieldworkbench" / "measurement_library.py").read_text(encoding="utf-8")
        dialogs = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        exposure_ui = (PROJECT_ROOT / "fieldworkbench" / "exposure_library.py").read_text(encoding="utf-8")
        import_ui = (PROJECT_ROOT / "fieldworkbench" / "waveform_import_dialog.py").read_text(encoding="utf-8")

        self.assertIn("class EditorWheelGuard(QObject)", window_utils)
        self.assertIn("def install_editor_wheel_guard(", window_utils)
        app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("install_application_editor_wheel_guard(application)", app_source)
        self.assertIn("install_editor_wheel_guard(self.window)", window_utils)
        self.assertIn("_field_workbench_editor_wheel_guard", window_utils)
        self.assertIn("QAbstractSpinBox", window_utils)
        self.assertIn("Qt.FocusPolicy.WheelFocus", window_utils)
        self.assertIn("editor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)", window_utils)
        self.assertIn("self.resize(1260, 860)", measurement_library)
        self.assertIn("self.resize(1320, 880)", dialogs)
        self.assertIn("mode_combo.installEventFilter(self._link_wheel_guard)", measurement_library)
        self.assertIn("source_combo.installEventFilter(self._link_wheel_guard)", measurement_library)
        self.assertIn("populate=populate_source_combo", measurement_library)
        self.assertIn("coil_id=coil_id,", measurement_library)
        self.assertIn("self._measurement_wheel_guard = install_editor_wheel_guard(self)", dialogs)
        self.assertIn("self.sequence_tree = QTreeWidget()", exposure_ui)
        self.assertIn('self.sequence_tree.setItemWidget(root, 1, duration)', exposure_ui)
        self.assertIn('duration.setSuffix(" s")', exposure_ui)
        self.assertIn('duration = float(self._steps[-1].get("duration_s", 1.0)) if self._steps else 1.0', exposure_ui)
        self.assertNotIn('QInputDialog.getDouble(self, "Add exposure step"', exposure_ui)
        self.assertIn("self.coil_list = QListWidget()", exposure_ui)
        self.assertIn("self.load_override_table = QTreeWidget()", exposure_ui)
        self.assertIn("self.load_override_table.setMinimumHeight(155)", exposure_ui)
        self.assertIn("self.load_override_table.setMaximumHeight(220)", exposure_ui)
        self.assertNotIn('QPushButton("Edit waveform…")', exposure_ui)
        self.assertIn("Resistance override (Ω)", exposure_ui)
        self.assertIn("Inductance override (mH)", exposure_ui)
        self.assertNotIn("No voltage-source current or power limit is modeled yet", exposure_ui)
        self.assertIn("self._wheel_guard = install_editor_wheel_guard(self)", import_ui)
        self.assertIn("class _FormatBuilderWheelScrollFilter(QObject)", import_ui)

    def test_exposure_analysis_target_view_uses_persistent_stacked_pages(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        panel_source = (PROJECT_ROOT / "fieldworkbench" / "exposure_panel.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        self.assertIn("QStackedWidget", source)
        header_source = (PROJECT_ROOT / "fieldworkbench" / "exposure_analysis_header.py").read_text(encoding="utf-8")
        self.assertIn('QPushButton("← Analysis home"', header_source)
        self.assertIn('QPushButton("← Analysis targets"', header_source)
        self.assertIn("ExposureAnalysisHeaderPanel", source)
        self.assertNotIn('self.footer_back_button = QPushButton("Back")', source)
        self.assertNotIn('footer_close_button', source)
        self.assertIn('class ExposureAnalysisTargetView(QWidget)', source)
        self.assertIn('class ExposureAnalysisWindow(QDialog)', source)
        self.assertIn('configure_application_window(self)', source)
        self.assertNotIn('self.footer_back_button.setVisible(not on_home)', source)
        self.assertIn('self.analysis_footer_status = QWidget(self)', source)
        self.assertIn('self.analysis_footer_layout = QVBoxLayout(self.analysis_footer_status)', source)
        self.assertIn('class _AnalysisProgressRow(QWidget)', source)
        self.assertIn('ACTIVE_STATES = {"queued", "running", "pausing", "paused", "cancelling"}', source)
        self.assertIn('self.calculate_all_button = QPushButton("Calculate all uncached analyses", body)', source)
        self.assertIn('self._analysis_job_queue = []', source)
        self.assertIn('def _schedule_analysis_job(self, page, function):', source)
        self.assertIn('Queued — waiting for current calculation', source)
        self.assertIn('self.pause_button = QPushButton("Pause", self)', source)
        self.assertIn('self.pause_button.setText("Run" if pause_requested else "Pause")', source)
        self.assertNotIn('self.cancel_analysis_button = QPushButton("Cancel calculation")', source)
        self.assertIn('owner._analysis_progress_time_text = f"Processing time {_format_analysis_elapsed(elapsed)}"', source)
        self.assertIn('owner._analysis_progress_message = "Loaded saved result"', source)
        self.assertIn('owner._analysis_progress_time_text = "No recalculation"', source)
        self.assertIn('Original processing time {_format_analysis_elapsed(original_elapsed)}', source)
        self.assertIn('page._analysis_cache_changed_callback = self._analysis_cache_changed', source)
        self.assertIn('class _AnalysisCard(QFrame)', source)
        self.assertIn('self._analysis_home_cards[card_key] = card', source)
        self.assertIn('Open spectrum / FFT…', source)
        self.assertIn('<b>Method basis:</b>', source)
        self.assertIn('<b>Research precedent:</b>', source)
        self.assertIn('<b>Reference frameworks:</b>', source)
        self.assertNotIn('self.analysis_table.setHorizontalHeaderLabels(["Analysis", "Status", "Action"])', source)
        self.assertIn('def _analysis_home_cache_statuses(cache):', source)
        self.assertIn('self._refresh_home_cache_status()', source)
        self.assertIn('self.preparation_state_label = QLabel(f"Preparation: {preparation_state}"', source)
        self.assertIn('Save failed — calculated result is only in this window.', source)
        self.assertIn('self.retry_save_button = QPushButton("Retry save", self)', source)
        self.assertNotIn('manual_run=True', source)
        self.assertNotIn('recalculate_button', source)
        self.assertNotIn('def run_profile(', source)
        self.assertIn('class ExposureSummaryPage(_CancellableAnalysisPage):', source)
        self.assertIn('self.region_panel = ExposureRegionPanel(', source)
        self.assertIn('self.region_panel.configure_exposure_summary()', source)
        self.assertIn('set_exposure_summary_results(results)', source)
        self.assertIn('EXPOSURE_SUMMARY_HEADER_LABELS', panel_source)
        self.assertIn('("Mean RMS |B|", 1)', panel_source)
        self.assertIn('("Exposure RMS |B|", 2)', panel_source)
        self.assertIn('("B²-time", 6)', panel_source)
        self.assertIn('"widths": [250, 128, 142, 128, 112, 124, 132, 72]', panel_source)
        self.assertNotIn('Filter and Highlight rules use the same controls as Brain Areas', panel_source)
        self.assertIn('self.metrics_note.hide()', panel_source)
        self.assertIn('class _WrappedToolTipHeaderView(QHeaderView):', panel_source)
        self.assertIn('QToolTip.showText(event.globalPos(), wrapped, self)', panel_source)
        self.assertIn('self.toggle_filters_button = QPushButton("Show/hide filters")', panel_source)
        self.assertIn('self.filter_heading.setText("Filters / highlights")', panel_source)
        self.assertIn('self.region_sections_splitter.insertWidget(0, self.table_section)', panel_source)
        self.assertIn('self._set_exposure_summary_filters_visible(False)', panel_source)
        self.assertIn('all_regions=True', source)
        self.assertIn('self._show_region_browser and not on_home and not owns_region_browser', source)
        self.assertIn('self.region_tree.hide()\n        self.selection_note.hide()', source)
        self.assertIn('return analyse_regions_threshold_duration(owner.payload, request_key[2], progress_callback=progress)', source)
        self.assertIn('self._show_page("spectrum")', source)
        self.assertIn('self._show_page("weighted")', source)
        self.assertIn('self._show_page("fft")', source)
        self.assertNotIn("page.setWindowFlags(Qt.WindowType.Widget)", source)
        self.assertIn("class _CancellableAnalysisPage(QWidget):", source)
        self.assertIn("class FFTSpectrumPage(_CancellableAnalysisPage):", source)
        self.assertIn("class SpectrumAnalysisPage(_CancellableAnalysisPage):", source)
        self.assertIn("class TimingAnalysisPage(_CancellableAnalysisPage):", source)
        self.assertIn("class ThresholdDurationPage(_CancellableAnalysisPage):", source)
        self.assertIn("class ExposureSummaryPage(_CancellableAnalysisPage):", source)
        self.assertIn("class FrequencyWeightedPage(_CancellableAnalysisPage):", source)
        self.assertIn("class FrequencyWeightedResultsWindow(QDialog):", source)
        self.assertNotIn("class _CancellableAnalysisDialog", source)
        self.assertNotIn("ExposureSummaryDialog", source)
        self.assertNotIn("SpectrumAnalysisDialog", source)
        self.assertNotIn("FFTSpectrumDialog", source)
        self.assertNotIn("TimingAnalysisDialog", source)
        self.assertNotIn("ThresholdDurationDialog", source)
        self.assertNotIn("FrequencyWeightedDialog", source)
        self.assertNotIn("_embed_dialog_page", source)
        self.assertIn("def _configure_analysis_page(self, page, page_key):", source)
        self.assertIn("Object chooser and analysis workspace", help_topics)
        self.assertIn("Exposure analysis home", help_topics)
        self.assertIn("literature or reference framework", help_topics)
        self.assertIn("← Analysis home", help_topics)


    def test_frequency_weighted_reference_queue_and_cached_browsing_are_present(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        self.assertIn('self._weighted_pending_requests = []', source)
        self.assertIn('QPushButton("Calculate all uncached references", self)', source)
        self.assertIn('def calculate_all_uncached_profiles(self, *_args):', source)
        self.assertIn('def _analysis_job_released(self):', source)
        self.assertIn('def _analysis_cancel_requested(self):', source)
        self.assertIn('view.setText("Queue")', source)
        self.assertIn('view.setText("Open")', source)
        self.assertIn('tree.setEnabled(not busy or isinstance(page, FrequencyWeightedPage))', source)
        self.assertIn('self._advance_analysis_job_queue', source)
        self.assertIn('Calculate all uncached references', help_topics)
        self.assertIn('Cached results remain fully browsable', help_topics)


    def test_frequency_weighted_persistent_cache_is_compact(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        self.assertIn('def _compact_frequency_weighted_result(result):', source)
        self.assertIn('"weighted_spatial_rms_uT",', source)
        self.assertIn('"spectral_top_rows"', source)
        self.assertIn('dtype=np.float32', source)
        self.assertIn('str(request_key[0]) == "weighted"', source)
        self.assertIn('data.get("spectral_top_rows", ())', source)
        self.assertNotIn('ratio = np.asarray(data["spectral_ratio_by_bin"]', source)
        self.assertIn('compact derived cache', help_topics)
        self.assertIn('30 strongest spectral rows', help_topics)


    def test_frequency_weighted_results_use_right_side_viewer_splitter(self) -> None:
        source = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")

        self.assertIn('self.result_splitter = QSplitter(Qt.Orientation.Horizontal, self.result_window)', source)
        self.assertIn('class FrequencyWeightedResultsWindow(QDialog):', source)
        self.assertIn('self.result_window = FrequencyWeightedResultsWindow(self)', source)
        self.assertIn('self.result_splitter.setObjectName("frequencyWeightedResultSplitter")', source)
        self.assertIn('self.result_splitter.setChildrenCollapsible(False)', source)
        self.assertIn('self.result_splitter.addWidget(self.result_text_panel)', source)
        self.assertIn('self.result_splitter.addWidget(self.result_plot_panel)', source)
        self.assertIn('self.plot = PlotView(self.result_plot_panel)', source)
        self.assertIn('result_text_layout.addWidget(self.table, 1)', source)
        self.assertIn('self.result_splitter.setStretchFactor(0, 3)', source)
        self.assertIn('self.result_splitter.setStretchFactor(1, 2)', source)


    def test_fft_setting_changes_rehome_2d_axes_without_rerun_action(self) -> None:
        actions = (PROJECT_ROOT / "fieldworkbench" / "exposure_actions.py").read_text(encoding="utf-8")
        plot_view = (PROJECT_ROOT / "fieldworkbench" / "plot_view.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        self.assertNotIn('QPushButton("Recalculate FFT")', actions)
        self.assertNotIn('self.recalculate_button', actions)
        self.assertIn('self.samples.valueChanged.connect(self.recalculate)', actions)
        self.assertIn('self.plot.request_reset_2d_view_on_next_figure()', actions)
        self.assertIn('QPushButton("Fullscreen viewer")', actions)
        self.assertIn('self.plot.toggle_fullscreen()', actions)
        self.assertIn('true-fullscreen interactive copy of the FFT plot', help_topics)
        self.assertIn('"autorange": max_hz is None', actions)
        self.assertIn('"autorange": True', actions)
        self.assertIn('def request_reset_2d_view_on_next_figure', plot_view)
        self.assertIn('reset2dView', plot_view)
        self.assertIn('Plot max frequency', help_topics)


    def test_measurement_calibration_correction_links_are_wired_into_scene_and_ui(self) -> None:
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")
        manager = (PROJECT_ROOT / "fieldworkbench" / "measurement_library.py").read_text(encoding="utf-8")
        help_topics = (PROJECT_ROOT / "fieldworkbench" / "help_topics.py").read_text(encoding="utf-8")

        self.assertIn('"correction_mode": "custom"', adapter)
        self.assertNotIn('"correction_dataset_id"', adapter)
        self.assertIn('"correction_calibration_id": ""', adapter)
        self.assertIn('"custom_field_scale_factor": 1.0', adapter)
        self.assertIn('def set_coil_correction_calibration', adapter)
        self.assertNotIn('def set_coil_correction_dataset', adapter)
        self.assertIn('def measurement_calibration_linked_coils', adapter)
        self.assertIn('def fit_measurement_composite_calibration', adapter)
        self.assertIn('def measurement_calibration_status', adapter)
        self.assertNotIn('can only be linked to coils included in its fitted target group', adapter)
        self.assertIn('correction_mode.addItem("Custom", "custom")', main_window)
        self.assertIn('correction_mode.addItem("Calibration", "calibration")', main_window)
        self.assertIn('QLabel("3 · Coil correction links")', manager)
        self.assertIn('QGroupBox("Fit coil group")', manager)
        self.assertIn('"Fit coils"', manager)
        self.assertIn('Attach any ready Calibration / Composite fit to any coil in the scene.', manager)
        self.assertIn('mode_combo.addItem("Calibration", "calibration")', manager)
        self.assertIn('Each source selection equal', manager)
        self.assertIn('Calibration / Composite fits', help_topics)

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
        self.assertIn('"2D field map"', help_topics)
        self.assertIn('"3D field map"', help_topics)
        self.assertIn('"Brain model and atlas"', help_topics)
        self.assertIn('"Playback overview"', help_topics)
        self.assertIn('"Exposure analysis"', help_topics)
        self.assertIn('"Spectrum and FFT"', help_topics)
        self.assertIn('"Timing and transitions"', help_topics)
        self.assertIn('"Threshold and duration"', help_topics)
        self.assertIn('"Frequency-weighted screening"', help_topics)
        self.assertIn('"General spatial statistics"', help_topics)
        self.assertIn('"Exposure statistics and research basis"', help_topics)
        self.assertIn("Filter</b> is an exclusion rule", help_topics)
        self.assertIn("Exposure RMS = sqrt(B²-time / exposure time)", help_topics)
        self.assertNotIn("numerical analysis timeline", help_topics)
        self.assertNotIn("Whole-playback metrics are in", help_topics)
        self.assertIn("Röösli", help_topics)



    def test_dataset_manager_is_project_native_and_model_editor_routes_to_it(self) -> None:
        manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(encoding="utf-8")
        measurement_library = (PROJECT_ROOT / "fieldworkbench" / "measurement_library.py").read_text(encoding="utf-8")
        main_window = (PROJECT_ROOT / "fieldworkbench" / "main_window.py").read_text(encoding="utf-8")

        self.assertIn("class DatasetManagerWindow", measurement_library)
        self.assertIn("ProjectContext", measurement_library)
        self.assertIn("embedded=True", measurement_library)
        self.assertIn("DatasetManagerWindow", manager)
        self.assertIn('register_window("dataset_manager", window)', manager)
        self.assertIn("def open_datasets(", manager)
        self.assertIn("window.open_dataset(dataset_id)", manager)
        self.assertIn("window.open_calibration(calibration_id)", manager)
        self.assertNotIn("suite_dataset_manager_requested", main_window)
        self.assertIn("def external_measurement_datasets_changed", main_window)

    def test_waveform_editor_is_project_native_and_reuses_global_theme(self) -> None:
        editor_root = PROJECT_ROOT / "fieldworkbench" / "waveform_editor"
        self.assertTrue((editor_root / "ui" / "main_window.py").is_file())
        self.assertTrue((editor_root / "ui" / "native_editor.py").is_file())
        self.assertFalse((editor_root / "ui" / "builder.py").exists())
        self.assertFalse((editor_root / "ui" / "free_form_canvas.py").exists())
        self.assertTrue((editor_root / "ui" / "formatting.py").is_file())
        self.assertTrue((editor_root / "ui" / "preview.py").is_file())
        self.assertTrue((editor_root / "ui" / "embedded_imports.py").is_file())
        self.assertTrue((editor_root / "core" / "native.py").is_file())
        self.assertTrue((editor_root / "core" / "encoding.py").is_file())
        self.assertFalse((editor_root / "core" / "waveforms.py").exists())
        self.assertFalse((editor_root / "core" / "model.py").exists())
        self.assertFalse((editor_root / "ui" / "preferences_state.py").exists())
        self.assertFalse((editor_root / "ui" / "plot.py").exists())
        self.assertFalse((editor_root / "ui" / "dialogs").exists())

        imported_modules: set[str] = set()
        for source_path in editor_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module.split(".")[0])
        self.assertNotIn("matplotlib", imported_modules)

        main_window = (editor_root / "ui" / "main_window.py").read_text(encoding="utf-8")
        theme_bridge = (editor_root / "ui" / "theme.py").read_text(encoding="utf-8")
        self.assertIn("class WaveformEditorWindow", main_window)
        self.assertIn("ProjectContext", main_window)
        self.assertIn("WaveformImportDialog", main_window)
        self.assertIn("export_selected_waveform", main_window)
        self.assertIn("export_csv_waveform", main_window)
        self.assertIn("save_waveform_document", main_window)
        self.assertIn("NativeWaveformEditor", main_window)
        self.assertIn('catalogue_buttons.addWidget(self.create_resource_button, 0, 0)', main_window)
        self.assertIn('catalogue_buttons.addWidget(self.edit_button, 0, 1)', main_window)
        self.assertIn('catalogue_buttons.addWidget(self.rename_resource_button, 1, 0)', main_window)
        self.assertIn('catalogue_buttons.addWidget(self.duplicate_resource_button, 1, 1)', main_window)
        self.assertIn('catalogue_buttons.addWidget(self.import_resource_button, 2, 0)', main_window)
        self.assertIn('catalogue_buttons.addWidget(self.export_button, 2, 1)', main_window)
        self.assertIn('catalogue_buttons.addWidget(self.delete_resource_button, 3, 0, 1, 2)', main_window)
        self.assertIn('catalogue_buttons.addWidget(self.close_button, 4, 0, 1, 2)', main_window)
        self.assertIn('waveformEditorLeftStack', main_window)
        self.assertIn('QGroupBox("Waveform encoding"', main_window)
        self.assertIn('QGroupBox("Waveform information", self.lower_stack)', main_window)
        self.assertIn('"Create waveform", self.encoding_setup_group', main_window)
        self.assertIn('self.edit_encoding_button = QPushButton("Edit"', main_window)
        self.assertIn('QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed', main_window)
        self.assertIn('waveformWorkflowBar', main_window)
        self.assertIn('waveformEditorCentral', main_window)
        self.assertIn('self.workflow_bar.setFrameShadow(QFrame.Shadow.Sunken)', main_window)
        self.assertNotIn('self.workflow_actions = QWidget', main_window)
        self.assertIn('self.rename_resource_button.setEnabled(catalogue_selection_enabled)', main_window)
        self.assertIn('self.duplicate_resource_button.setEnabled(catalogue_selection_enabled)', main_window)
        self.assertIn('self.delete_resource_button.setEnabled(catalogue_selection_enabled)', main_window)
        self.assertIn('waveformWorkflowMode', main_window)
        self.assertIn('"Waveform navigator"', main_window)
        self.assertIn('"Waveform encoding"', main_window)
        self.assertIn('"Waveform editor"', main_window)
        self.assertIn('"Save waveform", self.authoring_actions', main_window)
        self.assertIn('QGroupBox("Components", self.edit_options_group)', main_window)
        self.assertIn('waveformChunkList', main_window)
        self.assertIn('QGroupBox("Component actions", self.edit_options_group)', main_window)
        self.assertIn('self.edit_chunk_button', main_window)
        self.assertIn('self.clone_chunk_button', main_window)
        self.assertIn('self.move_chunk_up_button', main_window)
        self.assertIn('self.move_chunk_down_button', main_window)
        self.assertIn('def move_selected_chunk_up', main_window)
        self.assertIn('def move_selected_chunk_down', main_window)
        self.assertIn('self.delete_chunk_button', main_window)
        self.assertNotIn('self.import_chunk_button', main_window)
        self.assertNotIn('self.import_dac_action', main_window)
        self.assertNotIn('self.import_dac_chunk_button', main_window)
        self.assertIn('def clone_selected_chunk', main_window)
        self.assertIn('def delete_selected_chunk', main_window)
        self.assertNotIn('self.edit_authored_button', main_window)
        self.assertIn('self.import_tool_button', main_window)
        self.assertIn('("Import waveform…", self.import_native_waveform, "import")', main_window)
        self.assertNotIn('self.import_dac_tool_button', main_window)
        self.assertIn('self.notes_panel.setVisible(False)', main_window)
        self.assertIn('button.setMinimumSize(130, 82)', main_window)
        self.assertIn('elif kind == "import"', (editor_root / "ui" / "widgets.py").read_text(encoding="utf-8"))
        self.assertNotIn('self.task_label', main_window)
        self.assertIn('waveformEditorLowerStack', main_window)
        self.assertIn('self.lower_stack.setCurrentWidget(self.encoding_setup_group)', main_window)
        self.assertIn('self.lower_stack.setCurrentWidget(self.toolbox_group)', main_window)
        self.assertIn('self._encoding_confirmed', main_window)
        self.assertIn('waveformEmbeddedToolStack', main_window)
        self.assertIn('EmbeddedGeneratorPanel', main_window)
        embedded_tools = (editor_root / "ui" / "embedded_tools.py").read_text(encoding="utf-8")
        preview_source = (editor_root / "ui" / "preview.py").read_text(encoding="utf-8")
        self.assertIn('def add_free_form_point', preview_source)
        self.assertIn('self.owner.native_editor.plot.add_free_form_point()', embedded_tools)
        self.assertIn('"Duration (ms)"', embedded_tools)
        self.assertIn('"Cycle count"', embedded_tools)
        self.assertIn('def _periodic_cycle_count_changed', embedded_tools)
        self.assertIn('def _periodic_duration_changed', embedded_tools)
        self.assertIn('def _validate_action_potential_inputs', embedded_tools)
        self.assertIn('set_external_invalid', embedded_tools)
        self.assertNotIn('Define one stylized event.', embedded_tools)
        self.assertIn('self.controls.addWidget(timing_group, 0, 0)', embedded_tools)
        self.assertIn('self.controls.addWidget(levels_group, 1, 0)', embedded_tools)
        self.assertIn('focus_generated_preview', embedded_tools)
        self.assertIn('"Reset free-form curve"', embedded_tools)
        self.assertIn('QMessageBox.StandardButton.Cancel', embedded_tools)
        self.assertIn('EmbeddedWaveformImportPanel', main_window)
        self.assertIn('EmbeddedRuntimeFormatPanel', main_window)
        self.assertIn('on_runtime_profile_source=self._open_embedded_runtime_profile_source', main_window)
        embedded_imports = (editor_root / "ui" / "embedded_imports.py").read_text(encoding="utf-8")
        self.assertNotIn('("dac", "")', embedded_imports)
        self.assertIn('builtin_waveform_format_profiles()', embedded_imports)
        waveform_import_dialog = (PROJECT_ROOT / "fieldworkbench" / "waveform_import_dialog.py").read_text(encoding="utf-8")
        self.assertIn('builtin_waveform_format_profiles()', waveform_import_dialog)
        self.assertIn('"Waveform encoding"', embedded_imports)
        self.assertIn('self.target_quantity', embedded_imports)
        self.assertIn('self.target_unit', embedded_imports)
        self.assertIn('self._raw_dac_codes = self.profile.profile_id == "builtin.dac_8bit"', embedded_imports)
        self.assertIn('"Point delay (ms)" if self._raw_dac_codes', embedded_imports)
        self.assertIn('f"Code {self.profile.runtime_raw_min:g} ="', embedded_imports)
        self.assertIn('raw["output_quantity"] = self.target_quantity', embedded_imports)
        self.assertNotIn('form.addRow("Signal type", self.quantity)', embedded_imports)
        self.assertNotIn('form.addRow("Units", self.unit)', embedded_imports)
        self.assertNotIn('table_form.addRow("Signal type", self.quantity)', embedded_imports)
        self.assertNotIn('table_form.addRow("Units", self.units)', embedded_imports)
        self.assertIn('def _preview_embedded_import', main_window)
        self.assertIn('def _apply_embedded_import', main_window)
        self.assertIn('def _preview_embedded_generator', main_window)
        self.assertIn('def _apply_embedded_generator', main_window)
        self.assertIn('self.native_editor.show_generated_preview', main_window)
        self.assertIn('self.native_editor.replace_generated_recipe_segment', main_window)
        self.assertIn('self.native_editor.clone_recipe_segment', main_window)
        self.assertIn('self.native_editor.delete_recipe_segment', main_window)
        self.assertIn('waveformEditorNotesSplitter', main_window)
        self.assertIn('waveformToggleNotesButton', main_window)
        self.assertIn('self.notes_stack.setCurrentWidget(self.edit_notes)', main_window)
        self.assertIn('Qt.ContextMenuPolicy.ActionsContextMenu', main_window)
        self.assertNotIn('Replace the current unsaved working waveform', main_window)
        self.assertNotIn('self.browse_actions', main_window)
        self.assertNotIn("Authoring mode", main_window)
        self.assertNotIn("8-bit authoring", main_window)
        self.assertNotIn("16-bit authoring", main_window)
        self.assertIn("RuntimeFormatImportDialog", waveform_import_dialog)
        self.assertIn("DacExportDialog", main_window)
        self.assertIn('shortcut="Esc"', main_window)
        self.assertIn("Apply or Cancel the active waveform tool preview before saving.", main_window)
        self.assertIn("Discard the active waveform tool preview and continue closing?", main_window)
        self.assertNotIn("_dialog_cache", main_window)
        self.assertNotIn("NativeSquareDialog", main_window)
        self.assertNotIn("NativeSineDialog", main_window)
        self.assertNotIn("NativeDCDialog", main_window)
        self.assertNotIn("NativeActionPotentialDialog", main_window)
        self.assertNotIn("NativeFreeFormDialog", main_window)
        native_editor = (editor_root / "ui" / "native_editor.py").read_text(encoding="utf-8")
        formatting = (editor_root / "ui" / "formatting.py").read_text(encoding="utf-8")
        preview = (editor_root / "ui" / "preview.py").read_text(encoding="utf-8")
        dialogs = (editor_root / "ui" / "native_dialogs.py").read_text(encoding="utf-8")
        self.assertIn("class NativeWaveformEditor", native_editor)
        self.assertIn("class _SignalState", native_editor)
        self.assertIn("WaveformSeries", native_editor)
        self.assertIn("WaveformPreviewCanvas", native_editor)
        self.assertIn("def current_signal", native_editor)
        self.assertIn("def show_generated_preview", native_editor)
        self.assertIn("def focus_generated_preview", native_editor)
        self.assertIn("def focus_time_range", preview)
        self.assertIn("def show_document_preview", native_editor)
        self.assertIn("def restore_working_preview", native_editor)
        embedded = (editor_root / "ui" / "embedded_tools.py").read_text(encoding="utf-8")
        self.assertIn("class EmbeddedGeneratorPanel", embedded)
        embedded_imports = (editor_root / "ui" / "embedded_imports.py").read_text(encoding="utf-8")
        self.assertIn("class EmbeddedWaveformImportPanel", embedded_imports)
        self.assertIn("class EmbeddedRuntimeFormatPanel", embedded_imports)
        self.assertIn("documents_from_text_table", embedded_imports)
        self.assertIn("encoded_segment", embedded_imports)
        self.assertIn("waveformEncodedSamplesEditor", embedded_imports)
        self.assertIn("def show_recipe_segment_preview", native_editor)
        self.assertIn("def replace_recipe_segment", native_editor)
        self.assertIn("def build_trace", embedded)
        self.assertNotIn('waveformFreeFormAnchorTable', embedded)
        self.assertIn('freeFormEdited', preview)
        self.assertIn('set_free_form_editor', preview)
        self.assertIn('def _position_to_data_unbounded', preview)
        self.assertIn('def _free_form_proxy_point', preview)
        self.assertIn('plot_rect.adjusted(-0.5, -0.5, 0.5, 0.5).contains(handle)', preview)
        self.assertIn('near-zero contact', preview)
        self.assertIn('def reset_selected_free_form_tangent', preview)
        self.assertIn('def fit_free_form_handles', preview)
        self.assertIn('Double-click inside the free-form component', embedded)
        self.assertIn('Vertical position', embedded)
        self.assertIn('def _selected_point_value_changed', embedded)
        self.assertNotIn('Add free-form component', embedded)
        self.assertIn('Reset selected tangent', embedded)
        self.assertIn('Fit handles', embedded)
        self.assertIn('waveformInlineImportScroll', embedded_imports)
        self.assertNotIn("class _ChannelTableModel", native_editor)
        self.assertNotIn("QTableView", native_editor)
        self.assertNotIn("Paste time/value", native_editor)
        self.assertNotIn("set_data_editor_visible", native_editor)
        self.assertIn("def format_time_span", formatting)
        self.assertIn("class WaveformPreviewCanvas", preview)
        self.assertNotIn("WaveformBuilderDialog", dialogs)
        self.assertNotIn("NativeSquareDialog", dialogs)
        self.assertNotIn("NativeSineDialog", dialogs)
        self.assertNotIn("NativeDCDialog", dialogs)
        self.assertNotIn("NativeActionPotentialDialog", dialogs)
        self.assertNotIn("NativeFreeFormDialog", dialogs)
        self.assertNotIn("recipe_from_legacy_generation", native_editor)
        self.assertNotIn('metadata.get("generation")', main_window)
        self.assertIn("Sample interval (ms)", embedded_imports)  # declarative runtime-timing import
        self.assertNotIn("Sample interval", embedded)
        self.assertIn("plot_theme_colors", theme_bridge)

    def test_waveform_resources_are_wired_through_project_manager_and_scene_metadata(self) -> None:
        manager = (PROJECT_ROOT / "fieldworkbench" / "project_manager.py").read_text(encoding="utf-8")
        adapter = (PROJECT_ROOT / "fieldworkbench" / "studio_adapter.py").read_text(encoding="utf-8")
        context = (PROJECT_ROOT / "fieldworkbench" / "project_context.py").read_text(encoding="utf-8")
        waveform_io = (PROJECT_ROOT / "fieldworkbench" / "waveform_io.py").read_text(encoding="utf-8")

        self.assertIn("WaveformEditorWindow", manager)
        self.assertIn("def open_waveforms", manager)
        self.assertIn('register_window("waveform_editor", window)', manager)
        self.assertIn("WORKBENCH_METADATA_VERSION = 26", adapter)
        self.assertIn('metadata["waveforms"]', adapter)
        self.assertIn("def list_waveforms", adapter)
        self.assertIn("def save_waveform_document", adapter)
        self.assertIn("def delete_waveforms", adapter)
        self.assertIn("def _waveform_resources", context)
        self.assertNotIn("waveform_document_to_payload", waveform_io)
        self.assertIn("waveform_signal_to_payload", waveform_io)
        self.assertIn("waveform_document_from_payload", waveform_io)


if __name__ == "__main__":
    unittest.main()
