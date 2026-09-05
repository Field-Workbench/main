"""Offline Plotly view for Qt.

Plotly's JavaScript is written into a temporary local HTML file.  This avoids
both a CDN dependency and QWebEngine's small setHtml/data-URL size limit.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import tempfile
import weakref
from pathlib import Path

import plotly.io as pio
from plotly.offline import get_plotlyjs
from PySide6.QtCore import QObject, QSettings, QTimer, QUrl, Signal, Slot, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QBoxLayout,
    QApplication,
    QFileDialog,
    QMessageBox,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .theme import current_theme, plot_theme_colors, theme_colors

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - exercised only by minimal Qt installs
    QWebEngineView = None

try:
    from PySide6.QtWebChannel import QWebChannel
except ImportError:  # pragma: no cover - exercised only by minimal Qt installs
    QWebChannel = None


_ACTIVE_PLOT_VIEWS: "weakref.WeakSet[PlotView]" = weakref.WeakSet()
_PLOT_VIEW_SHUTDOWN_CONNECTED = False


def _cleanup_plot_views_for_shutdown() -> None:
    """Retire every Plotly/WebEngine surface before QApplication teardown."""

    for view in list(_ACTIVE_PLOT_VIEWS):
        try:
            view.cleanup()
        except RuntimeError:
            # A parent can finish deleting a view during the same shutdown turn.
            pass


def _ensure_plot_view_shutdown_hook() -> None:
    global _PLOT_VIEW_SHUTDOWN_CONNECTED
    if _PLOT_VIEW_SHUTDOWN_CONNECTED:
        return
    application = QApplication.instance()
    if application is None:
        return
    application.aboutToQuit.connect(_cleanup_plot_views_for_shutdown)
    _PLOT_VIEW_SHUTDOWN_CONNECTED = True


class _PlotBridge(QObject):
    """Small JavaScript-to-Qt bridge for 3D picking and preview visibility."""

    selection_requested = Signal(str, bool)
    visibility_requested = Signal(str, bool)
    image_ready = Signal(str)
    image_failed = Signal(str)
    image_save_requested = Signal(int, int)
    view_state_ready = Signal(str, str)
    live_view_changed = Signal(str)

    @Slot(str, bool)
    def requestSelection(self, object_id: str, additive: bool) -> None:  # noqa: N802 - JS API
        self.selection_requested.emit(object_id, additive)

    @Slot(str, bool)
    def requestVisibility(self, object_id: str, visible: bool) -> None:  # noqa: N802 - JS API
        self.visibility_requested.emit(object_id, visible)

    @Slot(str)
    def deliverImage(self, data_url: str) -> None:  # noqa: N802 - JS API
        self.image_ready.emit(data_url)

    @Slot(str)
    def deliverImageError(self, detail: str) -> None:  # noqa: N802 - JS API
        self.image_failed.emit(detail)

    @Slot(int, int)
    def requestImageSave(self, width: int, height: int) -> None:  # noqa: N802 - JS API
        self.image_save_requested.emit(int(width), int(height))

    @Slot(str, str)
    def deliverViewState(self, request_id: str, payload: str) -> None:  # noqa: N802 - JS API
        self.view_state_ready.emit(str(request_id), str(payload))

    @Slot(str)
    def reportLiveView(self, payload: str) -> None:  # noqa: N802 - JS API
        self.live_view_changed.emit(str(payload))



class _FullscreenPlotHost(QWidget):
    """True-fullscreen host for a separately constructed PlotView.

    The working architecture remains the independently constructed PlotView in a
    second native window that belongs to the owning dialog's modal family.  Because
    the WebEngine surface is born in this host and never reparented, the host can now
    make the final transition to Qt's true fullscreen state without moving the live
    browser surface between native windows.
    """

    exit_requested = Signal()

    def __init__(self, owner=None):
        flags = Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        super().__init__(owner, flags)
        self._allow_close = False
        self.setWindowTitle("Field Workbench — Fullscreen viewer")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self._exit_shortcuts = []
        for key in ("F11", "Escape"):
            shortcut = QShortcut(QKeySequence(key), self)
            # QWebEngine can consume Escape/F11 before a parent window shortcut.
            # ApplicationShortcut keeps the exit keys reliable while this host lives.
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(self.exit_requested.emit)
            self._exit_shortcuts.append(shortcut)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        self.exit_requested.emit()


class PlotView(QStackedWidget):
    """A reusable Plotly surface with a readable fallback/error page."""

    MODEBAR_BUTTONS_TO_REMOVE = (
        "sendDataToCloud",
        # Field Workbench owns camera fitting.  Plotly's built-in house restores
        # its private initial camera, which is subtly different from our Home.
        "resetCameraDefault3d",
        "resetCameraLastSave3d",
    )

    load_finished = Signal(bool)
    selection_requested = Signal(str, bool)
    visibility_requested = Signal(str, bool)
    image_ready = Signal(str)
    image_failed = Signal(str)
    fullscreen_changed = Signal(bool)
    live_3d_view_changed = Signal(dict)
    live_2d_view_changed = Signal(dict)

    def __init__(
        self,
        parent=None,
        *,
        include_3d_return_axes: bool = False,
        synchronize_legend_visibility: bool = False,
        allow_fullscreen: bool = True,
        image_filename: str = "field-workbench-plot",
    ):
        super().__init__(parent)
        self._workbench_theme = current_theme()
        self._include_3d_return_axes = bool(include_3d_return_axes)
        self._synchronize_legend_visibility = bool(synchronize_legend_visibility)
        self._allow_fullscreen = bool(allow_fullscreen)
        self._image_filename = self._sanitized_image_filename(image_filename)
        self._pending_modebar_export_path: Path | None = None
        self._temporary_directory = Path(tempfile.mkdtemp(prefix="field-workbench-plot-"))
        self._ready = False
        self._pending_figure_json: str | None = None
        self._pending_figure_is_3d = False
        self._pending_figure_coalesce = False
        self._render_request_serial = 0
        self._pending_render_request_id = 0
        # The first real scene should use precisely the same fitted camera as Home.
        # Plotly's natural initial camera is close, but visibly not identical.
        self._home_on_next_figure = True
        # Transform edits should expand/recenter scene ranges without throwing
        # away the user's current viewing direction.  This is separate from
        # Home, which intentionally restores the canonical isometric camera.
        self._fit_bounds_on_next_figure = False
        # Structural scene changes must not reuse Plotly's existing WebGL trace
        # objects.  In-place ``Plotly.react`` is ideal for selection overlays,
        # but adding/removing a Mesh3d can leave renderer state attached to a
        # trace that no longer owns it.  Structural updates therefore render
        # into a fresh plot surface and swap it in only after Plotly succeeds.
        # The live scene is never purged before its replacement is ready.
        self._clean_rebuild_on_next_figure = True

        # Do not move a live QWebEngineView between top-level windows. Chromium's
        # native surface can render after such a reparent while mouse/keyboard input
        # remains attached to the old window (and some Linux graphics stacks can
        # crash outright). The viewer-window experiment therefore gets a second
        # PlotView created directly in a separate top-level host. The original
        # viewer stays put and is merely disabled until that window closes.
        self._fullscreen_host: _FullscreenPlotHost | None = None
        self._fullscreen_clone: PlotView | None = None
        self._last_figure_json: str | None = None
        self._last_figure_is_3d = False
        self._pending_view_state: dict | None = None
        self._view_request_serial = 0
        self._pending_view_callbacks: dict[str, object] = {}
        self._cleaned_up = False
        # Small Plotly.restyle/relayout overrides layered over the most recent
        # complete figure. Keeping this state separate from the large source JSON
        # lets selectors update a handful of style values without parsing,
        # serializing, or retransmitting their geometry. It is also replayed into
        # theme refreshes and the independent fullscreen renderer.
        self._incremental_trace_state: dict[int, dict[str, object]] = {}
        self._incremental_layout_state: dict[str, object] = {}
        self._incremental_update_serial = 0

        self._fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self._fullscreen_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._fullscreen_shortcut.setEnabled(self._allow_fullscreen)
        self._fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        self._fullscreen_escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self._fullscreen_escape_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._fullscreen_escape_shortcut.setEnabled(self._allow_fullscreen)
        self._fullscreen_escape_shortcut.activated.connect(self.exit_fullscreen)

        self.message = QTextBrowser()
        self.message.setOpenExternalLinks(True)
        self.message.setHtml(
            "<div style='font:16px sans-serif;padding:32px'>"
            "Preparing the viewer…</div>"
        )
        self._style_message_surface()
        self.addWidget(self.message)

        self.web = None
        self._channel = None
        self._bridge = None
        _ACTIVE_PLOT_VIEWS.add(self)
        _ensure_plot_view_shutdown_hook()
        if QWebEngineView is not None:
            self.web = QWebEngineView()
            self.web.setContextMenuPolicy(self.contextMenuPolicy())
            if QWebChannel is not None:
                self._bridge = _PlotBridge(self)
                self._bridge.selection_requested.connect(self.selection_requested.emit)
                self._bridge.visibility_requested.connect(self.visibility_requested.emit)
                self._bridge.image_ready.connect(self._bridge_image_ready)
                self._bridge.image_failed.connect(self._bridge_image_failed)
                self._bridge.image_save_requested.connect(self._modebar_image_save_requested)
                self._bridge.view_state_ready.connect(self._bridge_view_state_ready)
                self._bridge.live_view_changed.connect(self._bridge_live_view_changed)
                self._channel = QWebChannel(self.web.page())
                self._channel.registerObject("fieldWorkbenchBridge", self._bridge)
                self.web.page().setWebChannel(self._channel)
            self.web.loadFinished.connect(self._page_loaded)
            self.addWidget(self.web)
            self._load_persistent_page()

    @staticmethod
    def _sanitized_image_filename(value: str) -> str:
        """Return a portable basename for a Plotly PNG suggestion."""
        name = str(value or "field-workbench-plot").strip()
        if name.lower().endswith(".png"):
            name = name[:-4]
        name = re.sub(r"[^A-Za-z0-9._ -]+", "-", name)
        name = re.sub(r"\s+", " ", name).strip(" .-_")
        return name or "field-workbench-plot"

    def _browser_theme_values(self, theme: str | None = None) -> dict[str, str]:
        """Return CSS values for the browser chrome around a Plotly surface."""

        selected = self._workbench_theme if theme is None else str(theme)
        plot = plot_theme_colors(selected)
        chrome = theme_colors(selected)
        if selected == "dark":
            return {
                "figure": plot["figure"],
                "text": chrome["text"],
                "muted": chrome["muted"],
                "panel": "rgba(42,48,55,0.96)",
                "controls": chrome["panel_alternate"],
                "border": chrome["border"],
                "track": chrome["disabled_border"],
                "sweep": chrome["focus"],
                "disabled_bg": chrome["disabled_bg"],
                "disabled_border": chrome["disabled_border"],
                "disabled_text": chrome["disabled"],
                "shadow": "rgba(0,0,0,0.32)",
            }
        return {
            "figure": plot["figure"],
            "text": "#334155",
            "muted": "#475569",
            "panel": "rgba(255,255,255,0.94)",
            "controls": "#ffffff",
            "border": "#e2e8f0",
            "track": "#e2e8f0",
            "sweep": "#64748b",
            "disabled_bg": "#e5e7eb",
            "disabled_border": "#d1d5db",
            "disabled_text": "#9ca3af",
            "shadow": "rgba(15,23,42,0.10)",
        }

    def _style_message_surface(self) -> None:
        colours = plot_theme_colors(self._workbench_theme)
        text_colour = colours["text"] if self._workbench_theme == "dark" else "#334155"
        self.message.setStyleSheet(
            "QTextBrowser {"
            f"background-color: {colours['figure']}; color: {text_colour};"
            "border: none;"
            "}"
        )

    def _apply_browser_chrome_theme(self) -> None:
        if self.web is None or not self._ready:
            return
        values = self._browser_theme_values()
        encoded = json.dumps(values, separators=(",", ":"))
        script = f"""
(() => {{
  const values = {encoded};
  const root = document.documentElement;
  for (const [name, value] of Object.entries(values)) {{
    root.style.setProperty('--fwb-' + name.replaceAll('_', '-'), value);
  }}
}})();
"""
        try:
            self.web.page().runJavaScript(script)
        except RuntimeError:
            pass

    def apply_workbench_theme(self, theme: str) -> None:
        """Refresh browser-backed plot surfaces after an application theme change."""

        selected = str(theme).strip().lower()
        if selected not in {"light", "dark"}:
            return
        changed = selected != self._workbench_theme
        self._workbench_theme = selected
        self._style_message_surface()
        self._apply_browser_chrome_theme()
        if (
            not changed
            or self._last_figure_json is None
            or self.web is None
            or self.currentWidget() is not self.web
        ):
            return
        try:
            figure = json.loads(self._last_figure_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(figure, dict):
            self.set_figure(
                figure,
                coalesce=True,
                preserve_incremental=True,
            )

    def _themed_figure_json(self, source_json: str) -> str:
        """Apply WaveBuilder's deliberately light plot surface in Dark mode."""

        if self._workbench_theme != "dark":
            return source_json
        try:
            figure = json.loads(source_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return source_json
        if not isinstance(figure, dict):
            return source_json
        colours = plot_theme_colors("dark")
        layout = figure.setdefault("layout", {})
        if not isinstance(layout, dict):
            return source_json
        layout["paper_bgcolor"] = colours["figure"]
        layout["plot_bgcolor"] = colours["axes"]
        font = layout.setdefault("font", {})
        if isinstance(font, dict):
            font["color"] = colours["text"]

        # Set only surface/axis defaults. Explicit trace, annotation, and hidden
        # axis colours remain untouched, preserving engineering overlays.
        axis_keys = {
            key
            for key in layout
            if re.fullmatch(r"[xy]axis\d*", str(key))
        }
        axis_keys.update(("xaxis", "yaxis"))
        for key in axis_keys:
            axis = layout.setdefault(key, {})
            if not isinstance(axis, dict):
                continue
            axis.setdefault("gridcolor", colours["major_grid"])
            axis.setdefault("zerolinecolor", colours["reference"])
            axis.setdefault("linecolor", colours["reference"])
            axis.setdefault("tickcolor", colours["reference"])

        for key, scene in tuple(layout.items()):
            if not re.fullmatch(r"scene\d*", str(key)) or not isinstance(scene, dict):
                continue
            scene["bgcolor"] = colours["axes"]
            for axis_name in ("xaxis", "yaxis", "zaxis"):
                axis = scene.setdefault(axis_name, {})
                if not isinstance(axis, dict):
                    continue
                axis.setdefault("gridcolor", colours["major_grid"])
                axis.setdefault("zerolinecolor", colours["reference"])
                axis.setdefault("linecolor", colours["reference"])
                axis.setdefault("tickcolor", colours["reference"])
        return json.dumps(figure, separators=(",", ":"), ensure_ascii=False)

    def _modebar_image_save_requested(self, width: int, height: int) -> None:
        """Open a native Save As dialog for Plotly's camera-modebar action."""
        settings = QSettings()
        last_directory = str(settings.value("plotView/lastPngDirectory", "") or "").strip()
        suggested_name = f"{self._image_filename}.png"
        initial_path = (
            str(Path(last_directory) / suggested_name)
            if last_directory and Path(last_directory).is_dir()
            else suggested_name
        )
        filename, _ = QFileDialog.getSaveFileName(
            self.window(),
            "Save plot as PNG",
            initial_path,
            "PNG image (*.png)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".png"):
            filename += ".png"
        destination = Path(filename)
        settings.setValue("plotView/lastPngDirectory", str(destination.parent))
        self._pending_modebar_export_path = destination
        render_width = max(1, int(width) if int(width) > 0 else self.width())
        render_height = max(1, int(height) if int(height) > 0 else self.height())
        if not self.request_png(width=render_width, height=render_height, scale=2.0):
            self._pending_modebar_export_path = None
            QMessageBox.warning(
                self.window(),
                "Save plot as PNG",
                "The plot is not ready to render yet.",
            )

    def _bridge_image_ready(self, data_url: str) -> None:
        """Save mode-bar PNGs locally, while preserving the public image signal API."""
        path = self._pending_modebar_export_path
        if path is None:
            self.image_ready.emit(data_url)
            return
        self._pending_modebar_export_path = None
        try:
            marker = "base64,"
            if marker not in data_url:
                raise ValueError("The renderer returned an unsupported image format.")
            encoded = data_url.split(marker, 1)[1]
            path.write_bytes(base64.b64decode(encoded, validate=True))
        except Exception as error:  # noqa: BLE001 - filesystem/render boundary
            QMessageBox.warning(
                self.window(),
                "Save plot as PNG",
                f"Could not save the plot image:\n{error}",
            )

    def _bridge_image_failed(self, detail: str) -> None:
        """Route mode-bar failures locally and explicit rendering failures outward."""
        if self._pending_modebar_export_path is None:
            self.image_failed.emit(detail)
            return
        self._pending_modebar_export_path = None
        QMessageBox.warning(
            self.window(),
            "Save plot as PNG",
            f"Could not render the plot image:\n{detail}",
        )

    def _load_persistent_page(self) -> None:
        """Load Plotly once; later scene changes use in-place Plotly.react calls."""
        html_path = self._temporary_directory / "plot-view.html"
        values = self._browser_theme_values()
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    :root {{
      --fwb-figure:{values['figure']}; --fwb-text:{values['text']};
      --fwb-muted:{values['muted']}; --fwb-panel:{values['panel']};
      --fwb-controls:{values['controls']}; --fwb-border:{values['border']};
      --fwb-track:{values['track']}; --fwb-sweep:{values['sweep']};
      --fwb-disabled-bg:{values['disabled_bg']};
      --fwb-disabled-border:{values['disabled_border']};
      --fwb-disabled-text:{values['disabled_text']}; --fwb-shadow:{values['shadow']};
    }}
    html, body {{
      width:100%; height:100%; margin:0; overflow:hidden;
      background:var(--fwb-figure); color:var(--fwb-text);
    }}
    body {{ position:relative; }}
    .plot-surface {{
      position:absolute; inset:0; width:100%; height:100%;
      background:var(--fwb-figure);
    }}
    #loading {{
      position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
      font-family:Arial, Helvetica, sans-serif; color:var(--fwb-muted);
      pointer-events:none; z-index:7;
    }}
    #loading-panel {{
      width:min(360px, calc(100% - 48px)); box-sizing:border-box; padding:18px 22px;
      border:1px solid var(--fwb-border); border-radius:10px; background:var(--fwb-panel);
      box-shadow:0 8px 28px var(--fwb-shadow);
    }}
    #loading-title {{ font-size:16px; font-weight:600; color:var(--fwb-text); }}
    #loading-track {{
      position:relative; height:5px; margin:14px 0 0; overflow:hidden; border-radius:999px;
      background:var(--fwb-track);
    }}
    #loading-sweep {{
      position:absolute; top:0; bottom:0; left:-36%; width:36%; border-radius:999px;
      background:var(--fwb-sweep); animation:field-workbench-loading-sweep 1.15s ease-in-out infinite;
    }}
    @keyframes field-workbench-loading-sweep {{
      0% {{ transform:translateX(0); }}
      100% {{ transform:translateX(380%); }}
    }}
    #waveform-controls {{
      position:absolute; left:0; right:0; bottom:0; height:70px; display:none;
      align-items:center; justify-content:center; gap:18px; box-sizing:border-box;
      background:var(--fwb-controls); color:var(--fwb-text);
      border-top:1px solid var(--fwb-border); z-index:8;
      font-family:Arial, Helvetica, sans-serif;
    }}
    .waveform-control {{
      min-width:132px; height:42px; padding:0 22px; border-radius:7px;
      border:1px solid transparent; font-size:17px; font-weight:600; cursor:pointer;
      transition:background-color 100ms ease, border-color 100ms ease, color 100ms ease;
    }}
    #waveform-play:not(:disabled) {{ background:#2e7d32; border-color:#246528; color:white; }}
    #waveform-stop:not(:disabled) {{ background:#c62828; border-color:#a82121; color:white; }}
    .waveform-control:disabled {{
      background:var(--fwb-disabled-bg); border-color:var(--fwb-disabled-border);
      color:var(--fwb-disabled-text); cursor:default;
    }}
    #waveform-playback-info {{
      min-width:190px; color:var(--fwb-muted); font-size:14px; font-weight:600; text-align:left;
    }}
  </style>
  <script>{get_plotlyjs()}</script>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head>
<body>
  <div id="plot" class="plot-surface"></div>
  <div id="loading">
    <div id="loading-panel">
      <div id="loading-title">Preparing the viewer…</div>
      <div id="loading-track"><div id="loading-sweep"></div></div>
    </div>
  </div>
  <div id="waveform-controls" aria-label="Waveform playback controls">
    <button id="waveform-play" class="waveform-control" type="button">▶ Play</button>
    <button id="waveform-stop" class="waveform-control" type="button" disabled>■ Stop</button>
    <span id="waveform-playback-info"></span>
  </div>
  <script>
    window.fieldWorkbenchBridge = null;
    window.fieldWorkbenchDisposed = false;
    window.fieldWorkbenchDispose = function() {{
      if (window.fieldWorkbenchDisposed) return;
      window.fieldWorkbenchDisposed = true;
      window.fieldWorkbenchLatestCoalescedRenderId = Number.MAX_SAFE_INTEGER;
      for (const plot of document.querySelectorAll('.plot-surface')) {{
        if (!plot || !plot.data || !window.Plotly) continue;
        try {{ Plotly.purge(plot); }} catch (error) {{ console.warn(error); }}
      }}
      window.fieldWorkbenchRenderQueue = Promise.resolve();
    }};
    // Plotly render calls are asynchronous.  Serialize updates so a large STL
    // import cannot finish out of order with startup or selection rendering.
    window.fieldWorkbenchRenderQueue = Promise.resolve();
    window.fieldWorkbenchLoadingToken = null;
    window.fieldWorkbenchBeginLoading = function(token) {{
      const loading = document.getElementById('loading');
      const title = document.getElementById('loading-title');
      const track = document.getElementById('loading-track');
      if (!loading) return;
      window.fieldWorkbenchLoadingToken = token;
      if (title) title.textContent = 'Preparing the viewer…';
      if (track) track.style.display = 'block';
      loading.style.display = 'flex';
    }};

    window.fieldWorkbenchEndLoading = function(token) {{
      if (window.fieldWorkbenchLoadingToken !== token) return;
      window.fieldWorkbenchLoadingToken = null;
      const loading = document.getElementById('loading');
      if (loading) loading.style.display = 'none';
    }};

    window.fieldWorkbenchFailLoading = function(token, detail) {{
      if (window.fieldWorkbenchLoadingToken !== token) return;
      const title = document.getElementById('loading-title');
      const track = document.getElementById('loading-track');
      if (title) title.textContent = 'Unable to update plot: ' + String(detail || 'Unknown renderer error');
      if (track) track.style.display = 'none';
    }};

    if (typeof QWebChannel !== 'undefined' && window.qt && qt.webChannelTransport) {{
      new QWebChannel(qt.webChannelTransport, channel => {{
        window.fieldWorkbenchBridge = channel.objects.fieldWorkbenchBridge;
      }});
    }}

    window.fieldWorkbenchSetWaveformRunning = function(running) {{
      const play = document.getElementById('waveform-play');
      const stop = document.getElementById('waveform-stop');
      if (!play || !stop) return;
      play.disabled = Boolean(running);
      stop.disabled = !Boolean(running);
    }};

    window.fieldWorkbenchConfigureWaveformControls = function(plot, figure) {{
      const controls = document.getElementById('waveform-controls');
      const play = document.getElementById('waveform-play');
      const stop = document.getElementById('waveform-stop');
      const info = document.getElementById('waveform-playback-info');
      const loading = document.getElementById('loading');
      if (!controls || !play || !stop) return Promise.resolve(plot);

      const meta = figure && figure.layout && figure.layout.meta;
      const settings = meta && meta.field_workbench_waveform;
      if (!settings || !plot) {{
        controls.style.display = 'none';
        if (plot) plot.style.height = '100%';
        if (loading) loading.style.bottom = '0';
        if (info) info.textContent = '';
        return Promise.resolve(plot);
      }}

      controls.style.display = 'flex';
      plot.style.height = 'calc(100% - 70px)';
      if (loading) loading.style.bottom = '70px';
      window.fieldWorkbenchSetWaveformRunning(false);

      const playbackFps = Math.max(0.1, Number(settings.playback_fps) || 60);
      const frameDuration = 1000.0 / playbackFps;
      const realTimeMultiplier = Math.max(0, Number(settings.real_time_multiplier) || 0);
      const updateMode = String(settings.update_mode || 'animate');
      const sourceFrames = Array.isArray(figure && figure.frames) ? figure.frames : [];
      const totalFrames = Math.max(0, sourceFrames.length || Number(settings.frame_count) || 0);
      const sliderSyncInterval = 100.0;
      const fpsText = playbackFps >= 10 ? playbackFps.toFixed(0) : playbackFps.toFixed(2);
      const speedText = realTimeMultiplier >= 100
        ? realTimeMultiplier.toFixed(0)
        : (realTimeMultiplier >= 10 ? realTimeMultiplier.toFixed(1) : realTimeMultiplier.toFixed(3));
      const baseInfo = `Target ${{fpsText}} FPS • ${{speedText}}× real time`;
      if (info) info.textContent = baseInfo;

      plot.__fieldWorkbenchWaveformToken = (plot.__fieldWorkbenchWaveformToken || 0) + 1;
      plot.__fieldWorkbenchWaveformUpdateMode = updateMode;
      plot.__fieldWorkbenchWaveformSourceFrames = sourceFrames;

      const setInfo = measuredFps => {{
        if (!info) return;
        if (Number.isFinite(measuredFps) && measuredFps > 0) {{
          const measuredText = measuredFps >= 10 ? measuredFps.toFixed(1) : measuredFps.toFixed(2);
          info.textContent = `${{baseInfo}} • rendering ${{measuredText}} FPS`;
        }} else {{
          info.textContent = baseInfo;
        }}
      }};
      plot.__fieldWorkbenchWaveformSetInfo = setInfo;

      const syncSlider = frameIndex => {{
        const slider = plot.layout && plot.layout.sliders && plot.layout.sliders[0];
        if (!slider || Number(slider.active) === frameIndex) return Promise.resolve();
        plot.__fieldWorkbenchWaveformSettingSlider = true;
        return Promise.resolve(
          Plotly.relayout(plot, {{'sliders[0].active': frameIndex}})
        ).finally(() => {{ plot.__fieldWorkbenchWaveformSettingSlider = false; }});
      }};

      const restyleFrame = (frameIndex, updateSlider = true) => {{
        if (updateMode !== 'restyle' || !sourceFrames.length) return Promise.resolve();
        const boundedIndex = Math.max(0, Math.min(sourceFrames.length - 1, Number(frameIndex) || 0));
        const frame = sourceFrames[boundedIndex];
        if (!frame) return Promise.resolve();
        const frameData = Array.isArray(frame.data) ? frame.data : [];
        const traceIndices = Array.isArray(frame.traces) ? frame.traces : [];
        let chain = Promise.resolve();
        for (let index = 0; index < frameData.length; index += 1) {{
          const traceIndex = traceIndices[index];
          if (!Number.isInteger(traceIndex)) continue;
          const update = frameData[index] || {{}};
          const payload = {{}};
          for (const [key, value] of Object.entries(update)) payload[key] = [value];
          chain = chain.then(() => Plotly.restyle(plot, payload, [traceIndex]));
        }}
        return chain.then(() => {{
          plot.__fieldWorkbenchWaveformFrameIndex = boundedIndex;
          return updateSlider ? syncSlider(boundedIndex) : undefined;
        }});
      }};
      plot.__fieldWorkbenchWaveformRestyleFrame = restyleFrame;

      const animateFrame = (frameIndex, updateSlider = true) => {{
        if (!sourceFrames.length) return Promise.resolve();
        const boundedIndex = Math.max(0, Math.min(sourceFrames.length - 1, Number(frameIndex) || 0));
        const frame = sourceFrames[boundedIndex];
        if (!frame) return Promise.resolve();
        return Promise.resolve(
          Plotly.animate(
            plot,
            [frame.name],
            {{
              frame: {{duration: 0, redraw: true}},
              transition: {{duration: 0}},
              mode: 'immediate'
            }}
          )
        ).then(() => {{
          plot.__fieldWorkbenchWaveformFrameIndex = boundedIndex;
          return updateSlider ? syncSlider(boundedIndex) : undefined;
        }});
      }};

      const applyFrame = (frameIndex, updateSlider = true) => (
        updateMode === 'restyle'
          ? restyleFrame(frameIndex, updateSlider)
          : animateFrame(frameIndex, updateSlider)
      );
      plot.__fieldWorkbenchWaveformApplyFrame = applyFrame;

      if (!plot.__fieldWorkbenchWaveformSliderBound) {{
        plot.__fieldWorkbenchWaveformSliderBound = true;
        plot.on('plotly_sliderchange', eventData => {{
          if (plot.__fieldWorkbenchWaveformSettingSlider) return;
          const stepIndex = eventData && eventData.step && Array.isArray(eventData.step.args)
            ? Number(eventData.step.args[0])
            : NaN;
          const active = Number.isFinite(stepIndex)
            ? stepIndex
            : (eventData && eventData.slider ? Number(eventData.slider.active) : 0);
          ++plot.__fieldWorkbenchWaveformToken;
          window.fieldWorkbenchSetWaveformRunning(false);
          const refreshInfo = plot.__fieldWorkbenchWaveformSetInfo;
          if (refreshInfo) refreshInfo(NaN);
          if (plot.__fieldWorkbenchWaveformUpdateMode === 'restyle') {{
            const apply = plot.__fieldWorkbenchWaveformApplyFrame;
            if (apply) Promise.resolve(apply(active, false)).catch(error => console.warn(error));
          }} else {{
            plot.__fieldWorkbenchWaveformFrameIndex = active;
          }}
        }});
      }}

      const currentFrameIndex = () => {{
        let frameIndex = Number(plot.__fieldWorkbenchWaveformFrameIndex);
        if (!Number.isInteger(frameIndex)) {{
          frameIndex = plot.layout && plot.layout.sliders && plot.layout.sliders[0]
            ? Number(plot.layout.sliders[0].active) || 0
            : 0;
        }}
        return Math.max(0, Math.min(Math.max(0, totalFrames - 1), frameIndex));
      }};

      play.onclick = () => {{
        if (play.disabled || totalFrames < 2 || !sourceFrames.length) return;
        const token = ++plot.__fieldWorkbenchWaveformToken;
        window.fieldWorkbenchSetWaveformRunning(true);
        setInfo(NaN);

        let startIndex = currentFrameIndex();
        const lastIndex = totalFrames - 1;
        const resetPromise = startIndex >= lastIndex
          ? Promise.resolve(applyFrame(0, true)).then(() => {{ startIndex = 0; }})
          : Promise.resolve();

        resetPromise.then(() => {{
          if (plot.__fieldWorkbenchWaveformToken !== token) return;
          const wallStart = performance.now();
          let lastRenderedIndex = startIndex;
          let rendering = false;
          let renderedTransitions = 0;
          let lastInfoUpdate = wallStart;
          let lastSliderSync = wallStart;
          const playbackDuration = (lastIndex - startIndex) * frameDuration;

          const finish = () => {{
            if (plot.__fieldWorkbenchWaveformToken !== token) return;
            const elapsed = Math.max(1, performance.now() - wallStart);
            const measuredFps = renderedTransitions * 1000.0 / elapsed;
            window.fieldWorkbenchSetWaveformRunning(false);
            setInfo(measuredFps);
          }};

          const tick = now => {{
            if (plot.__fieldWorkbenchWaveformToken !== token) return;
            const elapsed = Math.max(0, now - wallStart);
            const desiredIndex = Math.min(
              lastIndex,
              startIndex + Math.floor(elapsed / frameDuration + 1e-9)
            );
            const reachedWallEnd = elapsed >= playbackDuration;

            if (!rendering && desiredIndex > lastRenderedIndex) {{
              const requestedIndex = reachedWallEnd ? lastIndex : desiredIndex;
              const updateSlider = reachedWallEnd || (now - lastSliderSync >= sliderSyncInterval);
              rendering = true;
              Promise.resolve(applyFrame(requestedIndex, updateSlider)).then(() => {{
                if (plot.__fieldWorkbenchWaveformToken !== token) return;
                rendering = false;
                lastRenderedIndex = requestedIndex;
                renderedTransitions += 1;
                const completedAt = performance.now();
                if (updateSlider) lastSliderSync = completedAt;
                if (completedAt - lastInfoUpdate >= 250.0) {{
                  setInfo(renderedTransitions * 1000.0 / Math.max(1, completedAt - wallStart));
                  lastInfoUpdate = completedAt;
                }}
                if (reachedWallEnd && requestedIndex >= lastIndex) {{
                  finish();
                }}
              }}).catch(error => {{
                console.warn(error);
                if (plot.__fieldWorkbenchWaveformToken === token) {{
                  window.fieldWorkbenchSetWaveformRunning(false);
                  setInfo(NaN);
                }}
              }});
            }} else if (reachedWallEnd && !rendering && lastRenderedIndex >= lastIndex) {{
              finish();
              return;
            }}

            if (plot.__fieldWorkbenchWaveformToken === token) requestAnimationFrame(tick);
          }};
          requestAnimationFrame(tick);
        }}).catch(error => {{
          console.warn(error);
          if (plot.__fieldWorkbenchWaveformToken === token) {{
            window.fieldWorkbenchSetWaveformRunning(false);
            setInfo(NaN);
          }}
        }});
      }};

      stop.onclick = () => {{
        if (stop.disabled) return;
        ++plot.__fieldWorkbenchWaveformToken;
        window.fieldWorkbenchSetWaveformRunning(false);
        setInfo(NaN);
        const frameIndex = currentFrameIndex();
        Promise.resolve(syncSlider(frameIndex)).catch(error => console.warn(error));
        if (updateMode === 'restyle') return;
        Promise.resolve(Plotly.animate(
          plot,
          [null],
          {{
            frame: {{duration: 0, redraw: false}},
            transition: {{duration: 0}},
            mode: 'immediate'
          }}
        )).catch(error => console.warn(error));
      }};

      // The surface was initially rendered at full height. Resize once after the
      // playback bar claims its space so Plotly immediately refits the axes.
      try {{ Plotly.Plots.resize(plot); }} catch (error) {{ console.warn(error); }}
      return Promise.resolve(plot);
    }};

    window.fieldWorkbenchRenderExtentTrace = function(data) {{
      // Keep every 3-D plot backed by an actual finite WebGL volume.  Explicit
      // Plotly axis ranges are not enough on all drivers: a stack of perfectly
      // coincident planar scatter3d traces can still make Plotly normalize the
      // scene as a zero-thickness plane, especially when that plane is away from
      // the origin.  A fully transparent helper proved too easy for Plotly/WebGL
      // to optimize out, so retain one *practically invisible* eight-corner
      // scaffold with tiny but non-zero marker opacity.  It is render-only and
      // never enters the Workbench scene/model/physics.
      //
      // The scaffold follows the current geometry rather than anchoring to the
      // world origin.  That gives the same anti-degeneracy benefit without
      // forcing an assembly at X=1000 mm to zoom out just to include (0,0,0).
      const source = (data || []).filter(trace => !(
        trace && trace.meta && trace.meta.field_workbench_render_extent
      ));
      const bounds = {{x: [Infinity, -Infinity], y: [Infinity, -Infinity], z: [Infinity, -Infinity]}};
      let has3dCoordinates = false;
      for (const trace of source) {{
        if (!trace || trace.visible === false || trace.visible === 'legendonly') continue;
        const type = String(trace.type || '').toLowerCase();
        if (!['scatter3d', 'mesh3d', 'surface', 'cone', 'streamtube', 'volume', 'isosurface'].includes(type)) continue;
        let traceHasFinite3dPoint = false;
        for (const axis of ['x', 'y', 'z']) {{
          const values = trace[axis];
          if (!values || (!Array.isArray(values) && !ArrayBuffer.isView(values))) continue;
          for (const raw of values) {{
            const value = Number(raw);
            if (!Number.isFinite(value)) continue;
            bounds[axis][0] = Math.min(bounds[axis][0], value);
            bounds[axis][1] = Math.max(bounds[axis][1], value);
            traceHasFinite3dPoint = true;
          }}
        }}
        has3dCoordinates = has3dCoordinates || traceHasFinite3dPoint;
      }}
      if (!has3dCoordinates) return source;
      const spans = ['x', 'y', 'z'].map(axis => bounds[axis][1] - bounds[axis][0]);
      const finiteSpans = spans.filter(Number.isFinite);
      const maxSpan = Math.max(1, ...finiteSpans);
      const flatTolerance = Math.max(1e-9, maxSpan * 1e-9);
      const centres = {{}};
      const halfSpans = {{}};
      for (const axis of ['x', 'y', 'z']) {{
        let [low, high] = bounds[axis];
        if (!Number.isFinite(low) || !Number.isFinite(high)) {{
          low = -0.5 * maxSpan;
          high = 0.5 * maxSpan;
        }}
        centres[axis] = 0.5 * (low + high);
        const rawHalf = 0.5 * Math.max(0, high - low);
        // Preserve genuine scene extents, but never allow any renderer axis to
        // have zero/near-zero raw data span.  Flat axes get a cube-like depth
        // tied to the largest populated engineering span.
        halfSpans[axis] = rawHalf <= flatTolerance
          ? Math.max(0.5, 0.5 * maxSpan)
          : Math.max(0.5, rawHalf);
      }}
      const helper = {{x: [], y: [], z: []}};
      for (const sx of [-1, 1]) {{
        for (const sy of [-1, 1]) {{
          for (const sz of [-1, 1]) {{
            helper.x.push(centres.x + sx * halfSpans.x);
            helper.y.push(centres.y + sy * halfSpans.y);
            helper.z.push(centres.z + sz * halfSpans.z);
          }}
        }}
      }}
      return source.concat([{{
        type: 'scatter3d',
        mode: 'markers',
        x: helper.x,
        y: helper.y,
        z: helper.z,
        // Do not use opacity:0 here. Some Plotly/WebGL paths discard a fully
        // transparent trace before their internal 3-D normalization step. Eight
        // sub-pixel markers at 0.1% opacity are visually imperceptible but remain
        // real renderer geometry, just like adding bobbin thickness did in the
        // user's reproduction.
        marker: {{size: 0.1, opacity: 0.001, color: '#000000'}},
        hoverinfo: 'skip',
        hovertemplate: null,
        showlegend: false,
        name: '__Field Workbench render extent__',
        uid: 'field-workbench-render-extent',
        meta: {{field_workbench_render_extent: true}}
      }}]);
    }};

    window.fieldWorkbenchApplyPickingPolicy = function(plot) {{
      // Brain View can keep ordinary scene meshes visible as anatomical/coil
      // context while asking the cursor to interact only with LPBA40 shells.
      // Plotly's public hoverinfo='skip' suppresses labels/events, but GL3D still
      // draws those meshes into its shared depth-tested pick buffer.  Marked
      // Workbench traces therefore have only their private drawPick hook disabled;
      // their ordinary opaque/transparent render hooks remain untouched.
      const plotScene = plot && plot._fullLayout && plot._fullLayout.scene;
      const scene = plotScene && plotScene._scene;
      const glplot = scene && scene.glplot;
      const objects = glplot && glplot.objects;
      if (!Array.isArray(objects)) return plot;

      let changed = false;
      for (const object of objects) {{
        const trace = object && object._trace && object._trace.data;
        const meta = trace && trace.meta;
        const ignorePicking = Boolean(
          meta && meta.field_workbench_ignore_picking
        );
        if (ignorePicking) {{
          if (typeof object.drawPick === 'function' &&
              !object.__fieldWorkbenchOriginalDrawPick) {{
            object.__fieldWorkbenchOriginalDrawPick = object.drawPick;
          }}
          if (object.drawPick) {{
            object.drawPick = null;
            changed = true;
          }}
        }} else if (!object.drawPick &&
                   typeof object.__fieldWorkbenchOriginalDrawPick === 'function') {{
          object.drawPick = object.__fieldWorkbenchOriginalDrawPick;
          changed = true;
        }}
      }}

      // gl-plot3d caches an off-screen colour/depth pick buffer. update() marks
      // that buffer dirty so the next animation frame rebuilds it without the
      // ignored context meshes.  Rendering itself is unchanged.
      if (changed && typeof glplot.update === 'function') {{
        try {{ glplot.update(); }} catch (error) {{ console.warn(error); }}
      }}
      return plot;
    }};

    window.fieldWorkbenchFittedView = function(data, camera, hiddenAxis) {{
      const bounds = {{x: [Infinity, -Infinity], y: [Infinity, -Infinity], z: [Infinity, -Infinity]}};
      for (const trace of (data || [])) {{
        if (!trace || trace.visible === false || trace.visible === 'legendonly') continue;
        for (const axis of ['x', 'y', 'z']) {{
          const values = trace[axis];
          if (!values || (!Array.isArray(values) && !ArrayBuffer.isView(values))) continue;
          for (const raw of values) {{
            const value = Number(raw);
            if (!Number.isFinite(value)) continue;
            bounds[axis][0] = Math.min(bounds[axis][0], value);
            bounds[axis][1] = Math.max(bounds[axis][1], value);
          }}
        }}
      }}
      const spans = ['x', 'y', 'z'].map(axis => bounds[axis][1] - bounds[axis][0]);
      const finiteSpans = spans.filter(Number.isFinite);
      const maxSpan = Math.max(1, ...finiteSpans);
      const flatTolerance = Math.max(1e-9, maxSpan * 1e-9);
      const ranges = {{}};
      for (const axis of ['x', 'y', 'z']) {{
        let [low, high] = bounds[axis];
        if (!Number.isFinite(low) || !Number.isFinite(high)) {{
          [low, high] = [-0.5 * maxSpan, 0.5 * maxSpan];
        }}
        const centre = 0.5 * (low + high);
        const rawHalf = 0.5 * Math.max(0, high - low);
        // A genuinely planar scene has no meaningful visual thickness along
        // its normal.  Do not squeeze that axis into a thin sliver: give it the
        // same engineering span as the largest populated axis.  Besides being
        // easier to read, this avoids the distorted/twisted axis presentation
        // and scatter3d clipping that can occur for off-origin coplanar coils.
        const fittedHalf = rawHalf <= flatTolerance
          ? Math.max(0.5, 0.5 * maxSpan)
          : Math.max(0.5, rawHalf);
        const half = fittedHalf * 1.12;
        ranges[axis] = [centre - half, centre + half];
      }}
      return {{
        'scene.xaxis.range': ranges.x,
        'scene.yaxis.range': ranges.y,
        'scene.zaxis.range': ranges.z,
        'scene.xaxis.visible': hiddenAxis !== 'x',
        'scene.yaxis.visible': hiddenAxis !== 'y',
        'scene.zaxis.visible': hiddenAxis !== 'z',
        // Plotly natively defines aspectmode='data' as drawing the scene axes
        // in proportion to their explicit ranges.  That is exactly Workbench's
        // equal-mm requirement, and avoids the fragile manual aspectratio path
        // for scatter3d scenes.  The ranges above already pad flat axes to a
        // real cube-like depth before Plotly derives the aspect.
        'scene.aspectmode': 'data',
        'scene.camera': camera
      }};
    }};

    window.fieldWorkbenchFitScene = function(plot, camera, hiddenAxis) {{
      return Plotly.relayout(
        plot,
        window.fieldWorkbenchFittedView(plot.data || [], camera, hiddenAxis)
      );
    }};

    window.fieldWorkbenchHomeCamera = {{
      center: {{x: 0, y: 0, z: 0}},
      eye: {{x: 1.35, y: -1.35, z: 1.1}},
      up: {{x: 0, y: 0, z: 1}},
      projection: {{type: 'perspective'}}
    }};

    window.fieldWorkbenchHome = function(plot) {{
      return window.fieldWorkbenchFitScene(
        plot,
        window.fieldWorkbenchHomeCamera,
        null
      );
    }};

    window.fieldWorkbenchCurrentView = function(plot) {{
      if (!plot || !plot._fullLayout) return null;
      const clone = value => JSON.parse(JSON.stringify(value));
      const scene = plot._fullLayout.scene;
      if (scene) {{
        // Plotly writes interactive orbit/pan/zoom changes back to gd.layout.
        // Prefer that public layout camera; _fullLayout.scene.camera can lag behind
        // on some Plotly/WebEngine combinations.
        const layoutScene = plot && plot.layout && plot.layout.scene;
        const liveCamera = layoutScene && layoutScene.camera
          ? layoutScene.camera
          : scene.camera;
        return {{
          'scene.xaxis.range': clone(scene.xaxis.range),
          'scene.yaxis.range': clone(scene.yaxis.range),
          'scene.zaxis.range': clone(scene.zaxis.range),
          'scene.xaxis.visible': scene.xaxis.visible,
          'scene.yaxis.visible': scene.yaxis.visible,
          'scene.zaxis.visible': scene.zaxis.visible,
          'scene.aspectmode': scene.aspectmode,
          'scene.camera': clone(liveCamera)
        }};
      }}
      // 2D field maps use ordinary Cartesian axes. Capture the live ranges after
      // pan/zoom so the Camera section can author the equivalent centre + zoom.
      const updates = {{}};
      for (const axisName of ['xaxis', 'yaxis']) {{
        const fullAxis = plot._fullLayout[axisName];
        const layoutAxis = plot.layout && plot.layout[axisName];
        const liveRange = layoutAxis && Array.isArray(layoutAxis.range)
          ? layoutAxis.range
          : (fullAxis && fullAxis.range);
        if (Array.isArray(liveRange) && liveRange.length >= 2) {{
          updates[axisName + '.range'] = clone(liveRange);
        }}
      }}
      return Object.keys(updates).length ? updates : null;
    }};

    window.fieldWorkbenchPublishLiveCamera = function(plot) {{
      if (!plot || !window.fieldWorkbenchBridge || !window.fieldWorkbenchBridge.reportLiveView) return;
      const view = window.fieldWorkbenchCurrentView(plot);
      if (view && (view['scene.camera'] || view['xaxis.range'] || view['yaxis.range'])) {{
        window.fieldWorkbenchBridge.reportLiveView(JSON.stringify(view));
      }}
    }};

    window.fieldWorkbenchLayoutWithView = function(layout, view) {{
      if (!view) return layout || {{}};
      const clone = value => JSON.parse(JSON.stringify(value));
      const nextLayout = clone(layout || {{}});
      if (view['scene.camera']) {{
        const scene = nextLayout.scene = nextLayout.scene || {{}};
        for (const axis of ['x', 'y', 'z']) {{
          const axisLayout = scene[axis + 'axis'] = scene[axis + 'axis'] || {{}};
          axisLayout.range = clone(view['scene.' + axis + 'axis.range']);
          axisLayout.autorange = false;
          axisLayout.visible = view['scene.' + axis + 'axis.visible'];
        }}
        scene.aspectmode = 'data';
        // An upstream Magpylib layout can carry a stale manual aspectratio.
        // Remove it whenever Workbench restores a fitted 3D view so Plotly's
        // native range-proportional data mode is unambiguous.
        delete scene.aspectratio;
        scene.camera = clone(view['scene.camera']);
        return nextLayout;
      }}
      // Preserve a live 2D pan/zoom across Plotly.react in exactly the same way
      // that the 3D path preserves its camera.
      for (const axisName of ['xaxis', 'yaxis']) {{
        const range = view[axisName + '.range'];
        if (!Array.isArray(range) || range.length < 2) continue;
        const axisLayout = nextLayout[axisName] = nextLayout[axisName] || {{}};
        axisLayout.range = clone(range);
        axisLayout.autorange = false;
      }}
      return nextLayout;
    }};
  </script>
</body>
</html>"""
        html_path.write_text(html, encoding="utf-8")
        self.web.load(QUrl.fromLocalFile(str(html_path.resolve())))

    def _page_loaded(self, ok: bool) -> None:
        self._ready = bool(ok)
        self.load_finished.emit(bool(ok))
        if ok:
            self._apply_browser_chrome_theme()
            if self._pending_figure_json is not None:
                self._push_pending_figure()
            elif self._incremental_trace_state or self._incremental_layout_state:
                self._push_incremental_state()

    def _bridge_view_state_ready(self, request_id: str, payload: str) -> None:
        """Resolve one live-view request delivered through the WebChannel bridge."""
        callback = self._pending_view_callbacks.pop(str(request_id), None)
        if not callable(callback):
            return
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = None
        QTimer.singleShot(0, lambda: callback(value))

    def _bridge_live_view_changed(self, payload: str) -> None:
        """Publish an interactive 3D camera or 2D pan/zoom move."""
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(value, dict):
            return
        if isinstance(value.get("scene.camera"), dict):
            self.live_3d_view_changed.emit(value)
            return
        if isinstance(value.get("xaxis.range"), list) or isinstance(
            value.get("yaxis.range"), list
        ):
            self.live_2d_view_changed.emit(value)

    def request_3d_view(self, callback) -> None:
        """Return the live Plotly 3D view state.

        Prefer the existing WebChannel bridge rather than depending on the
        QWebEnginePage.runJavaScript result-callback overload.  The latter has
        behaved inconsistently across PySide/WebEngine builds.
        """
        if self.web is None or not self._ready:
            QTimer.singleShot(0, lambda: callback(None))
            return
        script_body = (
            "const plot=document.getElementById('plot'); "
            "return plot && window.fieldWorkbenchCurrentView ? "
            "window.fieldWorkbenchCurrentView(plot) : null;"
        )
        if self._bridge is not None:
            self._view_request_serial += 1
            request_id = f"view-{self._view_request_serial}"
            self._pending_view_callbacks[request_id] = callback
            request_json = json.dumps(request_id)
            script = (
                "(() => { "
                + script_body
                + " })()"
            )
            bridge_script = f"""
(() => {{
  let value = null;
  try {{
    value = {script};
  }} catch (error) {{
    console.warn(error);
  }}
  if (window.fieldWorkbenchBridge && window.fieldWorkbenchBridge.deliverViewState) {{
    window.fieldWorkbenchBridge.deliverViewState(
      {request_json},
      JSON.stringify(value)
    );
  }}
  return value;
}})();
"""
            def callback_fallback(value) -> None:
                pending = self._pending_view_callbacks.pop(request_id, None)
                if callable(pending):
                    QTimer.singleShot(0, lambda: pending(value))

            try:
                # Return the same value as a fallback for WebEngine builds where
                # the WebChannel delivery is delayed/unavailable.  Whichever path
                # completes first removes the pending callback; the other becomes
                # a harmless no-op.
                self.web.page().runJavaScript(bridge_script, callback_fallback)
                return
            except TypeError:
                # Some PySide releases do not expose the callback overload
                # consistently.  The WebChannel path remains the primary one.
                try:
                    self.web.page().runJavaScript(bridge_script)
                    QTimer.singleShot(
                        500,
                        lambda: (
                            callback_fallback(None)
                            if request_id in self._pending_view_callbacks
                            else None
                        ),
                    )
                    return
                except RuntimeError:
                    self._pending_view_callbacks.pop(request_id, None)
            except RuntimeError:
                self._pending_view_callbacks.pop(request_id, None)

        # Minimal-install fallback when QtWebChannel is unavailable.
        try:
            self.web.page().runJavaScript(
                "(() => { " + script_body + " })();",
                callback,
            )
        except (RuntimeError, TypeError):
            QTimer.singleShot(0, lambda: callback(None))

    def request_2d_view(self, callback) -> None:
        """Return the live Plotly 2D x/y axis ranges."""
        self.request_3d_view(callback)

    def show_message(self, heading: str, detail: str = "") -> None:
        self.message.setHtml(
            "<div style='font-family:sans-serif;padding:36px'>"
            f"<h2>{heading}</h2><p>{detail}</p></div>"
        )
        self.setCurrentWidget(self.message)

    def set_figure(
        self,
        figure: dict,
        *,
        coalesce: bool = False,
        preserve_incremental: bool = False,
    ) -> None:
        if self.web is None:
            self.show_message(
                "Interactive view unavailable",
                "Install the PySide6 Addons package to enable Qt WebEngine.",
            )
            return
        if not preserve_incremental:
            self._incremental_trace_state.clear()
            self._incremental_layout_state.clear()
        three_dimensional_types = {
            "scatter3d", "mesh3d", "surface", "cone", "streamtube", "volume", "isosurface"
        }
        self._pending_figure_is_3d = bool(
            figure.get("layout", {}).get("scene")
            or any(
                str(trace.get("type", "")).lower() in three_dimensional_types
                for trace in figure.get("data", [])
                if isinstance(trace, dict)
            )
        )
        self._render_request_serial += 1
        self._pending_render_request_id = self._render_request_serial
        self._pending_figure_coalesce = bool(coalesce)
        # Plotly.py removes trace uid values by default.  Field Workbench uses
        # those identities to stop Plotly.react from reusing per-face Mesh3d
        # colours when structural edits move traces to different indices.
        source_json = pio.to_json(
            figure,
            validate=False,
            pretty=False,
            remove_uids=False,
        )
        self._pending_figure_json = self._themed_figure_json(source_json)
        # Keep a replayable copy for the independent fullscreen renderer.  The
        # unthemed source allows an already-open viewer to move cleanly in both
        # directions when Preferences changes Light/Dark.
        self._last_figure_json = source_json
        self._last_figure_is_3d = bool(self._pending_figure_is_3d)
        self.setCurrentWidget(self.web)
        if self._ready:
            self._push_pending_figure()

    def _push_pending_figure(self) -> None:
        if self.web is None or not self._ready or self._pending_figure_json is None:
            return
        payload = json.dumps(self._pending_figure_json)
        is_three_dimensional = bool(self._pending_figure_is_3d)
        coalesce_render = bool(self._pending_figure_coalesce)
        render_request_id = int(self._pending_render_request_id)
        modebar_buttons_to_remove = list(self.MODEBAR_BUTTONS_TO_REMOVE)
        native_image_save = self._bridge is not None
        if native_image_save:
            # Replace Plotly's direct browser download with a Workbench-owned
            # native Save As dialog.  This prevents anonymous newplot.png files
            # and gives every embedded PlotView the same predictable behavior.
            modebar_buttons_to_remove.append("toImage")
        config = json.dumps(
            {
                "responsive": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": modebar_buttons_to_remove,
            }
        )
        apply_home = "true" if self._home_on_next_figure and is_three_dimensional else "false"
        apply_bounds_fit = (
            "true"
            if self._fit_bounds_on_next_figure and is_three_dimensional and not self._home_on_next_figure
            else "false"
        )
        include_return_axes = (
            "true"
            if is_three_dimensional and self._include_3d_return_axes
            else "false"
        )
        synchronize_legend_visibility = (
            "true" if self._synchronize_legend_visibility else "false"
        )
        clean_rebuild = "true" if self._clean_rebuild_on_next_figure else "false"
        native_image_save_js = "true" if native_image_save else "false"
        coalesce_render_js = "true" if coalesce_render else "false"
        self._home_on_next_figure = False
        self._fit_bounds_on_next_figure = False
        self._clean_rebuild_on_next_figure = False
        self._pending_figure_is_3d = False
        self._pending_figure_coalesce = False
        script = f"""
(() => {{
  const figure = JSON.parse({payload});
  const renderRequestId = {render_request_id};
  const coalesceRender = {coalesce_render_js};
  if (coalesceRender) {{
    // Interactive selector controls can change much faster than Plotly/WebEngine
    // can paint.  Mark this request as the newest one immediately; older queued
    // selector renders will then become cheap no-ops instead of being replayed.
    window.fieldWorkbenchLatestCoalescedRenderId = renderRequestId;
  }}
  const renderStillCurrent = () => (
    !window.fieldWorkbenchDisposed &&
    (!coalesceRender ||
     window.fieldWorkbenchLatestCoalescedRenderId === renderRequestId)
  );
  const cleanRebuild = {clean_rebuild};
  const isThreeDimensional = {'true' if is_three_dimensional else 'false'};
  const applyHome = {apply_home};
  const applyBoundsFit = {apply_bounds_fit};
  const includeReturnAxes = {include_return_axes};
  const synchronizeLegendVisibility = {synchronize_legend_visibility};
  const nativeImageSave = {native_image_save_js};
  const loading = document.getElementById('loading');
  window.fieldWorkbenchBeginLoading(renderRequestId);
  const config = {config};
  if (nativeImageSave) {{
    config.modeBarButtonsToAdd = config.modeBarButtonsToAdd || [];
    config.modeBarButtonsToAdd.push({{
      name: 'Save plot as PNG…',
      icon: (Plotly.Icons && Plotly.Icons['camera-retro']) || {{
        width: 1000,
        height: 1000,
        path: 'm518 386q0 8-5 13t-13 5q-37 0-63-27t-26-63q0-8 5-13t13-5 12 5 5 13q0 23 16 38t38 16q8 0 13 5t5 13z m125-73q0-59-42-101t-101-42-101 42-42 101 42 101 101 42 101-42 42-101z',
        transform: 'matrix(1 0 0 -1 0 850)'
      }},
      click: plot => {{
        const rect = plot.getBoundingClientRect();
        if (window.fieldWorkbenchBridge) {{
          window.fieldWorkbenchBridge.requestImageSave(
            Math.max(1, Math.round(rect.width || plot.clientWidth || 1)),
            Math.max(1, Math.round(rect.height || plot.clientHeight || 1))
          );
        }}
      }}
    }});
  }}

  if (includeReturnAxes) {{
    // Plotly's stock 3D reset buttons restore Plotly's private initial camera,
    // which does not match Field Workbench's fitted engineering view.  Add a
    // mode-bar Home control that uses the same Return axes operation as the
    // main scene viewer instead.
    config.modeBarButtonsToAdd = config.modeBarButtonsToAdd || [];
    config.modeBarButtonsToAdd.push(
      {{
        name: 'Return axes',
        icon: (Plotly.Icons && Plotly.Icons.home) || {{
          width: 928.6,
          height: 1000,
          path: 'm786 296v-267q0-15-11-26t-25-10h-214v214h-143v-214h-214q-15 0-25 10t-11 26v267q0 1 0 2t0 2l321 264 321-264q1-1 1-4z m124 39l-34-41q-5-5-12-6h-2q-7 0-12 3l-386 322-386-322q-7-4-13-4-7 2-12 7l-35 41q-4 5-3 13t6 12l401 334q18 15 42 15t43-15l136-114v109q0 8 5 13t13 5h107q8 0 13-5t5-13v-227l122-102q5-5 6-12t-4-13z',
          transform: 'matrix(1 0 0 -1 0 850)'
        }},
        click: plot => window.fieldWorkbenchHome(plot)
      }}
    );
  }}

  const replaceAnimationFrames = plot => {{
    if (!plot) return Promise.resolve(plot);
    const waveformMeta = figure && figure.layout && figure.layout.meta
      ? figure.layout.meta.field_workbench_waveform
      : null;
    const needsPlotlyFrames = !waveformMeta || String(waveformMeta.update_mode || 'animate') !== 'restyle';
    const existingFrames = (
      plot._transitionData && Array.isArray(plot._transitionData._frames)
        ? plot._transitionData._frames.length
        : 0
    );
    const clearExisting = existingFrames
      ? Plotly.deleteFrames(plot, Array.from({{length: existingFrames}}, (_, index) => index))
      : Promise.resolve();
    return Promise.resolve(clearExisting)
      .then(() => {{
        if (needsPlotlyFrames && Array.isArray(figure.frames) && figure.frames.length) {{
          return Plotly.addFrames(plot, figure.frames);
        }}
        return undefined;
      }})
      .then(() => plot);
  }};

  const renderFigure = () => {{
    if (!renderStillCurrent()) return null;
    let plot = document.getElementById('plot');
    if (!plot) throw new Error('The active plot surface is missing.');
    if (isThreeDimensional) {{
      figure.data = window.fieldWorkbenchRenderExtentTrace(figure.data || []);
    }}
    const currentView = plot.data ? window.fieldWorkbenchCurrentView(plot) : null;
    let finalView = currentView;
    if (applyHome) {{
      finalView = window.fieldWorkbenchFittedView(
        figure.data || [],
        window.fieldWorkbenchHomeCamera,
        null
      );
    }} else if (applyBoundsFit && currentView) {{
      // A moved object may leave the existing fixed Plotly axis ranges.  Re-fit
      // the complete scene while preserving the user's camera direction and any
      // hidden axis from a Front/Right/Top preset.
      const hiddenAxis = ['x', 'y', 'z'].find(
        axis => currentView['scene.' + axis + 'axis.visible'] === false
      ) || null;
      finalView = window.fieldWorkbenchFittedView(
        figure.data || [],
        currentView['scene.camera'],
        hiddenAxis
      );
    }}
    if (finalView) {{
      // Put the final camera and ranges into the incoming layout before Plotly
      // paints.  A post-render relayout exposes Plotly's default camera for one
      // frame, which looks like a brief jump to an almost-Home view.
      figure.layout = window.fieldWorkbenchLayoutWithView(
        figure.layout || {{}},
        finalView
      );
    }}

    if (!cleanRebuild) {{
      return Promise.resolve()
        .then(() => Plotly.react(
          plot,
          figure.data || [],
          figure.layout || {{}},
          config
        ))
        .then(() => plot);
    }}

    // Render structural changes into a separate surface.  The old scene stays
    // alive and visible until the complete replacement has succeeded.  This
    // avoids the Qt WebEngine failure mode where purge + immediate newPlot on
    // the same element leaves only stale HTML legend content and no 3D scene.
    const replacement = document.createElement('div');
    replacement.className = 'plot-surface';
    replacement.style.visibility = 'hidden';
    plot.parentNode.insertBefore(replacement, loading || null);

    return Promise.resolve()
      .then(() => Plotly.newPlot(
        replacement,
        figure.data || [],
        figure.layout || {{}},
        config
      ))
      .then(() => {{
        plot.removeAttribute('id');
        replacement.id = 'plot';
        replacement.style.visibility = 'visible';
        try {{ Plotly.purge(plot); }} catch (error) {{ console.warn(error); }}
        plot.remove();
        return replacement;
      }})
      .catch(error => {{
        try {{ Plotly.purge(replacement); }} catch (purgeError) {{ console.warn(purgeError); }}
        replacement.remove();
        throw error;
      }});
  }};

  window.fieldWorkbenchRenderQueue = (window.fieldWorkbenchRenderQueue || Promise.resolve())
    .catch(() => undefined)
    .then(renderFigure)
    .then(plot => {{
      return replaceAnimationFrames(plot);
    }})
    .then(plot => {{
      return window.fieldWorkbenchConfigureWaveformControls(plot, figure);
    }})
    .then(plot => {{
      return window.fieldWorkbenchApplyPickingPolicy(plot);
    }})
    .then(plot => {{
      if (!plot || !renderStillCurrent()) return plot;
      if (!plot.__fieldWorkbenchPickingBound) {{
        plot.__fieldWorkbenchPickingBound = true;
        plot.on('plotly_click', eventData => {{
          const point = eventData && eventData.points && eventData.points[0];
          const meta = point && ((point.data && point.data.meta) ||
                                 (point.fullData && point.fullData.meta));
          const objectId = meta && meta.field_workbench_object_id;
          if (objectId && window.fieldWorkbenchBridge) {{
            const mouseEvent = eventData.event || {{}};
            const additive = Boolean(mouseEvent.ctrlKey || mouseEvent.metaKey || mouseEvent.shiftKey);
            window.fieldWorkbenchBridge.requestSelection(String(objectId), additive);
          }}
        }});
      }}
      if (!plot.__fieldWorkbenchCameraSyncBound) {{
        plot.__fieldWorkbenchCameraSyncBound = true;
        let cameraPublishTimer = null;
        const cameraUpdate = eventData => {{
          const keys = Object.keys(eventData || {{}});
          const cameraMoved = keys.some(
            key => key === 'scene.camera' || key.startsWith('scene.camera.')
          );
          const mapMoved = keys.some(
            key => key === 'xaxis.range' || key.startsWith('xaxis.range[') ||
                   key === 'yaxis.range' || key.startsWith('yaxis.range[') ||
                   key === 'xaxis.autorange' || key === 'yaxis.autorange'
          );
          if (!cameraMoved && !mapMoved) return;
          if (cameraPublishTimer !== null) return;
          cameraPublishTimer = window.setTimeout(() => {{
            cameraPublishTimer = null;
            window.fieldWorkbenchPublishLiveCamera(plot);
          }}, 40);
        }};
        // relayouting keeps the numeric controls visibly following an orbit/pan/zoom;
        // relayout supplies the final camera on Plotly builds that omit relayouting.
        plot.on('plotly_relayouting', cameraUpdate);
        plot.on('plotly_relayout', cameraUpdate);
      }}
      if (!plot.__fieldWorkbenchLegendVisibilityBound) {{
        plot.__fieldWorkbenchLegendVisibilityBound = true;
        plot.on('plotly_legendclick', eventData => {{
          if (!synchronizeLegendVisibility) return undefined;
          const curveNumber = Number(eventData && eventData.curveNumber);
          const trace = Number.isInteger(curveNumber) && plot.data
            ? plot.data[curveNumber]
            : null;
          const meta = trace && trace.meta;
          const objectId = meta && meta.field_workbench_object_id;
          if (objectId && window.fieldWorkbenchBridge) {{
            const currentlyVisible = trace.visible !== 'legendonly' && trace.visible !== false;
            window.fieldWorkbenchBridge.requestVisibility(
              String(objectId),
              !currentlyVisible
            );
          }}
          // In synchronized previews the native checklist owns visibility. Do not
          // let Plotly also maintain an independent legend-only state.
          return false;
        }});
        plot.on('plotly_legenddoubleclick', () => (
          synchronizeLegendVisibility ? false : undefined
        ));
      }}
      if (includeReturnAxes) {{
        // Plotly appends custom mode-bar buttons after its built-ins. Move the
        // Workbench Home/Return axes control into the first group so it is the
        // leftmost tool in every viewer that exposes it.
        const homeButton = Array.from(plot.querySelectorAll('.modebar-btn')).find(
          button => button.getAttribute('data-title') === 'Return axes'
        );
        const firstGroup = plot.querySelector('.modebar-group');
        if (homeButton && firstGroup) {{
          firstGroup.insertBefore(homeButton, firstGroup.firstChild);
        }}
      }}
      window.fieldWorkbenchEndLoading(renderRequestId);
    }})
    .catch(error => {{
      window.fieldWorkbenchFailLoading(renderRequestId, 'Renderer error: ' + error);
    }});
}})();
"""
        self._pending_figure_json = None
        self.web.page().runJavaScript(script)
        if self._incremental_trace_state or self._incremental_layout_state:
            # Both operations append to the same browser-side promise queue, so
            # these lightweight overrides cannot overtake the complete render.
            self._push_incremental_state()
        if self._pending_view_state is not None:
            state = self._pending_view_state
            self._pending_view_state = None
            self._apply_view_state(state)

    def is_fullscreen_view(self) -> bool:
        """Return whether an independent fullscreen renderer is active."""
        return self._fullscreen_host is not None

    def toggle_fullscreen(self) -> None:
        """Toggle an F11-style fullscreen view of this PlotView only."""
        if not self._allow_fullscreen:
            return
        if self.is_fullscreen_view():
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def _capture_view_state(self, callback) -> None:
        """Return the current camera/zoom state without moving the WebEngine view."""
        if self.web is None or not self._ready:
            callback(None)
            return
        script = r"""
(() => {
  const plot = document.getElementById('plot');
  if (!plot || !plot._fullLayout) return null;
  const clone = value => JSON.parse(JSON.stringify(value));
  const scene = plot._fullLayout.scene;
  if (scene) {
    return {
      kind: '3d',
      updates: window.fieldWorkbenchCurrentView(plot)
    };
  }
  const updates = {};
  for (const axisName of ['xaxis', 'yaxis']) {
    const axis = plot._fullLayout[axisName];
    if (axis && axis.range) updates[axisName + '.range'] = clone(axis.range);
  }
  return {kind: '2d', updates: updates};
})()
"""
        self.web.page().runJavaScript(script, callback)

    def capture_view_state(self, callback) -> None:
        """Asynchronously return the current camera/zoom state to a caller."""
        self._capture_view_state(callback)

    def _apply_view_state(self, state) -> None:
        """Apply a captured camera/zoom state after this PlotView finishes rendering."""
        if self.web is None or not state or not isinstance(state, dict):
            return
        updates = state.get("updates")
        if not isinstance(updates, dict) or not updates:
            return
        payload = json.dumps(updates)
        script = f"""
(() => {{
  const apply = () => {{
    const plot = document.getElementById('plot');
    if (!plot || !plot.data) return;
    return Plotly.relayout(plot, {payload});
  }};
  window.fieldWorkbenchRenderQueue = (window.fieldWorkbenchRenderQueue || Promise.resolve())
    .catch(() => undefined)
    .then(apply);
}})();
"""
        if self._ready:
            self.web.page().runJavaScript(script)
        else:
            # Defer until _push_pending_figure has queued the initial plot.  Merely
            # reacting to loadFinished is too early because that signal fires before
            # the pending Plotly render is submitted.
            self._pending_view_state = dict(state)

    def apply_view_state(self, state) -> None:
        """Restore a camera/zoom state captured by :meth:`capture_view_state`."""
        self._apply_view_state(state)

    def enter_fullscreen(self) -> None:
        """Open a fresh PlotView in an ordinary separate top-level test window."""
        if not self._allow_fullscreen or self.is_fullscreen_view():
            return

        # IMPORTANT: field-map dialogs are launched with QDialog.exec(), which makes
        # them application-modal. An unparented second top-level window can still
        # render behind such a dialog but Qt deliberately blocks its mouse/keyboard
        # input and activation. Parent the new *window* to the PlotView's owning
        # top-level window so it belongs to the modal family while remaining a
        # separate native window.
        owner = self.window()
        host = _FullscreenPlotHost(owner)
        host.exit_requested.connect(self.exit_fullscreen)

        # Construct the clone with the fullscreen host as its parent from the
        # beginning.  No live QWebEngineView crosses a native window boundary.
        clone = PlotView(
            host,
            include_3d_return_axes=self._include_3d_return_axes,
            synchronize_legend_visibility=self._synchronize_legend_visibility,
            allow_fullscreen=False,
            image_filename=self._image_filename,
        )
        clone.selection_requested.connect(self.selection_requested.emit)
        clone.visibility_requested.connect(self.visibility_requested.emit)
        clone.image_ready.connect(self.image_ready.emit)
        clone.image_failed.connect(self.image_failed.emit)
        host.layout().addWidget(clone)

        self._fullscreen_host = host
        self._fullscreen_clone = clone
        self.setEnabled(False)

        # Replay the latest Python-side figure into the independent renderer.
        # Capture the current camera/zoom from the windowed renderer separately
        # so entering fullscreen starts from what the user is actually looking at.
        if self._last_figure_json is not None:
            try:
                figure = json.loads(self._last_figure_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                figure = None
            if isinstance(figure, dict):
                clone.set_figure(figure)
                clone.apply_incremental_update(
                    trace_updates={
                        index: dict(updates)
                        for index, updates in self._incremental_trace_state.items()
                    },
                    layout_updates=dict(self._incremental_layout_state),
                )
            else:
                clone.show_message("Fullscreen viewer unavailable", "Unable to replay this plot.")
        else:
            clone.show_message("Fullscreen viewer", "No rendered plot is available yet.")

        self._capture_view_state(clone._apply_view_state)

        # The independent renderer is now proven to work as a normal and frameless
        # window.  Make only the host window truly fullscreen; the original PlotView
        # and its QWebEngineView remain untouched in their original native window.
        # Seed the host geometry onto the originating screen first so multi-monitor
        # window managers choose the same display for showFullScreen().
        try:
            screen = self.screen()
            geometry = screen.geometry() if screen is not None else None
        except (AttributeError, RuntimeError):
            geometry = None
        if geometry is not None:
            host.setGeometry(geometry)
        else:
            host.resize(1100, 760)

        host.showFullScreen()
        host.raise_()
        host.activateWindow()
        if clone.web is not None:
            clone.web.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        else:
            clone.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

        # Some window managers apply the transient/modal relationship after the
        # show event. Repeat activation on the next event-loop turn.
        QTimer.singleShot(0, self._reactivate_fullscreen_host)
        self.fullscreen_changed.emit(True)

    def _reactivate_fullscreen_host(self) -> None:
        """Repeat activation only while the detached host still exists.

        A queued bound C++ method keeps a Shiboken wrapper after Qt has deleted
        its widget.  Fast open/close cycles (and UI-test teardown) could
        therefore call ``raise_`` on an already-deleted host.  Queue this
        Python-side guard instead and resolve the current host at delivery time.
        """
        host = self._fullscreen_host
        if host is None:
            return
        try:
            host.raise_()
            host.activateWindow()
        except RuntimeError:
            # The owning native window may have been destroyed earlier in this
            # same event-loop turn.  There is nothing left to reactivate.
            return

    def exit_fullscreen(self) -> None:
        """Close the independent fullscreen renderer and reactivate this viewer."""
        host = self._fullscreen_host
        clone = self._fullscreen_clone
        if host is None:
            return

        # Clear state before deleting widgets so repeated Escape/F11/close events
        # cannot enter the teardown twice.
        self._fullscreen_host = None
        self._fullscreen_clone = None
        self.setEnabled(True)

        if clone is not None:
            # Keep the clone parented to its host through destruction as well;
            # even teardown should not move a live QWebEngine surface across a
            # native-window boundary.
            clone.cleanup()
            clone.deleteLater()

        host._allow_close = True
        host.hide()
        host.deleteLater()

        if self.web is not None:
            self.web.setFocus()
        else:
            self.setFocus()
        self.fullscreen_changed.emit(False)

    def request_home_on_next_figure(self) -> None:
        """Cleanly rebuild and apply the shared Home fit to the next figure."""
        self._home_on_next_figure = True
        self._clean_rebuild_on_next_figure = True

    def request_fit_bounds_on_next_figure(self) -> None:
        """Fit the next 3D scene to all visible geometry without changing view direction."""
        self._fit_bounds_on_next_figure = True

    def request_clean_rebuild_on_next_figure(self) -> None:
        """Atomically rebuild structural trace changes while retaining the view."""
        self._clean_rebuild_on_next_figure = True

    def apply_incremental_update(
        self,
        *,
        trace_updates: dict[int, dict[str, object]] | None = None,
        layout_updates: dict[str, object] | None = None,
    ) -> bool:
        """Apply persistent scalar overrides and send only the live delta.

        Trace geometry is intentionally unsupported here. The method is for
        compact properties such as color, opacity, visible, axis visibility, and
        title text. The complete accumulated state is retained so it can be
        replayed after a structural/theme rebuild or into a fullscreen clone,
        while ordinary interactions transmit only the values changed by this
        call.
        """

        delta_trace_updates: dict[int, dict[str, object]] = {}
        for raw_index, raw_updates in (trace_updates or {}).items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if index < 0 or not isinstance(raw_updates, dict) or not raw_updates:
                continue
            updates = dict(raw_updates)
            self._incremental_trace_state.setdefault(index, {}).update(updates)
            delta_trace_updates[index] = updates

        delta_layout_updates = (
            dict(layout_updates)
            if isinstance(layout_updates, dict) and layout_updates
            else {}
        )
        if delta_layout_updates:
            self._incremental_layout_state.update(delta_layout_updates)

        if not delta_trace_updates and not delta_layout_updates:
            return False
        if self.web is None:
            return False
        if not self._ready:
            # The persistent state above will be replayed after page load.
            return True
        return self._push_incremental_delta(
            trace_updates=delta_trace_updates,
            layout_updates=delta_layout_updates,
        )

    @staticmethod
    def _incremental_trace_groups(
        trace_updates: dict[int, dict[str, object]],
    ) -> list[dict[str, object]]:
        """Batch traces by property *shape*, retaining per-trace values.

        Grouping by identical values caused one expensive Plotly redraw for every
        LPBA40 major-region colour when a large selection was restored. Grouping
        by the set of property names instead lets a heterogeneous set of colours
        and opacities travel through one restyle/update call.
        """

        groups: dict[tuple[str, ...], dict[str, object]] = {}
        for index, updates in sorted(trace_updates.items()):
            if not isinstance(updates, dict) or not updates:
                continue
            keys = tuple(sorted(str(key) for key in updates))
            group = groups.setdefault(
                keys,
                {
                    "indices": [],
                    "updates": {key: [] for key in keys},
                },
            )
            group["indices"].append(int(index))
            values = group["updates"]
            for key in keys:
                values[key].append(updates[key])
        return list(groups.values())

    def _push_incremental_state(self) -> bool:
        """Replay the complete persistent override state after a full render."""
        return self._push_incremental_delta(
            trace_updates=self._incremental_trace_state,
            layout_updates=self._incremental_layout_state,
        )

    def _push_incremental_delta(
        self,
        *,
        trace_updates: dict[int, dict[str, object]] | None = None,
        layout_updates: dict[str, object] | None = None,
    ) -> bool:
        """Send one compact incremental delta through the browser render queue.

        Deltas are never discarded as stale: unlike the old implementation they
        do not contain the full accumulated state, so every queued delta must be
        applied in order. Trace groups with the same property names carry arrays
        of per-trace values, allowing e.g. dozens of LPBA40 shells with different
        colours to be updated in a single Plotly call.
        """
        if self.web is None or not self._ready:
            return False
        trace_groups = self._incremental_trace_groups(trace_updates or {})
        trace_payload = json.dumps(
            trace_groups,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        layout_payload = json.dumps(
            layout_updates or {},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self._incremental_update_serial += 1
        update_serial = self._incremental_update_serial
        script = f"""
(() => {{
  const updateSerial = {update_serial};
  const traceGroups = {trace_payload};
  const layoutUpdates = {layout_payload};
  const applyIncrementalDelta = () => {{
    const plot = document.getElementById('plot');
    if (!plot || !plot.data) return null;
    let chain = Promise.resolve();
    let layoutPending = Object.keys(layoutUpdates).length > 0;
    for (const group of traceGroups) {{
      const indices = (group.indices || []).filter(
        index => Number.isInteger(index) && index >= 0 && index < plot.data.length
      );
      if (!indices.length) continue;
      const updates = group.updates || {{}};
      if (layoutPending) {{
        chain = chain.then(() => Plotly.update(plot, updates, layoutUpdates, indices));
        layoutPending = false;
      }} else {{
        chain = chain.then(() => Plotly.restyle(plot, updates, indices));
      }}
    }}
    if (layoutPending) {{
      chain = chain.then(() => Plotly.relayout(plot, layoutUpdates));
    }}
    return chain.then(() => window.fieldWorkbenchApplyPickingPolicy(plot));
  }};
  window.fieldWorkbenchRenderQueue = (window.fieldWorkbenchRenderQueue || Promise.resolve())
    .catch(() => undefined)
    .then(applyIncrementalDelta)
    .catch(error => console.warn('Incremental Plotly update failed', updateSerial, error));
}})();
"""
        try:
            self.web.page().runJavaScript(script)
        except RuntimeError:
            return False
        return True

    def _run_plot_script(self, body: str) -> bool:
        if self.web is None or not self._ready:
            return False
        script = f"""
(() => {{
  const plot = document.getElementById('plot');
  if (!plot || !plot.data) return;
  {body}
}})();
"""
        self.web.page().runJavaScript(script)
        return True

    def home_view(self) -> bool:
        """Fit every axis and restore a perspective isometric camera."""
        return self._run_plot_script(
            """
  window.fieldWorkbenchHome(plot);
"""
        )

    def set_camera_preset(self, preset: str) -> bool:
        """Fit the scene and switch to a readable cardinal engineering view."""
        cameras = {
            "front": {
                "eye": {"x": 0, "y": -1.9, "z": 0},
                "up": {"x": 0, "y": 0, "z": 1},
                "hidden_axis": "y",
            },
            "right": {
                "eye": {"x": 1.9, "y": 0, "z": 0},
                "up": {"x": 0, "y": 0, "z": 1},
                "hidden_axis": "x",
            },
            "top": {
                "eye": {"x": 0, "y": 0, "z": 1.9},
                "up": {"x": 0, "y": 1, "z": 0},
                "hidden_axis": "z",
            },
        }
        camera = cameras.get(str(preset).lower())
        if camera is None:
            return False
        hidden_axis = camera.pop("hidden_axis")
        camera["center"] = {"x": 0, "y": 0, "z": 0}
        camera["projection"] = {"type": "perspective"}
        return self._run_plot_script(
            f"window.fieldWorkbenchFitScene(plot, {json.dumps(camera)}, "
            f"{json.dumps(hidden_axis)});"
        )

    def request_png(self, *, width: int, height: int, scale: float = 2.0) -> bool:
        """Ask Plotly for a high-resolution rendering of the current view."""
        if self._bridge is None:
            return False
        options = json.dumps(
            {
                "format": "png",
                "width": max(1, int(width)),
                "height": max(1, int(height)),
                "scale": max(0.25, float(scale)),
            }
        )
        return self._run_plot_script(
            f"""
  Plotly.toImage(plot, {options})
    .then(dataUrl => window.fieldWorkbenchBridge.deliverImage(dataUrl))
    .catch(error => window.fieldWorkbenchBridge.deliverImageError(String(error)));
"""
        )

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        _ACTIVE_PLOT_VIEWS.discard(self)
        # A parent window can be closed while its independent fullscreen clone
        # is open.  Tear that clone down first so no WebEngine host outlives its
        # owning Field Workbench viewer.
        self.exit_fullscreen()
        self._ready = False
        self._pending_figure_json = None
        self._pending_view_callbacks.clear()
        web = self.web
        self.web = None
        self._channel = None
        self._bridge = None
        if web is not None:
            # Purge Plotly while the page still owns a live WebGL context. Merely
            # deleting the temporary HTML directory left all GL objects for Qt's
            # late native-window teardown, which can no longer make that context
            # current and prints an OpenGL cleanup warning.
            try:
                web.page().runJavaScript(
                    "window.fieldWorkbenchDispose && window.fieldWorkbenchDispose();"
                )
            except RuntimeError:
                pass
            try:
                web.stop()
            except RuntimeError:
                pass
            try:
                self.removeWidget(web)
                web.hide()
            except RuntimeError:
                pass
            try:
                web.deleteLater()
            except RuntimeError:
                # Its native parent may already have completed deletion in the
                # same event-loop turn; there is then nothing left to queue.
                pass
        shutil.rmtree(self._temporary_directory, ignore_errors=True)

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        self.cleanup()
        super().closeEvent(event)
