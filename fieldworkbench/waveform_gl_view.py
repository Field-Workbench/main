"""Shared WebGL2 renderer for static field maps and waveform playback.

The viewer uploads a fixed field plus any field-per-ampere bases to WebGL once,
then changes only a small current-uniform vector during playback. 2D pan/zoom and
3D camera/slice state therefore remain independent of waveform playback.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .theme import current_theme, plot_theme_colors

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - minimal Qt installs
    QWebEngineView = None

try:
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
except ImportError:  # pragma: no cover - older/minimal Qt installs
    QWebEnginePage = None
    QWebEngineSettings = None

try:
    from PySide6.QtWebChannel import QWebChannel
except ImportError:  # pragma: no cover - older/minimal Qt installs
    QWebChannel = None


_GPU_PAGE_STALL_TIMEOUT_MS = 30_000
_GPU_PAGE_AUTOMATIC_RETRIES = 1
_ACTIVE_WAVEFORM_GL_VIEWS: "weakref.WeakSet[WaveformGLView]" = weakref.WeakSet()
_WAVEFORM_GL_SHUTDOWN_CONNECTED = False


def _cleanup_waveform_gl_views_for_shutdown() -> None:
    """Release WebGL pages before QApplication begins destroying native surfaces."""

    for view in list(_ACTIVE_WAVEFORM_GL_VIEWS):
        try:
            view.cleanup()
        except RuntimeError:
            # Qt may already have deleted a child through its native parent.
            pass


def _ensure_waveform_gl_shutdown_hook() -> None:
    global _WAVEFORM_GL_SHUTDOWN_CONNECTED
    if _WAVEFORM_GL_SHUTDOWN_CONNECTED:
        return
    application = QApplication.instance()
    if application is None:
        return
    application.aboutToQuit.connect(_cleanup_waveform_gl_views_for_shutdown)
    _WAVEFORM_GL_SHUTDOWN_CONNECTED = True


class _WaveformWebBridge(QObject):
    """Narrow JavaScript bridge for actions that belong in native Qt UI."""

    def __init__(self, owner: "WaveformGLView"):
        super().__init__(owner)
        self._owner = owner

    @Slot()
    def exportTraces(self) -> None:  # noqa: N802 - JavaScript-facing Qt slot
        self._owner.export_traces()

    @Slot(str)
    def cameraChanged(self, camera_json: str) -> None:  # noqa: N802 - JavaScript-facing Qt slot
        self._owner._camera_changed_from_web(camera_json)

    @Slot(int, float)
    def frameChanged(self, frame_index: int, time_s: float) -> None:  # noqa: N802
        self._owner._frame_changed_from_web(frame_index, time_s)

    @Slot(str)
    def gpuValidationReady(self, report_json: str) -> None:  # noqa: N802 - JavaScript-facing Qt slot
        self._owner._gpu_validation_ready_from_web(report_json)


class WaveformGLView(QStackedWidget):
    """Persistent GPU waveform view with interaction independent of playback."""

    ready = Signal(bool)
    cameraChanged = Signal(dict)
    frameChanged = Signal(int, float)

    def __init__(
        self,
        payload: dict[str, Any],
        parent=None,
        *,
        preview_mode: bool = False,
        initial_camera: dict[str, Any] | None = None,
        start_hibernated: bool = False,
        _source_html_path: Path | None = None,
    ):
        super().__init__(parent)
        self._workbench_theme = current_theme()
        browser_payload = dict(payload)
        browser_payload["workbench_theme"] = self._workbench_theme
        if preview_mode:
            browser_payload["export_preview"] = True
        if initial_camera is not None:
            browser_payload["initial_camera"] = dict(initial_camera)
        # The browser document is the durable copy of the large base64 field and
        # scene buffers. Keep only native actions/metadata in Python after writing
        # it; otherwise every open result retains a second complete field payload.
        self._payload = self._native_payload(browser_payload)
        self._temporary_directory: Path | None = None
        self._html_path: Path | None = None
        if _source_html_path is not None:
            self._html_path = Path(_source_html_path)
        else:
            self._temporary_directory = Path(
                tempfile.mkdtemp(prefix="field-workbench-waveform-gl-")
            )
            self._html_path = self._temporary_directory / "waveform-gl.html"
            self._html_path.write_text(
                self._html(browser_payload), encoding="utf-8"
            )
        del browser_payload
        self._preview_mode = bool(preview_mode)
        self._page_started = False
        self._last_load_progress = -1
        self._load_retry_count = 0
        self._load_failed = False
        self._cleaned_up = False
        self._load_watchdog = QTimer(self)
        self._load_watchdog.setSingleShot(True)
        self._load_watchdog.setInterval(_GPU_PAGE_STALL_TIMEOUT_MS)
        self._load_watchdog.timeout.connect(self._gpu_page_load_stalled)
        self._requested_lifecycle = (
            "discarded" if start_hibernated else "active"
        )
        self._lifecycle_retry_pending = False
        self._fullscreen_host: QWidget | None = None
        self._fullscreen_clone: WaveformGLView | None = None
        self._web_channel = None
        self._web_bridge = None
        self._gpu_validation_callback = None
        self._gpu_validation_timeout = None
        self._brain_highlight_label_ids: tuple[int, ...] = ()
        self.message = QLabel("Preparing GPU waveform view…")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._style_message_surface()
        self.addWidget(self.message)
        self.web = None
        _ACTIVE_WAVEFORM_GL_VIEWS.add(self)
        _ensure_waveform_gl_shutdown_hook()
        if QWebEngineView is None:
            self.message.setText(
                "GPU waveform playback requires the Qt WebEngine component used by Field Workbench's plot viewers."
            )
            QTimer.singleShot(0, lambda: self.ready.emit(False))
            return
        if self._requested_lifecycle == "active":
            self._create_web_view()
            self._start_page_load()
        else:
            self.message.setText(
                "Result hibernated to conserve memory — select this tab to load it."
            )

    def _create_web_view(self) -> None:
        """Create one fresh Chromium surface for this already-built local result."""

        if QWebEngineView is None or self._cleaned_up:
            return
        web = QWebEngineView()
        if QWebEngineSettings is not None:
            try:
                web.settings().setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                web.settings().setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            except (AttributeError, TypeError):
                pass
        web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        if QWebChannel is not None:
            channel = QWebChannel(web.page())
            bridge = _WaveformWebBridge(self)
            bridge.setParent(channel)
            channel.registerObject("fieldWorkbenchBridge", bridge)
            web.page().setWebChannel(channel)
            self._web_channel = channel
            self._web_bridge = bridge
        web.loadStarted.connect(lambda web=web: self._page_load_started(web))
        web.loadProgress.connect(
            lambda progress, web=web: self._page_load_progress(web, progress)
        )
        web.loadFinished.connect(
            lambda ok, web=web: self._page_loaded(web, ok)
        )
        try:
            web.page().renderProcessTerminated.connect(
                lambda status, exit_code, web=web: self._render_process_terminated(
                    web, status, exit_code
                )
            )
        except (AttributeError, RuntimeError):
            pass
        self.web = web
        self.addWidget(web)

    def _style_message_surface(self) -> None:
        colours = plot_theme_colors(self._workbench_theme)
        text_colour = colours["text"] if self._workbench_theme == "dark" else "#64748b"
        self.message.setStyleSheet(
            "QLabel {"
            f"font-size:16px; color:{text_colour}; "
            f"background-color:{colours['figure']};"
            "}"
        )

    def _apply_browser_theme(self) -> None:
        if not self._web_page_is_active(self.web):
            return
        encoded = json.dumps(self._workbench_theme)
        try:
            self.web.page().runJavaScript(
                "window.fieldWorkbenchSetTheme && "
                f"window.fieldWorkbenchSetTheme({encoded});"
            )
        except RuntimeError:
            pass

    def apply_workbench_theme(self, theme: str) -> None:
        """Refresh an already-open GPU viewer after Light/Dark changes."""

        selected = str(theme).strip().lower()
        if selected not in {"light", "dark"}:
            return
        self._workbench_theme = selected
        self._style_message_surface()
        self._apply_browser_theme()

    @staticmethod
    def _web_page_is_active(web) -> bool:
        """Return whether JavaScript may safely run on this WebEngine page."""

        if web is None or QWebEnginePage is None:
            return web is not None
        try:
            return web.page().lifecycleState() == QWebEnginePage.LifecycleState.Active
        except (AttributeError, RuntimeError):
            return False

    def _dispose_browser_gpu_resources(self, web) -> None:
        """Ask the page to release app-owned WebGL objects while its context is live."""

        if not self._web_page_is_active(web):
            return
        try:
            web.page().runJavaScript(
                "window.fieldWorkbenchDispose && window.fieldWorkbenchDispose();"
            )
        except RuntimeError:
            pass

    def _retire_web_view(self, web) -> None:
        """Stop and retire one renderer surface without touching the local HTML."""

        if web is None:
            return
        self._dispose_browser_gpu_resources(web)
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
            # Parent-driven teardown may already have deleted the native view.
            pass

    def _discard_web_view(self) -> None:
        """Release a hidden browser surface while retaining its local result HTML."""

        old = self.web
        self.web = None
        self._web_channel = None
        self._web_bridge = None
        self._page_started = False
        self._last_load_progress = -1
        self._load_watchdog.stop()
        self._retire_web_view(old)

    def _replace_web_view(self) -> None:
        self._discard_web_view()
        self._create_web_view()

    def _arm_load_watchdog(self) -> None:
        if self._requested_lifecycle == "active" and not self._cleaned_up:
            self._load_watchdog.start()

    def _page_load_started(self, web) -> None:
        if web is not self.web or self._cleaned_up:
            return
        self._page_started = True
        self._last_load_progress = 0
        self._load_failed = False
        self.message.setText("Loading GPU result…")
        self.setCurrentWidget(self.message)
        self._arm_load_watchdog()

    def _page_load_progress(self, web, progress: int) -> None:
        if web is not self.web or not self._page_started or self._cleaned_up:
            return
        # Chromium may emit duplicate loadProgress values while its renderer is
        # wedged. Only genuine forward progress earns another watchdog window;
        # otherwise repeated identical signals could keep a dead page alive forever.
        try:
            progress_value = int(progress)
        except (TypeError, ValueError):
            return
        if progress_value > self._last_load_progress:
            self._last_load_progress = progress_value
            self._arm_load_watchdog()

    def _gpu_page_load_stalled(self) -> None:
        """Recover an occasional Chromium/WebGL page startup that stops advancing."""

        if self._cleaned_up or self.web is None or not self._page_started:
            return
        if self._requested_lifecycle != "active":
            # A hidden result does not need to finish constructing a GPU context.
            self._discard_web_view()
            self.message.setText(
                "Result hibernated to conserve memory — select this tab to load it."
            )
            return
        if self._load_retry_count < _GPU_PAGE_AUTOMATIC_RETRIES:
            self._load_retry_count += 1
            self.message.setText("GPU viewer stalled — restarting renderer…")
            self.setCurrentWidget(self.message)
            self._replace_web_view()
            QTimer.singleShot(0, self._start_page_load)
            return
        # Never leave a tab spinning forever. A later tab re-selection gets a
        # fresh manual attempt without repeating the magnetic-field calculation.
        self._discard_web_view()
        self._load_failed = True
        self.message.setText(
            "The GPU viewer did not finish loading. Switch tabs and return to retry this result."
        )
        self.setCurrentWidget(self.message)
        self.ready.emit(False)

    @staticmethod
    def _native_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Retain only values used by native actions after HTML serialization."""

        keys = (
            "renderer",
            "static_view",
            "view_mode",
            "render_mode",
            "title",
            "centre_mm",
            "span_mm",
            "resolution",
            "observation",
            "observations",
            "excitation",
            "sequencer",
        )
        return {key: payload[key] for key in keys if key in payload}

    def can_export_video(self) -> bool:
        """Return whether this tab is an already-rendered animated map."""

        return (
            str(self._payload.get("view_mode", "")).lower() in {"2d", "3d"}
            and not bool(self._payload.get("static_view", False))
        )

    def export_payload(self) -> dict[str, Any]:
        """Recover the frozen animation payload from this tab's local HTML.

        Animated result tabs intentionally discard the large Python field arrays
        after writing their self-contained local WebGL document. Video export can
        therefore reuse the exact rendered result without repeating a magnetic-field
        solve by reading the payload back only when the user chooses Export video.
        """

        if not self.can_export_video() or self._html_path is None:
            raise ValueError("The selected tab is not an animated field-map result.")
        try:
            html = self._html_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"The rendered animation payload is unavailable: {error}") from error
        marker = '<script id="payload" type="application/json">'
        start = html.find(marker)
        if start < 0:
            raise ValueError("The rendered animation contains no recoverable payload.")
        start += len(marker)
        end = html.find("</script>", start)
        if end < 0:
            raise ValueError("The rendered animation payload is incomplete.")
        try:
            payload = json.loads(html[start:end])
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("The rendered animation payload could not be decoded.") from error
        if not isinstance(payload, dict):
            raise ValueError("The rendered animation payload is malformed.")
        payload.pop("export_preview", None)
        return payload

    def _start_page_load(self) -> None:
        if self._cleaned_up or self._html_path is None:
            return
        if self.web is None:
            self._create_web_view()
        if self.web is None or self._page_started:
            return
        self._load_failed = False
        self.message.setText("Loading GPU result…")
        self.setCurrentWidget(self.message)
        self._page_started = True
        self._last_load_progress = 0
        self._arm_load_watchdog()
        self.web.load(QUrl.fromLocalFile(str(self._html_path.resolve())))

    def _page_loaded(self, web, ok: bool) -> None:
        if web is not self.web or self._cleaned_up:
            return
        self._load_watchdog.stop()
        if ok:
            self._last_load_progress = 100
            self._load_retry_count = 0
            self._load_failed = False
            self.setCurrentWidget(web)
            self._apply_browser_theme()
            self._apply_brain_highlight()
            self._schedule_renderer_wake()
        elif self._requested_lifecycle == "active" and self._load_retry_count < _GPU_PAGE_AUTOMATIC_RETRIES:
            self._load_retry_count += 1
            self.message.setText("GPU viewer load failed — restarting renderer…")
            self.setCurrentWidget(self.message)
            self._replace_web_view()
            QTimer.singleShot(0, self._start_page_load)
            return
        else:
            self._discard_web_view()
            self._load_failed = True
            self.message.setText(
                "Unable to initialize the GPU waveform renderer. Switch tabs and return to retry."
            )
            self.setCurrentWidget(self.message)
        self.ready.emit(bool(ok and self.web is not None))
        # A result can finish while tab visibility is changing. Reapply the
        # controller's requested lifecycle after the load callback settles.
        QTimer.singleShot(0, self._apply_requested_lifecycle)

    def _render_process_terminated(self, web, _status, _exit_code: int) -> None:
        """Recover if Chromium loses the renderer backing the selected result."""

        if web is not self.web or self._cleaned_up:
            return
        if self._requested_lifecycle != "active":
            self._discard_web_view()
            return
        self._load_watchdog.stop()
        if self._load_retry_count < _GPU_PAGE_AUTOMATIC_RETRIES:
            self._load_retry_count += 1
            self.message.setText("GPU renderer stopped — restarting viewer…")
            self.setCurrentWidget(self.message)
            self._replace_web_view()
            QTimer.singleShot(0, self._start_page_load)
            return
        self._discard_web_view()
        self._load_failed = True
        self.message.setText(
            "The GPU renderer stopped unexpectedly. Switch tabs and return to retry this result."
        )
        self.setCurrentWidget(self.message)
        self.ready.emit(False)

    def _camera_changed_from_web(self, camera_json: str) -> None:
        try:
            camera = json.loads(str(camera_json))
        except (TypeError, ValueError):
            return
        if isinstance(camera, dict):
            self.cameraChanged.emit(camera)

    def _frame_changed_from_web(self, frame_index: int, time_s: float) -> None:
        if self._cleaned_up:
            return
        try:
            index = max(0, int(frame_index))
            value = float(time_s)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            value = 0.0
        self.frameChanged.emit(index, value)

    def _apply_brain_highlight(self) -> None:
        if not self._web_page_is_active(self.web):
            return
        encoded = json.dumps(list(self._brain_highlight_label_ids))
        try:
            self.web.page().runJavaScript(
                "window.fieldWorkbenchSetBrainHighlight && "
                f"window.fieldWorkbenchSetBrainHighlight({encoded});"
            )
        except RuntimeError:
            pass

    def set_brain_region_highlight(self, label_ids) -> None:
        """Highlight LPBA40 overlay samples in this result viewer only."""

        try:
            values = tuple(sorted({int(value) for value in label_ids}))
        except (TypeError, ValueError):
            values = ()
        self._brain_highlight_label_ids = values
        self._apply_brain_highlight()

    def _gpu_validation_ready_from_web(self, report_json: str) -> None:
        callback = self._gpu_validation_callback
        self._gpu_validation_callback = None
        timer = self._gpu_validation_timeout
        self._gpu_validation_timeout = None
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        if callback is None:
            return
        try:
            report = json.loads(str(report_json))
        except (TypeError, ValueError):
            report = {"ok": False, "error": "The GPU validation result could not be decoded."}
        QTimer.singleShot(0, lambda: callback(report))

    def _gpu_validation_timed_out(self) -> None:
        callback = self._gpu_validation_callback
        self._gpu_validation_callback = None
        timer = self._gpu_validation_timeout
        self._gpu_validation_timeout = None
        if timer is not None:
            timer.deleteLater()
        if callback is not None:
            callback({"ok": False, "error": "The WebGL renderer did not return GPU validation data."})

    def validate_gpu_volume(self, callback) -> None:
        """Compare Full Volume texture sampling against its source lattice values."""
        if not self._web_page_is_active(self.web):
            QTimer.singleShot(0, lambda: callback({"ok": False, "error": "The GPU viewer is unavailable."}))
            return
        if str(self._payload.get("render_mode", "")) != "volume":
            QTimer.singleShot(0, lambda: callback({"ok": False, "error": "GPU volume validation requires a Full Volume animation."}))
            return
        if self._gpu_validation_callback is not None:
            QTimer.singleShot(0, lambda: callback({"ok": False, "error": "A GPU validation request is already running."}))
            return
        self._gpu_validation_callback = callback
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(5000)
        timer.timeout.connect(self._gpu_validation_timed_out)
        self._gpu_validation_timeout = timer
        timer.start()
        try:
            self.web.page().runJavaScript(
                "window.fieldWorkbenchValidateGpuVolume && window.fieldWorkbenchValidateGpuVolume();"
            )
        except RuntimeError:
            timer.stop()
            self._gpu_validation_timed_out()

    def set_camera(self, camera: dict[str, Any]) -> None:
        """Apply an explicit fixed camera to an export preview."""
        if not self._web_page_is_active(self.web):
            return
        encoded = json.dumps(camera, separators=(",", ":"), ensure_ascii=False)
        try:
            self.web.page().runJavaScript(
                f"window.fieldWorkbenchSetCamera && window.fieldWorkbenchSetCamera({encoded});"
            )
        except RuntimeError:
            pass

    def request_camera(self, callback) -> None:
        """Return the live preview camera through a Qt WebEngine callback."""
        if not self._web_page_is_active(self.web):
            QTimer.singleShot(0, lambda: callback(None))
            return
        try:
            self.web.page().runJavaScript(
                "window.fieldWorkbenchGetCamera ? window.fieldWorkbenchGetCamera() : null;",
                callback,
            )
        except RuntimeError:
            QTimer.singleShot(0, lambda: callback(None))

    def capture_frame(
        self,
        *,
        frame_index: int,
        width: int,
        height: int,
        camera: dict[str, Any],
        callback,
        sequence_elapsed_s: float | None = None,
        timeline_elapsed_s: float | None = None,
    ) -> None:
        """Render one exact timeline frame and return it as a PNG data URL."""
        if not self._web_page_is_active(self.web):
            QTimer.singleShot(0, lambda: callback(None))
            return
        options = {
            "frame_index": int(frame_index),
            "width": int(width),
            "height": int(height),
            "camera": dict(camera),
        }
        if sequence_elapsed_s is not None:
            options["sequence_elapsed_s"] = float(sequence_elapsed_s)
        if timeline_elapsed_s is not None:
            options["timeline_elapsed_s"] = float(timeline_elapsed_s)
        encoded = json.dumps(options, separators=(",", ":"), ensure_ascii=False)
        try:
            self.web.page().runJavaScript(
                "window.fieldWorkbenchCaptureFrame "
                f"? window.fieldWorkbenchCaptureFrame({encoded}) : null;",
                callback,
            )
        except RuntimeError:
            QTimer.singleShot(0, lambda: callback(None))

    def _run_renderer_wake(self) -> None:
        """Force a visible WebEngine canvas to resize/repaint after tab activation.

        Qt WebEngine can occasionally initialize a WebGL canvas while its tab is
        still hidden. In that case Chromium may retain a black compositor surface
        until the first pointer event. Ask the page to resize and redraw explicitly
        instead of relying on user interaction to wake it.
        """
        web = self.web
        if not self._web_page_is_active(web):
            return
        try:
            web.update()
            web.page().runJavaScript(
                "window.fieldWorkbenchWaveformWake && window.fieldWorkbenchWaveformWake();"
            )
        except RuntimeError:
            pass

    def _schedule_renderer_wake(self) -> None:
        for delay_ms in (0, 60, 220, 500):
            QTimer.singleShot(delay_ms, self._run_renderer_wake)

    def refresh_viewport(self) -> None:
        """Resize/redraw after this viewer is made visible in a map tab."""
        self.updateGeometry()
        self.update()
        if self.web is not None:
            self.web.updateGeometry()
            self.web.update()
        self._schedule_renderer_wake()

    def showEvent(self, event):  # noqa: N802 - Qt API
        super().showEvent(event)
        self.set_backgrounded(False)

    def hideEvent(self, event):  # noqa: N802 - Qt API
        super().hideEvent(event)
        self.set_backgrounded(True)

    def stop_playback(self) -> None:
        """Stop this renderer's playback without changing its current frame."""
        web = self.web
        if not self._page_started or not self._web_page_is_active(web):
            return
        try:
            web.page().runJavaScript(
                "window.fieldWorkbenchWaveformStop && window.fieldWorkbenchWaveformStop();"
            )
        except RuntimeError:
            pass

    def set_backgrounded(self, backgrounded: bool, *, discard: bool = False) -> None:
        """Suspend or hibernate a hidden renderer; reactivate it when selected."""

        if backgrounded:
            self.stop_playback()
            # For 3D result tabs, release the entire Chromium surface ourselves
            # rather than placing QWebEnginePage in Discarded. Qt's transition out
            # of Discarded reloads asynchronously, leaving a window where queued
            # JavaScript calls target a page that explicitly refuses JavaScript.
            # Recreating QWebEngineView on selection is deterministic and still
            # reloads only the already-generated local HTML payload.
            if discard or self._requested_lifecycle == "discarded":
                self._requested_lifecycle = "discarded"
                self._discard_web_view()
                self.message.setText(
                    "Result hibernated to conserve memory — select this tab to load it."
                )
                self.setCurrentWidget(self.message)
                return
            self._requested_lifecycle = "frozen"
        else:
            self._requested_lifecycle = "active"
            if self._load_failed:
                self._load_failed = False
                self._load_retry_count = 0
                self._page_started = False
                self._last_load_progress = -1
        self._apply_requested_lifecycle()

    def _retry_requested_lifecycle(self) -> None:
        self._lifecycle_retry_pending = False
        self._apply_requested_lifecycle()

    def _apply_requested_lifecycle(self) -> None:
        requested = self._requested_lifecycle
        if requested == "discarded":
            self._discard_web_view()
            return
        web = self.web
        if requested == "active":
            if web is None:
                self._create_web_view()
                web = self.web
            if web is None:
                return
            if not self._page_started:
                self._start_page_load()
                return
        if web is None or not self._page_started or QWebEnginePage is None:
            return
        page = web.page()
        if requested != "active":
            try:
                page_visible = bool(page.isVisible())
            except (AttributeError, RuntimeError):
                page_visible = self.isVisible()
            if page_visible:
                # currentChanged can arrive just before WebEngine observes the
                # old tab's hide. Retry once after the Qt event has propagated.
                if not self._lifecycle_retry_pending:
                    self._lifecycle_retry_pending = True
                    QTimer.singleShot(0, self._retry_requested_lifecycle)
                return
        state = {
            "active": QWebEnginePage.LifecycleState.Active,
            "frozen": QWebEnginePage.LifecycleState.Frozen,
        }[requested]
        try:
            page.setLifecycleState(state)
        except (AttributeError, RuntimeError):
            return
        if requested == "active":
            self.refresh_viewport()

    def toggle_fullscreen(self) -> None:
        """Open/close an independent fullscreen GPU view without reparenting WebEngine."""
        if self._fullscreen_host is not None:
            self._close_fullscreen()
            return
        # A fullscreen view is an independent WebEngine/WebGL renderer. Explicitly
        # stop the embedded player first so a looping hidden canvas cannot continue
        # consuming GPU time behind the fullscreen clone.
        self.stop_playback()
        host = QWidget(self.window(), Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        host.setWindowTitle("Field Workbench — GPU waveform fullscreen")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        clone = WaveformGLView(
            self._payload,
            host,
            _source_html_path=self._html_path,
        )
        layout.addWidget(clone)
        shortcuts = []
        for key in ("Escape", "F11"):
            shortcut = QShortcut(QKeySequence(key), host)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(self._close_fullscreen)
            shortcuts.append(shortcut)
        host._field_workbench_shortcuts = shortcuts  # keep Qt wrappers alive
        host.destroyed.connect(lambda *_args: self._fullscreen_destroyed())
        self._fullscreen_host = host
        self._fullscreen_clone = clone
        host.showFullScreen()

    def _fullscreen_destroyed(self) -> None:
        self._fullscreen_host = None
        self._fullscreen_clone = None

    def _close_fullscreen(self) -> None:
        host, clone = self._fullscreen_host, self._fullscreen_clone
        self._fullscreen_host = None
        self._fullscreen_clone = None
        if clone is not None:
            clone.cleanup()
        if host is not None:
            host.close()
            host.deleteLater()

    def _trace_export_records(self) -> list[dict[str, Any]]:
        records = []
        observations = self._payload.get("observations")
        if isinstance(observations, list):
            records.extend(item for item in observations if isinstance(item, dict))
        elif isinstance(self._payload.get("observation"), dict):
            records.append(self._payload["observation"])
        excitation = self._payload.get("excitation")
        if isinstance(excitation, dict):
            records.append(excitation)
        sequencer = self._payload.get("sequencer")
        if isinstance(sequencer, dict):
            records.append(sequencer)
        return [
            item
            for item in records
            if isinstance(item.get("time_s"), list)
            and (
                isinstance(item.get("values"), list)
                or (
                    item.get("axis") == "sequencer"
                    and isinstance(item.get("step_numbers"), list)
                    and isinstance(item.get("active_coils"), list)
                )
            )
        ]

    @staticmethod
    def _trace_column_label(trace: dict[str, Any]) -> str:
        label = str(trace.get("label") or "Trace").strip() or "Trace"
        unit = str(trace.get("unit") or "").strip()
        if unit and unit not in label:
            return f"{label} ({unit})"
        return label

    @staticmethod
    def _write_trace_csv(path: Path, traces: list[dict[str, Any]]) -> None:
        if not traces:
            raise ValueError("Choose at least one trace to export.")
        common_time = [float(value) for value in traces[0].get("time_s", [])]
        if not common_time:
            raise ValueError("The selected traces do not contain a time axis.")

        headers = ["Time (s)"]
        columns: list[tuple[str, list[Any]]] = []
        for trace in traces:
            times = [float(value) for value in trace.get("time_s", [])]
            if len(times) != len(common_time):
                raise ValueError("The selected traces do not share one common time axis.")
            for first, second in zip(times, common_time, strict=True):
                tolerance = max(1.0e-12, abs(second) * 1.0e-10)
                if not math.isfinite(first) or abs(first - second) > tolerance:
                    raise ValueError("The selected traces do not share one common time axis.")

            if trace.get("axis") == "sequencer":
                steps = list(trace.get("step_numbers", trace.get("values", [])))
                active = [str(value) for value in trace.get("active_coils", [])]
                if len(steps) != len(common_time) or len(active) != len(common_time):
                    raise ValueError("The sequencer trace does not match the common time axis.")
                try:
                    step_values = [int(value) for value in steps]
                except (TypeError, ValueError) as error:
                    raise ValueError("The sequencer trace contains an invalid step number.") from error
                headers.extend(["Step number", "Active coils"])
                columns.append(("integer", step_values))
                columns.append(("text", active))
                continue

            values = [float(value) for value in trace.get("values", [])]
            if len(values) != len(common_time):
                raise ValueError("The selected traces do not share one common time axis.")
            if not all(math.isfinite(value) for value in values):
                raise ValueError("A selected trace contains non-finite values.")
            headers.append(WaveformGLView._trace_column_label(trace))
            columns.append(("float", values))

        output = Path(path)
        if output.suffix.lower() != ".csv":
            output = output.with_suffix(".csv")
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for index, time_value in enumerate(common_time):
                row: list[str] = [f"{time_value:.12g}"]
                for kind, values in columns:
                    value = values[index]
                    if kind == "float":
                        row.append(f"{float(value):.12g}")
                    elif kind == "integer":
                        row.append(str(int(value)))
                    else:
                        row.append(str(value))
                writer.writerow(row)

    def export_traces(self) -> None:
        traces = self._trace_export_records()
        if not traces:
            QMessageBox.information(self, "Export traces", "There are no waveform traces to export.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Export waveform traces")
        dialog.setMinimumWidth(440)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Choose the traces to include in the CSV file:"))

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        checklist = QWidget(scroll)
        checklist_layout = QVBoxLayout(checklist)
        checklist_layout.setContentsMargins(8, 8, 8, 8)
        checkboxes: list[QCheckBox] = []
        for trace in traces:
            check = QCheckBox(self._trace_column_label(trace), checklist)
            check.setChecked(True)
            checklist_layout.addWidget(check)
            checkboxes.append(check)
        checklist_layout.addStretch(1)
        scroll.setWidget(checklist)
        layout.addWidget(scroll, 1)

        selection_row = QHBoxLayout()
        select_all = QPushButton("Select all", dialog)
        clear_all = QPushButton("Clear all", dialog)
        select_all.clicked.connect(lambda: [check.setChecked(True) for check in checkboxes])
        clear_all.clicked.connect(lambda: [check.setChecked(False) for check in checkboxes])
        selection_row.addWidget(select_all)
        selection_row.addWidget(clear_all)
        selection_row.addStretch(1)
        layout.addLayout(selection_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected = [trace for trace, check in zip(traces, checkboxes, strict=True) if check.isChecked()]
        if not selected:
            QMessageBox.information(self, "Export traces", "Choose at least one trace to export.")
            return
        filename, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export waveform traces",
            "waveform_traces.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not filename:
            return
        try:
            self._write_trace_csv(Path(filename), selected)
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "Export traces", f"Could not export waveform traces:\n{error}")

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        _ACTIVE_WAVEFORM_GL_VIEWS.discard(self)
        self._load_watchdog.stop()
        self._close_fullscreen()
        web = self.web
        self.web = None
        self._web_channel = None
        self._web_bridge = None
        self._page_started = False
        if web is not None:
            # Release the page's own textures/buffers before Qt tears down the
            # native surface. This also makes application-exit cleanup less
            # dependent on a deferred WebEngine destructor finding a live GL context.
            self._retire_web_view(web)
        if self._temporary_directory is not None:
            shutil.rmtree(self._temporary_directory, ignore_errors=True)
            self._temporary_directory = None

    def closeEvent(self, event):  # noqa: N802 - Qt API
        self.cleanup()
        super().closeEvent(event)

    @staticmethod
    def _html(payload: dict[str, Any]) -> str:
        # Keep the full sequencer trace in the native payload for CSV export, but
        # do not ship thousands of repeated step/name samples into Chromium when
        # the realtime player already has the compact step definition. The web
        # lane only needs the common trace-axis endpoints plus playback_sequence.
        browser_payload = dict(payload)
        sequencer = browser_payload.get("sequencer")
        playback_sequence = browser_payload.get("playback_sequence")
        if isinstance(sequencer, dict) and isinstance(playback_sequence, dict):
            times = sequencer.get("time_s")
            if isinstance(times, list) and times:
                browser_payload["sequencer"] = {
                    "time_s": [times[0], times[-1]],
                    "label": sequencer.get("label", "Sequencer"),
                    "unit": sequencer.get("unit", ""),
                    "axis": "sequencer",
                }
        data = json.dumps(browser_payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
        template = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#fff;color:#334155;font-family:Arial,Helvetica,sans-serif}
  #app{display:flex;flex-direction:column;width:100%;height:100%;min-height:0}
  #title{height:38px;flex:0 0 38px;display:flex;align-items:center;justify-content:center;font-size:16px;color:#444}
  #viewer-row{position:relative;display:flex;flex:1 1 auto;min-height:220px;border-top:1px solid #eef2f7;border-bottom:1px solid #e5e7eb}
  #gl,#grid-overlay{position:absolute;inset:0;width:100%;height:100%}
  #gl{touch-action:none;cursor:grab}
  #grid-overlay{pointer-events:none;z-index:2}
  #gl.dragging{cursor:grabbing}
  #toolbar{position:absolute;top:10px;left:10px;display:flex;gap:7px;z-index:3}
  .tool{height:32px;padding:0 11px;border:1px solid #cbd5e1;border-radius:5px;background:rgba(255,255,255,.92);color:#334155;font-weight:600;cursor:pointer}
  .tool:hover{background:white;border-color:#94a3b8}
  #gpu-status{position:absolute;top:11px;right:82px;padding:5px 8px;border-radius:4px;background:rgba(255,255,255,.86);font-size:12px;color:#64748b;z-index:3}
  #colorbar{position:absolute;right:18px;top:48px;bottom:35px;width:48px;z-index:3;pointer-events:none}
  #gradient{position:absolute;right:0;top:20px;bottom:20px;width:18px;border:1px solid #64748b;background:linear-gradient(to top,#08306b 0%,#2563eb 18%,#06b6d4 36%,#22c55e 54%,#facc15 72%,#f97316 86%,#dc2626 100%)}
  #cmax,#cmin{position:absolute;right:23px;width:90px;text-align:right;font-size:11px;color:#475569}
  #cmax{top:16px;transform:translateY(-50%)} #cmin{bottom:16px;transform:translateY(50%)}
  #ctitle{position:absolute;right:0;top:0;font-size:11px;white-space:nowrap;transform:translateY(-100%)}
  #axis-hint{position:absolute;left:12px;bottom:10px;font-size:11px;color:#64748b;background:rgba(255,255,255,.78);padding:3px 5px;border-radius:3px}
  #observation-wrap{display:none;flex:0 0 190px;position:relative;background:white;border-bottom:1px solid #e5e7eb;overflow-x:hidden;overflow-y:hidden}
  #obs{display:block;width:100%;height:100%}
  #obs-stats{position:absolute;right:18px;top:6px;display:flex;gap:10px;align-items:center;font-size:11px;font-weight:600;color:#475569;background:rgba(255,255,255,.90);padding:3px 7px;max-width:72%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #obs-invert-control{display:flex;gap:4px;align-items:center;white-space:nowrap;font-weight:600}
  #controls{flex:0 0 auto;display:grid;grid-template-columns:auto auto auto minmax(180px,1fr);gap:10px 12px;align-items:center;padding:10px 14px 11px;background:white}
  .run{height:42px;min-width:118px;border-radius:7px;border:1px solid transparent;font-size:16px;font-weight:700;cursor:pointer}
  #play:not(:disabled){background:#2e7d32;border-color:#246528;color:#fff}
  #stop:not(:disabled){background:#c62828;border-color:#a82121;color:#fff}
  #stop.reset:not(:disabled){background:#475569;border-color:#334155;color:#fff}
  .run:disabled{background:#e5e7eb;border-color:#d1d5db;color:#9ca3af;cursor:default}
  #time-area{display:grid;grid-template-columns:auto minmax(120px,1fr) auto;gap:8px;align-items:center}
  #time-slider,#slice-slider{width:100%;accent-color:#2563eb}
  #time-text{min-width:150px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px;color:#475569}
  #loop-control{display:flex;gap:6px;align-items:center;font-size:13px;font-weight:600;color:#475569;white-space:nowrap}
  #speed-control{grid-column:1 / 3;display:flex;gap:8px;align-items:flex-start;font-size:12px;font-weight:600;color:#475569;white-space:nowrap}
  #speed-stack{display:flex;flex-direction:column;gap:3px;align-items:flex-start}
  #speed-row{display:flex;gap:8px;align-items:center}
  #playback-speed{height:30px;min-width:86px;padding:0 7px;border:1px solid #cbd5e1;border-radius:5px;background:#fff;color:#334155;font-weight:600}
  #export-traces{height:30px;padding:0 10px;border:1px solid #cbd5e1;border-radius:5px;background:#fff;color:#334155;font-weight:600;cursor:pointer}
  #export-traces:hover:not(:disabled){background:#f8fafc;border-color:#94a3b8}
  #export-traces:disabled{color:#9ca3af;background:#f8fafc;cursor:default}
  #secondary{grid-column:1 / -1;display:grid;grid-template-columns:auto minmax(140px,360px) auto 1fr;gap:9px;align-items:center;font-size:12px;color:#475569}
  #perf{grid-column:1 / -1;text-align:right}
  #perf{text-align:right;font-weight:600;color:#475569}
  #error{display:none;position:absolute;inset:20px;z-index:10;padding:20px;background:#fff3f3;border:1px solid #fecaca;color:#991b1b;white-space:pre-wrap;overflow:auto}
  #export-time{display:none;position:absolute;left:18px;bottom:16px;padding:5px 8px;border-radius:4px;background:rgba(255,255,255,.86);font-size:13px;font-weight:600;color:#334155;z-index:3}
  body.export-preview #title,body.export-preview #toolbar,body.export-preview #gpu-status,body.export-preview #axis-hint,body.export-preview #controls{display:none!important}
  body.export-preview #viewer-row{border:0;min-height:0}
  body.export-preview #export-time{display:block}
  html[data-workbench-theme="dark"],html[data-workbench-theme="dark"] body{background:#aeb3b8;color:#14181c}
  html[data-workbench-theme="dark"] #title,html[data-workbench-theme="dark"] #controls{background:#252b31;color:#e4e8ec}
  html[data-workbench-theme="dark"] #viewer-row{background:#aeb3b8;border-color:#48535e}
  html[data-workbench-theme="dark"] #observation-wrap{background:#aeb3b8;border-color:#48535e}
  html[data-workbench-theme="dark"] .tool{background:rgba(48,55,64,.94);border-color:#48535e;color:#e4e8ec}
  html[data-workbench-theme="dark"] .tool:hover{background:#394550;border-color:#6589a4}
  html[data-workbench-theme="dark"] #gpu-status,html[data-workbench-theme="dark"] #axis-hint,html[data-workbench-theme="dark"] #obs-stats,html[data-workbench-theme="dark"] #export-time{background:rgba(37,43,49,.92);color:#cbd2d8}
  html[data-workbench-theme="dark"] #cmax,html[data-workbench-theme="dark"] #cmin,html[data-workbench-theme="dark"] #time-text,html[data-workbench-theme="dark"] #loop-control,html[data-workbench-theme="dark"] #speed-control,html[data-workbench-theme="dark"] #secondary,html[data-workbench-theme="dark"] #perf{color:#cbd2d8}
  html[data-workbench-theme="dark"] #playback-speed,html[data-workbench-theme="dark"] #export-traces{background:#303740;border-color:#48535e;color:#e4e8ec}
  html[data-workbench-theme="dark"] #export-traces:hover:not(:disabled){background:#394550;border-color:#6589a4}
  html[data-workbench-theme="dark"] #export-traces:disabled,html[data-workbench-theme="dark"] .run:disabled{background:#292f35;border-color:#3b444d;color:#7d8791}
  html[data-workbench-theme="dark"] #error{background:#4a2f32;border-color:#b85c62;color:#f0b8bb}
</style>
</head>
<body>
<div id="app">
  <div id="title"></div>
  <div id="viewer-row">
    <canvas id="gl"></canvas>
    <canvas id="grid-overlay"></canvas>
    <div id="toolbar"><button id="home" class="tool">⌂ Home</button></div>
    <div id="gpu-status">WebGL2 GPU</div>
    <div id="colorbar"><span id="ctitle"></span><span id="cmax"></span><div id="gradient"></div><span id="cmin"></span></div>
    <div id="axis-hint">X = LR • Y = AP • Z = SI &nbsp; | &nbsp; Drag: orbit • right/middle drag: pan • wheel: zoom</div>
    <div id="export-time"></div>
    <div id="error"></div>
  </div>
  <div id="observation-wrap"><canvas id="obs"></canvas><div id="obs-stats"><label id="obs-invert-control"><input id="obs-invert" type="checkbox"> Invert</label><span id="obs-stats-text"></span></div></div>
  <div id="controls">
    <button id="play" class="run">▶ Play</button><button id="stop" class="run reset">↺ Reset</button>
    <label id="loop-control" title="Repeat the entire video timeline when it reaches the end"><input id="loop" type="checkbox"> Loop</label>
    <div id="time-area"><span>Video timeline</span><input id="time-slider" type="range"><span id="time-text"></span></div>
    <div id="speed-control"><div id="speed-stack"><div id="speed-row"><span>Playback speed</span><select id="playback-speed">
      <option value="0.10">0.10×</option><option value="0.25">0.25×</option>
      <option value="0.50">0.50×</option><option value="1" selected>1.00×</option>
      <option value="2">2.00×</option><option value="4">4.00×</option><option value="8">8.00×</option>
    </select></div></div><button id="export-traces" type="button">Export traces…</button></div>
    <div id="secondary">
      <span id="slice-label">Visible slices</span><input id="slice-slider" type="range"><span id="slice-count"></span>
      <span id="perf"></span>
    </div>
  </div>
</div>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(() => {
'use strict';
const P=JSON.parse(document.getElementById('payload').textContent);
const brainVideoTable=(P.brain_region_video_table&&P.brain_region_video_table.enabled)?P.brain_region_video_table:null;
const brainFrameMetricsPayload=(P.brain_region_frame_metrics&&typeof P.brain_region_frame_metrics==='object')?P.brain_region_frame_metrics:null;
let workbenchDark=String(P.workbench_theme||'light').toLowerCase()==='dark';
document.documentElement.setAttribute('data-workbench-theme',workbenchDark?'dark':'light');
const previewMode=P.export_preview===true;
const staticView=P.static_view===true;
if(previewMode)document.body.classList.add('export-preview');
if(staticView)document.body.classList.add('static-view');
const is2d=P.view_mode==='2d';
const renderMode=is2d?'slices':String(P.render_mode||'slices');
const isVolume=!is2d&&renderMode==='volume';
const isPointCloud=!is2d&&renderMode==='points';
const overlayPointCount=Math.max(0,Number(P.overlay_point_count)||0);
const hasOverlayFieldPoints=!isPointCloud&&overlayPointCount>0;
const showField=P.show_field!==false;
const showContours=is2d&&P.show_contours===true;
const contourCount=Math.max(1,Math.min(50,Number(P.contour_count)||10));
const contourLabels=showContours&&P.contour_labels===true;
if(!showField)document.getElementById('colorbar').style.display='none';
const canvas=document.getElementById('gl');
const gridCanvas=document.getElementById('grid-overlay');
const gridSpec=P.show_grid!==false&&P.grid&&P.grid.enabled!==false?P.grid:null;
const errorBox=document.getElementById('error');
const gl=canvas.getContext('webgl2',{alpha:false,antialias:true,premultipliedAlpha:false,preserveDrawingBuffer:previewMode,powerPreference:'high-performance'});
if(!gl){errorBox.style.display='block';errorBox.textContent='WebGL2 is unavailable. Check Qt WebEngine/GPU acceleration settings for this system.';document.getElementById('gpu-status').textContent='WebGL2 unavailable';return;}
document.getElementById('title').textContent=P.title||'Waveform playback';
document.getElementById('gpu-status').textContent=isVolume?'WebGL2 GPU • volume':(isPointCloud?'WebGL2 GPU • atlas points':'WebGL2 GPU');
const fmt=v=>{const a=Math.abs(Number(v));if(!Number.isFinite(a))return '—';if(a!==0&&(a>=1e4||a<1e-3))return Number(v).toExponential(3);return Number(v).toPrecision(5).replace(/\.?0+$/,'');};
document.getElementById('cmin').textContent=fmt(P.colour_min);
document.getElementById('cmax').textContent=fmt(P.colour_max);
document.getElementById('ctitle').textContent=(P.logarithmic?'log₁₀ ':'')+(P.component_label||'Field')+' ('+(P.unit||'')+')';
if(P.signed_component){document.getElementById('gradient').style.background='linear-gradient(to top,#2563eb 0%,#f8fafc 50%,#dc2626 100%)';}
function decodeB64(text,Type){const raw=atob(text||'');const bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Type(bytes.buffer);}
const brainMetricShape=brainFrameMetricsPayload&&Array.isArray(brainFrameMetricsPayload.shape)?brainFrameMetricsPayload.shape.map(Number):[];
const brainTableMode=String(brainVideoTable&&brainVideoTable.mode||'frame').toLowerCase();
const brainTableRows=brainVideoTable&&Array.isArray(brainVideoTable.rows)?brainVideoTable.rows:[];
const brainMetricKeys=brainFrameMetricsPayload&&Array.isArray(brainFrameMetricsPayload.keys)?brainFrameMetricsPayload.keys.map(String):brainTableRows.map(row=>String(row&&row.key||''));
const brainMetricValues=(brainVideoTable&&brainFrameMetricsPayload&&brainMetricShape.length===3)?decodeB64(brainFrameMetricsPayload.values_b64,Float32Array):null;
const brainSummaryValues=(brainVideoTable&&Array.isArray(brainVideoTable.summary_values))?brainVideoTable.summary_values:null;
const brainMetricSampleIndices=(brainVideoTable&&brainFrameMetricsPayload&&brainFrameMetricsPayload.sampled_frame_indices_b64)?decodeB64(brainFrameMetricsPayload.sampled_frame_indices_b64,Int32Array):null;
const brainMetricSourceFrameCount=Math.max(0,Number(brainFrameMetricsPayload&&brainFrameMetricsPayload.source_frame_count)||0);
const hasBrainSummary=!!(brainSummaryValues&&brainSummaryValues.length===brainMetricKeys.length);
const hasBrainVideoTable=!!(brainVideoTable&&((brainTableMode==='summary'&&hasBrainSummary)||(brainMetricValues&&brainMetricShape.length===3&&brainMetricShape[1]===brainMetricKeys.length)));
const brainMetricRowIndex=new Map(brainMetricKeys.map((key,index)=>[key,index]));
const brainVideoRows=hasBrainVideoTable&&Array.isArray(brainVideoTable.rows)?brainVideoTable.rows.map(row=>({...row,key:String(row.key||''),parent_key:row.parent_key===null?null:String(row.parent_key||''),name:String(row.name||''),side:String(row.side||'')})):[];
const brainVideoRowByKey=new Map(brainVideoRows.map(row=>[row.key,row]));
const brainVideoChildren=new Map();for(const row of brainVideoRows){const parent=row.parent_key===null?'__root__':row.parent_key;if(!brainVideoChildren.has(parent))brainVideoChildren.set(parent,[]);brainVideoChildren.get(parent).push(row);}
function shader(type,source){const sh=gl.createShader(type);gl.shaderSource(sh,source);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(sh)||'Shader compile failed');return sh;}
function program(vs,fs){const v=shader(gl.VERTEX_SHADER,vs),f=shader(gl.FRAGMENT_SHADER,fs),p=gl.createProgram();gl.attachShader(p,v);gl.attachShader(p,f);gl.linkProgram(p);gl.deleteShader(v);gl.deleteShader(f);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p)||'Shader link failed');return p;}
function transformProgram(vs,fs,varyings){const v=shader(gl.VERTEX_SHADER,vs),f=shader(gl.FRAGMENT_SHADER,fs),p=gl.createProgram();gl.attachShader(p,v);gl.attachShader(p,f);gl.transformFeedbackVaryings(p,varyings,gl.INTERLEAVED_ATTRIBS);gl.linkProgram(p);gl.deleteShader(v);gl.deleteShader(f);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p)||'Transform-feedback shader link failed');return p;}
let disposed=false;
const MAX_DRIVE_COILS=12;
const fieldProgram=program(`#version 300 es
precision highp float;
in vec3 aPosition;in vec3 aFixed;in vec3 aBasis0;in vec3 aBasis1;in vec3 aBasis2;in vec3 aBasis3;in vec3 aBasis4;in vec3 aBasis5;in vec3 aBasis6;in vec3 aBasis7;in vec3 aBasis8;in vec3 aBasis9;in vec3 aBasis10;in vec3 aBasis11;uniform mat4 uMVP;uniform float uCurrents[12];uniform int uDriveCount;uniform float uPointSize;out vec3 vField;
void main(){vField=aFixed;if(uDriveCount>0)vField+=uCurrents[0]*aBasis0;if(uDriveCount>1)vField+=uCurrents[1]*aBasis1;if(uDriveCount>2)vField+=uCurrents[2]*aBasis2;if(uDriveCount>3)vField+=uCurrents[3]*aBasis3;if(uDriveCount>4)vField+=uCurrents[4]*aBasis4;if(uDriveCount>5)vField+=uCurrents[5]*aBasis5;if(uDriveCount>6)vField+=uCurrents[6]*aBasis6;if(uDriveCount>7)vField+=uCurrents[7]*aBasis7;if(uDriveCount>8)vField+=uCurrents[8]*aBasis8;if(uDriveCount>9)vField+=uCurrents[9]*aBasis9;if(uDriveCount>10)vField+=uCurrents[10]*aBasis10;if(uDriveCount>11)vField+=uCurrents[11]*aBasis11;gl_Position=uMVP*vec4(aPosition,1.0);gl_PointSize=uPointSize;}`, `#version 300 es
precision highp float;in vec3 vField;uniform int uComponent;uniform float uScale;uniform float uMin;uniform float uMax;uniform bool uLog;uniform float uOpacityLow;uniform float uOpacityHigh;uniform float uOpacityLowLevel;uniform float uOpacityHighLevel;uniform bool uShowContours;uniform int uContourCount;uniform bool uPoint;out vec4 outColor;
vec3 cm(float t){t=clamp(t,0.0,1.0);vec3 c0=vec3(0.0314,0.1882,0.4196),c1=vec3(0.1451,0.3882,0.9216),c2=vec3(0.0235,0.7137,0.8314),c3=vec3(0.1333,0.7725,0.3686),c4=vec3(0.9804,0.8000,0.0824),c5=vec3(0.9765,0.4510,0.0863),c6=vec3(0.8627,0.1490,0.1490);if(t<.18)return mix(c0,c1,t/.18);if(t<.36)return mix(c1,c2,(t-.18)/.18);if(t<.54)return mix(c2,c3,(t-.36)/.18);if(t<.72)return mix(c3,c4,(t-.54)/.18);if(t<.86)return mix(c4,c5,(t-.72)/.14);return mix(c5,c6,(t-.86)/.14);}
vec3 cms(float t){t=clamp(t,0.0,1.0);vec3 neg=vec3(0.1451,0.3882,0.9216),zero=vec3(0.9725,0.9804,0.9882),pos=vec3(0.8627,0.1490,0.1490);return t<.5?mix(neg,zero,t*2.0):mix(zero,pos,(t-.5)*2.0);}
float cv(vec3 v){if(uComponent==0)return length(v);if(uComponent==1)return v.x;if(uComponent==2)return v.y;if(uComponent==3)return v.z;if(uComponent==4)return abs(v.x);if(uComponent==5)return abs(v.y);return abs(v.z);}
float opacityForStrength(float strength){float lo=clamp(uOpacityLowLevel,0.0,1.0),hi=clamp(uOpacityHighLevel,0.0,1.0);if(strength<=lo)return uOpacityLow;if(strength>=hi)return uOpacityHigh;if(hi<=lo)return uOpacityHigh;return mix(uOpacityLow,uOpacityHigh,(strength-lo)/(hi-lo));}
void main(){if(uPoint){vec2 q=gl_PointCoord*2.0-1.0;if(dot(q,q)>1.0)discard;}float s=cv(vField)*uScale;bool signedComp=uComponent>=1&&uComponent<=3;if(uLog)s=log(max(s,1e-30))/log(10.0);float t=clamp((s-uMin)/max(1e-30,uMax-uMin),0.0,1.0);float strength=signedComp?clamp(abs(s)/max(1e-30,max(abs(uMin),abs(uMax))),0.0,1.0):t;float a=opacityForStrength(strength);vec3 colour=signedComp?cms(t):cm(t);if(uShowContours&&uContourCount>0&&t>0.0005&&t<0.9995){float bands=float(uContourCount+1);float phase=fract(t*bands);float distanceToLine=min(phase,1.0-phase);float aa=max(fwidth(t*bands)*1.15,0.006);float line=1.0-smoothstep(aa,aa*1.8,distanceToLine);colour=mix(colour,vec3(0.04,0.05,0.06),0.82*line);}outColor=vec4(colour,a);}`);
const solidProgram=program(`#version 300 es
precision highp float;in vec3 aPosition;uniform mat4 uMVP;out vec3 vWorld;void main(){vWorld=aPosition;gl_Position=uMVP*vec4(aPosition,1.0);}`, `#version 300 es
precision highp float;in vec3 vWorld;uniform vec4 uColour;uniform bool uPoint;out vec4 outColor;void main(){if(uPoint){vec2 q=gl_PointCoord*2.0-1.0;if(dot(q,q)>1.0)discard;}vec3 n=normalize(cross(dFdx(vWorld),dFdy(vWorld)));float light=.58+.42*abs(dot(n,normalize(vec3(.35,.55,.76))));outColor=vec4(uColour.rgb*light,uColour.a);}`);
const lineProgram=program(`#version 300 es
precision highp float;in vec3 aPosition;uniform mat4 uMVP;uniform float uPointSize;void main(){gl_Position=uMVP*vec4(aPosition,1.0);gl_PointSize=uPointSize;}`, `#version 300 es
precision highp float;uniform vec4 uColour;uniform bool uPoint;out vec4 outColor;void main(){if(uPoint){vec2 q=gl_PointCoord*2.0-1.0;if(dot(q,q)>1.0)discard;}outColor=uColour;}`);
const volumeProgram=isVolume?program(`#version 300 es
precision highp float;const vec2 P[3]=vec2[3](vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));out vec2 vUV;void main(){vec2 p=P[gl_VertexID];vUV=p*.5+.5;gl_Position=vec4(p,0.0,1.0);}`, `#version 300 es
precision highp float;precision highp sampler3D;in vec2 vUV;uniform mat4 uInvVP;uniform vec3 uCamera;uniform vec3 uBoxMin;uniform vec3 uBoxMax;uniform sampler3D uFixedTex;uniform sampler3D uBasisTex0;uniform sampler3D uBasisTex1;uniform sampler3D uBasisTex2;uniform sampler3D uBasisTex3;uniform sampler3D uBasisTex4;uniform sampler3D uBasisTex5;uniform sampler3D uBasisTex6;uniform sampler3D uBasisTex7;uniform sampler3D uBasisTex8;uniform sampler3D uBasisTex9;uniform sampler3D uBasisTex10;uniform sampler3D uBasisTex11;uniform float uCurrents[12];uniform int uDriveCount;uniform int uComponent;uniform float uScale;uniform float uMin;uniform float uMax;uniform bool uLog;uniform float uOpacityLow;uniform float uOpacityHigh;uniform float uOpacityLowLevel;uniform float uOpacityHighLevel;uniform int uSurfaceCount;uniform int uSteps;out vec4 outColor;
vec3 cm(float t){t=clamp(t,0.0,1.0);vec3 c0=vec3(0.0314,0.1882,0.4196),c1=vec3(0.1451,0.3882,0.9216),c2=vec3(0.0235,0.7137,0.8314),c3=vec3(0.1333,0.7725,0.3686),c4=vec3(0.9804,0.8000,0.0824),c5=vec3(0.9765,0.4510,0.0863),c6=vec3(0.8627,0.1490,0.1490);if(t<.18)return mix(c0,c1,t/.18);if(t<.36)return mix(c1,c2,(t-.18)/.18);if(t<.54)return mix(c2,c3,(t-.36)/.18);if(t<.72)return mix(c3,c4,(t-.54)/.18);if(t<.86)return mix(c4,c5,(t-.72)/.14);return mix(c5,c6,(t-.86)/.14);}
vec3 cms(float t){t=clamp(t,0.0,1.0);vec3 neg=vec3(0.1451,0.3882,0.9216),zero=vec3(0.9725,0.9804,0.9882),pos=vec3(0.8627,0.1490,0.1490);return t<.5?mix(neg,zero,t*2.0):mix(zero,pos,(t-.5)*2.0);}
float cv(vec3 v){if(uComponent==0)return length(v);if(uComponent==1)return v.x;if(uComponent==2)return v.y;if(uComponent==3)return v.z;if(uComponent==4)return abs(v.x);if(uComponent==5)return abs(v.y);return abs(v.z);}
vec2 hitBox(vec3 ro,vec3 rd){vec3 inv=1.0/rd;vec3 a=(uBoxMin-ro)*inv,b=(uBoxMax-ro)*inv;vec3 lo=min(a,b),hi=max(a,b);return vec2(max(max(lo.x,lo.y),lo.z),min(min(hi.x,hi.y),hi.z));}
vec3 latticeTc(vec3 u){vec3 dims=vec3(textureSize(uFixedTex,0));return (clamp(u,vec3(0.0),vec3(1.0))*(dims-vec3(1.0))+vec3(0.5))/dims;}
float opacityForStrength(float strength){float lo=clamp(uOpacityLowLevel,0.0,1.0),hi=clamp(uOpacityHighLevel,0.0,1.0);if(strength<=lo)return uOpacityLow;if(strength>=hi)return uOpacityHigh;if(hi<=lo)return uOpacityHigh;return mix(uOpacityLow,uOpacityHigh,(strength-lo)/(hi-lo));}
void main(){vec2 ndc=vUV*2.0-1.0;vec4 n4=uInvVP*vec4(ndc,-1.0,1.0),f4=uInvVP*vec4(ndc,1.0,1.0);vec3 nearP=n4.xyz/n4.w,farP=f4.xyz/f4.w;vec3 ro=uCamera,rd=normalize(farP-nearP);vec2 h=hitBox(ro,rd);float t0=max(h.x,0.0),t1=h.y;if(t1<=t0){outColor=vec4(1.0);return;}float dt=(t1-t0)/float(max(uSteps,1));float surfaceDenom=float(max(uSurfaceCount-1,1));float bestStrength=-1.0,bestQ=0.0,bestAlpha=0.0;for(int i=0;i<128;i++){if(i>=uSteps)break;float d=t0+(float(i)+.5)*dt;vec3 wp=ro+rd*d;vec3 tc=latticeTc((wp-uBoxMin)/max(uBoxMax-uBoxMin,vec3(1e-12)));vec3 vf=texture(uFixedTex,tc).xyz;if(uDriveCount>0)vf+=uCurrents[0]*texture(uBasisTex0,tc).xyz;if(uDriveCount>1)vf+=uCurrents[1]*texture(uBasisTex1,tc).xyz;if(uDriveCount>2)vf+=uCurrents[2]*texture(uBasisTex2,tc).xyz;if(uDriveCount>3)vf+=uCurrents[3]*texture(uBasisTex3,tc).xyz;if(uDriveCount>4)vf+=uCurrents[4]*texture(uBasisTex4,tc).xyz;if(uDriveCount>5)vf+=uCurrents[5]*texture(uBasisTex5,tc).xyz;if(uDriveCount>6)vf+=uCurrents[6]*texture(uBasisTex6,tc).xyz;if(uDriveCount>7)vf+=uCurrents[7]*texture(uBasisTex7,tc).xyz;if(uDriveCount>8)vf+=uCurrents[8]*texture(uBasisTex8,tc).xyz;if(uDriveCount>9)vf+=uCurrents[9]*texture(uBasisTex9,tc).xyz;if(uDriveCount>10)vf+=uCurrents[10]*texture(uBasisTex10,tc).xyz;if(uDriveCount>11)vf+=uCurrents[11]*texture(uBasisTex11,tc).xyz;float s=cv(vf)*uScale;bool signedComp=uComponent>=1&&uComponent<=3;if(uLog)s=log(max(s,1e-30))/log(10.0);if(s<uMin||s>uMax)continue;float q=clamp((s-uMin)/max(1e-30,uMax-uMin),0.0,1.0);float surfaceQ=floor(q*surfaceDenom+0.5)/surfaceDenom;float surfaceS=mix(uMin,uMax,surfaceQ);float strength=signedComp?clamp(abs(surfaceS)/max(1e-30,max(abs(uMin),abs(uMax))),0.0,1.0):surfaceQ;float a=opacityForStrength(strength);if(a<=0.0001)continue;if(strength>bestStrength){bestStrength=strength;bestQ=surfaceQ;bestAlpha=a;}}if(bestStrength<0.0){outColor=vec4(1.0);return;}vec3 c=(uComponent>=1&&uComponent<=3)?cms(bestQ):cm(bestQ);outColor=vec4(mix(vec3(1.0),c,bestAlpha),1.0);}`):null;
const fieldLoc={pos:gl.getAttribLocation(fieldProgram,'aPosition'),fixed:gl.getAttribLocation(fieldProgram,'aFixed'),bases:Array.from({length:MAX_DRIVE_COILS},(_,i)=>gl.getAttribLocation(fieldProgram,'aBasis'+i)),mvp:gl.getUniformLocation(fieldProgram,'uMVP'),currents:gl.getUniformLocation(fieldProgram,'uCurrents[0]'),driveCount:gl.getUniformLocation(fieldProgram,'uDriveCount'),component:gl.getUniformLocation(fieldProgram,'uComponent'),scale:gl.getUniformLocation(fieldProgram,'uScale'),min:gl.getUniformLocation(fieldProgram,'uMin'),max:gl.getUniformLocation(fieldProgram,'uMax'),log:gl.getUniformLocation(fieldProgram,'uLog'),opacityLow:gl.getUniformLocation(fieldProgram,'uOpacityLow'),opacityHigh:gl.getUniformLocation(fieldProgram,'uOpacityHigh'),opacityLowLevel:gl.getUniformLocation(fieldProgram,'uOpacityLowLevel'),opacityHighLevel:gl.getUniformLocation(fieldProgram,'uOpacityHighLevel'),showContours:gl.getUniformLocation(fieldProgram,'uShowContours'),contourCount:gl.getUniformLocation(fieldProgram,'uContourCount'),point:gl.getUniformLocation(fieldProgram,'uPoint'),pointSize:gl.getUniformLocation(fieldProgram,'uPointSize')};
const solidLoc={pos:gl.getAttribLocation(solidProgram,'aPosition'),mvp:gl.getUniformLocation(solidProgram,'uMVP'),colour:gl.getUniformLocation(solidProgram,'uColour'),point:gl.getUniformLocation(solidProgram,'uPoint')};
const lineLoc={pos:gl.getAttribLocation(lineProgram,'aPosition'),mvp:gl.getUniformLocation(lineProgram,'uMVP'),colour:gl.getUniformLocation(lineProgram,'uColour'),point:gl.getUniformLocation(lineProgram,'uPoint'),pointSize:gl.getUniformLocation(lineProgram,'uPointSize')};
const volumeLoc=isVolume?{invVP:gl.getUniformLocation(volumeProgram,'uInvVP'),camera:gl.getUniformLocation(volumeProgram,'uCamera'),boxMin:gl.getUniformLocation(volumeProgram,'uBoxMin'),boxMax:gl.getUniformLocation(volumeProgram,'uBoxMax'),fixedTex:gl.getUniformLocation(volumeProgram,'uFixedTex'),basisTex:Array.from({length:MAX_DRIVE_COILS},(_,i)=>gl.getUniformLocation(volumeProgram,'uBasisTex'+i)),currents:gl.getUniformLocation(volumeProgram,'uCurrents[0]'),driveCount:gl.getUniformLocation(volumeProgram,'uDriveCount'),component:gl.getUniformLocation(volumeProgram,'uComponent'),scale:gl.getUniformLocation(volumeProgram,'uScale'),min:gl.getUniformLocation(volumeProgram,'uMin'),max:gl.getUniformLocation(volumeProgram,'uMax'),log:gl.getUniformLocation(volumeProgram,'uLog'),opacityLow:gl.getUniformLocation(volumeProgram,'uOpacityLow'),opacityHigh:gl.getUniformLocation(volumeProgram,'uOpacityHigh'),opacityLowLevel:gl.getUniformLocation(volumeProgram,'uOpacityLowLevel'),opacityHighLevel:gl.getUniformLocation(volumeProgram,'uOpacityHighLevel'),surfaceCount:gl.getUniformLocation(volumeProgram,'uSurfaceCount'),steps:gl.getUniformLocation(volumeProgram,'uSteps')}:null;
function buffer(data,target=gl.ARRAY_BUFFER){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b;}
const R=Number(P.resolution),L=Number(P.slice_count),vertsPerSlice=R*R;
let posBuf=null,fixedBuf=null,basisBufs=[],idxBuf=null,gridIndices=new Uint32Array();
let overlayPosBuf=null,overlayFixedBuf=null,overlayBasisBufs=[],overlayPositionsData=null,overlayLabelsData=null,brainHighlightBuf=null,brainHighlightCount=0;
let planarPositions=null,planarFixed=null,planarBases=[];
if(!isVolume){const positions=decodeB64(P.positions,Float32Array),fixed=decodeB64(P.fixed_vectors,Float32Array),basisPayload=(P.basis_vectors_b64&&P.basis_vectors_b64.length?P.basis_vectors_b64:[P.basis_vectors]),decodedBases=basisPayload.slice(0,MAX_DRIVE_COILS).map(value=>decodeB64(value,Float32Array));posBuf=buffer(positions);fixedBuf=buffer(fixed);basisBufs=decodedBases.map(value=>buffer(value));if(is2d&&contourLabels){planarPositions=positions;planarFixed=fixed;planarBases=decodedBases;}if(!isPointCloud){const idx=[];for(let y=0;y<R-1;y++)for(let x=0;x<R-1;x++){const a=y*R+x,b=a+1,c=a+R,d=c+1;idx.push(a,c,b,b,c,d);}gridIndices=new Uint32Array(idx);idxBuf=buffer(gridIndices,gl.ELEMENT_ARRAY_BUFFER);}}
if(hasOverlayFieldPoints){const overlayPositions=decodeB64(P.overlay_positions,Float32Array),overlayFixed=decodeB64(P.overlay_fixed_vectors,Float32Array),overlayBasisPayload=(P.overlay_basis_vectors_b64&&P.overlay_basis_vectors_b64.length?P.overlay_basis_vectors_b64:[P.overlay_basis_vectors]),decodedOverlayBases=overlayBasisPayload.slice(0,MAX_DRIVE_COILS).map(value=>decodeB64(value,Float32Array));overlayPositionsData=overlayPositions;overlayLabelsData=decodeB64(P.overlay_point_label_ids_b64||'',Int32Array);overlayPosBuf=buffer(overlayPositions);overlayFixedBuf=buffer(overlayFixed);overlayBasisBufs=decodedOverlayBases.map(value=>buffer(value));}
let volumeTextureFilter='unknown';
function texture3d(data){const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_3D,tex);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);const linear=!!gl.getExtension('OES_texture_float_linear');volumeTextureFilter=linear?'LINEAR':'NEAREST';gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,linear?gl.LINEAR:gl.NEAREST);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,linear?gl.LINEAR:gl.NEAREST);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);while(gl.getError()!==gl.NO_ERROR){}gl.texImage3D(gl.TEXTURE_3D,0,gl.RGB32F,R,R,R,0,gl.RGB,gl.FLOAT,data);if(gl.getError()===gl.NO_ERROR)return tex;const rgba=new Float32Array((data.length/3)*4);for(let i=0,j=0;i<data.length;i+=3,j+=4){rgba[j]=data[i];rgba[j+1]=data[i+1];rgba[j+2]=data[i+2];rgba[j+3]=0;}gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA32F,R,R,R,0,gl.RGBA,gl.FLOAT,rgba);if(gl.getError()!==gl.NO_ERROR)throw new Error('Unable to allocate the 3D floating-point field texture.');return tex;}
let volumeFixedTex=null,volumeBasisTex=[],volumeFixedData=null,volumeBasisData=[];
if(isVolume){volumeFixedData=decodeB64(P.volume_fixed_vectors,Float32Array);volumeFixedTex=texture3d(volumeFixedData);const basisPayload=(P.volume_basis_vectors_b64&&P.volume_basis_vectors_b64.length?P.volume_basis_vectors_b64:[P.volume_basis_vectors]);volumeBasisData=basisPayload.slice(0,MAX_DRIVE_COILS).map(value=>decodeB64(value,Float32Array));volumeBasisTex=volumeBasisData.map(value=>texture3d(value));}
const sceneMeshes=(P.scene?.meshes||[]).map(m=>({...m,pbuf:buffer(decodeB64(m.positions,Float32Array)),ibuf:buffer(decodeB64(m.indices,Uint32Array),gl.ELEMENT_ARRAY_BUFFER)}));
const sceneLines=(P.scene?.lines||[]).map(m=>({...m,pbuf:buffer(decodeB64(m.positions,Float32Array))}));
const scenePoints=(P.scene?.points||[]).map(m=>({...m,pbuf:buffer(decodeB64(m.positions,Float32Array))}));
const gridBuffer=gridSpec?gl.createBuffer():null;
function mat4Perspective(fovy,aspect,near,far){const f=1/Math.tan(fovy/2),nf=1/(near-far);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]);}
function mat4Ortho(left,right,bottom,top,near,far){const lr=1/(left-right),bt=1/(bottom-top),nf=1/(near-far);return new Float32Array([-2*lr,0,0,0,0,-2*bt,0,0,0,0,2*nf,0,(left+right)*lr,(top+bottom)*bt,(far+near)*nf,1]);}
function vsub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}function vadd(a,b){return[a[0]+b[0],a[1]+b[1],a[2]+b[2]]}function vmul(a,s){return[a[0]*s,a[1]*s,a[2]*s]}function vdot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function vcross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function vnorm(a){const n=Math.hypot(...a)||1;return vmul(a,1/n)}
function lookAt(eye,target,up){const z=vnorm(vsub(eye,target)),x=vnorm(vcross(up,z)),y=vcross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-vdot(x,eye),-vdot(y,eye),-vdot(z,eye),1]);}
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)o[c*4+r]=a[0*4+r]*b[c*4+0]+a[1*4+r]*b[c*4+1]+a[2*4+r]*b[c*4+2]+a[3*4+r]*b[c*4+3];return o;}
function inverse(m){const o=new Float32Array(16),a00=m[0],a01=m[1],a02=m[2],a03=m[3],a10=m[4],a11=m[5],a12=m[6],a13=m[7],a20=m[8],a21=m[9],a22=m[10],a23=m[11],a30=m[12],a31=m[13],a32=m[14],a33=m[15],b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10,b03=a01*a12-a02*a11,b04=a01*a13-a03*a11,b05=a02*a13-a03*a12,b06=a20*a31-a21*a30,b07=a20*a32-a22*a30,b08=a20*a33-a23*a30,b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;if(!det)return null;det=1/det;o[0]=(a11*b11-a12*b10+a13*b09)*det;o[1]=(a02*b10-a01*b11-a03*b09)*det;o[2]=(a31*b05-a32*b04+a33*b03)*det;o[3]=(a22*b04-a21*b05-a23*b03)*det;o[4]=(a12*b08-a10*b11-a13*b07)*det;o[5]=(a00*b11-a02*b08+a03*b07)*det;o[6]=(a32*b02-a30*b05-a33*b01)*det;o[7]=(a20*b05-a22*b02+a23*b01)*det;o[8]=(a10*b10-a11*b08+a13*b06)*det;o[9]=(a01*b08-a00*b10-a03*b06)*det;o[10]=(a30*b04-a31*b02+a33*b00)*det;o[11]=(a21*b02-a20*b04-a23*b00)*det;o[12]=(a11*b07-a10*b09-a12*b06)*det;o[13]=(a00*b09-a01*b07+a02*b06)*det;o[14]=(a31*b01-a30*b03-a32*b00)*det;o[15]=(a20*b03-a21*b01+a22*b00)*det;return o;}
const centre=(P.centre_mm||[0,0,0]).map(Number),half=Number(P.span_mm)/2;
let bmin=[centre[0]-half,centre[1]-half,centre[2]-half],bmax=[centre[0]+half,centre[1]+half,centre[2]+half];if(!is2d&&P.scene?.bounds_min&&P.scene?.bounds_max){for(let i=0;i<3;i++){bmin[i]=Math.min(bmin[i],Number(P.scene.bounds_min[i]));bmax[i]=Math.max(bmax[i],Number(P.scene.bounds_max[i]));}}
const fitTarget=is2d?centre.slice():[(bmin[0]+bmax[0])/2,(bmin[1]+bmax[1])/2,(bmin[2]+bmax[2])/2],fitRadius=Math.max(1,is2d?Number(P.span_mm)/2:Math.hypot(bmax[0]-bmin[0],bmax[1]-bmin[1],bmax[2]-bmin[2])/2);
const frameTimes=decodeB64(P.frame_times_b64,Float64Array),frameCurrents=decodeB64(P.frame_currents_b64,Float32Array),freeFrameCurrents=decodeB64(P.free_frame_currents_b64||P.frame_currents_b64,Float32Array),driveCount=Math.max(1,Math.min(MAX_DRIVE_COILS,Number(P.drive_coil_count)||1)),N=Math.min(Number(P.frame_count)||frameTimes.length,frameTimes.length,Math.floor(frameCurrents.length/driveCount),Math.floor(freeFrameCurrents.length/driveCount));
const sequenceConfig=P.playback_sequence&&Array.isArray(P.playback_sequence.steps)&&P.playback_sequence.steps.length?P.playback_sequence:null;
let currents=new Float32Array(MAX_DRIVE_COILS);for(let j=0;j<driveCount;j++)currents[j]=N?Number(frameCurrents[j]):0;
let target=fitTarget.slice(),yaw=-Math.PI/4,pitch=.42,distance=fitRadius*2.7,zoom2d=1,visibleSlices=Math.max(1,L),dirty=true,fovY=Math.PI/4,captureSize=null;
const opacityEnabled=P.intensity_opacity_enabled!==false,initialOpacityLow=Number(P.volume_opacity_lower),initialOpacityHigh=Number(P.volume_opacity_upper),initialOpacityLowLevel=Number(P.volume_opacity_lower_level),initialOpacityHighLevel=Number(P.volume_opacity_upper_level);let volumeOpacityLow=Math.max(0,Math.min(1,Number.isFinite(initialOpacityLow)?initialOpacityLow:0.15)),volumeOpacityHigh=Math.max(0,Math.min(1,Number.isFinite(initialOpacityHigh)?initialOpacityHigh:0.40)),volumeOpacityLowLevel=Math.max(0,Math.min(1,Number.isFinite(initialOpacityLowLevel)?initialOpacityLowLevel:0.0)),volumeOpacityHighLevel=Math.max(0,Math.min(1,Number.isFinite(initialOpacityHighLevel)?initialOpacityHighLevel:1.0));
const plane=P.plane||'xy';const view2d={xy:{eye:[0,0,1],up:[0,1,0],right:[1,0,0],screenUp:[0,1,0]},xz:{eye:[0,-1,0],up:[0,0,1],right:[1,0,0],screenUp:[0,0,1]},yz:{eye:[1,0,0],up:[0,0,1],right:[0,1,0],screenUp:[0,0,1]}}[plane]||{eye:[0,0,1],up:[0,1,0],right:[1,0,0],screenUp:[0,1,0]};
function eyePosition(){if(is2d)return vadd(target,vmul(view2d.eye,Math.max(1,Number(P.span_mm)*2)));const cp=Math.cos(pitch);return[target[0]+distance*cp*Math.cos(yaw),target[1]+distance*cp*Math.sin(yaw),target[2]+distance*Math.sin(pitch)];}
function cameraState(){const state={position:eyePosition().map(Number),target:target.map(Number),fov_deg:fovY*180/Math.PI};if(is2d)state.zoom_2d=zoom2d;return state;}
function applyCamera(raw){if(!raw||!Array.isArray(raw.position)||!Array.isArray(raw.target))return false;const eye=raw.position.map(Number),nextTarget=raw.target.map(Number),fov=Number(raw.fov_deg??45);if(eye.length!==3||nextTarget.length!==3||![...eye,...nextTarget,fov].every(Number.isFinite))return false;const delta=vsub(eye,nextTarget),nextDistance=Math.hypot(...delta);if(nextDistance<=1e-6)return false;target=nextTarget;distance=nextDistance;yaw=Math.atan2(delta[1],delta[0]);pitch=Math.max(-1.48,Math.min(1.48,Math.atan2(delta[2],Math.hypot(delta[0],delta[1]))));fovY=Math.max(10,Math.min(120,fov))*Math.PI/180;if(is2d&&Number.isFinite(Number(raw.zoom_2d)))zoom2d=Math.max(.01,Math.min(100,Number(raw.zoom_2d)));dirty=true;return true;}
window.fieldWorkbenchSetCamera=raw=>{const ok=applyCamera(raw);if(ok)render();return ok;};
window.fieldWorkbenchGetCamera=()=>cameraState();
if(P.initial_camera)applyCamera(P.initial_camera);
const cameraTimeline=P.camera_timeline&&P.camera_timeline.enabled===true&&Array.isArray(P.camera_timeline.points)&&P.camera_timeline.points.length?P.camera_timeline:null;
const scriptedTimeline=!!(cameraTimeline&&cameraTimeline.scripted===true);
function timelineHermite(p0,p1,p2,p3,t0,t1,t2,t3,t,hasPrev,hasNext){const span=t2-t1;if(!(span>1e-12))return Number(p1);const u=Math.max(0,Math.min(1,(t-t1)/span)),u2=u*u,u3=u2*u,m1=hasPrev&&t2>t0+1e-12?(p2-p0)/(t2-t0):(p2-p1)/span,m2=hasNext&&t3>t1+1e-12?(p3-p1)/(t3-t1):(p2-p1)/span;return(2*u3-3*u2+1)*p1+(u3-2*u2+u)*span*m1+(-2*u3+3*u2)*p2+(u3-u2)*span*m2;}
function timelineCameraAt(timeS){if(!cameraTimeline)return null;const limit=Number(cameraTimeline.duration_s),all=cameraTimeline.points,points=Number.isFinite(limit)?all.filter(point=>(Number(point.time_s)||0)<=limit+1e-12):all;if(!points.length)return null;const t=Math.max(0,Number(timeS)||0);let li=0,ri=0;if(points.length>1&&t>(Number(points[0].time_s)||0)){if(t>=(Number(points[points.length-1].time_s)||0)){li=ri=points.length-1;}else{ri=1;while(ri<points.length&&(Number(points[ri].time_s)||0)<t)ri++;li=Math.max(0,ri-1);}}const left=points[li],right=points[ri],a=left.camera||{},b=right.camera||a;if(li===ri){const result={position:(a.position||[0,0,1]).map(Number),target:(a.target||[0,0,0]).map(Number),fov_deg:45};if(is2d)result.zoom_2d=Math.max(.01,Number(a.zoom_2d??1));return result;}const pi=Math.max(0,li-1),ni=Math.min(points.length-1,ri+1),prev=points[pi],next=points[ni],p=prev.camera||a,n=next.camera||b,t0=Number(prev.time_s)||0,t1=Number(left.time_s)||0,t2=Number(right.time_s)||t1,t3=Number(next.time_s)||t2,interp=(p0,p1,p2,p3)=>timelineHermite(Number(p0),Number(p1),Number(p2),Number(p3),t0,t1,t2,t3,t,li>0,ri+1<points.length),vec=(key,fallback)=>{const v0=p[key]||fallback,v1=a[key]||fallback,v2=b[key]||v1,v3=n[key]||v2;if(fallback.every((_,i)=>Math.abs(Number(v2[i])-Number(v1[i]))<=1e-12))return v1.map(Number);return fallback.map((_,i)=>interp(v0[i],v1[i],v2[i],v3[i]));};const result={position:vec('position',[0,0,1]),target:vec('target',[0,0,0]),fov_deg:45};if(is2d)result.zoom_2d=Math.max(.01,interp(p.zoom_2d??1,a.zoom_2d??1,b.zoom_2d??a.zoom_2d??1,n.zoom_2d??b.zoom_2d??1));return result;}
function timelinePlaybackSpeedAt(timeS){if(!cameraTimeline)return 1;const t=Math.max(0,Number(timeS)||0);let speed=1;for(const point of cameraTimeline.points){if((Number(point.time_s)||0)<=t+1e-12)speed=Math.max(1e-6,Number(point.playback_speed)||1);else break;}return speed;}
function applyTimelineCamera(timeS){const camera=timelineCameraAt(timeS);if(camera)applyCamera(camera);}
function mvp(){const viewWidth=captureSize?canvas.width:canvas.clientWidth,viewHeight=captureSize?canvas.height:canvas.clientHeight,aspect=Math.max(.1,viewWidth/Math.max(1,viewHeight)),eye=eyePosition();if(is2d){const h=half*zoom2d,hh=aspect>=1?h:h/aspect,hw=aspect>=1?h*aspect:h,proj=mat4Ortho(-hw,hw,-hh,hh,-Math.max(1000,Number(P.span_mm)*10),Math.max(1000,Number(P.span_mm)*10));return mul(proj,lookAt(eye,target,view2d.up));}const near=Math.max(.01,distance/10000),far=Math.max(1000,distance+fitRadius*12),proj=mat4Perspective(fovY,aspect,near,far);return mul(proj,lookAt(eye,target,[0,0,1]));}
function gridBounds(){if(!gridSpec)return null;const lo=(gridSpec.box_min_mm||[]).map(Number),hi=(gridSpec.box_max_mm||[]).map(Number);return lo.length===3&&hi.length===3&&[...lo,...hi].every(Number.isFinite)?{lo,hi}:null;}
function pushSegment(target,a,b){target.push(a[0],a[1],a[2],b[0],b[1],b[2]);}
function gridPlaneVertices(){
  const bounds=gridBounds();if(!bounds)return new Float32Array();
  if(is2d&&Array.isArray(gridSpec.plane_segments_mm)){const out=[];for(const segment of gridSpec.plane_segments_mm){if(!Array.isArray(segment)||segment.length!==2)continue;const a=segment[0],b=segment[1];if(!Array.isArray(a)||!Array.isArray(b)||a.length!==3||b.length!==3)continue;pushSegment(out,a.map(Number),b.map(Number));}return new Float32Array(out);}
  const {lo,hi}=bounds,eye=eyePosition(),far=[eye[0]>=centre[0]?lo[0]:hi[0],eye[1]>=centre[1]?lo[1]:hi[1],eye[2]>=centre[2]?lo[2]:hi[2]],axes=Array.isArray(gridSpec.axes)?gridSpec.axes:[],out=[];
  const ticks=index=>{const values=Array.isArray(axes[index]?.tick_values_mm)?axes[index].tick_values_mm.map(Number):[];const epsilon=Math.max(1e-9,Math.abs(hi[index]-lo[index])*1e-9);return values.filter(value=>Number.isFinite(value)&&value>lo[index]+epsilon&&value<hi[index]-epsilon);};
  for(const y of ticks(1))pushSegment(out,[far[0],y,lo[2]],[far[0],y,hi[2]]);
  for(const z of ticks(2))pushSegment(out,[far[0],lo[1],z],[far[0],hi[1],z]);
  for(const x of ticks(0))pushSegment(out,[x,far[1],lo[2]],[x,far[1],hi[2]]);
  for(const z of ticks(2))pushSegment(out,[lo[0],far[1],z],[hi[0],far[1],z]);
  for(const x of ticks(0))pushSegment(out,[x,lo[1],far[2]],[x,hi[1],far[2]]);
  for(const y of ticks(1))pushSegment(out,[lo[0],y,far[2]],[hi[0],y,far[2]]);
  return new Float32Array(out);
}
function drawGrid(M){
  if(!gridBuffer)return;
  const vertices=gridPlaneVertices();if(!vertices.length)return;
  gl.useProgram(lineProgram);gl.uniformMatrix4fv(lineLoc.mvp,false,M);gl.bindBuffer(gl.ARRAY_BUFFER,gridBuffer);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(lineLoc.pos);gl.vertexAttribPointer(lineLoc.pos,3,gl.FLOAT,false,0,0);gl.uniform4f(lineLoc.colour,.72,.77,.83,.62);gl.uniform1i(lineLoc.point,0);gl.uniform1f(lineLoc.pointSize,1);gl.depthMask(false);try{gl.lineWidth(1);}catch(_){}gl.drawArrays(gl.LINES,0,vertices.length/3);gl.depthMask(true);
}
function projectGridPoint(M,point,width,height){const x=point[0],y=point[1],z=point[2],cx=M[0]*x+M[4]*y+M[8]*z+M[12],cy=M[1]*x+M[5]*y+M[9]*z+M[13],cw=M[3]*x+M[7]*y+M[11]*z+M[15];if(!Number.isFinite(cw)||Math.abs(cw)<1e-12)return null;return[(cx/cw*.5+.5)*width,(1-(cy/cw*.5+.5))*height];}
function gridAxisPoint(axis,value,bounds){const eye=eyePosition(),nearX=eye[0]>=centre[0]?bounds.hi[0]:bounds.lo[0],nearY=eye[1]>=centre[1]?bounds.hi[1]:bounds.lo[1],floorZ=eye[2]>=centre[2]?bounds.lo[2]:bounds.hi[2];if(axis===0)return[value,nearY,floorZ];if(axis===1)return[nearX,value,floorZ];return[nearX,nearY,value];}
let contourLabelCacheKey='',contourLabelWorld=[];
function contourScalarAt(index){if(!planarFixed)return NaN;const o=index*3;let vx=Number(planarFixed[o]),vy=Number(planarFixed[o+1]),vz=Number(planarFixed[o+2]);for(let j=0;j<Math.min(driveCount,planarBases.length);j++){const basis=planarBases[j],current=Number(currents[j])||0;vx+=current*Number(basis[o]);vy+=current*Number(basis[o+1]);vz+=current*Number(basis[o+2]);}let value;if(P.component==='x')value=vx;else if(P.component==='y')value=vy;else if(P.component==='z')value=vz;else if(P.component==='abs_x')value=Math.abs(vx);else if(P.component==='abs_y')value=Math.abs(vy);else if(P.component==='abs_z')value=Math.abs(vz);else value=Math.hypot(vx,vy,vz);value*=P.field==='B'?1e6:1;if(P.logarithmic)value=Math.log10(Math.max(value,1e-30));return value;}
function updateContourLabelWorld(){if(!contourLabels||!planarPositions||!planarFixed||R<2){contourLabelWorld=[];return;}const key=Array.from(currents.slice(0,driveCount)).map(value=>Number(value).toPrecision(8)).join(',');if(key===contourLabelCacheKey)return;contourLabelCacheKey=key;const total=R*R,values=new Float64Array(total);for(let i=0;i<total;i++)values[i]=contourScalarAt(i);const rows=Array.from(new Set([Math.floor((R-1)/2),Math.floor((R-1)/3),Math.floor(2*(R-1)/3),0,R-1])).filter(v=>v>=0&&v<R),cols=rows,low=Number(P.colour_min),high=Number(P.colour_max),found=[];function crossing(aIndex,bIndex,target){const a=values[aIndex],b=values[bIndex];if(!Number.isFinite(a)||!Number.isFinite(b)||Math.abs(b-a)<1e-30||((target<a&&target<b)||(target>a&&target>b)))return null;const q=Math.max(0,Math.min(1,(target-a)/(b-a))),ao=aIndex*3,bo=bIndex*3;return[0,1,2].map(k=>Number(planarPositions[ao+k])+(Number(planarPositions[bo+k])-Number(planarPositions[ao+k]))*q);}for(let level=1;level<=contourCount;level++){const targetValue=low+(high-low)*level/(contourCount+1);let point=null;for(const row of rows){for(let col=0;col<R-1&&!point;col++)point=crossing(row*R+col,row*R+col+1,targetValue);if(point)break;}if(!point){for(const col of cols){for(let row=0;row<R-1&&!point;row++)point=crossing(row*R+col,(row+1)*R+col,targetValue);if(point)break;}}if(point)found.push({point,label:fmt(targetValue)});}contourLabelWorld=found;}
function drawContourLabels(M,x,width,height,scale){updateContourLabelWorld();if(!contourLabelWorld.length)return;const placed=[];x.font='600 '+Math.round(10*scale)+'px Arial, sans-serif';x.textAlign='center';x.textBaseline='middle';for(const item of contourLabelWorld){const point=projectGridPoint(M,item.point,width,height);if(!point||point[0]<8*scale||point[0]>width-8*scale||point[1]<8*scale||point[1]>height-8*scale)continue;if(placed.some(other=>Math.hypot(other[0]-point[0],other[1]-point[1])<32*scale))continue;placed.push(point);x.lineWidth=Math.max(2,3*scale);x.strokeStyle='rgba(255,255,255,.92)';x.strokeText(item.label,point[0],point[1]);x.fillStyle='#111827';x.fillText(item.label,point[0],point[1]);}}
function draw2dGridLabels(M,x,width,height,scale){if(!gridSpec||!Array.isArray(gridSpec.plane_axes)||gridSpec.plane_axes.length!==2)return;const bounds=gridBounds();if(!bounds)return;const first=Number(gridSpec.plane_axes[0]),second=Number(gridSpec.plane_axes[1]),normal=Number(gridSpec.normal_axis),axes=Array.isArray(gridSpec.axes)?gridSpec.axes:[],firstSpec=axes[first]||{},secondSpec=axes[second]||{},normalValue=Number(bounds.lo[normal]),makePoint=(firstValue,secondValue)=>{const point=[0,0,0];point[first]=firstValue;point[second]=secondValue;point[normal]=normalValue;return point;},drawText=(text,px,py,align='center')=>{x.textAlign=align;x.textBaseline='middle';x.lineWidth=Math.max(2,3*scale);x.strokeStyle='rgba(255,255,255,.90)';x.strokeText(text,px,py);x.fillStyle='#334155';x.fillText(text,px,py);};x.font=Math.round(10*scale)+'px Arial, sans-serif';const firstTicks=Array.isArray(firstSpec.tick_values_mm)?firstSpec.tick_values_mm.map(Number):[],firstLabels=Array.isArray(firstSpec.tick_labels)?firstSpec.tick_labels.map(String):firstTicks.map(fmt);for(let i=0;i<Math.min(firstTicks.length,firstLabels.length);i++){const point=projectGridPoint(M,makePoint(firstTicks[i],bounds.lo[second]),width,height);if(point&&point[0]>=8*scale&&point[0]<=width-8*scale&&point[1]>=0&&point[1]<=height-18*scale)drawText(firstLabels[i],point[0],point[1]+11*scale);}const secondTicks=Array.isArray(secondSpec.tick_values_mm)?secondSpec.tick_values_mm.map(Number):[],secondLabels=Array.isArray(secondSpec.tick_labels)?secondSpec.tick_labels.map(String):secondTicks.map(fmt);for(let i=0;i<Math.min(secondTicks.length,secondLabels.length);i++){const point=projectGridPoint(M,makePoint(bounds.lo[first],secondTicks[i]),width,height);if(point&&point[0]>=24*scale&&point[0]<=width&&point[1]>=8*scale&&point[1]<=height-8*scale)drawText(secondLabels[i],point[0]-9*scale,point[1],'right');}x.font='600 '+Math.round(11*scale)+'px Arial, sans-serif';const firstTitle=projectGridPoint(M,makePoint((bounds.lo[first]+bounds.hi[first])/2,bounds.lo[second]),width,height),secondTitle=projectGridPoint(M,makePoint(bounds.lo[first],(bounds.lo[second]+bounds.hi[second])/2),width,height);if(firstTitle)drawText(String(firstSpec.label||''),firstTitle[0],Math.min(height-9*scale,firstTitle[1]+27*scale));if(secondTitle)drawText(String(secondSpec.label||''),Math.max(35*scale,secondTitle[0]-34*scale),secondTitle[1]);}
function drawGridOverlay(M){
  const x=gridCanvas.getContext('2d'),width=gridCanvas.width,height=gridCanvas.height;x.clearRect(0,0,width,height);if(width<2||height<2)return;const scale=captureSize?Math.max(.75,Math.min(3,height/720)):Math.min(2,window.devicePixelRatio||1);if(is2d){if(gridSpec)draw2dGridLabels(M,x,width,height,scale);if(contourLabels)drawContourLabels(M,x,width,height,scale);return;}const bounds=gridBounds();if(!bounds)return;
  const axes=Array.isArray(gridSpec.axes)?gridSpec.axes:[],centrePx=projectGridPoint(M,centre,width,height)||[width/2,height/2];
  x.lineWidth=Math.max(1,scale);x.strokeStyle='rgba(71,85,105,.72)';x.fillStyle='#475569';x.textAlign='center';x.textBaseline='middle';
  for(let axis=0;axis<3;axis++){
    const spec=axes[axis]||{},ticks=Array.isArray(spec.tick_values_mm)?spec.tick_values_mm.map(Number):[],labels=Array.isArray(spec.tick_labels)?spec.tick_labels.map(String):ticks.map(fmt),minimum=Number(spec.minimum_mm??bounds.lo[axis]),maximum=Number(spec.maximum_mm??bounds.hi[axis]),midpoint=(minimum+maximum)/2,midPx=projectGridPoint(M,gridAxisPoint(axis,midpoint,bounds),width,height);if(!midPx)continue;
    let dx=midPx[0]-centrePx[0],dy=midPx[1]-centrePx[1],length=Math.hypot(dx,dy);if(length<1){dx=axis===2?-1:0;dy=axis===2?0:1;length=1;}const ox=dx/length,oy=dy/length;
    x.font=Math.round(11*scale)+'px Arial, sans-serif';
    for(let index=0;index<Math.min(ticks.length,labels.length);index++){const value=ticks[index];if(!Number.isFinite(value))continue;const point=projectGridPoint(M,gridAxisPoint(axis,value,bounds),width,height);if(!point)continue;x.beginPath();x.moveTo(point[0],point[1]);x.lineTo(point[0]+ox*5*scale,point[1]+oy*5*scale);x.stroke();const tx=Math.max(24*scale,Math.min(width-24*scale,point[0]+ox*13*scale)),ty=Math.max(9*scale,Math.min(height-9*scale,point[1]+oy*13*scale));x.lineWidth=Math.max(2,3*scale);x.strokeStyle='rgba(255,255,255,.92)';x.strokeText(labels[index],tx,ty);x.fillStyle='#475569';x.fillText(labels[index],tx,ty);x.lineWidth=Math.max(1,scale);x.strokeStyle='rgba(71,85,105,.72)';}
    const title=String(spec.label||(['X (LR, mm)','Y (AP, mm)','Z (SI, mm)'][axis]));x.font='600 '+Math.round(12*scale)+'px Arial, sans-serif';const titleX=Math.max(48*scale,Math.min(width-48*scale,midPx[0]+ox*31*scale)),titleY=Math.max(12*scale,Math.min(height-12*scale,midPx[1]+oy*31*scale));x.lineWidth=Math.max(2,4*scale);x.strokeStyle='rgba(255,255,255,.94)';x.strokeText(title,titleX,titleY);x.fillStyle='#334155';x.fillText(title,titleX,titleY);x.lineWidth=Math.max(1,scale);x.strokeStyle='rgba(71,85,105,.72)';
  }
}
function attrib(loc,buf,offset=0){gl.bindBuffer(gl.ARRAY_BUFFER,buf);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,offset);}
function sceneAlpha(item){const materialAlpha=Number(item?.colour?.[3]),objectAlpha=Number(item?.opacity);const a=Number.isFinite(materialAlpha)?materialAlpha:1,b=Number.isFinite(objectAlpha)?objectAlpha:1;return Math.max(0,Math.min(1,a*b));}
function setScene4(loc,item){const v=item.colour||[.39,.45,.55,1];gl.uniform4f(loc,Number(v[0]),Number(v[1]),Number(v[2]),sceneAlpha(item));}
function drawMeshItem(m){attrib(solidLoc.pos,m.pbuf);setScene4(solidLoc.colour,m);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.ibuf);gl.drawElements(gl.TRIANGLES,Number(m.index_count),gl.UNSIGNED_INT,0);}
function drawLineItem(l){attrib(lineLoc.pos,l.pbuf);setScene4(lineLoc.colour,l);gl.uniform1i(lineLoc.point,0);gl.uniform1f(lineLoc.pointSize,1);try{gl.lineWidth(Math.min(4,Number(l.width)||1));}catch(e){}gl.drawArrays(gl.LINES,0,Number(l.count));}
function drawPointItem(p){attrib(lineLoc.pos,p.pbuf);setScene4(lineLoc.colour,p);gl.uniform1i(lineLoc.point,1);gl.uniform1f(lineLoc.pointSize,Number(p.size)||4);gl.drawArrays(gl.POINTS,0,Number(p.count));}
function drawScene(M){
  // Opaque scene primitives populate the depth buffer normally. Transparent
  // scene objects are then blended in stable scene order with depth writes
  // disabled. Do not camera-sort transparent objects: changing the camera must
  // never make an object win/lose the depth buffer and visibly pop in or out.
  gl.depthMask(true);
  gl.useProgram(solidProgram);gl.uniformMatrix4fv(solidLoc.mvp,false,M);gl.uniform1i(solidLoc.point,0);
  for(const m of sceneMeshes){if(sceneAlpha(m)>=.999)drawMeshItem(m);}
  gl.useProgram(lineProgram);gl.uniformMatrix4fv(lineLoc.mvp,false,M);
  for(const l of sceneLines){if(sceneAlpha(l)>=.999)drawLineItem(l);}
  for(const p of scenePoints){if(sceneAlpha(p)>=.999)drawPointItem(p);}

  gl.depthMask(false);
  gl.useProgram(solidProgram);gl.uniformMatrix4fv(solidLoc.mvp,false,M);gl.uniform1i(solidLoc.point,0);
  for(const m of sceneMeshes){const a=sceneAlpha(m);if(a>.001&&a<.999)drawMeshItem(m);}
  gl.useProgram(lineProgram);gl.uniformMatrix4fv(lineLoc.mvp,false,M);
  for(const l of sceneLines){const a=sceneAlpha(l);if(a>.001&&a<.999)drawLineItem(l);}
  for(const p of scenePoints){const a=sceneAlpha(p);if(a>.001&&a<.999)drawPointItem(p);}
  gl.depthMask(true);
}
const sliceAxisIndex='xyz'.indexOf(P.slice_axis),sliceCentre=Number(P.centre_mm[sliceAxisIndex]);const sliceOrder=(isVolume||isPointCloud)?[]:Array.from({length:L},(_,i)=>i).sort((a,b)=>Math.abs(Number(P.slice_positions_mm[a])-sliceCentre)-Math.abs(Number(P.slice_positions_mm[b])-sliceCentre));
function bindFieldBasisBuffers(buffers,offset=0){for(let j=0;j<MAX_DRIVE_COILS;j++){const loc=fieldLoc.bases[j];if(loc<0)continue;if(j<buffers.length){attrib(loc,buffers[j],offset);}else{gl.disableVertexAttribArray(loc);gl.vertexAttrib3f(loc,0,0,0);}}}
function setFieldUniforms(M,pointMode,pointSize){gl.useProgram(fieldProgram);gl.uniformMatrix4fv(fieldLoc.mvp,false,M);gl.uniform1fv(fieldLoc.currents,currents);gl.uniform1i(fieldLoc.driveCount,driveCount);gl.uniform1i(fieldLoc.component,{magnitude:0,x:1,y:2,z:3,abs_x:4,abs_y:5,abs_z:6}[P.component]??0);gl.uniform1f(fieldLoc.scale,P.field==='B'?1e6:1);gl.uniform1f(fieldLoc.min,Number(P.colour_min));gl.uniform1f(fieldLoc.max,Number(P.colour_max));gl.uniform1i(fieldLoc.log,P.logarithmic?1:0);gl.uniform1f(fieldLoc.opacityLow,(is2d||!opacityEnabled)?1.0:volumeOpacityLow);gl.uniform1f(fieldLoc.opacityHigh,(is2d||!opacityEnabled)?1.0:volumeOpacityHigh);gl.uniform1f(fieldLoc.opacityLowLevel,volumeOpacityLowLevel);gl.uniform1f(fieldLoc.opacityHighLevel,volumeOpacityHighLevel);gl.uniform1i(fieldLoc.showContours,showContours?1:0);gl.uniform1i(fieldLoc.contourCount,contourCount);gl.uniform1i(fieldLoc.point,pointMode?1:0);gl.uniform1f(fieldLoc.pointSize,pointSize);}
function drawField(M){setFieldUniforms(M,isPointCloud,Math.max(1,Math.min(16,Number(P.point_size)||4)));gl.depthMask(false);if(isPointCloud){attrib(fieldLoc.pos,posBuf);attrib(fieldLoc.fixed,fixedBuf);bindFieldBasisBuffers(basisBufs);gl.drawArrays(gl.POINTS,0,Math.max(0,Number(P.point_count)||0));gl.depthMask(true);return;}gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idxBuf);const chosen=new Set(sliceOrder.slice(0,visibleSlices));const eye=eyePosition();const drawOrder=Array.from(chosen).sort((a,b)=>{const axis='xyz'.indexOf(P.slice_axis),da=Math.abs(eye[axis]-Number(P.slice_positions_mm[a])),db=Math.abs(eye[axis]-Number(P.slice_positions_mm[b]));return db-da;});for(const layer of drawOrder){const off=layer*vertsPerSlice*3*4;attrib(fieldLoc.pos,posBuf,off);attrib(fieldLoc.fixed,fixedBuf,off);bindFieldBasisBuffers(basisBufs,off);gl.drawElements(gl.TRIANGLES,gridIndices.length,gl.UNSIGNED_INT,0);}gl.depthMask(true);}
function drawOverlayFieldPoints(M){if(!hasOverlayFieldPoints||!overlayPosBuf||!overlayFixedBuf)return;setFieldUniforms(M,true,Math.max(1,Math.min(16,Number(P.overlay_point_size)||Number(P.point_size)||4)));gl.depthMask(false);attrib(fieldLoc.pos,overlayPosBuf);attrib(fieldLoc.fixed,overlayFixedBuf);bindFieldBasisBuffers(overlayBasisBufs);gl.drawArrays(gl.POINTS,0,overlayPointCount);gl.depthMask(true);}
function drawBrainHighlight(M){if(!brainHighlightBuf||brainHighlightCount<=0)return;gl.useProgram(lineProgram);gl.uniformMatrix4fv(lineLoc.mvp,false,M);attrib(lineLoc.pos,brainHighlightBuf);gl.uniform4f(lineLoc.colour,.980,.800,.082,1.0);gl.uniform1i(lineLoc.point,1);gl.uniform1f(lineLoc.pointSize,Math.max(5.5,(Number(P.overlay_point_size)||Number(P.point_size)||4)+1.5));gl.depthMask(false);gl.drawArrays(gl.POINTS,0,brainHighlightCount);gl.depthMask(true);}
window.fieldWorkbenchSetBrainHighlight=labelIds=>{brainHighlightCount=0;if(brainHighlightBuf){gl.deleteBuffer(brainHighlightBuf);brainHighlightBuf=null;}if(!hasOverlayFieldPoints||!overlayPositionsData||!overlayLabelsData||!Array.isArray(labelIds)||!labelIds.length){dirty=true;return;}const wanted=new Set(labelIds.map(Number)),selected=[];const count=Math.min(overlayPointCount,overlayLabelsData.length,Math.floor(overlayPositionsData.length/3));for(let i=0;i<count;i++){if(!wanted.has(Number(overlayLabelsData[i])))continue;const off=i*3;selected.push(overlayPositionsData[off],overlayPositionsData[off+1],overlayPositionsData[off+2]);}if(selected.length){const values=new Float32Array(selected);brainHighlightBuf=buffer(values);brainHighlightCount=values.length/3;}dirty=true;};
function drawVolume(M){const inv=inverse(M);if(!inv)return;const eye=eyePosition(),steps=Math.max(1,Math.min(128,Number(P.volume_ray_steps)||64));gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.useProgram(volumeProgram);gl.uniformMatrix4fv(volumeLoc.invVP,false,inv);gl.uniform3f(volumeLoc.camera,eye[0],eye[1],eye[2]);const lo=P.volume_box_min_mm||[centre[0]-half,centre[1]-half,centre[2]-half],hi=P.volume_box_max_mm||[centre[0]+half,centre[1]+half,centre[2]+half];gl.uniform3f(volumeLoc.boxMin,Number(lo[0]),Number(lo[1]),Number(lo[2]));gl.uniform3f(volumeLoc.boxMax,Number(hi[0]),Number(hi[1]),Number(hi[2]));gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,volumeFixedTex);gl.uniform1i(volumeLoc.fixedTex,0);for(let j=0;j<MAX_DRIVE_COILS;j++){const loc=volumeLoc.basisTex[j];if(loc===null)continue;if(j<volumeBasisTex.length){gl.activeTexture(gl.TEXTURE0+j+1);gl.bindTexture(gl.TEXTURE_3D,volumeBasisTex[j]);gl.uniform1i(loc,j+1);}else{gl.uniform1i(loc,0);}}gl.uniform1fv(volumeLoc.currents,currents);gl.uniform1i(volumeLoc.driveCount,driveCount);gl.uniform1i(volumeLoc.component,{magnitude:0,x:1,y:2,z:3,abs_x:4,abs_y:5,abs_z:6}[P.component]??0);gl.uniform1f(volumeLoc.scale,P.field==='B'?1e6:1);gl.uniform1f(volumeLoc.min,Number(P.colour_min));gl.uniform1f(volumeLoc.max,Number(P.colour_max));gl.uniform1i(volumeLoc.log,P.logarithmic?1:0);gl.uniform1f(volumeLoc.opacityLow,opacityEnabled?volumeOpacityLow:1.0);gl.uniform1f(volumeLoc.opacityHigh,opacityEnabled?volumeOpacityHigh:1.0);gl.uniform1f(volumeLoc.opacityLowLevel,volumeOpacityLowLevel);gl.uniform1f(volumeLoc.opacityHighLevel,volumeOpacityHighLevel);gl.uniform1i(volumeLoc.surfaceCount,Math.max(2,Math.min(40,Number(P.volume_surface_count)||R)));gl.uniform1i(volumeLoc.steps,steps);gl.drawArrays(gl.TRIANGLES,0,3);}
function resize(){const dpr=Math.min(2,window.devicePixelRatio||1),w=captureSize?captureSize.width:Math.max(1,Math.floor(canvas.clientWidth*dpr)),h=captureSize?captureSize.height:Math.max(1,Math.floor(canvas.clientHeight*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h);dirty=true;}if(gridCanvas.width!==w||gridCanvas.height!==h){gridCanvas.width=w;gridCanvas.height=h;dirty=true;}}
let fpsCount=0,fpsStamp=performance.now(),measuredFps=0;
function render(){resize();const M=mvp();if(workbenchDark)gl.clearColor(174/255,179/255,184/255,1);else gl.clearColor(1,1,1,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(is2d){gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);if(showField)drawField(M);if(showField)drawOverlayFieldPoints(M);drawBrainHighlight(M);drawGrid(M);drawScene(M);}else if(isVolume){if(showField)drawVolume(M);gl.clear(gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);if(showField)drawOverlayFieldPoints(M);drawBrainHighlight(M);drawGrid(M);drawScene(M);}else{gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);drawGrid(M);drawScene(M);if(showField)drawField(M);if(showField)drawOverlayFieldPoints(M);drawBrainHighlight(M);}drawGridOverlay(M);fpsCount++;const now=performance.now();if(now-fpsStamp>=1000){measuredFps=fpsCount*1000/(now-fpsStamp);fpsCount=0;fpsStamp=now;updatePerf();}dirty=false;}
let dragging=false,button=0,lastX=0,lastY=0;
canvas.addEventListener('pointerdown',e=>{dragging=true;button=e.button;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');e.preventDefault();});
canvas.addEventListener('pointerup',e=>{dragging=false;canvas.classList.remove('dragging');try{canvas.releasePointerCapture(e.pointerId)}catch(_){}emitCameraChanged();});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;if(is2d){const worldPerPixel=Math.max(1e-9,(Number(P.span_mm)*zoom2d)/Math.max(1,canvas.clientHeight));target=vadd(target,vadd(vmul(view2d.right,-dx*worldPerPixel),vmul(view2d.screenUp,dy*worldPerPixel)));}else if(button===0&&!e.shiftKey){yaw-=dx*.007;pitch=Math.max(-1.48,Math.min(1.48,pitch+dy*.007));}else{const eye=eyePosition(),forward=vnorm(vsub(target,eye)),right=vnorm(vcross(forward,[0,0,1])),up=vnorm(vcross(right,forward)),scale=distance*.0016;target=vadd(target,vadd(vmul(right,-dx*scale),vmul(up,dy*scale)));}dirty=true;e.preventDefault();});
canvas.addEventListener('wheel',e=>{if(is2d){zoom2d*=Math.exp(e.deltaY*.001);zoom2d=Math.max(.03,Math.min(50,zoom2d));}else{distance*=Math.exp(e.deltaY*.001);distance=Math.max(fitRadius*.08,Math.min(fitRadius*80,distance));}dirty=true;emitCameraChanged();e.preventDefault();},{passive:false});
document.getElementById('home').onclick=()=>{target=fitTarget.slice();yaw=-Math.PI/4;pitch=.42;distance=fitRadius*2.7;fovY=Math.PI/4;zoom2d=1;dirty=true;emitCameraChanged();};
const timeSlider=document.getElementById('time-slider'),sliceSlider=document.getElementById('slice-slider'),timeText=document.getElementById('time-text'),sliceText=document.getElementById('slice-count'),sliceLabel=document.getElementById('slice-label'),secondary=document.getElementById('secondary'),play=document.getElementById('play'),stop=document.getElementById('stop'),loop=document.getElementById('loop'),playbackSpeed=document.getElementById('playback-speed'),perf=document.getElementById('perf');
if(is2d){sliceLabel.style.display='none';sliceSlider.style.display='none';sliceText.style.display='none';secondary.style.gridTemplateColumns='1fr';document.getElementById('axis-hint').textContent=((P.axis_labels||['Horizontal','Vertical']).join(' • '))+'   |   Drag: pan • wheel: zoom';}
if(!is2d){document.getElementById('axis-hint').textContent=(P.axis_labels||['X (LR, mm)','Y (AP, mm)','Z (SI, mm)']).join(' • ')+'   |   Drag: orbit • right/middle drag: pan • wheel: zoom'+(opacityEnabled?' • intensity-linked opacity':'');}if(isVolume||isPointCloud){sliceLabel.style.display='none';sliceSlider.style.display='none';sliceText.style.display='none';}
if(staticView){document.getElementById('play').style.display='none';document.getElementById('stop').style.display='none';document.getElementById('loop-control').style.display='none';document.getElementById('time-area').style.display='none';document.getElementById('speed-control').style.display='none';perf.style.display='none';if(isVolume||!showField){document.getElementById('controls').style.display='none';}else{document.getElementById('controls').style.display='block';secondary.style.display='grid';secondary.style.gridTemplateColumns='auto minmax(140px,1fr) auto';secondary.style.margin='0';}}
sliceSlider.min=1;sliceSlider.max=Math.max(1,L);sliceSlider.step=1;sliceSlider.value=Math.max(1,L);sliceSlider.disabled=L<=1;
const firstFrameTime=N?Number(frameTimes[0]):0,lastFrameTime=N?Number(frameTimes[N-1]):firstFrameTime,waveformDuration=Math.max(0,lastFrameTime-firstFrameTime),scriptDuration=scriptedTimeline?Math.max(0,Number(cameraTimeline.duration_s)||waveformDuration):waveformDuration;
let frameIndex=0,running=false,runStart=0,lastAdvanceNow=0,playbackElapsed=0,waveformElapsed=0,runStartElapsed=0,renderedSequenceElapsed=0,livePlaybackSpeed=1.0,endedNaturally=false;
const stimulusLoop=!!(scriptedTimeline&&cameraTimeline&&cameraTimeline.loop_waveform);
const configuredRealTimeMultiplier=Math.max(1e-12,Number(P.real_time_multiplier)||1),timelineSourceDuration=Math.max(0,scriptedTimeline?scriptDuration:waveformDuration),videoTimelineDuration=timelineSourceDuration/configuredRealTimeMultiplier;
timeSlider.min=0;timeSlider.max=Math.max(0,videoTimelineDuration);timeSlider.step=.01;timeSlider.value=0;
function timelinePhysicalPhase(elapsed){const value=Math.max(0,Number(elapsed)||0);if(!(waveformDuration>0))return 0;return Math.min(waveformDuration,value);}
function effectiveRealTimeMultiplier(elapsed=playbackElapsed){if(scriptedTimeline)return configuredRealTimeMultiplier*livePlaybackSpeed;return configuredRealTimeMultiplier*livePlaybackSpeed*timelinePlaybackSpeedAt(timelinePhysicalPhase(elapsed));}
function waveformAdvanceForScriptInterval(startS,endS){let start=Math.max(0,Number(startS)||0),end=Math.max(start,Number(endS)||0),total=0,cursor=start;if(!cameraTimeline||end<=start)return 0;for(const point of cameraTimeline.points){const boundary=Number(point.time_s)||0;if(boundary<=cursor+1e-12)continue;if(boundary>=end-1e-12)break;total+=(boundary-cursor)*timelinePlaybackSpeedAt(cursor);cursor=boundary;}if(end>cursor)total+=(end-cursor)*timelinePlaybackSpeedAt(cursor);return total;}
function waveformElapsedAtScriptTime(timeS){return waveformAdvanceForScriptInterval(0,Math.max(0,Number(timeS)||0));}
function runningPlaybackElapsed(now){if(!cameraTimeline)return runStartElapsed+(now-runStart)/1000*effectiveRealTimeMultiplier();const current=Number(now)||performance.now();if(lastAdvanceNow<=0)lastAdvanceNow=current;const delta=Math.max(0,(current-lastAdvanceNow)/1000);if(scriptedTimeline){const start=playbackElapsed,end=start+delta*configuredRealTimeMultiplier*livePlaybackSpeed;waveformElapsed+=waveformAdvanceForScriptInterval(start,end);playbackElapsed=end;}else{playbackElapsed+=delta*effectiveRealTimeMultiplier(playbackElapsed);waveformElapsed=playbackElapsed;}lastAdvanceNow=current;return playbackElapsed;}
function sequenceStepAt(elapsed){
  if(!sequenceConfig)return null;
  const steps=sequenceConfig.steps,cycle=Number(sequenceConfig.cycle_s)||steps.reduce((total,step)=>total+Math.max(0,Number(step.duration_s)||0),0),tolerance=Math.max(1e-12,cycle*1e-10);
  if(!(cycle>0))return -1;
  let phase=Math.max(0,Number(elapsed)||0);
  if((scriptedTimeline&&stimulusLoop)||(!scriptedTimeline&&sequenceConfig.loop===true)){phase=((phase%cycle)+cycle)%cycle;if(Math.abs(phase-cycle)<=tolerance||Math.abs(phase)<=tolerance)phase=0;}
  else if(phase>=cycle-tolerance)return -1;
  let boundary=0;
  for(let index=0;index<steps.length;index++){boundary+=Math.max(0,Number(steps[index].duration_s)||0);if(phase<boundary-tolerance||index===steps.length-1)return index;}
  return -1;
}
function sequenceElapsedForFrame(index,elapsed){
  if(scriptedTimeline)return Math.max(0,waveformElapsed);
  const frameElapsed=Math.max(0,(N?Number(frameTimes[index]):firstFrameTime)-firstFrameTime);
  if(!(waveformDuration>0))return frameElapsed;
  return frameElapsed;
}
function formatPlaybackClock(seconds){
  const value=Math.max(0,Number(seconds)||0),hundredths=Math.round(value*100),hours=Math.floor(hundredths/360000),minutes=Math.floor((hundredths-hours*360000)/6000),secs=(hundredths%6000)/100,secondsText=secs.toFixed(2).padStart(5,'0');
  return hours>0?String(hours)+':'+String(minutes).padStart(2,'0')+':'+secondsText:String(minutes)+':'+secondsText;
}
function videoTimelinePosition(){
  if(!(configuredRealTimeMultiplier>0))return 0;
  let source=Math.max(0,Number(playbackElapsed)||0);
  source=Math.min(timelineSourceDuration,source);
  return Math.max(0,Math.min(videoTimelineDuration,source/configuredRealTimeMultiplier));
}
function updateTime(){const t=N?Number(frameTimes[frameIndex]):0,videoT=videoTimelinePosition(),clock=formatPlaybackClock(videoT)+' / '+formatPlaybackClock(videoTimelineDuration);timeText.textContent=clock;document.getElementById('export-time').textContent='Video timeline '+clock;timeSlider.value=videoT;drawObservation(t,false,renderedSequenceElapsed);}
function setFrame(i,elapsed=playbackElapsed,applyTimeline=true){
  frameIndex=Math.max(0,Math.min(N-1,Math.round(i)));playbackElapsed=Math.max(0,Number(elapsed)||0);renderedSequenceElapsed=sequenceElapsedForFrame(frameIndex,playbackElapsed);currents.fill(0);
  const driveElapsed=scriptedTimeline?waveformElapsed:playbackElapsed,tolerance=Math.max(1e-12,waveformDuration*1e-10),firstPass=driveElapsed<waveformDuration-tolerance||(frameIndex===N-1&&driveElapsed<=waveformDuration+tolerance),stepIndex=sequenceStepAt(renderedSequenceElapsed),step=stepIndex===null||stepIndex<0?null:sequenceConfig.steps[stepIndex],active=step?new Set((step.active_coil_indices||[]).map(Number)):null,source=sequenceConfig&&!firstPass?freeFrameCurrents:frameCurrents;
  for(let j=0;j<driveCount;j++)currents[j]=N&&(!sequenceConfig||firstPass||(active&&active.has(j)))?Number(source[frameIndex*driveCount+j]):0;
  if(cameraTimeline&&applyTimeline)applyTimelineCamera(scriptedTimeline?playbackElapsed:Math.max(0,(N?Number(frameTimes[frameIndex]):firstFrameTime)-firstFrameTime));
  updateTime();emitFrameChanged();dirty=true;
}
function updateStopButton(){stop.disabled=false;stop.textContent=running?'■ Stop':'↺ Reset';stop.classList.toggle('reset',!running);}
function stopRun(){if(running){playbackElapsed=Math.max(0,runningPlaybackElapsed(performance.now()));running=false;setFrame(frameIndex,playbackElapsed);}play.disabled=false;endedNaturally=false;updateStopButton();updatePerf();}
function resetPlayback(){running=false;endedNaturally=false;playbackElapsed=0;waveformElapsed=0;runStartElapsed=0;lastAdvanceNow=0;setFrame(0,0);play.disabled=false;updateStopButton();updatePerf();}
window.fieldWorkbenchWaveformStop=stopRun;
timeSlider.oninput=()=>{if(running)stopRun();endedNaturally=false;const videoT=Math.max(0,Math.min(videoTimelineDuration,Number(timeSlider.value)||0)),elapsed=Math.min(timelineSourceDuration,videoT*configuredRealTimeMultiplier);playbackElapsed=elapsed;waveformElapsed=scriptedTimeline?waveformElapsedAtScriptTime(elapsed):elapsed;const waveDuration=Math.max(1e-30,waveformDuration),phase=stimulusLoop?((waveformElapsed%waveDuration)+waveDuration)%waveDuration:Math.min(waveDuration,waveformElapsed),fraction=phase/waveDuration,index=Math.max(0,Math.min(N-1,Math.round(fraction*(N-1))));setFrame(index,playbackElapsed);updateStopButton();};
sliceSlider.oninput=()=>{visibleSlices=Number(sliceSlider.value);sliceText.textContent=visibleSlices+' / '+L;dirty=true;};sliceText.textContent=visibleSlices+' / '+L;
play.onclick=()=>{if(N<2)return;if(endedNaturally)resetPlayback();running=true;endedNaturally=false;play.disabled=true;runStart=performance.now();lastAdvanceNow=runStart;runStartElapsed=playbackElapsed;updateStopButton();updatePerf();};stop.onclick=()=>{if(running)stopRun();else resetPlayback();};
playbackSpeed.onchange=()=>{const now=performance.now();if(running){playbackElapsed=Math.max(0,runningPlaybackElapsed(now));runStart=now;runStartElapsed=playbackElapsed;}const next=Number(playbackSpeed.value);livePlaybackSpeed=Number.isFinite(next)&&next>0?next:1.0;updatePerf();};
function updatePerf(){if(staticView){perf.textContent='';return;}const targetFps=Number(P.playback_fps)||60;perf.textContent='Target '+fmt(targetFps)+' FPS • GPU '+(measuredFps?measuredFps.toFixed(1):'—')+' FPS';}
const obsWrap=document.getElementById('observation-wrap'),obsCanvas=document.getElementById('obs'),obsStatsText=document.getElementById('obs-stats-text'),obsInvert=document.getElementById('obs-invert');
const scrollObservationTraces=P.scroll_observation_traces===true;
const observationTraces=Array.isArray(P.observations)?P.observations.slice():(P.observation?[P.observation]:[]);if(P.excitation)observationTraces.push(P.excitation);
const sequencerTrace=P.sequencer&&typeof P.sequencer==='object'?P.sequencer:null;
const hasObservationData=observationTraces.length>0||!!sequencerTrace;let obsBase=null,observationCaptureSize=null;
function exportTraceFraction(){if(!hasObservationData)return 0;if(sequencerTrace&&observationTraces.length)return .28;if(sequencerTrace)return .14;return .23;}
const exportTracesButton=document.getElementById('export-traces');let nativeBridge=null;exportTracesButton.disabled=!hasObservationData;function emitFrameChanged(){if(nativeBridge&&nativeBridge.frameChanged)nativeBridge.frameChanged(frameIndex,N?Number(frameTimes[frameIndex]):0);}if(typeof QWebChannel!=='undefined'&&window.qt&&window.qt.webChannelTransport){new QWebChannel(window.qt.webChannelTransport,channel=>{nativeBridge=channel.objects.fieldWorkbenchBridge||null;exportTracesButton.disabled=!nativeBridge||!hasObservationData;emitFrameChanged();});}else{exportTracesButton.disabled=true;}exportTracesButton.onclick=()=>{if(nativeBridge&&hasObservationData)nativeBridge.exportTraces();};
let gpuValidationProgram=null,gpuValidationLoc=null;
function scalarCpuAt(index){let x=Number(volumeFixedData[index*3]||0),y=Number(volumeFixedData[index*3+1]||0),z=Number(volumeFixedData[index*3+2]||0);for(let c=0;c<driveCount&&c<volumeBasisData.length;c++){const b=volumeBasisData[c],a=Number(currents[c]||0),off=index*3;x+=a*Number(b[off]||0);y+=a*Number(b[off+1]||0);z+=a*Number(b[off+2]||0);}const comp=String(P.component||'magnitude');let value=comp==='x'?x:comp==='y'?y:comp==='z'?z:comp==='abs_x'?Math.abs(x):comp==='abs_y'?Math.abs(y):comp==='abs_z'?Math.abs(z):Math.hypot(x,y,z);return value*(P.field==='B'?1e6:1);}
function normaliseScalar(value){let s=Number(value);if(P.logarithmic)s=Math.log10(Math.max(s,1e-30));return Math.max(0,Math.min(1,(s-Number(P.colour_min))/Math.max(1e-30,Number(P.colour_max)-Number(P.colour_min))));}
function colourRgb(t){t=Math.max(0,Math.min(1,Number(t)||0));const mix3=(a,b,u)=>a.map((v,i)=>v+(b[i]-v)*u),c0=[8,48,107],c1=[37,99,235],c2=[6,182,212],c3=[34,197,94],c4=[250,204,21],c5=[249,115,22],c6=[220,38,38];if(P.signed_component){const neg=[37,99,235],zero=[248,250,252],pos=[220,38,38];return (t<.5?mix3(neg,zero,t*2):mix3(zero,pos,(t-.5)*2)).map(v=>Math.round(v));}let c;if(t<.18)c=mix3(c0,c1,t/.18);else if(t<.36)c=mix3(c1,c2,(t-.18)/.18);else if(t<.54)c=mix3(c2,c3,(t-.36)/.18);else if(t<.72)c=mix3(c3,c4,(t-.54)/.18);else if(t<.86)c=mix3(c4,c5,(t-.72)/.14);else c=mix3(c5,c6,(t-.86)/.14);return c.map(v=>Math.round(v));}
function ensureGpuValidationProgram(){if(gpuValidationProgram)return;const vs=`#version 300 es
precision highp float;precision highp sampler3D;uniform sampler3D uFixedTex;uniform sampler3D uBasisTex0;uniform sampler3D uBasisTex1;uniform sampler3D uBasisTex2;uniform sampler3D uBasisTex3;uniform sampler3D uBasisTex4;uniform sampler3D uBasisTex5;uniform sampler3D uBasisTex6;uniform sampler3D uBasisTex7;uniform sampler3D uBasisTex8;uniform sampler3D uBasisTex9;uniform sampler3D uBasisTex10;uniform sampler3D uBasisTex11;uniform float uCurrents[12];uniform int uDriveCount;uniform int uComponent;uniform float uScale;uniform int uResolution;out float vScalar;float cv(vec3 v){if(uComponent==0)return length(v);if(uComponent==1)return v.x;if(uComponent==2)return v.y;if(uComponent==3)return v.z;if(uComponent==4)return abs(v.x);if(uComponent==5)return abs(v.y);return abs(v.z);}void main(){int r=max(uResolution,2),index=gl_VertexID,x=index%r,y=(index/r)%r,z=index/(r*r);vec3 tc=(vec3(float(x),float(y),float(z))+vec3(0.5))/float(r);vec3 vf=texture(uFixedTex,tc).xyz;if(uDriveCount>0)vf+=uCurrents[0]*texture(uBasisTex0,tc).xyz;if(uDriveCount>1)vf+=uCurrents[1]*texture(uBasisTex1,tc).xyz;if(uDriveCount>2)vf+=uCurrents[2]*texture(uBasisTex2,tc).xyz;if(uDriveCount>3)vf+=uCurrents[3]*texture(uBasisTex3,tc).xyz;if(uDriveCount>4)vf+=uCurrents[4]*texture(uBasisTex4,tc).xyz;if(uDriveCount>5)vf+=uCurrents[5]*texture(uBasisTex5,tc).xyz;if(uDriveCount>6)vf+=uCurrents[6]*texture(uBasisTex6,tc).xyz;if(uDriveCount>7)vf+=uCurrents[7]*texture(uBasisTex7,tc).xyz;if(uDriveCount>8)vf+=uCurrents[8]*texture(uBasisTex8,tc).xyz;if(uDriveCount>9)vf+=uCurrents[9]*texture(uBasisTex9,tc).xyz;if(uDriveCount>10)vf+=uCurrents[10]*texture(uBasisTex10,tc).xyz;if(uDriveCount>11)vf+=uCurrents[11]*texture(uBasisTex11,tc).xyz;vScalar=cv(vf)*uScale;gl_Position=vec4(0.0,0.0,0.0,1.0);}`;const fs=`#version 300 es
precision highp float;out vec4 outColor;void main(){outColor=vec4(0.0);}`;gpuValidationProgram=transformProgram(vs,fs,['vScalar']);gpuValidationLoc={fixedTex:gl.getUniformLocation(gpuValidationProgram,'uFixedTex'),basisTex:Array.from({length:MAX_DRIVE_COILS},(_,i)=>gl.getUniformLocation(gpuValidationProgram,'uBasisTex'+i)),currents:gl.getUniformLocation(gpuValidationProgram,'uCurrents[0]'),driveCount:gl.getUniformLocation(gpuValidationProgram,'uDriveCount'),component:gl.getUniformLocation(gpuValidationProgram,'uComponent'),scale:gl.getUniformLocation(gpuValidationProgram,'uScale'),resolution:gl.getUniformLocation(gpuValidationProgram,'uResolution')};}
function gpuVolumeValidationReport(){if(!isVolume||!volumeFixedData)return{ok:false,error:'The selected animated viewer is not using Full Volume rendering.'};ensureGpuValidationProgram();const count=R*R*R,outBuffer=gl.createBuffer(),gpu=new Float32Array(count);gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,outBuffer);gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER,count*4,gl.STREAM_READ);gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,outBuffer);gl.useProgram(gpuValidationProgram);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,volumeFixedTex);gl.uniform1i(gpuValidationLoc.fixedTex,0);for(let j=0;j<MAX_DRIVE_COILS;j++){const loc=gpuValidationLoc.basisTex[j];if(loc===null)continue;if(j<volumeBasisTex.length){gl.activeTexture(gl.TEXTURE0+j+1);gl.bindTexture(gl.TEXTURE_3D,volumeBasisTex[j]);gl.uniform1i(loc,j+1);}else gl.uniform1i(loc,0);}gl.uniform1fv(gpuValidationLoc.currents,currents);gl.uniform1i(gpuValidationLoc.driveCount,driveCount);gl.uniform1i(gpuValidationLoc.component,{magnitude:0,x:1,y:2,z:3,abs_x:4,abs_y:5,abs_z:6}[P.component]??0);gl.uniform1f(gpuValidationLoc.scale,P.field==='B'?1e6:1);gl.uniform1i(gpuValidationLoc.resolution,R);gl.enable(gl.RASTERIZER_DISCARD);gl.beginTransformFeedback(gl.POINTS);gl.drawArrays(gl.POINTS,0,count);gl.endTransformFeedback();gl.disable(gl.RASTERIZER_DISCARD);gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,outBuffer);gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER,0,gpu);gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,null);gl.deleteBuffer(outBuffer);let sum2=0,maxError=-1,worst=0,peak=0,relativeMax=0,relativeSum2=0,relativeCount=0,maxColourError=0,maxCpu=-Infinity,maxCpuIndex=0;const cpu=new Float64Array(count);for(let i=0;i<count;i++){const expected=scalarCpuAt(i),actual=Number(gpu[i]),err=Math.abs(actual-expected);cpu[i]=expected;sum2+=err*err;const mag=Math.abs(expected);if(mag>peak)peak=mag;if(expected>maxCpu){maxCpu=expected;maxCpuIndex=i;}if(err>maxError){maxError=err;worst=i;}if(mag>1e-6){const rel=err/mag;relativeMax=Math.max(relativeMax,rel);relativeSum2+=rel*rel;relativeCount++;}maxColourError=Math.max(maxColourError,Math.abs(normaliseScalar(actual)-normaliseScalar(expected)));}const coords=index=>{const x=index%R,y=Math.floor(index/R)%R,z=Math.floor(index/(R*R)),lo=P.volume_box_min_mm||P.centre_mm.map((v)=>Number(v)-Number(P.span_mm)/2),hi=P.volume_box_max_mm||P.centre_mm.map((v)=>Number(v)+Number(P.span_mm)/2),f=R>1?[x/(R-1),y/(R-1),z/(R-1)]:[0,0,0];return{index,grid:[x,y,z],position_mm:f.map((u,k)=>Number(lo[k])+u*(Number(hi[k])-Number(lo[k])))};};const selected=[Math.floor(count/2),maxCpuIndex,worst,0,count-1];const seen=new Set(),probes=[];for(const index of selected){if(index<0||index>=count||seen.has(index))continue;seen.add(index);const expected=Number(cpu[index]),actual=Number(gpu[index]),tCpu=normaliseScalar(expected),tGpu=normaliseScalar(actual);probes.push({...coords(index),cpu_value:expected,gpu_value:actual,error:Math.abs(actual-expected),cpu_colour_fraction:tCpu,gpu_colour_fraction:tGpu,cpu_rgb:colourRgb(tCpu),gpu_rgb:colourRgb(tGpu)});}const absTolerance=Math.max(1e-5,peak*1e-3),rms=Math.sqrt(sum2/Math.max(1,count)),rmsRel=Math.sqrt(relativeSum2/Math.max(1,relativeCount));return{ok:true,passed:maxError<=absTolerance&&maxColourError<=0.005,resolution:R,point_count:count,field:String(P.field||''),component:String(P.component||''),unit:String(P.unit||''),texture_filter:volumeTextureFilter,currents_a:Array.from(currents.slice(0,driveCount),Number),peak_field:peak,max_abs_error:maxError,rms_abs_error:rms,max_relative_error_pct:relativeMax*100,rms_relative_error_pct:rmsRel*100,max_colour_fraction_error:maxColourError,absolute_tolerance:absTolerance,colour_min:Number(P.colour_min),colour_max:Number(P.colour_max),probes};}
window.fieldWorkbenchValidateGpuVolume=()=>{let report;try{report=gpuVolumeValidationReport();}catch(error){report={ok:false,error:String(error&&error.message||error)}}if(nativeBridge&&nativeBridge.gpuValidationReady)nativeBridge.gpuValidationReady(JSON.stringify(report));return !!report.ok;};
function emitCameraChanged(){if(nativeBridge&&nativeBridge.cameraChanged)nativeBridge.cameraChanged(JSON.stringify(cameraState()));}
function clippedCanvasText(x,text,maxWidth){const source=String(text||'');if(x.measureText(source).width<=maxWidth)return source;let low=0,high=source.length;while(low<high){const middle=Math.ceil((low+high)/2),candidate=source.slice(0,middle)+'…';if(x.measureText(candidate).width<=maxWidth)low=middle;else high=middle-1;}return source.slice(0,low)+'…';}
function brainMetricCacheFrame(frameIndex){const frames=Math.max(0,Number(brainMetricShape[0])||0);if(frames<=1)return 0;const source=Math.max(0,Math.round(Number(frameIndex)||0));if(brainMetricSampleIndices&&brainMetricSampleIndices.length===frames){let lo=0,hi=frames-1;while(lo<hi){const mid=(lo+hi)>>1;if(Number(brainMetricSampleIndices[mid])<source)lo=mid+1;else hi=mid;}if(lo>0&&Math.abs(Number(brainMetricSampleIndices[lo-1])-source)<=Math.abs(Number(brainMetricSampleIndices[lo])-source))return lo-1;return lo;}if(brainMetricSourceFrameCount>1)return Math.max(0,Math.min(frames-1,Math.round(source*(frames-1)/(brainMetricSourceFrameCount-1))));return Math.max(0,Math.min(frames-1,source));}
function brainMetricAt(frameIndex,key,column){const regionIndex=brainMetricRowIndex.get(String(key));if(regionIndex===undefined||column<1||column>7)return NaN;const metric=column-1;if(brainTableMode==='summary'&&hasBrainSummary){const row=brainSummaryValues[regionIndex];return Array.isArray(row)&&metric<row.length?Number(row[metric]):NaN;}const regions=Math.max(0,Number(brainMetricShape[1])||0),metrics=Math.max(0,Number(brainMetricShape[2])||0),frame=brainMetricCacheFrame(frameIndex);if(!brainMetricValues||metric>=metrics)return NaN;return Number(brainMetricValues[(frame*regions+regionIndex)*metrics+metric]);}
function brainRowMatchesRule(row,frameIndex,rule){const metric=rule.metric,op=String(rule.operator||''),threshold=rule.value;if(metric==='side')return op==='='&&String(row.side||'').toLowerCase()===String(threshold||'').toLowerCase();const value=brainMetricAt(frameIndex,row.key,Number(metric)),limit=Number(threshold);if(!Number.isFinite(value)||!Number.isFinite(limit))return false;if(op==='>')return value>limit;if(op==='<')return value<limit;return false;}
function brainRowMatchesFilters(row,frameIndex){const rules=Array.isArray(brainVideoTable.filters)?brainVideoTable.filters:[];return rules.every(rule=>brainRowMatchesRule(row,frameIndex,rule));}
function brainRowHighlightColour(row,frameIndex){const rules=Array.isArray(brainVideoTable.highlights)?brainVideoTable.highlights:[];let colour='';for(const rule of rules)if(brainRowMatchesRule(row,frameIndex,rule))colour=String(rule.colour||'');return colour;}
function brainHighlightTextColour(colour,fallback){const match=/^#([0-9a-f]{6})$/i.exec(String(colour||''));if(!match)return fallback;const n=parseInt(match[1],16),r=(n>>16)&255,g=(n>>8)&255,b=n&255,l=(.2126*r+.7152*g+.0722*b)/255;return l>.56?'#111827':'#f8fafc';}
function brainVisibleRows(frameIndex){if(!hasBrainVideoTable)return[];const rules=Array.isArray(brainVideoTable.filters)?brainVideoTable.filters:[],expanded=new Set(Array.isArray(brainVideoTable.expanded_keys)?brainVideoTable.expanded_keys.map(String):[]),visibleCache=new Map(),sortColumn=Math.max(0,Math.min(7,Number(brainVideoTable.sort_column)||0)),descending=String(brainVideoTable.sort_order||'ascending').toLowerCase()==='descending';function filterVisible(row){if(visibleCache.has(row.key))return visibleCache.get(row.key);let childVisible=false;for(const child of(brainVideoChildren.get(row.key)||[]))childVisible=filterVisible(child)||childVisible;const excluded=rules.length&&brainRowMatchesFilters(row,frameIndex),visible=!excluded||childVisible;visibleCache.set(row.key,visible);return visible;}function sortedChildren(parent){const rows=(brainVideoChildren.get(parent)||[]).filter(filterVisible);rows.sort((a,b)=>{let cmp=0;if(sortColumn===0)cmp=a.name.localeCompare(b.name,undefined,{sensitivity:'base'});else{const av=brainMetricAt(frameIndex,a.key,sortColumn),bv=brainMetricAt(frameIndex,b.key,sortColumn),af=Number.isFinite(av),bf=Number.isFinite(bv);if(af&&bf)cmp=av-bv;else if(af!==bf)cmp=af?-1:1;else cmp=a.name.localeCompare(b.name,undefined,{sensitivity:'base'});}return descending?-cmp:cmp;});return rows;}const output=[];function append(row){if(!filterVisible(row))return;output.push(row);if(expanded.has(row.key))for(const child of sortedChildren(row.key))append(child);}for(const root of sortedChildren('__root__'))append(root);return output;}
function brainMetricText(value,column){if(!Number.isFinite(value))return '—';const units=Array.isArray(brainVideoTable&&brainVideoTable.units)?brainVideoTable.units:['','µT','µT','µT','µT','%','%','cm³'],unit=String(units[column]||'');return fmt(value)+(unit?' '+unit:'');}
function drawBrainVideoTable(x,originX,width,height,frameIndex){if(!hasBrainVideoTable||width<20)return;const scale=Math.max(.70,Math.min(1.55,height/900)),pad=Math.round(12*scale),titleSize=Math.round(18*scale),noteSize=Math.round(10*scale),headerH=Math.round(29*scale),rowH=Math.round(24*scale),headerY=pad+titleSize+Math.round(18*scale),tableY=headerY+Math.round(8*scale),rows=brainVisibleRows(frameIndex),visibleColumns=(Array.isArray(brainVideoTable.visible_columns)?brainVideoTable.visible_columns.map(Number):[0,1,2,3,4,5,6,7]).filter(c=>c>=0&&c<8),columns=visibleColumns.length?visibleColumns:[0],sourceWidths=Array.isArray(brainVideoTable.column_widths)?brainVideoTable.column_widths.map(Number):[220,92,92,92,92,104,96,92],usable=Math.max(1,width-pad*2),rawTotal=columns.reduce((sum,c)=>sum+Math.max(35,sourceWidths[c]||90),0),colWidths=columns.map(c=>usable*Math.max(35,sourceWidths[c]||90)/rawTotal),headers=Array.isArray(brainVideoTable.headers)?brainVideoTable.headers:['Structure','RMS |B|','Mean |B|','P95 |B|','Peak |B|','Uniformity','Direction','Volume'],dark=workbenchDark;x.fillStyle=dark?'#2f353b':'#f8fafc';x.fillRect(originX,0,width,height);x.strokeStyle=dark?'#64748b':'#cbd5e1';x.lineWidth=Math.max(1,scale);x.beginPath();x.moveTo(originX+.5,0);x.lineTo(originX+.5,height);x.stroke();x.fillStyle=dark?'#f8fafc':'#1e293b';x.font='700 '+titleSize+'px Arial';x.textAlign='left';x.textBaseline='alphabetic';x.fillText(String(brainVideoTable.title||'Brain areas'),originX+pad,pad+titleSize);const frameTime=N?Number(frameTimes[Math.max(0,Math.min(N-1,frameIndex))]):0;x.fillStyle=dark?'#cbd5e1':'#64748b';x.font=noteSize+'px Arial';x.textAlign='right';x.fillText('Frame '+(frameIndex+1)+'/'+Math.max(1,N)+' • '+fmt(frameTime)+' s',originX+width-pad,pad+titleSize);let cx=originX+pad;x.fillStyle=dark?'#283038':'#334155';x.fillRect(cx,tableY,usable,headerH);x.font='600 '+Math.round(11*scale)+'px Arial';x.textBaseline='middle';for(let i=0;i<columns.length;i++){const column=columns[i],cw=colWidths[i],sortMark=column===Number(brainVideoTable.sort_column)?(String(brainVideoTable.sort_order)==='descending'?' ▼':' ▲'):'';x.fillStyle='#f8fafc';x.textAlign=column===0?'left':'center';const tx=column===0?cx+Math.round(6*scale):cx+cw/2;x.fillText(clippedCanvasText(x,String(headers[column]||'')+sortMark,Math.max(8,cw-Math.round(10*scale))),tx,tableY+headerH/2);x.strokeStyle=dark?'#475569':'#64748b';x.strokeRect(cx,tableY,cw,headerH);cx+=cw;}const bodyTop=tableY+headerH,available=Math.max(0,height-bodyTop-pad),maxRows=Math.max(0,Math.floor(available/rowH)),shown=rows.slice(0,maxRows),rowFont=Math.max(9,Math.round(11*scale));for(let r=0;r<shown.length;r++){const row=shown[r],y=bodyTop+r*rowH,highlightColour=brainRowHighlightColour(row,frameIndex),defaultText=dark?'#f1f5f9':'#1e293b';cx=originX+pad;x.fillStyle=highlightColour||(dark?(r%2?'#3b4249':'#343b42'):(r%2?'#e2e8f0':'#f1f5f9'));x.fillRect(cx,y,usable,rowH);for(let i=0;i<columns.length;i++){const column=columns[i],cw=colWidths[i];x.strokeStyle=dark?'#4b5563':'#cbd5e1';x.strokeRect(cx,y,cw,rowH);x.font=rowFont+'px Arial';x.fillStyle=highlightColour?brainHighlightTextColour(highlightColour,defaultText):defaultText;x.textBaseline='middle';if(column===0){const indent=Math.max(0,Number(row.depth)||0)*Math.round(11*scale),hasChildren=(brainVideoChildren.get(row.key)||[]).length>0,expanded=new Set(Array.isArray(brainVideoTable.expanded_keys)?brainVideoTable.expanded_keys.map(String):[]).has(row.key);let prefix='';if(hasChildren)prefix=expanded?'▾ ':'▸ ';const text=prefix+row.name;x.textAlign='left';x.fillText(clippedCanvasText(x,text,Math.max(8,cw-indent-Math.round(10*scale))),cx+Math.round(5*scale)+indent,y+rowH/2);}else{x.textAlign='right';x.fillText(clippedCanvasText(x,brainMetricText(brainMetricAt(frameIndex,row.key,column),column),Math.max(8,cw-Math.round(8*scale))),cx+cw-Math.round(4*scale),y+rowH/2);}cx+=cw;}}if(rows.length>shown.length&&maxRows>0){const footerY=bodyTop+(maxRows-1)*rowH;x.fillStyle=dark?'rgba(47,53,59,.94)':'rgba(248,250,252,.94)';x.fillRect(originX+pad,footerY,usable,rowH);x.fillStyle=dark?'#cbd5e1':'#64748b';x.font='600 '+rowFont+'px Arial';x.textAlign='center';x.fillText('… '+(rows.length-shown.length+1)+' more rows',originX+pad+usable/2,footerY+rowH/2);}}
const drawBrainVideoTableBase=drawBrainVideoTable;
drawBrainVideoTable=function(x,originX,width,height,frameIndex){
  drawBrainVideoTableBase(x,originX,width,height,frameIndex);
  if(brainTableMode!=='summary'||!hasBrainVideoTable)return;
  const scale=Math.max(.70,Math.min(1.55,height/900)),pad=Math.round(12*scale),titleSize=Math.round(18*scale),noteSize=Math.round(10*scale),dark=workbenchDark,lines=Array.isArray(brainVideoTable.summary_lines)?brainVideoTable.summary_lines:[],note=lines.length?String(lines[0]).replace(/^Exposure time:\s*/,'Full playback • '):'Full playback',left=originX+Math.round(width*.48),right=originX+width-pad,top=pad,bottom=pad+titleSize+Math.round(3*scale);
  x.fillStyle=dark?'#2f353b':'#f8fafc';x.fillRect(left,top,Math.max(0,right-left+2),Math.max(1,bottom-top));x.fillStyle=dark?'#cbd5e1':'#64748b';x.font=noteSize+'px Arial';x.textAlign='right';x.textBaseline='alphabetic';x.fillText(clippedCanvasText(x,note,Math.max(20,right-left)),right,pad+titleSize);
};
function exportFrameCanvas(index,width,height,viewerWidth,tableWidth,sceneHeight,traceHeight){
  const output=document.createElement('canvas');output.width=width;output.height=height;const x=output.getContext('2d');x.fillStyle=workbenchDark?'#20252b':'#fff';x.fillRect(0,0,width,height);x.drawImage(canvas,0,0,viewerWidth,sceneHeight);if(gridSpec)x.drawImage(gridCanvas,0,0,viewerWidth,sceneHeight);
  const scale=Math.max(.75,Math.min(3,sceneHeight/720)),barW=Math.round(18*scale),barH=Math.round(Math.max(80,Math.min(sceneHeight-64,sceneHeight*.58))),barX=Math.round(viewerWidth-30*scale-barW),barY=Math.round((sceneHeight-barH)/2),gradient=x.createLinearGradient(0,barY+barH,0,barY);
  if(P.signed_component){gradient.addColorStop(0,'#2563eb');gradient.addColorStop(.5,'#f8fafc');gradient.addColorStop(1,'#dc2626');}else{gradient.addColorStop(0,'#08306b');gradient.addColorStop(.18,'#2563eb');gradient.addColorStop(.36,'#06b6d4');gradient.addColorStop(.54,'#22c55e');gradient.addColorStop(.72,'#facc15');gradient.addColorStop(.86,'#f97316');gradient.addColorStop(1,'#dc2626');}
  x.fillStyle=gradient;x.fillRect(barX,barY,barW,barH);x.strokeStyle='#64748b';x.lineWidth=Math.max(1,scale);x.strokeRect(barX,barY,barW,barH);
  const fontSize=Math.round(12*scale);x.font=fontSize+'px Arial';x.fillStyle=workbenchDark?'#e2e8f0':'#334155';x.textAlign='right';x.textBaseline='middle';x.fillText(fmt(P.colour_max),barX-Math.round(7*scale),barY);x.fillText(fmt(P.colour_min),barX-Math.round(7*scale),barY+barH);x.textBaseline='alphabetic';x.fillText((P.logarithmic?'log₁₀ ':'')+(P.component_label||'Field')+' ('+(P.unit||'')+')',viewerWidth-Math.round(18*scale),Math.max(fontSize+4,barY-Math.round(10*scale)));
  const videoT=videoTimelinePosition(),label='Video timeline '+formatPlaybackClock(videoT)+' / '+formatPlaybackClock(videoTimelineDuration);x.font='600 '+Math.round(14*scale)+'px Arial';x.textAlign='left';const pad=Math.round(7*scale),labelWidth=x.measureText(label).width;x.fillStyle=workbenchDark?'rgba(47,53,59,.90)':'rgba(255,255,255,.86)';x.fillRect(Math.round(18*scale),sceneHeight-Math.round(18*scale)-fontSize-pad*2,labelWidth+pad*2,fontSize+pad*2);x.fillStyle=workbenchDark?'#f1f5f9':'#334155';x.fillText(label,Math.round(18*scale)+pad,sceneHeight-Math.round(18*scale)-pad);
  if(traceHeight>0){x.drawImage(obsCanvas,0,0,viewerWidth,traceHeight,0,sceneHeight,viewerWidth,traceHeight);x.strokeStyle='#cbd5e1';x.lineWidth=Math.max(1,scale);x.beginPath();x.moveTo(0,sceneHeight+.5);x.lineTo(viewerWidth,sceneHeight+.5);x.stroke();const stats=String(obsStatsText.textContent||'');if(stats){const statsFont=Math.max(10,Math.round(11*Math.max(.85,height/1080)));x.font='600 '+statsFont+'px Arial';const maxWidth=Math.round(viewerWidth*.70),text=clippedCanvasText(x,stats,maxWidth),textWidth=x.measureText(text).width,statsPad=Math.max(4,Math.round(5*height/1080)),statsX=viewerWidth-Math.round(18*height/1080)-textWidth-statsPad*2,statsY=sceneHeight+Math.round(6*height/1080);x.fillStyle='rgba(255,255,255,.90)';x.fillRect(statsX,statsY,textWidth+statsPad*2,statsFont+statsPad*2);x.fillStyle='#475569';x.textAlign='left';x.textBaseline='top';x.fillText(text,statsX+statsPad,statsY+statsPad);}}
  if(tableWidth>0)drawBrainVideoTable(x,viewerWidth,tableWidth,height,index);
  return output;
}
window.fieldWorkbenchCaptureFrame=options=>{if(!previewMode||!options)return null;const width=Math.max(2,Math.min(7680,Math.round(Number(options.width)||0))),height=Math.max(2,Math.min(4320,Math.round(Number(options.height)||0))),index=Math.max(0,Math.min(N-1,Math.round(Number(options.frame_index)||0)));if(width%2||height%2)return null;if(options.camera&&!applyCamera(options.camera))return null;const tableWidth=hasBrainVideoTable?Math.max(420,Math.min(Math.round(width*.42),Math.round(width*.48))):0,viewerWidth=Math.max(2,width-tableWidth),traceHeight=hasObservationData?Math.max(2,Math.min(height-2,Math.round(height*exportTraceFraction()))):0,sceneHeight=height-traceHeight,requestedSequenceElapsed=Number(options.sequence_elapsed_s),requestedTimelineElapsed=Number(options.timeline_elapsed_s),sourceElapsed=Math.max(0,Number(frameTimes[index])-firstFrameTime),captureElapsed=Number.isFinite(requestedTimelineElapsed)?Math.max(0,requestedTimelineElapsed):(Number.isFinite(requestedSequenceElapsed)?Math.max(0,requestedSequenceElapsed):sourceElapsed);try{captureSize={width:viewerWidth,height:sceneHeight};observationCaptureSize=traceHeight?{width:viewerWidth,height:traceHeight}:null;if(scriptedTimeline){playbackElapsed=captureElapsed;waveformElapsed=Number.isFinite(requestedSequenceElapsed)?Math.max(0,requestedSequenceElapsed):sourceElapsed;}setFrame(index,captureElapsed,false);render();gl.finish();return exportFrameCanvas(index,width,height,viewerWidth,tableWidth,sceneHeight,traceHeight).toDataURL('image/png');}catch(_){return null;}finally{observationCaptureSize=null;captureSize=null;resize();render();}};
const obsColours=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2','#4f46e5','#be123c'];
function observationInverted(){return !!(obsInvert&&obsInvert.checked);}
function sequencerStateAt(cursorT,sequenceT){if(sequenceConfig){const index=sequenceStepAt(sequenceT);if(index===null)return null;if(index<0)return{step:0,active:'None',activeList:[],durationS:0};const step=sequenceConfig.steps[index]||{},labels=(step.active_coils||[]).map(String);return{step:index+1,active:labels.length?labels.join(', '):'None',activeList:labels,durationS:Math.max(0,Number(step.duration_s)||0)};}if(!sequencerTrace)return null;const ts=(sequencerTrace.time_s||[]).map(Number),steps=sequencerTrace.step_numbers||sequencerTrace.values||[],active=sequencerTrace.active_coils||[];if(!ts.length)return null;let lo=0,hi=ts.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(Number(ts[mid])<=cursorT)lo=mid;else hi=mid-1;}const step=Number(steps[lo])||0,activeText=String(active[lo]??'None')||'None',labels=activeText==='None'?[]:activeText.split(/,\s*/).filter(Boolean);return{step,active:activeText,activeList:labels,durationS:0};}
function updateObservationStats(cursorT,sequenceT){const parts=observationTraces.map(t=>{const s=t.stats||{},u=t.unit||'';return String(t.label||'Trace')+': Mean/DC '+fmt(Number(s.mean||0))+' '+u+', RMS '+fmt(Number(s.rms||0))+' '+u+', Pk-pk '+fmt(Number(s.pk_pk||0))+' '+u;});const state=sequencerStateAt(cursorT,sequenceT);if(state&&sequencerTrace)parts.push(state.step>0?'Sequencer: Step '+state.step+' — '+state.active:'Sequencer: Off — '+state.active);const text=parts.join('   |   ');obsStatsText.textContent=text;obsStatsText.title=text;}
function sequencerLaneHeight(){if(!sequencerTrace)return 0;let maxCoils=1;if(sequenceConfig)maxCoils=Math.max(1,...sequenceConfig.steps.map(step=>(step.active_coils||[]).length));else{for(const value of (sequencerTrace.active_coils||[])){const text=String(value??'None');if(text&&text!=='None')maxCoils=Math.max(maxCoils,text.split(/,\s*/).filter(Boolean).length);}}return 26+14*maxCoils;}
function setupObservation(){if(!hasObservationData)return;obsWrap.style.display='block';const laneExtra=Math.max(0,sequencerLaneHeight()-32);if(scrollObservationTraces&&!previewMode){const desired=Math.max(190,150+Math.max(0,observationTraces.length-1)*34+laneExtra),viewport=Math.min(300,desired);obsWrap.style.flexBasis=viewport+'px';obsWrap.style.overflowY=desired>viewport?'auto':'hidden';obsCanvas.style.height=desired+'px';}else{obsWrap.style.overflowY='hidden';obsCanvas.style.height='100%';obsWrap.style.flexBasis=previewMode?(exportTraceFraction()*100).toFixed(1)+'%':(sequencerTrace?(observationTraces.length?(230+laneExtra)+'px':(115+laneExtra)+'px'):'190px');}if(!observationTraces.length)document.getElementById('obs-invert-control').style.display='none';drawObservation(N?Number(frameTimes[frameIndex]):0,true,renderedSequenceElapsed);}
function traceRange(traces){let lo=Infinity,hi=-Infinity;for(const t of traces){for(const raw of (t.values||[])){const v=Number(raw);if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}}if(!Number.isFinite(lo)||!Number.isFinite(hi)){lo=0;hi=1;}if(lo===hi){const d=Math.max(1e-12,Math.abs(lo)*.05||1);lo-=d;hi+=d;}return[lo,hi];}
function sequenceNextBoundary(elapsed){
  if(!sequenceConfig)return Infinity;
  const steps=sequenceConfig.steps||[],cycle=Number(sequenceConfig.cycle_s)||steps.reduce((total,step)=>total+Math.max(0,Number(step.duration_s)||0),0),tolerance=Math.max(1e-12,cycle*1e-10);
  if(!(cycle>0)||!steps.length)return Infinity;
  const raw=Number(elapsed)||0,index=sequenceStepAt(raw);
  if(index===null||index<0)return Infinity;
  let end=0;for(let i=0;i<=index;i++)end+=Math.max(0,Number(steps[i].duration_s)||0);
  if(raw<0)return end;
  const loops=!!((scriptedTimeline&&stimulusLoop)||(!scriptedTimeline&&sequenceConfig.loop===true));
  if(!loops)return end;
  let phase=((raw%cycle)+cycle)%cycle;if(Math.abs(phase-cycle)<=tolerance||Math.abs(phase)<=tolerance)phase=0;
  const cycleStart=raw-phase,next=cycleStart+end;
  return next>raw+tolerance?next:next+cycle;
}
function sequencerLaneSamples(cursorT,sequenceT){
  if(!sequencerTrace)return null;
  const sourceTimes=sequencerTrace.time_s||[];
  if(!sourceTimes.length)return null;
  if(sequenceConfig){
    const requestedT0=Number(arguments[2]),requestedT1=Number(arguments[3]),fallbackT0=Number(sourceTimes[0]),fallbackT1=Number(sourceTimes[sourceTimes.length-1]),t0=Number.isFinite(requestedT0)?requestedT0:fallbackT0,t1=Number.isFinite(requestedT1)?requestedT1:fallbackT1;
    const ts=[],steps=[],active=[],activeLists=[],durations=[],span=Math.max(1e-12,t1-t0),epsilon=Math.max(1e-10,span*1e-10),maxSegments=4096;
    let traceT=t0,guard=0;
    while(traceT<t1-epsilon&&guard++<maxSegments){
      const sequenceAtTrace=sequenceT+(traceT-cursorT),state=sequencerStateAt(traceT,sequenceAtTrace);
      ts.push(traceT);steps.push(state?Number(state.step)||0:0);active.push(state?String(state.active||'None'):'None');activeLists.push(state&&Array.isArray(state.activeList)?state.activeList.slice():[]);durations.push(state?Math.max(0,Number(state.durationS)||0):0);
      const nextSequence=sequenceNextBoundary(sequenceAtTrace);
      if(!Number.isFinite(nextSequence))break;
      const delta=nextSequence-sequenceAtTrace;
      if(!(delta>epsilon)){traceT=Math.min(t1,traceT+epsilon);continue;}
      traceT=Math.min(t1,traceT+delta);
    }
    if(!ts.length){const state=sequencerStateAt(t0,sequenceT+(t0-cursorT));ts.push(t0);steps.push(state?Number(state.step)||0:0);active.push(state?String(state.active||'None'):'None');activeLists.push(state&&Array.isArray(state.activeList)?state.activeList.slice():[]);durations.push(state?Math.max(0,Number(state.durationS)||0):0);}
    return{ts,steps,active,activeLists,durations};
  }
  const ts=sourceTimes.map(Number),active=(sequencerTrace.active_coils||[]).map(String),activeLists=active.map(value=>value==='None'?[]:value.split(/,\s*/).filter(Boolean));
  return{ts,steps:(sequencerTrace.step_numbers||sequencerTrace.values||[]).map(Number),active,activeLists,durations:active.map(()=>0)};
}
function formatSequencerDuration(durationS){const ms=Math.max(0,Number(durationS)||0)*1000;if(ms<=0)return'';const rounded=Math.round(ms);return(Math.abs(ms-rounded)<.001?String(rounded):fmt(ms))+' ms';}
function drawSequencerLane(x,W,t0,t1,padL,padR,laneTop,laneHeight,cursorT,sequenceT){const lane=sequencerLaneSamples(cursorT,sequenceT,t0,t1);if(!lane)return;const ts=lane.ts,steps=lane.steps,active=lane.active,activeLists=lane.activeLists||[],durations=lane.durations||[],xFor=t=>padL+(t-t0)/(t1-t0||1)*(W-padL-padR),headerH=21,rowH=14;x.fillStyle='#f8fafc';x.fillRect(padL,laneTop,W-padL-padR,laneHeight);x.strokeStyle='#cbd5e1';x.strokeRect(padL,laneTop,W-padL-padR,laneHeight);let start=0;for(let i=1;i<=ts.length;i++){const changed=i===ts.length||Number(steps[i])!==Number(steps[start])||String(active[i]??'')!==String(active[start]??'');if(!changed)continue;const left=xFor(ts[start]),right=i<ts.length?xFor(ts[i]):xFor(t1),width=Math.max(0,right-left);x.fillStyle=(Number(steps[start])%2)?'rgba(37,99,235,.10)':'rgba(100,116,139,.10)';x.fillRect(left,laneTop,width,laneHeight);x.strokeStyle='rgba(100,116,139,.28)';x.beginPath();x.moveTo(left,laneTop);x.lineTo(left,laneTop+laneHeight);x.stroke();if(width>28){x.save();x.beginPath();x.rect(left+2,laneTop+1,Math.max(0,width-4),laneHeight-2);x.clip();const step=Number(steps[start])||0,duration=formatSequencerDuration(durations[start]),header=step>0?'Step '+step+(duration?' - '+duration:''):'Off';x.font='600 10px Arial';x.fillStyle='#334155';x.fillText(clippedCanvasText(x,header,Math.max(0,width-10)),left+5,laneTop+14);x.strokeStyle='rgba(100,116,139,.20)';x.beginPath();x.moveTo(left,laneTop+headerH);x.lineTo(right,laneTop+headerH);x.stroke();const labels=Array.isArray(activeLists[start])&&activeLists[start].length?activeLists[start]:['None'];x.font='10px Arial';for(let row=0;row<labels.length;row++){const y=laneTop+headerH+11+row*rowH;if(y>laneTop+laneHeight-3)break;x.fillStyle='#475569';x.fillText(clippedCanvasText(x,String(labels[row]),Math.max(0,width-10)),left+5,y);}x.restore();}start=i;}}
function drawObservation(cursorT,force=false,sequenceT=renderedSequenceElapsed){if(!hasObservationData)return;updateObservationStats(cursorT,sequenceT);const capture=observationCaptureSize,inverted=observationInverted(),dpr=capture?1:Math.min(2,window.devicePixelRatio||1),rect=capture||obsCanvas.getBoundingClientRect(),w=Math.max(1,Math.floor(rect.width*dpr)),h=Math.max(1,Math.floor(rect.height*dpr));if(force||obsCanvas.width!==w||obsCanvas.height!==h||!obsBase){obsCanvas.width=w;obsCanvas.height=h;const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d');x.scale(dpr,dpr);const W=rect.width,H=rect.height,fieldTraces=observationTraces.filter(t=>t.axis!=='excitation'),excitationTraces=observationTraces.filter(t=>t.axis==='excitation'),hasExcitation=excitationTraces.length>0,hasSignals=observationTraces.length>0,padL=62,padR=hasExcitation?62:18,padT=34+(scrollObservationTraces?Math.max(0,Math.ceil(observationTraces.length/3)-1)*15:0),padB=20,sequenceGap=sequencerTrace&&hasSignals?10:0,sequenceHeight=sequencerLaneHeight(),graphBottom=hasSignals?H-padB-sequenceHeight-sequenceGap:padT,allSources=sequencerTrace?[...observationTraces,sequencerTrace]:observationTraces,allTimes=allSources.flatMap(t=>(t.time_s||[]).map(Number).filter(Number.isFinite)),t0=allTimes.length?Math.min(...allTimes):0,t1=allTimes.length?Math.max(...allTimes):1,fieldRange=traceRange(fieldTraces),excRange=traceRange(excitationTraces);x.fillStyle='#fff';x.fillRect(0,0,W,H);const yFor=(v,range)=>{const f=(v-range[0])/(range[1]-range[0]||1);return padT+(inverted?f:1-f)*(graphBottom-padT);};if(hasSignals){x.strokeStyle='#e5e7eb';x.strokeRect(padL,padT,W-padL-padR,Math.max(1,graphBottom-padT));for(let ti=0;ti<observationTraces.length;ti++){const t=observationTraces[ti],ts=(t.time_s||[]).map(Number),vs=(t.values||[]).map(Number),range=t.axis==='excitation'?excRange:fieldRange;x.strokeStyle=obsColours[ti%obsColours.length];x.lineWidth=t.axis==='excitation'?1.8:1.5;x.setLineDash(t.axis==='excitation'?[6,3]:[]);x.beginPath();let moved=false;for(let i=0;i<Math.min(ts.length,vs.length);i++){if(!Number.isFinite(ts[i])||!Number.isFinite(vs[i]))continue;const px=padL+(ts[i]-t0)/(t1-t0||1)*(W-padL-padR),py=yFor(vs[i],range);if(moved)x.lineTo(px,py);else{x.moveTo(px,py);moved=true;}}x.stroke();x.setLineDash([]);}x.font='11px Arial';x.fillStyle='#475569';const fieldUnit=fieldTraces[0]?.unit||'',excUnit=excitationTraces[0]?.unit||'';if(fieldTraces.length){x.fillText(fmt(inverted?fieldRange[0]:fieldRange[1])+' '+fieldUnit,3,padT+4);x.fillText(fmt(inverted?fieldRange[1]:fieldRange[0])+' '+fieldUnit,3,graphBottom);}if(hasExcitation){x.textAlign='right';x.fillText(fmt(inverted?excRange[0]:excRange[1])+' '+excUnit,W-3,padT+4);x.fillText(fmt(inverted?excRange[1]:excRange[0])+' '+excUnit,W-3,graphBottom);x.textAlign='left';}let lx=padL,ly=18;for(let ti=0;ti<observationTraces.length;ti++){const t=observationTraces[ti],label=String(t.label||'Trace');x.fillStyle=obsColours[ti%obsColours.length];const short=label.length>34?label.slice(0,31)+'…':label,itemW=Math.min(220,Math.max(90,x.measureText(short).width+24));if(scrollObservationTraces&&lx>padL&&lx+itemW>W-padR){lx=padL;ly+=15;}x.fillText(short,lx,ly);lx+=itemW;if(!scrollObservationTraces&&lx>W-padR-90)break;}}
const laneTop=sequencerTrace?(hasSignals?graphBottom+sequenceGap:padT+16):0;if(sequencerTrace){x.font='11px Arial';x.fillStyle='#475569';x.fillText('Sequencer',padL,hasSignals?laneTop-2:16);}obsBase={canvas:c,t0,t1,padL,padR,padT,padB,W,H,dpr,graphBottom,laneTop,sequenceHeight};}const ctx=obsCanvas.getContext('2d');ctx.clearRect(0,0,obsCanvas.width,obsCanvas.height);ctx.drawImage(obsBase.canvas,0,0);ctx.save();ctx.scale(obsBase.dpr,obsBase.dpr);if(sequencerTrace)drawSequencerLane(ctx,obsBase.W,obsBase.t0,obsBase.t1,obsBase.padL,obsBase.padR,obsBase.laneTop,obsBase.sequenceHeight,cursorT,sequenceT);const px=obsBase.padL+(cursorT-obsBase.t0)/(obsBase.t1-obsBase.t0||1)*(obsBase.W-obsBase.padL-obsBase.padR);ctx.strokeStyle='#0f172a';ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(px,obsBase.padT);const cursorBottom=sequencerTrace?obsBase.laneTop+obsBase.sequenceHeight:obsBase.H-obsBase.padB;ctx.lineTo(px,cursorBottom);ctx.stroke();ctx.restore();}
obsInvert.onchange=()=>{obsBase=null;drawObservation(N?Number(frameTimes[frameIndex]):0,true,renderedSequenceElapsed);};
function tick(now){if(disposed)return;if(running&&N>1){let elapsed=Math.max(0,runningPlaybackElapsed(now));const waveDuration=Math.max(1e-30,waveformDuration),endDuration=Math.max(1e-30,timelineSourceDuration);if(elapsed>=endDuration){if(loop.checked){elapsed=((elapsed%endDuration)+endDuration)%endDuration;playbackElapsed=elapsed;waveformElapsed=scriptedTimeline?waveformElapsedAtScriptTime(elapsed):elapsed;lastAdvanceNow=Number(now)||performance.now();const phase=stimulusLoop?((waveformElapsed%waveDuration)+waveDuration)%waveDuration:Math.min(waveDuration,waveformElapsed),fraction=phase/waveDuration,i=Math.max(0,Math.min(N-1,Math.round(fraction*(N-1))));setFrame(i,elapsed);endedNaturally=false;}else{playbackElapsed=endDuration;waveformElapsed=scriptedTimeline?waveformElapsedAtScriptTime(endDuration):waveDuration;const phase=stimulusLoop?((waveformElapsed%waveDuration)+waveDuration)%waveDuration:Math.min(waveDuration,waveformElapsed),fraction=phase/waveDuration,i=Math.max(0,Math.min(N-1,Math.round(fraction*(N-1))));setFrame(i,endDuration);running=false;endedNaturally=true;play.disabled=false;updateStopButton();updatePerf();}}else{playbackElapsed=elapsed;if(!scriptedTimeline)waveformElapsed=elapsed;const phase=stimulusLoop?((waveformElapsed%waveDuration)+waveDuration)%waveDuration:Math.min(waveDuration,waveformElapsed),fraction=phase/waveDuration,i=Math.max(0,Math.min(N-1,Math.round(fraction*(N-1))));if(i!==frameIndex||sequenceStepAt(sequenceElapsedForFrame(i,elapsed))!==sequenceStepAt(renderedSequenceElapsed)||scriptedTimeline)setFrame(i,elapsed);}}if(dirty||running||dragging)render();if(!disposed)requestAnimationFrame(tick);}
window.fieldWorkbenchDispose=()=>{if(disposed)return true;disposed=true;running=false;dragging=false;try{if(gpuValidationProgram)gl.deleteProgram(gpuValidationProgram);[fieldProgram,solidProgram,lineProgram,volumeProgram].filter(Boolean).forEach(value=>gl.deleteProgram(value));[posBuf,fixedBuf,idxBuf,gridBuffer,overlayPosBuf,overlayFixedBuf,brainHighlightBuf].filter(Boolean).forEach(value=>gl.deleteBuffer(value));basisBufs.filter(Boolean).forEach(value=>gl.deleteBuffer(value));overlayBasisBufs.filter(Boolean).forEach(value=>gl.deleteBuffer(value));sceneMeshes.forEach(value=>{if(value.pbuf)gl.deleteBuffer(value.pbuf);if(value.ibuf)gl.deleteBuffer(value.ibuf);});sceneLines.forEach(value=>{if(value.pbuf)gl.deleteBuffer(value.pbuf);});scenePoints.forEach(value=>{if(value.pbuf)gl.deleteBuffer(value.pbuf);});[volumeFixedTex,...volumeBasisTex].filter(Boolean).forEach(value=>gl.deleteTexture(value));gl.bindBuffer(gl.ARRAY_BUFFER,null);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,null);gl.bindTexture(gl.TEXTURE_3D,null);gl.flush();}catch(_error){}return true;};
document.addEventListener('freeze',()=>{if(running)stopRun();});
window.addEventListener('resize',()=>{if(disposed)return;obsBase=null;dirty=true;drawObservation(N?Number(frameTimes[frameIndex]):0,true,renderedSequenceElapsed);});
window.fieldWorkbenchSetTheme=theme=>{workbenchDark=String(theme||'light').toLowerCase()==='dark';document.documentElement.setAttribute('data-workbench-theme',workbenchDark?'dark':'light');if(disposed)return;obsBase=null;dirty=true;resize();render();drawObservation(N?Number(frameTimes[frameIndex]):0,true,renderedSequenceElapsed);};
window.fieldWorkbenchWaveformWake=()=>{if(disposed)return;obsBase=null;dirty=true;resize();render();drawObservation(N?Number(frameTimes[frameIndex]):0,true,renderedSequenceElapsed);};
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&!disposed)requestAnimationFrame(window.fieldWorkbenchWaveformWake);});
window.addEventListener('pagehide',()=>window.fieldWorkbenchDispose(),{once:true});
setupObservation();updateTime();updateStopButton();updatePerf();render();requestAnimationFrame(tick);
})();
</script>
</body>
</html>'''
        return template.replace("__PAYLOAD__", data)
