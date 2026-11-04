"""Shared result-panel lifecycle and optional waveform tools for map windows."""
from __future__ import annotations

import copy
import math
import time
import threading
from datetime import datetime, timezone
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QRadioButton,
    QProgressBar, QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QPlainTextEdit, QTreeWidget, QTreeWidgetItem, QWidget,
)
from .exposure_analysis_header import ExposureAnalysisHeaderPanel
from .analysis_region_viewer import AnalysisRegionViewer
from .brain_analysis import lpba40_region_hierarchy
from .spatial_analysis import (
    AnalysisRegionNode, RegionalAnalysisError, analysis_region_hierarchy_from_payload,
)
from .exposure_panel import ExposureRegionPanel, ExposureResultTab, REGION_KEY_ROLE
from .region_analysis import (
    FFT_WINDOWS, FREQUENCY_WEIGHTING_PROFILES, MeasurementRegions,
    analyse_regions_exposure_summary, analyse_regions_spectrum, analyse_regions_waveform,
    analyse_regions_threshold_duration, analyse_regions_frequency_weighted,
)
from .plot_view import PlotView
from .window_utils import centre_window_on_parent, configure_application_window, enable_standard_window_controls


def result_view(widget):
    return widget.viewer if isinstance(widget, ExposureResultTab) else widget


def _long_authoritative_analysis(payload):
    """Return True when a page may need to replay a reduced long exposure."""
    return bool(
        isinstance(payload, dict)
        and payload.get("analysis_trace_reduced", False)
        and isinstance(payload.get("authoritative_analysis"), dict)
    )


class _AnalysisCancelled(RuntimeError):
    pass


class _AnalysisWorker(QObject):
    """Honor pause/cancel controls at numerical progress boundaries and before delivery."""
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(object)

    def __init__(self, function, cancel_event=None, pause_event=None):
        super().__init__()
        self.function = function
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.pause_event = pause_event if pause_event is not None else threading.Event()
        self._last_progress = 0.
        self._last_message = None
        self._pause_announced = False

    def _wait_if_paused(self):
        if not self.pause_event.is_set():
            return
        if not self._pause_announced:
            self.progress.emit({"_analysis_pause_state": "paused"})
            self._pause_announced = True
        while self.pause_event.is_set():
            if self.cancel_event.wait(.05):
                raise _AnalysisCancelled()
        if self.cancel_event.is_set():
            raise _AnalysisCancelled()
        if self._pause_announced:
            self.progress.emit({"_analysis_pause_state": "resumed"})
            self._pause_announced = False

    def _progress(self, update):
        if self.cancel_event.is_set():
            raise _AnalysisCancelled()
        self._wait_if_paused()
        if self.cancel_event.is_set():
            raise _AnalysisCancelled()
        now = time.monotonic()
        message = str(update.get("message", update.get("stage", ""))) if isinstance(update, dict) else str(update)
        if now - self._last_progress >= .05 or message != self._last_message:
            self.progress.emit(update)
            self._last_progress, self._last_message = now, message

    @Slot()
    def run(self):
        try:
            if self.cancel_event.is_set():
                raise _AnalysisCancelled()
            self._wait_if_paused()
            result = self.function(self._progress)
            if self.cancel_event.is_set():
                raise _AnalysisCancelled()
            self._wait_if_paused()
        except _AnalysisCancelled:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 - worker boundary
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class _AnalysisDelivery(QObject):
    """GUI-thread gate: Cancel also wins over an already queued completion."""
    def __init__(self, owner, event):
        super().__init__(owner)
        self.owner, self.cancel_event = owner, event

    @Slot(object)
    def completed(self, result):
        if self.cancel_event.is_set():
            self.cancelled()
            return
        self.owner._analysis_completed(result)
        self._finish()

    @Slot(str)
    def failed(self, message):
        if self.cancel_event.is_set():
            self.cancelled()
            return
        self.owner._analysis_failed(message)
        self._finish()

    @Slot()
    def cancelled(self):
        owner = self.owner
        elapsed = _analysis_elapsed_seconds(owner)
        owner._analysis_last_elapsed_s = float(elapsed)
        owner._set_analysis_busy(False)
        owner._analysis_pause_started_at = None
        owner._analysis_progress_state = "cancelled"
        owner._analysis_progress_message = "Cancelled — saved results kept"
        owner.progress_label.setText(owner._analysis_progress_message)
        owner.progress_bar.setRange(0, 1)
        owner.progress_bar.setValue(0)
        owner._analysis_progress_time_text = f"Stopped after {_format_analysis_elapsed(elapsed)}"
        owner.analysis_time_label.setText(owner._analysis_progress_time_text)
        _publish_analysis_status(owner)
        self._finish()

    def _finish(self):
        finished = getattr(self.owner, "_analysis_job_finished_callback", None)
        if callable(finished):
            finished(self.owner)
        self.deleteLater()

    @Slot(object)
    def progress(self, update):
        if self.cancel_event.is_set():
            return
        if isinstance(update, dict) and update.get("_analysis_pause_state"):
            state = str(update.get("_analysis_pause_state"))
            if state == "paused":
                _set_analysis_pause_acknowledged(self.owner, True)
            elif state == "resumed":
                _set_analysis_pause_acknowledged(self.owner, False)
            return
        self.owner._analysis_progress(update)


# Keep threads alive until they exit, even if a parent window is destroyed.
_ACTIVE_ANALYSIS_JOBS = {}


def _request_analysis_cancellation(owner):
    cancel_requested = getattr(owner, "_analysis_cancel_requested", None)
    if not getattr(owner, "_analysis_busy", False):
        # A page-local logical queue can briefly be between worker submissions.
        # Closing/cancelling in that gap must still discard its pending work.
        if callable(cancel_requested) and getattr(owner, "_weighted_pending_requests", None):
            cancel_requested()
        return
    if callable(cancel_requested):
        cancel_requested()
    if str(getattr(owner, "_analysis_progress_state", "")) == "queued":
        cancel_queued = getattr(owner, "_analysis_queue_cancel_callback", None)
        if callable(cancel_queued):
            cancel_queued(owner)
        return
    event = getattr(owner, "_analysis_cancel_event", None)
    if event is None:
        return
    event.set()
    pause_event = getattr(owner, "_analysis_pause_event", None)
    if pause_event is not None:
        pause_event.clear()
    owner._analysis_progress_state = "cancelling"
    owner._analysis_progress_message = "Cancelling after the current calculation block…"
    owner.progress_label.setText(owner._analysis_progress_message)
    _publish_analysis_status(owner)


def _toggle_analysis_pause(owner):
    """Pause/resume one background analysis at its next progress boundary."""
    pause_event = getattr(owner, "_analysis_pause_event", None)
    cancel_event = getattr(owner, "_analysis_cancel_event", None)
    if (not getattr(owner, "_analysis_busy", False) or pause_event is None
            or (cancel_event is not None and cancel_event.is_set())):
        return
    if pause_event.is_set():
        pause_event.clear()
        owner._analysis_progress_state = "running"
        owner._analysis_progress_message = str(
            getattr(owner, "_analysis_running_message", "") or "Calculating…"
        )
    else:
        pause_event.set()
        owner._analysis_progress_state = "pausing"
        owner._analysis_progress_message = "Pausing after the current calculation block…"
    owner.progress_label.setText(owner._analysis_progress_message)
    _publish_analysis_status(owner)


def _set_analysis_pause_acknowledged(owner, paused):
    now = time.monotonic()
    if paused:
        if getattr(owner, "_analysis_pause_started_at", None) is None:
            owner._analysis_pause_started_at = now
        owner._analysis_progress_state = "paused"
        running_message = str(getattr(owner, "_analysis_running_message", "") or "Calculating…")
        owner._analysis_progress_message = f"Paused — {running_message}"
    else:
        pause_started = getattr(owner, "_analysis_pause_started_at", None)
        if pause_started is not None:
            owner._analysis_paused_total_s = float(getattr(owner, "_analysis_paused_total_s", 0.0)) + max(
                0.0, now - float(pause_started)
            )
        owner._analysis_pause_started_at = None
        owner._analysis_progress_state = "running"
        owner._analysis_progress_message = str(
            getattr(owner, "_analysis_running_message", "") or "Calculating…"
        )
    owner.progress_label.setText(owner._analysis_progress_message)
    _refresh_analysis_elapsed(owner)


def _launch_analysis_job(owner, function):
    """Launch one calculation immediately. Queueing is handled by the target view."""
    thread = QThread()
    event = threading.Event()
    pause_event = threading.Event()
    owner._analysis_cancel_event = event
    owner._analysis_pause_event = pause_event
    worker = _AnalysisWorker(function, event, pause_event)
    delivery = _AnalysisDelivery(owner, event)
    worker.moveToThread(thread)
    _ACTIVE_ANALYSIS_JOBS[thread] = worker
    owner.destroyed.connect(lambda *_: (event.set(), pause_event.clear()))
    thread.started.connect(worker.run)
    worker.completed.connect(delivery.completed)
    worker.failed.connect(delivery.failed)
    worker.cancelled.connect(delivery.cancelled)
    worker.progress.connect(delivery.progress)
    for terminal in (worker.completed, worker.failed, worker.cancelled):
        terminal.connect(worker.deleteLater)
        terminal.connect(thread.quit)

    def cleanup():
        _ACTIVE_ANALYSIS_JOBS.pop(thread, None)
        thread.deleteLater()

    thread.finished.connect(cleanup)
    thread.start()
    _publish_analysis_status(owner)
    return thread


def _start_analysis_job(owner, function):
    """Submit a calculation through the owning target's single-worker queue when available."""
    scheduler = getattr(owner, "_analysis_job_scheduler", None)
    if not callable(scheduler) and isinstance(owner, QWidget):
        ancestor = owner.parentWidget()
        while ancestor is not None:
            scheduler = getattr(ancestor, "_schedule_analysis_job", None)
            if callable(scheduler):
                break
            ancestor = ancestor.parentWidget()
    if callable(scheduler):
        return scheduler(owner, function)
    return _launch_analysis_job(owner, function)


def _format_analysis_elapsed(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}:{remainder:04.1f}"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{remainder:04.1f}"


def _publish_analysis_status(owner):
    busy = bool(getattr(owner, "_analysis_busy", False))
    state = str(getattr(owner, "_analysis_progress_state", "idle"))
    cancel_event = getattr(owner, "_analysis_cancel_event", None)
    pause_event = getattr(owner, "_analysis_pause_event", None)
    queued_cancel = getattr(owner, "_analysis_queue_cancel_callback", None)
    can_cancel = busy and (
        (state == "queued" and callable(queued_cancel))
        or (cancel_event is not None and not cancel_event.is_set())
    )
    can_pause = state != "queued" and can_cancel and pause_event is not None

    cancel = getattr(owner, "analysis_cancel_button", None)
    if cancel is not None:
        cancel.setVisible(busy and cancel_event is not None)
        cancel.setEnabled(can_cancel)
    pause = getattr(owner, "analysis_pause_button", None)
    if pause is not None:
        pause.setVisible(busy and pause_event is not None)
        pause.setEnabled(can_pause)
        pause.setText("Run" if pause_event is not None and pause_event.is_set() else "Pause")

    callback = getattr(owner, "_analysis_status_callback", None)
    if not callable(callback):
        return
    bar = getattr(owner, "progress_bar", None)
    callback({
        "state": state,
        "message": str(getattr(owner, "_analysis_progress_message", "")),
        "time_text": str(getattr(owner, "_analysis_progress_time_text", "")),
        "minimum": int(bar.minimum()) if isinstance(bar, QProgressBar) else 0,
        "maximum": int(bar.maximum()) if isinstance(bar, QProgressBar) else 0,
        "value": int(bar.value()) if isinstance(bar, QProgressBar) else 0,
        "pause_requested": bool(pause_event is not None and pause_event.is_set()),
        "can_pause": bool(can_pause),
        "can_cancel": bool(can_cancel),
    })


def _analysis_elapsed_seconds(owner):
    started = getattr(owner, "_analysis_progress_started_at", None)
    if started is None:
        return 0.0
    now = time.monotonic()
    paused = float(getattr(owner, "_analysis_paused_total_s", 0.0) or 0.0)
    pause_started = getattr(owner, "_analysis_pause_started_at", None)
    if pause_started is not None:
        paused += max(0.0, now - float(pause_started))
    return max(0.0, now - float(started) - paused)


def _refresh_analysis_elapsed(owner):
    if str(getattr(owner, "_analysis_progress_state", "")) not in {
        "running", "pausing", "paused", "cancelling",
    }:
        return
    if getattr(owner, "_analysis_progress_started_at", None) is None:
        return
    elapsed = _analysis_elapsed_seconds(owner)
    text = f"Elapsed {_format_analysis_elapsed(elapsed)}"
    owner._analysis_progress_time_text = text
    time_label = getattr(owner, "analysis_time_label", None)
    if isinstance(time_label, QLabel):
        time_label.setText(text)
    _publish_analysis_status(owner)


def _add_analysis_progress(root, parent):
    """Add a reusable progress/status row used by analysis pages.

    When a page is embedded in the Exposure Analysis workspace this local row is
    hidden and mirrored into the target-view footer while work is active.
    """
    container = QWidget(parent)
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    label = QLabel("", container)
    label.setObjectName("hint")
    bar = QProgressBar(container)
    bar.setTextVisible(False)
    bar.setRange(0, 0)
    time_label = QLabel("", container)
    time_label.setObjectName("hint")
    time_label.setMinimumWidth(112)
    time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(label)
    row.addWidget(bar, 1)
    row.addWidget(time_label)
    pause = QPushButton("Pause", container)
    pause.clicked.connect(lambda: _toggle_analysis_pause(parent))
    pause.hide()
    row.addWidget(pause)
    cancel = QPushButton("Cancel", container)
    cancel.clicked.connect(lambda: _request_analysis_cancellation(parent))
    cancel.hide()
    row.addWidget(cancel)
    parent.analysis_pause_button = pause
    parent.analysis_cancel_button = cancel
    root.addWidget(container)
    container.hide()

    parent._analysis_progress_container = container
    parent.analysis_time_label = time_label
    parent._analysis_progress_state = "idle"
    parent._analysis_progress_message = ""
    parent._analysis_progress_time_text = ""
    parent._analysis_progress_started_at = None
    parent._analysis_pause_started_at = None
    parent._analysis_paused_total_s = 0.0
    parent._analysis_running_message = ""
    parent._analysis_pause_event = None
    parent._analysis_cancel_event = None
    parent._analysis_last_elapsed_s = None
    parent._analysis_progress_embedded = False
    parent._analysis_cache_changed_callback = None
    parent._analysis_status_callback = None
    parent._analysis_job_scheduler = None
    parent._analysis_job_finished_callback = None
    parent._analysis_queue_cancel_callback = None
    timer = QTimer(parent)
    timer.setInterval(200)
    timer.timeout.connect(lambda p=parent: _refresh_analysis_elapsed(p))
    parent._analysis_progress_timer = timer
    label._analysis_owner = parent
    return label, bar


def _set_analysis_progress(label, bar, visible, message="Calculating…", update=None):
    owner = getattr(label, "_analysis_owner", None)
    if not visible:
        _finish_analysis_progress(label, bar)
        return

    if owner is not None and str(getattr(owner, "_analysis_progress_state", "")) not in {
        "running", "pausing", "paused", "cancelling",
    }:
        owner._analysis_progress_started_at = time.monotonic()
        owner._analysis_pause_started_at = None
        owner._analysis_paused_total_s = 0.0
        timer = getattr(owner, "_analysis_progress_timer", None)
        if isinstance(timer, QTimer):
            timer.start()
    if owner is not None:
        pause_event = getattr(owner, "_analysis_pause_event", None)
        if pause_event is None or not pause_event.is_set():
            owner._analysis_progress_state = "running"

    text = str(message)
    completed = total = None
    if isinstance(update, dict):
        text = str(update.get("message") or update.get("stage") or message)
        try:
            completed = int(update.get("completed"))
            total = int(update.get("total"))
        except (TypeError, ValueError):
            completed = total = None
    label.setText(text)
    if owner is not None:
        owner._analysis_running_message = text
    if completed is not None and total is not None and total > 0:
        maximum = min(total, 1_000_000)
        bar.setRange(0, maximum)
        bar.setValue(max(0, min(maximum, int(completed / total * maximum))))
    else:
        bar.setRange(0, 0)
    if owner is not None:
        if str(getattr(owner, "_analysis_progress_state", "")) not in {"pausing", "paused"}:
            owner._analysis_progress_message = text
        container = getattr(owner, "_analysis_progress_container", None)
        if isinstance(container, QWidget) and not bool(getattr(owner, "_analysis_progress_embedded", False)):
            container.show()
        _refresh_analysis_elapsed(owner)
    else:
        label.show()
        bar.show()


def _finish_analysis_progress(label, bar):
    owner = getattr(label, "_analysis_owner", None)
    if owner is None:
        label.hide()
        bar.hide()
        return
    if str(getattr(owner, "_analysis_progress_state", "")) != "running":
        return
    timer = getattr(owner, "_analysis_progress_timer", None)
    if isinstance(timer, QTimer):
        timer.stop()
    elapsed = _analysis_elapsed_seconds(owner)
    owner._analysis_last_elapsed_s = float(elapsed)
    owner._analysis_pause_started_at = None
    owner._analysis_progress_state = "completed"
    owner._analysis_progress_message = "Completed"
    owner._analysis_progress_time_text = f"Processing time {_format_analysis_elapsed(elapsed)}"
    label.setText(owner._analysis_progress_message)
    bar.setRange(0, 1)
    bar.setValue(1)
    owner.analysis_time_label.setText(owner._analysis_progress_time_text)
    if not bool(getattr(owner, "_analysis_progress_embedded", False)):
        owner._analysis_progress_container.show()
    _publish_analysis_status(owner)


# Each saved Exposure Analysis record already identifies one prepared target.
# These keys identify one derived settings configuration within that record.
def _regional_config_key(request_key):
    return (request_key[0], "__all_regions__", *request_key[2:])


def _analysis_cached_result(cache, request_key):
    if request_key and request_key[0] in ("summary", "spectrum", "timing", "threshold", "weighted"):
        group = cache.get(_regional_config_key(request_key))
        if isinstance(group, dict):
            return group.get("regions", {}).get(str(request_key[1]))
    return cache.get(request_key)


def _start_regional_analysis(owner, request_key):
    owner._analysis_request_key = request_key
    owner._set_analysis_busy(True)

    def calculate(progress):
        kind = request_key[0]
        if kind == "summary":
            return analyse_regions_exposure_summary(owner.payload, progress_callback=progress)
        if kind == "spectrum":
            return analyse_regions_spectrum(owner.payload, fft_samples=request_key[2],
                window=request_key[3], detrend=request_key[4], progress_callback=progress)
        if kind == "threshold":
            return analyse_regions_threshold_duration(owner.payload, request_key[2], progress_callback=progress)
        return analyse_regions_waveform(owner.payload, threshold_percent=request_key[2],
                                        progress_callback=progress)

    _start_analysis_job(owner, calculate)


def _complete_regional_analysis(owner, results, apply, *, all_regions=False):
    request_key = owner._analysis_request_key
    owner._set_analysis_busy(False)
    # Publish a configuration atomically only when every prepared region succeeded.
    if (not isinstance(results, dict) or set(results) != set(owner.payload["regions"])
            or not all(isinstance(value, dict) for value in results.values())):
        owner._analysis_failed("The calculation did not return all regions. Return to the analysis home and reopen this method to try again.")
        return
    _store_analysis_cache_result(owner, _regional_config_key(request_key), {"regions": results})
    current = owner._request_key()
    if current and _regional_config_key(current) == _regional_config_key(request_key):
        apply(results if all_regions else results[str(current[1])])
    elif not all_regions:
        refresh = getattr(owner, "refresh", None) or getattr(owner, "recalculate", None)
        if callable(refresh):
            QTimer.singleShot(0, refresh)


def _analysis_cache_meta_key(request_key):
    if not isinstance(request_key, tuple):
        return None
    return ("__cache_meta__",) + tuple(request_key)


def _analysis_cache_metadata(cache, request_key):
    if not isinstance(cache, dict):
        return {}
    if request_key and _regional_config_key(request_key) in cache:
        request_key = _regional_config_key(request_key)
    meta_key = _analysis_cache_meta_key(request_key)
    value = cache.get(meta_key) if meta_key is not None else None
    return value if isinstance(value, dict) else {}


def _analysis_cache_result_keys(cache):
    """Return persisted derived-result keys, excluding reserved cache metadata rows."""
    if not isinstance(cache, dict):
        return ()
    return tuple(
        key for key in cache
        if isinstance(key, tuple) and key and str(key[0]) != "__cache_meta__"
    )


_WEIGHTED_FULL_ARRAY_KEYS = {
    "weighted_spatial_rms_uT",
    "frequency_hz",
    "bin_rms_uT",
    "reference_rms_by_bin_uT",
    "spectral_ratio_by_bin",
}


def _compact_frequency_weighted_result(result):
    """Return the persistent/display form of one frequency-weighted region result.

    The weighting calculation needs the full spectrum while it is running, but the
    persistent weighted cache does not. A matching Spectrum/FFT cache already owns
    those full spectral arrays. Keep authoritative scalar results at float64
    precision, one bounded normalized display trace, and only the strongest 30
    spectral rows required by the results table.
    """
    if not isinstance(result, dict):
        return result
    compact = {
        key: value for key, value in result.items()
        if key not in _WEIGHTED_FULL_ARRAY_KEYS
    }

    ratio = np.asarray(result.get("spectral_ratio_by_bin", ()), dtype=np.float64)
    frequency = np.asarray(result.get("frequency_hz", ()), dtype=np.float64)
    bin_rms = np.asarray(result.get("bin_rms_uT", ()), dtype=np.float64)
    refs = np.asarray(result.get("reference_rms_by_bin_uT", ()), dtype=np.float64)
    if ratio.ndim == frequency.ndim == bin_rms.ndim == refs.ndim == 1 and (
        len(ratio) == len(frequency) == len(bin_rms) == len(refs) and len(ratio)
    ):
        indices = [
            int(i) for i in np.argsort(ratio)[::-1][:30]
            if ratio[i] > 1e-15 and np.isfinite(refs[i])
        ]
        compact["spectral_top_rows"] = (
            np.column_stack((frequency[indices], bin_rms[indices], refs[indices], ratio[indices]))
            if indices else np.empty((0, 4), dtype=np.float64)
        )
    else:
        existing_rows = np.asarray(result.get("spectral_top_rows", ()), dtype=np.float64)
        compact["spectral_top_rows"] = (
            existing_rows if existing_rows.ndim == 2 and existing_rows.shape[1:] == (4,)
            else np.empty((0, 4), dtype=np.float64)
        )

    # This trace is explicitly a bounded plot preview; the scalar peak was found
    # using the full analysed samples. Float32 is ample for screen plotting and
    # halves the only profile/region-specific large array retained in this cache.
    compact["weighted_index"] = np.asarray(
        result.get("weighted_index", ()), dtype=np.float32
    )
    compact["time_s"] = np.asarray(result.get("time_s", ()), dtype=np.float64)
    compact["weighted_cache_format"] = "compact-v1"
    return compact


def _compact_frequency_weighted_cache_group(data):
    """Compact all regional results in a weighted all-regions cache row."""
    if not isinstance(data, dict) or not isinstance(data.get("regions"), dict):
        return data
    return {
        **data,
        "regions": {
            str(key): _compact_frequency_weighted_result(value)
            for key, value in data["regions"].items()
        },
    }


def _analysis_home_cache_statuses(cache):
    """Summarize target-wide derived caches for the descriptive analysis home."""
    keys = [
        key for key in _analysis_cache_result_keys(cache)
        if len(key) >= 2 and str(key[1]) == "__all_regions__"
    ]

    def count(kind):
        return sum(1 for key in keys if str(key[0]) == kind)

    def configured_status(kind):
        total = count(kind)
        if total <= 0:
            return "Not calculated"
        if total == 1:
            return "Cached"
        return f"Cached • {total:,} settings"

    return {
        "summary": "Cached" if count("summary") else "Not calculated",
        "spectrum": configured_status("spectrum"),
        "timing": configured_status("timing"),
        "threshold": configured_status("threshold"),
        "weighted": configured_status("weighted"),
    }


def _store_analysis_cache_result(owner, request_key, data, *, notify=True, elapsed_s=None):
    """Store one derived result and notify the persistent Exposure Analysis target."""
    if request_key is None or not isinstance(getattr(owner, "analysis_cache", None), dict):
        return
    if isinstance(request_key, tuple) and request_key and str(request_key[0]) == "weighted":
        data = _compact_frequency_weighted_cache_group(data)
    owner.analysis_cache[request_key] = data
    meta_key = _analysis_cache_meta_key(request_key)
    if meta_key is not None:
        elapsed = elapsed_s if elapsed_s is not None else getattr(owner, "_analysis_last_elapsed_s", None)
        metadata = {
            "cached_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            elapsed_value = float(elapsed)
        except (TypeError, ValueError):
            elapsed_value = float("nan")
        if np.isfinite(elapsed_value) and elapsed_value >= 0.0:
            metadata["processing_time_s"] = elapsed_value
        owner.analysis_cache[meta_key] = metadata
    callback = getattr(owner, "_analysis_cache_changed_callback", None)
    if notify and callable(callback):
        callback(request_key)


def _set_analysis_cached(label, bar, cache=None, request_key=None):
    owner = getattr(label, "_analysis_owner", None)
    if owner is None:
        return
    timer = getattr(owner, "_analysis_progress_timer", None)
    if isinstance(timer, QTimer):
        timer.stop()
    owner._analysis_progress_started_at = None
    owner._analysis_pause_started_at = None
    owner._analysis_paused_total_s = 0.0
    owner._analysis_progress_state = "cached"
    owner._analysis_progress_message = "Loaded saved result"
    meta = _analysis_cache_metadata(cache, request_key)
    try:
        original_elapsed = float(meta.get("processing_time_s"))
    except (TypeError, ValueError):
        original_elapsed = float("nan")
    if np.isfinite(original_elapsed) and original_elapsed >= 0.0:
        owner._analysis_progress_time_text = (
            f"Original processing time {_format_analysis_elapsed(original_elapsed)}"
        )
    else:
        owner._analysis_progress_time_text = "No recalculation"
    label.setText(owner._analysis_progress_message)
    bar.setRange(0, 1)
    bar.setValue(1)
    owner.analysis_time_label.setText(owner._analysis_progress_time_text)
    if not bool(getattr(owner, "_analysis_progress_embedded", False)):
        owner._analysis_progress_container.show()
    _publish_analysis_status(owner)


def _set_analysis_queued(owner):
    """Keep a submitted page busy while it waits for the target's one worker slot."""
    timer = getattr(owner, "_analysis_progress_timer", None)
    if isinstance(timer, QTimer):
        timer.stop()
    owner._analysis_progress_started_at = None
    owner._analysis_pause_started_at = None
    owner._analysis_paused_total_s = 0.0
    owner._analysis_cancel_event = None
    owner._analysis_pause_event = None
    owner._analysis_progress_state = "queued"
    owner._analysis_progress_message = "Queued — waiting for current calculation"
    owner._analysis_progress_time_text = "Waiting"
    owner.progress_label.setText(owner._analysis_progress_message)
    owner.progress_bar.setRange(0, 1)
    owner.progress_bar.setValue(0)
    owner.analysis_time_label.setText(owner._analysis_progress_time_text)
    refresh_profiles = getattr(owner, "_refresh_profiles", None)
    if callable(refresh_profiles):
        refresh_profiles()
    _publish_analysis_status(owner)


def _set_analysis_failed(label, bar):
    owner = getattr(label, "_analysis_owner", None)
    if owner is None:
        return
    timer = getattr(owner, "_analysis_progress_timer", None)
    if isinstance(timer, QTimer):
        timer.stop()
    elapsed = _analysis_elapsed_seconds(owner)
    owner._analysis_last_elapsed_s = float(elapsed)
    owner._analysis_pause_started_at = None
    owner._analysis_progress_state = "failed"
    owner._analysis_progress_message = "Analysis failed"
    owner._analysis_progress_time_text = f"Stopped after {_format_analysis_elapsed(elapsed)}"
    label.setText(owner._analysis_progress_message)
    bar.setRange(0, 1)
    bar.setValue(0)
    owner.analysis_time_label.setText(owner._analysis_progress_time_text)
    if not bool(getattr(owner, "_analysis_progress_embedded", False)):
        owner._analysis_progress_container.show()
    _publish_analysis_status(owner)


def _make_analysis_region_tree(owner):
    tree = QTreeWidget(owner)
    tree.setHeaderLabels(["Region"])
    tree.setMinimumWidth(185)
    tree.setUniformRowHeights(True)
    items = {}
    nodes = {}

    def metadata_lines(node):
        metadata = node.metadata if isinstance(getattr(node, "metadata", None), dict) else {}
        lines = []
        kind = str(getattr(node, "region_kind", "") or "")
        if kind == "near_surface_volume":
            lines.append("Near-surface volume")
        elif kind == "near_surface_volume_set":
            lines.append("Near-surface region set")
        try:
            depth = float(metadata.get("sampling_depth_mm"))
        except (TypeError, ValueError):
            depth = float("nan")
        if math.isfinite(depth) and depth > 0.0:
            lines.append(f"Sampling depth: {depth:,.6g} mm inward")
        try:
            volume = float(metadata.get("volume_cm3"))
        except (TypeError, ValueError):
            volume = float("nan")
        if math.isfinite(volume) and volume > 0.0:
            lines.append(f"Sampled volume: {volume:,.6g} cm³")
        try:
            area = float(metadata.get("surface_area_cm2"))
        except (TypeError, ValueError):
            area = float("nan")
        if math.isfinite(area) and area > 0.0:
            lines.append(f"Surface area: {area:,.6g} cm²")
        source_count = metadata.get("source_voxel_count")
        retained_count = metadata.get("retained_sample_count")
        try:
            source_count = int(source_count)
        except (TypeError, ValueError):
            source_count = 0
        try:
            retained_count = int(retained_count)
        except (TypeError, ValueError):
            retained_count = 0
        if source_count > 0:
            lines.append(f"Source voxels: {source_count:,}")
        if retained_count > 0:
            lines.append(f"Retained field samples: {retained_count:,}")
        return lines

    def add_node(node, parent):
        children = [add_node(child, None) for child in node.children]
        children = [child for child in children if child is not None]
        if node.key not in owner.payload["regions"] and not children:
            return None
        item = QTreeWidgetItem([node.name])
        details = metadata_lines(node)
        if details:
            item.setToolTip(0, node.name + "\n" + "\n".join(details))
        if node.key in owner.payload["regions"]:
            item.setData(0, Qt.ItemDataRole.UserRole, node.key)
            items[node.key] = item
            nodes[node.key] = node
        else:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        for child in children:
            item.addChild(child)
        if parent is not None:
            parent.addChild(item)
        return item

    context = owner.payload.get("exposure_analysis_context") if isinstance(owner.payload, dict) else None
    context = context if isinstance(context, dict) else {}
    hierarchy_payload = context.get("region_hierarchy")
    try:
        if isinstance(hierarchy_payload, dict):
            hierarchy = analysis_region_hierarchy_from_payload(hierarchy_payload)
        else:
            hierarchy = lpba40_region_hierarchy()
        root = add_node(hierarchy, None)
    except (RegionalAnalysisError, OSError, ValueError):
        root = None
    if root is not None:
        tree.addTopLevelItem(root)
        root.setExpanded(True)
    for key, entry in owner.payload["regions"].items():
        if key not in items:
            item = QTreeWidgetItem([entry["name"]])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            tree.addTopLevelItem(item)
            items[key] = item
    tree._region_items = items
    tree._region_nodes = nodes
    tree.currentItemChanged.connect(owner._tree_region_changed)
    return tree



class _CancellableAnalysisPage(QWidget):
    """Embedded analysis page with cooperative pause/cancel background controls.

    Analysis tools are pages owned by :class:`ExposureAnalysisTargetView`, not
    dialogs.  Navigation and window lifetime belong to the surrounding target
    view/window; the page only owns its calculation state.
    """

    def cancel_analysis(self):
        _request_analysis_cancellation(self)

    def cleanup(self):
        self.cancel_analysis()


class FFTSpectrumPage(_CancellableAnalysisPage):
    """Interactive spectrum viewer backed by the frozen waveform-analysis payload."""

    WINDOW_LABELS = {
        "rectangular": "Rectangular",
        "hann": "Hann",
        "hamming": "Hamming",
        "blackman": "Blackman",
    }

    def __init__(self, payload, selected_key=None, parent=None, analysis_cache=None):
        super().__init__(parent)
        self.payload = payload
        self.analysis_cache = analysis_cache if isinstance(analysis_cache, dict) else {}
        self._spectrum = None
        self._plot_revision_serial = 0
        self._analysis_busy = False
        self._analysis_request_key = None

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.region = QComboBox()
        for key, entry in payload["regions"].items():
            self.region.addItem(entry["name"], key)
        index = self.region.findData(selected_key)
        if index >= 0:
            self.region.setCurrentIndex(index)
        form.addRow("Region", self.region)

        mode_widget = QWidget(self)
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_mode = QRadioButton("Auto")
        self.custom_mode = QRadioButton("Custom")
        self.auto_mode.setChecked(True)
        mode_layout.addWidget(self.auto_mode)
        mode_layout.addWidget(self.custom_mode)
        mode_layout.addStretch(1)
        form.addRow("FFT settings", mode_widget)

        if payload.get("analysis_trace_reduced", False) and isinstance(payload.get("authoritative_analysis"), dict):
            maximum_samples = 65536
        else:
            maximum_samples = max(2, min(int(payload.get("shape", [3])[0]) - 1, 65536))
        self.samples = QSpinBox()
        self.samples.setRange(2, maximum_samples)
        self.samples.setValue(maximum_samples)
        self.samples.setKeyboardTracking(False)
        self.samples.setToolTip(
            "Uniform samples per FFT record. Short records use one physical record; long periodic "
            "exposures use one verified cycle; other long exposures divide the full duration into "
            "equal records. The analysis never samples faster than the authoritative source clock."
        )
        form.addRow("FFT samples", self.samples)

        self.window = QComboBox()
        for key in FFT_WINDOWS:
            self.window.addItem(self.WINDOW_LABELS.get(key, key.title()), key)
        form.addRow("Window", self.window)

        self.detrend = QCheckBox("Remove mean field before FFT")
        self.detrend.setChecked(True)
        form.addRow("Detrend", self.detrend)

        self.max_frequency = QDoubleSpinBox()
        self.max_frequency.setRange(0.0, 1.0e9)
        self.max_frequency.setDecimals(6)
        self.max_frequency.setValue(0.0)
        self.max_frequency.setSpecialValueText("Nyquist")
        self.max_frequency.setSuffix(" Hz")
        self.max_frequency.setKeyboardTracking(False)
        self.max_frequency.setToolTip("0 / Nyquist shows the complete calculated frequency range.")
        form.addRow("Plot max frequency", self.max_frequency)

        display_widget = QWidget(self)
        display_layout = QHBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        self.show_dc = QCheckBox("Show DC")
        self.show_dc.setChecked(True)
        self.log_y = QCheckBox("Log Y")
        display_layout.addWidget(self.show_dc)
        display_layout.addWidget(self.log_y)
        display_layout.addStretch(1)
        form.addRow("Display", display_widget)
        root.addLayout(form)

        self.progress_label, self.progress_bar = _add_analysis_progress(root, self)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("hint")
        root.addWidget(self.info)

        self.plot = PlotView(self)
        root.addWidget(self.plot, 1)

        plot_actions = QHBoxLayout()
        plot_actions.addStretch(1)
        self.fullscreen_button = QPushButton("Fullscreen viewer")
        self.fullscreen_button.setToolTip(
            "Open a separate true-fullscreen interactive FFT plot (Esc/F11 closes it)"
        )
        self.fullscreen_button.clicked.connect(self._toggle_plot_fullscreen)
        plot_actions.addWidget(self.fullscreen_button)
        root.addLayout(plot_actions)

        self.auto_mode.toggled.connect(self._mode_changed)
        self.custom_mode.toggled.connect(self._mode_changed)
        self.region.currentIndexChanged.connect(self._region_changed)
        # Plot-range and display controls do not change the FFT itself.  Apply them
        # immediately because these controls do not change the cached FFT calculation.
        self.max_frequency.valueChanged.connect(
            lambda *_args: self._refresh_plot(reset_view=True)
        )
        self.show_dc.toggled.connect(
            lambda *_args: self._refresh_plot(reset_view=True)
        )
        self.log_y.toggled.connect(
            lambda *_args: self._refresh_plot(reset_view=True)
        )
        # Calculation settings are cache keys. Changing them loads a matching saved
        # configuration or calculates that configuration automatically; an immutable
        # saved analysis does not expose a rerun action.
        self.samples.valueChanged.connect(self.recalculate)
        self.window.currentIndexChanged.connect(self.recalculate)
        self.detrend.toggled.connect(self.recalculate)
        self._mode_changed()

    def _mode_changed(self, *_args):
        custom = self.custom_mode.isChecked()
        # These three controls change the FFT calculation.  Plot max frequency is
        # deliberately display-only, so it remains available in Auto mode too.
        for control in (self.samples, self.window, self.detrend):
            control.setEnabled(custom)
        self.max_frequency.setEnabled(True)
        if self.region.currentData() is not None:
            self.recalculate()
        else:
            self._refresh_plot(reset_view=True)

    def _region_changed(self, *_args):
        self.recalculate()

    def _request_key(self):
        key = self.region.currentData()
        if key is None:
            return None
        if self.auto_mode.isChecked():
            return ("spectrum", str(key), None, "rectangular", True)
        return (
            "spectrum", str(key), int(self.samples.value()),
            str(self.window.currentData()), bool(self.detrend.isChecked()),
        )

    def _set_analysis_busy(self, busy, message="Calculating FFT…"):
        self._analysis_busy = bool(busy)
        _set_analysis_progress(self.progress_label, self.progress_bar, busy, message)
        for control in (
            self.region, self.auto_mode, self.custom_mode, self.samples, self.window,
            self.detrend,
        ):
            control.setEnabled(not busy)
        if not busy:
            custom = self.custom_mode.isChecked()
            self.samples.setEnabled(custom)
            self.window.setEnabled(custom)
            self.detrend.setEnabled(custom)

    def recalculate(self, *_args):
        request_key = self._request_key()
        if request_key is None or self._analysis_busy:
            return
        cached = _analysis_cached_result(self.analysis_cache, request_key)
        if isinstance(cached, dict):
            _set_analysis_cached(
                self.progress_label, self.progress_bar, self.analysis_cache, request_key
            )
            self._apply_spectrum(cached)
            return
        _start_regional_analysis(self, request_key)

    def _analysis_progress(self, update):
        if not self._analysis_busy:
            return
        _set_analysis_progress(
            self.progress_label, self.progress_bar, True, "Calculating FFT…", update
        )

    def _analysis_completed(self, spectrum):
        _complete_regional_analysis(self, spectrum, self._apply_spectrum)

    def _analysis_failed(self, message):
        self._set_analysis_busy(False)
        _set_analysis_failed(self.progress_label, self.progress_bar)
        self._show_analysis_error(str(message))

    def _show_analysis_error(self, message):
        self._spectrum = None
        self.info.setText(str(message))
        self.plot.show_message("Unable to calculate FFT", str(message))

    def _apply_spectrum(self, spectrum):
        self._spectrum = spectrum
        # Keep the custom editor synchronized with the exact source-supported count.
        for control in (self.samples, self.window, self.detrend):
            control.blockSignals(True)
        self.samples.setRange(2, max(2, int(spectrum["max_fft_samples"])))
        if self.auto_mode.isChecked():
            self.samples.setValue(int(spectrum["fft_samples"]))
            self.window.setCurrentIndex(self.window.findData(spectrum["window"]))
            self.detrend.setChecked(bool(spectrum["detrended"]))
        for control in (self.samples, self.window, self.detrend):
            control.blockSignals(False)
        mode = str(spectrum.get("analysis_mode", "whole_record"))
        if mode == "verified_periodic_cycle":
            mode_text = (
                f"Verified cycle • {spectrum['spectral_record_duration_s']:.6g} s record"
                f" from {spectrum['physical_duration_s']:.6g} s exposure"
            )
        elif mode == "segmented_full_duration":
            mode_text = (
                f"Full duration • {spectrum['segment_count']:,} × "
                f"{spectrum['spectral_record_duration_s']:.6g} s records"
            )
        else:
            mode_text = f"Whole record • {spectrum['physical_duration_s']:.6g} s"
        self.info.setText(
            f"{mode_text} • {spectrum['fft_samples']:,} samples/record\n"
            f"Δf {spectrum['frequency_resolution_hz']:.6g} Hz • "
            f"Nyquist {spectrum['nyquist_hz']:.6g} Hz • "
            f"DC {spectrum['dc_rms_uT']:.6g} µT"
        )
        self._refresh_plot(reset_view=True)

    def _refresh_plot(self, *_args, reset_view=False):
        if not isinstance(self._spectrum, dict):
            return
        frequency = np.asarray(self._spectrum["frequency_hz"], dtype=float)
        values = np.asarray(self._spectrum["bin_rms_uT"], dtype=float).copy()
        if len(values) and self.show_dc.isChecked():
            # The analysis intentionally reports true (unwindowed) DC separately.
            # Put that value at 0 Hz so the graph can show DC alongside AC bins.
            values[0] = float(self._spectrum["dc_rms_uT"])
        elif len(values):
            frequency = frequency[1:]
            values = values[1:]

        max_hz = None
        if self.max_frequency.value() > 0:
            max_hz = min(float(self.max_frequency.value()), float(self._spectrum["nyquist_hz"]))
            mask = frequency <= max_hz + np.finfo(float).eps * max(1.0, max_hz)
            frequency = frequency[mask]
            values = values[mask]
        if self.log_y.isChecked():
            plotted_values = np.where(values > 0, values, np.nan)
        else:
            plotted_values = values

        region_name = self.region.currentText() or str(self.region.currentData())
        trace = {
            "type": "bar",
            "x": frequency.tolist(),
            "y": plotted_values.tolist(),
            "name": "Spatial RMS contribution",
            "hovertemplate": "%{x:.8g} Hz<br>%{y:.8g} µT<extra></extra>",
        }
        resolution = float(self._spectrum["frequency_resolution_hz"])
        if resolution > 0:
            trace["width"] = resolution * 0.82
        xaxis = {
            "title": {"text": "Frequency (Hz)"},
            "rangemode": "tozero",
            "autorange": max_hz is None,
        }
        if max_hz is not None:
            xaxis["range"] = [0.0, max_hz]
        yaxis = {
            "title": {"text": "Spatial RMS contribution (µT)"},
            "rangemode": "tozero",
            "autorange": True,
        }
        if self.log_y.isChecked():
            yaxis.pop("rangemode", None)
            yaxis["type"] = "log"
        # Plotly preserves zoom/range state while ``uirevision`` is unchanged.
        # A single constant token therefore caused an explicit Plot max frequency
        # range to be ignored after the first full-Nyquist render.  Tie the token to
        # every setting that legitimately changes the axes/data so those changes
        # reset the Plotly view, while ordinary repaints with identical settings
        # still preserve a user's manual zoom.
        if reset_view:
            self._plot_revision_serial += 1
        revision = "|".join(
            (
                "waveform-fft-spectrum",
                str(self.region.currentData()),
                str(int(self._spectrum["fft_samples"])),
                str(self._spectrum["window"]),
                "detrend" if self._spectrum["detrended"] else "mean",
                "nyquist" if max_hz is None else f"max={max_hz:.12g}",
                "dc" if self.show_dc.isChecked() else "no-dc",
                "log" if self.log_y.isChecked() else "linear",
                f"view={self._plot_revision_serial}",
            )
        )
        figure = {
            "data": [trace],
            "layout": {
                "template": "plotly_white",
                "title": {"text": f"FFT spectrum — {region_name}", "x": 0.5},
                "margin": {"l": 80, "r": 30, "t": 60, "b": 70},
                "xaxis": xaxis,
                "yaxis": yaxis,
                "bargap": 0.04,
                "showlegend": False,
                "uirevision": revision,
            },
        }
        if reset_view:
            self.plot.request_reset_2d_view_on_next_figure()
        self.plot.set_figure(figure, coalesce=True)

    def _toggle_plot_fullscreen(self, *_args):
        """Open/close the shared PlotView true-fullscreen renderer for the FFT plot."""
        self.plot.toggle_fullscreen()

    def cleanup(self):
        self.plot.cleanup()
        super().cleanup()


class SpectrumAnalysisPage(_CancellableAnalysisPage):
    """On-demand vector-spectrum analysis from the frozen exposure payload."""

    fftRequested = Signal()

    def __init__(self, payload, selected_key=None, parent=None, analysis_cache=None):
        super().__init__(parent)
        self.payload = payload
        self.analysis_cache = analysis_cache if isinstance(analysis_cache, dict) else {}
        self._analysis_busy = False
        self._analysis_request_key = None
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.region = QComboBox()
        for key, entry in payload["regions"].items():
            self.region.addItem(entry["name"], key)
        index = self.region.findData(selected_key)
        if index >= 0:
            self.region.setCurrentIndex(index)
        form.addRow("Region", self.region)
        root.addLayout(form)

        note = QLabel(
            "Shows the frequencies present in this region. The table lists the 30 strongest components; "
            "View FFT… opens the full spectrum and its settings."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.progress_label, self.progress_bar = _add_analysis_progress(root, self)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(150)
        root.addWidget(self.summary)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Frequency (Hz)", "Spatial RMS contribution (µT)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 180)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.fft_button = QPushButton("View FFT…", self)
        self.fft_button.clicked.connect(lambda _checked=False: self.fftRequested.emit())
        actions.addWidget(self.fft_button)
        root.addLayout(actions)
        self.region.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def _request_key(self):
        key = self.region.currentData()
        return None if key is None else ("spectrum", str(key), None, "rectangular", True)

    def _set_analysis_busy(self, busy, message="Calculating spectrum…"):
        self._analysis_busy = bool(busy)
        self.region.setEnabled(not busy)
        self.fft_button.setEnabled(not busy)
        _set_analysis_progress(self.progress_label, self.progress_bar, busy, message)

    def refresh(self, *_args):
        request_key = self._request_key()
        if request_key is None or self._analysis_busy:
            return
        cached = _analysis_cached_result(self.analysis_cache, request_key)
        if isinstance(cached, dict):
            _set_analysis_cached(
                self.progress_label, self.progress_bar, self.analysis_cache, request_key
            )
            self._apply_spectrum(cached)
            return
        _start_regional_analysis(self, request_key)

    def _analysis_progress(self, update):
        if self._analysis_busy:
            _set_analysis_progress(
                self.progress_label, self.progress_bar, True, "Calculating spectrum…", update
            )

    def _analysis_completed(self, data):
        _complete_regional_analysis(self, data, self._apply_spectrum)

    def _analysis_failed(self, message):
        self._set_analysis_busy(False)
        _set_analysis_failed(self.progress_label, self.progress_bar)
        self._show_analysis_error(str(message))

    def _show_analysis_error(self, message):
        self.summary.setPlainText(str(message))
        self.table.setRowCount(0)

    def _apply_spectrum(self, data):
        mode = str(data.get("analysis_mode", "whole_record"))
        if mode == "verified_periodic_cycle":
            mode_line = (
                f"Verified steady-state cycle: {data['spectral_record_duration_s']:.6g} s "
                f"from {data['physical_duration_s']:.6g} s physical exposure."
            )
            if data.get("spectral_sampling_reduced", False):
                mode_line += (
                    f" Analysis Nyquist is {data['nyquist_hz']:.6g} Hz versus source "
                    f"{data['source_nyquist_hz']:.6g} Hz because the per-record FFT cap is active."
                )
        elif mode == "segmented_full_duration":
            mode_line = (
                f"Segmented full-duration spectrum: {data['segment_count']:,} records × "
                f"{data['spectral_record_duration_s']:.6g} s; vector spectral power averaged."
            )
            if data.get("spectral_sampling_reduced", False):
                mode_line += (
                    f" Analysis Nyquist is {data['nyquist_hz']:.6g} Hz versus source "
                    f"{data['source_nyquist_hz']:.6g} Hz because the bounded spectral-work budget is active."
                )
        else:
            mode_line = f"Whole physical record: {data['physical_duration_s']:.6g} s."
        self.summary.setPlainText(
            f"DC spatial RMS: {data['dc_rms_uT']:.6g} µT\n"
            f"{mode_line}\n"
            f"FFT: {data['fft_samples']:,} samples/record; bin spacing {data['frequency_resolution_hz']:.6g} Hz; "
            f"Nyquist {data['nyquist_hz']:.6g} Hz.\n"
            "Automatic spectrum: rectangular window with mean removal. Non-periodic records can leak between bins; "
            "custom windowing and display controls are available in View FFT…."
        )
        power = np.asarray(data["bin_rms_uT"], dtype=float)
        indices = [
            int(i)
            for i in np.argsort(power[1:])[::-1][:30] + 1
            if power[i] > 1e-15
        ]
        self.table.setRowCount(len(indices))
        for row, index in enumerate(indices):
            self.table.setItem(row, 0, QTableWidgetItem(f"{data['frequency_hz'][index]:.8g}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{power[index]:.8g}"))


class TimingAnalysisPage(_CancellableAnalysisPage):
    """Vector slew and drive pulse/gap timing for a frozen exposure region."""

    def __init__(self, payload, selected_key=None, parent=None, analysis_cache=None):
        super().__init__(parent)
        self.payload = payload
        self.analysis_cache = analysis_cache if isinstance(analysis_cache, dict) else {}
        self._analysis_busy = False
        self._analysis_request_key = None
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.region = QComboBox()
        for key, entry in payload["regions"].items():
            self.region.addItem(entry["name"], key)
        index = self.region.findData(selected_key)
        if index >= 0:
            self.region.setCurrentIndex(index)
        form.addRow("Region", self.region)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 99.99)
        self.threshold.setValue(1.0)
        self.threshold.setSuffix(" % of each drive's peak |current|")
        self.threshold.setKeyboardTracking(False)
        self.threshold.setToolTip(
            "Used only to identify drive-current pulse/gap crossings. It is not a tissue or field threshold."
        )
        form.addRow("Drive pulse threshold", self.threshold)
        root.addLayout(form)

        note = QLabel(
            "Shows how quickly the field changes in this region, along with drive-current pulse lengths and gaps."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.progress_label, self.progress_bar = _add_analysis_progress(root, self)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        root.addWidget(self.summary, 1)
        self.region.currentIndexChanged.connect(self.refresh)
        self.threshold.valueChanged.connect(self.refresh)
        self.refresh()

    def _request_key(self):
        key = self.region.currentData()
        if key is None:
            return None
        return ("timing", str(key), round(float(self.threshold.value()), 9))

    def _set_analysis_busy(self, busy, message="Analysing timing…"):
        self._analysis_busy = bool(busy)
        self.region.setEnabled(not busy)
        self.threshold.setEnabled(not busy)
        _set_analysis_progress(self.progress_label, self.progress_bar, busy, message)

    def refresh(self, *_args):
        request_key = self._request_key()
        if request_key is None or self._analysis_busy:
            return
        cached = _analysis_cached_result(self.analysis_cache, request_key)
        if isinstance(cached, dict):
            _set_analysis_cached(
                self.progress_label, self.progress_bar, self.analysis_cache, request_key
            )
            self._apply_timing(cached)
            return
        _start_regional_analysis(self, request_key)

    def _analysis_progress(self, update):
        if self._analysis_busy:
            _set_analysis_progress(
                self.progress_label, self.progress_bar, True, "Analysing timing…", update
            )

    def _analysis_completed(self, data):
        _complete_regional_analysis(self, data, self._apply_timing)

    def _analysis_failed(self, message):
        self._set_analysis_busy(False)
        _set_analysis_failed(self.progress_label, self.progress_bar)
        self.summary.setPlainText(str(message))

    def _apply_timing(self, data):
        lines = [
            f"Maximum interval spatial-RMS |dB/dt|: {data['peak_spatial_rms_slew_uT_s']:.6g} µT/s "
            f"(interval starting at {data['slew_time_s']:.6g} s)",
            "Slew is a regional RMS, not the worst individual point in the object.",
            f"Largest interval between samples: {data['max_interval_s']:.6g} s. "
            "Transitions faster than the physical analysis sampling cannot be resolved.",
            "",
            "Drive timing (linear current threshold crossings; not tissue pulse duration):",
        ]
        for drive in data["drive_timing"]:
            lines.append(
                f"{drive['id']}: {drive['pulses']:,} pulse(s), threshold {drive['threshold_a']:.6g} A; "
                f"active {drive['active_s']:.6g} s, inactive {drive['inactive_s']:.6g} s"
            )
            if drive["pulses"]:
                lines.append(
                    f"  Pulse widths: {drive['pulse_width_min_s']:.6g}–{drive['pulse_width_max_s']:.6g} s "
                    f"(mean {drive['pulse_width_mean_s']:.6g} s)"
                )
            if drive["gap_count"]:
                lines.append(
                    f"  Between-pulse gaps: {drive['gap_width_min_s']:.6g}–{drive['gap_width_max_s']:.6g} s "
                    f"(mean {drive['gap_width_mean_s']:.6g} s)"
                )
            if drive.get("timing_preview_truncated"):
                lines.append("  Repeated timing was summarized without expanding every pulse.")
        self.summary.setPlainText("\n".join(lines))


class ThresholdDurationPage(_CancellableAnalysisPage):
    """User-defined time-above-field-threshold analysis."""

    def __init__(self, payload, selected_key=None, parent=None, analysis_cache=None):
        super().__init__(parent)
        self.payload = payload
        self.analysis_cache = analysis_cache if isinstance(analysis_cache, dict) else {}
        self._analysis_busy = False
        self._analysis_request_key = None
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.region = QComboBox()
        for key, entry in payload["regions"].items():
            self.region.addItem(entry["name"], key)
        index = self.region.findData(selected_key)
        if index >= 0:
            self.region.setCurrentIndex(index)
        form.addRow("Region", self.region)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0e9)
        self.threshold.setDecimals(6)
        self.threshold.setValue(1.0)
        self.threshold.setSuffix(" µT spatial RMS")
        self.threshold.setKeyboardTracking(False)
        form.addRow("Field threshold", self.threshold)
        root.addLayout(form)

        note = QLabel(
            "Measures how long the selected region's spatial-RMS |B| is above a threshold you choose. "
            "It reports total time, fraction of exposure, excursion count, and the longest uninterrupted interval. "
            "The threshold is descriptive and is not treated as a biological or regulatory boundary."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.progress_label, self.progress_bar = _add_analysis_progress(root, self)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(190)
        root.addWidget(self.summary)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Excursion", "Start (s)", "End (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        self.region.currentIndexChanged.connect(self.refresh)
        self.threshold.valueChanged.connect(self.refresh)
        self.refresh()

    def _request_key(self):
        key = self.region.currentData()
        if key is None:
            return None
        return ("threshold", str(key), round(float(self.threshold.value()), 9))

    def _set_analysis_busy(self, busy, message="Analysing threshold duration…"):
        self._analysis_busy = bool(busy)
        self.region.setEnabled(not busy)
        self.threshold.setEnabled(not busy)
        _set_analysis_progress(self.progress_label, self.progress_bar, busy, message)

    def refresh(self, *_args):
        request_key = self._request_key()
        if request_key is None or self._analysis_busy:
            return
        cached = _analysis_cached_result(self.analysis_cache, request_key)
        if isinstance(cached, dict):
            _set_analysis_cached(
                self.progress_label, self.progress_bar, self.analysis_cache, request_key
            )
            self._apply_threshold(cached)
            return
        _start_regional_analysis(self, request_key)

    def _analysis_progress(self, update):
        if self._analysis_busy:
            _set_analysis_progress(
                self.progress_label, self.progress_bar, True,
                "Analysing threshold duration…", update
            )

    def _analysis_completed(self, data):
        _complete_regional_analysis(self, data, self._apply_threshold)

    def _analysis_failed(self, message):
        self._set_analysis_busy(False)
        _set_analysis_failed(self.progress_label, self.progress_bar)
        self._show_analysis_error(str(message))

    def _show_analysis_error(self, message):
        self.summary.setPlainText(str(message))
        self.table.setRowCount(0)

    def _apply_threshold(self, data):
        self.summary.setPlainText(
            f"Threshold: {data['threshold_uT']:.6g} µT regional spatial RMS\n"
            f"Physical exposure duration: {data['duration_s']:.6g} s\n"
            f"Time above threshold: {data['time_above_s']:.6g} s ({data['percent_above']:.6g} %)\n"
            f"Longest uninterrupted interval: {data['longest_interval_s']:.6g} s\n"
            f"Excursions above threshold: {data['excursion_count']:,}\n"
            f"Peak regional spatial RMS: {data['peak_spatial_rms_uT']:.6g} µT at {data['peak_time_s']:.6g} s\n"
            f"Samples: {data['source_sample_count']:,}; largest interval {data['max_source_interval_s']:.6g} s.\n\n"
            "Threshold-crossing times are estimated between samples. Faster excursions "
            "cannot be recovered from a timebase that did not sample them."
        )
        intervals = data["intervals_s"]
        if data.get("intervals_truncated"):
            self.summary.appendPlainText(
                f"\nThe table shows only the first {len(intervals):,} excursions; "
                "the totals above include every repeated excursion."
            )
        self.table.setRowCount(len(intervals))
        for row, (start, end) in enumerate(intervals):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{start:.9g}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{end:.9g}"))


class ExposureSummaryPage(_CancellableAnalysisPage):
    """All-region exposure summary using the shared regional metric panel."""

    def __init__(
        self, payload, selected_key=None, parent=None, analysis_cache=None
    ):
        super().__init__(parent)
        self.payload = payload
        self.analysis_cache = analysis_cache if isinstance(analysis_cache, dict) else {}
        self._analysis_busy = False
        self._analysis_request_key = None
        context = payload.get("exposure_analysis_context") if isinstance(payload, dict) else None
        context = context if isinstance(context, dict) else {}
        self._target_kind = str(context.get("target_kind") or "regional")
        self._brain_target = self._target_kind == "brain_lpba40" or (
            str(context.get("atlas") or "").strip().upper() == "LPBA40"
        )
        self._surface_target = self._target_kind == "surface_region_volume"

        # Keep the common hidden selector API so selection in this all-region
        # table stays synchronized with the region-specific analysis pages.
        self.region = QComboBox(self)
        for key, entry in payload["regions"].items():
            self.region.addItem(entry["name"], key)
        index = self.region.findData(selected_key)
        if index >= 0:
            self.region.setCurrentIndex(index)
        self.region.hide()

        hierarchy = self._summary_hierarchy()
        region_mode = (
            "brain" if self._brain_target
            else "surface" if self._surface_target
            else "measurement"
        )
        self.region_panel = ExposureRegionPanel(
            hierarchy, self, region_mode=region_mode
        )
        self.region_panel.configure_exposure_summary()
        self.table = self.region_panel.table
        self.filter_table = self.region_panel.filter_table
        self.region_panel.table.itemSelectionChanged.connect(self._summary_region_selected)
        self.region.currentIndexChanged.connect(self._sync_table_selection)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.region_panel, 1)
        self.progress_label, self.progress_bar = _add_analysis_progress(root, self)
        self._owns_region_browser = True
        self._sync_table_selection()
        self.refresh()

    def _summary_hierarchy(self):
        context = (
            self.payload.get("exposure_analysis_context")
            if isinstance(self.payload, dict) else None
        )
        context = context if isinstance(context, dict) else {}
        hierarchy_payload = context.get("region_hierarchy")
        if isinstance(hierarchy_payload, dict):
            try:
                return analysis_region_hierarchy_from_payload(hierarchy_payload)
            except (RegionalAnalysisError, TypeError, ValueError):
                pass
        if self._brain_target:
            return lpba40_region_hierarchy()
        children = []
        for index, (key, entry) in enumerate(self.payload.get("regions", {}).items(), start=1):
            children.append(
                AnalysisRegionNode(str(key), str(entry.get("name") or key), (index,), ())
            )
        return MeasurementRegions(tuple(children))

    def _request_key(self):
        key = self.region.currentData()
        if key is None and self.region.count():
            key = self.region.itemData(0)
        return None if key is None else ("summary", str(key))

    def _summary_region_selected(self):
        item = self.table.currentItem()
        if item is None:
            return
        key = item.data(0, REGION_KEY_ROLE)
        if key is None:
            return
        index = self.region.findData(key)
        if index >= 0 and index != self.region.currentIndex():
            self.region.setCurrentIndex(index)

    def _sync_table_selection(self, *_args):
        item = self.region_panel._items_by_key.get(str(self.region.currentData()))
        if item is None:
            return
        self.table.blockSignals(True)
        self.table.setCurrentItem(item)
        item.setSelected(True)
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self.table.scrollToItem(item)
        self.table.blockSignals(False)

    def _set_analysis_busy(self, busy, message="Calculating exposure summary…"):
        self._analysis_busy = bool(busy)
        self.region_panel.setEnabled(not busy)
        _set_analysis_progress(self.progress_label, self.progress_bar, busy, message)

    def refresh(self, *_args):
        if self._analysis_busy:
            return
        request_key = self._request_key()
        if request_key is None:
            return
        group = self.analysis_cache.get(_regional_config_key(request_key))
        regions = group.get("regions") if isinstance(group, dict) else None
        if isinstance(regions, dict):
            _set_analysis_cached(
                self.progress_label, self.progress_bar, self.analysis_cache, request_key
            )
            self._apply_summary_group(regions)
            return
        _start_regional_analysis(self, request_key)

    def _analysis_progress(self, update):
        if self._analysis_busy:
            _set_analysis_progress(
                self.progress_label, self.progress_bar, True,
                "Calculating exposure summary…", update
            )

    def _analysis_completed(self, data):
        _complete_regional_analysis(
            self, data, self._apply_summary_group, all_regions=True
        )

    def _analysis_failed(self, message):
        self._set_analysis_busy(False)
        _set_analysis_failed(self.progress_label, self.progress_bar)
        self.region_panel.set_exposure_summary_status(str(message))

    def _apply_summary_group(self, results):
        self.region_panel.set_exposure_summary_results(results)
        self._sync_table_selection()


class FrequencyWeightedResultsWindow(QDialog):
    """Modeless result browser owned by its frequency-weighting page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Frequency-weighted results")
        self.resize(1150, 800)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)


class FrequencyWeightedPage(_CancellableAnalysisPage):
    """Queue reference-profile calculations and browse cached regional results."""

    def __init__(self, payload, selected_key=None, parent=None, analysis_cache=None):
        super().__init__(parent)
        self.payload = payload
        self.analysis_cache = analysis_cache if isinstance(analysis_cache, dict) else {}
        self._analysis = None
        self._analysis_busy = False
        self._analysis_request_key = None
        self._weighted_pending_requests = []
        self._weighted_pending_set = set()
        self._weighted_open_after = set()
        self._weighted_followup_pending = False

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.region = QComboBox(self)
        for key, entry in payload["regions"].items():
            self.region.addItem(entry["name"], key)
        index = self.region.findData(selected_key)
        if index >= 0:
            self.region.setCurrentIndex(index)
        self.region.hide()  # Retain target selection synchronization through the existing API.

        self.profile = QComboBox(self)
        for key, definition in FREQUENCY_WEIGHTING_PROFILES.items():
            self.profile.addItem(definition["label"], key)
        self.profile.hide()  # Selection is made by the per-profile actions below.

        mode_widget = QWidget(self)
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_mode = QRadioButton("Auto")
        self.custom_mode = QRadioButton("Custom")
        self.auto_mode.setChecked(True)
        mode_layout.addWidget(self.auto_mode)
        mode_layout.addWidget(self.custom_mode)
        mode_layout.addStretch(1)
        form.addRow("Sampling", mode_widget)

        if payload.get("analysis_trace_reduced", False) and isinstance(payload.get("authoritative_analysis"), dict):
            maximum_samples = 65536
        else:
            maximum_samples = max(2, min(int(payload.get("shape", [3])[0]) - 1, 65536))
        self.samples = QSpinBox()
        self.samples.setRange(2, maximum_samples)
        self.samples.setValue(maximum_samples)
        self.samples.setKeyboardTracking(False)
        self.samples.setToolTip(
            "Samples used for each frequency-analysis record. Auto chooses the sample count. "
            "See Help for how long exposures are analysed."
        )
        form.addRow("Samples per record", self.samples)
        root.addLayout(form)

        note = QLabel(
            "Field Workbench compares the exposure in the chosen region to the selected reference standard. "
            "This is not a substitute for a professional safety assessment."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.progress_label, self.progress_bar = _add_analysis_progress(root, self)

        self.profile_table = QTableWidget(len(FREQUENCY_WEIGHTING_PROFILES), 3, self)
        self.profile_table.setHorizontalHeaderLabels(["Reference profile", "Status", "Action"])
        self.profile_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.profile_table.verticalHeader().hide()
        self.profile_table.setWordWrap(False)
        header = self.profile_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._profile_actions = {}
        for row, (key, definition) in enumerate(FREQUENCY_WEIGHTING_PROFILES.items()):
            label_item = QTableWidgetItem(definition["label"])
            label_item.setToolTip(definition["label"])
            self.profile_table.setItem(row, 0, label_item)
            view = QPushButton("Calculate")
            view.setToolTip(
                "Calculate this reference for the current sampling settings. Cached references can be opened while other calculations run."
            )
            view.clicked.connect(lambda _checked=False, k=key: self._profile_action_clicked(k))
            self.profile_table.setCellWidget(row, 2, view)
            self._profile_actions[key] = (row, view)
        self.region_tree = self._make_region_tree()
        overview = QSplitter(Qt.Orientation.Horizontal, self)
        overview.addWidget(self.region_tree)
        overview.addWidget(self.profile_table)
        overview.setStretchFactor(0, 1)
        overview.setStretchFactor(1, 3)
        overview.setSizes([250, 780])
        overview.setChildrenCollapsible(False)
        root.addWidget(overview, 1)
        profile_controls = QHBoxLayout()
        self.calculate_all_profiles_button = QPushButton("Calculate all uncached references", self)
        self.calculate_all_profiles_button.setObjectName("calculateAllFrequencyWeightedReferences")
        self.calculate_all_profiles_button.setToolTip(
            "Queue every reference profile that is not already cached for the current sampling settings."
        )
        self.calculate_all_profiles_button.clicked.connect(self.calculate_all_uncached_profiles)
        profile_controls.addWidget(self.calculate_all_profiles_button)
        profile_controls.addStretch(1)
        root.addLayout(profile_controls)
        hint = QLabel(
            "Cached references remain openable while another reference is running. Uncached references can be queued; "
            "sampling settings stay locked until the queued frequency-weighting work finishes."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        root.addWidget(self.error_label)
        self.result_window = FrequencyWeightedResultsWindow(self)
        result_root = QVBoxLayout(self.result_window)
        self.result_splitter = QSplitter(Qt.Orientation.Horizontal, self.result_window)
        self.result_splitter.setObjectName("frequencyWeightedResultSplitter")
        self.result_splitter.setChildrenCollapsible(False)

        self.result_text_panel = QWidget()
        self.result_text_panel.setObjectName("frequencyWeightedResultsPanel")
        result_text_layout = QVBoxLayout(self.result_text_panel)
        result_text_layout.setContentsMargins(0, 0, 0, 0)
        result_text_layout.setSpacing(8)

        self.summary = QPlainTextEdit(self.result_text_panel)
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(150)
        self.summary.setMaximumHeight(250)
        result_text_layout.addWidget(self.summary)

        self.table = QTableWidget(0, 4, self.result_text_panel)
        self.table.setHorizontalHeaderLabels(
            ["Frequency (Hz)", "Spatial RMS (µT)", "Reference RMS (µT)", "Ratio (%)"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 125)
        self.table.setColumnWidth(1, 140)
        self.table.setColumnWidth(2, 150)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        result_text_layout.addWidget(self.table, 1)

        self.result_plot_panel = QWidget()
        self.result_plot_panel.setObjectName("frequencyWeightedViewerPanel")
        result_plot_layout = QVBoxLayout(self.result_plot_panel)
        result_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot = PlotView(self.result_plot_panel)
        result_plot_layout.addWidget(self.plot, 1)

        self.result_splitter.addWidget(self.result_text_panel)
        self.result_splitter.addWidget(self.result_plot_panel)
        self.result_splitter.setStretchFactor(0, 3)
        self.result_splitter.setStretchFactor(1, 2)
        self.result_splitter.setSizes([620, 430])
        self.result_region_tree = self._make_region_tree()
        result_browser = QSplitter(Qt.Orientation.Horizontal, self.result_window)
        result_browser.addWidget(self.result_region_tree)
        result_browser.addWidget(self.result_splitter)
        result_browser.setStretchFactor(0, 1)
        result_browser.setStretchFactor(1, 4)
        result_browser.setSizes([230, 920])
        result_browser.setChildrenCollapsible(False)
        result_root.addWidget(result_browser, 1)
        self._view_profile = None
        self._view_sample_spec = None

        self.region.currentIndexChanged.connect(self._region_changed)
        self.samples.valueChanged.connect(self._refresh_profiles)
        self.auto_mode.toggled.connect(self._sampling_mode_changed)
        self.custom_mode.toggled.connect(self._sampling_mode_changed)
        self._region_changed()
        self._sampling_mode_changed()

    def _make_region_tree(self):
        return _make_analysis_region_tree(self)

    def _tree_region_changed(self, item, _previous):
        if item is not None:
            index = self.region.findData(item.data(0, Qt.ItemDataRole.UserRole))
            if index >= 0:
                self.region.setCurrentIndex(index)

    def _region_changed(self, *_args):
        key = self.region.currentData()
        for tree in (self.region_tree, self.result_region_tree):
            item = tree._region_items.get(key)
            if item is not None:
                tree.blockSignals(True)
                tree.setCurrentItem(item)
                ancestor = item.parent()
                while ancestor is not None:
                    ancestor.setExpanded(True)
                    ancestor = ancestor.parent()
                tree.scrollToItem(item)
                tree.blockSignals(False)
        self._refresh_profiles()
        if self.result_window.isVisible() and self._view_profile is not None:
            self._show_region_result()

    def _region_display_name(self):
        item = self.region_tree._region_items.get(self.region.currentData())
        names = []
        while item is not None:
            names.append(item.text(0))
            item = item.parent()
        return " / ".join(reversed(names)) or self.region.currentText()

    def _show_region_result(self):
        key = ("weighted", str(self.region.currentData()), self._view_profile, self._view_sample_spec)
        data = _analysis_cached_result(self.analysis_cache, key)
        if not isinstance(data, dict):
            self.summary.setPlainText("This profile has not been calculated for the selected sampling settings.")
            self.table.setRowCount(0)
            self.plot.show_message("No cached result for this region")
            return
        self._apply_analysis(data)
        target_name = self.payload.get("exposure_analysis_context", {}).get("target_name", "")
        prefix = f"{target_name} — " if target_name and not self._region_display_name().startswith(target_name) else ""
        self.result_window.setWindowTitle(f"{prefix}{self._region_display_name()} — {data['profile_label']}")

    def _current_sample_spec(self):
        return None if self.auto_mode.isChecked() else int(self.samples.value())

    def _request_key_for(self, profile, sample_spec):
        key = self.region.currentData()
        if key is None or profile is None:
            return None
        return ("weighted", str(key), str(profile), sample_spec)

    @staticmethod
    def _request_identity(profile, sample_spec):
        return (str(profile), sample_spec)

    def _active_request_identity(self):
        request_key = self._analysis_request_key
        if (not self._analysis_busy or not isinstance(request_key, tuple)
                or len(request_key) < 4):
            return None
        return self._request_identity(request_key[2], request_key[3])

    def _sampling_queue_active(self):
        return bool(self._analysis_busy or self._weighted_pending_requests)

    def _update_sampling_controls(self):
        locked = self._sampling_queue_active()
        self.profile.setEnabled(not locked)
        self.auto_mode.setEnabled(not locked)
        self.custom_mode.setEnabled(not locked)
        self.samples.setEnabled(not locked and self.custom_mode.isChecked())

    def _sampling_mode_changed(self, *_args):
        self._update_sampling_controls()
        self._refresh_profiles()

    def _refresh_profiles(self, *_args):
        sample_spec = self._current_sample_spec()
        active_identity = self._active_request_identity()
        pending = set(self._weighted_pending_set)
        page_cancelling = (
            self._analysis_busy and self._analysis_progress_state == "cancelling"
        )
        any_uncached_available = False
        for profile, (row, view) in self._profile_actions.items():
            request_key = self._request_key_for(profile, sample_spec)
            cached = isinstance(_analysis_cached_result(self.analysis_cache, request_key), dict)
            count = sum(
                isinstance(
                    _analysis_cached_result(
                        self.analysis_cache,
                        ("weighted", str(k), profile, sample_spec),
                    ),
                    dict,
                )
                for k in self.payload["regions"]
            )
            total = len(self.payload["regions"])
            identity = self._request_identity(profile, sample_spec)
            active = identity == active_identity
            queued_active = active and self._analysis_progress_state == "queued"
            cancelling = active and self._analysis_progress_state == "cancelling"
            running = active and self._analysis_progress_state in {
                "running", "pausing", "paused",
            }
            queued_pending = identity in pending
            status = (
                "Queued…" if queued_active or queued_pending else
                "Cancelling…" if cancelling else
                "Running…" if running else
                (f"Ready • {count}/{total} regions" if count else "Not calculated")
            )
            meta = _analysis_cache_metadata(self.analysis_cache, request_key)
            if cached and not active and not queued_pending and "processing_time_s" in meta:
                status += " • " + _format_analysis_elapsed(meta["processing_time_s"])
            self.profile_table.setItem(row, 1, QTableWidgetItem(status))

            if cached:
                view.setText("Open")
                view.setToolTip(
                    "Open this cached reference result. Cached results remain available while another reference calculates."
                )
                view.setEnabled(True)
            elif queued_active or queued_pending:
                view.setText("Queued")
                view.setToolTip("This reference is already waiting in the analysis queue.")
                view.setEnabled(False)
            elif cancelling:
                view.setText("Cancelling")
                view.setToolTip("This reference is being cancelled.")
                view.setEnabled(False)
            elif running:
                view.setText("Running")
                view.setToolTip("This reference is currently being calculated.")
                view.setEnabled(False)
            elif page_cancelling:
                view.setText("Wait")
                view.setToolTip("Wait for the current frequency-weighting cancellation to finish before queueing more references.")
                view.setEnabled(False)
            else:
                any_uncached_available = True
                if self._analysis_busy or self._weighted_pending_requests:
                    view.setText("Queue")
                    view.setToolTip(
                        "Add this reference to the frequency-weighting queue. It will run when earlier work finishes."
                    )
                else:
                    view.setText("Calculate")
                    view.setToolTip(
                        "Calculate this reference for the current sampling settings."
                    )
                view.setEnabled(True)

        self.calculate_all_profiles_button.setEnabled(any_uncached_available)
        self._update_sampling_controls()

    def _profile_action_clicked(self, profile):
        request_key = self._request_key(profile)
        if isinstance(_analysis_cached_result(self.analysis_cache, request_key), dict):
            self.view_profile(profile)
        else:
            self.ensure_profile_cached(profile)

    def calculate_all_uncached_profiles(self, *_args):
        """Queue every uncached reference for the current sampling configuration."""
        for profile in FREQUENCY_WEIGHTING_PROFILES:
            request_key = self._request_key(profile)
            if not isinstance(_analysis_cached_result(self.analysis_cache, request_key), dict):
                self.ensure_profile_cached(profile)
        self._refresh_profiles()

    def ensure_profile_cached(self, profile, *, open_result=False):
        """Open a cached profile or queue its calculation without duplicating requests."""
        index = self.profile.findData(profile)
        if index >= 0 and not self._sampling_queue_active():
            self.profile.setCurrentIndex(index)
        sample_spec = self._current_sample_spec()
        request_key = self._request_key_for(profile, sample_spec)
        if request_key is None:
            return
        data = _analysis_cached_result(self.analysis_cache, request_key)
        identity = self._request_identity(profile, sample_spec)
        if isinstance(data, dict):
            self._view_profile = str(profile)
            self._view_sample_spec = sample_spec
            # Browsing a saved reference must never overwrite the progress state
            # of a different profile that is running or queued on this page.
            if not self._sampling_queue_active():
                _set_analysis_cached(
                    self.progress_label, self.progress_bar, self.analysis_cache, request_key
                )
            if open_result:
                self._show_region_result()
                self.result_window.show()
                self.result_window.raise_()
                self.result_window.activateWindow()
            self._refresh_profiles()
            return

        if open_result:
            self._weighted_open_after.add(identity)
        if identity == self._active_request_identity() or identity in self._weighted_pending_set:
            self._refresh_profiles()
            return

        self.error_label.clear()
        self._weighted_pending_requests.append(identity)
        self._weighted_pending_set.add(identity)
        self._refresh_profiles()
        self._start_next_weighted_request()

    def view_profile(self, profile):
        self.ensure_profile_cached(profile, open_result=True)

    def _request_key(self, profile=None):
        profile = self.profile.currentData() if profile is None else profile
        return self._request_key_for(profile, self._current_sample_spec())

    def _set_analysis_busy(self, busy, message="Calculating frequency weighting…"):
        self._analysis_busy = bool(busy)
        _set_analysis_progress(self.progress_label, self.progress_bar, busy, message)
        self._refresh_profiles()

    def _start_next_weighted_request(self):
        if self._analysis_busy or not self._weighted_pending_requests:
            self._refresh_profiles()
            return
        profile, sample_spec = self._weighted_pending_requests.pop(0)
        self._weighted_pending_set.discard((profile, sample_spec))
        self._weighted_followup_pending = False
        self._start_weighted_request(profile, sample_spec)

    def _start_weighted_request(self, profile, sample_spec):
        request_key = self._request_key_for(profile, sample_spec)
        if request_key is None:
            self._refresh_profiles()
            return
        index = self.profile.findData(profile)
        if index >= 0:
            self.profile.setCurrentIndex(index)
        self._analysis_request_key = request_key
        fft_samples = sample_spec

        spectrum_key = ("spectrum", "__all_regions__", fft_samples, "rectangular", True)

        def calculate(progress):
            # Resolve the shared spectrum when this job actually reaches the
            # worker, not when it is submitted. A Spectrum/FFT or earlier
            # weighted job ahead of us in the FIFO queue may have populated the
            # matching cache while this reference was waiting.
            spectra = self.analysis_cache.get(spectrum_key, {}).get("regions")
            spectrum_elapsed = None
            if spectra is None:
                spectrum_started = time.perf_counter()
                spectra = analyse_regions_spectrum(
                    self.payload, fft_samples=fft_samples, window="rectangular",
                    detrend=True, progress_callback=progress,
                )
                spectrum_elapsed = time.perf_counter() - spectrum_started
            results = analyse_regions_frequency_weighted(
                self.payload, profile=profile, fft_samples=fft_samples,
                progress_callback=progress, spectra=spectra,
            )
            return {
                "results": results, "spectrum": spectra, "spectrum_key": spectrum_key,
                "spectrum_elapsed_s": spectrum_elapsed,
            }

        self._set_analysis_busy(True)
        if (self._analysis_progress_embedded or _long_authoritative_analysis(self.payload)
                or len(self.payload["regions"]) > 1):
            _start_analysis_job(self, calculate)
            return
        try:
            data = calculate(None)
        except (ValueError, KeyError, TypeError) as error:
            self._analysis_failed(str(error))
            return
        self._analysis_completed(data)

    def recalculate(self, *_args):
        # Retained as a small compatibility shim for internal callers: derived
        # weighting configurations are immutable and are calculated only if absent.
        self.ensure_profile_cached(self.profile.currentData())

    def _analysis_progress(self, update):
        if self._analysis_busy:
            _set_analysis_progress(
                self.progress_label, self.progress_bar, True,
                "Calculating frequency weighting…", update
            )

    def _queue_weighted_followup(self):
        self._weighted_followup_pending = bool(self._weighted_pending_requests)
        self._refresh_profiles()
        if (self._weighted_followup_pending
                and not callable(getattr(self, "_analysis_job_finished_callback", None))):
            QTimer.singleShot(0, self._analysis_job_released)

    def _analysis_job_released(self):
        """Submit the next reference after the shared worker slot has been released."""
        if self._analysis_busy or not self._weighted_followup_pending:
            return
        self._weighted_followup_pending = False
        self._start_next_weighted_request()

    def _analysis_cancel_requested(self):
        """Cancel means stop the current reference and discard this page's pending references."""
        self._weighted_pending_requests.clear()
        self._weighted_pending_set.clear()
        self._weighted_open_after.clear()
        self._weighted_followup_pending = False
        self._refresh_profiles()

    def _analysis_completed(self, data):
        request_key = self._analysis_request_key
        self._set_analysis_busy(False)
        spectrum = None
        if isinstance(data, dict) and "results" in data:
            spectrum = data.get("spectrum")
            spectrum_key = data.get("spectrum_key")
            spectrum_elapsed = data.get("spectrum_elapsed_s")
            data = data["results"]
        if (not isinstance(data, dict) or set(data) != set(self.payload["regions"])
                or not all(isinstance(result, dict) and "weighted_peak_index" in result
                           for result in data.values())):
            _set_analysis_failed(self.progress_label, self.progress_bar)
            self._show_analysis_error("Frequency-weighted analysis returned incomplete results.")
            self._queue_weighted_followup()
            return
        if spectrum is not None:
            if (set(spectrum) != set(self.payload["regions"]) or
                    spectrum_key != ("spectrum", "__all_regions__", request_key[3], "rectangular", True)):
                _set_analysis_failed(self.progress_label, self.progress_bar)
                self._show_analysis_error("The calculation did not return a matching spectrum.")
                self._queue_weighted_followup()
                return
            if spectrum_key not in self.analysis_cache:
                _store_analysis_cache_result(
                    self, spectrum_key, {"regions": spectrum},
                    notify=False, elapsed_s=spectrum_elapsed,
                )
        _store_analysis_cache_result(self, _regional_config_key(request_key), {"regions": data})
        identity = self._request_identity(request_key[2], request_key[3])
        self._refresh_profiles()
        if identity in self._weighted_open_after:
            self._weighted_open_after.discard(identity)
            self._view_profile = request_key[2]
            self._view_sample_spec = request_key[3]
            self._show_region_result()
            self.result_window.show()
            self.result_window.raise_()
            self.result_window.activateWindow()
        elif self.result_window.isVisible():
            self._show_region_result()
        self._queue_weighted_followup()

    def _analysis_failed(self, message):
        request_key = self._analysis_request_key
        if isinstance(request_key, tuple) and len(request_key) >= 4:
            self._weighted_open_after.discard(
                self._request_identity(request_key[2], request_key[3])
            )
        self._set_analysis_busy(False)
        _set_analysis_failed(self.progress_label, self.progress_bar)
        self._show_analysis_error(str(message))
        self._queue_weighted_followup()

    def _show_analysis_error(self, message):
        self.error_label.setText(f"Unable to calculate: {message}")
        self._refresh_profiles()

    def _apply_analysis(self, data):
        self._analysis = data
        key = self.region.currentData()
        profile = data["profile"]

        if data["full_band_resolved"]:
            coverage = "full 1 Hz–100 kHz LF band resolved"
        else:
            coverage = (
                f"resolved approximately {data['resolved_band_min_hz']:.6g}–"
                f"{data['resolved_band_max_hz']:.6g} Hz; content outside that range is not represented"
            )
        mode = str(data.get("analysis_mode", "whole_record"))
        if mode == "verified_periodic_cycle":
            mode_line = (
                f"Verified steady-state cycle: {data['spectral_record_duration_s']:.6g} s from a "
                f"{data['physical_duration_s']:.6g} s repeating exposure."
            )
        elif mode == "segmented_full_duration":
            mode_line = (
                f"Full exposure analysed in {data['segment_count']:,} records of "
                f"{data['spectral_record_duration_s']:.6g} s."
            )
        else:
            mode_line = f"Whole physical record: {data['physical_duration_s']:.6g} s."
        context = self.payload.get("exposure_analysis_context", {})
        lines = [
            f"Results for: {context.get('target_name', self._region_display_name())} — {self._region_display_name()}",
            f"Reference: {data['profile_label']}",
            ("Sampling: Auto" if self._view_sample_spec is None else
             f"Sampling: Custom — {self._view_sample_spec:,} samples per record"),
            f"Source: {data['source']}",
            f"Phase-aware weighted-peak index: {data['weighted_peak_index']:.6g} "
            f"({data['weighted_peak_percent']:.4g} %) at {data['weighted_peak_time_s']:.6g} s",
            f"Weighted equivalent regional spatial RMS at the peak: {data['weighted_peak_equivalent_uT']:.6g} µT; "
            f"50 Hz reference = {data['reference_rms_uT']:.6g} µT RMS "
            f"({data['reference_peak_uT']:.6g} µT peak).",
            f"Conservative spectral-summation index: {data['spectral_sum_index']:.6g} "
            f"({data['spectral_sum_percent']:.4g} %)",
            mode_line,
            f"Spectral records: {data['fft_samples']:,} samples; Δf {data['frequency_resolution_hz']:.6g} Hz; "
            f"weighted-peak transfer Δf {data['weighting_frequency_resolution_hz']:.6g} Hz; "
            f"Nyquist {data['nyquist_hz']:.6g} Hz; {coverage}.",
            f"Source sample interval: {data['max_source_interval_s']:.6g} s "
            f"(source Nyquist {data['source_nyquist_hz']:.6g} Hz).",
        ]
        if context.get("sampling_mode") == "user_defined":
            lines.append("These results describe the custom sample points, not the whole volume.")
        if data.get("spectral_sampling_reduced", False):
            lines.append(
                "This analysis uses fewer time samples than the source. "
                "The Nyquist frequency above is the highest frequency represented."
            )
        if data.get("weighted_trace_reduced", False):
            lines.append(
                "The plotted weighted-index trace is evenly thinned for display only; the peak search used every analysed sample."
            )
        if data["below_band_rms_uT"] > 1e-12:
            lines.append(
                f"Non-DC spectrum below 1 Hz: {data['below_band_rms_uT']:.6g} µT RMS (outside this LF weighting band)."
            )
        if data["above_band_rms_uT"] > 1e-12:
            lines.append(
                f"Spectrum above 100 kHz: {data['above_band_rms_uT']:.6g} µT RMS (not included in this LF screen)."
            )
        lines.extend(
            [
                "",
                "Interpretation: 1.0 / 100% is the selected reference/action-level curve after frequency weighting. "
                "A regional index below 1 is not by itself a compliance finding; localized/non-uniform exposure may require "
                "maximum-field assessment or internal induced-E dosimetry.",
            ]
        )
        self.summary.setPlainText("\n".join(lines))

        time_s = np.asarray(data["time_s"], dtype=float)
        percent = np.asarray(data["weighted_index"], dtype=float) * 100.0
        figure = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": time_s.tolist(),
                    "y": percent.tolist(),
                    "name": "Weighted regional index",
                    "hovertemplate": "%{x:.8g} s<br>%{y:.8g} %<extra></extra>",
                }
            ],
            "layout": {
                "template": "plotly_white",
                "title": {"text": f"Frequency-weighted regional index — {self._region_display_name()}", "x": 0.5},
                "margin": {"l": 80, "r": 30, "t": 60, "b": 70},
                "xaxis": {"title": {"text": "Physical time (s)"}},
                "yaxis": {"title": {"text": "Selected reference level (%)"}, "rangemode": "tozero"},
                "shapes": [
                    {
                        "type": "line",
                        "xref": "paper",
                        "x0": 0,
                        "x1": 1,
                        "y0": 100,
                        "y1": 100,
                        "line": {"dash": "dash", "width": 1.5},
                    }
                ],
                "showlegend": False,
                "uirevision": str((key, profile, int(data["fft_samples"]), data.get("analysis_mode"))),
            },
        }
        self.plot.set_figure(figure, coalesce=True)

        rows = np.asarray(data.get("spectral_top_rows", ()), dtype=float)
        if rows.ndim != 2 or rows.shape[1:] != (4,):
            rows = np.empty((0, 4), dtype=float)
        self.table.setRowCount(len(rows))
        for row, (frequency_hz, bin_rms_uT, reference_uT, ratio) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(f"{frequency_hz:.8g}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{bin_rms_uT:.8g}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{reference_uT:.8g}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{ratio * 100.0:.8g}"))

    def cleanup(self):
        self.result_window.close()
        self.plot.cleanup()
        super().cleanup()


class _AnalysisProgressRow(QWidget):
    """One independently controlled background calculation in the shared footer."""

    ACTIVE_STATES = {"queued", "running", "pausing", "paused", "cancelling"}

    def __init__(self, title, page, parent=None):
        super().__init__(parent)
        self.title = str(title)
        self.page = page
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.label = QLabel("", self)
        self.label.setObjectName("hint")
        self.label.setMinimumWidth(220)
        self.bar = QProgressBar(self)
        self.bar.setTextVisible(False)
        self.bar.setMinimumWidth(180)
        self.time_label = QLabel("", self)
        self.time_label.setObjectName("hint")
        self.time_label.setMinimumWidth(125)
        self.time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(lambda: _toggle_analysis_pause(self.page))
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(lambda: _request_analysis_cancellation(self.page))

        row.addWidget(self.label)
        row.addWidget(self.bar, 1)
        row.addWidget(self.time_label)
        row.addWidget(self.pause_button)
        row.addWidget(self.cancel_button)
        self.hide()

    def update_state(self, state):
        state = dict(state or {})
        progress_state = str(state.get("state") or "idle")
        active = progress_state in self.ACTIVE_STATES
        self.setVisible(active)
        if not active:
            return

        message = str(state.get("message") or "Calculating…")
        self.label.setText(f"{self.title}: {message}")
        self.time_label.setText(str(state.get("time_text") or ""))

        minimum = int(state.get("minimum", 0) or 0)
        maximum = int(state.get("maximum", 0) or 0)
        value = int(state.get("value", 0) or 0)
        self.bar.setRange(minimum, maximum)
        if maximum > minimum:
            self.bar.setValue(max(minimum, min(maximum, value)))

        pause_requested = bool(state.get("pause_requested", False))
        self.pause_button.setText("Run" if pause_requested else "Pause")
        self.pause_button.setVisible(progress_state != "queued")
        self.pause_button.setEnabled(bool(state.get("can_pause", False)))
        self.cancel_button.setEnabled(bool(state.get("can_cancel", False)))


class _AnalysisCard(QFrame):
    """Explanation-first launcher with target-wide cache/run state."""

    def __init__(
        self, title, question, description, basis_html, button_text, callback,
        parent=None, *, retry_callback=None,
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("exposureAnalysisCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)

        heading = QLabel(f"<b>{title}</b>", self)
        heading.setTextFormat(Qt.TextFormat.RichText)
        heading.setObjectName("sectionHeading")
        layout.addWidget(heading)

        prompt = QLabel(question, self)
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        detail = QLabel(description, self)
        detail.setWordWrap(True)
        detail.setObjectName("hint")
        layout.addWidget(detail)

        basis = QLabel(basis_html, self)
        basis.setWordWrap(True)
        basis.setOpenExternalLinks(True)
        basis.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(basis)

        cache_row = QHBoxLayout()
        cache_row.setContentsMargins(0, 0, 0, 0)
        self.cache_status = QLabel("Cache: Not calculated", self)
        self.cache_status.setObjectName("hint")
        self.cache_status.setWordWrap(True)
        cache_row.addWidget(self.cache_status, 1)
        self.retry_save_button = QPushButton("Retry save", self)
        if callable(retry_callback):
            self.retry_save_button.clicked.connect(retry_callback)
        self.retry_save_button.hide()
        cache_row.addWidget(self.retry_save_button)
        layout.addLayout(cache_row)

        self.activity_status = QLabel("", self)
        self.activity_status.setObjectName("hint")
        self.activity_status.setWordWrap(True)
        self.activity_status.hide()
        layout.addWidget(self.activity_status)
        layout.addStretch(1)

        self.open_button = QPushButton(button_text, self)
        self.open_button.clicked.connect(callback)
        layout.addWidget(self.open_button)

    def set_cache_status(self, text, *, activity=None, retry=False):
        self.cache_status.setText(f"Cache: {str(text or 'Not calculated')}")
        activity_text = str(activity or "").strip()
        self.activity_status.setText(activity_text)
        self.activity_status.setVisible(bool(activity_text))
        self.retry_save_button.setVisible(bool(retry))


class ExposureAnalysisTargetView(QWidget):
    """One prepared exposure-analysis target with its own region browser and tools."""

    busyChanged = Signal(bool)

    AGNIR_URL = "https://assets.publishing.service.gov.uk/media/5a7d9bd4e5274a676d533128/RCE-1_Power_frequency_electromagnetic_fields__melatonin_and_the_risk_of_breast_cancer_.pdf"
    ICNIRP_2010_URL = "https://www.icnirp.org/cms/upload/publications/ICNIRPLFgdl.pdf"
    ICNIRP_PULSED_URL = "https://www.icnirp.org/cms/upload/publications/ICNIRPpulsed.pdf"
    EU_2013_URL = "https://eur-lex.europa.eu/eli/dir/2013/35/oj/eng"
    ZHANG_URL = "https://pubmed.ncbi.nlm.nih.gov/9209718/"
    KAUNE_URL = "https://onlinelibrary.wiley.com/doi/abs/10.1002/bem.46"
    ROOSLI_URL = "https://link.springer.com/article/10.1186/1476-069X-7-35"

    PAGE_TITLES = {
        "home": "Exposure analysis home",
        "summary": "Exposure summary",
        "spectrum": "Spectrum",
        "fft": "FFT spectrum",
        "timing": "Timing & transitions",
        "threshold": "Threshold & duration",
        "weighted": "Frequency-weighted screening",
    }
    PAGE_EXPLANATIONS = {
        "home": (
            "Choose an analysis below to characterize this prepared exposure. "
            "Calculations are automatically cached so each configuration only needs to run once."
        ),
        "summary": "Review the descriptive field statistics calculated for the prepared target.",
        "spectrum": "Review the regional frequency content of the prepared exposure.",
        "fft": "Inspect the configurable full FFT spectrum for the prepared exposure.",
        "timing": "Review field-change timing, transitions, pulse widths and gaps.",
        "threshold": "Measure how long prepared regions remain above a selected field threshold.",
        "weighted": "Compare the prepared exposure with a selected frequency-dependent reference framework.",
    }

    def __init__(
        self, payload, selected_key=None, parent=None, *, adapter=None,
        analysis_cache: dict | None = None, saved_record=None, on_cache_changed=None,
        on_back_to_targets=None, on_region_viewer_requested=None,
        on_region_selection_changed=None,
    ):
        super().__init__(parent)
        payload_copy = copy.deepcopy(payload) if isinstance(payload, dict) else payload
        if isinstance(payload_copy, dict) and isinstance(saved_record, dict):
            context = payload_copy.setdefault("exposure_analysis_context", {})
            if isinstance(context, dict):
                context["saved_analysis_id"] = str(saved_record.get("id") or "")
                context["saved_analysis_name"] = str(saved_record.get("name") or "")
                context["saved_analysis_status"] = str(saved_record.get("status") or "stale")
        self.payload = payload_copy
        self.workspace_title = "Exposure analysis"
        self.page_titles = dict(self.PAGE_TITLES)
        context = self.payload.get("exposure_analysis_context") if isinstance(self.payload, dict) else None
        exposure_name = str(context.get("exposure_name") or "").strip() if isinstance(context, dict) else ""
        self.window_title = self.workspace_title + (f" — {exposure_name}" if exposure_name else "")
        self.project_adapter = adapter
        self._on_cache_changed = on_cache_changed
        self._on_back_to_targets = on_back_to_targets
        self._on_region_viewer_requested = on_region_viewer_requested
        self._on_region_selection_changed = on_region_selection_changed
        self._owned_region_viewer: AnalysisRegionViewer | None = None
        self._saved_analysis_id = (
            str(saved_record.get("id") or "") if isinstance(saved_record, dict) else ""
        )
        self.analysis_cache = copy.deepcopy(analysis_cache) if isinstance(analysis_cache, dict) else {}
        self._cache_persist_error_shown = False
        self._cache_persist_error = ""
        self._persisted_cache_keys = set(self.analysis_cache)
        self._dirty_cache_keys = set()
        self._pages = {}
        self._analysis_home_cards = {}
        self._pages_cleaned = False
        self._analysis_job_queue = []
        self._analysis_active_page = None
        self._analysis_preview_request_serial = 0
        self.current_page_key = "home"

        root = QVBoxLayout(self)

        self.analysis_page_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.analysis_page_splitter.setObjectName("exposureAnalysisPreparedVerticalSplitter")
        self.analysis_page_splitter.setChildrenCollapsible(False)

        self.analysis_overview_panel = self._build_preparation_overview()
        self.analysis_header = self.analysis_overview_panel
        self.analysis_page_splitter.addWidget(self.analysis_overview_panel)

        lower_panel = QWidget(self.analysis_page_splitter)
        lower_layout = QVBoxLayout(lower_panel)
        lower_layout.setContentsMargins(0, 0, 0, 0)

        self.cache_save_note = QLabel("", lower_panel)
        self.cache_save_note.setObjectName("hint")
        self.cache_save_note.setWordWrap(True)
        self.cache_save_note.hide()
        lower_layout.addWidget(self.cache_save_note)

        self.selection_note = QLabel(lower_panel)
        self.selection_note.setObjectName("hint")
        self.selection_note.setWordWrap(True)
        lower_layout.addWidget(self.selection_note)

        self.stack = QStackedWidget(lower_panel)
        self.home_page = self._build_home_page(selected_key)
        self.stack.addWidget(self.home_page)
        self._pages["home"] = self.home_page
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal, lower_panel)
        self.workspace_splitter.addWidget(self.region_tree)
        self.workspace_splitter.addWidget(self.stack)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 3)
        self.workspace_splitter.setSizes([260, 820])
        self.workspace_splitter.setChildrenCollapsible(False)
        self._show_region_browser = len(self.payload.get("regions", {})) > 1
        # Region selection has no meaning on the guided home page. Keep the
        # browser available for the individual tools, but let the home page use
        # the full width of the workspace.
        self.region_tree.hide()
        self.selection_note.hide()
        lower_layout.addWidget(self.workspace_splitter, 1)

        self.analysis_page_splitter.addWidget(lower_panel)
        self.analysis_page_splitter.setStretchFactor(0, 1)
        self.analysis_page_splitter.setStretchFactor(1, 2)
        self.analysis_page_splitter.setSizes([235, 465])
        root.addWidget(self.analysis_page_splitter, 1)

        self._analysis_footer_states = {}
        self._analysis_progress_rows = {}
        self.analysis_footer_status = QWidget(self)
        self.analysis_footer_layout = QVBoxLayout(self.analysis_footer_status)
        self.analysis_footer_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_footer_layout.setSpacing(4)
        self.analysis_footer_status.hide()
        root.addWidget(self.analysis_footer_status)

        self._queue_analysis_target_preview()
        self._refresh_home_cache_status()

    @staticmethod
    def _format_exposure_duration(payload):
        context = payload.get("exposure_analysis_context") if isinstance(payload, dict) else None
        context = context if isinstance(context, dict) else {}
        try:
            duration = float(context.get("physical_duration_s"))
        except (TypeError, ValueError):
            duration = float("nan")
        if not np.isfinite(duration) or duration < 0.0:
            return "Unknown"
        rounded = round(duration)
        duration_text = (
            f"{int(rounded):,}"
            if np.isclose(duration, rounded, rtol=1e-10, atol=1e-10)
            else f"{duration:,.6g}"
        )
        return f"{duration_text} s"

    def _build_preparation_overview(self):
        context = self.payload.get("exposure_analysis_context") if isinstance(self.payload, dict) else None
        context = context if isinstance(context, dict) else {}
        panel = ExposureAnalysisHeaderPanel(
            self,
            title=self.page_titles["home"],
            explanation=(
                "Choose an analysis below to characterize this prepared exposure. "
                "Calculations are automatically cached so each configuration only needs to run once."
            ),
            show_targets_button=callable(self._on_back_to_targets),
            show_home_button=False,
            preview_object_name="exposureAnalysisPreparedTargetPreview",
        )
        panel.setObjectName("exposureAnalysisPreparationOverview")

        self.targets_button = panel.targets_button
        if callable(self._on_back_to_targets):
            self.targets_button.clicked.connect(self._on_back_to_targets)
        self.back_button = panel.home_button
        self.back_button.clicked.connect(self.show_home)
        self.page_heading = panel.heading_label

        target_name = str(context.get("target_name") or "Prepared target")
        exposure_name = str(context.get("exposure_name") or "Exposure")
        panel.set_identity(target_name, exposure_name)
        self.preparation_target_label = panel.target_label
        self.preparation_exposure_label = panel.exposure_label

        target_kind = str(context.get("target_kind") or "")
        target_summary = []
        if target_kind == "brain_lpba40":
            target_summary.append("Type: Brain Areas")
        elif target_kind == "surface_region_volume":
            target_summary.append("Type: Surface-region exposure")
        elif target_kind == "measurement_volume":
            target_summary.append("Type: Measurement volume")
        atlas = str(context.get("atlas") or "").strip()
        if atlas:
            target_summary.append(f"Atlas: {atlas}")
        target_summary.append(f"Full exposure: {self._format_exposure_duration(self.payload)}")
        panel.set_stats_text(" • ".join(target_summary))

        prep_stats = QWidget(panel.info_panel)
        prep_layout = QGridLayout(prep_stats)
        prep_layout.setContentsMargins(0, 0, 0, 0)
        prep_layout.setHorizontalSpacing(24)
        prep_layout.setVerticalSpacing(4)

        saved_status = str(context.get("saved_analysis_status") or "").strip().lower()
        if saved_status == "current":
            preparation_state = "Current"
        elif saved_status:
            preparation_state = "Stale — historical result"
        else:
            preparation_state = "In memory"
        self.preparation_state_label = QLabel(f"Preparation: {preparation_state}", prep_stats)

        self.preparation_atlas_label = QLabel(f"Atlas: {atlas}" if atlas else "", prep_stats)
        self.preparation_atlas_label.hide()
        self.preparation_duration_label = QLabel(
            f"Full exposure: {self._format_exposure_duration(self.payload)}", prep_stats
        )
        self.preparation_duration_label.hide()
        self.preparation_depth_label = None
        if target_kind == "surface_region_volume":
            try:
                depth_mm = float(context.get("sampling_depth_mm"))
            except (TypeError, ValueError):
                depth_mm = float("nan")
            if np.isfinite(depth_mm) and depth_mm > 0.0:
                self.preparation_depth_label = QLabel(
                    f"Surface depth: {depth_mm:,.6g} mm inward", prep_stats
                )

        region_count = len(self.payload.get("regions", {})) if isinstance(self.payload, dict) else 0
        self.preparation_regions_label = QLabel(f"Prepared regions: {region_count:,}", prep_stats)
        try:
            sample_count = int(context.get("sample_count", 0) or 0)
        except (TypeError, ValueError):
            sample_count = 0
        sample_text = f"{sample_count:,}" if sample_count > 0 else "—"
        self.preparation_samples_label = QLabel(f"Sample points: {sample_text}", prep_stats)

        prep_layout.addWidget(self.preparation_state_label, 0, 0)
        prep_layout.addWidget(self.preparation_regions_label, 0, 1)
        prep_layout.addWidget(self.preparation_samples_label, 0, 2)
        if self.preparation_depth_label is not None:
            prep_layout.addWidget(self.preparation_depth_label, 1, 0)

        try:
            preparation_time = float(context.get("preparation_processing_time_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            preparation_time = 0.0
        if np.isfinite(preparation_time) and preparation_time > 0.0:
            self.preparation_time_label = QLabel(
                f"Preparation time: {_format_analysis_elapsed(preparation_time)}", prep_stats
            )
            self.preparation_time_label.setObjectName("hint")
            prep_layout.addWidget(self.preparation_time_label, 1, 1)
        else:
            self.preparation_time_label = None
        panel.extra_layout.addWidget(prep_stats)

        self.analysis_target_preview = panel.preview
        self.analysis_target_preview.region_viewer_requested.connect(self._request_region_viewer)
        panel.region_viewer_button.clicked.connect(
            lambda: self._request_region_viewer(self._analysis_target_id())
        )

        if saved_status and saved_status != "current":
            panel.set_preview_note(
                "Live model geometry. This saved preparation is stale, so the display may "
                "differ from the geometry used for its numerical results."
            )
        self.analysis_target_preview_note = panel.preview_note
        return panel

    def set_visual_active(
        self,
        active: bool,
        *,
        inactive_detail: str | None = None,
    ) -> None:
        """Keep only the currently visible embedded target preview active."""
        panel = getattr(self, "analysis_overview_panel", None)
        if panel is not None:
            panel.set_preview_rendering_active(
                bool(active),
                inactive_detail=inactive_detail,
            )

    def _queue_analysis_target_preview(self):
        """Resolve the prepared target's display scene after the header can paint."""
        self._analysis_preview_request_serial += 1
        serial = self._analysis_preview_request_serial
        preview = getattr(self, "analysis_target_preview", None)
        if preview is None:
            return
        context = self.payload.get("exposure_analysis_context") if isinstance(self.payload, dict) else None
        context = context if isinstance(context, dict) else {}
        target_id = str(context.get("target_id") or "").strip()
        target_name = str(context.get("target_name") or "Prepared target").strip()
        if self.project_adapter is None or not target_id:
            self.analysis_overview_panel.region_viewer_button.setEnabled(False)
            preview.show_empty(
                "3D target preview unavailable",
                "The saved preparation does not have a live model target available in this workspace.",
            )
            return
        preview.show_loading(f"Loading {target_name}…")
        QTimer.singleShot(0, lambda: self._resolve_analysis_target_preview(target_id, serial))

    def _resolve_analysis_target_preview(self, target_id, serial):
        if serial != self._analysis_preview_request_serial:
            return
        preview = getattr(self, "analysis_target_preview", None)
        if preview is None:
            return
        try:
            scene = self.project_adapter.analysis_region_scene(str(target_id))
            object_figure = self.project_adapter.analysis_target_preview_figure(str(target_id))
        except Exception as error:  # noqa: BLE001 - visualization must not block saved analysis
            if serial == self._analysis_preview_request_serial:
                self.analysis_overview_panel.region_viewer_button.setEnabled(False)
                preview.show_empty("3D target preview unavailable", str(error))
            return
        if serial == self._analysis_preview_request_serial:
            preview.set_scene(scene, object_figure=object_figure)
            self.analysis_overview_panel.region_viewer_button.setEnabled(True)
            self._sync_analysis_region_visuals()

    def _analysis_target_id(self):
        context = self.payload.get("exposure_analysis_context") if isinstance(self.payload, dict) else None
        context = context if isinstance(context, dict) else {}
        return str(context.get("target_id") or "").strip()

    def _visual_region_key(self):
        """Return the region that auxiliary 3D views should emphasize.

        Exposure Analysis home deliberately shows the whole object. Method pages
        follow the shared regional result selection. This is display state only.
        """
        if self.current_page_key == "home":
            return None
        key = self._selected_key()
        return str(key) if key else None

    def _sync_analysis_region_visuals(self):
        """One-way sync: analysis selection -> compact preview / Region Viewer."""
        target_id = self._analysis_target_id()
        selected_key = self._visual_region_key()
        preview = getattr(self, "analysis_target_preview", None)
        if preview is not None and preview.scene is not None and preview.scene.target_id == target_id:
            try:
                preview.set_selected_region(selected_key)
            except Exception:
                pass
        viewer = self._owned_region_viewer
        if viewer is not None and viewer.target_id == target_id:
            viewer.set_selected_region(selected_key)
        if callable(self._on_region_selection_changed):
            self._on_region_selection_changed(target_id, selected_key)

    def _region_viewer_context_note(self):
        context = self.payload.get("exposure_analysis_context") if isinstance(self.payload, dict) else None
        context = context if isinstance(context, dict) else {}
        status = str(context.get("saved_analysis_status") or "").strip().lower()
        if status == "current":
            return "Live model geometry for this current saved preparation."
        if status:
            return (
                "Live model geometry. This saved preparation is stale, so this display may "
                "differ from the geometry used for its frozen numerical results."
            )
        return "Live model geometry; prepared numerical samples are frozen separately."

    def _request_region_viewer(self, target_id):
        """Forward to the owner or open one live standalone Region Viewer."""
        target_id = str(target_id or "").strip()
        if not target_id:
            return
        selected_key = self._visual_region_key()
        context_note = self._region_viewer_context_note()
        if callable(self._on_region_viewer_requested):
            self._on_region_viewer_requested(target_id, selected_key, context_note)
            return
        if self.project_adapter is None:
            return
        try:
            # Resolve explicitly on open so deleted/edited model targets cannot
            # be masked by a compact preview scene captured earlier.
            scene = self.project_adapter.analysis_region_scene(target_id)
        except Exception as error:  # noqa: BLE001 - auxiliary visualization boundary
            if self._owned_region_viewer is not None and self._owned_region_viewer.target_id == target_id:
                self._owned_region_viewer.show_unavailable("Prepared target", str(error))
            QMessageBox.warning(self, "Region viewer", str(error))
            return
        if self._owned_region_viewer is None:
            self._owned_region_viewer = AnalysisRegionViewer()
            self._owned_region_viewer.viewer_hidden.connect(
                lambda: self.set_visual_active(True)
            )
        self.set_visual_active(
            False,
            inactive_detail="Preview is disabled while the Region Viewer is active.",
        )
        if context_note:
            self._owned_region_viewer.present(
                scene, selected_region_key=selected_key, context_note=context_note
            )
        else:
            self._owned_region_viewer.present(scene, selected_region_key=selected_key)

    def _build_home_page(self, selected_key):
        page = QWidget(self.stack)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        context = self.payload.get("exposure_analysis_context") if isinstance(self.payload, dict) else None
        context = context if isinstance(context, dict) else {}

        self.region = QComboBox(self)
        for key, entry in self.payload["regions"].items():
            self.region.addItem(entry["name"], key)
        index = self.region.findData(selected_key)
        if index >= 0:
            self.region.setCurrentIndex(index)
        self.region.hide()
        self.region_tree = _make_analysis_region_tree(self)
        if context.get("target_name") and self.region_tree.topLevelItemCount() == 1:
            item = self.region_tree.topLevelItem(0)
            if not item.text(0).startswith(str(context["target_name"])):
                item.setText(0, str(context["target_name"]) + " — " + item.text(0))
        self.region.currentIndexChanged.connect(self._home_region_changed)

        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(4, 4, 8, 8)
        body_layout.setSpacing(10)

        characterize = QLabel("Characterize the exposure", body)
        characterize.setObjectName("dialogHeading")
        body_layout.addWidget(characterize)
        characterize_note = QLabel(
            "Descriptive analyses answer different questions about what the exposure delivered. "
            "They are complementary measurements, not interchangeable dose or pass/fail metrics.",
            body,
        )
        characterize_note.setObjectName("hint")
        characterize_note.setWordWrap(True)
        body_layout.addWidget(characterize_note)

        calculate_all_row = QHBoxLayout()
        calculate_all_note = QLabel(
            "Calculations run one at a time. Uncached methods wait in the queue and start automatically.",
            body,
        )
        calculate_all_note.setObjectName("hint")
        calculate_all_note.setWordWrap(True)
        calculate_all_row.addWidget(calculate_all_note, 1)
        self.calculate_all_button = QPushButton("Calculate all uncached analyses", body)
        self.calculate_all_button.setToolTip(
            "Queue the current/default configuration of each analysis that is not already cached."
        )
        self.calculate_all_button.clicked.connect(self.calculate_all_uncached)
        calculate_all_row.addWidget(self.calculate_all_button)
        body_layout.addLayout(calculate_all_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        card_specs = [
            ("summary", _AnalysisCard(
                "Exposure summary",
                "How much field was delivered over the exposure?",
                "Mean, RMS, peak, B-time, B²-time and physical duration from the authoritative regional exposure source.",
                f'<b>Method basis:</b> cumulative/TWA/peak exposure descriptors — <a href="{self.AGNIR_URL}">AGNIR/HPA</a> · <a href="{self.ROOSLI_URL}">Röösli et al.</a>',
                "Open exposure summary…",
                self.open_exposure_summary,
                body, retry_callback=self._retry_cache_save,
            )),
            ("spectrum", _AnalysisCard(
                "Spectrum / FFT",
                "Which frequencies are present in this exposure?",
                "Regional XYZ spectrum, strongest RMS frequency components and a configurable full FFT view.",
                f'<b>Method basis:</b> harmonic/frequency characterization — <a href="{self.AGNIR_URL}">AGNIR/HPA</a> · <a href="{self.ICNIRP_2010_URL}">ICNIRP 2010</a>',
                "Open spectrum / FFT…",
                self.open_spectrum,
                body, retry_callback=self._retry_cache_save,
            )),
            ("timing", _AnalysisCard(
                "Timing & transitions",
                "How quickly does the field change, and how are pulses and gaps arranged?",
                "Regional vector slew plus drive-current pulse counts, widths, active time and between-pulse gaps.",
                f'<b>Method basis:</b> transients and complex/pulsed waveform characterization — <a href="{self.ICNIRP_PULSED_URL}">ICNIRP pulsed-field guidance</a>',
                "Open timing analysis…",
                self.open_timing,
                body, retry_callback=self._retry_cache_save,
            )),
            ("threshold", _AnalysisCard(
                "Threshold & duration",
                "How long was this region above a field level I choose?",
                "Total time above threshold, fraction of exposure, excursion count and the longest uninterrupted interval.",
                f'<b>Research precedent:</b> threshold/duration and temporal-variability descriptors — <a href="{self.ZHANG_URL}">Zhang et al.</a> · <a href="{self.KAUNE_URL}">Kaune et al.</a>',
                "Open threshold analysis…",
                self.open_threshold,
                body, retry_callback=self._retry_cache_save,
            )),
        ]
        for card_index, (card_key, card) in enumerate(card_specs):
            self._analysis_home_cards[card_key] = card
            grid.addWidget(card, card_index // 2, card_index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        body_layout.addLayout(grid)

        compare = QLabel("Compare against a reference framework", body)
        compare.setObjectName("dialogHeading")
        body_layout.addWidget(compare)
        compare_note = QLabel(
            "Reference-level screening is kept separate from descriptive exposure statistics so "
            "cumulative metrics are not mistaken for guideline comparison or biological dose.",
            body,
        )
        compare_note.setObjectName("hint")
        compare_note.setWordWrap(True)
        body_layout.addWidget(compare_note)

        weighted = _AnalysisCard(
            "Frequency-weighted screening",
            "How does this complex waveform compare with a frequency-dependent exposure guideline?",
            "Phase-aware weighted peak and conservative spectral summation using a selected ICNIRP 2010 or EU 2013/35/EU external-B curve.",
            f'<b>Reference frameworks:</b> <a href="{self.ICNIRP_2010_URL}">ICNIRP 2010</a> · <a href="{self.EU_2013_URL}">EU Directive 2013/35/EU</a>',
            "Open frequency-weighted screening…",
            self.open_frequency_weighted,
            body, retry_callback=self._retry_cache_save,
        )
        self._analysis_home_cards["weighted"] = weighted
        body_layout.addWidget(weighted)

        coverage_note = QLabel(
            "Each analysis configuration is calculated once and cached for all prepared regions of this object. "
            "Opening an uncached method or changing a calculation setting starts that configuration automatically; "
            "region selection only chooses which saved regional result you are viewing. Exposure Summary shows all regions together.",
            body,
        )
        coverage_note.setObjectName("hint")
        coverage_note.setWordWrap(True)
        body_layout.addWidget(coverage_note)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        self._sync_region_tree()
        return page

    def _tree_region_changed(self, item, _previous):
        if item is not None:
            index = self.region.findData(item.data(0, Qt.ItemDataRole.UserRole))
            if index >= 0:
                self.region.setCurrentIndex(index)

    def _sync_region_tree(self):
        item = self.region_tree._region_items.get(self._selected_key())
        if item is None:
            return
        self.region_tree.blockSignals(True)
        self.region_tree.setCurrentItem(item)
        names = [item.text(0)]
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            names.append(parent.text(0))
            parent = parent.parent()
        self.region_tree.scrollToItem(item)
        self.region_tree.blockSignals(False)
        selection_text = "Results for: " + " / ".join(reversed(names))
        node = getattr(self.region_tree, "_region_nodes", {}).get(str(self._selected_key()))
        metadata = node.metadata if node is not None and isinstance(node.metadata, dict) else {}
        if str(getattr(node, "region_kind", "") or "") == "near_surface_volume":
            details = []
            try:
                details.append(f"{float(metadata.get('sampling_depth_mm')):,.6g} mm inward")
            except (TypeError, ValueError):
                pass
            try:
                details.append(f"{float(metadata.get('volume_cm3')):,.6g} cm³ sampled volume")
            except (TypeError, ValueError):
                pass
            try:
                details.append(f"{float(metadata.get('surface_area_cm2')):,.6g} cm² surface")
            except (TypeError, ValueError):
                pass
            try:
                retained = int(metadata.get("retained_sample_count", 0) or 0)
            except (TypeError, ValueError):
                retained = 0
            try:
                source = int(metadata.get("source_voxel_count", 0) or 0)
            except (TypeError, ValueError):
                source = 0
            if retained:
                details.append(f"{retained:,} retained samples")
            if source:
                details.append(f"{source:,} source voxels")
            if details:
                selection_text += " • " + " • ".join(details)
        self.selection_note.setText(selection_text)

    def _selected_key(self):
        return self.region.currentData()

    @staticmethod
    def _cache_key_method(key):
        if not isinstance(key, tuple) or not key:
            return None
        parts = key[1:] if str(key[0]) == "__cache_meta__" else key
        if not parts:
            return None
        kind = str(parts[0])
        return kind if kind in {"summary", "spectrum", "timing", "threshold", "weighted"} else None

    def _dirty_cache_methods(self):
        return {
            method for method in (self._cache_key_method(key) for key in self._dirty_cache_keys)
            if method is not None
        }

    def _refresh_home_cache_status(self):
        if not self._analysis_home_cards:
            return

        statuses = _analysis_home_cache_statuses(self.analysis_cache)
        dirty_methods = self._dirty_cache_methods() if self._cache_persist_error else set()
        for kind, card in self._analysis_home_cards.items():
            pages = [self._pages.get(kind)]
            if kind == "spectrum":
                pages.append(self._pages.get("fft"))
            pages = [page for page in pages if page is not None]
            states = {
                str(getattr(page, "_analysis_progress_state", "")) for page in pages
            }
            running = "running" in states
            queued = "queued" in states
            cancelled = "cancelled" in states
            failed = "failed" in states

            activity = None
            retry = False
            if kind in dirty_methods:
                activity = "Save failed — calculated result is only in this window."
                retry = True
            elif "cancelling" in states:
                activity = "Cancelling…"
            elif "paused" in states:
                activity = "Paused."
            elif "pausing" in states:
                activity = "Pausing…"
            elif running:
                activity = "Running…"
            elif queued:
                activity = "Queued — waiting for the current calculation."
            elif cancelled:
                activity = (
                    "Last run cancelled — existing cached results were kept."
                    if statuses.get(kind) != "Not calculated"
                    else "Last run cancelled — no partial result was cached."
                )
            elif failed:
                activity = (
                    "Last run failed — existing cached results were kept."
                    if statuses.get(kind) != "Not calculated"
                    else "Last run failed — no result was cached."
                )
            card.set_cache_status(
                statuses.get(kind, "Not calculated"), activity=activity, retry=retry
            )

        busy = any(
            getattr(page, "_analysis_busy", False)
            or bool(getattr(page, "_weighted_pending_requests", ()))
            for page in self._pages.values()
        )
        if hasattr(self, "region_tree"):
            self.region_tree.setEnabled(not busy)
        if hasattr(self, "targets_button"):
            self.targets_button.setEnabled(not busy)
        for page in self._pages.values():
            tree = getattr(page, "result_region_tree", None)
            if isinstance(tree, QTreeWidget):
                # The frequency-weighted result browser is a read-only view of
                # already cached data. Keep it navigable while another
                # reference is calculating; setup-region controls remain locked.
                tree.setEnabled(not busy or isinstance(page, FrequencyWeightedPage))
        self.busyChanged.emit(busy)

    def _home_region_changed(self, *_args):
        """Keep already-created analysis pages on the region selected at home."""
        key = self._selected_key()
        self._sync_region_tree()
        for page_key, page in tuple(self._pages.items()):
            if page_key == "home":
                continue
            combo = getattr(page, "region", None)
            if not isinstance(combo, QComboBox):
                continue
            index = combo.findData(key)
            if index >= 0 and index != combo.currentIndex():
                combo.setCurrentIndex(index)
        self._refresh_home_cache_status()
        self._sync_analysis_region_visuals()

    def _page_region_changed(self, page):
        """Carry a region changed within an analysis page back to the target view and sibling pages."""
        combo = getattr(page, "region", None)
        if not isinstance(combo, QComboBox):
            return
        key = combo.currentData()
        index = self.region.findData(key)
        if index >= 0 and index != self.region.currentIndex():
            self.region.setCurrentIndex(index)
        else:
            self._sync_analysis_region_visuals()

    def _schedule_analysis_job(self, page, function):
        """Run at most one derived Exposure Analysis calculation at a time."""
        page._analysis_job_scheduler = self._schedule_analysis_job
        page._analysis_job_finished_callback = self._analysis_job_finished
        page._analysis_queue_cancel_callback = self._cancel_queued_analysis
        if self._analysis_active_page is None:
            self._analysis_active_page = page
            return _launch_analysis_job(page, function)
        self._analysis_job_queue.append((page, function))
        _set_analysis_queued(page)
        return None

    def _analysis_job_finished(self, page):
        if self._analysis_active_page is page:
            self._analysis_active_page = None
        QTimer.singleShot(0, lambda p=page: self._advance_analysis_job_queue(p))

    def _advance_analysis_job_queue(self, completed_page):
        # Give work that was already waiting in the target-wide FIFO queue the
        # next worker slot.  A page with its own logical follow-up queue (the
        # frequency-weighting reference queue) may then submit its next item
        # behind that work without bypassing the shared single-worker policy.
        self._start_next_analysis_job()
        released = getattr(completed_page, "_analysis_job_released", None)
        if callable(released):
            released()

    def _start_next_analysis_job(self):
        if self._analysis_active_page is not None:
            return
        while self._analysis_job_queue:
            page, function = self._analysis_job_queue.pop(0)
            if not bool(getattr(page, "_analysis_busy", False)):
                continue
            if str(getattr(page, "_analysis_progress_state", "")) != "queued":
                continue
            self._analysis_active_page = page
            message = str(
                getattr(page, "_analysis_running_message", "") or "Calculating…"
            )
            _set_analysis_progress(page.progress_label, page.progress_bar, True, message)
            refresh_profiles = getattr(page, "_refresh_profiles", None)
            if callable(refresh_profiles):
                refresh_profiles()
            _launch_analysis_job(page, function)
            return

    def _cancel_queued_analysis(self, page):
        removed = False
        retained = []
        for queued_page, function in self._analysis_job_queue:
            if queued_page is page:
                removed = True
            else:
                retained.append((queued_page, function))
        self._analysis_job_queue = retained
        if not removed:
            return
        page._set_analysis_busy(False)
        timer = getattr(page, "_analysis_progress_timer", None)
        if isinstance(timer, QTimer):
            timer.stop()
        page._analysis_progress_started_at = None
        page._analysis_pause_started_at = None
        page._analysis_progress_state = "cancelled"
        page._analysis_progress_message = "Cancelled before calculation started"
        page._analysis_progress_time_text = "Not started"
        page.progress_label.setText(page._analysis_progress_message)
        page.progress_bar.setRange(0, 1)
        page.progress_bar.setValue(0)
        page.analysis_time_label.setText(page._analysis_progress_time_text)
        _publish_analysis_status(page)

    def calculate_all_uncached(self, *_args):
        """Queue one current/default configuration for every uncached home-card method."""
        for page_key in ("summary", "spectrum", "timing", "threshold"):
            page = self._pages.get(page_key)
            if page is None:
                page = self._create_page(page_key)
            elif not getattr(page, "_analysis_busy", False):
                refresh = getattr(page, "refresh", None) or getattr(page, "recalculate", None)
                if callable(refresh):
                    refresh()

        weighted = self._pages.get("weighted")
        if weighted is None:
            weighted = self._create_page("weighted")
        if not getattr(weighted, "_analysis_busy", False):
            weighted.ensure_profile_cached(weighted.profile.currentData())
        self._refresh_home_cache_status()

    def _configure_analysis_page(self, page, page_key):
        """Attach one true QWidget analysis page to the shared target workspace."""
        page._analysis_page_key = str(page_key)
        combo = getattr(page, "region", None)
        if isinstance(combo, QComboBox):
            combo.hide()
            for form in page.findChildren(QFormLayout):
                label = form.labelForField(combo)
                if label is not None:
                    label.hide()
        if isinstance(page, FrequencyWeightedPage):
            page.region_tree.hide()
        # Each active calculation gets its own row in the shared target footer,
        # rather than duplicating a progress row inside the embedded page.
        local_progress = getattr(page, "_analysis_progress_container", None)
        if isinstance(local_progress, QWidget):
            page._analysis_progress_embedded = True
            local_progress.hide()
        page._analysis_status_callback = (
            lambda state, p=page: self._analysis_page_status_changed(p, state)
        )
        page._analysis_cache_changed_callback = self._analysis_cache_changed
        page._analysis_job_scheduler = self._schedule_analysis_job
        page._analysis_job_finished_callback = self._analysis_job_finished
        page._analysis_queue_cancel_callback = self._cancel_queued_analysis
        _publish_analysis_status(page)
        if isinstance(combo, QComboBox):
            combo.currentIndexChanged.connect(
                lambda _index, p=page: self._page_region_changed(p)
            )
        return page

    def _retry_cache_save(self, *_args):
        self._analysis_cache_changed()

    def _analysis_cache_changed(self, _request_key=None):
        """Persist completed target results and expose any unsaved method clearly."""
        if self.project_adapter is None or not self._saved_analysis_id:
            self._refresh_home_cache_status()
            return

        self._dirty_cache_keys.update(set(self.analysis_cache) - self._persisted_cache_keys)
        if _request_key is not None:
            self._dirty_cache_keys.update((_request_key, _analysis_cache_meta_key(_request_key)))
        if not self._dirty_cache_keys:
            self._cache_persist_error = ""
            self.cache_save_note.hide()
            self._refresh_home_cache_status()
            return

        try:
            record = self.project_adapter.exposure_analysis(self._saved_analysis_id)
            merged = dict(record.get("analysis_cache") or {})
            for key in self._dirty_cache_keys:
                if key in self.analysis_cache:
                    merged[key] = self.analysis_cache[key]
            self.project_adapter.update_exposure_analysis_cache(self._saved_analysis_id, merged)
            self.analysis_cache.clear()
            self.analysis_cache.update(merged)
        except Exception as error:  # noqa: BLE001 - project persistence boundary
            self._cache_persist_error = str(error)
            self.cache_save_note.setText(
                "Save failed. Calculated results remain available in this window, but the project "
                "cache was not updated. Use Retry save on an affected analysis card. "
                + self._cache_persist_error
            )
            self.cache_save_note.show()
            self._refresh_home_cache_status()
            if not self._cache_persist_error_shown:
                self._cache_persist_error_shown = True
                QMessageBox.warning(
                    self,
                    "Cache Exposure Analysis",
                    "The analysis result was calculated, but Field Workbench could not update its project cache:\n\n"
                    + self._cache_persist_error,
                )
            return

        self._persisted_cache_keys = set(self.analysis_cache)
        self._dirty_cache_keys.clear()
        self._cache_persist_error_shown = False
        self._cache_persist_error = ""
        self.cache_save_note.hide()
        self._refresh_home_cache_status()
        if callable(self._on_cache_changed):
            self._on_cache_changed(self._saved_analysis_id)

    def _analysis_page_status_changed(self, page, state):
        state = dict(state or {})
        self._analysis_footer_states[id(page)] = state
        page_key = str(getattr(page, "_analysis_page_key", "") or "")
        title = self.page_titles.get(page_key, "Analysis")
        row = self._analysis_progress_rows.get(id(page))
        progress_state = str(state.get("state") or "idle")
        if row is None and progress_state in _AnalysisProgressRow.ACTIVE_STATES:
            row = _AnalysisProgressRow(title, page, self.analysis_footer_status)
            self.analysis_footer_layout.addWidget(row)
            self._analysis_progress_rows[id(page)] = row
        if row is not None:
            row.update_state(state)
        active = any(
            str(item.get("state") or "idle") in _AnalysisProgressRow.ACTIVE_STATES
            for item in self._analysis_footer_states.values()
        )
        self.analysis_footer_status.setVisible(active)
        self._refresh_home_cache_status()

    def _create_page(self, page_key):
        selected = self._selected_key()
        cache_keys_before = set(self.analysis_cache.keys())
        if page_key == "summary":
            page = ExposureSummaryPage(
                self.payload, selected, self.stack, analysis_cache=self.analysis_cache
            )
        elif page_key == "spectrum":
            page = SpectrumAnalysisPage(self.payload, selected, self.stack, analysis_cache=self.analysis_cache)
        elif page_key == "fft":
            page = FFTSpectrumPage(self.payload, selected, self.stack, analysis_cache=self.analysis_cache)
        elif page_key == "timing":
            page = TimingAnalysisPage(self.payload, selected, self.stack, analysis_cache=self.analysis_cache)
        elif page_key == "threshold":
            page = ThresholdDurationPage(self.payload, selected, self.stack, analysis_cache=self.analysis_cache)
        elif page_key == "weighted":
            page = FrequencyWeightedPage(self.payload, selected, self.stack, analysis_cache=self.analysis_cache)
        else:
            raise KeyError(page_key)
        self._configure_analysis_page(page, page_key)
        if set(self.analysis_cache.keys()) != cache_keys_before:
            # Short analyses can finish synchronously inside the page constructor,
            # before _configure_analysis_page() has attached the persistence callback.
            self._analysis_cache_changed(None)
        if page_key == "spectrum":
            page.fftRequested.connect(self.open_fft)
        self.stack.addWidget(page)
        self._pages[page_key] = page
        # Page constructors can submit their automatic first calculation before
        # _configure_analysis_page() has attached the target-view status callback.
        # Refresh once the page is registered so shared selection controls reflect
        # that already-running/queued job immediately, rather than waiting for the
        # worker's first progress signal.
        self._refresh_home_cache_status()
        return page

    def _show_page(self, page_key):
        page = self._pages.get(page_key)
        if page is None:
            page = self._create_page(page_key)
        combo = getattr(page, "region", None)
        if isinstance(combo, QComboBox):
            index = combo.findData(self._selected_key())
            if index >= 0 and index != combo.currentIndex():
                combo.setCurrentIndex(index)
        if page_key not in ("home", "weighted"):
            refresh = getattr(page, "refresh", None) or getattr(page, "recalculate", None)
            if callable(refresh):
                refresh()
        self.current_page_key = page_key
        on_home = page_key == "home"
        self.analysis_overview_panel.set_page(
            self.page_titles[page_key],
            self.PAGE_EXPLANATIONS.get(page_key, "Review this prepared exposure analysis."),
        )
        self.analysis_overview_panel.set_navigation(
            targets=callable(self._on_back_to_targets),
            home=not on_home,
        )
        self.stack.setCurrentWidget(page)
        owns_region_browser = bool(getattr(page, "_owns_region_browser", False))
        self.region_tree.setVisible(
            self._show_region_browser and not on_home and not owns_region_browser
        )
        self.selection_note.setVisible(not on_home and not owns_region_browser)
        self._sync_analysis_region_visuals()
        return page

    def show_home(self, *_args):
        self._refresh_home_cache_status()
        self.current_page_key = "home"
        self.analysis_overview_panel.set_page(
            self.page_titles["home"],
            self.PAGE_EXPLANATIONS["home"],
        )
        self.analysis_overview_panel.set_navigation(
            targets=callable(self._on_back_to_targets),
            home=False,
        )
        self.region_tree.hide()
        self.selection_note.hide()
        self.stack.setCurrentWidget(self.home_page)
        self._sync_analysis_region_visuals()

    def _cancel_running_analyses(self):
        for page in self._pages.values():
            _request_analysis_cancellation(page)

    def cancel_running_analyses(self):
        self._cancel_running_analyses()

    def is_busy(self):
        return any(
            getattr(page, "_analysis_busy", False)
            or bool(getattr(page, "_weighted_pending_requests", ()))
            for page in self._pages.values()
        )

    def open_exposure_summary(self, *_args):
        return self._show_page("summary")

    def open_spectrum(self, *_args):
        return self._show_page("spectrum")

    def open_fft(self, *_args):
        return self._show_page("fft")

    def open_timing(self, *_args):
        return self._show_page("timing")

    def open_threshold(self, *_args):
        return self._show_page("threshold")

    def open_frequency_weighted(self, *_args):
        return self._show_page("weighted")

    def _cleanup_pages(self):
        if self._pages_cleaned:
            return
        self._pages_cleaned = True
        for page in tuple(self._pages.values()):
            cleanup = getattr(page, "cleanup", None)
            if callable(cleanup):
                cleanup()

    def cleanup(self):
        self.cancel_running_analyses()
        self._analysis_preview_request_serial += 1
        preview = getattr(self, "analysis_target_preview", None)
        if preview is not None:
            preview.cleanup()
        viewer = self._owned_region_viewer
        self._owned_region_viewer = None
        if viewer is not None:
            viewer.cleanup()
            viewer.close()
        self._cleanup_pages()


class ExposureAnalysisWindow(QDialog):
    """Taskbar-visible modeless window for one prepared Exposure Analysis resource.

    Saved Results manages this window's lifetime explicitly rather than making
    it a native transient child.  The analysis pages themselves are normal
    QWidget children of :class:`ExposureAnalysisTargetView`; this shell is the
    only top-level window needed for a reopened saved analysis.
    """

    def __init__(
        self, payload, selected_key=None, *, adapter=None,
        analysis_cache=None, saved_record=None, on_cache_changed=None,
    ):
        super().__init__(None)
        # Reopened analyses are normal application windows.  Saved Results
        # retains/closes them logically, but no native transient-parent is used
        # so they remain visible in the desktop taskbar/window group.
        configure_application_window(self)
        self.analysis_view = ExposureAnalysisTargetView(
            payload,
            selected_key=selected_key,
            parent=self,
            adapter=adapter,
            analysis_cache=analysis_cache,
            saved_record=saved_record,
            on_cache_changed=on_cache_changed,
        )
        self.setWindowTitle(self.analysis_view.window_title)
        self.resize(1080, 840)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.analysis_view, 1)

    def reject(self):
        self.analysis_view.cleanup()
        super().reject()

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        # Background analysis jobs are independently retained until their worker
        # threads stop.  Closing an owned window therefore requests cancellation
        # but never vetoes owner/application shutdown.
        self.analysis_view.cleanup()
        super().closeEvent(event)


class ExposureActionsMixin:
    def _wrap_render_result_view(self, viewer, job, isolated_result):
        payload = isolated_result.get("payload", {})
        data = payload.get("measurement_regions")
        if not isinstance(data, dict):
            return viewer
        hierarchy = MeasurementRegions(tuple(
            AnalysisRegionNode(
                obj["key"], obj["name"], (obj["label_id"],),
                tuple(
                    AnalysisRegionNode(child["key"], child["name"], (child["label_id"],))
                    for child in obj.get("children", [])
                ),
            )
            for obj in data["objects"]
        ))
        panel = ExposureRegionPanel(hierarchy, measurement=True)
        panel.analysis_data = data.get("waveform")
        if isinstance(data.get("metrics"), dict):
            panel.set_frame_metric_data(data["metrics"])
        else:
            panel.mode_controls.setVisible(bool(data["objects"]))
        viewer.frameChanged.connect(panel.request_frame)
        panel.region_selected.connect(viewer.set_brain_region_highlight)
        return ExposureResultTab(viewer, panel)

    def _current_region_tab(self):
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, ExposureResultTab) else None

    def _add_region_actions(self, layout):
        self.regions_button = QPushButton("Show / hide regions")
        self.regions_button.setVisible(False)
        self.regions_button.clicked.connect(self._toggle_regions)
        layout.addWidget(self.regions_button)

    def _update_region_actions(self):
        if not hasattr(self, "regions_button"):
            return
        self.regions_button.setVisible(self._current_region_tab() is not None)

    def _toggle_regions(self, *_args):
        tab = self._current_region_tab()
        if tab is not None:
            tab.toggle_regions()

    def _export_region_video(self):
        # Preserve panel state while releasing the Chromium surface for the
        # existing single-renderer video-export workflow.
        from .waveform_gl_view import WaveformGLView
        from .waveform_video_dialog import WaveformVideoExportDialog
        tab = self._current_region_tab()
        if tab is None or not tab.viewer.can_export_video():
            return
        view = tab.viewer
        view.stop_playback()

        def launch(camera):
            try:
                payload = view.export_payload()
                if isinstance(camera, dict):
                    payload["initial_camera"] = copy.deepcopy(camera)
                if tab.splitter.sizes()[1] > 0:
                    payload["brain_region_video_table"] = tab.region_panel.video_table_snapshot()
                    frames = tab.region_panel.video_frame_metric_payload()
                    if isinstance(frames, dict):
                        payload["brain_region_frame_metrics"] = frames
                else:
                    payload["brain_region_video_table"] = None
            except Exception as error:
                QMessageBox.warning(self, "Unable to export video", str(error))
                return
            sizes = tab.splitter.sizes()
            placeholder = QLabel("Video export in progress…")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tab.splitter.replaceWidget(0, placeholder)
            view.cleanup()
            view.deleteLater()

            def run():
                dialog = WaveformVideoExportDialog(payload, parent=self)
                centre_window_on_parent(dialog, self)
                try:
                    if dialog.exec() == QDialog.DialogCode.Accepted and dialog.output_path is not None:
                        self.calculation_status.showMessage(f"Video exported to {dialog.output_path}")
                finally:
                    dialog.deleteLater()
                    restored = WaveformGLView(payload, initial_camera=camera if isinstance(camera, dict) else None)
                    tab.splitter.replaceWidget(0, restored)
                    tab.viewer = restored
                    connector = getattr(self, "_connect_result_camera_viewer", None)
                    if callable(connector):
                        connector(restored)
                    restored.frameChanged.connect(tab.region_panel.request_frame)
                    tab.region_panel.region_selected.connect(restored.set_brain_region_highlight)
                    tab.region_panel._selection_changed()
                    tab.splitter.setSizes(sizes)
                    restored.refresh_viewport()
                    placeholder.deleteLater()
                    self._render_tab_changed(self.tabs.currentIndex())
            QTimer.singleShot(0, run)
        view.request_camera(launch)
