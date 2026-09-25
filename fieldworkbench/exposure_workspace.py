"""Wizard-style entry point for authoritative saved-exposure analysis."""
from __future__ import annotations

import copy
import math
import threading
import time

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .brain_analysis import lpba40_region_hierarchy
from .exposure_actions import ExposureAnalysisTargetView
from .exposure_analysis import (
    ExposureAnalysisCancelled,
    exposure_analysis_targets,
    prepare_exposure_analysis_payload,
)
from .exposures import normalize_exposure_protocol
from .window_utils import configure_application_window


# Keep preparation workers alive independently of the auxiliary window so that
# closing the window can request cancellation without destroying a running QThread.
_ACTIVE_EXPOSURE_PREPARATION_JOBS = {}


class _ExposureAnalysisBuildWorker(QObject):
    """Run one frozen preparation in a worker thread."""

    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(object)

    def __init__(self, adapter, exposure, *, target_id):
        super().__init__()
        self.adapter = adapter
        self.exposure = copy.deepcopy(exposure)
        self.target_id = str(target_id)
        self.cancel_event = threading.Event()

    def request_cancellation(self):
        self.cancel_event.set()

    @Slot()
    def run(self):
        try:
            result = prepare_exposure_analysis_payload(
                self.adapter,
                self.exposure,
                target_id=self.target_id,
                progress_callback=self.progress.emit,
                cancel_requested=self.cancel_event.is_set,
            )
        except ExposureAnalysisCancelled:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 - worker boundary
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class ExposureAnalysisWorkspace(QDialog):
    """Taskbar-visible workspace for choosing/preparing saved-exposure analyses."""

    def __init__(self, adapter, exposure, *, project_label=None, on_cache_changed=None):
        super().__init__(None)
        # This is a real Field Workbench workspace, not a transient dialog.
        # Keep it native-parentless so desktop environments give it its own
        # taskbar entry; Exposure Editor retains/closes it as the logical owner.
        configure_application_window(self)
        self.adapter = adapter
        self.exposure = exposure
        self.project_label = str(project_label or "Current project model")
        self.on_cache_changed = on_cache_changed
        self._target_views = {}
        self._active_view = None
        self._items = {}
        self._targets = {target["id"]: target for target in exposure_analysis_targets(adapter)}
        self._duration = float(normalize_exposure_protocol(exposure.get("protocol"))["duration_s"])

        self._preparation_thread: QThread | None = None
        self._preparation_worker: _ExposureAnalysisBuildWorker | None = None
        self._preparation_target_id = ""
        self._preparation_started_at: float | None = None
        self._preparation_cancel_requested = False
        self._prepared_record = None

        self.setWindowTitle("Exposure analysis — " + str(exposure.get("name") or "Exposure"))
        self.resize(1080, 700)
        self.setMinimumSize(760, 520)

        root = QVBoxLayout(self)
        self.stack = QStackedWidget(self)
        self.chooser_page = self._build_chooser_page()
        self.preparation_page = self._build_preparation_page()
        self.stack.addWidget(self.chooser_page)
        self.stack.addWidget(self.preparation_page)
        root.addWidget(self.stack, 1)

        if self._items:
            self.target_tree.setCurrentItem(next(iter(self._items.values())))
        else:
            self.target_title.setText("Target: No analysis objects are available")
            self.target_details.setText(
                "Add an SRI24 brain or assign a supported volume the Measurement role in the model."
            )
            self.saved_analysis_status.setText("")
            self.create_button.setEnabled(False)
            self.open_analysis_button.setEnabled(False)

    def _build_chooser_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        heading = QLabel("Create exposure analysis", page)
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        intro = QLabel(
            "Choose a target, then open one of its saved analyses or create a fresh analysis preparation. "
            "Selecting an object only browses its information and saved work; it does not calculate fields.",
            page,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Keep the selected target/exposure identity above both tables.  The
        # tables then answer two separate questions: which object, and which
        # saved analysis of that object?
        self.target_title = QLabel("Target: Choose an object", page)
        self.target_title.setObjectName("dialogHeading")
        self.target_title.setWordWrap(True)
        layout.addWidget(self.target_title)

        self.exposure_summary = QLabel(
            f"Exposure: {self.exposure.get('name') or 'Exposure'} • Full duration: {self._format_duration(self._duration)}",
            page,
        )
        self.exposure_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.exposure_summary)

        self.target_details = QLabel("", page)
        self.target_details.setWordWrap(True)
        self.target_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.target_details)

        self.saved_analysis_status = QLabel("", page)
        self.saved_analysis_status.setObjectName("hint")
        self.saved_analysis_status.setWordWrap(True)
        layout.addWidget(self.saved_analysis_status)
        layout.addSpacing(6)

        split = QSplitter(Qt.Orientation.Horizontal, page)

        object_panel = QWidget(split)
        object_layout = QVBoxLayout(object_panel)
        object_layout.setContentsMargins(0, 0, 6, 0)
        object_heading = QLabel("Available objects", object_panel)
        object_heading.setObjectName("sectionHeading")
        object_layout.addWidget(object_heading)
        self.target_tree = QTreeWidget(object_panel)
        self.target_tree.setHeaderLabels(["Object", "Type"])
        self.target_tree.setRootIsDecorated(False)
        self.target_tree.setItemsExpandable(False)
        self.target_tree.setAlternatingRowColors(True)
        self.target_tree.setMinimumWidth(270)
        self.target_tree.setColumnWidth(0, 205)
        object_layout.addWidget(self.target_tree, 1)
        self.create_button = QPushButton("Create new analysis", object_panel)
        self.create_button.setObjectName("primaryButton")
        self.create_button.setToolTip(
            "Prepare a fresh analysis from the current exposure and target; existing saved analyses are preserved"
        )
        self.create_button.clicked.connect(self._create_selected_analysis)
        object_layout.addWidget(self.create_button)

        saved_panel = QWidget(split)
        saved_layout = QVBoxLayout(saved_panel)
        saved_layout.setContentsMargins(6, 0, 0, 0)
        saved_heading = QLabel("Saved analyses for selected object", saved_panel)
        saved_heading.setObjectName("sectionHeading")
        saved_layout.addWidget(saved_heading)
        self.saved_analysis_tree = QTreeWidget(saved_panel)
        self.saved_analysis_tree.setObjectName("exposureWorkspaceSavedAnalysisTree")
        self.saved_analysis_tree.setHeaderLabels(
            ["Name", "Status", "Cached work", "Updated (UTC)", "Stored size"]
        )
        self.saved_analysis_tree.setRootIsDecorated(False)
        self.saved_analysis_tree.setItemsExpandable(False)
        self.saved_analysis_tree.setAlternatingRowColors(True)
        self.saved_analysis_tree.setColumnWidth(0, 235)
        self.saved_analysis_tree.setColumnWidth(1, 80)
        self.saved_analysis_tree.setColumnWidth(2, 135)
        self.saved_analysis_tree.setColumnWidth(3, 145)
        self.saved_analysis_tree.setColumnWidth(4, 90)
        self.saved_analysis_tree.itemSelectionChanged.connect(self._saved_selection_changed)
        self.saved_analysis_tree.itemDoubleClicked.connect(self._open_saved_analysis_item)
        saved_layout.addWidget(self.saved_analysis_tree, 1)
        self.saved_analysis_hint = QLabel(
            "Saved analyses for the selected exposure and object appear here. Current entries match the live project; stale entries remain available as historical results.",
            saved_panel,
        )
        self.saved_analysis_hint.setObjectName("hint")
        self.saved_analysis_hint.setWordWrap(True)
        saved_layout.addWidget(self.saved_analysis_hint)
        self.open_analysis_button = QPushButton("Open selected analysis", saved_panel)
        self.open_analysis_button.setObjectName("accentButton")
        self.open_analysis_button.clicked.connect(self._open_selected_analysis)
        self.open_analysis_button.setEnabled(False)
        saved_layout.addWidget(self.open_analysis_button)

        split.addWidget(object_panel)
        split.addWidget(saved_panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([330, 690])
        layout.addWidget(split, 1)

        for target in self._targets.values():
            type_label = "Brain Areas" if target["kind"] == "brain_lpba40" else "Measurement volume"
            item = QTreeWidgetItem([target["label"], type_label])
            item.setData(0, Qt.ItemDataRole.UserRole, target["id"])
            self.target_tree.addTopLevelItem(item)
            self._items[target["id"]] = item

        self._saved_rows_by_id = {}
        self.target_tree.currentItemChanged.connect(self._selected)
        self.create_button.setEnabled(bool(self._items))
        return page

    def _build_preparation_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 10, 18, 10)

        self.preparation_title = QLabel("Preparing exposure analysis", page)
        self.preparation_title.setObjectName("dialogHeading")
        self.preparation_title.setWordWrap(True)
        layout.addWidget(self.preparation_title)

        self.preparation_summary = QLabel("", page)
        self.preparation_summary.setWordWrap(True)
        self.preparation_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.preparation_summary)
        layout.addSpacing(12)

        self.preparation_progress_label = QLabel("Starting preparation…", page)
        self.preparation_progress_label.setWordWrap(True)
        layout.addWidget(self.preparation_progress_label)

        self.preparation_progress_bar = QProgressBar(page)
        self.preparation_progress_bar.setRange(0, 0)
        layout.addWidget(self.preparation_progress_bar)

        self.preparation_progress_detail = QLabel("", page)
        self.preparation_progress_detail.setObjectName("hint")
        self.preparation_progress_detail.setWordWrap(True)
        self.preparation_progress_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.preparation_progress_detail)
        layout.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_preparation_button = QPushButton("Cancel preparation", page)
        self.cancel_preparation_button.clicked.connect(self._cancel_preparation)
        buttons.addWidget(self.cancel_preparation_button)
        layout.addLayout(buttons)
        return page

    @staticmethod
    def _format_duration(value):
        value = float(value)
        rounded = round(value)
        if math.isclose(value, rounded, rel_tol=1.0e-10, abs_tol=1.0e-10):
            return f"{int(rounded):,} s"
        return f"{value:,.6g} s"

    @staticmethod
    def _format_bytes(value):
        try:
            size = max(0, int(value))
        except (TypeError, ValueError):
            return "—"
        units = ("B", "KiB", "MiB", "GiB")
        amount = float(size)
        unit = units[0]
        for candidate in units:
            unit = candidate
            if amount < 1024.0 or candidate == units[-1]:
                break
            amount /= 1024.0
        return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"

    @staticmethod
    def _display_timestamp(value):
        text = str(value or "").replace("T", " ").replace("Z", "")
        if "." in text:
            text = text.split(".", 1)[0]
        return text or "—"

    @staticmethod
    def _cached_work_text(record):
        labels = {
            "summary": "Summary",
            "spectrum": "Spectrum/FFT",
            "timing": "Timing",
            "threshold": "Threshold",
            "weighted": "Frequency weighted",
        }
        counts = record.get("cached_tool_counts")
        counts = counts if isinstance(counts, dict) else {}
        parts = []
        known = 0
        for key, label in labels.items():
            try:
                count = max(0, int(counts.get(key, 0) or 0))
            except (TypeError, ValueError):
                count = 0
            known += count
            if count:
                parts.append(label if count == 1 else f"{label} ×{count:,}")
        try:
            total = max(0, int(record.get("cached_result_count", 0) or 0))
        except (TypeError, ValueError):
            total = known
        if total > known:
            parts.append(f"Other ×{total - known:,}")
        if total <= 0:
            return "Preparation only", "Preparation cache is stored; no derived analysis page has been calculated yet."
        tool_count = len(parts)
        compact = f"{tool_count:,} tool{'s' if tool_count != 1 else ''} • {total:,} config{'s' if total != 1 else ''}"
        return compact, "Preparation cache plus: " + ", ".join(parts)

    def _saved_analysis_rows(self, target_id):
        exposure_id = str(self.exposure.get("id") or "")
        try:
            rows = self.adapter.list_exposure_analyses(live_status=True)
        except Exception:
            return []
        matches = [
            row for row in rows
            if str(row.get("exposure_id") or "") == exposure_id
            and str(row.get("target_id") or "") == str(target_id)
        ]
        return sorted(
            matches,
            key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""),
            reverse=True,
        )

    @staticmethod
    def _saved_analysis_summary(rows):
        if not rows:
            return "Saved analyses: None"
        current = sum(str(row.get("status") or "").lower() == "current" for row in rows)
        stale = len(rows) - current
        parts = []
        if current:
            parts.append(f"{current:,} current")
        if stale:
            parts.append(f"{stale:,} stale")
        return f"Saved analyses: {len(rows):,} — " + " • ".join(parts)

    def _measurement_details(self, target_id):
        settings = self.adapter.measurement_properties(str(target_id))
        geometry = self.adapter.measurement_geometry_info(str(target_id))
        sample = self.adapter.measurement_sample_points(str(target_id))
        mode = str(settings.get("sampling_mode") or "generated")
        if mode == "user_defined":
            sampling = "Custom XYZ points"
        else:
            quality = str(sample.get("quality") or settings.get("quality") or "generated")
            sampling = f"Generated ({quality})"
        count = len(sample.get("world_points_m", ()))
        return [
            "Type: Measurement volume",
            f"Shape: {geometry['shape_label']} — {geometry['size_summary']}",
            f"Volume: {float(geometry['volume_cm3']):,.6g} cm³",
            f"Sampling: {sampling}",
            f"Sample points: {count:,}",
        ]

    def _brain_details(self):
        hierarchy = lpba40_region_hierarchy()
        return [
            "Type: Brain Areas",
            "Atlas: LPBA40",
            f"Atlas regions: {len(hierarchy.label_ids):,} labelled regions plus grouped views",
            "Sampling: up to 900 volume-weighted samples per atlas label during preparation",
        ]

    def _refresh_saved_analysis_tree(self, rows):
        previous = self._selected_saved_analysis_id()
        self.saved_analysis_tree.clear()
        self._saved_rows_by_id = {
            str(row.get("id") or ""): copy.deepcopy(row)
            for row in rows if str(row.get("id") or "")
        }
        selected = None
        for record in rows:
            analysis_id = str(record.get("id") or "")
            status = str(record.get("status") or "stale").lower()
            cached_work, cache_tooltip = self._cached_work_text(record)
            item = QTreeWidgetItem([
                str(record.get("name") or analysis_id or "Exposure analysis"),
                "Current" if status == "current" else "Stale",
                cached_work,
                self._display_timestamp(record.get("updated_at") or record.get("created_at")),
                self._format_bytes(record.get("compressed_bytes")),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, analysis_id)
            item.setToolTip(2, cache_tooltip)
            if status == "current":
                item.setForeground(1, QBrush(QColor("#15803d")))
                item.setToolTip(1, "Matches the current exposure, model and target dependencies.")
            else:
                item.setForeground(1, QBrush(QColor("#b45309")))
                item.setToolTip(1, "Historical saved analysis. It remains readable but is not reused as a current preparation.")
            self.saved_analysis_tree.addTopLevelItem(item)
            if analysis_id == previous:
                selected = item
        if selected is not None:
            self.saved_analysis_tree.setCurrentItem(selected)
        elif self.saved_analysis_tree.topLevelItemCount():
            self.saved_analysis_tree.setCurrentItem(self.saved_analysis_tree.topLevelItem(0))
        self._saved_selection_changed()

    def _refresh_target_info(self, target_id):
        target = self._targets.get(str(target_id))
        if target is None:
            self.target_title.setText("Target: Choose an object")
            self.target_details.clear()
            self.saved_analysis_status.clear()
            self._refresh_saved_analysis_tree([])
            self.create_button.setEnabled(False)
            return
        self.target_title.setText("Target: " + str(target["label"]))
        try:
            lines = self._brain_details() if target["kind"] == "brain_lpba40" else self._measurement_details(target_id)
            self.target_details.setText(" • ".join(lines))
        except Exception as error:  # noqa: BLE001 - browsing should remain usable for invalid sampling settings
            type_label = "Brain Areas" if target["kind"] == "brain_lpba40" else "Measurement volume"
            self.target_details.setText(
                f"Type: {type_label} • Target details could not be completed: {error}"
            )
        rows = self._saved_analysis_rows(target_id)
        self.saved_analysis_status.setText(self._saved_analysis_summary(rows))
        self._refresh_saved_analysis_tree(rows)
        self.create_button.setEnabled(not self._busy())

    def _selected(self, item, *_args):
        if item is None:
            self._refresh_target_info(None)
            return
        self._refresh_target_info(item.data(0, Qt.ItemDataRole.UserRole))

    def _selected_saved_analysis_id(self):
        item = self.saved_analysis_tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _saved_selection_changed(self, *_args):
        self.open_analysis_button.setEnabled(
            not self._busy() and self._selected_saved_analysis_id() is not None
        )

    def _discard_target_view(self, analysis_id, view):
        self._target_views.pop(str(analysis_id), None)
        if self._active_view is view:
            self._active_view = None
        self.stack.removeWidget(view)
        view.cleanup()
        view.deleteLater()

    def _load_saved_analysis_view(self, analysis_id):
        analysis_id = str(analysis_id or "")
        if not analysis_id:
            return None
        view = self._target_views.get(analysis_id)
        if view is not None:
            saved = next(
                (row for row in self.adapter.exposure_analyses if str(row.get("id") or "") == analysis_id),
                None,
            )
            changed_cache = saved is not None and saved.get("cache_sha256") != getattr(view, "_workspace_cache_sha", None)
            if (saved is None or changed_cache) and not view._dirty_cache_keys:
                self._discard_target_view(analysis_id, view)
                view = None
        if view is None:
            try:
                record = self.adapter.exposure_analysis(analysis_id)
            except Exception:
                return None
            view = self._attach_record(record)
        return view

    def _attach_record(self, record):
        analysis_id = str(record.get("id") or "")
        if not analysis_id:
            raise ValueError("Saved Exposure Analysis record has no identifier.")
        existing = self._target_views.get(analysis_id)
        if existing is not None:
            self._discard_target_view(analysis_id, existing)
        view = ExposureAnalysisTargetView(
            record["payload"],
            parent=self.stack,
            adapter=self.adapter,
            analysis_cache=record.get("analysis_cache"),
            saved_record=record,
            on_cache_changed=lambda changed_id, key=analysis_id: self._cache_changed(key, changed_id),
        )
        view._workspace_cache_sha = record.get("cache_sha256")
        view.busyChanged.connect(lambda _busy: self._sync_busy())
        self.stack.addWidget(view)
        self._target_views[analysis_id] = view
        return view

    def _show_target_view(self, view):
        self._active_view = view
        view.show_home()
        self.stack.setCurrentWidget(view)
        self._sync_busy()

    def _open_analysis_id(self, analysis_id):
        if self._busy():
            return
        view = self._load_saved_analysis_view(analysis_id)
        if view is None:
            QMessageBox.warning(
                self,
                "Open exposure analysis",
                "The selected saved analysis is no longer available.",
            )
            item = self.target_tree.currentItem()
            if item is not None:
                self._refresh_target_info(item.data(0, Qt.ItemDataRole.UserRole))
            return
        self._show_target_view(view)

    def _open_selected_analysis(self, *_args):
        analysis_id = self._selected_saved_analysis_id()
        if analysis_id:
            self._open_analysis_id(analysis_id)

    def _open_saved_analysis_item(self, item, _column):
        analysis_id = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if analysis_id:
            self._open_analysis_id(str(analysis_id))

    def _create_selected_analysis(self, *_args):
        item = self.target_tree.currentItem()
        if item is None or self._busy():
            return
        target_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        if target_id:
            self._start_preparation(target_id)

    def _start_preparation(self, target_id):
        if self._preparation_thread is not None:
            return
        target = self._targets.get(str(target_id))
        if target is None:
            return

        self._preparation_target_id = str(target_id)
        self._preparation_started_at = time.monotonic()
        self._preparation_cancel_requested = False
        self._prepared_record = None

        self.preparation_title.setText(f"Preparing {target['label']}")
        self.preparation_summary.setText(
            f"Exposure: {self.exposure.get('name') or 'Exposure'}\n"
            f"Model: {self.project_label}\n"
            f"Full exposure: {self._format_duration(self._duration)}"
        )
        self.preparation_progress_label.setText("Starting exposure analysis preparation…")
        self.preparation_progress_detail.clear()
        self.preparation_progress_bar.setRange(0, 0)
        self.cancel_preparation_button.setText("Cancel preparation")
        self.cancel_preparation_button.setEnabled(True)

        self._active_view = None
        self.stack.setCurrentWidget(self.preparation_page)
        self.target_tree.setEnabled(False)
        self.saved_analysis_tree.setEnabled(False)
        self.create_button.setEnabled(False)
        self.open_analysis_button.setEnabled(False)

        frozen_adapter = self.adapter.detached_copy()
        worker = _ExposureAnalysisBuildWorker(frozen_adapter, self.exposure, target_id=target_id)
        thread = QThread()
        worker.moveToThread(thread)
        self._preparation_worker = worker
        self._preparation_thread = thread
        _ACTIVE_EXPOSURE_PREPARATION_JOBS[thread] = worker
        self.destroyed.connect(lambda *_args, w=worker: w.request_cancellation())
        thread.started.connect(worker.run)
        worker.progress.connect(self._preparation_progress)
        worker.completed.connect(self._preparation_completed)
        worker.failed.connect(self._preparation_failed)
        worker.cancelled.connect(self._preparation_cancelled)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._preparation_thread_finished)

        def release_job(t=thread):
            _ACTIVE_EXPOSURE_PREPARATION_JOBS.pop(t, None)
            t.deleteLater()

        thread.finished.connect(release_job)
        thread.start()

    @Slot(object)
    def _preparation_progress(self, update):
        if self._preparation_cancel_requested:
            return
        if not isinstance(update, dict):
            self.preparation_progress_label.setText(str(update))
            return
        self.preparation_progress_label.setText(str(update.get("stage") or update.get("message") or "Calculating…"))
        self.preparation_progress_detail.setText(str(update.get("detail") or ""))
        try:
            completed = int(update.get("completed"))
            total = int(update.get("total"))
        except (TypeError, ValueError):
            completed = total = 0
        if total > 0:
            maximum = min(total, 1_000_000)
            self.preparation_progress_bar.setRange(0, maximum)
            self.preparation_progress_bar.setValue(max(0, min(maximum, int(completed / total * maximum))))
        else:
            self.preparation_progress_bar.setRange(0, 0)

    @Slot(object)
    def _preparation_completed(self, payload):
        if self._preparation_cancel_requested:
            self._preparation_cancelled()
            return
        result = dict(payload) if isinstance(payload, dict) else None
        if result is None:
            self._preparation_failed("Preparation did not produce an analysis payload.")
            return

        elapsed = max(0.0, time.monotonic() - self._preparation_started_at) if self._preparation_started_at else 0.0
        context = result.setdefault("exposure_analysis_context", {})
        if isinstance(context, dict):
            context["preparation_processing_time_s"] = float(elapsed)
        try:
            analysis_id = self.adapter.save_exposure_analysis_resource(result, analysis_cache={})
            self._prepared_record = self.adapter.exposure_analysis(analysis_id)
        except Exception as error:  # noqa: BLE001 - persistence boundary
            self._prepared_record = None
            self.preparation_progress_bar.setRange(0, 100)
            self.preparation_progress_bar.setValue(0)
            self.preparation_progress_label.setText("Preparation finished but could not be saved.")
            self.preparation_progress_detail.setText(str(error))
            self.cancel_preparation_button.setText("Back to objects")
            return

        self.preparation_progress_bar.setRange(0, 100)
        self.preparation_progress_bar.setValue(100)
        self.preparation_progress_label.setText("Exposure analysis is ready.")
        sample_count = int(context.get("sample_count", 0) or 0) if isinstance(context, dict) else 0
        self.preparation_progress_detail.setText(
            f"Prepared {sample_count:,} sample points and saved the analysis preparation."
            if sample_count else "Analysis preparation is ready and saved."
        )
        self.cancel_preparation_button.setEnabled(False)

    @Slot(str)
    def _preparation_failed(self, message):
        if self._preparation_cancel_requested:
            self._preparation_cancelled()
            return
        self._prepared_record = None
        self.preparation_progress_bar.setRange(0, 100)
        self.preparation_progress_bar.setValue(0)
        self.preparation_progress_label.setText("Exposure analysis preparation failed.")
        self.preparation_progress_detail.setText(str(message))
        self.cancel_preparation_button.setText("Back to objects")
        self.cancel_preparation_button.setEnabled(True)

    @Slot()
    def _preparation_cancelled(self):
        self._prepared_record = None
        self.preparation_progress_bar.setRange(0, 100)
        self.preparation_progress_bar.setValue(0)
        self.preparation_progress_label.setText("Exposure analysis preparation cancelled.")
        self.preparation_progress_detail.clear()
        self.cancel_preparation_button.setText("Back to objects")
        self.cancel_preparation_button.setEnabled(True)

    @Slot()
    def _preparation_thread_finished(self):
        self._preparation_thread = None
        self._preparation_worker = None

        record = self._prepared_record
        if isinstance(record, dict):
            target_id = self._preparation_target_id
            self._prepared_record = None
            view = self._attach_record(record)
            if callable(self.on_cache_changed):
                self.on_cache_changed(record.get("id"))
            self._refresh_target_info(target_id)
            self._show_target_view(view)
        else:
            self._sync_busy()

    def _cancel_preparation(self, *_args):
        if self._preparation_thread is None:
            self._show_objects()
            return
        if self._preparation_cancel_requested:
            return
        self._preparation_cancel_requested = True
        self.cancel_preparation_button.setEnabled(False)
        self.preparation_progress_label.setText("Cancelling exposure analysis safely…")
        self.preparation_progress_detail.setText("The current field-sampling batch will finish before the worker stops.")
        worker = self._preparation_worker
        if worker is not None:
            worker.request_cancellation()

    def _show_objects(self, *_args):
        if self._preparation_thread is not None or self._analysis_busy():
            return
        self._active_view = None
        self.stack.setCurrentWidget(self.chooser_page)
        self.target_tree.setEnabled(True)
        item = self.target_tree.currentItem()
        if item is not None:
            self._refresh_target_info(item.data(0, Qt.ItemDataRole.UserRole))

    def _cache_changed(self, view_id, analysis_id):
        view = self._target_views.get(str(view_id))
        if view is not None:
            record = next(
                (row for row in self.adapter.exposure_analyses if str(row.get("id") or "") == str(analysis_id)),
                {},
            )
            view._workspace_cache_sha = record.get("cache_sha256")
        item = self.target_tree.currentItem()
        if item is not None:
            self._refresh_target_info(item.data(0, Qt.ItemDataRole.UserRole))
        if callable(self.on_cache_changed):
            self.on_cache_changed(analysis_id)

    def _analysis_busy(self):
        return any(view.is_busy() for view in self._target_views.values())

    def _busy(self):
        return self._preparation_thread is not None or self._analysis_busy()

    def _sync_busy(self):
        busy = self._busy()
        self.target_tree.setEnabled(not busy)
        self.saved_analysis_tree.setEnabled(not busy)
        self.create_button.setEnabled(not busy and self.target_tree.currentItem() is not None)
        self.open_analysis_button.setEnabled(not busy and self._selected_saved_analysis_id() is not None)

    def _cleanup(self):
        if self._preparation_thread is not None:
            self._cancel_preparation()
        for view in tuple(self._target_views.values()):
            view.cleanup()
        self._target_views.clear()
        self._active_view = None

    def reject(self):
        self._cleanup()
        super().reject()

    def closeEvent(self, event):
        # Preparation/analysis workers are retained independently until they stop,
        # so closing this owned auxiliary window never vetoes owner/application shutdown.
        self._cleanup()
        super().closeEvent(event)

