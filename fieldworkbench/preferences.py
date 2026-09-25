"""Application-wide Field Workbench preferences.

The Project Manager owns these settings in suite mode.  They intentionally
exclude viewer-specific presentation controls such as fluxline cleanup, which
belong to the Maps workspace instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .dialogs import OptimizerWorkerBenchmarkDialog, configure_field_render_worker_limit
from .optimizer import DEFAULT_OPTIMIZER_WORKER_COUNT, available_optimizer_workers
from .studio_adapter import (
    COIL_MODELING_LABELS,
    DEFAULT_SOLVER_MEMORY_BUDGET_MB,
    MAX_SOLVER_MEMORY_BUDGET_MB,
    MIN_SOLVER_MEMORY_BUDGET_MB,
    StudioAdapter,
    StudioOperationError,
)
from .theme import (
    THEME_CHOICES,
    apply_workbench_theme,
    current_theme,
    save_theme_preference,
)
from .window_utils import enable_standard_window_controls


def _setting_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class GlobalPreferences:
    """Resolved application-wide numerical/resource preferences."""

    coil_modeling_method: str
    solver_memory_budget_mb: int
    worker_count: int
    solver_diagnostics_enabled: bool


def read_global_preferences(settings: QSettings | None = None) -> GlobalPreferences:
    source = settings if settings is not None else QSettings()
    method = str(source.value("coil_modeling/method", "auto"))
    if method not in COIL_MODELING_LABELS:
        method = "auto"
    try:
        memory = int(source.value("solver/memoryBudgetMb", DEFAULT_SOLVER_MEMORY_BUDGET_MB))
    except (TypeError, ValueError):
        memory = DEFAULT_SOLVER_MEMORY_BUDGET_MB
    memory = max(MIN_SOLVER_MEMORY_BUDGET_MB, min(MAX_SOLVER_MEMORY_BUDGET_MB, memory))
    try:
        workers = int(source.value("optimizer/workerCount", DEFAULT_OPTIMIZER_WORKER_COUNT))
    except (TypeError, ValueError):
        workers = DEFAULT_OPTIMIZER_WORKER_COUNT
    workers = max(1, min(available_optimizer_workers(), workers))
    diagnostics = _setting_bool(source.value("solver/diagnosticsEnabled", False), False)
    return GlobalPreferences(method, memory, workers, diagnostics)


def apply_global_preferences_to_adapter(
    adapter: StudioAdapter,
    preferences: GlobalPreferences | None = None,
    *,
    settings: QSettings | None = None,
) -> None:
    """Apply only application-global settings to one live adapter."""

    resolved = preferences or read_global_preferences(settings)
    try:
        adapter.set_coil_modeling_method(resolved.coil_modeling_method)
    except StudioOperationError:
        adapter.set_coil_modeling_method("auto")
    try:
        adapter.set_solver_memory_budget_mb(resolved.solver_memory_budget_mb)
    except StudioOperationError:
        adapter.set_solver_memory_budget_mb(DEFAULT_SOLVER_MEMORY_BUDGET_MB)
    try:
        adapter.set_optimizer_worker_count(resolved.worker_count)
    except StudioOperationError:
        adapter.set_optimizer_worker_count(DEFAULT_OPTIMIZER_WORKER_COUNT)
    adapter.set_solver_diagnostics_enabled(resolved.solver_diagnostics_enabled)


def _unique_adapters(adapters: Iterable[StudioAdapter]) -> list[StudioAdapter]:
    result: list[StudioAdapter] = []
    seen: set[int] = set()
    for adapter in adapters:
        if not isinstance(adapter, StudioAdapter) or id(adapter) in seen:
            continue
        seen.add(id(adapter))
        result.append(adapter)
    return result


def open_global_preferences(
    parent: QWidget,
    settings: QSettings,
    adapters: Iterable[StudioAdapter] = (),
    *,
    dialog_factory: Callable[[QWidget], QDialog] | None = None,
) -> GlobalPreferences | None:
    """Show the suite-wide Preferences dialog and apply accepted settings."""

    live_adapters = _unique_adapters(adapters)

    # Keep dialog construction injectable.  A number of UI smoke tests replace
    # the caller's QDialog with a non-blocking test double; without carrying that
    # factory across this shared helper, the real modal dialog enters exec() in
    # the offscreen Qt backend and the test suite waits forever.
    make_dialog = dialog_factory or QDialog
    dialog = make_dialog(parent)
    enable_standard_window_controls(dialog)
    dialog.setWindowTitle("Preferences")
    layout = QVBoxLayout(dialog)
    form = QFormLayout()

    theme_selector = QComboBox()
    theme_selector.setObjectName("themeSelector")
    for label, theme_id in THEME_CHOICES:
        theme_selector.addItem(label, theme_id)
    theme_selector.setCurrentIndex(max(0, theme_selector.findData(current_theme())))
    theme_selector.setToolTip(
        "Applies immediately to Project Manager and every open Field Workbench window."
    )

    def apply_selected_theme(_index: int) -> None:
        selected_theme = theme_selector.currentData()
        if selected_theme is None:
            return
        selected = str(selected_theme)
        apply_workbench_theme(parent, selected)
        save_theme_preference(selected, settings)

    theme_selector.currentIndexChanged.connect(apply_selected_theme)
    form.addRow("Theme", theme_selector)

    current_preferences = read_global_preferences(settings)

    model = QComboBox()
    for value in ("auto", "exact", "bundled", "current_sheet", "centreline"):
        model.addItem(COIL_MODELING_LABELS[value], value)
    model.setCurrentIndex(max(0, model.findData(current_preferences.coil_modeling_method)))
    form.addRow("Coil field model", model)

    memory_budget = QSpinBox()
    memory_budget.setRange(MIN_SOLVER_MEMORY_BUDGET_MB, MAX_SOLVER_MEMORY_BUDGET_MB)
    memory_budget.setSingleStep(64)
    memory_budget.setSuffix(" MB")
    memory_budget.setValue(current_preferences.solver_memory_budget_mb)
    memory_budget.setToolTip(
        "Target maximum temporary memory for one detailed Magpylib kernel call. "
        "Lower values use smaller point/source batches and are slower but safer."
    )
    form.addRow("Solver temporary-memory budget", memory_budget)

    worker_row = QWidget()
    worker_layout = QHBoxLayout(worker_row)
    worker_layout.setContentsMargins(0, 0, 0, 0)
    workers = QSpinBox()
    workers.setRange(1, available_optimizer_workers())
    workers.setValue(current_preferences.worker_count)
    workers.setSuffix(" workers")
    workers.setToolTip(
        "Maximum concurrent workers used by optimizer candidate batches and queued "
        "2D/3D render jobs. A value of 1 keeps both workloads serial. Counts may exceed "
        "the machine's reported logical CPU count; use Benchmark to find the fastest "
        "practical value. Each worker owns private solver state, so memory use rises with it."
    )
    worker_layout.addWidget(workers)
    benchmark = QPushButton("Benchmark…")
    benchmark.setToolTip(
        "Measure packaged-scene optimizer throughput using the field model and memory "
        "budget currently selected in this Preferences window."
    )
    worker_layout.addWidget(benchmark)
    worker_layout.addStretch(1)
    form.addRow("Worker concurrency", worker_row)

    def run_benchmark() -> None:
        benchmark_dialog = OptimizerWorkerBenchmarkDialog(
            coil_modeling_method=str(model.currentData()),
            solver_memory_budget_mb=memory_budget.value(),
            current_workers=workers.value(),
            parent=dialog,
        )
        if (
            benchmark_dialog.exec() == QDialog.DialogCode.Accepted
            and benchmark_dialog.recommended_workers is not None
        ):
            workers.setValue(benchmark_dialog.recommended_workers)

    benchmark.clicked.connect(run_benchmark)

    diagnostics = QCheckBox("Record solver diagnostics to a temporary JSONL log")
    diagnostics.setChecked(current_preferences.solver_diagnostics_enabled)
    form.addRow("Diagnostics", diagnostics)

    first_adapter = live_adapters[0] if live_adapters else None
    diagnostics_path = QLabel(
        first_adapter.solver_diagnostics_path()
        if first_adapter is not None and first_adapter.solver_diagnostics_path()
        else "A log path will be created when diagnostics are enabled."
    )
    diagnostics_path.setWordWrap(True)
    diagnostics_path.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse
        | Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    diagnostics_path.setObjectName("hint")
    form.addRow("Current log", diagnostics_path)

    note = QLabel(
        "These settings are application-wide. Fluxline cleanup is configured in Maps "
        "when a fluxline display is selected."
    )
    note.setWordWrap(True)
    note.setObjectName("hint")
    form.addRow(note)

    layout.addLayout(form)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    resolved = GlobalPreferences(
        coil_modeling_method=str(model.currentData()),
        solver_memory_budget_mb=memory_budget.value(),
        worker_count=workers.value(),
        solver_diagnostics_enabled=diagnostics.isChecked(),
    )
    settings.setValue("coil_modeling/method", resolved.coil_modeling_method)
    settings.setValue("solver/memoryBudgetMb", resolved.solver_memory_budget_mb)
    settings.setValue("optimizer/workerCount", resolved.worker_count)
    settings.setValue("solver/diagnosticsEnabled", resolved.solver_diagnostics_enabled)
    settings.sync()
    configure_field_render_worker_limit(resolved.worker_count)
    for adapter in live_adapters:
        apply_global_preferences_to_adapter(adapter, resolved)
    return resolved
