"""Crash-isolated process entry point for detailed 2D/3D field renders.

This module deliberately has no Qt imports. A native fault in Magpylib, NumPy,
SciPy, or a platform math library therefore terminates only the spawned render
process rather than the Field Workbench GUI process.
"""

from __future__ import annotations

import copy
import faulthandler
import os
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

from .studio_adapter import (
    FieldCalculationCancelled,
    StudioAdapter,
    StudioOperationError,
)
from .process_ipc import PipeCancellationSignal
from .waveform import build_3d_gpu_static, build_prepared_gpu_payload


def _send_message(connection: Any, message: dict[str, Any]) -> bool:
    """Best-effort IPC that never replaces the underlying solver outcome."""

    try:
        connection.send(message)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def run_field_render_process(
    connection: Any,
    result_path: str,
    crash_log_path: str,
    document: dict[str, Any],
    preferences: dict[str, Any],
    render_kind: str,
    request: dict[str, Any],
    cancel_connection: Any,
) -> None:
    """Calculate one render and atomically stage its result for the GUI thread."""

    cancel_event = PipeCancellationSignal(cancel_connection)
    output = Path(result_path)
    temporary_output = output.with_name(output.name + ".tmp")
    crash_log = Path(crash_log_path)
    crash_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = False

    with crash_log.open("w", encoding="utf-8") as crash_stream:
        crash_stream.write(
            "Field Workbench isolated map render process\n"
            f"pid={os.getpid()} kind={render_kind}\n"
        )
        crash_stream.flush()
        faulthandler.enable(file=crash_stream, all_threads=True)

        def progress(update: dict[str, Any]) -> None:
            if cancel_event.is_set():
                raise FieldCalculationCancelled(
                    "The field-map calculation was cancelled by the user."
                )
            if not _send_message(
                connection,
                {"kind": "progress", "update": dict(update)},
            ):
                raise FieldCalculationCancelled(
                    "The render controller closed while the calculation was running."
                )
            if cancel_event.is_set():
                raise FieldCalculationCancelled(
                    "The field-map calculation was cancelled by the user."
                )

        try:
            if cancel_event.is_set():
                raise FieldCalculationCancelled("Calculation cancelled.")
            adapter = StudioAdapter(document)
            # StudioAdapter owns a deep scene copy. Release the deserialized IPC
            # document before the large field arrays are allocated.
            document = {}
            adapter.set_coil_modeling_method(
                str(preferences.get("coil_modeling_method", "auto"))
            )
            adapter.set_solver_memory_budget_mb(
                int(preferences.get("solver_memory_budget_mb", 1024))
            )
            adapter.set_fluxline_minimum_field_cutoff(
                bool(
                    preferences.get(
                        "fluxline_minimum_field_cutoff_enabled", True
                    )
                ),
                float(
                    preferences.get(
                        "fluxline_minimum_field_cutoff_percent", 0.05
                    )
                ),
            )
            adapter.set_fluxline_conductor_cutoff(
                bool(
                    preferences.get("fluxline_conductor_cutoff_enabled", True)
                ),
                float(preferences.get("fluxline_conductor_cutoff_mm", 0.5)),
            )
            if bool(preferences.get("solver_diagnostics_enabled", False)):
                adapter.set_solver_diagnostics_enabled(True)
                diagnostic_path = adapter.solver_diagnostics_path()
                if diagnostic_path:
                    _send_message(
                        connection,
                        {"kind": "diagnostics", "path": diagnostic_path},
                    )

            result: dict[str, Any]
            if str(render_kind) in {"static", "brain_static"}:
                payload = build_3d_gpu_static(
                    adapter,
                    map_settings=copy.deepcopy(request["map_settings"]),
                    active_coil_ids=list(request.get("active_coil_ids", [])),
                    initial_camera=copy.deepcopy(request.get("initial_camera")),
                    progress_callback=progress,
                )
                result = {"payload": payload}
                field_vectors = payload.pop("_field_vectors_t", None)
                if field_vectors is not None:
                    result["field_vectors_t"] = field_vectors
            elif str(render_kind) == "2d_static":
                map_settings = copy.deepcopy(request.get("map_settings", {}))
                progress(
                    {
                        "stage": "Preparing 2D sample grid…",
                        "render_task": "grid",
                        "completed": 0,
                        "total": 1,
                        "unit": "render step",
                    }
                )
                progress(
                    {
                        "stage": "2D sample grid ready",
                        "render_task": "grid",
                        "render_task_complete": True,
                        "completed": 1,
                        "total": 1,
                        "unit": "render step",
                    }
                )
                figure = adapter.field_map(
                    **map_settings,
                    active_coil_ids=list(request.get("active_coil_ids", [])),
                    progress_callback=progress,
                )
                progress(
                    {
                        "stage": "Plotly map package ready",
                        "render_task": "assemble",
                        "render_task_complete": True,
                        "completed": 1,
                        "total": 1,
                        "unit": "render step",
                    }
                )
                result = {"figure": figure}
            elif str(render_kind) == "waveform":
                payload = build_prepared_gpu_payload(
                    adapter,
                    request,
                    progress_callback=progress,
                )
                result = {"payload": payload}
            else:
                raise StudioOperationError(
                    f"Unknown background render kind: {render_kind}"
                )
            if cancel_event.is_set():
                raise FieldCalculationCancelled("Calculation cancelled.")

            result.update(
                {
                    "solver_report": adapter.coil_modeling_report(),
                    "solver_elapsed_seconds": time.perf_counter() - started,
                }
            )
            with temporary_output.open("wb") as stream:
                pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_output, output)
            completed = True
            _send_message(connection, {"kind": "completed"})
        except FieldCalculationCancelled:
            _send_message(connection, {"kind": "cancelled"})
        except StudioOperationError as error:
            if cancel_event.is_set() or "cancelled" in str(error).lower():
                _send_message(connection, {"kind": "cancelled"})
            else:
                _send_message(
                    connection,
                    {
                        "kind": "failed",
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
        except BaseException as error:  # noqa: BLE001 - spawned process boundary
            _send_message(
                connection,
                {
                    "kind": "failed",
                    "message": (
                        f"Unexpected isolated render error "
                        f"({type(error).__name__}): {error}"
                    ),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            if not completed:
                for path in (temporary_output, output):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
            try:
                faulthandler.disable()
            except RuntimeError:
                pass
            try:
                connection.close()
            except OSError:
                pass
            cancel_event.close()
