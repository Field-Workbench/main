"""Lightweight process-launch and cancellation primitives for map renders.

This module intentionally imports no Qt or numerical packages.  Source launches
on POSIX can therefore start a clean fork server before QtWebEngine creates its
large thread/process tree, and render cancellation can use ordinary OS pipes
instead of resource-tracked named semaphores.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Any


_FORKSERVER_PRIMED = False
_FORKSERVER_UNAVAILABLE = False
_FORKSERVER_FAILURE: str | None = None


class PipeCancellationSignal:
    """Expose the ``is_set`` interface over a one-way multiprocessing pipe."""

    def __init__(self, receive_connection: Any):
        self._connection = receive_connection
        self._cancelled = False

    def is_set(self) -> bool:
        if self._cancelled:
            return True
        connection = self._connection
        if connection is None:
            return True
        try:
            if connection.poll():
                try:
                    connection.recv_bytes()
                except EOFError:
                    pass
                self._cancelled = True
        except (OSError, ValueError):
            # A vanished controller must not leave an orphan render running.
            self._cancelled = True
        return self._cancelled

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except (OSError, ValueError):
                pass


def _forkserver_probe() -> None:
    """Importable no-op target used to start the POSIX fork server safely."""


def field_render_process_context():
    """Return the safest supported multiprocessing context for render jobs."""

    if (
        os.name != "nt"
        and not getattr(sys, "frozen", False)
        and not _FORKSERVER_UNAVAILABLE
        and "forkserver" in multiprocessing.get_all_start_methods()
    ):
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context("spawn")


def prime_field_render_process_context() -> bool:
    """Start the POSIX fork server before Qt/WebEngine starts any threads."""

    global _FORKSERVER_PRIMED, _FORKSERVER_UNAVAILABLE, _FORKSERVER_FAILURE
    if _FORKSERVER_PRIMED:
        return True
    context = field_render_process_context()
    if context.get_start_method() != "forkserver":
        _FORKSERVER_PRIMED = True
        return True

    process = context.Process(
        target=_forkserver_probe,
        name="FieldRenderForkserverProbe",
    )
    started = False
    try:
        process.start()
        started = True
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)
        if process.exitcode != 0:
            _FORKSERVER_UNAVAILABLE = True
            _FORKSERVER_FAILURE = f"probe exit code {process.exitcode}"
            return False
        _FORKSERVER_PRIMED = True
        return True
    except (OSError, RuntimeError) as error:
        _FORKSERVER_UNAVAILABLE = True
        _FORKSERVER_FAILURE = f"{type(error).__name__}: {error}"
        return False
    finally:
        if not started or not process.is_alive():
            try:
                process.close()
            except (AttributeError, ValueError):
                pass
