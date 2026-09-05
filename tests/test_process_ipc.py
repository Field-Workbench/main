"""Tests for semaphore-free map-render process control."""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
import unittest

from fieldworkbench.process_ipc import (
    PipeCancellationSignal,
    field_render_process_context,
    prime_field_render_process_context,
)
import fieldworkbench.process_ipc as process_ipc_module


def _wait_for_pipe_cancellation(cancel_receive, status_send) -> None:
    signal = PipeCancellationSignal(cancel_receive)
    try:
        status_send.send_bytes(b"ready")
        deadline = time.monotonic() + 5.0
        while not signal.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        status_send.send_bytes(b"cancelled" if signal.is_set() else b"timeout")
    finally:
        signal.close()
        status_send.close()


class ProcessIpcTests(unittest.TestCase):
    def test_pipe_cancellation_signal_has_no_semaphore_dependency(self) -> None:
        context = field_render_process_context()
        receive_connection, send_connection = context.Pipe(duplex=False)
        signal = PipeCancellationSignal(receive_connection)
        try:
            self.assertFalse(signal.is_set())
            send_connection.send_bytes(b"cancel")
            self.assertTrue(signal.is_set())
            self.assertTrue(signal.is_set())
        finally:
            signal.close()
            send_connection.close()

    def test_closed_controller_pipe_cancels_orphan(self) -> None:
        context = field_render_process_context()
        receive_connection, send_connection = context.Pipe(duplex=False)
        signal = PipeCancellationSignal(receive_connection)
        send_connection.close()
        try:
            deadline = time.monotonic() + 1.0
            while not signal.is_set() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(signal.is_set())
        finally:
            signal.close()

    def test_render_context_primes_and_cancels_concurrent_children(self) -> None:
        primed = prime_field_render_process_context()
        context = field_render_process_context()
        if (
            primed
            and os.name != "nt"
            and not getattr(sys, "frozen", False)
            and "forkserver" in multiprocessing.get_all_start_methods()
        ):
            self.assertEqual(context.get_start_method(), "forkserver")
        else:
            self.assertEqual(context.get_start_method(), "spawn")
            if not primed:
                self.assertIsNotNone(process_ipc_module._FORKSERVER_FAILURE)

        jobs = []
        try:
            for index in range(8):
                cancel_receive, cancel_send = context.Pipe(duplex=False)
                status_receive, status_send = context.Pipe(duplex=False)
                process = context.Process(
                    target=_wait_for_pipe_cancellation,
                    args=(cancel_receive, status_send),
                    name=f"PipeCancellationStress-{index}",
                )
                process.start()
                cancel_receive.close()
                status_send.close()
                jobs.append((process, cancel_send, status_receive))

            for _process, _cancel_send, status_receive in jobs:
                self.assertTrue(status_receive.poll(10.0))
                self.assertEqual(status_receive.recv_bytes(), b"ready")
            for _process, cancel_send, _status_receive in jobs:
                cancel_send.send_bytes(b"cancel")
            for process, _cancel_send, status_receive in jobs:
                self.assertTrue(status_receive.poll(10.0))
                self.assertEqual(status_receive.recv_bytes(), b"cancelled")
                process.join(timeout=10.0)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
        finally:
            for process, cancel_send, status_receive in jobs:
                cancel_send.close()
                status_receive.close()
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                if not process.is_alive():
                    try:
                        process.close()
                    except ValueError:
                        pass


if __name__ == "__main__":
    unittest.main()
