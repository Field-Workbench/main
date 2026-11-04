"""Loopback listener; authenticated requests are handed to the GUI dispatcher."""

from __future__ import annotations

import hmac
import os
import secrets
import socketserver
import threading
import time
from uuid import uuid4

from . import PROTOCOL_VERSION
from .connection import runtime_directory
from .protocol import MAX_MESSAGE_BYTES, CommandError, dumps, failure, loads


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 16

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(30)
        identifier = None
        try:
            line = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
            if not line or not line.endswith(b"\n") or len(line) > MAX_MESSAGE_BYTES:
                raise CommandError(
                    "invalid_request", "Incomplete or oversized command request."
                )
            payload = loads(line)
            if not isinstance(payload, dict):
                raise CommandError("invalid_request", "Request must be an object.")
            identifier = payload.get("request_id")
            if not isinstance(identifier, str) or not 1 <= len(identifier) <= 128:
                identifier = None
                raise CommandError(
                    "invalid_request", "A request_id of 1–128 characters is required."
                )
            token = payload.pop("token", None)
            if not isinstance(token, str) or not hmac.compare_digest(
                token, self.server.token
            ):
                raise CommandError("unauthorized", "Invalid instance token.")
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise CommandError(
                    "protocol_mismatch", "Unsupported command protocol version."
                )
            if (
                set(payload) != {"protocol_version", "request_id", "command", "params"}
                or not isinstance(payload["command"], str)
                or not isinstance(payload["params"], dict)
            ):
                raise CommandError(
                    "invalid_request", "Expected command and params fields."
                )
            response = self.server.dispatch(payload)
        except CommandError as error:
            response = failure(error, identifier)
        except (OSError, ValueError, TypeError, RecursionError) as error:
            response = failure(CommandError("invalid_request", str(error)), identifier)
        try:
            encoded = (dumps(response) + "\n").encode()
            if len(encoded) > MAX_MESSAGE_BYTES:
                encoded = (
                    dumps(
                        failure(
                            CommandError(
                                "response_too_large",
                                "Use project.export or job.export for this result.",
                            ),
                            identifier,
                        )
                    )
                    + "\n"
                ).encode()
            self.wfile.write(encoded)
        except (OSError, ValueError):
            pass  # Client disconnected; no automatic replay of a completed edit.


class LocalServer:
    def __init__(self, dispatch, *, instance_id=None, directory=None):
        self.instance_id = instance_id or uuid4().hex
        self.directory = runtime_directory(directory, create=True)
        self.path = self.directory / f"{self.instance_id}.json"
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.token = secrets.token_hex(32)
        self._server.dispatch = dispatch
        self._server.slots = threading.BoundedSemaphore(16)
        self._thread = None
        self._published = False

    def start(self):
        from .. import __version__

        descriptor = {
            "instance_id": self.instance_id,
            "protocol_version": PROTOCOL_VERSION,
            "port": self._server.server_address[1],
            "pid": os.getpid(),
            "version": __version__,
            "started_at": time.time(),
            "token": self._server.token,
        }
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._published = True
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(dumps(descriptor))
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="FWB-command-listener",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self._server.server_close()
            if self._published:
                self.path.unlink(missing_ok=True)
                self._published = False
            raise

    def close(self):
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=1)
            self._thread = None
        self._server.server_close()
        if self._published:
            self.path.unlink(missing_ok=True)
            self._published = False
