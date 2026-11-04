"""Authenticated loopback transport and private per-user instance discovery.

The CLI/MCP side imports only this standard-library module and the command catalog.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

from . import PROTOCOL_VERSION
from .protocol import MAX_MESSAGE_BYTES, CommandError, dumps, loads

DEFAULT_TIMEOUT = 30.0


def runtime_directory(value=None, *, create=False):
    if value is None:
        base = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir())
        user = (
            str(os.getuid())
            if hasattr(os, "getuid")
            else hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
        )
        value = base / f"field-workbench-{user}"
    directory = Path(value).expanduser().absolute()
    if create:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise CommandError(
            "invalid_runtime_directory",
            "Runtime directory must not be a symbolic link.",
        )
    if directory.exists() and os.name != "nt":
        stat = directory.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise CommandError(
                "invalid_runtime_directory",
                "Runtime directory must be owned by you and private (mode 0700).",
            )
    return directory


def read_descriptor(path):
    if path.is_symlink():
        raise ValueError("Symbolic link descriptor")
    stat = path.stat()
    if stat.st_size > 16384:
        raise ValueError("Oversized descriptor")
    if os.name != "nt" and (stat.st_uid != os.getuid() or stat.st_mode & 0o077):
        raise ValueError("Instance descriptor is not private")
    descriptor = loads(path.read_bytes())
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("protocol_version") != PROTOCOL_VERSION
        or not re.fullmatch(r"[a-f0-9]{32}", descriptor.get("instance_id", ""))
        or path.stem != descriptor["instance_id"]
        or not isinstance(descriptor.get("port"), int)
        or not 1 <= descriptor["port"] <= 65535
        or not isinstance(descriptor.get("token"), str)
        or len(descriptor["token"]) < 32
    ):
        raise ValueError("Invalid instance descriptor")
    return descriptor


def request(
    descriptor, command, params=None, *, timeout=DEFAULT_TIMEOUT, request_id=None
):
    identifier = request_id or uuid4().hex
    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": identifier,
        "token": descriptor["token"],
        "command": command,
        "params": params if params is not None else {},
    }
    payload = (dumps(envelope) + "\n").encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise CommandError("request_too_large", "Request exceeds the message limit.")
    try:
        # The host is fixed, not trusted from a discovery file.
        with socket.create_connection(
            ("127.0.0.1", descriptor["port"]), timeout=timeout
        ) as connection:
            connection.settimeout(timeout)
            connection.sendall(payload)
            with connection.makefile("rb") as stream:
                line = stream.readline(MAX_MESSAGE_BYTES + 1)
        if not line or len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
            raise CommandError(
                "invalid_response",
                "FWB returned an incomplete or oversized response.",
                request_id=identifier,
            )
        response = loads(line)
        if (
            not isinstance(response, dict)
            or response.get("protocol_version") != PROTOCOL_VERSION
            or response.get("request_id") != identifier
            or not isinstance(response.get("ok"), bool)
        ):
            raise CommandError(
                "invalid_response",
                "FWB returned an invalid response envelope.",
                request_id=identifier,
            )
        return response
    except (OSError, ValueError) as error:
        raise CommandError(
            "connection_failed",
            str(error),
            request_id=identifier,
            outcome_unknown=True,
            recovery="Inspect live state. Reuse this request_id only with the identical command and parameters while this instance retains it; never blindly retry edits.",
        ) from error


def discover(directory=None):
    root = runtime_directory(directory)
    instances = []
    for path in sorted(root.glob("*.json")):
        try:
            descriptor = read_descriptor(path)
            response = request(descriptor, "system.info", timeout=1.0)
            if (
                response.get("ok")
                and response["result"]["instance_id"] == descriptor["instance_id"]
            ):
                instances.append(descriptor)
        except (OSError, ValueError, TypeError, KeyError, CommandError):
            continue  # Stale/crashed instances are ignored, never blindly deleted.
    return instances


def select_instance(instance_id=None, directory=None):
    if instance_id:
        if not re.fullmatch(r"[a-f0-9]{32}", instance_id):
            raise CommandError(
                "invalid_instance", "Use a complete instance ID from instances."
            )
        try:
            return read_descriptor(runtime_directory(directory) / f"{instance_id}.json")
        except (OSError, ValueError, TypeError):
            raise CommandError(
                "instance_not_found",
                "Instance is unavailable or its descriptor is invalid.",
            ) from None
    instances = discover(directory)
    if len(instances) != 1:
        raise CommandError(
            "instance_required",
            "Specify --instance when zero or multiple live FWB instances are found.",
            instances=[d["instance_id"] for d in instances],
        )
    return instances[0]


class CommandClient:
    def __init__(self, instance_id=None, directory=None, timeout=DEFAULT_TIMEOUT):
        self.instance_id, self.directory, self.timeout = instance_id, directory, timeout
        self._descriptor = None
        self._selection_lock = threading.Lock()

    def call(self, command, params=None, *, request_id=None):
        # Pin a connection once selected; switching the active GUI tab or starting
        # another FWB process must never redirect an established MCP session.
        with self._selection_lock:
            if self._descriptor is None:
                self._descriptor = select_instance(self.instance_id, self.directory)
        return request(
            self._descriptor,
            command,
            params,
            timeout=self.timeout,
            request_id=request_id,
        )
