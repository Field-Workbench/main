"""Transport-independent JSON contract. No Qt, numerical, or MCP imports."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION

MAX_MESSAGE_BYTES = 8 * 1024 * 1024
RUNTIME_SETTINGS = (
    "coil_modeling_method",
    "solver_memory_budget_mb",
    "optimizer_worker_count",
    "solver_diagnostics_enabled",
    "fluxline_minimum_field_cutoff_enabled",
    "fluxline_minimum_field_cutoff_percent",
    "fluxline_conductor_cutoff_enabled",
    "fluxline_conductor_cutoff_mm",
)


class CommandError(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.details = code, details

    def payload(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


def json_value(value: Any) -> Any:
    """Normalize numerical results without silently stringifying unknown objects.

    Undefined/nonfinite numerical outputs become JSON null, including in exports.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    raise TypeError(f"Unsupported result type: {type(value).__name__}")


def dumps(value: Any) -> str:
    return json.dumps(
        json_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def loads(value: str | bytes) -> Any:
    def reject(constant: str):
        raise ValueError(f"Nonfinite JSON constant: {constant}")

    return json.loads(value, parse_constant=reject)


def revision(adapter: Any) -> str:
    payload = {
        "document_signature": adapter.signature(),
        "runtime": {key: getattr(adapter, key, None) for key in RUNTIME_SETTINGS},
    }
    return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()


def failure(error: CommandError, request_id: str | None = None) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": error.payload(),
    }


def validate(value: Any, schema: dict, path: str = "params") -> None:
    """Validate the deliberately small JSON Schema vocabulary used by the registry.

    Schema definitions below use only types, properties/required/additionalProperties,
    enum, items, min/maxItems, minLength, minimum/maximum. MCP receives the same schema.
    """
    kind = schema.get("type")
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ),
        "null": lambda: value is None,
    }
    if kind and not checks[kind]():
        raise CommandError("invalid_params", f"{path} must be {kind}.")
    if "enum" in schema and value not in schema["enum"]:
        raise CommandError("invalid_params", f"{path} must be one of {schema['enum']}.")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise CommandError("invalid_params", f"{path}.{key} is required.")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise CommandError(
                    "invalid_params", f"Unknown parameter: {path}.{key}."
                )
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", math.inf
        ):
            raise CommandError(
                "invalid_params", f"{path} has an invalid number of items."
            )
        if "items" in schema:
            for index, item in enumerate(value):
                validate(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise CommandError("invalid_params", f"{path} must not be empty.")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (
            not math.isfinite(value)
            or value < schema.get("minimum", -math.inf)
            or value > schema.get("maximum", math.inf)
        )
    ):
        raise CommandError("invalid_params", f"{path} is out of range.")
    # Check arbitrary JSON values too, notably inside model-specific settings.
    if isinstance(value, (dict, list)):

        def finite(item):
            if isinstance(item, float) and not math.isfinite(item):
                raise CommandError(
                    "invalid_params", f"{path} contains a nonfinite number."
                )
            if isinstance(item, dict):
                for child in item.values():
                    finite(child)
            if isinstance(item, list):
                for child in item:
                    finite(child)

        finite(value)
