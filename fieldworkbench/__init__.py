"""Field Workbench prototype package."""

from __future__ import annotations

from typing import Any

__all__ = ["StudioAdapter"]
__version__ = "0.13.0.355"


def __getattr__(name: str) -> Any:
    """Load the heavy numerical adapter only when callers actually request it."""

    if name == "StudioAdapter":
        from .studio_adapter import StudioAdapter

        return StudioAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
