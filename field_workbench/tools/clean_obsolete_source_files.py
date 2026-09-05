#!/usr/bin/env python3
"""Remove files deleted by newer Field Workbench source releases.

ZIP extraction replaces files but cannot delete paths that disappeared from a newer
release. Run this only from the Field Workbench source tree after overlaying a new
source ZIP onto an older checkout.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBSOLETE_PATHS = (
    Path("fieldworkbench/camera_timeline_dialog.py"),
)


def main() -> int:
    removed = 0
    for relative in OBSOLETE_PATHS:
        path = ROOT / relative
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Removed obsolete directory: {relative}")
            removed += 1
        elif path.exists():
            path.unlink()
            print(f"Removed obsolete file: {relative}")
            removed += 1
    for cache in ROOT.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    print(f"Obsolete source paths removed: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
