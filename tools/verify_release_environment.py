#!/usr/bin/env python3
"""Refuse a Windows release build from the wrong or inconsistent environment."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = (ROOT / "requirements.txt", ROOT / "requirements-build.txt")


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_requirements() -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for path in REQUIREMENTS:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "-r ")):
                continue
            match = re.fullmatch(r"([A-Za-z0-9._-]+)==([^;\s]+)", line)
            if match is None:
                raise ValueError(f"Release requirement is not exactly pinned: {line}")
            result[normalized(match.group(1))] = (match.group(1), match.group(2))
    return result


def main() -> int:
    errors: list[str] = []
    if os.name != "nt":
        errors.append("The release batch must run on Windows.")
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 is required; found {platform.python_version()}.")
    if not os.environ.get("VIRTUAL_ENV"):
        errors.append("No active virtual environment was detected.")

    for key, (display_name, expected) in pinned_requirements().items():
        try:
            actual = md.version(display_name)
        except md.PackageNotFoundError:
            errors.append(f"Missing pinned distribution: {display_name}=={expected}")
            continue
        if actual != expected:
            errors.append(f"{display_name}: expected {expected}, found {actual}")

    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        errors.append("pip check failed:\n" + (check.stdout or check.stderr).strip())

    if errors:
        print("RELEASE ENVIRONMENT CHECK FAILED")
        for error in errors:
            print(f"  - {error}")
        print("\nCreate a clean venv and install requirements-build.txt exactly.")
        return 1

    print(f"Release environment OK: Python {platform.python_version()}")
    for _key, (name, version) in sorted(pinned_requirements().items()):
        print(f"  {name}=={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
