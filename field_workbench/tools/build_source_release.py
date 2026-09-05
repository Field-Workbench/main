#!/usr/bin/env python3
"""Create a clean, matching Field Workbench source-release ZIP."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".spec"}
REQUIRED_FILES = {
    "LICENSE.txt", "PROJECT_NOTICE.txt", "THIRD_PARTY_NOTICES.txt",
    "requirements.txt", "requirements-build.txt", "build_windows.bat",
    "requirements-mesh-build.txt", "BUILDING_WINDOWS.md", "app.py",
}


def project_version() -> str:
    source = (ROOT / "fieldworkbench" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not determine Field Workbench version.")
    return match.group(1)


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix().casefold())


def main() -> int:
    missing = sorted(name for name in REQUIRED_FILES if not (ROOT / name).is_file())
    if missing:
        raise RuntimeError("Source release is missing: " + ", ".join(missing))

    version = project_version()
    output = ROOT.parent / f"FieldWorkbench_{version}_Source.zip"
    files = included_files()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT)
            archive.write(path, Path("field_workbench") / relative)

    print(f"Created: {output}")
    print(f"Files:   {len(files)}")
    print(f"Size:    {output.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
