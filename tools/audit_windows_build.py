#!/usr/bin/env python3
"""Size and content guard for the Windows Field Workbench onedir release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


UNEXPECTED_MODULES = (
    "pandas", "IPython", "jupyter", "notebook",
    "nbformat", "dash", "kaleido", "pyvista", "trimesh", "xarray",
    "polars", "pyarrow", "sympy", "numba", "skimage", "pytest", "PyQt5",
    "PyQt6", "PySide2",
)
REQUIRED_GLOBS = (
    "FieldWorkbench.exe",
    "**/default_scene.magpy.json",
    "**/plotly.min.js",
    "**/Qt6WebEngineCore.dll",
    "**/QtWebEngineProcess.exe",
    "**/qtwebengine_resources*.pak",
    "LICENSE.txt",
    "PROJECT_NOTICE.txt",
    "THIRD_PARTY_NOTICES.txt",
    "licenses/MANIFEST.txt",
    "**/fieldworkbench/assets/ffmpeg/windows-x86_64/ffmpeg.exe",
    "licenses/FFmpeg-n8.1.2-x264/BUILD_INFO.txt",
    "licenses/FFmpeg-n8.1.2-x264/GPL-2.0.txt",
    "licenses/FFmpeg-n8.1.2-x264/GPL-3.0.txt",
    "licenses/FFmpeg-n8.1.2-x264/ZLIB_LICENSE.txt",
)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def module_present(analysis_text: str, module: str) -> bool:
    return bool(re.search(rf"\(['\"]{re.escape(module)}(?:\.|['\"])", analysis_text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--max-mb", type=float, default=800.0)
    args = parser.parse_args()

    analysis_text = args.analysis.read_text(encoding="utf-8", errors="replace")
    dist_dir = args.dist.resolve()
    total_bytes = directory_size(dist_dir)
    total_mb = total_bytes / (1024 * 1024)
    unexpected = [name for name in UNEXPECTED_MODULES if module_present(analysis_text, name)]
    missing = [pattern for pattern in REQUIRED_GLOBS if not any(dist_dir.glob(pattern))]

    top_items: list[tuple[int, str]] = []
    for item in dist_dir.iterdir():
        size = directory_size(item) if item.is_dir() else item.stat().st_size
        top_items.append((size, item.name))
    top_items.sort(reverse=True)

    lines = [
        "FIELD WORKBENCH WINDOWS BUILD REPORT",
        "====================================",
        f"Total onedir size: {total_mb:.1f} MiB ({total_bytes:,} bytes)",
        f"Configured maximum: {args.max_mb:.1f} MiB",
        "",
        "Largest top-level entries:",
    ]
    lines.extend(f"  {size / (1024 * 1024):8.1f} MiB  {name}" for size, name in top_items[:12])
    lines.extend(["", "Unexpected optional Python stacks:"])
    lines.append("  " + (", ".join(unexpected) if unexpected else "none detected"))
    lines.extend(["", "Missing required release files:"])
    lines.append("  " + (", ".join(missing) if missing else "none"))
    report = "\n".join(lines) + "\n"
    (dist_dir / "BUILD_REPORT.txt").write_text(report, encoding="utf-8")
    print(report)

    failed = False
    if total_mb > args.max_mb:
        print("ERROR: Release exceeds the configured size ceiling.")
        failed = True
    if unexpected:
        print("ERROR: Optional heavyweight packages leaked into the frozen analysis.")
        failed = True
    if missing:
        print("ERROR: Required runtime or notice files are absent.")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
