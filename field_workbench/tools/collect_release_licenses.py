#!/usr/bin/env python3
"""Collect legal material for distributions actually present in a PyInstaller build.

The script runs inside the exact release venv after PyInstaller. It maps module
names in Analysis-00.toc back to installed distributions, copies their legal
metadata while preserving paths, inventories native Qt components, and writes
the binary's version-specific third-party notice. It performs no network access.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import shutil
import sys
from pathlib import Path


LEGAL_WORDS = (
    "license", "licence", "copying", "notice", "copyright", "authors",
    "contributors", "third-party", "third_party", "thirdparty", "sbom",
    "spdx", "credits",
)
REQUIRED_DISTS = {
    "magpylib", "magpylib-studio", "numpy", "plotly", "pyside6",
    "pyside6-addons", "pyside6-essentials", "pygeomag", "scipy", "shiboken6",
}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def safe_rel(path: str) -> Path:
    return Path(*(part for part in Path(path).parts if part not in (".", "..")))


def module_roots(analysis: Path) -> set[str]:
    text = analysis.read_text(encoding="utf-8", errors="replace")
    # Analysis TOCs contain tuples beginning with a module name. Restricting
    # matches to identifier-shaped dotted names avoids treating file paths as
    # importable modules.
    names = re.findall(r"\(['\"]([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)['\"]\s*,", text)
    return {name.split(".", 1)[0] for name in names}


def installed_distributions() -> dict[str, md.Distribution]:
    result: dict[str, md.Distribution] = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if name:
            result[norm(name)] = dist
    return result


def selected_distributions(roots: set[str]) -> list[md.Distribution]:
    package_map = md.packages_distributions()
    selected_names: set[str] = set()
    for root in roots:
        for dist_name in package_map.get(root, ()):
            selected_names.add(norm(dist_name))

    # These packages are unambiguously part of every Field Workbench release;
    # including them also protects against unusual TOC formatting or namespace
    # package metadata that does not map cleanly through packages_distributions.
    selected_names.update(REQUIRED_DISTS)

    installed = installed_distributions()
    missing = sorted(name for name in REQUIRED_DISTS if name not in installed)
    if missing:
        raise RuntimeError("Required release distributions are missing: " + ", ".join(missing))
    return [installed[name] for name in sorted(selected_names) if name in installed]


def looks_legal(entry: str) -> bool:
    low = entry.lower()
    return any(word in low for word in LEGAL_WORDS)


def copy_distribution_legal_files(dist: md.Distribution, destination: Path) -> int:
    name = dist.metadata.get("Name", "unknown")
    folder = destination / f"{name}-{dist.version}"
    copied = 0
    for entry in dist.files or ():
        entry_text = str(entry)
        if not looks_legal(entry_text):
            continue
        source = Path(dist.locate_file(entry))
        if not source.is_file():
            continue
        target = folder / safe_rel(entry_text)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1

    folder.mkdir(parents=True, exist_ok=True)
    fields = ("Name", "Version", "License", "License-Expression", "Home-page", "Project-URL")
    metadata_lines: list[str] = []
    for field in fields:
        for value in dist.metadata.get_all(field) or ():
            metadata_lines.append(f"{field}: {value}")
    (folder / "DISTRIBUTION_METADATA.txt").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )
    return copied


def copy_python_license(destination: Path) -> Path:
    candidates = [
        Path(sys.base_prefix) / "LICENSE.txt",
        Path(sys.base_prefix) / "LICENSE",
        Path(sys.prefix) / "LICENSE.txt",
        Path(sys.prefix) / "LICENSE",
    ]
    for candidate in candidates:
        if candidate.is_file():
            target = destination / f"Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-LICENSE.txt"
            shutil.copy2(candidate, target)
            return target
    raise RuntimeError("Could not find the Python license in the release installation.")


def qt_inventory(dist_dir: Path) -> tuple[list[str], bool]:
    qt_files = sorted(
        path.name for path in dist_dir.rglob("*")
        if path.is_file()
        and (path.name.startswith("Qt6") or path.name.startswith("qtwebengine"))
        and path.suffix.lower() in {".dll", ".exe", ".pak"}
    )
    webengine = any(name.lower() == "qt6webenginecore.dll" for name in qt_files)
    return sorted(set(qt_files), key=str.casefold), webengine


def license_label(dist: md.Distribution) -> str:
    value = (
        dist.metadata.get("License-Expression")
        or dist.metadata.get("License")
        or "see collected metadata/license files"
    ).replace("\n", " ").strip()
    if len(value) > 240:
        return "see collected metadata/license files"
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--notice-template", type=Path, required=True)
    args = parser.parse_args()

    analysis = args.analysis.resolve()
    dist_dir = args.dist.resolve()
    if not analysis.is_file() or not dist_dir.is_dir():
        print("ERROR: PyInstaller analysis or output directory is missing.")
        return 1

    license_dir = dist_dir / "licenses"
    if license_dir.exists():
        shutil.rmtree(license_dir)
    license_dir.mkdir(parents=True)

    try:
        distributions = selected_distributions(module_roots(analysis))
        python_license = copy_python_license(license_dir)
        copied_counts = {
            dist.metadata.get("Name", "unknown"): copy_distribution_legal_files(dist, license_dir)
            for dist in distributions
        }
    except (OSError, RuntimeError) as error:
        print(f"ERROR: Could not construct the release license bundle: {error}")
        return 1

    qt_files, has_webengine = qt_inventory(dist_dir)
    if not has_webengine:
        print("ERROR: Qt6WebEngineCore.dll was not found in the frozen release.")
        return 1

    manifest_lines = [
        "FIELD WORKBENCH RELEASE LICENSE MANIFEST",
        "Generated from PyInstaller Analysis-00.toc and the active build venv.",
        "",
        f"Python license: {python_license.name}",
        "",
        "Bundled Python distributions:",
    ]
    notice_lines = ["", "Bundled Python distributions", "----------------------------"]
    for dist in distributions:
        name = dist.metadata.get("Name", "unknown")
        manifest_lines.append(f"  {name}=={dist.version}  legal files: {copied_counts[name]}")
        notice_lines.append(f"  {name} {dist.version} — {license_label(dist)}")

    manifest_lines.extend(["", "Detected Qt/Qt WebEngine runtime files:"])
    manifest_lines.extend(f"  {name}" for name in qt_files)
    manifest_lines.extend([
        "",
        "Qt/Chromium note:",
        "  Qt WebEngine incorporates Chromium and other third-party code.",
        "  Preserve Qt's packaged resource files and all collected PySide6/Qt",
        "  legal, credits, NOTICE, SBOM, and SPDX material with the release.",
        "  Official versioned inventory: https://doc.qt.io/qt-6/qtwebengine-licensing.html",
        "",
    ])
    (license_dir / "MANIFEST.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    notice_lines.extend(["", "Detected Qt/Qt WebEngine runtime files", "--------------------------------------"])
    notice_lines.extend(f"  {name}" for name in qt_files)
    template = args.notice_template.read_text(encoding="utf-8").rstrip()
    (dist_dir / "THIRD_PARTY_NOTICES.txt").write_text(
        template + "\n" + "\n".join(notice_lines) + "\n", encoding="utf-8"
    )

    print(f"Collected licenses for {len(distributions)} bundled Python distributions.")
    print(f"Detected {len(qt_files)} Qt/Qt WebEngine runtime files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
