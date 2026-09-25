#!/usr/bin/env python3
"""Run pinned Magpylib and Magpylib Studio upstream suites in clean venvs.

This is intentionally separate from Field Workbench's normal tests. It clones
the exact release tags named by requirements.txt, installs each project's test
dependencies in a temporary virtual environment, and runs its own pytest suite.
Network access, Git, and enough time/disk space for optional visualization test
dependencies are required.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
from contextlib import nullcontext
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"\n>>> {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def pinned_version(package: str) -> str:
    pattern = re.compile(rf"^{re.escape(package)}==([^\s#]+)$", re.IGNORECASE)
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1)
    raise RuntimeError(f"No exact {package} pin found in {REQUIREMENTS}")


def environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def dependency_group(project: Path, group_name: str) -> list[str]:
    """Read a PEP 735 group, including nested groups, from pyproject.toml."""
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    groups = pyproject.get("dependency-groups", {})

    def expand(name: str, active: set[str]) -> list[str]:
        if name in active:
            raise RuntimeError(f"Recursive dependency group in {project}: {name}")
        result: list[str] = []
        for item in groups.get(name, []):
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and "include-group" in item:
                result.extend(expand(str(item["include-group"]), active | {name}))
        return result

    dependencies = expand(group_name, set())
    if dependencies:
        return dependencies
    optional = pyproject.get("project", {}).get("optional-dependencies", {})
    return [str(item) for item in optional.get(group_name, [])]


def clone_release(repository: str, tag: str, destination: Path) -> None:
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            repository,
            str(destination),
        ]
    )


def new_environment(path: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    python = environment_python(path)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    return python


def run_magpylib(workspace: Path, version_number: str) -> None:
    source = workspace / "magpylib-source"
    clone_release(
        "https://github.com/magpylib/magpylib.git", version_number, source
    )
    python = new_environment(workspace / "magpylib-venv")
    run([str(python), "-m", "pip", "install", "-e", str(source)])
    dependencies = dependency_group(source, "test")
    if not dependencies:
        raise RuntimeError("The Magpylib release does not declare a test dependency group.")
    run([str(python), "-m", "pip", "install", *dependencies])
    run([str(python), "-m", "pytest", "-q", str(source / "tests")], cwd=source)


def run_studio(workspace: Path, studio_version: str, magpylib_version: str) -> None:
    source = workspace / "magpylib-studio-source"
    clone_release(
        "https://github.com/magpylib/magpylib-studio.git",
        f"v{studio_version}",
        source,
    )
    python = new_environment(workspace / "magpylib-studio-venv")
    run([str(python), "-m", "pip", "install", "-e", f"{source}[dev]"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            f"magpylib=={magpylib_version}",
        ]
    )
    run([str(python), "-m", "pip", "check"])
    run([str(python), "-m", "pytest", "-q", str(source / "tests")], cwd=source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("all", "magpylib", "studio"),
        default="all",
        help="Dependency suite to run (default: all).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Preserve clones and environments in this directory instead of a temporary one.",
    )
    args = parser.parse_args()

    if shutil.which("git") is None:
        parser.error("Git is required to fetch the exact upstream release sources.")

    magpylib_version = pinned_version("magpylib")
    studio_version = pinned_version("magpylib-studio")
    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=True)
        context = nullcontext(args.workdir.resolve())
    else:
        context = tempfile.TemporaryDirectory(prefix="field-workbench-upstream-")

    with context as selected:
        workspace = Path(selected)
        print(
            "Validating upstream releases "
            f"magpylib {magpylib_version} and magpylib-studio {studio_version}\n"
            f"Workspace: {workspace}",
            flush=True,
        )
        if args.target in {"all", "magpylib"}:
            run_magpylib(workspace, magpylib_version)
        if args.target in {"all", "studio"}:
            run_studio(workspace, studio_version, magpylib_version)

    print("\nAll requested upstream dependency tests passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"\nUpstream validation failed with exit code {error.returncode}.")
        raise SystemExit(error.returncode) from error
