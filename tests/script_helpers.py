# SPDX-License-Identifier: GPL-3.0-only

"""Helpers to test the example shell scripts without Docker, SSH, ROS2 or a GPU.

The scripts are run with fake commands (``docker``, ``ssh``, ``python``, ``ros2``)
that only record their arguments, which is enough to check what each script would
launch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Skip a whole test module when the scripts cannot run (no bash, or Windows).
requires_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the deployment scripts need bash",
)


def fake_command(directory: Path, name: str, exit_code: int = 0) -> Path:
    """Create an executable ``name`` that writes its arguments, one per line, to
    ``<directory>/<name>.args`` (appending, so several calls can be told apart)."""

    script = directory / name
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" >> "{directory}/{name}.args"\n'
        f'echo "---" >> "{directory}/{name}.args"\n'
        f"exit {exit_code}\n"
    )
    script.chmod(0o755)
    return script


def calls(directory: Path, name: str) -> list[list[str]]:
    """Argument lists of every call recorded by the fake command ``name``."""

    recorded = directory / f"{name}.args"
    if not recorded.exists():
        return []
    calls, current = [], []
    for line in recorded.read_text().splitlines():
        if line == "---":
            calls.append(current)
            current = []
        else:
            current.append(line)
    return calls


def run_script(
    script: Path, args: list[str], tmp_path: Path, **env
) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **env}
    # Fake commands first on PATH; scripts must not rely on the caller's cwd.
    full_env["PATH"] = f"{tmp_path}{os.pathsep}{full_env['PATH']}"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=tmp_path,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def requirement_names(path: Path) -> set[str]:
    """Requirement names (with extras, without version) of a requirements file."""

    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[<>=!~ ]", line, maxsplit=1)[0].lower())
    return names


def has_pair(args: list[str], flag: str, value: str) -> bool:
    return any(a == flag and b == value for a, b in zip(args, args[1:]))
