# SPDX-License-Identifier: GPL-3.0-only

"""The benchmark scripts must keep running; their numbers are not checked."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(REPO / "benchmarks" / script), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_transport_benchmark_reports_both_settings(tmp_path):
    out = tmp_path / "t.csv"

    result = run("bench_transport.py", "--rounds", "5", "--csv", str(out))

    assert result.returncode == 0, result.stderr
    rows = out.read_text().splitlines()
    assert rows[0] == "tcp_nodelay,median_ms,p95_ms,max_ms"
    assert [r.split(",")[0] for r in rows[1:]] == ["False", "True"]


def test_decoupling_benchmark_reports_both_modes(tmp_path):
    out = tmp_path / "d.csv"

    result = run("bench_decoupling.py", "--steps", "2", "--csv", str(out))

    assert result.returncode == 0, result.stderr
    modes = {line.split(",")[0] for line in out.read_text().splitlines()[1:]}
    # sb3_subproc is added only when Stable-Baselines3 is installed.
    assert {"coupled", "decoupled"} <= modes <= {"coupled", "decoupled", "sb3_subproc"}
