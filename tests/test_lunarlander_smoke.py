# SPDX-License-Identifier: GPL-3.0-only

import subprocess
import sys
import time
from pathlib import Path

import pytest

from spindecoupler import BaseCommPoint


def _terminate_process(proc: subprocess.Popen) -> None:
    """Terminate a subprocess best-effort without hanging the test."""

    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _skip_if_lunarlander_not_available() -> None:
    pytest.importorskip("stable_baselines3")
    gym = pytest.importorskip("gymnasium")

    try:
        env = gym.make("LunarLander-v3")
        env.reset(seed=0)
        env.close()
    except Exception as exc:
        pytest.skip(f"LunarLander-v3 unavailable in this runner: {exc}")


def _run_agent_with_retries(
    agent_cmd: list[str], repo_root: Path, attempts: int = 60, delay_s: float = 0.5
) -> subprocess.CompletedProcess:
    """Retry agent launch to absorb RL-server startup jitter in CI."""

    last: subprocess.CompletedProcess | None = None
    for _ in range(attempts):
        result = subprocess.run(
            agent_cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            return result
        last = result
        if "connection refused" not in (result.stdout + result.stderr).lower():
            return result
        time.sleep(delay_s)

    assert last is not None
    return last


def test_lunarlander_example_smoke(free_tcp_port):
    """Run decoupled LunarLander example with very small timesteps."""

    _skip_if_lunarlander_not_available()

    repo_root = Path(__file__).resolve().parents[1]
    rl_script = repo_root / "examples" / "lunar_lander" / "rl_side_lunarlander.py"
    agent_script = repo_root / "examples" / "lunar_lander" / "agent_side_lunarlander.py"
    # Use the same IP resolution path as subprocesses (no localhost monkeypatch)
    # so RL-side bind address and agent-side connect target are consistent.
    host_ip = BaseCommPoint.get_ip()

    rl_cmd = [
        sys.executable,
        str(rl_script),
        "--port",
        str(free_tcp_port),
        "--timesteps",
        "256",
        "--timeout",
        "5.0",
        "--debug",
    ]
    agent_cmd = [
        sys.executable,
        str(agent_script),
        "--ip",
        host_ip,
        "--port",
        str(free_tcp_port),
        "--rl-step-period",
        "0.002",
        "--control-period",
        "0.001",
        "--timeout",
        "5.0",
        "--debug",
    ]

    rl_proc = subprocess.Popen(
        rl_cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        agent = _run_agent_with_retries(agent_cmd=agent_cmd, repo_root=repo_root)
        rl_out, _ = rl_proc.communicate(timeout=180)
    finally:
        _terminate_process(rl_proc)

    assert agent.returncode == 0, agent.stdout + "\n" + agent.stderr
    assert "finish command received" in agent.stdout.lower()
    assert "sent transition obs action=" in agent.stdout.lower()
    assert "reward=" in agent.stdout.lower()

    assert "training finished" in rl_out.lower()
    assert "model saved" in rl_out.lower()
    assert "rew_agent=" in rl_out.lower()
