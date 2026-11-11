# SPDX-License-Identifier: GPL-3.0-only

"""Dependency-light smoke tests for the real Burger example."""

import os
import py_compile
import subprocess
from pathlib import Path

import pytest
from script_helpers import (
    calls,
    fake_command,
    has_pair,
    requirement_names,
    requires_bash,
    run_script,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "burger_real"
CONTRACT_DIR = EXAMPLE_DIR / "ros2" / "burger_real_agent"


def test_burger_real_python_entrypoints_compile():
    for path in (
        EXAMPLE_DIR / "rl_side_burger_real.py",
        CONTRACT_DIR / "agent_node.py",
    ):
        py_compile.compile(str(path), doraise=True)


def test_burger_real_contract_matches_coppelia_action_observation_shape():
    import sys

    sys.path.insert(0, str(CONTRACT_DIR))
    from contract import NUM_LIDAR_SECTORS, OBS_DIM  # noqa: PLC0415

    assert OBS_DIM == NUM_LIDAR_SECTORS + 2 == 10
    assert CONTRACT_DIR.joinpath("package.xml").is_file()
    assert CONTRACT_DIR.joinpath("setup.py").is_file()


def test_burger_real_exploitation_has_no_training_call():
    source = (EXAMPLE_DIR / "rl_side_burger_real.py").read_text()

    assert "SAC.load" in source
    assert "PPO.load" not in source
    assert "deterministic=True" in source
    assert "model.learn(" not in source


# Deployment scripts (run with fake commands: no ROS2, GPU or SSH needed).
@requires_bash
@pytest.mark.parametrize(
    "script", sorted(EXAMPLE_DIR.glob("*.sh")), ids=lambda p: p.name
)
def test_burger_real_scripts_are_valid_bash_and_executable(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert os.access(script, os.X_OK), f"{script} is not executable"


def test_burger_real_ships_the_trained_policy():
    assert (EXAMPLE_DIR / "training_8rays_fixedtime.zip").stat().st_size > 0


def test_burger_real_requirements_are_split_per_host():
    rl = requirement_names(EXAMPLE_DIR / "requirements-rl.txt")
    agent = requirement_names(EXAMPLE_DIR / "requirements-agent.txt")

    assert "stable-baselines3" in rl
    assert "stable-baselines3" not in agent and "torch" not in agent


def test_ros_packages_have_real_maintainers():
    for manifest in (
        CONTRACT_DIR / "package.xml",
        REPO_ROOT
        / "examples"
        / "burger_real_sim"
        / "ros2"
        / "burger_sim_bridge"
        / "package.xml",
    ):
        assert "example.com" not in manifest.read_text(), manifest


@requires_bash
def test_run_rl_remote_uses_the_included_model_by_default(tmp_path):
    fake_python = fake_command(tmp_path, "fakepython")

    result = run_script(
        EXAMPLE_DIR / "run_rl_remote.sh",
        ["--episodes", "5"],
        tmp_path,
        PYTHON=str(fake_python),
        DEVICE="cpu",
        PORT="50123",
    )

    assert result.returncode == 0, result.stderr
    (call,) = calls(tmp_path, "fakepython")
    assert call[0].endswith("examples/burger_real/rl_side_burger_real.py")
    assert has_pair(
        call, "--model-path", "examples/burger_real/training_8rays_fixedtime.zip"
    )
    assert has_pair(call, "--port", "50123")
    assert has_pair(call, "--device", "cpu")
    # A reset waits for the operator to publish the next goal: long by default.
    assert has_pair(call, "--timeout", "300")
    assert has_pair(call, "--episodes", "5")


@requires_bash
def test_run_rl_remote_accepts_a_model_path_and_timeout(tmp_path):
    fake_python = fake_command(tmp_path, "fakepython")

    run_script(
        EXAMPLE_DIR / "run_rl_remote.sh",
        ["my_policy.zip"],
        tmp_path,
        PYTHON=str(fake_python),
        TIMEOUT="45",
    )

    (call,) = calls(tmp_path, "fakepython")
    assert has_pair(call, "--model-path", "my_policy.zip")
    assert has_pair(call, "--timeout", "45")


@requires_bash
@pytest.mark.parametrize(
    ("args", "expected_ip", "forwarded"),
    [
        ([], "127.0.0.1", []),
        (["192.0.2.7"], "192.0.2.7", []),
        (["--debug"], "127.0.0.1", ["--debug"]),  # an option is not an IP
        (["192.0.2.7", "--sensor-timeout", "2.0"], "192.0.2.7", ["--sensor-timeout"]),
    ],
)
def test_run_agent_ros2_picks_the_host_and_forwards_options(
    tmp_path, args, expected_ip, forwarded
):
    fake_command(tmp_path, "ros2")

    result = run_script(EXAMPLE_DIR / "run_agent_ros2.sh", args, tmp_path, PORT="50123")

    assert result.returncode == 0, result.stderr
    (call,) = calls(tmp_path, "ros2")
    assert call[:3] == ["run", "burger_real_agent", "agent_node"]
    assert "use_sim_time:=false" in call  # the real robot uses wall-clock time
    assert has_pair(call, "--ip", expected_ip)
    assert has_pair(call, "--port", "50123")
    for option in forwarded:
        assert option in call


@requires_bash
def test_run_agent_ros2_can_follow_a_simulated_clock(tmp_path):
    fake_command(tmp_path, "ros2")

    run_script(EXAMPLE_DIR / "run_agent_ros2.sh", [], tmp_path, USE_SIM_TIME="true")

    (call,) = calls(tmp_path, "ros2")
    assert "use_sim_time:=true" in call


@requires_bash
def test_burger_real_tunnel_forwards_the_requested_ports(tmp_path):
    fake_command(tmp_path, "ssh")

    run_script(EXAMPLE_DIR / "tunnel.sh", ["user@host"], tmp_path)
    run_script(EXAMPLE_DIR / "tunnel.sh", ["user@host", "49054", "50000"], tmp_path)

    first, second = calls(tmp_path, "ssh")
    assert has_pair(first, "-L", "49054:127.0.0.1:49054")
    assert has_pair(second, "-L", "50000:127.0.0.1:49054")


def test_agent_node_reconnects_after_a_lost_connection():
    source = (CONTRACT_DIR / "agent_node.py").read_text()

    assert "except ConnectionLost" in source
    assert "reconnect(timeout=" in source
