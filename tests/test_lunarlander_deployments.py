# SPDX-License-Identifier: GPL-3.0-only

"""Tests of the LunarLander deployment files (remote host and container).

Neither Docker, a GPU nor an SSH server is needed: the scripts are run with fake
``docker``, ``ssh`` and ``python`` commands that only record their arguments,
which is enough to check what each script would launch. The files the Docker image
is built from are checked statically.
"""

from __future__ import annotations

import os
import re
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
EXAMPLES = REPO_ROOT / "examples"
BASE = EXAMPLES / "lunar_lander"
REMOTE = EXAMPLES / "lunar_lander_remote"
CONTAINER = EXAMPLES / "lunar_lander_container"

pytestmark = requires_bash

ALL_SCRIPTS = sorted(
    p for directory in (BASE, REMOTE, CONTAINER) for p in directory.glob("*.sh")
)


# Helpers: run a script with a fake command that records its arguments.
# Static checks.
@pytest.mark.parametrize(
    "script", ALL_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_scripts_are_valid_bash_and_executable(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert os.access(script, os.X_OK), f"{script} is not executable"


def test_split_requirements_install_only_their_half():
    """The remote/container RL host needs no Box2D; the agent host needs no SB3."""

    for directory in (REMOTE, CONTAINER):
        rl = requirement_names(directory / "requirements-rl.txt")
        agent = requirement_names(directory / "requirements-agent.txt")
        assert "stable-baselines3" in rl and "gymnasium[box2d]" not in rl
        assert "gymnasium[box2d]" in agent and "stable-baselines3" not in agent


def test_base_requirements_have_what_the_example_imports():
    names = requirement_names(BASE / "requirements.txt")

    assert "stable-baselines3" in names and "gymnasium[box2d]" in names
    assert not any("coppeliasim" in n for n in names)  # a Burger leftover


def test_dockerfile_copies_only_existing_files_and_has_a_dockerignore():
    dockerfile = (CONTAINER / "Dockerfile").read_text()

    copies = re.findall(r"^COPY\s+(.+)$", dockerfile, flags=re.MULTILINE)
    assert copies, "the Dockerfile copies nothing"
    for line in copies:
        *sources, _dest = line.split()
        for source in sources:
            assert (REPO_ROOT / source).exists(), f"COPY source missing: {source}"

    entrypoint = re.search(r'^ENTRYPOINT \["(.+)"\]', dockerfile, flags=re.MULTILINE)
    assert entrypoint is not None
    assert (CONTAINER / Path(entrypoint.group(1)).name).is_file()

    ignored = (REPO_ROOT / ".dockerignore").read_text().split()
    assert ".venv" in ignored and ".git" in ignored


# Bare-host deployment.
def test_run_rl_remote_launches_the_rl_side(tmp_path):
    fake_python = fake_command(tmp_path, "fakepython")

    result = run_script(
        REMOTE / "run_rl_remote.sh",
        ["B", "--timesteps", "20000"],
        tmp_path,
        PYTHON=str(fake_python),
        DEVICE="cpu",
        PORT="50123",
    )

    assert result.returncode == 0, result.stderr
    assert "profile B" in result.stdout
    (call,) = calls(tmp_path, "fakepython")
    assert call[0].endswith("examples/lunar_lander/rl_side_lunarlander.py")
    assert has_pair(call, "--port", "50123")
    assert has_pair(call, "--device", "cpu")
    assert has_pair(call, "--timeout", "10")
    assert has_pair(call, "--timesteps", "20000")


def test_run_rl_remote_profile_is_optional(tmp_path):
    fake_python = fake_command(tmp_path, "fakepython")

    result = run_script(
        REMOTE / "run_rl_remote.sh",
        ["--timesteps", "5"],
        tmp_path,
        PYTHON=str(fake_python),
    )

    assert result.returncode == 0, result.stderr
    assert "profile A" in result.stdout
    (call,) = calls(tmp_path, "fakepython")
    assert has_pair(call, "--timesteps", "5")
    assert "A" not in call  # the profile letter is not forwarded to the RL side


@pytest.mark.parametrize(
    ("args", "expected_ip", "forwarded"),
    [
        ([], "127.0.0.1", []),
        (["192.0.2.7"], "192.0.2.7", []),
        (["--debug"], "127.0.0.1", ["--debug"]),  # an option is not an IP
        (["192.0.2.7", "--render"], "192.0.2.7", ["--render"]),
    ],
)
def test_run_agent_local_picks_the_host_and_forwards_options(
    tmp_path, args, expected_ip, forwarded
):
    fake_python = fake_command(tmp_path, "fakepython")

    result = run_script(
        REMOTE / "run_agent_local.sh",
        args,
        tmp_path,
        PYTHON=str(fake_python),
        PORT="50123",
        TIMEOUT="7",
    )

    assert result.returncode == 0, result.stderr
    (call,) = calls(tmp_path, "fakepython")
    assert call[0].endswith("examples/lunar_lander/agent_side_lunarlander.py")
    assert has_pair(call, "--ip", expected_ip)
    assert has_pair(call, "--port", "50123")
    assert has_pair(call, "--timeout", "7")
    for option in forwarded:
        assert option in call
    assert "--render" in call or "--render" not in args  # never added by default


def test_agent_does_not_render_unless_asked(tmp_path):
    fake_python = fake_command(tmp_path, "fakepython")

    run_script(REMOTE / "run_agent_local.sh", [], tmp_path, PYTHON=str(fake_python))

    (call,) = calls(tmp_path, "fakepython")
    assert "--render" not in call


def test_tunnel_forwards_the_requested_ports(tmp_path):
    fake_command(tmp_path, "ssh")

    default = run_script(REMOTE / "tunnel.sh", ["user@host"], tmp_path)
    custom = run_script(REMOTE / "tunnel.sh", ["user@host", "49054", "50000"], tmp_path)

    assert default.returncode == 0 and custom.returncode == 0
    first, second = calls(tmp_path, "ssh")
    assert has_pair(first, "-L", "49054:127.0.0.1:49054")
    assert has_pair(second, "-L", "50000:127.0.0.1:49054")
    assert first[-1] == "user@host"
    assert "-v" not in first  # no debug verbosity


def test_tunnel_requires_a_host(tmp_path):
    fake_command(tmp_path, "ssh")

    result = run_script(REMOTE / "tunnel.sh", [], tmp_path)

    assert result.returncode != 0
    assert "usage" in result.stderr.lower()


# Container deployment.
def test_container_profile_a_uses_host_networking_and_the_gpu(tmp_path):
    fake_command(tmp_path, "docker")
    model_dir = tmp_path / "models"

    result = run_script(
        CONTAINER / "run_rl_container.sh",
        ["A", "--timesteps", "20000", "--model-path", str(model_dir / "m.zip")],
        tmp_path,
        PORT="50123",
    )

    assert result.returncode == 0, result.stderr
    run = [c for c in calls(tmp_path, "docker") if c[0] == "run"][0]
    assert has_pair(run, "--gpus", "all")
    assert has_pair(run, "--network", "host")
    assert has_pair(run, "-v", f"{model_dir}:/output")
    assert "lunarlander-rl" in run
    rl_args = run[run.index("lunarlander-rl") + 1 :]
    assert has_pair(rl_args, "--port", "50123")
    assert has_pair(rl_args, "--device", "cuda")
    assert has_pair(rl_args, "--timeout", "10")
    assert has_pair(rl_args, "--timesteps", "20000")
    assert has_pair(rl_args, "--model-path", "/output/m.zip")
    assert "--network" not in rl_args and "A" not in rl_args
    assert model_dir.is_dir()  # created so the mount has a source


def test_container_profile_b_publishes_only_on_loopback(tmp_path):
    fake_command(tmp_path, "docker")

    result = run_script(
        CONTAINER / "run_rl_container.sh", ["B"], tmp_path, PORT="50123"
    )

    assert result.returncode == 0, result.stderr
    run = [c for c in calls(tmp_path, "docker") if c[0] == "run"][0]
    assert has_pair(run, "-p", "127.0.0.1:50123:50123")
    assert "--network" not in run


def test_container_cpu_does_not_request_the_nvidia_runtime(tmp_path):
    fake_command(tmp_path, "docker")

    run_script(CONTAINER / "run_rl_container.sh", [], tmp_path, DEVICE="cpu")

    run = [c for c in calls(tmp_path, "docker") if c[0] == "run"][0]
    assert "--gpus" not in run
    assert has_pair(run[run.index("lunarlander-rl") + 1 :], "--device", "cpu")


def test_container_default_model_goes_to_the_output_directory(tmp_path):
    fake_command(tmp_path, "docker")

    run_script(CONTAINER / "run_rl_container.sh", [], tmp_path)

    run = [c for c in calls(tmp_path, "docker") if c[0] == "run"][0]
    expected = (CONTAINER / "output").resolve()
    assert has_pair(run, "-v", f"{expected}:/output")
    assert has_pair(run, "--model-path", "/output/ppo_lunar_lander_decoupled.zip")
    # The directory was created by the script; remove the artifact of this test.
    if expected.is_dir() and not any(expected.iterdir()):
        expected.rmdir()


def test_container_builds_the_image_when_it_is_missing(tmp_path):
    # `docker image inspect` fails => the script must build before running.
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" >> "{tmp_path}/docker.args"\n'
        f'echo "---" >> "{tmp_path}/docker.args"\n'
        '[[ "$1 $2" == "image inspect" ]] && exit 1\n'
        "exit 0\n"
    )
    fake.chmod(0o755)

    result = run_script(
        CONTAINER / "run_rl_container.sh", ["B"], tmp_path, DEVICE="cpu"
    )

    assert result.returncode == 0, result.stderr
    verbs = [c[0] for c in calls(tmp_path, "docker")]
    assert verbs == ["image", "build", "run"]
    build = calls(tmp_path, "docker")[1]
    assert has_pair(build, "-f", "examples/lunar_lander_container/Dockerfile")
    assert has_pair(build, "-t", "lunarlander-rl")
    assert build[-1] == "."


def test_container_agent_scripts_reuse_the_bare_host_ones(tmp_path):
    fake_python = fake_command(tmp_path, "fakepython")
    fake_command(tmp_path, "ssh")

    agent = run_script(
        CONTAINER / "run_agent_local.sh",
        ["192.0.2.7", "--debug"],
        tmp_path,
        PYTHON=str(fake_python),
    )
    tunnel = run_script(
        CONTAINER / "tunnel.sh", ["user@host", "49054", "50000"], tmp_path
    )

    assert agent.returncode == 0, agent.stderr
    assert tunnel.returncode == 0, tunnel.stderr
    (agent_call,) = calls(tmp_path, "fakepython")
    assert has_pair(agent_call, "--ip", "192.0.2.7") and "--debug" in agent_call
    (tunnel_call,) = calls(tmp_path, "ssh")
    assert has_pair(tunnel_call, "-L", "50000:127.0.0.1:49054")
