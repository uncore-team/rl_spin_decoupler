# SPDX-License-Identifier: GPL-3.0-only

"""Dependency-light smoke tests for the real Burger example."""

import py_compile
from pathlib import Path

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

    assert "PPO.load" in source
    assert "deterministic=True" in source
    assert "model.learn(" not in source
