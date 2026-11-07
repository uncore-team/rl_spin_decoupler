# SPDX-License-Identifier: GPL-3.0-only

"""Dependency-free smoke tests for the CoppeliaSim Burger example.

The example needs CoppeliaSim and Stable-Baselines3, so it cannot run here. These
tests guard the static contracts that are easy to break unnoticed: valid Python,
the shared scene, the launcher, and the constants that code in different places
must keep in agreement.
"""

import ast
import py_compile
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "burger_coppeliasim"
# The scene is shared with examples/burger_real_sim.
SCENE_DIR = REPO_ROOT / "examples" / "_common" / "burger_scene"
REAL_CONTRACT = (
    REPO_ROOT
    / "examples"
    / "burger_real"
    / "ros2"
    / "burger_real_agent"
    / "contract.py"
)


def _constants(path: Path) -> dict:
    """Evaluate the plain top-level constants of a dependency-free module."""
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def _scene_laser_max_scan_distance() -> float:
    """Read the scene laser's ``config["max_scan_distance"]`` without importing it."""
    tree = ast.parse((SCENE_DIR / "laser.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "config"
            and isinstance(node.value, ast.Dict)
        ):
            for key, value in zip(node.value.keys, node.value.values):
                if getattr(key, "value", None) == "max_scan_distance":
                    return float(ast.literal_eval(value))
    raise AssertionError("config['max_scan_distance'] not found in laser.py")


def test_burger_coppeliasim_entrypoints_compile():
    for filename in (
        "agent_side_burger_coppeliasim.py",
        "rl_side_burger_coppeliasim.py",
        "reward.py",
        "episode_config.py",
    ):
        py_compile.compile(str(EXAMPLE_DIR / filename), doraise=True)


def test_burger_coppeliasim_scene_is_available():
    scene_path = SCENE_DIR / "scene.ttt"

    assert scene_path.is_file()
    assert scene_path.stat().st_size > 0


def test_episode_config_matches_the_scene_lidar_range():
    """episode_config.LIDAR_MAX_RANGE must equal the scene's max_scan_distance."""
    config = _constants(EXAMPLE_DIR / "episode_config.py")

    assert config["LIDAR_MAX_RANGE"] == _scene_laser_max_scan_distance()


def test_episode_config_matches_the_real_robot_contract():
    """A policy trained here must run unchanged on the real robot."""
    config = _constants(EXAMPLE_DIR / "episode_config.py")
    real = _constants(REAL_CONTRACT)

    assert config["NUM_LIDAR_SECTORS"] == real["NUM_LIDAR_SECTORS"]
    assert config["OBS_DIM"] == real["OBS_DIM"]
    assert config["LIDAR_MAX_RANGE"] == real["LIDAR_MAX_RANGE"]
    assert config["COLLISION_LIDAR_THRESHOLD"] == real["COLLISION_LIDAR_THRESHOLD"]
    assert config["GOAL_DIST_THRESHOLD"] == real["GOAL_DIST_THRESHOLD"]
    assert config["MAX_LINEAR_VELOCITY"] == real["LINEAR_VELOCITY_LIMIT"]
    assert config["MAX_ANGULAR_VELOCITY"] == real["ANGULAR_VELOCITY_LIMIT"]


def test_demo_script_is_valid_bash():
    script = EXAMPLE_DIR / "run_demo.sh"

    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)
