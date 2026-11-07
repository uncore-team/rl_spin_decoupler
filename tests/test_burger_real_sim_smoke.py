# SPDX-License-Identifier: GPL-3.0-only

"""Dependency-free smoke tests for the shared Burger scene and the ROS2 bridge.

The scene scripts run inside CoppeliaSim and the bridge needs ROS2, so none of
them can be imported here. These tests only guard the static contracts that are
easy to break unnoticed: valid Python, a ROS2-agnostic scene, the functions the
scene wrappers must re-export, and the bridge/real-agent contract agreement.
"""

from __future__ import annotations

import ast
import py_compile
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = REPO_ROOT / "examples" / "_common" / "burger_scene"
SIM_DIR = REPO_ROOT / "examples" / "burger_real_sim"
BRIDGE_DIR = SIM_DIR / "ros2" / "burger_sim_bridge"
AGENT_CONTRACT = (
    REPO_ROOT
    / "examples"
    / "burger_real"
    / "ros2"
    / "burger_real_agent"
    / "contract.py"
)

SCENE_MODULES = (
    "common",
    "arena",
    "obstacles",
    "target",
    "burger",
    "laser",
    "orchestrator",
)

# Functions each object's wrapper (inside scene.ttt) re-exports and external
# code calls with sim.callScriptFunction. Keep in sync with the README table.
WRAPPER_EXPORTS = {
    "arena": {"getExternalWallParams"},
    "obstacles": {"rebuild"},
    "target": {"reset_pose"},
    "burger": {"set_velocity", "reset_pose", "stop", "get_odom_pose"},
    "laser": {"get_rl_observation"},
    "orchestrator": {
        "reset_episode",
        "step_episode",
        "get_state",
        "ros_reset",
        "ros_step",
        "get_ros_state",
        "get_static_transforms",
    },
}


def _top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _constants(path: Path) -> dict:
    """Evaluate the plain top-level constants of a dependency-free module."""
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def test_scene_scripts_compile():
    for module in SCENE_MODULES:
        py_compile.compile(str(SCENE_DIR / f"{module}.py"), doraise=True)


def test_scene_file_is_available():
    scene = SCENE_DIR / "scene.ttt"
    assert scene.is_file()
    assert scene.stat().st_size > 0


def test_scene_is_ros2_agnostic():
    for module in SCENE_MODULES:
        source = (SCENE_DIR / f"{module}.py").read_text()
        assert "require('simROS2')" not in source and 'require("simROS2")' not in source
        assert "createPublisher" not in source and "createSubscription" not in source


def test_scene_modules_define_the_functions_their_wrappers_export():
    for module, exported in WRAPPER_EXPORTS.items():
        missing = exported - _top_level_functions(SCENE_DIR / f"{module}.py")
        assert not missing, f"{module}.py does not define {sorted(missing)}"


def test_bridge_sources_compile():
    for filename in ("bridge_node.py", "contract.py", "setup.py"):
        py_compile.compile(str(BRIDGE_DIR / filename), doraise=True)
    assert (BRIDGE_DIR / "package.xml").is_file()


def test_bridge_contract_matches_the_real_agent_contract():
    bridge = _constants(BRIDGE_DIR / "contract.py")
    agent = _constants(AGENT_CONTRACT)

    assert bridge["SCAN_TOPIC"] == agent["SCAN_TOPIC"]
    assert bridge["ODOM_TOPIC"] == agent["ODOM_TOPIC"]
    assert bridge["GOAL_TOPIC"] == agent["GOAL_TOPIC"]
    assert bridge["CMD_VEL_TOPIC"] == agent["CMD_VEL_TOPIC"]
    assert bridge["ODOM_FRAME"] == agent["ODOM_FRAME"]
    assert bridge["MAP_FRAME"] == agent["GOAL_FRAME"]
    assert bridge["LINEAR_VELOCITY_LIMIT"] == agent["LINEAR_VELOCITY_LIMIT"]
    assert bridge["ANGULAR_VELOCITY_LIMIT"] == agent["ANGULAR_VELOCITY_LIMIT"]
    assert bridge["LIDAR_MAX_RANGE"] == agent["LIDAR_MAX_RANGE"]


def test_demo_script_is_valid_bash():
    script = SIM_DIR / "run_demo.sh"
    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_demo_script_puts_the_repository_on_pythonpath_for_the_ros_nodes():
    """`ros2 run` uses the system Python, which does not see the virtualenv's
    editable install (a .pth file that PYTHONPATH ignores): the launcher must put
    the directory holding spindecoupler.py on PYTHONPATH itself."""

    script = (SIM_DIR / "run_demo.sh").read_text()

    assert 'export PYTHONPATH="$REPO_ROOT:' in script
    assert (REPO_ROOT / "spindecoupler.py").is_file()
