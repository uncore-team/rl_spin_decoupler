# SPDX-License-Identifier: GPL-3.0-only
"""Episode orchestrator (CoppeliaSim child script).

Coordinates `reset_episode()` / `step_episode(action)`, called every episode
reset / control tick by agent_side_burger_coppeliasim.py through
rl_spin_decoupler.

Design rule: this script never decides whether an episode has ended. That
decision belongs entirely to the agent-side process (see reward.py:
is_terminated / is_truncated), by explicit project design. This script only:

  1. Delegates "how to physically reset/move an object" to that object's own
     script (burger.py::reset_pose, target.py::reset_pose,
     obstacles.py::rebuild) instead of reimplementing it here.
  2. Samples *where* the robot and target spawn each episode, respecting the
     arena's real, current wall dimensions (read from arena.py through
     getExternalWallParams(), not a hardcoded constant).
  3. Assembles the 10-dim observation vector (8 LiDAR sectors + distance +
     relative angle to target) every control tick.
"""

import common

import math
import random
from typing import Any, Dict, List, Tuple

# Global sim object, set once in init() and used in all other functions.
sim = None

# Cached object/script handles, resolved once in sysCall_init instead of on
# every reset_episode()/step_episode() call.
robot_obj = None
target_obj = None
burger_script = None
target_script = None
laser_script = None
obstacles_script = None
wall_script = None

MIN_TARGET_DIST = 0.75           # Minimum robot-target distance at spawn (m).
TARGET_SAMPLING_ATTEMPTS = 200  # Bounded retries; see _sample_target_position.
MIN_SPAWN_HALF_EXTENT = 0.05    # Never return a degenerate/negative spawn area.


# ----------------------------------------------------------------------------
# CoppeliaSim lifecycle
# ----------------------------------------------------------------------------
def init(simulation: Any):
    global sim, robot_obj, target_obj
    global burger_script, target_script, laser_script, obstacles_script, wall_script

    sim = simulation # Store the sim object for later use in other functions.

    robot_obj = sim.getObject(common.SCENE_NAMES["robot"])
    target_obj = sim.getObject(common.SCENE_NAMES["target"])

    burger_script = sim.getScript(sim.scripttype_childscript, robot_obj)
    target_script = sim.getScript(sim.scripttype_childscript, target_obj)
    laser_script = sim.getScript(
        sim.scripttype_childscript, sim.getObject(common.SCENE_NAMES["laser"])
    )
    obstacles_script = sim.getScript(
        sim.scripttype_childscript, sim.getObject(common.SCENE_NAMES["obstacles"])
    )
    wall_script = sim.getScript(
        sim.scripttype_childscript, sim.getObject(common.SCENE_NAMES["walls"])
    )


def actuation():
    pass


def sensing():
    pass


def cleanup():
    pass


# ----------------------------------------------------------------------------
# Spawn sampling
# ----------------------------------------------------------------------------
def _spawn_bounds(radius: float, wall_cfg: Dict[str, Any]) -> Tuple[float, float]:
    half_x = wall_cfg["scene_x_dim"] / 2.0 - wall_cfg["scene_walls_thickness"] - radius
    half_y = wall_cfg["scene_y_dim"] / 2.0 - wall_cfg["scene_walls_thickness"] - radius
    return max(half_x, MIN_SPAWN_HALF_EXTENT), max(half_y, MIN_SPAWN_HALF_EXTENT)


def _sample_robot_position(half_x: float, half_y: float) -> Tuple[float, float, float]:
    """
    Sample a random robot spawn position (x, y) and yaw angle in the arena.
    The robot is spawned at least `common.ROBOT_RADIUS` away from the walls.
    """
    rx = random.uniform(-half_x, half_x)
    ry = random.uniform(-half_y, half_y)
    ryaw = random.uniform(-math.pi, math.pi)
    return (rx, ry, ryaw)


def _sample_target_position(rx: float, ry: float, half_x: float, half_y: float) -> Tuple[float, float]:
    """Sample a target position at least MIN_TARGET_DIST away from (rx, ry).

    Bounded retry loop, which can spin forever if MIN_TARGET_DIST doesn't
    geometrically fit the configured arena. Falls back to the farthest
    candidate seen across all attempts if none satisfies the constraint.
    """
    best_xy = (0.0, 0.0)
    best_dist = -1.0
    for _ in range(TARGET_SAMPLING_ATTEMPTS):
        tx = random.uniform(-half_x, half_x)
        ty = random.uniform(-half_y, half_y)
        dist = math.hypot(tx - rx, ty - ry)
        if dist >= MIN_TARGET_DIST:
            return tx, ty
        if dist > best_dist:
            best_xy, best_dist = (tx, ty), dist
    return best_xy


# ----------------------------------------------------------------------------
# RL-facing API
# ----------------------------------------------------------------------------
def reset_episode() -> List[float]:
    """Called once per episode by agent_side (via rl_spin_decoupler)."""
    wall_cfg = sim.callScriptFunction("getExternalWallParams", wall_script)

    robot_half_x, robot_half_y = _spawn_bounds(common.ROBOT_RADIUS, wall_cfg)
    rx, ry, ryaw = _sample_robot_position(robot_half_x, robot_half_y)
    sim.callScriptFunction("reset_pose", burger_script, rx, ry, ryaw)

    target_half_x, target_half_y = _spawn_bounds(common.TARGET_RADIUS, wall_cfg)
    tx, ty = _sample_target_position(rx, ry, target_half_x, target_half_y)
    sim.callScriptFunction("reset_pose", target_script, tx, ty)

    # Obstacles are rebuilt last so they can avoid the just-placed robot/target.
    sim.callScriptFunction("rebuild", obstacles_script)

    return get_state()


def step_episode(action: List[float]) -> List[float]:
    """Called every control tick by agent_side. Applies `action`, returns the observation.

    Termination/truncation are NOT decided here on purpose -- see the module
    docstring. This only reports raw sensor/geometry state; agent-side
    (reward.py) is the sole authority on whether the episode is over.
    """
    sim.callScriptFunction("set_velocity", burger_script, float(action[0]), float(action[1]))
    return get_state()


def get_state() -> List[float]:
    """Assemble the 10-dim observation: 8 LiDAR sectors + distance + relative angle."""
    lidar = sim.callScriptFunction("get_rl_observation", laser_script)

    robot_pos = sim.getObjectPosition(robot_obj, sim.handle_world)
    robot_ori = sim.getObjectOrientation(robot_obj, sim.handle_world)
    target_pos = sim.getObjectPosition(target_obj, sim.handle_world)

    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    dist = math.hypot(dx, dy)
    # robot_ori[2] is the robot's yaw (Z rotation) for a ground robot kept
    # upright, matching how reset_pose()/burger.py set its orientation.
    relative_angle = _wrap_to_pi(math.atan2(dy, dx) - robot_ori[2])

    return list(lidar) + [dist, relative_angle]


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
