# SPDX-License-Identifier: GPL-3.0-only
"""Episode orchestrator (CoppeliaSim child script).

Single, ROS2-agnostic entry point for driving the Burger arena. It exposes
two surfaces over the ZMQ remote API, and nothing else decides when an
episode ends; that decision lives outside the scene:

  1. RL training surface (used by examples/burger_coppeliasim through
     rl_spin_decoupler):
       - reset_episode()        -> 10-dim observation
       - step_episode(action)   -> 10-dim observation

  2. ROS2 bridge surface (used by examples/burger_real_sim's bridge node):
       - ros_reset()            -> rich state dict (new episode)
       - ros_step(action)       -> rich state dict (one control tick)
       - get_ros_state()        -> rich state dict (no side effects)
       - get_static_transforms()-> fixed base_link -> base_scan transform

Design rules kept from the original script:
  - It delegates "how to physically reset/move an object" to that object's
    own script (burger.py::reset_pose, target.py::reset_pose,
    obstacles.py::rebuild) instead of reimplementing it here.
  - Spawn sampling always respects the arena's real, current wall dimensions
    (read from arena.py through getExternalWallParams()), never a hardcoded
    constant.
"""

import math
import random
from typing import Any, Dict, List, Tuple

import common

# Global sim object, set once in init() and used in all other functions.
sim = None

# Cached object/script handles, resolved once in init() instead of on every
# reset_episode()/step_episode() call.
robot_obj = None
target_obj = None
laser_obj = None
burger_script = None
target_script = None
laser_script = None
obstacles_script = None
wall_script = None

MIN_TARGET_DIST = 0.75  # Minimum robot-target distance at spawn (m).
TARGET_SAMPLING_ATTEMPTS = 200  # Bounded retries; see _sample_target_position.
MIN_SPAWN_HALF_EXTENT = 0.05  # Never return a degenerate/negative spawn area.


def init(simulation: Any):
    global sim, robot_obj, target_obj, laser_obj
    global burger_script, target_script, laser_script, obstacles_script, wall_script

    sim = simulation  # Store the sim object for later use in other functions.

    robot_obj = sim.getObject(common.SCENE_NAMES["robot"])
    target_obj = sim.getObject(common.SCENE_NAMES["target"])
    laser_obj = sim.getObject(common.SCENE_NAMES["laser"])

    burger_script = sim.getScript(sim.scripttype_childscript, robot_obj)
    target_script = sim.getScript(sim.scripttype_childscript, target_obj)
    laser_script = sim.getScript(sim.scripttype_childscript, laser_obj)
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


def _spawn_bounds(radius: float, wall_cfg: Dict[str, Any]) -> Tuple[float, float]:
    half_x = wall_cfg["scene_x_dim"] / 2.0 - wall_cfg["scene_walls_thickness"] - radius
    half_y = wall_cfg["scene_y_dim"] / 2.0 - wall_cfg["scene_walls_thickness"] - radius
    return max(half_x, MIN_SPAWN_HALF_EXTENT), max(half_y, MIN_SPAWN_HALF_EXTENT)


def _sample_robot_position(half_x: float, half_y: float) -> Tuple[float, float, float]:
    """Sample a random robot spawn (x, y) and yaw in the arena.

    The robot is spawned at least `common.ROBOT_RADIUS` away from the walls.
    """
    rx = random.uniform(-half_x, half_x)
    ry = random.uniform(-half_y, half_y)
    ryaw = random.uniform(-math.pi, math.pi)
    return (rx, ry, ryaw)


def _sample_target_position(
    rx: float, ry: float, half_x: float, half_y: float
) -> Tuple[float, float]:
    """Sample a target position at least MIN_TARGET_DIST away from (rx, ry).

    Bounded retry loop. Falls back to the farthest candidate seen across all
    attempts if none satisfies the constraint (e.g. MIN_TARGET_DIST doesn't
    geometrically fit the configured arena).
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


def _reset_placement() -> None:
    """Reposition robot/target/obstacles for a new episode (no observation)."""
    wall_cfg = sim.callScriptFunction("getExternalWallParams", wall_script)

    robot_half_x, robot_half_y = _spawn_bounds(common.ROBOT_RADIUS, wall_cfg)
    rx, ry, ryaw = _sample_robot_position(robot_half_x, robot_half_y)
    sim.callScriptFunction("reset_pose", burger_script, rx, ry, ryaw)

    target_half_x, target_half_y = _spawn_bounds(common.TARGET_RADIUS, wall_cfg)
    tx, ty = _sample_target_position(rx, ry, target_half_x, target_half_y)
    sim.callScriptFunction("reset_pose", target_script, tx, ty)

    # Obstacles are rebuilt last so they can avoid the just-placed robot/target.
    sim.callScriptFunction("rebuild", obstacles_script)


# RL training surface (burger_coppeliasim, via rl_spin_decoupler)
def reset_episode() -> List[float]:
    """Called once per episode by agent_side (via rl_spin_decoupler)."""
    _reset_placement()
    return get_state()


def step_episode(action: List[float]) -> List[float]:
    """Called every control tick by agent_side: apply `action`, return the observation.

    Termination and truncation are not decided here. The scene only reports raw
    sensor and geometry state, and the agent and RL sides decide when the
    episode is over.
    """
    sim.callScriptFunction(
        "set_velocity", burger_script, float(action[0]), float(action[1])
    )
    return get_state()


def get_state() -> List[float]:
    """Assemble the model observation: distance, angle, then 8 LiDAR sectors."""
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

    return [dist, relative_angle] + list(lidar)


# ROS2 bridge surface (burger_real_sim's bridge node)
def ros_reset() -> Dict[str, Any]:
    """Stop, reposition everything for a new mission, and return fresh state."""
    sim.callScriptFunction("stop", burger_script)
    _reset_placement()
    return get_ros_state()


def ros_step(action: List[float]) -> Dict[str, Any]:
    """Apply one /cmd_vel (linear, angular) and return fresh state (one round-trip)."""
    sim.callScriptFunction(
        "set_velocity", burger_script, float(action[0]), float(action[1])
    )
    return get_ros_state()


def get_ros_state() -> Dict[str, Any]:
    """Aggregate everything the ROS2 bridge needs, in one ZMQ round-trip.

    Keys:
      - sim_time:          CoppeliaSim time (s), for /clock and message stamps.
      - lidar:             8 sector ranges (m) -> /scan.
      - robot_world_pos/quat, target_world_pos: world-frame geometry, used by
        the bridge's operator role (goal/collision checks) and to publish the
        goal once per mission.
      - odom_pos/quat:     base_link in odom -> /odom pose + odom->base_link TF.
      - map_odom_pos/quat: odom in map    -> map->odom TF.
    """
    lidar = sim.callScriptFunction("get_rl_observation", laser_script)
    odom = sim.callScriptFunction("get_odom_pose", burger_script)

    return {
        "sim_time": float(sim.getSimulationTime()),
        "lidar": [float(v) for v in lidar],
        "robot_world_pos": [
            float(v) for v in sim.getObjectPosition(robot_obj, sim.handle_world)
        ],
        "robot_world_quat": [
            float(v) for v in sim.getObjectQuaternion(robot_obj, sim.handle_world)
        ],
        "target_world_pos": [
            float(v) for v in sim.getObjectPosition(target_obj, sim.handle_world)
        ],
        "odom_pos": [float(v) for v in odom["pos"]],
        "odom_quat": [float(v) for v in odom["quat"]],
        "map_odom_pos": [float(v) for v in odom["map_pos"]],
        "map_odom_quat": [float(v) for v in odom["map_quat"]],
    }


def get_static_transforms() -> Dict[str, Any]:
    """Return the fixed base_link -> base_scan transform (the LiDAR mount offset).

    Read once by the bridge at startup and published on /tf_static, so rviz2
    can place /scan (base_scan would otherwise be a frame disconnected from
    base_link/map).
    """
    return {
        "base_scan_pos": [
            float(v) for v in sim.getObjectPosition(laser_obj, robot_obj)
        ],
        "base_scan_quat": [
            float(v) for v in sim.getObjectQuaternion(laser_obj, robot_obj)
        ],
    }


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
