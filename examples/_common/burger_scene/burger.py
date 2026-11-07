# SPDX-License-Identifier: GPL-3.0-only
"""TurtleBot3 Burger low-level driver (CoppeliaSim child script).

Owns wheel velocity control, the odometry reference dummy, and ground-truth
trajectory drawing. `reset_pose(x, y, yaw)` is the single place that knows
*how* to physically place the robot for a new episode (motors, dynamics,
odometry frame); orchestrator.py decides *where* (spawn sampling within the
real arena bounds) and calls this function instead of teleporting the robot
itself.

This script is ROS2-agnostic. It produces robot-level data only:
- wheel velocity control driven through `set_velocity()`;
- odometry, as the robot's pose relative to `odom_dummy` (a reference frame
  re-anchored at every `reset_pose()`), exposed through `get_odom_pose()`.

The external ROS2 bridge (examples/burger_real_sim) reads that odometry via
orchestrator.get_ros_state() and turns it into `/odom` + the
`map -> odom -> base_link` TF chain. Likewise `/cmd_vel` does not arrive
here through a simROS2 subscription: the bridge receives it and calls
`set_velocity()` through the ZMQ remote API, exactly like the training
example drives `step_episode()`.
"""

from typing import Any, Dict, List

WHEEL_BASE = 0.16
WHEEL_RADIUS = 0.033


class State:
    sim = None
    robot = None
    footprint = None
    motor_left = None
    motor_right = None

    odom_dummy = None
    initial_pos: List[float] = []
    pose_gt = None


def set_velocity(v: float, w: float) -> None:
    """Apply inverse differential-drive kinematics and drive the wheel joints.

    v: linear velocity (m/s), w: angular velocity (rad/s)
    """
    v_left = v - (w * WHEEL_BASE / 2.0)
    v_right = v + (w * WHEEL_BASE / 2.0)

    State.sim.setJointTargetVelocity(State.motor_left, v_left / WHEEL_RADIUS)
    State.sim.setJointTargetVelocity(State.motor_right, v_right / WHEEL_RADIUS)

    # Trace the ground-truth trajectory.
    p = State.sim.getObjectPosition(State.footprint, State.sim.handle_world)
    State.sim.addDrawingObjectItem(State.pose_gt, p)


def stop() -> None:
    """Zero the wheels immediately.

    Called by orchestrator.ros_reset() (via the bridge's operator role) right
    before a mission reset, so the robot doesn't carry a stale velocity into
    the teleport.
    """
    set_velocity(0.0, 0.0)


def reset_pose(x: float, y: float, yaw: float = 0.0) -> None:
    """Teleport the robot to (x, y, yaw) in world coordinates for a new episode.

    Keeps the original spawn height recorded in init() and re-anchors the
    odometry reference frame (odom_dummy) at the new pose. Stops the wheels
    and forces a dynamics reset first, so the physics engine doesn't fight
    the teleport.
    """
    sim = State.sim

    sim.setJointTargetVelocity(State.motor_left, 0.0)
    sim.setJointTargetVelocity(State.motor_right, 0.0)

    world_pos = [float(x), float(y), State.initial_pos[2]]
    sim.setObjectPosition(State.odom_dummy, sim.handle_world, world_pos)
    sim.setObjectOrientation(State.odom_dummy, sim.handle_world, [0.0, 0.0, float(yaw)])

    # Place the robot at the origin of the (just-moved) odometry frame.
    sim.setObjectPosition(State.robot, State.odom_dummy, [0.0, 0.0, 0.0])
    sim.setObjectQuaternion(State.robot, State.odom_dummy, [0.0, 0.0, 0.0, 1.0])

    # Force a dynamics refresh to avoid physics jerks right after teleporting.
    for obj in (State.robot, State.motor_left, State.motor_right):
        sim.resetDynamicObject(obj)

    # Clear the drawn trajectory line.
    sim.addDrawingObjectItem(State.pose_gt, None)


# Odometry readout (consumed by orchestrator.get_ros_state / the ROS2 bridge)
def get_odom_pose() -> Dict[str, List[float]]:
    """Return the robot's pose in the odometry frame, plus that frame in world.

    - `pos` / `quat`: the robot pose relative to `odom_dummy` (i.e. the pose
      of `base_link` in `odom`), which is what `/odom` carries.
    - `map_pos` / `map_quat`: `odom_dummy`'s own world pose (i.e. the pose of
      `odom` in `map`), which is what the `map -> odom` TF carries. It is
      re-anchored at every reset_pose(), matching a real navigation stack
      where odometry drifts within a fixed map.
    """
    sim = State.sim
    return {
        "pos": list(sim.getObjectPosition(State.robot, State.odom_dummy)),
        "quat": list(sim.getObjectQuaternion(State.robot, State.odom_dummy)),
        "map_pos": list(sim.getObjectPosition(State.odom_dummy, sim.handle_world)),
        "map_quat": list(sim.getObjectQuaternion(State.odom_dummy, sim.handle_world)),
    }


def init(simulation: Any) -> None:
    State.sim = simulation
    sim = State.sim

    State.robot = sim.getObject("..")
    State.footprint = sim.getObject("/base_link_visual")
    State.motor_left = sim.getObject("/wheel_left_joint")
    State.motor_right = sim.getObject("/wheel_right_joint")

    State.initial_pos = sim.getObjectPosition(State.robot, sim.handle_world)

    # Reference dummy for odometry, re-anchored at every reset_pose() call.
    robot_alias = sim.getObjectAlias(State.robot, 3)
    State.odom_dummy = sim.createDummy(0.01)
    sim.setObjectAlias(State.odom_dummy, f"{robot_alias}_initial_pose")

    p = sim.getObjectPosition(State.footprint, sim.handle_world)
    q = sim.getObjectQuaternion(State.footprint, sim.handle_world)
    sim.setObjectPosition(State.odom_dummy, sim.handle_world, p)
    sim.setObjectQuaternion(State.odom_dummy, sim.handle_world, q)

    # Ground-truth trajectory drawing.
    State.pose_gt = sim.addDrawingObject(
        sim.drawing_linestrip, 3, 0, sim.handle_world, 1000, [1, 0, 0]
    )

    print("[Coppelia] Burger robot script initialized.")


def actuation() -> None:
    pass


def sensing() -> None:
    pass


def cleanup() -> None:
    pass
