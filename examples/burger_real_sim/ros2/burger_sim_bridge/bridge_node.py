# SPDX-License-Identifier: GPL-3.0-only
"""CoppeliaSim <-> ROS2 bridge: makes the simulated Burger look like a real one.

The scene (``examples/_common/burger_scene``) knows nothing about ROS2. This node
connects to it through the ZMQ remote API and exposes the same ROS2 contract as
the real TurtleBot3 Burger, so the real-robot agent
(``examples/burger_real/ros2/burger_real_agent``) runs unchanged. It plays the
two roles that on a real robot belong to different parties:

1. **Robot drivers** (stateless). Every control tick it makes ONE batched
   ``orchestrator.ros_step`` call (applies the latest ``/cmd_vel`` and returns
   the whole state), then publishes ``/scan``, ``/odom``, TF and ``/clock``.
   ``/scan`` and ``/odom`` are rate-limited in simulation time (5 Hz and 20 Hz
   by default) as a real robot would.

2. **Operator** (stateful). The agent decides when an episode ends (goal
   reached, collision, truncation) and stops commanding motion when it does.
   The operator watches for that: once ``/cmd_vel`` has been idle for
   ``IDLE_RESET_S`` it repositions robot, target and obstacles
   (``orchestrator.ros_reset``) and then offers the new ``/goal_pose`` until the
   agent starts moving again. A real operator likewise waits for the robot to
   stop before moving it, so the world never changes mid-episode.

The only simulation-specific seam is ``ros_reset`` (the operator's "hands"). On
a real robot a person or a mission node publishes the goal, and this whole node
is replaced by the robot's own drivers.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from enum import Enum

import rclpy
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .contract import (
    ANGULAR_VELOCITY_LIMIT,
    BASE_FRAME,
    CLOCK_TOPIC,
    CMD_VEL_TOPIC,
    CONTROL_HZ,
    GOAL_TOPIC,
    LIDAR_MAX_RANGE,
    LINEAR_VELOCITY_LIMIT,
    MAP_FRAME,
    ODOM_FRAME,
    ODOM_HZ,
    ODOM_TOPIC,
    SCAN_FRAME,
    SCAN_HZ,
    SCAN_TOPIC,
)

# Operator timing, in simulation seconds.
IDLE_RESET_S = 1.0  # idle /cmd_vel for this long => the episode is over, reset
AWAIT_REPUBLISH_S = 0.5  # while waiting for the agent, re-offer the goal this often
CMD_MOVING_EPS = 1e-3  # |cmd_vel| above this counts as "the agent is moving"


class _RateGate:
    """Lets an action through at most ``hz`` times per simulation second."""

    def __init__(self, hz: float) -> None:
        self._period = 1.0 / hz if hz > 0 else 0.0
        self._last: float | None = None

    def due(self, sim_time: float) -> bool:
        if self._last is None or sim_time - self._last >= self._period:
            self._last = sim_time
            return True
        return False

    def reset(self) -> None:
        self._last = None


class _Phase(Enum):
    AWAIT_START = 0  # robot parked; offering the goal until the agent moves
    RUNNING = 1  # robot driven by the agent's /cmd_vel


class BurgerSimBridge(Node):
    """Drivers + operator for the simulated Burger (see the module docstring)."""

    def __init__(
        self,
        sim_host: str,
        sim_port: int,
        scene_path: str | None,
        scan_hz: float,
        odom_hz: float,
        control_hz: float,
    ) -> None:
        super().__init__("burger_sim_bridge")
        self._control_period = 1.0 / control_hz
        self._scan_gate = _RateGate(scan_hz)
        self._odom_gate = _RateGate(odom_hz)

        # CoppeliaSim, over the ZMQ remote API.
        self._sim = RemoteAPIClient(host=sim_host, port=sim_port).require("sim")
        if scene_path:
            self._load_scene(scene_path)
        self._orchestrator = self._sim.getScript(
            self._sim.scripttype_childscript, self._sim.getObject("/Orchestrator")
        )

        # ROS2 interface. The goal is latched so late joiners (e.g. rviz2) get it.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._scan_pub = self.create_publisher(LaserScan, SCAN_TOPIC, 10)
        self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self._goal_pub = self.create_publisher(PoseStamped, GOAL_TOPIC, latched)
        self._clock_pub = self.create_publisher(Clock, CLOCK_TOPIC, 10)
        self._tf = TransformBroadcaster(self)
        self._tf_static = StaticTransformBroadcaster(self)
        # Queue of 1: only the most recent command matters.
        self.create_subscription(Twist, CMD_VEL_TOPIC, self._on_cmd_vel, 1)

        self._cmd = [0.0, 0.0]  # latest (linear, angular) command
        self._phase = _Phase.AWAIT_START
        self._idle_since: float | None = None  # sim time /cmd_vel went idle
        self._goal: list[float] | None = None  # current goal in the map frame
        self._last_goal_pub = 0.0  # sim time of the last goal publication
        self._prev_odom: tuple[list[float], float, float] | None = None  # pos, yaw, t

        self._publish_static_tf()
        self._start_new_mission()
        self.get_logger().info("burger_sim_bridge ready: offering the first goal")

    def _load_scene(self, scene_path: str) -> None:
        path = os.path.abspath(scene_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene file not found: {path}")
        if self._sim.getSimulationState() != self._sim.simulation_stopped:
            self._sim.stopSimulation()
            while self._sim.getSimulationState() != self._sim.simulation_stopped:
                time.sleep(0.05)
        self._sim.loadScene(path)
        # Real-time mode keeps simulation time close to wall-clock time, which
        # is what the wall-clock-driven agent assumes.
        self._sim.setBoolParam(self._sim.boolparam_realtime_simulation, True)
        self._sim.startSimulation()
        self.get_logger().info(f"Loaded scene '{path}' and started the simulation")

    def _call(self, function: str, *args):
        return self._sim.callScriptFunction(function, self._orchestrator, *args)

    def spin(self) -> None:
        """Run until interrupted; the simulation is stopped on exit."""
        try:
            while rclpy.ok():
                started = time.monotonic()
                rclpy.spin_once(self, timeout_sec=0.0)
                self._tick()
                # Sleep only up to the loop-rate cap: normally ros_step already
                # took longer (it blocks until the next simulation step).
                time.sleep(
                    max(0.0, self._control_period - (time.monotonic() - started))
                )
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self._sim.stopSimulation()
            except Exception:  # noqa: BLE001 - best effort during shutdown
                pass

    def _tick(self) -> None:
        if self._phase is _Phase.RUNNING:
            state = self._call("ros_step", self._cmd)
            self._publish_state(state)
            self._update_idle(state["sim_time"])
            if (
                self._idle_since is not None
                and state["sim_time"] - self._idle_since >= IDLE_RESET_S
            ):
                self._start_new_mission()
        else:  # AWAIT_START: parked, offering the goal.
            state = self._call("ros_step", [0.0, 0.0])
            self._publish_state(state)
            if state["sim_time"] - self._last_goal_pub >= AWAIT_REPUBLISH_S:
                self._publish_goal(state["sim_time"])
            if self._agent_moving():
                self._idle_since = None
                self._phase = _Phase.RUNNING
                self.get_logger().info("Agent started moving: episode running")

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd = [
            max(-LINEAR_VELOCITY_LIMIT, min(LINEAR_VELOCITY_LIMIT, msg.linear.x)),
            max(-ANGULAR_VELOCITY_LIMIT, min(ANGULAR_VELOCITY_LIMIT, msg.angular.z)),
        ]

    def _agent_moving(self) -> bool:
        return abs(self._cmd[0]) > CMD_MOVING_EPS or abs(self._cmd[1]) > CMD_MOVING_EPS

    def _update_idle(self, sim_time: float) -> None:
        """Remember since when the agent has not been commanding motion."""
        if self._agent_moving():
            self._idle_since = None
        elif self._idle_since is None:
            self._idle_since = sim_time

    def _start_new_mission(self) -> None:
        """Reposition the world and offer a fresh goal to the agent."""
        self._cmd = [0.0, 0.0]
        self._idle_since = None
        self._prev_odom = None
        self._odom_gate.reset()
        self._scan_gate.reset()
        state = self._call("ros_reset")
        self._goal = list(state["target_world_pos"])
        self._publish_state(state)
        self._publish_goal(state["sim_time"])
        self._phase = _Phase.AWAIT_START
        self.get_logger().info("New mission: world repositioned, awaiting the agent")

    @staticmethod
    def _stamp(sim_time: float):
        return Time(seconds=sim_time).to_msg()

    def _publish_state(self, state: dict) -> None:
        sim_time = state["sim_time"]
        stamp = self._stamp(sim_time)

        clock = Clock()
        clock.clock = stamp
        self._clock_pub.publish(clock)

        if self._odom_gate.due(sim_time):
            self._publish_odom(state, stamp)
            self._tf.sendTransform(
                [
                    self._transform(
                        MAP_FRAME,
                        ODOM_FRAME,
                        state["map_odom_pos"],
                        state["map_odom_quat"],
                        stamp,
                    ),
                    self._transform(
                        ODOM_FRAME,
                        BASE_FRAME,
                        state["odom_pos"],
                        state["odom_quat"],
                        stamp,
                    ),
                ]
            )
        if self._scan_gate.due(sim_time):
            self._publish_scan(state["lidar"], stamp)

    def _publish_odom(self, state: dict, stamp) -> None:
        pos, quat, sim_time = state["odom_pos"], state["odom_quat"], state["sim_time"]
        linear_x, angular_z = self._twist_from_motion(pos, quat, sim_time)
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = ODOM_FRAME
        msg.child_frame_id = BASE_FRAME
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = (
            pos
        )
        (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ) = quat
        msg.twist.twist.linear.x = linear_x
        msg.twist.twist.angular.z = angular_z
        self._odom_pub.publish(msg)

    def _publish_scan(self, ranges: list[float], stamp) -> None:
        n = len(ranges)
        increment = 2.0 * math.pi / n
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = SCAN_FRAME
        # Sector centres: the first sector is centred at -pi + increment / 2.
        msg.angle_min = -math.pi + increment / 2.0
        msg.angle_max = msg.angle_min + (n - 1) * increment
        msg.angle_increment = increment
        msg.scan_time = 1.0 / SCAN_HZ
        msg.range_min = 0.0
        msg.range_max = LIDAR_MAX_RANGE
        msg.ranges = [float(r) for r in ranges]
        msg.intensities = [0.0] * n
        self._scan_pub.publish(msg)

    def _publish_goal(self, sim_time: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self._stamp(sim_time)
        msg.header.frame_id = MAP_FRAME
        msg.pose.position.x, msg.pose.position.y = self._goal[0], self._goal[1]
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self._last_goal_pub = sim_time

    def _publish_static_tf(self) -> None:
        mount = self._call("get_static_transforms")
        stamp = self.get_clock().now().to_msg()
        self._tf_static.sendTransform(
            self._transform(
                BASE_FRAME,
                SCAN_FRAME,
                mount["base_scan_pos"],
                mount["base_scan_quat"],
                stamp,
            )
        )

    def _twist_from_motion(self, pos, quat, sim_time):
        """Estimate the odometry twist from consecutive poses (finite differences)."""
        yaw = math.atan2(
            2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
            1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
        )
        linear_x = angular_z = 0.0
        if self._prev_odom is not None:
            prev_pos, prev_yaw, prev_time = self._prev_odom
            dt = max(sim_time - prev_time, 1e-6)
            linear_x = math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) / dt
            dyaw = (yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi
            angular_z = dyaw / dt
        self._prev_odom = (pos, yaw, sim_time)
        return linear_x, angular_z

    @staticmethod
    def _transform(parent: str, child: str, pos, quat, stamp) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        (
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        ) = pos
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = quat
        return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoppeliaSim <-> ROS2 Burger bridge")
    parser.add_argument(
        "--sim-host", default="127.0.0.1", help="CoppeliaSim ZMQ remote API host"
    )
    parser.add_argument(
        "--sim-port", type=int, default=23000, help="CoppeliaSim ZMQ remote API port"
    )
    parser.add_argument(
        "--scene-path",
        default=None,
        help="Scene (.ttt) to load and start. "
        "Omit to use the scene already open in CoppeliaSim.",
    )
    parser.add_argument(
        "--scan-hz", type=float, default=SCAN_HZ, help="/scan rate (simulation time)"
    )
    parser.add_argument(
        "--odom-hz",
        type=float,
        default=ODOM_HZ,
        help="/odom and TF rate (simulation time)",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=CONTROL_HZ,
        help="Upper bound for the bridge loop rate",
    )
    # `ros2 run ... --ros-args ... -- <our flags>`: drop the ROS arguments first.
    return parser.parse_args(remove_ros_args(sys.argv)[1:])


def _raise_keyboard_interrupt(*_) -> None:
    raise KeyboardInterrupt


def main() -> None:
    args = parse_args()
    # Let `kill` (SIGTERM) stop the simulation cleanly, like Ctrl+C does.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    bridge = None
    try:
        bridge = BurgerSimBridge(
            sim_host=args.sim_host,
            sim_port=args.sim_port,
            scene_path=args.scene_path,
            scan_hz=args.scan_hz,
            odom_hz=args.odom_hz,
            control_hz=args.control_hz,
        )
        bridge.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
