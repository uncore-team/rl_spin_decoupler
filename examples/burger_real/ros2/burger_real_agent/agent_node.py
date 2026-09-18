"""ROS2 agent node for the real TurtleBot3 Burger example."""

from __future__ import annotations

import argparse
import math
import time
from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from spindecoupler import AgentSide, BaseCommPoint

from .contract import (
    ANGULAR_VELOCITY_LIMIT,
    CMD_VEL_TOPIC,
    COLLISION_LIDAR_THRESHOLD,
    GOAL_FRAME,
    GOAL_TOPIC,
    LIDAR_MAX_RANGE,
    LINEAR_VELOCITY_LIMIT,
    NUM_LIDAR_SECTORS,
    OBS_DIM,
    ODOM_FRAME,
    ODOM_TOPIC,
    SCAN_TOPIC,
    normalize_angle,
)


class StepState(Enum):
    READY_FOR_RL_COMMAND = 0
    EXECUTING_LAST_ACTION = 1
    AFTER_RESET = 2


class BurgerRealAgent(Node):
    """Bridge ROS2 sensors/commands and the decoupler agent-side protocol."""

    def __init__(
        self,
        ip: str,
        port: int,
        rl_step_period: float,
        control_period: float,
        timeout: float,
        sensor_timeout: float,
        debug: bool,
    ):
        super().__init__("burger_real_agent")
        if rl_step_period <= control_period:
            raise ValueError(
                "rl_step_period must be strictly greater than control_period"
            )

        self._rl_step_period = rl_step_period
        self._control_period = control_period
        self._timeout = timeout
        self._sensor_timeout = sensor_timeout
        self._debug = debug
        self._comm = AgentSide(ip, port, verbose=debug)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._scan_callback, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, 10)
        self.create_subscription(PoseStamped, GOAL_TOPIC, self._goal_callback, 10)

        self._scan: LaserScan | None = None
        self._odom: Odometry | None = None
        self._goal: PoseStamped | None = None
        self._sensor_sequence = 0
        self._last_sensor_time = 0.0
        self._state = StepState.READY_FOR_RL_COMMAND
        self._last_action = [0.0, 0.0]
        self._last_action_start = time.monotonic()
        self._obs = np.zeros((OBS_DIM,), dtype=np.float32)
        self._term_latched = False
        self._trunc_latched = False
        self._frozen = False

        self._wait_for_sensor_data()
        self._refresh_observation()

    def _scan_callback(self, message: LaserScan) -> None:
        self._scan = message
        self._mark_sensor_update()

    def _odom_callback(self, message: Odometry) -> None:
        self._odom = message
        self._mark_sensor_update()

    def _goal_callback(self, message: PoseStamped) -> None:
        self._goal = message
        self._mark_sensor_update()

    def _mark_sensor_update(self) -> None:
        self._sensor_sequence += 1
        self._last_sensor_time = time.monotonic()

    def _wait_for_sensor_data(self, previous_sequence: int = -1) -> None:
        while rclpy.ok():
            self._spin_once()
            if (
                self._scan is not None
                and self._odom is not None
                and self._goal is not None
                and self._sensor_sequence > previous_sequence
            ):
                self._refresh_observation()
                return
        raise RuntimeError("ROS2 shutdown while waiting for Burger sensor data")

    def _spin_once(self) -> None:
        rclpy.spin_once(self, timeout_sec=0.0)

    def _refresh_observation(self) -> None:
        if self._scan is None or self._odom is None or self._goal is None:
            raise RuntimeError("Burger observation requires scan, odometry and goal")
        goal_x, goal_y = self._goal_in_odom()
        position = self._odom.pose.pose.position
        orientation = self._odom.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        dx = goal_x - position.x
        dy = goal_y - position.y
        distance = math.hypot(dx, dy)
        relative_angle = normalize_angle(math.atan2(dy, dx) - yaw)
        sectors = self._scan_sectors(self._scan)
        self._obs = np.asarray(sectors + [distance, relative_angle], dtype=np.float32)

    def _goal_in_odom(self) -> tuple[float, float]:
        assert self._goal is not None
        goal = self._goal.pose.position
        if self._goal.header.frame_id in ("", ODOM_FRAME):
            return float(goal.x), float(goal.y)
        if self._goal.header.frame_id != GOAL_FRAME:
            raise RuntimeError(
                f"Unsupported goal frame '{self._goal.header.frame_id}'; "
                f"expected '{GOAL_FRAME}' or '{ODOM_FRAME}'"
            )
        try:
            transform = self._tf_buffer.lookup_transform(
                ODOM_FRAME, GOAL_FRAME, rclpy.time.Time()
            )
        except TransformException as exc:
            raise RuntimeError("TF map -> odom is unavailable") from exc
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        transform_yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        transformed_x = (
            translation.x
            + math.cos(transform_yaw) * goal.x
            - math.sin(transform_yaw) * goal.y
        )
        transformed_y = (
            translation.y
            + math.sin(transform_yaw) * goal.x
            + math.cos(transform_yaw) * goal.y
        )
        return float(transformed_x), float(transformed_y)

    @staticmethod
    def _scan_sectors(scan: LaserScan) -> list[float]:
        values = np.asarray(scan.ranges, dtype=np.float32)
        values[~np.isfinite(values)] = LIDAR_MAX_RANGE
        values = np.clip(values, 0.0, LIDAR_MAX_RANGE)
        sectors: list[float] = []
        sector_width = 2.0 * math.pi / NUM_LIDAR_SECTORS
        for sector_index in range(NUM_LIDAR_SECTORS):
            start = -math.pi + sector_index * sector_width
            end = start + sector_width
            angles = scan.angle_min + np.arange(values.size) * scan.angle_increment
            selected = values[(angles >= start) & (angles < end)]
            sectors.append(
                float(np.min(selected)) if selected.size else LIDAR_MAX_RANGE
            )
        return sectors

    def _build_payload(self) -> dict[str, object]:
        return {
            "observation": self._obs.tolist(),
            "terminated": bool(self._term_latched),
            "truncated": bool(self._trunc_latched),
        }

    def _publish_action(self) -> None:
        message = Twist()
        message.linear.x = float(
            np.clip(self._last_action[0], -LINEAR_VELOCITY_LIMIT, LINEAR_VELOCITY_LIMIT)
        )
        message.angular.z = float(
            np.clip(
                self._last_action[1], -ANGULAR_VELOCITY_LIMIT, ANGULAR_VELOCITY_LIMIT
            )
        )
        self._cmd_pub.publish(message)

    def _publish_stop(self) -> None:
        self._last_action = [0.0, 0.0]
        self._publish_action()

    def _run_control_tick(self) -> None:
        self._spin_once()
        sensor_age = time.monotonic() - self._last_sensor_time
        if self._last_sensor_time <= 0.0 or sensor_age > self._sensor_timeout:
            self._trunc_latched = True
            self._frozen = True
            self._publish_stop()
            return
        self._refresh_observation()
        if float(np.min(self._obs[:NUM_LIDAR_SECTORS])) <= COLLISION_LIDAR_THRESHOLD:
            self._term_latched = True
            self._frozen = True
            self._publish_stop()
            return
        if not self._frozen:
            self._publish_action()

    def spinloop(self) -> None:
        self.get_logger().info("Running real Burger ROS2 agent")
        try:
            while rclpy.ok():
                now = time.monotonic()
                self._run_control_tick()
                if self._state == StepState.EXECUTING_LAST_ACTION:
                    if now - self._last_action_start >= self._rl_step_period:
                        self._comm.stepSendObs(
                            self._build_payload(), agenttime=time.time()
                        )
                        self._state = StepState.READY_FOR_RL_COMMAND
                elif self._state == StepState.AFTER_RESET:
                    self._comm.resetSendObs(
                        self._build_payload(), agenttime=time.time()
                    )
                    self._state = StepState.READY_FOR_RL_COMMAND
                else:
                    command = self._comm.readWhatToDo(timeout=self._timeout)
                    if command is not None:
                        what, payload = command
                        if what == AgentSide.WhatToDo.RESET_SEND_OBS:
                            sequence = self._sensor_sequence
                            self._term_latched = False
                            self._trunc_latched = False
                            self._frozen = False
                            self._publish_stop()
                            self._wait_for_sensor_data(sequence)
                            self._last_action_start = time.monotonic()
                            self._state = StepState.AFTER_RESET
                        elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                            self._comm.stepSendLastActDur(
                                time.monotonic() - self._last_action_start
                            )
                            self._last_action_start = time.monotonic()
                            self._last_action = [float(payload[0]), float(payload[1])]
                            self._state = StepState.EXECUTING_LAST_ACTION
                        elif what == AgentSide.WhatToDo.FINISH:
                            return
                time.sleep(self._control_period)
        finally:
            self._publish_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 real Burger agent")
    parser.add_argument("--ip", default=BaseCommPoint.get_ip())
    parser.add_argument("--port", type=int, default=49054)
    parser.add_argument("--rl-step-period", type=float, default=0.08)
    parser.add_argument("--control-period", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--sensor-timeout", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    agent = None
    try:
        agent = BurgerRealAgent(
            ip=args.ip,
            port=args.port,
            rl_step_period=args.rl_step_period,
            control_period=args.control_period,
            timeout=args.timeout,
            sensor_timeout=args.sensor_timeout,
            debug=args.debug,
        )
        agent.spinloop()
    finally:
        if agent is not None:
            agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
