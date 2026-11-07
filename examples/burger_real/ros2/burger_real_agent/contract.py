"""Shared Burger real-robot observation and topic contract."""

from __future__ import annotations

import math

NUM_LIDAR_SECTORS = 8
OBS_DIM = NUM_LIDAR_SECTORS + 2

# Contract used by training_8rays_fixedtime.zip: distance, angle, then rays.
LIDAR_MAX_RANGE = 8.0
COLLISION_LIDAR_THRESHOLD = 0.16
GOAL_DIST_THRESHOLD = 0.10

SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
GOAL_TOPIC = "/goal_pose"
CMD_VEL_TOPIC = "/cmd_vel"
ODOM_FRAME = "odom"
GOAL_FRAME = "map"

LINEAR_VELOCITY_LIMIT = 0.15
ANGULAR_VELOCITY_LIMIT = 0.6


def normalize_angle(angle: float) -> float:
    """Return an angle in [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
