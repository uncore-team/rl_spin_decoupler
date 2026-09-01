# SPDX-License-Identifier: GPL-3.0-only

"""Shared episode constants for the decoupled CoppeliaSim Burger example.

These values define the physical/task contract that BOTH sides of the decoupler
must agree on, so they live in a single neutral module (no reward or RL logic
here) imported by:

- agent_side_burger_coppeliasim.py: uses ``COLLISION_LIDAR_THRESHOLD`` to detect
  a physics collision locally and report it in the payload, and the observation
  dimensions to size its observation buffer.
- reward.py (RL side): uses both thresholds for reward shaping and for the
  task-level goal check.
- rl_side_burger_coppeliasim.py: uses the observation dimensions and
  ``LIDAR_MAX_RANGE`` to build the Gymnasium observation space.

Keeping them here prevents the agent's collision check and the RL-side reward
from silently drifting apart.
"""

from __future__ import annotations

# Observation layout: NUM_LIDAR_SECTORS LiDAR readings + distance + angle.
NUM_LIDAR_SECTORS = 8
OBS_DIM = NUM_LIDAR_SECTORS + 2  # 8 LiDAR + distance + relative angle -> 10

# Maximum LiDAR range in METERS. The scene's laser proxy reports each sector as
# the nearest return distance in meters, clamped to this value, so it is also
# the upper bound of the LiDAR entries in the observation space.
#
# IMPORTANT: this MUST equal ``config["max_scan_distance"]`` in the scene child
# script ``scene/laser.py``. That script runs inside CoppeliaSim and cannot
# import this module, so the two values are kept in sync by hand. If you change
# the LiDAR range, change it in BOTH places.
LIDAR_MAX_RANGE = 1.0

# LiDAR distance in METERS at/under which the robot is considered to have
# collided (with LIDAR_MAX_RANGE == 1.0 this is numerically in [0, 1], but the
# quantity is metric, not a normalized fraction). This is a PHYSICS event:
# detected agent-side and transported in the payload as ``terminated``.
COLLISION_LIDAR_THRESHOLD = 0.14

# Metric distance (m) to the target at/under which the goal is reached. This is
# a TASK event: derived RL-side from the observation.
GOAL_DIST_THRESHOLD = 0.10
