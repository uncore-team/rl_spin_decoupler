# SPDX-License-Identifier: GPL-3.0-only

"""Constants shared by both sides of the CoppeliaSim Burger example.

The agent side and the RL side must agree on these values, so they live in one
neutral module (no reward or RL logic) imported by:

- ``agent_side_burger_coppeliasim.py``: the collision threshold (to detect a
  physics collision locally and report it in the payload) and the observation
  size.
- ``reward.py`` (RL side): the thresholds, for reward shaping and the goal check.
- ``rl_side_burger_coppeliasim.py``: the bounds of the Gymnasium observation and
  action spaces.

Keeping them here prevents the agent's collision check and the RL-side reward
from silently drifting apart.

The values follow the real-robot contract of ``examples/burger_real`` (a test
checks it), so a policy trained here runs unchanged on the real robot and on the
simulated stand-in of ``examples/burger_real_sim``.
"""

from __future__ import annotations

# Observation layout: [distance_to_goal, relative_goal_angle, ray_1 ... ray_8].
NUM_LIDAR_SECTORS = 8
OBS_DIM = NUM_LIDAR_SECTORS + 2

# Maximum LiDAR range in meters: each sector reports the nearest return, clamped to
# this value. It has to equal ``config["max_scan_distance"]`` in the scene script
# ``examples/_common/burger_scene/laser.py``, which runs inside CoppeliaSim and
# cannot import this module; a test compares the two.
LIDAR_MAX_RANGE = 8.0

# A LiDAR reading of this many meters or less is a collision. The agent detects it
# and sends it in the payload as ``terminated``.
COLLISION_LIDAR_THRESHOLD = 0.16

# The goal is reached at this many meters from the target or less. The RL side
# decides it from the observation.
GOAL_DIST_THRESHOLD = 0.10

# Bounds of the observation and action spaces.
MAX_GOAL_DISTANCE = 3.0  # m; the arena diagonal is about 2.7 m
MAX_LINEAR_VELOCITY = 0.15  # m/s (forward only)
MAX_ANGULAR_VELOCITY = 0.6  # rad/s, in both directions
