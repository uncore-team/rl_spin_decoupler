# SPDX-License-Identifier: GPL-3.0-only

"""RL-side reward and task-termination logic for the decoupled CoppeliaSim
Burger demo.

This module runs on the RL side. It computes reward and the task-level goal
check from the 10-dim observation received over the transport. It stays aligned
with the base LunarLander example: reward lives on the RL side and the agent is
agnostic to the learning task.

Split of responsibilities for this robot demo:
- Collision is a PHYSICS event owned by the agent (detected from its LiDAR
  stream and transported in the payload as ``terminated``). ``compute_reward``
  still applies a collision penalty for shaping, but it does not own the
  termination decision for a collision.
- Reaching the goal is a TASK event derived here from the observation
  (``is_goal_reached``).
- Truncation by step budget is an RL-side concern (``is_truncated``).

Shared physical thresholds and observation dimensions live in
``episode_config.py`` so the agent's collision check and this module cannot
drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from episode_config import (
    COLLISION_LIDAR_THRESHOLD,
    GOAL_DIST_THRESHOLD,
    NUM_LIDAR_SECTORS,
    OBS_DIM,
)

# Defined once for readability and reuse.
ObsType = np.ndarray | list[float] | tuple[float, ...] | dict[str, Any]


def _normalize_obs(obs: ObsType) -> np.ndarray:
    """Normalize accepted observation formats into a 1D float32 array.

    Accepted forms:
    - Raw vector-like obs with ``OBS_DIM`` elements
      (``NUM_LIDAR_SECTORS`` LiDAR + distance + angle).
    - Dict payload with key ``observation`` containing that vector.
    """
    if isinstance(obs, dict):
        obs = obs.get("observation", obs)

    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape != (OBS_DIM,):
        raise ValueError(f"Expected observation shape ({OBS_DIM},), got {arr.shape}")
    return arr


def compute_reward(
    obs: ObsType,
    action: ObsType,
    prev_obs: ObsType | None = None,
    lat: float | None = None,
) -> float:
    """Compute the RL-side reward from a 10-dim observation.

    Observation convention:
    [NUM_LIDAR_SECTORS LiDAR readings (min per sector), distance_to_target,
     relative_angle]

    Reward terms:
    - Strong penalty/bonus for collision or reaching the goal.
    - Bonus for scalar progress toward the goal (requires prev_obs).
    - Penalties for misalignment, action effort, and time.

    ``lat`` is accepted for compatibility with timing-aware experiments (same as
    in the LunarLander example) but does not currently participate in the
    computation.
    """
    cur = _normalize_obs(obs)
    lidar_min = float(np.min(cur[0:NUM_LIDAR_SECTORS]))
    dist = float(cur[NUM_LIDAR_SECTORS])
    angle = float(cur[NUM_LIDAR_SECTORS + 1])

    # 1. Strong penalty/bonus for collision or success.
    if lidar_min <= COLLISION_LIDAR_THRESHOLD:
        return -100.0
    if dist <= GOAL_DIST_THRESHOLD:
        return 200.0

    # 2. Bonus for scalar progress toward the goal.
    reward = 0.0
    if prev_obs is not None:
        prev = _normalize_obs(prev_obs)
        prev_dist = float(prev[NUM_LIDAR_SECTORS])
        progress = prev_dist - dist
        reward += progress * 50.0

    # 3. Penalties for misalignment, action effort and elapsed time.
    action_arr = np.asarray(action, dtype=np.float32)
    reward -= 0.05 * abs(angle)
    reward -= 0.01 * float(np.linalg.norm(action_arr))
    reward -= 0.1  # per-step penalty

    return float(reward)


def is_goal_reached(obs: ObsType) -> bool:
    """Return whether the task goal (reaching the target) is satisfied.

    Collision termination is NOT decided here: it is a physics event reported by
    the agent in the payload. The RL side ORs this task check with that flag.
    """
    cur = _normalize_obs(obs)
    dist = float(cur[NUM_LIDAR_SECTORS])
    return bool(dist <= GOAL_DIST_THRESHOLD)


def is_truncated(step_count: int, max_steps: int) -> bool:
    """Return whether the episode is truncated by the RL-side step budget."""
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    return bool(step_count >= max_steps)
