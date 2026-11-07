# SPDX-License-Identifier: GPL-3.0-only

"""
Reward and task-level termination for the Burger example, computed on the RL side.

They are derived from the 10-value observation that arrives over the socket. The
agent reports a collision itself, in the payload, because it is the one that sees
it; the reward still penalizes a collision, but the decision to end the episode
comes from the agent's flag. Reaching the goal (``is_goal_reached``) and the step
budget (``is_truncated``) are decided here.

The thresholds come from ``episode_config.py``, shared with the agent so that the
two cannot drift apart.
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

ObsType = np.ndarray | list[float] | tuple[float, ...] | dict[str, Any]


def _normalize_obs(obs: ObsType) -> np.ndarray:
    """Normalize an observation into a 1D float32 array of ``OBS_DIM`` values.

    Accepts a vector-like observation (``[distance, angle, ray_1..ray_8]``) or a
    dict payload whose ``observation`` key holds that vector.
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
    """
    Reward for the last step, from the observation
    ``[distance_to_goal, relative_goal_angle, ray_1 ... ray_8]``.

    A collision gives -100 and reaching the goal +200. Otherwise the reward is 50 times
    the progress toward the goal since ``prev_obs``, minus 0.05 times the angle error,
    0.01 times the norm of the action and 0.1 per step. ``lat`` is accepted for
    timing-aware experiments and not used.
    """
    cur = _normalize_obs(obs)
    lidar_min = float(np.min(cur[2 : 2 + NUM_LIDAR_SECTORS]))
    dist = float(cur[0])
    angle = float(cur[1])

    if lidar_min <= COLLISION_LIDAR_THRESHOLD:
        return -100.0
    if dist <= GOAL_DIST_THRESHOLD:
        return 200.0

    reward = 0.0
    if prev_obs is not None:
        prev = _normalize_obs(prev_obs)
        prev_dist = float(prev[0])
        progress = prev_dist - dist
        reward += progress * 50.0

    action_arr = np.asarray(action, dtype=np.float32)
    reward -= 0.05 * abs(angle)
    reward -= 0.01 * float(np.linalg.norm(action_arr))
    reward -= 0.1  # per-step penalty

    return float(reward)


def is_goal_reached(obs: ObsType) -> bool:
    """
    Whether the robot is within the goal distance.

    A collision is not decided here: the agent reports it in the payload.
    """
    cur = _normalize_obs(obs)
    dist = float(cur[0])
    return bool(dist <= GOAL_DIST_THRESHOLD)


def is_truncated(step_count: int, max_steps: int) -> bool:
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    return bool(step_count >= max_steps)
