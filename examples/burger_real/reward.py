"""RL-side reward and termination logic for the real Burger example."""

from __future__ import annotations

from typing import Any

import numpy as np
from contract import (
    COLLISION_LIDAR_THRESHOLD,
    GOAL_DIST_THRESHOLD,
    NUM_LIDAR_SECTORS,
    OBS_DIM,
)

ObsType = np.ndarray | list[float] | tuple[float, ...] | dict[str, Any]


def _normalize_obs(obs: ObsType) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs.get("observation", obs)
    array = np.asarray(obs, dtype=np.float32)
    if array.shape != (OBS_DIM,):
        raise ValueError(f"Expected observation shape ({OBS_DIM},), got {array.shape}")
    return array


def compute_reward(
    obs: ObsType,
    action: ObsType,
    prev_obs: ObsType | None = None,
    lat: float | None = None,
) -> float:
    current = _normalize_obs(obs)
    lidar_min = float(np.min(current[:NUM_LIDAR_SECTORS]))
    distance = float(current[NUM_LIDAR_SECTORS])
    angle = float(current[NUM_LIDAR_SECTORS + 1])

    if lidar_min <= COLLISION_LIDAR_THRESHOLD:
        return -100.0
    if distance <= GOAL_DIST_THRESHOLD:
        return 200.0

    reward = 0.0
    if prev_obs is not None:
        previous = _normalize_obs(prev_obs)
        reward += float(previous[NUM_LIDAR_SECTORS] - distance) * 50.0

    reward -= 0.05 * abs(angle)
    reward -= 0.01 * float(np.linalg.norm(np.asarray(action, dtype=np.float32)))
    reward -= 0.1
    return float(reward)


def is_goal_reached(obs: ObsType) -> bool:
    current = _normalize_obs(obs)
    return bool(current[NUM_LIDAR_SECTORS] <= GOAL_DIST_THRESHOLD)


def is_truncated(step_count: int, max_steps: int) -> bool:
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    return bool(step_count >= max_steps)
