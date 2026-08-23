# SPDX-License-Identifier: GPL-3.0-only

"""Agent-side reward and episode-termination logic for the decoupled
CoppeliaSim Burger demo.

Structure mirrors examples/lunar_lander/reward.py from the rl_spin_decoupler
repo, with one deliberate difference: in that example this module is
imported by the RL side (the agent stays "reward-agnostic"), while here it
is imported by agent_side_burger_coppeliasim.py. That is an explicit design
decision for this project (termination is decided in agent-side), not an
oversight or an accidental deviation from the reference example.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Defined once for readability and reuse.
ObsType = np.ndarray | list[float] | tuple[float, ...] | dict[str, Any]

# Episode thresholds, shared between compute_reward() and is_terminated().
COLLISION_LIDAR_THRESHOLD = 0.14
GOAL_DIST_THRESHOLD = 0.10


def _normalize_obs(obs: ObsType) -> np.ndarray:
    """Normalize accepted observation formats into a 1D float32 array of size 10.

    Accepted forms:
    - Raw vector-like obs with 10 elements (8 LiDAR + distance + angle).
    - Dict payload with key ``observation`` containing that vector.
    """
    if isinstance(obs, dict):
        obs = obs.get("observation", obs)

    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape != (10,):
        raise ValueError(f"Expected observation shape (10,), got {arr.shape}")
    return arr


def compute_reward(
    obs: ObsType,
    action: ObsType,
    prev_obs: ObsType | None = None,
    lat: float | None = None,
) -> float:
    """Compute the agent-side reward from a 10-dim observation.

    Observation convention:
    [8 LiDAR readings (min per sector), distance_to_target, relative_angle]

    Reward terms:
    - Strong penalty/bonus for collision or reaching the goal.
    - Bonus for scalar progress toward the goal (requires prev_obs).
    - Penalties for misalignment, action effort, and time.

    ``lat`` is accepted for compatibility with timing-aware experiments
    (same as in the LunarLander example) but does not currently participate
    in the computation.
    """
    cur = _normalize_obs(obs)
    lidar_min = float(np.min(cur[0:8]))
    dist = float(cur[8])
    angle = float(cur[9])

    # 1. Strong penalty/bonus for collision or success.
    if lidar_min <= COLLISION_LIDAR_THRESHOLD:
        return -100.0
    if dist <= GOAL_DIST_THRESHOLD:
        return 200.0

    # 2. Bonus for scalar progress toward the goal.
    reward = 0.0
    if prev_obs is not None:
        prev = _normalize_obs(prev_obs)
        prev_dist = float(prev[8])
        progress = prev_dist - dist
        reward += progress * 50.0

    # 3. Penalties for misalignment, action effort and elapsed time.
    action_arr = np.asarray(action, dtype=np.float32)
    reward -= 0.05 * abs(angle)
    reward -= 0.01 * float(np.linalg.norm(action_arr))
    reward -= 0.1  # per-step penalty

    return float(reward)


def is_terminated(obs: ObsType) -> bool:
    """Return whether the episode ended due to a collision or reaching the goal."""
    cur = _normalize_obs(obs)
    lidar_min = float(np.min(cur[0:8]))
    dist = float(cur[8])
    return bool(lidar_min <= COLLISION_LIDAR_THRESHOLD or dist <= GOAL_DIST_THRESHOLD)


def is_truncated(step_count: int, max_steps: int) -> bool:
    """Return whether the episode is truncated by the step budget."""
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    return bool(step_count >= max_steps)
