# SPDX-License-Identifier: GPL-3.0-only

"""
Reward and task-level termination for the LunarLander example, computed on the
RL side.

Only the observation and the action that arrive over the socket are used. The
agent knows nothing about the task: it sends the observation, its timing and the
physics flags of Gymnasium (crash, out of bounds, time limit).

The reward is a rough shaping inspired by LunarLander's, enough to show the
decoupling; it does not reproduce Gymnasium's own.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

ObsType = np.ndarray | list[float] | tuple[float, ...] | dict[str, Any]


def _normalize_obs(
    obs: ObsType,
) -> np.ndarray:
    """
    Return the observation as a float32 vector of 8 values. A dict with an
    ``observation`` key is accepted too.
    """

    if isinstance(obs, dict):
        obs = obs.get("observation", obs)

    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape != (8,):
        raise ValueError(f"Expected observation shape (8,), got {arr.shape}")
    return arr


def compute_reward(
    obs: ObsType,
    action: int | None,
    prev_obs: ObsType | None = None,
    lat: float | None = None,
) -> float:
    """
    Reward for the last step.

    The observation is ``[x, y, vx, vy, angle, angular_vel, left_leg, right_leg]``.
    It is negative in proportion to the distance from the landing area, the speed and
    the tilt, positive for each leg in contact, and includes a small progress term
    that compares with ``prev_obs`` and a small cost for firing an engine.
    ``lat`` is accepted for timing-aware experiments and barely used.
    """

    cur = _normalize_obs(obs)
    prev = _normalize_obs(prev_obs) if prev_obs is not None else None

    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = [float(v) for v in cur]

    distance_penalty = 1.6 * math.sqrt(x * x + y * y)
    velocity_penalty = 1.1 * math.sqrt(vx * vx + vy * vy)
    attitude_penalty = 0.8 * abs(angle) + 0.3 * abs(ang_vel)
    leg_bonus = 0.25 * left_leg + 0.25 * right_leg

    reward = -(distance_penalty + velocity_penalty + attitude_penalty) + leg_bonus

    if prev is not None:
        px, py, pvx, pvy, pangle, pang_vel, _, _ = [float(v) for v in prev]
        prev_cost = (
            1.6 * math.sqrt(px * px + py * py)
            + 1.1 * math.sqrt(pvx * pvx + pvy * pvy)
            + 0.8 * abs(pangle)
            + 0.3 * abs(pang_vel)
        )
        cur_cost = distance_penalty + velocity_penalty + attitude_penalty
        reward += 0.5 * (prev_cost - cur_cost)

    if action is not None:
        # Discrete LunarLander: 0=no-op, 1=left engine, 2=main engine, 3=right engine.
        if int(action) == 2:
            reward -= 0.03
        elif int(action) in (1, 3):
            reward -= 0.01

    if lat is not None and lat <= 0.0:
        # Guard against a degenerate LAT from the caller.
        reward -= 0.001

    if is_terminated(obs):
        reward += 80.0

    return float(reward)


def is_terminated(
    obs: ObsType,
) -> bool:
    """
    Landing check, made from the observation.

    It holds when both legs touch, speed and tilt are low, and the lander is near the
    pad. The RL side combines it with the physics flag sent by the agent.
    """

    arr = _normalize_obs(obs)
    x, y, vx, vy, angle, _, left_leg, right_leg = [float(v) for v in arr]

    both_legs = left_leg > 0.5 and right_leg > 0.5
    low_speed = abs(vx) <= 0.25 and abs(vy) <= 0.25
    stable_attitude = abs(angle) <= 0.25
    near_ground = y <= 0.10 and abs(x) <= 0.30

    return bool(both_legs and low_speed and stable_attitude and near_ground)


def is_truncated(step_count: int, max_steps: int) -> bool:
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    return bool(step_count >= max_steps)
