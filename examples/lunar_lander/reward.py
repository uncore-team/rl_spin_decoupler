# SPDX-License-Identifier: GPL-3.0-only

"""RL-side reward and episode logic for the decoupled LunarLander example.

This module intentionally computes reward and done conditions on the RL side,
using only observation/action data coming from the decoupled transport layer.

Important:
- This reward is a didactic approximation inspired by LunarLander shaping.
- It is sufficient to illustrate decoupling (agent does not send reward), but
  it does not attempt to exactly replicate Gymnasium's internal reward.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

# Defined once for readability and reuse.
#
# Observations are transported by the decoupler as plain, picklable Python
# objects (see ``agent_side_lunarlander.py``, which sends ``obs.tolist()``).
# This module therefore accepts any length-8 numeric sequence (list, tuple, or
# NumPy array) or a dict payload carrying that vector under ``observation``. No
# third-party package is required to evaluate the reward.
ObsType = Union[Sequence[float], dict]

_OBS_SIZE = 8


def _normalize_obs(
    obs: ObsType,
) -> list[float]:
    """Normalize accepted observation formats into a list of 8 floats.

    Accepted forms:
    - Raw vector-like obs with 8 numeric elements (list, tuple, NumPy array...).
    - Dict payload with key ``observation`` containing that vector.
    """

    if isinstance(obs, dict):
        obs = obs.get("observation", obs)

    try:
        values = [float(v) for v in obs]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Observation is not a numeric vector: {obs!r}") from exc

    if len(values) != _OBS_SIZE:
        raise ValueError(
            f"Expected observation of length {_OBS_SIZE}, got {len(values)}"
        )
    return values


def compute_reward(
    obs: ObsType,
    action: int | None,
    prev_obs: ObsType | None = None,
    lat: float | None = None,
) -> float:
    """Compute a didactic RL-side reward from LunarLander observations.

    Observation convention:
    [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]

    Reward terms (approximate shaping):
    - Penalize distance from landing area (x, y -> 0).
    - Penalize translational velocity (vx, vy).
    - Penalize tilt and angular velocity.
    - Bonus for leg contacts.
    - Optional small fuel-like action penalty.

    The reward can use ``prev_obs`` to build a progress term. ``lat`` is accepted
    for compatibility with timing-aware experiments but is intentionally optional.
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
        # Keep numeric behavior robust if a caller passes degenerate LAT.
        reward -= 0.001

    if is_terminated(obs):
        reward += 80.0

    return float(reward)


def is_terminated(
    obs: ObsType,
) -> bool:
    """Return termination condition decided on RL side from observation data.

    Termination policy used in this example:
    - Terminate when a stable landing signature is observed: both legs in
      contact, low speed, and small tilt near ground.
    """

    arr = _normalize_obs(obs)
    x, y, vx, vy, angle, _, left_leg, right_leg = [float(v) for v in arr]

    both_legs = left_leg > 0.5 and right_leg > 0.5
    low_speed = abs(vx) <= 0.25 and abs(vy) <= 0.25
    stable_attitude = abs(angle) <= 0.25
    near_ground = y <= 0.10 and abs(x) <= 0.30

    return bool(both_legs and low_speed and stable_attitude and near_ground)


def is_truncated(step_count: int, max_steps: int) -> bool:
    """Return whether episode is truncated by RL-side step budget."""

    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    return bool(step_count >= max_steps)
