# SPDX-License-Identifier: GPL-3.0-only

"""RL-side reward and task-termination logic for the first-order plant example.

Reward and task-level termination are computed on the RL side from the
observation received over the decoupler transport. The agent stays agnostic to
the learning task: it only transports observations, timing, and any
physics/hardware termination flags (this synthetic plant has none, so those
flags are always ``False``).
"""

from __future__ import annotations

from typing import Any

ObsType = dict[str, Any]
ActionType = dict[str, float]


def _plant_state(obs: ObsType) -> float:
    """Extract the plant state from an observation payload."""

    return float(obs.get("plant_state", 0.0))


def compute_reward(
    obs: ObsType,
    action: ActionType | None,
    prev_obs: ObsType | None = None,
    lat: float | None = None,
) -> float:
    """Compute reward from the current first-order plant observation.

    ``action``, ``prev_obs``, and ``lat`` match the LunarLander reward API so
    the RL-side reward modules share a consistent call contract.
    """

    _ = action
    _ = prev_obs
    _ = lat
    return -abs(_plant_state(obs))


def is_terminated(obs: ObsType, tolerance: float = 0.05) -> bool:
    """Return whether the plant has reached the target tolerance."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    return abs(_plant_state(obs)) <= tolerance
