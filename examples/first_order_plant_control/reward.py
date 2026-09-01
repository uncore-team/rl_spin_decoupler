# SPDX-License-Identifier: GPL-3.0-only

"""Agent-side reward and episode logic for the first-order plant example."""

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
    agent-side reward modules have a consistent call contract.
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