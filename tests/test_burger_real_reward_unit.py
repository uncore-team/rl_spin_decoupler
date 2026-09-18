# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for the real Burger reward contract."""

import importlib.util
import sys
from pathlib import Path

import pytest

CONTRACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "burger_real"
    / "ros2"
    / "burger_real_agent"
)
sys.path.insert(0, str(CONTRACT_DIR))

spec = importlib.util.spec_from_file_location(
    "burger_real_reward", CONTRACT_DIR.parents[1] / "reward.py"
)
assert spec is not None and spec.loader is not None
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


def observation(*, lidar_min: float = 1.0, distance: float = 1.0) -> list[float]:
    return [lidar_min] + [1.0] * 7 + [distance, 0.0]


def test_collision_and_goal_boundaries():
    assert reward.compute_reward(observation(lidar_min=0.10), [0.0, 0.0]) == -100.0
    assert reward.compute_reward(observation(distance=0.05), [0.0, 0.0]) == 200.0
    assert reward.is_goal_reached(observation(distance=0.05))
    assert not reward.is_goal_reached(observation(distance=0.2))


def test_progress_is_rewarded_and_budget_is_checked():
    assert (
        reward.compute_reward(
            observation(distance=0.6), [0.0, 0.0], observation(distance=0.8)
        )
        > -0.1
    )
    assert reward.is_truncated(10, 10)
    assert not reward.is_truncated(9, 10)


def test_invalid_observation_and_budget_are_rejected():
    with pytest.raises(ValueError, match=r"Expected observation shape \(10,\)"):
        reward.compute_reward([0.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="max_steps"):
        reward.is_truncated(1, 0)
