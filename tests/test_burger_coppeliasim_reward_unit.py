# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for the CoppeliaSim Burger RL-side reward contract."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_module(module_name: str, filename: str):
    examples_dir = Path(__file__).resolve().parents[1] / "examples" / "burger_coppeliasim"
    sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location(
        module_name, examples_dir / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Burger example module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward = _load_module("burger_reward", "reward.py")


def _observation(
    *, lidar_min: float = 1.0, distance: float = 1.0, angle: float = 0.0
) -> list[float]:
    return [lidar_min] + [1.0] * 7 + [distance, angle]


def test_compute_reward_accepts_dict_payload_and_numpy_action():
    observation = _observation(distance=0.8)
    payload = {"observation": observation}

    reward_from_list = reward.compute_reward(
        observation, np.array([0.1, 0.0]), prev_obs=observation
    )
    reward_from_payload = reward.compute_reward(
        payload, np.array([0.1, 0.0]), prev_obs=payload
    )

    assert reward_from_payload == reward_from_list


def test_compute_reward_bonuses_progress_toward_goal():
    previous = _observation(distance=0.8)
    current = _observation(distance=0.6)

    assert reward.compute_reward(current, [0.0, 0.0], previous) > -0.1


def test_compute_reward_penalizes_collision():
    collision = _observation(lidar_min=reward.COLLISION_LIDAR_THRESHOLD - 0.01)
    clear = _observation(lidar_min=reward.COLLISION_LIDAR_THRESHOLD + 0.01)

    assert reward.compute_reward(collision, [0.0, 0.0]) < reward.compute_reward(
        clear, [0.0, 0.0]
    )


def test_compute_reward_rewards_goal_reached():
    goal = _observation(distance=reward.GOAL_DIST_THRESHOLD - 0.01)

    assert reward.compute_reward(goal, [0.0, 0.0]) == 200.0


def test_goal_and_truncation_boundaries():
    assert reward.is_goal_reached(
        _observation(distance=reward.GOAL_DIST_THRESHOLD - 0.01)
    )
    assert not reward.is_goal_reached(
        _observation(distance=reward.GOAL_DIST_THRESHOLD + 0.01)
    )
    assert not reward.is_truncated(step_count=9, max_steps=10)
    assert reward.is_truncated(step_count=10, max_steps=10)


def test_reward_rejects_wrong_observation_shape():
    with pytest.raises(ValueError, match=r"Expected observation shape \(10,\)"):
        reward.compute_reward([0.0, 1.0], [0.0, 0.0])


def test_truncation_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="max_steps"):
        reward.is_truncated(step_count=1, max_steps=0)
