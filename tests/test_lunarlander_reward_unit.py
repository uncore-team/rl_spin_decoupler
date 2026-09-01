# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for the agent-side LunarLander reward logic.

Observations are built here as plain Python lists of floats, which is exactly
the payload structure the agent transports over the decoupler
(see ``examples/lunar_lander/agent_side_lunarlander.py``, which sends
``obs.tolist()``). Keeping these tests dependency-free means the agent-side reward
logic can be exercised without installing NumPy, matching the "pure-stdlib core"
philosophy of the library.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_reward_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "lunar_lander" / "reward.py"
    spec = importlib.util.spec_from_file_location("lunarlander_reward", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load lunar_lander reward module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward = _load_reward_module()

# Observation convention:
# [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]


def test_compute_reward_prefers_better_observation_progress():
    prev_obs = [0.6, 0.7, 0.8, -0.7, 0.6, 0.5, 0.0, 0.0]
    obs_bad = [0.7, 0.8, 0.9, -0.8, 0.7, 0.6, 0.0, 0.0]
    obs_good = [0.2, 0.2, 0.1, -0.1, 0.05, 0.02, 1.0, 1.0]

    r_bad = reward.compute_reward(obs_bad, action=0, prev_obs=prev_obs, lat=0.05)
    r_good = reward.compute_reward(obs_good, action=0, prev_obs=prev_obs, lat=0.05)

    assert r_good > r_bad


def test_compute_reward_penalizes_main_engine_use():
    obs = [0.2, 0.2, 0.1, -0.1, 0.05, 0.02, 0.0, 0.0]

    r_no_engine = reward.compute_reward(obs, action=0, prev_obs=obs)
    r_main_engine = reward.compute_reward(obs, action=2, prev_obs=obs)

    assert r_main_engine < r_no_engine


def test_compute_reward_accepts_dict_payload():
    # The reward must also accept the dict-wrapped transport form.
    obs_list = [0.2, 0.2, 0.1, -0.1, 0.05, 0.02, 0.0, 0.0]
    obs_dict = {"observation": obs_list}

    assert reward.compute_reward(
        obs_dict, action=0, prev_obs=obs_dict
    ) == reward.compute_reward(obs_list, action=0, prev_obs=obs_list)


def test_normalize_obs_rejects_wrong_length():
    with pytest.raises(ValueError, match=r"Expected observation shape \(8,\)"):
        reward._normalize_obs([0.0, 1.0, 2.0])


def test_is_terminated_true_for_stable_landing_signature():
    landing_obs = [0.01, 0.02, 0.05, -0.04, 0.03, 0.01, 1.0, 1.0]
    assert reward.is_terminated(landing_obs)


def test_is_terminated_false_for_unstable_non_landing_state():
    non_terminal_obs = [0.4, 0.8, 0.6, -0.7, 0.6, 0.3, 0.0, 0.0]
    assert not reward.is_terminated(non_terminal_obs)


def test_is_truncated_threshold_and_validation():
    assert not reward.is_truncated(step_count=9, max_steps=10)
    assert reward.is_truncated(step_count=10, max_steps=10)
    with pytest.raises(ValueError, match="max_steps"):
        reward.is_truncated(step_count=1, max_steps=0)
