# SPDX-License-Identifier: GPL-3.0-only

import importlib.util
from pathlib import Path


def _load_fopcontrol_module():
    """Load the first-order plant agent-side reward module."""

    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "examples" / "first_order_plant_control" / "reward.py"
    )
    spec = importlib.util.spec_from_file_location("fopcontrol_reward", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load fopcontrol reward module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fopcontrol_reward = _load_fopcontrol_module()


def test_compute_reward_penalizes_absolute_plant_state():
    obs_near = {"plant_state": 0.1}
    obs_far = {"plant_state": -0.7}

    r_near = fopcontrol_reward.compute_reward(obs_near, action=None)
    r_far = fopcontrol_reward.compute_reward(obs_far, action=None)

    assert r_near > r_far


def test_goal_reached_respects_tolerance_threshold():
    obs_inside = {"plant_state": 0.049}
    obs_outside = {"plant_state": 0.06}

    assert fopcontrol_reward.is_terminated(obs_inside, tolerance=0.05)
    assert not fopcontrol_reward.is_terminated(obs_outside, tolerance=0.05)
