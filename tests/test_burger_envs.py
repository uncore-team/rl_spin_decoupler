# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the Burger environments built on ``DecoupledEnv``.

The RL side of each example is loaded from its folder and given a fake ``RLSide``, so
neither CoppeliaSim, ROS2 nor a socket is needed. The two examples have a ``reward``
module of the same name, so each test loads its own and removes it afterwards.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import numpy as np  # noqa: E402

from spindecoupler import ConnectionLost  # noqa: E402
from spindecoupler_gym import DecoupledEnv  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SIBLING_MODULES = ("reward", "episode_config", "contract")

CASES = {
    "coppeliasim": (
        EXAMPLES / "burger_coppeliasim",
        "rl_side_burger_coppeliasim.py",
        "DecoupledCoppeliaBurgerEnv",
    ),
    "real": (
        EXAMPLES / "burger_real",
        "rl_side_burger_real.py",
        "DecoupledBurgerRealEnv",
    ),
}


def observation(distance=1.0, lidar_min=1.0):
    return [distance, 0.0, lidar_min] + [1.0] * 7


class FakeRL:
    """Stand-in for RLSide that records what the environment asks for."""

    def __init__(self):
        self.actions = []
        self.finished = 0
        self.reconnects = 0
        self.fail_steps = 0
        self.next_payload = {"observation": observation()}

    def resetGetObs(self, timeout):
        return {"observation": observation()}, 5.0

    def stepSendActGetObs(self, action, timeout):
        if self.fail_steps:
            self.fail_steps -= 1
            raise ConnectionLost("step lost")
        self.actions.append(action)
        return 0.05, self.next_payload, 0.0, 6.0

    def reconnect(self):
        self.reconnects += 1

    def stepExpFinished(self, timeout):
        self.finished += 1


@pytest.fixture
def make_env(monkeypatch):
    """Build the environment of one example around a fake RLSide."""
    fake = FakeRL()

    def build(case, **kwargs):
        folder, filename, class_name = CASES[case]
        for name in SIBLING_MODULES:
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.syspath_prepend(str(folder))
        spec = importlib.util.spec_from_file_location(
            f"burger_{case}_rl", folder / filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "RLSide", lambda port, verbose=False: fake)
        params = {"port": 1, "timeout": 1.0, "max_steps": 500, "debug": False}
        params.update(kwargs)
        return getattr(module, class_name)(**params), sys.modules["reward"]

    yield build, fake
    for name in SIBLING_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


def test_environment_is_a_decoupled_env_and_resets(make_env, case):
    build, _ = make_env
    env, _ = build(case)

    obs, info = env.reset()

    assert isinstance(env, DecoupledEnv)
    assert obs.dtype == np.float32 and obs.shape == (10,)
    assert info["t_agent"] == 5.0


def test_step_sends_a_plain_list_and_computes_the_reward_here(make_env, case):
    build, fake = make_env
    env, reward = build(case)
    first, _ = env.reset()
    action = np.array([0.1, 0.2], dtype=np.float32)
    fake.next_payload = {"observation": observation(distance=0.8)}

    obs, got, terminated, truncated, info = env.step(action)

    assert type(fake.actions[0]) is list
    assert all(type(value) is float for value in fake.actions[0])
    assert got == pytest.approx(
        reward.compute_reward(obs, fake.actions[0], first, 0.05)
    )
    assert (terminated, truncated) == (False, False)
    assert info["lat"] == pytest.approx(0.05)
    assert info["connection_lost"] is False


def test_episode_ends_on_collision_goal_and_step_budget(make_env, case):
    build, fake = make_env
    env, _ = build(case, max_steps=2)
    action = np.array([0.1, 0.0], dtype=np.float32)
    env.reset()

    fake.next_payload = {"observation": observation(), "terminated": True}
    assert env.step(action)[2] is True

    fake.next_payload = {"observation": observation(distance=0.05)}
    assert env.step(action)[2] is True

    env.reset()
    fake.next_payload = {"observation": observation()}
    assert env.step(action)[3] is False
    assert env.step(action)[3] is True


def test_payload_without_a_dict_is_an_observation(make_env, case):
    build, fake = make_env
    env, _ = build(case)
    env.reset()
    fake.next_payload = observation()

    assert env.step(np.array([0.1, 0.0], dtype=np.float32))[0].shape == (10,)


def test_wrong_observation_shape_is_rejected(make_env, case):
    build, fake = make_env
    env, _ = build(case)
    env.reset()
    fake.next_payload = {"observation": [1.0, 2.0, 3.0]}

    with pytest.raises(ValueError, match="Invalid observation shape"):
        env.step(np.array([0.1, 0.0], dtype=np.float32))


def test_lost_connection_truncates_the_step_and_reconnects(make_env, case):
    build, fake = make_env
    env, _ = build(case)
    first, _ = env.reset()
    fake.fail_steps = 1

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.1, 0.0], dtype=np.float32)
    )

    assert obs is first and reward == 0.0
    assert (terminated, truncated) == (False, True)
    assert info["connection_lost"] is True
    assert fake.reconnects == 1


def test_close_sends_finish_once(make_env, case):
    build, fake = make_env
    env, _ = build(case)

    env.close()
    env.close()

    assert fake.finished == 1
