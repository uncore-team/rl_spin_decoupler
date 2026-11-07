# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the Gymnasium environment of ``spindecoupler_gym``."""

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

from spindecoupler_gym import DecoupledEnv  # noqa: E402


class FakeRL:
    """Stand-in for RLSide that records what the environment asks for."""

    def __init__(self):
        self.actions = []
        self.finished = 0
        self.fail_finish = False

    def resetGetObs(self, timeout):
        return [1.0, 2.0], 5.0

    def stepSendActGetObs(self, action, timeout):
        self.actions.append(action)
        return 0.05, [3.0, 4.0], 0.0, 6.0

    def stepExpFinished(self, timeout):
        self.finished += 1
        if self.fail_finish:
            raise RuntimeError("gone")


class Env(DecoupledEnv):
    def compute_reward(self, obs, action, prev_obs, lat):
        return float(obs[0] - prev_obs[0]) + lat


def make_env(rl=None, **kwargs):
    return Env(
        rl or FakeRL(),
        spaces.Box(-np.inf, np.inf, shape=(2,)),
        spaces.Discrete(2),
        **kwargs,
    )


def test_reset_returns_observation_and_agent_time():
    obs, info = make_env().reset(seed=1)

    assert obs == [1.0, 2.0]
    assert info == {"t_agent": 5.0, "step_count": 0}


def test_step_computes_reward_here_and_reports_timing():
    rl = FakeRL()
    env = make_env(rl)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(1)

    assert obs == [3.0, 4.0]
    assert reward == pytest.approx(2.05)  # 3.0 - 1.0 + lat
    assert (terminated, truncated) == (False, False)
    assert rl.actions == [1]
    assert info["lat"] == 0.05
    assert info["t_agent"] == 6.0
    assert info["t_wall"] > 0
    assert info["step_count"] == 1


def test_truncates_after_max_steps():
    env = make_env(max_steps=2)
    env.reset()

    assert env.step(0)[3] is False
    assert env.step(0)[3] is True


def test_parse_payload_flags_and_is_terminated_hooks():
    class Flagged(Env):
        def parse_payload(self, payload):
            return payload, True, True

        def is_terminated(self, obs):
            return False

    env = Flagged(FakeRL(), spaces.Box(-1, 1, shape=(2,)), spaces.Discrete(2))
    env.reset()

    _obs, _reward, terminated, truncated, info = env.step(0)

    assert terminated is True and truncated is True
    assert info["terminated_physics"] and info["truncated_physics"]


def test_compute_reward_must_be_overridden():
    env = DecoupledEnv(FakeRL(), spaces.Discrete(2), spaces.Discrete(2))
    env.reset()

    with pytest.raises(NotImplementedError):
        env.step(0)


def test_close_sends_finish_once_and_tolerates_failure():
    rl = FakeRL()
    env = make_env(rl)

    env.close()
    env.close()
    assert rl.finished == 1

    rl2 = FakeRL()
    rl2.fail_finish = True
    make_env(rl2).close()  # logged, not raised
    assert rl2.finished == 1
