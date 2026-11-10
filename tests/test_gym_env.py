# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the Gymnasium environment of ``spindecoupler_gym``."""

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

from spindecoupler import ConnectionLost  # noqa: E402
from spindecoupler_gym import DecoupledEnv  # noqa: E402


class FakeRL:
    """Stand-in for RLSide that records what the environment asks for."""

    def __init__(self):
        self.actions = []
        self.finished = 0
        self.fail_finish = False
        self.fail_steps = 0
        self.fail_resets = 0
        self.reconnects = 0

    def resetGetObs(self, timeout):
        if self.fail_resets:
            self.fail_resets -= 1
            raise ConnectionLost("reset lost")
        return [1.0, 2.0], 5.0

    def reconnect(self):
        self.reconnects += 1

    def stepSendActGetObs(self, action, timeout):
        if self.fail_steps:
            self.fail_steps -= 1
            raise ConnectionLost("step lost")
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


def test_connection_loss_propagates_without_reconnect():
    rl = FakeRL()
    env = make_env(rl)
    env.reset()
    rl.fail_steps = 1
    with pytest.raises(ConnectionLost):
        env.step(0)
    rl.fail_resets = 1
    with pytest.raises(ConnectionLost):
        env.reset()
    assert rl.reconnects == 0


def test_step_truncates_and_reconnects_after_a_lost_connection():
    rl = FakeRL()
    env = make_env(rl, reconnect=True)
    first_obs, _ = env.reset()
    rl.fail_steps = 1

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs is first_obs
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["connection_lost"] is True
    assert np.isnan(info["lat"]) and np.isnan(info["t_agent"])
    assert env.connection_losses == 1
    assert rl.reconnects == 1
    env.reset()
    _, _, _, truncated, info = env.step(0)
    assert truncated is False
    assert info["connection_lost"] is False


def test_reset_retries_once_after_a_lost_connection():
    rl = FakeRL()
    env = make_env(rl, reconnect=True)
    rl.fail_resets = 1

    obs, info = env.reset()

    assert list(obs) == [1.0, 2.0]
    assert rl.reconnects == 1
    assert env.connection_losses == 1

    rl.fail_resets = 2
    with pytest.raises(ConnectionLost):
        env.reset()
