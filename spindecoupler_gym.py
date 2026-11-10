# SPDX-License-Identifier: GPL-3.0-only

"""
Gymnasium environment on top of :class:`spindecoupler.RLSide`.

Every example repeats the same wrapper: ``reset()`` and ``step()`` forward to the
agent, the reward is computed from what comes back, and ``close()`` tells the agent
that the experiment is over. :class:`DecoupledEnv` is that wrapper. A task only
overrides the hooks that carry its own logic: how to read the agent's payload, the
reward and the end of the episode.

Gymnasium is imported here and not in the core module, so ``spindecoupler`` itself
still needs nothing beyond the standard library.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym

from spindecoupler import ConnectionLost, RLSide, enable_console_logging

logger = logging.getLogger("spindecoupler")


class DecoupledEnv(gym.Env):
    """
    Environment whose ``reset()`` and ``step()`` are answered by an agent process.

    Reward and the task-level end of the episode are decided on the RL side, from
    the observation: the agent only senses and acts. The ``info`` returned by
    ``step()`` carries the timing values of the protocol: ``lat`` (duration of the
    previous action) and ``t_agent`` (time of the observation), both by the agent's
    clock, and ``t_wall``, the RL process clock when the reply arrived.

    Override these hooks:

    - :meth:`parse_payload`, if the agent sends more than the bare observation,
    - :meth:`compute_reward` (required),
    - :meth:`is_terminated` and :meth:`is_truncated`.

    Args:
        rl_side: Connected :class:`~spindecoupler.RLSide`. Build it with
            ``RLSide(port)``; that call waits for the agent.
        observation_space: Gymnasium space of the observations.
        action_space: Gymnasium space of the actions.
        timeout: Seconds to wait for each reply of the agent.
        max_steps: Steps per episode before it is truncated, or ``None`` for no
            limit on the RL side.
        debug: Log one line per reset and step (the ``spindecoupler`` logger at
            debug level, shown on the standard output).
        reconnect: Survive a lost connection. If it drops during ``step()``, that
            step is returned truncated (previous observation, zero reward,
            ``info["connection_lost"]`` true and ``lat`` and ``t_agent`` NaN), the
            connection is restored with :meth:`~spindecoupler.RLSide.reconnect`, and
            the next ``reset()`` starts a new episode. A drop during ``reset()`` is
            retried once. With ``False`` (the default),
            :class:`~spindecoupler.ConnectionLost` propagates.

    Attributes:
        connection_losses: How many lost connections were recovered.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rl_side: RLSide,
        observation_space: gym.Space,
        action_space: gym.Space,
        timeout: float = 10.0,
        max_steps: int | None = None,
        debug: bool = False,
        reconnect: bool = False,
    ):
        super().__init__()
        if debug:
            enable_console_logging(logging.DEBUG)
        self._rl = rl_side
        self.observation_space = observation_space
        self.action_space = action_space
        self._timeout = timeout
        self._max_steps = max_steps
        self._step_count = 0
        self._prev_obs: Any = None
        self._finished = False
        self._reconnect = reconnect
        self.connection_losses = 0

    def _recover(self, exc: ConnectionLost):
        """Count a lost connection and wait for the agent to come back."""
        self.connection_losses += 1
        logger.warning("connection lost (%s); reconnecting", exc)
        self._rl.reconnect()

    def parse_payload(self, payload: Any) -> tuple[Any, bool, bool]:
        """
        Split what the agent sent into ``(observation, terminated, truncated)``.

        The default takes the payload as the observation and no flags. Override it
        when the agent also reports physics flags, such as a crash or a time limit.
        """
        return payload, False, False

    def compute_reward(self, obs: Any, action: Any, prev_obs: Any, lat: float) -> float:
        """Reward of the last step, from the new and the previous observation."""
        raise NotImplementedError("compute_reward must be overridden")

    def is_terminated(self, obs: Any) -> bool:
        """True when the task is over (success or failure), judged from ``obs``."""
        return False

    def is_truncated(self, step_count: int, max_steps: int | None) -> bool:
        """True when the episode ran out of steps. The default uses ``max_steps``."""
        return max_steps is not None and step_count >= max_steps

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Ask the agent to reset and return its first observation and ``info``."""
        super().reset(seed=seed)
        try:
            payload, ato = self._rl.resetGetObs(timeout=self._timeout)
        except ConnectionLost as exc:
            if not self._reconnect:
                raise
            self._recover(exc)
            payload, ato = self._rl.resetGetObs(timeout=self._timeout)
        obs, _terminated, _truncated = self.parse_payload(payload)
        self._step_count = 0
        self._prev_obs = obs
        logger.debug("reset ato=%.6f", ato)
        return obs, {"t_agent": ato, "step_count": 0}

    def step(self, action: Any):
        """Send ``action`` to the agent and return the Gymnasium 5-tuple."""
        # The reward slot of the protocol is not used: it is computed here.
        try:
            lat, payload, _rew, ato = self._rl.stepSendActGetObs(
                action, timeout=self._timeout
            )
        except ConnectionLost as exc:
            if not self._reconnect:
                raise
            self._recover(exc)
            self._step_count += 1
            info = {
                "lat": float("nan"),
                "t_agent": float("nan"),
                "t_wall": time.time(),
                "step_count": self._step_count,
                "terminated_physics": False,
                "truncated_physics": False,
                "connection_lost": True,
            }
            return self._prev_obs, 0.0, False, True, info
        obs, term_physics, trunc_physics = self.parse_payload(payload)
        reward = self.compute_reward(obs, action, self._prev_obs, float(lat))
        self._step_count += 1
        terminated = bool(term_physics or self.is_terminated(obs))
        truncated = bool(
            trunc_physics or self.is_truncated(self._step_count, self._max_steps)
        )
        self._prev_obs = obs
        info = {
            "lat": float(lat),
            "t_agent": ato,
            "t_wall": time.time(),
            "step_count": self._step_count,
            "terminated_physics": bool(term_physics),
            "truncated_physics": bool(trunc_physics),
            "connection_lost": False,
        }
        logger.debug(
            "step lat=%.6f ato=%.6f rew=%.6f terminated=%s truncated=%s",
            lat,
            ato,
            reward,
            terminated,
            truncated,
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        """Tell the agent that the experiment is over. Only the first call acts."""
        if self._finished:
            return
        self._finished = True
        try:
            self._rl.stepExpFinished(timeout=self._timeout)
        except Exception as exc:
            logger.warning("could not send FINISH to the agent: %s", exc)
