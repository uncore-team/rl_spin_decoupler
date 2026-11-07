# SPDX-License-Identifier: GPL-3.0-only

"""
Template for the RL side of a decoupled setup.

Start this process first: RLSide listens and waits for the agent to connect.
Each "..." marks something you have to supply. The agent counterpart is in
skeleton_agent_side.py.
"""

import time
from typing import Optional

from spindecoupler import RLSide


class RLEnv:
    """Gym-style environment whose reset() and step() are served by the agent."""

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._commstoagent = RLSide(49054, verbose=debug)  # waits for the agent

    def resetGetObs(self):
        obs0, t_agent = self._commstoagent.resetGetObs()
        return obs0, t_agent

    def stepSendActGetObs(self, action):
        """Return (o_{t+1}, t_agent, t_wall, LAT)."""
        lat, obs_next, _rew_from_agent, t_agent = (
            self._commstoagent.stepSendActGetObs(action)
        )
        t_wall = time.time()  # RL clock; t_agent and lat use the agent's
        return obs_next, t_agent, t_wall, lat

    def stepExpFinished(self):
        self._commstoagent.stepExpFinished()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs0, t_agent = self.resetGetObs()
        if self._debug:
            print(f"reset: t_agent={t_agent}")
        return obs0, {"t_agent": t_agent}

    def step(self, action):
        obs, t_agent, t_wall, lat = self.stepSendActGetObs(action)
        reward = self._compute_reward(obs, action, lat)
        terminated = self._is_terminal(obs)
        truncated = False
        info = {"t_agent": t_agent, "t_wall": t_wall, "lat": lat}
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs, action, lat: float) -> float:
        return ...

    def _is_terminal(self, obs) -> bool:
        return ...


if __name__ == "__main__":
    env = RLEnv(debug=True)

    model = ...
    model.learn(total_timesteps=1_000)
    model.save(...)

    env.stepExpFinished()
