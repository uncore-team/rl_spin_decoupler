# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from spindecoupler import RLSide

# ============================================================================
# Reward and episode-termination detection (terminated/truncated) are
# computed in agent-side and arrive over the protocol:
#   - `rew` -> RLSide.stepSendActGetObs
#   - `terminated`/`truncated` -> observation payload (see _parse_payload)
# See agent_side_burger_coppeliasim.py (compute_reward, is_terminated,
# is_truncated, both imported there from reward.py).
# ============================================================================


# ============================================================================
# DECOUPLED GYMNASIUM ENVIRONMENT
# ============================================================================
class DecoupledCoppeliaBurgerEnv(gym.Env):
    """Gymnasium environment backed by rl_spin_decoupler communications for CoppeliaSim."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int,
        timeout: float = 10.0,
        debug: bool = False,
    ):
        super().__init__()
        self._timeout = timeout
        self._debug = debug
        self._comm = RLSide(port, verbose=debug)
        self._finished = False
        self._step_count = 0

        # Observation (10 dim): 8 LiDAR + distance + angle
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 8 + [0.0, -np.pi], dtype=np.float32),
            high=np.array([1.0] * 8 + [10.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        # Continuous action (2 dim): [linear velocity (m/s), angular velocity (rad/s)]
        self.action_space = spaces.Box(
            low=np.array([-0.22, -2.84], dtype=np.float32),
            high=np.array([0.22, 2.84], dtype=np.float32),
            dtype=np.float32,
        )

    def _parse_payload(self, payload: Any) -> tuple[np.ndarray, bool, bool]:
        if isinstance(payload, dict):
            obs_payload = payload.get("observation", payload)
            terminated = bool(payload.get("terminated", False))
            truncated = bool(payload.get("truncated", False))
        else:
            obs_payload = payload
            terminated = False
            truncated = False

        obs = np.asarray(obs_payload, dtype=np.float32)
        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"Invalid observation shape {obs.shape}; "
                f"expected {self.observation_space.shape}"
            )
        return obs, terminated, truncated

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        payload, ato = self._comm.resetGetObs(timeout=self._timeout)
        obs, _terminated, _truncated = self._parse_payload(payload)
        self._step_count = 0

        info: dict[str, Any] = {"t_agent": ato, "step_count": self._step_count}
        if self._debug:
            print(f"[RL] reset ato={ato:.6f} obs={obs.tolist()}")
        return obs, info

    def step(self, action: np.ndarray):
        action_list = action.tolist()  # Send the continuous action vector.
        lat, payload, agent_reward, ato = self._comm.stepSendActGetObs(
            action_list, timeout=self._timeout
        )
        obs, terminated, truncated = self._parse_payload(payload)
        reward = float(agent_reward)

        self._step_count += 1  # Informational only (see info["step_count"]).

        t_wall = time.time()
        info: dict[str, Any] = {
            "lat": float(lat),
            "t_agent": ato,
            "t_wall": t_wall,
            "step_count": self._step_count,
            "reward_from_agent": True,
        }

        if self._debug:
            print(
                f"[RL] step action={action_list} lat={lat:.6f}s "
                f"ato={ato:.6f} rew_agent={reward:.6f}"
            )

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._finished:
            return
        self._finished = True
        try:
            self._comm.stepExpFinished(timeout=self._timeout)
        except Exception as exc:
            if self._debug:
                print(f"[RL] close warning: {exc}")


def parse_args() -> argparse.Namespace:
    # Note: the per-episode step budget (--max-steps) is now configured in
    # agent_side_burger_coppeliasim.py, which is what computes `truncated`.
    parser = argparse.ArgumentParser(description="PPO RL-side CoppeliaSim Burger Demo")
    parser.add_argument("--port", type=int, default=49054, help="TCP port for RL-side server")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total SB3 timesteps")
    parser.add_argument("--timeout", type=float, default=5.0, help="Communication timeout")
    parser.add_argument("--model-path", type=str, default="ppo_coppelia_burger.zip", help="Output model path")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = DecoupledCoppeliaBurgerEnv(
        port=args.port,
        timeout=args.timeout,
        debug=args.debug,
    )
    try:
        env.reset(seed=args.seed)
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, learning_rate=3e-4)
        model.learn(total_timesteps=args.timesteps)

        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"[RL] Training finished. Model saved to: {model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()