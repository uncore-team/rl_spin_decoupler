# SPDX-License-Identifier: GPL-3.0-only

"""RL-side SB3 demo using rl_spin_decoupler and LunarLander.

Run this process first. It starts the RL-side server and trains PPO on a
Gymnasium-like environment wrapper that exchanges data with an external
agent process via sockets.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from reward import compute_reward, is_terminated, is_truncated
from stable_baselines3 import PPO

from spindecoupler import RLSide


class DecoupledLunarLanderEnv(gym.Env):
    """Gymnasium environment backed by rl_spin_decoupler communications.

    The RL algorithm interacts with this object as with a standard Gymnasium env.
    Actions/observations are transported over sockets, but the learning signal
    (reward/termination/truncation) is generated here on RL side from observation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int,
        timeout: float = 10.0,
        max_steps: int = 1000,
        debug: bool = False,
    ):
        super().__init__()
        self._timeout = timeout
        self._max_steps = max_steps
        self._debug = debug
        self._comm = RLSide(port, verbose=debug)
        self._finished = False
        self._step_count = 0
        self._prev_obs: np.ndarray | None = None

        # LunarLander-v3 uses 8 floats in observation and 4 discrete actions.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def _parse_payload(self, payload: Any) -> np.ndarray:
        """Decode agent payload into a fixed-size observation vector."""

        obs_payload = (
            payload.get("observation", payload)
            if isinstance(payload, dict)
            else payload
        )
        obs = np.asarray(obs_payload, dtype=np.float32)
        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"Invalid observation shape {obs.shape}; "
                f"expected {self.observation_space.shape}"
            )
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Request reset on agent side and return the initial observation."""

        super().reset(seed=seed)
        payload, ato = self._comm.resetGetObs(timeout=self._timeout)
        obs = self._parse_payload(payload)
        self._step_count = 0
        self._prev_obs = obs.copy()

        info: dict[str, Any] = {}
        info.update({"t_agent": ato, "step_count": self._step_count})
        if self._debug:
            print(f"[RL] reset ato={ato:.6f} obs={obs.tolist()}")
        return obs, info

    def step(self, action):
        """Send action, receive observation, and compute RL-side learning signal.

        Central design choice:
        - Agent transports observation/time (and LAT timing).
        - Reward/terminated/truncated are computed here from received observation.
        """

        lat, payload, _agent_rew_unused, ato = self._comm.stepSendActGetObs(
            int(action), timeout=self._timeout
        )
        obs = self._parse_payload(payload)

        reward = compute_reward(
            obs=obs,
            action=int(action),
            prev_obs=self._prev_obs,
            lat=float(lat),
        )

        self._step_count += 1
        terminated = is_terminated(obs)
        truncated = is_truncated(self._step_count, self._max_steps)
        self._prev_obs = obs.copy()

        t_wall = time.time()
        info: dict[str, Any] = {}
        info.update(
            {
                "lat": float(lat),
                "t_agent": ato,
                "t_wall": t_wall,
                "step_count": self._step_count,
                "reward_from_rl_logic": True,
            }
        )

        if self._debug:
            print(
                "[RL] step action={} lat={:.6f}s ato={:.6f} "
                "wall={:.6f} rew_rl={:.6f}".format(
                    int(action), lat, ato, t_wall, reward
                )
            )

        return obs, float(reward), terminated, truncated, info

    def close(self):
        """Signal experiment end to agent side once."""

        if self._finished:
            return
        self._finished = True
        try:
            self._comm.stepExpFinished(timeout=self._timeout)
        except Exception as exc:
            if self._debug:
                print(f"[RL] close warning: {exc}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this executable demo."""

    parser = argparse.ArgumentParser(
        description="SB3 PPO RL-side demo with decoupled LunarLander agent"
    )
    parser.add_argument(
        "--port", type=int, default=49054, help="TCP port for RL-side server"
    )
    parser.add_argument(
        "--timesteps", type=int, default=20_000, help="Total SB3 training timesteps"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum RL steps per episode before truncation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds per decoupler communication operation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="examples/lunar_lander/ppo_lunar_lander_decoupled.zip",
        help="Output path for the trained SB3 model",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed used by PPO and env reset"
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Optional post-training rollout steps to report mean LAT (0 disables)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose RL-side logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device passed to SB3 (auto/cpu/cuda)",
    )
    return parser.parse_args()


def run_rollout(model: PPO, env: DecoupledLunarLanderEnv, rollout_steps: int) -> None:
    """Run a short rollout and print mean LAT for timing observability."""

    obs, info = env.reset()
    _ = info
    lats: list[float] = []

    for _ in range(rollout_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        lats.append(float(info["lat"]))
        if terminated or truncated:
            obs, info = env.reset()
            _ = info

    if lats:
        mean_lat = float(np.mean(np.asarray(lats, dtype=np.float32)))
        print(
            f"[RL] rollout finished. mean LAT over {len(lats)} steps: {mean_lat:.6f}s"
        )


def main() -> None:
    """Train PPO using the decoupled Gymnasium wrapper."""

    args = parse_args()
    env = DecoupledLunarLanderEnv(
        port=args.port,
        timeout=args.timeout,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    try:
        env.reset(seed=args.seed)
        # 20k steps are kept small for quick execution. Solving LunarLander often
        # needs around 1e6 timesteps with tuned hyperparameters and reward design.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        # Report the effective device so the container demo proves the GPU is
        # wired up (note: PPO with MlpPolicy on LunarLander barely benefits from
        # a GPU; this split-host example is a deployment template, not a speedup).
        print(f"[RL] requested device={args.device}, effective device={model.device}")
        model.learn(total_timesteps=args.timesteps)

        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"[RL] training finished. Model saved to: {model_path}")

        if args.rollout_steps > 0:
            run_rollout(model, env, args.rollout_steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
