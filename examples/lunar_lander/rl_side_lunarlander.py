# SPDX-License-Identifier: GPL-3.0-only

"""
RL side of the LunarLander example: PPO training.

Run it first. It opens the server socket and trains a Stable-Baselines3 PPO
policy on a Gymnasium environment whose ``step()`` and ``reset()`` are answered
by the agent process (agent_side_lunarlander.py) through spindecoupler, and waits
up to 60 s for the agent to connect.

Reward and the task-level end of the episode are computed here, from the
observation (see reward.py). The agent only sends the observation, its timing and
the physics flags of Gymnasium.
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
    """
    Gymnasium environment whose steps are served by the agent.

    Reward comes from the received observation. An episode terminates when the agent
    reports a physics end or the lander has landed, and is truncated when the agent
    reports its time limit or the RL-side step budget runs out.
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

    def _parse_payload(self, payload: Any) -> tuple[np.ndarray, bool, bool]:
        """Split the agent's payload into observation, terminated and truncated."""

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
        self._prev_obs = obs.copy()

        info: dict[str, Any] = {"t_agent": ato, "step_count": self._step_count}
        if self._debug:
            print(f"[RL] reset ato={ato:.6f} obs={obs.tolist()}")
        return obs, info

    def step(self, action):
        # The transported reward slot is unused: reward is computed here.
        lat, payload, _rew_unused, ato = self._comm.stepSendActGetObs(
            int(action), timeout=self._timeout
        )
        obs, term_phys, trunc_phys = self._parse_payload(payload)

        reward = compute_reward(obs, int(action), self._prev_obs, float(lat))

        self._step_count += 1
        terminated = bool(term_phys or is_terminated(obs))
        truncated = bool(trunc_phys or is_truncated(self._step_count, self._max_steps))
        self._prev_obs = obs.copy()

        t_wall = time.time()
        info: dict[str, Any] = {
            "lat": float(lat),
            "t_agent": ato,
            "t_wall": t_wall,
            "step_count": self._step_count,
            "terminated_physics": bool(term_phys),
            "truncated_physics": bool(trunc_phys),
        }

        if self._debug:
            print(
                "[RL] step action={} lat={:.6f}s ato={:.6f} "
                "wall={:.6f} rew={:.6f} term={} trunc={}".format(
                    int(action), lat, ato, t_wall, reward, terminated, truncated
                )
            )

        return obs, reward, terminated, truncated, info

    def close(self):
        """Tell the agent the experiment is over, once."""

        if self._finished:
            return
        self._finished = True
        try:
            self._comm.stepExpFinished(timeout=self._timeout)
        except Exception as exc:
            if self._debug:
                print(f"[RL] close warning: {exc}")


def parse_args() -> argparse.Namespace:
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
        "--n-steps",
        type=int,
        default=2048,
        help="PPO rollout length: steps collected between policy updates. PPO "
        "always completes a whole rollout, so this is also the minimum number of "
        "steps a run executes. Use a small value for quick runs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for each agent reply. Raise it if the agent can "
        "be slow to answer (e.g. a remote host over a slow link).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="examples/lunar_lander/ppo_lunar_lander_decoupled.zip",
        help="Where to save the trained SB3 model (.zip)",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed used by PPO and env reset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for SB3 (auto|cpu|cuda). Default auto-detects a GPU.",
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
    return parser.parse_args()


def run_rollout(model: PPO, env: DecoupledLunarLanderEnv, rollout_steps: int) -> None:
    """Run ``rollout_steps`` steps with the trained policy and print the mean LAT."""

    obs, _info = env.reset()
    lats: list[float] = []

    for _ in range(rollout_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        lats.append(float(info["lat"]))
        if terminated or truncated:
            obs, _info = env.reset()

    if lats:
        mean_lat = float(np.mean(np.asarray(lats, dtype=np.float32)))
        print(
            f"[RL] rollout finished. mean LAT over {len(lats)} steps: {mean_lat:.6f}s"
        )


def main() -> None:
    args = parse_args()
    env = DecoupledLunarLanderEnv(
        port=args.port,
        timeout=args.timeout,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    try:
        env.reset(seed=args.seed)
        # The default of 20k steps keeps the demo short. Solving LunarLander often
        # needs around 1e6 steps with tuned hyperparameters and reward design.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=min(64, args.n_steps),
        )
        print(f"[RL] torch device: {model.device}")
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
