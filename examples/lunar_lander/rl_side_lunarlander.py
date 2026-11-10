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
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from reward import compute_reward, is_terminated, is_truncated
from stable_baselines3 import PPO

from spindecoupler import RLSide
from spindecoupler_gym import DecoupledEnv


class DecoupledLunarLanderEnv(DecoupledEnv):
    """
    LunarLander whose steps are served by the agent.

    Reward comes from the received observation (reward.py). An episode terminates
    when the agent reports a physics end or the lander has landed, and is truncated
    when the agent reports its time limit or the RL-side step budget runs out.
    """

    def __init__(
        self,
        port: int,
        timeout: float = 10.0,
        max_steps: int = 1000,
        debug: bool = False,
        reconnect: bool = True,
    ):
        super().__init__(
            RLSide(port, verbose=debug),
            # LunarLander-v3: 8 floats in the observation and 4 discrete actions.
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            ),
            action_space=spaces.Discrete(4),
            timeout=timeout,
            max_steps=max_steps,
            debug=debug,
            reconnect=reconnect,
        )

    def parse_payload(self, payload: Any) -> tuple[np.ndarray, bool, bool]:
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

    def step(self, action):
        # Gymnasium hands the discrete action as a NumPy integer; send a plain int.
        return super().step(int(action))

    def compute_reward(self, obs, action, prev_obs, lat):
        return compute_reward(obs, action, prev_obs, lat)

    def is_terminated(self, obs):
        return is_terminated(obs)

    def is_truncated(self, step_count, max_steps):
        return is_truncated(step_count, max_steps)


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
