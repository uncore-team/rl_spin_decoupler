# SPDX-License-Identifier: GPL-3.0-only
"""RL side of the CoppeliaSim Burger example: SAC training.

Trains a Stable-Baselines3 SAC policy to drive the TurtleBot3 Burger to a goal
among obstacles. The simulator lives in another process (the agent side,
``agent_side_burger_coppeliasim.py``); this process talks to it through the
``spindecoupler`` TCP socket, which makes the environment look like an ordinary
Gymnasium one.

Who decides what:

* reward and the task-level termination are computed HERE, in ``reward.py``:
  ``compute_reward``, ``is_goal_reached`` (goal) and ``is_truncated`` (step
  budget);
* the agent reports the PHYSICS termination in the payload (``terminated`` =
  collision, detected from its LiDAR).

An episode ends when either side says so. Start this process first: it waits up
to 60 s for the agent to connect. The trained model is saved at the end.

The environment is built on ``DecoupledEnv``. If the connection drops, the episode in
progress is truncated and this process waits up to 60 s for the agent to come back.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from episode_config import (
    COLLISION_LIDAR_THRESHOLD,
    LIDAR_MAX_RANGE,
    MAX_ANGULAR_VELOCITY,
    MAX_GOAL_DISTANCE,
    MAX_LINEAR_VELOCITY,
    NUM_LIDAR_SECTORS,
)
from gymnasium import spaces
from reward import compute_reward, is_goal_reached, is_truncated
from stable_baselines3 import SAC

from spindecoupler import RLSide
from spindecoupler_gym import DecoupledEnv


class DecoupledCoppeliaBurgerEnv(DecoupledEnv):
    """Gymnasium environment backed by the decoupler socket (see module docstring).

    Observation (10 floats): ``[distance_to_goal, relative_goal_angle, ray_1, ...,
    ray_8]``; the rays are 45-degree LiDAR sector minima in meters.
    Action (2 floats): ``[linear_x (m/s), angular_z (rad/s)]``.
    """

    def __init__(
        self,
        port: int,
        timeout: float = 10.0,
        max_steps: int = 500,
        debug: bool = False,
        reconnect: bool = True,
    ):
        # The lower bound of the rays is the collision threshold: a smaller
        # reading ends the episode.
        observation_space = spaces.Box(
            low=np.array(
                [0.0, -np.pi] + [COLLISION_LIDAR_THRESHOLD] * NUM_LIDAR_SECTORS,
                dtype=np.float32,
            ),
            high=np.array(
                [MAX_GOAL_DISTANCE, np.pi] + [LIDAR_MAX_RANGE] * NUM_LIDAR_SECTORS,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Forward velocity only (no reversing); turning in both directions.
        action_space = spaces.Box(
            low=np.array([0.0, -MAX_ANGULAR_VELOCITY], dtype=np.float32),
            high=np.array(
                [MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY], dtype=np.float32
            ),
            dtype=np.float32,
        )
        super().__init__(
            RLSide(port, verbose=debug),
            observation_space=observation_space,
            action_space=action_space,
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
        # The agent receives a plain list of floats, not a NumPy array.
        return super().step(action.tolist())

    def compute_reward(self, obs, action, prev_obs, lat):
        return compute_reward(obs, action, prev_obs, lat)

    def is_terminated(self, obs):
        return is_goal_reached(obs)

    def is_truncated(self, step_count, max_steps):
        return is_truncated(step_count, max_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAC RL-side CoppeliaSim Burger Demo")
    parser.add_argument(
        "--port", type=int, default=49054, help="TCP port to listen on for the agent"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total SB3 timesteps"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum RL steps per episode before truncation (RL-side budget)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for each agent reply. A reset is the slowest one "
        "(the scene is rebuilt); raise it if the simulator can be slow.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="sac_coppelia_burger.zip",
        help="Where to save the trained model (.zip)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for SB3 (auto|cpu|cuda). Default auto-detects a GPU.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = DecoupledCoppeliaBurgerEnv(
        port=args.port,
        timeout=args.timeout,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    try:
        env.reset(seed=args.seed)
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            device=args.device,
        )
        print(f"[RL] torch device: {model.device}")
        model.learn(total_timesteps=args.timesteps)

        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"[RL] Training finished. Model saved to: {model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
