"""Remote inference process for a real TurtleBot3 Burger."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

CONTRACT_DIR = Path(__file__).resolve().parent / "ros2" / "burger_real_agent"
sys.path.insert(0, str(CONTRACT_DIR))

from contract import (  # noqa: E402
    LIDAR_MAX_RANGE,
    NUM_LIDAR_SECTORS,
)
from reward import compute_reward, is_goal_reached, is_truncated  # noqa: E402

from spindecoupler import RLSide  # noqa: E402


class DecoupledBurgerRealEnv(gym.Env):
    """Gymnasium view over the ROS2 agent and the decoupler socket."""

    metadata = {"render_modes": []}

    def __init__(self, port: int, timeout: float, max_steps: int, debug: bool):
        super().__init__()
        self._timeout = timeout
        self._max_steps = max_steps
        self._debug = debug
        self._comm = RLSide(port, verbose=debug)
        self._finished = False
        self._step_count = 0
        self._prev_obs: np.ndarray | None = None
        self.observation_space = spaces.Box(
            low=np.array([0.0] * NUM_LIDAR_SECTORS + [0.0, -np.pi], dtype=np.float32),
            high=np.array(
                [LIDAR_MAX_RANGE] * NUM_LIDAR_SECTORS + [10.0, np.pi],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-0.22, -2.84], dtype=np.float32),
            high=np.array([0.22, 2.84], dtype=np.float32),
            dtype=np.float32,
        )

    def _parse_payload(self, payload: Any) -> tuple[np.ndarray, bool, bool]:
        if isinstance(payload, dict):
            observation = payload.get("observation", payload)
            terminated = bool(payload.get("terminated", False))
            truncated = bool(payload.get("truncated", False))
        else:
            observation = payload
            terminated = False
            truncated = False
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != self.observation_space.shape:
            raise ValueError(
                f"Invalid observation shape {array.shape}; "
                f"expected {self.observation_space.shape}"
            )
        return array, terminated, truncated

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        payload, agent_time = self._comm.resetGetObs(timeout=self._timeout)
        observation, _, _ = self._parse_payload(payload)
        self._step_count = 0
        self._prev_obs = observation.copy()
        return observation, {"t_agent": agent_time, "step_count": 0}

    def step(self, action: np.ndarray):
        latency, payload, _, agent_time = self._comm.stepSendActGetObs(
            action.tolist(), timeout=self._timeout
        )
        observation, physics_terminated, physics_truncated = self._parse_payload(
            payload
        )
        reward = compute_reward(observation, action, self._prev_obs, float(latency))
        self._step_count += 1
        terminated = bool(physics_terminated or is_goal_reached(observation))
        truncated = bool(
            physics_truncated or is_truncated(self._step_count, self._max_steps)
        )
        self._prev_obs = observation.copy()
        info = {
            "lat": float(latency),
            "t_agent": agent_time,
            "t_wall": time.time(),
            "step_count": self._step_count,
            "terminated_physics": physics_terminated,
            "truncated_physics": physics_truncated,
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._finished:
            return
        self._finished = True
        self._comm.stepExpFinished(timeout=self._timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real Burger policy exploitation")
    parser.add_argument("--port", type=int, default=49054)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = DecoupledBurgerRealEnv(
        port=args.port,
        timeout=args.timeout,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    model = PPO.load(args.model_path, env=environment, device=args.device)
    try:
        observation, _ = environment.reset()
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                observation, _ = environment.reset()
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()


if __name__ == "__main__":
    main()
