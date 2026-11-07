"""RL side of the real TurtleBot3 Burger example: policy exploitation only.

This process loads an already-trained Stable-Baselines3 SAC policy and runs
deterministic inference; it never calls ``learn()``. It talks to the ROS2 agent
(``ros2/burger_real_agent``) through the ``spindecoupler`` TCP socket:

* the agent sends observations (plus physics ``terminated``/``truncated`` flags);
* this process computes the reward and the task-level termination (goal reached,
  step budget), chooses the next action, and sends it back.

The same process works unchanged against the real robot and against the
CoppeliaSim stand-in of ``examples/burger_real_sim``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

CONTRACT_DIR = Path(__file__).resolve().parent / "ros2" / "burger_real_agent"
sys.path.insert(0, str(CONTRACT_DIR))

from contract import (  # noqa: E402
    LIDAR_MAX_RANGE,
    NUM_LIDAR_SECTORS,
)
from reward import compute_reward, is_goal_reached, is_truncated  # noqa: E402

from spindecoupler import RLSide  # noqa: E402


class DecoupledBurgerRealEnv(gym.Env):
    """Gymnasium view over the ROS2 agent and the decoupler socket.

    Observation (10 floats): ``[distance_to_goal, relative_goal_angle,
    ray_1, ..., ray_8]``, where the rays are eight 45-degree minimum LiDAR sectors
    in meters.
    Action (2 floats): ``[linear_x (m/s), angular_z (rad/s)]``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int,
        timeout: float,
        max_steps: int,
        debug: bool,
    ):
        super().__init__()
        self._timeout = timeout
        self._max_steps = max_steps
        self._debug = debug
        self._comm = RLSide(port, verbose=debug)
        self._finished = False
        self._step_count = 0
        self._prev_obs: np.ndarray | None = None
        # The 0.16 m lower bound on the rays is the collision threshold.
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi] + [0.16] * NUM_LIDAR_SECTORS, dtype=np.float32),
            high=np.array(
                [3.0, np.pi] + [LIDAR_MAX_RANGE] * NUM_LIDAR_SECTORS,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.6], dtype=np.float32),
            high=np.array([0.15, 0.6], dtype=np.float32),
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
    parser.add_argument(
        "--port", type=int, default=49054, help="TCP port to listen on for the agent"
    )
    parser.add_argument(
        "--model-path", required=True, help="Trained SB3 SAC model (.zip)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Step budget per episode before truncation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each agent reply. Raise it if the agent can be "
        "slow to answer (e.g. a simulator running below real time).",
    )
    parser.add_argument(
        "--device", default="auto", help="PyTorch device: auto, cpu or cuda"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Stop cleanly after this many episodes (targets); 0 runs forever.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose transport logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = DecoupledBurgerRealEnv(
        port=args.port,
        timeout=args.timeout,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    model = SAC.load(args.model_path, env=environment, device=args.device)
    max_episodes = max(0, int(args.episodes))
    try:
        episodes_done = 0
        observation, _ = environment.reset()
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                episodes_done += 1
                print(
                    f"[RL] episode {episodes_done} finished "
                    f"(terminated={terminated}, truncated={truncated})"
                )
                if max_episodes and episodes_done >= max_episodes:
                    print(f"[RL] reached {max_episodes} episode(s); finishing")
                    break
                observation, _ = environment.reset()
    except KeyboardInterrupt:
        pass
    finally:
        # Sends FINISH to the agent (via close -> stepExpFinished), so the agent
        # leaves its spin loop and the whole stack can tear down cleanly.
        environment.close()


if __name__ == "__main__":
    main()
