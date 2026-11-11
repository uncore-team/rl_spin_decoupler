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
from pathlib import Path
from typing import Any

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
from spindecoupler_gym import DecoupledEnv  # noqa: E402


class DecoupledBurgerRealEnv(DecoupledEnv):
    """Gymnasium view over the ROS2 agent and the decoupler socket.

    Observation (10 floats): ``[distance_to_goal, relative_goal_angle,
    ray_1, ..., ray_8]``, where the rays are eight 45-degree minimum LiDAR sectors
    in meters.
    Action (2 floats): ``[linear_x (m/s), angular_z (rad/s)]``.
    """

    def __init__(
        self,
        port: int,
        timeout: float,
        max_steps: int,
        debug: bool,
        reconnect: bool = True,
    ):
        # The 0.16 m lower bound on the rays is the collision threshold.
        observation_space = spaces.Box(
            low=np.array([0.0, -np.pi] + [0.16] * NUM_LIDAR_SECTORS, dtype=np.float32),
            high=np.array(
                [3.0, np.pi] + [LIDAR_MAX_RANGE] * NUM_LIDAR_SECTORS,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=np.array([0.0, -0.6], dtype=np.float32),
            high=np.array([0.15, 0.6], dtype=np.float32),
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
            observation, _, terminated, truncated, info = environment.step(action)
            if info.get("connection_lost"):
                print("[RL] connection lost; episode dropped, starting a new one")
                observation, _ = environment.reset()
                continue
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
