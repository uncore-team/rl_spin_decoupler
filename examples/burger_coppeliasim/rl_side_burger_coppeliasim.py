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

from episode_config import LIDAR_MAX_RANGE, NUM_LIDAR_SECTORS, OBS_DIM
from reward import compute_reward, is_goal_reached, is_truncated
from spindecoupler import RLSide

# ============================================================================
# Reward and task-level termination are computed HERE, on the RL side:
#   - reward           -> reward.py::compute_reward
#   - goal reached     -> reward.py::is_goal_reached (task termination)
#   - step-budget      -> reward.py::is_truncated (RL-side truncation)
# The agent transports the PHYSICS termination in the observation payload:
#   - collision        -> payload["terminated"] (see _parse_payload)
# The RL side ORs the physics flag with its task/budget decisions.
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
        max_steps: int = 500,
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

        # Observation (OBS_DIM): NUM_LIDAR_SECTORS LiDAR + distance + angle.
        # LiDAR entries are metric distances clamped to LIDAR_MAX_RANGE (must
        # match scene/laser.py::config["max_scan_distance"]).
        self.observation_space = spaces.Box(
            low=np.array([0.0] * NUM_LIDAR_SECTORS + [0.0, -np.pi], dtype=np.float32),
            high=np.array(
                [LIDAR_MAX_RANGE] * NUM_LIDAR_SECTORS + [10.0, np.pi],
                dtype=np.float32,
            ),
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
        self._prev_obs = obs.copy()

        info: dict[str, Any] = {"t_agent": ato, "step_count": self._step_count}
        if self._debug:
            print(f"[RL] reset ato={ato:.6f} obs={obs.tolist()}")
        return obs, info

    def step(self, action: np.ndarray):
        action_list = action.tolist()  # Send the continuous action vector.
        # The transported reward slot is unused: reward is computed here.
        lat, payload, _rew_unused, ato = self._comm.stepSendActGetObs(
            action_list, timeout=self._timeout
        )
        obs, term_phys, trunc_phys = self._parse_payload(payload)

        reward = compute_reward(obs, action, self._prev_obs, float(lat))

        self._step_count += 1
        # terminated = physics collision (from agent) OR task goal (from obs).
        terminated = bool(term_phys or is_goal_reached(obs))
        # truncated = physics truncation (from agent) OR RL-side step budget.
        truncated = bool(
            trunc_phys or is_truncated(self._step_count, self._max_steps)
        )
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
                f"[RL] step action={action_list} lat={lat:.6f}s "
                f"ato={ato:.6f} rew={reward:.6f} term={terminated} trunc={truncated}"
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
    parser = argparse.ArgumentParser(description="PPO RL-side CoppeliaSim Burger Demo")
    parser.add_argument("--port", type=int, default=49054, help="TCP port for RL-side server")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total SB3 timesteps")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum RL steps per episode before truncation (RL-side budget)",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Communication timeout")
    parser.add_argument("--model-path", type=str, default="ppo_coppelia_burger.zip", help="Output model path")
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
        model = PPO(
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
