# SPDX-License-Identifier: GPL-3.0-only

"""RL side of the first-order plant example.

Start it first; it listens until `agent_side_fopcontrol.py` connects. It sends
alternating targets and prints LAT, ATO and the wall-clock time of every step.
The reward and the success check are computed here, from the observation only.
"""

from __future__ import annotations

import argparse
import time

from spindecoupler import RLSide


def _compute_reward(obs: dict) -> float:
    """Reward: minus the distance of the plant state to zero."""

    return -abs(float(obs.get("plant_state", 0.0)))


def _goal_reached(obs: dict, tolerance: float = 0.05) -> bool:
    """True when the plant state is within `tolerance` of zero."""

    return abs(float(obs.get("plant_state", 0.0))) <= tolerance


def run_episode(
    host_port: int,
    num_steps: int,
    timeout: float,
) -> None:
    """Run one episode against the agent.

    Args:
        host_port: TCP port the RL side listens on.
        num_steps: Maximum number of actions sent after the reset.
        timeout: Seconds to wait for each agent reply.
    """

    rl = RLSide(host_port, verbose=True)
    obs0, ato0 = rl.resetGetObs(timeout=timeout)
    print(f"[RL] reset -> obs={obs0} ato={ato0:.6f}")

    total_rew = 0.0
    for step_idx in range(num_steps):
        action: dict[str, float] = {
            "target": 0.8 if step_idx % 2 == 0 else -0.4,
            "gain": 0.25,
        }
        lat, obs, _agent_rew_unused, ato = rl.stepSendActGetObs(action, timeout=timeout)
        t_wall = time.time()
        rew = _compute_reward(obs)
        terminated = _goal_reached(obs)
        total_rew += rew
        print(
            "[RL] step={:02d} action={} lat={:.6f}s ato={:.6f} t_wall={:.6f} "
            "obs={} rew_rl={:.6f} terminated={}".format(
                step_idx,
                action,
                lat,
                ato,
                t_wall,
                obs,
                rew,
                terminated,
            )
        )
        if terminated:
            print("[RL] goal reached -> ending episode")
            break

    rl.stepExpFinished(timeout=timeout)
    print(f"[RL] finished -> total_reward={total_rew:.6f}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""

    parser = argparse.ArgumentParser(
        description="RL side of the first-order plant example"
    )
    parser.add_argument(
        "--port", type=int, default=49054, help="TCP port the RL side listens on"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="maximum number of actions to send"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="seconds to wait for each agent reply",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_episode(host_port=args.port, num_steps=args.steps, timeout=args.timeout)
