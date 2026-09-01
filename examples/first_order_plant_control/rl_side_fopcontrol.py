# SPDX-License-Identifier: GPL-3.0-only

"""Executable RL-side demo for rl_spin_decoupler.

Run this process first, then start
examples/first_order_plant_control/agent_side_fopcontrol.py in another shell.
The script drives a short episode and prints LAT/ATO/t_wall per step.

Reward and task-level termination are computed HERE, on the RL side, from the
observation received over the transport (see reward.py). The agent only
transports observations, timing, and physics/hardware termination flags (always
``False`` for this synthetic plant).
"""

from __future__ import annotations

import argparse
import time

from reward import compute_reward, is_terminated
from spindecoupler import RLSide


def _unwrap(payload):
    """Return the inner observation dict and the physics termination flags.

    The agent wraps every observation as
    ``{"observation": <obs>, "terminated": bool, "truncated": bool}`` so the
    payload layout is identical across all examples.
    """

    if isinstance(payload, dict) and "observation" in payload:
        return (
            payload["observation"],
            bool(payload.get("terminated", False)),
            bool(payload.get("truncated", False)),
        )
    # Backwards-compatible fallback: a bare observation dict.
    return payload, False, False


def run_episode(
    host_port: int,
    num_steps: int,
    timeout: float,
) -> None:
    """Run a short RL loop against the agent-side demo.

    Args:
            host_port: TCP port where the RL-side server listens.
            num_steps: Number of actions sent after reset.
            timeout: Communication timeout per operation in seconds.
    """

    rl = RLSide(host_port, verbose=True)
    payload0, ato0 = rl.resetGetObs(timeout=timeout)
    obs, _term, _trunc = _unwrap(payload0)
    prev_obs = obs
    print(f"[RL] reset -> obs={obs} ato={ato0:.6f}")

    total_rew = 0.0
    for step_idx in range(num_steps):
        action: dict[str, float] = {
            "target": 0.8 if step_idx % 2 == 0 else -0.4,
            "gain": 0.25,
        }
        # The transported reward slot is unused: reward is computed here.
        lat, payload, _rew_unused, ato = rl.stepSendActGetObs(
            action, timeout=timeout
        )
        obs, term_phys, trunc_phys = _unwrap(payload)
        t_wall = time.time()

        rew = compute_reward(obs, action, prev_obs, lat)
        terminated = term_phys or is_terminated(obs)
        truncated = trunc_phys
        total_rew += rew
        prev_obs = obs

        print(
            "[RL] step={:02d} action={} lat={:.6f}s ato={:.6f} t_wall={:.6f} "
            "obs={} rew={:.6f} terminated={} truncated={}".format(
                step_idx,
                action,
                lat,
                ato,
                t_wall,
                obs,
                rew,
                terminated,
                truncated,
            )
        )
        if terminated or truncated:
            print("[RL] episode boundary reached -> ending episode")
            break

    rl.stepExpFinished(timeout=timeout)
    print(f"[RL] finished -> total_reward={total_rew:.6f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""

    parser = argparse.ArgumentParser(description="RL-side demo for rl_spin_decoupler")
    parser.add_argument(
        "--port", type=int, default=49054, help="TCP port for RL-side server"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of RL actions to send"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Timeout in seconds for each communication operation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_episode(host_port=args.port, num_steps=args.steps, timeout=args.timeout)
