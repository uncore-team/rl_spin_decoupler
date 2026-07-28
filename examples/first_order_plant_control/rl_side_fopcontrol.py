# SPDX-License-Identifier: GPL-3.0-only

"""Executable RL-side demo for rl_spin_decoupler.

Run this process first, then start examples/first_order_plant_control/agent_side_fopcontrol.py in another shell.
The script drives a short episode and prints LAT/ATO/t_wall per step.
"""

from __future__ import annotations

import argparse
import time

from spindecoupler import RLSide


def _compute_reward(obs: dict) -> float:
	"""Compute reward on RL side from observation only."""

	return -abs(float(obs.get("plant_state", 0.0)))


def _goal_reached(obs: dict, tolerance: float = 0.05) -> bool:
	"""Decide RL-side success condition from observation only."""

	return abs(float(obs.get("plant_state", 0.0))) <= tolerance


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
