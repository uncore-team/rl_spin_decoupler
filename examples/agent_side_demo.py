# SPDX-License-Identifier: GPL-3.0-only

"""Executable agent-side demo for rl_spin_decoupler.

Run this process after examples/rl_side_demo.py is already listening.
The agent simulates a fast control loop and reports observations plus reward.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum

from spindecoupler import AgentSide, BaseCommPoint


class StepState(Enum):
	"""Internal state machine for the simulated agent loop."""

	READY_FOR_RL_COMMAND = 0
	EXECUTING_LAST_ACTION = 1
	AFTER_RESET = 2


@dataclass
class ToyPlant:
	"""Tiny first-order plant advanced by the control loop.

	The plant state follows:
		x[t+1] = x[t] + alpha * (u - x[t])
	where ``u`` is the current action target.
	"""

	state: float = 0.0
	alpha: float = 0.12

	def reset(self) -> None:
		"""Reset the plant state for a new episode."""

		self.state = 0.0

	def apply_target(self, target: float) -> None:
		"""Advance one control tick towards the target."""

		self.state = self.state + self.alpha * (target - self.state)


class SimulatedAgent:
	"""High-rate agent loop that communicates with the RL-side endpoint."""

	def __init__(
		self, ip: str, port: int, rl_step_period: float, control_period: float
	):
		if rl_step_period <= control_period:
			raise ValueError(
				"rl_step_period must be strictly greater than control_period"
			)

		self._comm = AgentSide(ip, port, verbose=True)
		self._plant = ToyPlant()
		self._rl_step_period = rl_step_period
		self._control_period = control_period
		self._state = StepState.READY_FOR_RL_COMMAND
		self._last_action_start = time.time()
		self._current_target = 0.0
		self._current_gain = 0.25

	def _build_observation(self) -> dict:
		"""Build an observation payload sent to RL."""

		return {
			"plant_state": self._plant.state,
			"target": self._current_target,
			"gain": self._current_gain,
		}

	def _compute_reward(self) -> float:
		"""Compute a simple reward that prefers the state near zero."""

		return -abs(self._plant.state)

	def _apply_control_tick(self) -> None:
		"""Apply one fast control update."""

		self._plant.apply_target(self._current_target)

	def spin(self, timeout: float = 1.0) -> None:
		"""Run the agent loop until FINISH is received."""

		print("[AGENT] running control loop...")
		while True:
			now = time.time()
			self._apply_control_tick()

			if self._state == StepState.EXECUTING_LAST_ACTION:
				if now - self._last_action_start >= self._rl_step_period:
					obs = self._build_observation()
					rew = self._compute_reward()
					self._comm.stepSendObs(obs, agenttime=now, rew=rew)
					self._state = StepState.READY_FOR_RL_COMMAND

			elif self._state == StepState.AFTER_RESET:
				self._comm.resetSendObs(self._build_observation(), agenttime=now)
				self._state = StepState.READY_FOR_RL_COMMAND

			elif self._state == StepState.READY_FOR_RL_COMMAND:
				command = self._comm.readWhatToDo(timeout=timeout)
				if command is not None:
					what, payload = command
					if what == AgentSide.WhatToDo.RESET_SEND_OBS:
						self._plant.reset()
						self._current_target = 0.0
						self._last_action_start = now
						self._state = StepState.AFTER_RESET
					elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
						lat = now - self._last_action_start
						self._comm.stepSendLastActDur(lat)
						self._last_action_start = now
						self._current_target = float(payload.get("target", 0.0))
						self._current_gain = float(payload.get("gain", 0.25))
						self._plant.alpha = max(0.01, min(0.9, self._current_gain))
						self._state = StepState.EXECUTING_LAST_ACTION
					elif what == AgentSide.WhatToDo.FINISH:
						print("[AGENT] finish command received")
						return

			time.sleep(self._control_period)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the agent-side demo."""

	parser = argparse.ArgumentParser(
		description="Agent-side control-loop demo for rl_spin_decoupler"
	)
	parser.add_argument(
		"--ip", default=BaseCommPoint.get_ip(), help="RL-side IPv4 address"
	)
	parser.add_argument("--port", type=int, default=49054, help="RL-side TCP port")
	parser.add_argument(
		"--rl-step-period",
		type=float,
		default=0.2,
		help="Seconds each RL action should remain active on the agent",
	)
	parser.add_argument(
		"--control-period",
		type=float,
		default=0.02,
		help="Fast control-loop period in seconds",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		default=1.0,
		help="Timeout used by readWhatToDo() when data is pending",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	agent = SimulatedAgent(
		ip=args.ip,
		port=args.port,
		rl_step_period=args.rl_step_period,
		control_period=args.control_period,
	)
	agent.spin(timeout=args.timeout)
