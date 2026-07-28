# SPDX-License-Identifier: GPL-3.0-only

"""Executable agent-side demo for rl_spin_decoupler.

Run this process after examples/first_order_plant_control/rl_side_fopcontrol.py is already listening.
The agent simulates a fast control loop and reports observations only.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum

from spindecoupler import AgentSide, BaseCommPoint


@dataclass
class AgentVizRecord:
	"""Snapshot of agent-side dynamics for the optional live visualizer."""

	r_tick: int
	plant_state: float
	target: float
	alpha: float
	last_action_age: float


class AgentVisualizer:
	"""Matplotlib UI that displays control-loop evolution on the agent side."""

	def __init__(self, refresh_every: int = 5):
		if refresh_every <= 0:
			raise ValueError("refresh_every must be > 0")

		self._refresh_every = refresh_every
		self._records: list[AgentVizRecord] = []
		self._draw_count = 0

		try:
			import matplotlib.pyplot as plt
		except Exception as exc:  # pragma: no cover - runtime dependency
			raise RuntimeError(
				"Matplotlib is required for --plot. Install it with: pip install matplotlib"
			) from exc

		self._plt = plt
		self._plt.ion()
		self._fig, (self._ax_state, self._ax_timing) = self._plt.subplots(2, 1, figsize=(10, 7))
		self._fig.suptitle("Agent-side first-order plant monitor")

	def update(self, record: AgentVizRecord, step_state: StepState) -> None:
		"""Append a sample and refresh the figure at the configured cadence."""

		self._records.append(record)
		self._draw_count += 1
		if self._draw_count % self._refresh_every != 0:
			return

		ticks = [r.r_tick for r in self._records]
		plant = [r.plant_state for r in self._records]
		target = [r.target for r in self._records]
		alpha = [r.alpha for r in self._records]
		age_ms = [1_000.0 * r.last_action_age for r in self._records]

		self._ax_state.clear()
		self._ax_state.plot(ticks, plant, label="plant_state", linewidth=2.0)
		self._ax_state.plot(ticks, target, label="target", linestyle="--", linewidth=1.8)
		self._ax_state.plot(ticks, alpha, label="gain(alpha)", linewidth=1.4, alpha=0.9)
		self._ax_state.set_ylabel("State / Gain")
		self._ax_state.grid(True, alpha=0.25)
		self._ax_state.legend(loc="best")

		self._ax_timing.clear()
		self._ax_timing.plot(ticks, age_ms, label="time since last RL action (ms)", color="tab:orange")
		self._ax_timing.set_xlabel("Control tick")
		self._ax_timing.set_ylabel("Timing")
		self._ax_timing.grid(True, alpha=0.25)
		self._ax_timing.legend(loc="best")

		self._fig.suptitle(
			"Agent-side first-order plant monitor | "
			f"mode={step_state.name} | target={record.target:.3f}"
		)
		self._fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
		self._plt.pause(0.001)

	def close(self, hold_window: bool) -> None:
		"""Finalize the plot and optionally keep the window open."""

		if hold_window:
			self._plt.ioff()
			self._plt.show()
		else:
			self._plt.close(self._fig)


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
		self,
		ip: str,
		port: int,
		rl_step_period: float,
		control_period: float,
		plot: bool,
		plot_refresh: int,
		plot_hold: bool,
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
		self._tick_counter = 0
		self._plot_hold = plot_hold
		self._visualizer = AgentVisualizer(refresh_every=plot_refresh) if plot else None

	def _build_observation(self) -> dict:
		"""Build an observation payload sent to RL."""

		return {
			"plant_state": self._plant.state,
			"target": self._current_target,
			"gain": self._current_gain,
		}

	def _apply_control_tick(self) -> None:
		"""Apply one fast control update."""

		self._plant.apply_target(self._current_target)
		self._tick_counter += 1

	def _update_visualization(self, now: float) -> None:
		"""Refresh optional GUI with latest local plant/control-loop values."""

		if self._visualizer is None:
			return
		record = AgentVizRecord(
			r_tick=self._tick_counter,
			plant_state=self._plant.state,
			target=self._current_target,
			alpha=self._plant.alpha,
			last_action_age=now - self._last_action_start,
		)
		self._visualizer.update(record, step_state=self._state)

	def spin(self, timeout: float = 1.0) -> None:
		"""Run the agent loop until FINISH is received."""

		print("[AGENT] running control loop...")
		try:
			while True:
				now = time.time()
				self._apply_control_tick()
				self._update_visualization(now)

				if self._state == StepState.EXECUTING_LAST_ACTION:
					if now - self._last_action_start >= self._rl_step_period:
						obs = self._build_observation()
						self._comm.stepSendObs(obs, agenttime=now)
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
		finally:
			if self._visualizer is not None:
				self._visualizer.close(hold_window=self._plot_hold)


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
	parser.add_argument(
		"--plot",
		action="store_true",
		help="Show a live Matplotlib GUI for agent-side control dynamics",
	)
	parser.add_argument(
		"--plot-refresh",
		type=int,
		default=5,
		help="Redraw GUI every N control ticks (used with --plot)",
	)
	parser.add_argument(
		"--no-plot-hold",
		action="store_true",
		help="Close plot automatically when FINISH is received",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	agent = SimulatedAgent(
		ip=args.ip,
		port=args.port,
		rl_step_period=args.rl_step_period,
		control_period=args.control_period,
		plot=args.plot,
		plot_refresh=args.plot_refresh,
		plot_hold=not args.no_plot_hold,
	)
	agent.spin(timeout=args.timeout)
