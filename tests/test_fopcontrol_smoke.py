# SPDX-License-Identifier: GPL-3.0-only

import subprocess
import sys
from pathlib import Path

from spindecoupler import BaseCommPoint


def _terminate_process(proc: subprocess.Popen) -> None:
	"""Terminate a subprocess best-effort without hanging the test."""

	if proc.poll() is not None:
		return
	proc.terminate()
	try:
		proc.wait(timeout=2)
	except subprocess.TimeoutExpired:
		proc.kill()
		proc.wait(timeout=2)


def test_fopcontrol_example_smoke(free_tcp_port):
	"""Run first-order control example as real processes and verify completion."""

	repo_root = Path(__file__).resolve().parents[1]
	rl_script = repo_root / "examples" / "first_order_plant_control" / "rl_side_fopcontrol.py"
	agent_script = repo_root / "examples" / "first_order_plant_control" / "agent_side_fopcontrol.py"
	host_ip = BaseCommPoint.get_ip()

	rl_cmd = [
		sys.executable,
		str(rl_script),
		"--port",
		str(free_tcp_port),
		"--steps",
		"3",
		"--timeout",
		"2.0",
	]
	agent_cmd = [
		sys.executable,
		str(agent_script),
		"--ip",
		host_ip,
		"--port",
		str(free_tcp_port),
		"--rl-step-period",
		"0.08",
		"--control-period",
		"0.01",
		"--timeout",
		"0.5",
	]

	rl_proc = subprocess.Popen(
		rl_cmd,
		cwd=repo_root,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)
	try:
		agent = subprocess.run(
			agent_cmd,
			cwd=repo_root,
			capture_output=True,
			text=True,
			timeout=20,
		)
		rl_out, _ = rl_proc.communicate(timeout=20)
	finally:
		_terminate_process(rl_proc)

	assert agent.returncode == 0, agent.stdout + "\n" + agent.stderr
	assert "finish command received" in agent.stdout.lower()

	assert "[RL] reset" in rl_out
	assert "[RL] step=00" in rl_out
	assert "[RL] finished" in rl_out
