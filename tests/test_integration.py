import socket
import threading
import time

import pytest

from spindecoupler import AgentSide, BaseCommPoint, RLSide


def connect_agent_with_retry(ip, port, attempts=50, delay=0.05):
	last_error = None
	for _ in range(attempts):
		try:
			return AgentSide(ip, port)
		except RuntimeError as exc:
			last_error = exc
			time.sleep(delay)
	raise last_error


def test_rl_agent_full_cycle_over_tcp(free_tcp_port, force_localhost):
	results = {}
	errors = []
	actions = [
		{"move": "left"},
		{"move": "right"},
		{"move": "stop"},
	]

	def rl_worker():
		try:
			rl = RLSide(free_tcp_port, verbose=True)
			results["reset"] = rl.resetGetObs(timeout=2.0)
			results["steps"] = [
				rl.stepSendActGetObs(action, timeout=2.0)
				for action in actions
			]
			rl.stepExpFinished(timeout=2.0)
		except BaseException as exc:
			errors.append(exc)

	rl_thread = threading.Thread(target=rl_worker, daemon=True)
	rl_thread.start()

	agent = connect_agent_with_retry(BaseCommPoint.get_ip(), free_tcp_port)
	agent._verbose = True
	received_actions = []
	step_index = 0
	deadline = time.time() + 5.0

	while time.time() < deadline:
		command = agent.readWhatToDo(timeout=0.2)
		if command is None:
			time.sleep(0.01)
			continue

		what, payload = command
		if what == AgentSide.WhatToDo.RESET_SEND_OBS:
			agent.resetSendObs({"phase": "reset"}, agenttime=10.0)
		elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
			received_actions.append(payload)
			lat = 0.25 + step_index
			agent.stepSendLastActDur(lat)
			agent.stepSendObs(
				{"echo": payload, "index": step_index},
				agenttime=20.0 + step_index,
				rew=30.0 + step_index,
			)
			step_index += 1
		elif what == AgentSide.WhatToDo.FINISH:
			break

	rl_thread.join(timeout=5.0)
	agent._rlcomm.end()

	assert not rl_thread.is_alive(), "RL thread did not finish"
	assert errors == []
	assert results["reset"] == ({"phase": "reset"}, 10.0)
	assert received_actions == actions
	assert results["steps"] == [
		(0.25, {"echo": {"move": "left"}, "index": 0}, 30.0, 20.0),
		(1.25, {"echo": {"move": "right"}, "index": 1}, 31.0, 21.0),
		(2.25, {"echo": {"move": "stop"}, "index": 2}, 32.0, 22.0),
	]


def test_rl_side_raises_after_abrupt_agent_disconnect(free_tcp_port, force_localhost):
	results = {}

	def rl_worker():
		rl = RLSide(free_tcp_port)
		with pytest.raises(RuntimeError, match="after-reset observation") as excinfo:
			rl.resetGetObs(timeout=0.5)
		results["error"] = str(excinfo.value)

	rl_thread = threading.Thread(target=rl_worker, daemon=True)
	rl_thread.start()

	agent = connect_agent_with_retry(BaseCommPoint.get_ip(), free_tcp_port)
	agent._rlcomm.end()
	rl_thread.join(timeout=3.0)

	assert not rl_thread.is_alive(), "RL thread did not stop after disconnect"
	assert "Error reading after-reset observation from the agent." in results["error"]


def test_server_commpoint_reports_malformed_network_payload(free_tcp_port, force_localhost):
	results = {}
	ready = threading.Event()

	def low_level_worker():
		from rl_spin_decoupler.socketcomms.comms import ServerCommPoint

		server = ServerCommPoint(free_tcp_port)
		ready.set()
		assert server.begin(1.0) == ""
		results["read"] = server.readData(timeout=0.5)
		server.end()

	thread = threading.Thread(target=low_level_worker, daemon=True)
	thread.start()
	assert ready.wait(timeout=1.0)

	with socket.create_connection(("127.0.0.1", free_tcp_port), timeout=1.0) as sock:
		sock.sendall(b"not-a-valid-pickle")

	thread.join(timeout=3.0)

	assert not thread.is_alive(), "Server thread did not finish"
	assert results["read"][0]
	assert results["read"][1] is None