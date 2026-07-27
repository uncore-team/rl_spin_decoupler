# SPDX-License-Identifier: GPL-3.0-only

import pickle
import socket
import struct

import pytest
from rl_spin_decoupler.socketcomms.comms import ClientCommPoint, ServerCommPoint

import spindecoupler as spindecoupler_module
from spindecoupler import AgentSide, BaseCommPoint, RLSide


class FakeSocket:
	def __init__(self, recv_payloads=None, send_error=None, recv_error=None):
		self.recv_payloads = list(recv_payloads or [])
		self.send_error = send_error
		self.recv_error = recv_error
		self.sent_payloads = []
		self.timeout_history = []
		self.closed = False

	def send(self, data):
		if self.send_error is not None:
			raise self.send_error
		self.sent_payloads.append(data)

	def sendall(self, data):
		self.send(data)

	def recv(self, _size):
		if self.recv_error is not None:
			raise self.recv_error
		if self.recv_payloads:
			return self.recv_payloads.pop(0)
		return b""

	def settimeout(self, value):
		self.timeout_history.append(value)

	def close(self):
		self.closed = True


class FakeIPSocket:
	def __init__(self, *, ip="192.168.1.44", connect_error=None):
		self.ip = ip
		self.connect_error = connect_error
		self.closed = False
		self.timeout = None

	def settimeout(self, value):
		self.timeout = value

	def connect(self, _address):
		if self.connect_error is not None:
			raise self.connect_error

	def getsockname(self):
		return (self.ip, 12345)

	def close(self):
		self.closed = True


class FakeServerSocket:
	def __init__(self, accepted_socket=None, accept_error=None):
		self.accepted_socket = accepted_socket or FakeSocket()
		self.accept_error = accept_error
		self.timeout_history = []

	def settimeout(self, value):
		self.timeout_history.append(value)

	def accept(self):
		if self.accept_error is not None:
			raise self.accept_error
		return self.accepted_socket, ("127.0.0.1", 40000)

	def close(self):
		return None


class FakeSocketFactory:
	def __init__(self, sock):
		self.sock = sock

	def __call__(self, *_args, **_kwargs):
		return self.sock


class FakeComm:
	def __init__(self, *, check=True, read_results=None, send_results=None):
		self.check = check
		self.read_results = list(read_results or [])
		self.send_results = list(send_results or [])
		self.sent = []

	def checkDataToRead(self):
		return self.check

	def readData(self, timeout):
		if not self.read_results:
			raise AssertionError("No more read results configured")
		return self.read_results.pop(0)

	def sendData(self, data):
		self.sent.append(data)
		if self.send_results:
			return self.send_results.pop(0)
		return ""

	def end(self):
		return ""


def make_base_point():
	point = BaseCommPoint(kind=BaseCommPoint.Kind.SERVER)
	point._begun = True
	return point


def framed(payload):
	body = pickle.dumps(payload)
	return [struct.pack("!I", len(body)), body]


def make_agent_side(fake_comm):
	agent = object.__new__(AgentSide)
	agent._verbose = False
	agent._rlcomm = fake_comm
	return agent


def make_rl_side(fake_comm):
	rl = object.__new__(RLSide)
	rl._verbose = False
	rl._rlcomm = fake_comm
	return rl


def test_base_commpoint_validates_constructor_arguments():
	with pytest.raises(TypeError):
		BaseCommPoint(kind="server")
	with pytest.raises(ValueError):
		BaseCommPoint(kind=BaseCommPoint.Kind.SERVER, datachunkmaxsize=0)
	with pytest.raises(ValueError):
		BaseCommPoint(kind=BaseCommPoint.Kind.SERVER, port=19999)
	with pytest.raises(ValueError):
		BaseCommPoint(kind=BaseCommPoint.Kind.SERVER, ipv4="bad-ip")


def test_base_commpoint_prevents_copy():
	with pytest.raises(NotImplementedError):
		BaseCommPoint(kind=BaseCommPoint.Kind.SERVER).__copy__()


def test_print_info_and_set_debug(capsys):
	point = BaseCommPoint(kind=BaseCommPoint.Kind.SERVER)
	assert point._debug is False
	point.setDebug(True)
	assert point._debug is True
	point._printInfo("hello")
	assert "hello" in capsys.readouterr().out


def test_get_ip_returns_socket_address(monkeypatch):
	fake_socket = FakeIPSocket(ip="10.0.0.8")
	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.socket.socket",
		FakeSocketFactory(fake_socket),
	)

	assert BaseCommPoint.get_ip() == "10.0.0.8"
	assert fake_socket.closed is True


def test_get_ip_falls_back_to_loopback(monkeypatch):
	fake_socket = FakeIPSocket(connect_error=OSError("unreachable"))
	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.socket.socket",
		FakeSocketFactory(fake_socket),
	)

	assert BaseCommPoint.get_ip() == "127.0.0.1"
	assert fake_socket.closed is True


def test_send_data_serializes_dictionary():
	point = make_base_point()
	point._sock = FakeSocket()
	payload = {"stepkind": "reset", "nested": {"obs": 1}}

	assert point.sendData(payload) == ""
	frame = point._sock.sent_payloads[0]
	assert struct.unpack("!I", frame[:4])[0] == len(frame[4:])
	assert pickle.loads(frame[4:]) == payload


def test_send_data_wraps_socket_errors():
	point = make_base_point()
	point._sock = FakeSocket(send_error=OSError("boom"))

	assert "boom" in point.sendData({"x": 1})


def test_send_data_debug_path(monkeypatch):
	point = make_base_point()
	point._sock = FakeSocket()
	point._debug = True
	messages = []
	point._printInfo = messages.append

	assert point.sendData({"x": 1}) == ""
	assert messages == ["Sending 21 bytes...", "\tSent ok."]


def test_send_data_requires_begun_connection():
	point = BaseCommPoint(kind=BaseCommPoint.Kind.SERVER)

	with pytest.raises(RuntimeError, match="not-begun"):
		point.sendData({"x": 1})


def test_read_data_deserializes_and_resets_timeout():
	point = make_base_point()
	point._sock = FakeSocket(recv_payloads=framed({"obs": 7}))

	res, data = point.readData(timeout=1.5)

	assert res == ""
	assert data == {"obs": 7}
	assert point._sock.timeout_history == [1.5, None]


def test_read_data_with_non_positive_timeout_uses_blocking_mode():
	point = make_base_point()
	point._sock = FakeSocket(recv_payloads=framed({"obs": 3}))

	res, data = point.readData(timeout=0.0)

	assert res == ""
	assert data == {"obs": 3}
	assert point._sock.timeout_history == [None, None]


def test_read_data_reports_connection_closed():
	point = make_base_point()
	point._sock = FakeSocket(recv_payloads=[b""])

	res, data = point.readData(timeout=0.5)

	assert "Connection closed while receiving" in res
	assert data is None


def test_read_data_debug_path():
	point = make_base_point()
	point._sock = FakeSocket(recv_payloads=framed({"obs": 9}))
	point._debug = True
	messages = []
	point._printInfo = messages.append

	assert point.readData(timeout=0.5) == ("", {"obs": 9})
	assert messages == ["Receiving...", "\tReceived 23 bytes."]


def test_read_data_reports_malformed_payload():
	point = make_base_point()
	point._sock = FakeSocket(recv_payloads=[struct.pack("!I", 12), b"not-a-pickle"])

	res, data = point.readData(timeout=0.5)

	assert res
	assert data is None


def test_read_data_requires_begun_connection():
	point = BaseCommPoint(kind=BaseCommPoint.Kind.SERVER)

	with pytest.raises(RuntimeError, match="not-begun"):
		point.readData(timeout=1.0)


def test_check_data_to_read_reports_select_result(monkeypatch):
	point = make_base_point()
	point._sock = FakeSocket()

	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.select.select",
		lambda *_args, **_kwargs: ([point._sock], [], []),
	)
	assert point.checkDataToRead() is True

	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.select.select",
		lambda *_args, **_kwargs: ([], [], []),
	)
	assert point.checkDataToRead() is False


def test_check_data_to_read_debug_path(monkeypatch):
	point = make_base_point()
	point._sock = FakeSocket()
	point._debug = True
	messages = []
	point._printInfo = messages.append
	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.select.select",
		lambda *_args, **_kwargs: ([], [], []),
	)

	assert point.checkDataToRead() is False
	assert messages == ["Peeking..."]


def test_check_data_to_read_requires_begun_connection():
	point = BaseCommPoint(kind=BaseCommPoint.Kind.SERVER)

	with pytest.raises(RuntimeError, match="not-begun"):
		point.checkDataToRead()


def test_server_begin_returns_timeout_without_connection():
	server = object.__new__(ServerCommPoint)
	server._begun = False
	server._basesock = FakeServerSocket(accept_error=socket.timeout())

	assert server.begin(0.5) == "timeout"
	assert server._basesock.timeout_history == [0.5, None]


def test_server_begin_returns_other_accept_errors():
	server = object.__new__(ServerCommPoint)
	server._begun = False
	server._basesock = FakeServerSocket(accept_error=OSError("accept failed"))

	assert "accept failed" in server.begin(0.5)


def test_server_constructor_retries_busy_port(monkeypatch, capsys):
	state = {"bind_calls": 0}

	class RetrySocket(FakeSocket):
		def bind(self, _address):
			state["bind_calls"] += 1
			if state["bind_calls"] == 1:
				error = OSError("in use")
				error.errno = socket.errno.EADDRINUSE
				raise error

		def listen(self, backlog):
			self.backlog = backlog

	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.socket.socket",
		FakeSocketFactory(RetrySocket()),
	)
	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.time.sleep", lambda _seconds: None
	)
	monkeypatch.setattr(BaseCommPoint, "get_ip", classmethod(lambda cls: "127.0.0.1"))

	server = ServerCommPoint(24000)

	assert state["bind_calls"] == 2
	assert server._basesock.backlog == 1
	assert "Retrying in 13 secs (1)" in capsys.readouterr().out


def test_server_constructor_raises_non_retryable_socket_error(monkeypatch, capsys):
	class BrokenBindSocket(FakeSocket):
		def bind(self, _address):
			error = OSError("bad bind")
			error.errno = 999
			raise error

		def listen(self, _backlog):
			raise AssertionError("listen should not be called")

	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.socket.socket",
		FakeSocketFactory(BrokenBindSocket()),
	)
	monkeypatch.setattr(BaseCommPoint, "get_ip", classmethod(lambda cls: "127.0.0.1"))

	with pytest.raises(OSError, match="bad bind"):
		ServerCommPoint(24000)

	assert "Socket error: bad bind" in capsys.readouterr().out


def test_server_constructor_aborts_after_too_many_busy_retries(monkeypatch, capsys):
	state = {"bind_calls": 0}

	class AlwaysBusySocket(FakeSocket):
		def bind(self, _address):
			state["bind_calls"] += 1
			error = OSError("still in use")
			error.errno = socket.errno.EADDRINUSE
			raise error

		def listen(self, _backlog):
			raise AssertionError("listen should not be called")

	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.socket.socket",
		FakeSocketFactory(AlwaysBusySocket()),
	)
	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.time.sleep", lambda _seconds: None
	)
	monkeypatch.setattr(BaseCommPoint, "get_ip", classmethod(lambda cls: "127.0.0.1"))

	with pytest.raises(OSError, match="still in use"):
		ServerCommPoint(24000)

	assert state["bind_calls"] == 11
	out = capsys.readouterr().out
	assert "Too many tries. Aborting" in out
	assert "Retrying in 13 secs (10)" in out


def test_server_begin_validates_positive_timeout():
	server = object.__new__(ServerCommPoint)
	server._begun = False
	server._basesock = FakeServerSocket()
	server.end = lambda: ""

	with pytest.raises(ValueError, match="Timeoutaccept must be > 0.0"):
		server.begin(0.0)


def test_server_begin_accepts_connection_and_end_closes_socket():
	accepted = FakeSocket()
	server = object.__new__(ServerCommPoint)
	server._begun = False
	server._basesock = FakeServerSocket(accepted_socket=accepted)

	assert server.begin(1.0) == ""
	assert server._begun is True
	assert server._sock is accepted
	assert server.end() == ""
	assert accepted.closed is True
	assert server._begun is False


def test_server_end_returns_close_error():
	accepted = FakeSocket()

	def broken_close():
		raise OSError("close failed")

	accepted.close = broken_close
	server = object.__new__(ServerCommPoint)
	server._begun = True
	server._sock = accepted

	assert "close failed" in server.end()


def test_server_and_client_string_representations():
	server = object.__new__(ServerCommPoint)
	server._servip = "127.0.0.1"
	server._port = 24000
	server._begun = False
	assert "Server listening at 127.0.0.1:24000" in str(server)

	client = object.__new__(ClientCommPoint)
	client._myip = "127.0.0.1"
	client._ipv4 = "127.0.0.1"
	client._port = 24001
	client._begun = True
	assert "Client at 127.0.0.1 to connect to 127.0.0.1:24001" in str(client)


def test_client_begin_and_end(monkeypatch):
	fake_socket = FakeSocket()
	fake_socket.connect_calls = []

	def connect(address):
		fake_socket.connect_calls.append(address)

	fake_socket.connect = connect
	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.socket.socket",
		FakeSocketFactory(fake_socket),
	)

	client = ClientCommPoint("127.0.0.1", 24000)

	assert client.begin() == ""
	assert fake_socket.connect_calls[-1] == ("127.0.0.1", 24000)
	assert client.end() == ""
	assert fake_socket.closed is True


def test_client_begin_reports_connection_error(monkeypatch):
	fake_socket = FakeSocket()

	def connect(_address):
		raise OSError("connect failed")

	fake_socket.connect = connect
	monkeypatch.setattr(
		"rl_spin_decoupler.socketcomms.comms.socket.socket",
		FakeSocketFactory(fake_socket),
	)

	client = ClientCommPoint("127.0.0.1", 24000)

	assert "connect failed" in client.begin()


def test_client_end_returns_close_error():
	client = object.__new__(ClientCommPoint)
	client._begun = True
	client._sock = FakeSocket()

	def broken_close():
		raise OSError("client close failed")

	client._sock.close = broken_close

	assert "client close failed" in client.end()


def test_rlside_init_verbose_and_error_path(monkeypatch, capsys):
	class FakeServerCommPoint:
		def __init__(self, port):
			self.port = port

		def begin(self, timeoutaccept):
			assert timeoutaccept == 60.0
			return "timeout"

		def end(self):
			return ""

	monkeypatch.setattr(spindecoupler_module, "ServerCommPoint", FakeServerCommPoint)

	with pytest.raises(RuntimeError, match="No agent connection: timeout"):
		RLSide(24000, verbose=True)

	assert (
		"RL decoupler enabled. Waiting for agent connection..."
		in capsys.readouterr().out
	)


def test_agentside_init_verbose_and_error_path(monkeypatch, capsys):
	class FakeClientCommPoint:
		def __init__(self, ip, port):
			self.ip = ip
			self.port = port

		def begin(self):
			return "connect failed"

		def end(self):
			return ""

	monkeypatch.setattr(spindecoupler_module, "ClientCommPoint", FakeClientCommPoint)

	with pytest.raises(
		RuntimeError, match="Error starting connection with RL. connect failed"
	):
		AgentSide("127.0.0.1", 24000, verbose=True)

	assert "Agent decoupler enabled." in capsys.readouterr().out


def test_agentside_init_verbose_success_path(monkeypatch, capsys):
	class FakeClientCommPoint:
		def __init__(self, ip, port):
			self.ip = ip
			self.port = port

		def begin(self):
			return ""

		def end(self):
			return ""

	monkeypatch.setattr(spindecoupler_module, "ClientCommPoint", FakeClientCommPoint)

	agent = AgentSide("127.0.0.1", 24000, verbose=True)

	out = capsys.readouterr().out
	assert "Agent decoupler enabled." in out
	assert "Agent decoupler connected to RL decoupler" in out
	AgentSide.__del__(agent)


def test_agentside_read_what_to_do_returns_none_when_no_data():
	agent = make_agent_side(FakeComm(check=False))

	assert agent.readWhatToDo() is None


@pytest.mark.parametrize(
	"indicator, expected",
	[
		(
			{"stepkind": "step", "action": {"a": 1}},
			(AgentSide.WhatToDo.REC_ACTION_SEND_OBS, {"a": 1}),
		),
		({"stepkind": "reset"}, (AgentSide.WhatToDo.RESET_SEND_OBS, None)),
		({"stepkind": "finish"}, (AgentSide.WhatToDo.FINISH, None)),
	],
)
def test_agentside_read_what_to_do_maps_commands(indicator, expected):
	fake_comm = FakeComm(check=True, read_results=[("", indicator)])
	agent = make_agent_side(fake_comm)

	assert agent.readWhatToDo(timeout=0.75) == expected


def test_agentside_read_what_to_do_wraps_comm_errors():
	fake_comm = FakeComm(check=True, read_results=[("broken", None)])
	agent = make_agent_side(fake_comm)

	with pytest.raises(RuntimeError, match="Error receiving what-to-do"):
		agent.readWhatToDo(timeout=0.25)


def test_agentside_read_what_to_do_rejects_unknown_indicator():
	fake_comm = FakeComm(check=True, read_results=[("", {"stepkind": "other"})])
	agent = make_agent_side(fake_comm)

	with pytest.raises(ValueError, match="Unknown what-to-do indicator"):
		agent.readWhatToDo()


def test_agentside_send_helpers_wrap_errors():
	fake_comm = FakeComm(send_results=["lat failed", "obs failed", "reset failed"])
	agent = make_agent_side(fake_comm)

	with pytest.raises(RuntimeError, match="lat to RL"):
		agent.stepSendLastActDur(0.5)
	with pytest.raises(RuntimeError, match="observation/reward to RL"):
		agent.stepSendObs({"obs": 1}, agenttime=1.0, rew=2.0)
	with pytest.raises(RuntimeError, match="observation to RL"):
		agent.resetSendObs({"obs": 2}, agenttime=3.0)


def test_agentside_send_helpers_emit_expected_payloads():
	fake_comm = FakeComm()
	agent = make_agent_side(fake_comm)

	agent.stepSendLastActDur(0.25)
	agent.stepSendObs({"obs": 1}, agenttime=2.0, rew=4.0)
	agent.resetSendObs({"obs": 2}, agenttime=3.0)

	assert fake_comm.sent == [
		{"lat": 0.25},
		{"obs": {"obs": 1}, "rew": 4.0, "ato": 2.0},
		{"obs": {"obs": 2}, "ato": 3.0},
	]


def test_rlside_reset_get_obs_and_step_send_act_get_obs():
	fake_comm = FakeComm(
		read_results=[
			("", {"obs": {"state": "reset"}, "ato": 1.5}),
			("", {"lat": 0.2}),
			("", {"obs": {"state": "step"}, "rew": 3.0, "ato": 2.5}),
		]
	)
	rl = make_rl_side(fake_comm)

	assert rl.resetGetObs(timeout=0.4) == ({"state": "reset"}, 1.5)
	assert rl.stepSendActGetObs({"move": 1}, timeout=0.4) == (
		0.2,
		{"state": "step"},
		3.0,
		2.5,
	)
	assert fake_comm.sent == [
		{"stepkind": "reset"},
		{"stepkind": "step", "action": {"move": 1}},
	]


def test_rlside_methods_wrap_communication_errors():
	rl = make_rl_side(FakeComm(send_results=["send reset failed"]))
	with pytest.raises(RuntimeError, match="sending what to do"):
		rl.resetGetObs()

	rl = make_rl_side(FakeComm(read_results=[("read reset failed", None)]))
	with pytest.raises(RuntimeError, match="after-reset observation"):
		rl.resetGetObs()

	rl = make_rl_side(FakeComm(send_results=["send step failed"]))
	with pytest.raises(RuntimeError, match="sending step action"):
		rl.stepSendActGetObs(action=1)

	rl = make_rl_side(FakeComm(read_results=[("lat failed", None)]))
	with pytest.raises(RuntimeError, match="last action duration"):
		rl.stepSendActGetObs(action=1)

	rl = make_rl_side(FakeComm(read_results=[("", {"lat": 0.1}), ("obs failed", None)]))
	with pytest.raises(RuntimeError, match="step observation"):
		rl.stepSendActGetObs(action=1)


def test_rlside_step_exp_finished_emits_finish_command():
	fake_comm = FakeComm()
	rl = make_rl_side(fake_comm)

	rl.stepExpFinished(timeout=0.1)

	assert fake_comm.sent == [{"stepkind": "finish"}]


def test_destructors_tolerate_partial_initialization():
	RLSide.__del__(object.__new__(RLSide))
	AgentSide.__del__(object.__new__(AgentSide))


def test_rlside_destructor_closes_connection_and_reports_errors(capsys):
	fake_comm = FakeComm()
	fake_comm.end = lambda: "shutdown failed"
	rl = make_rl_side(fake_comm)
	rl._verbose = True

	RLSide.__del__(rl)

	out = capsys.readouterr().out
	assert "Error closing communications with the agent: shutdown failed" in out
	assert "Communications closed in the RL side." in out


def test_agentside_destructor_closes_connection_and_can_raise():
	state = {"calls": 0}
	fake_comm = FakeComm()

	def end_once():
		state["calls"] += 1
		if state["calls"] == 1:
			return "shutdown failed"
		return ""

	fake_comm.end = end_once
	agent = make_agent_side(fake_comm)

	with pytest.raises(RuntimeError, match="Error stopping connection with RL"):
		AgentSide.__del__(agent)


def test_agentside_destructor_verbose_path(capsys):
	fake_comm = FakeComm()
	agent = make_agent_side(fake_comm)
	agent._verbose = True

	AgentSide.__del__(agent)

	assert "Connection with RL finished." in capsys.readouterr().out
