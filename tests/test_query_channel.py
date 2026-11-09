# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the query channel (RLSideQuery and AgentSideQuery)."""

import threading
import time

import pytest

import spindecoupler as spindecoupler_module
from spindecoupler import AgentSideQuery, BaseCommPoint, RLSideQuery


class FakeComm:
    def __init__(self, *, read_results=None, send_results=None, end_results=None):
        self.read_results = list(read_results or [])
        self.send_results = list(send_results or [])
        self.end_results = list(end_results or [])
        self.begin_results = []
        self.sent = []
        self.read_timeouts = []
        self.begin_timeouts = []
        self.end_calls = 0
        self.options = None

    def readData(self, timeout):
        self.read_timeouts.append(timeout)
        if not self.read_results:
            raise AssertionError("No more read results configured")
        return self.read_results.pop(0)

    def sendData(self, data):
        self.sent.append(data)
        if self.send_results:
            return self.send_results.pop(0)
        return ""

    def begin(self, timeoutaccept=None):
        self.begin_timeouts.append(timeoutaccept)
        if self.begin_results:
            return self.begin_results.pop(0)
        return ""

    def end(self):
        self.end_calls += 1
        if self.end_results:
            return self.end_results.pop(0)
        return ""


def make_server(fake_comm, verbose=False, accept_timeout=60.0):
    server = object.__new__(RLSideQuery)
    server._verbose = verbose
    server._accept_timeout = accept_timeout
    server._srv = fake_comm
    return server


def make_client(fake_comm, verbose=False):
    client = object.__new__(AgentSideQuery)
    client._verbose = verbose
    client._comm = fake_comm
    return client


def connect_client_with_retry(ip, port, attempts=50, delay=0.05):
    last_error = None
    for _ in range(attempts):
        try:
            return AgentSideQuery(ip, port)
        except RuntimeError as exc:
            last_error = exc
            time.sleep(delay)
    raise last_error


def query(obs):
    return {"stepkind": "query", "obs": obs}


def test_server_init_verbose_success_path(monkeypatch, capsys):
    fake_comm = FakeComm()

    def factory(port, **options):
        fake_comm.options = options
        return fake_comm

    monkeypatch.setattr(spindecoupler_module, "ServerCommPoint", factory)

    server = RLSideQuery(24000, verbose=True, accept_timeout=7.0)

    out = capsys.readouterr().out
    assert "Waiting for the agent on port 24000" in out
    assert "Agent connected." in out
    assert fake_comm.begin_timeouts == [7.0]
    RLSideQuery.__del__(server)


def test_server_init_raises_without_agent(monkeypatch):
    fake_comm = FakeComm()
    fake_comm.begin_results = ["timeout"]
    monkeypatch.setattr(
        spindecoupler_module, "ServerCommPoint", lambda port, **o: fake_comm
    )

    with pytest.raises(
        RuntimeError, match="No agent connection for query channel: timeout"
    ):
        RLSideQuery(24000)


def test_server_init_rejects_non_positive_accept_timeout():
    with pytest.raises(ValueError, match="accept_timeout"):
        RLSideQuery(24000, accept_timeout=0.0)


def test_receive_query_returns_the_observation():
    fake_comm = FakeComm(read_results=[("", query([0.1, 0.2]))])
    server = make_server(fake_comm)

    assert server.receive_query(timeout=0.5) == [0.1, 0.2]
    assert fake_comm.read_timeouts == [0.5]


def test_receive_query_waits_without_limit_by_default():
    fake_comm = FakeComm(read_results=[("", query(1))])
    server = make_server(fake_comm)

    server.receive_query()

    assert fake_comm.read_timeouts == [-1.0]


def test_receive_query_wraps_errors_and_rejects_other_messages():
    server = make_server(FakeComm(read_results=[("broken", None)]))
    with pytest.raises(RuntimeError, match="Error reading query from agent: broken"):
        server.receive_query()

    for message in ({"stepkind": "step"}, {"obs": 1}, "text", None):
        server = make_server(FakeComm(read_results=[("", message)]))
        with pytest.raises(RuntimeError, match="Unexpected message"):
            server.receive_query()


def test_send_action_emits_payload_and_wraps_errors():
    fake_comm = FakeComm(send_results=["", "send failed"])
    server = make_server(fake_comm)

    server.send_action({"linear": 0.1, "angular": 0.2})
    with pytest.raises(RuntimeError, match="query channel\\): send failed"):
        server.send_action([[0.1, 0.2]])

    assert fake_comm.sent == [
        {"action": {"linear": 0.1, "angular": 0.2}},
        {"action": [[0.1, 0.2]]},
    ]


def test_reconnect_verbose_success_path(capsys):
    fake_comm = FakeComm()
    server = make_server(fake_comm, verbose=True)

    server.reconnect(timeoutaccept=5.0)

    out = capsys.readouterr().out
    assert "Waiting for the agent to reconnect" in out
    assert "Agent reconnected." in out
    assert fake_comm.end_calls == 1
    assert fake_comm.begin_timeouts == [5.0]


def test_reconnect_uses_the_constructor_timeout_by_default():
    fake_comm = FakeComm()
    server = make_server(fake_comm, accept_timeout=9.0)

    server.reconnect()

    assert fake_comm.begin_timeouts == [9.0]


def test_reconnect_wraps_errors():
    server = make_server(FakeComm(end_results=["close failed"]))
    with pytest.raises(RuntimeError, match="Error closing query channel"):
        server.reconnect()

    fake_comm = FakeComm()
    fake_comm.begin_results = ["timeout"]
    server = make_server(fake_comm)
    with pytest.raises(RuntimeError, match="No agent reconnection"):
        server.reconnect()


def test_server_destructor_reports_errors(capsys):
    server = make_server(FakeComm(end_results=["shutdown failed"]), True)

    RLSideQuery.__del__(server)

    out = capsys.readouterr().out
    assert "Error closing the connection: shutdown failed" in out
    assert "Communications closed." in out


def test_client_init_verbose_success_path(monkeypatch, capsys):
    fake_comm = FakeComm()
    monkeypatch.setattr(
        spindecoupler_module, "ClientCommPoint", lambda ip, port, **o: fake_comm
    )

    client = AgentSideQuery("127.0.0.1", 24000, verbose=True)

    assert "Connected to the query server." in capsys.readouterr().out
    AgentSideQuery.__del__(client)


def test_client_init_raises_on_connection_error(monkeypatch):
    fake_comm = FakeComm()
    fake_comm.begin_results = ["connect failed"]
    monkeypatch.setattr(
        spindecoupler_module, "ClientCommPoint", lambda ip, port, **o: fake_comm
    )

    with pytest.raises(RuntimeError, match="connecting to the RL side: connect failed"):
        AgentSideQuery("127.0.0.1", 24000)


def test_query_action_round_trip():
    fake_comm = FakeComm(read_results=[("", {"action": {"linear": 0.3}})])
    client = make_client(fake_comm)

    assert client.query_action([1.0, 2.0], timeout=0.5) == {"linear": 0.3}
    assert fake_comm.sent == [query([1.0, 2.0])]
    assert fake_comm.read_timeouts == [0.5]


def test_query_action_wraps_errors():
    client = make_client(FakeComm(send_results=["send failed"]))
    with pytest.raises(RuntimeError, match="Error sending the query"):
        client.query_action([1.0])

    client = make_client(FakeComm(read_results=[("read failed", None)]))
    with pytest.raises(RuntimeError, match="Error receiving the action"):
        client.query_action([1.0])


def test_client_destructor_reports_errors(capsys):
    client = make_client(FakeComm(end_results=["shutdown failed"]), True)

    AgentSideQuery.__del__(client)

    out = capsys.readouterr().out
    assert "Error closing the connection: shutdown failed" in out
    assert "Communications closed." in out


def test_destructors_tolerate_partial_initialization():
    RLSideQuery.__del__(object.__new__(RLSideQuery))
    AgentSideQuery.__del__(object.__new__(AgentSideQuery))


def test_options_reach_the_transport(monkeypatch):
    seen = {}

    def server_factory(port, **options):
        seen["server"] = options
        return FakeComm()

    def client_factory(ip, port, **options):
        seen["client"] = options
        return FakeComm()

    monkeypatch.setattr(spindecoupler_module, "ServerCommPoint", server_factory)
    monkeypatch.setattr(spindecoupler_module, "ClientCommPoint", client_factory)
    options = {"max_frame_size": 5, "allow_pickle": True, "send_timeout": 2.0}

    RLSideQuery(24000, **options)
    AgentSideQuery("127.0.0.1", 24000, **options)

    assert seen["server"] == options
    assert seen["client"] == options


def test_round_trip_and_reconnect_over_tcp(free_tcp_port, force_localhost):
    received = []
    errors = []
    server_ready = threading.Event()

    def server_worker():
        try:
            server = RLSideQuery(free_tcp_port, accept_timeout=5.0)
            server_ready.set()
            for _ in range(2):  # one query per client connection
                obs = server.receive_query(timeout=2.0)
                received.append(obs)
                server.send_action([2 * x for x in obs])
                if len(received) == 1:
                    with pytest.raises(RuntimeError):
                        server.receive_query(timeout=2.0)  # first client left
                    server.reconnect(timeoutaccept=2.0)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    first = connect_client_with_retry(BaseCommPoint.get_ip(), free_tcp_port)
    assert server_ready.wait(timeout=2.0)
    assert first.query_action([1.0, 2.0], timeout=2.0) == [2.0, 4.0]
    first._comm.end()

    second = connect_client_with_retry(BaseCommPoint.get_ip(), free_tcp_port)
    assert second.query_action([3.0], timeout=2.0) == [6.0]
    second._comm.end()

    thread.join(timeout=5.0)

    assert not thread.is_alive(), "Query server thread did not finish"
    assert errors == []
    assert received == [[1.0, 2.0], [3.0]]


def test_numpy_batches_cross_the_default_allowlist(free_tcp_port, force_localhost):
    np = pytest.importorskip("numpy")
    answers = []

    def server_worker():
        server = RLSideQuery(free_tcp_port, accept_timeout=5.0)
        obs = server.receive_query(timeout=2.0)
        server.send_action(obs.sum(axis=1))

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()
    client = connect_client_with_retry(BaseCommPoint.get_ip(), free_tcp_port)
    answers.append(client.query_action(np.ones((3, 4)), timeout=2.0))
    client._comm.end()
    thread.join(timeout=5.0)

    assert answers[0].tolist() == [4.0, 4.0, 4.0]
