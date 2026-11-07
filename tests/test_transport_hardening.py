# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the transport limits and checks.

Frame size limit, protocol header, restricted unpickling, send timeout, address
reuse, configurable accept timeout and logging.
"""

import io
import logging
import os
import pickle
import socket
import struct
import threading
import time
import types

import pytest
from rl_spin_decoupler.socketcomms import comms as comms_module
from rl_spin_decoupler.socketcomms.comms import (
    PROTOCOL_VERSION,
    BaseCommPoint,
    ClientCommPoint,
    ServerCommPoint,
    enable_console_logging,
)

import spindecoupler as spindecoupler_module
from spindecoupler import AgentSide, RLSide


class Evil:
    """Unpickling this would call ``os.getcwd``: a stand-in for any attack."""

    def __reduce__(self):
        return (os.getcwd, ())


def connected_pair(port, **options):
    """Return a connected (server, client) pair over loopback."""
    server = ServerCommPoint(port, **options)
    thread = threading.Thread(target=server.begin, args=(5.0,), daemon=True)
    thread.start()
    client = None
    for _ in range(50):
        client = ClientCommPoint("127.0.0.1", port, **options)
        if not client.begin():
            break
        time.sleep(0.05)
    thread.join(timeout=5.0)
    assert client is not None and client._begun and server._begun
    return server, client


@pytest.fixture
def pair(free_tcp_port, force_localhost):
    server, client = connected_pair(free_tcp_port)
    yield server, client
    client.end()
    server.end()


def test_round_trip_keeps_plain_data(pair):
    server, client = pair
    payload = {"a": [1, 2.5, "x", None, True], "t": (1, 2), "s": {3}, "b": b"\x00"}

    assert client.sendData(payload) == ""

    assert server.readData(2.0) == ("", payload)


def test_numpy_arrays_and_scalars_are_accepted(pair):
    np = pytest.importorskip("numpy")
    server, client = pair
    payload = {"obs": np.arange(6, dtype=np.float32).reshape(2, 3), "r": np.float64(2)}

    assert client.sendData(payload) == ""

    res, data = server.readData(2.0)
    assert res == ""
    assert (data["obs"] == payload["obs"]).all()
    assert data["obs"].dtype == np.float32
    assert data["r"] == 2.0


def test_unknown_globals_are_refused_by_default(pair):
    server, client = pair

    assert client.sendData({"x": Evil()}) == ""

    res, data = server.readData(2.0)
    assert "refused to unpickle" in res
    assert "allow_pickle=True" in res
    assert data is None


def test_allow_pickle_accepts_any_pickle(free_tcp_port, force_localhost):
    server, client = connected_pair(free_tcp_port, allow_pickle=True)
    try:
        assert client.sendData({"x": Evil()}) == ""
        res, data = server.readData(2.0)
        assert res == ""
        assert data["x"] == os.getcwd()
    finally:
        client.end()
        server.end()


def test_oversized_incoming_frame_closes_the_connection(pair):
    server, client = pair
    server._max_frame_size = 100
    # A header announcing 4 GiB, as a corrupt or hostile peer could send.
    client._sock.sendall(struct.pack("!2sBI", b"SD", PROTOCOL_VERSION, 0xFFFFFFFF))

    res, data = server.readData(2.0)

    assert "exceeds the limit of 100 bytes" in res
    assert data is None
    assert server._begun is False  # unusable: the stream is out of step
    with pytest.raises(RuntimeError, match="not-begun"):
        server.readData(0.1)


def test_oversized_outgoing_payload_is_not_sent(pair):
    server, client = pair
    client._max_frame_size = 100

    err = client.sendData({"blob": "x" * 500})

    assert "exceeds the limit of 100 bytes" in err
    assert client._begun is True  # nothing was written, the stream is intact
    assert client.sendData({"ok": 1}) == ""
    assert server.readData(2.0) == ("", {"ok": 1})


def test_foreign_peer_is_rejected_by_magic(pair):
    server, client = pair
    client._sock.sendall(b"GET / HTTP/1.1\r\n")

    res, data = server.readData(2.0)

    assert "does not speak this protocol" in res
    assert data is None
    assert server._begun is False


def test_protocol_version_mismatch_is_reported(pair):
    server, client = pair
    client._sock.sendall(struct.pack("!2sBI", b"SD", PROTOCOL_VERSION + 1, 0))

    res, data = server.readData(2.0)

    assert f"protocol version {PROTOCOL_VERSION + 1} received" in res
    assert f"{PROTOCOL_VERSION} expected" in res
    assert data is None
    assert server._begun is False


def test_send_timeout_closes_the_connection(free_tcp_port, force_localhost):
    server, client = connected_pair(free_tcp_port, send_timeout=0.3)
    try:
        # The server never reads, so the socket buffers fill up.
        big = {"blob": b"x" * (8 * 1024 * 1024)}
        errors = []
        for _ in range(50):
            err = client.sendData(big)
            if err:
                errors.append(err)
                break
        assert errors and "send timed out" in errors[0]
        assert client._begun is False
    finally:
        client.end()
        server.end()


def test_send_timeout_zero_restores_blocking_mode(pair):
    server, client = pair
    client._send_timeout = 0

    assert client.sendData({"x": 1}) == ""
    assert client._sock.gettimeout() is None


def test_constructor_validates_new_options():
    kind = BaseCommPoint.Kind.SERVER
    for bad in (0, -1, 2**32, 1.5, True, "1"):
        with pytest.raises(ValueError, match="max. frame size"):
            BaseCommPoint(kind=kind, max_frame_size=bad)
    with pytest.raises(ValueError, match="send timeout"):
        BaseCommPoint(kind=kind, send_timeout="soon")
    with pytest.raises(ValueError, match="send timeout"):
        BaseCommPoint(kind=kind, send_timeout=True)


@pytest.mark.skipif(os.name != "posix", reason="SO_REUSEADDR is set only on POSIX")
def test_server_socket_reuses_address(free_tcp_port, force_localhost):
    server = ServerCommPoint(free_tcp_port)
    try:
        reuse = server._basesock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        assert reuse != 0
    finally:
        server._basesock.close()


@pytest.mark.skipif(os.name != "posix", reason="SO_REUSEADDR is set only on POSIX")
def test_server_restarts_right_after_a_connection(free_tcp_port, force_localhost):
    server, client = connected_pair(free_tcp_port)
    client.sendData({"x": 1})
    assert server.readData(2.0)[0] == ""
    # The side that closes first keeps the port in TIME_WAIT.
    server.end()
    client.end()
    server._basesock.close()

    again = ServerCommPoint(free_tcp_port)  # would raise EADDRINUSE without reuse
    again._basesock.close()


def test_server_socket_skips_reuse_off_posix(
    monkeypatch, free_tcp_port, force_localhost
):
    monkeypatch.setattr(comms_module, "os", types.SimpleNamespace(name="nt"))
    server = ServerCommPoint(free_tcp_port)
    try:
        reuse = server._basesock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        assert reuse == 0
    finally:
        server._basesock.close()


def test_rlside_accept_timeout_is_configurable(free_tcp_port, force_localhost):
    start = time.time()

    with pytest.raises(RuntimeError, match="No agent connection: timeout"):
        RLSide(free_tcp_port, accept_timeout=0.3)

    assert time.time() - start < 5.0


def test_rlside_rejects_non_positive_accept_timeout(free_tcp_port, force_localhost):
    with pytest.raises(ValueError, match="accept_timeout"):
        RLSide(free_tcp_port, accept_timeout=0.0)


def test_endpoints_forward_transport_options(monkeypatch):
    seen = {}

    class FakeServer:
        def __init__(self, port, **options):
            seen["server"] = options

        def begin(self, timeoutaccept):
            return ""

        def end(self):
            return ""

    class FakeClient:
        def __init__(self, ip, port, **options):
            seen["client"] = options

        def begin(self):
            return ""

        def end(self):
            return ""

    monkeypatch.setattr(spindecoupler_module, "ServerCommPoint", FakeServer)
    monkeypatch.setattr(spindecoupler_module, "ClientCommPoint", FakeClient)
    options = {"max_frame_size": 5, "allow_pickle": True, "send_timeout": 2.0}

    RLSide(24000, **options)
    AgentSide("127.0.0.1", 24000, **options)

    assert seen["server"] == options
    assert seen["client"] == options


def test_verbose_logs_go_to_the_library_logger(free_tcp_port, force_localhost, caplog):
    with caplog.at_level(logging.INFO, logger="spindecoupler"):
        with pytest.raises(RuntimeError):
            RLSide(free_tcp_port, verbose=True, accept_timeout=0.2)

    messages = [r.getMessage() for r in caplog.records if r.name == "spindecoupler"]
    assert any("Waiting for agent connection" in m for m in messages)


def test_enable_console_logging_is_idempotent():
    log = comms_module.logger
    handlers = list(log.handlers)
    level = log.level
    try:
        enable_console_logging(logging.INFO)
        enable_console_logging(logging.DEBUG)
        added = [h for h in log.handlers if h not in handlers]
        assert len(added) <= 1
        assert log.level == logging.DEBUG
    finally:
        log.handlers[:] = handlers
        log.setLevel(level)


def test_check_data_to_read_error_names_the_operation():
    point = BaseCommPoint(kind=BaseCommPoint.Kind.SERVER)

    with pytest.raises(RuntimeError, match="Cannot check data"):
        point.checkDataToRead()


def test_pickle_of_default_protocol_still_loads_plain_data():
    # Guards the allowlist against a Python version that pickles plain data
    # with a global.
    for proto in range(3, pickle.HIGHEST_PROTOCOL + 1):
        data = {"a": [1, (2, 3)], "s": {1}, "f": frozenset({2}), "c": 1j}
        blob = pickle.dumps(data, protocol=proto)
        loaded = comms_module._RestrictedUnpickler(io.BytesIO(blob)).load()
        assert loaded == data
