# SPDX-License-Identifier: GPL-3.0-only

"""Tests for TCP keepalive and ``is_connected()`` of the comm points."""

import socket
import threading

import pytest
from rl_spin_decoupler.socketcomms.comms import (
    KEEPALIVE_COUNT,
    KEEPALIVE_IDLE,
    KEEPALIVE_INTERVAL,
)

from spindecoupler import BaseCommPoint, ClientCommPoint, ServerCommPoint


class OptionSocket:
    """Records setsockopt calls; options in ``refused`` raise OSError."""

    def __init__(self, refused=()):
        self.options = []
        self.refused = set(refused)

    def setsockopt(self, level, option, value):
        if option in self.refused:
            raise OSError("refused")
        self.options.append((level, option, value))


def _set_constants(monkeypatch, **constants):
    for name in ("TCP_KEEPIDLE", "TCP_KEEPALIVE", "TCP_KEEPINTVL", "TCP_KEEPCNT"):
        monkeypatch.delattr(socket, name, raising=False)
    for name, value in constants.items():
        monkeypatch.setattr(socket, name, value, raising=False)


def test_keepalive_sets_all_options_where_linux_has_them(monkeypatch):
    _set_constants(monkeypatch, TCP_KEEPIDLE=101, TCP_KEEPINTVL=102, TCP_KEEPCNT=103)
    sock = OptionSocket()

    BaseCommPoint._enable_keepalive(sock)

    assert sock.options == [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, 101, KEEPALIVE_IDLE),
        (socket.IPPROTO_TCP, 102, KEEPALIVE_INTERVAL),
        (socket.IPPROTO_TCP, 103, KEEPALIVE_COUNT),
    ]


def test_keepalive_uses_the_macos_idle_option(monkeypatch):
    _set_constants(monkeypatch, TCP_KEEPALIVE=201, TCP_KEEPINTVL=102, TCP_KEEPCNT=103)
    sock = OptionSocket()

    BaseCommPoint._enable_keepalive(sock)

    assert (socket.IPPROTO_TCP, 201, KEEPALIVE_IDLE) in sock.options
    assert len(sock.options) == 4


def test_keepalive_without_tuning_options_sets_only_the_flag(monkeypatch):
    _set_constants(monkeypatch)
    sock = OptionSocket()

    BaseCommPoint._enable_keepalive(sock)

    assert sock.options == [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]


def test_keepalive_ignores_a_refused_tuning_option(monkeypatch):
    _set_constants(monkeypatch, TCP_KEEPIDLE=101, TCP_KEEPINTVL=102, TCP_KEEPCNT=103)
    sock = OptionSocket(refused=[102])

    BaseCommPoint._enable_keepalive(sock)

    assert (socket.IPPROTO_TCP, 101, KEEPALIVE_IDLE) in sock.options
    assert (socket.IPPROTO_TCP, 103, KEEPALIVE_COUNT) in sock.options
    assert len(sock.options) == 3


def test_keepalive_flag_failure_is_not_hidden():
    sock = OptionSocket(refused=[socket.SO_KEEPALIVE])

    with pytest.raises(OSError):
        BaseCommPoint._enable_keepalive(sock)


def test_both_ends_enable_keepalive_and_report_connection(
    free_tcp_port, force_localhost
):
    server = ServerCommPoint(free_tcp_port)
    client = ClientCommPoint(BaseCommPoint.get_ip(), free_tcp_port)
    assert not server.is_connected()
    assert not client.is_connected()

    thread = threading.Thread(target=server.begin, args=(5.0,), daemon=True)
    thread.start()
    try:
        assert client.begin() == ""
        thread.join(timeout=5.0)
        assert server.is_connected()
        assert client.is_connected()
        for point in (server, client):
            flag = point._sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            assert flag != 0
    finally:
        client.end()
        server.end()
    assert not server.is_connected()
    assert not client.is_connected()
