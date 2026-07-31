# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the reconnect/cleanup semantics of the socket comm points.

``begin()`` on both the server and the client must close any previously
established connection before opening a new one. ``end()`` must be a safe no-op
when the point was never begun.
"""

import threading
import time

from spindecoupler import BaseCommPoint, ClientCommPoint, ServerCommPoint


def _connect_client_with_retry(ip, port, attempts=50, delay=0.05):
    last = None
    for _ in range(attempts):
        client = ClientCommPoint(ip, port)
        err = client.begin()
        if not err:
            return client
        last = err
        time.sleep(delay)
    raise RuntimeError(f"client could not connect: {last}")


def test_server_end_is_noop_when_not_begun(free_tcp_port, force_localhost):
    """A freshly constructed (never begun) server closes cleanly."""

    server = ServerCommPoint(free_tcp_port)
    try:
        assert server.end() == ""  # falls through the not-begun branch
    finally:
        server.end()


def test_client_begin_reconnects_when_already_begun(free_tcp_port, force_localhost):
    """Calling begin() twice on a client tears down and reconnects."""

    ip = BaseCommPoint.get_ip()
    server = ServerCommPoint(free_tcp_port)
    errors = []

    def server_worker():
        try:
            server.begin(timeoutaccept=5.0)
        except BaseException as exc:  # pragma: no cover - surfaced via errors list
            errors.append(exc)

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    client = _connect_client_with_retry(ip, free_tcp_port)
    try:
        assert client._begun is True
        err = client.begin()  # already begun -> close then reconnect
        assert err == ""
        assert client._begun is True
    finally:
        client.end()
        server.end()
        thread.join(timeout=2.0)
    assert not errors


def test_server_begin_reconnects_when_already_begun(free_tcp_port, force_localhost):
    """Calling begin() twice on a server accepts a fresh client connection."""

    ip = BaseCommPoint.get_ip()
    server = ServerCommPoint(free_tcp_port)
    errors = []
    first_accepted = threading.Event()

    def server_worker():
        try:
            server.begin(timeoutaccept=5.0)  # accept first client
            first_accepted.set()
            server.begin(timeoutaccept=5.0)  # already begun -> close then re-accept
        except BaseException as exc:  # pragma: no cover - surfaced via errors list
            errors.append(exc)

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    client1 = _connect_client_with_retry(ip, free_tcp_port)
    assert first_accepted.wait(timeout=5.0)
    client2 = _connect_client_with_retry(ip, free_tcp_port)
    try:
        time.sleep(0.1)
        assert server._begun is True
    finally:
        client1.end()
        client2.end()
        server.end()
        thread.join(timeout=2.0)
    assert not errors


def test_server_binds_to_all_interfaces_by_default(free_tcp_port, force_localhost):
    """The server always binds to 0.0.0.0, regardless of get_ip()'s value."""

    server = ServerCommPoint(free_tcp_port)
    assert server._ipv4 == "0.0.0.0"
    # get_ip() (mocked to 127.0.0.1 here) is still used for display purposes.
    assert server._servip == "127.0.0.1"

    errors = []

    def server_worker():
        try:
            server.begin(timeoutaccept=5.0)
        except BaseException as exc:  # pragma: no cover - surfaced via errors list
            errors.append(exc)

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    # The listening socket is on 0.0.0.0, so a loopback client still reaches it.
    client = _connect_client_with_retry("127.0.0.1", free_tcp_port)
    try:
        assert client._begun is True
    finally:
        client.end()
        server.end()
        thread.join(timeout=2.0)
    assert not errors
