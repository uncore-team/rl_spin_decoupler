# SPDX-License-Identifier: GPL-3.0-only

import random
import socket

import pytest

from spindecoupler import BaseCommPoint


@pytest.fixture
def force_localhost(monkeypatch):
    monkeypatch.setattr(BaseCommPoint, "get_ip", classmethod(lambda cls: "127.0.0.1"))


@pytest.fixture
def free_tcp_port():
    for _ in range(100):
        port = random.randint(20000, 49151)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not allocate a free TCP port in the supported range")
