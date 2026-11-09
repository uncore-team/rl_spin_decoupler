# SPDX-License-Identifier: GPL-3.0-only

"""Helpers shared by the benchmark scripts."""

import os
import platform
import random
import socket
import sys
import time


def free_port() -> int:
    """Return a TCP port that the RL side can bind right now.

    The port is checked the way ``ServerCommPoint`` binds it (all interfaces, with
    ``SO_REUSEADDR`` on POSIX) and taken below 32768, outside the ephemeral range
    that Linux uses for the client ends of connections, so that a connection made
    in the meantime cannot take it.
    """
    for _ in range(100):
        port = random.randint(20000, 32767)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if os.name == "posix":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free TCP port in 20000-32767")


def connect_agent(port: int, timeout: float = 10.0, **options):
    """Create an ``AgentSide`` on loopback, retrying until the RL side listens.

    The RL side is started in another thread, so the agent may try to connect
    before the server socket exists.
    """
    from spindecoupler import AgentSide

    deadline = time.monotonic() + timeout
    while True:
        try:
            return AgentSide("127.0.0.1", port, **options)
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def environment() -> str:
    """One line that identifies the machine, to store next to the results."""
    return "{} {}, Python {}".format(
        platform.system(), platform.release(), sys.version.split()[0]
    )


def percentile(sorted_values, fraction: float) -> float:
    """Value at ``fraction`` (0 to 1) of an already sorted list."""
    return sorted_values[
        min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    ]
