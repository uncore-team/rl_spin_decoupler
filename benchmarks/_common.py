# SPDX-License-Identifier: GPL-3.0-only

"""Helpers shared by the benchmark scripts."""

import platform
import random
import socket
import sys


def free_port() -> int:
    """Return a TCP port in the range the library accepts and nobody is using."""
    for _ in range(100):
        port = random.randint(20000, 49151)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free TCP port in 20000-49151")


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
