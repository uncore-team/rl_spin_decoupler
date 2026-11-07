# SPDX-License-Identifier: GPL-3.0-only

"""Cost of one protocol step over loopback, with and without TCP_NODELAY.

A step is a request from the RL side followed by two small replies sent back to
back (LAT, then the observation), the pattern of ``stepSendActGetObs``. With
Nagle's algorithm on, the second reply waits for the delayed ACK of the first.

Usage: python benchmarks/bench_transport.py [--rounds 300] [--csv out.csv]
"""

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import environment, free_port, percentile  # noqa: E402

from spindecoupler import BaseCommPoint, ClientCommPoint, ServerCommPoint  # noqa: E402


def measure(nodelay: bool, rounds: int):
    """Return the duration in milliseconds of ``rounds`` steps."""
    original = BaseCommPoint.__dict__["_disable_nagle"]  # the staticmethod object
    if not nodelay:
        BaseCommPoint._disable_nagle = staticmethod(lambda sock: None)
    try:
        port = free_port()
        server = ServerCommPoint(port)
        accepting = threading.Thread(target=server.begin, args=(10.0,))
        accepting.start()
        client = ClientCommPoint("127.0.0.1", port)
        while client.begin():
            time.sleep(0.05)
        accepting.join()

        def agent():
            for _ in range(rounds):
                client.readData(10.0)
                client.sendData({"lat": 0.08})
                client.sendData({"obs": [0.0] * 10, "rew": 0.0, "ato": 1.0})

        worker = threading.Thread(target=agent)
        worker.start()
        times = []
        for _ in range(rounds):
            start = time.perf_counter()
            server.sendData({"stepkind": "step", "action": 1})
            server.readData(10.0)
            server.readData(10.0)
            times.append((time.perf_counter() - start) * 1000)
        worker.join()
        client.end()
        server.end()
        server._basesock.close()
        return times
    finally:
        BaseCommPoint._disable_nagle = original


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--csv", help="write the summary to this CSV file")
    args = parser.parse_args()

    rows = []
    print(f"# {environment()}, {args.rounds} steps per row")
    print("tcp_nodelay,median_ms,p95_ms,max_ms")
    for nodelay in (False, True):
        times = sorted(measure(nodelay, args.rounds))
        row = (
            nodelay,
            statistics.median(times),
            percentile(times, 0.95),
            times[-1],
        )
        rows.append(row)
        print("{},{:.3f},{:.3f},{:.3f}".format(*row))
    if args.csv:
        with open(args.csv, "w") as out:
            out.write("tcp_nodelay,median_ms,p95_ms,max_ms\n")
            for row in rows:
                out.write("{},{:.3f},{:.3f},{:.3f}\n".format(*row))


if __name__ == "__main__":
    main()
