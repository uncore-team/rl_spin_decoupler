# SPDX-License-Identifier: GPL-3.0-only

"""Does TCP_NODELAY change the LAT when each action is held for a while?

``bench_transport.py`` shows that, without TCP_NODELAY, a step whose two replies
are sent back to back waits for the delayed ACK of the receiver. This script
checks the loop of the TurtleBot3 example, where that does not happen: the agent
runs a spin loop (50 ms), holds each action for an RL step (300 ms) and reports
the LAT as the skeleton agent does. Each side then has at most one message in
flight, so the median LAT should not depend on TCP_NODELAY.

Usage: python benchmarks/bench_nagle_lat.py [--steps 40] [--csv out.csv]
"""

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import connect_agent, environment, free_port, percentile  # noqa: E402

from spindecoupler import AgentSide, BaseCommPoint, RLSide  # noqa: E402

RL_STEP = 0.3  # s, how long the agent holds each action
SPIN_PERIOD = 0.05  # s, period of the agent's loop


def run_agent(agent):
    """Agent loop of the skeleton: hold each action, then report LAT and obs."""
    state, last_t = "READY", time.time()
    while True:
        now = time.time()
        if state == "EXEC":
            if now - last_t >= RL_STEP:
                agent.stepSendObs([0.0] * 10, agenttime=now)
                state = "READY"
        elif state == "READY":
            command = agent.readWhatToDo()
            if command:
                what, _ = command
                if what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                    agent.stepSendLastActDur(now - last_t)
                    last_t, state = now, "EXEC"
                elif what == AgentSide.WhatToDo.RESET_SEND_OBS:
                    last_t, state = now, "RESET"
                else:  # FINISH
                    return
        elif state == "RESET":
            agent.resetSendObs([0.0] * 10, agenttime=now)
            state = "READY"
        time.sleep(SPIN_PERIOD)


def measure(nodelay, steps):
    """Return the LATs in milliseconds reported over ``steps`` RL steps."""
    original = BaseCommPoint.__dict__["_disable_nagle"]  # the staticmethod object
    if not nodelay:
        BaseCommPoint._disable_nagle = staticmethod(lambda sock: None)
    try:
        port = free_port()
        server = {}
        accepting = threading.Thread(
            target=lambda: server.setdefault("rl", RLSide(port, accept_timeout=20.0)),
            daemon=True,
        )
        accepting.start()
        agent = connect_agent(port)
        accepting.join(timeout=30.0)
        rl = server["rl"]
        worker = threading.Thread(target=run_agent, args=(agent,), daemon=True)
        worker.start()
        rl.resetGetObs(timeout=10.0)
        lats = []
        for _ in range(steps):
            lat, _, _, _ = rl.stepSendActGetObs(0, timeout=10.0)
            lats.append(lat * 1000)
        rl.stepExpFinished()
        worker.join()
        return lats[1:]  # the first LAT spans the reset, not an action
    finally:
        BaseCommPoint._disable_nagle = original


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--csv", help="write the summary to this CSV file")
    args = parser.parse_args()

    print(f"# {environment()}, {args.steps} RL steps per row")
    print(f"# hold {RL_STEP * 1000:.0f} ms, agent loop {SPIN_PERIOD * 1000:.0f} ms")
    header = "tcp_nodelay,median_lat_ms,p95_lat_ms,max_lat_ms"
    print(header)
    rows = []
    for nodelay in (False, True):
        lats = measure(nodelay, args.steps)
        ordered = sorted(lats)
        row = (
            nodelay,
            statistics.median(ordered),
            percentile(ordered, 0.95),
            ordered[-1],
        )
        rows.append(row)
        print("{},{:.1f},{:.1f},{:.1f}".format(*row))
    if args.csv:
        with open(args.csv, "w") as out:
            out.write(header + "\n")
            for row in rows:
                out.write("{},{:.1f},{:.1f},{:.1f}\n".format(*row))


if __name__ == "__main__":
    main()
