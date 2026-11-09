# SPDX-License-Identifier: GPL-3.0-only

"""What decoupling buys: a control loop that keeps its rate while the learner thinks.

An agent runs a control loop with a fixed period (20 ms). Every RL step (100 ms)
it reports an observation and receives the next action; the learner needs
``think`` seconds to produce it. Two agents are compared, both on the library:

- coupled: after sending the observation the agent waits for the action and does
  not run the control loop meanwhile, as a synchronous Gym environment would.
- decoupled: the agent polls ``readWhatToDo`` once per control tick, which returns
  at once when nothing has arrived, and keeps the loop running.
- sb3_subproc (only if Stable-Baselines3 is installed): the same control loop
  inside a Gymnasium environment run by SB3's ``SubprocVecEnv``. The environment
  lives in its own process, but it only runs inside ``step()``, so the loop stops
  while the learner computes, as in the coupled case.

For each learner time it reports the gaps between consecutive control ticks.

Usage: python benchmarks/bench_decoupling.py [--steps 30] [--csv out.csv]
       [--plot out.png]
"""

import argparse
import multiprocessing
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import environment, free_port, percentile  # noqa: E402

from spindecoupler import AgentSide, RLSide  # noqa: E402

CONTROL_PERIOD = 0.02  # s, the control loop of the agent
RL_STEP_PERIOD = 0.1  # s, how long each action stays applied before the next report
THINK_TIMES = (0.0, 0.02, 0.05, 0.1, 0.2)  # s, time the learner needs per decision


def run_learner(port, steps, think, ready):
    """RL side: ``steps`` decisions, each taking ``think`` seconds."""
    rl = RLSide(port, accept_timeout=20.0)
    ready.set()
    rl.resetGetObs(timeout=10.0)
    for _ in range(steps):
        time.sleep(think)  # stands for the policy and the update
        rl.stepSendActGetObs(0, timeout=10.0)
    rl.stepExpFinished()


def run_agent(port, coupled):
    """Agent side. Return the timestamps of its control ticks after the reset."""
    agent = AgentSide("127.0.0.1", port)
    ticks = []
    started = False
    last_report = time.perf_counter()
    while True:
        command = agent.readWhatToDo(timeout=10.0)
        if command is None:
            if coupled and started:
                time.sleep(0.0005)  # waiting for the action; the loop does not run
            else:
                if started:
                    ticks.append(time.perf_counter())
                time.sleep(CONTROL_PERIOD)
            continue
        what, _ = command
        if what == AgentSide.WhatToDo.FINISH:
            return ticks
        if what == AgentSide.WhatToDo.RESET_SEND_OBS:
            agent.resetSendObs([0.0], agenttime=time.perf_counter())
            started = True
            last_report = time.perf_counter()
        else:  # a new action: hold it for RL_STEP_PERIOD, then report
            agent.stepSendLastActDur(time.perf_counter() - last_report)
            last_report = time.perf_counter()
            while time.perf_counter() - last_report < RL_STEP_PERIOD:
                ticks.append(time.perf_counter())
                time.sleep(CONTROL_PERIOD)
            agent.stepSendObs([0.0], agenttime=time.perf_counter())
            last_report = time.perf_counter()


def measure(think, coupled, steps):
    """Return the gaps in milliseconds between consecutive control ticks."""
    port = free_port()
    ready = threading.Event()
    learner = threading.Thread(target=run_learner, args=(port, steps, think, ready))
    learner.start()
    ready.wait(5.0)
    time.sleep(0.2)
    ticks = run_agent(port, coupled)
    learner.join()
    return [(b - a) * 1000 for a, b in zip(ticks, ticks[1:])]


def measure_sb3_subproc(think, steps):
    """Gaps in milliseconds of the same loop inside an SB3 ``SubprocVecEnv``."""
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import SubprocVecEnv

    class ControlLoopEnv(gym.Env):
        """Runs the control ticks of one RL step inside ``step()``."""

        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def __init__(self):
            self.ticks = []

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            start = time.perf_counter()
            while time.perf_counter() - start < RL_STEP_PERIOD:
                self.ticks.append(time.perf_counter())
                time.sleep(CONTROL_PERIOD)
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        def get_ticks(self):
            return self.ticks

    # fork, where available, reuses the modules already imported (torch takes long
    # to import again); the environment still runs only inside step().
    start = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
    vec = SubprocVecEnv([ControlLoopEnv], start_method=start)
    try:
        vec.reset()
        for _ in range(steps):
            time.sleep(think)  # stands for the policy and the update
            vec.step(np.zeros(1, dtype=np.int64))
        ticks = vec.env_method("get_ticks")[0]
    finally:
        vec.close()
    return [(b - a) * 1000 for a, b in zip(ticks, ticks[1:])]


def sb3_available():
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        return False
    return True


def summarize(gaps):
    ordered = sorted(gaps)
    missed = sum(1 for g in gaps if g > 1.5 * CONTROL_PERIOD * 1000)
    return percentile(ordered, 0.5), percentile(ordered, 0.99), ordered[-1], missed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--csv", help="write the results to this CSV file")
    parser.add_argument(
        "--plot", help="save a figure to this PNG file (needs Matplotlib)"
    )
    args = parser.parse_args()

    print(f"# {environment()}, {args.steps} RL steps per row")
    print(
        f"# control period {CONTROL_PERIOD * 1000:.0f} ms, "
        f"RL step {RL_STEP_PERIOD * 1000:.0f} ms"
    )
    header = "mode,think_ms,median_gap_ms,p99_gap_ms,max_gap_ms,missed_ticks"
    print(header)
    rows = []
    modes = ["coupled", "decoupled"]
    if sb3_available():
        modes.append("sb3_subproc")
    else:
        print("# Stable-Baselines3 not installed: sb3_subproc skipped")
    for mode in modes:
        for think in THINK_TIMES:
            if mode == "sb3_subproc":
                gaps = measure_sb3_subproc(think, args.steps)
            else:
                gaps = measure(think, mode == "coupled", args.steps)
            median, p99, worst, missed = summarize(gaps)
            row = (mode, think * 1000, median, p99, worst, missed)
            rows.append(row)
            print("{},{:.0f},{:.1f},{:.1f},{:.1f},{}".format(*row))
    if args.csv:
        with open(args.csv, "w") as out:
            out.write(header + "\n")
            for row in rows:
                out.write("{},{:.0f},{:.1f},{:.1f},{:.1f},{}\n".format(*row))
    if args.plot:
        plot(rows, args.plot)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for mode, style in (
        ("coupled", "o-"),
        ("decoupled", "s--"),
        ("sb3_subproc", "^:"),
    ):
        data = [r for r in rows if r[0] == mode]
        if not data:
            continue
        ax.plot([r[1] for r in data], [r[4] for r in data], style, label=mode)
    ax.axhline(CONTROL_PERIOD * 1000, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("learner time per decision (ms)")
    ax.set_ylabel("longest gap between control ticks (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
