# Benchmarks

Three scripts that reproduce the timing claims of the library. They use only the
standard library (the plot of the second one needs Matplotlib, and its
Stable-Baselines3 baseline runs only if that package is installed) and run on one machine
over the loopback interface, so they take about a minute. Absolute numbers depend on
the machine and the operating system; the shape of the results does not.

From the repository root:

```bash
python benchmarks/bench_transport.py --csv benchmarks/results_transport.csv
python benchmarks/bench_decoupling.py --steps 20 --csv benchmarks/results_decoupling.csv \
    --plot benchmarks/decoupling.png
python benchmarks/bench_nagle_lat.py --csv benchmarks/results_nagle_lat.csv
```

## Cost of a step (`bench_transport.py`)

A protocol step is a request from the RL side followed by two small replies sent back
to back. The script times 300 steps with `TCP_NODELAY` off and on, by replacing
`BaseCommPoint._disable_nagle` for the first run.

| `TCP_NODELAY` | median | p95 | max |
|---|---|---|---|
| off | 44.000 ms | 44.116 ms | 48.073 ms |
| on | 0.138 ms | 0.204 ms | 0.351 ms |

Measured on Linux 6.18 (WSL2) with Python 3.10. Without the option each step waits for
the delayed ACK of the receiver (about 40 ms on Linux), and the agent's `LAT` includes
that wait. See `socketcomms/README.md`.

## What decoupling buys (`bench_decoupling.py`)

An agent runs a control loop with a period of 20 ms. Every 100 ms it reports an
observation and receives the next action, which the learner needs a given time to
produce. Three agents are compared:

- Coupled: after reporting, the agent waits for the action and does not run its loop,
  as a synchronous Gym environment does.
- Decoupled: the agent calls `readWhatToDo()` on every tick. It returns at once when
  nothing has arrived, so the loop keeps its period and the action in force is held.
- SB3 `SubprocVecEnv` (only if Stable-Baselines3 is installed): the same loop inside a
  Gymnasium environment run by Stable-Baselines3 in its own process. The environment
  runs only inside `step()`, so moving it to another process does not keep the loop
  running while the learner computes.

The table gives the longest gap between two consecutive control ticks and the number of
ticks later than 1.5 periods, over 20 RL steps per row (`results_decoupling.csv`).

| Learner time per decision | Coupled | Decoupled | SB3 `SubprocVecEnv` |
|---|---|---|---|
| 0 ms | 21.0 ms (0) | 20.3 ms (0) | 20.6 ms (0) |
| 20 ms | 40.9 ms (19) | 20.4 ms (0) | 40.7 ms (19) |
| 50 ms | 70.8 ms (19) | 20.2 ms (0) | 70.5 ms (19) |
| 100 ms | 121.2 ms (19) | 20.2 ms (0) | 120.6 ms (19) |
| 200 ms | 220.9 ms (19) | 20.2 ms (0) | 220.7 ms (19) |

Each cell is the longest gap, with the late ticks in brackets. The gap of the coupled
agent, and of the same loop under `SubprocVecEnv`, grows with the learner time, because
the loop stops while it waits. The decoupled loop keeps its 20 ms period whatever the learner does, which is
what a real robot or a simulator that advances in real time needs. `decoupling.png`
plots the same data.

The benchmark measures the control loop, not learning performance. The learner here
only sleeps, so the result is the same for any algorithm.

## `TCP_NODELAY` and the LAT of a held action (`bench_nagle_lat.py`)

`bench_transport.py` sends the two replies of a step back to back, which is where Nagle's
algorithm costs a delayed ACK. In the TurtleBot3 example the agent instead holds each
action for an RL step before it reports, so each side has at most one message in
flight. The script reproduces that loop (300 ms hold, 50 ms agent loop, LAT reported as
the skeleton agent does) and compares the LAT with `TCP_NODELAY` off and on, over 40 RL
steps per row (`results_nagle_lat.csv`).

| `TCP_NODELAY` | median LAT | p95 | max |
|---|---|---|---|
| off | 350.6 ms | 350.8 ms | 3287.4 ms |
| on | 350.6 ms | 350.7 ms | 350.9 ms |

The median and p95 do not depend on the option: in this loop the delayed ACK does not
enter the step. The LAT is 350 ms rather than 300 ms because the agent notices the end
of the hold on its next 50 ms tick. The single maximum of about 3 s is a stall of the
host (WSL2): repeated runs show it once in a while with either setting.

