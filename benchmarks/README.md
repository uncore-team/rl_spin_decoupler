# Benchmarks

Two scripts that reproduce the timing claims of the library. They use only the
standard library (the plot of the second one needs Matplotlib) and run on one machine
over the loopback interface, so they take about a minute. Absolute numbers depend on
the machine and the operating system; the shape of the results does not.

From the repository root:

```bash
python benchmarks/bench_transport.py --csv benchmarks/results_transport.csv
python benchmarks/bench_decoupling.py --csv benchmarks/results_decoupling.csv \
    --plot benchmarks/decoupling.png
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
produce. Both agents use the library:

- Coupled: after reporting, the agent waits for the action and does not run its loop,
  as a synchronous Gym environment does.
- Decoupled: the agent calls `readWhatToDo()` on every tick. It returns at once when
  nothing has arrived, so the loop keeps its period and the action in force is held.

The table gives the longest gap between two consecutive control ticks and the number of
ticks later than 1.5 periods, over 20 RL steps per row (`results_decoupling.csv`).

| Learner time per decision | Coupled: longest gap (late ticks) | Decoupled: longest gap (late ticks) |
|---|---|---|
| 0 ms | 20.9 ms (0) | 20.2 ms (0) |
| 20 ms | 40.9 ms (19) | 20.3 ms (0) |
| 50 ms | 70.8 ms (19) | 20.8 ms (0) |
| 100 ms | 120.9 ms (19) | 20.3 ms (0) |
| 200 ms | 221.1 ms (19) | 20.3 ms (0) |

The gap of the coupled agent grows with the learner time, because the loop stops while
it waits. The decoupled loop keeps its 20 ms period whatever the learner does, which is
what a real robot or a simulator that advances in real time needs. `decoupling.png`
plots the same data.

The benchmark measures the control loop, not learning performance. The learner here
only sleeps, so the result is the same for any algorithm.
