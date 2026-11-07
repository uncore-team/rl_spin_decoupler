# First-order plant control

The smallest example of the library. A fast control loop on the agent side drives a
simulated first-order plant, and the RL side sends a new target every 0.2 s. It needs
no dependencies besides the package, so it is the quickest way to see the protocol
working.

```
 rl_side_fopcontrol.py  <--- TCP :49054 ---  agent_side_fopcontrol.py
 (targets, reward)                           (50 Hz loop, plant x' = x + a(u - x))
```

The agent sends observations only. The RL side computes the reward (minus the
distance of the plant state to zero) and decides when the goal is reached.

## Installation

```bash
pip install -e .
```

The live plot of the agent also needs Matplotlib:

```bash
pip install -r examples/first_order_plant_control/requirements.txt
```

## Running

One command starts both sides and stops them when the episode ends:

```bash
./examples/first_order_plant_control/run_demo.sh
```

Or by hand, in two terminals from the repository root. Start the RL side first:

```bash
python examples/first_order_plant_control/rl_side_fopcontrol.py --port 49054 --steps 20
python examples/first_order_plant_control/agent_side_fopcontrol.py --port 49054
```

| Option | Script | Meaning |
|---|---|---|
| `--steps` | RL | Maximum number of actions (20) |
| `--timeout` | both | Seconds to wait for the other side (3 on the RL side, 1 on the agent) |
| `--rl-step-period` | agent | Seconds each action stays active (0.2) |
| `--control-period` | agent | Control-loop period (0.02) |
| `--plot` | agent | Live plot of the plant, target, gain and action age |
| `--plot-refresh N` | agent | Redraw every N control ticks |
| `--no-plot-hold` | agent | Close the plot when the episode ends |

## What you should see

The RL side prints one line per step with the action, `lat`, `ato`, the wall-clock
time, the observation and the reward. The episode ends after `--steps` actions or
when the plant state is within 0.05 of zero. The agent prints
`[AGENT] finish command received`.

## References

- [Main README](../../README.md)
- [LunarLander example](../lunar_lander/README.md), a larger one with PPO
