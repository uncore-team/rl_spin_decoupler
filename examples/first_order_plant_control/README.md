# First-order plant control with rl_spin_decoupler

This complete, executable example controls a synthetic first-order plant with
separate RL-side and agent-side processes on one machine.

## What it demonstrates

The agent simulates the discrete plant
$x_{t+1}=x_t+\alpha(u-x_t)$, where $x$ is the plant state, $u$ is the target
requested by the RL side, and $\alpha$ is the gain. Each fast agent control
tick moves the state a fraction of the remaining distance toward the target.

- `rl_side_fopcontrol.py` opens the `RLSide` server, alternates target
  requests, and computes reward and task-level termination from observations.
- `agent_side_fopcontrol.py` runs the high-rate plant/control loop and sends
  observations and timing. It does not compute reward.
- `reward.py` defines the RL-side objective using tracking performance and
  latency.

The example makes differing RL-step and agent control-loop rates visible
without requiring a physical device or an external simulator.

## Requirements and installation

Install the package from the repository root:

```bash
pip install -e .
```

The optional live agent plot needs Matplotlib:

```bash
pip install -r examples/first_order_plant_control/requirements.txt
```

On WSL or a headless Linux host, the plot also needs system Tk bindings, for
example `sudo apt install python3-tk`. Do not add that system package to
`requirements.txt`.

## Running (correct order)

Open two terminals at the repository root. Start the RL side first:

```bash
python examples/first_order_plant_control/rl_side_fopcontrol.py --port 49054 --steps 20
```

Then start the agent side:

```bash
python examples/first_order_plant_control/agent_side_fopcontrol.py --port 49054
```

Add `--plot` to the agent command to show plant state, target, gain, and
action timing. `--plot-refresh N` controls redraw cadence, and
`--no-plot-hold` closes the plot when the RL side sends `FINISH`.

## What you should see

- An RL reset observation followed by step logs with action, plant state,
  reward, `lat`, `ato`, and `t_wall`.
- A final total-reward line once the RL side sends `FINISH`.
- With `--plot`, the agent-side state moving toward each target and timing
  since the most recent RL action.

## Scope of the example

The plant is deliberately simple and deterministic; it is a communication and
timing demonstration, not a trained controller or a model of a real system.
The RL side uses a fixed alternating policy for a short episode rather than a
learning algorithm.

## References

- Main project README: [../../README.md](../../README.md)
- Library API: [../../docs/api.rst](../../docs/api.rst)