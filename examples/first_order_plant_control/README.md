# First-order plant control example

This example contains two cooperating scripts:

- `rl_side_fopcontrol.py`: RL-side server process (start first).
- `agent_side_fopcontrol.py`: agent-side client process (start second).

Design note:

- The agent sends only observations (plus timing fields handled by the
	protocol).
- The RL side computes reward and decides whether the objective has been
	reached.

## Run

From the repository root, use two terminals.

Terminal 1:

```bash
python examples/first_order_plant_control/rl_side_fopcontrol.py --port 49054 --steps 20
```

Optional: to display a live GUI on the agent side (plant state, target, gain,
and action timing), install Matplotlib and add `--plot` in Terminal 2.

```bash
pip install -r examples/first_order_plant_control/requirements.txt
```

Useful GUI flags:

- `--plot-refresh N`: redraw every `N` control ticks.
- `--no-plot-hold`: close the plot automatically when `FINISH` is received.

Terminal 2:

```bash
python examples/first_order_plant_control/agent_side_fopcontrol.py --port 49054 --plot
```
