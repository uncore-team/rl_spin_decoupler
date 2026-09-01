# Examples

This folder contains complete end-to-end examples with two real Python
processes (RL side + agent side) synchronized over localhost TCP sockets.

## Available examples

- [first_order_plant_control/README.md](first_order_plant_control/README.md): lightweight baseline example with a synthetic first-order plant where agent sends observations and RL computes reward/goal.
- [lunar_lander/README.md](lunar_lander/README.md): Gymnasium + Stable-Baselines3 example where the agent only transports observations/timing and reward/termination are computed on RL side.
- [lunar_lander_container/README.md](lunar_lander_container/README.md): split deployment of the LunarLander example, with the RL side running in an NVIDIA/CUDA container on a remote GPU host and the agent running locally.

## Quick run (first_order_plant_control)

From the repository root, first install the package so the examples can
`import spindecoupler` (its core is pure standard library, so this pulls no
runtime dependencies):

```bash
pip install -e .
```

Then open two terminals.

Optional dependencies for the agent-side GUI:

```bash
pip install -r examples/first_order_plant_control/requirements.txt
```

Terminal 1 (RL side first):

```bash
python examples/first_order_plant_control/rl_side_fopcontrol.py --port 49054 --steps 20
```

Terminal 2 (agent side second):

```bash
python examples/first_order_plant_control/agent_side_fopcontrol.py --port 49054
```

To launch with the graphical monitor:

```bash
python examples/first_order_plant_control/agent_side_fopcontrol.py --port 49054 --plot
```
