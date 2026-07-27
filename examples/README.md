# Example: end-to-end decoupled run

This folder contains a complete and executable example of RL Spin Decoupler with two real Python processes communicating through localhost TCP sockets.

- [rl_side_demo.py](rl_side_demo.py): RL-side process (server, started first).
- [agent_side_demo.py](agent_side_demo.py): Agent-side process (client, started second).

The agent side simulates a fast control loop with a tiny first-order plant. The RL side sends alternating targets and receives:

- LAT: real duration of the previous action in the agent clock.
- ATO: agent timestamp when observation was sampled.
- t_wall: RL local timestamp when response arrives.

## Run

From the repository root, open two terminals.

Terminal 1 (RL side first):

```bash
python examples/rl_side_demo.py --port 49054 --steps 20
```

Terminal 2 (agent side second):

```bash
python examples/agent_side_demo.py --port 49054
```

If you need to set the IP explicitly (for remote runs), use the same interface
address where the RL side is listening:

```bash
python examples/agent_side_demo.py --ip <rl_host_ip> --port 49054
```

## Expected output

RL side prints one line per step with LAT/ATO/t_wall and observation values, for example:

```text
[RL] step=03 action={'target': -0.4, 'gain': 0.25} lat=0.203001s ato=1722080000.501000 t_wall=1722080000.502410 obs={'plant_state': 0.112, 'target': -0.4, 'gain': 0.25} rew=-0.112000
```

Agent side prints connection lifecycle messages and exits after receiving `FINISH`.
