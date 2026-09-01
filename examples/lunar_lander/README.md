# LunarLander + Stable-Baselines3 with rl_spin_decoupler

This complete, executable example demonstrates the RL/agent decoupling pattern
with Gymnasium `LunarLander-v3` and SB3 (PPO).

## What it demonstrates

Example architecture:

- `rl_side_lunarlander.py`: RL process. Opens the server socket with `RLSide`,
  trains PPO, and computes the learning signal.
- `agent_side_lunarlander.py`: agent/environment process. Runs the spin loop,
  applies actions, and publishes observations.
- `reward.py`: reward, `terminated`, and `truncated` logic on the RL side.

Central design choice:

- The agent only transports observations and agent time (plus LAT).
- Reward and termination are computed on the RL side from the received
  observation.

This is consistent with the decoupler philosophy: the agent stays agnostic to
the learning task, and the RL-specific logic lives in the RL process.

## Requirements and installation

These dependencies are optional and apply only to this example. The core of
`rl_spin_decoupler` remains pure standard library.

Installation (install the package first so the scripts can
`import spindecoupler`, then the example-only dependencies):

```bash
pip install -e .
pip install -r examples/lunar_lander/requirements.txt
```

Box2D note:

`gymnasium[box2d]` may require system build tools (for example `swig`,
`build-essential`, `python3-dev`, or the equivalents for your distribution).

## Running (correct order)

Open two terminals at the repository root.

1) Terminal 1 (first, RL side)

```bash
python examples/lunar_lander/rl_side_lunarlander.py
```

2) Terminal 2 (afterwards, agent side)

```bash
python examples/lunar_lander/agent_side_lunarlander.py
```

If you want to see the graphical part (the LunarLander window), enable
rendering in the agent process:

```bash
python examples/lunar_lander/agent_side_lunarlander.py --render
```

Note: on environments without a display (for example some servers or WSL
without an X/Wayland server), rendering may not be available.

Why this order:

`RLSide` opens the server socket and blocks waiting for a connection;
`AgentSide` connects as a client.

## What you should see

- SB3/PPO training logs in the RL process.
- Per-step timing metadata (including `lat`) inside `info`.
- If you use `--rollout-steps`, a summary with the mean LAT at the end of the
  rollout.

## Scope of the example

This example prioritizes illustrating the decoupling pattern, not reaching SOTA.

- The reward is reconstructed on the RL side and is a didactic approximation.
- Robustly solving LunarLander normally requires many more steps and
  hyperparameter tuning.

## References

- Main project README: [../../README.md](../../README.md)
- Library API: [../../docs/api.rst](../../docs/api.rst)
