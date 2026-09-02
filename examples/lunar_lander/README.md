# LunarLander + Stable-Baselines3 with rl_spin_decoupler

This complete, executable example runs Gymnasium `LunarLander-v3` and SB3 PPO
on one machine, with the RL process and the environment agent in separate
processes.

## What it demonstrates

- `rl_side_lunarlander.py` opens the `RLSide` server socket, trains PPO, and 
  computes reward plus task-level termination and truncation.
- `agent_side_lunarlander.py` runs the spin loop and LunarLander physics,
  applies actions, and publishes observations and timing.
- `reward.py` keeps learning-task logic on the RL side. The agent does not
  compute reward; it only reports Gymnasium physics termination flags.

This separation keeps the agent independent of the learning objective while
the RL side owns the policy and learning signal.

## Requirements and installation

Install the package first, then the dependencies specific to this example:

```bash
pip install -e .
pip install -r examples/lunar_lander/requirements.txt
```

`gymnasium[box2d]` may require system build tools such as `swig`,
`build-essential`, and `python3-dev`, or their equivalents for your
distribution.

The RL script uses PyTorch's automatic device selection by default. To force
CPU training, run it with `--device cpu`. For GPU training, install a PyTorch
wheel compatible with the local NVIDIA driver.

## Running (correct order)

From the repository root, open two terminals and start the RL side first:

```bash
python examples/lunar_lander/rl_side_lunarlander.py
```

Then start the agent side:

```bash
python examples/lunar_lander/agent_side_lunarlander.py
```

To display the LunarLander window, add `--render` to the agent command.
Rendering may not be available on a headless host or WSL without an X/Wayland
server. `RLSide` must start first because it listens for the agent connection.

## What you should see

- SB3/PPO training logs in the RL process.
- Per-step timing metadata, including `lat`, in the environment `info` dict.
- A mean-LAT summary after a rollout when `--rollout-steps` is used.
- The LunarLander window when the agent runs with `--render` and a display is
  available.

## Scope of the example

This example demonstrates process decoupling, rather than optimizing a
LunarLander policy. Its RL-side reward reconstruction is didactic; reliably
solving LunarLander generally needs more training steps and hyperparameter
tuning.

## References

- Main project README: [../../README.md](../../README.md)
- Library API: [../../docs/api.rst](../../docs/api.rst)