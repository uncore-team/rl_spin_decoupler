# Examples

This folder contains complete end-to-end examples with two real Python
processes (RL side + agent side) synchronized over localhost TCP sockets.

## Available examples

- [lunar_lander/README.md](lunar_lander/README.md): Gymnasium + Stable-Baselines3
-  example where the agent only transports observations/timing and reward/termination 
are computed on RL side.
- [lunar_lander_container/README.md](lunar_lander_container/README.md): split
 deployment of the LunarLander example, with the RL side running in an NVIDIA/CUDA
 container on a remote GPU host and the agent running locally.
- [lunar_lander_remote/README.md](lunar_lander_remote/README.md): split deployment
 of the LunarLander example across remote and local hosts without a container.
- [burger_coppeliasim_remote/README.md](burger_coppeliasim_remote/README.md): split
 deployment of the CoppeliaSim Burger example across remote and local hosts.
- [burger_coppeliasim_container/README.md](burger_coppeliasim_container/README.md): split
 deployment of the CoppeliaSim Burger example with the RL side in an NVIDIA container.
- [burger_real/README.md](burger_real/README.md): ROS2 real Burger agent with a
 pre-trained policy running on a remote RL host.

## Quick run (LunarLander)

From the repository root, install the package and the dependencies for the
LunarLander example:

```bash
pip install -e .
pip install -r examples/lunar_lander/requirements.txt
```

Terminal 1 (RL side first):

```bash
python examples/lunar_lander/rl_side_lunarlander.py
```

Terminal 2 (agent side second):

```bash
python examples/lunar_lander/agent_side_lunarlander.py --render
```

The remote and container deployment variants have their own setup and
networking instructions in their README files.
