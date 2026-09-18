# CoppeliaSim Burger split deployment: RL in a remote container, agent local

This example runs the RL side of the CoppeliaSim Burger demo in an NVIDIA/CUDA
container on a remote GPU host. The local agent still owns CoppeliaSim, the
scene, control loop, and sensor observations.

## Requirements

The remote host needs Docker, an NVIDIA driver compatible with CUDA 12.4, and
the NVIDIA Container Toolkit. Configure and test the runtime as described in
the [Lunar Lander container example](../lunar_lander_container/README.md).

On the local host, install the package, agent dependencies, and CoppeliaSim:

```bash
pip install -e .
pip install -r examples/burger_coppeliasim_container/requirements-agent.txt
```

The local agent loads `examples/burger_coppeliasim/scene/scene.ttt` by default.
Start CoppeliaSim before starting the agent.

## Running

Start the remote RL container first.

### Profile A: same LAN

On the remote host:

```bash
./examples/burger_coppeliasim_container/run_rl_container.sh A --timesteps 20000
```

Then on the local host:

```bash
./examples/burger_coppeliasim_container/run_agent_local.sh <REMOTE_LAN_IP>
```

The container uses host networking in this profile.

### Profile B: SSH tunnel

On the remote host:

```bash
./examples/burger_coppeliasim_container/run_rl_container.sh B --timesteps 20000
```

On the local host:

```bash
./examples/burger_coppeliasim_container/tunnel.sh user@remote-host
./examples/burger_coppeliasim_container/run_agent_local.sh
```

Profile B publishes the container server only on remote loopback. `PORT`
defaults to `49054`, `DEVICE` defaults to `cuda`, and arguments after `A` or
`B` are forwarded to `rl_side_burger_coppeliasim.py`.

The transport uses `pickle`; connect only trusted endpoints.
