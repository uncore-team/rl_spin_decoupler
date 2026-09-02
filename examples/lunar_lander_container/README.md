# LunarLander split deployment: RL in a remote GPU container, agent local

This example runs the decoupled LunarLander demo across two machines. The RL
side runs in an NVIDIA/CUDA container on a remote Ubuntu 22.04 GPU host, while
the local agent runs Gymnasium `LunarLander-v3` and its Box2D physics.

## What it demonstrates

- `run_rl_container.sh` starts the remote `RLSide` server in a GPU-enabled
  Docker container, trains PPO, and computes the learning signal.
- `run_agent_local.sh` starts the local agent, which owns the simulation and
  transports observations, physics flags, and timing.
- Profile A connects directly on a shared LAN; Profile B publishes only the
  remote loopback port and forwards it over SSH with `tunnel.sh`.

`RLSide` is the socket server, so the remote host listens and the local agent
initiates the connection.

## Requirements and installation

The remote host needs a recent NVIDIA driver for CUDA 12.4 (approximately
550+), Docker, and the NVIDIA Container Toolkit. Configure the runtime once:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

On the local host, install the package and the agent dependencies:

```bash
pip install -e .
pip install -r examples/lunar_lander_container/requirements-agent.txt
```

The container installs the RL-side package and dependencies itself. It does
not need `gymnasium[box2d]` because the local agent runs the simulation.

## Running (correct order)

Start the remote RL container before the local agent. It waits up to 60
seconds for the agent connection.

### Profile A: same LAN

On the remote host, from the repository root:

```bash
./examples/lunar_lander_container/run_rl_container.sh A --timesteps 20000
```

Allow TCP port `49054` from the local machine in the remote firewall. Then on
the local host run:

```bash
./examples/lunar_lander_container/run_agent_local.sh <REMOTE_LAN_IP>
```

This profile uses Docker host networking, so the container listens directly on
the remote host's LAN interface.

### Profile B: SSH tunnel

On the remote host:

```bash
./examples/lunar_lander_container/run_rl_container.sh B --timesteps 20000
```

On the local host, open the tunnel in one terminal:

```bash
./examples/lunar_lander_container/tunnel.sh user@remote-host
```

Then start the agent in a second terminal:

```bash
./examples/lunar_lander_container/run_agent_local.sh
```

Profile B publishes the container server only on remote loopback, so nothing
is exposed publicly. `PORT` defaults to `49054`, `DEVICE` defaults to `cuda`,
and arguments after `A` or `B` are forwarded to `rl_side_lunarlander.py`.

## What you should see

- The remote container reports `nvidia-smi`, the effective PyTorch device, and
  SB3/PPO training logs.
- The local agent runs LunarLander and can render with `--render`.
- Per-step `lat`, `ato`, and remote `t_wall` values are available in `info`.

## Scope of the example

This is a manual deployment template, not a CI-tested workflow. PPO with
LunarLander's small MLP benefits little from a GPU; the example demonstrates a
GPU-capable, cross-host architecture that is more useful for larger policies.
Remote `t_wall` and local `ATO` are separate clock domains, so synchronize
hosts with NTP or chrony and measure round-trip time separately for timing
analysis. The transport uses `pickle`, so connect only trusted endpoints.

## References

- Base LunarLander example: [../lunar_lander/README.md](../lunar_lander/README.md)
- Bare-host deployment: [../lunar_lander_remote/README.md](../lunar_lander_remote/README.md)
- Main project README: [../../README.md](../../README.md)