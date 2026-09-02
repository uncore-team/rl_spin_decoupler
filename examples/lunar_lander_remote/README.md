# LunarLander split deployment: RL on a remote host, agent local

This example runs the decoupled LunarLander demo across two machines. The RL
side runs directly in Python on a remote GPU-capable host, while the local
agent runs Gymnasium `LunarLander-v3`. No image is built and no container is
launched.

## What it demonstrates

- `run_rl_remote.sh` starts the remote `RLSide` server, PPO training, and
  RL-side reward computation.
- `run_agent_local.sh` starts the local agent, which owns LunarLander Box2D
  physics and transports observations, physics flags, and timing.
- Profile A connects directly over a shared LAN; Profile B forwards the
  connection through SSH with `tunnel.sh`.

`RLSide` is the socket server, so the remote host listens and the local agent
initiates the connection.

## Requirements and installation

On the remote host, install the package and the RL dependencies in the Python
environment that has the desired PyTorch build:

```bash
pip install -e .
pip install -r examples/lunar_lander_remote/requirements-rl.txt
```

On the local host, install the package and agent dependencies:

```bash
pip install -e .
pip install -r examples/lunar_lander_remote/requirements-agent.txt
```

For GPU training, install a PyTorch wheel compatible with the remote NVIDIA
driver. PPO with this small MLP also works with `DEVICE=cpu`.

## Running (correct order)

Start the remote RL side before the local agent. It waits up to 60 seconds for
the agent connection.

### Profile A: same LAN

On the remote host, from the repository root:

```bash
./examples/lunar_lander_remote/run_rl_remote.sh A --timesteps 20000
```

Allow TCP port `49054` from the local machine in the remote firewall. Then on
the local host run:

```bash
./examples/lunar_lander_remote/run_agent_local.sh <REMOTE_LAN_IP>
```

### Profile B: SSH tunnel

On the remote host:

```bash
./examples/lunar_lander_remote/run_rl_remote.sh B --timesteps 20000
```

On the local host, open the tunnel in one terminal:

```bash
./examples/lunar_lander_remote/tunnel.sh user@remote-host
```

Then start the agent in a second terminal:

```bash
./examples/lunar_lander_remote/run_agent_local.sh
```

The server binds to `0.0.0.0`; therefore Profile B needs a remote firewall
rule that blocks external access to port `49054` while allowing loopback. The
SSH tunnel forwards the local port to the remote loopback endpoint.

`PORT` defaults to `49054`, `DEVICE` defaults to `cuda`, and arguments after
`A` or `B` are forwarded to `rl_side_lunarlander.py`.

## What you should see

- The remote RL process reports the effective PyTorch device and SB3/PPO
  training logs.
- The local agent runs LunarLander and can render with `--render`.
- Per-step `lat`, `ato`, and remote `t_wall` values are available in `info`.

## Scope of the example

This is a manual deployment template, not a CI-tested workflow. It exercises
independent clock domains: remote `t_wall` and local `ATO` are not directly
comparable unless the hosts synchronize their clocks. Use NTP or chrony and
measure round-trip time separately for timing analysis. The transport uses
`pickle`, so connect only trusted endpoints.

## References

- Base LunarLander example: [../lunar_lander/README.md](../lunar_lander/README.md)
- Container deployment: [../lunar_lander_container/README.md](../lunar_lander_container/README.md)
- Main project README: [../../README.md](../../README.md)