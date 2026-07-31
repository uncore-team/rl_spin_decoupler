# LunarLander split deployment: RL on a remote GPU container, agent local

This example runs the same decoupled LunarLander demo as
[`../lunar_lander/`](../lunar_lander/README.md), but split across two machines:

- The RL side (`RLSide`, the socket **server**) runs in an NVIDIA/CUDA
  container on a **remote** Ubuntu 22.04 host with a GPU. It trains PPO and
  computes the learning signal.
- The agent side (`AgentSide`, the socket **client**) runs on your **local**
  Ubuntu 22.04 host. It executes the Gymnasium `LunarLander-v3` environment
  (Box2D physics, optional rendering) and only transports observations/timing.

Because `RLSide` is the server, the **remote host listens** and the **local
host initiates** the connection. That direction drives the firewall / port /
tunnel setup below.

## Dependencies: what runs where

The RL container needs only Stable-Baselines3 + PyTorch (and gymnasium for its
spaces/`Env` base class). LunarLander's Box2D physics runs in the agent process
on the local host, so the container needs no `swig` / `gymnasium[box2d]`. This
keeps the image lean.

## About the GPU

PPO with an `MlpPolicy` over an 8-dimensional observation is a tiny model, and
a GPU barely accelerates it (the host/device transfer for small batches often
cancels any gain). This example is a **deployment template** that demonstrates
the full stack — NVIDIA Container Toolkit, a CUDA-enabled PyTorch in a
container, and cross-host decoupling — which generalizes to GPU-hungry
workloads (e.g. SAC / CNN policies). The RL script prints its effective torch
device on startup so you can confirm the GPU is wired up.

## Prerequisites (remote host, once)

The host NVIDIA driver must be recent enough for CUDA 12.4 (driver >= ~550),
Docker must be installed, and the NVIDIA Container Toolkit configured:

```bash
nvidia-smi   # host driver works?

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# verify GPU visibility inside a container:
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Two network profiles (one image)

The container image is network-agnostic; the profile is chosen at run time.

### Profile A — same LAN (default, no library change required in practice)

The container shares the host network stack (`--network host`) and the agent
connects to the remote host's LAN IP.

Remote host (from the repository root):

```bash
./examples/lunar_lander_container/run_rl_container.sh A --timesteps 20000
```

Open the port to your local machine on the remote host, e.g. with ufw:

```bash
sudo ufw allow from <LOCAL_IP> to any port 49054 proto tcp
```

Local host (from the repository root):

```bash
./examples/lunar_lander_container/run_agent_local.sh <REMOTE_LAN_IP>
```

Note: `--network host` shares the host's network namespace (no network
isolation for the container, and `--port` occupies that port directly on the
remote host). That is normal and appropriate for a dedicated compute box you
control.

### Profile B — across networks, via SSH tunnel (recommended when not on one LAN)

The container publishes the server only on the remote loopback, and an SSH
local-forward carries the connection. Nothing is exposed publicly.

Remote host:

```bash
./examples/lunar_lander_container/run_rl_container.sh B --timesteps 20000
```

Local host, in one terminal (open the tunnel):

```bash
./examples/lunar_lander_container/tunnel.sh user@remote-host
```

Local host, in another terminal (agent connects to 127.0.0.1 by default):

```bash
./examples/lunar_lander_container/run_agent_local.sh
```

`RLSide` / `ServerCommPoint` always bind to `0.0.0.0` (every interface) by
design, so both profiles work unmodified: this is what lets the server accept
the container's NAT-forwarded connection in Profile B and the direct LAN
connection in Profile A, with no configuration needed.

## Start order and connection window

Start the RL container first: `RLSide` blocks on `accept()` for up to 60
seconds waiting for the agent. Then (open the tunnel if using Profile B and)
start the agent. The agent client raises on a refused connection; the wrapper
scripts run a single attempt, so if you miss the window, just relaunch the
agent.

Use `--rl-step-period 0.08` (as the wrapper does) rather than the tiny values
from the local smoke test, to leave room for real network round-trip time.

## Timing across two machines

This is the first example that genuinely exercises the two cross-host clock
domains the library was built to preserve. `t_wall` comes from the RL (remote)
clock and `ATO` from the agent (local) clock; the two machines are not
synchronized, so `t_wall - ATO` mixes clock offset with latency. To analyze
timing seriously, enable NTP/chrony on both hosts and measure round-trip time
separately. Per-step `lat` / `ato` / `t_wall` are available in the `info` dict.

## Security note

The transport uses `pickle` over the socket, so it must only be used between
trusted endpoints. Both ends here are your own machines, and Profile B keeps
the channel inside an SSH tunnel.

## Scope and CI

Unlike the other examples, this one is **not** exercised in CI (it needs a GPU,
a container runtime, and two hosts). Validate it manually. To sanity-check the
logic without any of that, run the plain two-terminal local variant first:

```bash
# terminal 1
python examples/lunar_lander/rl_side_lunarlander.py --device cpu --timesteps 2000
# terminal 2
python examples/lunar_lander/agent_side_lunarlander.py
```

## References

- Base LunarLander example: [../lunar_lander/README.md](../lunar_lander/README.md)
- Main project README: [../../README.md](../../README.md)
