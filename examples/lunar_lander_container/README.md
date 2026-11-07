# LunarLander with the RL side in a GPU container

The [LunarLander example](../lunar_lander/README.md) with its two processes on two
machines. The RL side, which trains PPO and computes the reward, runs in an
NVIDIA/CUDA Docker container on a remote Ubuntu 22.04 GPU host. The agent, which runs
the LunarLander physics, runs on the local host. It is the container counterpart of
[`lunar_lander_remote`](../lunar_lander_remote/README.md).

```
 local host                                     remote GPU host (Docker)
 agent_side_lunarlander.py  --- TCP :49054 -->  container: rl_side_lunarlander.py
 (LunarLander physics)      <--- or SSH ----    (PPO, reward, CUDA)
```

`RLSide` is the socket server, so the container listens and the local agent connects. The
container must therefore start first.

## Installation

The remote host needs an NVIDIA driver recent enough for CUDA 12.4 (550 or newer), Docker
and the NVIDIA Container Toolkit. Configure the runtime once:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The image is built the first time the container starts, or with `./build_image.sh`. It
downloads PyTorch, so the first build takes a while and the image is several GB. It
holds the RL side only and has no Box2D.

The local host needs the package and the agent dependencies:

```bash
pip install -e .
pip install -r examples/lunar_lander_container/requirements-agent.txt
```

`gymnasium[box2d]` may need `swig`, `build-essential` and `python3-dev` to build. The
agent scripts run `python3` from the active environment (`PYTHON` overrides it).

## Running

Start the container first. The RL side waits 60 seconds for the agent once it is
listening.

### Same LAN (profile A)

On the remote host, from the repository root:

```bash
./examples/lunar_lander_container/run_rl_container.sh A --timesteps 20000
```

The container uses host networking, so it listens on the host's LAN interface. Allow TCP
port 49054 from the local machine in the remote firewall. On the local host:

```bash
./examples/lunar_lander_container/run_agent_local.sh <REMOTE_LAN_IP>
```

### SSH tunnel (profile B)

On the remote host:

```bash
./examples/lunar_lander_container/run_rl_container.sh B --timesteps 20000
```

The port is published on the remote loopback only, so nothing is exposed. On the local
host, open the tunnel in one terminal and leave it running:

```bash
./examples/lunar_lander_container/tunnel.sh user@remote-host
```

Then start the agent in another terminal. It connects to `127.0.0.1` by default:

```bash
./examples/lunar_lander_container/run_agent_local.sh
```

If the local port is busy, pick another one with
`./tunnel.sh user@remote-host 49054 50000` and start the agent with `PORT=50000`.

### Options

Arguments after the profile go to `rl_side_lunarlander.py`. Arguments after the host go
to `agent_side_lunarlander.py`, for instance `--render` (it needs a display).

| Variable | Default | Script | Meaning |
|---|---|---|---|
| `PORT` | `49054` | both | TCP port of the RL side |
| `DEVICE` | `cuda` | container | `cuda` passes `--gpus all`; `cpu` runs without the NVIDIA runtime |
| `TIMEOUT` | `10` | both | RL: seconds to wait for each agent reply. Agent: seconds to wait for a pending command |
| `IMAGE` | `lunarlander-rl` | container | Image name |
| `PYTHON` | `python3` | agent | Interpreter with the agent dependencies |

On a slow or distant link raise `TIMEOUT`. The RL side waits 60 seconds for the agent to
connect (the `accept_timeout` of `RLSide`, not exposed by the script), so start the agent promptly.

### The trained model

The model is saved on the remote host and survives the container. `--model-path` takes a
path on the host, and its directory is mounted into the container. The default is
`examples/lunar_lander_container/output/ppo_lunar_lander_decoupled.zip`. The files belong
to root.

## Trying it on one machine

To see it work without a second host, this runs the container and the agent on one
computer (profile A) and stops them when training ends. It needs Docker, and with
`DEVICE=cpu` it does not need the NVIDIA runtime:

```bash
DEVICE=cpu ./examples/lunar_lander_container/run_local_demo.sh
```

It takes the options of [`../lunar_lander/run_demo.sh`](../lunar_lander/README.md#options)
(`TIMESTEPS`, `N_STEPS` and so on). The model goes to `output/`.

## What you should see

In the container, the `nvidia-smi` table, the PyTorch device and the PPO logs. On the
local host, `[AGENT] running LunarLander loop` and, when training ends,
`[AGENT] finish command received`.

## Notes

PPO with LunarLander's small network gains little from a GPU. The example is there to
show a GPU-capable setup across hosts, which pays off with larger policies. The remote
`t_wall` and the local `t_agent` come from different clocks, so synchronize the hosts
before comparing them. The transport uses `pickle`, so connect only endpoints you trust.
These scripts are a starting point to adapt to your infrastructure.

## References

- [LunarLander example](../lunar_lander/README.md)
- [Bare-host variant](../lunar_lander_remote/README.md)
- [Main README](../../README.md)
