# CoppeliaSim Burger with the RL side in a GPU container

The [CoppeliaSim Burger training example](../burger_coppeliasim/README.md) with its two
processes on two machines. The RL side, which trains SAC and computes the reward, runs in
an NVIDIA/CUDA Docker container on a remote Ubuntu 22.04 GPU host. The agent runs on the
local host next to CoppeliaSim and owns the simulator, the scene and the control loop. It
is the container counterpart of
[`burger_coppeliasim_remote`](../burger_coppeliasim_remote/README.md).

```
 local host                                               remote GPU host (Docker)
 CoppeliaSim <-ZMQ-> agent_side_burger_coppeliasim.py --TCP :49054-> container: rl_side_burger_coppeliasim.py
 (scene, physics)                                     <-- or SSH --  (SAC, reward, CUDA)
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
downloads PyTorch, so the first build takes a while and the image is several GB. It holds
the RL side only, without the simulator.

The local host needs CoppeliaSim Edu 4.x (its ZMQ remote API is enabled by default) and
the agent dependencies:

```bash
pip install -e .
pip install -r examples/burger_coppeliasim_container/requirements-agent.txt
```

The agent scripts run `python3` from the active environment (`PYTHON` overrides it).

## Running

Open CoppeliaSim on the local host first. Its GUI is fine and no scene needs to be loaded,
because the agent loads `examples/_common/burger_scene/scene.ttt` itself. CoppeliaSim runs
the scene's Python scripts with the first `python3` on its `PATH`, which needs `pyzmq` and
`cbor2`; start it from a shell where your virtualenv is active, or the scene fails with
`script is not initialized`.

Then start the container, whose RL side waits 60 seconds for the agent once it is
listening.

### Same LAN (profile A)

On the remote host, from the repository root:

```bash
./examples/burger_coppeliasim_container/run_rl_container.sh A --timesteps 20000
```

The container uses host networking, so it listens on the host's LAN interface. Allow TCP
port 49054 from the local machine in the remote firewall. On the local host:

```bash
./examples/burger_coppeliasim_container/run_agent_local.sh <REMOTE_LAN_IP>
```

### SSH tunnel (profile B)

On the remote host:

```bash
./examples/burger_coppeliasim_container/run_rl_container.sh B --timesteps 20000
```

The port is published on the remote loopback only, so nothing is exposed. On the local
host, open the tunnel in one terminal and leave it running:

```bash
./examples/burger_coppeliasim_container/tunnel.sh user@remote-host
```

Then start the agent in another terminal. It connects to `127.0.0.1` by default:

```bash
./examples/burger_coppeliasim_container/run_agent_local.sh
```

If the local port is busy, pick another one with
`./tunnel.sh user@remote-host 49054 50000` and start the agent with `PORT=50000`.

### Options

Arguments after the profile go to `rl_side_burger_coppeliasim.py`. Arguments after the host
go to `agent_side_burger_coppeliasim.py`.

| Variable | Default | Script | Meaning |
|---|---|---|---|
| `PORT` | `49054` | both | TCP port of the RL side |
| `DEVICE` | `cuda` | container | `cuda` passes `--gpus all`; `cpu` runs without the NVIDIA runtime |
| `TIMEOUT` | RL `30`, agent `10` | both | RL: seconds to wait for each agent reply. Agent: seconds to wait for a pending command |
| `IMAGE` | `burger-coppelia-rl` | container | Image name |
| `SIM_HOST`, `SIM_PORT` | `127.0.0.1`, `23000` | agent | Where the agent finds CoppeliaSim |

The slowest reply is the first reset, when the agent has to start the simulation and
rebuild the arena, hence the RL default of 30 seconds. Raise it on a slow or distant link.
The container waits 60 seconds for the agent to connect (the `accept_timeout` of `RLSide`, not exposed by the script), so start the agent promptly.

### The trained model

The model is saved on the remote host and survives the container. `--model-path` takes a
path on the host, and its directory is mounted into the container. The default is
`examples/burger_coppeliasim_container/output/sac_coppelia_burger.zip`. The files belong to
root.

## Trying it on one machine

To see it work without a second host, this opens CoppeliaSim, runs the container and the
agent on one computer (profile A) and stops everything when training ends. It needs
Docker, and with `DEVICE=cpu` it does not need the NVIDIA runtime:

```bash
DEVICE=cpu ./examples/burger_coppeliasim_container/run_local_demo.sh
```

It takes the options of [`../burger_coppeliasim/run_demo.sh`](../burger_coppeliasim/README.md#options)
(`TIMESTEPS` and so on), with 2000 steps by default. The model goes to `output/`.

## Notes

SAC with this small network gains little from a GPU. The example shows a GPU-capable setup
across hosts, which pays off with larger policies. The agent runs the simulation on the
wall clock, so CoppeliaSim has to keep up with real time on the local host. The transport
uses `pickle`, so connect only endpoints you trust. These scripts are a starting point to
adapt to your infrastructure.

## References

- [Burger training example](../burger_coppeliasim/README.md)
- [Bare-host variant](../burger_coppeliasim_remote/README.md)
- [The same split with LunarLander](../lunar_lander_container/README.md)
