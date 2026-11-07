# CoppeliaSim Burger on two hosts

The [CoppeliaSim Burger training example](../burger_coppeliasim/README.md) with its two
processes on two machines. The RL side, which trains SAC and computes the reward, runs on
a remote host that usually has a GPU. The agent runs on the local host next to
CoppeliaSim and owns the simulator, the scene and the control loop. No container is used;
[`burger_coppeliasim_container`](../burger_coppeliasim_container/README.md) is the variant
that does.

```
 local host                                               remote host
 CoppeliaSim <-ZMQ-> agent_side_burger_coppeliasim.py --TCP :49054-> rl_side_burger_coppeliasim.py
 (scene, physics)                                     <-- or SSH --  (SAC, reward, GPU)
```

`RLSide` is the socket server, so the remote host listens and the local agent connects.
The RL side must therefore start first.

## Installation

On the remote host, in the Python environment that has the PyTorch build you want (for GPU
training, install a CUDA build of PyTorch first):

```bash
pip install -e .
pip install -r examples/burger_coppeliasim_remote/requirements-rl.txt
```

On the local host, install CoppeliaSim Edu 4.x (its ZMQ remote API is enabled by default)
and the agent dependencies:

```bash
pip install -e .
pip install -r examples/burger_coppeliasim_remote/requirements-agent.txt
```

The remote host needs no simulator, and the local host needs no Stable-Baselines3. The
scripts run `python3` from the active environment; set `PYTHON` to use another
interpreter.

## Running

Open CoppeliaSim on the local host first. Its GUI is fine and no scene needs to be loaded,
because the agent loads `examples/_common/burger_scene/scene.ttt` itself. CoppeliaSim runs
the scene's Python scripts with the first `python3` on its `PATH`, which needs `pyzmq` and
`cbor2`; start it from a shell where your virtualenv is active, or the scene fails with
`script is not initialized`.

Then start the RL side, which waits 60 seconds for the agent once it is listening.

### Same LAN (profile A)

On the remote host, from the repository root:

```bash
./examples/burger_coppeliasim_remote/run_rl_remote.sh A --timesteps 20000
```

Allow TCP port 49054 from the local machine in the remote firewall. On the local host:

```bash
./examples/burger_coppeliasim_remote/run_agent_local.sh <REMOTE_LAN_IP>
```

### SSH tunnel (profile B)

Use this when the port cannot be exposed. On the remote host:

```bash
./examples/burger_coppeliasim_remote/run_rl_remote.sh B --timesteps 20000
```

On the local host, open the tunnel in one terminal and leave it running:

```bash
./examples/burger_coppeliasim_remote/tunnel.sh user@remote-host
```

Then start the agent in another terminal. It connects to `127.0.0.1` by default:

```bash
./examples/burger_coppeliasim_remote/run_agent_local.sh
```

The server binds to every interface, so in this profile the remote firewall should block
the port from outside and allow loopback. If the local port is busy, pick another one with
`./tunnel.sh user@remote-host 49054 50000` and start the agent with `PORT=50000`.

### Options

Arguments after the profile go to `rl_side_burger_coppeliasim.py` (`--timesteps`,
`--model-path` and so on). Arguments after the host go to
`agent_side_burger_coppeliasim.py`, for instance `--debug`.

| Variable | Default | Script | Meaning |
|---|---|---|---|
| `PORT` | `49054` | both | TCP port of the RL side |
| `DEVICE` | `cuda` | RL | PyTorch device: `cuda`, `cpu` or `auto` |
| `TIMEOUT` | RL `30`, agent `10` | both | RL: seconds to wait for each agent reply. Agent: seconds to wait for a pending command |
| `SIM_HOST`, `SIM_PORT` | `127.0.0.1`, `23000` | agent | Where the agent finds CoppeliaSim |
| `PYTHON` | `python3` | both | Interpreter with the dependencies |

The model is saved on the remote host at `--model-path` (default `sac_coppelia_burger.zip`
in the current directory).

The slowest reply is the first reset, when the agent has to start the simulation and
rebuild the arena, hence the RL default of 30 seconds. Raise it on a slow or distant link.
The 60-second wait for the agent to connect cannot be changed, so start the agent promptly.

## Trying it on one machine

To see it work without a second host, this opens CoppeliaSim, runs both scripts on one
computer over the loopback interface (profile A) and stops everything when training ends:

```bash
./examples/burger_coppeliasim_remote/run_local_demo.sh
```

It takes the options of [`../burger_coppeliasim/run_demo.sh`](../burger_coppeliasim/README.md#options)
(`TIMESTEPS`, `DEVICE` and so on), with `DEVICE=cpu` and 2000 steps by default.

## Notes

The agent runs the simulation on the wall clock, so CoppeliaSim has to keep up with real
time on the local host. The transport uses `pickle`, so connect only endpoints you trust.
These scripts are a starting point to adapt to your network.

## References

- [Burger training example](../burger_coppeliasim/README.md)
- [Container variant](../burger_coppeliasim_container/README.md)
- [The same split with LunarLander](../lunar_lander_remote/README.md)
