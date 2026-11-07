# LunarLander on two hosts

The [LunarLander example](../lunar_lander/README.md) with its two processes on two
machines. The RL side, which trains PPO and computes the reward, runs on a remote host
that usually has a GPU. The agent, which runs the LunarLander physics, runs on the local
host. No container is used; [`lunar_lander_container`](../lunar_lander_container/README.md)
is the variant that does.

```
 local host                                     remote host
 agent_side_lunarlander.py  --- TCP :49054 -->  rl_side_lunarlander.py
 (LunarLander physics)      <--- or SSH ----    (PPO, reward, GPU)
```

`RLSide` is the socket server, so the remote host listens and the local agent connects.
The RL side must therefore start first.

## Installation

On the remote host, in the Python environment that has the PyTorch build you want (for
GPU training, install a CUDA build of PyTorch first):

```bash
pip install -e .
pip install -r examples/lunar_lander_remote/requirements-rl.txt
```

On the local host:

```bash
pip install -e .
pip install -r examples/lunar_lander_remote/requirements-agent.txt
```

The remote host needs no Box2D, and the local host needs no Stable-Baselines3.
`gymnasium[box2d]` may need `swig`, `build-essential` and `python3-dev` to build. The
scripts run `python3` from the active environment; set `PYTHON` to use another
interpreter.

## Running

Start the RL side first. It waits 60 seconds for the agent once it is listening.

### Same LAN (profile A)

On the remote host, from the repository root:

```bash
./examples/lunar_lander_remote/run_rl_remote.sh A --timesteps 20000
```

Allow TCP port 49054 from the local machine in the remote firewall. On the local host:

```bash
./examples/lunar_lander_remote/run_agent_local.sh <REMOTE_LAN_IP>
```

### SSH tunnel (profile B)

Use this when the port cannot be exposed. On the remote host:

```bash
./examples/lunar_lander_remote/run_rl_remote.sh B --timesteps 20000
```

On the local host, open the tunnel in one terminal and leave it running:

```bash
./examples/lunar_lander_remote/tunnel.sh user@remote-host
```

Then start the agent in another terminal. It connects to `127.0.0.1` by default:

```bash
./examples/lunar_lander_remote/run_agent_local.sh
```

The server binds to every interface, so in this profile the remote firewall should block
the port from outside and allow loopback; the tunnel reaches it through the loopback. If
the local port is busy, pick another one with `./tunnel.sh user@remote-host 49054 50000`
and start the agent with `PORT=50000`.

### Options

Arguments after the profile go to `rl_side_lunarlander.py`. Arguments after the host go
to `agent_side_lunarlander.py`, for instance `--render` to show the window (it needs a
display).

| Variable | Default | Script | Meaning |
|---|---|---|---|
| `PORT` | `49054` | both | TCP port of the RL side |
| `DEVICE` | `cuda` | RL | PyTorch device: `cuda`, `cpu` or `auto` |
| `TIMEOUT` | `10` | both | RL: seconds to wait for each agent reply. Agent: seconds to wait for a pending command |
| `PYTHON` | `python3` | both | Interpreter with the dependencies |

On a slow or distant link raise `TIMEOUT`: the RL side gives up when an agent reply takes
longer. The wait of 60 seconds for the agent to connect (the `accept_timeout` of
`RLSide`, not exposed by the script) is short on a slow start, so start the agent promptly; if it expires,
start both again.

## Trying it on one machine

To see it work without a second host, this runs both scripts on one computer over the
loopback interface (profile A) and stops them when training ends:

```bash
./examples/lunar_lander_remote/run_local_demo.sh
```

It takes the options of [`../lunar_lander/run_demo.sh`](../lunar_lander/README.md#options)
(`TIMESTEPS`, `N_STEPS`, `DEVICE` and so on), with `DEVICE=cpu` by default.

## What you should see

On the remote host, the PyTorch device and the PPO logs. On the local host,
`[AGENT] running LunarLander loop` and, when training ends,
`[AGENT] finish command received`. Each step's `lat`, `t_agent` and `t_wall` are in the
`info` dict of the environment.

## Notes

`t_wall` (remote) and `t_agent` (local) come from different clocks and are not directly
comparable. Synchronize the hosts with NTP or chrony, and measure the round-trip time
separately, if you analyze timing. The transport uses `pickle`, so connect only endpoints
you trust. These scripts are a starting point to adapt to your network.

## References

- [LunarLander example](../lunar_lander/README.md)
- [Container variant](../lunar_lander_container/README.md)
- [Main README](../../README.md)
