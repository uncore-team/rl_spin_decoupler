# CoppeliaSim Burger split deployment: RL on a remote host, agent local

This example runs the CoppeliaSim Burger demo across two machines. The RL
side runs directly in Python on a remote host, while the local agent owns
CoppeliaSim, the scene, control loop, and sensor observations.

## Requirements

Install the package and RL dependencies on the remote host:

```bash
pip install -e .
pip install -r examples/burger_coppeliasim_remote/requirements-rl.txt
```

On the local host, install the package, the agent dependencies, and a local
CoppeliaSim installation with ZMQ remote API enabled:

```bash
pip install -e .
pip install -r examples/burger_coppeliasim_remote/requirements-agent.txt
```

The local agent loads `examples/burger_coppeliasim/scene/scene.ttt` by default.
Start CoppeliaSim on the local host before starting the agent.

## Running

Start the remote RL side first. It waits for the agent connection.

### Profile A: same LAN

On the remote host:

```bash
./examples/burger_coppeliasim_remote/run_rl_remote.sh A --timesteps 20000
```

On the local host, allow TCP port `49054` and run:

```bash
./examples/burger_coppeliasim_remote/run_agent_local.sh <REMOTE_LAN_IP>
```

### Profile B: SSH tunnel

On the remote host:

```bash
./examples/burger_coppeliasim_remote/run_rl_remote.sh B --timesteps 20000
```

On the local host, open the tunnel and then start the agent in another terminal:

```bash
./examples/burger_coppeliasim_remote/tunnel.sh user@remote-host
./examples/burger_coppeliasim_remote/run_agent_local.sh
```

`PORT` defaults to `49054`, `DEVICE` defaults to `cuda`, and arguments after
`A` or `B` are forwarded to `rl_side_burger_coppeliasim.py`. Use `SIM_HOST`
and `SIM_PORT` to point the local agent at CoppeliaSim when it is not running
on `127.0.0.1:23000`.

The transport uses `pickle`; connect only trusted endpoints.
