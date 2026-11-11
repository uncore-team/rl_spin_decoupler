# CoppeliaSim Burger: training a navigation policy

A simulated TurtleBot3 Burger has to reach a goal marker inside a small arena with
random obstacles, using an 8-sector LiDAR and the position of the goal. A Soft
Actor-Critic (SAC) policy is trained for the task, with the simulator and the learner in
separate processes that talk through the `spindecoupler` socket:

```
 CoppeliaSim            agent side                       RL side
 (scene, physics) <-ZMQ-> applies the action,  --TCP-->  SAC and reward
                          returns observation  <--TCP--  (Gymnasium env)
```

The agent (`agent_side_burger_coppeliasim.py`) owns the simulator. It loads the scene,
applies the action chosen by the RL side, keeps the simulation stepping and returns the
observation. It computes no reward. The RL side (`rl_side_burger_coppeliasim.py`) trains
the policy; to Stable-Baselines3 it is an ordinary Gymnasium environment whose `step()`
and `reset()` are answered by the agent over the socket.

This is the pattern of the [LunarLander example](../lunar_lander/README.md) with a
physics simulator in place of a Gym environment. The model trained here is the one that
[`burger_real`](../burger_real/README.md) runs on a real robot and
[`burger_real_sim`](../burger_real_sim/README.md) on a simulated one.

One script starts everything:

```bash
./examples/burger_coppeliasim/run_demo.sh
```

It opens CoppeliaSim with its GUI, loads the arena, starts the RL side and the agent,
trains for 2000 steps (about 7 minutes on a laptop) and saves the model.

## Installation

Tested on Ubuntu 22.04 under WSL2 with CoppeliaSim Edu 4.10. ROS2 is not needed.

1. Install CoppeliaSim Edu 4.x from https://www.coppeliarobotics.com/ in
   `/opt/CoppeliaSim_Edu` (otherwise set `COPPELIASIM_SH`). The ZMQ remote API add-on
   is enabled by default. The GUI needs a graphical session.
2. From the repository root, install the Python packages:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   pip install -r examples/burger_coppeliasim/requirements.txt
   ```

CoppeliaSim runs the scene's Python scripts with the first `python3` on its `PATH`, and
they need `pyzmq` and `cbor2`, which the requirements install. If you open CoppeliaSim
yourself, start it from a shell where the virtualenv is active, or the scene fails with
`script is not initialized`. The demo script takes care of this when it launches
CoppeliaSim.

## Running

```bash
./examples/burger_coppeliasim/run_demo.sh                    # 2000 steps
TIMESTEPS=100000 ./examples/burger_coppeliasim/run_demo.sh   # a real training run
```

The script checks its prerequisites and starts CoppeliaSim (a running instance on port
23000 is reused and left open), then the RL side, then the agent, which loads
`examples/_common/burger_scene/scene.ttt`, starts the simulation and connects. The output
of each component is shown with a prefix and kept in a log directory whose path is
printed at the end. When training ends the RL side saves the model and tells the agent to
stop, and the script exits. Ctrl+C stops it earlier.

2000 steps only show that the loop works; the resulting policy is not useful. Training
runs at 5 to 6 steps per second, because each step lasts 0.08 s of real time and includes
a gradient update, so the RL side's default of 100000 steps takes about five hours.

### Options

| Variable | Default | Meaning |
|---|---|---|
| `TIMESTEPS` | `2000` | Training steps |
| `MODEL_OUT` | `examples/burger_coppeliasim/sac_coppelia_burger.zip` | Where the model is saved |
| `DEVICE` | `cpu` | PyTorch device: `cpu`, `cuda` or `auto` |
| `COPPELIASIM_SH` | `/opt/CoppeliaSim_Edu/coppeliaSim.sh` | CoppeliaSim launcher |
| `COPPELIASIM_ARGS` | empty | Extra CoppeliaSim arguments, for instance `-h` for headless |
| `KEEP_SIM_OPEN` | `0` | `1` leaves CoppeliaSim open afterwards, if the script started it |
| `LOG_DIR` | new directory under `/tmp` | Per-component logs |
| `SIM_PORT`, `RL_PORT` | `23000`, `49054` | CoppeliaSim API port and RL-agent TCP port |

### Timeouts

| Variable | Default | Meaning |
|---|---|---|
| `RL_TIMEOUT` | `30` | Seconds the RL side waits for each agent reply (its own default is 5) |
| `STARTUP_TIMEOUT` | `180` | Seconds the script waits for CoppeliaSim and the RL side to come up |

The slowest reply is the first reset, when the agent has to start the simulation and
rebuild the arena. The library gives the agent 60 seconds to connect once the RL side
listens; since the agent loads the scene before connecting, the script starts it right
after the RL side.

If a component exits, the script says which one and where its log is.

If the agent restarts or the network drops, the connection is lost. The environment is
built on `DecoupledEnv` with `reconnect=True`: the episode in progress is truncated and the
RL side waits up to 60 seconds for the agent. The agent stops the robot and retries for
`--reconnect-timeout` seconds (180 by default, since the RL side may take minutes to
restart), and training goes on with a new episode.

## Task definition

The observation has 10 floats: `[distance_to_goal, relative_goal_angle, ray_1 ... ray_8]`.
The rays are the minimum LiDAR distance in eight 45-degree sectors, in meters, up to 8 m.
The action has 2 floats, `[linear_x, angular_z]`, in `[0, 0.15]` m/s and `[-0.6, 0.6]`
rad/s.

An episode ends in one of three ways:

- Collision, when a ray reads 0.16 m or less. The agent detects it and reports it in the
  payload as `terminated`.
- Goal reached, when the distance to the goal is 0.10 m or less. The RL side decides it.
- Truncation after 500 steps, also decided by the RL side.

The reward (`reward.py`) is -100 for a collision and +200 for reaching the goal.
Otherwise it is 50 times the progress toward the goal, minus 0.05 times the angle error,
0.01 times the action norm and 0.1 per step. Each episode places the robot, the goal (at
least 0.75 m away) and the obstacles at random. Both sides read the thresholds and bounds
from `episode_config.py`, so the agent's collision check and the reward cannot drift
apart.

## How the agent works

The agent runs two nested loops on the wall clock. A fast one (`--control-period`, 10 ms)
steps the simulation with the current action, so the robot keeps moving while the RL side
thinks. A slower one (`--rl-step-period`, 80 ms) decides when to send the next
observation. The agent switches CoppeliaSim to real-time mode when it loads the scene, so
that simulation time follows the wall clock.

On a collision the agent freezes the robot and waits for the reset. It never resets by
itself; the RL side asks for every reset.

## Running the pieces by hand

Open CoppeliaSim first; no scene needs to be loaded. Then, from the repository root with
the virtualenv active, in two terminals:

```bash
python examples/burger_coppeliasim/rl_side_burger_coppeliasim.py \
  --timesteps 2000 --timeout 30 --device cpu                          # waits 60 s for the agent
python examples/burger_coppeliasim/agent_side_burger_coppeliasim.py \
  --ip 127.0.0.1 --port 49054
```

Both accept `--help`. Without `--model-path`, the RL side saves `sac_coppelia_burger.zip`
in the current directory.

## Files

```
run_demo.sh                       starts everything
agent_side_burger_coppeliasim.py  drives CoppeliaSim for the RL side
rl_side_burger_coppeliasim.py     Gymnasium wrapper and SAC training
reward.py                         reward and task termination (RL side)
episode_config.py                 thresholds and bounds shared by both sides
requirements.txt                  Python dependencies
```

The scene is in [`examples/_common/burger_scene`](../_common/burger_scene/README.md).
The example also runs split across two hosts, in
[`burger_coppeliasim_remote`](../burger_coppeliasim_remote/README.md) and
[`burger_coppeliasim_container`](../burger_coppeliasim_container/README.md).

## Notes

The agent and the RL side use the wall clock, so the simulation has to keep up with real
time for the dynamics to be consistent. On a laptop the scene needs about 35 ms of
computation per 50 ms simulation step; on a slower machine each action covers less
simulated time. `--seed` seeds the SAC policy only: the placement of robot, goal and
obstacles is random in every episode. The transport uses `pickle`, so connect only
endpoints you trust.
