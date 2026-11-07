# Real Burger agent against a simulated robot

[`burger_real`](../burger_real/README.md) runs a trained policy on a real TurtleBot3
through a ROS2 agent. This example runs the same agent and the same RL process with no
hardware. A CoppeliaSim scene stands in for the robot, and a small bridge node gives it
the ROS2 interface of the real one: `/scan`, `/odom`, `/cmd_vel`, `/goal_pose` and TF.

One script starts everything:

```bash
./examples/burger_real_sim/run_demo.sh
```

It opens CoppeliaSim with its GUI, loads the arena, starts the bridge, the RL side with a
pre-trained SAC policy, and the ROS2 agent, runs 5 episodes of one target each, and shuts
everything down.

```
 CoppeliaSim scene        burger_sim_bridge         burger_real_agent       RL side
 (_common/burger_scene)   (drivers + operator)      (ROS2 node, as is)      (SAC policy, as is)
 physics, LiDAR  <-ZMQ->  /scan /odom /tf /clock -> observation --TCP-->   action
 robot, target            /goal_pose  <- /cmd_vel <- action      <--TCP--
```

## Installation

Tested on Ubuntu 22.04 under WSL2 with ROS2 Humble and CoppeliaSim Edu 4.10.

1. Install ROS2 Humble and colcon:
   `sudo apt install ros-humble-ros-base python3-colcon-common-extensions`.
2. Install CoppeliaSim Edu 4.x from https://www.coppeliarobotics.com/ in
   `/opt/CoppeliaSim_Edu` (otherwise set `COPPELIASIM_SH`). The ZMQ remote API add-on is
   enabled by default. The GUI needs a graphical session.
3. From the repository root, install the Python packages. The trained policy
   `examples/burger_real/training_8rays_fixedtime.zip` is already in the repository.

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   pip install -r examples/burger_real_sim/requirements.txt
   ```

CoppeliaSim runs the scene's Python scripts with the first `python3` on its `PATH`, and
they need `pyzmq` and `cbor2`, which the requirements install. If you open CoppeliaSim
yourself, start it from a shell where the virtualenv is active, or the scene fails with
`script is not initialized`. The demo script takes care of this when it launches
CoppeliaSim.

## Running

```bash
./examples/burger_real_sim/run_demo.sh                 # 5 episodes
EPISODES=3 ./examples/burger_real_sim/run_demo.sh
EPISODES=0 ./examples/burger_real_sim/run_demo.sh      # until Ctrl+C
```

The script checks its prerequisites, builds the two ROS2 packages (a few seconds), and
starts CoppeliaSim (a running instance on port 23000 is reused and left open), the
bridge, which loads `scene.ttt` and starts the simulation, the RL side, which loads the
policy, and the agent, waiting for each to be ready before starting the next. The output
of every component is shown with a prefix and kept in a log directory whose path is
printed at the end. When the RL side has completed `EPISODES` episodes it tells the agent
to stop, and the script exits. Ctrl+C stops it earlier.

### Options

| Variable | Default | Meaning |
|---|---|---|
| `EPISODES` | `5` | Episodes (one target each) to run; `0` means no limit |
| `MODEL_PATH` | `examples/burger_real/training_8rays_fixedtime.zip` | Trained SAC policy |
| `DEVICE` | `cpu` | PyTorch device: `cpu`, `cuda` or `auto` |
| `COPPELIASIM_SH` | `/opt/CoppeliaSim_Edu/coppeliaSim.sh` | CoppeliaSim launcher |
| `COPPELIASIM_ARGS` | empty | Extra CoppeliaSim arguments, for instance `-h` for headless |
| `ROS_SETUP` | `/opt/ros/humble/setup.bash` | ROS2 setup script |
| `KEEP_SIM_OPEN` | `0` | `1` leaves CoppeliaSim open afterwards, if the script started it |
| `LOG_DIR` | new directory under `/tmp` | Per-component logs |
| `SIM_PORT`, `RL_PORT` | `23000`, `49054` | CoppeliaSim API port and RL-agent TCP port |

### Timeouts

| Variable | Default | Meaning |
|---|---|---|
| `RL_TIMEOUT` | `60` | Seconds the RL side waits for each agent reply |
| `SENSOR_TIMEOUT` | `10` | Seconds of sensor silence before the agent truncates an episode |
| `STARTUP_TIMEOUT` | `180` | Seconds the script waits for CoppeliaSim, the bridge and the RL side to come up |

The slowest reply is a reset: the agent waits for the bridge to reposition the world and
offer a new goal. The library gives the agent 60 seconds to connect once the RL side
listens, so the script starts the agent right after it. If a component exits, the script
says which one and where its log is.

## ROS2 interface

It is the interface of the real robot:

| Topic | Type | Bridge | Rate (simulation time) |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | subscribes | |
| `/scan` | `sensor_msgs/LaserScan` | publishes | 5 Hz, like a TurtleBot3 LDS-01 |
| `/odom` | `nav_msgs/Odometry` | publishes | 20 Hz |
| `/goal_pose` | `geometry_msgs/PoseStamped`, frame `map` | publishes | at each mission start, every 0.5 s until the agent moves |
| `/clock` | `rosgraph_msgs/Clock` | publishes | each simulation step |
| TF | `map` to `odom` to `base_link`, and static `base_link` to `base_scan` | publishes | 20 Hz |

`/scan` carries the 8 LiDAR sector minima (45 degrees each, up to 8 m). The rates are
enforced in simulation time, and the scene steps at 20 Hz, which bounds how fresh `/odom`
can be. The agent runs with `use_sim_time:=true` to follow `/clock`.

## How it works

The scene ([`_common/burger_scene`](../_common/burger_scene/README.md)) is plain
CoppeliaSim with no ROS2 in it. The bridge drives it over the ZMQ remote API with one
batched call per cycle, and plays two roles that belong to different parties on a real
robot.

As the robot's drivers, it applies the latest `/cmd_vel`, reads the whole state back and
publishes the topics above.

As the operator, it deals with the end of episodes. The agent alone decides when an
episode is over (goal reached, collision, step budget) and stops commanding motion when it
is. When `/cmd_vel` has been idle for 1 s of simulation time, the bridge repositions
robot, target and obstacles, and offers the new `/goal_pose` until the agent moves again.
On every reset the agent blocks until a new goal arrives, just as it would wait for a
person to publish one. The world is only moved while the robot is already stopped.

The one thing specific to the simulation is that reposition command. With a real robot the
bridge disappears: the robot's own drivers publish the topics, and a person or a mission
node publishes the goals.

## Running the pieces by hand

Open CoppeliaSim first; no scene needs to be loaded. Then, from the repository root with
the virtualenv active, one terminal for each:

```bash
# 1. bridge: loads the scene and starts the simulation
source /opt/ros/humble/setup.bash
(cd examples/burger_real_sim/ros2 && colcon build --symlink-install)
source examples/burger_real_sim/ros2/install/setup.bash
ros2 run burger_sim_bridge bridge_node --scene-path examples/_common/burger_scene/scene.ttt

# 2. RL side (waits 60 s for the agent)
python examples/burger_real/rl_side_burger_real.py \
  --model-path examples/burger_real/training_8rays_fixedtime.zip --device cpu \
  --timeout 60 --episodes 5

# 3. agent
source /opt/ros/humble/setup.bash
(cd examples/burger_real/ros2 && colcon build --symlink-install)
source examples/burger_real/ros2/install/setup.bash
ros2 run burger_real_agent agent_node --ros-args -p use_sim_time:=true -- \
  --ip 127.0.0.1 --port 49054 --sensor-timeout 10
```

Leave out `--scene-path` to run the bridge on the scene already open in CoppeliaSim. If
other ROS2 systems run on your network, set a private `ROS_DOMAIN_ID` in every terminal so
they do not find each other.

## Files

```
run_demo.sh                    starts everything
requirements.txt               Python dependencies
ros2/burger_sim_bridge/        ROS2 package with bridge_node.py and contract.py
```

The scene is in [`examples/_common/burger_scene`](../_common/burger_scene/README.md), and
the agent and RL side that run here are in [`examples/burger_real`](../burger_real/README.md).

## Notes

The agent and the RL side use the wall clock (each action lasts 0.08 s of real time), so
the simulation has to keep up with real time for the dynamics to resemble the real robot.
On a laptop the scene needs about 35 ms of computation per 50 ms simulation step, so it
runs in real time. That relies on `laser.py` computing the LiDAR sectors with plain
arithmetic: a simulator call for each LiDAR point made a step four times slower, and the
simulation ran at about a third of real time. On a slower machine the simulation falls
behind real time and each action covers less simulated time; the demo still runs, but
episodes are more likely to end by truncation.

The scene assigns LiDAR points to the eight sectors by their angle in the robot frame. A
policy trained on a scene that used each sensor's own frame has to be revalidated or
retrained here. Robot, target and obstacles are placed at random in every mission, and the
demo is not seeded.
