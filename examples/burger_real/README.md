# Real TurtleBot3 Burger: running a trained policy

A trained navigation policy drives a real TurtleBot3 Burger to the goals it is given. The
process on the robot is a ROS2 node, the agent. The policy runs in a separate RL process,
possibly on another machine, which only does inference and never learns. They talk
through the `spindecoupler` socket:

```
 Burger host (ROS2 Humble)                            RL host
 /scan /odom /goal_pose /tf --> agent_node --TCP:49054--> rl_side_burger_real.py
 /cmd_vel                   <--            <--or SSH----  (SAC policy, reward)
```

The decoupled contract is the one of [`burger_coppeliasim`](../burger_coppeliasim/README.md),
where the policy is trained in simulation. A pre-trained policy,
`training_8rays_fixedtime.zip`, is included in this folder.

To try this agent and RL process without a robot, [`burger_real_sim`](../burger_real_sim/README.md)
runs them against a simulated Burger with one script.

Safety: a learned policy is driving a real robot. Check the topics and the TF with
`/cmd_vel` disconnected first, start in a clear space and keep a physical emergency stop
within reach. The agent clips the velocity to 0.15 m/s and 0.6 rad/s.

## How a mission works

The agent keeps the robot still until a `/goal_pose` arrives. Then it drives with the
policy until one of these happens:

- The goal is reached (distance of 0.10 m or less). The RL side decides it.
- A collision: a LiDAR sector reads 0.16 m or less. The agent decides it and stops the
  robot.
- The sensors go silent: none of `/scan`, `/odom` or `/goal_pose` arrives for
  `--sensor-timeout` (1 s). The agent stops the robot and truncates the attempt.
- The step budget (500 steps) runs out. The RL side decides it.

At the end of an attempt the RL side resets, and the agent stops and waits for the next
`/goal_pose` message before starting again. What counts is a new message, not new
coordinates, so publishing the same goal twice starts a new mission. Whoever acts as the
operator (a person with `ros2 topic pub`, a mission node, or the simulator bridge) starts
every mission by publishing a goal.

## ROS2 contract

The agent publishes the action as `geometry_msgs/msg/Twist` on `/cmd_vel` and subscribes to:

| Topic | Type | Notes |
|---|---|---|
| `/scan` | `sensor_msgs/msg/LaserScan` | 360 degrees, reduced to 8 sector minima of 45 degrees, clamped to 8 m |
| `/odom` | `nav_msgs/msg/Odometry` | pose of the robot in `odom` |
| `/goal_pose` | `geometry_msgs/msg/PoseStamped` | frame `map` (or `odom`) |
| TF | `map` to `odom` | to express the goal in the odometry frame |

The observation has 10 floats, `[distance_to_goal, relative_goal_angle, ray_1 ... ray_8]`,
with the rays in meters. The action is `[linear_x, angular_z]`, limited to `[0.0, 0.15]`
m/s and `[-0.6, 0.6]` rad/s. A model has to be trained with these observation and action
spaces. The constants are in `ros2/burger_real_agent/contract.py`.

## Installation

On the Burger host, with ROS2 Humble, install the Python dependencies and build the agent
package:

```bash
source /opt/ros/humble/setup.bash
pip install -e .                       # spindecoupler, from the repository root
pip install -r examples/burger_real/requirements-agent.txt
cd examples/burger_real/ros2
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The robot needs live publishers for the topics above and a valid `map` to `odom`
transform from your localization stack. Check them with `ros2 topic hz` and
`ros2 run tf2_ros tf2_echo map odom` before enabling `/cmd_vel`.

On the RL host, from the repository root (for GPU inference, install a CUDA build of
PyTorch first):

```bash
pip install -e .
pip install -r examples/burger_real/requirements-rl.txt
```

## Running

Start the RL side first; it waits 60 seconds for the agent once it is listening. On the RL
host:

```bash
./examples/burger_real/run_rl_remote.sh                  # the included model
./examples/burger_real/run_rl_remote.sh my_policy.zip    # another model
```

On the Burger host, after sourcing ROS2 and the workspace:

```bash
./examples/burger_real/run_agent_ros2.sh <RL_HOST_IP>
```

Allow TCP port 49054 from the robot in the firewall of the RL host. If the port cannot be
exposed, forward it through SSH from the Burger host. Leave the tunnel running,

```bash
./examples/burger_real/tunnel.sh user@rl-host
```

and start the agent in another terminal; it connects to `127.0.0.1` by default:

```bash
./examples/burger_real/run_agent_ros2.sh
```

The RL side binds to every interface, so with a tunnel the firewall of the RL host should
block the port from outside. If the local port is busy, pick another one with
`./tunnel.sh user@rl-host 49054 50000` and start the agent with `PORT=50000`.

Publish a goal to start a mission:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}"
```

### Options

Arguments after the model go to `rl_side_burger_real.py`, for instance `--episodes 5` to
stop after five missions. Arguments after the host go to `agent_node`, for instance
`--sensor-timeout 2.0`.

| Variable | Default | Script | Meaning |
|---|---|---|---|
| `PORT` | `49054` | both | TCP port of the RL side |
| `DEVICE` | `cuda` | RL | PyTorch device: `cuda`, `cpu` or `auto` |
| `TIMEOUT` | `300` | RL | Seconds to wait for each agent reply |
| `USE_SIM_TIME` | `false` | agent | `true` when the topics come from a simulator's `/clock` |
| `PYTHON` | `python3` | RL | Interpreter with the RL dependencies |

### Timeouts

The RL `TIMEOUT` is 300 s because on every reset the agent blocks until an operator
publishes a new `/goal_pose`, and the RL side waits for that reply. It has to be longer
than the time you need between missions; with an automatic mission node you can lower it.

The agent's `--sensor-timeout` (1 s) is the longest silence from the sensors before it
truncates the attempt; raise it on slow or lossy links. `--rl-step-period` (0.08 s) is how
long each action is applied, and `--control-period` (0.01 s) is how often the sensors are
polled and the action is re-published. The RL side waits 60 seconds for the agent to
connect, a limit fixed in `spindecoupler`.

## Files

```
rl_side_burger_real.py        RL side: loads the policy and runs inference
reward.py                     reward, goal check and step budget (RL side)
ros2/burger_real_agent/       ROS2 package with agent_node.py and contract.py
run_rl_remote.sh              starts the RL side
run_agent_ros2.sh             starts the agent on the Burger
tunnel.sh                     SSH port forward, for when the port cannot be exposed
requirements-agent.txt        Python dependencies of the Burger host
requirements-rl.txt           Python dependencies of the RL host
training_8rays_fixedtime.zip  pre-trained SAC policy
```

## Notes

The transport uses `pickle`, so connect only endpoints you trust. The RL-side `t_wall` and
the agent-side `t_agent` come from different clocks; synchronize the hosts with NTP or
chrony before comparing them.
