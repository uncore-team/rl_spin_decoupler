# Real TurtleBot3 Burger: ROS2 policy exploitation

This example uses the same decoupled contract as `burger_coppeliasim`, but the
agent is a ROS2 node running on a real TurtleBot3 Burger. The remote RL process
loads an already-trained PPO policy and performs deterministic inference only.
It never calls `learn()`.

## Contract

The agent publishes actions as `geometry_msgs/msg/Twist` on `/cmd_vel`, and
subscribes to:

- `/scan` (`sensor_msgs/msg/LaserScan`)
- `/odom` (`nav_msgs/msg/Odometry`)
- `/goal_pose` (`geometry_msgs/msg/PoseStamped`, frame `map`)

The observation is the same 10-value vector as CoppeliaSim:
8 uniform minimum LiDAR sectors, distance to goal, and relative goal angle.
The policy action is `[linear_x, angular_z]`, clipped to the Burger limits.
The agent transforms the goal from `map` to `odom` using TF `map -> odom`.

If sensor data become stale, the agent publishes zero velocity and truncates
the episode. A collision detected by LiDAR terminates it. After an episode,
physically reposition the robot; the next RL `reset()` confirms the reset and
the node waits for fresh ROS messages before continuing.

## ROS2 Humble agent setup

On the Burger host, source ROS2 Humble and build the package:

```bash
source /opt/ros/humble/setup.bash
cd examples/burger_real/ros2
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
pip install -r ../requirements-agent.txt
```

The Burger must have active publishers for the topics above and a valid
`map -> odom` TF. Test the topic and TF setup before enabling `/cmd_vel`.

## Remote RL setup

On the remote host:

```bash
pip install -e .
pip install -r examples/burger_real/requirements-rl.txt
```

Copy the previously trained SB3 model to the remote host. Start the RL process
first:

```bash
./examples/burger_real/run_rl_remote.sh /path/to/ppo_coppelia_burger.zip
```

The model must have been trained with the same 10-value observation and
2-value continuous action spaces.

## Networking

For a shared LAN, allow TCP port `49054` on the remote host and run on the
Burger host:

```bash
./examples/burger_real/run_agent_ros2.sh <REMOTE_LAN_IP>
```

For an SSH tunnel, run the remote RL process bound to its normal server port,
then on the Burger host:

```bash
./examples/burger_real/tunnel.sh user@remote-host
./examples/burger_real/run_agent_ros2.sh
```

Set `PORT` to use another port. The transport uses `pickle`; connect only
trusted endpoints. Begin with low velocity limits and a physical emergency
stop during first trials.
