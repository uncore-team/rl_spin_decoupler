# SPDX-License-Identifier: GPL-3.0-only
"""Topic, frame and rate constants of the simulated Burger.

Topic names, frames and velocity limits have to match the real-robot contract
(``examples/burger_real/ros2/burger_real_agent/contract.py``), so the agent and
a policy run unchanged against the simulator and against the real robot.
"""

SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
GOAL_TOPIC = "/goal_pose"
CMD_VEL_TOPIC = "/cmd_vel"
CLOCK_TOPIC = "/clock"

# Frames: map -> odom -> base_link -> base_scan.
MAP_FRAME = "map"
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
SCAN_FRAME = "base_scan"

# Velocity limits applied to /cmd_vel (same as the real agent).
LINEAR_VELOCITY_LIMIT = 0.15  # m/s
ANGULAR_VELOCITY_LIMIT = 0.6  # rad/s

# LiDAR: eight sectors, maximum range in meters (laser.py `max_scan_distance`).
LIDAR_MAX_RANGE = 8.0

# Publication rates, in *simulation* time. The scene steps at dt = 0.05 s
# (20 Hz), the ceiling for fresh odometry; /scan matches a TurtleBot3 LDS-01.
SCAN_HZ = 5.0
ODOM_HZ = 20.0

# Upper bound for the bridge loop. The loop is normally paced by the scene
# (one blocking ZMQ call per tick), so this only matters on a fast machine.
CONTROL_HZ = 20.0
