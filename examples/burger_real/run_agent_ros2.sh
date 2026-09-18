#!/usr/bin/env bash
set -e

# Run on the Burger host after sourcing ROS2 Humble and the built workspace.
# Profile A: ./run_agent_ros2.sh <REMOTE_LAN_IP>
# Profile B: ./run_agent_ros2.sh 127.0.0.1
RL_IP="${1:-127.0.0.1}"
PORT="${PORT:-49054}"
shift || true

exec ros2 run burger_real_agent agent_node \
  --ros-args -p use_sim_time:=false -- \
  --ip "$RL_IP" --port "$PORT" "$@"
