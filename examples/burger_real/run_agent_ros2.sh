#!/usr/bin/env bash
# Start the ROS2 agent on the Burger host, after the RL side is listening. Source
# ROS2 Humble and the built workspace first (see README.md).
#
# Usage:   ./run_agent_ros2.sh [RL_HOST] [agent options...]
#   RL_HOST   IP of the RL host (LAN). Omit it, or use 127.0.0.1, when the RL side
#             is reached through an SSH tunnel (see tunnel.sh) or runs on this host.
# Options go to agent_node, e.g. --sensor-timeout 2.0 or --debug.
#
# Environment variables (defaults in brackets):
#   PORT          TCP port of the RL side                             [49054]
#   USE_SIM_TIME  true when the topics come from a simulator's /clock    [false]
set -euo pipefail

RL_HOST="127.0.0.1"
if [[ $# -gt 0 && "$1" != -* ]]; then RL_HOST="$1"; shift; fi
PORT="${PORT:-49054}"

exec ros2 run burger_real_agent agent_node \
  --ros-args -p "use_sim_time:=${USE_SIM_TIME:-false}" -- \
  --ip "$RL_HOST" --port "$PORT" "$@"
