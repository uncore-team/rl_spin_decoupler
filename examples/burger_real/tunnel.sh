#!/usr/bin/env bash
# Forward a local port to the RL side's port on the RL host through SSH, for when
# that port cannot be exposed. Leave it running, then start run_agent_ros2.sh (it
# connects to 127.0.0.1 by default). Run it on the Burger host.
#
# Usage:   ./tunnel.sh user@rl-host [REMOTE_PORT] [LOCAL_PORT]
#   REMOTE_PORT   port of the RL side on the RL host (default $PORT, else 49054)
#   LOCAL_PORT    local port to listen on (default REMOTE_PORT). If it differs,
#                 start the agent with PORT=<LOCAL_PORT>.
set -euo pipefail
REMOTE="${1:?usage: ./tunnel.sh user@rl-host [remote_port] [local_port]}"
REMOTE_PORT="${2:-${PORT:-49054}}"
LOCAL_PORT="${3:-$REMOTE_PORT}"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$REMOTE"
