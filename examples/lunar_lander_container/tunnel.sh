#!/usr/bin/env bash
set -e
# Profile B only: forward a local port to the remote host's loopback, where the
# container publishes the RL server via `-p 127.0.0.1:PORT:PORT`.
#
# Usage:
#   ./tunnel.sh user@remote-host [PORT]
# Then run the agent against 127.0.0.1 (the default of run_agent_local.sh).
REMOTE="${1:?usage: ./tunnel.sh user@remote-host [port]}"
PORT="${2:-49054}"
exec ssh -N -L "${PORT}:127.0.0.1:${PORT}" "$REMOTE"
