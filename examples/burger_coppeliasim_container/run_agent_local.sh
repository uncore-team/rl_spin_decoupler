#!/usr/bin/env bash
set -e
# Run the CoppeliaSim agent on the LOCAL host, after the remote RL container listens.
#
# Usage:
#   Profile A (same LAN):   ./run_agent_local.sh <REMOTE_LAN_IP>
#   Profile B (SSH tunnel): ./run_agent_local.sh
#
# Env overrides: PORT (default 49054), SIM_HOST (default 127.0.0.1),
# SIM_PORT (default 23000).
AGENT_IP="${1:-127.0.0.1}"
PORT="${PORT:-49054}"
SIM_HOST="${SIM_HOST:-127.0.0.1}"
SIM_PORT="${SIM_PORT:-23000}"
shift || true

exec python examples/burger_coppeliasim/agent_side_burger_coppeliasim.py \
    --ip "$AGENT_IP" --port "$PORT" \
    --sim-host "$SIM_HOST" --sim-port "$SIM_PORT" \
    --rl-step-period 0.08 --control-period 0.01 --timeout 10.0 "$@"
