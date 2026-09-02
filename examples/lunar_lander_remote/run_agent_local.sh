#!/usr/bin/env bash
set -e
# Run the agent (client) on the LOCAL host, after the remote RL side listens.
#
# Usage:
#   Profile A (same LAN):   ./run_agent_local.sh <REMOTE_LAN_IP>
#   Profile B (SSH tunnel): ./run_agent_local.sh     # defaults to 127.0.0.1
#
# Extra args are forwarded to the agent script, e.g.:
#   ./run_agent_local.sh --debug
#
# Env overrides: PORT (default 49054).
AGENT_IP="${1:-127.0.0.1}"
PORT="${PORT:-49054}"
shift || true

exec python examples/lunar_lander/agent_side_lunarlander.py \
    --ip "$AGENT_IP" --port "$PORT" \
    --rl-step-period 0.08 --control-period 0.01 --timeout 10.0 --render "$@"
    