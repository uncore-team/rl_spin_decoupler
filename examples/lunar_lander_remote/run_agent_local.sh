#!/usr/bin/env bash
# Start the agent (the LunarLander simulation) on the LOCAL host, after the RL side
# is listening.
#
# Usage:   ./run_agent_local.sh [RL_HOST] [agent options...]
#   RL_HOST   IP of the RL host (profile A). Omit it, or use 127.0.0.1, for profile B
#             (SSH tunnel, see tunnel.sh) or to try everything on one machine.
# Options go to agent_side_lunarlander.py, e.g. --render or --debug.
#
# Environment variables (defaults in brackets):
#   PORT     TCP port of the RL side                                  [49054]
#   TIMEOUT  seconds to wait for a pending RL command                    [10]
#   PYTHON   Python interpreter that has the agent dependencies      [python3]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root

RL_HOST="127.0.0.1"
if [[ $# -gt 0 && "$1" != -* ]]; then RL_HOST="$1"; shift; fi
PORT="${PORT:-49054}"
TIMEOUT="${TIMEOUT:-10}"

exec "${PYTHON:-python3}" examples/lunar_lander/agent_side_lunarlander.py \
  --ip "$RL_HOST" --port "$PORT" \
  --rl-step-period 0.08 --control-period 0.01 --timeout "$TIMEOUT" "$@"
