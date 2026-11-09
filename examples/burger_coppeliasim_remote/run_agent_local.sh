#!/usr/bin/env bash
# Start the agent on the LOCAL host, next to CoppeliaSim, after the RL side is
# listening. The agent loads the arena scene into the running CoppeliaSim.
#
# Usage:   ./run_agent_local.sh [RL_HOST] [agent options...]
#   RL_HOST   IP of the RL host (profile A). Omit it, or use 127.0.0.1, for profile B
#             (SSH tunnel, see tunnel.sh) or to try everything on one machine.
# Options go to agent_side_burger_coppeliasim.py, e.g. --debug.
#
# Start CoppeliaSim first (its GUI is fine); the ZMQ remote API must be enabled.
#
# Environment variables (defaults in brackets):
#   PORT      TCP port of the RL side                                 [49054]
#   SIM_HOST  CoppeliaSim ZMQ remote API host                     [127.0.0.1]
#   SIM_PORT  CoppeliaSim ZMQ remote API port                          [23000]
#   TIMEOUT   seconds to wait for a pending RL command                    [10]
#   PYTHON    Python interpreter that has the agent dependencies      [python3]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root

RL_HOST="127.0.0.1"
if [[ $# -gt 0 && "$1" != -* ]]; then RL_HOST="$1"; shift; fi
PORT="${PORT:-49054}"
SIM_HOST="${SIM_HOST:-127.0.0.1}"
SIM_PORT="${SIM_PORT:-23000}"
TIMEOUT="${TIMEOUT:-10}"

exec "${PYTHON:-python3}" examples/burger_coppeliasim/agent_side_burger_coppeliasim.py \
  --ip "$RL_HOST" --port "$PORT" --sim-host "$SIM_HOST" --sim-port "$SIM_PORT" \
  --rl-step-period 0.08 --control-period 0.01 --timeout "$TIMEOUT" "$@"
