#!/usr/bin/env bash
# One-command demo: the RL side and the agent of the first-order plant example.
#
# Usage:   ./examples/first_order_plant_control/run_demo.sh
# Options (environment variables, defaults in brackets):
#   STEPS     maximum number of actions        [20]
#   RL_PORT   TCP port of the RL side          [49054]

set -euo pipefail

STEPS="${STEPS:-20}"
RL_PORT="${RL_PORT:-49054}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="$REPO_ROOT/examples/first_order_plant_control"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$REPO_ROOT/.venv/bin/python"; else PYTHON_BIN="python3"; fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$EXAMPLE_DIR/rl_side_fopcontrol.py" --port "$RL_PORT" --steps "$STEPS" &
RL_PID=$!
trap 'kill "$RL_PID" 2>/dev/null || true' EXIT

# Connecting before the server listens is refused, so wait for the port first.
# `ss` only inspects the socket table; probing the port would look like the agent.
for _ in $(seq 100); do
  [[ -n "$(ss -H -ltn "sport = :$RL_PORT" 2>/dev/null)" ]] && break
  kill -0 "$RL_PID" 2>/dev/null || { echo "the RL side exited during startup" >&2; exit 1; }
  sleep 0.2
done
"$PYTHON_BIN" "$EXAMPLE_DIR/agent_side_fopcontrol.py" --ip 127.0.0.1 --port "$RL_PORT"
wait "$RL_PID"
