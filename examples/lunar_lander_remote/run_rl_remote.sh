#!/usr/bin/env bash
set -e
# Run the RL side on the REMOTE host without Docker.
#
# Usage:
#   ./run_rl_remote.sh
#
# Extra args are forwarded to the RL script, e.g.:
#   ./run_rl_remote.sh --timesteps 20000
#
# Env overrides: PORT (default 49054), DEVICE (default cuda).
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
shift || true

exec python examples/lunar_lander/rl_side_lunarlander.py \
    --port "$PORT" --device "$DEVICE" --timeout 10.0 "$@"
