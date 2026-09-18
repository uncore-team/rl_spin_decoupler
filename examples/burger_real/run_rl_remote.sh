#!/usr/bin/env bash
set -e

# Run policy exploitation on the remote RL host.
# Usage: ./run_rl_remote.sh --model-path /path/to/policy.zip
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"

exec python examples/burger_real/rl_side_burger_real.py \
  --port "$PORT" --device "$DEVICE" --model-path "$1" "${@:2}"
