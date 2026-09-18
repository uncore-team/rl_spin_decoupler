#!/usr/bin/env bash
set -e
# Run the Burger RL side on the REMOTE host without Docker.
#
# Usage:
#   ./run_rl_remote.sh [A|B] [rl-side options...]
#
# Profile A is the direct LAN mode. Profile B is intended for tunnel.sh; both
# modes use the same server bind behavior in the RL-side Python process.
# Env overrides: PORT (default 49054), DEVICE (default cuda).
PROFILE="${1:-A}"
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
shift || true

case "$PROFILE" in
    A|B)
        exec python examples/burger_coppeliasim/rl_side_burger_coppeliasim.py \
            --port "$PORT" --device "$DEVICE" --timeout 10.0 "$@"
        ;;
    *)
        echo "usage: $0 [A|B] [rl-side options...]" >&2
        exit 2
        ;;
esac
