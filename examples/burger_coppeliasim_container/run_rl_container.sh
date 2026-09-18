#!/usr/bin/env bash
set -e
# Run the Burger RL-side container on the REMOTE (GPU) host.
#
# Usage:
#   Profile A (same LAN):       ./run_rl_container.sh A
#   Profile B (SSH tunnel):     ./run_rl_container.sh B
#
# Extra args are forwarded to the RL script, for example --timesteps 20000.
# Env overrides: PORT (default 49054), DEVICE (default cuda).
PROFILE="${1:-A}"
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
shift || true

if ! docker image inspect burger-coppelia-rl >/dev/null 2>&1; then
    echo "Image 'burger-coppelia-rl' not found, building it now..."
    bash examples/burger_coppeliasim_container/build_image.sh
fi

COMMON=(--rm --gpus all)
RLARGS=(--port "$PORT" --device "$DEVICE" --timeout 10.0 "$@")

case "$PROFILE" in
    A)
        echo "[profile A] --network host; the agent connects to this host's LAN IP"
        exec docker run "${COMMON[@]}" --network host burger-coppelia-rl "${RLARGS[@]}"
        ;;
    B)
        echo "[profile B] -p 127.0.0.1:${PORT}:${PORT}; connect through an SSH tunnel"
        exec docker run "${COMMON[@]}" -p 127.0.0.1:"$PORT":"$PORT" burger-coppelia-rl "${RLARGS[@]}"
        ;;
    *)
        echo "usage: $0 [A|B] [rl-side options...]" >&2
        exit 2
        ;;
esac
