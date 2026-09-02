#!/usr/bin/env bash
set -e
# Run the RL-side container on the REMOTE (GPU) host. If the image does not exist, it will be built first.
#
# Usage:
#   Profile A (same LAN):       ./run_rl_container.sh A (default)
#   Profile B (SSH tunnel):     ./run_rl_container.sh B
#
# Extra args are forwarded to the RL script, e.g.:
#   ./run_rl_container.sh A --timesteps 20000
#
# Env overrides: PORT (default 49054), DEVICE (default cuda).
PROFILE="${1:-A}"
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
shift || true

# Check that the image exists, otherwise build it.
if ! docker image inspect lunarlander-rl >/dev/null 2>&1; then
    echo "Image 'lunarlander-rl' not found, building it now..."

    # Call build_image.sh to build the image
    bash examples/lunar_lander_container/build_image.sh
fi

COMMON=(--rm --gpus all)
RLARGS=(--port "$PORT" --device "$DEVICE" --timeout 10.0 "$@")

case "$PROFILE" in
    A)
        echo "[profile A] --network host; the agent connects to this host's LAN IP"
        exec docker run "${COMMON[@]}" --network host lunarlander-rl "${RLARGS[@]}"
        ;;
    B)
        echo "[profile B] -p 127.0.0.1:${PORT}:${PORT}; connect the agent through an SSH tunnel"
        exec docker run "${COMMON[@]}" -p 127.0.0.1:"$PORT":"$PORT" lunarlander-rl "${RLARGS[@]}"
        ;;
    *)
        echo "usage: $0 [A|B] [rl-side options...]" >&2
        exit 2
        ;;
esac
