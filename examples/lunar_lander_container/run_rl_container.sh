#!/usr/bin/env bash
set -e
# Build and run the RL-side container on the REMOTE (GPU) host.
#
# Usage:
#   ./run_rl_container.sh              # Profile A (same LAN) -- default
#   ./run_rl_container.sh B            # Profile B (SSH tunnel)
# Extra args are forwarded to the RL script, e.g.:
#   ./run_rl_container.sh A --timesteps 20000
#
# Env overrides: PORT (default 49054), DEVICE (default cuda).
PROFILE="${1:-A}"
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
shift || true

# Build context is the REPO ROOT so the image can COPY spindecoupler + example.
# Run this script from the repository root.
docker build -f examples/lunar_lander_container/Dockerfile -t lunarlander-rl .

COMMON=(--rm --gpus all)
RLARGS=(--port "$PORT" --device "$DEVICE" --timeout 10.0 "$@")

if [ "$PROFILE" = "A" ]; then
    echo "[profile A] --network host; the agent connects to this host's LAN IP"
    exec docker run "${COMMON[@]}" --network host lunarlander-rl "${RLARGS[@]}"
else
    echo "[profile B] -p 127.0.0.1:${PORT}:${PORT}; connect the agent through an SSH tunnel"
    exec docker run "${COMMON[@]}" -p 127.0.0.1:"$PORT":"$PORT" lunarlander-rl "${RLARGS[@]}"
fi
