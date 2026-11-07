#!/usr/bin/env bash
# Start the RL side of the LunarLander example in a Docker container, on the
# REMOTE host. The image is built first if it does not exist.
#
# Usage:   ./run_rl_container.sh [A|B] [rl-side options...]
#   A   host networking: the agent connects to this host's LAN IP (default)
#   B   the port is published on the loopback only: the agent connects through
#       an SSH tunnel
# Remaining arguments go to rl_side_lunarlander.py, e.g. --timesteps 20000.
#
# The trained model is saved on the host: --model-path takes a HOST path (default
# examples/lunar_lander_container/output/ppo_lunar_lander_decoupled.zip), whose
# directory is mounted into the container. Files are owned by root.
#
# Environment variables (defaults in brackets):
#   PORT     TCP port of the RL side                                  [49054]
#   DEVICE   cuda (needs the NVIDIA runtime, passes --gpus all), or cpu  [cuda]
#   TIMEOUT  seconds to wait for each agent reply; raise it on slow links [10]
#   IMAGE    image name                                      [lunarlander-rl]
#
# The RL side waits 60 s for the agent after it starts listening.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root

PROFILE="A"
case "${1:-}" in A|B) PROFILE="$1"; shift ;; esac
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
TIMEOUT="${TIMEOUT:-10}"
IMAGE="${IMAGE:-lunarlander-rl}"

# Split --model-path (a host path) from the arguments forwarded to the RL side.
MODEL="$PWD/examples/lunar_lander_container/output/ppo_lunar_lander_decoupled.zip"
ARGS=()
while (( $# )); do
  case "$1" in
    --model-path)   MODEL="$2"; shift 2 ;;
    --model-path=*) MODEL="${1#*=}"; shift ;;
    *)              ARGS+=("$1"); shift ;;
  esac
done
mkdir -p "$(dirname "$MODEL")"
MODEL_DIR="$(cd "$(dirname "$MODEL")" && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image '$IMAGE' not found; building it (this takes a while)..."
  bash examples/lunar_lander_container/build_image.sh
fi

DOCKER_ARGS=(--rm -v "$MODEL_DIR:/output")
[[ "$DEVICE" == cpu ]] || DOCKER_ARGS+=(--gpus all)
case "$PROFILE" in
  A)
    echo "[profile A] host networking: the agent connects to this host's LAN IP, port $PORT"
    DOCKER_ARGS+=(--network host) ;;
  B)
    echo "[profile B] port $PORT published on 127.0.0.1 only: connect the agent through an SSH tunnel"
    DOCKER_ARGS+=(-p "127.0.0.1:$PORT:$PORT") ;;
esac

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
  --port "$PORT" --device "$DEVICE" --timeout "$TIMEOUT" \
  --model-path "/output/$(basename "$MODEL")" "${ARGS[@]}"
