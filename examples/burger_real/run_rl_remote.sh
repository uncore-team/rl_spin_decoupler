#!/usr/bin/env bash
# Start the RL side of the real-Burger example on the RL host: it loads a trained
# SAC policy and runs inference only (no learning).
#
# Usage:   ./run_rl_remote.sh [MODEL.zip] [rl-side options...]
#   MODEL.zip   trained SB3 SAC model (default: training_8rays_fixedtime.zip, which
#               is included in this folder)
# Remaining arguments go to rl_side_burger_real.py, e.g. --episodes 5.
#
# Environment variables (defaults in brackets):
#   PORT     TCP port the agent connects to                           [49054]
#   DEVICE   PyTorch device: cuda | cpu | auto                         [cuda]
#   TIMEOUT  seconds to wait for each agent reply. On every reset the agent waits
#            until an operator publishes a NEW /goal_pose, so this must exceed
#            the time the operator needs between missions                [300]
#   PYTHON   Python interpreter that has the RL-side dependencies   [python3]
#
# The RL side waits 60 s for the agent after it starts listening.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root

MODEL="examples/burger_real/training_8rays_fixedtime.zip"
if [[ $# -gt 0 && "$1" != -* ]]; then MODEL="$1"; shift; fi
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
TIMEOUT="${TIMEOUT:-300}"

exec "${PYTHON:-python3}" examples/burger_real/rl_side_burger_real.py \
  --port "$PORT" --device "$DEVICE" --timeout "$TIMEOUT" --model-path "$MODEL" "$@"
