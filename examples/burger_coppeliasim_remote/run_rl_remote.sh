#!/usr/bin/env bash
# Start the RL side of the CoppeliaSim Burger example on the REMOTE host (no container).
#
# Usage:   ./run_rl_remote.sh [A|B] [rl-side options...]
#   A   the agent connects over the LAN to this host's IP (default)
#   B   the agent connects through an SSH tunnel; block the port in the firewall
# The profile only changes the message printed: the server always listens on all
# interfaces. Remaining arguments go to rl_side_burger_coppeliasim.py, e.g.
# --timesteps 20000.
#
# Environment variables (defaults in brackets):
#   PORT     TCP port of the RL side                                  [49054]
#   DEVICE   PyTorch device: cuda | cpu | auto                         [cuda]
#   TIMEOUT  seconds to wait for each agent reply. The slowest is the first reset
#            (the agent starts the simulation); raise it on slow links     [30]
#   PYTHON   Python interpreter that has the RL-side dependencies   [python3]
#
# The RL side waits 60 s for the agent after it starts listening.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root

PROFILE="A"
case "${1:-}" in A|B) PROFILE="$1"; shift ;; esac
PORT="${PORT:-49054}"
DEVICE="${DEVICE:-cuda}"
TIMEOUT="${TIMEOUT:-30}"

case "$PROFILE" in
  A) echo "[profile A] the agent connects to this host's LAN IP, port $PORT" ;;
  B) echo "[profile B] the agent connects through an SSH tunnel to port $PORT; keep it closed in the firewall" ;;
esac

exec "${PYTHON:-python3}" examples/burger_coppeliasim/rl_side_burger_coppeliasim.py \
  --port "$PORT" --device "$DEVICE" --timeout "$TIMEOUT" "$@"
