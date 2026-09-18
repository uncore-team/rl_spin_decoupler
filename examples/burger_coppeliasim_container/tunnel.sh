#!/usr/bin/env bash
set -e
# Profile B only: forward a local port to the remote host loopback, where the
# container publishes the RL server via -p 127.0.0.1:PORT:PORT.
#
# Usage:
#   ./tunnel.sh user@remote-host [PORT]
REMOTE="${1:?usage: ./tunnel.sh user@remote-host [port]}"
PORT="${2:-49054}"
exec ssh -v -N -L "${PORT}:127.0.0.1:${PORT}" "$REMOTE"
