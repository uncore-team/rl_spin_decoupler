#!/usr/bin/env bash
set -e

REMOTE="${1:?usage: ./tunnel.sh user@remote-host [port]}"
PORT="${2:-49054}"
exec ssh -N -L "${PORT}:127.0.0.1:${PORT}" "$REMOTE"
