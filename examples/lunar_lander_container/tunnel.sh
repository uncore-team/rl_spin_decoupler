#!/usr/bin/env bash
# Profile B only: forward a local port to the RL container's port on the remote
# host (published on its loopback). Identical to the bare-host deployment, so this
# simply runs ../lunar_lander_remote/tunnel.sh.
#
# Usage:   ./tunnel.sh user@remote-host [PORT]
exec "$(dirname "${BASH_SOURCE[0]}")/../lunar_lander_remote/tunnel.sh" "$@"
