#!/usr/bin/env bash
# Try this deployment on ONE machine: the "remote" RL host and the "local" agent
# host are both this computer, talking over the loopback interface (profile A).
# It opens CoppeliaSim, then runs run_rl_remote.sh and run_agent_local.sh exactly
# as you would on two hosts, and shuts everything down when training ends.
#
# Usage:   ./run_local_demo.sh
# Options are the environment variables of ../burger_coppeliasim/run_demo.sh
# (TIMESTEPS, DEVICE, RL_TIMEOUT, KEEP_SIM_OPEN, ...), with DEVICE=cpu by default.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_LAUNCHER="$DIR/run_rl_remote.sh A"
export AGENT_LAUNCHER="$DIR/run_agent_local.sh 127.0.0.1"
exec "$DIR/../burger_coppeliasim/run_demo.sh" "$@"
