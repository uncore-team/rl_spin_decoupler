#!/usr/bin/env bash
# Try this deployment on ONE machine: the "remote" GPU host and the "local" agent
# host are both this computer (profile A, loopback). It opens CoppeliaSim, then
# runs run_rl_container.sh and run_agent_local.sh exactly as you would on two
# hosts, and stops everything when training ends. Needs Docker; DEVICE=cpu does
# not need the NVIDIA runtime.
#
# Usage:   ./run_local_demo.sh
# Options are the environment variables of ../burger_coppeliasim/run_demo.sh
# (TIMESTEPS, DEVICE, RL_TIMEOUT, KEEP_SIM_OPEN, ...). The model is saved in
# ./output/ unless MODEL_OUT says otherwise.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_LAUNCHER="$DIR/run_rl_container.sh A"
export AGENT_LAUNCHER="$DIR/run_agent_local.sh 127.0.0.1"
export MODEL_OUT="${MODEL_OUT:-$DIR/output/sac_coppelia_burger.zip}"
exec "$DIR/../burger_coppeliasim/run_demo.sh" "$@"
