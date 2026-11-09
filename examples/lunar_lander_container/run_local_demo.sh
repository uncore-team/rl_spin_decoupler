#!/usr/bin/env bash
# Try this deployment on ONE machine: the "remote" GPU host and the "local" agent
# host are both this computer (profile A, loopback). It runs run_rl_container.sh
# and run_agent_local.sh exactly as you would on two hosts, and stops both when
# training ends. Needs Docker; DEVICE=cpu does not need the NVIDIA runtime.
#
# Usage:   ./run_local_demo.sh
# Options are the environment variables of ../lunar_lander/run_demo.sh
# (TIMESTEPS, N_STEPS, DEVICE, RENDER, RL_TIMEOUT, ...). The model is saved in
# ./output/ unless MODEL_OUT says otherwise.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_LAUNCHER="$DIR/run_rl_container.sh A"
export AGENT_LAUNCHER="$DIR/run_agent_local.sh 127.0.0.1"
export MODEL_OUT="${MODEL_OUT:-$DIR/output/ppo_lunar_lander_decoupled.zip}"
exec "$DIR/../lunar_lander/run_demo.sh" "$@"
