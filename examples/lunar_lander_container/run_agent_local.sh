#!/usr/bin/env bash
# Start the agent (the LunarLander simulation) on the LOCAL host, after the RL
# container is listening. Identical to the bare-host deployment, so this simply
# runs ../lunar_lander_remote/run_agent_local.sh; see that script for the options.
#
# Usage:   ./run_agent_local.sh [RL_HOST] [agent options...]
exec "$(dirname "${BASH_SOURCE[0]}")/../lunar_lander_remote/run_agent_local.sh" "$@"
