#!/usr/bin/env bash
# Start the agent on the LOCAL host, next to CoppeliaSim, after the RL container is
# listening. Identical to the bare-host deployment, so this simply runs
# ../burger_coppeliasim_remote/run_agent_local.sh; see that script for the options.
#
# Usage:   ./run_agent_local.sh [RL_HOST] [agent options...]
exec "$(dirname "${BASH_SOURCE[0]}")/../burger_coppeliasim_remote/run_agent_local.sh" "$@"
