#!/usr/bin/env bash
# One-command demo: PPO training on LunarLander with the simulator in a separate process.
#
# Starts, in dependency order, and tears everything down on exit:
#
#   1. RL side     opens the socket server and trains PPO (reward is computed here)
#   2. Agent side  runs Gymnasium LunarLander-v3 and serves the RL side
#
# The demo trains for TIMESTEPS steps, optionally prints the mean decoupling
# latency over a short rollout, and stops by itself. See README.md.
#
# Usage:   ./examples/lunar_lander/run_demo.sh
# Options (environment variables, defaults in brackets):
#   TIMESTEPS       training steps                                            [1024]
#   N_STEPS         PPO rollout length (steps between policy updates)         [256]
#   ROLLOUT_STEPS   steps of a final rollout that reports the mean latency   [100]
#   DEVICE          PyTorch device: cpu | cuda | auto                         [cpu]
#   RENDER          1 = open the LunarLander window (needs a display)          [0]
#   MODEL_OUT       where to save the model [examples/lunar_lander/ppo_lunar_lander_decoupled.zip]
#   LOG_DIR         where per-component logs are written            [a new /tmp dir]
# Timeouts (see "Timeouts" in README.md):
#   RL_TIMEOUT      seconds the RL side waits for each agent reply                [10]
#   AGENT_TIMEOUT   seconds the agent waits for a pending RL command               [10]
#   STARTUP_TIMEOUT seconds to wait for the RL side to be listening              [180]
#   RL_PORT         TCP port of the RL side                                   [49054]
# Deployment hooks, used by the lunar_lander_remote and lunar_lander_container
# examples to try a split deployment on one machine (they receive the training
# options as trailing arguments, and PORT, DEVICE and TIMEOUT as variables):
#   RL_LAUNCHER     command that starts the RL side instead of rl_side_lunarlander.py
#   AGENT_LAUNCHER  command that starts the agent instead of agent_side_lunarlander.py

set -euo pipefail

TIMESTEPS="${TIMESTEPS:-1024}"
N_STEPS="${N_STEPS:-256}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-100}"
DEVICE="${DEVICE:-cpu}"
RENDER="${RENDER:-0}"
RL_TIMEOUT="${RL_TIMEOUT:-10}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-10}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
RL_PORT="${RL_PORT:-49054}"
RL_LAUNCHER="${RL_LAUNCHER:-}"
AGENT_LAUNCHER="${AGENT_LAUNCHER:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="$REPO_ROOT/examples/lunar_lander"
MODEL_OUT="${MODEL_OUT:-$EXAMPLE_DIR/ppo_lunar_lander_decoupled.zip}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/lunar_lander.XXXXXX)}"
mkdir -p "$LOG_DIR"

say() { echo "[run] $*"; }
die() { echo "[run] ERROR: $*" >&2; exit 1; }

# Pre-flight checks: fail early with an actionable message.
# True if something listens on TCP port $1. It only inspects the socket table:
# actually connecting to the RL port would be taken by the RL side as "the agent".
port_open() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
command -v ss >/dev/null || die "'ss' not found (sudo apt install iproute2)"
port_open "$RL_PORT" && die "port $RL_PORT is already in use (another demo running? set RL_PORT)"

# Python: the repository virtualenv if present, else the system one.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$REPO_ROOT/.venv/bin/python"; else PYTHON_BIN="python3"; fi
export PYTHONUNBUFFERED=1   # show each component's output immediately
export PYTHON="$PYTHON_BIN" # inherited by the deployment launchers

if [[ -z "$RL_LAUNCHER" && -z "$AGENT_LAUNCHER" ]]; then
  "$PYTHON_BIN" -c "import stable_baselines3, gymnasium, Box2D, spindecoupler" 2>/dev/null \
    || die "Python packages missing; see 'Installation' in examples/lunar_lander/README.md"
fi
if [[ "$RENDER" == "1" && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  die "RENDER=1 needs a display (set DISPLAY), or leave RENDER unset"
fi

# Process management.
PIDS=()   # components started by this script

# Run a component in the background; its output is logged to $LOG_DIR/<name>.log
# and echoed to the terminal prefixed with [<name>].
run_logged() {
  local name="$1"; shift
  "$@" > >(tee "$LOG_DIR/$name.log" | sed -u "s/^/[$name] /") 2>&1 &
  PIDS+=("$!")
  LAST_PID="$!"
}

# Print a process and all of its descendants. Launchers and containers spawn the
# real program as a child, so signalling only the stored PID would orphan it. The
# list is collected BEFORE signalling, while the tree is intact.
tree_pids() {
  local pid="$1" child
  echo "$pid"
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do tree_pids "$child"; done
}

# Wait until every given PID has exited, up to $1 seconds.
wait_gone() {
  local deadline=$((SECONDS + $1)) pid alive; shift
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "$@"; do [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1; done
    (( alive )) || return 0
    sleep 0.3
  done
  return 1
}

cleanup() {
  trap - EXIT INT TERM
  set +e  # a vanished PID must not abort the teardown half-way
  say "Shutting down (logs: $LOG_DIR)"
  local pid victims=()
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && victims+=($(tree_pids "$pid")); done
  for pid in "${victims[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null; done
  wait_gone 8 "${victims[@]:-}" || for pid in "${victims[@]}"; do kill -KILL "$pid" 2>/dev/null; done
  true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# 1. RL side: opens the TCP server and trains. It accepts the agent for 60 s
#    (fixed in spindecoupler), so the agent is started right after.
TRAIN_ARGS=(--timesteps "$TIMESTEPS" --n-steps "$N_STEPS" --rollout-steps "$ROLLOUT_STEPS"
            --model-path "$MODEL_OUT")
export PORT="$RL_PORT" DEVICE TIMEOUT="$RL_TIMEOUT"   # read by the deployment launchers
say "Starting the RL side (PPO, $TIMESTEPS steps, rollout $N_STEPS) ..."
if [[ -n "$RL_LAUNCHER" ]]; then
  read -ra RL_CMD <<< "$RL_LAUNCHER"
  run_logged rl "${RL_CMD[@]}" "${TRAIN_ARGS[@]}"
else
  run_logged rl "$PYTHON_BIN" "$EXAMPLE_DIR/rl_side_lunarlander.py" \
    --port "$RL_PORT" --device "$DEVICE" --timeout "$RL_TIMEOUT" "${TRAIN_ARGS[@]}"
fi
RL_PID="$LAST_PID"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until port_open "$RL_PORT"; do
  kill -0 "$RL_PID" 2>/dev/null || die "the RL side exited during startup (see $LOG_DIR/rl.log)"
  (( SECONDS < deadline )) || die "timed out waiting for the RL side to listen on port $RL_PORT"
  sleep 0.5
done

# 2. Agent side: runs LunarLander and serves the RL side.
AGENT_EXTRA=()
[[ "$RENDER" == "1" ]] && AGENT_EXTRA+=(--render)
say "Starting the agent ..."
export TIMEOUT="$AGENT_TIMEOUT"   # read by the agent launchers
if [[ -n "$AGENT_LAUNCHER" ]]; then
  read -ra AGENT_CMD <<< "$AGENT_LAUNCHER"
  run_logged agent "${AGENT_CMD[@]}" "${AGENT_EXTRA[@]}"
else
  run_logged agent "$PYTHON_BIN" "$EXAMPLE_DIR/agent_side_lunarlander.py" \
    --ip 127.0.0.1 --port "$RL_PORT" --timeout "$AGENT_TIMEOUT" "${AGENT_EXTRA[@]}"
fi
AGENT_PID="$LAST_PID"

# Run until training (and the optional rollout) finishes: the RL side then sends
# FINISH to the agent. Also stop if either component dies. Ctrl+C stops all.
say "Training. Ctrl+C to stop."
while kill -0 "$RL_PID" 2>/dev/null && kill -0 "$AGENT_PID" 2>/dev/null; do
  sleep 1
done

status=0
# Normal end: the RL side finishes and exits, and the agent exits right after
# receiving FINISH. Give the RL process a moment to exit on its own before
# treating the loop exit as a failure.
if wait_gone 10 "$RL_PID"; then
  wait "$RL_PID" || status=$?
  wait_gone 20 "$AGENT_PID" || true
else
  say "The agent stopped unexpectedly (log: $LOG_DIR/agent.log)"
  status=1
fi

if (( status == 0 )); then
  say "Demo finished. Model saved to: $MODEL_OUT"
else
  say "Demo failed (exit status $status)."
fi
exit "$status"
