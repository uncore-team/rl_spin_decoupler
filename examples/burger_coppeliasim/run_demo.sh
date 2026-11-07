#!/usr/bin/env bash
# One-command demo: train a SAC policy to drive a simulated TurtleBot3 Burger.
#
# Starts, in dependency order, and tears everything down on exit:
#
#   1. CoppeliaSim (GUI)   the physics simulator (skipped if already running)
#   2. RL side             trains the SAC policy and saves the model
#   3. Agent side          loads the arena scene, runs the simulation for the RL side
#
# The demo trains for TIMESTEPS steps and stops by itself. See README.md.
#
# Usage:   ./examples/burger_coppeliasim/run_demo.sh
# Options (environment variables, defaults in brackets):
#   TIMESTEPS       training steps; the default only demonstrates the loop       [2000]
#   MODEL_OUT       where to save the trained model        [examples/burger_coppeliasim/sac_coppelia_burger.zip]
#   DEVICE          PyTorch device for training: cpu | cuda | auto               [cpu]
#   COPPELIASIM_SH  CoppeliaSim launcher                [/opt/CoppeliaSim_Edu/coppeliaSim.sh]
#   COPPELIASIM_ARGS extra CoppeliaSim arguments, e.g. "-h" for headless          []
#   KEEP_SIM_OPEN   1 = leave CoppeliaSim open after the demo (if we started it)   [0]
#   LOG_DIR         where per-component logs are written            [a new /tmp dir]
# Timeouts (see "Timeouts" in README.md):
#   RL_TIMEOUT      seconds the RL side waits for each agent reply                [30]
#   STARTUP_TIMEOUT seconds to wait for CoppeliaSim / the RL side to be up       [180]
#   SIM_PORT, RL_PORT  CoppeliaSim ZMQ API port / RL-agent TCP port      [23000, 49054]
# Deployment hooks, used by the burger_coppeliasim_remote and
# burger_coppeliasim_container examples to try a split deployment on one machine
# (they receive --timesteps/--model-path as trailing arguments, and PORT, DEVICE,
# TIMEOUT, SIM_HOST, SIM_PORT as variables):
#   RL_LAUNCHER     command that starts the RL side instead of rl_side_burger_coppeliasim.py
#   AGENT_LAUNCHER  command that starts the agent instead of agent_side_burger_coppeliasim.py

set -euo pipefail

TIMESTEPS="${TIMESTEPS:-2000}"
DEVICE="${DEVICE:-cpu}"
COPPELIASIM_SH="${COPPELIASIM_SH:-/opt/CoppeliaSim_Edu/coppeliaSim.sh}"
COPPELIASIM_ARGS="${COPPELIASIM_ARGS:-}"
KEEP_SIM_OPEN="${KEEP_SIM_OPEN:-0}"
SIM_PORT="${SIM_PORT:-23000}"
RL_PORT="${RL_PORT:-49054}"
RL_TIMEOUT="${RL_TIMEOUT:-30}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
RL_LAUNCHER="${RL_LAUNCHER:-}"
AGENT_LAUNCHER="${AGENT_LAUNCHER:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="$REPO_ROOT/examples/burger_coppeliasim"
MODEL_OUT="${MODEL_OUT:-$EXAMPLE_DIR/sac_coppelia_burger.zip}"
SCENE_PATH="$REPO_ROOT/examples/_common/burger_scene/scene.ttt"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/burger_coppeliasim.XXXXXX)}"
mkdir -p "$LOG_DIR"

say() { echo "[run] $*"; }
die() { echo "[run] ERROR: $*" >&2; exit 1; }

# Pre-flight checks: fail early with an actionable message.
# True if something listens on TCP port $1. It only inspects the socket table:
# actually connecting to the RL port would be taken by the RL side as "the agent".
port_open() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
command -v ss >/dev/null || die "'ss' not found (sudo apt install iproute2)"

[[ -f "$SCENE_PATH" ]] || die "scene not found: $SCENE_PATH"
port_open "$RL_PORT" && die "port $RL_PORT is already in use (another demo running? set RL_PORT)"

# Python: the repository virtualenv if present, else the system one.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$REPO_ROOT/.venv/bin/python"; else PYTHON_BIN="python3"; fi
export PYTHONUNBUFFERED=1   # show each component's output immediately

export PYTHON="$PYTHON_BIN"   # inherited by the deployment launchers
if [[ -z "$RL_LAUNCHER" && -z "$AGENT_LAUNCHER" ]]; then
  "$PYTHON_BIN" -c "import stable_baselines3, spindecoupler, coppeliasim_zmqremoteapi_client" 2>/dev/null \
    || die "Python packages missing; see 'Installation' in examples/burger_coppeliasim/README.md"
fi

# Process management.
PIDS=()      # components started by this script (RL side, agent side)
SIM_PID=""   # CoppeliaSim, only if this script started it

# Run a component in the background; its output is logged to $LOG_DIR/<name>.log
# and echoed to the terminal prefixed with [<name>].
run_logged() {
  local name="$1"; shift
  "$@" > >(tee "$LOG_DIR/$name.log" | sed -u "s/^/[$name] /") 2>&1 &
  PIDS+=("$!")
  LAST_PID="$!"
}

# Print a process and all of its descendants. coppeliaSim.sh spawns the real
# program as a child, so signalling only the stored PID would orphan it. The
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
  # 1. Our components. SIGTERM makes the agent stop the simulation cleanly.
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && victims+=($(tree_pids "$pid")); done
  for pid in "${victims[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null; done
  wait_gone 8 "${victims[@]:-}" || for pid in "${victims[@]}"; do kill -KILL "$pid" 2>/dev/null; done
  # 2. CoppeliaSim, only if we started it. It ignores SIGTERM/SIGINT (it is a GUI
  #    application), so ask it to quit through its API and force it as a last resort.
  if [[ -n "$SIM_PID" && "$KEEP_SIM_OPEN" != "1" ]]; then
    victims=($(tree_pids "$SIM_PID"))
    timeout 20 "$PYTHON_BIN" -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
RemoteAPIClient(port=$SIM_PORT).require('sim').quitSimulator()" >/dev/null 2>&1 || true
    wait_gone 15 "${victims[@]}" || for pid in "${victims[@]}"; do kill -KILL "$pid" 2>/dev/null; done
  fi
  true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# 1. CoppeliaSim with its GUI. Reused if already running on $SIM_PORT.
if port_open "$SIM_PORT"; then
  say "CoppeliaSim already running on port $SIM_PORT; reusing it (it will be left open)"
else
  [[ -x "$COPPELIASIM_SH" ]] || die "CoppeliaSim launcher not found: $COPPELIASIM_SH (set COPPELIASIM_SH)"
  if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" && "$COPPELIASIM_ARGS" != *-h* ]]; then
    die "no display for the CoppeliaSim GUI (set DISPLAY, or COPPELIASIM_ARGS=-h for headless)"
  fi
  say "Starting CoppeliaSim (log: $LOG_DIR/coppeliasim.log) ..."
  # CoppeliaSim runs the scene's Python scripts with the first `python3` on ITS
  # PATH, and they need pyzmq and cbor2/cbor (installed with the requirements).
  # Put the virtualenv first so that is the interpreter it finds.
  SIM_PATH="$PATH"
  [[ "$PYTHON_BIN" == */* ]] && SIM_PATH="$(dirname "$PYTHON_BIN"):$PATH"
  PATH="$SIM_PATH" python3 -c "
import zmq
try:
    import cbor2
except ModuleNotFoundError:
    import cbor" 2>/dev/null \
    || die "the python3 CoppeliaSim would use lacks pyzmq/cbor2; see 'Installation' in examples/burger_coppeliasim/README.md"
  # shellcheck disable=SC2086
  PATH="$SIM_PATH" "$COPPELIASIM_SH" $COPPELIASIM_ARGS >"$LOG_DIR/coppeliasim.log" 2>&1 &
  SIM_PID="$!"
  deadline=$((SECONDS + STARTUP_TIMEOUT))
  until port_open "$SIM_PORT"; do
    kill -0 "$SIM_PID" 2>/dev/null || die "CoppeliaSim exited during startup (see $LOG_DIR/coppeliasim.log)"
    (( SECONDS < deadline )) || die "timed out waiting for the CoppeliaSim ZMQ API on port $SIM_PORT"
    sleep 1
  done
fi

# 2. RL side: opens the TCP server and trains. It accepts the agent for 60 s
#    (fixed in spindecoupler), so the agent is started right after.
TRAIN_ARGS=(--timesteps "$TIMESTEPS" --model-path "$MODEL_OUT")
export PORT="$RL_PORT" DEVICE TIMEOUT="$RL_TIMEOUT" SIM_PORT   # read by the launchers
say "Starting the RL side (SAC, $TIMESTEPS steps, model: $MODEL_OUT) ..."
if [[ -n "$RL_LAUNCHER" ]]; then
  read -ra RL_CMD <<< "$RL_LAUNCHER"
  run_logged rl "${RL_CMD[@]}" "${TRAIN_ARGS[@]}"
else
  run_logged rl "$PYTHON_BIN" "$EXAMPLE_DIR/rl_side_burger_coppeliasim.py" \
    --port "$RL_PORT" --device "$DEVICE" --timeout "$RL_TIMEOUT" "${TRAIN_ARGS[@]}"
fi
RL_PID="$LAST_PID"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until port_open "$RL_PORT"; do
  kill -0 "$RL_PID" 2>/dev/null || die "the RL side exited during startup (see $LOG_DIR/rl.log)"
  (( SECONDS < deadline )) || die "timed out waiting for the RL side to listen on port $RL_PORT"
  sleep 0.5
done

# 3. Agent side: loads the scene into CoppeliaSim and serves the RL side.
say "Starting the agent (loads $SCENE_PATH) ..."
if [[ -n "$AGENT_LAUNCHER" ]]; then
  read -ra AGENT_CMD <<< "$AGENT_LAUNCHER"
  run_logged agent "${AGENT_CMD[@]}"
else
  run_logged agent "$PYTHON_BIN" "$EXAMPLE_DIR/agent_side_burger_coppeliasim.py" \
    --ip 127.0.0.1 --port "$RL_PORT" --sim-port "$SIM_PORT" --scene-path "$SCENE_PATH"
fi
AGENT_PID="$LAST_PID"

# Run until training finishes (the RL side then saves the model and sends FINISH
# to the agent), or either component dies. Ctrl+C stops everything.
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
