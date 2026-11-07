#!/usr/bin/env bash
# One-command demo: the real-robot ROS2 agent driving a simulated TurtleBot3 Burger.
#
# Starts, in dependency order, and tears everything down on exit:
#
#   1. CoppeliaSim (GUI)         the physics simulator (skipped if already running)
#   2. burger_sim_bridge         loads scene.ttt, exposes /scan /odom /cmd_vel /goal_pose /tf
#   3. RL side                   loads the trained SAC policy (examples/burger_real)
#   4. burger_real_agent         the unchanged real-robot ROS2 agent (examples/burger_real)
#
# The demo runs EPISODES episodes (one target each) and then stops by itself. See README.md.
#
# Usage:   ./examples/burger_real_sim/run_demo.sh
# Options (environment variables, defaults in brackets):
#   EPISODES        episodes (one target each) to run, then stop; 0 = until Ctrl+C  [5]
#   MODEL_PATH      trained SAC policy (.zip)   [examples/burger_real/training_8rays_fixedtime.zip]
#   DEVICE          PyTorch device for the policy: cpu | cuda | auto              [cpu]
#   COPPELIASIM_SH  CoppeliaSim launcher                [/opt/CoppeliaSim_Edu/coppeliaSim.sh]
#   COPPELIASIM_ARGS extra CoppeliaSim arguments, e.g. "-h" for headless          []
#   ROS_SETUP       ROS2 Humble setup script                  [/opt/ros/humble/setup.bash]
#   KEEP_SIM_OPEN   1 = leave CoppeliaSim open after the demo (if we started it)   [0]
#   LOG_DIR         where per-component logs are written            [a new /tmp dir]
# Timeouts (see "Timeouts" in README.md; the defaults suit a slow simulator):
#   RL_TIMEOUT      seconds the RL side waits for each agent reply                [60]
#   SENSOR_TIMEOUT  seconds of sensor silence before the agent truncates          [10]
#   STARTUP_TIMEOUT seconds to wait for CoppeliaSim / bridge / RL side to be up  [180]
#   SIM_PORT, RL_PORT  CoppeliaSim ZMQ API port / RL-agent TCP port      [23000, 49054]

set -euo pipefail

EPISODES="${EPISODES:-5}"
DEVICE="${DEVICE:-cpu}"
COPPELIASIM_SH="${COPPELIASIM_SH:-/opt/CoppeliaSim_Edu/coppeliaSim.sh}"
COPPELIASIM_ARGS="${COPPELIASIM_ARGS:-}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
KEEP_SIM_OPEN="${KEEP_SIM_OPEN:-0}"
SIM_PORT="${SIM_PORT:-23000}"
RL_PORT="${RL_PORT:-49054}"
RL_TIMEOUT="${RL_TIMEOUT:-60}"
SENSOR_TIMEOUT="${SENSOR_TIMEOUT:-10}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/examples/burger_real/training_8rays_fixedtime.zip}"
SCENE_PATH="$REPO_ROOT/examples/_common/burger_scene/scene.ttt"
AGENT_WS="$REPO_ROOT/examples/burger_real/ros2"
BRIDGE_WS="$REPO_ROOT/examples/burger_real_sim/ros2"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/burger_real_sim.XXXXXX)}"
mkdir -p "$LOG_DIR"

say() { echo "[run] $*"; }
die() { echo "[run] ERROR: $*" >&2; exit 1; }

# Pre-flight checks: fail early with an actionable message.
# True if something listens on TCP port $1. It only inspects the socket table:
# actually connecting to the RL port would be taken by the RL side as "the agent".
port_open() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
command -v ss >/dev/null || die "'ss' not found (sudo apt install iproute2)"

[[ -f "$ROS_SETUP" ]]  || die "ROS2 setup file not found: $ROS_SETUP (set ROS_SETUP)"
[[ -f "$MODEL_PATH" ]] || die "policy not found: $MODEL_PATH (set MODEL_PATH)"
[[ -f "$SCENE_PATH" ]] || die "scene not found: $SCENE_PATH"
port_open "$RL_PORT"   && die "port $RL_PORT is already in use (another demo running? set RL_PORT)"

# ROS2 setup files read variables that are unset in a fresh shell.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u
command -v colcon >/dev/null || die "colcon not found (sudo apt install python3-colcon-common-extensions)"

# Python for the RL side: the repo virtualenv if present, else the system one.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  # `ros2 run` executes the nodes with the system Python (its shebang), which
  # does not see a virtualenv; expose the venv's packages to it explicitly.
  VENV_SITE="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  export PYTHONPATH="$VENV_SITE:${PYTHONPATH:-}"
else
  PYTHON_BIN="python3"
fi
# `pip install -e .` only adds a .pth file to the venv, which PYTHONPATH does not
# process: put the repository root (where spindecoupler.py lives) on it directly.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1   # show each component's output immediately

# Checked from a neutral directory, so that the current one cannot satisfy an import.
(cd / && "$PYTHON_BIN" -c "import stable_baselines3, spindecoupler") 2>/dev/null \
  || die "RL-side Python packages missing; see 'Installation' in examples/burger_real_sim/README.md"
(cd / && /usr/bin/python3 -c "import rclpy, numpy, spindecoupler, coppeliasim_zmqremoteapi_client") 2>/dev/null \
  || die "ROS-side Python packages missing (rclpy/numpy/spindecoupler/coppeliasim_zmqremoteapi_client); see 'Installation' in examples/burger_real_sim/README.md"

# Process management.
PIDS=()      # components started by this script (bridge, RL side, agent)
SIM_PID=""   # CoppeliaSim, only if this script started it

# Run a component in the background; its output is logged to $LOG_DIR/<name>.log
# and echoed to the terminal prefixed with [<name>].
run_logged() {
  local name="$1"; shift
  "$@" > >(tee "$LOG_DIR/$name.log" | sed -u "s/^/[$name] /") 2>&1 &
  PIDS+=("$!")
  LAST_PID="$!"
}

# Print a process and all of its descendants. `ros2 run` and coppeliaSim.sh
# spawn the real program as a child, so signalling only the stored PID would
# orphan it. The list is collected BEFORE signalling, while the tree is intact.
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
  # 1. Our components. SIGTERM makes the bridge stop the simulation cleanly.
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && victims+=($(tree_pids "$pid")); done
  for pid in "${victims[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null; done
  wait_gone 8 "${victims[@]:-}" || for pid in "${victims[@]}"; do kill -KILL "$pid" 2>/dev/null; done
  # 2. CoppeliaSim, only if we started it. It ignores SIGTERM/SIGINT (it is a GUI
  #    application), so ask it to quit through its API and force it as a last resort.
  if [[ -n "$SIM_PID" && "$KEEP_SIM_OPEN" != "1" ]]; then
    victims=($(tree_pids "$SIM_PID"))
    timeout 20 /usr/bin/python3 -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
RemoteAPIClient(port=$SIM_PORT).require('sim').quitSimulator()" >/dev/null 2>&1 || true
    wait_gone 15 "${victims[@]}" || for pid in "${victims[@]}"; do kill -KILL "$pid" 2>/dev/null; done
  fi
  true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Wait until a log file contains a pattern, or fail if the process dies / times out.
wait_for_log() {
  local file="$1" pattern="$2" pid="$3" what="$4" deadline=$((SECONDS + STARTUP_TIMEOUT))
  until grep -q "$pattern" "$file" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || die "$what exited during startup (see $file)"
    (( SECONDS < deadline )) || die "timed out after ${STARTUP_TIMEOUT}s waiting for $what (see $file)"
    sleep 0.5
  done
}

# 1. Build the two ROS2 packages (fast when nothing changed).
build() {
  local workspace="$1" package="$2"
  say "Building $package ..."
  (cd "$workspace" && colcon build --symlink-install --packages-select "$package") \
    >"$LOG_DIR/colcon_$package.log" 2>&1 \
    || { cat "$LOG_DIR/colcon_$package.log" >&2; die "colcon build of $package failed"; }
}
build "$BRIDGE_WS" burger_sim_bridge
build "$AGENT_WS" burger_real_agent
set +u
# shellcheck disable=SC1091
source "$BRIDGE_WS/install/setup.bash"
# shellcheck disable=SC1091
source "$AGENT_WS/install/setup.bash"
set -u

# 2. CoppeliaSim with its GUI. Reused if already running on $SIM_PORT.
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
    || die "the python3 CoppeliaSim would use lacks pyzmq/cbor2; see 'Installation' in examples/burger_real_sim/README.md"
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

# 3. Bridge: loads the scene, starts the simulation, publishes the ROS2 topics.
say "Starting the bridge (loads $SCENE_PATH) ..."
run_logged bridge ros2 run burger_sim_bridge bridge_node \
  --sim-port "$SIM_PORT" --scene-path "$SCENE_PATH"
BRIDGE_PID="$LAST_PID"
wait_for_log "$LOG_DIR/bridge.log" "burger_sim_bridge ready" "$BRIDGE_PID" "the bridge"

# 4. RL side: opens the TCP server, then loads the policy. It accepts the agent
#    for 60 s (fixed in spindecoupler), so the agent is started right after.
say "Starting the RL side (policy: $MODEL_PATH, episodes: $EPISODES) ..."
run_logged rl "$PYTHON_BIN" "$REPO_ROOT/examples/burger_real/rl_side_burger_real.py" \
  --port "$RL_PORT" --model-path "$MODEL_PATH" --device "$DEVICE" \
  --timeout "$RL_TIMEOUT" --episodes "$EPISODES"
RL_PID="$LAST_PID"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until port_open "$RL_PORT"; do
  kill -0 "$RL_PID" 2>/dev/null || die "the RL side exited during startup (see $LOG_DIR/rl.log)"
  (( SECONDS < deadline )) || die "timed out waiting for the RL side to listen on port $RL_PORT"
  sleep 0.5
done

# 5. The real-robot agent, unchanged. use_sim_time follows the bridge's /clock.
say "Starting the ROS2 agent ..."
run_logged agent ros2 run burger_real_agent agent_node --ros-args -p use_sim_time:=true -- \
  --ip 127.0.0.1 --port "$RL_PORT" --sensor-timeout "$SENSOR_TIMEOUT"
AGENT_PID="$LAST_PID"

# Run until the RL side has finished its episodes (it then sends FINISH to the
# agent), or any component dies. Ctrl+C stops everything.
say "Running. Ctrl+C to stop."
while kill -0 "$RL_PID" 2>/dev/null && kill -0 "$AGENT_PID" 2>/dev/null \
      && kill -0 "$BRIDGE_PID" 2>/dev/null; do
  sleep 1
done

status=0
# Normal end: the RL side finishes its episodes and exits, and the agent exits
# right after receiving FINISH. Give the RL process a moment to exit on its own
# before treating the loop exit as a failure.
if wait_gone 10 "$RL_PID"; then
  wait "$RL_PID" || status=$?
  wait_gone 20 "$AGENT_PID" || true
else
  for component in bridge:$BRIDGE_PID agent:$AGENT_PID; do
    kill -0 "${component#*:}" 2>/dev/null || say "The ${component%%:*} stopped unexpectedly (log: $LOG_DIR/${component%%:*}.log)"
  done
  status=1
fi

if (( status == 0 )); then say "Demo finished."; else say "Demo failed (exit status $status)."; fi
exit "$status"
