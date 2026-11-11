# SPDX-License-Identifier: GPL-3.0-only
"""Agent side of the CoppeliaSim Burger example.

This process owns the simulator. It connects to a running CoppeliaSim through
the ZMQ remote API, loads the shared arena scene, and then, on behalf of the RL
side (``rl_side_burger_coppeliasim.py``, connected through ``spindecoupler``):

* applies the continuous action ``[linear m/s, angular rad/s]`` chosen by the RL
  side to the robot, and keeps stepping the simulation with it;
* returns the 10-value observation ``[distance_to_goal, relative_goal_angle,
  ray_1..ray_8]`` after every RL step and after every reset;
* repositions robot, target and obstacles when the RL side asks for a reset.

Start CoppeliaSim first (its GUI is fine), then the RL side, then this script;
``run_demo.sh`` does all of it.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from enum import Enum
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from episode_config import COLLISION_LIDAR_THRESHOLD, NUM_LIDAR_SECTORS, OBS_DIM

from spindecoupler import AgentSide, BaseCommPoint, ConnectionLost

# The scene is shared with examples/burger_real_sim. It is located from this file
# so that the working directory does not matter.
DEFAULT_SCENE_PATH = str(
    Path(__file__).resolve().parent.parent / "_common" / "burger_scene" / "scene.ttt"
)


class StepState(Enum):
    """Phases of the decoupled control loop."""

    READY_FOR_RL_COMMAND = 0  # waiting for the next command from the RL side
    EXECUTING_LAST_ACTION = 1  # holding the last action for one RL step period
    AFTER_RESET = 2  # world repositioned; first observation about to be sent


class CoppeliaBurgerAgent:
    """
    Drives the simulated Burger on behalf of the RL side.

    It follows the same loop as the LunarLander agent: a fast control loop keeps the
    simulation stepping with the current action (`control_period`) while a slower
    one (`rl_step_period`) decides when to send the next observation.

    It computes no reward. It does detect the one thing only it can see, a
    collision, from the LiDAR readings (COLLISION_LIDAR_THRESHOLD in
    episode_config.py), and sends it as ``terminated``. On a collision it freezes the
    robot and waits for the reset instead of resetting by itself. The RL side adds
    the goal check and its own step budget.
    """

    def __init__(
        self,
        ip: str,
        port: int,
        rl_step_period: float,
        control_period: float,
        timeout: float,
        sim_host: str = "127.0.0.1",
        sim_port: int = 23000,
        scene_path: str | None = None,
        debug: bool = False,
        reconnect_timeout: float = 180.0,
    ):
        if rl_step_period <= control_period:
            raise ValueError(
                "rl_step_period must be strictly greater than control_period"
            )

        self._debug = debug
        self._timeout = timeout
        self._reconnect_timeout = reconnect_timeout
        self._rl_step_period = rl_step_period
        self._control_period = control_period

        self._client = RemoteAPIClient(host=sim_host, port=sim_port)
        self._sim = self._client.require("sim")

        # The scene has to be loaded before /Orchestrator can be looked up.
        self._load_scene(scene_path)
        orchestrator_object = self._sim.getObject("/Orchestrator")
        if orchestrator_object <= 0:
            raise RuntimeError(
                "The loaded scene has no '/Orchestrator' object; use "
                "examples/_common/burger_scene/scene.ttt (see its README)."
            )
        self._orchestrator_handle = self._sim.getScript(
            self._sim.scripttype_childscript, orchestrator_object
        )
        if self._orchestrator_handle <= 0:
            raise RuntimeError(
                "The '/Orchestrator' object has no child script; use "
                "examples/_common/burger_scene/scene.ttt (see its README)."
            )

        self._comm = AgentSide(ip, port, verbose=debug)

        self._state = StepState.READY_FOR_RL_COMMAND
        self._last_action = self._null_action()
        self._last_action_start = time.time()

        self._obs = np.zeros((OBS_DIM,), dtype=np.float32)
        # Physics termination latched during the action-repeat window and
        # reported at the next RL-step boundary; frozen until the RL reset.
        self._term_latched = False
        self._frozen = False

        self._reset_workspace()

    def _load_scene(self, scene_path: str | None) -> None:
        """
        Stop any running simulation, load the scene and switch on real-time mode.

        Real-time mode makes the simulation clock follow the wall clock, which is what
        the loop in ``spinloop()`` assumes. Without it CoppeliaSim would step as fast as
        it can and simulation time would drift away from real time.
        """
        path = os.path.abspath(scene_path) if scene_path else DEFAULT_SCENE_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene file not found: {path}")

        if self._sim.getSimulationState() != self._sim.simulation_stopped:
            self._sim.stopSimulation()
            while self._sim.getSimulationState() != self._sim.simulation_stopped:
                time.sleep(0.05)

        self._sim.loadScene(path)
        self._sim.setBoolParam(self._sim.boolparam_realtime_simulation, True)
        print(f"[AGENT] Loaded scene '{path}' (real-time simulation enabled)")

    def _null_action(self) -> list[float]:
        return [0.0, 0.0]

    def _apply_action(self, action: list[float]) -> None:
        self._last_action = [float(action[0]), float(action[1])]

    def _build_observation(self) -> np.ndarray:
        return self._obs

    def _is_collision(self, obs: np.ndarray) -> bool:
        lidar_min = float(np.min(obs[2 : 2 + NUM_LIDAR_SECTORS]))
        return lidar_min <= COLLISION_LIDAR_THRESHOLD

    def _build_transport_payload(
        self, terminated: bool = False, truncated: bool = False
    ) -> dict[str, object]:
        return {
            "observation": self._build_observation().tolist(),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def _reset_workspace(self) -> None:
        """Reset the episode in CoppeliaSim and keep the first observation."""
        if self._sim.getSimulationState() == self._sim.simulation_stopped:
            self._sim.startSimulation()

        raw_obs = self._sim.callScriptFunction(
            "reset_episode", self._orchestrator_handle
        )
        self._obs = np.asarray(raw_obs, dtype=np.float32)
        self._term_latched = False
        self._frozen = False
        self._last_action = self._null_action()

    def _run_control_tick(self) -> None:
        """
        Step the simulation once with the current action.

        The orchestrator only returns the observation; the collision is detected here
        from it. Once raised, the episode stays frozen until the reset arrives.
        """
        if self._frozen:
            return

        raw_obs = self._sim.callScriptFunction(
            "step_episode", self._orchestrator_handle, self._last_action
        )
        self._obs = np.asarray(raw_obs, dtype=np.float32)
        if self._is_collision(self._obs):
            self._term_latched = True
            self._frozen = True

    def spinloop(self) -> None:
        """
        Serve the RL side until it sends FINISH.

        Each iteration steps the simulation, so the robot keeps moving while the RL side
        thinks, and then either sends the observation (once ``rl_step_period`` has
        elapsed) or polls for the next command.
        """
        print("[AGENT] Running CoppeliaSim Burger loop")
        try:
            while True:
                try:
                    now = time.time()
                    self._run_control_tick()

                    if self._state == StepState.EXECUTING_LAST_ACTION:
                        if now - self._last_action_start >= self._rl_step_period:
                            # A collided episode stays frozen until the RL side
                            # resets it.
                            self._comm.stepSendObs(
                                self._build_transport_payload(
                                    terminated=self._term_latched, truncated=False
                                ),
                                agenttime=now,
                            )
                            if self._debug:
                                print(
                                    f"[AGENT] Sent obs action={self._last_action} "
                                    f"terminated={self._term_latched}"
                                )
                            self._state = StepState.READY_FOR_RL_COMMAND

                    elif self._state == StepState.AFTER_RESET:
                        self._comm.resetSendObs(
                            self._build_transport_payload(), agenttime=now
                        )
                        if self._debug:
                            print("[AGENT] Sent reset observation")
                        self._state = StepState.READY_FOR_RL_COMMAND

                    elif self._state == StepState.READY_FOR_RL_COMMAND:
                        command = self._comm.readWhatToDo(timeout=self._timeout)
                        if command is not None:
                            what, payload = command
                            if what == AgentSide.WhatToDo.RESET_SEND_OBS:
                                self._reset_workspace()
                                self._last_action_start = now
                                self._state = StepState.AFTER_RESET
                            elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                                lat = now - self._last_action_start
                                self._comm.stepSendLastActDur(lat)
                                self._last_action_start = now
                                self._apply_action(payload)
                                self._state = StepState.EXECUTING_LAST_ACTION
                            elif what == AgentSide.WhatToDo.FINISH:
                                print("[AGENT] Finish command received")
                                return
                except ConnectionLost:
                    # Holding the last action while disconnected is not safe.
                    self._apply_action(self._null_action())
                    print("[AGENT] Connection lost; reconnecting")
                    self._comm.reconnect(timeout=self._reconnect_timeout)
                    self._state = StepState.READY_FOR_RL_COMMAND
                time.sleep(self._control_period)
        finally:
            self._sim.stopSimulation()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoppeliaSim Agent-side wrapper")
    parser.add_argument(
        "--ip", default=BaseCommPoint.get_ip(), help="RL-side IPv4 address"
    )
    parser.add_argument("--port", type=int, default=49054, help="RL-side TCP port")
    parser.add_argument(
        "--rl-step-period",
        type=float,
        default=0.08,
        help="Seconds each RL action is executed (wall clock); "
        "must exceed --control-period",
    )
    parser.add_argument(
        "--control-period",
        type=float,
        default=0.01,
        help="Seconds between simulation steps / polls (wall clock)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for a pending RL command before stepping again",
    )
    parser.add_argument(
        "--scene-path",
        type=str,
        default=None,
        help="Scene (.ttt) to load (default: examples/_common/burger_scene/scene.ttt)",
    )
    parser.add_argument(
        "--sim-host", default="127.0.0.1", help="CoppeliaSim ZMQ remote API host"
    )
    parser.add_argument(
        "--sim-port", type=int, default=23000, help="CoppeliaSim ZMQ remote API port"
    )
    parser.add_argument(
        "--reconnect-timeout",
        type=float,
        default=180.0,
        help="Seconds to keep retrying after a lost connection; the RL side may need "
        "minutes to restart (it imports PyTorch)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def _raise_keyboard_interrupt(*_) -> None:
    raise KeyboardInterrupt


def main() -> None:
    args = parse_args()
    # Let `kill` (SIGTERM) stop the simulation cleanly, like Ctrl+C does.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        agent = CoppeliaBurgerAgent(
            ip=args.ip,
            port=args.port,
            rl_step_period=args.rl_step_period,
            control_period=args.control_period,
            timeout=args.timeout,
            sim_host=args.sim_host,
            sim_port=args.sim_port,
            scene_path=args.scene_path,
            debug=args.debug,
            reconnect_timeout=args.reconnect_timeout,
        )
        agent.spinloop()
    except KeyboardInterrupt:
        print("[AGENT] Interrupted")


if __name__ == "__main__":
    main()
