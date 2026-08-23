# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import argparse
import time
from enum import Enum
from typing import List

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from spindecoupler import AgentSide, BaseCommPoint

from reward import compute_reward, is_terminated, is_truncated


class StepState(Enum):
    READY_FOR_RL_COMMAND = 0
    EXECUTING_LAST_ACTION = 1
    AFTER_RESET = 2


class CoppeliaBurgerAgent:
    """Agent-side process controlling a CoppeliaSim TurtleBot3 Burger.

    Runs a decoupled state machine identical in spirit to the repo's
    LunarLander example: a fast control loop (`control_period`) keeps the
    simulator stepping in real time while a slower RL loop
    (`rl_step_period`) asynchronously decides the next action.

    Reward and episode termination/truncation are computed here, in
    agent-side (see reward.py), not on the RL side and not by the
    CoppeliaSim orchestrator script -- that is a deliberate project choice,
    documented in reward.py and orchestrator.py.
    """

    def __init__(
        self,
        ip: str,
        port: int,
        rl_step_period: float,
        control_period: float,
        timeout: float,
        max_steps: int = 500,
        sim_host: str = "127.0.0.1",
        sim_port: int = 23000,
        debug: bool = False,
    ):
        if rl_step_period <= control_period:
            raise ValueError("rl_step_period must be strictly greater than control_period")

        self._debug = debug
        self._timeout = timeout
        self._rl_step_period = rl_step_period
        self._control_period = control_period
        self._max_steps = max_steps

        # 1. ZMQ connection to CoppeliaSim.
        self._client = RemoteAPIClient(host=sim_host, port=sim_port)
        self._sim = self._client.require("sim")
        self._orchestrator_handle = self._sim.getScript(
            self._sim.scripttype_childscript, self._sim.getObject("/Orchestrator")
        )

        # 2. Decoupling socket (AgentSide).
        self._comm = AgentSide(ip, port, verbose=debug)

        self._state = StepState.READY_FOR_RL_COMMAND
        self._last_action = self._null_action()
        self._last_action_start = time.time()

        self._obs = np.zeros((10,), dtype=np.float32)
        self._prev_obs: np.ndarray | None = None
        self._step_count = 0
        self._reset_pending_after_publish = False

        self._reset_workspace()

    def _null_action(self) -> List[float]:
        """Neutral action: [v=0.0 m/s, w=0.0 rad/s]."""
        return [0.0, 0.0]

    def _apply_action(self, action: List[float]) -> None:
        """Store the continuous action to be applied in the simulator."""
        self._last_action = [float(action[0]), float(action[1])]

    def _build_observation(self) -> np.ndarray:
        return self._obs

    def _build_transport_payload(
        self, terminated: bool = False, truncated: bool = False
    ) -> dict[str, object]:
        return {
            "observation": self._build_observation().tolist(),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def _reset_workspace(self) -> None:
        """Trigger the CoppeliaSim episode reset and store the first observation."""
        if self._sim.getSimulationState() == self._sim.simulation_stopped:
            self._sim.startSimulation()

        raw_obs = self._sim.callScriptFunction("reset_episode", self._orchestrator_handle)
        self._obs = np.asarray(raw_obs, dtype=np.float32)
        # The first step of the new episode compares against the post-reset
        # observation, same as DecoupledCoppeliaBurgerEnv.reset() used to.
        self._prev_obs = self._obs.copy()
        self._step_count = 0

        self._reset_pending_after_publish = False
        self._last_action = self._null_action()

    def _run_control_tick(self) -> None:
        """Run one control tick: apply the current action and read fresh sensors.

        This does NOT decide whether the episode is over -- orchestrator.py's
        step_episode() only ever returns the raw observation. Termination is
        computed once per RL step, in spinloop(), from that observation.
        """
        if self._reset_pending_after_publish:
            return

        raw_obs = self._sim.callScriptFunction(
            "step_episode", self._orchestrator_handle, self._last_action
        )
        self._obs = np.asarray(raw_obs, dtype=np.float32)

    def spinloop(self) -> None:
        """Run the decoupled state machine, identical in spirit to LunarLander."""
        print("[AGENT] Running CoppeliaSim Burger loop")
        try:
            while True:
                now = time.time()
                self._run_control_tick()

                if self._state == StepState.EXECUTING_LAST_ACTION:
                    if now - self._last_action_start >= self._rl_step_period:
                        self._step_count += 1
                        terminated = bool(is_terminated(self._obs))
                        truncated = is_truncated(self._step_count, self._max_steps)
                        if terminated:
                            self._reset_pending_after_publish = True

                        reward = compute_reward(
                            obs=self._obs,
                            action=np.asarray(self._last_action, dtype=np.float32),
                            prev_obs=self._prev_obs,
                            lat=now - self._last_action_start,
                        )
                        self._comm.stepSendObs(
                            self._build_transport_payload(
                                terminated=terminated, truncated=truncated
                            ),
                            agenttime=now,
                            rew=reward,
                        )
                        if self._debug:
                            print(
                                f"[AGENT] Sent transition obs action={self._last_action} "
                                f"reward={reward:.4f} terminated={terminated} "
                                f"truncated={truncated} step={self._step_count}"
                            )

                        # The observation just published becomes the "prev_obs"
                        # reference for the next reward computation.
                        self._prev_obs = self._obs.copy()

                        if self._reset_pending_after_publish:
                            self._reset_workspace()
                            self._last_action_start = now
                            self._apply_action(self._null_action())
                            if self._debug:
                                print("[AGENT] Local auto-reset after termination")

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

                time.sleep(self._control_period)
        finally:
            self._sim.stopSimulation()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoppeliaSim Agent-side wrapper")
    parser.add_argument("--ip", default=BaseCommPoint.get_ip(), help="RL-side IPv4 address")
    parser.add_argument("--port", type=int, default=49054, help="RL-side TCP port")
    parser.add_argument("--rl-step-period", type=float, default=0.08, help="RL action step period")
    parser.add_argument("--control-period", type=float, default=0.01, help="Sim tick period")
    parser.add_argument("--timeout", type=float, default=1.0, help="Timeout readWhatToDo")
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode (defines truncation here; no longer exists in rl_side)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = CoppeliaBurgerAgent(
        ip=args.ip,
        port=args.port,
        rl_step_period=args.rl_step_period,
        control_period=args.control_period,
        timeout=args.timeout,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    try:
        agent.spinloop()
    except KeyboardInterrupt:
        print("[AGENT] Interrupted by user")


if __name__ == "__main__":
    main()
