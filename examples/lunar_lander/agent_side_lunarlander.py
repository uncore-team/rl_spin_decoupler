# SPDX-License-Identifier: GPL-3.0-only

"""
Agent side of the LunarLander example: the simulation.

Runs Gymnasium's LunarLander-v3 and serves the RL side (rl_side_lunarlander.py,
which has to be listening already). A fast loop steps the physics with the
current action, and commands from the RL side are polled without blocking.

The agent computes no reward. It reports what only the environment knows,
Gymnasium's ``terminated`` (crash, out of bounds) and ``truncated`` (time limit),
with each observation; the RL side adds the landing check and its own step
budget. When the physics ends an episode the agent freezes and waits for the RL
side to reset it, instead of resetting by itself.
"""

from __future__ import annotations

import argparse
import signal
import time
from enum import Enum

import gymnasium as gym
import numpy as np

from spindecoupler import AgentSide, BaseCommPoint


class StepState(Enum):
    """Phases of the loop that serves the RL side."""

    READY_FOR_RL_COMMAND = 0
    EXECUTING_LAST_ACTION = 1
    AFTER_RESET = 2


class LunarLanderAgent:
    """Runs LunarLander-v3 and answers the commands of the RL side."""

    def __init__(
        self,
        ip: str,
        port: int,
        rl_step_period: float,
        control_period: float,
        timeout: float,
        seed: int = 7,
        render: bool = False,
        debug: bool = False,
    ):
        if rl_step_period <= control_period:
            raise ValueError(
                "rl_step_period must be strictly greater than control_period"
            )

        self._debug = debug
        self._timeout = timeout
        self._rl_step_period = rl_step_period
        self._control_period = control_period
        self._episode_seed = seed
        self._render_enabled = render
        self._render_available = render

        self._comm = AgentSide(ip, port, verbose=debug)
        env_kwargs = {"render_mode": "human"} if render else {}
        self._env = gym.make("LunarLander-v3", **env_kwargs)

        self._state = StepState.READY_FOR_RL_COMMAND
        self._last_action = self._null_action()
        self._last_action_start = time.time()

        self._obs = np.zeros((8,), dtype=np.float32)
        # Physics termination flags latched during the action-repeat window and
        # reported at the next RL-step boundary.
        self._term_latched = False
        self._trunc_latched = False
        # While frozen, the finished episode is not stepped; cleared on reset.
        self._frozen = False

        self._reset_workspace()

    def _build_observation(self) -> np.ndarray:
        return self._obs

    def _null_action(self) -> int:
        return 0

    def _apply_action(self, action: int) -> None:
        """Keep ``action`` until the RL side sends another one."""

        self._last_action = int(action)

    def _reset_workspace(self) -> None:
        """Reset the environment and keep its first observation."""

        obs, _info = self._env.reset(seed=self._episode_seed)
        self._episode_seed += 1
        self._obs = np.asarray(obs, dtype=np.float32)
        self._term_latched = False
        self._trunc_latched = False
        self._frozen = False
        self._last_action = self._null_action()

    def _build_transport_payload(
        self, terminated: bool = False, truncated: bool = False
    ) -> dict[str, object]:
        """Observation plus the physics flags, as the RL side expects it."""

        return {
            "observation": self._build_observation().tolist(),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def _run_control_tick(self) -> None:
        """
        Step the physics once with the current action.

        Gymnasium's own reward is dropped because the RL side computes it. The
        ``terminated`` and ``truncated`` flags are latched, and once one is raised the
        episode stays frozen until the reset arrives.
        """

        if self._frozen:
            return

        obs, _reward, terminated, truncated, _info = self._env.step(
            int(self._last_action)
        )
        self._obs = np.asarray(obs, dtype=np.float32)
        self._try_render_frame()
        if terminated or truncated:
            self._term_latched = self._term_latched or bool(terminated)
            self._trunc_latched = self._trunc_latched or bool(truncated)
            self._frozen = True

    def _try_render_frame(self) -> None:
        if not self._render_enabled or not self._render_available:
            return
        try:
            self._env.render()
        except Exception as exc:
            self._render_available = False
            print(f"[AGENT] render disabled at runtime: {exc}")

    def spinloop(self) -> None:
        """Serve the RL side until it sends FINISH."""

        print("[AGENT] running LunarLander loop")
        try:
            while True:
                now = time.time()
                self._run_control_tick()

                if self._state == StepState.EXECUTING_LAST_ACTION:
                    if now - self._last_action_start >= self._rl_step_period:
                        # A finished episode stays frozen until the RL side resets it.
                        self._comm.stepSendObs(
                            self._build_transport_payload(
                                terminated=self._term_latched,
                                truncated=self._trunc_latched,
                            ),
                            agenttime=now,
                        )
                        if self._debug:
                            print(
                                "[AGENT] sent transition obs action={} "
                                "term={} trunc={}".format(
                                    self._last_action,
                                    self._term_latched,
                                    self._trunc_latched,
                                )
                            )
                        self._state = StepState.READY_FOR_RL_COMMAND

                elif self._state == StepState.AFTER_RESET:
                    self._comm.resetSendObs(
                        self._build_transport_payload(), agenttime=now
                    )
                    if self._debug:
                        print("[AGENT] sent reset observation")
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
                            self._apply_action(int(payload))
                            self._state = StepState.EXECUTING_LAST_ACTION
                        elif what == AgentSide.WhatToDo.FINISH:
                            print("[AGENT] finish command received")
                            return

                time.sleep(self._control_period)
        finally:
            self._env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent-side LunarLander wrapper for rl_spin_decoupler"
    )
    parser.add_argument(
        "--ip", default=BaseCommPoint.get_ip(), help="RL-side IPv4 address"
    )
    parser.add_argument("--port", type=int, default=49054, help="RL-side TCP port")
    parser.add_argument(
        "--rl-step-period",
        type=float,
        default=0.08,
        help="Seconds each RL action remains active before sending observation",
    )
    parser.add_argument(
        "--control-period",
        type=float,
        default=0.01,
        help="Fast spin-loop period in seconds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for a pending RL command before stepping again",
    )
    parser.add_argument("--seed", type=int, default=7, help="Initial environment seed")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable Gymnasium human rendering in the agent process",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose agent logs"
    )
    return parser.parse_args()


def _raise_keyboard_interrupt(*_) -> None:
    raise KeyboardInterrupt


def main() -> None:
    args = parse_args()
    # Let `kill` (SIGTERM) stop the agent cleanly, like Ctrl+C does.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        agent = LunarLanderAgent(
            ip=args.ip,
            port=args.port,
            rl_step_period=args.rl_step_period,
            control_period=args.control_period,
            timeout=args.timeout,
            seed=args.seed,
            render=args.render,
            debug=args.debug,
        )
        agent.spinloop()
    except KeyboardInterrupt:
        print("[AGENT] interrupted")


if __name__ == "__main__":
    main()
