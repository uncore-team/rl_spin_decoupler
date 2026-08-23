# SPDX-License-Identifier: GPL-3.0-only

"""Agent-side demo wrapping Gymnasium LunarLander-v3.

Run this process after rl_side_lunarlander.py is listening. The agent executes a
faster control loop and exchanges commands/observations with the RL process
through rl_spin_decoupler.

Design choice for internal Gymnasium termination:
- If LunarLander reaches terminated/truncated internally, the agent keeps that
        state only as local bookkeeping to avoid stepping a finished episode.
- Episode-level termination/truncation decisions still remain on RL side, from
        observation traces (see reward.py: is_terminated/is_truncated).
"""

from __future__ import annotations

import argparse
import time
from enum import Enum

import gymnasium as gym
import numpy as np

from spindecoupler import AgentSide, BaseCommPoint

from reward import compute_reward


class StepState(Enum):
    """State machine for handling decoupler commands inside the spin loop."""

    READY_FOR_RL_COMMAND = 0
    EXECUTING_LAST_ACTION = 1
    AFTER_RESET = 2


class LunarLanderAgent:
    """High-rate agent process around Gymnasium LunarLander-v3."""

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
        self._prev_obs: np.ndarray | None = None
        self._env_done_latched = False
        self._reset_pending_after_publish = False

        self._reset_workspace()

    def _build_observation(self) -> np.ndarray:
        """Return the latest LunarLander observation (8-dim vector)."""

        return self._obs

    def _null_action(self) -> int:
        """Return a neutral valid action for discrete LunarLander."""

        return 0

    def _apply_action(self, action: int) -> None:
        """Store the action to be applied on each future control tick.

        The spin loop maps one control tick to one ``env.step(action)`` call
        (1:1 mapping). The current action is held constant until a new RL action
        arrives.
        """

        self._last_action = int(action)

    def _reset_workspace(self) -> None:
        """Reset internal environment state and cache the first observation."""

        obs, info = self._env.reset(seed=self._episode_seed)
        _ = info
        self._episode_seed += 1
        self._obs = np.asarray(obs, dtype=np.float32)
        # The first step of the new episode compares against the post-reset
        # observation, same as DecoupledLunarLanderEnv.reset() used to (back
        # when it tracked _prev_obs on RL side).
        self._prev_obs = self._obs.copy()
        self._env_done_latched = False
        self._reset_pending_after_publish = False
        self._last_action = self._null_action()

    def _build_transport_payload(self) -> list[float]:
        """Build the transport payload with observation only."""

        return self._build_observation().tolist()

    def _run_control_tick(self) -> None:
        """Advance one simulator tick with the latest action."""

        if self._reset_pending_after_publish:
            return

        obs, reward, terminated, truncated, info = self._env.step(
            int(self._last_action)
        )
        # Gymnasium's own reward is intentionally discarded: this project's
        # reward signal is reward.py::compute_reward, computed below in
        # spinloop() at the RL-step boundary, not Gymnasium's native reward.
        _ = reward
        _ = info
        self._obs = np.asarray(obs, dtype=np.float32)
        self._env_done_latched = bool(terminated or truncated)
        self._try_render_frame()
        if self._env_done_latched:
            self._reset_pending_after_publish = True

    def _try_render_frame(self) -> None:
        """Render a frame when enabled, disabling on runtime display errors."""

        if not self._render_enabled or not self._render_available:
            return
        try:
            self._env.render()
        except Exception as exc:
            self._render_available = False
            print(f"[AGENT] render disabled at runtime: {exc}")

    def spinloop(self) -> None:
        """Run the agent loop until FINISH is received."""

        print("[AGENT] running LunarLander loop")
        try:
            while True:
                now = time.time()
                self._run_control_tick()

                if self._state == StepState.EXECUTING_LAST_ACTION:
                    if now - self._last_action_start >= self._rl_step_period:
                        reward = compute_reward(
                            obs=self._obs,
                            action=self._last_action,
                            prev_obs=self._prev_obs,
                            lat=now - self._last_action_start,
                        )
                        self._comm.stepSendObs(
                            self._build_transport_payload(),
                            agenttime=now,
                            rew=reward,
                        )
                        if self._debug:
                            print(
                                "[AGENT] sent transition obs action={} "
                                "reward={:.4f} internal_done={}".format(
                                    self._last_action,
                                    reward,
                                    self._env_done_latched,
                                )
                            )

                        # The observation just published becomes the
                        # "prev_obs" reference for the next reward computation.
                        self._prev_obs = self._obs.copy()

                        if self._reset_pending_after_publish:
                            # Policy: after signaling env_done once, locally
                            # auto-reset to keep simulator running and avoid
                            # stepping a done episode.
                            self._reset_workspace()
                            self._last_action_start = now
                            self._apply_action(self._null_action())
                            if self._debug:
                                print("[AGENT] local auto-reset after env_done signal")
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
    """Parse command-line arguments for the agent-side process."""

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
        help="Timeout for readWhatToDo() once data are pending",
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


def main() -> None:
    """Entry point for the executable agent-side demo."""

    args = parse_args()
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
    try:
        agent.spinloop()
    except KeyboardInterrupt:
        print("[AGENT] interrupted by user")


if __name__ == "__main__":
    main()
