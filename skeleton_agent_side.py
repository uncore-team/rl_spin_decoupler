# SPDX-License-Identifier: GPL-3.0-only

"""
Template for the agent side of a decoupled setup.

Start this process after the RL side, which is the one that listens. The agent
runs its own fast loop and polls the RL side without blocking; each "..." marks
something you have to supply. The RL counterpart is in skeleton_rl_side.py.
"""

import time
from enum import Enum
from typing import Any

from spindecoupler import AgentSide, BaseCommPoint, ConnectionLost


class Agent:
    """Control loop of the robot or simulator."""

    class StepState(Enum):
        READYFORRLCOMMAND = 0  # waiting for the next command from the RL side
        EXECUTINGLASTACTION = 1  # holding the last action until the RL step ends
        AFTERRESET = 2  # reset done; the first observation is still to be sent

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._rltimestep = 0.1  # seconds per RL action; must exceed the control one
        self._control_timestep = 0.02  # seconds per iteration of the local loop
        if self._rltimestep <= self._control_timestep:
            raise ValueError("RL timestep must be > control timestep")

        self._stepstate = Agent.StepState.READYFORRLCOMMAND
        self._lastaction = self._null_action()
        self._lastactiont0 = time.time()
        self._commstoRL = AgentSide(BaseCommPoint.get_ip(), 49054, verbose=debug)

    def step(self) -> Any:
        """One iteration of the loop: advance the state machine, apply the action."""
        now = time.time()

        if self._stepstate == Agent.StepState.EXECUTINGLASTACTION:
            if now - self._lastactiont0 >= self._rltimestep:
                self._commstoRL.stepSendObs(self._build_observation(), agenttime=now)
                self._stepstate = Agent.StepState.READYFORRLCOMMAND

        elif self._stepstate == Agent.StepState.READYFORRLCOMMAND:
            whattodo = self._commstoRL.readWhatToDo()  # None if nothing is pending
            if whattodo is not None:
                what, payload = whattodo

                if what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                    # LAT must be sent before the observation.
                    self._commstoRL.stepSendLastActDur(now - self._lastactiont0)
                    self._lastactiont0 = now
                    self._lastaction = payload
                    self._stepstate = Agent.StepState.EXECUTINGLASTACTION

                elif what == AgentSide.WhatToDo.RESET_SEND_OBS:
                    self._reset_workspace()
                    self._lastaction = self._null_action()
                    self._lastactiont0 = now
                    self._stepstate = Agent.StepState.AFTERRESET

                elif what == AgentSide.WhatToDo.FINISH:
                    raise RuntimeError("Experiment finished")

        elif self._stepstate == Agent.StepState.AFTERRESET:
            self._commstoRL.resetSendObs(self._build_observation(), agenttime=now)
            self._stepstate = Agent.StepState.READYFORRLCOMMAND

        self._apply_action(self._lastaction)
        return self._lastaction

    def spinloop(self) -> None:
        while True:
            try:
                self.step()
            except ConnectionLost:
                # Holding the last action while disconnected is not safe.
                self._lastaction = self._null_action()
                self._apply_action(self._lastaction)
                self._commstoRL.reconnect()
                self._stepstate = Agent.StepState.READYFORRLCOMMAND
            time.sleep(self._control_timestep)

    def _build_observation(self):
        return ...

    def _apply_action(self, action: Any) -> None:
        ...

    def _reset_workspace(self) -> None:
        ...

    def _null_action(self):
        return ...


if __name__ == "__main__":
    Agent(debug=True).spinloop()
