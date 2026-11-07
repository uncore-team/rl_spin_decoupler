Tutorial: End-to-End Loop
=========================

This tutorial follows one complete cycle: a reset, some steps, and the end of the
experiment. The setup is an RL environment wrapper in one process and the control
loop of a robot or simulator in another, with the timing measured by the agent
reaching the learner.

RL-side wrapper
---------------

The RL process owns the server socket and blocks until the agent connects.

.. code-block:: python

   import time
   from spindecoupler import RLSide


   class RLEnv:
       def __init__(self) -> None:
           self._comms = RLSide(49054, verbose=True)

       def reset(self):
           obs, ato = self._comms.resetGetObs()
           return obs, {"t_agent": ato}

       def step(self, action):
           lat, obs, rew, ato = self._comms.stepSendActGetObs(action)
           t_wall = time.time()
           info = {"lat": lat, "t_agent": ato, "t_wall": t_wall}
           terminated = False
           truncated = False
           return obs, rew, terminated, truncated, info

       def close(self):
           self._comms.stepExpFinished()

Agent-side loop
---------------

The agent polls the socket on every iteration and carries on with its own work when the
RL side has sent nothing.

.. code-block:: python

   import time
   from spindecoupler import AgentSide, BaseCommPoint


   class Agent:
       def __init__(self) -> None:
           self._last_action_start = time.time()
           self._comms = AgentSide(BaseCommPoint.get_ip(), 49054, verbose=True)

       def reset_workspace(self) -> None:
           pass

       def build_observation(self):
           return {"position": 0.0, "velocity": 0.0}

       def run(self) -> None:
           while True:
               message = self._comms.readWhatToDo(timeout=0.5)
               if message is None:
                   continue

               what, payload = message
               now = time.time()

               if what == AgentSide.WhatToDo.RESET_SEND_OBS:
                   self.reset_workspace()
                   self._comms.resetSendObs(self.build_observation(), agenttime=now)

               elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                   lat = now - self._last_action_start
                   self._comms.stepSendLastActDur(lat)
                   # Apply payload here.
                   self._last_action_start = now
                   self._comms.stepSendObs(self.build_observation(), agenttime=now, rew=0.0)

               elif what == AgentSide.WhatToDo.FINISH:
                   break

Order of execution
------------------

1. Start the RL process, so the server socket exists.
2. Start the agent process, which connects to it.
3. Call ``resetGetObs()`` on the RL side.
4. Call ``stepSendActGetObs(action)`` for each RL step.
5. Call ``stepExpFinished()`` once, when learning is over.

Why the timing values matter
----------------------------

The two processes do not share a notion of time, and the library keeps their clocks
apart. ``LAT`` is how long the previous action really lasted on the agent, ``ATO`` is
when the agent sampled the observation, and ``t_wall`` is something the RL code can add
to record when the answer arrived locally.
