Tutorial: End-to-End Loop
=========================

This tutorial shows the complete control cycle that the package is designed to
support.

Scenario
--------

Assume you have:

- an RL environment wrapper running in one process,
- a robot or simulator control loop running in another process,
- a need to preserve timing information from the agent side.

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

The agent loop polls the socket without blocking when no RL command is pending.

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

Execution order
---------------

1. Start the RL process first so the server socket is ready.
2. Start the agent process and connect to the RL socket.
3. Call ``resetGetObs()`` from the RL side.
4. For each RL step, call ``stepSendActGetObs(action)``.
5. When learning is finished, call ``stepExpFinished()`` once.

Why the timing metadata matters
-------------------------------

The RL process and the agent process do not necessarily share the same notion of
time. The library keeps those clocks separate instead of pretending they are the
same.

- ``LAT`` tells the RL side how long the previous action actually executed on
  the agent side.
- ``ATO`` tells the RL side when the observation was sampled on the agent side.
- ``t_wall`` can be added by the RL code itself to record when the data arrived
  locally.