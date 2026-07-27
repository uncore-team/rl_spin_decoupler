Quick Start
===========

Minimal control flow
--------------------

The library coordinates two processes:

1. An RL process that decides when to reset and which action to apply next.
2. An agent process that executes actions, samples observations, and optionally
   computes rewards.

At a high level, the RL side does this:

.. code-block:: python

   from spindecoupler import RLSide

   rl = RLSide(port=49054, verbose=True)
   obs0, ato0 = rl.resetGetObs()

   action = {"move": "left"}
   lat, obs1, rew1, ato1 = rl.stepSendActGetObs(action)
   rl.stepExpFinished()

And the agent side does this:

.. code-block:: python

   from spindecoupler import AgentSide, BaseCommPoint

   agent = AgentSide(BaseCommPoint.get_ip(), 49054, verbose=True)

   while True:
       whattodo = agent.readWhatToDo()
       if whattodo is None:
           continue

       what, payload = whattodo
       if what == AgentSide.WhatToDo.RESET_SEND_OBS:
           agent.resetSendObs({"state": "reset"}, agenttime=0.0)
       elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
           agent.stepSendLastActDur(lat=0.1)
           agent.stepSendObs({"state": payload}, agenttime=0.1, rew=0.0)
       elif what == AgentSide.WhatToDo.FINISH:
           break

Timing values
-------------

- ``LAT`` is the duration of the previous action, measured by the agent clock.
- ``ATO`` is the agent-side time when an observation was sampled.
- ``t_wall`` is not produced by the library itself; it is typically captured by
  user code on the RL side with ``time.time()`` when the RL process receives a
  result.