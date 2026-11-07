Quick Start
===========

Two processes take part. The RL process decides when to reset and which action to apply
next; the agent process applies the actions, samples observations and, if it wants,
computes a reward. Start the RL process first, because it is the one that listens.

The RL side:

.. code-block:: python

   from spindecoupler import RLSide

   rl = RLSide(port=49054, verbose=True)
   obs0, ato0 = rl.resetGetObs()

   action = {"move": "left"}
   lat, obs1, rew1, ato1 = rl.stepSendActGetObs(action)
   rl.stepExpFinished()

The agent side:

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

``LAT`` is the duration of the previous action, and ``ATO`` is the time at which an
observation was sampled; the agent's clock measures both. ``t_wall`` is not produced by
the library: RL code usually reads it with ``time.time()`` when a result arrives, and it
belongs to the RL process's clock.

The repository has complete, runnable examples in ``examples/README.md``.
