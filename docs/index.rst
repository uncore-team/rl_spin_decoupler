RL Spin Decoupler
=================

RL Spin Decoupler is a small Python library that synchronizes a reinforcement-learning
loop with an agent loop that runs in another process and may follow a different clock.
The learner and the agent exchange commands, actions and observations over a TCP
connection, and the agent reports how long each action really lasted and when each
observation was taken.

These pages cover installation, a first example, a longer tutorial, how the protocol
works, and the API reference generated from the docstrings.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial
   how-it-works
   api
