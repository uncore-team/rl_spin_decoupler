Related work
============

This page places the library among the tools that a reinforcement-learning (RL)
practitioner would otherwise combine. It is not a survey, and the references should be
checked against the final bibliography.

Environment interfaces and learning libraries
---------------------------------------------

Gymnasium [Towers2024]_ defines the ``reset()``/``step()`` interface that most RL
libraries consume, and Stable-Baselines3 [Raffin2021]_ provides the algorithms used in
the examples. Both assume that the environment lives in the learner's process and
advances only when ``step()`` is called. That fits a simulator that the learner owns,
and it breaks down when the environment is a robot, or a simulator with its own clock,
that must keep running while the learner computes.

RL Spin Decoupler keeps the Gymnasium interface on the learner side
(``spindecoupler_gym.DecoupledEnv``) and moves the environment to another process that
runs its own loop. The learner is not modified: any algorithm that accepts a Gymnasium
environment works.

Distributed RL frameworks
-------------------------

RLlib [Liang2018]_ separates policy evaluation from training and can serve a policy to
an external environment over the network. Those frameworks aim at scale (many
environment workers feeding one learner) and at throughput. This library has a narrower
goal: one learner and one agent whose clocks differ, with the timing of each step made
explicit. It has no dependencies besides the standard library, so it can be added to an
existing training script, or to a robot that cannot install a framework.

Robotics middleware
-------------------

Gym-gazebo [Zamora2016]_ and similar projects connect RL to robots through ROS. ROS
solves transport and message definition, and a learner on top of it still has to decide
when an action starts, how long it lasts, and which observation belongs to which action.
The ``burger_real`` and ``burger_real_sim`` examples show the library on top of ROS 2: the
ROS node is the agent, and the decoupler carries actions, observations and timing
between it and the learner.

Time in RL
----------

Real-time RL [Ramstedt2019]_ studies agents that must act while the environment keeps
changing, and delay-aware formulations [Chen2021]_ extend the Markov decision process
with the delay between observing and acting. The surveys of real-world RL [DulacArnold2021]_
list delays and asynchrony among the obstacles to deployment. The library does not
propose a new algorithm for them. It supplies the measurements such formulations need:
the real duration of every action (``LAT``) and the time at which each observation was
taken (``ATO``), both by the agent's clock, so the learner can build time-aware states
or analyze the timing afterwards.

Position
--------

In short, the library provides three things together: the Gymnasium-compatible learner
side, an agent side whose loop is never blocked by the learner, and the timing values
in the protocol. Each exists elsewhere in some form. The combination, in a package with
no mandatory dependencies, is what the examples and the benchmarks
(:doc:`benchmarks`) exercise.

References
----------

.. [Towers2024] M. Towers et al. Gymnasium: A standard interface for reinforcement
   learning environments. arXiv:2407.17032, 2024.
.. [Raffin2021] A. Raffin et al. Stable-Baselines3: Reliable reinforcement learning
   implementations. Journal of Machine Learning Research 22(268), 2021.
.. [Liang2018] E. Liang et al. RLlib: Abstractions for distributed reinforcement
   learning. ICML, 2018.
.. [Zamora2016] I. Zamora et al. Extending the OpenAI Gym for robotics: a toolkit for
   reinforcement learning using ROS and Gazebo. arXiv:1608.05742, 2016.
.. [Ramstedt2019] S. Ramstedt and C. Pal. Real-time reinforcement learning. NeurIPS,
   2019.
.. [Chen2021] B. Chen et al. Delay-aware model-based reinforcement learning for
   continuous control. Neurocomputing 450, 2021.
.. [DulacArnold2021] G. Dulac-Arnold et al. Challenges of real-world reinforcement
   learning: definitions, benchmarks and analysis. Machine Learning 110, 2021.
