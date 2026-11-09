# Improvements over the JA2026 preliminary version

A preliminary version of the method behind this package, with an early validation on a
robotic manipulator, was presented at the Jornadas de Automática 2026 (Bañuls-Arias et
al., 2026). The released package extends that work in the following ways.

1. Two-sided architecture. `RLSide` (server) and `AgentSide` (client) communicate over
   TCP, and the agent side follows a documented state machine (`RESET_SEND_OBS`,
   `REC_ACTION_SEND_OBS`, `FINISH`) with non-blocking polling based on `select`.

2. Timing in the protocol. The Last Action Time (LAT) and the Agent Time of Observation
   (ATO) travel with every step, which makes time-aware (augmented) MDP formulations
   possible.

3. No mandatory dependencies. The core uses only the standard library. Stable-Baselines3
   and Gymnasium are optional and needed only by user-side training scripts.

4. Packaging. Installation is configured in `pyproject.toml` (Python 3.8 or newer), with
   tagged releases, and the library is tested on Linux, macOS and Windows.

5. Tests and continuous integration. Unit, integration, state-machine and example tests
   cover 100 % of the library code (`spindecoupler.py` and `socketcomms/comms.py`). They
   run in GitHub Actions on Linux, macOS and Windows for Python 3.8 to 3.13, with a 95 %
   coverage gate and Codecov reporting.

6. Documentation. A Sphinx site (https://uncore-team.github.io/rl_spin_decoupler/) has
   installation, quick start, tutorial, how-it-works and API pages.

7. Examples. Each one runs end to end through the decoupler, reports LAT and ATO at every
   step, and has a script that starts the whole stack:
   - A first-order plant controlled by a fast agent loop: the smallest example, with no
     dependencies besides the package.
   - Gymnasium LunarLander trained with Stable-Baselines3 PPO, in which the agent only
     transports observations and timing while the RL side computes reward and episode
     termination, plus two split versions with the RL side on a remote host, directly or
     in an NVIDIA/CUDA container (this exercises the separate clock domains).
   - A simulated TurtleBot3 Burger in CoppeliaSim trained with SAC, with the same two
     split versions.
   - The trained policy driving a real Burger through ROS2, and the same agent run
     against the simulated robot through a ROS2 bridge, so that it can be tried without
     hardware.

8. Transport. Messages are length-prefixed, sent with `sendall` and received with
   exact-byte reads before `pickle.loads`, which fixes the truncation and
   desynchronization that the earlier single-`recv` implementation could produce with
   payloads larger than the chunk size or fragmented by the network. Both ends also set
   `TCP_NODELAY`: without it, each step paid a wait of about 40 ms (Nagle's algorithm
   against the receiver's delayed ACK), and the exchange of one step took 44 ms on
   loopback instead of 0.14 ms.

9. Gymnasium environment. `spindecoupler_gym.DecoupledEnv` is the wrapper that the
   examples repeated: a task only supplies the reward and the end of the episode. The
   LunarLander example uses it.

10. Benchmarks. Two scripts reproduce the timing claims: the cost of a step with and
    without `TCP_NODELAY`, and the regularity of the agent's control loop against the
    time the learner needs, coupled and decoupled.

11. Query channel. `RLSideQuery` and `AgentSideQuery` let the agent request the action of
    a trained policy for an observation, with reconnection, for deployment.

12. Transport hardening. Frames carry a header with a magic number and a protocol
    version, so a foreign peer or a mismatched release fails with a clear message. The
    frame size is bounded (`max_frame_size`), incoming pickles are restricted to an
    allowlist unless `allow_pickle=True`, a stuck peer cannot block a send forever
    (`send_timeout`), the server socket sets `SO_REUSEADDR` on POSIX, the wait for the
    agent is configurable (`accept_timeout`), and the `verbose` output goes through
    `logging`.
