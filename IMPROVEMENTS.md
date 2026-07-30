# Improvements and extensions over the JA2026 preliminary version

A preliminary version of the method underlying this package, together with an early validation on a robotic manipulator, was presented at the Jornadas de Automática 2026 (Bañuls-Arias et al., 2026). The released software package (v1.2.1) extends that work with:

1. **Consolidated two-sided architecture.** Stable `RLSide` (server) / `AgentSide`
   (client) split over TCP sockets, with a documented client state machine
   (`RESET_SEND_OBS`, `REC_ACTION_SEND_OBS`, `FINISH`) and non-blocking polling via `select`.
2. **Public, documented API with temporal instrumentation.** First-class reporting of
   the Last Action Time (LAT) and Agent Time of Observation (ATO) in the communication
   protocol, enabling time-aware (augmented) MDP formulations.
3. **Pure-stdlib core.** No mandatory third-party dependencies; Stable-Baselines3 and
   Gymnasium are optional, user-side dependencies only.
4. **Packaging and distribution.** `pyproject.toml`-based installation (Python >=3.8);
   tagged releases (v1.2.1); support for Linux, macOS, and Windows.
5. **Testing and continuous integration.** Unit, integration, state-machine, and
   example-smoke tests (69 tests) with 100% coverage of the library code
   (`spindecoupler.py`, `socketcomms/comms.py`), run in CI (GitHub Actions) on
   Linux, macOS, and Windows for Python 3.8-3.12, with a 95% coverage gate and
   Codecov reporting.
6. **Documentation.** Browsable Sphinx documentation website
   (https://uncore-team.github.io/rl_spin_decoupler/) with installation, quickstart,
   tutorial, how-it-works, and API reference pages.
7. **Runnable example suite.** Two self-contained, graded examples wired end-to-end
   through the decoupler with per-step LAT/ATO telemetry: closed-loop control of a
   first-order plant (pure software, no simulator), and the classic Gymnasium
   LunarLander benchmark trained with Stable-Baselines3 (PPO), where the agent only
   transports observations/timing while reward and episode termination are computed
   on the RL side.
8. **Communication-layer robustness (v1.2.1).** The socket transport now uses
   length-prefixed framing with `sendall` and exact-byte receives before
   `pickle.loads`, fixing potential message truncation/desynchronization of the
   previous single-`recv` implementation for payloads larger than the chunk size or
   fragmented by the network.
