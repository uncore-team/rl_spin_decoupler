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
   tagged releases (v1.2.0); support for Linux, macOS, and Windows.
5. **Testing and continuous integration.** Unit, integration, state-machine, and
   example-smoke tests (56 tests) with 100% coverage of the library code
   (`spindecoupler.py`, `socketcomms/comms.py`), run in CI (GitHub Actions) on
   Linux, macOS, and Windows for Python 3.8-3.12, with a 95% coverage gate and
   Codecov reporting.
6. **Documentation.** Browsable Sphinx documentation website
   (https://uncore-team.github.io/rl_spin_decoupler/) with installation, quickstart,
   tutorial, how-it-works, and API reference pages.
7. **Runnable example suite.** Three self-contained, graded examples: closed-loop
   control of a first-order plant (pure software, no simulator), the classic Gymnasium
   LunarLander benchmark, and the complete sim-to-real navigation example of the paper
   (TurtleBot3 Burger, SAC, CoppeliaSim -> physical robot) with its telemetry/LAT
   analysis scripts. `[PENDIENTE: el ejemplo TurtleBot3 aun no esta en el repo]`
8. **Communication-layer robustness (v1.2.1).** The socket transport now uses
   length-prefixed framing with `sendall` and exact-byte receives before
   `pickle.loads`, fixing potential message truncation/desynchronization of the
   previous single-`recv` implementation for payloads larger than the chunk size or
   fragmented by the network.
