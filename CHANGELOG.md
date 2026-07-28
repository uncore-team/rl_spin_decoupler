# Changelog

All notable changes to this repository are documented in this file.

## [1.2.1] - 2026-07-28
### Added
- New end-to-end examples under `examples/`, including `first_order_plant_control` and `lunar_lander`, with runnable RL-side and agent-side scripts.
- API documentation pages under `docs/` (including `docs/api.rst`) and generated HTML documentation in `docs/_build/html/`.

### Changed
- Bumped package version to `1.2.1` in `pyproject.toml`.
- Updated exported module `__version__` to resolve dynamically from installed package metadata.
- In `socketcomms/comms.py`, switched to length-prefixed framing with `sendall` and improved receive robustness by reading header and payload with exact-byte semantics before `pickle.loads`.

## [1.2.0] - 2026-05-21
### Added
- Installation, requirements, download, and quick-start documentation in `README.md`.
- Empty `requirements.txt` file to prepare optional user-side dependencies.

### Changed
- Updated the module semantic version to `v1.2.0`.
- Refined the `skeleton_rl_side.py` and `skeleton_agent_side.py` examples to better reflect the recommended integration flow.
- Minor adjustment in `__init__.py` for package exports.

## [1.1.0] - 2025-03-28
### Added
- Added communication of the agent timestamp with each observation so the RL side can work with the agent's actual timing.

### Changed
- Corrected the previous semantic version to consolidate the `1.1.x` line.
- Improved the agent skeleton with a more polished usage example.
- Small documentation and communication-message adjustments in `socketcomms`.

## [1.0.0] - 2025-03-10
### Added
- Persistent connection between the RL and agent processes instead of short-lived exchanges.
- Additional data exchange between both sides, including actual duration of the previous action and optional reward from the agent side.
- Example skeletons in `skeleton_rl_side.py` and `skeleton_agent_side.py` to ease integration.

### Changed
- Reversed the operation sequence to "send action and then receive observation", which is the correct RL loop semantics.
- `readWhatToDo()` became non-blocking when no pending commands are available.
- Adjusted timeouts and communication implementation details for the new workflow.
- Renamed and refined the main module text to reflect the final RL-centered approach.

## [0.0.1] - 2025-02-07
### Added
- Initial release of the `spindecoupler.py` module to decouple the RL loop from the agent loop.
- `socketcomms` submodule with socket-based client/server communication primitives.
- Initial project and communication-submodule documentation.

### Changed
- Early adjustment in `spindecoupler.py` after the initial upload.
- Small formatting fixes in the initial documentation.