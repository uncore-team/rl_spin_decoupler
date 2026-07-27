# RL Spin Decoupler

[![Version](https://img.shields.io/github/v/release/uncore-team/rl_spin_decoupler?label=version)](https://github.com/uncore-team/rl_spin_decoupler/releases)
[![License](https://img.shields.io/github/license/uncore-team/rl_spin_decoupler)](https://github.com/uncore-team/rl_spin_decoupler/blob/main/LICENSE)
[![CI](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/ci.yml/badge.svg)](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/ci.yml)
[![Docs](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/docs.yml/badge.svg)](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/docs.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/ci.yml)

Documentation: https://uncore-team.github.io/rl_spin_decoupler/

Example code: [tutorial](docs/tutorial.rst) | [RL skeleton](skeleton_rl_side.py) | [Agent skeleton](skeleton_agent_side.py)


This is a simple Python module that allows to sync an **RL algorithm** (e.g., from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)) to an **agent**, physical or simulated, that is able to get observations and execute actions.

Usually, this would not be necessary since in many cases you can implement in the same environment class all what is needed for both RL and the agent, but there are scenarios where the agent needs to execute some *spin loop*, in its own thread, while RL executes its own. Both loops may be quite difficult to put in sync. **RL Spin Decoupler** is intended for those cases.

Its use is pretty simple: you will have a Python program running RL and another one -a different process- running the agent. The former will use the `RLSide` class of this module, and the latter the `AgentSide` class in order to communicate to each other (communications are implemented with [sockets](https://docs.python.org/3/library/socket.html)). You can find further explanations in the code about when and how to call the methods of these classes in order to sync both processes. 

In addition, two files called 'skeleton_...' contain incomplete implementations of a case of use of the decoupler.

Besides facilitating the link between different processes that carry out RL and agent simulation/control, this decoupler may be useful as well when **different timings are involved and matter** (for instance, if the agent must execute some action at a given time, while the RL algorithm has no notion of that).

Although **RL Spin Decoupler** has been implemented and tested with Stable Baselines3 in mind, it is quite general and could be used with other RL libraries: it assumes that there is a `step()` method at each RL step and a `reset()` method when an episode starts. The parameters and results of those methods are the same that Stable Baselines3 uses.

## Dependencies

The core library uses only Python standard library modules and does not require third-party packages.

External RL packages (for example Stable-Baselines3 or Gymnasium) are optional and only needed for user-side training scripts built on top of this library.

## Requirements

- Python 3.8 or newer.
- No mandatory third-party dependencies for the core communication library.

## Installation

Clone the repository and move into the project folder:

```bash
git clone https://github.com/uncore-team/rl_spin_decoupler.git
cd rl_spin_decoupler
```

Optional: if you want to install optional RL packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Editable install for development:

```bash
pip install -e .
```

Developer verification

Run the test suite with coverage:

```bash
pip install -e ".[dev]"
pytest
```

Build the documentation locally:

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

The generated documentation home page is `docs/_build/html/index.html`.

Run lint and formatting checks locally:

```bash
python -m ruff check .
python -m ruff format --check .
```

## Download from GitHub

If you are new to GitHub, you can get this library in two simple ways:

1. With `git` (recommended)

```bash
git clone https://github.com/uncore-team/rl_spin_decoupler.git
cd rl_spin_decoupler
```

2. Without `git` (download ZIP)

- Open the repository page in GitHub.
- Click the green `Code` button.
- Select `Download ZIP`.
- Extract the ZIP file to a local folder.

After downloading, open the project folder in your editor and start using `spindecoupler.py` and the skeleton files as a base.

## Quick Start

1. Implement your agent and RL wrappers from the provided skeleton files.
2. Start the RL side process first.
3. Start the agent side process second.
4. Run learning and close communications with `stepExpFinished()` when finished.

## Templates

These templates illustrate a typical decoupled setup: the RL process drives training at a lower frequency, while the agent process runs a faster control loop and exchanges observations/actions through the communication wrappers.

### RL Side Skeleton: Minimalist Gym-like wrapper using RL Spin Decoupler

This skeleton wraps the RL-facing communication API in a Gym-like environment, so you can plug your learning loop with minimal glue code.

~~~python
""" RL side skeleton. Orchestrates the learning process. """
# imports
...
import time
from typing import Optional

from spindecoupler import RLSide  # RL side comms wrapper


class RLEnv:
	"""Minimal Gym-like wrapper around the decoupled communication API."""

	def __init__(self, debug: bool = False):
		self._debug = debug
		self._commstoagent = RLSide(49054, verbose=debug)  # blocks until agent connects

	def resetGetObs(self):
		"""Paper-level API name kept explicitly in the skeleton."""
		obs0, t_agent = self._commstoagent.resetGetObs()
		return obs0, t_agent

	def stepSendActGetObs(self, action):
		"""Paper-level tuple: (o_{t+1}, t_agent, t_wall, LAT)"""
		lat, obs_next, _rew_from_agent, t_agent = self._commstoagent.stepSendActGetObs(action)
		t_wall = time.time()
		return obs_next, t_agent, t_wall, lat

	def stepExpFinished(self):
		"""Graceful end of experiment/socket lifecycle."""
		self._commstoagent.stepExpFinished()

	# -- Gym-like API --
	def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
		obs0, t_agent = self.resetGetObs()
		info = {"t_agent": t_agent}
		if self._debug:
			print(f"reset -> t_agent={t_agent}")
		return obs0, info

	def step(self, action):
		o_t1, t_agent, t_wall, lat = self.stepSendActGetObs(action)
		reward = self._compute_reward(o_t1, action, lat)
		terminated = self._is_terminal(o_t1)
		truncated = False
		info = {"t_agent": t_agent, "t_wall": t_wall, "lat": lat}
		return o_t1, reward, terminated, truncated, info

	# -- User hooks --
	def _compute_reward(self, obs, action, lat: float) -> float:
		return ...

	def _is_terminal(self, obs) -> bool:
		return ...


# -- Entry point --
if __name__ == "__main__":
	print("Learning...")
	env = RLEnv(debug=True)

	model = ...
	numstepsexp = 1_000
	model.learn(total_timesteps=numstepsexp)
	model.save(...)

	env.stepExpFinished()
~~~

### Agent Side Skeleton: Real-time control loop with asynchronous polling

This skeleton models a high-frequency control loop that polls RL commands asynchronously and reports observations at the configured RL timestep.

Use it as a starting point: keep the communication calls as shown, and customize the user hooks (`_compute_reward`, `_is_terminal`, `_build_observation`, `_apply_action`, `_reset_workspace`, `_null_action`) for your specific robot or simulator.

~~~python
""" Agent side skeleton. Manages the high-frequency spin loop. """
# imports
...
import time
from enum import Enum
from typing import Any

from spindecoupler import AgentSide, BaseCommPoint  # Agent side comms wrapper


class Agent:
	"""Robot/simulator control loop that talks to the RL process."""

	class StepState(Enum):
		"""States of the agent during its step() execution."""
		READYFORRLCOMMAND = 0  # Waiting for new RL command
		EXECUTINGLASTACTION = 1  # Executing the last received action
		AFTERRESET = 2  # Just reset; must send observation

	def __init__(self, debug: bool = False) -> None:
		self._debug = debug
		self._rltimestep = 0.1  # seconds between RL actions (must be > control timestep)
		self._control_timestep = 0.02  # seconds of local control loop
		if self._rltimestep <= self._control_timestep:
			raise ValueError("RL timestep must be > control timestep")

		self._stepstate = Agent.StepState.READYFORRLCOMMAND
		self._lastaction = self._null_action()
		self._lastactiont0 = time.time()
		self._commstoRL = AgentSide(BaseCommPoint.get_ip(), 49054, verbose=debug)

	def readWhatToDo(self):
		"""Paper-level API: poll for commands from RL side."""
		return self._commstoRL.readWhatToDo()

	def stepSendLastActDur(self, lat: float) -> None:
		"""Paper-level API: report actual execution time of previous action."""
		self._commstoRL.stepSendLastActDur(lat)

	def step(self) -> Any:
		"""Main agent tick: manages state transitions and communication."""
		now_wall = time.time()
		act = self._lastaction

		if self._stepstate == Agent.StepState.EXECUTINGLASTACTION:
			# Check if action duration threshold reached; send observation if so
			if now_wall - self._lastactiont0 >= self._rltimestep:
				observation = self._build_observation()
				self._commstoRL.stepSendObs(observation, agenttime=now_wall)
				self._stepstate = Agent.StepState.READYFORRLCOMMAND

		elif self._stepstate == Agent.StepState.READYFORRLCOMMAND:
			# Poll for new commands from RL
			whattodo = self.readWhatToDo()
			if whattodo is not None:
				what, payload = whattodo
				if what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
					# Receive action and report LAT of previous action
					lat = now_wall - self._lastactiont0
					self.stepSendLastActDur(lat)
					self._lastactiont0 = now_wall
					self._lastaction = payload
					self._stepstate = Agent.StepState.EXECUTINGLASTACTION

				elif what == AgentSide.WhatToDo.RESET_SEND_OBS:
					# Reset episode and prepare to send observation
					self._reset_workspace()
					act = self._null_action()
					self._lastactiont0 = now_wall
					self._stepstate = Agent.StepState.AFTERRESET

				elif what == AgentSide.WhatToDo.FINISH:
					raise RuntimeError("Experiment finished")

		elif self._stepstate == Agent.StepState.AFTERRESET:
			# Send observation after reset
			observation = self._build_observation()
			self._commstoRL.resetSendObs(observation, agenttime=now_wall)
			self._stepstate = Agent.StepState.READYFORRLCOMMAND

		self._apply_action(self._lastaction)
		return self._lastaction

	def spinloop(self) -> None:
		"""Typical control loop."""
		while True:
			self.step()
			time.sleep(self._control_timestep)

	# -- user hooks --
	def _build_observation(self):
		return ...  # dict-like observation

	def _apply_action(self, action: Any) -> None:
		...  # send action to low-level controller

	def _reset_workspace(self) -> None:
		...  # reset robot/simulation state

	def _null_action(self):
		return ...


# -- Entry point --
if __name__ == "__main__":
	agent = Agent(debug=True)
	agent.spinloop()
~~~

## Contributing

Contributions are welcome through pull requests. Please:

1. Create a feature branch from `main`.
2. Keep changes focused and documented.
3. Open a pull request with a clear description of the motivation and changes.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
