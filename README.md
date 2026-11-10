# RL Spin Decoupler

[![Version](https://img.shields.io/github/v/release/uncore-team/rl_spin_decoupler?label=version)](https://github.com/uncore-team/rl_spin_decoupler/releases)
[![License](https://img.shields.io/github/license/uncore-team/rl_spin_decoupler)](https://github.com/uncore-team/rl_spin_decoupler/blob/main/LICENSE)
[![CI](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/ci.yml/badge.svg)](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/ci.yml)
[![Docs](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/docs.yml/badge.svg)](https://github.com/uncore-team/rl_spin_decoupler/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/uncore-team/rl_spin_decoupler/branch/main/graph/badge.svg?token=ENURQ2GN3T)](https://codecov.io/gh/uncore-team/rl_spin_decoupler)

Documentation: https://uncore-team.github.io/rl_spin_decoupler/

RL Spin Decoupler synchronizes a reinforcement-learning algorithm (for instance one
from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)) with
an agent, real or simulated, that observes and acts in a different process.

Most of the time none of this is needed: the environment class can hold both the
learner's side and the agent. It becomes necessary when the agent has to run its own
spin loop, for example a controller that must command a robot at a fixed rate, and
that loop is hard to keep in step with the `reset()` and `step()` calls of the
learner. The library puts the two in separate processes and joins them with a TCP
socket. The RL process uses the `RLSide` class and the agent process uses
`AgentSide`.

It is also useful when timing matters. With every step the agent reports how long the
previous action really lasted (`LAT`) and when the observation was taken (`ATO`),
both measured with its own clock, so the learner can work with the actual timing
instead of assuming one.

The library was developed with Stable-Baselines3 in mind, but it only assumes a
`reset()` at the start of each episode and a `step()` for each RL step, with the usual
Gymnasium arguments and results, so it can be used with other RL libraries.

## Requirements

Python 3.8 or newer. The core library uses only the standard library. Stable-Baselines3
and Gymnasium are needed only by the training scripts that you write on top of it.

## Installation

```bash
git clone https://github.com/uncore-team/rl_spin_decoupler.git
cd rl_spin_decoupler
pip install -e .
```

Without git, download the ZIP from the GitHub page (Code, then Download ZIP) and extract
it. To install the optional RL packages listed in `requirements.txt`, run
`pip install -r requirements.txt`.

## Quick start

Start the RL side first, because it listens, and the agent second.

```python
# RL process
from spindecoupler import RLSide

rl = RLSide(port=49054, verbose=True)   # waits 60 s for the agent (accept_timeout)
obs, ato = rl.resetGetObs()
lat, obs, rew, ato = rl.stepSendActGetObs({"move": "left"})
rl.stepExpFinished()
```

```python
# agent process
from spindecoupler import AgentSide, BaseCommPoint

agent = AgentSide(BaseCommPoint.get_ip(), 49054, verbose=True)
while True:
    command = agent.readWhatToDo()      # None when the RL side sent nothing
    if command is None:
        continue
    what, payload = command
    if what == AgentSide.WhatToDo.RESET_SEND_OBS:
        agent.resetSendObs({"state": "reset"}, agenttime=0.0)
    elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
        agent.stepSendLastActDur(lat=0.1)
        agent.stepSendObs({"state": payload}, agenttime=0.1)
    elif what == AgentSide.WhatToDo.FINISH:
        break
```

[`skeleton_rl_side.py`](skeleton_rl_side.py) and
[`skeleton_agent_side.py`](skeleton_agent_side.py) are templates of a complete
integration: a Gym-style environment on the RL side, and a control loop with its state
machine on the agent side. The [tutorial](docs/tutorial.rst) walks through the same
cycle.

## Examples

The [examples](examples/README.md) run end to end, each with a script that starts
everything:

- [First-order plant](examples/first_order_plant_control/README.md): the smallest
  example, with no dependencies besides the package.
- [LunarLander](examples/lunar_lander/README.md) trained with Stable-Baselines3 PPO.
  [`lunar_lander_remote`](examples/lunar_lander_remote/README.md) and
  [`lunar_lander_container`](examples/lunar_lander_container/README.md) split it
  across two hosts, the second with the RL side in a GPU container.
- [CoppeliaSim Burger](examples/burger_coppeliasim/README.md): SAC training of a
  navigation policy for a simulated TurtleBot3, with the same two split variants
  ([remote](examples/burger_coppeliasim_remote/README.md),
  [container](examples/burger_coppeliasim_container/README.md)).
- [Real Burger](examples/burger_real/README.md): the trained policy driving a real
  robot through ROS2, and [the same agent against a simulated
  robot](examples/burger_real_sim/README.md), which needs no hardware.

In the LunarLander and Burger examples the agent only transports observations and
timing, while reward and the task-level end of the episode are computed on the RL side.

## Gymnasium environment

Each example wraps `RLSide` in a Gymnasium environment. `spindecoupler_gym.DecoupledEnv`
is that wrapper (it needs `pip install gymnasium`; the core module does not). A task
subclasses it and overrides `compute_reward()`, and optionally `parse_payload()`,
`is_terminated()` and `is_truncated()`. `step()` returns the timing values of the
protocol in `info`. The [LunarLander example](examples/lunar_lander/rl_side_lunarlander.py)
is a complete use.

## Benchmarks

[`benchmarks/`](benchmarks/README.md) has two scripts that reproduce the timing claims: a
protocol step takes 0.14 ms instead of 44 ms thanks to `TCP_NODELAY`, and a decoupled
agent keeps its control loop period while the learner takes up to 200 ms per decision,
where a coupled agent stalls for that long. See also the
[related work](docs/related-work.rst).

## Development

```bash
pip install -e ".[dev]"
pytest                                    # tests, with a 95 % coverage gate
python -m ruff check . && python -m ruff format --check .

pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

The documentation home page is then `docs/_build/html/index.html`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and the writing style, and
[CHANGELOG.md](CHANGELOG.md) for the release history.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
