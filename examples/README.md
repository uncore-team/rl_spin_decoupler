# Examples

Every example runs two real processes, an RL side and an agent side, joined by the
decoupler's socket. Most of them have a `run_demo.sh` that starts the whole stack and
stops it when the run is over.

| Example | What it shows |
|---|---|
| [`first_order_plant_control`](first_order_plant_control/README.md) | The smallest example: control of a first-order plant, with no extra dependencies. |
| [`lunar_lander`](lunar_lander/README.md) | Gymnasium LunarLander trained with Stable-Baselines3 PPO. The agent runs the physics, the RL side computes reward. |
| [`lunar_lander_remote`](lunar_lander_remote/README.md) | The same, with the RL side on another host. |
| [`lunar_lander_container`](lunar_lander_container/README.md) | The same, with the RL side in an NVIDIA container on another host. |
| [`burger_coppeliasim`](burger_coppeliasim/README.md) | SAC training of a navigation policy for a TurtleBot3 simulated in CoppeliaSim. |
| [`burger_coppeliasim_remote`](burger_coppeliasim_remote/README.md) | The same, with the RL side on another host. |
| [`burger_coppeliasim_container`](burger_coppeliasim_container/README.md) | The same, with the RL side in a container. |
| [`burger_real`](burger_real/README.md) | The trained policy driving a real TurtleBot3 through ROS2. |
| [`burger_real_sim`](burger_real_sim/README.md) | The agent and policy of `burger_real` against a simulated robot, with no hardware. |

`_common/burger_scene` holds the CoppeliaSim scene shared by the Burger examples.

## Quick run

From the repository root, install the package and the dependencies of the LunarLander
example, then run the demo. It trains for a few minutes and stops by itself:

```bash
pip install -e .
pip install -r examples/lunar_lander/requirements.txt
./examples/lunar_lander/run_demo.sh
```

To start the two processes by hand, RL side first:

```bash
python examples/lunar_lander/rl_side_lunarlander.py
python examples/lunar_lander/agent_side_lunarlander.py --render
```

## What each family needs

The LunarLander examples need only Python packages. The Burger examples that use
CoppeliaSim also need CoppeliaSim Edu 4.x installed, and `burger_real` and
`burger_real_sim` need ROS2 Humble. The split variants have a `run_local_demo.sh` that
runs both halves on one machine, which is also the quickest way to check that your
setup works before using two hosts.

The scripts that start CoppeliaSim assume it is in `/opt/CoppeliaSim_Edu`; set
`COPPELIASIM_SH` if it is elsewhere.
