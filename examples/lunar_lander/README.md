# LunarLander with Stable-Baselines3

Gymnasium's `LunarLander-v3` is simulated in one process, the agent, and a
Stable-Baselines3 PPO policy is trained in another, the RL side. Here both run on the
same machine and talk through the `spindecoupler` socket:

```
 agent side                                 RL side
 LunarLander physics  -- observation -->    PPO and reward
                      <--    action   --    (Gymnasium env wrapper)
```

The agent (`agent_side_lunarlander.py`) applies the action chosen by the RL side, keeps
stepping the physics, and returns the observation with its timing. It computes no
reward. The RL side (`rl_side_lunarlander.py`) owns the policy and the learning signal;
to Stable-Baselines3 it is an ordinary Gymnasium environment whose `step()` and
`reset()` are answered by the agent over the socket.

The example is about the decoupling and not about solving LunarLander. The RL-side
reward in `reward.py` is an approximation written for the demonstration, not
Gymnasium's own.

Everything starts with one script:

```bash
./examples/lunar_lander/run_demo.sh
```

It trains for 1024 steps (about 3 minutes), prints the mean latency of a short rollout,
saves the model and stops.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r examples/lunar_lander/requirements.txt
```

`gymnasium[box2d]` builds the Box2D physics engine, which may need `swig`,
`build-essential` and `python3-dev` (on Ubuntu, `sudo apt install swig build-essential
python3-dev`). The demo trains on the CPU; for a GPU, install a PyTorch build that
matches your driver and set `DEVICE=cuda`.

## Running

```bash
./examples/lunar_lander/run_demo.sh
TIMESTEPS=20000 N_STEPS=2048 ./examples/lunar_lander/run_demo.sh
RENDER=1 ./examples/lunar_lander/run_demo.sh      # show the LunarLander window
```

The script starts the RL side, waits until it listens, and starts the agent. The output
of each is shown with a prefix and kept in a log directory whose path is printed at the
end. When training finishes the RL side saves the model and tells the agent to stop,
and the script exits. Ctrl+C stops it earlier.

### Options

| Variable | Default | Meaning |
|---|---|---|
| `TIMESTEPS` | `1024` | Training steps |
| `N_STEPS` | `256` | PPO rollout length. PPO always completes a rollout, so this is also the least a run executes |
| `ROLLOUT_STEPS` | `100` | Steps of a final rollout that reports the mean latency; `0` skips it |
| `DEVICE` | `cpu` | PyTorch device: `cpu`, `cuda` or `auto` |
| `RENDER` | `0` | `1` opens the LunarLander window and needs a display |
| `MODEL_OUT` | `examples/lunar_lander/ppo_lunar_lander_decoupled.zip` | Where the model is saved |
| `LOG_DIR` | new directory under `/tmp` | Per-component logs |
| `RL_PORT` | `49054` | TCP port of the RL side |

### Timeouts

| Variable | Default | Meaning |
|---|---|---|
| `RL_TIMEOUT` | `10` | Seconds the RL side waits for each agent reply |
| `AGENT_TIMEOUT` | `10` | Seconds the agent waits for a command that has started to arrive |
| `STARTUP_TIMEOUT` | `180` | Seconds the script waits for the RL side to listen (importing PyTorch takes most of it) |

While PPO updates its policy, which takes the RL side a few seconds, nothing is
exchanged. This does no harm: the agent keeps stepping the physics (a finished episode
stays frozen) and waits for the next command. One limit is fixed in the library: the RL
side waits 60 seconds for the agent after it starts listening, which is why the agent is
started right away.

If the agent restarts or the network drops, the connection is lost. The environment is
built with `reconnect=True`: the episode in progress is truncated, the RL side waits up to
60 seconds for the agent, and the agent stops its lander (null action) and reconnects.
Training then goes on with a new episode.

## What you should see

PPO logs from Stable-Baselines3 in the RL process, then
`[RL] training finished. Model saved to: ...` and
`[RL] rollout finished. mean LAT over N steps: ...`, then
`[AGENT] finish command received`.

LAT is the duration of the previous action, measured by the agent between two
consecutive actions. It is at least `rl_step_period` (0.08 s) and includes the time the
RL side takes to answer. The `info` dict returned by `step()` carries `lat`, `t_agent`
(agent time of the observation) and `t_wall` (RL-side time). They come from different
clocks, so compare them across hosts only if the clocks are synchronized.

## How it works

The observation has 8 floats: `x`, `y`, `vx`, `vy`, `angle`, `angular_velocity` and the
contact of each leg. The action is `Discrete(4)`: do nothing, left engine, main engine,
right engine.

Who decides that an episode is over:

- The agent reports crashes and out-of-bounds positions (Gymnasium's `terminated`) and
  the Gymnasium time limit (`truncated`) in each payload.
- The RL side decides that the lander has landed, from the observation
  (`is_terminated` in `reward.py`), and that the 1000-step budget is spent
  (`is_truncated`).

An episode ends when either side says so. When the agent reports a physics end it
freezes and waits for the reset; it never resets by itself.

The agent runs two nested loops on the wall clock. A fast one (`--control-period`,
10 ms) steps the physics with the current action. A slower one (`--rl-step-period`,
80 ms) decides when to send the next observation, so each RL action lasts about 8
physics steps. The RL side thus sees a delayed, lower-rate view of a system that keeps
running.

## Running the pieces by hand

In two terminals, RL side first because it listens:

```bash
python examples/lunar_lander/rl_side_lunarlander.py --device cpu
python examples/lunar_lander/agent_side_lunarlander.py [--render]
```

Both accept `--help`. The RL side trains for 20 000 steps by default, which is a
demonstration; solving LunarLander usually takes around a million steps and tuned
hyperparameters.

## Files

```
run_demo.sh                 starts everything
agent_side_lunarlander.py   LunarLander physics, serves the RL side
rl_side_lunarlander.py      Gymnasium wrapper and PPO training
reward.py                   reward, landing check and step budget
requirements.txt            Python dependencies
```

The same example split across two hosts is in
[`lunar_lander_remote`](../lunar_lander_remote/README.md) and
[`lunar_lander_container`](../lunar_lander_container/README.md).

## Notes

The socket transport uses `pickle`, so connect only endpoints you trust. `--seed` seeds
PPO and the first environment reset, and the agent increases the seed by one on every
reset.
