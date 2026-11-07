# Burger arena scene

The CoppeliaSim scene used by the Burger examples: a small arena with a TurtleBot3 Burger,
a goal marker, randomly placed cylindrical obstacles and an 8-sector LiDAR built from
three vision sensors. [`burger_coppeliasim`](../../burger_coppeliasim/README.md) drives it
directly to train a policy, and [`burger_real_sim`](../../burger_real_sim/README.md) puts a
ROS2 bridge in front of it so that it looks like a real robot.

Nothing inside the scene publishes ROS2 topics or decides when an episode ends. An outside
process resets and steps it through CoppeliaSim's ZMQ remote API, so it is the same scene
for both examples. It needs CoppeliaSim 4.x with that add-on, which is on by default at
port 23000.

CoppeliaSim runs the scene's Python scripts with the first `python3` on its `PATH`, and
they need `pyzmq` and `cbor2`, which the examples' requirements install. If you open the
scene yourself, start CoppeliaSim from a shell where your virtualenv is active, or the
scene fails with `script is not initialized`. The demo scripts do this when they launch
CoppeliaSim. You do not need to press Play: the examples load the scene and start the
simulation themselves.

## Files

```
scene.ttt        object tree, physics, three vision sensors, wrapper scripts
orchestrator.py  the public interface: reset, step, state
arena.py         the four walls
obstacles.py     random obstacles, kept clear of the robot and the target
target.py        the goal marker and its placement
burger.py        wheel control, odometry frame
laser.py         the 8-sector LiDAR
common.py        object paths, footprint radii, config helpers
```

## How the scripts hook into the scene

The Python files are not stored inside the `.ttt`. Each object has a short wrapper child
script that imports its module and forwards CoppeliaSim's callbacks, which is why the
files must stay next to `scene.ttt`. The wrapper also re-exports the functions that
others call with `sim.callScriptFunction`, because CoppeliaSim only finds functions
defined in the wrapper's own namespace.

| Object | Module | Functions the wrapper re-exports |
|---|---|---|
| `/Walls` | `arena` | `getExternalWallParams` |
| `/Obstacles` | `obstacles` | `rebuild` |
| `/Target` | `target` | `reset_pose` |
| `/Burger` | `burger` | `set_velocity`, `reset_pose`, `stop`, `get_odom_pose` |
| `/Burger/scan_joint/Laser` | `laser` | `get_rl_observation` |
| `/Orchestrator` | `orchestrator` | `reset_episode`, `step_episode`, `get_state`, `ros_reset`, `ros_step`, `get_ros_state`, `get_static_transforms` |

All wrappers look alike. This is the one of `/Target`:

```python
import target

def sysCall_init():
    sim = require('sim')
    target.init(sim)

def sysCall_actuation():
    target.actuation()

def sysCall_sensing():
    target.sensing()

def sysCall_cleanup():
    target.cleanup()

def reset_pose(x, y):
    return target.reset_pose(x, y)
```

To make another function callable from outside, re-export it in the wrapper inside
CoppeliaSim and save the scene.

## Interface of the orchestrator

Call a function with `sim.callScriptFunction(name, orchestrator_script, *args)`.

For training (`burger_coppeliasim`):

| Function | Returns |
|---|---|
| `reset_episode()` | observation after repositioning robot, target and obstacles |
| `step_episode([v, w])` | observation after applying the linear and angular velocity |
| `get_state()` | current observation |

The observation has 10 floats, `[distance_to_goal, relative_goal_angle, ray_1 ... ray_8]`,
with the rays in meters up to 8 m.

For the ROS2 bridge (`burger_real_sim`):

| Function | Returns |
|---|---|
| `ros_reset()` | state after stopping the robot and repositioning everything |
| `ros_step([v, w])` | state after applying the velocity, in one round trip |
| `get_ros_state()` | current state |
| `get_static_transforms()` | the fixed `base_link` to `base_scan` transform |

The state is a dict with `sim_time`, `lidar` (8 ranges), `robot_world_pos` and
`robot_world_quat`, `target_world_pos`, `odom_pos` and `odom_quat` (`base_link` in `odom`),
and `map_odom_pos` and `map_odom_quat` (`odom` in `map`).

The scene never ends an episode. Reaching the goal, a collision and the step budget are
decided by whoever drives it.

## Odometry

`burger.reset_pose()` moves a dummy object, `odom_dummy`, to the robot's new pose, and the
odometry is the robot's pose relative to that dummy. As on a real robot with a navigation
stack, the odometry starts from zero at every episode while the goal stays fixed in the
global `map` frame.

## LiDAR

Three perspective vision sensors (`sensor1`, `sensor2`, `sensor3`) under the `Laser` mount
cover 120 degrees each, 360 degrees in total. Their filter scripts extract 120 range points
per sensor. `laser.py` places every point in one of eight 45-degree sectors by its angle in
the robot frame and keeps the minimum range per sector. Using the angle in each sensor's
own frame would mix up the sectors, because the three sensors face different directions.

The transform from sensor to robot is applied with plain arithmetic on one matrix per
sensor. A `sim.*` call for each point would mean about 360 inter-process calls per step
and slow the simulation to about a third of real time. The far clipping distance is set to
`max_scan_distance` (8 m) when the script starts, which overrides the value saved in the
`.ttt`.

## Configuration

The size of the arena, the obstacles and the target are the `_SCHEMA` defaults in
`arena.py`, `obstacles.py` and `target.py`: 1.2 m by 2.4 m, and 8 obstacles of 12 cm. Edit
them there to change the scenario. Each module also has a helper to change them at runtime
(`setExternalWallParams`, `setObsParams`, `setTargetParams`); re-export it in the wrapper
first if you want to call it from outside.
