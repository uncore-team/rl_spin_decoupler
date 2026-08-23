# SPDX-License-Identifier: GPL-3.0-only
"""Randomized cylindrical obstacles generator (CoppeliaSim child script).

Rebuilds a set of static cylinder obstacles inside the arena, either on a
jittered grid or fully at random, avoiding the robot's and target's current
positions. Reads the arena's *actual* size through
`arena.py::getExternalWallParams()` instead of a hardcoded arena size.
"""

import common

import math
import random
from typing import Any, Dict, List, Optional, Tuple

sim = None
self_handle = None

_SCHEMA: Dict[str, Any] = {
    "n_obstacles": {"default": 8, "minimum": 0, "maximum": 200, "type": "int"},
    "diam_obstacles": {"default": 0.12, "minimum": 0.01, "maximum": 1.0, "type": "float"},
    "height_obstacles": {"default": 0.25, "minimum": 0.01, "maximum": 2.0, "type": "float"},
    "flag_grid": {"default": True, "type": "bool"},
    "grid_visible": {"default": False, "type": "bool"},
    "quads_x": {"default": 2, "minimum": 1, "maximum": 8, "type": "int"},
    "quads_y": {"default": 2, "minimum": 1, "maximum": 8, "type": "int"},
    "grid_rows_per_quad": {"default": 5, "minimum": 1, "maximum": 10, "type": "int"},
    "grid_cols_per_quad": {"default": 5, "minimum": 1, "maximum": 10, "type": "int"},
}

# Extra clearance added on top of the two radii being compared, so shapes
# never end up touching even at the boundary of "valid".
SAFETY_MARGIN = 0.08

# Fallback arena size used only if arena.py's config can't be read (e.g. the
# wall object doesn't exist yet). Matches arena.py's own schema defaults.
_FALLBACK_ARENA_SIZE = (1.2, 2.4, 0.05)


def _defaults() -> Dict[str, Any]:
    return {key: spec["default"] for key, spec in _SCHEMA.items()}


def _with_defaults(user_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stored = common.read_config(sim, self_handle, _defaults())
    return common.merge_config(_defaults(), stored, user_cfg, _SCHEMA)


def _read_arena_size() -> Tuple[float, float, float]:
    """Return (scene_x_dim, scene_y_dim, scene_walls_thickness) from arena.py."""
    try:
        wall_obj = sim.getObject(common.SCENE_NAMES["walls"])
        wall_script = sim.getScript(sim.scripttype_childscript, wall_obj)
        wall_cfg = sim.callScriptFunction("getExternalWallParams", wall_script)
        return (
            float(wall_cfg["scene_x_dim"]),
            float(wall_cfg["scene_y_dim"]),
            float(wall_cfg["scene_walls_thickness"]),
        )
    except Exception:
        return _FALLBACK_ARENA_SIZE


def _collect_occupied_positions() -> List[Tuple[float, float, float]]:
    """Collect (x, y, safety_radius) for the robot and the target, if present."""
    occupied = []
    targets = [
        (common.SCENE_NAMES["robot"], common.ROBOT_RADIUS),
        (common.SCENE_NAMES["target"], common.TARGET_RADIUS),
    ]
    for name, radius in targets:
        try:
            handle = sim.getObject(name)
            pos = sim.getObjectPosition(handle, sim.handle_world)
            occupied.append((pos[0], pos[1], radius))
        except Exception:
            pass
    return occupied


def _is_position_valid(
    rx: float, ry: float, obs_radius: float,
    occupied: List[Tuple[float, float, float]], max_x: float, max_y: float,
) -> bool:
    if abs(rx) > max_x or abs(ry) > max_y:
        return False
    for ox, oy, oradius in occupied:
        if math.hypot(rx - ox, ry - oy) < (obs_radius + oradius + SAFETY_MARGIN):
            return False
    return True


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def rebuild(
    config: Optional[Dict[str, Any]] = None,
    positions: Optional[List[Tuple[float, float]]] = None,
) -> int:
    cfg = _with_defaults(config)
    common.remove_child_shapes(sim, self_handle)

    fx, fy, wall_thick = _read_arena_size()

    obs_radius = cfg["diam_obstacles"] / 2.0
    max_x = (fx / 2.0) - wall_thick - obs_radius - 0.02
    max_y = (fy / 2.0) - wall_thick - obs_radius - 0.02

    occupied = _collect_occupied_positions()
    cells: List[Tuple[float, float]] = []

    if cfg["flag_grid"]:
        pw, ph = fx / cfg["quads_x"], fy / cfg["quads_y"]
        step_x = pw / (cfg["grid_cols_per_quad"] + 1)
        step_y = ph / (cfg["grid_rows_per_quad"] + 1)

        x_coords = [
            -fx / 2.0 + ix * pw + dx * step_x
            for ix in range(cfg["quads_x"])
            for dx in range(1, cfg["grid_cols_per_quad"] + 1)
        ]
        y_coords = [
            -fy / 2.0 + iy * ph + dy * step_y
            for iy in range(cfg["quads_y"])
            for dy in range(1, cfg["grid_rows_per_quad"] + 1)
        ]
        cells = [(x, y) for x in x_coords for y in y_coords if abs(x) <= max_x and abs(y) <= max_y]

    def _place(rx: float, ry: float, idx: int) -> None:
        obs = sim.createPrimitiveShape(
            sim.primitiveshape_cylinder,
            [cfg["diam_obstacles"], cfg["diam_obstacles"], cfg["height_obstacles"]],
            0,
        )
        sim.setObjectPosition(obs, sim.handle_world, [rx, ry, cfg["height_obstacles"] / 2.0])
        sim.setObjectAlias(obs, f"Obstacle{idx}")
        sim.setObjectParent(obs, self_handle, True)
        sim.setObjectSpecialProperty(
            obs,
            sim.objectspecialproperty_collidable
            | sim.objectspecialproperty_measurable
            | sim.objectspecialproperty_detectable,
        )
        sim.setObjectInt32Param(obs, sim.shapeintparam_respondable, 1)
        sim.setObjectInt32Param(obs, sim.shapeintparam_static, 1)
        occupied.append((rx, ry, obs_radius))

    placed = 0
    to_place = cfg["n_obstacles"]

    if positions:
        for rx, ry in positions:
            placed += 1
            _place(rx, ry, placed)
    elif cfg["flag_grid"]:
        random.shuffle(cells)
        for rx, ry in cells:
            if placed >= to_place:
                break
            if _is_position_valid(rx, ry, obs_radius, occupied, max_x, max_y):
                placed += 1
                _place(rx, ry, placed)
    else:
        for _ in range(to_place):
            for _ in range(100):  # bounded retry per obstacle
                rx, ry = random.uniform(-max_x, max_x), random.uniform(-max_y, max_y)
                if _is_position_valid(rx, ry, obs_radius, occupied, max_x, max_y):
                    placed += 1
                    _place(rx, ry, placed)
                    break

    return self_handle


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def setObsParams(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update obstacle config and rebuild -- only if `params` is given.

    Calling with no arguments returns the current config without side
    effects (safe to use as a getter).
    """
    cur = _with_defaults(params)
    if params:
        common.write_config(sim, self_handle, cur)
        rebuild(cur)
    return cur


# ----------------------------------------------------------------------------
# CoppeliaSim lifecycle
# ----------------------------------------------------------------------------
def init(simulation: Any) -> None:
    global sim, self_handle

    sim = simulation
    self_handle = sim.getObject(".")

    common.write_schema(sim, self_handle, _SCHEMA)
    cfg = _with_defaults()
    common.write_config(sim, self_handle, cfg)
    rebuild(cfg)


def actuation() -> None:
    pass


def sensing() -> None:
    pass


def cleanup() -> None:
    pass
