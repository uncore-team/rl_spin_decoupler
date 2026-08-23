# SPDX-License-Identifier: GPL-3.0-only
"""Target marker (CoppeliaSim child script).

Builds/rebuilds the flat disk that marks the goal position, exposes its
config via `setTargetParams()`, and exposes `reset_pose(x, y)` so
orchestrator.py can relocate it every episode without reaching into this
object's internals (no more `sim.setObjectPosition` calls from outside).
"""

import common

from typing import Any, Dict, Optional

sim = None
self_handle = None

# World-frame Z offset so the marker sits just above the floor, independent
# of its own (very thin) configured `height`.
GROUND_CLEARANCE = 0.05

_SCHEMA: Dict[str, Any] = {
    "outer_disk_rad": {"default": 0.05, "minimum": 0.005, "maximum": 1.0, "type": "float"},
    "height": {"default": 0.001, "minimum": 0.0001, "maximum": 0.01, "type": "float"},
}


def _defaults() -> Dict[str, Any]:
    return {key: spec["default"] for key, spec in _SCHEMA.items()}


def _with_defaults(user_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stored = common.read_config(sim, self_handle, _defaults())
    return common.merge_config(_defaults(), stored, user_cfg, _SCHEMA)


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def rebuild(config: Optional[Dict[str, Any]] = None) -> None:
    cfg = _with_defaults(config)
    common.remove_child_shapes(sim, self_handle)

    h, rad = cfg["height"], cfg["outer_disk_rad"]
    if rad <= 0:
        return

    disk = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [rad * 2.0, rad * 2.0, h], 0)
    sim.setObjectAlias(disk, "Outer_disk")
    sim.setObjectParent(disk, self_handle, False)
    sim.setObjectPosition(disk, self_handle, [0.0, 0.0, h / 2.0])
    sim.setObjectInt32Param(disk, sim.shapeintparam_respondable, 0)  # non-collidable, passive
    sim.setObjectColor(disk, 0, sim.colorcomponent_ambient_diffuse, [0.0, 0.0, 1.0])

    prop = (sim.getObjectProperty(disk) | sim.objectproperty_ignoreviewfitting) & (~sim.objectproperty_selectable)
    sim.setObjectProperty(disk, prop)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def setTargetParams(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cur = _with_defaults(params)
    if params:
        common.write_config(sim, self_handle, cur)
        rebuild(cur)
    return cur


def reset_pose(x: float, y: float) -> None:
    """Move the target marker to (x, y) in world coordinates for a new episode.

    Called by orchestrator.py, which decides *where* the target spawns; this
    function only knows *how* to place the marker (fixed ground clearance).
    """
    sim.setObjectPosition(self_handle, sim.handle_world, [float(x), float(y), GROUND_CLEARANCE])


# ----------------------------------------------------------------------------
# CoppeliaSim lifecycle
# ----------------------------------------------------------------------------
def init(simulation: Any) -> None:
    global sim, self_handle

    sim = simulation
    self_handle = sim.getObject("..")

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
