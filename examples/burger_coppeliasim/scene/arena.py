# SPDX-License-Identifier: GPL-3.0-only
"""Arena walls generator (CoppeliaSim child script).

Builds/rebuilds the four-wall rectangular arena centered on the world
origin, and exposes its config (size, wall thickness/height, color) as a
schema-validated dict that other scripts (obstacles.py, orchestrator.py)
query through `getExternalWallParams()` instead of reading this object's
custom data directly.
"""

import common

from typing import Any, Dict, List, Optional

sim = None
self_handle = None

_SCHEMA: Dict[str, Any] = {
    "scene_x_dim": {"default": 1.2, "minimum": 0.5, "maximum": 100.0, "type": "float"},
    "scene_y_dim": {"default": 2.4, "minimum": 0.5, "maximum": 100.0, "type": "float"},
    "scene_walls_thickness": {"default": 0.05, "minimum": 0.05, "maximum": 1.0, "type": "float"},
    "scene_walls_height": {"default": 0.25, "minimum": 0.01, "maximum": 5.0, "type": "float"},
    "color": {"default": [1.0, 1.0, 1.0], "type": "color"},
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

    # Keep the /ExternalWall container centered at the world origin so the
    # walls stay symmetric around (0, 0) regardless of where it was dragged
    # in the scene hierarchy.
    sim.setObjectPosition(self_handle, sim.handle_world, [0.0, 0.0, 0.0])
    sim.setObjectOrientation(self_handle, sim.handle_world, [0.0, 0.0, 0.0])

    common.remove_child_shapes(sim, self_handle)

    x, y, t, h = cfg["scene_x_dim"], cfg["scene_y_dim"], cfg["scene_walls_thickness"], cfg["scene_walls_height"]
    col = cfg["color"]
    walls: List[int] = []

    # -Y / +Y walls
    p1 = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [x + t * 2.0, t, h], 0)
    sim.setObjectParent(p1, self_handle, False)
    sim.setObjectColor(p1, 0, sim.colorcomponent_ambient_diffuse, col)
    sim.setObjectPosition(p1, self_handle, [0.0, -y * 0.5 - t * 0.5, h * 0.5])
    walls.append(p1)

    p2 = sim.copyPasteObjects([p1], 0)[0]
    sim.setObjectParent(p2, self_handle, False)
    sim.setObjectColor(p2, 0, sim.colorcomponent_ambient_diffuse, col)
    sim.setObjectPosition(p2, self_handle, [0.0, y * 0.5 + t * 0.5, h * 0.5])
    walls.append(p2)

    # -X / +X walls
    p3 = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [t, y + t * 2.0, h], 0)
    sim.setObjectParent(p3, self_handle, False)
    sim.setObjectColor(p3, 0, sim.colorcomponent_ambient_diffuse, col)
    sim.setObjectPosition(p3, self_handle, [-x * 0.5 - t * 0.5, 0.0, h * 0.5])
    walls.append(p3)

    p4 = sim.copyPasteObjects([p3], 0)[0]
    sim.setObjectParent(p4, self_handle, False)
    sim.setObjectColor(p4, 0, sim.colorcomponent_ambient_diffuse, col)
    sim.setObjectPosition(p4, self_handle, [x * 0.5 + t * 0.5, 0.0, h * 0.5])
    walls.append(p4)

    grp = sim.groupShapes(walls)
    sim.setObjectParent(grp, self_handle, True)
    sim.setObjectAlias(grp, "WallsGroup")

    prop = (sim.getObjectProperty(grp) | sim.objectproperty_ignoreviewfitting) & (~sim.objectproperty_selectable)
    sim.setObjectProperty(grp, prop)
    sim.setObjectSpecialProperty(
        grp,
        sim.objectspecialproperty_collidable
        | sim.objectspecialproperty_measurable
        | sim.objectspecialproperty_detectable,
    )
    sim.setObjectInt32Param(grp, sim.shapeintparam_respondable, 1)
    sim.setObjectInt32Param(grp, sim.shapeintparam_static, 1)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def getExternalWallParams() -> Dict[str, Any]:
    """Read-only getter: current wall config. Never rebuilds the geometry."""
    return common.read_config(sim, self_handle, _defaults())


def setExternalWallParams(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update wall config and rebuild the geometry -- but only if `params` is given.

    Calling this with no arguments used to still trigger a full wall
    teardown/rebuild, which made it unusable as a safe read-only getter (see
    getExternalWallParams above for that use case instead).
    """
    cur = _with_defaults(params)
    common.write_config(sim, self_handle, cur)
    if params:
        rebuild(cur)
    return cur


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
