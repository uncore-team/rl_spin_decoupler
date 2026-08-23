# SPDX-License-Identifier: GPL-3.0-only
"""Shared helpers and constants for the CoppeliaSim Burger arena scripts.

Import this module from every config-driven child script (arena.py,
obstacles.py, target.py) and from orchestrator.py:

    import common

It is a plain Python module with no `sysCall_*` handlers of its own. It
exists so the config-schema boilerplate that used to be copy-pasted across
arena.py / obstacles.py / target.py, and the scene's object names / shared
physical constants, live in exactly one place instead of N slightly
different copies.

NOTE on how this module reaches the other scripts: this project links each
scene script as an external .py file (via CoppeliaSim's file-based /
"include" scripting, see the accompanying delivery notes) rather than
embedding code in the .ttt. That is what makes a plain `import scene_common`
resolve here. If your CoppeliaSim setup instead requires an explicit
`require`/`include` directive for Python child scripts, add it before this
import -- the functions below don't care how they were pulled in.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Canonical scene object paths.
#
# These are best-effort names inferred from the original scripts (they used
# to disagree with each other -- e.g. orchestrator.py referred to "/tu_robot"
# and "/diana" while obstacles.py referred to "/Burger" and "/Target").
# Verify these against your actual scene tree and edit ONLY here if they
# differ; every other script resolves objects through this dict instead of
# hardcoding a path of its own.
# ---------------------------------------------------------------------------
SCENE_NAMES: Dict[str, str] = {
    "robot": "/Burger",
    "target": "/Target",
    "walls": "/Walls",
    "obstacles": "/Obstacles",
    "laser": "/Burger/Laser",
    "orchestrator": "/Orchestrator",
}

# Physical footprint radii (meters), shared between obstacles.py (placement
# safety margins) and orchestrator.py (spawn-area bounds), so both agree on
# how much clearance the robot/target need without duplicating the numbers.
ROBOT_RADIUS = 0.25
TARGET_RADIUS = 0.20


def clamp(value: float, minimum: Optional[float], maximum: Optional[float]) -> float:
    """Clamp `value` into [minimum, maximum]; a None bound is ignored."""
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _coerce(default: Any, value: Any) -> Any:
    """Coerce `value` to the type of `default`.

    Two special cases the naive `type(default)(value)` gets wrong:
    - bool: `bool("false")` is True in plain Python, which would silently
      corrupt a boolean field coming from a JSON/string-based config
      source. The strings "false"/"0"/"" (case-insensitive) are treated as
      False explicitly.
    - list (e.g. an RGB color): only accepted if `value` is already a list
      of the same length as `default`; otherwise falls back to `default`
      rather than raising or silently truncating/padding.
    """
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "")
        return bool(value)

    if isinstance(default, list):
        if isinstance(value, list) and len(value) == len(default):
            return list(value)
        return default

    try:
        return type(default)(value)
    except (TypeError, ValueError):
        return default


def merge_config(
    defaults: Dict[str, Any],
    stored: Optional[Dict[str, Any]],
    user_cfg: Optional[Dict[str, Any]],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge `user_cfg` over `stored` over `defaults`, coercing and clamping.

    This is the common body behind every `_with_defaults()` helper that used
    to be duplicated across arena.py / obstacles.py / target.py.
    """
    stored = stored or {}
    user_cfg = user_cfg or {}
    schema = schema or {}
    merged: Dict[str, Any] = {}

    for key, default_value in defaults.items():
        raw = user_cfg.get(key, stored.get(key, default_value))
        value = _coerce(default_value, raw)
        if (
            key in schema
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            value = clamp(value, schema[key].get("minimum"), schema[key].get("maximum"))
        merged[key] = value

    return merged


def read_config(sim: Any, handle: int, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Read the `__config__` custom data, falling back to `defaults` if absent."""
    return sim.readCustomTableData(handle, "__config__") or dict(defaults)


def write_config(sim: Any, handle: int, config: Dict[str, Any]) -> None:
    sim.writeCustomTableData(handle, "__config__", config)


def write_schema(sim: Any, handle: int, schema: Dict[str, Any]) -> None:
    sim.writeCustomBufferData(handle, "__schema__", sim.packTable(schema))


def read_schema(sim: Any, handle: int) -> Dict[str, Any]:
    raw = sim.readCustomBufferData(handle, "__schema__")
    return sim.unpackTable(raw) if raw else {}


def remove_child_shapes(sim: Any, node: int) -> None:
    """Remove every shape object under `node` (used before rebuilding generated geometry).

    Filters by `sim.object_shape_type` consistently -- one of the scripts
    this replaces used `sim.handle_all` instead, which would also delete any
    non-shape children (dummies, sensors, ...) if that node ever grew one.
    """
    children = sim.getObjectsInTree(node, sim.object_shape_type, 1) or []
    valid = [child for child in children if child is not None and child > 0]
    if valid:
        sim.removeObjects(valid)
