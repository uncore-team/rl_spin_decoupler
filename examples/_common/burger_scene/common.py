# SPDX-License-Identifier: GPL-3.0-only
"""Shared constants and config helpers for the Burger arena scripts.

Imported as ``import common`` by every other script of the scene. It is a plain
module with no ``sysCall_*`` handlers.

The scene is intentionally ROS2-agnostic: no script here imports ``simROS2`` or
decides when an episode ends. Everything is driven from outside through the
ZMQ remote API (see ``orchestrator.py`` for the public surface), so the very
same scene serves the RL training example (``examples/burger_coppeliasim``) and
the ROS2 bridge example (``examples/burger_real_sim``).

Each object's child script inside ``scene.ttt`` is a tiny wrapper that does
``import <module>`` and forwards the ``sysCall_*`` hooks, so the ``.py`` files
must stay next to the ``.ttt`` (see README.md for the wrapper contract).
"""

from __future__ import annotations

from typing import Any

# Scene object paths. CoppeliaSim resolves a partial alias path, so
# "/Burger/Laser" also matches "/Burger/scan_joint/Laser". Every script looks
# objects up through this dict; edit it only here if the scene tree changes.
SCENE_NAMES: dict[str, str] = {
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


def clamp(value: float, minimum: float | None, maximum: float | None) -> float:
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
    defaults: dict[str, Any],
    stored: dict[str, Any] | None,
    user_cfg: dict[str, Any] | None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge `user_cfg` over `stored` over `defaults`, coercing and clamping.

    This is the common body behind every `_with_defaults()` helper that used
    to be duplicated across arena.py / obstacles.py / target.py.
    """
    stored = stored or {}
    user_cfg = user_cfg or {}
    schema = schema or {}
    merged: dict[str, Any] = {}

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


def read_config(sim: Any, handle: int, defaults: dict[str, Any]) -> dict[str, Any]:
    """Read the `__config__` custom data, falling back to `defaults` if absent."""
    return sim.readCustomTableData(handle, "__config__") or dict(defaults)


def write_config(sim: Any, handle: int, config: dict[str, Any]) -> None:
    sim.writeCustomTableData(handle, "__config__", config)


def write_schema(sim: Any, handle: int, schema: dict[str, Any]) -> None:
    sim.writeCustomBufferData(handle, "__schema__", sim.packTable(schema))


def read_schema(sim: Any, handle: int) -> dict[str, Any]:
    raw = sim.readCustomBufferData(handle, "__schema__")
    return sim.unpackTable(raw) if raw else {}


def remove_child_shapes(sim: Any, node: int) -> None:
    """Remove every shape under `node` (used before rebuilding generated geometry).

    It filters by `sim.object_shape_type` so that dummies or sensors added under
    the node later are not deleted along with the generated shapes.
    """
    children = sim.getObjectsInTree(node, sim.object_shape_type, 1) or []
    valid = [child for child in children if child is not None and child > 0]
    if valid:
        sim.removeObjects(valid)
