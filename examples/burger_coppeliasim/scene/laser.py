# SPDX-License-Identifier: GPL-3.0-only
"""8-sector LiDAR-like proxy built from 3 vision sensors (CoppeliaSim child script).

Exposes `get_rl_observation() -> list[float]`: the minimum detected distance
per 45-degree sector (8 sectors covering 360 degrees), clamped to
`config["max_scan_distance"]`.
"""

import math
from typing import Any, Dict, List, Optional

sim = None
robot_handle: Optional[int] = None
vision_sensor_handles: List[int] = []

# Created in sysCall_init but not currently wired into any vision sensor's
# "entity to render" filter, so it has no effect on what the sensors detect
# yet. Left in place because it looks like an intentional, half-finished
# self-occlusion filter (it explicitly subtracts the robot's own body via
# handle_tree/robot_handle below). Wire it into
# `sim.setObjectInt32Param(sensor, sim.visionintparam_entity_to_render,
# collection_handle)` in rebuild() if that's the intent, or drop it.
collection_handle: Optional[int] = None

config: Dict[str, Any] = {
    "max_scan_distance": 1.0,
    "scanning_angle_deg": 360.0,
    "show_lines": [False, False, False],
    "show_points": False,
    "color_lines": [1.0, 0.0, 0.0],
    "color_points": [1.0, 0.0, 0.0],
}

# 8 sectors of 45 degrees each.
sectors = [1.0] * 8
lines: Optional[int] = None
points: Optional[int] = None
not_first_here: bool = False


# ----------------------------------------------------------------------------
# RL-facing API
# ----------------------------------------------------------------------------
def get_rl_observation() -> List[float]:
    """Public API for orchestrator.py: 8 floats, min distance per sector."""
    return sectors


def _process_sensor_data(sensor_handle: int, show_line: bool, show_pt: bool) -> float:
    global sectors
    min_dist_overall = math.inf
    _res, _, aux2 = sim.readVisionSensor(sensor_handle)

    if aux2 and len(aux2) >= 2:
        m1 = sim.getObjectMatrix(sensor_handle, sim.handle_world)
        width, height = int(aux2[0]), int(aux2[1])
        p_zero = sim.multiplyVector(m1, [0.0, 0.0, 0.0])

        for j in range(height):
            for i in range(width):
                w = 2 + 4 * (j * width + i)
                v1, v2, v3, v4 = aux2[w : w + 4]

                if v4 <= config["max_scan_distance"]:
                    p_world = sim.multiplyVector(m1, [v1, v2, v3])
                    if show_line and lines:
                        sim.addDrawingObjectItem(
                            lines, [p_zero[0], p_zero[1], p_zero[2], p_world[0], p_world[1], p_world[2]]
                        )
                    if show_pt and points:
                        sim.addDrawingObjectItem(points, p_world)

                    # Angle relative to the sensor's own local frame.
                    angle = math.atan2(v2, v1)
                    sector_idx = int((math.degrees(angle) + 180) / 45) % 8
                    sectors[sector_idx] = min(sectors[sector_idx], v4)

                    if v4 < min_dist_overall:
                        min_dist_overall = v4

    return min_dist_overall


# ----------------------------------------------------------------------------
# CoppeliaSim lifecycle
# ----------------------------------------------------------------------------
def rebuild() -> None:
    global lines, points
    if lines:
        sim.removeDrawingObject(lines)
    if points:
        sim.removeDrawingObject(points)

    dist = config["max_scan_distance"]
    angle = math.radians(config["scanning_angle_deg"]) / 3.0  # split across 3 sensors

    for s in vision_sensor_handles:
        sim.setObjectFloatParam(s, sim.visionfloatparam_far_clipping, dist)
        sim.setObjectFloatParam(s, sim.visionfloatparam_perspective_angle, angle)

    lines = sim.addDrawingObject(sim.drawing_lines, 1, 0, -1, 1000, config["color_lines"])
    points = sim.addDrawingObject(sim.drawing_points, 3, 0, -1, 1000, config["color_points"])


def init(simulation: Any) -> None:
    global sim, robot_handle, vision_sensor_handles, collection_handle
    sim = simulation

    robot_handle = sim.getObject("../../..")
    vision_sensor_handles = [sim.getObject(f"../sensor{i + 1}") for i in range(3)]

    collection_handle = sim.createCollection(0)
    sim.addItemToCollection(collection_handle, sim.handle_all, -1, 0)
    sim.addItemToCollection(collection_handle, sim.handle_tree, robot_handle, 1)

    rebuild()

def actuation() -> None:
    pass

def sensing() -> None:
    global sectors, not_first_here
    sectors = [config["max_scan_distance"]] * 8  # reset to max range every tick

    if not_first_here:
        if lines:
            sim.addDrawingObjectItem(lines, None)
        if points:
            sim.addDrawingObjectItem(points, None)

        for i, s in enumerate(vision_sensor_handles):
            _process_sensor_data(s, config["show_lines"][i], config["show_points"])

    not_first_here = True


def cleanup() -> None:
    if lines:
        sim.removeDrawingObject(lines)
    if points:
        sim.removeDrawingObject(points)
