# SPDX-License-Identifier: GPL-3.0-only

"""Smoke tests for the CoppeliaSim Burger example layout."""

import py_compile
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "burger_coppeliasim"


def test_burger_coppeliasim_entrypoints_compile():
    for filename in (
        "agent_side_burger_coppeliasim.py",
        "rl_side_burger_coppeliasim.py",
    ):
        py_compile.compile(str(EXAMPLE_DIR / filename), doraise=True)


def test_burger_coppeliasim_scene_is_available():
    scene_path = EXAMPLE_DIR / "scene" / "scene.ttt"

    assert scene_path.is_file()
    assert scene_path.stat().st_size > 0
