"""
The demonstration's world, exported once for every test that reads the scene.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.robosimstudio_tiago_demo.scene import PhysicsScene
from experiments.robosimstudio_tiago_demo.world import TiagoApartment

# %% the exported scene


@pytest.fixture(scope="session")
def tiago_apartment() -> TiagoApartment:
    """
    The TIAGo in the apartment, built from the descriptions the environment provides.
    """
    return TiagoApartment.build()


@pytest.fixture(scope="session")
def scene_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Where the scene of :func:`tiago_apartment` is exported to.
    """
    return tmp_path_factory.mktemp("robosimstudio_tiago_demo") / "scene.xml"


@pytest.fixture(scope="session")
def physics_scene(tiago_apartment: TiagoApartment, scene_file: Path) -> PhysicsScene:
    """
    The exported scene of :func:`tiago_apartment`, as written to :func:`scene_file`.
    """
    return PhysicsScene.export(tiago_apartment, scene_file)
