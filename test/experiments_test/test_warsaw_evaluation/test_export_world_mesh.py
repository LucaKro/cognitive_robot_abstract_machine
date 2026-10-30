"""
Writing a world as a scene a modelling tool can open.

The point of the scene is that a body can be clicked and identified, so what matters is
that every body arrives as its own node and that the node says both which object it is
and what the world decided the object is.
"""

from __future__ import annotations

from pathlib import Path

import trimesh
from semantic_digital_twin.adapters.world_mesh_exporter import WorldMeshExtractor
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.warsaw.evaluation.export_world_mesh import (
    named_by_class,
    write_world_mesh,
)
from experiments.warsaw.painting import Coloring

# %% a world with a named body and an unnamed one


def world_with_a_drawer_and_an_unnamed_body() -> World:
    """
    Build a drawer holding a handle, and one body the world names nothing.
    """
    world = World.create_with_root_body("root")
    bodies = {
        name: Body(
            name=PrefixedName(name),
            visual=ShapeCollection([Box(scale=Scale(0.4, 0.4, 0.2))]),
        )
        for name in ("drawer_1", "handle_1", "nobody_named_me")
    }
    with world.modify_world():
        for body in bodies.values():
            world.add_connection(FixedConnection(parent=world.root, child=body))
        handle = Handle(name=PrefixedName("a_handle"), root=bodies["handle_1"])
        world.add_semantic_annotation_recursively(handle)
        world.add_semantic_annotation_recursively(
            Drawer(name=PrefixedName("a_drawer"), root=bodies["drawer_1"])
        )
    return world


# %% what a node is called


def test_a_body_is_named_for_what_it_is_as_well_as_which_one_it_is():
    """
    A click in the outliner shows the node's name and nothing else, so the name has to
    carry both or the scene answers only half the question.
    """
    snapshot = WorldMeshExtractor().extract(world_with_a_drawer_and_an_unnamed_body())

    named = {body.name for body in named_by_class(snapshot).body_meshes}

    assert "drawer_1 [Drawer]" in named
    assert "handle_1 [Handle]" in named


def test_a_body_the_world_names_nothing_keeps_the_name_it_has():
    """
    Appending an empty class would say the body was named something, when what happened
    is that nothing named it.
    """
    snapshot = WorldMeshExtractor().extract(world_with_a_drawer_and_an_unnamed_body())

    named = {body.name for body in named_by_class(snapshot).body_meshes}

    assert "nobody_named_me" in named


def test_naming_the_bodies_leaves_the_world_alone():
    """
    The scene is a way of looking at a world, not a change to it.
    """
    world = world_with_a_drawer_and_an_unnamed_body()
    snapshot = WorldMeshExtractor().extract(world)

    named_by_class(snapshot)

    assert {str(body.name) for body in world.bodies} >= {"drawer_1", "handle_1"}


# %% what is written


def test_every_body_arrives_as_a_node_of_its_own(tmp_path: Path):
    """
    One node per body is what lets one be hidden to see what stands behind it.
    """
    written = write_world_mesh(
        world_with_a_drawer_and_an_unnamed_body(), tmp_path, Coloring.BY_CLASS
    )

    nodes = set(trimesh.load(written).graph.nodes)

    assert {"drawer_1 [Drawer]", "handle_1 [Handle]", "nobody_named_me"} <= nodes


def test_the_scene_is_written_beside_what_says_which_body_is_which(tmp_path: Path):
    """
    Clicking answers most questions; the manifest answers the rest without opening
    anything.
    """
    written = write_world_mesh(
        world_with_a_drawer_and_an_unnamed_body(), tmp_path, Coloring.UNCHANGED
    )

    assert written.exists()
    assert (tmp_path / "manifest.json").exists()
