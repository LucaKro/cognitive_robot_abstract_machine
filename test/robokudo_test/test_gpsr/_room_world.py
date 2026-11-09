"""
A world with two rooms side by side, standing in for the world the knowledge side
shares.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Room,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% two rooms


@dataclass
class TwoRoomWorld:
    """
    A kitchen covering x from 0 to 4 m and an office covering x from 4 to 7 m, both
    reaching from y = 0 to y = 3 m in the world frame.
    """

    kitchen_name: str = "iai_kitchen"
    """
    Name of the kitchen.
    """

    office_name: str = "office"
    """
    Name of the office.
    """

    def build(self) -> World:
        """
        :return: A world holding the two rooms.
        """
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
        with world.modify_world():
            rooms = [
                self._room(world, self.kitchen_name, minimum_x=0.0, maximum_x=4.0),
                self._room(world, self.office_name, minimum_x=4.0, maximum_x=7.0),
            ]
            world.add_semantic_annotations(rooms)
        return world

    @staticmethod
    def _room(world: World, name: str, minimum_x: float, maximum_x: float) -> Room:
        """
        :param world: The world that receives the room.
        :param name: The name of the room.
        :param minimum_x: Where the floor starts along x, in meters.
        :param maximum_x: Where the floor ends along x, in meters.
        :return: A room whose floor reaches from *minimum_x* to *maximum_x* and from y =
            0 to y = 3 m.
        """
        width = maximum_x - minimum_x
        floor = Floor.create_with_new_body_from_polytope_in_world(
            name=f"{name}_floor",
            world=world,
            floor_polytope=[
                Point3(0, 0, 0),
                Point3(0, 3, 0),
                Point3(width, 3, 0),
                Point3(width, 0, 0),
            ],
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=minimum_x + width / 2.0, y=1.5
            ),
        )
        return Room(floor=floor, name=PrefixedName(name))
