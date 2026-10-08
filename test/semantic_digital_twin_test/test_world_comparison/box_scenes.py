from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlacedBox:
    """
    A box-shaped body fixed to the root of a world.
    """

    name: str
    """
    The name of the body.
    """

    position: tuple[float, float, float]
    """
    Where the box's centre lies in the world frame, in metres.
    """

    size: tuple[float, float, float] = (0.3, 0.3, 0.3)
    """
    The edge lengths of the box, in metres.
    """

    origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    How far the body's origin lies from the box's centre, in metres.

    The box stays where :attr:`position` puts it; only the body frame moves.
    """

    def add_to(self, world: World) -> Body:
        """
        Add this box as a body fixed to the root of the world.

        :param world: The world to add the box to.
        :return: The body that was added.
        """
        x, y, z = self.position
        offset_x, offset_y, offset_z = self.origin_offset
        box = Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-offset_x, y=-offset_y, z=-offset_z
            ),
            scale=Scale(*self.size),
        )
        body = Body(
            name=PrefixedName(self.name),
            visual=ShapeCollection([box]),
            collision=ShapeCollection([box]),
        )
        with world.modify_world():
            world.add_body(body)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=x + offset_x, y=y + offset_y, z=z + offset_z
                    ),
                )
            )
        return body


@dataclass
class BoxScene:
    """
    A world made only of boxes fixed to its root.
    """

    boxes: list[PlacedBox] = field(default_factory=list)
    """
    The boxes the world holds.
    """

    def create_world(self) -> World:
        """
        :return: A new world holding every box of this scene.
        """
        world = World.create_with_root_body("map")
        for box in self.boxes:
            box.add_to(world)
        return world
