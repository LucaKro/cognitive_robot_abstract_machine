from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Iterable, Optional, Self

import numpy as np
from probabilistic_model.probabilistic_circuit.rx.helper import uniform_measure_of_event
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from typing_extensions import List

from krrood.entity_query_language.entity import entity, let
from krrood.entity_query_language.quantify_entity import an
from .mixins import (
    HasBody,
    HasSupportingSurface,
    HasRegion,
    HasDrawers,
    HasDoors,
    HasHandle,
    HasCorpus,
)
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..reasoning.predicates import InsideOf
from ..spatial_types import Point3
from ..utils import Direction
from ..world_description.geometry import Scale
from ..world_description.shape_collection import BoundingBoxCollection
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
)


@dataclass(eq=False)
class IsPerceivable:
    """
    A mixin class for semantic annotations that can be perceived.
    """

    class_label: Optional[str] = field(default=None, kw_only=True)
    """
    The exact class label of the perceived object.
    """


@dataclass(eq=False)
class Handle(HasBody):

    @classmethod
    def create_with_new_body(
        cls, name: PrefixedName, scale: Scale, thickness: float
    ) -> Self:
        handle_event = cls._create_handle_geometry(scale=scale).as_composite_set()

        inner_box = cls._create_handle_geometry(
            scale=scale, thickness=thickness
        ).as_composite_set()

        handle_event -= inner_box

        handle = Body(name=name)
        collision = BoundingBoxCollection.from_event(handle, handle_event).as_shapes()
        handle.collision = collision
        handle.visual = collision

    @classmethod
    def _create_handle_geometry(
        cls, scale: Scale, thickness: float = 0.0
    ) -> SimpleEvent:
        """
        Create a box event representing the handle.

        :param scale: The scale of the handle.
        :param thickness: The thickness of the handle walls.
        """

        x_interval = closed(0, scale.x - thickness)
        y_interval = closed(
            -scale.y / 2 + thickness,
            scale.y / 2 - thickness,
        )

        z_interval = closed(-scale.z / 2, scale.z / 2)

        return SimpleEvent(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )


@dataclass(eq=False)
class Fridge(
    HasCorpus,
    HasDoors,
):
    """
    A semantic annotation representing a fridge that has a door and a body.
    """

    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Aperture(HasRegion):
    """
    A semantic annotation that represents an opening in a physical entity.
    An example is like a hole in a wall that can be used to enter a room.
    """


@dataclass(eq=False)
class Door(HasBody, HasHandle):
    """
    A door is a physical entity that has covers an opening, has a movable body and a handle.
    """

    @classmethod
    def create_with_new_body(
        cls,
        name: PrefixedName,
        scale: Scale,
    ) -> Self:
        door_event = scale.to_simple_event().as_composite_set()
        body = Body(name=name)
        bounding_box_collection = BoundingBoxCollection.from_event(body, door_event)
        collision = bounding_box_collection.as_shapes()
        body.collision = collision
        body.visual = collision

        return cls(body=body)


@dataclass(eq=False)
class DoubleDoor(SemanticAnnotation):
    left_door: Door
    right_door: Door


@dataclass(eq=False)
class Drawer(HasCorpus, HasHandle):

    @property
    def opening_direction(self) -> Direction:
        return Direction.Z


############################### subclasses to Furniture


@dataclass(eq=False)
class Furniture(SemanticAnnotation, ABC): ...


@dataclass(eq=False)
class Table(Furniture):
    """
    A semantic annotation that represents a table.
    """

    def points_on_table(self, amount: int = 100) -> List[Point3]:
        """
        Get points that are on the table.

        :amount: The number of points to return.
        :returns: A list of points that are on the table.
        """
        area_of_table = BoundingBoxCollection.from_shapes(self.body.collision)
        event = area_of_table.event
        p = uniform_measure_of_event(event)
        p = p.marginal(SpatialVariables.xy)
        samples = p.sample(amount)
        z_coordinate = np.full(
            (amount, 1), max([b.max_z for b in area_of_table]) + 0.01
        )
        samples = np.concatenate((samples, z_coordinate), axis=1)
        return [Point3(*s, reference_frame=self.body) for s in samples]


@dataclass(eq=False)
class Cabinet(HasCorpus, Furniture, HasDrawers, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Dresser(HasCorpus, Furniture, HasDrawers, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Cupboard(HasCorpus, Furniture, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Wardrobe(HasCorpus, Furniture, HasDrawers, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Floor(HasSupportingSurface):

    @classmethod
    def create_with_new_body(cls, name: PrefixedName, scale: Scale) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given scale.

        :param name: The name of the floor body.
        :param scale: The scale defining the floor polytope.
        """
        polytope = scale.to_bounding_box().get_points()
        return cls.create_with_new_body_fron_polytope(
            name=name, floor_polytope=polytope
        )

    @classmethod
    def create_with_new_body_fron_polytope(
        cls, name: PrefixedName, floor_polytope: List[Point3]
    ) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given list of Point3.

        :param name: The name of the floor body.
        :param floor_polytope: A list of 3D points defining the floor poly
        """
        room_body = Body.from_3d_points(name=name, points_3d=floor_polytope)
        return cls(body=room_body)


@dataclass(eq=False)
class Room(SemanticAnnotation):
    """
    A semantic annotation that represents a closed area with a specific purpose
    """

    floor: Floor
    """
    The room's floor.
    """


@dataclass(eq=False)
class Wall(HasBody):

    @classmethod
    def create_with_new_body(
        cls, name: PrefixedName, scale: Scale, *args, **kwargs
    ) -> Self:
        wall_body = Body(name=name)
        wall_event = cls._create_wall_event(scale)
        wall_collision = BoundingBoxCollection.from_event(
            wall_body, wall_event
        ).as_shapes()

        wall_body.collision = wall_collision
        wall_body.visual = wall_collision

        return cls(body=wall_body, *args, **kwargs)

    @property
    def doors(self) -> Iterable[Door]:
        door = let(Door, self._world.semantic_annotations)
        query = an(entity(door), InsideOf(self.body, door.entry_way.region)() > 0.1)
        return query.evaluate()

    @classmethod
    def _create_wall_event(cls, scale: Scale) -> Event:
        """
        Return a wall event created from its scale. The height origin is on the ground, not in the center of the wall.
        """
        x_interval = closed(-scale.x / 2, scale.x / 2)
        y_interval = closed(-scale.y / 2, scale.y / 2)
        z_interval = closed(0, scale.z)

        wall_event = SimpleEvent(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )
        return wall_event.as_composite_set()
