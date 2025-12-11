from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import reduce
from operator import or_

from random_events.product_algebra import *
from typing_extensions import Type

from krrood.entity_query_language.entity import (
    let,
    entity,
    not_,
    in_,
    for_all,
)
from krrood.entity_query_language.quantify_entity import an
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import (
    TransformationMatrix,
    Vector3,
    Point3,
)
from ..world import World
from ..world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from ..world_description.degree_of_freedom import DegreeOfFreedom
from ..world_description.geometry import Scale
from ..world_description.shape_collection import BoundingBoxCollection, ShapeCollection
from ..world_description.world_entity import (
    Body,
    Region,
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from ..semantic_annotations.semantic_annotations import (
        Handle,
        Dresser,
        Drawer,
        Door,
        Wall,
        DoubleDoor,
        Room,
        Floor,
    )


@dataclass
class HasDoorLikeFactories(ABC):
    """
    Mixin for factories receiving multiple DoorLikeFactories.
    """

    door_like_factory_configs: List[DoorLikeConfigForParentFactory] = field(
        default_factory=list, hash=False
    )
    """
    The door factories used to create the doors.
    """

    def add_doorlike_semantic_annotation_to_world(
        self,
        parent_world: World,
    ):
        """
        Adds door-like semantic annotations to the parent world.
        """
        for config in self.door_like_factory_configs:
            match config:
                case DoorConfigForParentFactory():
                    self._add_door_to_world(
                        door_factory=config.factory_instance,
                        parent_T_door=config.parent_T_child,
                        opening_axis=config.hinge_axis,
                        parent_world=parent_world,
                    )
                case DoubleDoorConfigForParentFactory():
                    self._add_double_door_to_world(
                        door_factory=config.factory_instance,
                        parent_T_double_door=config.parent_T_child,
                        parent_world=parent_world,
                    )

    def _add_double_door_to_world(
        self,
        door_factory: DoubleDoorFactory,
        parent_T_double_door: TransformationMatrix,
        parent_world: World,
    ):
        """
        Adds a double door to the parent world by extracting the doors from the double door world, moving the
        relevant bodies and semantic annotations to the parent world.
        :param door_factory: The factory used to create the double door.
        :param parent_T_double_door: The transformation matrix defining the double door's position and orientation relative
        to the parent world.
        :param parent_world: The world to which the double door will be added.
        """
        parent_root = parent_world.root
        double_door_world = door_factory.create()
        double_door = double_door_world.get_semantic_annotations_by_type(DoubleDoor)[0]
        doors = [double_door.left_door, double_door.right_door]
        new_worlds = []
        new_connections = []
        new_dofs = []
        for door in doors:
            new_world, new_connection, new_dof = self._move_door_into_new_world(
                parent_root, door, parent_T_double_door
            )
            new_worlds.append(new_world)
            new_connections.append(new_connection)
            new_dofs.append(new_dof)

        with parent_world.modify_world():

            with double_door_world.modify_world():

                for new_door_world, new_parent_C_left, new_dof in zip(
                    new_worlds, new_connections, new_dofs
                ):
                    parent_world.add_degree_of_freedom(new_dof)
                    parent_world.merge_world(new_door_world, new_parent_C_left)

                double_door_world.remove_semantic_annotation(double_door)
                parent_world.add_semantic_annotation(double_door)

    @staticmethod
    def _move_door_into_new_world(
        new_parent: KinematicStructureEntity,
        door: Door,
        parent_T_double_door: TransformationMatrix,
    ):
        """
        Move a door from a double door world into a new world with a revolute connection.

        :param new_parent: Entity that will be the new parent of the door
        :param door: The door to be moved into a new world
        :param parent_T_double_door: Original transform from parent to double door
        :return:
        """
        double_door_world = door._world
        door_hinge_kse = door.body.parent_kinematic_structure_entity
        double_door_C_door: RevoluteConnection = door_hinge_kse.parent_connection
        double_door_T_door = double_door_C_door.parent_T_connection_expression
        parent_T_door = parent_T_double_door @ double_door_T_door
        old_dof = double_door_C_door.dof
        door_world = double_door_world.move_branch_to_new_world(door_hinge_kse)

        new_dof = DegreeOfFreedom(
            name=old_dof.name,
            lower_limits=old_dof.lower_limits,
            upper_limits=old_dof.upper_limits,
        )

        new_parent_C_left = RevoluteConnection(
            parent=new_parent,
            child=door_hinge_kse,
            parent_T_connection_expression=parent_T_door,
            multiplier=double_door_C_door.multiplier,
            offset=double_door_C_door.offset,
            axis=double_door_C_door.axis,
            dof_id=new_dof.id,
        )

        with double_door_world.modify_world(), door_world.modify_world():
            double_door_world.remove_semantic_annotation(door)
            door_world.add_semantic_annotation(door)

        return door_world, new_parent_C_left, new_dof

    def remove_doors_from_world(
        self, parent_world: World, wall_event_thickness: float = 0.1
    ):
        """
        Remove the door volumes from all bodies in the world that are not doors.

        :param parent_world: The world from which to remove the door volumes.
        :param wall_event_thickness: The thickness of the wall event used to create the door events.
        """
        doors: List[Door] = parent_world.get_semantic_annotations_by_type(Door)
        if not doors:
            return
        all_doors_event = self._build_all_doors_event_from_semantic_annotations(
            doors, wall_event_thickness
        )

        all_bodies_not_door = self._get_all_bodies_excluding_doors_from_world(
            parent_world
        )

        if not all_doors_event.is_empty():
            self._remove_doors_from_bodies(all_bodies_not_door, all_doors_event)

    @staticmethod
    def _get_all_bodies_excluding_doors_from_world(world: World) -> List[Body]:
        """
        Return all bodies in the world that are not part of any door semantic annotation.

        :param world: The world from which to get the bodies.
        :return: A list of bodies that are not part of any door semantic annotation.
        """
        all_doors = let(Door, domain=world.semantic_annotations)
        other_body = let(type_=Body, domain=world.bodies)
        door_bodies = all_doors.bodies
        bodies_without_excluded_bodies_query = an(
            entity(other_body, for_all(door_bodies, not_(in_(other_body, door_bodies))))
        )

        filtered_bodies = list(bodies_without_excluded_bodies_query.evaluate())
        return filtered_bodies

    def _build_all_doors_event_from_semantic_annotations(
        self, doors: List[Door], wall_event_thickness: float = 0.1
    ) -> Event:
        """
        Build a single event representing all doors by combining the events of each door.

        :param doors: The list of door semantic annotations to build the event from.
        :param wall_event_thickness: The thickness of the wall event used to create the door events.
        :return: An event representing all doors.
        """
        door_events = [
            self._build_single_door_event(door, wall_event_thickness) for door in doors
        ]
        if door_events:
            return reduce(or_, door_events)
        return Event()

    @staticmethod
    def _build_single_door_event(
        door: Door, wall_event_thickness: float = 0.1
    ) -> Event:
        """
        Build an event representing a single door by creating a bounding box event around the door's collision shapes

        :param door: The door semantic annotation to build the event from.
        :param wall_event_thickness: The thickness of the wall event used to create the door event.
        :return: An event representing the door.
        """
        door_event = door.body.collision.as_bounding_box_collection_in_frame(
            door._world.root
        ).event

        door_plane_spatial_variables = SpatialVariables.yz
        door_thickness_spatial_variable = SpatialVariables.x.value
        door_event = door_event.marginal(door_plane_spatial_variables)
        door_event.fill_missing_variables([door_thickness_spatial_variable])
        thickness_event = SimpleEvent(
            {
                door_thickness_spatial_variable: closed(
                    -wall_event_thickness / 2, wall_event_thickness / 2
                )
            }
        ).as_composite_set()
        thickness_event.fill_missing_variables(door_plane_spatial_variables)
        door_event = door_event & thickness_event

        return door_event

    def _remove_doors_from_bodies(self, bodies: List[Body], all_doors_event: Event):
        """
        Remove the door volumes from the given bodies by subtracting the all_doors_event from each body's collision event.

        :param bodies: The list of bodies from which to remove the door volumes.
        :param all_doors_event: The event representing all doors.
        """
        for body in bodies:
            self._remove_door_from_body(body, all_doors_event)

    @staticmethod
    def _remove_door_from_body(body: Body, all_doors_event: Event):
        """
        Remove the door volumes from the given body by subtracting the all_doors_event from the body's collision event.

        :param body: The body from which to remove the door volumes.
        :param all_doors_event: The event representing all doors.
        """
        root = body._world.root
        body_event = (
            body.collision.as_bounding_box_collection_in_frame(root).event
            - all_doors_event
        )
        new_collision = BoundingBoxCollection.from_event(root, body_event).as_shapes()
        body.collision = new_collision
        body.visual = new_collision


@dataclass
class DresserConfigForParentFactory(ConfigForParentFactory):
    """
    Configuration for creating a dresser from a parent factory.
    """

    factory_instance: DresserFactory


@dataclass
class DresserFactory(
    SemanticAnnotationFactory[Dresser, DresserConfigForParentFactory],
    HasDoorLikeFactories,
    HasDrawerFactories,
):
    """
    Factory for creating a dresser with drawers, and doors.
    """

    def _create(self, world: World) -> World:
        """
        Return a world with a dresser at its root. The dresser consists of a container, potentially drawers, and doors.
        Assumes that the number of drawers matches the number of drawer transforms.
        """

        dresser_world = self._make_dresser_world()
        dresser_world.name = world.name
        return self._make_interior(dresser_world)

    def _make_interior(self, world: World) -> World:
        """
        Create the interior of the dresser by subtracting the drawers and doors from the container, and filling  with
        the remaining space.

        :param world: The world containing the dresser body as its root.
        """
        dresser_body: Body = world.root
        container_event = dresser_body.collision.as_bounding_box_collection_at_origin(
            TransformationMatrix(reference_frame=dresser_body)
        ).event

        container_footprint = self._subtract_bodies_from_container_footprint(
            world, container_event
        )

        container_event = self._fill_container_body(
            container_footprint, container_event
        )

        collision_shapes = BoundingBoxCollection.from_event(
            dresser_body, container_event
        ).as_shapes()
        dresser_body.collision = collision_shapes
        dresser_body.visual = collision_shapes
        return world

    def _subtract_bodies_from_container_footprint(
        self, world: World, container_event: Event
    ) -> Event:
        """
        Subtract the bounding boxes of all bodies in the world from the container event,
        except for the dresser body itself. This creates a frontal footprint of the container

        :param world: The world containing the dresser body as its root.
        :param container_event: The event representing the container.

        :return: An event representing the footprint of the container after subtracting other bodies.
        """
        dresser_body = world.root

        container_footprint = container_event.marginal(SpatialVariables.yz)

        for body in world.bodies_with_enabled_collision:
            if body == dresser_body:
                continue
            body_footprint = body.collision.as_bounding_box_collection_at_origin(
                TransformationMatrix(reference_frame=dresser_body)
            ).event.marginal(SpatialVariables.yz)
            container_footprint -= body_footprint
            if container_footprint.is_empty():
                return Event()

        return container_footprint

    def _fill_container_body(
        self, container_footprint: Event, container_event: Event
    ) -> Event:
        """
        Expand container footprint into 3d space and fill the space of the resulting container body.

        :param container_footprint: The footprint of the container in the yz-plane.
        :param container_event: The event representing the container.

        :return: An event representing the container body with the footprint filled in the x-direction.
        """

        container_footprint.fill_missing_variables([SpatialVariables.x.value])

        depth_interval = container_event.bounding_box()[SpatialVariables.x.value]
        limiting_event = SimpleEvent(
            {SpatialVariables.x.value: depth_interval}
        ).as_composite_set()
        limiting_event.fill_missing_variables(SpatialVariables.yz)

        container_event |= container_footprint & limiting_event

        return container_event
