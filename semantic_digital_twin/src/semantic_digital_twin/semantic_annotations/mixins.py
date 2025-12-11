from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from functools import reduce
from operator import or_
from typing import (
    List,
    Optional,
    Self,
    Iterable,
    Tuple,
    Union,
)

import numpy as np
import trimesh
from numpy._typing import NDArray
from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_simple_event,
)
from random_events.interval import closed, SimpleInterval, Bound
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous
from typing_extensions import TYPE_CHECKING, assert_never

from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..exceptions import InvalidAxisError
from ..spatial_types import Point3, TransformationMatrix, Vector3
from ..spatial_types.derivatives import DerivativeMap
from ..utils import Direction
from ..world import World
from ..world_description.connections import (
    RevoluteConnection,
    FixedConnection,
    PrismaticConnection,
)
from ..world_description.degree_of_freedom import DegreeOfFreedom
from ..world_description.geometry import Scale, BoundingBox
from ..world_description.shape_collection import BoundingBoxCollection
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
)

if TYPE_CHECKING:
    from .semantic_annotations import (
        Drawer,
        Door,
        Handle,
    )


class IntervalConstants:
    """
    Predefined intervals for semantic directions.
    """

    ZERO_DIRAC = (0, 0, Bound.CLOSED, Bound.CLOSED)
    ZERO_TO_ONE_THIRD = (0, 1 / 3, Bound.CLOSED, Bound.CLOSED)
    ONE_THIRD_TO_TWO_THIRD = (1 / 3, 2 / 3, Bound.OPEN, Bound.OPEN)
    HALF_DIRAC = (0.5, 0.5, Bound.CLOSED, Bound.CLOSED)
    TWO_THIRD_TO_ONE = (2 / 3, 1, Bound.CLOSED, Bound.CLOSED)
    ONE_DIRAC = (1, 1, Bound.CLOSED, Bound.CLOSED)


class SemanticDirection(Enum): ...


class HorizontalSemanticDirection(SimpleInterval, SemanticDirection):
    """
    Semantic directions for horizontal positioning.
    """

    FULLY_LEFT = IntervalConstants.ZERO_DIRAC
    LEFT = IntervalConstants.ZERO_TO_ONE_THIRD
    CENTER = IntervalConstants.ONE_THIRD_TO_TWO_THIRD
    FULLY_CENTER = IntervalConstants.HALF_DIRAC
    RIGHT = IntervalConstants.TWO_THIRD_TO_ONE
    FULLY_RIGHT = IntervalConstants.ONE_DIRAC


class VerticalSemanticDirection(SimpleInterval, SemanticDirection):
    """
    Semantic directions for vertical positioning.
    """

    FULLY_BOTTOM = IntervalConstants.ZERO_DIRAC
    BOTTOM = IntervalConstants.ZERO_TO_ONE_THIRD
    CENTER = IntervalConstants.ONE_THIRD_TO_TWO_THIRD
    FULLY_CENTER = IntervalConstants.HALF_DIRAC
    TOP = IntervalConstants.TWO_THIRD_TO_ONE
    FULLY_TOP = IntervalConstants.ONE_DIRAC


@dataclass
class SemanticPositionDescription:
    """
    Describes a position by mapping semantic concepts (RIGHT, CENTER, LEFT, TOP, BOTTOM) to instances of
    random_events.intervals.SimpleInterval, which are then used to "zoom" into specific regions of an event.
    Each DirectionInterval divides the original event into three parts, either vertically or horizontally, and zooms into
    one of them depending on which specific direction was chosen.
    The sequence of zooms is defined by the order of directions in the horizontal_direction_chain and
    vertical_direction_chain lists.
    Finally, we can sample aa 2d pose from the resulting event
    """

    horizontal_direction_chain: List[HorizontalSemanticDirection]
    """
    Describes the sequence of zooms in the horizontal direction (Y axis).
    """

    vertical_direction_chain: List[VerticalSemanticDirection]
    """
    Describes the sequence of zooms in the vertical direction (Z axis).
    """

    @staticmethod
    def _zoom_interval(base: SimpleInterval, target: SimpleInterval) -> SimpleInterval:
        """
        Zoom 'base' interval by the percentage interval 'target' (0..1),
        preserving the base's boundary styles.

        :param base: The base interval to be zoomed in.
        :param target: The target interval defining the zoom percentage (0..1).
        :return: A new SimpleInterval representing the zoomed-in interval.
        """
        span = base.upper - base.lower
        new_lower = base.lower + span * target.lower
        new_upper = base.lower + span * target.upper
        return SimpleInterval(new_lower, new_upper, base.left, base.right)

    def _apply_zoom(self, simple_event: SimpleEvent) -> SimpleEvent:
        """
        Apply zooms in order and return the resulting intervals.

        :param simple_event: The event to zoom in.
        :return: A SimpleEvent containing the resulting intervals after applying all zooms.
        """
        simple_events = [
            self._apply_zoom_in_one_direction(
                axis,
                assignment.simple_sets[0],
            )
            for axis, assignment in simple_event.items()
        ]

        if not simple_events:
            return SimpleEvent()

        return reduce(or_, simple_events)

    def _apply_zoom_in_one_direction(
        self, axis: Continuous, current_interval: SimpleInterval
    ) -> SimpleEvent:
        """
        Apply zooms in one direction (Y, horizontal or Z, vertical) in order and return the resulting interval.

        :param axis: The axis to zoom in (SpatialVariables.y or SpatialVariables.z).
        :param current_interval: The current interval to zoom in.
        :return: A SimpleEvent containing the resulting interval after applying all zooms in the specified direction.
        """
        if axis == SpatialVariables.y.value:
            directions = self.horizontal_direction_chain
        elif axis == SpatialVariables.z.value:
            directions = self.vertical_direction_chain
        else:
            raise NotImplementedError

        for step in directions:
            current_interval = self._zoom_interval(current_interval, step)

        return SimpleEvent({axis: current_interval})

    def sample_point_from_event(self, event: Event) -> Tuple[float, float]:
        """
        Sample a 2D point from the given event by applying the zooms defined in the semantic position description.

        :param event: The event to sample from.
        :return: A sampled 2D point as a tuple (y, z).
        """
        simple_event = self._apply_zoom(event.bounding_box())
        event_circuit = uniform_measure_of_simple_event(simple_event)
        return event_circuit.sample(amount=1)[0]


@dataclass(eq=False)
class HasBody(SemanticAnnotation, ABC):
    """
    Abstract base class for all household objects. Each semantic annotation refers to a single Body.
    Each subclass automatically derives a MatchRule from its own class name and
    the names of its HouseholdObject ancestors. This makes specialized subclasses
    naturally more specific than their bases.
    """

    body: Body = field(kw_only=True)

    @property
    def bodies(self) -> Iterable[Body]:
        return [self.body]

    @classmethod
    @abstractmethod
    def create_with_new_body(cls, *args, **kwargs) -> Self: ...


@dataclass(eq=False)
class HasRegion(SemanticAnnotation, ABC):
    """
    Abstract base class for all household objects. Each semantic annotation refers to a single Body.
    Each subclass automatically derives a MatchRule from its own class name and
    the names of its HouseholdObject ancestors. This makes specialized subclasses
    naturally more specific than their bases.
    """

    region: Region = field(kw_only=True)

    @classmethod
    @abstractmethod
    def create_with_new_region(cls, *args, **kwargs) -> Self: ...


@dataclass
class _HasActiveConnection(ABC):

    connection_T_child: TransformationMatrix

    def _create_door_upper_lower_limits(
        self, parent_T_hinge: TransformationMatrix, opening_axis: Vector3
    ) -> Tuple[DerivativeMap[float], DerivativeMap[float]]:
        """
        Return the upper and lower limits for the door's degree of freedom.

        :param parent_T_hinge: The transformation matrix defining the door's pivot point relative to the parent world.
        :param opening_axis: The axis around which the door opens.

        :return: The upper and lower limits for the door's degree of freedom.
        """

        # upper and lower limit need to be chosen based on the pivot point of the door
        match opening_axis.to_np().tolist():
            case [0, 1, 0, 0]:
                sign = np.sign(parent_T_hinge.to_position().to_np()[2])
                lower_limit_position, upper_limit_position = (
                    (-np.pi / 2, 0) if sign > 0 else (0, np.pi / 2)
                )
            case [0, 0, 1, 0]:
                sign = np.sign(parent_T_hinge.to_position().to_np()[1])
                lower_limit_position, upper_limit_position = (
                    (-np.pi / 2, 0) if sign < 0 else (0, np.pi / 2)
                )
            case _:
                raise InvalidAxisError(axis=opening_axis)

        lower_limits = DerivativeMap[float]()
        upper_limits = DerivativeMap[float]()
        lower_limits.position = lower_limit_position
        upper_limits.position = upper_limit_position

        return upper_limits, lower_limits

    @staticmethod
    def _create_drawer_upper_lower_limits(
        drawer_factory: DrawerFactory,
    ) -> Tuple[DerivativeMap[float], DerivativeMap[float]]:
        """
        Return the upper and lower limits for the drawer's degree of freedom.
        """
        lower_limits = DerivativeMap[float]()
        upper_limits = DerivativeMap[float]()
        lower_limits.position = 0.0
        upper_limits.position = (
            drawer_factory.container_factory_config.factory_instance.scale.x * 0.75
        )

        return upper_limits, lower_limits


@dataclass(eq=False)
class HasDrawers(_HasActiveConnection):
    """
    A mixin class for semantic annotations that have drawers.
    """

    drawers: List[Drawer] = field(default_factory=list, hash=False, kw_only=True)

    def _add_drawer_to_world(
        self,
        drawer_factory: DrawerFactory,
        parent_T_drawer: TransformationMatrix,
        parent_world: World,
    ):

        lower_limits, upper_limits = self._create_drawer_upper_lower_limits(
            drawer_factory
        )
        drawer_world = drawer_factory.create()
        parent_root = parent_world.root
        child_root = drawer_world.root

        parent_T_drawer.reference_frame = parent_root

        dof = DegreeOfFreedom(
            name=PrefixedName(
                f"{child_root.name.name}_connection", child_root.name.prefix
            ),
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
        with parent_world.modify_world():
            parent_world.add_degree_of_freedom(dof)
            connection = PrismaticConnection(
                parent=parent_root,
                child=child_root,
                parent_T_connection_expression=parent_T_drawer,
                multiplier=1.0,
                offset=0.0,
                axis=Vector3.X(),
                dof_id=dof.id,
            )

            parent_world.merge_world(drawer_world, connection)


@dataclass(eq=False)
class HasDoors(_HasActiveConnection):
    """
    A mixin class for semantic annotations that have doors.
    """

    doors: List[Door] = field(default_factory=list, hash=False, kw_only=True)

    def add_door(
        self,
        door_factory: DoorFactory,
        parent_T_door: TransformationMatrix,
        opening_axis: Vector3,
        parent_world: World,
    ):
        """
        Adds a door to the parent world using a new door hinge body with a revolute connection.

        :param door_factory: The factory used to create the door.
        :param parent_T_door: The transformation matrix defining the door's position and orientation relative
        to the parent world.
        :param parent_world: The world to which the door will be added.
        """
        with parent_world.modify_world():
            door_world, parent_T_hinge = self._add_hinge_to_door(
                door_factory, parent_T_door, opening_axis
            )

            upper_limits, lower_limits = self._create_door_upper_lower_limits(
                parent_T_hinge, opening_axis
            )

            root = door_world.root
            dof = DegreeOfFreedom(
                name=PrefixedName(f"{root.name.name}_connection", root.name.prefix),
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            )
            with parent_world.modify_world():
                parent_world.add_degree_of_freedom(dof)
                connection = RevoluteConnection(
                    parent=parent_world.root,
                    child=root,
                    parent_T_connection_expression=parent_T_hinge,
                    multiplier=1.0,
                    offset=0.0,
                    axis=opening_axis,
                    dof_id=dof.id,
                )

                parent_world.merge_world(door_world, connection)

    def _add_hinge_to_door(
        self,
        door_factory: DoorFactory,
        parent_T_door: TransformationMatrix,
        opening_axis: Vector3,
    ):
        """
        Adds a hinge to the door. The hinge's pivot point is on the opposite side of the handle.
        :param door_factory: The factory used to create the door.
        :param parent_T_door: The transformation matrix defining the door's position and orientation relative
        :param opening_axis: The axis around which the door opens.
        """
        door_world = door_factory.create()
        root = door_world.root
        semantic_door_annotation: Door = door_world.get_semantic_annotations_by_type(
            Door
        )[0]

        door_hinge = Body(
            name=PrefixedName(f"{root.name.name}_door_hinge", root.name.prefix)
        )
        parent_T_hinge = self._calculate_door_pivot_point(
            semantic_door_annotation,
            parent_T_door,
            door_factory.scale,
            opening_axis,
        )
        hinge_T_door = parent_T_hinge.inverse() @ parent_T_door

        hinge_door_connection = FixedConnection(
            parent=door_hinge,
            child=root,
            parent_T_connection_expression=hinge_T_door,
        )
        with door_world.modify_world():
            door_world.add_connection(hinge_door_connection)

        return door_world, parent_T_hinge

    def _calculate_door_pivot_point(
        self,
        semantic_door_annotation: Door,
        parent_T_door: TransformationMatrix,
        scale: Scale,
        opening_axis: Vector3,
    ) -> TransformationMatrix:
        """
        Calculate the door pivot point based on the handle position and the door scale. The pivot point is on the opposite
        side of the handle.

        :param semantic_door_annotation: The door semantic annotation containing the handle.
        :param parent_T_door: The transformation matrix defining the door's position and orientation.
        :param scale: The scale of the door.
        :param opening_axis: The axis along which the door is open.

        :return: The transformation matrix defining the door's pivot point.
        """
        connection = semantic_door_annotation.handle.body.parent_connection
        door_P_handle: NDArray[float] = (
            connection.origin_expression.to_position().to_np()
        )

        match opening_axis.to_np().tolist():
            case [0, 1, 0, 0]:
                sign = np.sign(-1 * door_P_handle[2]) if door_P_handle[2] != 0 else 1
                offset = sign * (scale.z / 2)
                parent_P_hinge = parent_T_door.to_np()[:3, 3] + np.array([0, 0, offset])
            case [0, 0, 1, 0]:
                sign = np.sign(-1 * door_P_handle[1]) if door_P_handle[1] != 0 else 1
                offset = sign * (scale.y / 2)
                parent_P_hinge = parent_T_door.to_np()[:3, 3] + np.array([0, offset, 0])
            case _:
                raise InvalidAxisError(axis=opening_axis)

        parent_T_hinge = TransformationMatrix.from_point_rotation_matrix(
            Point3(*parent_P_hinge)
        )

        return parent_T_hinge


HandlePosition = Union[SemanticPositionDescription, TransformationMatrix]


@dataclass(eq=False)
class HasHandle(ABC):

    handle: Optional[Handle] = None
    """
    The handle of the semantic annotation.
    """

    def add_handle(
        self: HasBody | Self,
        handle: Handle,
        handle_position: HandlePosition,
    ):
        """
        Adds a handle to the parent world with a fixed connection.

        :param parent_T_handle: The transformation matrix defining the handle's position and orientation relative
        to the parent world.
        :param parent_world: The world to which the handle will be added.
        """
        connection = FixedConnection(
            parent=self.body,
            child=handle.body,
            parent_T_connection_expression=self._resolve_parent_T_handle(
                handle_position
            ),
        )

    def _resolve_parent_T_handle(
        self: HasBody, handle_position: HandlePosition
    ) -> Optional[TransformationMatrix]:
        """
        Return a transformation matrix that defines the position and orientation of the handle relative to its parent.
        :raises: NotImplementedError if the handle direction is Z or NEGATIVE_Z.
        """

        match handle_position:
            case SemanticPositionDescription():
                min_bounds, max_bounds = self.body.combined_mesh.bounding_box.bounds
                min_point = Point3.from_iterable(min_bounds)
                max_point = Point3.from_iterable(max_bounds)
                body_bounding_box = BoundingBox.from_min_max(min_point, max_point)
                body_as_event = body_bounding_box.simple_event.as_composite_set()
                sampled_2d_point = handle_position.sample_point_from_event(
                    body_as_event.marginal(SpatialVariables.yz)
                )

                return TransformationMatrix.from_xyz_rpy(
                    x=body_bounding_box.depth / 2,
                    y=sampled_2d_point[0],
                    z=sampled_2d_point[1],
                )
            case TransformationMatrix():
                return handle_position
            case _:
                assert_never(handle_position)


@dataclass(eq=False)
class HasSupportingSurface(HasBody, ABC):
    """
    A semantic annotation that represents a supporting surface.
    """

    supporting_surface: Region = field(init=False)

    def __post_init__(self):
        self.recalculate_supporting_surface()

    def recalculate_supporting_surface(
        self,
        upward_threshold: float = 0.95,
        clearance_threshold: float = 0.5,
        min_surface_area: float = 0.0225,  # 15cm x 15cm
    ):
        mesh = self.body.combined_mesh
        if mesh is None:
            return
        # --- Find upward-facing faces ---
        normals = mesh.face_normals
        upward_mask = normals[:, 2] > upward_threshold

        if not upward_mask.any():
            return

        # --- Find connected upward-facing regions ---
        upward_face_indices = np.nonzero(upward_mask)[0]
        submesh_up = mesh.submesh([upward_face_indices], append=True)
        face_groups = submesh_up.split(only_watertight=False)

        # Compute total area for each group
        large_groups = [g for g in face_groups if g.area >= min_surface_area]

        if not large_groups:
            return

        # --- Merge qualifying upward-facing submeshes ---
        candidates = trimesh.util.concatenate(large_groups)

        # --- Check vertical clearance using ray casting ---
        face_centers = candidates.triangles_center
        ray_origins = face_centers + np.array([0, 0, 0.01])  # small upward offset
        ray_dirs = np.tile([0, 0, 1], (len(ray_origins), 1))

        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs
        )

        # Compute distances to intersections (if any)
        distances = np.full(len(ray_origins), np.inf)
        distances[index_ray] = np.linalg.norm(
            locations - ray_origins[index_ray], axis=1
        )

        # Filter faces with enough space above
        clear_mask = (distances > clearance_threshold) | np.isinf(distances)

        if not clear_mask.any():
            raise ValueError(
                "No upward-facing surfaces with sufficient clearance found."
            )

        candidates_filtered = candidates.submesh([clear_mask], append=True)

        # --- Build the region ---
        points_3d = [
            Point3(
                x,
                y,
                z,
                reference_frame=self.body,
            )
            for x, y, z in candidates_filtered.vertices
        ]

        self.supporting_surface = Region.from_3d_points(
            name=PrefixedName(
                f"{self.body.name.name}_supporting_surface_region",
                self.body.name.prefix,
            ),
            points_3d=points_3d,
        )


@dataclass(eq=False)
class HasCorpus(HasSupportingSurface, ABC):

    @property
    @abstractmethod
    def opening_direction(self) -> Direction: ...

    @classmethod
    def create_with_new_body(
        cls, name: PrefixedName, scale: Scale, wall_thickness: float = 0.01
    ) -> Self:
        container_event = cls._create_container_event(scale, wall_thickness)

        corpus_body = Body(name=name)
        collision_shapes = BoundingBoxCollection.from_event(
            corpus_body, container_event
        ).as_shapes()
        corpus_body.collision = collision_shapes
        corpus_body.visual = collision_shapes

        semantic_container_annotation = cls(body=corpus_body, name=name)

        return semantic_container_annotation

    @classmethod
    def _create_container_event(cls, scale: Scale, wall_thickness: float) -> Event:
        """
        Return an event representing a container with walls of a specified thickness.
        """
        outer_box = scale.to_simple_event()
        inner_scale = Scale(
            scale.x - wall_thickness,
            scale.y - wall_thickness,
            scale.z - wall_thickness,
        ).to_simple_event(self.opening_direction, wall_thickness)

        inner_box = cls._extend_inner_event_in_direction(
            outer_scale=scale, inner_scale=inner_scale
        )

        container_event = outer_box.as_composite_set() - inner_box.as_composite_set()

        return container_event
