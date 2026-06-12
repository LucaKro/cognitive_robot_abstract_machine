from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import Generic, List, Optional, Self, Type, TypeVar, Union
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
    Region,
)

if TYPE_CHECKING:
    from random_events.product_algebra import Event

    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Connection

T = TypeVar("T", bound=KinematicStructureEntity)


@dataclass
class KinematicStructureEntitySpec(ABC, Generic[T]):
    """
    Declarative, world-independent description of a kinematic structure entity.

    A spec is reusable: every materialization copies the prototype shapes and the
    pose, so the spec never becomes bound to an entity or world.
    """

    name: Union[str, PrefixedName]
    """
    The name of entities created from this spec. Can be overridden per spawn.
    """

    shapes: List[Shape] = field(default_factory=list)
    """
    Prototype shapes with origins expressed in the entity frame.
    """

    pose: Optional[HomogeneousTransformationMatrix] = None
    """
    Default placement of created entities in the parent frame. Overridden by the
    pose passed to spawn or to an annotation factory. None means identity.
    """

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PrefixedName(self.name)

    def _copied_shapes(self) -> List[Shape]:
        """
        Fresh, frame-unbound copies of the prototype shapes.
        """
        return [shape.copy_for_world(None) for shape in self.shapes]

    def _copied_pose(
        self, override: Optional[HomogeneousTransformationMatrix] = None
    ) -> HomogeneousTransformationMatrix:
        """
        A fresh pose to bind to a connection: the override if given, else this
        spec's default pose, else identity — always copied, because binding
        mutates the pose's reference and child frame in place.

        :param override: A pose taking precedence over the spec's default.
        :return: The copied pose.
        """
        pose = override if override is not None else self.pose
        if pose is None:
            return HomogeneousTransformationMatrix()
        return HomogeneousTransformationMatrix(pose.to_np())

    @abstractmethod
    def _to_entity(self, name: Optional[Union[str, PrefixedName]] = None) -> T:
        """
        Create a new, world-independent entity from this spec.

        :param name: Optional name override.
        :return: The created entity.
        """

    def spawn(
        self,
        world: World,
        *,
        parent: Optional[KinematicStructureEntity] = None,
        pose: Optional[HomogeneousTransformationMatrix] = None,
        name: Optional[Union[str, PrefixedName]] = None,
        connection_type: Type[Connection] = FixedConnection,
    ) -> T:
        """
        Create a new entity from this spec and connect it to a world.

        Can be called standalone or inside an open ``world.modify_world()`` block.

        :param world: The world to add the entity to.
        :param parent: The entity to connect the new entity to. Defaults to the
                       world's root, which requires the world to be non-empty.
        :param pose: The pose of the entity in the parent frame, taking precedence
                     over the spec's default pose. The pose must be expressed in
                     the parent frame; transform it first otherwise.
        :param name: Optional name override for the entity.
        :param connection_type: The connection used to attach the entity, e.g.
                                Connection6DoF for a freely movable body. Connection
                                types whose create_with_dofs requires extra arguments
                                (hinges, sliders) are not supported here; use the
                                semantic annotation API for those.
        :return: The created entity, connected to the world.
        """
        entity = self._to_entity(name=name)

        with world.modify_world():
            connection_parent = parent or world.root
            parent_T_entity = self._copied_pose(pose)
            parent_T_entity.reference_frame = connection_parent
            parent_T_entity.child_frame = entity
            connection = connection_type.create_with_dofs(
                world=world,
                parent=connection_parent,
                child=entity,
                parent_T_connection_expression=parent_T_entity,
            )
            world.add_connection(connection)
        return entity

    @classmethod
    def from_event(cls, name: Union[str, PrefixedName], event: Event) -> Self:
        """
        Spec whose shapes are the bounding boxes of a random event.

        This is the construction used by semantic annotations with composite
        geometry (hollow handles, container cases, walls minus apertures, ...).

        :param name: The name of the entity.
        :param event: The event describing the geometry, in the entity frame.
        :return: The created spec.
        """
        # BoundingBoxCollection requires a reference frame, so the shapes are
        # built around a throwaway body and unbound again for the spec.
        anchor = Body(name=PrefixedName("spec_anchor"))
        shapes = BoundingBoxCollection.from_event(anchor, event).as_shapes()
        return cls(name=name, shapes=[shape.copy_for_world(None) for shape in shapes])


@dataclass
class BodySpec(KinematicStructureEntitySpec[Body]):
    """
    Declarative, world-independent description of a :class:`Body`.

    The shapes are used for both collision and visual geometry (one shared
    :class:`ShapeCollection`, like :meth:`Body.from_shape_collection`); set
    `visual_shapes` when visual geometry should differ from collision geometry.

    The 90% case::

        box = BodySpec.box("crate", Scale(0.2, 0.2, 0.2))
        body = box.spawn(world, pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0))
    """

    inertial: Optional[Inertial] = None
    """
    Inertia properties of created bodies. None means the Body default.
    """

    visual_shapes: Optional[List[Shape]] = None
    """
    Visual shapes when they differ from `shapes`. None shares `shapes` for both
    collision and visual (one collection); an empty list means no visual geometry.
    """

    def to_body(self, name: Optional[Union[str, PrefixedName]] = None) -> Body:
        """
        Create a new, world-independent body from this spec.

        :param name: Optional name override, e.g. for spawning multiple bodies
                     from the same spec.
        :return: The created body.
        """
        if isinstance(name, str):
            name = PrefixedName(name)
        if self.visual_shapes is None:
            body = Body.from_shape_collection(
                name=name or self.name,
                shape_collection=ShapeCollection(self._copied_shapes()),
            )
        else:
            body = Body(
                name=name or self.name,
                collision=ShapeCollection(self._copied_shapes()),
                visual=ShapeCollection(
                    [shape.copy_for_world(None) for shape in self.visual_shapes]
                ),
            )
        if self.inertial is not None:
            body.inertial = deepcopy(self.inertial)
        return body

    def _to_entity(self, name: Optional[Union[str, PrefixedName]] = None) -> Body:
        return self.to_body(name=name)

    @classmethod
    def box(
        cls,
        name: Union[str, PrefixedName],
        scale: Scale,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> BodySpec:
        """
        Spec for a body with a single box shape.

        :param name: The name of the body.
        :param scale: The extents of the box.
        :param color: The color of the box.
        :param origin: The origin of the box in the body frame. Defaults to identity.
        :return: The created spec.
        """
        return cls(
            name=name,
            shapes=[
                Box(
                    origin=origin or HomogeneousTransformationMatrix(),
                    scale=scale,
                    color=color or Color(),
                )
            ],
        )

    @classmethod
    def sphere(
        cls,
        name: Union[str, PrefixedName],
        radius: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> BodySpec:
        """
        Spec for a body with a single sphere shape.

        :param name: The name of the body.
        :param radius: The radius of the sphere.
        :param color: The color of the sphere.
        :param origin: The origin of the sphere in the body frame. Defaults to identity.
        :return: The created spec.
        """
        return cls(
            name=name,
            shapes=[
                Sphere(
                    origin=origin or HomogeneousTransformationMatrix(),
                    radius=radius,
                    color=color or Color(),
                )
            ],
        )

    @classmethod
    def cylinder(
        cls,
        name: Union[str, PrefixedName],
        width: float,
        height: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> BodySpec:
        """
        Spec for a body with a single cylinder shape.

        :param name: The name of the body.
        :param width: The diameter of the cylinder.
        :param height: The height of the cylinder.
        :param color: The color of the cylinder.
        :param origin: The origin of the cylinder in the body frame. Defaults to identity.
        :return: The created spec.
        """
        return cls(
            name=name,
            shapes=[
                Cylinder(
                    origin=origin or HomogeneousTransformationMatrix(),
                    width=width,
                    height=height,
                    color=color or Color(),
                )
            ],
        )

    @classmethod
    def mesh(
        cls,
        name: Union[str, PrefixedName],
        filename: str,
        scale: Optional[Scale] = None,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> BodySpec:
        """
        Spec for a body with a single mesh shape loaded from a file.

        :param name: The name of the body.
        :param filename: The path of the mesh file.
        :param scale: The scale applied to the mesh.
        :param color: The color of the mesh.
        :param origin: The origin of the mesh in the body frame. Defaults to identity.
        :return: The created spec.
        """
        return cls(
            name=name,
            shapes=[
                Mesh(
                    origin=origin or HomogeneousTransformationMatrix(),
                    filename=filename,
                    scale=scale or Scale(),
                    color=color or Color(),
                )
            ],
        )


@dataclass
class RegionSpec(KinematicStructureEntitySpec[Region]):
    """
    Declarative, world-independent description of a :class:`Region`.

    The shapes become the region's area.
    """

    def to_region(self, name: Optional[Union[str, PrefixedName]] = None) -> Region:
        """
        Create a new, world-independent region from this spec.

        :param name: Optional name override.
        :return: The created region.
        """
        if isinstance(name, str):
            name = PrefixedName(name)
        return Region.from_shape_collection(
            name=name or self.name, shape_collection=ShapeCollection(self._copied_shapes())
        )

    def _to_entity(self, name: Optional[Union[str, PrefixedName]] = None) -> Region:
        return self.to_region(name=name)
