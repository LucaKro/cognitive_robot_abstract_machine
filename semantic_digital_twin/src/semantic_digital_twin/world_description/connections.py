from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from typing_extensions import List, TYPE_CHECKING, Union, Optional, Self, Dict, Any

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.degree_of_freedom_ownership import (
    DegreeOfFreedomOwnership,
    DegreeOfFreedomRole,
)
from semantic_digital_twin.world_description.world_entity import (
    Connection,
    KinematicStructureEntity,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    Quaternion,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


class HasUpdateState(ABC):
    """
    Mixin class for connections that need state updated which are not trivial integrations.
    Typically needed for connections that use active and passive degrees of freedom.
    Look at OmniDrive for an example usage.
    """

    @abstractmethod
    def update_state(self, dt: float) -> None:
        """
        Allows the connection to update the state of its dofs.
        An integration update for active dofs will have happened before this method is called.
        Write directly into self._world.state, but don't touch dofs that don't belong to this connection.
        :param dt: Time passed since last update.
        """
        pass


@dataclass(eq=False)
class FixedConnection(Connection):
    """
    Has 0 degrees of freedom.
    """

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        *,
        name: Optional[PrefixedName] = None,
        **kwargs,
    ) -> Self:
        return FixedConnection(parent=parent, child=child, name=name, **kwargs)


@dataclass(eq=False)
class ActiveConnection(Connection):
    """
    Has one or more degrees of freedom that can be actively controlled, e.g., robot joints.
    """

    @property
    def has_hardware_interface(self) -> bool:
        """
        Whether this connection is linked to a controller and can therefore respond to control commands.

        E.g. the caster wheels of a PR2 are active, because they have a DOF, but they are not directly controlled.
        Instead a the omni drive connection is directly controlled and a low level controller translates these commands
        to commands for the caster wheels.

        A door hinge is also active but cannot be controlled.
        """
        return any(dof.has_hardware_interface for dof in self.dofs)

    @has_hardware_interface.setter
    def has_hardware_interface(self, value: bool) -> None:
        for dof in self.dofs:
            dof.has_hardware_interface = value

    @property
    def is_controlled(self):
        return self.has_hardware_interface


@dataclass(eq=False)
class ActiveConnection1DOF(ActiveConnection, ABC):
    """
    Superclass for active connections with 1 degree of freedom.
    """

    axis: Vector3 = field(kw_only=True)
    """
    Connection moves along this axis, should be a unit vector.
    The axis is defined relative to the local reference frame of the parent KinematicStructureEntity.
    """

    multiplier: float = 1.0
    """
    Movement along the axis is multiplied by this value. Useful if Connections share DoFs.
    """

    offset: float = 0.0
    """
    Movement along the axis is offset by this value. Useful if Connections share DoFs.
    """

    dynamics: JointDynamics = field(default_factory=JointDynamics)
    """
    Dynamic properties of the joint.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["axis"] = to_json(self.axis)
        result["multiplier"] = self.multiplier
        result["offset"] = self.offset
        result["dynamics"] = to_json(self.dynamics)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)

        return cls(
            **{
                "name": from_json(data["name"]),
                "parent": tracker.get_world_entity_with_id(
                    id=from_json(data["parent_id"])
                ),
                "child": tracker.get_world_entity_with_id(
                    id=from_json(data["child_id"])
                ),
                "parent_T_connection_expression": from_json(
                    data["parent_T_connection_expression"], **kwargs
                ),
                "connection_T_child_expression": from_json(
                    data["connection_T_child_expression"], **kwargs
                ),
                "degrees_of_freedom": from_json(data["degrees_of_freedom"], **kwargs),
            },
            axis=from_json(data["axis"], **kwargs),
            multiplier=data["multiplier"],
            offset=data["offset"],
            dynamics=from_json(data["dynamics"], **kwargs),
        )

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        *,
        axis: Vector3,
        name: Optional[PrefixedName] = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: Optional[DegreeOfFreedomLimits] = None,
        **kwargs,
    ) -> Self:
        """
        Creates and returns an instance of the class with associated degrees of freedom
        (DOFs) based on the specified parameters. This method facilitates initializing
        a kinematic relationship between a parent and a child entity, augmented by
        an axis representation and configurable properties such as multiplier and offset.

        :param world: The motion world in which to add the degree of freedom.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param axis: The axis vector defining the joint relation.
        :param name: Optional specific name for the DOF entity. If not provided, a
                     default name is generated based on the parent and child.
        :param multiplier: A scaling factor applied to the DOF's motion. Defaults to 1.0.
        :param offset: A constant offset value applied to the DOF's motion. Defaults to 0.0.
        :return: An instance of the class representing the defined relationship with
                 its DOF added to the world.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        dof = DegreeOfFreedom(name=PrefixedName("dof", str(name)), limits=dof_limits)
        world.add_degree_of_freedom(dof)
        connection = cls(
            name=name,
            parent=parent,
            child=child,
            axis=axis,
            multiplier=multiplier,
            offset=offset,
            degrees_of_freedom=DegreeOfFreedomOwnership.single_active(dof),
            **kwargs,
        )
        return connection

    def __post_init__(self):
        super().__post_init__()
        if self.multiplier is None:
            self.multiplier = 1.0
        if self.offset is None:
            self.offset = 0.0

    @property
    def dof_id(self) -> UUID:
        """
        :return: The id of the degree of freedom controlling movement along the axis.
        """
        return self.raw_dof.id

    @property
    def dof(self) -> DegreeOfFreedom:
        """
        A reference to the Degree of Freedom associated with this connection.
        .. warning:: WITH multiplier and offset applied.
        """
        result = deepcopy(self.raw_dof)
        result.variables = self.raw_dof.variables * self.multiplier
        if self.multiplier < 0:
            # if multiplier is negative, we need to swap the limits
            result.limits.lower, result.limits.upper = (
                result.limits.upper,
                result.limits.lower,
            )
        result.limits.lower = result.limits.lower * self.multiplier
        result.limits.upper = result.limits.upper * self.multiplier

        result.variables.position += self.offset
        if result.limits.lower.position is not None:
            result.limits.lower.position = result.limits.lower.position + self.offset
        if result.limits.upper.position is not None:
            result.limits.upper.position = result.limits.upper.position + self.offset
        return result

    @property
    def raw_dof(self) -> DegreeOfFreedom:
        """
        A reference to the Degree of Freedom associated with this connection.
        .. warning:: WITHOUT multiplier and offset applied.
        """
        return self.degrees_of_freedom.dof_for(DegreeOfFreedomRole.MAIN)

    @property
    def position(self) -> float:
        return (
            self._world.state[self.raw_dof.id].position * self.multiplier + self.offset
        )

    @position.setter
    def position(self, value: float) -> None:
        self._world.state[self.raw_dof.id].position = (
            value - self.offset
        ) / self.multiplier
        self._world.notify_state_change()

    @property
    def velocity(self) -> float:
        return self._world.state[self.raw_dof.id].velocity * self.multiplier

    @velocity.setter
    def velocity(self, value: float) -> None:
        self._world.state[self.raw_dof.id].velocity = value / self.multiplier
        self._world.notify_state_change()

    @property
    def acceleration(self) -> float:
        return self._world.state[self.raw_dof.id].acceleration * self.multiplier

    @acceleration.setter
    def acceleration(self, value: float) -> None:
        self._world.state[self.raw_dof.id].acceleration = value / self.multiplier
        self._world.notify_state_change()

    @property
    def jerk(self) -> float:
        return self._world.state[self.raw_dof.id].jerk * self.multiplier

    @jerk.setter
    def jerk(self, value: float) -> None:
        self._world.state[self.raw_dof.id].jerk = value / self.multiplier
        self._world.notify_state_change()


@dataclass(eq=False)
class PrismaticConnection(ActiveConnection1DOF):
    """
    Allows translation along an axis.
    """

    def add_to_world(self, world: World):
        super().add_to_world(world)

        translation_axis = self.axis * self.dof.variables.position
        self._kinematics = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=translation_axis[0],
            y=translation_axis[1],
            z=translation_axis[2],
            child_frame=self.child,
        )


@dataclass(eq=False)
class RevoluteConnection(ActiveConnection1DOF):
    """
    Allows rotation about an axis.
    """

    def add_to_world(self, world: World):
        super().add_to_world(world)

        self._kinematics = HomogeneousTransformationMatrix.from_xyz_axis_angle(
            axis=self.axis,
            angle=self.dof.variables.position,
            child_frame=self.child,
        )


@dataclass(eq=False)
class Connection6DoF(Connection):
    """
    Has full 6 degrees of freedom, that cannot be actively controlled.
    Useful for synchronizing with transformations from external providers.
    """

    @property
    def x(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.X)

    @property
    def y(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.Y)

    @property
    def z(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.Z)

    @property
    def qx(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.QX)

    @property
    def qy(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.QY)

    @property
    def qz(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.QZ)

    @property
    def qw(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.QW)

    @property
    def x_id(self) -> UUID:
        return self.x.id

    @property
    def y_id(self) -> UUID:
        return self.y.id

    @property
    def z_id(self) -> UUID:
        return self.z.id

    @property
    def qx_id(self) -> UUID:
        return self.qx.id

    @property
    def qy_id(self) -> UUID:
        return self.qy.id

    @property
    def qz_id(self) -> UUID:
        return self.qz.id

    @property
    def qw_id(self) -> UUID:
        return self.qw.id

    def add_to_world(self, world: World):
        super().add_to_world(world)
        parent_P_child = Point3(
            x=self.x.variables.position,
            y=self.y.variables.position,
            z=self.z.variables.position,
        )
        parent_R_child = Quaternion(
            x=self.qx.variables.position,
            y=self.qy.variables.position,
            z=self.qz.variables.position,
            w=self.qw.variables.position,
        ).to_rotation_matrix()
        self._kinematics = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=parent_P_child,
            rotation_matrix=parent_R_child,
            child_frame=self.child,
        )

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        *,
        name: Optional[PrefixedName] = None,
        parent_T_connection_expression: Optional[
            HomogeneousTransformationMatrix
        ] = None,
        **kwargs,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom (DoFs)
        for the provided parent and child kinematic entities within the specified world.

        This method initializes and adds the required degrees of freedom to the world,
        and sets their properties accordingly. It generates a name for the connection if
        none is provided, and ensures valid initial state for relevant degrees of freedom.

        :param world: The World object where the degrees of freedom are added and modified.
        :param parent: The KinematicStructureEntity serving as the parent.
        :param child: The KinematicStructureEntity serving as the child.
        :param name: An optional PrefixedName for the connection. If None, it will be
                     auto-generated based on the parent and child names.
        :param parent_T_connection_expression: Optional transformation matrix specifying
                                               the connection relationship between parent
                                               and child entities.
        :return: A new instance of the class representing the parent-child connection with
                 automatically defined degrees of freedom.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)

        stringified_name = str(name)
        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        z = DegreeOfFreedom(name=PrefixedName("z", stringified_name))
        world.add_degree_of_freedom(z)
        qx = DegreeOfFreedom(name=PrefixedName("qx", stringified_name))
        world.add_degree_of_freedom(qx)
        qy = DegreeOfFreedom(name=PrefixedName("qy", stringified_name))
        world.add_degree_of_freedom(qy)
        qz = DegreeOfFreedom(name=PrefixedName("qz", stringified_name))
        world.add_degree_of_freedom(qz)
        qw = DegreeOfFreedom(name=PrefixedName("qw", stringified_name))
        world.add_degree_of_freedom(qw)
        world.state[qw.id].position = 1.0

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            name=name,
            degrees_of_freedom=DegreeOfFreedomOwnership.create(
                passive={
                    DegreeOfFreedomRole.X: x,
                    DegreeOfFreedomRole.Y: y,
                    DegreeOfFreedomRole.Z: z,
                    DegreeOfFreedomRole.QX: qx,
                    DegreeOfFreedomRole.QY: qy,
                    DegreeOfFreedomRole.QZ: qz,
                    DegreeOfFreedomRole.QW: qw,
                }
            ),
        )

    @property
    def origin(self) -> HomogeneousTransformationMatrix:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: Union[NpMatrix4x4, HomogeneousTransformationMatrix]
    ) -> None:
        if not isinstance(transformation, HomogeneousTransformationMatrix):
            transformation = HomogeneousTransformationMatrix(data=transformation)
        position = transformation.to_position().to_np()
        orientation = transformation.to_rotation_matrix().to_quaternion().to_np()
        self._world.state[self.x.id].position = position[0]
        self._world.state[self.y.id].position = position[1]
        self._world.state[self.z.id].position = position[2]
        self._world.state[self.qx.id].position = orientation[0]
        self._world.state[self.qy.id].position = orientation[1]
        self._world.state[self.qz.id].position = orientation[2]
        self._world.state[self.qw.id].position = orientation[3]
        self._world.notify_state_change()


@dataclass(eq=False)
class WheeledDrive(ActiveConnection, HasUpdateState, ABC):
    """
    Superclass for connections that describe a drive, e.g., an omnidirectional drive or a differential drive.
    """

    @property
    def x(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.X)

    @property
    def y(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.Y)

    @property
    def roll(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.ROLL)

    @property
    def pitch(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.PITCH)

    @property
    def yaw(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.YAW)

    @property
    def x_velocity(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.X_VELOCITY)

    @property
    def origin(self) -> HomogeneousTransformationMatrix:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: Union[NpMatrix4x4, HomogeneousTransformationMatrix]
    ) -> None:
        """
        Overwrites the origin of the connection by projecting it onto the plane the drive can move in.

        .. note::
            A planar drive only represents x, y, and yaw, so the z, roll, and pitch components of
            ``transformation`` are intentionally ignored.
        """
        if isinstance(transformation, np.ndarray):
            transformation = HomogeneousTransformationMatrix(data=transformation)
        position = transformation.to_position()
        _, _, yaw = transformation.to_rotation_matrix().to_rpy()
        self._world.state[self.x.id].position = position.x
        self._world.state[self.y.id].position = position.y
        self._world.state[self.yaw.id].position = yaw
        self._world.notify_state_change()

    def get_free_variable_names(self) -> List[UUID]:
        return [self.x.id, self.y.id, self.yaw.id]

    @property
    def has_hardware_interface(self) -> bool:
        return self.x_velocity.has_hardware_interface

    @has_hardware_interface.setter
    def has_hardware_interface(self, value: bool) -> None:
        for dof in self.active_dofs:
            dof.has_hardware_interface = value


@dataclass(eq=False)
class OmniDrive(WheeledDrive):
    """
    A connection describing an omnidirectional drive.
    It can rotate about its z-axis and drive on the x-y plane simultaneously.
    - x/y: Passive dofs describing the measured odometry with respect to parent frame.
        We assume that the robot can't fly, and we can't measure its z-axis position, so z=0.
        The odometry sensors typically provide velocity measurements with respect to the child frame,
        therefore the velocity values of x/y must stay 0.
    - x_vel/y_vel: The measured and commanded velocity is represented with respect to the child frame with these
        active dofs. It must be ensured that their position values stay 0.
    - roll/pitch: Some robots, like the PR2, have sensors to measure pitch and roll using an IMU,
        we therefore have passive dofs for them.
    - yaw: Since the robot can only rotate about its z-axis, we don't need different dofs for position and velocity of yaw.
        They are combined into one active dof.
    """

    @property
    def y_velocity(self) -> DegreeOfFreedom:
        return self._dof_by_role(DegreeOfFreedomRole.Y_VELOCITY)

    def add_to_world(self, world: World):
        super().add_to_world(world)
        odom_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x.variables.position,
            y=self.y.variables.position,
            yaw=self.yaw.variables.position,
        )
        bf_T_bf_vel = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x_velocity.variables.position, y=self.y_velocity.variables.position
        )
        bf_vel_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0,
            y=0,
            z=0,
            roll=self.roll.variables.position,
            pitch=self.pitch.variables.position,
            yaw=0,
        )
        self._kinematics = odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf
        self._kinematics.child_frame = self.child

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        *,
        name: Optional[PrefixedName] = None,
        parent_T_connection_expression: Optional[
            HomogeneousTransformationMatrix
        ] = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
        **kwargs,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom
        (DOFs) for translation on the x and y axes, rotation along roll, pitch, and yaw
        axes, and velocity limits for translation and rotation.

        This method modifies the provided world to add all required degrees of freedom
        and their limits, based on the provided settings. Names for the degrees of
        freedom are auto-generated using the stringified version of the provided name
        or its default setting.

        :param world: The world where the configuration is being applied, and degrees of freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        stringified_name = str(name)
        lower_translation_limits = DerivativeMap()
        lower_translation_limits.velocity = -translation_velocity_limits
        upper_translation_limits = DerivativeMap()
        upper_translation_limits.velocity = translation_velocity_limits
        lower_rotation_limits = DerivativeMap()
        lower_rotation_limits.velocity = -rotation_velocity_limits
        upper_rotation_limits = DerivativeMap()
        upper_rotation_limits.velocity = rotation_velocity_limits

        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        roll = DegreeOfFreedom(name=PrefixedName("roll", stringified_name))
        world.add_degree_of_freedom(roll)
        pitch = DegreeOfFreedom(name=PrefixedName("pitch", stringified_name))
        world.add_degree_of_freedom(pitch)
        yaw = DegreeOfFreedom(
            name=PrefixedName("yaw", stringified_name),
            limits=DegreeOfFreedomLimits(
                lower=lower_rotation_limits,
                upper=upper_rotation_limits,
            ),
        )
        world.add_degree_of_freedom(yaw)

        x_vel = DegreeOfFreedom(
            name=PrefixedName("x_vel", stringified_name),
            limits=DegreeOfFreedomLimits(
                lower=lower_translation_limits,
                upper=upper_translation_limits,
            ),
        )
        world.add_degree_of_freedom(x_vel)
        y_vel = DegreeOfFreedom(
            name=PrefixedName("y_vel", stringified_name),
            limits=DegreeOfFreedomLimits(
                lower=lower_translation_limits,
                upper=upper_translation_limits,
            ),
        )
        world.add_degree_of_freedom(y_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            name=name,
            degrees_of_freedom=DegreeOfFreedomOwnership.create(
                active={
                    DegreeOfFreedomRole.X_VELOCITY: x_vel,
                    DegreeOfFreedomRole.Y_VELOCITY: y_vel,
                    DegreeOfFreedomRole.YAW: yaw,
                },
                passive={
                    DegreeOfFreedomRole.X: x,
                    DegreeOfFreedomRole.Y: y,
                    DegreeOfFreedomRole.ROLL: roll,
                    DegreeOfFreedomRole.PITCH: pitch,
                },
            ),
        )

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_velocity.id].position = 0
        state[self.y_velocity.id].position = 0

        x_vel = state[self.x_velocity.id].velocity
        y_vel = state[self.y_velocity.id].velocity
        delta = state[self.yaw.id].position
        x_velocity = np.cos(delta) * x_vel - np.sin(delta) * y_vel
        state[self.x.id].position += x_velocity * dt
        y_velocity = np.sin(delta) * x_vel + np.cos(delta) * y_vel
        state[self.y.id].position += y_velocity * dt


@dataclass(eq=False)
class DifferentialDrive(WheeledDrive):
    """
    A connection describing a differential drive.
    It can rotate around its z-axis and drive in x-direction. It allows movement in the x-y plane.
    """

    def add_to_world(self, world: World):
        super().add_to_world(world)
        odom_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x.variables.position,
            y=self.y.variables.position,
            yaw=self.yaw.variables.position,
        )
        bf_T_bf_vel = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x_velocity.variables.position,
        )
        bf_vel_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0,
            y=0,
            z=0,
            roll=self.roll.variables.position,
            pitch=self.pitch.variables.position,
            yaw=0,
        )
        self._kinematics = odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf
        self._kinematics.child_frame = self.child

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        *,
        name: Optional[PrefixedName] = None,
        parent_T_connection_expression: Optional[
            HomogeneousTransformationMatrix
        ] = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
        **kwargs,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated DoFs for translation on the x-axis,
        rotation along roll, pitch, and yaw axes, and velocity limits for translation and rotation.

        :param world: The world where the configuration is being applied, and degrees of freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        stringified_name = str(name)
        lower_translation_limits = DerivativeMap()
        lower_translation_limits.velocity = -translation_velocity_limits
        upper_translation_limits = DerivativeMap()
        upper_translation_limits.velocity = translation_velocity_limits
        lower_rotation_limits = DerivativeMap()
        lower_rotation_limits.velocity = -rotation_velocity_limits
        upper_rotation_limits = DerivativeMap()
        upper_rotation_limits.velocity = rotation_velocity_limits

        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        roll = DegreeOfFreedom(name=PrefixedName("roll", stringified_name))
        world.add_degree_of_freedom(roll)
        pitch = DegreeOfFreedom(name=PrefixedName("pitch", stringified_name))
        world.add_degree_of_freedom(pitch)
        yaw = DegreeOfFreedom(
            name=PrefixedName("yaw", stringified_name),
            limits=DegreeOfFreedomLimits(
                lower=lower_rotation_limits,
                upper=upper_rotation_limits,
            ),
        )
        world.add_degree_of_freedom(yaw)

        x_vel = DegreeOfFreedom(
            name=PrefixedName("x_vel", stringified_name),
            limits=DegreeOfFreedomLimits(
                lower=lower_translation_limits,
                upper=upper_translation_limits,
            ),
        )
        world.add_degree_of_freedom(x_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            name=name,
            degrees_of_freedom=DegreeOfFreedomOwnership.create(
                active={
                    DegreeOfFreedomRole.X_VELOCITY: x_vel,
                    DegreeOfFreedomRole.YAW: yaw,
                },
                passive={
                    DegreeOfFreedomRole.X: x,
                    DegreeOfFreedomRole.Y: y,
                    DegreeOfFreedomRole.ROLL: roll,
                    DegreeOfFreedomRole.PITCH: pitch,
                },
            ),
        )

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_velocity.id].position = 0

        x_vel = state[self.x_velocity.id].velocity
        delta = state[self.yaw.id].position
        x_velocity = np.cos(delta) * x_vel
        state[self.x.id].position += x_velocity * dt
        y_velocity = np.sin(delta) * x_vel
        state[self.y.id].position += y_velocity * dt
