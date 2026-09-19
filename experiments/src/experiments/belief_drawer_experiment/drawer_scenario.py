"""
The drawer one run of the belief experiment is performed on, and the three conditions it
is performed under.

The world is built here rather than supplied, so that what the tests exercise is what
the sweep runs. It is a planar arm and a sliding drawer rather than a full robot: every
robot in this repository is described by a ROS package that is not on PyPI, and the
experiment's claim is about how a weight behaves rather than about a particular arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import List, Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.grasp_likelihood_source import GraspLikelihoodSource
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% what one run varies


class DrawerCondition(StrEnum):
    """
    How a run weighs the grip on the drawer handle, and what the grasp behind it is
    doing.
    """

    UNCONDITIONAL_GRIP = "unconditional_grip"
    """
    The grip carries its full weight whatever the evidence says, which is what
    :class:`~giskardpy.motion_statechart.goals.open_close.Open` does today and is why it
    cannot represent a grasp that failed.
    """

    BELIEVED_GRIP = "believed_grip"
    """
    The grip's weight follows a belief the measurements keep confirming.
    """

    FAILING_GRIP = "failing_grip"
    """
    The grip's weight follows a belief whose measurements have stopped supporting a
    grasp.
    """

    @property
    def weighs_the_grip_by_belief(self) -> bool:
        """
        :return: Whether the run builds a belief and lets the grip's weight follow it.
        """
        return self is not DrawerCondition.UNCONDITIONAL_GRIP

    def share_of_hits(self, holding: float, failing: float) -> Optional[float]:
        """
        :param holding: The share of rays reported while a grasp holds.
        :param failing: The share of rays reported while it fails.
        :return: What the likelihood reports under this condition, or nothing where no
            grasp is measured at all. Reporting nothing is what makes
            :attr:`UNCONDITIONAL_GRIP` the baseline for both other conditions at once:
            stock ``Open`` reads no likelihood, so it runs the same way whether the
            grasp holds or fails.
        """
        if not self.weighs_the_grip_by_belief:
            return None
        return holding if self is DrawerCondition.BELIEVED_GRIP else failing


@dataclass(frozen=True)
class ArmConfiguration:
    """
    A posture the arm starts a run from.
    """

    name: str
    """
    What the posture is called in a result, so a row can be read back to a run.
    """

    shoulder: float
    """
    Where the shoulder joint starts, in radians.
    """

    elbow: float
    """
    Where the elbow joint starts, in radians.
    """

    wrist: float
    """
    Where the wrist joint starts, in radians.
    """


# %% a likelihood the experiment decides, in place of a raycast


@dataclass(eq=False, repr=False)
class ScriptedLikelihood(MotionStatechartNode, GraspLikelihoodSource):
    """
    Publishes the share of rays the condition dictates, in place of the raycast a live
    measurement samples.

    ..note:: The raycast is not usable here.
        :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_in_gripper`
        deduplicates the bodies its rays hit before counting them, so it answers ``0``
        or ``1 / sample_size`` rather than the share its docstring promises, and a
        likelihood capped that low collapses the belief under every condition alike.
    """

    share_of_hits: float = field(kw_only=True)
    """
    What share of the rays are to hit the body on every cycle.
    """

    sample_size: int = field(default=100, kw_only=True)
    """
    How many rays that share is reported out of, which is how far a reading is trusted.
    """

    _likelihood: Optional[FloatVariable] = field(default=None, init=False, repr=False)
    """
    The variable the share is published to, created while building.
    """

    @property
    def likelihood(self) -> FloatVariable:
        return self._likelihood

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self._likelihood = FloatVariable(f"{self.unique_name}_scripted_likelihood")
        context.float_variable_data.register_expression(self._likelihood)
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext) -> None:
        self._publish(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        self._publish(context)
        return None

    def _publish(self, context: MotionStatechartContext) -> None:
        """
        Writes the scripted share into the variable carrying it.

        :param context: The context holding the float variable data to write to.
        """
        context.float_variable_data.set_value(self._likelihood, self.share_of_hits)


# %% the world a run is performed in

ARM_SEGMENT_LENGTH = 0.45
"""
How long each of the arm's two segments is, in metres, so its reach is twice this.
"""

ARM_HEIGHT = 0.5
"""
How high above the base the arm and the drawer handle sit, in metres.
"""

CABINET_DISTANCE = 0.95
"""
How far in front of the base the cabinet's centre stands, in metres, placing the handle
within reach of a bent arm rather than at the arm's singular full extension.
"""

DRAWER_TRAVEL = 0.3
"""
How far the drawer slides open, in metres.

It exceeds what the arm can follow, which is what lets a run distinguish a grip that
keeps holding from one that gives way.
"""


@dataclass
class DrawerWorld:
    """
    A planar arm in front of a cabinet whose drawer slides away from it, together with
    the frames a run reads its outcome off.
    """

    world: World
    """
    The world the motion is executed in.
    """

    gripper: Body
    """
    The body the arm holds the handle with.
    """

    handle: Body
    """
    The drawer handle the grip is on.
    """

    mechanism: PrismaticConnection
    """
    The connection the drawer slides along, whose position is how far it is open.
    """

    arm: List[ActiveConnection1DOF]
    """
    The joints the arm's travel is summed over.
    """

    gripper_opening: JointState
    """
    The gripper state that would mean nothing is held, which the belief decays toward
    the prior while it is reached.
    """

    @classmethod
    def of(cls, cabinet_yaw: float, arm_configuration: ArmConfiguration) -> DrawerWorld:
        """
        Build the world with the cabinet turned by the given angle and the arm in the
        given posture.

        :param cabinet_yaw: How far the cabinet is turned about the vertical, in
            radians.
        :param arm_configuration: The posture the arm starts in.
        :return: The world, ready to execute a motion in.
        """
        world = World()
        with world.modify_world():
            base = cls._box("base", (0.1, 0.1, 0.1))
            world.add_body(base)
            shoulder = cls._box("shoulder", (ARM_SEGMENT_LENGTH, 0.05, 0.05))
            elbow = cls._box("elbow", (ARM_SEGMENT_LENGTH, 0.05, 0.05))
            gripper = cls._box("gripper", (0.04, 0.04, 0.04))
            arm = [
                cls._revolute(
                    world,
                    base,
                    shoulder,
                    HomogeneousTransformationMatrix.from_xyz_rpy(z=ARM_HEIGHT),
                ),
                cls._revolute(
                    world,
                    shoulder,
                    elbow,
                    HomogeneousTransformationMatrix.from_xyz_rpy(x=ARM_SEGMENT_LENGTH),
                ),
                cls._revolute(
                    world,
                    elbow,
                    gripper,
                    HomogeneousTransformationMatrix.from_xyz_rpy(x=ARM_SEGMENT_LENGTH),
                ),
            ]
            finger = cls._box("finger", (0.02, 0.02, 0.06))
            finger_connection = PrismaticConnection.create_with_dofs(
                parent=gripper,
                child=finger,
                axis=Vector3.Y(),
                world=world,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(x=0.03)
                ),
                dof_limits=cls._limits(0.0, 0.05, 0.1),
            )
            world.add_connection(finger_connection)

            handle, mechanism = cls._add_drawer(world, base, cabinet_yaw)
            MinimalRobot.from_world(world)

        for connection, position in zip(
            arm,
            (
                arm_configuration.shoulder,
                arm_configuration.elbow,
                arm_configuration.wrist,
            ),
        ):
            world.state[connection.dof.id].position = position
        world.notify_state_change()

        return cls(
            world=world,
            gripper=gripper,
            handle=handle,
            mechanism=mechanism,
            arm=arm,
            gripper_opening=JointState.from_mapping(
                {finger_connection: finger_connection.dof.limits.upper.position}
            ),
        )

    @property
    def mechanism_travel(self) -> float:
        """
        :return: How far the drawer is currently open, in metres.
        """
        return float(self.world.state[self.mechanism.dof.id].position)

    @property
    def grip_offset(self) -> float:
        """
        :return: How far the gripper currently is from the handle, in metres, which is
            how far a grip that stopped following has been left behind.
        """
        gripper_position = self.world.compute_forward_kinematics_np(
            self.world.root, self.gripper
        )[:3, 3]
        handle_position = self.world.compute_forward_kinematics_np(
            self.world.root, self.handle
        )[:3, 3]
        return float(np.linalg.norm(gripper_position - handle_position))

    @property
    def arm_positions(self) -> np.ndarray:
        """
        :return: Where each of the arm's joints currently stands, in radians.
        """
        return np.array(
            [
                float(self.world.state[connection.dof.id].position)
                for connection in self.arm
            ]
        )

    @property
    def gripper_touches_handle(self) -> bool:
        """
        :return: Whether the gripper and the handle are in contact, read off the world's
            own collision detector rather than off the belief.
        """
        return bool(contact(self.gripper, self.handle))

    @staticmethod
    def _box(name: str, scale: tuple[float, float, float]) -> Body:
        """
        :param name: What the body is called in the world.
        :param scale: How large the box is along each axis, in metres.
        :return: A body whose collision geometry is that box.
        """
        return Body(
            name=PrefixedName(name),
            collision=ShapeCollection(shapes=[Box(scale=Scale(*scale))]),
        )

    @staticmethod
    def _limits(lower: float, upper: float, velocity: float) -> DegreeOfFreedomLimits:
        """
        :param lower: The lowest position allowed.
        :param upper: The highest position allowed.
        :param velocity: How fast the degree of freedom may move, either way.
        :return: Limits bounding position and velocity and leaving the higher
            derivatives unbounded, as the repository's own prismatic fixtures do.
        """
        return DegreeOfFreedomLimits(
            lower=DerivativeMap(
                position=lower, velocity=-velocity, acceleration=None, jerk=None
            ),
            upper=DerivativeMap(
                position=upper, velocity=velocity, acceleration=None, jerk=None
            ),
        )

    @classmethod
    def _revolute(
        cls,
        world: World,
        parent: Body,
        child: Body,
        parent_T_connection: HomogeneousTransformationMatrix,
    ) -> RevoluteConnection:
        """
        Add one of the arm's joints, turning about the vertical.

        :param world: The world the joint is added to.
        :param parent: The body the joint turns relative to.
        :param child: The body the joint turns.
        :param parent_T_connection: Where the joint sits on its parent.
        :return: The connection that was added.
        """
        connection = RevoluteConnection.create_with_dofs(
            parent=parent,
            child=child,
            axis=Vector3.Z(),
            world=world,
            parent_T_connection_expression=parent_T_connection,
            dof_limits=cls._limits(-2.6, 2.6, 1.0),
        )
        world.add_connection(connection)
        return connection

    @classmethod
    def _add_drawer(
        cls, world: World, base: Body, cabinet_yaw: float
    ) -> tuple[Body, PrismaticConnection]:
        """
        Add the cabinet, its sliding drawer and the handle on its front.

        The drawer slides away from the arm, so following it eventually runs the arm out
        of reach, which is the conflict a grip's weight has to win or lose.

        :param world: The world the drawer is added to.
        :param base: The body the cabinet is placed relative to.
        :param cabinet_yaw: How far the cabinet is turned about the vertical, in
            radians.
        :return: The handle and the connection the drawer slides along.
        """
        case = cls._box("cabinet", (0.4, 0.5, 0.5))
        world.add_connection(
            FixedConnection(
                parent=base,
                child=case,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=CABINET_DISTANCE, z=ARM_HEIGHT, yaw=cabinet_yaw
                    )
                ),
            )
        )
        front = cls._box("drawer_front", (0.05, 0.45, 0.4))
        mechanism = PrismaticConnection.create_with_dofs(
            parent=case,
            child=front,
            axis=Vector3.X(),
            world=world,
            parent_T_connection_expression=(
                HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.2)
            ),
            dof_limits=cls._limits(0.0, DRAWER_TRAVEL, 0.3),
        )
        world.add_connection(mechanism)
        handle = cls._box("handle", (0.04, 0.2, 0.04))
        world.add_connection(
            FixedConnection(
                parent=front,
                child=handle,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.05)
                ),
            )
        )
        return handle, mechanism
