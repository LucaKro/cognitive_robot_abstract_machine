from __future__ import division

from dataclasses import dataclass, field
from typing import Optional

from krrood.symbolic_math.symbolic_math import trinary_logic_and
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from giskardpy.motion_statechart.beliefs.grasp import GraspBelief
from giskardpy.motion_statechart.beliefs.grasp_weighted_tasks import (
    GraspWeightedCartesianPose,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState


@dataclass(eq=False, repr=False)
class Open(Goal):
    """
    Open a 1-dof mechanism in an environment by driving its degree of freedom towards
    its upper limit while keeping the end effector fixed relative to the grasped part.

    Assumes that the grasped part (e.g. a handle or a bottle cap) has already been
    grasped. Works with any mechanism whose grasped part hangs below an
    :class:`ActiveConnection1DOF`, e.g. drawers, doors, or screw caps.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    End effector that is grasping the handle.
    """

    environment_link: KinematicStructureEntity = field(kw_only=True)
    """
    Name of the handle that was grasped.
    """

    goal_joint_state: Optional[float] = field(default=None, kw_only=True)
    """
    Goal state for the mechanism.

    default is the limit this goal drives towards.
    """

    mechanism_weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Weight of the goal driving the degree of freedom of the mechanism.

    Below collision avoidance, because following a mechanism contorts the arm against
    whatever is around it, and at a higher weight the solver buys the trajectory by
    pushing the arm through what is in its way.
    """

    grasp_weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Weight of the goal keeping the end effector fixed relative to the grasped part.

    Above collision avoidance, because at a lower weight the solver buys clearance by
    letting the end effector drift off the grasped part, and the two move independently.
    """

    grasp_belief: Optional[GraspBelief] = field(default=None, kw_only=True)
    """
    The belief about whether the grasped part is held, which scales
    :attr:`grasp_weight`.

    Left unset, the grip carries that weight unconditionally, which asserts the grasp
    this goal's docstring assumes. Set, a grasp that fails or degrades takes the grip
    with it, so the mechanism is no longer followed against a part that is not held.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.connection = self.environment_link.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        self.goal_joint_state = self._reachable_goal_joint_state()

        self._add_children_to_motion_statechart(
            [
                JointPositionList(
                    name="hinge goal",
                    goal_state=JointState.from_mapping(
                        {self.connection: self.goal_joint_state}
                    ),
                    weight=self.mechanism_weight,
                ),
                self._hold_handle_task(),
            ]
        )

    def _hold_handle_task(self) -> CartesianPose:
        """
        :return: The task keeping the end effector on the grasped part, weighted by the
            belief that it is held where one was given.
        """
        arguments = dict(
            name="hold handle",
            root_link=self.environment_link,
            tip_link=self.tip_link,
            goal_pose=Pose(reference_frame=self.tip_link),
            weight=self.grasp_weight,
        )
        if self.grasp_belief is None:
            return CartesianPose(**arguments)
        return GraspWeightedCartesianPose(**arguments, grasp_belief=self.grasp_belief)

    def _reachable_goal_joint_state(self) -> float:
        """
        :return: The commanded goal state, clamped to the limit this goal drives towards.
        """
        limit = self.connection.dof.limits.upper.position
        if self.goal_joint_state is None:
            return limit
        return min(limit, self.goal_joint_state)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True once both the degree of freedom and the grip
        on the grasped part reached their goals.

        This goal ends neither of them, so a part that keeps running is judged by what
        it observes now and stops counting once it drifts away from its goal again. A
        part something *else* ended keeps counting, because its verdict outlasts it.
        """
        return NodeArtifacts(
            observation=trinary_logic_and(*[node.goal_reached for node in self.nodes])
        )


@dataclass(eq=False, repr=False)
class Close(Open):
    """
    Close a 1-dof mechanism in an environment by driving its degree of freedom towards
    its lower limit while keeping the end effector fixed relative to the grasped part.

    Assumes that the grasped part (e.g. a handle or a bottle cap) has already been
    grasped. Works with any mechanism whose grasped part hangs below an
    :class:`ActiveConnection1DOF`, e.g. drawers, doors, or screw caps.
    """

    def _reachable_goal_joint_state(self) -> float:
        limit = self.connection.dof.limits.lower.position
        if self.goal_joint_state is None:
            return limit
        return max(limit, self.goal_joint_state)
