from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from typing_extensions import List

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.beliefs.grasp import GraspBelief
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import ConvergingTask, MotionStatechartNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPosition,
)

# %% carrying as much weight as a grasp is believed in


@dataclass(eq=False, repr=False)
class GraspWeightedTask(ConvergingTask, ABC):
    """
    A task that carries as much of its weight as a body is currently believed to be
    held.

    The solver stops buying a constraint the moment the grasp behind it stops being
    credible, which is what lets a motion that depends on holding something give way on
    its own rather than being driven until something else notices.
    """

    grasp_belief: GraspBelief = field(kw_only=True)
    """
    The belief about whether the body is held, whose probability the weight follows.
    """

    @property
    def prerequisite_nodes(self) -> List[MotionStatechartNode]:
        """
        :return: The belief, which publishes the probability only while it builds.
        """
        return [self.grasp_belief]

    @property
    def constraint_weight(self) -> sm.ScalarData:
        """
        :return: The weight, scaled by how likely a grasp currently is, so a ruled-out
            grasp leaves the constraint carrying nothing.
        """
        return self.weight * self.grasp_belief.probability


@dataclass(eq=False, repr=False)
class GraspWeightedCartesianPosition(GraspWeightedTask, CartesianPosition):
    """
    Holds a tip link at a goal position for as long as a grasp is believed in.
    """


@dataclass(eq=False, repr=False)
class GraspWeightedCartesianOrientation(GraspWeightedTask, CartesianOrientation):
    """
    Holds a tip link at a goal orientation for as long as a grasp is believed in.
    """


@dataclass(eq=False, repr=False)
class GraspWeightedCartesianPose(CartesianPose):
    """
    Holds a tip link at a goal pose for as long as a grasp is believed in.

    Position and orientation stay separate tasks, as they are without a belief, and both
    follow the same probability.
    """

    grasp_belief: GraspBelief = field(kw_only=True)
    """
    The belief about whether the body is held, which both halves weigh themselves by.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Builds the grip from halves that read the belief, in place of the plain ones.

        :param context: Provides the world the root link defaults to.
        """
        if self.root_link is None:
            self.root_link = context.world.root
        self.nodes = [
            GraspWeightedCartesianPosition(
                name=f"{self.name}/position",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_point=self.goal_pose.to_position(),
                reference_velocity=self.reference_linear_velocity,
                threshold=self.translation_threshold,
                weight=self.weight,
                binding_policy=self.binding_policy,
                grasp_belief=self.grasp_belief,
            ),
            GraspWeightedCartesianOrientation(
                name=f"{self.name}/orientation",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_orientation=self.goal_pose.to_rotation_matrix(),
                reference_velocity=self.reference_angular_velocity,
                threshold=self.orientation_threshold,
                weight=self.weight,
                binding_policy=self.binding_policy,
                grasp_belief=self.grasp_belief,
            ),
        ]
        # Skips CartesianPose.expand, which would overwrite the halves just built with
        # ones that do not read the belief, and resumes at Parallel.
        super(CartesianPose, self).expand(context)
