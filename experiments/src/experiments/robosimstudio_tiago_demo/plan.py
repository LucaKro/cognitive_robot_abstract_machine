"""
What the TIAGo does: carry the milk from the counter to the kitchen island.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoMoveSim
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import IsAmongTheClosestGraspsTo
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose

from experiments.robosimstudio_tiago_demo.world import TiagoApartment, UprightPose

# %% transport


@dataclass(frozen=True)
class MilkTransport:
    """
    The TIAGo carries the milk from the counter to the kitchen island with its right
    arm.
    """

    pick_up_stand: UprightPose = UprightPose(1.419, 1.934, yaw=0.570)
    """
    Where the TIAGo stands to pick the milk up: the pose
    :class:`~coraplex.robot_plans.actions.composite.transporting.TransportAction`'s own
    search chose for this world in a kinematically simulated run.
    """

    turning_point: UprightPose = UprightPose(4.6, 2.0, yaw=-2.981)
    """
    Where the TIAGo turns to face the island before driving up to it, far enough from
    the island that the milk it holds at its side clears it.
    """

    place_stand: UprightPose = UprightPose(3.828, 1.996, yaw=-2.981)
    """
    Where the TIAGo stands to put the milk down, as far from :attr:`place_target` as the
    pose the same search chose for a target 20 cm closer to the island's edge.
    """

    place_target: UprightPose = UprightPose(2.9, 2.2, 1.04, yaw=math.pi)
    """
    Where the milk is put down on the island, turned half a turn: just above where it
    rests on the island's top.
    """

    sampling_seed: int = 0
    """
    The seed anything the plan leaves open is sampled with, so a run is repeatable.
    """

    def context(self, apartment: TiagoApartment) -> Context:
        """
        :param apartment: The world the plan is performed in.
        :return: The context the plan is built and performed in, driving the TIAGo's
            differential base as such.
        """
        return Context(
            apartment.world,
            apartment.robot,
            alternative_motion_mappings=[TiagoMoveSim],
            sampling_seed=self.sampling_seed,
        )

    def plan(self, apartment: TiagoApartment, context: Context) -> PlanNode:
        """
        Pick the milk up, raise the torso again before parking the arms with the milk in
        hand so it clears the counter's edge, carry it over and put it down.

        :param apartment: The world the plan is performed in.
        :param context: The context to build the plan in.
        :return: The plan.
        """
        root = apartment.world.root
        milk = apartment.world.get_semantic_annotations_by_type(Milk)[0]
        arms = apartment.robot.all_arms
        pick_up_stand = self.pick_up_stand.to_pose(root)
        return sequential(
            [
                ParkArmsAction(arms),
                MoveTorsoAction(TorsoState.HIGH),
                NavigateAction(pick_up_stand),
                PickUpAction(
                    self.closest_grasp(milk, pick_up_stand),
                    apartment.robot.get_right_arm_if_specified(),
                ),
                MoveTorsoAction(TorsoState.HIGH),
                ParkArmsAction(arms),
                NavigateAction(self.turning_point.to_pose(root)),
                NavigateAction(self.place_stand.to_pose(root)),
                PlaceAction(
                    object_designator=milk,
                    target_location=self.place_target.to_pose(root),
                ),
                ParkArmsAction(arms),
            ],
            context,
        )

    @staticmethod
    def closest_grasp(milk: Milk, standing_position: Pose) -> GraspCandidate:
        """
        :param milk: The milk to grasp.
        :param standing_position: Where the robot stands while grasping.
        :return: The grasp on the milk closest to where the robot stands.
        """
        grasps = milk.grasp_candidates()
        [grasp] = [
            grasp
            for grasp in grasps
            if IsAmongTheClosestGraspsTo(
                grasp, standing_position, grasps, number_of_grasps=1
            )()
        ]
        return grasp
