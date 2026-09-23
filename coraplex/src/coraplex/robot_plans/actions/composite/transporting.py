from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from typing_extensions import Optional, Any

from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    variable,
)
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms
from coraplex.locations.base import DeferredLocation, Location
from coraplex.locations.factories import accessing_location, reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import HasGraspChoice, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TransportAction(ActionDescription, HasGraspChoice, HasApproachesGraspPoses):
    """
    Transports an object to a position using an arm.
    """

    target_location: Pose = field(kw_only=True)
    """
    Target Location to which the object should be transported.

    The navigation this action plans aims at the same grasp the pick-up takes, so a
    caller that worked out which grasp is reachable passes it as :attr:`grasp` and
    both follow it.
    """

    def inside_container(self) -> List[Body]:
        bodies = []
        object_body = self.grasp.graspable.root
        for body in self.world.bodies:
            if body == object_body:
                continue
            if InsideOf(object_body, body).compute_containment_ratio() > 0.9:
                bodies.append(body)
        return bodies

    def _make_open_container_actions(self, container: Body) -> List:
        """
        :param container: The container body in which the object is located.
        :return: The actions needed to open the given container, empty if the container is not a known drawer.
        """
        drawer_annotation = an(
            entity(
                drawer := variable(Drawer, domain=self.world.semantic_annotations)
            ).where(drawer.root == container)
        )
        drawer_annotation = list(drawer_annotation.evaluate())
        if len(drawer_annotation) == 0:
            return []
        handle = drawer_annotation[0].handle

        return [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=accessing_location(
                        container=drawer_annotation[0],
                        context=self.context,
                        arm=ViewManager.get_arm_view(self.arm, self.robot),
                    ),
                ),
                keep_joint_states=True,
            ),
            OpenAction(handle, self.arm),
        ]

    @property
    def _action_plan(self) -> PlanNode:

        children = []
        for container in self.inside_container():
            children.extend(self._make_open_container_actions(container))

        children.extend(
            [
                ParkArmsAction(Arms.BOTH),
                a(NavigateAction)(
                    target_location=variable(
                        Pose, domain=DeferredLocation(self._pick_up_location)
                    ),
                    keep_joint_states=True,
                ),
                a(PickUpAction)(
                    grasp=self.grasp,
                    arm=self.arm,
                    approach_clearance=self.approach_clearance,
                    retreat_distance=self.retreat_distance,
                ),
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                self._make_navigate_action_for_placing(self.grasp),
                a(PlaceAction)(
                    object_designator=self.grasp.graspable,
                    target_location=self.target_location,
                    arm=self.arm,
                ),
                ParkArmsAction(Arms.BOTH),
            ]
        )

        return sequential(children)

    def _pick_up_location(self) -> Location:
        """
        :return: The standing poses from which the arm reaches :attr:`grasp` on
            the object where it is.
        """
        return reachability_location(
            grasp=self.grasp,
            context=self.context,
            arm=ViewManager.get_arm_view(self.arm, self.robot),
            approach_clearance=self.approach_clearance,
            retreat_distance=self.retreat_distance,
        )

    def _make_navigate_action_for_placing(self, chosen_grasp: GraspPose):
        """
        :param chosen_grasp: The grasp the pick-up was told to take. The grasp the gripper
            actually holds the object by is preferred once the navigation runs, since
            the pick-up may have corrected it.
        :return: The navigate action that will be used to place the object.
        """
        return a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: reachability_location(
                        grasp=self._grasp_held_now() or chosen_grasp,
                        context=self.context,
                        arm=ViewManager.get_arm_view(self.arm, self.robot),
                        destination=self.target_location,
                        approach_clearance=self.approach_clearance,
                        retreat_distance=self.retreat_distance,
                    )
                ),
            ),
            keep_joint_states=True,
        )

    def _grasp_held_now(self) -> Optional[GraspPose]:
        """
        :return: The grasp the gripper holds the object by, or ``None``
            when it is not holding it yet.
        """
        held = ViewManager.get_end_effector_view(self.arm, self.robot).grasp_on(
            self.grasp.graspable.root
        )
        if held is None:
            return None
        return GraspPose(self.grasp.graspable, held)


@dataclass
class PickAndPlaceAction(ActionDescription):
    """
    Transports an object to a position using an arm without moving the base of
    the robot.
    """

    graspable_object: HasGraspPoses
    """
    The annotation of the object that should be transported.
    """

    target_location: Pose
    """
    Target Location to which the object should be transported.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                PickUpAction(self.graspable_object.grasp_poses()[0], self.arm),
                ParkArmsAction(Arms.BOTH),
                PlaceAction(self.graspable_object, self.target_location, self.arm),
                ParkArmsAction(Arms.BOTH),
            ]
        )


@dataclass
class MoveAndPlaceAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the target and place the
    object.
    """

    standing_position: Pose
    """
    The pose to stand before trying to pick up the object.
    """
    object_designator: HasGraspPoses
    """
    The annotation of the object to pick up.
    """
    target_location: Pose
    """
    The location to place the object.
    """
    arm: Arms
    """
    The arm to use.
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(self.standing_position, self.keep_joint_states),
                FaceAtAction(self.target_location, self.keep_joint_states),
                PlaceAction(self.object_designator, self.target_location, self.arm),
            ]
        )


@dataclass
class MoveAndPickUpAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the object and pick it
    up.
    """

    standing_position: Pose
    """
    The pose to stand before trying to pick up the object.
    """
    grasp: GraspPose
    """
    The grasp to take hold by, which also names the object to pick up.
    """
    arm: Arms
    """
    The arm to use.
    """
    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(self.standing_position, self.keep_joint_states),
                FaceAtAction(
                    self.grasp.graspable.root.global_pose, self.keep_joint_states
                ),
                PickUpAction(self.grasp, self.arm),
            ]
        )
