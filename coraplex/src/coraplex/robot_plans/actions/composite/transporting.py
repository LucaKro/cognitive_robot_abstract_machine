from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from typing_extensions import Any

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
from coraplex.robot_plans.actions.core.pick_up import HasGraspChoice, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TransportAction(ActionDescription, HasGraspChoice, HasApproachesGraspPoses):
    """
    Transports an object to a position using an arm.

    Where the robot stands for each step is left open and grounded together with the
    step itself, so a standing pose is only taken once the pick-up, place or opening
    from it has been tried.
    """

    target_location: Pose = field(kw_only=True)
    """
    Target Location to which the object should be transported.
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
        return [
            a(MoveAndOpenAction)(
                standing_position=variable(
                    Pose,
                    domain=accessing_location(
                        container=drawer_annotation[0],
                        context=self.context,
                        arm=ViewManager.get_arm_view(self.arm, self.robot),
                    ),
                ),
                handle=drawer_annotation[0].handle,
                arm=self.arm,
                keep_joint_states=True,
            ),
        ]

    @property
    def _action_plan(self) -> PlanNode:

        children = []
        for container in self.inside_container():
            children.extend(self._make_open_container_actions(container))

        children.extend(
            [
                ParkArmsAction(Arms.BOTH),
                a(MoveAndPickUpAction)(
                    standing_position=variable(
                        Pose, domain=DeferredLocation(self._pick_up_location)
                    ),
                    grasp=self.grasp,
                    arm=self.arm,
                    keep_joint_states=True,
                    approach_clearance=self.approach_clearance,
                    retreat_distance=self.retreat_distance,
                ),
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                a(MoveAndPlaceAction)(
                    standing_position=variable(
                        Pose, domain=DeferredLocation(self._place_location)
                    ),
                    object_designator=self.grasp.graspable,
                    target_location=self.target_location,
                    arm=self.arm,
                    keep_joint_states=True,
                ),
                ParkArmsAction(Arms.BOTH),
            ]
        )

        return sequential(children)

    def _pick_up_location(self) -> Location:
        """
        :return: The standing poses around the object where it is.
        """
        return reachability_location(
            target_pose=self.grasp.graspable.root.global_pose,
            context=self.context,
            arm=ViewManager.get_arm_view(self.arm, self.robot),
        )

    def _place_location(self) -> Location:
        """
        :return: The standing poses around :attr:`target_location`.
        """
        return reachability_location(
            target_pose=self.target_location,
            context=self.context,
            arm=ViewManager.get_arm_view(self.arm, self.robot),
        )


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
    Navigate to `standing_position`, facing the target, and place the object.
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
                FaceAtAction(
                    self.target_location,
                    self.keep_joint_states,
                    standing_position=self.standing_position,
                ),
                PlaceAction(self.object_designator, self.target_location, self.arm),
            ]
        )


@dataclass
class MoveAndPickUpAction(ActionDescription, HasApproachesGraspPoses):
    """
    Navigate to `standing_position`, facing the object, and pick it up.
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
                FaceAtAction(
                    self.grasp.graspable.root.global_pose,
                    self.keep_joint_states,
                    standing_position=self.standing_position,
                ),
                PickUpAction(
                    self.grasp,
                    self.arm,
                    approach_clearance=self.approach_clearance,
                    retreat_distance=self.retreat_distance,
                ),
            ]
        )


@dataclass
class MoveAndOpenAction(ActionDescription):
    """
    Navigate to `standing_position`, facing the handle, and open its container.
    """

    standing_position: Pose
    """
    The pose to stand at while opening the container.
    """

    handle: Handle
    """
    The handle of the container to open.
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
                FaceAtAction(
                    self.handle.root.global_pose,
                    self.keep_joint_states,
                    standing_position=self.standing_position,
                ),
                OpenAction(self.handle, self.arm),
            ]
        )
