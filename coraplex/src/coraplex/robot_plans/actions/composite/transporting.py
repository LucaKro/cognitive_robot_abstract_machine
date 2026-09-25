from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from typing_extensions import Any, Self

from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    variable,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ReachFraction
from coraplex.exceptions import NothingToPlace
from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.query.match import Match
from krrood.patterns.field_metadata import JSONMetadata
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
class BoundsItsCandidates:
    """
    Adds a limit on how many candidates a step tries.

    A candidate is tried by running the step with it, so a step that succeeds with none
    would otherwise try every one it is offered.
    """

    candidates_to_try: int = field(default=50, kw_only=True)
    """
    How many candidates a step tries before giving up.
    """

    def _bound_candidates(self, *steps: Any) -> None:
        """
        Limit every step that tries candidates to :attr:`candidates_to_try` of them.

        :param steps: The steps.
        """
        for step in steps:
            if isinstance(step, Match):
                step.expression.limit(self.candidates_to_try)


@dataclass
class TransportAction(ActionDescription, BoundsItsCandidates):
    """
    Picks an object up with one step and puts it down with another.
    """

    pick_up: MoveAndPickUpAction = field(
        metadata=JSONMetadata(serialize=False).as_dict()
    )
    """
    The step that picks the object up.
    """

    place: MoveAndPlaceAction = field(metadata=JSONMetadata(serialize=False).as_dict())
    """
    The step that puts down what :attr:`pick_up` picked up.
    """

    @classmethod
    def from_grasp(
        cls, grasp: GraspPose, target_location: Pose, arm: Arms, context: Context
    ) -> Self:
        """
        A transport that takes an object by `grasp` to `target_location`, standing
        wherever each step can be carried out from.

        :param grasp: The grasp to take the object by.
        :param target_location: Where to put the object down.
        :param arm: The arm that carries the object.
        :param context: The context the standing poses are drawn in.
        :return: The transport, standing near the object to pick it up and near the
            target to place it.
        """
        arm_view = ViewManager.get_arm_view(arm, context.robot)
        return cls(
            pick_up=a(MoveAndPickUpAction)(
                standing_position=variable(
                    Pose,
                    domain=ReachabilityLocation(
                        Pose(reference_frame=grasp.graspable.root),
                        arm_view,
                        context=context,
                    ),
                ),
                grasp=grasp,
                arm=arm,
            ),
            place=a(MoveAndPlaceAction)(
                standing_position=variable(
                    Pose,
                    domain=ReachabilityLocation(
                        target_location, arm_view, context=context
                    ),
                ),
                target_location=target_location,
                arm=arm,
            ),
        )

    @property
    def _action_plan(self) -> PlanNode:
        self._bound_candidates(self.pick_up, self.place)
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                self.pick_up,
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                self.place,
                ParkArmsAction(Arms.BOTH),
            ]
        )


@dataclass
class PickAndPlaceAction(ActionDescription, BoundsItsCandidates):
    """
    Picks an object up with one step and puts it down with another, without moving the
    base of the robot.
    """

    pick_up: PickUpAction = field(metadata=JSONMetadata(serialize=False).as_dict())
    """
    The step that picks the object up.
    """

    place: PlaceAction = field(metadata=JSONMetadata(serialize=False).as_dict())
    """
    The step that puts down what :attr:`pick_up` picked up.
    """

    @property
    def _action_plan(self) -> PlanNode:
        self._bound_candidates(self.pick_up, self.place)
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                self.pick_up,
                ParkArmsAction(Arms.BOTH),
                self.place,
                ParkArmsAction(Arms.BOTH),
            ]
        )


@dataclass
class MoveAndPlaceAction(ActionDescription):
    """
    Navigate to `standing_position`, facing the target, and place what the arm holds.
    """

    standing_position: Pose
    """
    The pose to stand at while placing.
    """

    target_location: Pose
    """
    The location to place the object.
    """

    arm: Arms
    """
    The arm that holds the object.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                FaceAtAction(
                    self.target_location,
                    standing_position=self.standing_position,
                ),
                PlaceAction(self._placed_object(), self.target_location, self.arm),
            ]
        )

    def _placed_object(self) -> HasGraspPoses:
        """
        The object :attr:`arm` puts down.

        A plan is built before it runs, so when the arm holds nothing yet, the object is
        the one the pick-up before this step is going to give it.

        :return: An annotation of the object held, or about to be held. Any annotation
            of the held body names the same object to put down.
        :raises NothingToPlace: If the arm holds nothing and no pick-up precedes this
            step.
        """
        held_body = ViewManager.get_end_effector_view(self.arm, self.robot).held_body
        if held_body is not None:
            return next(
                annotation
                for annotation in self.world.get_semantic_annotations_by_type(
                    HasGraspPoses
                )
                if annotation.root is held_body
            )
        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        if previous_pick is None:
            raise NothingToPlace(self.arm)
        return previous_pick.designator.grasp.graspable


@dataclass
class MoveAndPickUpAction(
    ActionDescription, HasApproachesGraspPoses, BoundsItsCandidates
):
    """
    Navigate to `standing_position`, facing the object, and pick it up, opening the
    drawer it is in first.
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

    @property
    def _action_plan(self) -> PlanNode:
        children = []
        for container in self._containers_around_the_object():
            children.extend(self._make_open_container_actions(container))
        children.extend(
            [
                FaceAtAction(
                    self.grasp.graspable.root.global_pose,
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
        return sequential(children)

    def _containers_around_the_object(self) -> List[Body]:
        """
        :return: The bodies the object to pick up lies inside of.
        """
        object_body = self.grasp.graspable.root
        return [
            body
            for body in self.world.bodies
            if body != object_body
            and InsideOf(object_body, body).compute_containment_ratio() > 0.9
        ]

    def _make_open_container_actions(self, container: Body) -> List[Match]:
        """
        :param container: A body the object lies inside of.
        :return: The step opening it, from a standing pose tried together with the
            opening, or nothing if the container is not a known drawer.
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
        open_the_drawer = a(MoveAndOpenAction)(
            standing_position=variable(
                Pose,
                domain=ReachabilityLocation(
                    Pose(reference_frame=handle.root),
                    ViewManager.get_arm_view(self.arm, self.robot),
                    ReachFraction.ACCESSING,
                    context=self.context,
                ),
            ),
            handle=handle,
            arm=self.arm,
        )
        self._bound_candidates(open_the_drawer)
        return [open_the_drawer]


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

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                FaceAtAction(
                    self.handle.root.global_pose,
                    standing_position=self.standing_position,
                ),
                OpenAction(self.handle, self.arm),
            ]
        )
