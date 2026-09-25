"""
How a transport moves to an object, fetches it and puts it down.
"""

import numpy as np
import pytest
from typing_extensions import Callable, List, Type

from krrood.entity_query_language.factories import a, variable
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ReachFraction
from coraplex.exceptions import NothingToPlace
from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.factories import sequential
from coraplex.plans.underspecified import UnderspecifiedNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    PickAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Milk,
)
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% where the robot stands is tried together with what it does there


def _underspecified_steps(transport: TransportAction) -> List[Type[ActionDescription]]:
    """
    :return: The action types of the steps the transport leaves to be grounded, in
        order.
    """
    return [
        child.designator_type
        for child in transport._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ]


def _pick_up_the_milk(world: World, context: Context) -> MoveAndPickUpAction:
    """
    :return: A pick-up of the milk, standing wherever its trial finds one that works.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    return a(MoveAndPickUpAction)(
        standing_position=variable(
            Pose,
            domain=ReachabilityLocation(
                Pose(reference_frame=milk.root),
                context.robot.right_arm,
                context=context,
            ),
        ),
        grasp=milk.grasp_poses()[0],
        arm=context.robot.right_arm,
    )


def _place_at(target: Pose, context: Context) -> MoveAndPlaceAction:
    """
    :return: A place at `target`, standing wherever its trial finds one that works.
    """
    return a(MoveAndPlaceAction)(
        standing_position=variable(
            Pose,
            domain=ReachabilityLocation(
                target,
                context.robot.right_arm,
                context=context,
            ),
        ),
        target_location=target,
        arm=context.robot.right_arm,
    )


def _transport_of_the_milk(world: World, context: Context) -> TransportAction:
    return TransportAction(
        pick_up=_pick_up_the_milk(world, context),
        place=_place_at(Pose(reference_frame=world.root), context),
    )


def test_a_transport_grounds_the_steps_it_is_given(mutable_model_world):
    """
    The caller decides what is left open in each step, so the transport grounds the
    steps it was given rather than steps of its own.
    """
    world, robot, context = mutable_model_world
    transport = _transport_of_the_milk(world, context)
    sequential([transport], context)

    assert _underspecified_steps(transport) == [
        MoveAndPickUpAction,
        MoveAndPlaceAction,
    ]


def test_a_transport_tries_a_bounded_number_of_candidates(mutable_model_world):
    """
    Each standing pose is tried by running the step from it, so a step that can succeed
    from nowhere has to give up after a fixed number of them.
    """
    world, robot, context = mutable_model_world
    transport = _transport_of_the_milk(world, context)
    sequential([transport], context)

    limits = [
        child.underspecified_action.expression._limit_
        for child in transport._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ]

    assert limits == [transport.candidates_to_try] * len(limits)
    assert limits


def test_a_transport_from_a_grasp_stands_around_the_object_then_the_target(
    mutable_model_world,
):
    """
    Built from a grasp alone, a transport leaves only where the robot stands open: close
    to the object for the pick-up, and close to the target for the place.
    """
    world, robot, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root)

    transport = TransportAction.from_grasp(
        milk.grasp_poses()[0], target, context.robot.right_arm, context
    )

    pick_up_location = transport.pick_up.kwargs["standing_position"]._domain_.domain
    place_location = transport.place.kwargs["standing_position"]._domain_.domain
    assert pick_up_location.target_pose.reference_frame is milk.root
    assert place_location.target_pose is target


# %% picking up and placing without moving


def _pick_and_place_of_the_milk(world: World, arm: Arm) -> PickAndPlaceAction:
    """
    :param arm: The arm that picks the milk up and puts it down.
    :return: A pick-and-place of the milk that tries every grasp it offers.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    return PickAndPlaceAction(
        pick_up=a(PickUpAction)(
            grasp=variable(GraspPose, domain=milk.grasp_poses()), arm=arm
        ),
        place=a(PlaceAction)(
            object_designator=milk,
            target_location=Pose(reference_frame=world.root),
            arm=arm,
        ),
    )


def test_a_pick_and_place_grounds_the_steps_it_is_given(mutable_model_world):
    world, robot, context = mutable_model_world
    pick_and_place = _pick_and_place_of_the_milk(world, robot.right_arm)
    sequential([pick_and_place], context)

    assert [
        child.designator_type
        for child in pick_and_place._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ] == [PickUpAction, PlaceAction]


def test_a_pick_and_place_tries_a_bounded_number_of_candidates(mutable_model_world):
    world, robot, context = mutable_model_world
    pick_and_place = _pick_and_place_of_the_milk(world, robot.right_arm)
    sequential([pick_and_place], context)

    limits = [
        child.underspecified_action.expression._limit_
        for child in pick_and_place._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ]

    assert limits == [pick_and_place.candidates_to_try] * len(limits)
    assert limits


# %% fetching an object out of a drawer

DRAWER = "cabinet10_drawer_top"
"""
The apartment drawer the transport opens on its way to the object inside it.
"""

DRAWER_HANDLE = "handle_cab10_t"
"""
The handle of :data:`DRAWER`.
"""


def _pick_up_near_a_drawer(world: World, context: Context) -> MoveAndPickUpAction:
    """
    :return: A pick-up of the milk in a world where :data:`DRAWER` is annotated.
    """
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(
                root=world.get_body_by_name(DRAWER),
                handle=Handle(root=world.get_body_by_name(DRAWER_HANDLE)),
            )
        )
    move_and_pick_up = MoveAndPickUpAction(
        standing_position=Pose(reference_frame=world.root),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        arm=context.robot.right_arm,
    )
    sequential([move_and_pick_up], context)
    return move_and_pick_up


def test_opening_a_container_on_the_way_is_tried_with_the_move_to_it(
    mutable_model_world,
):
    """
    An object inside a drawer is fetched by opening the drawer first, from a standing
    pose of its own.
    """
    world, robot, context = mutable_model_world
    move_and_pick_up = _pick_up_near_a_drawer(world, context)

    assert [
        action.type
        for action in move_and_pick_up._make_open_container_actions(
            world.get_body_by_name(DRAWER)
        )
    ] == [MoveAndOpenAction]


def test_opening_a_container_on_the_way_stands_where_it_is_opened_from(
    mutable_model_world,
):
    """
    The robot stands back for opening a container the way it does for any container,
    rather than as close as it would to grasp something that stays put.
    """
    world, robot, context = mutable_model_world
    move_and_pick_up = _pick_up_near_a_drawer(world, context)

    [open_on_the_way] = move_and_pick_up._make_open_container_actions(
        world.get_body_by_name(DRAWER)
    )
    location = open_on_the_way.kwargs["standing_position"]._domain_.domain

    assert location.reach_fraction == ReachFraction.ACCESSING


# %% moving to an object and picking it up


def test_move_and_pick_up_takes_the_grasp_it_was_given(mutable_model_world):
    """
    The caller chooses the grasp, so the pick-up at the end of the walk takes that one
    rather than whichever grasp the object happens to list first.
    """
    world, robot, context = mutable_model_world
    grasp = world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[-1]
    move_and_pick_up = MoveAndPickUpAction(
        standing_position=Pose(reference_frame=world.root),
        grasp=grasp,
        arm=context.robot.left_arm,
    )
    sequential([move_and_pick_up], context)

    pick_ups = [
        child
        for child in move_and_pick_up._action_plan.children
        if isinstance(getattr(child, "designator", None), PickUpAction)
    ]

    assert [pick_up.designator.grasp for pick_up in pick_ups] == [grasp]


def test_move_and_pick_up_approaches_with_the_clearances_it_was_given(
    mutable_model_world,
):
    world, robot, context = mutable_model_world
    move_and_pick_up = MoveAndPickUpAction(
        standing_position=Pose(reference_frame=world.root),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        arm=context.robot.left_arm,
        approach_clearance=0.07,
        retreat_distance=0.13,
    )
    sequential([move_and_pick_up], context)

    [pick_up] = [
        child.designator
        for child in move_and_pick_up._action_plan.children
        if isinstance(getattr(child, "designator", None), PickUpAction)
    ]

    assert (pick_up.approach_clearance, pick_up.retreat_distance) == (
        move_and_pick_up.approach_clearance,
        move_and_pick_up.retreat_distance,
    )


# %% placing what the arm holds


def _hold_the_milk(world: World, arm: Arm) -> Milk:
    """
    Put the milk in the gripper of `arm`, as a pick-up does.

    :return: The milk.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    tool_frame = arm.end_effector.tool_frame
    with world.modify_world():
        world.move_branch_with_fixed_connection(milk.root, tool_frame)
    return milk


def _placed_object(move_and_place: MoveAndPlaceAction):
    """
    :return: The object the place at the end of `move_and_place` puts down, once the
        plan it belongs to is expanded the way it is before it runs.
    """
    plan = move_and_place.plan_node.plan
    plan.root.notify()
    [place] = plan.get_nodes_by_designator_type(PlaceAction)
    return place.designator.object_designator


def test_a_move_and_place_places_what_the_arm_holds(mutable_model_world):
    world, robot, context = mutable_model_world
    milk = _hold_the_milk(world, context.robot.left_arm)
    move_and_place = MoveAndPlaceAction(
        standing_position=Pose(reference_frame=world.root),
        target_location=Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        arm=context.robot.left_arm,
    )
    sequential([move_and_place], context)

    assert _placed_object(move_and_place) is milk


def test_a_move_and_place_places_a_held_body_that_has_several_annotations(
    mutable_model_world,
):
    """
    A body can be described by more than one annotation that offers grasps, and any of
    them names the body to put down.
    """
    world, robot, context = mutable_model_world
    milk = _hold_the_milk(world, context.robot.left_arm)
    with world.modify_world():
        world.add_semantic_annotation(HasGraspPoses(root=milk.root))
    move_and_place = MoveAndPlaceAction(
        standing_position=Pose(reference_frame=world.root),
        target_location=Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        arm=context.robot.left_arm,
    )
    sequential([move_and_place], context)

    assert _placed_object(move_and_place).root is milk.root


def test_a_move_and_place_after_a_pick_up_places_what_it_picks_up(
    mutable_model_world,
):
    """
    A plan is built before it runs, so a place that follows a pick-up in one plan puts
    down what that pick-up is going to take.
    """
    world, robot, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    move_and_place = MoveAndPlaceAction(
        standing_position=Pose(reference_frame=world.root),
        target_location=Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        arm=context.robot.left_arm,
    )
    sequential(
        [
            MoveAndPickUpAction(
                standing_position=Pose(reference_frame=world.root),
                grasp=milk.grasp_poses()[0],
                arm=context.robot.left_arm,
            ),
            move_and_place,
        ],
        context,
    )

    assert _placed_object(move_and_place) is milk


def test_a_move_and_place_with_nothing_to_place_is_refused(mutable_model_world):
    world, robot, context = mutable_model_world
    move_and_place = MoveAndPlaceAction(
        standing_position=Pose(reference_frame=world.root),
        target_location=Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        arm=context.robot.left_arm,
    )
    sequential([move_and_place], context)

    with pytest.raises(NothingToPlace):
        move_and_place._action_plan


# %% a move-and-act step acts from where it moved to

STANDING_POSITION = (3.5, 1.5)
"""
Where the move-and-act steps are sent, away from where the robot starts.
"""


def _navigation_targets(action: ActionDescription) -> List[Pose]:
    """
    :return: Every standing pose `action` navigates to, including the ones of the
        actions it is built from.
    """
    action.plan_node.notify()
    return [
        node.designator.target_location
        for node in action.plan_node.plan.get_nodes_by_designator_type(NavigateAction)
    ]


def _standing_pose(world: World) -> Pose:
    return Pose.from_xyz_rpy(*STANDING_POSITION, 0.0, reference_frame=world.root)


def _placing_the_held_milk(world: World, context: Context) -> MoveAndPlaceAction:
    _hold_the_milk(world, context.robot.left_arm)
    return MoveAndPlaceAction(
        standing_position=_standing_pose(world),
        target_location=Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        arm=context.robot.left_arm,
    )


MOVE_AND_ACT_STEPS = {
    "pick up": lambda world, context: MoveAndPickUpAction(
        standing_position=_standing_pose(world),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        arm=context.robot.left_arm,
    ),
    "place": _placing_the_held_milk,
}


@pytest.mark.parametrize("build", MOVE_AND_ACT_STEPS.values(), ids=MOVE_AND_ACT_STEPS)
def test_a_move_and_act_step_only_ever_stands_where_it_was_sent(
    mutable_model_world, build: Callable[[World, Context], ActionDescription]
):
    """
    Its plan is built before the robot moves, so turning to face the target has to be
    worked out from where the robot is sent rather than from where it stands at first,
    or the robot is sent back there before it acts.
    """
    world, robot, context = mutable_model_world
    step = build(world, context)
    sequential([step], context)

    for target in _navigation_targets(step):
        np.testing.assert_allclose(
            target.to_position().to_np()[:2].ravel(), STANDING_POSITION
        )


def test_facing_from_a_standing_position_turns_towards_the_target(mutable_model_world):
    world, robot, context = mutable_model_world
    target = Pose.from_xyz_rpy(4.0, 2.5, 0.9, reference_frame=world.root)
    face_at = FaceAtAction(target, standing_position=_standing_pose(world))
    sequential([face_at], context)

    [turn] = _navigation_targets(face_at)

    heading = turn.to_rotation_matrix().to_np()[:2, 0]
    towards_target = target.to_position().to_np()[:2].ravel() - np.array(
        STANDING_POSITION
    )
    np.testing.assert_allclose(
        heading, towards_target / np.linalg.norm(towards_target), atol=1e-6
    )
