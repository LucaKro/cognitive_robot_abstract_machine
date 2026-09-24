"""
How a transport moves to an object, fetches it and puts it down.
"""

import numpy as np
import pytest
from typing_extensions import Callable, List, Type

from coraplex.datastructures.enums import Arms, ReachFraction
from coraplex.locations import factories
from coraplex.plans.factories import sequential
from coraplex.plans.underspecified import UnderspecifiedNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Milk,
)
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


def test_a_transport_tries_each_standing_pose_with_what_it_does_there(
    mutable_model_world,
):
    """
    A standing pose is only known to work once the pick-up or place from it has been
    tried, so each is grounded together with the move to it.
    """
    world, robot, context = mutable_model_world
    transport = TransportAction(
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        target_location=Pose(reference_frame=world.root),
        arm=Arms.RIGHT,
    )
    sequential([transport], context)

    assert _underspecified_steps(transport) == [
        MoveAndPickUpAction,
        MoveAndPlaceAction,
    ]


# %% fetching an object out of a drawer

DRAWER = "cabinet10_drawer_top"
"""
The apartment drawer the transport opens on its way to the object inside it.
"""

DRAWER_HANDLE = "handle_cab10_t"
"""
The handle of :data:`DRAWER`.
"""


def test_opening_a_container_on_the_way_stands_where_it_is_opened_from(
    mutable_model_world, monkeypatch
):
    """
    An object inside a drawer is fetched by opening the drawer first, and the robot
    stands back for that the way it does for any container rather than as close as it
    would to grasp something that stays put.
    """
    world, robot, context = mutable_model_world
    drawer_body = world.get_body_by_name(DRAWER)
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(
                root=drawer_body,
                handle=Handle(root=world.get_body_by_name(DRAWER_HANDLE)),
            )
        )
    transport = TransportAction(
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        target_location=Pose(reference_frame=world.root),
        arm=Arms.RIGHT,
    )
    sequential([transport], context)

    asked_for = {}
    build_location = factories.reachability_location
    monkeypatch.setattr(
        factories,
        "reachability_location",
        lambda *args, **kwargs: asked_for.update(kwargs)
        or build_location(*args, **kwargs),
    )

    transport._make_open_container_actions(drawer_body)

    assert asked_for["reach_fraction"] == ReachFraction.ACCESSING


def test_opening_a_container_on_the_way_is_tried_with_the_move_to_it(
    mutable_model_world,
):
    world, robot, context = mutable_model_world
    drawer_body = world.get_body_by_name(DRAWER)
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(
                root=drawer_body,
                handle=Handle(root=world.get_body_by_name(DRAWER_HANDLE)),
            )
        )
    transport = TransportAction(
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        target_location=Pose(reference_frame=world.root),
        arm=Arms.RIGHT,
    )
    sequential([transport], context)

    assert [
        action.type for action in transport._make_open_container_actions(drawer_body)
    ] == [MoveAndOpenAction]


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
        arm=Arms.LEFT,
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
        arm=Arms.LEFT,
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


MOVE_AND_ACT_STEPS = {
    "pick up": lambda world: MoveAndPickUpAction(
        standing_position=_standing_pose(world),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        arm=Arms.LEFT,
    ),
    "place": lambda world: MoveAndPlaceAction(
        standing_position=_standing_pose(world),
        object_designator=world.get_semantic_annotations_by_type(Milk)[0],
        target_location=Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        arm=Arms.LEFT,
    ),
}


@pytest.mark.parametrize("build", MOVE_AND_ACT_STEPS.values(), ids=MOVE_AND_ACT_STEPS)
def test_a_move_and_act_step_only_ever_stands_where_it_was_sent(
    mutable_model_world, build: Callable[[World], ActionDescription]
):
    """
    Its plan is built before the robot moves, so turning to face the target has to be
    worked out from where the robot is sent rather than from where it stands at first,
    or the robot is sent back there before it acts.
    """
    world, robot, context = mutable_model_world
    step = build(world)
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
