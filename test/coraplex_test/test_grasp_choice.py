"""
How a pick-up settles on the grasp it takes.
"""

import numpy as np

from krrood.entity_query_language.factories import evaluate_condition
from coraplex.datastructures.enums import Arms
from coraplex.locations.pose_validator import AreReachableBy
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose


def _reach_of(pick_up: PickUpAction) -> ReachAction:
    """
    :return: The reach the pick-up's plan performs.
    """
    [reach_node] = pick_up._action_plan.plan.get_nodes_by_designator_type(ReachAction)
    return reach_node.designator


def test_pick_up_takes_the_grasp_it_is_given(immutable_model_world):
    """
    A caller that settled on a grasp -- together with the pose the robot stands at, say
    -- has the pick-up take that one instead of ranking the object's grasps again.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root)

    pick_up = PickUpAction(milk, Arms.LEFT, grasp_pose=given)
    sequential([pick_up], context=context)

    assert pick_up.chosen_grasp_pose is given


def test_pick_up_lets_the_gripper_choose_when_given_none(immutable_model_world):
    """
    Without a grasp the pick-up takes the one the gripper ranks first, which is what a
    caller that does not know where the object may be held wants.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    pick_up = PickUpAction(milk, Arms.LEFT)
    sequential([pick_up], context=context)

    expected = ViewManager.get_end_effector_view(
        Arms.LEFT, context.robot
    ).grasp_poses_by_distance(milk)[0]
    np.testing.assert_allclose(
        pick_up.chosen_grasp_pose.to_homogeneous_matrix().to_np(),
        expected.to_homogeneous_matrix().to_np(),
    )


def test_pick_up_reaches_for_the_grasp_it_settled_on(immutable_model_world):
    """
    The grasp the pick-up chose is the one its plan reaches for, so a caller's choice
    reaches the motions rather than stopping at the action.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root)

    pick_up = PickUpAction(milk, Arms.LEFT, grasp_pose=given)
    sequential([pick_up], context=context)

    assert _reach_of(pick_up).grasp_pose is given


def test_pre_condition_checks_only_the_grasp_it_was_given(immutable_model_world):
    """
    A caller that named a grasp is asking for that grasp, so the pre-condition fails
    when it cannot be reached even though the object offers others that can be.

    Taking one of those instead would be performing a different action than the one
    described.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )
    end_effector = ViewManager.get_end_effector_view(Arms.LEFT, view)
    unreachable = end_effector.grasp_poses_by_distance(milk)[0]

    pick_up = PickUpAction(milk, Arms.LEFT, grasp_pose=unreachable)
    sequential([pick_up], context=context)

    assert not evaluate_condition(
        PickUpAction.pre_condition(
            pick_up.bound_variables, context, pick_up.designator_parameter
        )
    )


def test_pre_condition_takes_any_grasp_when_given_none(immutable_model_world):
    """
    A caller that named no grasp is asking for the object to be picked up however it
    can be, so the pre-condition holds as long as some grasp is reachable -- even when
    the one the gripper ranks first is not.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pick_up = PickUpAction(milk, Arms.LEFT)
    sequential([pick_up], context=context)

    assert evaluate_condition(
        PickUpAction.pre_condition(
            pick_up.bound_variables, context, pick_up.designator_parameter
        )
    )


def test_pick_up_reaches_for_a_grasp_it_can_perform(immutable_model_world):
    """
    The grasp the plan is built around is the one the pre-condition accepted, so a
    pick-up does not set off towards a grasp its own check just rejected.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )
    end_effector = ViewManager.get_end_effector_view(Arms.LEFT, view)
    ranked_first = end_effector.grasp_poses_by_distance(milk)[0]

    pick_up = PickUpAction(milk, Arms.LEFT)
    sequential([pick_up], context=context)

    assert not np.allclose(
        pick_up.chosen_grasp_pose.to_homogeneous_matrix().to_np(),
        ranked_first.to_homogeneous_matrix().to_np(),
    ), "the best-ranked grasp is unreachable here, so it must not be the one chosen"
    assert AreReachableBy.for_grasp(
        pick_up.chosen_grasp_pose,
        end_effector,
        pick_up.chosen_grasp_pose,
        context=context,
    )()
