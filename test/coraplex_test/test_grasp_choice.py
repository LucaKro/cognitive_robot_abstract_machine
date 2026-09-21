"""
How a pick-up settles on the grasp it takes.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from krrood.entity_query_language.factories import evaluate_condition, variable
from coraplex.datastructures.enums import Arms
from coraplex.locations import factories
from coraplex.locations.pose_validator import AreReachableBy
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import GraspPose
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def _reach_of(pick_up: PickUpAction) -> ReachAction:
    """
    :return: The reach the pick-up's plan performs.

    A pick-up reaches through the grasp it is built from, so the reach only appears
    once the plan below it has been expanded.
    """
    pick_up.plan_node.notify()
    [reach_node] = pick_up.plan_node.plan.get_nodes_by_designator_type(ReachAction)
    return reach_node.designator


def test_pick_up_takes_the_grasp_it_is_given(immutable_model_world):
    """
    A caller that settled on a grasp -- together with the pose the robot stands at, say
    -- has the pick-up take that one instead of ranking the object's grasps again.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = GraspPose(milk, Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root))

    pick_up = PickUpAction(given, Arms.LEFT)
    sequential([pick_up], context=context)

    assert pick_up.grasp is given


def test_pick_up_reaches_for_the_grasp_it_settled_on(immutable_model_world):
    """
    The grasp the pick-up chose is the one its plan reaches for, so a caller's choice
    reaches the motions rather than stopping at the action.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = GraspPose(milk, Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root))

    pick_up = PickUpAction(given, Arms.LEFT)
    sequential([pick_up], context=context)

    assert _reach_of(pick_up).grasp is given


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
    unreachable = end_effector.grasp_poses_by_distance(
        milk, context.motion_tolerances.default_tcp_position_threshold
    )[0]

    pick_up = PickUpAction(unreachable, Arms.LEFT)
    sequential([pick_up], context=context)

    assert not evaluate_condition(
        PickUpAction.pre_condition(
            pick_up.bound_variables, context, pick_up.designator_parameter
        )
    )


def test_pre_condition_judges_the_grasp_the_action_takes(immutable_model_world):
    """
    The pre-condition asks about the grasp the action takes.

    That the object offers others that could be reached is not the question, because the
    action would not take them.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pick_up = PickUpAction(milk.grasp_poses()[0], Arms.LEFT)
    sequential([pick_up], context=context)
    reaches_its_grasp = AreReachableBy.for_grasp(
        pick_up.grasp,
        ViewManager.get_arm_view(Arms.LEFT, view),
        context=context,
    )()

    assert (
        evaluate_condition(
            PickUpAction.pre_condition(
                pick_up.bound_variables, context, pick_up.designator_parameter
            )
        )
        is reaches_its_grasp
    )


def test_pick_up_keeps_its_grasp_even_when_it_cannot_be_reached(immutable_model_world):
    """
    The action takes the grasp it was given and no other.

    Quietly swapping in one that works would perform a different action than the one
    described, and whether the grasp can be reached is the pre-condition's question.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pick_up = PickUpAction(milk.grasp_poses()[0], Arms.LEFT)
    sequential([pick_up], context=context)

    np.testing.assert_allclose(
        _reach_of(pick_up).grasp.root_T_grasp.to_homogeneous_matrix().to_np(),
        milk.grasp_poses()[0].root_T_grasp.to_homogeneous_matrix().to_np(),
    )


# %% the grasp domain is asked at execution, not at plan build


def test_reachable_grasps_searches_at_execution_not_construction(monkeypatch):
    """
    Handing the domain to a variable must not start the search.

    The domain is wrapped rather than consumed, so a generator is what defers the work
    to the first ``next``. Searching eagerly would answer about the world the plan was
    built in rather than the one the transport runs in.
    """
    searches = []

    def record_and_refuse(*args, **kwargs):
        searches.append(True)
        raise AssertionError("the search must not run before the domain is consumed")

    monkeypatch.setattr(factories, "grasping_location", record_and_refuse)

    variable(Pose, domain=factories.ReachableGrasps(object(), object(), object()))

    assert searches == []


def test_reachable_grasps_sees_the_world_as_it_is_when_consumed(monkeypatch):
    """
    The grasps are the ones of the world at the moment they are asked for, not of the
    world the domain was built in.
    """
    moved = {"value": "before"}
    observed = []

    @dataclass
    class LocationStandingIn:
        """
        A location yielding one pose, whose validator kept a grasp.
        """

        validator: SimpleNamespace = field(
            default_factory=lambda: SimpleNamespace(reachable_grasp=None)
        )
        """
        The validator whose grasp the caller reads back.
        """

        def __iter__(self):
            observed.append(moved["value"])
            self.validator.reachable_grasp = SimpleNamespace(
                copy_for_world=lambda world: world
            )
            yield Pose.from_xyz_rpy(0.0, 0.0, 0.0)

    monkeypatch.setattr(
        factories, "grasping_location", lambda *args, **kwargs: LocationStandingIn()
    )

    grasps = factories.ReachableGrasps(
        object(), SimpleNamespace(world=object()), object()
    )
    moved["value"] = "after"
    next(iter(grasps), None)

    assert observed == ["after"]


def test_a_grasping_location_judges_whether_the_object_can_be_grasped(
    immutable_model_world,
):
    """
    The grasp that was found is kept on the validator, so a caller that wants it reads
    that validator back off the location.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    location = factories.grasping_location(
        milk, context, ViewManager.get_arm_view(Arms.RIGHT, view)
    )

    assert location.validator.graspable is milk


def test_reachable_grasps_yields_grasps_the_object_offers(immutable_model_world):
    """
    Every grasp handed out is one of the object's own, so a caller naming one names
    something the object actually admits.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    # Where test_pose_validator establishes the right arm can reach it.
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.7, 1.4, 1.0, reference_frame=world.root
    )

    grasp = next(
        iter(
            factories.ReachableGrasps(
                milk, context, ViewManager.get_arm_view(Arms.RIGHT, view)
            )
        ),
        None,
    )

    assert grasp is not None, "the milk is reachable, so a grasp must be found"
    assert any(
        np.allclose(
            grasp.root_T_grasp.to_homogeneous_matrix().to_np(),
            offered.root_T_grasp.to_homogeneous_matrix().to_np(),
        )
        for offered in milk.grasp_poses()
    )


def test_reachable_grasps_are_on_the_annotation_the_caller_named(
    immutable_model_world,
):
    """
    A grasp handed out names the object in the world the plan runs in, not in the copy
    the reach was judged in, so the object can be attached to the gripper that takes it.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    # Where test_pose_validator establishes the right arm can reach it.
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.7, 1.4, 1.0, reference_frame=world.root
    )

    grasp = next(
        iter(
            factories.ReachableGrasps(
                milk, context, ViewManager.get_arm_view(Arms.RIGHT, view)
            )
        ),
        None,
    )

    assert grasp.graspable is milk
