"""
What a transport does besides moving an object: whether it looks at the two places it
works at.
"""

import numpy as np
import pytest

from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import ActionNode, PlanNode, UnderspecifiedNode
from unittest.mock import Mock

from coraplex.robot_plans.actions.composite import transporting
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.locations.base import DeferredLocation
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from typing_extensions import List, Type


def performed_actions(plan: PlanNode) -> List[Type]:
    """
    The action each step of a plan performs, in order, whether the step already names a
    concrete action or still has to resolve one.
    """
    performed = []
    for child in plan.children:
        if isinstance(child, UnderspecifiedNode):
            performed.append(child.underspecified_action.type_)
        else:
            performed.append(type(child.designator))
    return performed


def transport_plan(world, context, look_at_operation_site: bool) -> PlanNode:
    """
    The plan of a transport carrying the milk to a fixed pose.
    """
    action = TransportAction(
        world.get_semantic_annotations_by_type(Milk)[0],
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        Arms.LEFT,
        look_at_operation_site=look_at_operation_site,
    )
    execute_single(action, context=context)
    return action._action_plan


# %% looking at what the transport works on
def test_a_transport_looks_nowhere_by_default(immutable_model_world):
    world, view, context = immutable_model_world
    assert LookAtAction not in performed_actions(transport_plan(world, context, False))


def test_a_looking_transport_looks_before_it_picks_and_before_it_places(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    performed = performed_actions(transport_plan(world, context, True))
    assert performed[performed.index(PickUpAction) - 1] is LookAtAction
    assert performed[performed.index(PlaceAction) - 1] is LookAtAction


def test_a_looking_transport_looks_at_the_object_and_then_at_the_target(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    plan = transport_plan(world, context, True)
    looks = [
        child.designator
        for child in plan.children
        if isinstance(child, ActionNode) and isinstance(child.designator, LookAtAction)
    ]
    at_object, at_target = looks
    assert np.allclose(
        at_object.target.to_np(),
        world.get_body_by_name("milk.stl").global_pose.to_np(),
        atol=1e-6,
    )
    assert np.allclose(
        at_target.target.to_np(),
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root).to_np(),
        atol=1e-6,
    )


# %% where the robot stands to put something down


def test_neither_stand_is_searched_for_while_the_plan_is_built(
    immutable_model_world, monkeypatch
):
    """
    A stand for putting an object down is searched for with that object in hand.

    Searched while the plan is still being built, the probe answers for a gripper that
    is still empty, and the object the robot turns up holding is then in the way of the
    stand that was approved without it.
    """
    world, view, context = immutable_model_world
    searches = []
    monkeypatch.setattr(
        transporting,
        "reachability_location",
        lambda *args, **kwargs: searches.append(args) or Mock(),
    )

    transport_plan(world, context, False)

    assert searches == []
