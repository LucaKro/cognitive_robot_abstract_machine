"""
Tests for the kitchen fridge demonstration's container handling: finding what has to be
opened to reach an object, and putting the opening and shutting into a plan that only
describes the transport.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.plan_node import ActionNode
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import entity, the, variable
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% loading the demo

DEMO_PATH = (
    Path(__file__).resolve().parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_kitchen_fridge_demo"
    / "demo.py"
)


def load_demo():
    """
    :return: The demo module, which lives outside any package and so cannot be imported.
    """
    spec = importlib.util.spec_from_file_location("kitchen_fridge_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = load_demo()


# %% the scene the demo acts in


@pytest.fixture(scope="module")
def demonstration() -> "demo.KitchenFridgeDemonstration":
    return demo.KitchenFridgeDemonstration(used_robot=PR2)


@pytest.fixture(scope="module")
def fridge_world(demonstration) -> World:
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    return world


@pytest.fixture(scope="module")
def fridge_context(demonstration, fridge_world) -> Context:
    """
    The plan context, built without a ROS node so nothing here needs a running
    controller.
    """
    robot = variable(demonstration.used_robot, domain=fridge_world.semantic_annotations)
    return Context(
        world=fridge_world,
        robot=the(entity(robot)).first(),
        evaluate_conditions=False,
    )


@pytest.fixture(scope="module")
def milk_body(fridge_world) -> Body:
    milk = variable(Milk, domain=fridge_world.semantic_annotations)
    return the(entity(milk)).first().root


@pytest.fixture(scope="module")
def milk_grasp(demonstration, fridge_context) -> GraspDescription:
    return GraspDescription(
        demonstration.milk_approach_direction,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(demonstration.milk_arm, fridge_context.robot),
    )


# %% finding what has to be opened


def test_handle_of_a_body_behind_a_door(demonstration, fridge_world, milk_body):
    """
    The milk stands in the fridge, so it is the fridge door's handle that has to be
    pulled.
    """
    assert (
        demonstration.handle_of_enclosing_container(milk_body, fridge_world)
        is demonstration.fridge_door(fridge_world).handle.root
    )


def test_handle_of_a_body_inside_a_drawer(demonstration, apartment_world_copy):
    """
    A drawer holds what it contains in its own case rather than behind it, and is found
    the same way a door is.
    """
    drawer = the(
        entity(variable(Drawer, domain=apartment_world_copy.semantic_annotations))
    ).first()
    spoon = the(
        entity(variable(Spoon, domain=apartment_world_copy.semantic_annotations))
    ).first()

    assert (
        demonstration.handle_of_enclosing_container(spoon.root, apartment_world_copy)
        is drawer.handle.root
    )


def test_body_standing_in_the_open_has_no_handle(demonstration, fridge_world):
    """
    The kitchen island surface is in nothing that opens, so there is nothing to pull.
    """
    island_surface = fridge_world.get_body_by_name(demonstration.island_surface_name)

    assert (
        demonstration.handle_of_enclosing_container(island_surface, fridge_world)
        is None
    )


# %% adding the opening and the shutting


def test_container_steps_enclose_the_given_steps(
    demonstration, fridge_world, fridge_context, milk_body, milk_grasp
):
    """
    The opening goes in front of the given steps and the shutting behind them, leaving
    those steps themselves untouched.
    """
    handle = demonstration.fridge_door(fridge_world).handle.root
    transport = [
        PickUpAction(milk_body, demonstration.milk_arm, milk_grasp),
        ParkArmsAction(Arms.BOTH),
    ]

    steps = demonstration.add_container_opening_and_closing(
        transport, milk_body, fridge_context
    )

    assert steps[3:-3] == transport
    assert steps[3] is transport[0]
    assert steps[0].factory is NavigateAction
    assert steps[-3].factory is NavigateAction
    assert isinstance(steps[2], ParkArmsAction)
    assert isinstance(steps[-1], ParkArmsAction)

    opening = steps[1]
    assert isinstance(opening, OpenAction)
    assert opening.object_designator is handle
    assert opening.arm is demonstration.door_arm
    assert opening.approach_direction is demonstration.handle_approach_direction
    assert opening.goal_joint_state == demonstration.door_opening_angle

    shutting = steps[-2]
    assert isinstance(shutting, CloseAction)
    assert shutting.object_designator is handle
    assert shutting.arm is demonstration.door_arm
    assert shutting.approach_direction is demonstration.handle_approach_direction
    assert shutting.goal_joint_state == demonstration.door_closing_angle


def test_nothing_is_added_for_a_body_in_the_open(
    demonstration, fridge_world, fridge_context
):
    """
    A body that stands in nothing that opens leaves the plan as it was.
    """
    island_surface = fridge_world.get_body_by_name(demonstration.island_surface_name)
    transport = [ParkArmsAction(Arms.BOTH)]

    assert (
        demonstration.add_container_opening_and_closing(
            transport, island_surface, fridge_context
        )
        is transport
    )


# %% the whole plan


def test_plan_opens_the_fridge_and_shuts_it_again(demonstration, fridge_context):
    """
    The plan describes the transport only, and comes out with the fridge opened before
    the robot drives up to the milk and shut again once the milk is put down.
    """
    plan = demonstration.build_plan(fridge_context)

    actions = [
        (
            type(node.designator)
            if isinstance(node, ActionNode)
            else node.underspecified_action.factory
        )
        for node in plan.children
    ]

    assert actions == [
        ParkArmsAction,
        MoveTorsoAction,
        NavigateAction,
        OpenAction,
        ParkArmsAction,
        NavigateAction,
        PickUpAction,
        ParkArmsAction,
        NavigateAction,
        PlaceAction,
        ParkArmsAction,
        NavigateAction,
        CloseAction,
        ParkArmsAction,
    ]
