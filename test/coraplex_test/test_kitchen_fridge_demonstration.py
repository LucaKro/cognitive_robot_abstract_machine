"""
Tests for the kitchen fridge demonstration's container handling: finding what has to be
opened to reach an object, and putting the opening and shutting into a plan that only
describes the transport.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.plan_node import ActionNode, UnderspecifiedNode
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import an, entity, the, variable
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Drawer,
    Fridge,
    Milk,
    ShelfLayer,
    Spoon,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% loading the demo


def load_demo():
    """
    :return: The demo module, which lives outside any package and so cannot be imported.
    """
    spec = importlib.util.spec_from_file_location(
        "kitchen_fridge_demo",
        Path(__file__).resolve().parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_kitchen_fridge_demo"
        / "demo.py",
    )
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
def milk_grasp(demonstration, fridge_context, milk_body) -> GraspDescription:
    return demonstration.grasps_for(milk_body, fridge_context)[0]


# %% building the scene


def test_the_kitchen_is_furnished_before_the_milk_arrives(demonstration):
    """
    Building the world stands the shelf layer in the fridge; the milk only turns up once
    the scene is populated.
    """
    world = demonstration.build_simulated_world()
    fridge = the(entity(variable(Fridge, domain=world.semantic_annotations))).first()
    shelf_layer = the(
        entity(variable(ShelfLayer, domain=world.semantic_annotations))
    ).first()

    assert shelf_layer in fridge.shelf_layers
    assert (
        next(
            an(entity(variable(Milk, domain=world.semantic_annotations))).evaluate(),
            None,
        )
        is None
    )
    assert not demonstration.is_scene_populated(world)

    demonstration.populate_scene(world)

    assert demonstration.is_scene_populated(world)


def test_table_rests_on_the_floor(fridge_world):
    """
    The kitchen hangs the table's mesh off its own centre, so building the world has to
    lift it out of the ground.
    """
    table = the(
        entity(variable(Table, domain=fridge_world.semantic_annotations))
    ).first()

    bounding_box = table.root.collision.as_bounding_box_collection_in_frame(
        fridge_world.root
    ).bounding_box()

    assert bounding_box.min_z == pytest.approx(0.0, abs=1e-6)
    assert bounding_box.max_z > 0.0


def test_the_milk_is_put_down_on_a_counter_top(demonstration, fridge_world):
    """
    The surface the demonstration places onto is one the reasoner recognises as a
    counter top, rather than an anonymous body it only knows by name.
    """
    counter_tops = fridge_world.get_semantic_annotations_by_type(CounterTop)

    assert fridge_world.get_body_by_name(demonstration.island_surface_name) in [
        counter_top.root for counter_top in counter_tops
    ]


def test_shelf_layer_is_coloured_by_its_specification(demonstration, fridge_world):
    """
    The colour comes from the body specification the shelf layer is built from, not from
    recolouring its shapes afterwards.
    """
    shelf_layer = the(
        entity(variable(ShelfLayer, domain=fridge_world.semantic_annotations))
    ).first()

    assert [shape.color for shape in shelf_layer.root.visual.shapes] == [
        demonstration.shelf_layer_color
    ]


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


def test_pick_up_grasps_follow_the_milk(demonstration, fridge_context, milk_body):
    """
    The pick-up searches over the grasps computed from where the milk actually stands,
    rather than over one assumed side of it.
    """
    plan = demonstration.build_plan(fridge_context)
    pick_up = next(
        node
        for node in plan.children
        if isinstance(node, UnderspecifiedNode)
        and node.underspecified_action.factory is PickUpAction
    ).underspecified_action

    assert list(pick_up.kwargs["grasp_description"]._domain_) == (
        demonstration.grasps_for(milk_body, fridge_context)
    )


def test_grasps_turn_with_the_body(demonstration, fridge_context, milk_body):
    """
    The side the gripper comes from is worked out from how the body is turned, which is
    what lets the plan survive the milk being put somewhere else.
    """
    end_effector = ViewManager.get_end_effector_view(
        demonstration.milk_arm, fridge_context.robot
    )
    as_placed = GraspDescription.calculate_grasp_descriptions(
        end_effector, milk_body.global_pose
    )
    turned = GraspDescription.calculate_grasp_descriptions(
        end_effector,
        (
            milk_body.global_pose.to_homogeneous_matrix()
            @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi)
        ).to_pose(),
    )

    assert as_placed == demonstration.grasps_for(milk_body, fridge_context)
    assert turned[0].approach_direction is not as_placed[0].approach_direction


# %% the demo adapts to where the milk stands


def milk_standing_in_the_open(
    offset_along_island: float = 0.4,
) -> "demo.KitchenFridgeDemonstration":
    """
    :param offset_along_island: How far along the island, in meters, the milk starts from
        where it is put down. Far enough that carrying it there is a real transport rather
        than a nudge, and still on the southern half of the island the demo places into.
    :return: A demonstration that starts the milk on the kitchen island rather than in the
        fridge, clear of the spot it is later placed at.
    """
    layout = demo.KitchenFridgeDemonstration(used_robot=PR2)
    world = layout.build_simulated_world()
    return demo.KitchenFridgeDemonstration(
        used_robot=PR2,
        milk_start_pose=layout.milk_pose_on_surface(
            world,
            layout.island_surface_name,
            layout.place_x_on_island,
            layout.place_y_on_island + offset_along_island,
            layout.island_approach_yaw,
        ),
    )


def test_plan_skips_the_container_for_a_milk_in_the_open():
    """
    A milk that stands in the open needs no door opened, so the plan is the transport
    alone.
    """
    demonstration = milk_standing_in_the_open()
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    robot = variable(demonstration.used_robot, domain=world.semantic_annotations)
    context = Context(
        world=world, robot=the(entity(robot)).first(), evaluate_conditions=False
    )

    actions = [
        (
            type(node.designator)
            if isinstance(node, ActionNode)
            else node.underspecified_action.factory
        )
        for node in demonstration.build_plan(context).children
    ]

    assert actions == [
        ParkArmsAction,
        MoveTorsoAction,
        NavigateAction,
        PickUpAction,
        ParkArmsAction,
        NavigateAction,
        PlaceAction,
        ParkArmsAction,
    ]


@pytest.mark.slow
def test_milk_turned_around_is_still_delivered():
    """
    Turning the carton on its shelf turns the side the gripper comes from with it, so
    the transport goes through unchanged.
    """
    demonstration = demo.KitchenFridgeDemonstration(
        used_robot=PR2,
        shelf_layer_T_milk=demo.KitchenFridgeDemonstration(
            used_robot=PR2
        ).shelf_layer_T_milk
        @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi),
    )

    world = demonstration.run()

    assert_milk_was_delivered(demonstration, world)
    assert (
        demonstration.fridge_door(world).root.parent_connection.position
        < demonstration.door_is_closed_angle
    )


@pytest.mark.slow
def test_milk_in_the_open_is_delivered():
    """
    With nothing to unpack, the robot still finds its standing poses and carries the
    milk to the island.
    """
    demonstration = milk_standing_in_the_open()

    assert_milk_was_delivered(demonstration, demonstration.run())


def assert_milk_was_delivered(demonstration, world) -> None:
    """
    Check that the milk ended up where the demonstration placed it.
    """
    milk = variable(Milk, domain=world.semantic_annotations)
    assert np.allclose(
        the(entity(milk)).first().root.global_pose.to_position(),
        demonstration.place_pose_on_island(world).to_position(),
        atol=demonstration.placement_tolerance,
    )
