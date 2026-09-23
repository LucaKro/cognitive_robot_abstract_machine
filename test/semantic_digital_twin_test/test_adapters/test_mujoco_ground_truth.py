"""
Tests for keeping the simulated ground truth of a connection out of the world: the
physics and the world hold their own values for an unobserved connection, and the
synchronizer records how far they diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pytest

from ...pytest_environment import runs_in_continuous_integration
from .test_mujoco_servos import _pendulum_world

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

mujoco_runs_only_in_continuous_integration = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

# %% an unobserved joint


@mujoco_runs_only_in_continuous_integration
def test_the_physics_does_not_overwrite_an_unobserved_joint_in_the_world():
    """
    Neither the hinge nor the mirrored hinge sharing its degree of freedom carries the
    physics into the world.
    """
    pendulum = _pendulum_world()
    world = pendulum.world
    simulation = MujocoSim(world=world, headless=True)
    simulation.synchronizer.unobserved_connections.add(pendulum.hinge)
    believed = 0.5
    simulation.start_stepped_simulation()
    try:
        simulator = simulation.simulator
        world.state[pendulum.hinge.raw_dof.id].position = believed
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=simulator.step_size))
        simulated = simulator.get_joint_value(pendulum.hinge.name.name).result
    finally:
        simulation.stop_simulation()

    assert world.state[pendulum.hinge.raw_dof.id].position == believed
    assert simulated == pytest.approx(0.0)


@mujoco_runs_only_in_continuous_integration
def test_the_physics_still_overwrites_an_observed_joint_in_the_world():
    pendulum = _pendulum_world()
    world = pendulum.world
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulator = simulation.simulator
        world.state[pendulum.hinge.raw_dof.id].position = 0.5
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=simulator.step_size))
        simulated = simulator.get_joint_value(pendulum.hinge.name.name).result
    finally:
        simulation.stop_simulation()

    assert world.state[pendulum.hinge.raw_dof.id].position == pytest.approx(simulated)


@mujoco_runs_only_in_continuous_integration
def test_the_divergence_of_an_unobserved_joint_is_logged():
    pendulum = _pendulum_world()
    world = pendulum.world
    simulation = MujocoSim(world=world, headless=True)
    simulation.synchronizer.unobserved_connections.add(pendulum.hinge)
    simulation.start_stepped_simulation()
    try:
        simulator = simulation.simulator
        world.state[pendulum.hinge.raw_dof.id].position = 0.5
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=simulator.step_size))
        simulated = simulator.get_joint_value(pendulum.hinge.name.name).result
        simulation_time = simulator.current_simulation_time
    finally:
        simulation.stop_simulation()

    record = simulation.synchronizer.divergence_log[-1]
    [divergence] = record.divergences
    assert record.simulation_time == pytest.approx(simulation_time)
    assert divergence.degree_of_freedom is pendulum.hinge.raw_dof
    assert divergence.world_position == world.state[pendulum.hinge.raw_dof.id].position
    assert divergence.physics_position == pytest.approx(simulated)


@mujoco_runs_only_in_continuous_integration
def test_nothing_is_logged_while_every_connection_is_observed():
    pendulum = _pendulum_world()
    simulation = MujocoSim(world=pendulum.world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=simulation.simulator.step_size))
    finally:
        simulation.stop_simulation()

    assert simulation.synchronizer.divergence_log == []


# %% an unobserved object pose


@dataclass
class FallingBoxWorld:
    """
    A world with a box held above a floor by nothing but its own free connection.

    ..note:: Building a MuJoCo simulation of the world gives the box a new free
        connection, so the connection is looked up from the box afterwards.
    """

    world: World
    """
    The world itself.
    """

    box: Body
    """
    The box.
    """


def _falling_box_world(height: float) -> FallingBoxWorld:
    """
    :param height: How high above the floor the box starts.
    :return: The world with the box at rest at that height.
    """
    world = World()
    floor_thickness = 0.02
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        floor = Body(name=PrefixedName("floor"))
        floor.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=-floor_thickness / 2, reference_frame=floor
                    ),
                    scale=Scale(2.0, 2.0, floor_thickness),
                )
            ],
            reference_frame=floor,
        )
        world.add_connection(FixedConnection(parent=root, child=floor))
        box = Body(name=PrefixedName("box"))
        box.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=box
                    ),
                    scale=Scale(0.1, 0.1, 0.1),
                )
            ],
            reference_frame=box,
        )
        box_connection = Connection6DoF.create_with_dofs(
            world=world, parent=root, child=box
        )
        world.add_connection(box_connection)
    box_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=height, reference_frame=root
    )
    return FallingBoxWorld(world=world, box=box)


@mujoco_runs_only_in_continuous_integration
def test_the_physics_does_not_move_an_unobserved_object_in_the_world():
    """
    The box falls in the physics while the world keeps it where it was believed to be.
    """
    height = 1.0
    falling = _falling_box_world(height)
    simulation = MujocoSim(world=falling.world, headless=True)
    box_connection = falling.box.parent_connection
    simulation.synchronizer.unobserved_connections.add(box_connection)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=0.3))
        simulated_height = simulation.simulator.get_body_position(
            falling.box.name.name
        ).result[2]
    finally:
        simulation.stop_simulation()

    believed_height = box_connection.origin.to_np()[2, 3]
    assert believed_height == pytest.approx(height)
    assert simulated_height < height - 0.3


@mujoco_runs_only_in_continuous_integration
def test_moving_an_unobserved_object_in_the_world_does_not_move_it_in_the_physics():
    """
    A belief about where the box is never teleports the box itself.
    """
    height = 1.0
    falling = _falling_box_world(height)
    simulation = MujocoSim(world=falling.world, headless=True)
    box_connection = falling.box.parent_connection
    simulation.synchronizer.unobserved_connections.add(box_connection)
    simulation.start_stepped_simulation()
    try:
        box_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.5, z=height, reference_frame=falling.world.root
        )
        simulation.step_simulation(timedelta(seconds=simulation.simulator.step_size))
        simulated_x = simulation.simulator.get_body_position(
            falling.box.name.name
        ).result[0]
    finally:
        simulation.stop_simulation()

    assert simulated_x == pytest.approx(0.0)
