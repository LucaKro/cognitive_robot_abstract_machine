"""
Tests of screw axis connections whose screw axis is estimated: held in the world state,
so that a new estimate moves the child without changing the world's model.
"""

from __future__ import annotations

# %% Imports
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Twist,
    Vector3,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.exceptions import (
    CommittedScrewAxisEstimateError,
    ConstantScrewAxisError,
    ConstantScrewAxisOffConnectionFrameError,
    EstimatedScrewAxisOffZAxisError,
    UnnormalizedScrewAxisError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.connection_properties import (
    ScrewAxisEstimateParameter,
)
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.derivatives import DerivativeMap
from ..test_articulation_estimation.test_articulation_estimation import (
    observation_noise,
    screw_motions,
)


# %% Fixtures
@pytest.fixture()
def estimated_connection(world_with_two_bodies) -> ScrewAxisConnection:
    """
    An estimated screw axis connection that starts out revolute about z.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewAxisConnection.create_estimated_with_dofs(
            world=world,
            parent=parent,
            child=child,
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        )
        world.add_connection(connection)
    return connection


def child_pose(connection: ScrewAxisConnection) -> np.ndarray:
    """
    :return: The pose of the connection's child relative to its parent.
    """
    return connection._world.compute_forward_kinematics_np(
        root=connection.parent, tip=connection.child
    )


def helical_along_z(pitch: float) -> Twist:
    """
    :return: The screw axis of a helical connection along z with ``pitch`` per
        revolution.
    """
    return ScrewAxisConnection.create_helical_screw_axis(axis=Vector3.Z(), pitch=pitch)


# %% Creating an estimated connection
def test_estimated_connection_moves_the_child_along_its_initial_screw_axis(
    estimated_connection,
):
    """
    Before any estimate is written, the connection moves the child as the screw axis it
    was created with.
    """
    estimated_connection.position = np.pi / 2

    assert_allclose(child_pose(estimated_connection)[:3, 0], [0, 1, 0], atol=1e-12)
    assert estimated_connection.is_estimated


def test_estimated_connection_holds_its_screw_axis_in_passive_degrees_of_freedom(
    estimated_connection,
):
    """
    The estimate is held by one passive degree of freedom of the connection for each of
    its numbers, which the world keeps.
    """
    world = estimated_connection._world

    assert set(estimated_connection.estimate_dofs) == set(ScrewAxisEstimateParameter)
    assert set(estimated_connection.passive_dofs) <= set(world.degrees_of_freedom)
    assert estimated_connection.raw_dof not in estimated_connection.passive_dofs


def test_estimated_connection_off_the_z_axis_raises(world_with_two_bodies):
    """
    An estimated screw axis lies on the z axis of the connection frame, so that its two
    numbers fix it.
    """
    world, parent, child = world_with_two_bodies
    with pytest.raises(EstimatedScrewAxisOffZAxisError):
        with world.modify_world():
            ScrewAxisConnection.create_estimated_with_dofs(
                world=world,
                parent=parent,
                child=child,
                screw_axis=ScrewAxisConnection.create_revolute_screw_axis(
                    axis=Vector3.X()
                ),
            )


def test_constant_connection_is_not_estimated(world_with_two_bodies):
    """
    A connection created with a constant screw axis has no passive degrees of freedom.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        )
        world.add_connection(connection)

    assert not connection.is_estimated
    assert connection.passive_dofs == []


# %% Writing an estimate
def test_new_estimate_moves_the_child_without_recompiling(estimated_connection):
    """
    A new estimate is a state change: the child follows it, and the forward kinematics
    are not compiled again.
    """
    world = estimated_connection._world
    compiled_forward_kinematics = world._forward_kinematic_manager.compiled_all_fks
    estimated_connection.position = 2 * np.pi

    estimated_connection.update_screw_axis_estimate(screw_axis=helical_along_z(0.01))

    assert_allclose(child_pose(estimated_connection)[:3, 3], [0, 0, 0.01], atol=1e-12)
    assert (
        world._forward_kinematic_manager.compiled_all_fks is compiled_forward_kinematics
    )


@pytest.mark.parametrize(
    "screw_axis, behaviour",
    [
        (ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()), "revolute"),
        (
            ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.Z()),
            "prismatic",
        ),
        (helical_along_z(0.01), "screw"),
    ],
)
def test_behaviour_follows_the_current_estimate(
    estimated_connection, screw_axis, behaviour
):
    """
    Whether the connection behaves like a revolute, prismatic or screw connection is
    read from the current estimate.
    """
    estimated_connection.update_screw_axis_estimate(screw_axis=screw_axis)

    assert [
        estimated_connection.behaves_like_a_revolute_connection,
        estimated_connection.behaves_like_a_prismatic_connection,
        estimated_connection.behaves_like_a_screw_connection,
    ] == [behaviour == kind for kind in ["revolute", "prismatic", "screw"]]


def test_current_screw_axis_is_the_estimate(estimated_connection):
    """
    The current screw axis of an estimated connection holds the numbers last written.
    """
    estimated_connection.update_screw_axis_estimate(screw_axis=helical_along_z(0.02))

    assert_allclose(
        estimated_connection.current_screw_axis.to_np(),
        helical_along_z(0.02).to_np(),
    )


def test_position_for_translation_uses_the_current_estimate(estimated_connection):
    """
    The position that advances the child by a distance follows the estimated pitch.
    """
    estimated_connection.update_screw_axis_estimate(screw_axis=helical_along_z(0.02))

    assert estimated_connection.position_for_translation(
        distance=0.02
    ) == pytest.approx(2 * np.pi)


def test_unnormalized_estimate_raises(estimated_connection):
    """
    An estimate is normalized as a constant screw axis is.
    """
    with pytest.raises(UnnormalizedScrewAxisError):
        estimated_connection.update_screw_axis_estimate(
            screw_axis=Twist(angular_velocity=Vector3(0, 0, 2))
        )


def test_estimate_off_the_z_axis_raises(estimated_connection):
    """
    An estimate lies on the z axis of the connection frame.
    """
    with pytest.raises(EstimatedScrewAxisOffZAxisError):
        estimated_connection.update_screw_axis_estimate(
            screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.Y())
        )


def test_estimate_of_a_constant_connection_raises(world_with_two_bodies):
    """
    A connection created with a constant screw axis has nothing to estimate.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        )
        world.add_connection(connection)

    with pytest.raises(ConstantScrewAxisError):
        connection.update_screw_axis_estimate(screw_axis=helical_along_z(0.01))


# %% Axis frame
def test_new_axis_frame_moves_the_axis_but_not_the_child_at_position_zero(
    estimated_connection,
):
    """
    An estimate of the axis frame places the axis, about which the child then moves,
    while the child's pose at position zero stays the connection's.
    """
    world = estimated_connection._world
    compiled_forward_kinematics = world._forward_kinematic_manager.compiled_all_fks
    pose_at_zero = child_pose(estimated_connection)

    estimated_connection.update_screw_axis_estimate(
        screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        connection_T_screw_axis=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
    )

    assert_allclose(child_pose(estimated_connection), pose_at_zero, atol=1e-12)
    estimated_connection.position = np.pi
    assert_allclose(child_pose(estimated_connection)[:3, 3], [1, 0, 0], atol=1e-12)
    assert (
        world._forward_kinematic_manager.compiled_all_fks is compiled_forward_kinematics
    )


def test_constant_screw_axis_in_another_frame_raises(world_with_two_bodies):
    """
    Only an estimated connection moves its axis frame; a constant screw axis is placed
    by the connection frame.
    """
    world, parent, child = world_with_two_bodies
    with pytest.raises(ConstantScrewAxisOffConnectionFrameError):
        ScrewAxisConnection(
            parent=parent,
            child=child,
            raw_dof=DegreeOfFreedom(name=PrefixedName("dof")),
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
            connection_T_screw_axis=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
        )


@pytest.mark.parametrize("fit_type", list(screw_motions()))
def test_written_fit_reproduces_the_observations(
    world_with_two_bodies, fit_type, observation_noise
):
    """
    A fit written into an estimated connection that holds the child at its first
    observed pose puts the child, at each fitted position, where it was observed.
    """
    world, parent, child = world_with_two_bodies
    observations = screw_motions()[fit_type].observe(noise=observation_noise, seed=7)
    with world.modify_world():
        connection = ScrewAxisConnection.create_estimated_with_dofs(
            world=world,
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                observations.parent_T_child[0]
            ),
            screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(
                axis=Vector3.Z()
            ),
        )
        world.add_connection(connection)
    fit = fit_type.fit(observations=observations, noise=observation_noise)

    fit.update_estimated_connection(connection=connection)

    for position, observed in zip(fit.positions, observations.parent_T_child):
        connection.position = position
        assert_allclose(child_pose(connection)[:3, 3], observed[:3, 3], atol=0.01)


# %% Committing an estimate
def test_committed_estimate_cannot_be_updated(estimated_connection):
    """
    Committing makes the estimate final until it is reopened.
    """
    world = estimated_connection._world
    with world.modify_world():
        estimated_connection.commit_screw_axis_estimate(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z())
        )

    with pytest.raises(CommittedScrewAxisEstimateError):
        estimated_connection.update_screw_axis_estimate(
            screw_axis=helical_along_z(0.01)
        )


def test_reopened_estimate_can_be_updated(estimated_connection):
    """
    A reopened estimate takes new estimates again.
    """
    world = estimated_connection._world
    with world.modify_world():
        estimated_connection.commit_screw_axis_estimate(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z())
        )
    with world.modify_world():
        estimated_connection.reopen_screw_axis_estimate()

    estimated_connection.update_screw_axis_estimate(screw_axis=helical_along_z(0.01))

    assert estimated_connection.behaves_like_a_screw_connection


def test_commit_writes_the_final_estimate(estimated_connection):
    """
    Committing writes the screw axis it is given, as the last estimate.
    """
    world = estimated_connection._world
    with world.modify_world():
        estimated_connection.commit_screw_axis_estimate(
            screw_axis=helical_along_z(0.03)
        )

    assert_allclose(
        estimated_connection.current_screw_axis.to_np(),
        helical_along_z(0.03).to_np(),
    )
    assert estimated_connection.screw_axis_estimate_is_committed


# %% Periodic positions
def test_uncommitted_revolute_estimate_has_no_periodic_position(estimated_connection):
    """
    While the estimate may still change, positions a revolution apart are distinct: the
    connection may yet turn out to be helical.
    """
    assert not estimated_connection.position_is_periodic


def test_committed_revolute_estimate_has_a_periodic_position(estimated_connection):
    """
    A committed revolute estimate without position limits returns to the same pose after
    a revolution.
    """
    world = estimated_connection._world
    with world.modify_world():
        estimated_connection.commit_screw_axis_estimate(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z())
        )

    assert estimated_connection.position_is_periodic


@pytest.mark.parametrize(
    "limits, periodic",
    [
        (None, True),
        (
            DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-1.0), upper=DerivativeMap(position=1.0)
            ),
            False,
        ),
    ],
    ids=["continuous", "limited"],
)
def test_constant_revolute_connection_has_a_periodic_position_unless_limited(
    world_with_two_bodies, limits, periodic
):
    """
    A constant revolute connection's position is periodic exactly when it has no
    position limits.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
            dof_limits=limits,
        )
        world.add_connection(connection)

    assert connection.position_is_periodic is periodic


# %% Serialization and copies
def test_copied_world_holds_the_estimate_in_its_own_degrees_of_freedom(
    estimated_connection,
):
    """
    A copy of the world moves the child by the copied estimate, and a new estimate in
    the copy leaves the original alone.
    """
    world = estimated_connection._world
    estimated_connection.update_screw_axis_estimate(screw_axis=helical_along_z(0.02))
    estimated_connection.position = 2 * np.pi

    copied_world = deepcopy(world)
    copied_connection = copied_world.get_connection_by_name(estimated_connection.name)
    copied_connection.update_screw_axis_estimate(
        screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.Z())
    )

    assert set(copied_connection.passive_dofs) <= set(copied_world.degrees_of_freedom)
    assert_allclose(child_pose(copied_connection)[:3, 3], [0, 0, 2 * np.pi])
    assert_allclose(child_pose(estimated_connection)[:3, 3], [0, 0, 0.02], atol=1e-12)


def test_estimated_connection_survives_a_json_round_trip(estimated_connection):
    """
    A connection read back from json stays estimated, held by the same degrees of
    freedom, and so at the same estimate.
    """
    world = estimated_connection._world
    estimated_connection.update_screw_axis_estimate(screw_axis=helical_along_z(0.02))

    restored_connection = ScrewAxisConnection.from_json(
        data=estimated_connection.to_json(),
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )

    assert restored_connection.is_estimated
    assert [dof.id for dof in restored_connection.passive_dofs] == [
        dof.id for dof in estimated_connection.passive_dofs
    ]
    assert_allclose(
        restored_connection.current_screw_axis.to_np(),
        helical_along_z(0.02).to_np(),
    )
