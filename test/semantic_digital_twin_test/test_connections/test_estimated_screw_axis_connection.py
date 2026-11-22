"""
Tests of estimated screw axis connections: the child's pose at position zero and the
screw axis, written in the child's frame, held in the world state, so that a new
estimate moves the child without changing the world's model.
"""

from __future__ import annotations

# %% Imports
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import RigidTransform, Rotation

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    ScrewAxis,
    Vector3,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.exceptions import (
    ConstantScrewAxisError,
    EstimatedParentTConnectionError,
    UnnormalizedScrewAxisError,
)
from semantic_digital_twin.world_description.connection_properties import (
    ScrewAxisEstimateParameter,
)
from semantic_digital_twin.world_description.connections import ScrewAxisConnection


# %% Fixtures and helpers
def screw_axis(entries: list[float]) -> ScrewAxis:
    """
    :param entries: The rotation and then the translation per unit of position.
    :return: The screw axis with those entries.
    """
    return ScrewAxis(
        rotation_per_unit_position=Vector3(*entries[:3]),
        translation_per_unit_position=Vector3(*entries[3:]),
    )


hinge_off_the_origin = [0.0, 0.0, 1.0, 0.0, -0.4, 0.0]
"""
A door's hinge in the door's frame: along z, through (0.4, 0, 0), so that the door's
origin circles it on a radius of 0.4 m.
"""

thread = [0.0, 0.0, 1.0, 0.0, -0.03, 0.004 / (2 * np.pi)]
"""
A jar lid's thread in the lid's frame: along z, through (0.03, 0, 0), advancing 4 mm per
revolution.
"""

slide = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
"""
A drawer's slide in the drawer's frame: along x.
"""


def pose(*, rotation_vector: list[float], position: list[float]) -> np.ndarray:
    """
    :return: The 4x4 pose with that rotation, as a rotation vector, and that position.
    """
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    result[:3, 3] = position
    return result


first_pose_at_zero = pose(rotation_vector=[0.1, -0.2, 0.3], position=[0.4, -0.3, 0.9])
"""
The child's pose at position zero in the first estimate.
"""


@pytest.fixture()
def estimated_connection(world_with_two_bodies) -> ScrewAxisConnection:
    """
    An estimated screw axis connection whose first estimate is a door's hinge.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewAxisConnection.create_estimated_with_dofs(
            world=world,
            parent=parent,
            child=child,
            child_pose_at_zero=HomogeneousTransformationMatrix(data=first_pose_at_zero),
            screw_axis=screw_axis(hinge_off_the_origin),
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


def expected_child_pose(
    *, pose_at_zero: np.ndarray, entries: list[float], position: float
) -> np.ndarray:
    """
    :return: The child's pose after moving ``position`` along the screw axis with
        ``entries``, written in the child's frame at ``pose_at_zero``, by scipy's
        exponential.
    """
    return (
        pose_at_zero
        @ RigidTransform.from_exp_coords(np.asarray(entries) * position).as_matrix()
    )


# %% The motion
@pytest.mark.parametrize("entries", [hinge_off_the_origin, thread, slide])
@pytest.mark.parametrize("position", [0.0, 0.8, -2.5])
def test_estimated_connection_moves_the_child_along_its_estimate(
    world_with_two_bodies, entries: list[float], position: float
):
    """
    The child sits at its pose at position zero moved along the screw axis, written in
    the child's frame, also when the axis misses the connection frame's origin.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewAxisConnection.create_estimated_with_dofs(
            world=world,
            parent=parent,
            child=child,
            child_pose_at_zero=HomogeneousTransformationMatrix(data=first_pose_at_zero),
            screw_axis=screw_axis(entries),
        )
        world.add_connection(connection)

    connection.position = position

    assert_allclose(
        child_pose(connection),
        expected_child_pose(
            pose_at_zero=first_pose_at_zero, entries=entries, position=position
        ),
        atol=1e-12,
    )


def test_estimate_is_held_in_passive_degrees_of_freedom(estimated_connection):
    """
    One passive degree of freedom holds each number of the estimate, and the seven of
    the child's pose at position zero are the only variables of the connection's pose
    relative to its parent.
    """
    world = estimated_connection._world

    assert estimated_connection.is_estimated
    assert set(estimated_connection.estimate_dofs) == set(ScrewAxisEstimateParameter)
    assert set(estimated_connection.passive_dofs) <= set(
        world.passive_degrees_of_freedom
    )
    assert len(estimated_connection.estimated_transform_variables) == 7
    assert estimated_connection.connection_T_child_expression.is_constant()


def test_constant_connection_is_not_estimated(world_with_two_bodies):
    """
    A connection with a constant screw axis holds no estimate.
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
    assert connection.estimated_transform_variables == []


# %% New estimates
def test_new_estimate_moves_the_child_without_recompiling(estimated_connection):
    """
    A new estimate, here from turning to sliding with a shifted pose at position zero,
    moves the child at the given position, while the world keeps its compiled forward
    kinematics and its model.
    """
    world = estimated_connection._world
    manager = world._forward_kinematic_manager
    compiled = manager.compiled_all_fks
    revision = world.get_world_model_manager().revision
    shifted_pose = pose(rotation_vector=[0.1, -0.2, 0.3], position=[0.41, -0.3, 0.88])

    estimated_connection.update_estimate(
        child_pose_at_zero=HomogeneousTransformationMatrix(data=shifted_pose),
        screw_axis=screw_axis(slide),
        position=0.35,
    )

    assert manager.compiled_all_fks is compiled
    assert world.get_world_model_manager().revision == revision
    assert estimated_connection.position == pytest.approx(0.35)
    assert_allclose(
        child_pose(estimated_connection),
        expected_child_pose(pose_at_zero=shifted_pose, entries=slide, position=0.35),
        atol=1e-12,
    )


def test_new_estimate_keeps_the_sign_of_the_quaternion(estimated_connection):
    """
    An estimate whose quaternion comes out with the other sign, the same orientation, is
    stored with the sign of the one before, so that the stored numbers do not jump.
    """
    estimate_dofs = estimated_connection.estimate_dofs
    state = estimated_connection._world.state
    quaternion_dofs = [
        estimate_dofs[parameter]
        for parameter in (
            ScrewAxisEstimateParameter.CHILD_QX,
            ScrewAxisEstimateParameter.CHILD_QY,
            ScrewAxisEstimateParameter.CHILD_QZ,
            ScrewAxisEstimateParameter.CHILD_QW,
        )
    ]
    before = np.array([state[dof.id].position for dof in quaternion_dofs])
    turned_by_nearly_a_revolution = pose(
        rotation_vector=list(
            (
                Rotation.from_rotvec([0.0, 0.0, 2 * np.pi - 0.01])
                * Rotation.from_matrix(first_pose_at_zero[:3, :3])
            ).as_rotvec()
        ),
        position=[0.4, -0.3, 0.9],
    )

    estimated_connection.update_estimate(
        child_pose_at_zero=HomogeneousTransformationMatrix(
            data=turned_by_nearly_a_revolution
        ),
        screw_axis=screw_axis(hinge_off_the_origin),
        position=0.0,
    )

    after = np.array([state[dof.id].position for dof in quaternion_dofs])
    assert after @ before > 0
    assert_allclose(
        child_pose(estimated_connection), turned_by_nearly_a_revolution, atol=1e-12
    )


def test_unnormalized_estimate_raises(estimated_connection):
    """
    A screw axis that turns the child by two radians per unit of position is not
    normalized, so it is refused.
    """
    with pytest.raises(UnnormalizedScrewAxisError):
        estimated_connection.update_estimate(
            child_pose_at_zero=HomogeneousTransformationMatrix(data=first_pose_at_zero),
            screw_axis=screw_axis([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
            position=0.0,
        )


def test_estimate_of_a_constant_connection_raises(world_with_two_bodies):
    """
    A connection with a constant screw axis takes no estimate.
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
        connection.update_estimate(
            child_pose_at_zero=HomogeneousTransformationMatrix(),
            screw_axis=screw_axis(slide),
            position=0.0,
        )


def test_replacing_the_pose_of_an_estimated_connection_raises(estimated_connection):
    """
    The pose of an estimated connection relative to its parent is part of its estimate,
    so it is not replaced.
    """
    world = estimated_connection._world
    with pytest.raises(EstimatedParentTConnectionError):
        with world.modify_world():
            estimated_connection.update_parent_T_connection(
                parent_T_connection=HomogeneousTransformationMatrix(
                    reference_frame=estimated_connection.parent
                )
            )


# %% What the connection reports about its estimate
@pytest.mark.parametrize(
    "entries, revolute, prismatic, screw",
    [
        (hinge_off_the_origin, True, False, False),
        (thread, False, False, True),
        (slide, False, True, False),
    ],
)
def test_behaviour_follows_the_current_estimate(
    estimated_connection,
    entries: list[float],
    revolute: bool,
    prismatic: bool,
    screw: bool,
):
    """
    The kind of motion is read from the current estimate, by its pitch, so a hinge whose
    axis misses the connection frame's origin still behaves like a revolute connection.
    """
    estimated_connection.update_estimate(
        child_pose_at_zero=HomogeneousTransformationMatrix(data=first_pose_at_zero),
        screw_axis=screw_axis(entries),
        position=0.0,
    )

    assert estimated_connection.behaves_like_a_revolute_connection is revolute
    assert estimated_connection.behaves_like_a_prismatic_connection is prismatic
    assert estimated_connection.behaves_like_a_screw_connection is screw


def test_current_screw_axis_is_the_estimate(estimated_connection):
    """
    The current screw axis is the latest estimate, in the child's frame at position
    zero.
    """
    estimated_connection.update_estimate(
        child_pose_at_zero=HomogeneousTransformationMatrix(data=first_pose_at_zero),
        screw_axis=screw_axis(thread),
        position=0.0,
    )

    assert_allclose(estimated_connection.current_screw_axis.to_np()[:6], thread)


def test_position_for_translation_uses_the_pitch_of_the_estimate(estimated_connection):
    """
    For an estimated thread, the position that advances the child by one pitch is one
    revolution, although the axis misses the connection frame's origin.
    """
    estimated_connection.update_estimate(
        child_pose_at_zero=HomogeneousTransformationMatrix(data=first_pose_at_zero),
        screw_axis=screw_axis(thread),
        position=0.0,
    )

    assert estimated_connection.position_for_translation(
        distance=0.004
    ) == pytest.approx(2 * np.pi)


# %% Serialization and copies
def test_copied_world_holds_the_estimate_in_its_own_degrees_of_freedom(
    estimated_connection,
):
    """
    A copy of the world moves the child by the copied estimate, and a new estimate in
    the copy leaves the original alone.
    """
    world = estimated_connection._world
    estimated_connection.position = 0.5

    copied_world = deepcopy(world)
    copied_connection = copied_world.get_connection_by_name(estimated_connection.name)
    copied_connection.update_estimate(
        child_pose_at_zero=HomogeneousTransformationMatrix(data=first_pose_at_zero),
        screw_axis=screw_axis(slide),
        position=0.2,
    )

    assert set(copied_connection.passive_dofs) <= set(copied_world.degrees_of_freedom)
    assert_allclose(
        child_pose(copied_connection),
        expected_child_pose(
            pose_at_zero=first_pose_at_zero, entries=slide, position=0.2
        ),
        atol=1e-12,
    )
    assert_allclose(
        child_pose(estimated_connection),
        expected_child_pose(
            pose_at_zero=first_pose_at_zero, entries=hinge_off_the_origin, position=0.5
        ),
        atol=1e-12,
    )


def test_estimated_connection_survives_a_json_round_trip(estimated_connection):
    """
    A connection read back from json stays estimated, held by the same degrees of
    freedom, and so at the same estimate.
    """
    world = estimated_connection._world

    restored_connection = ScrewAxisConnection.from_json(
        data=estimated_connection.to_json(),
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )

    assert restored_connection.is_estimated
    assert {dof.id for dof in restored_connection.passive_dofs} == {
        dof.id for dof in estimated_connection.passive_dofs
    }
    assert_allclose(
        restored_connection.current_screw_axis.to_np()[:6], hinge_off_the_origin
    )
