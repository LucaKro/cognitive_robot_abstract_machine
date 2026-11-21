"""
Tests of estimating how a child moves from noisy observations of its poses.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    ScrewAxis,
    Vector3,
)
from semantic_digital_twin.articulation_estimation.exceptions import (
    NotHomogeneousTransformationError,
    TooFewObservationsError,
)
from semantic_digital_twin.articulation_estimation.fits import ScrewMotionFit
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import UnnormalizedScrewAxisError
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.world_entity import Body, Connection


# %% Synthetic observations
@dataclass(kw_only=True)
class ObservedScrewMotion:
    """
    A child moved by a known screw axis connection and observed with noise.
    """

    screw_axis: ScrewAxis
    """
    The screw axis of the connection that moves the child.
    """

    positions: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.linspace(0.0, 1.2, 15)
    )
    """
    The positions at which the child is observed.
    """

    parent_T_connection: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.4, y=-0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3
        )
    )
    """
    The pose of the connection frame relative to the parent.
    """

    connection_T_child: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.5, y=0.1, z=-0.2, roll=-0.3, pitch=0.1, yaw=0.4
        )
    )
    """
    The pose of the child relative to the connection frame at position zero.
    """

    def observe(self, *, noise: ObservationNoise, seed: int) -> PoseObservations:
        """
        :return: The child's poses at :attr:`positions`, each disturbed by ``noise``.
        """
        world = World()
        parent = Body(name=PrefixedName("parent"))
        child = Body(name=PrefixedName("child"))
        with world.modify_world():
            connection = ScrewAxisConnection.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                parent_T_connection_expression=self.parent_T_connection,
                connection_T_child_expression=self.connection_T_child,
                screw_axis=self.screw_axis,
            )
            world.add_connection(connection)
        poses = []
        for position in self.positions:
            connection.position = position
            poses.append(world.compute_forward_kinematics_np(root=parent, tip=child))
        return PoseObservations(
            parent_T_child_stack=disturbed_poses(
                poses=np.stack(poses), noise=noise, seed=seed
            )
        )

    @property
    def axis_direction_in_parent(self) -> npt.NDArray[np.float64]:
        """
        The unit direction of the screw axis in the parent frame.
        """
        return (
            self.parent_T_connection.to_np()[:3, :3]
            @ self.screw_axis.unit_direction.to_np()[:3]
        )

    def distance_from_axis(self, *, point: Point3) -> float:
        """
        :return: How far ``point``, in the parent frame, lies from the screw axis.
        """
        _, distance = point.project_to_line(
            line_point=Point3(*self.parent_T_connection.to_np()[:3, 3]),
            line_direction=Vector3(*self.axis_direction_in_parent),
        )
        return float(distance)


def disturbed_poses(
    *, poses: npt.NDArray[np.float64], noise: ObservationNoise, seed: int
) -> npt.NDArray[np.float64]:
    """
    :return: ``poses`` with normally distributed errors of the size ``noise`` describes
        added to each position and rotation vector.
    """
    random_numbers = np.random.default_rng(seed)
    disturbed = poses.copy()
    disturbed[:, :3, 3] += random_numbers.normal(
        scale=noise.position_standard_deviation, size=(len(poses), 3)
    )
    disturbed[:, :3, :3] = (
        Rotation.from_matrix(poses[:, :3, :3])
        * Rotation.from_rotvec(
            random_numbers.normal(
                scale=noise.orientation_standard_deviation, size=(len(poses), 3)
            )
        )
    ).as_matrix()
    return disturbed


@pytest.fixture()
def observation_noise() -> ObservationNoise:
    """
    The noise of a good motion tracker: a millimeter and a quarter of a degree.
    """
    return ObservationNoise(
        position_standard_deviation=0.001, orientation_standard_deviation=0.005
    )


@pytest.fixture()
def exact_observation() -> ObservationNoise:
    """
    No noise, for observing a motion as it is.
    """
    return ObservationNoise(
        position_standard_deviation=0.0, orientation_standard_deviation=0.0
    )


def screw_motions() -> dict[str, ObservedScrewMotion]:
    """
    :return: A revolute, a prismatic and a helical motion, by those names.
    """
    axis = Vector3.unit_vector(1, 2, 2)
    return {
        "revolute": ObservedScrewMotion(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=axis)
        ),
        "prismatic": ObservedScrewMotion(
            screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(axis=axis),
            positions=np.linspace(0.0, 0.4, 15),
        ),
        "helical": ObservedScrewMotion(
            screw_axis=ScrewAxisConnection.create_helical_screw_axis(
                axis=axis, pitch=0.3
            )
        ),
    }


turning_motions = ["revolute", "helical"]
"""
The names of the motions of :func:`screw_motions` that turn the child.
"""


# %% Observations
def test_fewer_than_two_observations_raise():
    """
    One pose cannot show whether the child moves.
    """
    with pytest.raises(TooFewObservationsError):
        PoseObservations(parent_T_child_stack=np.eye(4)[np.newaxis])


def test_a_pose_that_is_not_a_homogeneous_transformation_raises():
    """
    A pose whose last row is not [0, 0, 0, 1] is not a rigid transformation.
    """
    stack = np.tile(np.eye(4), (3, 1, 1))
    stack[1, 3, 0] = 0.1

    with pytest.raises(NotHomogeneousTransformationError) as raised:
        PoseObservations(parent_T_child_stack=stack)

    assert raised.value.observation_indices == [1]


def test_round_off_in_the_last_row_of_a_pose_is_removed():
    """
    A last row that differs from [0, 0, 0, 1] by round-off is set to it exactly, in a
    copy of the given poses.
    """
    stack = np.tile(np.eye(4), (2, 1, 1))
    stack[1, 3] += 1e-15

    observations = PoseObservations(parent_T_child_stack=stack)

    assert np.array_equal(observations.parent_T_child_stack[:, 3], [[0, 0, 0, 1]] * 2)
    assert stack[1, 3, 3] != 1


# %% Recovered motion
@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_fit_predicts_the_motion_behind_noisy_observations(
    motion_name: str,
    observation_noise: ObservationNoise,
    exact_observation: ObservationNoise,
):
    """
    The poses the fit predicts lie closer to the true poses than the observed ones do,
    in position and in orientation.
    """
    observations = screw_motions()[motion_name].observe(noise=observation_noise, seed=1)

    fit = ScrewMotionFit.fit(observations=observations, noise=observation_noise)

    true_parent_T_child_stack = (
        screw_motions()[motion_name].observe(noise=exact_observation, seed=1)
    ).parent_T_child_stack
    for slices in (np.s_[:, :3, 3], np.s_[:, :3, :3]):
        prediction_error = (
            fit.predicted_parent_T_child_stack[slices]
            - true_parent_T_child_stack[slices]
        )
        observation_error = (
            observations.parent_T_child_stack[slices]
            - true_parent_T_child_stack[slices]
        )
        assert np.linalg.norm(prediction_error) < np.linalg.norm(observation_error)


@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_fit_is_not_pulled_away_by_jumps_of_the_tracker(
    motion_name: str,
    observation_noise: ObservationNoise,
    exact_observation: ObservationNoise,
):
    """
    Two observations that jumped by 5 cm and 0.2 rad leave the other poses the fit
    predicts within a few standard deviations of the true ones.
    """
    observations = screw_motions()[motion_name].observe(noise=observation_noise, seed=7)
    jumped_stack = observations.parent_T_child_stack.copy()
    jumped_indices = [4, 9]
    jumped_stack[jumped_indices, :3, 3] += 0.05 * np.array([[0, 0.6, 0.8], [1, 0, 0]])
    jumped_stack[jumped_indices, :3, :3] = (
        Rotation.from_rotvec([[0.2, 0, 0], [0, 0, -0.2]])
        * Rotation.from_matrix(jumped_stack[jumped_indices, :3, :3])
    ).as_matrix()

    fit = ScrewMotionFit.fit(
        observations=PoseObservations(parent_T_child_stack=jumped_stack),
        noise=observation_noise,
    )

    true_parent_T_child_stack = (
        screw_motions()[motion_name].observe(noise=exact_observation, seed=7)
    ).parent_T_child_stack
    kept_indices = np.setdiff1d(np.arange(len(jumped_stack)), jumped_indices)
    predicted = fit.predicted_parent_T_child_stack[kept_indices]
    true = true_parent_T_child_stack[kept_indices]
    assert np.max(np.linalg.norm(predicted[:, :3, 3] - true[:, :3, 3], axis=1)) < 0.003
    assert (
        np.max(
            (
                Rotation.from_matrix(predicted[:, :3, :3]).inv()
                * Rotation.from_matrix(true[:, :3, :3])
            ).magnitude()
        )
        < 0.01
    )


@pytest.mark.parametrize("motion_name", turning_motions)
def test_fit_recovers_the_screw_axis_of_a_turning_child(
    motion_name: str, observation_noise: ObservationNoise
):
    """
    For a child that turns, the fit recovers the line of the screw axis, how far the
    child advances along it per radian and the position at each observation.
    """
    motion = screw_motions()[motion_name]

    fit = ScrewMotionFit.fit(
        observations=motion.observe(noise=observation_noise, seed=2),
        noise=observation_noise,
    )

    assert_allclose(
        fit.screw_axis.unit_direction[:3], motion.axis_direction_in_parent, atol=0.01
    )
    assert (
        motion.distance_from_axis(point=fit.screw_axis.axis_point_closest_to_origin)
        < 0.01
    )
    assert float(fit.screw_axis.translation_per_radian) == pytest.approx(
        float(
            motion.screw_axis.translation_per_unit_position.dot(
                motion.screw_axis.unit_direction
            )
        ),
        abs=0.002,
    )
    assert_allclose(fit.positions, motion.positions, atol=0.01)


def test_fit_leaves_the_noise_in_the_screw_axis(observation_noise: ObservationNoise):
    """
    The fit describes the motion it sees and does not round it to a kind of connection:
    noise leaves a door a small pitch.
    """
    fit = ScrewMotionFit.fit(
        observations=screw_motions()["revolute"].observe(
            noise=observation_noise, seed=3
        ),
        noise=observation_noise,
    )

    translation_per_radian = float(fit.screw_axis.translation_per_radian)
    assert translation_per_radian != 0
    assert abs(translation_per_radian) < 0.002


@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_fit_of_exact_observations_has_no_error(
    motion_name: str, exact_observation: ObservationNoise
):
    """
    Poses observed without noise are reproduced exactly.
    """
    noise = ObservationNoise(
        position_standard_deviation=1.0, orientation_standard_deviation=1.0
    )
    observations = screw_motions()[motion_name].observe(noise=exact_observation, seed=0)

    fit = ScrewMotionFit.fit(observations=observations, noise=noise)

    assert_allclose(fit.standardized_errors, 0, atol=1e-6)


def test_fit_of_an_exact_translation_does_not_turn(
    exact_observation: ObservationNoise,
):
    """
    Without noise, a translation is fit as one: the child does not turn, and its
    positions are the distances it slid.
    """
    noise = ObservationNoise(
        position_standard_deviation=0.001, orientation_standard_deviation=0.005
    )
    motion = screw_motions()["prismatic"]

    fit = ScrewMotionFit.fit(
        observations=motion.observe(noise=exact_observation, seed=0), noise=noise
    )

    assert not fit.screw_axis.turns_body
    assert_allclose(
        fit.screw_axis.unit_direction[:3], motion.axis_direction_in_parent, atol=1e-9
    )
    assert_allclose(fit.positions, motion.positions, atol=1e-9)


def test_fit_of_an_exact_rotation_has_no_pitch(exact_observation: ObservationNoise):
    """
    Without noise, a rotation is fit without advancing along its axis.
    """
    noise = ObservationNoise(
        position_standard_deviation=0.001, orientation_standard_deviation=0.005
    )

    fit = ScrewMotionFit.fit(
        observations=screw_motions()["revolute"].observe(
            noise=exact_observation, seed=0
        ),
        noise=noise,
    )

    assert float(fit.screw_axis.translation_per_radian) == pytest.approx(0, abs=1e-9)


def test_fit_follows_a_rotation_past_half_a_revolution(
    observation_noise: ObservationNoise,
):
    """
    Positions keep growing past π, where the rotation vector of a pose flips, as long as
    consecutive observations are less than half a revolution apart.
    """
    motion = ObservedScrewMotion(
        screw_axis=ScrewAxisConnection.create_revolute_screw_axis(
            axis=Vector3.unit_vector(1, 2, 2)
        ),
        positions=np.linspace(0.0, 5.0, 15),
    )

    fit = ScrewMotionFit.fit(
        observations=motion.observe(noise=observation_noise, seed=6),
        noise=observation_noise,
    )

    assert_allclose(fit.positions, motion.positions, atol=0.01)


# %% Created connections
def fitted_world(*, fit: ScrewMotionFit) -> tuple[World, Body, Body, Connection]:
    """
    :return: A world in which a connection that ``fit`` creates joins a parent and a
        child.
    """
    world = World()
    parent = Body(name=PrefixedName("parent"))
    child = Body(name=PrefixedName("child"))
    with world.modify_world():
        connection = fit.create_connection(world=world, parent=parent, child=child)
        world.add_connection(connection)
    return world, parent, child, connection


@pytest.mark.parametrize("motion_name", list(screw_motions()))
@pytest.mark.parametrize("noise_name", ["observation_noise", "exact_observation"])
def test_created_connection_moves_the_child_as_the_fit_predicts(
    motion_name: str, noise_name: str, request: pytest.FixtureRequest
):
    """
    At each fitted position, the created connection puts the child where the fit
    predicts it for that observation, whether the fit turns the child or not.
    """
    noise = ObservationNoise(
        position_standard_deviation=0.001, orientation_standard_deviation=0.005
    )
    fit = ScrewMotionFit.fit(
        observations=screw_motions()[motion_name].observe(
            noise=request.getfixturevalue(noise_name), seed=4
        ),
        noise=noise,
    )
    world, parent, child, connection = fitted_world(fit=fit)

    for position, predicted in zip(fit.positions, fit.predicted_parent_T_child_stack):
        connection.position = position
        assert_allclose(
            world.compute_forward_kinematics_np(root=parent, tip=child),
            predicted,
            atol=1e-9,
        )


@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_created_connection_takes_the_fitted_screw_axis_as_it_is(
    motion_name: str, observation_noise: ObservationNoise
):
    """
    The connection frame sits on the fitted screw axis, along the parent's axes, and the
    connection's screw axis is the fitted one, moved there: the same rotation, and a
    translation along the axis by the fitted pitch.
    """
    fit = ScrewMotionFit.fit(
        observations=screw_motions()[motion_name].observe(
            noise=observation_noise, seed=5
        ),
        noise=observation_noise,
    )
    world, parent, child, connection = fitted_world(fit=fit)

    parent_T_connection = connection.parent_T_connection_expression.to_np()
    assert_allclose(parent_T_connection[:3, :3], np.eye(3))
    assert_allclose(
        parent_T_connection[:3, 3], fit.screw_axis.axis_point_closest_to_origin[:3]
    )
    rotation = fit.screw_axis.rotation_per_unit_position.to_np()[:3]
    assert_allclose(connection.screw_axis.rotation_per_unit_position[:3], rotation)
    expected_translation = (
        float(fit.screw_axis.translation_per_radian) * rotation
        if fit.screw_axis.turns_body
        else fit.screw_axis.translation_per_unit_position.to_np()[:3]
    )
    assert_allclose(
        connection.screw_axis.translation_per_unit_position[:3],
        expected_translation,
        atol=1e-12,
    )


def test_fit_with_an_unnormalized_screw_axis_raises(
    observation_noise: ObservationNoise,
):
    """
    A fit holds its screw axis normalized, so one that is not raises.
    """
    fit = ScrewMotionFit.fit(
        observations=screw_motions()["revolute"].observe(
            noise=observation_noise, seed=6
        ),
        noise=observation_noise,
    )

    with pytest.raises(UnnormalizedScrewAxisError):
        replace(
            fit,
            screw_axis=ScrewAxis(
                rotation_per_unit_position=Vector3(0, 0, 2),
                translation_per_unit_position=Vector3(),
            ),
        )
