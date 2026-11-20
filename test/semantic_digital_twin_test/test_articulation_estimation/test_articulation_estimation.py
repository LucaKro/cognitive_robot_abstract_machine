"""
Tests of estimating how a child moves from noisy observations of its poses.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    ScrewAxis,
    Vector3,
)
from semantic_digital_twin.articulation_estimation.estimator import (
    ArticulationEstimator,
)
from semantic_digital_twin.articulation_estimation.exceptions import (
    TooFewObservationsError,
)
from semantic_digital_twin.articulation_estimation.fits import (
    ArticulationFit,
    HelicalFit,
    PrismaticFit,
    RevoluteFit,
    RigidFit,
    RotatingFit,
    ScrewMotionFit,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
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

    positions: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.2, 15))
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
            parent_T_child=disturbed_poses(
                poses=np.stack(poses), noise=noise, seed=seed
            )
        )

    @property
    def axis_direction_in_parent(self) -> np.ndarray:
        """
        The unit direction of the screw axis in the parent frame.
        """
        return (
            self.parent_T_connection.to_np()[:3, :3]
            @ self.screw_axis.unit_direction.to_np()[:3]
        )

    @property
    def axis_point_closest_to_parent_origin(self) -> np.ndarray:
        """
        The point of the screw axis closest to the parent's origin.
        """
        direction = self.axis_direction_in_parent
        through = self.parent_T_connection.to_np()[:3, 3]
        return through - (through @ direction) * direction


def disturbed_poses(
    *, poses: np.ndarray, noise: ObservationNoise, seed: int
) -> np.ndarray:
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


def screw_motions() -> dict[type[ScrewMotionFit], ObservedScrewMotion]:
    """
    :return: For each concrete kind of screw motion fit, a motion of that kind.
    """
    axis = Vector3.unit_vector(1, 2, 2)
    return {
        RevoluteFit: ObservedScrewMotion(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=axis)
        ),
        PrismaticFit: ObservedScrewMotion(
            screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(axis=axis),
            positions=np.linspace(0.0, 0.4, 15),
        ),
        HelicalFit: ObservedScrewMotion(
            screw_axis=ScrewAxisConnection.create_helical_screw_axis(
                axis=axis, pitch=0.3
            )
        ),
    }


# %% Model selection
@pytest.mark.parametrize("fit_type", list(screw_motions()))
def test_most_likely_fit_is_the_kind_of_motion_observed(
    fit_type: type[ScrewMotionFit], observation_noise: ObservationNoise
):
    """
    The estimator recognises revolute, prismatic and helical motion from noisy poses,
    and prefers revolute over helical motion, which also explains a rotation.
    """
    observations = screw_motions()[fit_type].observe(noise=observation_noise, seed=0)

    fit = ArticulationEstimator(noise=observation_noise).most_likely_fit(
        observations=observations
    )

    assert type(fit) is fit_type


def test_most_likely_fit_of_a_resting_child_is_rigid(
    observation_noise: ObservationNoise,
):
    """
    A child observed in the same pose, up to noise, is rigidly attached.
    """
    observations = ObservedScrewMotion(
        screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        positions=np.zeros(15),
    ).observe(noise=observation_noise, seed=0)

    fit = ArticulationEstimator(noise=observation_noise).most_likely_fit(
        observations=observations
    )

    assert type(fit) is RigidFit


def test_fewer_than_two_observations_raise():
    """
    One pose cannot show whether the child moves.
    """
    with pytest.raises(TooFewObservationsError):
        PoseObservations(parent_T_child=np.eye(4)[np.newaxis])


# %% Recovered parameters
@pytest.mark.parametrize("fit_type", list(screw_motions()))
def test_fit_recovers_the_axis_direction_and_positions(
    fit_type: type[ScrewMotionFit], observation_noise: ObservationNoise
):
    """
    The fit recovers the direction of the screw axis and the position at each
    observation, both oriented so that the positions grow as they did.
    """
    motion = screw_motions()[fit_type]

    fit = fit_type.fit(
        observations=motion.observe(noise=observation_noise, seed=1),
        noise=observation_noise,
    )

    assert_allclose(fit.axis_direction, motion.axis_direction_in_parent, atol=0.01)
    assert_allclose(fit.positions, motion.positions, atol=0.01)


@pytest.mark.parametrize("fit_type", [RevoluteFit, HelicalFit])
def test_rotating_fit_recovers_the_axis_point_and_pitch(
    fit_type: type[RotatingFit], observation_noise: ObservationNoise
):
    """
    A rotating fit recovers where the screw axis lies and how far the child advances
    along it per radian.
    """
    motion = screw_motions()[fit_type]

    fit = fit_type.fit(
        observations=motion.observe(noise=observation_noise, seed=2),
        noise=observation_noise,
    )

    assert_allclose(
        fit.axis_point, motion.axis_point_closest_to_parent_origin, atol=0.01
    )
    assert fit.translation_per_radian == pytest.approx(
        float(
            motion.screw_axis.translation_per_unit_position.dot(
                motion.screw_axis.unit_direction
            )
        ),
        abs=0.002,
    )


def test_fit_of_exact_observations_has_no_error():
    """
    Poses observed without noise are reproduced exactly by the fit of their kind.
    """
    noise = ObservationNoise(
        position_standard_deviation=1.0, orientation_standard_deviation=1.0
    )
    motion = screw_motions()[HelicalFit]
    observations = motion.observe(
        noise=ObservationNoise(
            position_standard_deviation=0.0, orientation_standard_deviation=0.0
        ),
        seed=0,
    )

    fit = HelicalFit.fit(observations=observations, noise=noise)

    assert_allclose(fit.standardized_errors, 0, atol=1e-6)


# %% Created connections
def fitted_world(*, fit: ArticulationFit) -> tuple[World, Body, Body, Connection]:
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


@pytest.mark.parametrize("fit_type", list(screw_motions()))
def test_created_connection_moves_the_child_as_the_fit_predicts(
    fit_type: type[ScrewMotionFit], observation_noise: ObservationNoise
):
    """
    At each fitted position, the created connection puts the child where the fit
    predicts it for that observation.
    """
    fit = fit_type.fit(
        observations=screw_motions()[fit_type].observe(noise=observation_noise, seed=3),
        noise=observation_noise,
    )
    world, parent, child, connection = fitted_world(fit=fit)

    for position, predicted in zip(fit.positions, fit.predicted_parent_T_child):
        connection.position = position
        assert_allclose(
            world.compute_forward_kinematics_np(root=parent, tip=child),
            predicted,
            atol=1e-9,
        )


@pytest.mark.parametrize("fit_type", [RevoluteFit, HelicalFit])
def test_created_connection_frame_sits_on_the_axis_next_to_the_child(
    fit_type: type[RotatingFit], observation_noise: ObservationNoise
):
    """
    The connection frame of a rotating fit has its z axis along the screw axis and its x
    axis toward the child, so the child sits on the frame's positive x axis.
    """
    fit = fit_type.fit(
        observations=screw_motions()[fit_type].observe(noise=observation_noise, seed=4),
        noise=observation_noise,
    )
    world, parent, child, connection = fitted_world(fit=fit)

    connection_P_child = connection.connection_T_child_expression.to_np()[:3, 3]
    assert_allclose(fit.parent_T_connection[:3, 2], fit.axis_direction)
    assert_allclose(connection_P_child[1:], 0, atol=1e-9)
    assert connection_P_child[0] > 0


def test_rigid_fit_creates_a_connection_holding_the_mean_pose(
    observation_noise: ObservationNoise,
):
    """
    A rigid fit's connection holds the child at the fit's single pose.
    """
    fit = RigidFit.fit(
        observations=ObservedScrewMotion(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
            positions=np.zeros(5),
        ).observe(noise=observation_noise, seed=5),
        noise=observation_noise,
    )
    world, parent, child, _ = fitted_world(fit=fit)

    assert_allclose(
        world.compute_forward_kinematics_np(root=parent, tip=child),
        fit.reference_parent_T_child,
        atol=1e-9,
    )


def test_revolute_fit_follows_a_rotation_past_half_a_revolution(
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

    fit = RevoluteFit.fit(
        observations=motion.observe(noise=observation_noise, seed=6),
        noise=observation_noise,
    )

    assert_allclose(fit.positions, motion.positions, atol=0.01)
