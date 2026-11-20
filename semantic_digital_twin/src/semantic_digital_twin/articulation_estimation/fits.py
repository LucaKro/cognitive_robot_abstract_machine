"""
Fits of rigid, prismatic, revolute and helical motion to observed poses, compared as
in Sturm, Stachniss and Burgard, `A Probabilistic Framework for Learning Kinematic
Models of Articulated Objects <https://arxiv.org/abs/1405.7705>`__.
"""

from __future__ import annotations

# %% Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING, Self

from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Twist,
    Vector3,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ScrewAxisConnection,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import (
        Connection,
        KinematicStructureEntity,
    )


# %% Any kind of motion
@dataclass(kw_only=True)
class ArticulationFit(ABC):
    """
    One kind of motion of a child relative to its parent, fit to observed poses.
    """

    observations: PoseObservations
    """
    The poses the fit explains.
    """

    noise: ObservationNoise
    """
    The noise the observations are assumed to carry.
    """

    reference_parent_T_child: NpMatrix4x4
    """
    The pose of the child relative to its parent at position zero, or, for a child that
    does not move, its only pose.
    """

    @classmethod
    @abstractmethod
    def fit(cls, *, observations: PoseObservations, noise: ObservationNoise) -> Self:
        """
        :param observations: The observed poses.
        :param noise: The noise the observations carry.
        :return: The fit of this kind of motion under which the observations are most
            likely.
        """

    @classproperty
    def pose_parameter_count(cls) -> int:
        """
        The number of independent numbers in a pose: three for its position and three
        for its orientation.
        """
        return 6

    @property
    @abstractmethod
    def fitted_parameter_count(self) -> int:
        """
        The number of independent numbers the fit chose, the position at each
        observation included.
        """

    @property
    @abstractmethod
    def predicted_parent_T_child(self) -> np.ndarray:
        """
        The pose of the child the fit predicts for each observation, stacked to shape
        (n, 4, 4).
        """

    @abstractmethod
    def create_connection(
        self,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
    ) -> Connection:
        """
        Creates a connection that moves ``child`` as the fit does, with its degrees of
        freedom added to ``world``.

        :param world: The world in which to add the degrees of freedom.
        :param parent: The parent of the observed poses.
        :param child: The child of the observed poses.
        :return: The connection, not yet added to ``world``.
        """

    @property
    def standardized_errors(self) -> np.ndarray:
        """
        The errors of :attr:`predicted_parent_T_child`, see
        :meth:`ObservationNoise.standardized_pose_errors`.
        """
        return self.noise.standardized_pose_errors(
            observed=self.observations.parent_T_child,
            predicted=self.predicted_parent_T_child,
        )

    @property
    def bayesian_information_criterion(self) -> float:
        """
        The `Bayesian information criterion
        <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`__ of the fit, up
        to a constant that every fit of the same observations shares: the lower, the
        more likely the fit, after a penalty for each number it chose.
        """
        errors = self.standardized_errors
        return float(
            errors @ errors + self.fitted_parameter_count * np.log(errors.size)
        )


# %% No motion
@dataclass(kw_only=True)
class RigidFit(ArticulationFit):
    """
    The child does not move relative to its parent.
    """

    @classmethod
    def fit(cls, *, observations: PoseObservations, noise: ObservationNoise) -> Self:
        poses = observations.parent_T_child
        reference_parent_T_child = np.eye(4)
        reference_parent_T_child[:3, :3] = (
            Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix()
        )
        reference_parent_T_child[:3, 3] = poses[:, :3, 3].mean(axis=0)
        return cls(
            observations=observations,
            noise=noise,
            reference_parent_T_child=reference_parent_T_child,
        )

    @property
    def fitted_parameter_count(self) -> int:
        return self.pose_parameter_count

    @property
    def predicted_parent_T_child(self) -> np.ndarray:
        return np.broadcast_to(
            self.reference_parent_T_child, self.observations.parent_T_child.shape
        )

    def create_connection(
        self,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
    ) -> FixedConnection:
        return FixedConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                data=self.reference_parent_T_child
            ),
        )


# %% Motion along a screw axis
@dataclass(kw_only=True)
class ScrewMotionFit(ArticulationFit, ABC):
    """
    The child moves along a screw axis in proportion to a position that differs between
    observations, as a :class:`ScrewAxisConnection` moves it.

    At each of
    :attr:`positions`, the connection :meth:`create_connection` creates moves the child
    to the pose the fit predicts for that observation.
    """

    axis_direction: np.ndarray
    """
    The unit direction of the screw axis, in the parent frame.
    """

    positions: np.ndarray
    """
    The position at each observation, zero at the first and, as far as the axis
    direction can make it, positive at the last.
    """

    @classmethod
    @abstractmethod
    def initial_fit(
        cls, *, observations: PoseObservations, noise: ObservationNoise
    ) -> Self:
        """
        :param observations: The observed poses.
        :param noise: The noise the observations carry.
        :return: An approximation of :meth:`fit` in closed form, which treats the first
            observation as exact.
        """

    @classproperty
    @abstractmethod
    def axis_parameter_count(cls) -> int:
        """
        The number of independent numbers that place the screw axis and set its pitch.
        """

    @property
    @abstractmethod
    def axis_parameters(self) -> np.ndarray:
        """
        The numbers that place the screw axis and set its pitch, as the optimizer of
        :meth:`fit` varies them.
        """

    @abstractmethod
    def copy_with_axis_parameters(self, *, axis_parameters: np.ndarray) -> Self:
        """
        :param axis_parameters: The numbers that place the screw axis and set its pitch,
            in the layout of :attr:`axis_parameters`.
        :return: A copy of the fit with the screw axis they describe, its direction
            scaled to unit length.
        """

    @abstractmethod
    def displacements(self, *, positions: np.ndarray) -> np.ndarray:
        """
        :param positions: Positions along the screw axis.
        :return: For each position, the transformation in the parent frame that moves
            the child from its pose at position zero to its pose at that position,
            stacked to shape (n, 4, 4).
        """

    @property
    @abstractmethod
    def connection_origin(self) -> np.ndarray:
        """
        The point on the screw axis, in the parent frame, at which
        :meth:`create_connection` places the connection frame.
        """

    @property
    @abstractmethod
    def connection_screw_axis(self) -> Twist:
        """
        The screw axis of the connection :meth:`create_connection` creates, along the z
        axis of its connection frame.
        """

    @classmethod
    def fit(cls, *, observations: PoseObservations, noise: ObservationNoise) -> Self:
        initial_fit = cls.initial_fit(observations=observations, noise=noise)
        result = least_squares(
            fun=lambda parameter_vector: (
                initial_fit.copy_with_parameter_vector(
                    parameter_vector=parameter_vector
                ).standardized_errors
            ),
            x0=initial_fit.parameter_vector,
        )
        return initial_fit.copy_with_parameter_vector(parameter_vector=result.x)

    @property
    def fitted_parameter_count(self) -> int:
        return (
            self.pose_parameter_count
            + len(self.positions)
            - 1
            + self.axis_parameter_count
        )

    @property
    def parameter_vector(self) -> np.ndarray:
        """
        The numbers the optimizer of :meth:`fit` varies: the rotation vector and
        position of :attr:`reference_parent_T_child`, the positions after the first, and
        :attr:`axis_parameters`.
        """
        return np.concatenate(
            [
                Rotation.from_matrix(self.reference_parent_T_child[:3, :3]).as_rotvec(),
                self.reference_parent_T_child[:3, 3],
                self.positions[1:],
                self.axis_parameters,
            ]
        )

    def copy_with_parameter_vector(self, *, parameter_vector: np.ndarray) -> Self:
        """
        :param parameter_vector: The numbers the optimizer of :meth:`fit` varies, in the
            layout of :attr:`parameter_vector`.
        :return: A copy of the fit with the poses, positions and screw axis they
            describe.
        """
        axis_parameters_start = self.pose_parameter_count + len(self.positions) - 1
        reference_parent_T_child = np.eye(4)
        reference_parent_T_child[:3, :3] = Rotation.from_rotvec(
            parameter_vector[:3]
        ).as_matrix()
        reference_parent_T_child[:3, 3] = parameter_vector[
            3 : self.pose_parameter_count
        ]
        positions = np.concatenate(
            [[0.0], parameter_vector[self.pose_parameter_count : axis_parameters_start]]
        )
        return replace(
            self,
            reference_parent_T_child=reference_parent_T_child,
            positions=positions,
        ).copy_with_axis_parameters(
            axis_parameters=parameter_vector[axis_parameters_start:]
        )

    @property
    def predicted_parent_T_child(self) -> np.ndarray:
        return (
            self.displacements(positions=self.positions) @ self.reference_parent_T_child
        )

    @property
    def lowest_observed_position(self) -> float:
        """
        The lowest of :attr:`positions`.
        """
        return float(self.positions.min())

    @property
    def highest_observed_position(self) -> float:
        """
        The highest of :attr:`positions`.
        """
        return float(self.positions.max())

    @property
    def parent_T_connection(self) -> NpMatrix4x4:
        """
        The connection frame of :meth:`create_connection`: at :attr:`connection_origin`,
        its z axis along :attr:`axis_direction` and its x axis toward the child's origin
        at position zero, or, if that lies on the axis, perpendicular to the axis.
        """
        z_axis = self.axis_direction
        toward_child = self.reference_parent_T_child[:3, 3] - self.connection_origin
        toward_child -= (toward_child @ z_axis) * z_axis
        if np.isclose(np.linalg.norm(toward_child), 0):
            least_aligned_unit_vector = np.eye(3)[np.argmin(np.abs(z_axis))]
            toward_child = (
                least_aligned_unit_vector
                - (least_aligned_unit_vector @ z_axis) * z_axis
            )
        x_axis = toward_child / np.linalg.norm(toward_child)
        parent_T_connection = np.eye(4)
        parent_T_connection[:3, 0] = x_axis
        parent_T_connection[:3, 1] = np.cross(z_axis, x_axis)
        parent_T_connection[:3, 2] = z_axis
        parent_T_connection[:3, 3] = self.connection_origin
        return parent_T_connection

    def create_connection(
        self,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
    ) -> ScrewAxisConnection:
        parent_T_connection = self.parent_T_connection
        return ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                data=parent_T_connection
            ),
            connection_T_child_expression=HomogeneousTransformationMatrix(
                data=np.linalg.inv(parent_T_connection) @ self.reference_parent_T_child
            ),
            screw_axis=self.connection_screw_axis,
        )


# %% Translation
@dataclass(kw_only=True)
class PrismaticFit(ScrewMotionFit):
    """
    The child translates along a direction, so its positions are in meters.

    A translation has no axis position, so the connection frame sits at the child's
    origin at position zero.
    """

    @classmethod
    def initial_fit(
        cls, *, observations: PoseObservations, noise: ObservationNoise
    ) -> Self:
        translations = observations.displacements_from_first[:, :3, 3]
        direction = np.linalg.svd(translations)[2][0]
        direction *= np.copysign(1.0, translations[-1] @ direction)
        return cls(
            observations=observations,
            noise=noise,
            reference_parent_T_child=observations.parent_T_child[0],
            axis_direction=direction,
            positions=translations @ direction,
        )

    @classproperty
    def axis_parameter_count(cls) -> int:
        return 2

    @property
    def axis_parameters(self) -> np.ndarray:
        return self.axis_direction

    def copy_with_axis_parameters(self, *, axis_parameters: np.ndarray) -> Self:
        return replace(
            self, axis_direction=axis_parameters / np.linalg.norm(axis_parameters)
        )

    def displacements(self, *, positions: np.ndarray) -> np.ndarray:
        displacements = np.tile(np.eye(4), (len(positions), 1, 1))
        displacements[:, :3, 3] = positions[:, np.newaxis] * self.axis_direction
        return displacements

    @property
    def connection_origin(self) -> np.ndarray:
        return self.reference_parent_T_child[:3, 3]

    @property
    def connection_screw_axis(self) -> Twist:
        return ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.Z())


# %% Rotation
@dataclass(kw_only=True)
class RotatingFit(ScrewMotionFit, ABC):
    """
    The child rotates about an axis, and may advance along it in proportion, so its
    positions are in radians.

    The connection frame sits at the point of the axis closest to the child's origin at
    position zero.
    """

    axis_point: np.ndarray
    """
    The point of the screw axis closest to the parent's origin, in the parent frame.
    """

    translation_per_radian: float
    """
    How far the child advances along :attr:`axis_direction` per radian, in meters; Lynch
    and Park's pitch.
    """

    @classmethod
    def initial_fit(
        cls, *, observations: PoseObservations, noise: ObservationNoise
    ) -> Self:
        # A displacement rotates by R about the axis and translates by (I - R) q + h θ ŝ,
        # which is linear in the axis point q and the pitch h.
        displacements = observations.displacements_from_first
        rotation_vectors = Rotation.from_matrix(displacements[:, :3, :3]).as_rotvec()
        direction = np.linalg.svd(rotation_vectors)[2][0]
        angles = np.unwrap(rotation_vectors @ direction)
        orientation = np.copysign(1.0, angles[-1])
        direction *= orientation
        angles *= orientation
        coefficients = np.concatenate(
            [
                np.eye(3) - displacements[:, :3, :3],
                (angles[:, np.newaxis] * direction)[:, :, np.newaxis],
            ],
            axis=2,
        ).reshape(-1, 4)
        axis_point_perpendicular_to_direction = np.append(direction, 0.0)
        solution = np.linalg.lstsq(
            np.vstack([coefficients, axis_point_perpendicular_to_direction]),
            np.append(displacements[:, :3, 3].ravel(), 0.0),
            rcond=None,
        )[0]
        initial_fit = cls(
            observations=observations,
            noise=noise,
            reference_parent_T_child=observations.parent_T_child[0],
            axis_direction=direction,
            positions=angles,
            axis_point=solution[:3],
            translation_per_radian=float(solution[3]),
        )
        return initial_fit.copy_with_axis_parameters(
            axis_parameters=initial_fit.axis_parameters
        )

    def displacements(self, *, positions: np.ndarray) -> np.ndarray:
        rotations = Rotation.from_rotvec(
            positions[:, np.newaxis] * self.axis_direction
        ).as_matrix()
        displacements = np.tile(np.eye(4), (len(positions), 1, 1))
        displacements[:, :3, :3] = rotations
        displacements[:, :3, 3] = (
            np.eye(3) - rotations
        ) @ self.axis_point + self.translation_per_radian * positions[
            :, np.newaxis
        ] * self.axis_direction
        return displacements

    @property
    def connection_origin(self) -> np.ndarray:
        return (
            self.axis_point
            + (
                (self.reference_parent_T_child[:3, 3] - self.axis_point)
                @ self.axis_direction
            )
            * self.axis_direction
        )

    def copy_with_axis_parameters(self, *, axis_parameters: np.ndarray) -> Self:
        # The first six parameters are the direction and a point of the axis; the copy
        # keeps the point of the axis closest to the parent's origin.
        direction = axis_parameters[:3] / np.linalg.norm(axis_parameters[:3])
        return replace(
            self,
            axis_direction=direction,
            axis_point=axis_parameters[3:6]
            - (axis_parameters[3:6] @ direction) * direction,
        )


@dataclass(kw_only=True)
class RevoluteFit(RotatingFit):
    """
    The child rotates about an axis without advancing along it.
    """

    @classproperty
    def axis_parameter_count(cls) -> int:
        # Two for the direction, two for where the axis crosses the plane through the
        # parent's origin perpendicular to it.
        return 4

    @property
    def axis_parameters(self) -> np.ndarray:
        return np.concatenate([self.axis_direction, self.axis_point])

    def copy_with_axis_parameters(self, *, axis_parameters: np.ndarray) -> Self:
        """
        Extends :meth:`RotatingFit.copy_with_axis_parameters` by fixing
        :attr:`translation_per_radian` at zero.
        """
        return replace(
            super().copy_with_axis_parameters(axis_parameters=axis_parameters),
            translation_per_radian=0.0,
        )

    @property
    def connection_screw_axis(self) -> Twist:
        return ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z())


@dataclass(kw_only=True)
class HelicalFit(RotatingFit):
    """
    The child rotates about an axis and advances along it in proportion, as a screw
    does.
    """

    @classproperty
    def axis_parameter_count(cls) -> int:
        return RevoluteFit.axis_parameter_count + 1

    @property
    def axis_parameters(self) -> np.ndarray:
        return np.concatenate(
            [self.axis_direction, self.axis_point, [self.translation_per_radian]]
        )

    def copy_with_axis_parameters(self, *, axis_parameters: np.ndarray) -> Self:
        """
        Extends :meth:`RotatingFit.copy_with_axis_parameters`: the seventh parameter is
        :attr:`translation_per_radian`.
        """
        return replace(
            super().copy_with_axis_parameters(axis_parameters=axis_parameters),
            translation_per_radian=float(axis_parameters[6]),
        )

    @property
    def connection_screw_axis(self) -> Twist:
        return ScrewAxisConnection.create_helical_screw_axis(
            axis=Vector3.Z(), pitch=2 * np.pi * self.translation_per_radian
        )
