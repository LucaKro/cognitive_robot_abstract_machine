"""
Fits of a resting child and of motion along a screw axis to observed poses, compared as
in Sturm, Stachniss and Burgard, `A Probabilistic Framework for Learning Kinematic
Models of Articulated Objects <https://arxiv.org/abs/1405.7705>`__.
"""

from __future__ import annotations

# %% Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING, Self

import krrood.symbolic_math.symbolic_math as sm
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import CompiledFunction
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    ScrewAxis,
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
class ScrewMotionFit(ArticulationFit):
    """
    The child moves along a screw axis in proportion to a position that differs between
    observations, as a :class:`ScrewAxisConnection` moves it.

    It may turn, slide or both: the fit describes the motion and does not choose a kind
    of connection.
    """

    screw_axis: ScrewAxis
    """
    How the child moves per unit of position, in the parent frame; once :meth:`fit`
    returns, normalized as Lynch and Park define it.
    """

    positions: np.ndarray
    """
    The position at each observation, zero at the first and positive at the last.
    """

    displacement_along_screw_axis: CompiledFunction = field(repr=False, compare=False)
    """
    :meth:`~krrood_spatial_math.spatial_types.ScrewAxis.to_displacement`, compiled, see
    :meth:`compile_displacement_along_screw_axis`.
    """

    @classproperty
    def negligible_rotation(cls) -> float:
        """
        The rotation over the observed positions, in radians, below which :meth:`fit`
        takes the child not to turn at all.
        """
        return 1e-9

    @classmethod
    def fit(cls, *, observations: PoseObservations, noise: ObservationNoise) -> Self:
        initial_fit = cls.initial_fit(observations=observations, noise=noise)
        result = least_squares(
            fun=lambda parameter_vector: noise.standardized_pose_errors(
                observed=observations.parent_T_child,
                predicted=initial_fit.predicted_parent_T_child_at(
                    parameter_vector=parameter_vector
                ),
            ),
            x0=initial_fit.parameter_vector,
        )
        return initial_fit.copy_with_parameter_vector(
            parameter_vector=result.x
        ).copy_with_normalized_screw_axis()

    @classmethod
    def initial_fit(
        cls, *, observations: PoseObservations, noise: ObservationNoise
    ) -> Self:
        """
        :return: An approximation in closed form, which takes the first observation as
            exact and the steps between consecutive observations as motions along one
            screw axis, each by the change of position between them.
        """
        displacement_along_screw_axis = cls.compile_displacement_along_screw_axis()
        steps = observations.displacements_between_observations
        rotation_vectors = Rotation.from_matrix(steps[:, :3, :3]).as_rotvec()
        # A step turns by r = Δθ ω and moves by J(r) Δθ v, with J the left Jacobian of
        # r, whose columns the displacement along (r, unit vector) by 1 gives.
        left_jacobians = np.stack(
            [
                np.stack(
                    [
                        displacement_along_screw_axis(
                            np.concatenate([rotation_vector, unit_vector, [1.0]])
                        )[3:].copy()
                        for unit_vector in np.eye(3)
                    ],
                    axis=1,
                )
                for rotation_vector in rotation_vectors
            ]
        )
        translations = np.linalg.solve(left_jacobians, steps[:, :3, 3, np.newaxis])
        # Each row is Δθ (ω, v), weighted as the errors are, so the rows have rank one.
        standardized_steps = np.hstack(
            [
                rotation_vectors / noise.orientation_standard_deviation,
                translations[:, :, 0] / noise.position_standard_deviation,
            ]
        )
        left, singular_values, right = np.linalg.svd(
            standardized_steps, full_matrices=False
        )
        position_steps = left[:, 0] * singular_values[0]
        orientation = np.copysign(1.0, position_steps.sum())
        return cls(
            observations=observations,
            noise=noise,
            reference_parent_T_child=observations.parent_T_child[0],
            screw_axis=ScrewAxis(
                rotation_per_unit_position=Vector3(
                    *(orientation * right[0, :3] * noise.orientation_standard_deviation)
                ),
                translation_per_unit_position=Vector3(
                    *(orientation * right[0, 3:] * noise.position_standard_deviation)
                ),
            ),
            positions=np.concatenate([[0.0], np.cumsum(orientation * position_steps)]),
            displacement_along_screw_axis=displacement_along_screw_axis,
        )

    @classmethod
    def compile_displacement_along_screw_axis(cls) -> CompiledFunction:
        """
        :return: A function of the rotation and the translation per unit of position and
            of the position, seven numbers, that returns the rotation vector and the
            translation of the displacement along that screw axis by that position, in
            an array it overwrites on the next call.
        """
        rotation = [sm.FloatVariable(name=f"rotation_{axis}") for axis in "xyz"]
        translation = [sm.FloatVariable(name=f"translation_{axis}") for axis in "xyz"]
        position = sm.FloatVariable(name="position")
        displacement = ScrewAxis(
            rotation_per_unit_position=Vector3(*rotation),
            translation_per_unit_position=Vector3(*translation),
        ).to_displacement(position=position)
        return sm.Matrix(
            [
                *(displacement.rotation_vector[index] for index in range(3)),
                *(displacement.translation[index] for index in range(3)),
            ]
        ).compile(
            parameters=sm.VariableParameters.from_lists(
                rotation + translation + [position]
            )
        )

    @property
    def fitted_parameter_count(self) -> int:
        # Six numbers set the screw axis, less the scale it shares with the positions.
        return self.pose_parameter_count + len(self.positions) - 1 + 5

    @property
    def parameter_vector(self) -> np.ndarray:
        """
        The numbers the optimizer of :meth:`fit` varies: the rotation vector and
        position of :attr:`reference_parent_T_child`, the positions between the first
        and the last, and the six entries of :attr:`screw_axis`.
        """
        return np.concatenate(
            [
                Rotation.from_matrix(self.reference_parent_T_child[:3, :3]).as_rotvec(),
                self.reference_parent_T_child[:3, 3],
                self.positions[1:-1],
                self.screw_axis.to_np(),
            ]
        )

    def split_parameter_vector(
        self, *, parameter_vector: np.ndarray
    ) -> tuple[NpMatrix4x4, np.ndarray, np.ndarray]:
        """
        :param parameter_vector: Numbers in the layout of :attr:`parameter_vector`.
        :return: The pose of the child at position zero, the position at each
            observation and the six entries of the screw axis that they describe. The
            last position stays the fit's: scaling the screw axis and dividing the
            positions by the same factor would not change the motion.
        """
        positions_end = self.pose_parameter_count + len(self.positions) - 2
        reference_parent_T_child = np.eye(4)
        reference_parent_T_child[:3, :3] = Rotation.from_rotvec(
            parameter_vector[:3]
        ).as_matrix()
        reference_parent_T_child[:3, 3] = parameter_vector[
            3 : self.pose_parameter_count
        ]
        positions = np.concatenate(
            [
                [0.0],
                parameter_vector[self.pose_parameter_count : positions_end],
                self.positions[-1:],
            ]
        )
        return reference_parent_T_child, positions, parameter_vector[positions_end:]

    def predicted_parent_T_child_at(
        self, *, parameter_vector: np.ndarray
    ) -> np.ndarray:
        """
        :param parameter_vector: Numbers in the layout of :attr:`parameter_vector`.
        :return: The pose of the child they predict for each observation, stacked to
            shape (n, 4, 4), computed without building spatial types, as :meth:`fit`
            asks for it at every step of its optimizer.
        """
        reference_parent_T_child, positions, screw_axis_entries = (
            self.split_parameter_vector(parameter_vector=parameter_vector)
        )
        return (
            self.displacements(
                screw_axis_entries=screw_axis_entries, positions=positions
            )
            @ reference_parent_T_child
        )

    @property
    def predicted_parent_T_child(self) -> np.ndarray:
        return self.predicted_parent_T_child_at(parameter_vector=self.parameter_vector)

    def copy_with_parameter_vector(self, *, parameter_vector: np.ndarray) -> Self:
        """
        :param parameter_vector: Numbers in the layout of :attr:`parameter_vector`.
        :return: A copy of the fit with the poses, positions and screw axis they
            describe, see :meth:`split_parameter_vector`.
        """
        reference_parent_T_child, positions, screw_axis_entries = (
            self.split_parameter_vector(parameter_vector=parameter_vector)
        )
        return replace(
            self,
            reference_parent_T_child=reference_parent_T_child,
            positions=positions,
            screw_axis=ScrewAxis(
                rotation_per_unit_position=Vector3(*screw_axis_entries[:3]),
                translation_per_unit_position=Vector3(*screw_axis_entries[3:]),
            ),
        )

    def copy_with_normalized_screw_axis(self) -> Self:
        """
        :return: A copy of the fit describing the same motion, with the screw axis
            normalized as Lynch and Park define it and the positions scaled to match. A
            rotation below :attr:`negligible_rotation` is set to zero.
        """
        rotation, translation = np.split(self.screw_axis.to_np(), 2)
        if np.linalg.norm(rotation) * np.ptp(self.positions) < self.negligible_rotation:
            rotation = np.zeros(3)
        scale = np.linalg.norm(rotation if np.any(rotation) else translation)
        return replace(
            self,
            screw_axis=ScrewAxis(
                rotation_per_unit_position=Vector3(*(rotation / scale)),
                translation_per_unit_position=Vector3(*(translation / scale)),
            ),
            positions=self.positions * scale,
        )

    def displacements(
        self, *, screw_axis_entries: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """
        :param screw_axis_entries: The six entries of a screw axis, as
            :meth:`~krrood_spatial_math.spatial_types.ScrewAxis.to_np` returns them.
        :param positions: Positions along that screw axis.
        :return: For each position, the transformation in the parent frame that moves
            the child from its pose at position zero to its pose at that position,
            stacked to shape (n, 4, 4).
        """
        rotation_vectors_and_translations = np.stack(
            [
                self.displacement_along_screw_axis(
                    np.concatenate([screw_axis_entries, [position]])
                ).copy()
                for position in positions
            ]
        )
        displacements = np.tile(np.eye(4), (len(positions), 1, 1))
        displacements[:, :3, :3] = Rotation.from_rotvec(
            rotation_vectors_and_translations[:, :3]
        ).as_matrix()
        displacements[:, :3, 3] = rotation_vectors_and_translations[:, 3:]
        return displacements

    @property
    def turns_child(self) -> bool:
        """
        Whether the position turns the child.
        """
        return bool(np.any(self.screw_axis.rotation_per_unit_position.to_np()[:3]))

    @property
    def connection_origin(self) -> Point3:
        """
        The point, in the parent frame, at which :meth:`create_connection` places the
        connection frame: on the screw axis, closest to the child's origin at position
        zero, or that origin itself if the child does not turn.
        """
        child_origin = Point3(*self.reference_parent_T_child[:3, 3])
        if not self.turns_child:
            return child_origin
        axis_point, _ = child_origin.project_to_line(
            line_point=self.screw_axis.axis_point_closest_to_origin,
            line_direction=self.screw_axis.unit_direction,
        )
        return axis_point

    @property
    def parent_T_connection(self) -> HomogeneousTransformationMatrix:
        """
        The connection frame of :meth:`create_connection`: at :attr:`connection_origin`,
        its z axis along the screw axis and its x axis toward the child's origin at
        position zero, or, if that lies on the axis, perpendicular to the axis.
        """
        z_axis = self.screw_axis.unit_direction
        connection_origin = self.connection_origin
        toward_child = Point3(*self.reference_parent_T_child[:3, 3]) - connection_origin
        toward_child = toward_child - z_axis * toward_child.dot(z_axis)
        if np.isclose(float(toward_child.norm()), 0):
            toward_child = RotationMatrix.from_x_axis(z_axis).y_vector()
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=connection_origin,
            rotation_matrix=RotationMatrix.from_vectors(x=toward_child, z=z_axis),
        )

    @property
    def connection_screw_axis(self) -> ScrewAxis:
        """
        The screw axis of the connection :meth:`create_connection` creates, along the z
        axis of its connection frame.
        """
        if not self.turns_child:
            return ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.Z())
        return ScrewAxisConnection.create_helical_screw_axis(
            axis=Vector3.Z(),
            pitch=2 * np.pi * float(self.screw_axis.translation_per_radian),
        )

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
            parent_T_connection_expression=parent_T_connection,
            connection_T_child_expression=parent_T_connection.inverse()
            @ HomogeneousTransformationMatrix(data=self.reference_parent_T_child),
            screw_axis=self.connection_screw_axis,
        )
