"""
A fit of motion along a screw axis to observed poses of a child relative to its parent.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
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
from semantic_digital_twin.world_description.connections import ScrewAxisConnection

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


# %% The displacement along a screw axis, as numbers
@dataclass(kw_only=True)
class CompiledScrewDisplacement:
    """
    :meth:`~krrood_spatial_math.spatial_types.ScrewAxis.to_displacement`, compiled into
    a function of numbers, for code that evaluates it too often to build spatial types
    every time, such as each step of an optimizer.
    """

    function: CompiledFunction = field(repr=False, compare=False)
    """
    The function of the six entries of a screw axis and a position, see :meth:`create`.
    """

    @classmethod
    def create(cls) -> Self:
        """
        :return: The displacement compiled into a function of the rotation and the
            translation per unit of position and of the position, seven numbers, that
            returns the rotation vector and the translation of the displacement.
        """
        rotation = [sm.FloatVariable(name=f"rotation_{axis}") for axis in "xyz"]
        translation = [sm.FloatVariable(name=f"translation_{axis}") for axis in "xyz"]
        position = sm.FloatVariable(name="position")
        displacement = ScrewAxis(
            rotation_per_unit_position=Vector3(*rotation),
            translation_per_unit_position=Vector3(*translation),
        ).to_displacement(position=position)
        return cls(
            function=sm.Matrix(
                [
                    *(displacement.rotation_vector[index] for index in range(3)),
                    *(displacement.translation[index] for index in range(3)),
                ]
            ).compile(
                parameters=sm.VariableParameters.from_lists(
                    rotation + translation + [position]
                )
            )
        )

    def rotation_vector_and_translation(
        self, *, screw_axis_entries: npt.NDArray[np.float64], position: float
    ) -> npt.NDArray[np.float64]:
        """
        :param screw_axis_entries: The six entries of a screw axis, as
            :meth:`~krrood_spatial_math.spatial_types.ScrewAxis.to_np` returns them.
        :param position: How far the body moves along that screw axis.
        :return: The rotation vector and the translation of the displacement, six
            numbers. A copy, since the compiled function overwrites its output.
        """
        return self.function(np.concatenate([screw_axis_entries, [position]])).copy()

    def displacements(
        self,
        *,
        screw_axis_entries: npt.NDArray[np.float64],
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        :param screw_axis_entries: The six entries of a screw axis.
        :param positions: Positions along that screw axis.
        :return: For each position, the transformation that moves the body from its pose
            at position zero to its pose at that position, stacked to shape (n, 4, 4).
        """
        rotation_vectors_and_translations = np.stack(
            [
                self.rotation_vector_and_translation(
                    screw_axis_entries=screw_axis_entries, position=position
                )
                for position in positions
            ]
        )
        displacements = np.tile(np.eye(4), (len(positions), 1, 1))
        displacements[:, :3, :3] = Rotation.from_rotvec(
            rotation_vectors_and_translations[:, :3]
        ).as_matrix()
        displacements[:, :3, 3] = rotation_vectors_and_translations[:, 3:]
        return displacements

    def left_jacobians(
        self, *, rotation_vectors: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :param rotation_vectors: Rotation vectors r, stacked to shape (n, 3).
        :return: For each r, the left Jacobian J(r) of the rotation, stacked to shape
            (n, 3, 3): moving by one along a screw axis that turns by r translates the
            body by J(r) times the screw axis's translation per unit of position, so
            column k of J(r) is that translation for a unit translation along axis k.
        """
        return np.stack(
            [
                np.stack(
                    [
                        self.rotation_vector_and_translation(
                            screw_axis_entries=np.concatenate(
                                [rotation_vector, unit_vector]
                            ),
                            position=1.0,
                        )[3:]
                        for unit_vector in np.eye(3)
                    ],
                    axis=1,
                )
                for rotation_vector in rotation_vectors
            ]
        )


# %% The numbers the optimizer varies
@dataclass(kw_only=True)
class ScrewMotionParameters:
    """
    A motion along a screw axis as plain numbers, in the parent frame: the pose of the
    child at position zero, the position at each observation and the six entries of the
    screw axis.

    :meth:`ScrewMotionFit.fit` varies them without building spatial types.
    """

    reference_parent_T_child: npt.NDArray[np.float64]
    """
    The pose of the child at position zero, a homogeneous transformation matrix.
    """

    positions: npt.NDArray[np.float64]
    """
    The position at each observation, zero at the first and positive at the last.
    """

    screw_axis_entries: npt.NDArray[np.float64]
    """
    The rotation and then the translation per unit of position, six numbers.
    """

    @classproperty
    def negligible_rotation(cls) -> float:
        """
        The rotation over the observed positions, in radians, below which
        :meth:`copy_with_normalized_screw_axis` takes the child not to turn at all.
        """
        return 1e-9

    @classmethod
    def initial_guess(
        cls,
        *,
        observations: PoseObservations,
        noise: ObservationNoise,
        displacement: CompiledScrewDisplacement,
    ) -> Self:
        """
        :return: A guess in closed form, exact for observations without noise: the
            first observation as the pose at position zero, and the closest rank one
            approximation of :meth:`standardized_steps`.
        """
        left, singular_values, right = np.linalg.svd(
            cls.standardized_steps(
                observations=observations, noise=noise, displacement=displacement
            ),
            full_matrices=False,
        )
        position_steps = left[:, 0] * singular_values[0]
        orientation = np.copysign(1.0, position_steps.sum())
        return cls(
            reference_parent_T_child=observations.parent_T_child[0],
            positions=np.concatenate([[0.0], np.cumsum(orientation * position_steps)]),
            screw_axis_entries=orientation
            * right[0]
            * np.repeat(
                [
                    noise.orientation_standard_deviation,
                    noise.position_standard_deviation,
                ],
                3,
            ),
        )

    @classmethod
    def standardized_steps(
        cls,
        *,
        observations: PoseObservations,
        noise: ObservationNoise,
        displacement: CompiledScrewDisplacement,
    ) -> npt.NDArray[np.float64]:
        """
        :return: For each step between consecutive observations, a move by Δθ along a
            screw axis S, the six numbers Δθ S, divided by the noise as the errors are,
            stacked to shape (n - 1, 6). If one screw axis moves the child, the rows are
            multiples of one another; the rotation vector and the translation of the
            step give Δθ S's rotation, and its translation through the left Jacobian.
        """
        steps = observations.displacements_between_observations
        rotation_vectors = Rotation.from_matrix(steps[:, :3, :3]).as_rotvec()
        translations = np.linalg.solve(
            displacement.left_jacobians(rotation_vectors=rotation_vectors),
            steps[:, :3, 3, np.newaxis],
        )[:, :, 0]
        return np.hstack(
            [
                rotation_vectors / noise.orientation_standard_deviation,
                translations / noise.position_standard_deviation,
            ]
        )

    @property
    def vector(self) -> npt.NDArray[np.float64]:
        """
        The numbers as one vector: the rotation vector and the position of
        :attr:`reference_parent_T_child`, the positions between the first and the last,
        and :attr:`screw_axis_entries`.
        """
        return np.concatenate(
            [
                Rotation.from_matrix(self.reference_parent_T_child[:3, :3]).as_rotvec(),
                self.reference_parent_T_child[:3, 3],
                self.positions[1:-1],
                self.screw_axis_entries,
            ]
        )

    def copy_with_vector(self, *, vector: npt.NDArray[np.float64]) -> Self:
        """
        :param vector: Numbers in the layout of :attr:`vector`.
        :return: A copy with the numbers ``vector`` holds. The last position stays:
            scaling the screw axis and dividing the positions by the same factor would
            not change the motion.
        """
        positions_end = 6 + len(self.positions) - 2
        reference_parent_T_child = np.eye(4)
        reference_parent_T_child[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
        reference_parent_T_child[:3, 3] = vector[3:6]
        return replace(
            self,
            reference_parent_T_child=reference_parent_T_child,
            positions=np.concatenate(
                [[0.0], vector[6:positions_end], self.positions[-1:]]
            ),
            screw_axis_entries=vector[positions_end:],
        )

    def copy_with_normalized_screw_axis(self) -> Self:
        """
        :return: A copy describing the same motion, with the screw axis normalized as
            Lynch and Park define it and the positions scaled to match. A rotation below
            :attr:`negligible_rotation` is set to zero.
        """
        rotation, translation = np.split(self.screw_axis_entries, 2)
        if np.linalg.norm(rotation) * np.ptp(self.positions) < self.negligible_rotation:
            rotation = np.zeros(3)
        scale = np.linalg.norm(rotation if np.any(rotation) else translation)
        return replace(
            self,
            screw_axis_entries=np.concatenate([rotation, translation]) / scale,
            positions=self.positions * scale,
        )

    def predicted_parent_T_child(
        self, *, displacement: CompiledScrewDisplacement
    ) -> npt.NDArray[np.float64]:
        """
        :param displacement: The compiled displacement along the screw axis.
        :return: The pose of the child at each position, stacked to shape (n, 4, 4).
        """
        return (
            displacement.displacements(
                screw_axis_entries=self.screw_axis_entries, positions=self.positions
            )
            @ self.reference_parent_T_child
        )


# %% The fit
@dataclass(kw_only=True)
class ScrewMotionFit:
    """
    The child moves along a screw axis in proportion to a position that differs between
    observations, as a :class:`ScrewAxisConnection` moves it.

    It may turn, slide or both: the fit describes the motion and does not choose a kind
    of connection.
    """

    observations: PoseObservations
    """
    The poses the fit explains.
    """

    noise: ObservationNoise
    """
    The noise the observations are assumed to carry.
    """

    reference_parent_T_child: npt.NDArray[np.float64]
    """
    The pose of the child relative to its parent at position zero.
    """

    screw_axis: ScrewAxis
    """
    How the child moves per unit of position, in the parent frame, normalized as Lynch
    and Park define it.
    """

    positions: npt.NDArray[np.float64]
    """
    The position at each observation, zero at the first and positive at the last.
    """

    displacement: CompiledScrewDisplacement = field(repr=False, compare=False)
    """
    The displacement along the screw axis, compiled, to predict poses with.
    """

    @classmethod
    def fit(cls, *, observations: PoseObservations, noise: ObservationNoise) -> Self:
        """
        :param observations: The observed poses.
        :param noise: The noise the observations carry.
        :return: The motion along a screw axis under which the observations are most
            likely: least squares of the errors divided by the noise, from
            :meth:`ScrewMotionParameters.initial_guess`.
        """
        displacement = CompiledScrewDisplacement.create()
        initial_guess = ScrewMotionParameters.initial_guess(
            observations=observations, noise=noise, displacement=displacement
        )
        result = least_squares(
            fun=lambda vector: noise.standardized_pose_errors(
                observed=observations.parent_T_child,
                predicted=initial_guess.copy_with_vector(
                    vector=vector
                ).predicted_parent_T_child(displacement=displacement),
            ),
            x0=initial_guess.vector,
        )
        parameters = initial_guess.copy_with_vector(
            vector=result.x
        ).copy_with_normalized_screw_axis()
        return cls(
            observations=observations,
            noise=noise,
            reference_parent_T_child=parameters.reference_parent_T_child,
            screw_axis=ScrewAxis(
                rotation_per_unit_position=Vector3(*parameters.screw_axis_entries[:3]),
                translation_per_unit_position=Vector3(
                    *parameters.screw_axis_entries[3:]
                ),
            ),
            positions=parameters.positions,
            displacement=displacement,
        )

    @property
    def predicted_parent_T_child(self) -> npt.NDArray[np.float64]:
        """
        The pose of the child the fit predicts for each observation, stacked to shape
        (n, 4, 4).
        """
        return (
            self.displacement.displacements(
                screw_axis_entries=self.screw_axis.to_np(), positions=self.positions
            )
            @ self.reference_parent_T_child
        )

    @property
    def standardized_errors(self) -> npt.NDArray[np.float64]:
        """
        The errors of :attr:`predicted_parent_T_child`, see
        :meth:`ObservationNoise.standardized_pose_errors`.
        """
        return self.noise.standardized_pose_errors(
            observed=self.observations.parent_T_child,
            predicted=self.predicted_parent_T_child,
        )

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
        """
        Creates a connection that moves ``child`` as the fit does, with its degrees of
        freedom added to ``world``.

        :param world: The world in which to add the degrees of freedom.
        :param parent: The parent of the observed poses.
        :param child: The child of the observed poses.
        :return: The connection, not yet added to ``world``.
        """
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
