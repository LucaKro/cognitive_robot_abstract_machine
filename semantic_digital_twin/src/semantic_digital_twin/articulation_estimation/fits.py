"""
A fit of motion along a screw axis to observed poses of a child relative to its parent.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import RigidTransform, Rotation
from typing_extensions import TYPE_CHECKING, Self

from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    ScrewAxis,
    Vector3,
)
from semantic_digital_twin.exceptions import UnnormalizedScrewAxisError
from semantic_digital_twin.articulation_estimation.lie_group_derivatives import (
    ExponentialCoordinateStack,
    RotationVectorStack,
    TransformationStack,
    Vector3Stack,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.articulation_estimation.refinement import (
    LinearizedPoseErrors,
    ScrewMotionRefinement,
    ScrewMotionStep,
    SoftL1Loss,
)
from semantic_digital_twin.world_description.connections import ScrewAxisConnection

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
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
    def outlier_threshold(cls) -> float:
        """
        The error, in standard deviations, beyond which an error counts as an outlier,
        such as a jump of the tracker: 3. The initial guess caps each error there, and
        the fit's loss grows only linearly beyond it.
        """
        return 3.0

    @classproperty
    def sampled_pair_count(cls) -> int:
        """
        The most pairs of observations :meth:`initial_guesses` tries: 30.
        """
        return 30

    @classproperty
    def refined_guess_count(cls) -> int:
        """
        How many of the best :meth:`initial_guesses` the fit refines: 3.

        On recorded motions with many outliers, the best guess alone often led to a
        worse minimum.
        """
        return 3

    @classmethod
    def initial_guesses(
        cls, *, observations: PoseObservations, noise: ObservationNoise, count: int
    ) -> list[Self]:
        """
        :param count: How many guesses to return.
        :return: Of the guesses :meth:`guess_from_pair` makes for pairs of observations,
            the ``count`` whose errors, each capped at :attr:`outlier_threshold`, are
            smallest, the smallest first. It tries every pair, or
            :attr:`sampled_pair_count` of them, drawn with a fixed seed so that the fit
            is reproducible. A pair that includes an outlier gives a poor guess and
            loses, and a pair less than half a revolution apart gives an exact one for
            observations without noise.
        """
        first_indices, second_indices = np.triu_indices(
            len(observations.parent_T_child_stack), k=1
        )
        if len(first_indices) > cls.sampled_pair_count:
            sampled = np.random.default_rng(seed=0).choice(
                len(first_indices), size=cls.sampled_pair_count, replace=False
            )
            first_indices, second_indices = (
                first_indices[sampled],
                second_indices[sampled],
            )
        guesses = [
            cls.guess_from_pair(
                observations=observations,
                noise=noise,
                first_index=first_index,
                second_index=second_index,
            )
            for first_index, second_index in zip(first_indices, second_indices)
        ]
        return sorted(
            guesses,
            key=lambda guess: np.sum(
                np.minimum(
                    noise.standardized_pose_errors(
                        observed_stack=observations.parent_T_child_stack,
                        predicted_stack=guess.predicted_parent_T_child_stack,
                    )
                    ** 2,
                    cls.outlier_threshold**2,
                )
            ),
        )[:count]

    @classmethod
    def guess_from_pair(
        cls,
        *,
        observations: PoseObservations,
        noise: ObservationNoise,
        first_index: int,
        second_index: int,
    ) -> Self:
        r"""
        :param first_index: The index of an observation.
        :param second_index: The index of a later observation.
        :return: The guess that the child moves along the screw axis of the
            displacement between the two observations: its exponential coordinates
            :math:`\Delta\theta\, S`, taken as :math:`S`. The first observation is the
            pose at position zero, and the position at each observation sums the steps
            of :meth:`standardized_steps` projected onto :math:`S`.
        """
        stack = observations.parent_T_child_stack
        screw_axis_entries = RigidTransform.from_matrix(
            stack[second_index] @ np.linalg.inv(stack[first_index])
        ).as_exp_coords()
        standardized_screw_axis = (
            screw_axis_entries / noise.exponential_coordinate_standard_deviations
        )
        position_steps = (
            cls.standardized_steps(observations=observations, noise=noise)
            @ standardized_screw_axis
            / (standardized_screw_axis @ standardized_screw_axis)
        )
        orientation = np.copysign(1.0, position_steps.sum())
        return cls(
            reference_parent_T_child=stack[0],
            positions=np.concatenate([[0.0], np.cumsum(orientation * position_steps)]),
            screw_axis_entries=orientation * screw_axis_entries,
        )

    @classmethod
    def standardized_steps(
        cls, *, observations: PoseObservations, noise: ObservationNoise
    ) -> npt.NDArray[np.float64]:
        r"""
        :return: For each step between consecutive observations, a move by
            :math:`\Delta\theta` along a screw axis :math:`S`, the exponential
            coordinates :math:`\Delta\theta\, S`, divided by the noise as the errors
            are, stacked to shape (n - 1, 6). If one screw axis moves the child, the
            rows are multiples of one another.
        """
        return (
            RigidTransform.from_matrix(
                observations.displacements_between_observations
            ).as_exp_coords()
            / noise.exponential_coordinate_standard_deviations
        )

    def linearized_errors(
        self, *, observations: PoseObservations, noise: ObservationNoise
    ) -> LinearizedPoseErrors:
        r"""
        :param observations: The observed poses.
        :param noise: The noise they carry.
        :return: The standardized errors of the predicted poses and their exact
            derivatives. A twist :math:`\delta` that moves a predicted pose in the parent
            frame changes its position errors by :math:`t \times \delta_\omega -
            \delta_v` and its orientation errors by :math:`-J^{-1}(e) R^T \delta_\omega`,
            with :math:`t, R` the predicted position and orientation and :math:`e` the
            orientation error. A position moves the pose by :math:`S`, the screw axis by
            the left Jacobian of :math:`\theta S` times :math:`\theta`.
        """
        exponential_coordinates = np.outer(self.positions, self.screw_axis_entries)
        displacements = RigidTransform.from_exp_coords(
            exponential_coordinates
        ).as_matrix()
        predicted_stack = displacements @ self.reference_parent_T_child
        errors = noise.standardized_pose_errors(
            observed_stack=observations.parent_T_child_stack,
            predicted_stack=predicted_stack,
        ).reshape(-1, 6)
        twist_jacobians = np.zeros((len(self.positions), 6, 6))
        twist_jacobians[:, :3, :3] = Vector3Stack(
            entries=predicted_stack[:, :3, 3]
        ).skew_symmetric_matrices
        twist_jacobians[:, :3, 3:] = -np.eye(3)
        twist_jacobians[:, 3:, :3] = -RotationVectorStack(
            entries=errors[:, 3:] * noise.orientation_standard_deviation
        ).inverse_left_jacobians @ np.swapaxes(predicted_stack[:, :3, :3], 1, 2)
        twist_jacobians /= np.repeat(
            [noise.position_standard_deviation, noise.orientation_standard_deviation], 3
        )[:, None]
        return LinearizedPoseErrors(
            errors=errors,
            reference_pose_jacobians=twist_jacobians
            @ TransformationStack(matrices=displacements).adjoints,
            position_jacobians=twist_jacobians @ self.screw_axis_entries,
            screw_axis_jacobians=twist_jacobians
            @ (
                ExponentialCoordinateStack(
                    entries=exponential_coordinates
                ).left_jacobians
                * self.positions[:, None, None]
            ),
        )

    def copy_with_step(self, *, step: ScrewMotionStep) -> Self:
        """
        :param step: The change.
        :return: A copy changed by it, the pose at position zero moved by its twist in
            the parent frame.
        """
        return replace(
            self,
            reference_parent_T_child=RigidTransform.from_exp_coords(
                step.reference_pose_twist
            ).as_matrix()
            @ self.reference_parent_T_child,
            positions=self.positions + step.position_changes,
            screw_axis_entries=self.screw_axis_entries + step.screw_axis_change,
        )

    def copy_with_fixed_gauge(self, *, noise: ObservationNoise) -> Self:
        """
        :param noise: The noise, whose standard deviations give the screw axis entries
            a common unit.
        :return: A copy describing the same motion with the first position zero, the
            last not negative and the screw axis entries, divided by
            :attr:`ObservationNoise.exponential_coordinate_standard_deviations`, of
            length one.
        """
        positions = self.positions - self.positions[0]
        scale = np.copysign(
            np.linalg.norm(
                self.screw_axis_entries
                / noise.exponential_coordinate_standard_deviations
            ),
            positions[-1],
        )
        return replace(
            self,
            reference_parent_T_child=RigidTransform.from_exp_coords(
                self.positions[0] * self.screw_axis_entries
            ).as_matrix()
            @ self.reference_parent_T_child,
            positions=positions * scale,
            screw_axis_entries=self.screw_axis_entries / scale,
        )

    def copy_with_normalized_screw_axis(self) -> Self:
        """
        :return: A copy describing the same motion, with the screw axis normalized as
            Lynch and Park define it and the positions scaled to match. A rotation of at
            most :attr:`ScrewAxis.normalization_tolerance` radians per meter of
            translation is set to zero: a normalized screw axis with that little
            rotation counts as not turning the child.
        """
        rotation, translation = np.split(self.screw_axis_entries, 2)
        if np.linalg.norm(
            rotation
        ) <= ScrewAxis.normalization_tolerance * np.linalg.norm(translation):
            rotation = np.zeros(3)
        scale = np.linalg.norm(rotation if np.any(rotation) else translation)
        return replace(
            self,
            screw_axis_entries=np.concatenate([rotation, translation]) / scale,
            positions=self.positions * scale,
        )

    @property
    def predicted_parent_T_child_stack(self) -> npt.NDArray[np.float64]:
        """
        The pose of the child at each position, stacked to shape (n, 4, 4).
        """
        return (
            RigidTransform.from_exp_coords(
                np.outer(self.positions, self.screw_axis_entries)
            ).as_matrix()
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

    def __post_init__(self) -> None:
        """
        :raises UnnormalizedScrewAxisError: if :attr:`screw_axis` is not normalized.
        """
        if not self.screw_axis.is_normalized:
            raise UnnormalizedScrewAxisError(screw_axis=self.screw_axis)

    @classmethod
    def fit(cls, *, observations: PoseObservations, noise: ObservationNoise) -> Self:
        """
        :param observations: The observed poses.
        :param noise: The noise the observations carry.
        :return: The motion along a screw axis that explains the observations best:
            least squares of the errors divided by the noise, refined from the
            :attr:`ScrewMotionParameters.refined_guess_count` best
            :meth:`ScrewMotionParameters.initial_guesses`. The loss grows only linearly
            beyond :attr:`ScrewMotionParameters.outlier_threshold`, so that a few
            outliers, such as jumps of the tracker, do not pull the fit away.
        """
        parameters = (
            ScrewMotionRefinement(
                observations=observations,
                noise=noise,
                loss=SoftL1Loss(scale=ScrewMotionParameters.outlier_threshold),
            )
            .refine_lowest(
                starts=ScrewMotionParameters.initial_guesses(
                    observations=observations,
                    noise=noise,
                    count=ScrewMotionParameters.refined_guess_count,
                )
            )
            .copy_with_normalized_screw_axis()
        )
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
        )

    @property
    def predicted_parent_T_child_stack(self) -> npt.NDArray[np.float64]:
        """
        The pose of the child the fit predicts for each observation, stacked to shape
        (n, 4, 4).
        """
        return ScrewMotionParameters(
            reference_parent_T_child=self.reference_parent_T_child,
            positions=self.positions,
            screw_axis_entries=self.screw_axis.to_np(),
        ).predicted_parent_T_child_stack

    @property
    def standardized_errors(self) -> npt.NDArray[np.float64]:
        """
        The errors of :attr:`predicted_parent_T_child_stack`, see
        :meth:`ObservationNoise.standardized_pose_errors`.
        """
        return self.noise.standardized_pose_errors(
            observed_stack=self.observations.parent_T_child_stack,
            predicted_stack=self.predicted_parent_T_child_stack,
        )

    @property
    def parent_T_connection(self) -> HomogeneousTransformationMatrix:
        """
        The connection frame of :meth:`create_connection`: at the point of the screw
        axis closest to the parent's origin, along the parent's axes.

        A screw axis connection needs its axis to pass through the connection frame's
        origin.
        """
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=self.screw_axis.axis_point_closest_to_origin
        )

    @property
    def connection_screw_axis(self) -> ScrewAxis:
        r"""
        :attr:`screw_axis` in :attr:`parent_T_connection`'s frame: the same rotation
        :math:`\omega`, and the translation :math:`v + \omega \times q` of the point of
        the child at the frame's origin :math:`q`, which only advances along the axis.
        """
        rotation = self.screw_axis.rotation_per_unit_position
        return ScrewAxis(
            rotation_per_unit_position=rotation,
            translation_per_unit_position=self.screw_axis.translation_per_unit_position
            + rotation.cross(self.screw_axis.axis_point_closest_to_origin.vector3),
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
