"""
How well a screw motion fit recovers the joints of the RBO dataset.

The dataset has no ground truth for its joints' axes. Each joint motion is therefore
compared with a reference made from the fits of the *other* interactions with the same
object, recording date and joint, and with the fit of all its own observations.
"""

from __future__ import annotations

# %% imports
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from experiments.articulation_estimation.dataset import JointKind, JointMotion
from krrood.patterns.classproperty import classproperty
from semantic_digital_twin.articulation_estimation.fits import ScrewMotionFit
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)


# %% axes


@dataclass(kw_only=True)
class JointAxis:
    """
    A joint's normalized screw axis in its parent part's frame, oriented so that it can
    be compared with others of the same joint.
    """

    kind: JointKind
    """
    The kind of joint the dataset gives the motion.
    """

    entries: npt.NDArray[np.float64]
    """
    The rotation and then the translation per unit of position. For a prismatic joint
    the rotation is zero and the translation a unit vector.
    """

    @classmethod
    def from_screw_axis_entries(
        cls,
        *,
        kind: JointKind,
        entries: npt.NDArray[np.float64],
        child_position: npt.NDArray[np.float64],
    ) -> Self:
        """
        :param entries: A screw axis, which may turn a little even for a prismatic joint.
        :param child_position: Where the child part is, in the parent's frame.
        :return: The axis normalized for the joint's kind: a prismatic joint keeps only
            the direction in which the child's origin moves, a revolute joint its unit
            rotation.
        """
        if kind == JointKind.PRISMATIC:
            translation = cls.child_velocity(
                entries=entries, child_position=child_position
            )
            translation = translation / np.linalg.norm(translation)
            return cls(kind=kind, entries=np.concatenate([np.zeros(3), translation]))
        return cls(kind=kind, entries=entries / np.linalg.norm(entries[:3]))

    @classmethod
    def child_velocity(
        cls,
        *,
        entries: npt.NDArray[np.float64],
        child_position: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""
        :param entries: A screw axis :math:`(\omega, v)`.
        :param child_position: A point :math:`p` in the parent's frame.
        :return: How fast the point moves per unit of position, :math:`v + \omega \times
            p`. Unlike :math:`v` alone, the velocity of the parent's origin, it does not
            depend on where that origin lies when the axis turns.
        """
        return entries[3:] + np.cross(entries[:3], child_position)

    @property
    def direction(self) -> npt.NDArray[np.float64]:
        """
        The unit direction the joint turns about or slides along.
        """
        part = (
            self.entries[3:] if self.kind == JointKind.PRISMATIC else self.entries[:3]
        )
        return part / np.linalg.norm(part)

    @property
    def point(self) -> npt.NDArray[np.float64]:
        """
        For a revolute joint, the point of its axis closest to the parent's origin.
        """
        return np.cross(self.entries[:3], self.entries[3:])

    @property
    def pitch(self) -> float:
        """
        For a revolute joint, how far it advances along its axis per radian, in meters.
        """
        return float(self.entries[:3] @ self.entries[3:])

    def aligned_with(self, *, other: JointAxis) -> JointAxis:
        """
        :return: This axis, or the same motion in the opposite direction if that points
            the way ``other`` does.
        """
        if self.direction @ other.direction >= 0:
            return self
        return JointAxis(kind=self.kind, entries=-self.entries)

    @classmethod
    def combine(cls, *, axes: list[JointAxis]) -> Self:
        """
        :param axes: Axes of the same joint, of the same kind.
        :return: Their mean direction, and for revolute joints the median axis point and
            the median pitch, so that a single poor fit does not move the result.
        """
        kind = axes[0].kind
        aligned = [axis.aligned_with(other=axes[0]) for axis in axes]
        direction = np.mean([axis.direction for axis in aligned], axis=0)
        direction /= np.linalg.norm(direction)
        if kind == JointKind.PRISMATIC:
            return cls(kind=kind, entries=np.concatenate([np.zeros(3), direction]))
        point = np.median([axis.point for axis in aligned], axis=0)
        point -= direction * (point @ direction)
        pitch = float(np.median([axis.pitch for axis in aligned]))
        return cls(
            kind=kind,
            entries=np.concatenate(
                [direction, -np.cross(direction, point) + pitch * direction]
            ),
        )


@dataclass(kw_only=True)
class AxisErrors:
    """
    How far an estimated axis lies from a target axis of the same joint.
    """

    direction_degrees: float
    """
    The angle between the two directions.
    """

    axis_distance_millimeters: float | None
    """
    For a revolute joint, the distance between the axes next to the child part.
    """

    pitch_millimeters_per_radian: float | None
    """
    For a revolute joint, the difference in how far the axes advance per radian.
    """

    rotation_degrees_per_meter: float | None
    """
    For a prismatic joint, how much the estimated screw axis turns the child per meter of
    sliding, which a prismatic joint should not.
    """

    @classmethod
    def between(
        cls,
        *,
        estimated_entries: npt.NDArray[np.float64],
        target: JointAxis,
        child_position: npt.NDArray[np.float64],
    ) -> Self:
        """
        :param estimated_entries: The fitted screw axis, unnormalized for the kind.
        :param target: The axis to compare with.
        :param child_position: Where the child part is, in the parent's frame.
        """
        estimated = JointAxis.from_screw_axis_entries(
            kind=target.kind,
            entries=estimated_entries,
            child_position=child_position,
        ).aligned_with(other=target)
        direction_degrees = float(
            np.degrees(
                np.arctan2(
                    np.linalg.norm(np.cross(estimated.direction, target.direction)),
                    estimated.direction @ target.direction,
                )
            )
        )
        if target.kind == JointKind.PRISMATIC:
            rotation = estimated_entries[:3] / np.linalg.norm(
                JointAxis.child_velocity(
                    entries=estimated_entries, child_position=child_position
                )
            )
            return cls(
                direction_degrees=direction_degrees,
                axis_distance_millimeters=None,
                pitch_millimeters_per_radian=None,
                rotation_degrees_per_meter=float(np.degrees(np.linalg.norm(rotation))),
            )
        target_point_near_child = target.point + target.direction * (
            (child_position - target.point) @ target.direction
        )
        offset = target_point_near_child - estimated.point
        offset -= estimated.direction * (offset @ estimated.direction)
        return cls(
            direction_degrees=direction_degrees,
            axis_distance_millimeters=float(1e3 * np.linalg.norm(offset)),
            pitch_millimeters_per_radian=float(
                1e3 * abs(estimated.pitch - target.pitch)
            ),
            rotation_degrees_per_meter=None,
        )


# %% fits of whole interactions


@dataclass(kw_only=True)
class FullFit:
    """
    The fit of all observations of a joint motion, with the noise it estimates from its
    own errors.
    """

    motion: JointMotion
    """
    The joint motion.
    """

    noise: ObservationNoise
    """
    The noise estimated from the errors of the fit.
    """

    screw_axis_entries: npt.NDArray[np.float64]
    """
    The fitted screw axis with that noise, see :attr:`ScrewMotionFit.screw_axis`.
    """

    positions: npt.NDArray[np.float64]
    """
    The fitted position at each observation, see :attr:`ScrewMotionFit.positions`.
    """

    seconds: float
    """
    The time the last fit took.
    """

    @classmethod
    def from_motion(
        cls,
        *,
        motion: JointMotion,
        initial_noise: ObservationNoise,
        noise_iterations: int = 3,
    ) -> Self:
        """
        :param initial_noise: The noise assumed for the first fit.
        :param noise_iterations: How often to fit, each time with the noise estimated
            from the errors of the fit before.
        :return: The last fit. The noise is the median absolute error, scaled to a
            standard deviation for normal errors, so that outliers barely change it.
        """
        noise = initial_noise
        fit, seconds = None, 0.0
        for _ in range(noise_iterations):
            start = time.perf_counter()
            fit = ScrewMotionFit.fit(observations=motion.observations, noise=noise)
            seconds = time.perf_counter() - start
            errors = fit.standardized_errors.reshape(-1, 2, 3)
            noise = ObservationNoise(
                position_standard_deviation=noise.position_standard_deviation
                * cls.robust_standard_deviation(values=errors[:, 0].ravel()),
                orientation_standard_deviation=noise.orientation_standard_deviation
                * cls.robust_standard_deviation(values=errors[:, 1].ravel()),
            )
        return cls(
            motion=motion,
            noise=noise,
            screw_axis_entries=fit.screw_axis.to_np(),
            positions=fit.positions,
            seconds=seconds,
        )

    @classmethod
    def robust_standard_deviation(cls, *, values: npt.NDArray[np.float64]) -> float:
        """
        :return: The median absolute deviation, times 1.4826, the factor that makes it
            the standard deviation of normally distributed values.
        """
        return float(1.4826 * np.median(np.abs(values - np.median(values))))

    @property
    def axis(self) -> JointAxis:
        """
        The fitted axis, normalized for the joint's kind at the child's first observed
        position.
        """
        return JointAxis.from_screw_axis_entries(
            kind=self.motion.joint.kind,
            entries=self.screw_axis_entries,
            child_position=self.motion.observations.parent_T_child_stack[0, :3, 3],
        )

    @classproperty
    def excursion_median_window(cls) -> int:
        """
        The observations over which :attr:`excursion` takes the median: 25, a quarter of
        a second at the dataset's 100 Hz.
        """
        return 25

    @property
    def excursion(self) -> npt.NDArray[np.float64]:
        """
        At each observation, the farthest the joint has moved so far: the fitted angle for
        a revolute joint, the distance the child's origin travelled for a prismatic one,
        each the running median over :attr:`excursion_median_window` observations. The
        median drops outliers, whose positions can lie whole revolutions away, and a
        prismatic fit that turns a little has positions in radians.
        """
        if self.motion.joint.kind == JointKind.REVOLUTE:
            moved = self.positions * np.linalg.norm(self.screw_axis_entries[:3])
        else:
            child_positions = self.motion.observations.parent_T_child_stack[:, :3, 3]
            moved = np.linalg.norm(child_positions - child_positions[0], axis=1)
        half_window = self.excursion_median_window // 2
        medians = np.median(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(moved, half_window, mode="edge"),
                2 * half_window + 1,
            ),
            axis=1,
        )
        return np.maximum.accumulate(np.abs(medians - medians[0]))


# %% results


@dataclass(kw_only=True)
class PartialFitResult:
    """
    The fit of the observations up to a share of a joint motion's excursion.
    """

    excursion_fraction: float
    """
    The share of the motion's full excursion the observations reach.
    """

    observation_count: int
    """
    How many observations the fit saw.
    """

    errors_to_reference: AxisErrors
    """
    The errors against the reference made from the other interactions.
    """

    errors_to_full_fit: AxisErrors
    """
    The errors against the fit of all the motion's observations.
    """

    seconds: float
    """
    The time the fit took.
    """


@dataclass(kw_only=True)
class JointMotionResult:
    """
    The evaluation of one joint motion.
    """

    interaction_name: str
    """
    The name of the recording.
    """

    object_name: str
    """
    The name of the object.
    """

    recording_date: str
    """
    The day of the recording.
    """

    joint_index: int
    """
    The position of the joint in the object model's list of joints.
    """

    kind: JointKind
    """
    The kind of joint the dataset gives it.
    """

    observation_count: int
    """
    How many observations the motion has.
    """

    excursion: float
    """
    The farthest the joint moved from its first position, in radians or meters.
    """

    position_standard_deviation: float
    """
    The noise of the positions, estimated from the full fit, in meters.
    """

    orientation_standard_deviation: float
    """
    The noise of the orientations, estimated from the full fit, in radians.
    """

    full_fit_seconds: float
    """
    The time the full fit took.
    """

    full_fit_errors_to_reference: AxisErrors
    """
    The errors of the full fit against the reference made from the other interactions.
    """

    partial_fits: list[PartialFitResult] = field(default_factory=list)
    """
    The fits of the observations up to shares of the excursion.
    """


@dataclass(kw_only=True)
class JointEvaluation:
    """
    Evaluates one joint motion against the fits of the other interactions with the same
    joint.
    """

    full_fit: FullFit
    """
    The fit of all the motion's observations.
    """

    other_fits: list[FullFit]
    """
    The full fits of the other interactions with the same object, recording date and
    joint.
    """

    excursion_fractions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    """
    The shares of the excursion up to which the observations are fit again.
    """

    def evaluate(self) -> JointMotionResult:
        """
        :return: The errors of the full fit and of the partial fits.
        """
        motion = self.full_fit.motion
        reference = JointAxis.combine(axes=[other.axis for other in self.other_fits])
        full_axis = self.full_fit.axis.aligned_with(other=reference)
        child_position = motion.observations.parent_T_child_stack[0, :3, 3]
        excursion = self.full_fit.excursion
        return JointMotionResult(
            interaction_name=motion.interaction.name,
            object_name=motion.interaction.object_name,
            recording_date=motion.interaction.recording_date,
            joint_index=motion.joint_index,
            kind=motion.joint.kind,
            observation_count=len(motion.times),
            excursion=float(excursion[-1]),
            position_standard_deviation=self.full_fit.noise.position_standard_deviation,
            orientation_standard_deviation=self.full_fit.noise.orientation_standard_deviation,
            full_fit_seconds=self.full_fit.seconds,
            full_fit_errors_to_reference=AxisErrors.between(
                estimated_entries=self.full_fit.screw_axis_entries,
                target=reference,
                child_position=child_position,
            ),
            partial_fits=[
                self.partial_fit_result(
                    fraction=fraction,
                    observation_count=int(
                        np.argmax(excursion >= fraction * excursion[-1]) + 1
                    ),
                    reference=reference,
                    full_axis=full_axis,
                    child_position=child_position,
                )
                for fraction in self.excursion_fractions
            ],
        )

    def partial_fit_result(
        self,
        *,
        fraction: float,
        observation_count: int,
        reference: JointAxis,
        full_axis: JointAxis,
        child_position: npt.NDArray[np.float64],
    ) -> PartialFitResult:
        """
        :return: The errors of the fit of the first ``observation_count`` observations,
            with the noise the full fit estimated.
        """
        observations = PoseObservations(
            parent_T_child_stack=self.full_fit.motion.observations.parent_T_child_stack[
                : max(observation_count, PoseObservations.required_observation_count)
            ]
        )
        start = time.perf_counter()
        fit = ScrewMotionFit.fit(observations=observations, noise=self.full_fit.noise)
        seconds = time.perf_counter() - start
        entries = fit.screw_axis.to_np()
        return PartialFitResult(
            excursion_fraction=fraction,
            observation_count=len(observations.parent_T_child_stack),
            errors_to_reference=AxisErrors.between(
                estimated_entries=entries,
                target=reference,
                child_position=child_position,
            ),
            errors_to_full_fit=AxisErrors.between(
                estimated_entries=entries,
                target=full_axis,
                child_position=child_position,
            ),
            seconds=seconds,
        )
