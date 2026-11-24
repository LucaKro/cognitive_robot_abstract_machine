"""
The probability of each :class:`MotionKind` from a summary of observed poses, given by a
model trained beforehand on motions whose kind is known, such as a probabilistic
circuit.
"""

from __future__ import annotations

# %% Imports
import json
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, field, fields
from pathlib import Path

import numpy as np
import numpy.typing as npt
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from krrood.adapters.json_serializer import from_json, to_json
from krrood.patterns.classproperty import classproperty
from semantic_digital_twin.articulation_estimation.motion_kinds import (
    MotionKind,
    MotionKindClassifier,
    MotionKindProbabilities,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)


# %% The summary of a motion
@dataclass(kw_only=True)
class MotionSummary:
    """
    Numbers that describe observed poses, most of them relative to the noise, and that
    stay the same when the parent frame moves.

    A significance is the logarithm of one plus a test statistic in units of the noise;
    near zero, the data shows nothing.
    """

    turned_angle_significance: float
    """
    How far the child turned between the first and the last pose, in orientation noise.
    """

    log_turned_angle: float
    """
    The logarithm of the angle the child turned, in radians.
    """

    chord_significance: float
    """
    How far the child's origin moved between the first and the last pose, in position
    noise.
    """

    straight_line_deviation_significance: float
    """
    How far the child's origin strays from the straight line that fits its path best, in
    position noise.
    """

    axial_advance_significance: float
    """
    How significantly the child's origin advances along the turning axis as it turns.
    """

    log_implied_radius: float
    """
    The logarithm of the radius, in meters, of the circle that the chord and the turned
    angle imply.
    """

    log_position_noise: float
    """
    The logarithm of :attr:`ObservationNoise.position_standard_deviation`.
    """

    log_orientation_noise: float
    """
    The logarithm of :attr:`ObservationNoise.orientation_standard_deviation`.
    """

    log_observation_count: float
    """
    The logarithm of the number of observed poses.
    """

    orientation_change_significance: float
    """
    How significantly the orientation changes with the progress along the path, which
    does not assume that the child moves at a constant speed.
    """

    path_bending_significance: float
    """
    How significantly the path bends away from its main direction, against the progress
    along it.
    """

    @classproperty
    def smallest_turned_angle(cls) -> float:
        """
        Radians added to the turned angle before its logarithm, so that no turning stays
        finite.
        """
        return 1e-4

    @classproperty
    def smallest_radius(cls) -> float:
        """
        Meters added to the implied radius before its logarithm, and the smallest value
        the chord's divisor takes, so that a sliding motion's radius stays finite.
        """
        return 1e-3

    @classmethod
    def from_observations(
        cls, *, observations: PoseObservations, noise: ObservationNoise
    ) -> Self:
        """
        :param observations: The observed poses, of a child that moved.
        :param noise: The noise of the observations.
        :return: The summary of the observed motion.
        """
        stack = observations.parent_T_child_stack
        position_noise = noise.position_standard_deviation
        orientation_noise = noise.orientation_standard_deviation
        tiny = np.finfo(float).tiny
        rotations = Rotation.from_matrix(stack[:, :3, :3])
        steps = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
        _, _, rotation_directions = np.linalg.svd(steps, full_matrices=False)
        axis = rotation_directions[0] * np.sign(
            rotation_directions[0] @ steps.sum(axis=0) + tiny
        )
        angles = np.concatenate([[0.0], np.cumsum(steps @ axis)])
        moved = (stack[:, :3, 3] - stack[0, :3, 3]) @ stack[0, :3, :3]
        chord = np.linalg.norm(moved[-1])
        centred = moved - moved.mean(axis=0)
        _, singular_values, path_directions = np.linalg.svd(
            centred, full_matrices=False
        )
        straight_line_deviation = np.sqrt((singular_values[1:] ** 2).sum() / len(stack))
        angle_spread = angles - angles.mean()
        angle_variation = max(angle_spread @ angle_spread, tiny)
        axial_advance = (
            abs((angle_spread @ (moved @ axis)) / angle_variation)
            * np.sqrt(angle_variation)
            / position_noise
        )
        turned = abs(angles[-1])
        radius = chord / max(
            2 * abs(np.sin(min(turned, np.pi) / 2)), cls.smallest_radius
        )

        progress = centred @ path_directions[0]
        progress_spread = progress - progress.mean()
        progress_variation = max(progress_spread @ progress_spread, tiny)
        middle = rotations[len(stack) // 2]
        relative = (middle.inv() * rotations).as_rotvec()
        slopes = (progress_spread @ (relative - relative.mean(axis=0))) / (
            progress_variation
        )
        orientation_change = (
            np.sqrt((slopes @ slopes) * progress_variation) / orientation_noise
        )
        design = np.stack([np.ones_like(progress), progress, progress**2], axis=1)
        design = design / np.maximum(np.abs(design).max(axis=0), tiny)
        inverse = np.linalg.pinv(design.T @ design)
        coefficients = inverse @ design.T @ (centred @ path_directions[1:].T)
        path_bending = (
            np.sqrt(
                max((coefficients[2] @ coefficients[2]) / max(inverse[2, 2], tiny), 0.0)
            )
            / position_noise
        )
        return cls(
            turned_angle_significance=float(np.log1p(turned / orientation_noise)),
            log_turned_angle=float(np.log(turned + cls.smallest_turned_angle)),
            chord_significance=float(np.log1p(chord / position_noise)),
            straight_line_deviation_significance=float(
                np.log1p(straight_line_deviation / position_noise)
            ),
            axial_advance_significance=float(np.log1p(axial_advance)),
            log_implied_radius=float(np.log(radius + cls.smallest_radius)),
            log_position_noise=float(np.log(position_noise)),
            log_orientation_noise=float(np.log(orientation_noise)),
            log_observation_count=float(np.log(len(stack))),
            orientation_change_significance=float(np.log1p(orientation_change)),
            path_bending_significance=float(np.log1p(path_bending)),
        )

    @classproperty
    def field_names(cls) -> list[str]:
        """
        The names of the summary's numbers, in the order of :attr:`values`.
        """
        return [summary_field.name for summary_field in fields(cls)]

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """
        The summary's numbers, in the order of :attr:`field_names`.
        """
        return np.array(astuple(self))


# %% Calibration
@dataclass(kw_only=True)
class ProbabilityCalibration:
    """
    A correction of a classifier's probabilities fitted on motions it was not trained
    on: the log probabilities are divided by a temperature and shifted per kind.
    """

    temperature: float = 1.0
    """
    The number the log probabilities are divided by; above one, the probabilities move
    towards each other.
    """

    revolute_offset: float = 0.0
    """
    The number added to the revolute kind's divided log probability.
    """

    helical_offset: float = 0.0
    """
    The number added to the helical kind's divided log probability; the sliding kind's
    is the reference and stays.
    """

    def calibrated(
        self, *, log_probabilities: npt.NDArray[np.float64]
    ) -> MotionKindProbabilities:
        """
        :param log_probabilities: The classifier's log probabilities, in the order
            :class:`MotionKind` lists the kinds.
        :return: The calibrated probabilities.
        """
        return MotionKindProbabilities.from_log_weights(
            log_weights=log_probabilities / self.temperature
            + np.array([self.revolute_offset, self.helical_offset, 0.0])
        )


# %% Classifiers of the summary
@dataclass(kw_only=True)
class MotionSummaryClassifier(MotionKindClassifier, ABC):
    """
    A classifier trained beforehand on the standardized :class:`MotionSummary` of
    motions whose kind is known, whose probabilities are then calibrated.

    A model of its own gives only :meth:`log_probabilities`; a file written by
    :meth:`to_file` holds which model it is.
    """

    summary_mean: npt.NDArray[np.float64]
    """
    The mean of each summary number over the training motions, subtracted before the
    model sees it.
    """

    summary_standard_deviation: npt.NDArray[np.float64]
    """
    The standard deviation of each summary number over the training motions, which the
    centred number is divided by.
    """

    calibration: ProbabilityCalibration
    """
    The correction applied to the model's probabilities.
    """

    training_kind_share: MotionKindProbabilities = field(
        default_factory=lambda: MotionKindProbabilities(
            revolute=1 / 3, helical=1 / 3, sliding=1 / 3
        )
    )
    """
    The share of each kind among the training motions; equal unless the training said
    otherwise.
    """

    @property
    def training_probabilities(self) -> MotionKindProbabilities:
        return self.training_kind_share

    @classmethod
    def from_file(cls, *, path: Path) -> Self:
        """
        :param path: A JSON file written by :meth:`to_file`.
        :return: The classifier stored in the file, of the model it was written with.
        """
        return from_json(json.loads(path.read_text()))

    def to_file(self, *, path: Path) -> None:
        """
        Writes the classifier to ``path`` as JSON, which :meth:`from_file` reads.
        """
        path.write_text(json.dumps(to_json(self)))

    @abstractmethod
    def log_probabilities(
        self, *, standardized_summary: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :param standardized_summary: The summary's numbers, in the order of
            :attr:`MotionSummary.field_names`, standardized with :attr:`summary_mean`
            and :attr:`summary_standard_deviation`.
        :return: The log probability of each kind, in the order :class:`MotionKind`
            lists the kinds, up to a constant they share.
        """

    def kind_probabilities(
        self, *, observations: PoseObservations, noise: ObservationNoise
    ) -> MotionKindProbabilities:
        summary = MotionSummary.from_observations(
            observations=observations, noise=noise
        )
        log_probabilities = self.log_probabilities(
            standardized_summary=(summary.values - self.summary_mean)
            / self.summary_standard_deviation
        )
        return self.calibration.calibrated(
            log_probabilities=log_probabilities - log_probabilities.max()
        )


@dataclass(kw_only=True)
class MotionKindCircuit(MotionSummaryClassifier):
    """
    A probabilistic circuit over the standardized :class:`MotionSummary` numbers and
    :attr:`MotionKindProbabilities.kind_variable`, whose exact conditional of the kind
    gives the probabilities.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The circuit, with one continuous variable per :attr:`MotionSummary.field_names`, of
    the same name, and the kind variable.
    """

    def log_probabilities(
        self, *, standardized_summary: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        standardized = dict(zip(MotionSummary.field_names, standardized_summary))
        kind_variable = MotionKindProbabilities.kind_variable
        events = np.array(
            [
                [
                    (
                        self.circuit.encoded_point({kind_variable: kind})[kind_variable]
                        if variable.name == kind_variable.name
                        else standardized[variable.name]
                    )
                    for variable in self.circuit.variables
                ]
                for kind in MotionKind
            ],
            dtype=float,
        )
        return np.asarray(self.circuit.root.log_likelihood_of_nodes(events=events))[
            :, 0
        ]
