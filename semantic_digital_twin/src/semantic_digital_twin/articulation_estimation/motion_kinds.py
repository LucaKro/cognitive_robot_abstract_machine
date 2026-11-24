"""
Which kind of motion observed poses show, revolute, helical or sliding, weighed by the
Laplace evidence of a fit of each (`MacKay, Information Theory, Inference, and Learning
Algorithms, chapter 28 <https://www.inference.org.uk/itprnn/book.pdf#page=355>`__) or
given by a :class:`MotionKindClassifier`, and how uncertain each fit is.
"""

from __future__ import annotations

# %% Imports
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from scipy.optimize import least_squares
from scipy.spatial.transform import RigidTransform, Rotation
from typing_extensions import Self

from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import ScrewAxis, Vector3
from semantic_digital_twin.articulation_estimation.exceptions import (
    KindNotFittedError,
)
from semantic_digital_twin.articulation_estimation.fits import ScrewMotionFit
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)


logger = logging.getLogger(__name__)


# %% Kinds and priors
class MotionKind(StrEnum):
    """
    A kind of motion along a screw axis.
    """

    REVOLUTE = "revolute"
    """
    The child turns about the axis and does not advance along it: a door's hinge.
    """

    HELICAL = "helical"
    """
    The child turns about the axis and advances along it: a jar lid's thread.
    """

    SLIDING = "sliding"
    """
    The child advances along the axis and does not turn: a drawer's slide.
    """

    @property
    def turns_child(self) -> bool:
        """
        Whether a motion of this kind turns the child.
        """
        return self != MotionKind.SLIDING


@dataclass(kw_only=True)
class MotionKindPriors:
    """
    The priors of :class:`MotionKindMixture`.

    .. warning:: Placeholders: broad, fixed and the same for every object. They are to come
        from the semantic annotations of the observed object.
    """

    axis_distance_standard_deviation: float = 2.0
    """
    The standard deviation, in meters, of the 2-D Gaussian around the child's origin at
    position zero that the closest point of a turning axis is expected in.
    """

    pitch_standard_deviation: float = 0.01
    """
    The standard deviation, in meters per radian, of the Gaussian around zero that the
    pitch of a helical motion is expected in; 0.01 is about 6 cm per revolution.
    """

    travel_length: float = 0.05
    """
    A length, in meters, added in quadrature to how far one radian of turning moves the
    child's origin, so that a child turning about its own origin still travels; see
    :meth:`KindFit.fit`.
    """

    revolute_probability: float = 0.49
    """
    The probability of a revolute motion before any observation.
    """

    helical_probability: float = 0.02
    """
    The probability of a helical motion before any observation; small, since doors and
    drawers are far more common than screwed parts, so that a turning motion that the
    observations cannot tell apart is taken as revolute.
    """

    sliding_probability: float = 0.49
    """
    The probability of a sliding motion before any observation.
    """

    def probability_of(self, *, kind: MotionKind) -> float:
        """
        :return: The probability of ``kind`` before any observation.
        """
        return {
            MotionKind.REVOLUTE: self.revolute_probability,
            MotionKind.HELICAL: self.helical_probability,
            MotionKind.SLIDING: self.sliding_probability,
        }[kind]

    @property
    def kind_probabilities(self) -> MotionKindProbabilities:
        """
        The probability of each kind before any observation.
        """
        return MotionKindProbabilities(
            revolute=self.revolute_probability,
            helical=self.helical_probability,
            sliding=self.sliding_probability,
        )


# %% The probability of each kind
@dataclass(kw_only=True)
class MotionKindProbabilities:
    """
    The probability of each :class:`MotionKind` given observed poses.
    """

    revolute: float
    """
    The probability of a revolute motion.
    """

    helical: float
    """
    The probability of a helical motion.
    """

    sliding: float
    """
    The probability of a sliding motion.
    """

    @classmethod
    def from_log_weights(cls, *, log_weights: npt.NDArray[np.float64]) -> Self:
        """
        :param log_weights: The logarithm of a number proportional to each kind's
            probability, in the order :class:`MotionKind` lists the kinds.
        :return: The probabilities, normalized to add up to one.
        """
        weights = np.exp(log_weights - log_weights.max())
        revolute, helical, sliding = (weights / weights.sum()).tolist()
        return cls(revolute=revolute, helical=helical, sliding=sliding)

    def probability_of(self, *, kind: MotionKind) -> float:
        """
        :return: The probability of ``kind``.
        """
        return {
            MotionKind.REVOLUTE: self.revolute,
            MotionKind.HELICAL: self.helical,
            MotionKind.SLIDING: self.sliding,
        }[kind]

    def with_prior(self, *, prior: Self, replacing: Self) -> Self:
        """
        :param prior: The probability of each kind before any observation, to take.
        :param replacing: The probability of each kind before any observation that these
            probabilities assumed, to give up.
        :return: The probabilities had ``prior`` been assumed instead of ``replacing``.
        """
        weights = np.array(
            [
                self.probability_of(kind=kind)
                * prior.probability_of(kind=kind)
                / replacing.probability_of(kind=kind)
                for kind in MotionKind
            ]
        )
        revolute, helical, sliding = (weights / weights.sum()).tolist()
        return type(self)(revolute=revolute, helical=helical, sliding=sliding)

    @property
    def most_probable_kind(self) -> MotionKind:
        """
        The kind with the largest probability.
        """
        return max(MotionKind, key=lambda kind: self.probability_of(kind=kind))

    @classproperty
    def kind_variable(cls) -> Symbolic:
        """
        The variable :attr:`distribution` is over, whose values are the kinds.
        """
        return Symbolic("kind", domain=Set.from_iterable(list(MotionKind)))

    @property
    def distribution(self) -> MultinomialDistribution:
        """
        The probabilities as a distribution over :attr:`kind_variable`, whose
        probabilities are exact.
        """
        variable = self.kind_variable
        return MultinomialDistribution(
            (variable,),
            np.array(
                [
                    self.probability_of(kind=MotionKind(str(value)))
                    for value in variable.domain.simple_sets
                ]
            ),
        )


class MotionKindClassifier(ABC):
    """
    Gives the probability of each :class:`MotionKind` from observed poses, without
    fitting the kinds.
    """

    @property
    @abstractmethod
    def training_probabilities(self) -> MotionKindProbabilities:
        """
        The share of each kind among the motions the classifier learned from, which its
        probabilities assume as the probability before any observation.
        """

    @abstractmethod
    def kind_probabilities(
        self, *, observations: PoseObservations, noise: ObservationNoise
    ) -> MotionKindProbabilities:
        """
        :param observations: The observed poses, of a child that moved.
        :param noise: The noise of the observations.
        :return: The probability of each kind given the observations, assuming
            :attr:`training_probabilities` before them.
        """


# %% The chart of one kind
@dataclass(kw_only=True)
class ScrewMotionChart:
    """
    Plain numbers around a motion of one :class:`MotionKind`, all zero at that motion,
    named by :attr:`names`: two tilts of the axis, for a turning kind two offsets of the
    axis, for a helical one the pitch, and a small rotation and translation of the
    child's pose at position zero.
    """

    kind: MotionKind
    """
    The kind of motion the chart describes.
    """

    direction: npt.NDArray[np.float64]
    """
    The unit vector along the axis, in the parent frame.
    """

    axis_point: npt.NDArray[np.float64] | None = None
    """
    The point of a turning axis closest to the child's origin at position zero, in the
    parent frame; None for a sliding motion.
    """

    pose_at_zero: npt.NDArray[np.float64]
    """
    The child's pose at position zero, relative to the parent.
    """

    @property
    def basis(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Two unit vectors perpendicular to :attr:`direction` and to each other, which the
        tilts and the offsets are measured along.
        """
        helper = np.eye(3)[np.argmin(np.abs(self.direction))]
        first = np.cross(self.direction, helper)
        first /= np.linalg.norm(first)
        return first, np.cross(self.direction, first)

    @property
    def axis_names(self) -> list[str]:
        """
        The names of the chart's numbers that describe the axis, in order.
        """
        names = ["tilt_1", "tilt_2"]
        if self.kind.turns_child:
            names += ["axis_offset_1", "axis_offset_2"]
        if self.kind == MotionKind.HELICAL:
            names.append("pitch")
        return names

    @property
    def names(self) -> list[str]:
        """
        The names of all of the chart's numbers, in order.
        """
        return self.axis_names + [
            "child_rotation_x",
            "child_rotation_y",
            "child_rotation_z",
            "child_x",
            "child_y",
            "child_z",
        ]

    def motion(
        self, *, values: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | None,
        float,
    ]:
        """
        :param values: The chart's numbers, in the order of :attr:`names`.
        :return: The screw axis's six entries in the parent frame, the child's pose at
            position zero, the axis point closest to the child's origin there, or None for
            a sliding motion, and the pitch.
        """
        first, second = self.basis
        direction = (
            self.direction + np.tan(values[0]) * first + np.tan(values[1]) * second
        )
        direction /= np.linalg.norm(direction)
        axis_count = len(self.axis_names)
        pose_values = values[axis_count : axis_count + 6]
        pose_at_zero = np.eye(4)
        pose_at_zero[:3, :3] = (
            self.pose_at_zero[:3, :3]
            @ Rotation.from_rotvec(pose_values[:3]).as_matrix()
        )
        pose_at_zero[:3, 3] = self.pose_at_zero[:3, 3] + pose_values[3:]
        if not self.kind.turns_child:
            return np.concatenate([np.zeros(3), direction]), pose_at_zero, None, 0.0
        point = self.axis_point + values[2] * first + values[3] * second
        pitch = values[4] if self.kind == MotionKind.HELICAL else 0.0
        child_origin = pose_at_zero[:3, 3]
        closest = point + direction * ((child_origin - point) @ direction)
        entries = np.concatenate(
            [direction, -np.cross(direction, point) + pitch * direction]
        )
        return entries, pose_at_zero, closest, float(pitch)

    def centred_on(self, *, values: npt.NDArray[np.float64]) -> Self:
        """
        :param values: The chart's numbers of a motion of the same kind.
        :return: A chart of the same kind whose numbers are all zero at that motion.
        """
        entries, pose_at_zero, closest, _ = self.motion(values=values)
        return type(self)(
            kind=self.kind,
            direction=entries[:3] if self.kind.turns_child else entries[3:],
            axis_point=closest,
            pose_at_zero=pose_at_zero,
        )


# %% The fit of one kind
@dataclass(kw_only=True)
class KindFit:
    """
    The fit of one :class:`MotionKind` to observed poses, in its own chart, with its
    uncertainty and its evidence.
    """

    chart: ScrewMotionChart
    """
    The chart, centred on the fitted motion.
    """

    covariance: npt.NDArray[np.float64]
    """
    The covariance of the chart's numbers, from the Laplace approximation of the
    posterior, with the positions marginalized out.
    """

    log_evidence: float
    """
    The logarithm of the Laplace approximation of the evidence for this kind, up to
    terms every kind shares.
    """

    screw_motion: ScrewMotionFit
    """
    The fitted motion as a screw motion fit, which writes it into a connection.
    """

    deviation_in_noise_units: float
    """
    How far the observations stray from the fitted motion, as a multiple of their
    stated noise: the root of the mean squared standardized pose error per number left
    after fitting. About one if the kind explains the observations and the noise is
    stated right.
    """

    @classmethod
    def fit(
        cls,
        *,
        observations: PoseObservations,
        noise: ObservationNoise,
        priors: MotionKindPriors,
        chart: ScrewMotionChart,
        positions: npt.NDArray[np.float64],
    ) -> Self:
        r"""
        Fits the kind of ``chart`` and computes its evidence, leaving out the priors every
        kind shares: on the axis direction, the child's pose at zero and how far the child
        travels, where a turning position of one radian moves it by
        :math:`\sqrt{r^2 + h^2 + \ell^2}`, see :attr:`MotionKindPriors.travel_length`.

        :param observations: The observed poses.
        :param noise: The noise of the observations.
        :param priors: The priors on the axis distance and the pitch.
        :param chart: A chart of the kind to fit, centred near the motion.
        :param positions: A start for the position at each observation, zero at the first.
        :return: The fit.
        """
        stack = observations.parent_T_child_stack
        count = len(stack)
        chart_size = len(chart.names)

        def residuals(
            values: npt.NDArray[np.float64], centred: ScrewMotionChart
        ) -> npt.NDArray[np.float64]:
            """
            :return: The standardized pose errors, then the prior's terms, of the chart's
                numbers and the positions in ``values``.
            """
            entries, pose_at_zero, closest, pitch = centred.motion(
                values=values[:chart_size]
            )
            all_positions = np.concatenate([[0.0], values[chart_size:]])
            predicted = (
                RigidTransform.from_exp_coords(
                    np.outer(all_positions, entries)
                ).as_matrix()
                @ pose_at_zero
            )
            parts = [
                noise.standardized_pose_errors(
                    observed_stack=stack, predicted_stack=predicted
                )
            ]
            if centred.kind.turns_child:
                parts.append(
                    (closest - pose_at_zero[:3, 3])
                    / priors.axis_distance_standard_deviation
                )
            if centred.kind == MotionKind.HELICAL:
                parts.append(np.array([pitch / priors.pitch_standard_deviation]))
            return np.concatenate(parts)

        start = np.concatenate([np.zeros(chart_size), positions[1:]])
        result = least_squares(residuals, start, x_scale="jac", args=(chart,))
        centred = chart.centred_on(values=result.x[:chart_size])
        values = np.concatenate([np.zeros(chart_size), result.x[chart_size:]])
        result = least_squares(residuals, values, x_scale="jac", args=(centred,))
        centred = centred.centred_on(values=result.x[:chart_size])
        precision = result.jac.T @ result.jac
        pose_error_count = 6 * count
        deviation_in_noise_units = np.sqrt(
            np.sum(result.fun[:pose_error_count] ** 2)
            / max(pose_error_count - len(result.x), 1)
        )
        _, log_determinant = np.linalg.slogdet(precision)
        log_evidence = (
            -0.5 * float(np.sum(result.fun**2))
            - 0.5 * log_determinant
            + 0.5 * len(result.x) * np.log(2 * np.pi)
        )
        entries, pose_at_zero, closest, pitch = centred.motion(
            values=np.zeros(chart_size)
        )
        if centred.kind.turns_child:
            distance = np.linalg.norm(closest - pose_at_zero[:3, 3])
            travel_per_radian = np.sqrt(
                distance**2 + pitch**2 + priors.travel_length**2
            )
            log_evidence += -np.log(
                2 * np.pi * priors.axis_distance_standard_deviation**2
            )
            log_evidence += (count - 1) * np.log(travel_per_radian)
        if centred.kind == MotionKind.HELICAL:
            log_evidence += -0.5 * np.log(
                2 * np.pi * priors.pitch_standard_deviation**2
            )
        return cls(
            chart=centred,
            covariance=np.linalg.inv(precision)[:chart_size, :chart_size],
            log_evidence=float(log_evidence),
            screw_motion=ScrewMotionFit(
                observations=observations,
                noise=noise,
                reference_parent_T_child=pose_at_zero,
                screw_axis=ScrewAxis(
                    rotation_per_unit_position=Vector3(*entries[:3]),
                    translation_per_unit_position=Vector3(*entries[3:]),
                ),
                positions=np.concatenate([[0.0], result.x[chart_size:]]),
            ),
            deviation_in_noise_units=float(deviation_in_noise_units),
        )

    @property
    def distribution(self) -> MultivariateGaussianDistribution:
        """
        The Gaussian of the chart's numbers, around zero, the fitted motion.
        """
        return MultivariateGaussianDistribution(
            variables=tuple(Continuous(name) for name in self.chart.names),
            mean=np.zeros(len(self.chart.names)),
            covariance=Covariance.from_matrix(
                (self.covariance + self.covariance.T) / 2
            ),
        )


# %% The mixture
@dataclass(kw_only=True)
class MotionKindMixture:
    """
    The probability of each :class:`MotionKind` given observed poses, and the fits of
    the kinds that are likely enough to be fitted.
    """

    fits: list[KindFit]
    """
    The fit of each fitted kind, in the order :class:`MotionKind` lists them.
    """

    priors: MotionKindPriors
    """
    The priors the fits used.
    """

    kind_probabilities: MotionKindProbabilities
    """
    The probability of each kind given the observations.
    """

    @classmethod
    def fit(
        cls,
        *,
        observations: PoseObservations,
        noise: ObservationNoise,
        priors: MotionKindPriors | None = None,
        classifier: MotionKindClassifier | None = None,
        minimum_probability: float = 0.05,
    ) -> Self:
        """
        :param observations: The observed poses, of a child that moved.
        :param noise: The noise of the observations.
        :param priors: The priors, :class:`MotionKindPriors` if None.
        :param classifier: Gives the probability of each kind, whose
            :attr:`MotionKindClassifier.training_probabilities` are replaced by
            :attr:`MotionKindPriors.kind_probabilities`. If None, every kind is fitted
            and weighed by its evidence and :meth:`MotionKindPriors.probability_of`.
        :param minimum_probability: The probability below which a kind of the
            ``classifier`` is not fitted. The most probable kind is always fitted.
        :return: The mixture.
        """
        priors = priors or MotionKindPriors()
        if classifier is None:
            fits = cls.fit_kinds(
                kinds=list(MotionKind),
                observations=observations,
                noise=noise,
                priors=priors,
            )
            return cls(
                fits=fits,
                priors=priors,
                kind_probabilities=MotionKindProbabilities.from_log_weights(
                    log_weights=np.array(
                        [
                            fit.log_evidence
                            + np.log(priors.probability_of(kind=fit.chart.kind))
                            for fit in fits
                        ]
                    )
                ),
            ).warned_if_unexplained()
        kind_probabilities = classifier.kind_probabilities(
            observations=observations, noise=noise
        ).with_prior(
            prior=priors.kind_probabilities,
            replacing=classifier.training_probabilities,
        )
        kinds = [
            kind
            for kind in MotionKind
            if kind == kind_probabilities.most_probable_kind
            or kind_probabilities.probability_of(kind=kind) >= minimum_probability
        ]
        return cls(
            fits=cls.fit_kinds(
                kinds=kinds, observations=observations, noise=noise, priors=priors
            ),
            priors=priors,
            kind_probabilities=kind_probabilities,
        ).warned_if_unexplained()

    @classproperty
    def largest_explained_deviation(cls) -> float:
        """
        The largest :attr:`KindFit.deviation_in_noise_units` of a fit that explains its
        observations: large enough for noise stated a few times too small, small enough
        for a motion of no kind, which strays by many times the noise.
        """
        return 3.0

    @property
    def explains_observations(self) -> bool:
        """
        Whether the most probable kind's fit explains the observations, so that they
        stray from it by no more than :attr:`largest_explained_deviation`.

        If not, the observed motion is likely of no kind, such as a hinge whose axis
        moves, a motion that changes kind halfway, or a tracking error, or the noise is
        stated far too small.
        """
        return (
            self.most_probable.deviation_in_noise_units
            <= self.largest_explained_deviation
        )

    def warned_if_unexplained(self) -> Self:
        """
        Logs a warning if the mixture does not :attr:`explains_observations`.

        :return: The mixture.
        """
        if not self.explains_observations:
            logger.warning(
                "The observations stray from the most probable kind's fit, %s, by %.1f "
                "times their stated noise, more than %.1f: the motion is likely of no "
                "kind, or the noise is stated too small.",
                self.most_probable.chart.kind.value,
                self.most_probable.deviation_in_noise_units,
                self.largest_explained_deviation,
            )
        return self

    @classmethod
    def fit_kinds(
        cls,
        *,
        kinds: list[MotionKind],
        observations: PoseObservations,
        noise: ObservationNoise,
        priors: MotionKindPriors,
    ) -> list[KindFit]:
        """
        Fits each of ``kinds``, starting the turning kinds from
        :meth:`ScrewMotionFit.fit` and the sliding kind from the line through the first
        and the last observed origin.

        :param kinds: The kinds to fit, in the order :class:`MotionKind` lists them.
        :param observations: The observed poses, of a child that moved.
        :param noise: The noise of the observations.
        :param priors: The priors on the axis distance and the pitch.
        :return: The fit of each of ``kinds``.
        """
        stack = observations.parent_T_child_stack
        fits = []
        if any(kind.turns_child for kind in kinds):
            free = ScrewMotionFit.fit(observations=observations, noise=noise)
            free_entries = np.asarray(free.screw_axis.to_np(), dtype=float).flatten()[
                :6
            ]
            direction = free_entries[:3] / np.linalg.norm(free_entries[:3])
            point = np.cross(free_entries[:3], free_entries[3:]) / (
                free_entries[:3] @ free_entries[:3]
            )
            child_origin = free.reference_parent_T_child[:3, 3]
            closest = point + direction * ((child_origin - point) @ direction)
        for kind in kinds:
            if kind.turns_child:
                chart = ScrewMotionChart(
                    kind=kind,
                    direction=direction,
                    axis_point=closest,
                    pose_at_zero=free.reference_parent_T_child,
                )
                positions = free.positions
            else:
                chord = stack[-1, :3, 3] - stack[0, :3, 3]
                slide_direction = chord / np.linalg.norm(chord)
                chart = ScrewMotionChart(
                    kind=kind, direction=slide_direction, pose_at_zero=stack[0]
                )
                positions = (stack[:, :3, 3] - stack[0, :3, 3]) @ slide_direction
            fits.append(
                KindFit.fit(
                    observations=observations,
                    noise=noise,
                    priors=priors,
                    chart=chart,
                    positions=positions,
                )
            )
        return fits

    def fit_of(self, *, kind: MotionKind) -> KindFit:
        """
        :return: The fit of ``kind``.
        :raises KindNotFittedError: if ``kind`` was not fitted.
        """
        fitted = {fit.chart.kind: fit for fit in self.fits}
        if kind not in fitted:
            raise KindNotFittedError(kind=kind, fitted_kinds=list(fitted))
        return fitted[kind]

    @property
    def most_probable(self) -> KindFit:
        """
        The fit of the most probable kind.
        """
        return self.fit_of(kind=self.kind_probabilities.most_probable_kind)
