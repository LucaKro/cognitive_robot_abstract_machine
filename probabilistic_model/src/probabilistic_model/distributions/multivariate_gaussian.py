from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from random_events.interval import Interval, SimpleInterval, reals, singleton
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import Continuous, Variable
from scipy.stats import multivariate_normal
from typing_extensions import Any, Dict, Iterable, List, Mapping, Optional, Self, Tuple

from probabilistic_model.exceptions import (
    IntractableError,
    MeanAndCovarianceDisagreeError,
    UndefinedOperationError,
)
from probabilistic_model.probabilistic_model import (
    CenterType,
    MomentType,
    OrderType,
    ProbabilisticModel,
)
from probabilistic_model.quantities import Quantities, QuantityPair

# %% the estimate and the uncertainty a Gaussian is written in


@dataclass
class Mean:
    """
    What each of a distribution's quantities is expected to be.

    The estimates are kept against the quantities they belong to rather than as an array
    a reader has to index by row. :attr:`as_array` is where that becomes an array, which
    is the only place the arithmetic needs one.
    """

    quantities: Quantities
    """
    The quantities being estimated.
    """

    estimates: Dict[Continuous, float]
    """
    What each of them is expected to be, one entry per quantity.
    """

    @classmethod
    def of(cls, quantities: Quantities, estimates: Mapping[Continuous, float]) -> Self:
        """
        :param quantities: The quantities being estimated.
        :param estimates: What each of them is estimated at; one left out is zero.
        :return: Those estimates, against those quantities.
        :raises VariableNotInQuantitiesError: If an estimate names a quantity that is not
            one of them.
        """
        for variable in estimates:
            quantities.index_of(variable)
        return cls(
            quantities=quantities,
            estimates={
                variable: float(estimates.get(variable, 0.0)) for variable in quantities
            },
        )

    @classmethod
    def from_array(
        cls, quantities: Quantities, values: npt.NDArray[np.float64]
    ) -> Self:
        """
        Read an estimate back off the array the arithmetic produced.

        :param quantities: The quantities the array is laid out by.
        :param values: One number per quantity, in that layout.
        :return: Those estimates, against those quantities.
        """
        return cls(
            quantities=quantities,
            estimates={
                variable: float(values[position])
                for position, variable in enumerate(quantities)
            },
        )

    @property
    def as_array(self) -> npt.NDArray[np.float64]:
        """
        :return: The estimates as one number per quantity, in the layout order, for the
            arithmetic that needs an array.
        """
        return self.quantities.vector(self.estimates)

    def estimate_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: What it is estimated at.
        :raises VariableNotInQuantitiesError: If it is not one of these quantities.
        """
        self.quantities.index_of(variable)
        return self.estimates[variable]


@dataclass
class Covariance:
    """
    How uncertain a distribution's estimates are, and how far their errors move
    together.

    Like :class:`Mean`, the numbers are kept against the pairs of quantities they
    describe rather than as a matrix indexed by row and column. Both directions of a
    pair are kept separately, so an uncertainty that drifts out of symmetry stays
    visible rather than being quietly evened out on the way in.
    """

    quantities: Quantities
    """
    The quantities being estimated.
    """

    uncertainty: Dict[QuantityPair, float]
    """
    How uncertain each ordered pair of them is, one entry per pair, with a quantity
    paired with itself being its own variance.
    """

    @classmethod
    def of(
        cls, quantities: Quantities, uncertainty: Mapping[QuantityPair, float]
    ) -> Self:
        """
        :param quantities: The quantities being estimated.
        :param uncertainty: How uncertain each pair of them is; one left out is zero. Two
            quantities vary together by one number, so a pair given once fills its mirror
            as well.
        :return: That uncertainty, against those quantities.
        :raises VariableNotInQuantitiesError: If an entry names a quantity that is not
            one of them.
        """
        return cls.from_array(quantities, quantities.symmetric_matrix(uncertainty))

    @classmethod
    def from_array(
        cls, quantities: Quantities, values: npt.NDArray[np.float64]
    ) -> Self:
        """
        Read an uncertainty back off the matrix the arithmetic produced.

        Each direction of a pair is read on its own rather than averaged, so this
        records what the arithmetic actually produced.

        :param quantities: The quantities the matrix is laid out by.
        :param values: One number per ordered pair, in that layout.
        :return: That uncertainty, against those quantities.
        """
        return cls(
            quantities=quantities,
            uncertainty={
                (row, column): float(values[first, second])
                for first, row in enumerate(quantities)
                for second, column in enumerate(quantities)
            },
        )

    @property
    def as_array(self) -> npt.NDArray[np.float64]:
        """
        :return: The uncertainty as one number per ordered pair, in the layout order, for
            the arithmetic that needs a matrix.
        """
        return self.quantities.matrix(self.uncertainty)

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: How uncertain its estimate is on its own.
        :raises VariableNotInQuantitiesError: If it is not one of these quantities.
        """
        return self.between(variable, variable)

    def between(self, first: Continuous, second: Continuous) -> float:
        """
        :param first: One of the quantities.
        :param second: The other one.
        :return: How far their errors move together.
        :raises VariableNotInQuantitiesError: If either is not one of these quantities.
        """
        self.quantities.index_of(first)
        self.quantities.index_of(second)
        return self.uncertainty[first, second]


# %% what a sensor reported


@dataclass
class Reading:
    """
    One number a sensor reported, and what it says about a distribution's quantities.
    """

    value: float
    """
    The number the sensor reported.
    """

    contributions: Mapping[Continuous, float]
    """
    How much each quantity adds to that number if the estimate is exactly right, so a
    sensor reading one quantity itself contributes one of it and nothing else.
    """

    variance: float
    """
    How far this sensor's readings scatter around the truth, which is what decides how
    far the reading is allowed to move the estimate.
    """

    @classmethod
    def of_one_variable(
        cls, variable: Continuous, value: float, variance: float
    ) -> Self:
        """
        Build the reading of a sensor that reports one quantity itself.

        :param variable: The quantity that was read.
        :param value: What the sensor reported for it.
        :param variance: How far that sensor's readings scatter.
        :return: The reading.
        """
        return cls(value=value, contributions={variable: 1.0}, variance=variance)


# %% a Gaussian over several quantities at once


@dataclass
class MultivariateGaussianDistribution(ProbabilisticModel):
    """
    A Gaussian over one or more continuous quantities that may co-vary.

    Everything is named by quantity rather than by row, so a caller never counts
    positions. :meth:`conditional` fixes some of the quantities at a value and answers
    with the Gaussian over the rest; :meth:`conditional_on_readings` expresses a
    measurement through it.

    ..note:: Arrays whose rows are this distribution's quantities — those
        :meth:`log_likelihood` and :meth:`sample` take and return — are laid out in
        :attr:`variables` order, which is the order the quantities were named in and is
        not sorted.
    """

    mean: Mean
    """
    What each quantity is expected to be.
    """

    covariance: Covariance
    """
    How uncertain those expectations are.
    """

    def __post_init__(self):
        """
        :raises MeanAndCovarianceDisagreeError: If the estimate and the uncertainty are
            laid out by different quantities, in which case neither says anything about
            the other.
        """
        if self.mean.quantities != self.covariance.quantities:
            raise MeanAndCovarianceDisagreeError(
                mean_quantities=list(self.mean.quantities),
                covariance_quantities=list(self.covariance.quantities),
            )

    # %% what it is about

    @property
    def quantities(self) -> Quantities:
        """
        :return: The quantities this distribution is over, in their layout order.
        """
        return self.mean.quantities

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self.quantities.variables

    @property
    def support(self) -> Event:
        return SimpleEvent.from_data(
            {variable: reals() for variable in self.variables}
        ).as_composite_set()

    # %% building one

    @classmethod
    def of(
        cls,
        quantities: Quantities,
        estimates: Mapping[Continuous, float],
        uncertainty: Mapping[QuantityPair, float],
    ) -> Self:
        """
        Build a Gaussian over several quantities at once.

        :param quantities: The quantities being estimated.
        :param estimates: What each of them is expected to be; one left out is zero.
        :param uncertainty: How uncertain those expectations are: a quantity paired with
            itself is its own variance, and two different quantities are how far their
            errors move together.
        :return: The Gaussian over them.
        :raises VariableNotInQuantitiesError: If either names a quantity that is not one
            of them.
        """
        return cls(
            mean=Mean.of(quantities, estimates),
            covariance=Covariance.of(quantities, uncertainty),
        )

    @classmethod
    def of_one_variable(
        cls, variable: Continuous, mean: float, variance: float
    ) -> Self:
        """
        Build a Gaussian over a single quantity.

        :param variable: The quantity being estimated.
        :param mean: What it is expected to be.
        :param variance: How uncertain that expectation is.
        :return: The Gaussian over it.
        """
        return cls.of(
            quantities=Quantities.of(variable),
            estimates={variable: mean},
            uncertainty={(variable, variable): variance},
        )

    def mean_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: What it is expected to be.
        :raises VariableNotInQuantitiesError: If this distribution is not about it.
        """
        return self.mean.estimate_of(variable)

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: How uncertain its expectation is.
        :raises VariableNotInQuantitiesError: If this distribution is not about it.
        """
        return self.covariance.variance_of(variable)

    # %% density and probability

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        return np.atleast_1d(
            multivariate_normal.logpdf(
                events, mean=self.mean.as_array, cov=self.covariance.as_array
            )
        )

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        """
        The probability of an axis-aligned box has no closed form once the quantities
        co-vary, so it is integrated numerically. A quantity confined to several
        stretches makes several boxes, and their probabilities add.

        :param event: The box, or boxes, to measure.
        :return: How probable it is.
        """
        stretches_per_quantity = [
            self._simple_intervals_of(event, variable) for variable in self.variables
        ]
        return float(
            sum(
                self._probability_of_box(box)
                for box in itertools.product(*stretches_per_quantity)
            )
        )

    def _simple_intervals_of(
        self, event: SimpleEvent, variable: Variable
    ) -> Tuple[SimpleInterval, ...]:
        """
        :param event: The box the quantity is confined by.
        :param variable: The quantity to read the confinement of.
        :return: The unbroken stretches it is confined to.
        """
        interval: Interval = event[variable]
        return tuple(interval.simple_sets)

    def _probability_of_box(self, box: Tuple[SimpleInterval, ...]) -> float:
        """
        :param box: One unbroken stretch per quantity, in layout order.
        :return: How probable it is that every quantity falls in its own stretch.
        """
        lower = np.array([stretch.lower for stretch in box])
        upper = np.array([stretch.upper for stretch in box])
        probability = multivariate_normal.cdf(
            upper,
            mean=self.mean.as_array,
            cov=self.covariance.as_array,
            lower_limit=lower,
        )
        return max(float(probability), 0.0)

    def log_mode(self) -> Tuple[Event, float]:
        """
        A Gaussian is most likely exactly where it is expected to be.

        :return: The expectation, and the log-density there.
        """
        mode = SimpleEvent.from_data(
            {
                variable: singleton(self.mean.estimate_of(variable))
                for variable in self.variables
            }
        ).as_composite_set()
        return mode, float(self.log_likelihood(self.mean.as_array.reshape(1, -1))[0])

    # %% fixing quantities at a value

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[Self], float]:
        """
        Fix some of the quantities at the values given and answer with the Gaussian over
        the rest, which stays Gaussian.

        :param point: What each fixed quantity is known to be.
        :return: The Gaussian over the quantities left free, and the log-density of the
            values given.
        :raises VariableNotInQuantitiesError: If a fixed quantity is not one of this
            distribution's.
        :raises UndefinedOperationError: If every quantity is fixed, since there is no
            distribution over nothing.
        """
        fixed_rows = [self.quantities.index_of(variable) for variable in point]
        free_rows = [
            row for row in range(len(self.quantities)) if row not in set(fixed_rows)
        ]
        if not free_rows:
            raise UndefinedOperationError(self)

        fixed_at = np.array([float(point[variable]) for variable in point])
        free_given_fixed = self._gaussian_of(free_rows, fixed_rows, fixed_at)
        log_density_of_fixed = self._gaussian_of(
            fixed_rows, [], fixed_at
        ).log_likelihood(fixed_at.reshape(1, -1))[0]
        return free_given_fixed, float(log_density_of_fixed)

    def _gaussian_of(
        self,
        rows: List[int],
        given_rows: List[int],
        given_values: npt.NDArray[np.float64],
    ) -> Self:
        """
        :param rows: The rows the answer is over, in layout order.
        :param given_rows: The rows held at a value, in the order ``given_values`` uses.
        :param given_values: What those rows are held at.
        :return: The Gaussian over ``rows``, narrowed by whatever ``given_rows`` says.
        """
        quantities = Quantities.of(*[self.quantities.variables[row] for row in rows])
        mean = self.mean.as_array[rows]
        covariance = self.covariance.as_array[np.ix_(rows, rows)]

        if given_rows:
            cross = self.covariance.as_array[np.ix_(rows, given_rows)]
            among_given = self.covariance.as_array[np.ix_(given_rows, given_rows)]
            explained = cross @ np.linalg.inv(among_given)
            mean = mean + explained @ (given_values - self.mean.as_array[given_rows])
            covariance = covariance - explained @ cross.T

        return type(self)(
            mean=Mean.from_array(quantities, mean),
            covariance=Covariance.from_array(quantities, self._symmetrized(covariance)),
        )

    @staticmethod
    def _symmetrized(
        covariance: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        An uncertainty is symmetric by definition, so any difference between a matrix
        and its transpose here is rounding. Averaging the two removes it, and one that
        drifts out of symmetry at a control cycle's rate shows up much later as an
        unexplained uncertainty.

        :param covariance: The uncertainty as the arithmetic left it.
        :return: The same uncertainty, exactly symmetric.
        """
        return (covariance + covariance.T) / 2

    def conditional_on_readings(self, readings: List[Reading]) -> Self:
        """
        Correct the expectation with what sensors reported.

        The readings and the expectation are weighed against each other by how uncertain
        each is, so a sensor that is unsure of itself barely moves the estimate. This is
        :meth:`conditional` applied to the joint distribution over these quantities and
        what the sensors report, which is what a measurement update is.

        :param readings: What the sensors reported, and what each number says about
            these quantities. Reporting nothing answers with this distribution
            unchanged.
        :return: The corrected Gaussian over the same quantities.
        :raises VariableNotInQuantitiesError: If a reading names a quantity this
            distribution is not about.
        """
        if not readings:
            return self

        model = np.array(
            [self.quantities.vector(reading.contributions) for reading in readings]
        )
        scatter = np.diag([reading.variance for reading in readings])
        reported = np.array([reading.value for reading in readings])

        joint = self._joint_with_readings(model, scatter)
        estimated_rows = list(range(len(self.quantities)))
        reported_rows = list(range(len(self.quantities), len(joint.quantities)))
        return joint._gaussian_of(estimated_rows, reported_rows, reported)

    def _joint_with_readings(
        self,
        model: npt.NDArray[np.float64],
        scatter: npt.NDArray[np.float64],
    ) -> Self:
        """
        :param model: How much each quantity contributes to each reported number.
        :param scatter: How far each sensor's readings scatter.
        :return: The Gaussian over these quantities followed by what the sensors report.
        """
        joint_quantities = Quantities.of(
            *self.quantities.variables,
            *self._variables_for_readings(len(model)),
        )
        covariance = self.covariance.as_array
        joint_covariance = np.block(
            [
                [covariance, covariance @ model.T],
                [model @ covariance, model @ covariance @ model.T + scatter],
            ]
        )
        return type(self)(
            mean=Mean.from_array(
                joint_quantities,
                np.concatenate([self.mean.as_array, model @ self.mean.as_array]),
            ),
            covariance=Covariance.from_array(
                joint_quantities,
                self._symmetrized(joint_covariance),
            ),
        )

    def _variables_for_readings(self, amount: int) -> List[Continuous]:
        """
        A reading is a quantity of the joint distribution only while the measurement is
        being applied, so it is named here rather than carried in :class:`Reading`.

        :param amount: How many readings need a name.
        :return: That many names, none of which this distribution already uses.
        """
        taken = {variable.name for variable in self.variables}
        names = []
        for position in range(amount):
            name = f"reading {position}"
            while name in taken:
                name = f"{name} "
            taken.add(name)
            names.append(Continuous(name))
        return names

    # %% confining it to an event

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[ProbabilisticModel], float]:
        """
        Confine this Gaussian to an event.

        A Gaussian confined to anything other than the whole space is no longer a
        Gaussian, so this answers with a
        :class:`TruncatedMultivariateGaussianDistribution` rather than with one of its
        own kind.

        :param event: The event to confine it to.
        :param singleton_allowed: Ignored, since a Gaussian gives every single point
            probability zero.
        :return: The confined distribution and the log-probability of the event, or
            nothing at all if the event cannot happen.
        """
        event.fill_missing_variables(set(self.variables))
        probability = self.probability(event)
        if probability == 0.0:
            return None, -np.inf
        return (
            TruncatedMultivariateGaussianDistribution(untruncated=self, event=event),
            math.log(probability),
        )

    # %% reading fewer quantities

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        :param variables: The quantities to keep. They are kept in this distribution's
            own layout order, whatever order they are asked for in.
        :return: The Gaussian over only those quantities.
        :raises VariableNotInQuantitiesError: If one of them is not one of this
            distribution's.
        """
        kept = set(variables)
        rows = [
            self.quantities.index_of(variable)
            for variable in self.variables
            if variable in kept
        ]
        return self._gaussian_of(rows, [], np.array([]))

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        """
        Every moment asked for here is of one quantity on its own, so each is answered
        by that quantity's own marginal.

        :param order: The order of the moment of each quantity to answer for.
        :param center: What to take each of those moments about.
        :return: The moment of each quantity asked for.
        """
        from probabilistic_model.distributions.gaussian import GaussianDistribution

        moments = VariableMap()
        for variable in order:
            marginal = GaussianDistribution(
                variable=variable,
                location=self.mean_of(variable),
                scale=math.sqrt(self.variance_of(variable)),
            )
            moments[variable] = marginal.moment(
                VariableMap({variable: order[variable]}),
                VariableMap({variable: center[variable]}),
            )[variable]
        return moments

    # %% moving and stretching

    def apply_translation(self, translation: Dict[Variable, float]):
        """
        Move the expectation, leaving the uncertainty as it is.

        :param translation: How far to move each quantity; one left out does not move.
        :raises VariableNotInQuantitiesError: If it names a quantity this distribution
            is not about.
        """
        self.mean = Mean.from_array(
            self.quantities, self.mean.as_array + self.quantities.vector(translation)
        )

    def apply_scaling(self, scaling: Dict[Variable, float]):
        """
        Stretch the quantities, which stretches the expectation once and the uncertainty
        once per quantity it relates.

        :param scaling: What to multiply each quantity by; one left out keeps its size.
        :raises VariableNotInQuantitiesError: If it names a quantity this distribution
            is not about.
        """
        factors = np.ones(len(self.quantities))
        for variable, factor in scaling.items():
            factors[self.quantities.index_of(variable)] = factor
        self.mean = Mean.from_array(self.quantities, self.mean.as_array * factors)
        self.covariance = Covariance.from_array(
            self.quantities, self.covariance.as_array * np.outer(factors, factors)
        )

    def apply_linear_map(self, mapping: Mapping[QuantityPair, float]):
        """
        Make each quantity the weighted sum of the quantities it is mapped from, which
        is what carrying an estimate forward through a linear change does.

        :param mapping: How much each quantity on the right of a pair carries into the
            quantity on its left. :attr:`Quantities.unchanged` leaves every quantity as
            it is.
        :raises VariableNotInQuantitiesError: If it names a quantity this distribution
            is not about.
        """
        matrix = self.quantities.matrix(mapping)
        self.mean = Mean.from_array(self.quantities, matrix @ self.mean.as_array)
        self.covariance = Covariance.from_array(
            self.quantities,
            self._symmetrized(matrix @ self.covariance.as_array @ matrix.T),
        )

    def apply_added_uncertainty(self, uncertainty: Mapping[QuantityPair, float]):
        """
        Make the quantities less certain, which is what keeps a distribution nobody is
        measuring from staying confident forever.

        :param uncertainty: How much uncertainty to add to each pair of quantities; one
            left out gains none.
        :raises VariableNotInQuantitiesError: If it names a quantity this distribution
            is not about.
        """
        self.covariance = Covariance.from_array(
            self.quantities,
            self.covariance.as_array + self.quantities.symmetric_matrix(uncertainty),
        )

    # %% sampling

    def sample(self, amount: int) -> npt.NDArray:
        return multivariate_normal.rvs(
            mean=self.mean.as_array, cov=self.covariance.as_array, size=amount
        ).reshape(amount, len(self.quantities))

    def __copy__(self) -> Self:
        return type(self)(
            mean=Mean.from_array(self.quantities, self.mean.as_array.copy()),
            covariance=Covariance.from_array(
                self.quantities, self.covariance.as_array.copy()
            ),
        )


# %% a Gaussian that has been confined to an event


@dataclass
class TruncatedMultivariateGaussianDistribution(ProbabilisticModel):
    """
    A Gaussian confined to an event, which is what is left of one once part of the space
    is ruled out.

    It is not itself a Gaussian: cutting a correlated Gaussian along an axis leaves a
    shape no Gaussian has. What it keeps is the shape of the original inside the event,
    scaled up so that the event is certain.
    """

    untruncated: MultivariateGaussianDistribution
    """
    The Gaussian before anything was ruled out.
    """

    event: Event
    """
    Everything still considered possible.
    """

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self.untruncated.variables

    @property
    def support(self) -> Event:
        return self.event

    @property
    def probability_of_the_event(self) -> float:
        """
        :return: How probable the event was before it was the only thing left, which is
            what every density here is scaled up by.
        """
        return self.untruncated.probability(self.event)

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        inside = np.array([self.event.contains(event) for event in events])
        return np.where(
            inside,
            self.untruncated.log_likelihood(events)
            - math.log(self.probability_of_the_event),
            -np.inf,
        )

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        surviving = event.as_composite_set() & self.event
        return self.untruncated.probability(surviving) / self.probability_of_the_event

    def log_mode(self) -> Tuple[Event, float]:
        """
        The most likely point is the untruncated expectation whenever the event still
        contains it, since the density falls away from there in every direction.

        :return: That point and its log-density.
        :raises IntractableError: If the expectation was ruled out, which leaves the
            most likely point somewhere on the event's boundary and no closed form for
            it.
        """
        expectation = self.untruncated.mean.as_array
        if not self.event.contains(expectation):
            raise IntractableError(self)
        return (
            self.untruncated.log_mode()[0],
            float(self.log_likelihood(expectation.reshape(1, -1))[0]),
        )

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[ProbabilisticModel], float]:
        """
        Rule out more of the space.

        :param event: What is still to be considered possible.
        :param singleton_allowed: Ignored, since every single point has probability
            zero.
        :return: The further confined distribution and the log-probability of the event
            under this one, or nothing at all if nothing is left.
        """
        event.fill_missing_variables(set(self.variables))
        surviving = event & self.event
        probability = self.probability(surviving)
        if probability == 0.0:
            return None, -np.inf
        return (
            type(self)(untruncated=self.untruncated, event=surviving),
            math.log(probability),
        )

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[Self], float]:
        """
        :raises UndefinedOperationError: Always. Fixing a quantity of a confined Gaussian
            leaves the Gaussian conditional confined to the slice the event makes at that
            value, which nothing asks for yet.
        """
        raise UndefinedOperationError(self)

    def sample(self, amount: int) -> npt.NDArray:
        """
        Sample by drawing from the untruncated Gaussian and keeping what the event
        allows.

        :param amount: How many samples to draw.
        :return: That many samples, all of them inside the event.
        """
        kept = np.empty((0, len(self.variables)))
        while len(kept) < amount:
            drawn = self.untruncated.sample(amount)
            inside = np.array([self.event.contains(sample) for sample in drawn])
            kept = np.concatenate([kept, drawn[inside]])
        return kept[:amount]

    def __copy__(self) -> Self:
        from copy import copy

        return type(self)(untruncated=copy(self.untruncated), event=self.event)
