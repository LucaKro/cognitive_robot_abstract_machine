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
from typing_extensions import Any, Dict, Iterable, List, Optional, Self, Tuple

from probabilistic_model.exceptions import (
    IntractableError,
    ShapeMismatchError,
    UndefinedOperationError,
    VariableNotInDistributionError,
)
from probabilistic_model.probabilistic_model import (
    CenterType,
    MomentType,
    OrderType,
    ProbabilisticModel,
)

# %% a Gaussian over several variables at once


@dataclass
class MultivariateGaussianDistribution(ProbabilisticModel):
    """
    A multivariate Gaussian distribution over continuous random variables.

    Every tractable query is answered by :mod:`scipy.stats.multivariate_normal`, the
    same way :class:`~probabilistic_model.distributions.gaussian.GaussianDistribution`
    answers its own through :mod:`scipy.stats.norm`.
    """

    distribution_variables: Tuple[Continuous, ...]
    """
    The variables of the distribution.
    """

    mean: npt.NDArray
    """
    The mean.

    Its dimension corresponds to the variables in the same order.
    """

    covariance: npt.NDArray
    """
    The covariance matrix. Both of its dimensions correspond to the variables in the
    same order.
    """

    def __post_init__(self):
        """
        :raises ShapeMismatchError: If the mean or the covariance is not laid out by
            this distribution's variables.
        """
        self.distribution_variables = tuple(self.distribution_variables)
        self.mean = np.asarray(self.mean, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)

        amount = len(self.distribution_variables)
        if self.mean.shape != (amount,):
            raise ShapeMismatchError(self.mean.shape, (amount,))
        if self.covariance.shape != (amount, amount):
            raise ShapeMismatchError(self.covariance.shape, (amount, amount))

    @property
    def variables(self) -> Tuple[Continuous, ...]:
        return tuple(self.distribution_variables)

    @property
    def support(self) -> Event:
        return SimpleEvent.from_data(
            {variable: reals() for variable in self.variables}
        ).as_composite_set()

    @property
    def scipy_distribution(self) -> Any:
        """
        :return: The scipy distribution every tractable query is answered by.
        """
        return multivariate_normal(mean=self.mean, cov=self.covariance)

    # %% reading it by variable

    def index_of(self, variable: Continuous) -> int:
        """
        :param variable: The variable to locate.
        :return: The row it occupies in the mean and the covariance.
        :raises VariableNotInDistributionError: If this distribution is not over it.
        """
        if variable not in self.distribution_variables:
            raise VariableNotInDistributionError(
                variable=variable, variables=list(self.distribution_variables)
            )
        return self.distribution_variables.index(variable)

    def mean_of(self, variable: Continuous) -> float:
        """
        :param variable: The variable to read.
        :return: Its mean.
        :raises VariableNotInDistributionError: If this distribution is not over it.
        """
        return float(self.mean[self.index_of(variable)])

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The variable to read.
        :return: Its variance.
        :raises VariableNotInDistributionError: If this distribution is not over it.
        """
        return self.covariance_between(variable, variable)

    def covariance_between(self, first: Continuous, second: Continuous) -> float:
        """
        The covariance of two variables is an integral over both of them, and this
        distribution is one of the few for which it is tractable in closed form.

        :param first: One of the variables.
        :param second: The other one.
        :return: Their covariance.
        :raises VariableNotInDistributionError: If this distribution is not over either
            of them.
        """
        return float(self.covariance[self.index_of(first), self.index_of(second)])

    # %% density and probability

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        return np.atleast_1d(self.scipy_distribution.logpdf(events))

    def cumulative_distribution_function(self, events: npt.NDArray) -> npt.NDArray:
        return np.atleast_1d(self.scipy_distribution.cdf(events))

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        """
        The probability of an axis-aligned box under a correlated Gaussian has no closed
        form, so it is integrated numerically. A variable confined to several stretches
        makes several boxes, and their probabilities add.

        :param event: The box, or boxes, to measure.
        :return: How probable it is.
        """
        stretches_per_variable = [
            tuple(event[variable].simple_sets) for variable in self.variables
        ]
        return float(
            sum(
                self._probability_of_box(box)
                for box in itertools.product(*stretches_per_variable)
            )
        )

    def _probability_of_box(self, box: Tuple[SimpleInterval, ...]) -> float:
        """
        :param box: One unbroken stretch per variable, in this distribution's order.
        :return: How probable it is that every variable falls in its own stretch.
        """
        probability = multivariate_normal.cdf(
            np.array([stretch.upper for stretch in box]),
            mean=self.mean,
            cov=self.covariance,
            lower_limit=np.array([stretch.lower for stretch in box]),
        )
        return max(float(probability), 0.0)

    def log_mode(self) -> Tuple[Event, float]:
        """
        A Gaussian is most dense exactly at its mean.

        :return: The mean, and the log-density there.
        """
        mode = SimpleEvent.from_data(
            {variable: singleton(self.mean_of(variable)) for variable in self.variables}
        ).as_composite_set()
        return mode, float(self.log_likelihood(self.mean.reshape(1, -1))[0])

    # %% reading fewer variables

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        :param variables: The variables to keep. They are kept in this distribution's
            own order, whatever order they are asked for in.
        :return: The Gaussian over only those variables.
        :raises VariableNotInDistributionError: If this distribution is not over one of
            them.
        """
        kept = set(variables)
        return self._over_rows(
            [self.index_of(variable) for variable in self.variables if variable in kept]
        )

    def _over_rows(self, rows: List[int]) -> Self:
        """
        :param rows: The rows to keep, in this distribution's own order.
        :return: The Gaussian over the variables those rows belong to.
        """
        return type(self)(
            distribution_variables=tuple(
                self.distribution_variables[row] for row in rows
            ),
            mean=self.mean[rows],
            covariance=self.covariance[np.ix_(rows, rows)],
        )

    # %% fixing variables at a value

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[ProbabilisticModel], float]:
        """
        Fix some of the variables at the values given and answer with the distribution
        over the rest, which stays Gaussian.

        Fixing *every* variable leaves a point mass at the values given, which is a
        product of Dirac impulses rather than a Gaussian.

        :param point: What each fixed variable is known to be.
        :return: The distribution over whatever is left, and the log-density of the
            values given.
        :raises VariableNotInDistributionError: If a fixed variable is not one of this
            distribution's.
        """
        fixed_rows = [self.index_of(variable) for variable in point]
        free_rows = [
            row for row in range(len(self.variables)) if row not in set(fixed_rows)
        ]

        fixed_at = np.array([float(point[variable]) for variable in point])
        log_density = float(
            self._over_rows(fixed_rows).log_likelihood(fixed_at.reshape(1, -1))[0]
        )

        if not free_rows:
            return self._point_mass_at(point), log_density
        return self._conditioned(free_rows, fixed_rows, fixed_at), log_density

    def _point_mass_at(self, point: Dict[Variable, Any]) -> ProbabilisticModel:
        """
        :param point: What every variable is known to be.
        :return: The product of one Dirac impulse per variable, which is what a
            distribution conditioned on all of its own variables is.
        """
        from probabilistic_model.distributions.distributions import (
            DiracDeltaDistribution,
        )
        from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
            ProbabilisticCircuit,
            ProductUnit,
            leaf,
        )

        circuit = ProbabilisticCircuit()
        product = ProductUnit(probabilistic_circuit=circuit)
        for variable in self.variables:
            product.add_subcircuit(
                leaf(
                    DiracDeltaDistribution(
                        variable=variable, location=float(point[variable])
                    ),
                    circuit,
                )
            )
        return product.probabilistic_circuit

    def _conditioned(
        self,
        free_rows: List[int],
        fixed_rows: List[int],
        fixed_at: npt.NDArray,
    ) -> Self:
        """
        :param free_rows: The rows the answer is over, in this distribution's order.
        :param fixed_rows: The rows held at a value, in the order ``fixed_at`` uses.
        :param fixed_at: What those rows are held at.
        :return: The Gaussian over ``free_rows``, narrowed by what the fixed rows say.
        """
        free = self._over_rows(free_rows)
        cross = self.covariance[np.ix_(free_rows, fixed_rows)]
        explained = cross @ np.linalg.inv(
            self.covariance[np.ix_(fixed_rows, fixed_rows)]
        )
        return type(self)(
            distribution_variables=free.distribution_variables,
            mean=free.mean + explained @ (fixed_at - self.mean[fixed_rows]),
            covariance=self._symmetrized(free.covariance - explained @ cross.T),
        )

    @staticmethod
    def _symmetrized(covariance: npt.NDArray) -> npt.NDArray:
        """
        A covariance is symmetric by definition, so any difference between a matrix and
        its transpose is rounding. Averaging the two removes it exactly.

        :param covariance: The covariance as the arithmetic left it.
        :return: The same covariance, exactly symmetric.
        """
        return (covariance + covariance.T) / 2

    def conditional_on_measurement(
        self,
        model: npt.NDArray,
        measured: npt.NDArray,
        noise: npt.NDArray,
    ) -> Self:
        """
        Correct the mean with what was measured.

        The measurement and the mean are weighed against each other by how uncertain
        each is. This is :meth:`conditional` applied to the joint distribution over
        these variables together with what the measurement reports, which is what a
        measurement update is.

        :param model: How much each variable contributes to each measured number, with
            one row per number measured.
        :param measured: The numbers measured, one per row of ``model``.
        :param noise: How far the measurement scatters, one row and column per number
            measured.
        :return: The corrected Gaussian over the same variables.
        :raises ShapeMismatchError: If the three do not describe one measurement of
            these variables.
        """
        model = np.atleast_2d(np.asarray(model, dtype=float))
        measured = np.atleast_1d(np.asarray(measured, dtype=float))
        noise = np.atleast_2d(np.asarray(noise, dtype=float))

        amount = len(model)
        if model.shape != (amount, len(self.variables)):
            raise ShapeMismatchError(model.shape, (amount, len(self.variables)))
        if measured.shape != (amount,):
            raise ShapeMismatchError(measured.shape, (amount,))
        if noise.shape != (amount, amount):
            raise ShapeMismatchError(noise.shape, (amount, amount))

        if amount == 0:
            return self

        joint = self._joint_with_measurement(model, noise)
        return joint._conditioned(
            list(range(len(self.variables))),
            list(range(len(self.variables), len(joint.variables))),
            measured,
        )

    def _joint_with_measurement(self, model: npt.NDArray, noise: npt.NDArray) -> Self:
        """
        :param model: How much each variable contributes to each measured number.
        :param noise: How far the measurement scatters.
        :return: The Gaussian over these variables followed by what is measured.
        """
        return type(self)(
            distribution_variables=(
                *self.distribution_variables,
                *self._variables_for_measurement(len(model)),
            ),
            mean=np.concatenate([self.mean, model @ self.mean]),
            covariance=self._symmetrized(
                np.block(
                    [
                        [self.covariance, self.covariance @ model.T],
                        [
                            model @ self.covariance,
                            model @ self.covariance @ model.T + noise,
                        ],
                    ]
                )
            ),
        )

    def _variables_for_measurement(self, amount: int) -> List[Continuous]:
        """
        What is measured is a variable of the joint only while the measurement is being
        applied, so it is named here rather than asked of the caller.

        :param amount: How many measured numbers need a name.
        :return: That many names, none of which this distribution already uses.
        """
        taken = {variable.name for variable in self.variables}
        names = []
        for position in range(amount):
            name = f"measurement {position}"
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

    # %% moments

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        """
        Every moment asked for here is of one variable on its own, so each is answered
        by that variable's own marginal.

        :param order: The order of the moment of each variable to answer for.
        :param center: What to take each of those moments about.
        :return: The moment of each variable asked for.
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
        Move the mean, leaving the covariance as it is.

        :param translation: How far to move each variable; one left out does not move.
        :raises VariableNotInDistributionError: If it names a variable this distribution
            is not over.
        """
        for variable, distance in translation.items():
            self.mean[self.index_of(variable)] += distance

    def apply_scaling(self, scaling: Dict[Variable, float]):
        """
        Stretch the variables, which stretches the mean once and the covariance once per
        variable it relates.

        :param scaling: What to multiply each variable by; one left out keeps its size.
        :raises VariableNotInDistributionError: If it names a variable this distribution
            is not over.
        """
        factors = np.ones(len(self.variables))
        for variable, factor in scaling.items():
            factors[self.index_of(variable)] = factor
        self.mean = self.mean * factors
        self.covariance = self.covariance * np.outer(factors, factors)

    def apply_linear_map(self, mapping: npt.NDArray):
        """
        Make each variable the weighted sum of the variables it is mapped from, which is
        what carrying a distribution forward through a linear change does.

        :param mapping: How much each variable on the way in contributes to each
            variable on the way out, one row and column per variable. The identity
            leaves every variable as it is.
        :raises ShapeMismatchError: If it is not laid out by this distribution's
            variables.
        """
        mapping = np.asarray(mapping, dtype=float)
        amount = len(self.variables)
        if mapping.shape != (amount, amount):
            raise ShapeMismatchError(mapping.shape, (amount, amount))
        self.mean = mapping @ self.mean
        self.covariance = self._symmetrized(mapping @ self.covariance @ mapping.T)

    def apply_added_covariance(self, added: npt.NDArray):
        """
        Make the variables less certain, which is what keeps a distribution nobody is
        measuring from staying confident forever.

        :param added: The covariance to add, one row and column per variable.
        :raises ShapeMismatchError: If it is not laid out by this distribution's
            variables.
        """
        added = np.asarray(added, dtype=float)
        amount = len(self.variables)
        if added.shape != (amount, amount):
            raise ShapeMismatchError(added.shape, (amount, amount))
        self.covariance = self.covariance + added

    # %% sampling

    def sample(self, amount: int) -> npt.NDArray:
        return self.scipy_distribution.rvs(size=amount).reshape(
            amount, len(self.variables)
        )

    def __copy__(self) -> Self:
        return type(self)(
            distribution_variables=self.distribution_variables,
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
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
        The most likely point is the untruncated mean whenever the event still contains
        it, since the density falls away from there in every direction.

        :return: That point and its log-density.
        :raises IntractableError: If the mean was ruled out, which leaves the most
            likely point somewhere on the event's boundary and no closed form for it.
        """
        mean = self.untruncated.mean
        if not self.event.contains(mean):
            raise IntractableError(self)
        return (
            self.untruncated.log_mode()[0],
            float(self.log_likelihood(mean.reshape(1, -1))[0]),
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
        :raises UndefinedOperationError: Always. Fixing a variable of a confined Gaussian
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
