import math

import numpy as np
import pytest
from random_events.interval import closed, open_closed, singleton
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.variable import Continuous

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    TruncatedMultivariateGaussianDistribution,
)
from probabilistic_model.exceptions import (
    IntractableError,
    ShapeMismatchError,
    UndefinedOperationError,
    VariableNotInDistributionError,
)

# %% shared fixtures


@pytest.fixture
def horizontal() -> Continuous:
    return Continuous("horizontal")


@pytest.fixture
def vertical() -> Continuous:
    return Continuous("vertical")


@pytest.fixture
def independent(horizontal, vertical) -> MultivariateGaussianDistribution:
    """
    Two quantities that do not co-vary, so the joint is the product of two marginals and
    every answer can be checked against the univariate distribution already in this
    package.
    """
    return MultivariateGaussianDistribution(
        distribution_variables=(horizontal, vertical),
        mean=np.array([1.0, -2.0]),
        covariance=np.array([[4.0, 0.0], [0.0, 9.0]]),
    )


@pytest.fixture
def correlated(horizontal, vertical) -> MultivariateGaussianDistribution:
    """
    Standard quantities correlated by 0.6, which is the case with no closed form for the
    probability of a box.
    """
    return MultivariateGaussianDistribution(
        distribution_variables=(horizontal, vertical),
        mean=np.array([0.0, 0.0]),
        covariance=np.array([[1.0, 0.6], [0.6, 1.0]]),
    )


def marginal_of(
    distribution: MultivariateGaussianDistribution, variable: Continuous
) -> GaussianDistribution:
    """
    :return: The univariate distribution of one quantity, to check a joint answer
        against the implementation this package already has.
    """
    return GaussianDistribution(
        variable=variable,
        location=distribution.mean_of(variable),
        scale=math.sqrt(distribution.variance_of(variable)),
    )


def one_variable(
    variable: Continuous, mean: float, variance: float
) -> MultivariateGaussianDistribution:
    """
    :return: A distribution over a single quantity, which several tests need and which
        carries no layout worth restating at each of them.
    """
    return MultivariateGaussianDistribution(
        distribution_variables=(variable,),
        mean=np.array([mean]),
        covariance=np.array([[variance]]),
    )


# %% what the distribution is built from and holds


class TestBuildingADistribution:
    def test_the_estimate_and_the_uncertainty_are_read_by_quantity(
        self, independent, horizontal, vertical
    ):
        assert independent.mean_of(horizontal) == 1.0
        assert independent.mean_of(vertical) == -2.0
        assert independent.variance_of(horizontal) == 4.0
        assert independent.variance_of(vertical) == 9.0

    def test_a_covariance_is_read_in_either_direction(
        self, correlated, horizontal, vertical
    ):
        assert correlated.covariance_between(horizontal, vertical) == 0.6
        assert correlated.covariance_between(vertical, horizontal) == 0.6

    def test_a_distribution_about_one_quantity_needs_no_layout(self, horizontal):
        distribution = MultivariateGaussianDistribution(
            distribution_variables=(horizontal,),
            mean=np.array([3.0]),
            covariance=np.array([[0.25]]),
        )
        assert distribution.mean_of(horizontal) == 3.0
        assert distribution.variance_of(horizontal) == 0.25
        assert distribution.variables == (horizontal,)

    def test_a_mean_that_is_not_laid_out_by_the_variables_is_rejected(
        self, horizontal, vertical
    ):
        with pytest.raises(ShapeMismatchError) as error:
            MultivariateGaussianDistribution(
                distribution_variables=(horizontal, vertical),
                mean=np.array([0.0]),
                covariance=np.zeros((2, 2)),
            )
        assert error.value.expected_shape == (2,)
        assert error.value.received_shape == (1,)

    def test_a_covariance_that_is_not_laid_out_by_the_variables_is_rejected(
        self, horizontal, vertical
    ):
        with pytest.raises(ShapeMismatchError) as error:
            MultivariateGaussianDistribution(
                distribution_variables=(horizontal, vertical),
                mean=np.zeros(2),
                covariance=np.zeros((2, 3)),
            )
        assert error.value.expected_shape == (2, 2)
        assert error.value.received_shape == (2, 3)

    def test_reading_a_quantity_the_distribution_is_not_about_is_rejected(
        self, independent
    ):
        absent = Continuous("absent")
        with pytest.raises(VariableNotInDistributionError) as error:
            independent.mean_of(absent)
        assert error.value.variable == absent

    def test_the_variables_keep_the_layout_order(
        self, independent, horizontal, vertical
    ):
        assert independent.variables == (horizontal, vertical)

    def test_the_mean_and_covariance_are_laid_out_by_the_variables(
        self, independent, horizontal, vertical
    ):
        assert independent.mean.tolist() == [1.0, -2.0]
        assert independent.covariance.tolist() == [[4.0, 0.0], [0.0, 9.0]]


# %% density


class TestLikelihood:
    def test_the_joint_density_of_independent_quantities_is_the_product_of_the_marginals(
        self, independent, horizontal, vertical
    ):
        point = np.array([[0.5, -1.0]])
        expected = (
            marginal_of(independent, horizontal).likelihood(np.array([[0.5]]))[0]
            * marginal_of(independent, vertical).likelihood(np.array([[-1.0]]))[0]
        )
        assert independent.likelihood(point)[0] == pytest.approx(expected)

    def test_every_point_is_scored(self, independent):
        points = np.array([[0.0, 0.0], [1.0, -2.0], [5.0, 5.0]])
        assert independent.log_likelihood(points).shape == (3,)

    def test_the_mean_is_the_most_likely_point(self, correlated):
        at_mean = correlated.log_likelihood(np.array([[0.0, 0.0]]))[0]
        elsewhere = correlated.log_likelihood(np.array([[0.4, -0.7]]))[0]
        assert at_mean > elsewhere


# %% the probability of a box, which has no closed form when the quantities co-vary


class TestProbabilityOfABox:
    def test_the_whole_support_is_certain(self, correlated):
        assert correlated.probability(correlated.support) == pytest.approx(1.0)

    def test_a_box_over_independent_quantities_is_the_product_of_the_marginals(
        self, independent, horizontal, vertical
    ):
        event = SimpleEvent.from_data(
            {horizontal: closed(0.0, 2.0), vertical: closed(-3.0, 1.0)}
        )
        expected = marginal_of(independent, horizontal).probability_of_simple_event(
            event
        ) * marginal_of(independent, vertical).probability_of_simple_event(event)
        assert independent.probability_of_simple_event(event) == pytest.approx(expected)

    def test_a_quadrant_of_correlated_quantities_follows_the_orthant_formula(
        self, correlated, horizontal, vertical
    ):
        """
        The probability that two standard correlated quantities are both positive is
        ``1/4 + arcsin(correlation) / 2pi`` — the one box probability a correlated
        Gaussian has in closed form, so it checks the numerical integration against
        something other than itself.
        """
        both_positive = SimpleEvent.from_data(
            {
                horizontal: open_closed(0.0, np.inf),
                vertical: open_closed(0.0, np.inf),
            }
        )
        correlation = correlated.covariance_between(horizontal, vertical)
        assert correlated.probability_of_simple_event(both_positive) == pytest.approx(
            0.25 + math.asin(correlation) / (2 * math.pi)
        )

    def test_correlation_changes_the_probability_of_a_quadrant(
        self, correlated, independent, horizontal, vertical
    ):
        """
        Without this the integration could be ignoring the off-diagonal entirely and
        every other assertion here would still hold.
        """
        both_positive = SimpleEvent.from_data(
            {
                horizontal: open_closed(0.0, np.inf),
                vertical: open_closed(0.0, np.inf),
            }
        )
        assert correlated.probability_of_simple_event(both_positive) > 0.25

    def test_a_quantity_confined_to_two_stretches_sums_them(
        self, independent, horizontal, vertical
    ):
        whole_column = closed(-np.inf, np.inf)
        lower = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: whole_column}
        )
        upper = SimpleEvent.from_data(
            {horizontal: closed(3.0, 4.0), vertical: whole_column}
        )
        both = SimpleEvent.from_data(
            {
                horizontal: closed(0.0, 1.0) | closed(3.0, 4.0),
                vertical: whole_column,
            }
        )
        assert independent.probability_of_simple_event(both) == pytest.approx(
            independent.probability_of_simple_event(lower)
            + independent.probability_of_simple_event(upper)
        )

    def test_a_box_with_no_width_is_impossible(self, correlated, horizontal, vertical):
        flattened = SimpleEvent.from_data(
            {horizontal: singleton(0.0), vertical: closed(-1.0, 1.0)}
        )
        assert correlated.probability_of_simple_event(flattened) == 0.0


# %% the most likely point


class TestMode:
    def test_the_mode_is_the_mean(self, independent, horizontal, vertical):
        mode, likelihood = independent.mode()
        assert mode.contains(np.array([1.0, -2.0]))
        assert likelihood == pytest.approx(
            independent.likelihood(np.array([[1.0, -2.0]]))[0]
        )


# %% conditioning on a value, which is the measurement update


class TestConditioningOnAValue:
    def test_the_conditional_density_is_the_joint_divided_by_the_marginal(
        self, correlated, horizontal, vertical
    ):
        """
        The definition of conditioning, checked against the distribution's own joint
        density and its own marginal rather than against a second copy of the Gaussian
        conditioning formula.
        """
        given = 0.8
        conditioned, _ = correlated.conditional({vertical: given})
        somewhere = 0.3

        joint = correlated.likelihood(np.array([[somewhere, given]]))[0]
        marginal = correlated.marginal([vertical]).likelihood(np.array([[given]]))[0]
        assert conditioned.likelihood(np.array([[somewhere]]))[0] == pytest.approx(
            joint / marginal
        )

    def test_conditioning_leaves_only_the_free_quantities(
        self, correlated, horizontal, vertical
    ):
        conditioned, _ = correlated.conditional({vertical: 0.0})
        assert conditioned.variables == (horizontal,)

    def test_the_probability_returned_is_the_marginal_density_of_the_value(
        self, correlated, vertical
    ):
        given = 0.8
        _, probability = correlated.conditional({vertical: given})
        assert probability == pytest.approx(
            correlated.marginal([vertical]).likelihood(np.array([[given]]))[0]
        )

    def test_conditioning_an_uncorrelated_quantity_leaves_it_alone(
        self, independent, horizontal, vertical
    ):
        conditioned, _ = independent.conditional({vertical: 100.0})
        assert conditioned.mean_of(horizontal) == pytest.approx(
            independent.mean_of(horizontal)
        )
        assert conditioned.variance_of(horizontal) == pytest.approx(
            independent.variance_of(horizontal)
        )

    def test_conditioning_always_narrows_a_correlated_quantity(
        self, correlated, horizontal, vertical
    ):
        conditioned, _ = correlated.conditional({vertical: 0.0})
        assert conditioned.variance_of(horizontal) < correlated.variance_of(horizontal)

    def test_the_conditioned_uncertainty_stays_exactly_symmetric(self, horizontal):
        """
        Rounding is the only thing that can make a conditional covariance asymmetric,
        and an asymmetric one at a control cycle's rate shows up much later as an
        unexplained uncertainty.
        """
        first, second, given = horizontal, Continuous("second"), Continuous("given")
        distribution = MultivariateGaussianDistribution(
            distribution_variables=(first, second, given),
            mean=np.zeros(3),
            covariance=np.array(
                [
                    [1e8, 1e-2, 0.9e4],
                    [1e-2, 1e-8, 1e-5],
                    [0.9e4, 1e-5, 3.0],
                ]
            ),
        )
        conditioned, _ = distribution.conditional({given: 1.0})
        assert conditioned.covariance_between(
            first, second
        ) == conditioned.covariance_between(second, first)

    def test_conditioning_on_every_quantity_leaves_a_point_mass(
        self, independent, horizontal, vertical
    ):
        """
        Nothing is left free, so what remains is the Dirac impulse at the values given
        rather than a Gaussian over nothing.
        """
        conditioned, _ = independent.conditional({horizontal: 1.5, vertical: -0.5})
        assert set(conditioned.variables) == {horizontal, vertical}
        assert conditioned.likelihood(np.array([[1.5, -0.5]]))[0] == np.inf
        assert conditioned.likelihood(np.array([[0.0, 0.0]]))[0] == 0.0

    def test_conditioning_on_an_unknown_quantity_is_rejected(self, independent):
        absent = Continuous("absent")
        with pytest.raises(VariableNotInDistributionError) as error:
            independent.conditional({absent: 0.0})
        assert error.value.variable == absent


# %% correcting an estimate with what a sensor reported


class TestCorrectingWithAMeasurement:
    def test_a_measurement_pulls_the_mean_toward_what_was_measured(
        self, independent, horizontal
    ):
        corrected = independent.conditional_on_measurement(
            model=np.array([[1.0, 0.0]]),
            measured=np.array([5.0]),
            noise=np.array([[1.0]]),
        )
        assert independent.mean_of(horizontal) < corrected.mean_of(horizontal) < 5.0

    def test_a_measurement_always_leaves_the_mean_more_certain(
        self, independent, horizontal
    ):
        corrected = independent.conditional_on_measurement(
            model=np.array([[1.0, 0.0]]),
            measured=np.array([5.0]),
            noise=np.array([[4.0]]),
        )
        assert corrected.variance_of(horizontal) < independent.variance_of(horizontal)

    def test_a_measurement_of_equal_certainty_lands_halfway(self, horizontal):
        """
        With the mean and the measurement equally uncertain, neither outweighs the
        other, so the corrected mean is their midpoint exactly.
        """
        corrected = one_variable(horizontal, 0.0, 2.0).conditional_on_measurement(
            model=np.array([[1.0]]),
            measured=np.array([10.0]),
            noise=np.array([[2.0]]),
        )
        assert corrected.mean_of(horizontal) == pytest.approx(5.0)

    def test_repeated_measurements_accumulate_into_the_covariance(self, horizontal):
        """
        Checked against the information form — precisions add — rather than against a
        stored number, so the recursion is verified against an independent formulation
        of the same law instead of a second copy of itself.
        """
        starting_variance, measurement_variance, measurements = 1.0, 4.0, 100
        distribution = one_variable(horizontal, 0.0, starting_variance)
        for _ in range(measurements):
            distribution = distribution.conditional_on_measurement(
                model=np.array([[1.0]]),
                measured=np.array([1.0]),
                noise=np.array([[measurement_variance]]),
            )

        expected_precision = 1 / starting_variance + measurements / measurement_variance
        assert distribution.variance_of(horizontal) == pytest.approx(
            1 / expected_precision
        )

    def test_a_measurement_of_several_quantities_at_once_corrects_all_of_them(
        self, horizontal, vertical
    ):
        """
        A sensor reporting the sum of two quantities says nothing about either one
        alone, which is what stating a measurement model buys over reading a quantity
        itself.
        """
        distribution = MultivariateGaussianDistribution(
            distribution_variables=(horizontal, vertical),
            mean=np.zeros(2),
            covariance=np.eye(2),
        )
        corrected = distribution.conditional_on_measurement(
            model=np.array([[1.0, 1.0]]),
            measured=np.array([4.0]),
            noise=np.array([[1.0]]),
        )
        assert corrected.mean_of(horizontal) > 0.0
        assert corrected.mean_of(vertical) > 0.0
        assert corrected.mean_of(horizontal) == pytest.approx(
            corrected.mean_of(vertical)
        )

    def test_measuring_nothing_leaves_the_mean_alone(self, independent, horizontal):
        corrected = independent.conditional_on_measurement(
            model=np.zeros((0, 2)), measured=np.zeros(0), noise=np.zeros((0, 0))
        )
        assert corrected.mean_of(horizontal) == independent.mean_of(horizontal)
        assert corrected.variance_of(horizontal) == independent.variance_of(horizontal)

    def test_correcting_does_not_change_the_distribution_it_corrected(
        self, independent, horizontal
    ):
        before = independent.mean_of(horizontal)
        independent.conditional_on_measurement(
            model=np.array([[1.0, 0.0]]),
            measured=np.array([5.0]),
            noise=np.array([[1.0]]),
        )
        assert independent.mean_of(horizontal) == before

    def test_a_measurement_model_of_the_wrong_width_is_rejected(self, independent):
        with pytest.raises(ShapeMismatchError) as error:
            independent.conditional_on_measurement(
                model=np.array([[1.0]]),
                measured=np.array([0.0]),
                noise=np.array([[1.0]]),
            )
        assert error.value.expected_shape == (1, 2)
        assert error.value.received_shape == (1, 1)

    def test_measuring_more_numbers_than_the_model_describes_is_rejected(
        self, independent
    ):
        with pytest.raises(ShapeMismatchError) as error:
            independent.conditional_on_measurement(
                model=np.array([[1.0, 0.0]]),
                measured=np.array([0.0, 1.0]),
                noise=np.array([[1.0]]),
            )
        assert error.value.expected_shape == (1,)
        assert error.value.received_shape == (2,)


# %% reading fewer quantities than the distribution is about


class TestMarginal:
    def test_a_marginal_keeps_the_estimate_and_uncertainty_of_what_it_kept(
        self, correlated, horizontal
    ):
        marginal = correlated.marginal([horizontal])
        assert marginal.variables == (horizontal,)
        assert marginal.mean_of(horizontal) == correlated.mean_of(horizontal)
        assert marginal.variance_of(horizontal) == correlated.variance_of(horizontal)

    def test_a_marginal_is_laid_out_in_the_distribution_s_own_order(
        self, correlated, horizontal, vertical
    ):
        assert correlated.marginal([vertical, horizontal]).variables == (
            horizontal,
            vertical,
        )


# %% moments


class TestMoments:
    def test_the_expectation_is_the_estimate(self, independent, horizontal, vertical):
        expectation = independent.expectation()
        assert expectation[horizontal] == pytest.approx(1.0)
        assert expectation[vertical] == pytest.approx(-2.0)

    def test_the_variance_is_the_uncertainty(self, independent, horizontal, vertical):
        variance = independent.variance()
        assert variance[horizontal] == pytest.approx(4.0)
        assert variance[vertical] == pytest.approx(9.0)

    def test_a_higher_moment_matches_the_quantity_s_own_marginal(
        self, independent, horizontal
    ):
        order = VariableMap({horizontal: 4})
        center = VariableMap({horizontal: 0.0})
        assert independent.moment(order, center)[horizontal] == pytest.approx(
            marginal_of(independent, horizontal).moment(order, center)[horizontal]
        )


# %% moving and stretching


class TestTranslationAndScaling:
    def test_translating_moves_the_estimate_and_leaves_the_uncertainty(
        self, independent, horizontal, vertical
    ):
        independent.apply_translation({horizontal: 3.0})
        assert independent.mean_of(horizontal) == 4.0
        assert independent.mean_of(vertical) == -2.0
        assert independent.variance_of(horizontal) == 4.0

    def test_scaling_stretches_the_estimate_and_squares_into_the_uncertainty(
        self, independent, horizontal, vertical
    ):
        independent.apply_scaling({horizontal: 2.0})
        assert independent.mean_of(horizontal) == 2.0
        assert independent.variance_of(horizontal) == 16.0
        assert independent.variance_of(vertical) == 9.0

    def test_scaling_carries_into_a_shared_uncertainty_once_per_quantity(
        self, correlated, horizontal, vertical
    ):
        correlated.apply_scaling({horizontal: 2.0})
        assert correlated.covariance_between(horizontal, vertical) == pytest.approx(1.2)

    def test_a_quantity_left_out_of_a_scaling_keeps_its_size(
        self, independent, vertical
    ):
        independent.apply_scaling({})
        assert independent.mean_of(vertical) == -2.0
        assert independent.variance_of(vertical) == 9.0


# %% sampling


class TestSampling:
    def test_every_sample_carries_one_number_per_quantity(self, correlated):
        assert correlated.sample(7).shape == (7, 2)

    def test_samples_of_one_quantity_are_still_a_column(self, horizontal):
        assert one_variable(horizontal, 0.0, 1.0).sample(5).shape == (5, 1)

    def test_samples_fall_where_the_distribution_says_they_should(self, independent):
        np.random.seed(69)
        samples = independent.sample(20000)
        assert samples.mean(axis=0) == pytest.approx(np.array([1.0, -2.0]), abs=0.1)


# %% copying


class TestCopying:
    def test_a_copy_moves_without_moving_the_original(self, independent, horizontal):
        from copy import copy

        copied = copy(independent)
        copied.apply_translation({horizontal: 10.0})
        assert independent.mean_of(horizontal) == 1.0
        assert copied.mean_of(horizontal) == 11.0


# %% confining a distribution to an event


class TestTruncation:
    def test_truncating_answers_with_a_distribution_that_is_no_longer_gaussian(
        self, correlated, horizontal, vertical
    ):
        """
        A correlated Gaussian confined to a box is not a Gaussian, so truncation cannot
        answer with one of its own kind.
        """
        box = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, probability = correlated.truncated(box)
        assert isinstance(truncated, TruncatedMultivariateGaussianDistribution)
        assert probability == pytest.approx(correlated.probability(box))

    def test_an_impossible_event_leaves_nothing(self, correlated, horizontal, vertical):
        nothing = SimpleEvent.from_data(
            {horizontal: singleton(0.0), vertical: singleton(0.0)}
        ).as_composite_set()
        truncated, probability = correlated.truncated(nothing)
        assert truncated is None
        assert probability == 0.0

    def test_the_truncated_density_is_the_original_scaled_up_to_one(
        self, correlated, horizontal, vertical
    ):
        box = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, probability = correlated.truncated(box)
        inside = np.array([[0.5, 0.5]])
        assert truncated.likelihood(inside)[0] == pytest.approx(
            correlated.likelihood(inside)[0] / probability
        )

    def test_nothing_outside_the_event_can_happen(
        self, correlated, horizontal, vertical
    ):
        box = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        assert truncated.likelihood(np.array([[5.0, 5.0]]))[0] == 0.0

    def test_the_truncated_distribution_is_certain_of_its_own_event(
        self, correlated, horizontal, vertical
    ):
        box = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        assert truncated.probability(truncated.support) == pytest.approx(1.0)

    def test_the_mode_is_the_mean_when_the_mean_survived_the_truncation(
        self, correlated, horizontal, vertical
    ):
        box = SimpleEvent.from_data(
            {horizontal: closed(-1.0, 1.0), vertical: closed(-1.0, 1.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        mode, _ = truncated.mode()
        assert mode.contains(np.array([0.0, 0.0]))

    def test_the_mode_is_intractable_once_the_mean_is_cut_away(
        self, correlated, horizontal, vertical
    ):
        """
        The most likely point is then somewhere on the event's boundary, which has no
        closed form for a correlated Gaussian.
        """
        box = SimpleEvent.from_data(
            {horizontal: closed(3.0, 4.0), vertical: closed(3.0, 4.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        with pytest.raises(IntractableError):
            truncated.mode()

    def test_truncating_again_narrows_the_event(self, correlated, horizontal, vertical):
        box = SimpleEvent.from_data(
            {horizontal: closed(0.0, 2.0), vertical: closed(0.0, 2.0)}
        ).as_composite_set()
        smaller = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        narrowed, probability = truncated.truncated(smaller)
        assert narrowed.likelihood(np.array([[1.5, 1.5]]))[0] == 0.0
        assert probability == pytest.approx(
            correlated.probability(smaller) / correlated.probability(box)
        )

    def test_every_sample_falls_inside_the_event(
        self, correlated, horizontal, vertical
    ):
        np.random.seed(69)
        box = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        samples = truncated.sample(200)
        assert samples.shape == (200, 2)
        assert all(truncated.support.contains(sample) for sample in samples)

    def test_conditioning_a_truncated_distribution_is_not_answered(
        self, correlated, horizontal, vertical
    ):
        box = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        with pytest.raises(UndefinedOperationError):
            truncated.conditional({vertical: 0.5})


# %% carrying the quantities forward through a linear change


class TestLinearMap:
    def test_a_quantity_becomes_the_weighted_sum_of_the_others(
        self, independent, horizontal, vertical
    ):
        independent.apply_linear_map(np.array([[1.0, 2.0], [0.0, 1.0]]))
        assert independent.mean_of(horizontal) == pytest.approx(1.0 + 2.0 * -2.0)

    def test_a_linear_map_carries_into_the_covariance_on_both_sides(
        self, independent, horizontal
    ):
        """
        The variance of a sum of two independent quantities is the sum of theirs.
        """
        independent.apply_linear_map(np.array([[1.0, 1.0], [0.0, 1.0]]))
        assert independent.variance_of(horizontal) == pytest.approx(4.0 + 9.0)

    def test_the_identity_changes_nothing(self, correlated):
        before = correlated.covariance.copy()
        correlated.apply_linear_map(np.eye(2))
        assert correlated.covariance.tolist() == before.tolist()

    def test_a_linear_map_not_laid_out_by_the_variables_is_rejected(self, independent):
        with pytest.raises(ShapeMismatchError) as error:
            independent.apply_linear_map(np.eye(3))
        assert error.value.expected_shape == (2, 2)
        assert error.value.received_shape == (3, 3)


# %% growing less certain


class TestAddedCovariance:
    def test_added_covariance_accumulates(self, independent, horizontal):
        independent.apply_added_covariance(np.array([[1.5, 0.0], [0.0, 0.0]]))
        assert independent.variance_of(horizontal) == pytest.approx(5.5)

    def test_an_added_covariance_reaches_both_directions_of_a_pair(
        self, independent, horizontal, vertical
    ):
        independent.apply_added_covariance(np.array([[0.0, 0.5], [0.5, 0.0]]))
        assert independent.covariance_between(horizontal, vertical) == pytest.approx(
            0.5
        )
        assert independent.covariance_between(vertical, horizontal) == pytest.approx(
            0.5
        )

    def test_adding_nothing_leaves_the_covariance_alone(self, correlated):
        before = correlated.covariance.copy()
        correlated.apply_added_covariance(np.zeros((2, 2)))
        assert correlated.covariance.tolist() == before.tolist()

    def test_an_added_covariance_not_laid_out_by_the_variables_is_rejected(
        self, independent
    ):
        with pytest.raises(ShapeMismatchError) as error:
            independent.apply_added_covariance(np.zeros((3, 3)))
        assert error.value.expected_shape == (2, 2)
        assert error.value.received_shape == (3, 3)
