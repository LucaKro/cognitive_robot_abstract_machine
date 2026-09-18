import numpy as np
import pytest
from random_events.variable import Continuous

from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.gaussian import (
    BeliefArray,
    GaussianBelief,
    Measurement,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    RepeatedVariableInBeliefError,
    UnknownBeliefError,
    VariableNotInBeliefError,
    WrongBeliefShapeError,
)

# %% the shape a belief has to have


class TestBeliefShape:
    """
    A belief's variables, mean and covariance all describe the same quantities, so they
    have to agree on how many there are.
    """

    def test_a_belief_about_one_quantity_has_one_dimension(self):
        belief = GaussianBelief.of_one_variable(
            Continuous("grasp"), mean=0.4, variance=0.25
        )
        assert belief.dimensions == 1
        assert belief.mean.tolist() == [0.4]
        assert belief.covariance.tolist() == [[0.25]]

    def test_a_mean_with_a_row_per_variable_is_accepted(self):
        belief = GaussianBelief(
            variables=[Continuous("x"), Continuous("y")],
            mean=np.zeros(2),
            covariance=np.eye(2),
        )
        assert belief.dimensions == 2

    def test_a_mean_that_does_not_have_a_row_per_variable_is_rejected(self):
        with pytest.raises(WrongBeliefShapeError) as error:
            GaussianBelief(
                variables=[Continuous("x")],
                mean=np.zeros(2),
                covariance=np.eye(1),
            )
        assert error.value.array == BeliefArray.MEAN
        assert error.value.expected_shape == (1,)
        assert error.value.actual_shape == (2,)

    def test_a_quantity_named_twice_is_rejected(self):
        """
        Only one of the two rows could ever be read back, so the other's estimate would
        be carried along and never answered with.
        """
        repeated = Continuous("x")
        with pytest.raises(RepeatedVariableInBeliefError) as error:
            GaussianBelief(
                variables=[repeated, repeated],
                mean=np.zeros(2),
                covariance=np.eye(2),
            )
        assert error.value.variable == repeated

    def test_a_covariance_that_is_not_square_over_the_variables_is_rejected(self):
        with pytest.raises(WrongBeliefShapeError) as error:
            GaussianBelief(
                variables=[Continuous("x"), Continuous("y")],
                mean=np.zeros(2),
                covariance=np.eye(3),
            )
        assert error.value.array == BeliefArray.COVARIANCE
        assert error.value.expected_shape == (2, 2)
        assert error.value.actual_shape == (3, 3)


# %% reading one quantity out of a belief


class TestReadingOneQuantity:
    """
    A belief about several quantities answers about each of them by name, so a caller
    never has to know which row a quantity sits in.
    """

    def build_belief(self) -> GaussianBelief:
        """
        :return: A belief whose two variables have distinct means and variances.
        """
        return GaussianBelief(
            variables=[Continuous("base_x"), Continuous("base_yaw")],
            mean=np.array([1.5, -0.25]),
            covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
        )

    def test_the_estimate_of_a_named_quantity_is_its_own_row(self):
        belief = self.build_belief()
        assert belief.mean_of(Continuous("base_yaw")) == -0.25

    def test_the_uncertainty_of_a_named_quantity_is_its_own_diagonal_entry(self):
        belief = self.build_belief()
        assert belief.variance_of(Continuous("base_yaw")) == 0.09

    def test_a_quantity_the_belief_is_not_about_is_rejected(self):
        belief = self.build_belief()
        unestimated = Continuous("gripper_opening")
        with pytest.raises(VariableNotInBeliefError) as error:
            belief.mean_of(unestimated)
        assert error.value.variable == unestimated
        assert error.value.belief_variables == belief.variables


# %% carrying a belief to the next control cycle


class TestPredict:
    """
    Prediction moves the estimate the way the quantity is expected to change on its own,
    and admits that doing so makes it less certain.
    """

    def test_a_quantity_expected_not_to_change_keeps_its_estimate(self):
        belief = GaussianBelief.of_one_variable(
            Continuous("grasp"), mean=0.8, variance=0.01
        )
        belief.predict(transition=np.eye(1), process_noise=np.zeros((1, 1)))
        assert belief.mean.tolist() == [0.8]

    def test_the_process_noise_is_added_to_the_uncertainty(self):
        belief = GaussianBelief.of_one_variable(
            Continuous("grasp"), mean=0.8, variance=0.01
        )
        belief.predict(transition=np.eye(1), process_noise=np.full((1, 1), 0.02))
        assert belief.variance_of(Continuous("grasp")) == pytest.approx(0.03)

    def test_the_transition_scales_the_uncertainty_by_its_square(self):
        belief = GaussianBelief.of_one_variable(Continuous("x"), mean=1.0, variance=1.0)
        belief.predict(transition=np.full((1, 1), 2.0), process_noise=np.zeros((1, 1)))
        assert belief.mean_of(Continuous("x")) == 2.0
        assert belief.variance_of(Continuous("x")) == 4.0

    def test_an_offset_moves_the_estimate_without_adding_uncertainty(self):
        """
        Decaying toward a prior is affine rather than linear, so the offset is what lets
        a belief fall back to what is known when nothing is observing it.
        """
        grasp = Continuous("grasp")
        belief = GaussianBelief.of_one_variable(grasp, mean=1.0, variance=0.04)
        decay = 0.75
        prior = 0.5
        belief.predict(
            transition=np.full((1, 1), decay),
            process_noise=np.zeros((1, 1)),
            offset=np.array([(1 - decay) * prior]),
        )
        assert belief.mean_of(grasp) == pytest.approx(decay * 1.0 + (1 - decay) * prior)
        assert belief.variance_of(grasp) == pytest.approx(decay**2 * 0.04)

    def test_a_transition_that_does_not_fit_the_belief_is_rejected(self):
        belief = GaussianBelief.of_one_variable(Continuous("x"), mean=0.0, variance=1.0)
        with pytest.raises(WrongBeliefShapeError) as error:
            belief.predict(transition=np.eye(2), process_noise=np.zeros((1, 1)))
        assert error.value.array == BeliefArray.TRANSITION
        assert error.value.expected_shape == (1, 1)

    def test_an_offset_that_does_not_fit_the_belief_is_rejected(self):
        belief = GaussianBelief.of_one_variable(Continuous("x"), mean=0.0, variance=1.0)
        with pytest.raises(WrongBeliefShapeError) as error:
            belief.predict(
                transition=np.eye(1),
                process_noise=np.zeros((1, 1)),
                offset=np.zeros(2),
            )
        assert error.value.array == BeliefArray.OFFSET


# %% correcting a belief with a reading


class TestUpdate:
    """
    An update weighs a reading against the estimate by how uncertain each of them is.
    """

    def test_a_reading_as_uncertain_as_the_estimate_lands_halfway_between_them(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        belief.update(belief.measurement_of(quantity, value=1.0, variance=1.0))
        assert belief.mean_of(quantity) == pytest.approx(0.5)
        assert belief.variance_of(quantity) == pytest.approx(0.5)

    def test_a_reading_the_sensor_is_sure_of_almost_replaces_the_estimate(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        belief.update(belief.measurement_of(quantity, value=1.0, variance=1e-6))
        assert belief.mean_of(quantity) == pytest.approx(1.0, abs=1e-5)

    def test_a_reading_the_sensor_is_unsure_of_barely_moves_the_estimate(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        belief.update(belief.measurement_of(quantity, value=1.0, variance=1e6))
        assert belief.mean_of(quantity) == pytest.approx(0.0, abs=1e-5)

    def test_a_reading_never_makes_the_estimate_less_certain(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        variance_before = belief.variance_of(quantity)
        belief.update(belief.measurement_of(quantity, value=3.0, variance=4.0))
        assert belief.variance_of(quantity) < variance_before

    def build_correlated_belief(self) -> GaussianBelief:
        """
        :return: A belief about two quantities whose errors are related, so reading one
            of them says something about the other.
        """
        return GaussianBelief(
            variables=[Continuous("a"), Continuous("b")],
            mean=np.zeros(2),
            covariance=np.array([[2.0, 1.0], [1.0, 2.0]]),
        )

    def test_reading_one_quantity_moves_a_related_one(self):
        belief = self.build_correlated_belief()
        belief.update(belief.measurement_of(Continuous("a"), value=4.0, variance=2.0))
        assert belief.mean_of(Continuous("a")) == pytest.approx(2.0)
        assert belief.mean_of(Continuous("b")) == pytest.approx(1.0)

    def test_reading_one_quantity_also_sharpens_a_related_one(self):
        belief = self.build_correlated_belief()
        belief.update(belief.measurement_of(Continuous("a"), value=4.0, variance=2.0))
        assert belief.variance_of(Continuous("a")) == pytest.approx(1.0)
        assert belief.variance_of(Continuous("b")) == pytest.approx(1.75)

    def test_reading_one_quantity_leaves_an_unrelated_one_alone(self):
        belief = GaussianBelief(
            variables=[Continuous("a"), Continuous("b")],
            mean=np.array([0.0, 5.0]),
            covariance=np.diag([2.0, 3.0]),
        )
        belief.update(belief.measurement_of(Continuous("a"), value=4.0, variance=2.0))
        assert belief.mean_of(Continuous("b")) == pytest.approx(5.0)
        assert belief.variance_of(Continuous("b")) == pytest.approx(3.0)

    def test_a_reading_of_several_quantities_at_once_corrects_all_of_them(self):
        """
        A sensor need not read a quantity directly: the measurement model is what lets a
        reading of, here, the sum of two quantities say something about each.
        """
        belief = GaussianBelief(
            variables=[Continuous("a"), Continuous("b")],
            mean=np.zeros(2),
            covariance=np.eye(2),
        )
        belief.update(
            Measurement(
                value=np.array([2.0]),
                model=np.array([[1.0, 1.0]]),
                noise=np.eye(1),
            )
        )
        assert belief.mean_of(Continuous("a")) == pytest.approx(2.0 / 3.0)
        assert belief.mean_of(Continuous("b")) == pytest.approx(2.0 / 3.0)

    def test_repeated_readings_accumulate_into_the_uncertainty(self):
        """
        Every reading adds its own precision to the estimate's, so after many cycles the
        uncertainty is the one the precisions add up to.

        Checked against the sum of precisions rather than against a stored number, since
        that is the statement the recursion has to keep being equal to.
        """
        belief = self.build_correlated_belief()
        prior_covariance = belief.covariance.copy()
        noise_variance = 2.0
        reading_of_a = np.array([[1.0, 0.0]])
        corrections = 100
        for _ in range(corrections):
            belief.update(
                belief.measurement_of(
                    Continuous("a"), value=4.0, variance=noise_variance
                )
            )
        accumulated_precision = np.linalg.inv(prior_covariance) + corrections * (
            reading_of_a.T @ reading_of_a / noise_variance
        )
        assert belief.covariance == pytest.approx(np.linalg.inv(accumulated_precision))

    def test_the_uncertainty_stays_symmetric_over_many_corrections(self):
        """
        A covariance that drifts out of symmetry is no longer one, and at a control
        cycle's rate the drift compounds long before anything looks wrong.

        Rounding keeps it from being symmetric to the last bit, so the tolerance is that
        of the arithmetic rather than of the result.
        """
        belief = self.build_correlated_belief()
        for _ in range(100):
            belief.update(
                belief.measurement_of(Continuous("a"), value=4.0, variance=2.0)
            )
        assert belief.covariance == pytest.approx(belief.covariance.T, rel=1e-14)

    def test_the_uncertainty_stays_a_covariance_over_many_corrections(self):
        """
        No quantity, nor any combination of them, may end up with a negative variance.
        """
        belief = self.build_correlated_belief()
        for _ in range(100):
            belief.update(
                belief.measurement_of(Continuous("a"), value=4.0, variance=2.0)
            )
        assert min(np.linalg.eigvalsh(belief.covariance)) >= 0

    def test_a_reading_whose_noise_does_not_match_its_values_is_rejected(self):
        with pytest.raises(WrongBeliefShapeError) as error:
            Measurement(
                value=np.array([1.0, 2.0]),
                model=np.eye(2),
                noise=np.eye(1),
            )
        assert error.value.array == BeliefArray.MEASUREMENT_NOISE
        assert error.value.expected_shape == (2, 2)

    def test_a_reading_of_a_differently_sized_belief_is_rejected(self):
        belief = GaussianBelief.of_one_variable(Continuous("x"), mean=0.0, variance=1.0)
        with pytest.raises(WrongBeliefShapeError) as error:
            belief.update(
                Measurement(
                    value=np.array([1.0]),
                    model=np.array([[1.0, 1.0]]),
                    noise=np.eye(1),
                )
            )
        assert error.value.array == BeliefArray.MEASUREMENT_MODEL
        assert error.value.expected_shape == (1, 1)
        assert error.value.actual_shape == (1, 2)


# %% the beliefs a statechart carries across its cycles


class TestBeliefContext:
    """
    The context is what makes a belief outlive the cycle that produced it.
    """

    def test_a_belief_is_found_by_the_quantity_it_is_about(self):
        grasp = Continuous("grasp")
        belief = GaussianBelief.of_one_variable(grasp, mean=0.5, variance=0.1)
        context = BeliefContext()
        context.add(belief)
        assert context.require(grasp) is belief

    def test_a_belief_about_several_quantities_is_found_by_each_of_them(self):
        belief = GaussianBelief(
            variables=[Continuous("base_x"), Continuous("base_yaw")],
            mean=np.zeros(2),
            covariance=np.eye(2),
        )
        context = BeliefContext()
        context.add(belief)
        assert context.require(Continuous("base_x")) is belief
        assert context.require(Continuous("base_yaw")) is belief

    def test_a_quantity_nothing_estimates_is_rejected(self):
        context = BeliefContext()
        unestimated = Continuous("grasp")
        with pytest.raises(UnknownBeliefError) as error:
            context.require(unestimated)
        assert error.value.variable == unestimated

    def test_a_second_belief_about_the_same_quantity_is_rejected(self):
        grasp = Continuous("grasp")
        context = BeliefContext()
        context.add(GaussianBelief.of_one_variable(grasp, mean=0.5, variance=0.1))
        with pytest.raises(DuplicateBeliefError) as error:
            context.add(GaussianBelief.of_one_variable(grasp, mean=0.9, variance=0.2))
        assert error.value.variable == grasp

    def test_a_rejected_belief_leaves_the_quantities_it_shares_untouched(self):
        shared = Continuous("base_x")
        first = GaussianBelief.of_one_variable(shared, mean=1.0, variance=0.1)
        context = BeliefContext()
        context.add(first)
        with pytest.raises(DuplicateBeliefError):
            context.add(
                GaussianBelief(
                    variables=[Continuous("base_yaw"), shared],
                    mean=np.zeros(2),
                    covariance=np.eye(2),
                )
            )
        assert context.require(shared) is first
        with pytest.raises(UnknownBeliefError):
            context.require(Continuous("base_yaw"))

    def test_a_correction_is_visible_to_whoever_reads_the_belief_next(self):
        """
        The cycle that corrects a belief and the one that reads it are different ticks,
        so the correction has to reach the context rather than a copy of the belief.
        """
        grasp = Continuous("grasp")
        context = BeliefContext()
        context.add(GaussianBelief.of_one_variable(grasp, mean=0.0, variance=1.0))
        estimating = context.require(grasp)
        estimating.update(estimating.measurement_of(grasp, value=1.0, variance=1.0))
        assert context.require(grasp).mean_of(grasp) == pytest.approx(0.5)

    def test_it_is_reachable_through_the_motion_statechart_context(self):
        """
        A node only ever sees the statechart's context, so the beliefs have to arrive as
        one of its extensions.
        """
        beliefs = BeliefContext()
        statechart_context = MotionStatechartContext.empty()
        statechart_context.add_extension(beliefs)
        assert statechart_context.require_extension(BeliefContext) is beliefs
