import numpy as np
import pytest
from probabilistic_model.exceptions import VariableNotInDistributionError
from random_events.variable import Continuous

from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.gaussian import GaussianBelief, Reading
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    UnknownBeliefError,
)

# %% building a belief


class TestBuildingABelief:
    def test_a_belief_about_one_quantity_starts_where_it_was_told_to(self):
        grasp = Continuous("grasp")
        belief = GaussianBelief.of_one_variable(grasp, mean=0.4, variance=0.25)
        assert belief.mean_of(grasp) == 0.4
        assert belief.variance_of(grasp) == 0.25

    def test_a_quantity_left_out_starts_at_no_estimate_and_no_uncertainty(self):
        first, second = Continuous("a"), Continuous("b")
        belief = GaussianBelief.of(
            variables=(first, second),
            estimates={first: 1.5},
            uncertainty={(first, first): 0.04},
        )
        assert belief.mean_of(second) == 0.0
        assert belief.variance_of(second) == 0.0

    def test_the_uncertainty_shared_by_two_quantities_is_stated_once(self):
        first, second = Continuous("a"), Continuous("b")
        belief = GaussianBelief.of(
            variables=(first, second),
            estimates={},
            uncertainty={
                (first, first): 2.0,
                (second, second): 2.0,
                (first, second): 1.0,
            },
        )
        assert belief.covariance_between(first, second) == 1.0
        assert belief.covariance_between(second, first) == 1.0

    def test_a_quantity_the_belief_is_not_about_is_rejected(self):
        belief = GaussianBelief.of_one_variable(
            Continuous("base_x"), mean=1.5, variance=0.04
        )
        unestimated = Continuous("gripper_opening")
        with pytest.raises(VariableNotInDistributionError) as error:
            belief.mean_of(unestimated)
        assert error.value.variable == unestimated


# %% carrying a belief to the next control cycle


class TestPredict:
    """
    Prediction moves the estimate the way the quantity is expected to change on its own,
    and admits that doing so makes it less certain.
    """

    def test_a_quantity_expected_not_to_change_keeps_its_estimate(self):
        grasp = Continuous("grasp")
        belief = GaussianBelief.of_one_variable(grasp, mean=0.8, variance=0.01)
        belief.predict(transition=belief.unchanged, process_noise={})
        assert belief.mean_of(grasp) == 0.8

    def test_the_process_noise_is_added_to_the_uncertainty(self):
        grasp = Continuous("grasp")
        belief = GaussianBelief.of_one_variable(grasp, mean=0.8, variance=0.01)
        belief.predict(
            transition=belief.unchanged,
            process_noise={(grasp, grasp): 0.02},
        )
        assert belief.variance_of(grasp) == pytest.approx(0.03)

    def test_the_transition_scales_the_uncertainty_by_its_square(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=1.0, variance=1.0)
        belief.predict(transition={(quantity, quantity): 2.0}, process_noise={})
        assert belief.mean_of(quantity) == 2.0
        assert belief.variance_of(quantity) == 4.0

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
            transition={(grasp, grasp): decay},
            process_noise={},
            offset={grasp: (1 - decay) * prior},
        )
        assert belief.mean_of(grasp) == pytest.approx(decay * 1.0 + (1 - decay) * prior)
        assert belief.variance_of(grasp) == pytest.approx(decay**2 * 0.04)

    def test_the_uncertainty_stays_symmetric_on_every_cycle(self):
        """
        Carrying a covariance through a transition is symmetric in exact arithmetic and
        drifts out of it under rounding, most visibly between quantities held on very
        different scales.

        A covariance that is not symmetric is no longer one, so this has to hold on
        every cycle rather than by the end of a run.
        """
        first, second = Continuous("a"), Continuous("b")
        belief = GaussianBelief.of(
            variables=(first, second),
            estimates={},
            uncertainty={
                (first, first): 0.001,
                (second, second): 1000.0,
                (first, second): 0.1,
            },
        )
        transition = {
            (first, first): 0.1,
            (first, second): 0.1,
            (second, first): 0.1,
            (second, second): 0.2,
        }
        for _ in range(20):
            belief.predict(transition=transition, process_noise={})
            covariance = belief.distribution.covariance
            assert covariance.tolist() == covariance.T.tolist()

    def test_a_quantity_the_belief_is_not_about_is_rejected(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        stranger = Continuous("y")
        with pytest.raises(VariableNotInDistributionError) as error:
            belief.predict(transition={(stranger, stranger): 1.0}, process_noise={})
        assert error.value.variable == stranger


# %% correcting a belief with a reading


class TestUpdate:
    """
    An update weighs a reading against the estimate by how uncertain each of them is.
    """

    def test_a_reading_as_uncertain_as_the_estimate_lands_halfway_between_them(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        belief.update([Reading.of_one_variable(quantity, value=1.0, variance=1.0)])
        assert belief.mean_of(quantity) == pytest.approx(0.5)
        assert belief.variance_of(quantity) == pytest.approx(0.5)

    def test_a_reading_the_sensor_is_sure_of_almost_replaces_the_estimate(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        belief.update([Reading.of_one_variable(quantity, value=1.0, variance=1e-6)])
        assert belief.mean_of(quantity) == pytest.approx(1.0, abs=1e-5)

    def test_a_reading_the_sensor_is_unsure_of_barely_moves_the_estimate(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        belief.update([Reading.of_one_variable(quantity, value=1.0, variance=1e6)])
        assert belief.mean_of(quantity) == pytest.approx(0.0, abs=1e-5)

    def test_a_reading_never_makes_the_estimate_less_certain(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.0, variance=1.0)
        variance_before = belief.variance_of(quantity)
        belief.update([Reading.of_one_variable(quantity, value=3.0, variance=4.0)])
        assert belief.variance_of(quantity) < variance_before

    def test_reporting_nothing_leaves_the_estimate_alone(self):
        quantity = Continuous("x")
        belief = GaussianBelief.of_one_variable(quantity, mean=0.3, variance=1.0)
        belief.update([])
        assert belief.mean_of(quantity) == 0.3
        assert belief.variance_of(quantity) == 1.0

    def build_correlated_belief(self) -> GaussianBelief:
        """
        :return: A belief about two quantities whose errors are related, so reading one
            of them says something about the other.
        """
        first, second = Continuous("a"), Continuous("b")
        return GaussianBelief.of(
            variables=(first, second),
            estimates={},
            uncertainty={
                (first, first): 2.0,
                (second, second): 2.0,
                (first, second): 1.0,
            },
        )

    def test_reading_one_quantity_moves_a_related_one(self):
        belief = self.build_correlated_belief()
        belief.update(
            [Reading.of_one_variable(Continuous("a"), value=4.0, variance=2.0)]
        )
        assert belief.mean_of(Continuous("a")) == pytest.approx(2.0)
        assert belief.mean_of(Continuous("b")) == pytest.approx(1.0)

    def test_reading_one_quantity_also_sharpens_a_related_one(self):
        belief = self.build_correlated_belief()
        belief.update(
            [Reading.of_one_variable(Continuous("a"), value=4.0, variance=2.0)]
        )
        assert belief.variance_of(Continuous("a")) == pytest.approx(1.0)
        assert belief.variance_of(Continuous("b")) == pytest.approx(1.75)

    def test_reading_one_quantity_leaves_an_unrelated_one_alone(self):
        first, second = Continuous("a"), Continuous("b")
        belief = GaussianBelief.of(
            variables=(first, second),
            estimates={second: 5.0},
            uncertainty={(first, first): 2.0, (second, second): 3.0},
        )
        belief.update([Reading.of_one_variable(first, value=4.0, variance=2.0)])
        assert belief.mean_of(second) == pytest.approx(5.0)
        assert belief.variance_of(second) == pytest.approx(3.0)

    def test_a_reading_of_several_quantities_at_once_corrects_all_of_them(self):
        """
        A sensor need not read a quantity itself: what it contributes to is what lets a
        reading of, here, the sum of two quantities say something about each.
        """
        first, second = Continuous("a"), Continuous("b")
        belief = GaussianBelief.of(
            variables=(first, second),
            estimates={},
            uncertainty={(first, first): 1.0, (second, second): 1.0},
        )
        belief.update(
            [Reading(value=2.0, contributions={first: 1.0, second: 1.0}, variance=1.0)]
        )
        assert belief.mean_of(first) == pytest.approx(2.0 / 3.0)
        assert belief.mean_of(second) == pytest.approx(2.0 / 3.0)

    def test_repeated_readings_accumulate_into_the_uncertainty(self):
        """
        Every reading adds its own precision to the estimate's, so after many cycles the
        uncertainty is the one the precisions add up to.

        Checked against the sum of precisions rather than against a stored number, since
        that is the statement the recursion has to keep being equal to.
        """
        belief = self.build_correlated_belief()
        prior_covariance = belief.distribution.covariance.copy()
        noise_variance = 2.0
        reading_of_a = np.array([[1.0, 0.0]])
        corrections = 100
        for _ in range(corrections):
            belief.update(
                [
                    Reading.of_one_variable(
                        Continuous("a"), value=4.0, variance=noise_variance
                    )
                ]
            )
        accumulated_precision = np.linalg.inv(prior_covariance) + corrections * (
            reading_of_a.T @ reading_of_a / noise_variance
        )
        assert belief.distribution.covariance == pytest.approx(
            np.linalg.inv(accumulated_precision)
        )

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
                [Reading.of_one_variable(Continuous("a"), value=4.0, variance=2.0)]
            )
        assert belief.distribution.covariance == pytest.approx(
            belief.distribution.covariance.T, rel=1e-14
        )

    def test_the_uncertainty_stays_a_covariance_over_many_corrections(self):
        """
        No quantity, nor any combination of them, may end up with a negative variance.
        """
        belief = self.build_correlated_belief()
        for _ in range(100):
            belief.update(
                [Reading.of_one_variable(Continuous("a"), value=4.0, variance=2.0)]
            )
        assert min(np.linalg.eigvalsh(belief.distribution.covariance)) >= 0

    def test_a_reading_of_a_quantity_the_belief_is_not_about_is_rejected(self):
        belief = GaussianBelief.of_one_variable(Continuous("x"), mean=0.0, variance=1.0)
        stranger = Continuous("y")
        with pytest.raises(VariableNotInDistributionError) as error:
            belief.update([Reading.of_one_variable(stranger, value=1.0, variance=1.0)])
        assert error.value.variable == stranger


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
        first, second = Continuous("base_x"), Continuous("base_yaw")
        belief = GaussianBelief.of(
            variables=(first, second),
            estimates={},
            uncertainty={(first, first): 1.0, (second, second): 1.0},
        )
        context = BeliefContext()
        context.add(belief)
        assert context.require(first) is belief
        assert context.require(second) is belief

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
        other = Continuous("base_yaw")
        first = GaussianBelief.of_one_variable(shared, mean=1.0, variance=0.1)
        context = BeliefContext()
        context.add(first)
        with pytest.raises(DuplicateBeliefError):
            context.add(
                GaussianBelief.of(
                    variables=(other, shared),
                    estimates={},
                    uncertainty={(other, other): 1.0, (shared, shared): 1.0},
                )
            )
        assert context.require(shared) is first
        with pytest.raises(UnknownBeliefError):
            context.require(other)

    def test_a_correction_is_visible_to_whoever_reads_the_belief_next(self):
        """
        The cycle that corrects a belief and the one that reads it are different ticks,
        so the correction has to reach the context rather than a copy of the belief.
        """
        grasp = Continuous("grasp")
        context = BeliefContext()
        context.add(GaussianBelief.of_one_variable(grasp, mean=0.0, variance=1.0))
        context.require(grasp).update(
            [Reading.of_one_variable(grasp, value=1.0, variance=1.0)]
        )
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
