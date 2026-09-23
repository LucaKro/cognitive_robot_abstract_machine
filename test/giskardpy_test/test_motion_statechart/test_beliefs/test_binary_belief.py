import pytest

from giskardpy.motion_statechart.beliefs.belief import Statistic, VariableStatistic
from giskardpy.motion_statechart.beliefs.binary import (
    BinaryBelief,
    BinaryEvidence,
    BinaryTransition,
)
from giskardpy.motion_statechart.exceptions import (
    ImpossibleEvidenceError,
    NegativeLikelihoodError,
    ProbabilityOutOfRangeError,
)

# %% prediction


def test_prediction_mixes_persisting_and_arising():
    prior, persists, arises = 0.8, 0.9, 0.1
    belief = BinaryBelief.about("coupled", probability=prior)

    belief.predict(BinaryTransition(persists=persists, arises=arises))

    assert belief.probability == pytest.approx(prior * persists + (1 - prior) * arises)


def test_transition_rejects_a_probability_outside_the_unit_interval():
    with pytest.raises(ProbabilityOutOfRangeError):
        BinaryTransition(persists=1.2, arises=0.0)


# %% update


def test_update_follows_bayes_rule():
    prior = 0.5
    evidence = BinaryEvidence(likelihood_if_holds=0.9, likelihood_if_not=0.3)
    belief = BinaryBelief.about("coupled", probability=prior)

    belief.update([evidence])

    supporting = prior * evidence.likelihood_if_holds
    opposing = (1 - prior) * evidence.likelihood_if_not
    assert belief.probability == pytest.approx(supporting / (supporting + opposing))


def test_update_with_several_pieces_of_evidence_applies_each_of_them():
    first = BinaryEvidence(likelihood_if_holds=0.9, likelihood_if_not=0.3)
    second = BinaryEvidence(likelihood_if_holds=0.2, likelihood_if_not=0.6)
    combined = BinaryBelief.about("coupled", probability=0.5)
    one_at_a_time = BinaryBelief.about("coupled", probability=0.5)

    combined.update([first, second])
    one_at_a_time.update([first])
    one_at_a_time.update([second])

    assert combined.probability == pytest.approx(one_at_a_time.probability)


def test_evidence_impossible_under_the_belief_is_rejected():
    belief = BinaryBelief.about("coupled", probability=1.0)

    with pytest.raises(ImpossibleEvidenceError):
        belief.update([BinaryEvidence(likelihood_if_holds=0.0, likelihood_if_not=0.5)])


def test_rejected_evidence_leaves_the_belief_unchanged():
    prior = 0.5
    belief = BinaryBelief.about("coupled", probability=prior)

    with pytest.raises(ImpossibleEvidenceError):
        belief.update(
            [
                BinaryEvidence(likelihood_if_holds=0.9, likelihood_if_not=0.3),
                BinaryEvidence(likelihood_if_holds=0.0, likelihood_if_not=0.0),
            ]
        )

    assert belief.probability == prior


def test_evidence_rejects_a_negative_likelihood():
    with pytest.raises(NegativeLikelihoodError):
        BinaryEvidence(likelihood_if_holds=-0.1, likelihood_if_not=0.5)


# %% construction and statistics


def test_belief_rejects_a_probability_outside_the_unit_interval():
    with pytest.raises(ProbabilityOutOfRangeError):
        BinaryBelief.about("coupled", probability=-0.1)


def test_belief_about_a_state_is_over_one_variable_of_that_name():
    name = "coupled"
    belief = BinaryBelief.about(name, probability=0.5)

    assert [variable.name for variable in belief.variables] == [name]


def test_statistic_is_the_probability_that_the_state_holds():
    belief = BinaryBelief.about("coupled", probability=0.3)

    assert belief.statistics() == {
        VariableStatistic(belief.variable, Statistic.PROBABILITY): belief.probability
    }
