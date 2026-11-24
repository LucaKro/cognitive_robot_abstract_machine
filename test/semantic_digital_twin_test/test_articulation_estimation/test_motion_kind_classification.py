"""
Tests of giving the probability of each kind of motion with a classifier: the summary of
the observed poses it sees, the circuit that classifies them, and fitting only the kinds
it finds likely.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from random_events.variable import Continuous
from scipy.spatial.transform import RigidTransform, Rotation
from sortedcontainers import SortedSet

from semantic_digital_twin.articulation_estimation.exceptions import (
    KindNotFittedError,
)
from semantic_digital_twin.articulation_estimation.motion_kind_classification import (
    MotionKindCircuit,
    MotionSummary,
    MotionSummaryClassifier,
    ProbabilityCalibration,
)
from semantic_digital_twin.articulation_estimation.motion_kinds import (
    MotionKind,
    MotionKindClassifier,
    MotionKindMixture,
    MotionKindPriors,
    MotionKindProbabilities,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)

from .test_articulation_estimation import screw_motions
from .test_motion_kinds import noise


# %% Classifiers the tests use
@dataclass(kw_only=True)
class FixedKindProbabilities(MotionKindClassifier):
    """
    A classifier that gives the same probabilities whatever it observes.
    """

    probabilities: MotionKindProbabilities
    """
    The probabilities it gives.
    """

    training_share: MotionKindProbabilities = field(
        default_factory=lambda: MotionKindPriors().kind_probabilities
    )
    """
    The share of each kind it claims to have learned from; the default priors, so that
    the mixture leaves its probabilities as they are.
    """

    @property
    def training_probabilities(self) -> MotionKindProbabilities:
        return self.training_share

    def kind_probabilities(
        self, *, observations: PoseObservations, noise: ObservationNoise
    ) -> MotionKindProbabilities:
        return self.probabilities


@pytest.fixture
def untrained_circuit() -> MotionKindCircuit:
    """
    :return: A classifier over an untrained circuit of the summary's numbers and the
        kind, with the numbers left as they are and no calibration.
    """
    variables = SortedSet([Continuous(name) for name in MotionSummary.field_names])
    circuit = (
        RegionGraph(variables=variables, partitions=2, depth=2, repetitions=2, seed=0)
        .create_random_region_graph()
        .to_probabilistic_circuit(
            input_units=3,
            sum_units=3,
            predicted_variable=MotionKindProbabilities.kind_variable,
        )
    )
    field_count = len(MotionSummary.field_names)
    return MotionKindCircuit(
        circuit=circuit,
        summary_mean=np.zeros(field_count),
        summary_standard_deviation=np.ones(field_count),
        calibration=ProbabilityCalibration(),
    )


# %% The summary
def test_summary_does_not_depend_on_the_parent_frame():
    """
    Moving the parent frame, which moves every observed pose alike, leaves the summary
    as it is.
    """
    observations = screw_motions()["helical"].observe(noise=noise, seed=4)
    moved_parent = RigidTransform.from_components(
        [0.3, -1.2, 0.7], Rotation.from_rotvec([0.4, -0.2, 1.1])
    ).as_matrix()
    moved = PoseObservations(
        parent_T_child_stack=moved_parent @ observations.parent_T_child_stack
    )

    assert_allclose(
        MotionSummary.from_observations(observations=moved, noise=noise).values,
        MotionSummary.from_observations(observations=observations, noise=noise).values,
        atol=1e-9,
    )


def test_summary_of_a_revolute_motion_holds_its_turned_angle():
    """
    For a revolute motion observed almost without noise, the summary holds the angle it
    turned.
    """
    motion = screw_motions()["revolute"]
    almost_no_noise = ObservationNoise(
        position_standard_deviation=1e-9, orientation_standard_deviation=1e-9
    )
    summary = MotionSummary.from_observations(
        observations=motion.observe(noise=almost_no_noise, seed=4),
        noise=almost_no_noise,
    )

    assert summary.log_turned_angle == pytest.approx(
        np.log(motion.positions[-1] + MotionSummary.smallest_turned_angle), rel=1e-6
    )


# %% Calibration
def test_a_large_temperature_makes_the_kinds_equally_probable():
    """
    Dividing the log probabilities by a large temperature leaves every kind with a
    third.
    """
    calibrated = ProbabilityCalibration(temperature=1e9).calibrated(
        log_probabilities=np.log([0.98, 0.015, 0.005])
    )

    for kind in MotionKind:
        assert calibrated.probability_of(kind=kind) == pytest.approx(1 / 3)


# %% The circuit
def test_circuit_gives_probabilities_that_add_up_to_one(
    untrained_circuit: MotionKindCircuit,
):
    """
    The circuit gives every kind a probability, and they add up to one.
    """
    probabilities = untrained_circuit.kind_probabilities(
        observations=screw_motions()["revolute"].observe(noise=noise, seed=4),
        noise=noise,
    )

    assert sum(
        probabilities.probability_of(kind=kind) for kind in MotionKind
    ) == pytest.approx(1)


def test_circuit_read_from_its_file_gives_the_same_probabilities(
    untrained_circuit: MotionKindCircuit, tmp_path: Path
):
    """
    A circuit written to a file and read back as any summary classifier is a circuit
    again, and gives the same probabilities.
    """
    observations = screw_motions()["prismatic"].observe(noise=noise, seed=4)
    path = tmp_path / "motion_kind_circuit.json"
    untrained_circuit.to_file(path=path)
    read = MotionSummaryClassifier.from_file(path=path)

    assert isinstance(read, MotionKindCircuit)
    assert read.kind_probabilities(
        observations=observations, noise=noise
    ) == untrained_circuit.kind_probabilities(observations=observations, noise=noise)


# %% Fitting only the likely kinds
def test_only_the_likely_kinds_are_fitted():
    """
    With a classifier, the mixture takes its probabilities and fits only the kinds it
    finds at least as probable as the minimum.
    """
    probabilities = MotionKindProbabilities(
        revolute=0.004, helical=0.001, sliding=0.995
    )
    mixture = MotionKindMixture.fit(
        observations=screw_motions()["prismatic"].observe(noise=noise, seed=4),
        noise=noise,
        classifier=FixedKindProbabilities(probabilities=probabilities),
    )

    assert mixture.kind_probabilities == probabilities
    assert [fit.chart.kind for fit in mixture.fits] == [MotionKind.SLIDING]
    assert mixture.most_probable.chart.kind == MotionKind.SLIDING


def test_asking_for_a_kind_that_was_not_fitted_raises():
    """
    The fit of a kind the classifier ruled out cannot be asked for.
    """
    mixture = MotionKindMixture.fit(
        observations=screw_motions()["prismatic"].observe(noise=noise, seed=4),
        noise=noise,
        classifier=FixedKindProbabilities(
            probabilities=MotionKindProbabilities(
                revolute=0.004, helical=0.001, sliding=0.995
            )
        ),
    )

    with pytest.raises(KindNotFittedError):
        mixture.fit_of(kind=MotionKind.REVOLUTE)


# %% Priors
def test_the_default_priors_take_an_ambiguous_turning_motion_as_revolute():
    """
    A turning motion a classifier trained on equally many motions of each kind cannot
    tell apart is taken as revolute under the default priors, which make a helical
    motion rare.
    """
    classifier = FixedKindProbabilities(
        probabilities=MotionKindProbabilities(revolute=0.52, helical=0.48, sliding=0.0),
        training_share=MotionKindProbabilities(
            revolute=1 / 3, helical=1 / 3, sliding=1 / 3
        ),
    )
    mixture = MotionKindMixture.fit(
        observations=screw_motions()["revolute"].observe(noise=noise, seed=4),
        noise=noise,
        classifier=classifier,
    )

    assert mixture.kind_probabilities.revolute == pytest.approx(
        0.52 * 0.49 / (0.52 * 0.49 + 0.48 * 0.02)
    )
    assert [fit.chart.kind for fit in mixture.fits] == [MotionKind.REVOLUTE]


def test_priors_that_expect_a_screw_take_the_same_motion_as_helical():
    """
    Priors that expect a helical motion, such as those of a lid, turn the same
    classifier's answer into helical.
    """
    classifier = FixedKindProbabilities(
        probabilities=MotionKindProbabilities(revolute=0.52, helical=0.48, sliding=0.0),
        training_share=MotionKindProbabilities(
            revolute=1 / 3, helical=1 / 3, sliding=1 / 3
        ),
    )
    mixture = MotionKindMixture.fit(
        observations=screw_motions()["helical"].observe(noise=noise, seed=4),
        noise=noise,
        priors=MotionKindPriors(
            revolute_probability=0.1, helical_probability=0.8, sliding_probability=0.1
        ),
        classifier=classifier,
    )

    assert mixture.most_probable.chart.kind == MotionKind.HELICAL


def test_swapping_a_prior_back_gives_the_probabilities_again():
    """
    Replacing a prior and then replacing the new one by the old one leaves the
    probabilities as they were.
    """
    probabilities = MotionKindProbabilities(revolute=0.2, helical=0.3, sliding=0.5)
    old = MotionKindProbabilities(revolute=1 / 3, helical=1 / 3, sliding=1 / 3)
    new = MotionKindProbabilities(revolute=0.6, helical=0.1, sliding=0.3)
    swapped_back = probabilities.with_prior(prior=new, replacing=old).with_prior(
        prior=old, replacing=new
    )

    for kind in MotionKind:
        assert swapped_back.probability_of(kind=kind) == pytest.approx(
            probabilities.probability_of(kind=kind)
        )
