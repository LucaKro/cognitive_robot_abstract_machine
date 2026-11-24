"""
Tests of telling revolute, helical and sliding motions apart by the evidence of a fit of
each.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from scipy.spatial.transform import RigidTransform, Rotation

from semantic_digital_twin.articulation_estimation.motion_kinds import (
    MotionKind,
    MotionKindMixture,
    MotionKindPriors,
    MotionKindProbabilities,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.world_entity import Body
from krrood_spatial_math.spatial_types import Vector3

from .test_articulation_estimation import ObservedScrewMotion, screw_motions

noise = ObservationNoise(
    position_standard_deviation=0.001, orientation_standard_deviation=0.005
)
"""
The noise of a good motion tracker: a millimeter and a quarter of a degree.
"""

kind_of_motion = {
    "revolute": MotionKind.REVOLUTE,
    "prismatic": MotionKind.SLIDING,
    "helical": MotionKind.HELICAL,
}
"""
The kind each of :func:`screw_motions` is.
"""


def door_opened_by(degrees: float) -> ObservedScrewMotion:
    """
    :return: A door hinged 0.4 m from its origin, observed while it opens by ``degrees``.
    """
    return ObservedScrewMotion(
        screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        positions=np.linspace(0.0, np.radians(degrees), 15),
    )


# %% Tests
@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_the_observed_kind_is_the_most_probable(motion_name: str):
    """
    A revolute, a helical and a sliding motion each come out as their own kind.
    """
    mixture = MotionKindMixture.fit(
        observations=screw_motions()[motion_name].observe(noise=noise, seed=4),
        noise=noise,
    )

    assert mixture.most_probable.chart.kind == kind_of_motion[motion_name]
    assert (
        mixture.kind_probabilities.probability_of(kind=kind_of_motion[motion_name])
        > 0.5
    )


def test_kind_distribution_holds_the_probabilities():
    """
    The distribution over the kinds gives each kind its probability, and the
    probabilities add up to one.
    """
    probabilities = MotionKindMixture.fit(
        observations=screw_motions()["revolute"].observe(noise=noise, seed=4),
        noise=noise,
    ).kind_probabilities
    distribution = probabilities.distribution
    variable = MotionKindProbabilities.kind_variable

    assert sum(
        probabilities.probability_of(kind=kind) for kind in MotionKind
    ) == pytest.approx(1)
    for kind in MotionKind:
        event = SimpleEvent.from_data({variable: Set.from_iterable([kind])})
        assert distribution.probability(event.as_composite_set()) == pytest.approx(
            probabilities.probability_of(kind=kind)
        )


def test_each_kind_has_its_own_chart():
    """
    The revolute chart has ten numbers, the helical one the pitch besides, and the
    sliding one neither the axis offsets nor the pitch; each fit's Gaussian is over its
    chart.
    """
    mixture = MotionKindMixture.fit(
        observations=screw_motions()["helical"].observe(noise=noise, seed=4),
        noise=noise,
    )

    sizes = {fit.chart.kind: len(fit.chart.names) for fit in mixture.fits}
    assert sizes == {
        MotionKind.REVOLUTE: 10,
        MotionKind.HELICAL: 11,
        MotionKind.SLIDING: 8,
    }
    for fit in mixture.fits:
        assert {variable.name for variable in fit.distribution.variables} == set(
            fit.chart.names
        )


def test_most_probable_fit_is_written_into_an_estimated_connection():
    """
    The most probable kind's fit, written into an estimated connection, puts the child
    where that fit predicts it for each observation.
    """
    mixture = MotionKindMixture.fit(
        observations=screw_motions()["revolute"].observe(noise=noise, seed=4),
        noise=noise,
    )
    fit = mixture.most_probable.screw_motion
    world = World()
    parent = Body(name=PrefixedName("parent"))
    child = Body(name=PrefixedName("child"))
    with world.modify_world():
        connection = fit.create_estimated_connection(
            world=world, parent=parent, child=child
        )
        world.add_connection(connection)

    for position, predicted in zip(fit.positions, fit.predicted_parent_T_child_stack):
        connection.position = position
        assert_allclose(
            world.compute_forward_kinematics_np(root=parent, tip=child),
            predicted,
            atol=1e-9,
        )


def test_a_turn_within_the_noise_is_called_sliding():
    """
    A door opened by a quarter of a degree turns less than the noise of one orientation,
    so the simpler sliding kind explains it as well and wins.
    """
    mixture = MotionKindMixture.fit(
        observations=door_opened_by(0.25).observe(noise=noise, seed=4),
        noise=noise,
    )

    assert mixture.most_probable.chart.kind == MotionKind.SLIDING


def test_the_axis_prior_decides_at_the_limit_of_detection():
    """
    For a door opened by one degree, expecting the axis closer to the child makes the
    turning kinds more probable.
    """
    observations = door_opened_by(1.0).observe(noise=noise, seed=4)

    def turning_weight(axis_distance_standard_deviation: float) -> float:
        """
        :return: The weight of the turning kinds under that prior on the axis distance.
        """
        probabilities = MotionKindMixture.fit(
            observations=observations,
            noise=noise,
            priors=MotionKindPriors(
                axis_distance_standard_deviation=axis_distance_standard_deviation
            ),
        ).kind_probabilities
        return probabilities.revolute + probabilities.helical

    assert turning_weight(0.5) > turning_weight(5.0)


# %% Observations no kind explains
def turned_then_slid() -> PoseObservations:
    """
    :return: The noisy poses of a child that turns by 90 degrees about an axis 0.3 m
        away, then slides 0.2 m, like a lid that is unscrewed and then lifted off.
    """
    rng = np.random.default_rng(4)
    turning = np.concatenate(
        [[0.0, 0.0, 1.0], -np.cross([0.0, 0.0, 1.0], [0.3, 0.0, 0.0])]
    )
    turned = RigidTransform.from_exp_coords(
        np.outer(np.linspace(0.0, np.pi / 2, 10), turning)
    ).as_matrix()
    slid = (
        RigidTransform.from_exp_coords(
            np.outer(np.linspace(0.0, 0.2, 11)[1:], [0, 0, 0, 1.0, 0, 0])
        ).as_matrix()
        @ turned[-1]
    )
    stack = np.concatenate([turned, slid])
    stack[:, :3, 3] += rng.normal(scale=noise.position_standard_deviation, size=(20, 3))
    stack[:, :3, :3] = (
        Rotation.from_matrix(stack[:, :3, :3])
        * Rotation.from_rotvec(
            rng.normal(scale=noise.orientation_standard_deviation, size=(20, 3))
        )
    ).as_matrix()
    return PoseObservations(parent_T_child_stack=stack)


@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_a_motion_of_a_kind_is_explained_within_its_noise(motion_name: str):
    """
    Observations of a motion of one kind stray from that kind's fit by about their
    noise, so the mixture explains them.
    """
    mixture = MotionKindMixture.fit(
        observations=screw_motions()[motion_name].observe(noise=noise, seed=4),
        noise=noise,
    )

    assert mixture.most_probable.deviation_in_noise_units == pytest.approx(1, abs=0.4)
    assert mixture.explains_observations


def test_noise_stated_twice_too_small_is_still_explained():
    """
    Noise stated half as large as it is doubles the deviation, which still counts as
    explained.
    """
    understated = ObservationNoise(
        position_standard_deviation=noise.position_standard_deviation / 2,
        orientation_standard_deviation=noise.orientation_standard_deviation / 2,
    )
    mixture = MotionKindMixture.fit(
        observations=screw_motions()["revolute"].observe(noise=noise, seed=4),
        noise=understated,
    )

    assert mixture.most_probable.deviation_in_noise_units == pytest.approx(2, abs=0.6)
    assert mixture.explains_observations


def test_a_motion_that_turns_then_slides_is_not_explained(
    caplog: pytest.LogCaptureFixture,
):
    """
    A motion that turns and then slides strays from the fit of every kind by many times
    its noise, so the mixture does not explain it and warns.
    """
    mixture = MotionKindMixture.fit(observations=turned_then_slid(), noise=noise)

    assert (
        mixture.most_probable.deviation_in_noise_units
        > 3 * MotionKindMixture.largest_explained_deviation
    )
    assert not mixture.explains_observations
    assert "likely of no kind" in caplog.text
