"""
What the pipeline's rule allows before any model is asked: for each relation of the part
hierarchy, whether the rule settles it, leaves it to the model with the true whole
offered, or loses it.
"""

from __future__ import annotations

import pytest

from experiments.warsaw.articulate3d.label_readings import TrueClasses
from experiments.warsaw.articulate3d.measured_scene import SceneToJudge
from experiments.warsaw.articulate3d.rule_ceiling import (
    CandidateOutcome,
    RuleCeiling,
)
from experiments.warsaw.articulate3d.score import Share
from experiments.warsaw.articulate3d.whole_choices import FirstCandidateOfTheRule
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene

from .scenes import measured_kitchen_corner


def ceiling_with(settings, tmp_path, ontology, ground_truth):
    """
    :return: What the rule allows on the kitchen corner, with the true classes.
    """
    rule = FirstCandidateOfTheRule(
        step=MeasureScene(settings=settings, run=Run(directory=tmp_path)),
        ontology=ontology,
    )
    return RuleCeiling.of(
        [SceneToJudge(measured=measured_kitchen_corner(), ground_truth=ground_truth)],
        rule=rule,
        reading=TrueClasses(),
    )


@pytest.fixture
def ceiling(tmp_path, ontology, ground_truth):
    """
    :return: What the rule allows with the pipeline's settings, every candidate kept
        offered.
    """
    return ceiling_with(
        PipelineSettings(prefer_nested_wholes=False), tmp_path, ontology, ground_truth
    )


def test_a_whole_the_rule_takes_without_asking_is_settled(ceiling):
    assert ceiling.count(CandidateOutcome.TAKEN_RIGHT, representable=True) == 3


def test_a_handle_a_seam_from_its_drawer_is_left_to_the_model(ceiling):
    assert ceiling.count(CandidateOutcome.LEFT_TO_THE_MODEL, representable=True) == 1


def test_without_counting_the_seam_as_touching_the_wrong_whole_is_taken(
    tmp_path, ontology, ground_truth
):
    settings = PipelineSettings(largest_gap_counted_as_touching=0.0)

    ceiling = ceiling_with(settings, tmp_path, ontology, ground_truth)

    assert ceiling.count(CandidateOutcome.WRONG_WHOLE_TAKEN, representable=True) == 1


def test_a_part_of_no_known_class_has_no_candidate(ceiling):
    """
    The control's name covers different kinds of things, so neither its class nor its
    relation is judged, and the relation counts as not representable.
    """
    assert ceiling.count(CandidateOutcome.NO_CANDIDATE, representable=False) == 1


def test_a_part_with_no_whole_is_left_alone(ceiling):
    assert ceiling.count(CandidateOutcome.LEFT_WITHOUT_A_WHOLE) == 2


def test_the_ceiling_counts_what_the_rule_settles_or_offers(ceiling):
    assert ceiling.ceiling == Share(4, 4)


def test_without_counting_the_seam_as_touching_the_ceiling_drops(
    tmp_path, ontology, ground_truth
):
    settings = PipelineSettings(largest_gap_counted_as_touching=0.0)

    ceiling = ceiling_with(settings, tmp_path, ontology, ground_truth)

    assert ceiling.ceiling == Share(3, 4)


def test_preferring_nested_wholes_the_handle_takes_its_drawer_without_asking(
    tmp_path, ontology, ground_truth
):
    """
    The handle's drawer touches the cabinet the handle touches, so the drawer is its
    whole and nothing is left to the model.
    """
    ceiling = ceiling_with(PipelineSettings(), tmp_path, ontology, ground_truth)

    assert (
        ceiling.count(CandidateOutcome.TAKEN_RIGHT, representable=True),
        ceiling.count(CandidateOutcome.LEFT_TO_THE_MODEL, representable=True),
    ) == (ceiling.ceiling.total, 0)
