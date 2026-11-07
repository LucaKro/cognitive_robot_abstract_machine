"""
Tests for letting the human pipelines go on only when the CAS holds a human, and
answering with nobody otherwise.
"""

from __future__ import annotations

# %% Imports
from py_trees.common import Status
from robokudo.annotators.gpsr.check_for_human_hypothesis import (
    CheckForHumanHypothesis,
)
from robokudo.types.annotation import Classification
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene

# %% humans in the CAS


def test_a_human_in_the_cas_lets_the_pipeline_go_on(scene: GpsrScene) -> None:
    """
    A human in the CAS lets the pipeline go on.
    """
    annotator = CheckForHumanHypothesis()
    scene.place_in_pipeline(annotator)
    scene.add_human()

    assert annotator.update() == Status.SUCCESS
    assert not scene.has_answered()


def test_a_human_among_objects_lets_the_pipeline_go_on(scene: GpsrScene) -> None:
    """
    A human among objects lets the pipeline go on.
    """
    annotator = CheckForHumanHypothesis()
    scene.place_in_pipeline(annotator)
    scene.add_object(Classification(classname="cup"))
    scene.add_human()
    scene.add_object(Classification(classname="bowl"))

    assert annotator.update() == Status.SUCCESS
    assert not scene.has_answered()


# %% no human in the CAS


def test_only_objects_in_the_cas_answer_with_nobody_and_stop_the_pipeline(
    scene: GpsrScene,
) -> None:
    """
    With only objects in the CAS, the query is answered with nobody and the pipeline
    stops.
    """
    annotator = CheckForHumanHypothesis()
    scene.place_in_pipeline(annotator)
    scene.add_object(Classification(classname="cup"))

    status = annotator.update()

    assert status == Status.FAILURE
    assert list(scene.sent_answer().res) == []
    assert not scene.has_failed()


def test_an_empty_cas_answers_with_nobody_and_stops_the_pipeline(
    scene: GpsrScene,
) -> None:
    """
    With an empty CAS, the query is answered with nobody and the pipeline stops.
    """
    annotator = CheckForHumanHypothesis()
    scene.place_in_pipeline(annotator)

    status = annotator.update()

    assert status == Status.FAILURE
    assert list(scene.sent_answer().res) == []
    assert not scene.has_failed()
