"""
Choosing the whole each part hangs from without asking a model: by geometry alone, or by
the pipeline's own candidate rule with every question answered by its first candidate.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.warsaw.articulate3d.label_readings import TrueClasses
from experiments.warsaw.articulate3d.whole_choices import (
    FirstCandidateOfTheRule,
    LargerObjectTouchedMost,
)
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene

from .scenes import measured_kitchen_corner, measured_pair, measured_segment

# %% by geometry alone


@pytest.fixture
def geometry():
    """
    :return: The choice by geometry, with the pipeline's gaps.
    """
    return LargerObjectTouchedMost(settings=PipelineSettings())


def test_a_part_hangs_from_the_larger_object_it_touches_along_the_most_edges(geometry):
    wholes = geometry.wholes_of(measured_kitchen_corner(), {})

    assert (wholes["door_1"], wholes["drawer_1"], wholes["handle_2"]) == (
        "cabinet_1",
        "cabinet_1",
        "cabinet_1",
    )


def test_the_larger_object_is_the_one_with_the_larger_box_not_more_faces(geometry):
    """
    A door is scanned in more faces than the thin frame around it, but the frame's box
    encloses the door's, so the door hangs from the frame.
    """
    door = replace(
        measured_segment("door_1", "door", faces=200),
        minimum_corner=(0.05, 0.0, 0.0),
        maximum_corner=(0.95, 0.05, 2.0),
    )
    frame = replace(
        measured_segment("door_frame_1", "door_frame", faces=50),
        minimum_corner=(0.0, 0.0, 0.0),
        maximum_corner=(1.0, 0.1, 2.1),
    )
    scene = replace(
        measured_kitchen_corner(),
        segments=[door, frame],
        pairs=[measured_pair("door_frame_1", "door_1", touching_edges=40)],
    )

    assert geometry.wholes_of(scene, {}) == {"door_1": "door_frame_1"}


def test_an_object_with_no_larger_neighbour_hangs_from_nothing(geometry):
    wholes = geometry.wholes_of(measured_kitchen_corner(), {})

    assert "cabinet_1" not in wholes
    assert "switch_base_1" not in wholes


def test_a_larger_object_within_the_gap_is_taken_when_none_touches(geometry):
    measured = measured_kitchen_corner()
    gap = geometry.settings.largest_gap_to_a_whole
    apart = replace(
        measured,
        pairs=[
            one
            for one in measured.pairs
            if {one.one, one.other} != {"switch_base_1", "control_1"}
        ]
        + [measured_pair("switch_base_1", "control_1", distance=gap / 2)],
    )

    assert geometry.wholes_of(apart, {})["control_1"] == "switch_base_1"


# %% by the pipeline's rule


@pytest.fixture
def rule(tmp_path, ontology):
    """
    :return: The choice by the pipeline's rule, with the pipeline's settings.
    """
    return FirstCandidateOfTheRule(
        step=MeasureScene(settings=PipelineSettings(), run=Run(directory=tmp_path)),
        ontology=ontology,
    )


def test_the_rule_hangs_each_part_from_the_one_whole_the_ontology_allows(
    rule, ground_truth
):
    measured = measured_kitchen_corner()
    classes = TrueClasses().classes_of(measured, ground_truth)

    wholes = rule.wholes_of(measured, classes)

    assert (wholes["door_1"], wholes["handle_1"], wholes["drawer_1"]) == (
        "cabinet_1",
        "door_1",
        "cabinet_1",
    )


def test_a_question_is_answered_with_the_whole_touched_along_the_most_edges(
    rule, ground_truth
):
    """
    The handle touches the cabinet and lies a seam from its drawer, so both are offered,
    and the cabinet is listed first.
    """
    measured = measured_kitchen_corner()
    classes = TrueClasses().classes_of(measured, ground_truth)
    keeping_every_candidate = FirstCandidateOfTheRule(
        step=MeasureScene(
            settings=PipelineSettings(prefer_nested_wholes=False), run=rule.step.run
        ),
        ontology=rule.ontology,
    )

    assert keeping_every_candidate.wholes_of(measured, classes)["handle_2"] == (
        "cabinet_1"
    )


def test_preferring_nested_wholes_the_handle_hangs_from_its_drawer(rule, ground_truth):
    """
    The drawer the handle lies a seam from touches the cabinet the handle touches.
    """
    measured = measured_kitchen_corner()
    classes = TrueClasses().classes_of(measured, ground_truth)

    assert rule.wholes_of(measured, classes)["handle_2"] == "drawer_1"


def test_a_part_of_no_known_class_hangs_from_nothing(rule, ground_truth):
    measured = measured_kitchen_corner()
    classes = TrueClasses().classes_of(measured, ground_truth)

    assert "control_1" not in rule.wholes_of(measured, classes)
