"""
What a run's answers to the questions about wholes add: the run judged again with every
question answered by its first candidate, and how its answers to the questions that may
be answered with none came out.

The run here is of the kitchen corner. The rule took the door, the first handle and the
drawer without asking. The model chose the second handle's drawer over the cabinet
listed first, declined the switch base the control belongs to, and declined a door for
the switch base, which belongs to nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.warsaw.articulate3d.convert import ConvertedScene, SceneConversion
from experiments.warsaw.articulate3d.ground_truth import GroundTruthFile
from experiments.warsaw.articulate3d.run_answers import (
    ConfirmationOutcome,
    RunAnswers,
    RunAnswersJudged,
    main,
)
from experiments.warsaw.articulate3d.score import ScoreFile, Share
from experiments.warsaw.evaluation.graph import EvaluationGraph, EvaluationNode
from experiments.warsaw.pipeline.records import (
    Adjudications,
    ForcedMembership,
    MembershipAnswer,
    MembershipCandidate,
    MembershipQuestion,
    OpenQuestions,
    Relations,
)
from experiments.warsaw.pipeline.run import Run, RunFile

from .scenes import kitchen_corner

# %% a run of the kitchen corner


def question(part, wholes, may_belong_to_none):
    """
    :return: A question asking which of *wholes* the part belongs to.
    """
    return MembershipQuestion(
        name=part,
        part=part,
        candidates=[
            MembershipCandidate(name=one, field_name="parts") for one in wholes
        ],
        may_belong_to_none=may_belong_to_none,
    )


@pytest.fixture
def run_of_the_kitchen_corner(tmp_path, taxonomy, ground_truth):
    """
    :return: The run, the segment of each part, and the directory of the ground truth.
    """
    scene_directory = tmp_path / "scene"
    SceneConversion().write(kitchen_corner(), scene_directory)
    segment = {
        one.part_id: one.segment for one in ConvertedScene.beside(scene_directory).parts
    }
    cabinet, door, handle, drawer, handle_2, switch_base, control = (
        segment[part_id] for part_id in range(1, 8)
    )
    run = Run(directory=tmp_path / "run")
    run.directory.mkdir()
    run.write_json(
        RunFile.PROVENANCE, {"settings": {"scene_directory": str(scene_directory)}}
    )
    run.write_json(RunFile.TAXONOMY, taxonomy)
    np.savez(
        run.path(RunFile.SPLIT_FACES),
        **{segment[one.part_id]: np.asarray(one.faces) for one in ground_truth.parts},
    )
    run.write_record(
        RunFile.RELATIONS,
        Relations(
            scene="kitchen corner",
            forced=[
                ForcedMembership(part=door, whole=cabinet, field_name="doors"),
                ForcedMembership(part=handle, whole=door, field_name="handles"),
                ForcedMembership(part=drawer, whole=cabinet, field_name="drawers"),
            ],
        ),
    )
    run.write_record(
        RunFile.QUESTIONS,
        OpenQuestions(
            scene="kitchen corner",
            asked=[
                question(handle_2, [cabinet, drawer], may_belong_to_none=False),
                question(control, [switch_base], may_belong_to_none=True),
                question(switch_base, [door], may_belong_to_none=True),
            ],
        ),
    )
    run.write_record(
        RunFile.ADJUDICATIONS,
        Adjudications(
            scene="kitchen corner",
            model="a model",
            answered=[
                MembershipAnswer(name=handle_2, part=handle_2, whole=drawer),
                MembershipAnswer(name=control, part=control, whole=None),
                MembershipAnswer(name=switch_base, part=switch_base, whole=None),
            ],
        ),
    )
    classes = {
        cabinet: "Cabinet",
        door: "Door",
        handle: "Handle",
        drawer: "Drawer",
        handle_2: "Handle",
        switch_base: "ElectricalDevice",
        control: None,
    }
    parents = {door: cabinet, handle: door, drawer: cabinet, handle_2: drawer}
    run.write_record(
        RunFile.EVALUATION_GRAPH,
        EvaluationGraph(
            nodes=[
                EvaluationNode(
                    name=name,
                    input_label=name.rsplit("_", 1)[0],
                    predicted_class=predicted,
                    faces=0,
                    body_id=None,
                    annotation_applied=predicted is not None,
                    parent=parents.get(name),
                )
                for name, predicted in classes.items()
            ]
        ),
    )
    ground_truth_directory = tmp_path / "ground_truth"
    ground_truth_directory.mkdir()
    ground_truth.write(
        GroundTruthFile.SUFFIX.path(ground_truth_directory, ground_truth.scene_id)
    )
    return run, segment, ground_truth_directory


@pytest.fixture
def answers(run_of_the_kitchen_corner, ground_truth):
    """
    :return: The run's answers, beside its scene's ground truth.
    """
    run, _, _ = run_of_the_kitchen_corner
    return RunAnswers(run=run, ground_truth=ground_truth)


# %% every question answered by its first candidate


def test_every_question_takes_its_first_candidate(answers, run_of_the_kitchen_corner):
    _, segment, _ = run_of_the_kitchen_corner

    parents = {
        one.name: one.parent for one in answers.answered_by_first_candidate().nodes
    }

    assert (parents[segment[5]], parents[segment[7]], parents[segment[6]]) == (
        segment[1],
        segment[6],
        segment[2],
    )


def test_what_the_rule_took_without_asking_stays(answers, run_of_the_kitchen_corner):
    _, segment, _ = run_of_the_kitchen_corner

    parents = {
        one.name: one.parent for one in answers.answered_by_first_candidate().nodes
    }

    assert parents[segment[2]] == segment[1]


def test_answered_by_first_candidates_the_run_is_judged_as_the_run_is(answers):
    """
    Six parts are hung: the three the rule took, the second handle from the cabinet, the
    control from its switch base, and the switch base, which belongs to nothing, from a
    door.
    """
    judged = answers.judged()

    assert judged.without_answers.links.precision == Share(4, 6)
    assert judged.with_answers.links.precision == Share(4, 4)


# %% the questions that may be answered with none


def test_a_true_whole_declined_is_counted_as_declined(answers):
    assert answers.judged().confirmations(ConfirmationOutcome.TRUE_WHOLE_DECLINED) == 1


def test_a_wrong_whole_declined_is_counted_as_avoided(answers):
    assert answers.judged().confirmations(ConfirmationOutcome.WRONG_WHOLE_AVOIDED) == 1


def test_a_question_that_must_be_answered_is_no_confirmation(answers):
    judged = answers.judged()

    assert sum(one.count for one in judged.confirmation_outcomes) == 2


# %% every run, from the command line


def test_the_command_writes_what_each_run_s_answers_came_to_beside_the_run(
    run_of_the_kitchen_corner,
):
    run, _, ground_truth_directory = run_of_the_kitchen_corner

    main([str(run.directory), "--ground-truth", str(ground_truth_directory)])

    judged = RunAnswersJudged.from_json(
        json.loads((run.directory / ScoreFile.ANSWERS).read_text())
    )
    assert judged.confirmations(ConfirmationOutcome.TRUE_WHOLE_DECLINED) == 1
