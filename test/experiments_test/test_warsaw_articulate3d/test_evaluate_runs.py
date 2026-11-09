"""
Everything finished runs come to, from their directories: which runs a path holds, each
run scored and judged in its own directory, and the baselines and what the rule allows
judged on the scenes the runs were scored on.
"""

from __future__ import annotations

import pytest

from experiments.warsaw.articulate3d.baselines import Baseline, BaselineName
from experiments.warsaw.articulate3d.evaluate_runs import RunsEvaluation
from experiments.warsaw.articulate3d.exceptions import SceneNotMeasuredError
from experiments.warsaw.articulate3d.ground_truth import GroundTruthFile
from experiments.warsaw.articulate3d.label_readings import TrueClasses
from experiments.warsaw.articulate3d.measured_scene import MeasuredSceneFile
from experiments.warsaw.articulate3d.run_answers import RunAnswers, RunAnswersJudged
from experiments.warsaw.articulate3d.score import ClassLineage, SceneScore, ScoreFile
from experiments.warsaw.articulate3d.whole_choices import FirstCandidateOfTheRule
from experiments.warsaw.evaluation.results import SettingName
from experiments.warsaw.pipeline.batch import BatchLedger, BatchFile
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings

from .scenes import SceneName, measured_kitchen_corner
from .test_run_answers import run_of_the_kitchen_corner  # noqa: F401

# %% which runs a path holds


def run_written(directory, finished=True) -> Run:
    """
    :return: A run holding only its provenance, and its graph once it finished.
    """
    run = Run(directory=directory)
    run.directory.mkdir(parents=True)
    run.write_json(RunFile.PROVENANCE, {"settings": {}})
    if finished:
        run.path(RunFile.EVALUATION_GRAPH).write_text("{}")
    return run


@pytest.fixture
def evaluation(tmp_path) -> RunsEvaluation:
    """
    :return: An evaluation against an empty directory of ground truth.
    """
    return RunsEvaluation(ground_truth_directory=tmp_path / "ground_truth")


def test_a_run_given_is_evaluated(evaluation, tmp_path):
    run = run_written(tmp_path / "one")

    assert evaluation.finished_runs([run.directory]) == [run]


def test_a_directory_of_runs_gives_its_finished_runs(evaluation, tmp_path):
    finished = run_written(tmp_path / "runs" / "one")
    run_written(tmp_path / "runs" / "two", finished=False)

    assert evaluation.finished_runs([tmp_path / "runs"]) == [finished]


def test_a_batch_gives_the_runs_its_ledger_holds(evaluation, tmp_path):
    run = run_written(tmp_path / "runs" / "one")
    batch = tmp_path / "runs" / "batches" / "tuning"
    batch.mkdir(parents=True)
    BatchLedger().with_finished(tmp_path / "scene", run.directory).write(
        batch / BatchFile.LEDGER
    )

    assert [one.directory for one in evaluation.finished_runs([batch])] == [
        run.directory.resolve()
    ]


def test_a_run_given_twice_is_evaluated_once(evaluation, tmp_path):
    run = run_written(tmp_path / "runs" / "one")

    assert evaluation.finished_runs([run.directory, tmp_path / "runs"]) == [run]


# %% each run, judged in its own directory


def test_a_run_is_scored_and_judged_with_and_without_its_answers(
    run_of_the_kitchen_corner, ground_truth
):
    run, _, ground_truth_directory = run_of_the_kitchen_corner

    RunsEvaluation(ground_truth_directory=ground_truth_directory).judge(run)

    assert ScoreFile.SCENE.read(run, SceneScore) == SceneScore.of(
        run, ground_truth_directory
    )
    assert (
        ScoreFile.ANSWERS.read(run, RunAnswersJudged)
        == RunAnswers(run=run, ground_truth=ground_truth).judged()
    )


def test_a_run_of_a_scene_with_no_ground_truth_is_not_scored(
    run_of_the_kitchen_corner, tmp_path
):
    run, _, _ = run_of_the_kitchen_corner

    RunsEvaluation(ground_truth_directory=tmp_path / "elsewhere").judge(run)

    assert ScoreFile.SCENE.read(run, SceneScore) is None


# %% the rule's settings a method recorded


def test_the_rule_works_with_the_settings_a_method_recorded(evaluation):
    recorded = {
        SettingName.LARGEST_GAP_TO_A_WHOLE: 0.2,
        SettingName.LARGEST_GAP_COUNTED_AS_TOUCHING: 0.03,
        SettingName.CONFIRM_WHOLES_WITHIN_THE_GAP: False,
        SettingName.PREFER_NESTED_WHOLES: True,
    }

    assert evaluation.rule_settings(recorded) == PipelineSettings(
        largest_gap_to_a_whole=0.2,
        largest_gap_counted_as_touching=0.03,
        confirm_wholes_within_the_gap=False,
        prefer_nested_wholes=True,
    )


def test_a_method_from_before_nested_wholes_were_preferred_worked_without(evaluation):
    """
    A run made before the setting existed recorded nothing for it, and kept every
    candidate.
    """
    recorded = {
        SettingName.LARGEST_GAP_TO_A_WHOLE: 0.1,
        SettingName.LARGEST_GAP_COUNTED_AS_TOUCHING: 0.01,
        SettingName.CONFIRM_WHOLES_WITHIN_THE_GAP: True,
    }

    assert evaluation.rule_settings(recorded).prefer_nested_wholes is False


def test_a_method_that_did_not_record_the_rule_s_settings_has_no_rule(evaluation):
    [first, *_] = evaluation.settings_of_the_rule

    assert evaluation.rule_settings({first.value: 0.1}) is None


# %% the baselines, on the scenes the runs were scored on


@pytest.fixture
def measured_and_true(tmp_path, ground_truth):
    """
    :return: The evaluation of runs with the kitchen corner measured and its ground
        truth written.
    """
    measured_directory = tmp_path / "measured"
    measured_directory.mkdir()
    measured_kitchen_corner().write(
        MeasuredSceneFile.SUFFIX.path(measured_directory, SceneName.KITCHEN_CORNER)
    )
    truth_directory = tmp_path / "truth"
    truth_directory.mkdir()
    ground_truth.write(
        GroundTruthFile.SUFFIX.path(truth_directory, SceneName.KITCHEN_CORNER)
    )
    return RunsEvaluation(
        ground_truth_directory=truth_directory,
        measured_directory=measured_directory,
        baseline_names=[BaselineName.TRUE_CLASSES_AND_RULE],
    )


def test_a_baseline_is_judged_on_the_scenes_the_runs_were_scored_on(
    measured_and_true, ground_truth, taxonomy, tmp_path
):
    [group] = measured_and_true.baselines([SceneName.KITCHEN_CORNER])

    baseline = Baseline(
        reading=TrueClasses(),
        choice=FirstCandidateOfTheRule.from_settings(PipelineSettings(), tmp_path),
    )
    assert (group.name, group.scene_scores) == (
        BaselineName.TRUE_CLASSES_AND_RULE,
        [
            baseline.score(
                measured_kitchen_corner(),
                ground_truth,
                ClassLineage.from_taxonomy(taxonomy, {}),
            )
        ],
    )


def test_a_scene_scored_but_not_measured_is_refused(measured_and_true):
    with pytest.raises(SceneNotMeasuredError):
        measured_and_true.baselines(["a scene nobody measured"])
