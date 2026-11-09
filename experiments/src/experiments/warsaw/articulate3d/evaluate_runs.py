"""
Everything finished runs come to, as the tables of a document, from their directories.

Each run of a scene with a ground truth is scored, and judged with and without its
model's answers about wholes; both are kept in the run's directory. Given the measured
scenes, the baselines are judged on the scenes the runs were scored on, and so is what
the rule allows with each method's settings. The tables are those of
:mod:`experiments.warsaw.evaluation.results`.

Run ``python -m experiments.warsaw.articulate3d.evaluate_runs --help`` for its options.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from krrood.ormatic.utils import classproperty
from typing_extensions import Any, Sequence

from experiments.warsaw.articulate3d.baselines import BaselineName, Baselines
from experiments.warsaw.articulate3d.convert import ConvertedScene
from experiments.warsaw.articulate3d.exceptions import SceneNotMeasuredError
from experiments.warsaw.articulate3d.ground_truth import (
    GroundTruthFile,
    SceneGroundTruth,
)
from experiments.warsaw.articulate3d.label_readings import TrueClasses
from experiments.warsaw.articulate3d.measured_scene import (
    MeasuredSceneFile,
    SceneToJudge,
)
from experiments.warsaw.articulate3d.rule_ceiling import RuleCeiling
from experiments.warsaw.articulate3d.run_answers import RunAnswers
from experiments.warsaw.articulate3d.score import SceneScore, ScoreFile, ScoreOverRuns
from experiments.warsaw.articulate3d.splits import ValidationLock
from experiments.warsaw.articulate3d.whole_choices import FirstCandidateOfTheRule
from experiments.warsaw.bases import HasLogger
from experiments.warsaw.evaluation.results import (
    BaselineGroup,
    ResultTables,
    RunComparison,
    RunMeasurements,
    SettingName,
)
from experiments.warsaw.pipeline.batch import BatchFile, BatchLedger
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings

# %% evaluating runs


@dataclass
class RunsEvaluation(HasLogger):
    """
    Finished runs scored, judged and compared with the baselines and with what the rule
    allows.
    """

    ground_truth_directory: Path
    """
    The directory holding every scene's ground truth.
    """

    measured_directory: Path | None = None
    """
    The directory of measured scenes, without which no baseline or ceiling is judged.
    """

    renamed_conditions: dict[str, str] = field(default_factory=dict)
    """
    Per name a method or baseline is given on its own, what to call it instead.
    """

    baseline_names: list[BaselineName] = field(
        default_factory=lambda: list(BaselineName)
    )
    """
    The baselines to judge.
    """

    lock: ValidationLock = field(default_factory=ValidationLock)
    """
    What keeps the validation scenes closed until the code is frozen.
    """

    @classproperty
    def settings_of_the_rule(cls) -> list[SettingName]:
        """
        :return: The settings that change what the rule offers a part, given its classes.
        """
        return [
            SettingName.LARGEST_GAP_TO_A_WHOLE,
            SettingName.LARGEST_GAP_COUNTED_AS_TOUCHING,
            SettingName.CONFIRM_WHOLES_WITHIN_THE_GAP,
            SettingName.PREFER_NESTED_WHOLES,
        ]

    @classproperty
    def settings_runs_may_predate(cls) -> dict[SettingName, Any]:
        """
        :return: The rule's settings added after runs were already made, each with the
            value a run made before it worked with.
        """
        return {SettingName.PREFER_NESTED_WHOLES: False}

    def finished_runs(self, paths: Sequence[Path]) -> list[Run]:
        """
        :param paths: Runs, batches, and directories holding runs.
        :return: Every finished run among them, each once, in the order given.
        """
        found: dict[Path, Run] = {}
        for path in map(Path, paths):
            for run in self.collected_runs(path):
                found.setdefault(run.directory.resolve(), run)
        return list(found.values())

    def collected_runs(self, path: Path) -> list[Run]:
        """
        :param path: A run, a batch, or a directory holding runs.
        :return: The finished runs it holds: a batch's those in its ledger.
        """
        ledger = path / BatchFile.LEDGER
        if ledger.is_file():
            return [
                Run(directory=Path(one.run_directory))
                for one in BatchLedger.read_or_empty(ledger).finished
            ]
        if Run(directory=path).holds(RunFile.PROVENANCE):
            candidates = [Run(directory=path)]
        else:
            candidates = [
                Run(directory=one)
                for one in sorted(path.iterdir())
                if Run(directory=one).holds(RunFile.PROVENANCE)
            ]
        unfinished = [one for one in candidates if not self.is_finished(one)]
        for run in unfinished:
            self.logger.warning("%s did not finish and is left out", run.directory)
        return [one for one in candidates if one not in unfinished]

    @staticmethod
    def is_finished(run: Run) -> bool:
        """
        :param run: A run.
        :return: Whether it got to its end.
        """
        return run.holds(RunFile.EVALUATION_GRAPH)

    def judge(self, run: Run) -> None:
        """
        Score the run and judge it with and without its model's answers, keeping both in
        its directory, where its scene has a ground truth; a run told not to use the
        part-whole knowledge only scored, since it asked no question of the rule.

        :param run: A finished run.
        :raises ValidationSceneLockedError: If the run is of a validation scene and the
            checkout is not frozen.
        """
        scene_id = ConvertedScene.scene_id_of(run.scene_directory)
        truth = GroundTruthFile.SUFFIX.path(self.ground_truth_directory, scene_id)
        if not truth.is_file():
            self.logger.info("%s: %s has no ground truth", run.directory.name, scene_id)
            return
        ScoreFile.SCENE.write(
            run, SceneScore.of(run, self.ground_truth_directory, lock=self.lock)
        )
        if not run.uses_part_whole_knowledge:
            return
        answers = RunAnswers(
            run=run, ground_truth=SceneGroundTruth.read(truth), lock=self.lock
        )
        ScoreFile.ANSWERS.write(run, answers.judged())

    def scenes_to_judge(self, scene_ids: Sequence[str]) -> list[SceneToJudge]:
        """
        :param scene_ids: Scenes runs were scored on.
        :return: Each, measured, with its ground truth.
        :raises SceneNotMeasuredError: If a scene was not measured.
        """
        missing = [
            one
            for one in scene_ids
            if not MeasuredSceneFile.SUFFIX.path(self.measured_directory, one).is_file()
        ]
        if missing:
            raise SceneNotMeasuredError(
                scene_ids=missing, measured_directory=self.measured_directory
            )
        return SceneToJudge.all_in(
            self.measured_directory, self.ground_truth_directory, list(scene_ids)
        )

    def baselines(self, scene_ids: Sequence[str]) -> list[BaselineGroup]:
        """
        :param scene_ids: The scenes the runs were scored on.
        :return: Each baseline, judged on those scenes with the pipeline's settings.
        """
        scenes = self.scenes_to_judge(scene_ids)
        baselines = Baselines.from_settings(PipelineSettings(), self.measured_directory)
        return [
            BaselineGroup(
                self.renamed_conditions.get(name.value, name.value),
                baselines.named(name).scores(scenes, baselines.lineage),
            )
            for name in self.baseline_names
        ]

    def rule_settings(self, recorded: dict[str, Any]) -> PipelineSettings | None:
        """
        :param recorded: Every setting a run recorded.
        :return: The pipeline's settings with the rule's as the run recorded them, or
            nothing where it recorded one of them not that it could have, or was told not
            to use the part-whole knowledge the rule is made of.
        """
        if recorded.get(SettingName.USE_PART_WHOLE_KNOWLEDGE) is False:
            return None
        recorded = {**self.settings_runs_may_predate, **recorded}
        missing = [one for one in self.settings_of_the_rule if one not in recorded]
        if missing:
            return None
        return PipelineSettings(
            **{one.value: recorded[one] for one in self.settings_of_the_rule}
        )

    def rule_ceilings(self, comparison: RunComparison) -> dict[str, RuleCeiling]:
        """
        :param comparison: The runs compared.
        :return: Per method, what the rule allows with its settings on the scenes its
            runs were scored on, for every method whose runs recorded the rule's settings.
        """
        ceilings = {}
        for group in comparison.by_condition:
            settings = self.rule_settings(group.runs[0].settings)
            scene_ids = sorted({one.scene for one in group.runs if one.score})
            if settings is None:
                self.logger.info("%s did not record the rule's settings", group.name)
            if settings is None or not scene_ids:
                continue
            ceilings[group.name] = RuleCeiling.of(
                self.scenes_to_judge(scene_ids),
                rule=FirstCandidateOfTheRule.from_settings(
                    settings, self.measured_directory
                ),
                reading=TrueClasses(),
            )
        return ceilings

    def result_tables(self, paths: Sequence[Path]) -> ResultTables:
        """
        :param paths: Runs, batches, and directories holding runs.
        :return: The tables of every finished run among them, each run judged anew.
        """
        runs = self.finished_runs(paths)
        for run in runs:
            self.judge(run)
        comparison = RunComparison(
            runs=[RunMeasurements.of(one) for one in runs],
            renamed_conditions=self.renamed_conditions,
        )
        if self.measured_directory is None:
            return ResultTables(comparison)
        scored = sorted({one.scene for one in comparison.runs if one.score})
        return ResultTables(
            comparison,
            baselines=self.baselines(scored),
            ceilings=self.rule_ceilings(comparison),
        )


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for evaluating finished runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="Runs, batches, and directories holding runs",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="The directory holding every scene's ground truth",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the tables"
    )
    parser.add_argument(
        "--measured",
        type=Path,
        help="The directory of measured scenes, to judge the baselines and the rule",
    )
    parser.add_argument(
        "--baseline",
        type=BaselineName,
        choices=list(BaselineName),
        action="append",
        help="A baseline to judge; every one by default",
    )
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="METHOD=NAME",
        help="What to call a method or baseline in the tables",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Evaluate finished runs and write their tables, saying what each method came to.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the tables are written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    evaluation = RunsEvaluation(
        ground_truth_directory=parsed.ground_truth,
        measured_directory=parsed.measured,
        renamed_conditions=dict(one.split("=", 1) for one in parsed.rename),
        baseline_names=parsed.baseline or list(BaselineName),
    )
    tables = evaluation.result_tables(parsed.runs)
    for group in tables.comparison.by_condition:
        scores = [one.score for one in group.runs if one.score]
        logger.info(
            "== %s: %s runs, %s scored", group.name, group.run_count, len(scores)
        )
        if not scores:
            continue
        for line in ScoreOverRuns.over(scores).lines():
            logger.info("   %s", line)
    for path in tables.write(parsed.output):
        logger.info("%s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
