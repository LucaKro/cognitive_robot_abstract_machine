"""
What a run's answers to the questions about wholes add.

A run's rule takes some wholes without asking and puts the rest to the model. Judging
the run again with every question answered by its first candidate, the one touching the
part along the most edges, keeps everything else the run did, its readings of the labels
included, so the difference is the model's answers alone.

A question about wholes only found within the gap may be answered with none. Most of the
time one of them is the true whole, so declining is right only now and then; counting
how the model's answers to these came out says whether asking them pays off.

Run ``python -m experiments.warsaw.articulate3d.run_answers --help`` for its options.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import numpy as np

from experiments.warsaw.articulate3d.convert import ConvertedScene
from experiments.warsaw.articulate3d.ground_truth import (
    GroundTruthFile,
    SceneGroundTruth,
)
from experiments.warsaw.articulate3d.score import (
    ClassLineage,
    SceneScore,
    ScoreFile,
    ScoreOverRuns,
    Share,
)
from experiments.warsaw.articulate3d.splits import ValidationLock
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.pipeline.records import (
    Adjudications,
    MembershipQuestion,
    OpenQuestions,
    Relations,
)
from experiments.warsaw.pipeline.run import Run, RunFile

# %% how a question that may be answered with none came out


class ConfirmationOutcome(StrEnum):
    """
    How the model answered a question about wholes that may be answered with none.
    """

    TRUE_WHOLE_KEPT = "true whole kept"
    """
    The true whole was offered, and the model took it.
    """

    TRUE_WHOLE_DECLINED = "true whole declined"
    """
    The true whole was offered, and the model answered none.
    """

    OTHER_WHOLE_TAKEN = "true whole offered, another taken"
    """
    The true whole was offered, and the model took another candidate.
    """

    WRONG_WHOLE_AVOIDED = "true whole not offered, none taken"
    """
    The part has no whole, or another one, and the model answered none.
    """

    WRONG_WHOLE_TAKEN = "true whole not offered, one taken"
    """
    The part has no whole, or another one, and the model took a candidate.
    """


@dataclass(frozen=True)
class CountedConfirmation(JsonRecord):
    """
    How often the model answered a confirming question one way.
    """

    outcome: ConfirmationOutcome
    """
    How the answer came out.
    """

    count: int
    """
    How many questions it came out that way.
    """


# %% a run's answers, judged


@dataclass
class RunAnswersJudged(JsonRecord):
    """
    A run judged with its model's answers and with every question answered by its first
    candidate, and how its confirming questions came out.
    """

    run: str
    """
    The run judged.
    """

    with_answers: SceneScore
    """
    The run as it is.
    """

    without_answers: SceneScore
    """
    The run with every question about wholes answered by its first candidate.
    """

    confirmation_outcomes: list[CountedConfirmation] = field(default_factory=list)
    """
    How often each outcome came about among the questions that may be answered with
    none.
    """

    def confirmations(self, outcome: ConfirmationOutcome) -> int:
        """
        :param outcome: How a confirming question came out.
        :return: How many came out that way.
        """
        return sum(
            one.count for one in self.confirmation_outcomes if one.outcome is outcome
        )

    @property
    def true_whole_offered(self) -> Share:
        """
        :return: The confirming questions that offered the true whole: how often always
            taking a candidate would have been right.
        """
        offered = (
            ConfirmationOutcome.TRUE_WHOLE_KEPT,
            ConfirmationOutcome.TRUE_WHOLE_DECLINED,
            ConfirmationOutcome.OTHER_WHOLE_TAKEN,
        )
        return Share(
            sum(self.confirmations(one) for one in offered),
            sum(one.count for one in self.confirmation_outcomes),
        )


@dataclass
class RunAnswers:
    """
    A finished run's questions about wholes and its model's answers, beside the ground
    truth of its scene.
    """

    run: Run
    """
    The finished run.
    """

    ground_truth: SceneGroundTruth
    """
    The ground truth of the scene it ran on.
    """

    lock: ValidationLock = field(default_factory=ValidationLock)
    """
    What keeps the validation scenes closed until the code is frozen.
    """

    @cached_property
    def questions(self) -> list[MembershipQuestion]:
        """
        :return: The questions the run asked about wholes.
        """
        return self.run.read_record_if_written(
            RunFile.QUESTIONS, OpenQuestions(scene="")
        ).membership

    @cached_property
    def answers(self) -> dict[str, str | None]:
        """
        :return: Per part asked about, the whole the model answered, or nothing.
        """
        answered = self.run.read_record_if_written(
            RunFile.ADJUDICATIONS, Adjudications(scene="", model="")
        )
        return {one.part: one.whole for one in answered.membership}

    @cached_property
    def true_whole_of_segment(self) -> dict[str, str | None]:
        """
        :return: Per segment written from a part the ground truth keeps, the segment of
            its true whole, or nothing for a part with no whole.
        """
        segment = ConvertedScene.beside(self.run.scene_directory).segment_of_part
        whole_of_part = {one.part: one.whole for one in self.ground_truth.relations}
        return {
            segment[one.part_id]: segment.get(whole_of_part.get(one.part_id))
            for one in self.ground_truth.parts
            if one.part_id in segment
        }

    def answered_by_first_candidate(self) -> EvaluationGraph:
        """
        :return: The run's graph with every part hung as the rule takes it, or as the
            first candidate of the question about it.
        """
        forced = self.run.read_record_if_written(
            RunFile.RELATIONS, Relations(scene="")
        ).forced
        parents = {one.part: one.whole for one in forced} | {
            one.part: one.candidates[0].name for one in self.questions
        }
        graph = self.run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph)
        return replace(
            graph,
            nodes=[replace(one, parent=parents.get(one.name)) for one in graph.nodes],
        )

    def confirmation_outcome(self, question: MembershipQuestion) -> ConfirmationOutcome:
        """
        :param question: A question that may be answered with none.
        :return: How the model's answer to it came out.
        """
        true_whole = self.true_whole_of_segment.get(question.part)
        answer = self.answers.get(question.part)
        if true_whole in question.candidate_names:
            if answer == true_whole:
                return ConfirmationOutcome.TRUE_WHOLE_KEPT
            if answer is None:
                return ConfirmationOutcome.TRUE_WHOLE_DECLINED
            return ConfirmationOutcome.OTHER_WHOLE_TAKEN
        if answer is None:
            return ConfirmationOutcome.WRONG_WHOLE_AVOIDED
        return ConfirmationOutcome.WRONG_WHOLE_TAKEN

    def judged(self) -> RunAnswersJudged:
        """
        :return: The run judged with and without its model's answers, and how its
            confirming questions came out.
        :raises ValidationSceneLockedError: If the run is of a validation scene and the
            checkout is not frozen.
        """
        self.lock.check(self.ground_truth.scene_id)
        faces = np.load(self.run.path(RunFile.SPLIT_FACES))
        bodies = {name: faces[name] for name in faces.files}
        lineage = ClassLineage.of_run(self.run)
        counted = Counter(
            self.confirmation_outcome(one)
            for one in self.questions
            if one.may_belong_to_none
        )
        return RunAnswersJudged(
            run=self.run.directory.name,
            with_answers=SceneScore.between(
                run_name=self.run.directory.name,
                ground_truth=self.ground_truth,
                bodies=bodies,
                graph=self.run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph),
                lineage=lineage,
            ),
            without_answers=SceneScore.between(
                run_name=self.run.directory.name,
                ground_truth=self.ground_truth,
                bodies=bodies,
                graph=self.answered_by_first_candidate(),
                lineage=lineage,
            ),
            confirmation_outcomes=[
                CountedConfirmation(outcome=one, count=counted[one])
                for one in ConfirmationOutcome
                if counted[one]
            ],
        )


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for judging runs' answers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Finished runs' directories")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="The directory holding every scene's ground truth",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Judge each run with and without its model's answers, write what each came to into
    the run's directory, and say what they came to together.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once every run is judged.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    judged = []
    for directory in parsed.runs:
        run = Run(directory=directory)
        scene_id = ConvertedScene.scene_id_of(run.scene_directory)
        ground_truth = SceneGroundTruth.read(
            GroundTruthFile.SUFFIX.path(parsed.ground_truth, scene_id)
        )
        one = RunAnswers(run=run, ground_truth=ground_truth).judged()
        ScoreFile.ANSWERS.write(run, one)
        judged.append(one)
    for title, scores in (
        ("with the model's answers", [one.with_answers for one in judged]),
        (
            "every question answered by its first candidate",
            [one.without_answers for one in judged],
        ),
    ):
        logger.info("== %s", title)
        for line in ScoreOverRuns.over(scores).lines():
            logger.info("   %s", line)
    logger.info("== questions that may be answered with none")
    for outcome in ConfirmationOutcome:
        logger.info(
            "   %-36s %s",
            outcome.value,
            sum(one.confirmations(outcome) for one in judged),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
