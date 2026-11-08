"""
What the pipeline's candidate rule allows before any model is asked.

For every relation of the part hierarchy, the rule either takes the true whole without a
question, leaves the choice to the model with the true whole among the candidates, or
loses the relation: it takes another whole, asks without offering the true one, or
offers nothing. Read with the true classes, the share it settles or leaves to the model is
the most any model answering its questions can reach.

Run ``python -m experiments.warsaw.articulate3d.rule_ceiling --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Sequence

from experiments.warsaw.articulate3d.ground_truth import (
    Expressibility,
    SceneGroundTruth,
)
from experiments.warsaw.articulate3d.label_readings import LabelReading, TrueClasses
from experiments.warsaw.articulate3d.measured_scene import MeasuredScene, SceneToJudge
from experiments.warsaw.articulate3d.score import Share
from experiments.warsaw.articulate3d.whole_choices import FirstCandidateOfTheRule
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.pipeline.records import MembershipQuestion
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene, Memberships

# %% what the rule does with one part


class CandidateOutcome(StrEnum):
    """
    What the rule does with one part, judged against the part hierarchy.
    """

    TAKEN_RIGHT = "true whole taken without a question"
    """
    The part has a whole, and the rule takes it without asking.
    """

    LEFT_TO_THE_MODEL = "asked, true whole offered"
    """
    The part has a whole, and the rule asks with that whole among the candidates.
    """

    WRONG_WHOLE_TAKEN = "wrong whole taken without a question"
    """
    The part has a whole, and the rule takes another one without asking.
    """

    TRUE_WHOLE_NOT_OFFERED = "asked, true whole not offered"
    """
    The part has a whole, and the rule asks without offering it.
    """

    NO_CANDIDATE = "no candidate"
    """
    The part has a whole, and the rule offers none.
    """

    LEFT_WITHOUT_A_WHOLE = "no whole, none given"
    """
    The part has no whole, and the rule offers none.
    """

    GIVEN_A_WHOLE = "no whole, one taken without a question"
    """
    The part has no whole, and the rule takes one without asking.
    """

    ASKED_AND_MAY_DECLINE = "no whole, asked and may answer none"
    """
    The part has no whole, and the rule asks with none allowed as an answer.
    """

    ASKED_AND_MUST_CHOOSE = "no whole, asked and must choose one"
    """
    The part has no whole, and the rule asks without allowing none.
    """


@dataclass(frozen=True)
class PartOutcome:
    """
    What the rule did with one part, and whether the ontology can hold its relation.
    """

    outcome: CandidateOutcome
    """
    What the rule did.
    """

    representable: bool | None
    """
    Whether the ontology can hold the part's relation to its whole, or nothing for a part
    with no whole.
    """


@dataclass(frozen=True)
class CountedOutcome(JsonRecord):
    """
    How often the rule did one thing with a part.
    """

    outcome: CandidateOutcome
    """
    What the rule did.
    """

    representable: bool | None
    """
    Whether the ontology can hold the part's relation to its whole, or nothing for a part
    with no whole.
    """

    count: int
    """
    How many parts it did it with.
    """


# %% what the rule allows over many scenes


@dataclass
class RuleCeiling(JsonRecord):
    """
    What the rule did with every part of a set of scenes.
    """

    outcomes: list[CountedOutcome] = field(default_factory=list)
    """
    How often each outcome came about.
    """

    @classmethod
    def of(
        cls,
        scenes: Sequence[SceneToJudge],
        rule: FirstCandidateOfTheRule,
        reading: LabelReading,
    ) -> RuleCeiling:
        """
        :param scenes: The scenes, measured, with their ground truth.
        :param rule: The pipeline's rule.
        :param reading: How the labels are read as classes.
        :return: What the rule did with every part of them.
        """
        counted: Counter[PartOutcome] = Counter()
        for scene in scenes:
            memberships = rule.memberships(
                scene.measured,
                reading.classes_of(scene.measured, scene.ground_truth),
            )
            counted.update(
                cls.outcomes_in(scene.measured, scene.ground_truth, memberships)
            )
        return cls(
            outcomes=[
                CountedOutcome(
                    outcome=one.outcome, representable=one.representable, count=n
                )
                for one, n in sorted(
                    counted.items(),
                    key=lambda item: (
                        list(CandidateOutcome).index(item[0].outcome),
                        str(item[0].representable),
                    ),
                )
            ]
        )

    @staticmethod
    def outcomes_in(
        scene: MeasuredScene, ground_truth: SceneGroundTruth, memberships: Memberships
    ) -> list[PartOutcome]:
        """
        :param scene: The measured scene.
        :param ground_truth: Its ground truth.
        :param memberships: What the rule did in it.
        :return: For each part the ground truth keeps, what the rule did with it and
            whether its relation is one the ontology can hold.
        """
        segment = scene.converted.segment_of_part
        forced = {one.part: one.whole for one in memberships.forced}
        asked = {one.part: one for one in memberships.to_ask_about}
        relations = {
            relation.part: relation
            for relation in ground_truth.relations
            if relation.part in segment and relation.whole in segment
        }
        found = []
        for part in ground_truth.parts:
            if part.part_id not in segment:
                continue
            relation = relations.get(part.part_id)
            if relation is None:
                found.append(
                    PartOutcome(
                        outcome=RuleCeiling.outcome_without_whole(
                            segment[part.part_id], forced, asked
                        ),
                        representable=None,
                    )
                )
                continue
            hanging, holder = segment[relation.part], segment[relation.whole]
            if relation.expressibility is Expressibility.CARRIED:
                hanging, holder = holder, hanging
            found.append(
                PartOutcome(
                    outcome=RuleCeiling.outcome_with_whole(
                        hanging, holder, forced, asked
                    ),
                    representable=relation.expressibility.reachable,
                )
            )
        return found

    @staticmethod
    def outcome_with_whole(
        part: str,
        whole: str,
        forced: dict[str, str],
        asked: dict[str, MembershipQuestion],
    ) -> CandidateOutcome:
        """
        :param part: The segment hanging from the whole as the rule would mount it.
        :param whole: The segment it hangs from.
        :param forced: The wholes the rule takes without asking, by part.
        :param asked: The questions the rule asks, by part.
        :return: What the rule does with the part.
        """
        if part in forced:
            return (
                CandidateOutcome.TAKEN_RIGHT
                if forced[part] == whole
                else CandidateOutcome.WRONG_WHOLE_TAKEN
            )
        if part not in asked:
            return CandidateOutcome.NO_CANDIDATE
        if whole in asked[part].candidate_names:
            return CandidateOutcome.LEFT_TO_THE_MODEL
        return CandidateOutcome.TRUE_WHOLE_NOT_OFFERED

    @staticmethod
    def outcome_without_whole(
        part: str, forced: dict[str, str], asked: dict[str, MembershipQuestion]
    ) -> CandidateOutcome:
        """
        :param part: A segment the part hierarchy hangs from nothing.
        :param forced: The wholes the rule takes without asking, by part.
        :param asked: The questions the rule asks, by part.
        :return: What the rule does with it.
        """
        if part in forced:
            return CandidateOutcome.GIVEN_A_WHOLE
        if part not in asked:
            return CandidateOutcome.LEFT_WITHOUT_A_WHOLE
        if asked[part].may_belong_to_none:
            return CandidateOutcome.ASKED_AND_MAY_DECLINE
        return CandidateOutcome.ASKED_AND_MUST_CHOOSE

    def count(
        self, outcome: CandidateOutcome, representable: bool | None = None
    ) -> int:
        """
        :param outcome: An outcome.
        :param representable: Whether the relations counted are ones the ontology can
            hold; nothing for the outcomes of parts with no whole.
        :return: How many parts came to it.
        """
        return sum(
            one.count
            for one in self.outcomes
            if one.outcome is outcome and one.representable is representable
        )

    @property
    def ceiling(self) -> Share:
        """
        :return: Of the relations the ontology can hold, those the rule settles or leaves
            to the model with the true whole offered.
        """
        representable = sum(one.count for one in self.outcomes if one.representable)
        return Share(
            self.count(CandidateOutcome.TAKEN_RIGHT, representable=True)
            + self.count(CandidateOutcome.LEFT_TO_THE_MODEL, representable=True),
            representable,
        )

    def lines(self) -> list[str]:
        """
        :return: Every outcome with its count, and the ceiling, as lines of text.
        """
        heading = {True: "representable", False: "not representable", None: "no whole"}
        return [
            f"{heading[one.representable]:18s} {one.outcome.value:42s} {one.count}"
            for one in self.outcomes
        ] + [f"ceiling {self.ceiling.count}/{self.ceiling.total}"]


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for working out what the rule allows.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measured", type=Path, required=True, help="The directory of measured scenes"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="The directory holding every scene's ground truth",
    )
    parser.add_argument(
        "--output", type=Path, help="Where to write the counts, as JSON"
    )
    defaults = PipelineSettings()
    parser.add_argument(
        "--largest-gap-to-a-whole",
        type=float,
        default=defaults.largest_gap_to_a_whole,
        help="See PipelineSettings.largest_gap_to_a_whole",
    )
    parser.add_argument(
        "--largest-gap-counted-as-touching",
        type=float,
        default=defaults.largest_gap_counted_as_touching,
        help="See PipelineSettings.largest_gap_counted_as_touching",
    )
    parser.add_argument(
        "--confirm-wholes-within-the-gap",
        action=argparse.BooleanOptionalAction,
        default=defaults.confirm_wholes_within_the_gap,
        help="See PipelineSettings.confirm_wholes_within_the_gap",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Work out what the rule allows on every measured scene, with the true classes.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the counts are written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    settings = PipelineSettings(
        largest_gap_to_a_whole=parsed.largest_gap_to_a_whole,
        largest_gap_counted_as_touching=parsed.largest_gap_counted_as_touching,
        confirm_wholes_within_the_gap=parsed.confirm_wholes_within_the_gap,
    )
    load_annotation_modules()
    rule = FirstCandidateOfTheRule(
        step=MeasureScene(settings=settings, run=Run(directory=parsed.measured)),
        ontology=annotation_classes(SemanticAnnotation),
    )
    ceiling = RuleCeiling.of(
        SceneToJudge.all_in(parsed.measured, parsed.ground_truth),
        rule=rule,
        reading=TrueClasses(),
    )
    for line in ceiling.lines():
        logger.info(line)
    if parsed.output is not None:
        parsed.output.write_text(json.dumps(ceiling.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
