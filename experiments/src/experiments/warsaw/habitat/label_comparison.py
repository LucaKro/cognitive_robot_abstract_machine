"""
Lay a run's answers beside the dataset's own labels, object by object.

A count says how well a run did and never says where it went wrong. This writes what a
disagreement has to be judged from -- what the dataset called an object, what the run
answered about that same object, and whether the two were reconciled -- for every object
of a converted room, and gathers the disagreements so the largest is the first thing
read.

The two sides are compared pair by pair. HM3D numbers its objects and the converter names
each body after that number, so an answer and a label are already about the same object
and nothing has to be matched up first.

Every object is carried through, including one the run left unannotated, because a body
quietly dropped is a body the counts flatter.

python -m experiments.warsaw.habitat.label_comparison --run <run directory>

writes the comparison into that run, beside everything else it says.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import List, Optional, Tuple

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.label_vocabulary import (
    EmbeddingMatcher,
    LexicalMatcher,
    Matcher,
    spoken_class_name,
)
from experiments.warsaw.habitat.convert import CONVERTED_ROOM_FILE, ConvertedRoom
from experiments.warsaw.pipeline.records import Classifications
from experiments.warsaw.pipeline.run import Run, RunFile

COMPARISON_FILE = "label_comparison.md"
"""
What the comparison is written as, inside the run it is about.
"""

SCENE_RECORDS = "scene"
"""
Where a run keeps what its scene said about itself.
"""

# %% one object, as each side has it


@dataclass
class ComparedObject(JsonRecord):
    """
    One object of a room, as the dataset has it and as the run answered it.
    """

    segment: str
    """
    What both sides call it.
    """

    object_id: int
    """
    Which object of the whole scene this is, as HM3D numbers them.
    """

    truth: str
    """
    What the dataset's annotator called it.
    """

    predicted: Optional[str]
    """
    What the run answered, in the words a dataset would write it in, or None where the
    run left the body unannotated.
    """

    agrees: bool = False
    """
    Whether the answer and the label were judged to name the same kind of thing.

    Judged of this pair alone. Both sides are already about this one object, since HM3D
    numbers its objects and the converter names the body after that number, so there is
    nothing to match up first and no other label of the room has any say.
    """


# %% a whole room, compared


@dataclass
class LabelComparison(JsonRecord):
    """
    Every object of one converted room, as the dataset has it and as the run answered.
    """

    run: str = ""
    """
    The run this is of.
    """

    scene: str = ""
    """
    The building it was cut from.
    """

    room_id: int = 0
    """
    Which room of that building.
    """

    matcher: str = ""
    """
    What reconciled the two vocabularies, since the answer depends on it.
    """

    objects: List[ComparedObject] = field(default_factory=list)
    """
    Every object of the room.
    """

    @property
    def agreed(self) -> int:
        """
        :return: How many objects the two sides ended up naming the same.
        """
        return sum(1 for one in self.objects if one.agrees)

    def disagreements(self) -> List[Tuple[Tuple[str, Optional[str]], int]]:
        """
        :return: Per pair of what it was called and what it was answered, how many
            objects disagree that way, the largest first.
        """
        counted = Counter(
            (one.truth, one.predicted) for one in self.objects if not one.agrees
        )
        return counted.most_common()

    def as_markdown(self) -> str:
        """
        :return: The comparison as a page: what it is of, where the disagreement is, and
            then every object.
        """
        lines = [
            f"# {self.scene}, room {self.room_id}",
            "",
            f"Run `{self.run}`, vocabularies reconciled by {self.matcher or 'nothing'}.",
            "",
            f"**{self.agreed} of {len(self.objects)} objects agree.**",
            "",
            "## Where the disagreement is",
            "",
            "| called | answered | objects |",
            "|---|---|---:|",
        ]
        lines += [
            f"| {truth} | {predicted or '*left unannotated*'} | {count} |"
            for (truth, predicted), count in self.disagreements()
        ]
        lines += [
            "",
            "## Every object",
            "",
            "| object | called | answered | |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {one.segment} | {one.truth} | {one.predicted or '--'} "
            f"| {'ok' if one.agrees else 'no'} |"
            for one in self.objects
        ]
        return "\n".join(lines) + "\n"

    def write_beside(self, directory: Path) -> Path:
        """
        :param directory: The run to write into.
        :return: The file written.
        """
        written = Path(directory) / COMPARISON_FILE
        written.write_text(self.as_markdown())
        return written


# %% comparing one run


def compare_a_run(
    directory: Path, matcher: Optional[Matcher] = None, matcher_name: str = ""
) -> LabelComparison:
    """
    Read a run back against the room it was given.

    :param directory: The run's directory.
    :param matcher: What reconciles the two vocabularies, or None to compare the words
        as they stand.
    :param matcher_name: What to call that in the written comparison.
    :return: Every object of the room, as each side has it.
    """
    directory = Path(directory)
    room = ConvertedRoom.from_json(
        json.loads((directory / SCENE_RECORDS / CONVERTED_ROOM_FILE).read_text())
    )
    answered = {
        one.name: spoken_class_name(one.class_name) if one.class_name else None
        for one in Run(directory=directory)
        .read_record(RunFile.CLASSIFICATIONS, Classifications)
        .bodies
    }
    return LabelComparison(
        run=directory.name,
        scene=room.scene,
        room_id=room.room_id,
        matcher=matcher_name,
        objects=[
            ComparedObject(
                segment=one.segment,
                object_id=one.object_id,
                truth=one.label,
                predicted=answered.get(one.segment),
                agrees=names_the_same(answered.get(one.segment), one.label, matcher),
            )
            for one in room.objects
        ],
    )


def names_the_same(
    predicted: Optional[str], truth: str, matcher: Optional[Matcher]
) -> bool:
    """
    :param predicted: What the run answered about one object, or None where it answered
        nothing.
    :param truth: What the dataset's annotator called that same object.
    :param matcher: What reconciles the two vocabularies, or None to compare the words as
        they stand.
    :return: Whether the two name the same kind of thing.
    """
    if predicted is None:
        return False
    if matcher is None:
        return predicted == truth
    return matcher.means_the_same(predicted, truth)


# %% command-line entry point

MATCHERS = {"none": None, "wording": LexicalMatcher, "meaning": EmbeddingMatcher}
"""
What a comparison may reconcile the two vocabularies by.
"""


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for comparing a run against its room.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="The run directory to compare"
    )
    parser.add_argument(
        "--matcher",
        choices=sorted(MATCHERS),
        default="meaning",
        help="What reconciles the two vocabularies",
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Compare one run against the room it was given.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the comparison is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)
    building = MATCHERS[parsed.matcher]
    compared = compare_a_run(
        parsed.run,
        building() if building is not None else None,
        matcher_name=parsed.matcher,
    )
    written = compared.write_beside(parsed.run)
    logger = logging.getLogger(__name__)
    logger.info(
        "%s of %s objects agree; %s ways of disagreeing",
        compared.agreed,
        len(compared.objects),
        len(compared.disagreements()),
    )
    for (truth, predicted), count in compared.disagreements()[:5]:
        logger.info("  %2s  %s answered %s", count, truth, predicted or "nothing")
    logger.info("written to %s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
