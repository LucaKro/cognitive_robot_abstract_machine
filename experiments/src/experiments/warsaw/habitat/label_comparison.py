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

python -m experiments.warsaw.habitat.label_comparison --runs <run directory> ...

writes the comparison into each run, beside everything else it says, once per matcher --
by default both the meaning matcher and the head-noun matcher, so either set of numbers is
there to be read. ``--summary <file>`` adds every run up, per building and over all, into
one page. A list of runs may be given as ``@<file>``, one run per line.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from typing_extensions import Dict, List, Optional, Tuple

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.label_vocabulary import (
    EmbeddingMatcher,
    HeadNounMatcher,
    LexicalMatcher,
    Matcher,
    spoken_class_name,
)
from experiments.warsaw.habitat.convert import CONVERTED_ROOM_FILE, ConvertedRoom
from experiments.warsaw.pipeline.records import Classifications
from experiments.warsaw.pipeline.run import Run, RunFile

COMPARISON_NAME = "label_comparison"
"""
What a comparison is written under inside the run it is about, before the matcher that made
it.
"""

SCENE_RECORDS = "scene"
"""
Where a run keeps what its scene said about itself.
"""

# %% what may reconcile the two vocabularies


class ComparisonMatcher(StrEnum):
    """
    What a comparison may reconcile the two vocabularies by.
    """

    NONE = "none"
    """
    Nothing: the words have to be the same.
    """

    WORDING = "wording"
    """
    The words, through :class:`LexicalMatcher`.
    """

    MEANING = "meaning"
    """
    What the names mean, through :class:`EmbeddingMatcher`.
    """

    HEAD_NOUN = "head-noun"
    """
    What the names mean, refusing a shared word that names different things, through
    :class:`HeadNounMatcher`.
    """

    @property
    def page(self) -> str:
        """
        :return: The file a comparison by this matcher is written as, so that two matchers
            compared on one run keep a comparison each.
        """
        return f"{COMPARISON_NAME}_{self.value}.md"

    @property
    def record(self) -> str:
        """
        :return: The file the same comparison is kept in as data.
        """
        return f"{COMPARISON_NAME}_{self.value}.json"

    def built(self, meaning: EmbeddingMatcher) -> Optional[Matcher]:
        """
        :param meaning: The meaning matcher to use wherever one is needed, so that its
            encoder is loaded once however many matchers read meanings.
        :return: The matcher, or None where the words are compared as they stand.
        """
        if self is ComparisonMatcher.NONE:
            return None
        if self is ComparisonMatcher.WORDING:
            return LexicalMatcher()
        if self is ComparisonMatcher.MEANING:
            return meaning
        return HeadNounMatcher(meaning=meaning)


# %% how well a set of answers found what it was meant to


@dataclass(frozen=True)
class ClassificationScores(JsonRecord):
    """
    Precision, recall and their harmonic mean.

    An empty ratio is zero: a label nothing was answered as has no precision to speak of,
    and averaging it in as zero is what a macro average over labels does.
    """

    precision: float = 0.0
    """
    The share of what was answered that was right.
    """

    recall: float = 0.0
    """
    The share of what there was to find that was found.
    """

    f1_score: float = 0.0
    """
    The harmonic mean of the two.
    """

    @classmethod
    def from_counts(
        cls, true_positives: int, false_positives: int, false_negatives: int
    ) -> ClassificationScores:
        """
        :param true_positives: Answers that were right.
        :param false_positives: Answers that were wrong.
        :param false_negatives: What there was to find and was not found.
        :return: The scores those counts come to.
        """
        precision = cls.ratio(true_positives, true_positives + false_positives)
        recall = cls.ratio(true_positives, true_positives + false_negatives)
        return cls(
            precision=precision,
            recall=recall,
            f1_score=cls.ratio(2 * precision * recall, precision + recall),
        )

    @classmethod
    def mean_of(cls, scores: List[ClassificationScores]) -> ClassificationScores:
        """
        :param scores: Scores to average, one per label.
        :return: Each score averaged, every label weighing the same.
        """
        return cls(
            precision=cls.ratio(sum(one.precision for one in scores), len(scores)),
            recall=cls.ratio(sum(one.recall for one in scores), len(scores)),
            f1_score=cls.ratio(sum(one.f1_score for one in scores), len(scores)),
        )

    @staticmethod
    def ratio(numerator: float, denominator: float) -> float:
        """
        :param numerator: The value above the division line.
        :param denominator: The value below it.
        :return: Their ratio, or zero where there is nothing to divide by.
        """
        return numerator / denominator if denominator else 0.0


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

    means: List[str] = field(default_factory=list)
    """
    The labels of the room the answer names the same kind of thing as.

    What a false positive of another label is counted from: an answer that means
    ``cabinet`` for an object labelled ``sink`` is a cabinet found where there was none.
    """

    agrees: bool = False
    """
    Whether the answer and the label were judged to name the same kind of thing.

    Judged of this pair alone. Both sides are already about this one object, since HM3D
    numbers its objects and the converter names the body after that number, so there is
    nothing to match up first and no other label of the room has any say.
    """

    far_apart: bool = False
    """
    Whether the answer is further from the label in meaning than the matcher reads the
    wording for, so that the matcher could not have credited it however right it is.

    What sets a coarse answer apart from a judged one: ``wall decor`` for a picture is
    unrelated as words and near as a concept. An object given no answer is not far apart,
    since there is no meaning to be far from.
    """


# %% one label, counted


@dataclass(frozen=True)
class LabelTally(JsonRecord):
    """
    How often one label was found, missed and taken for another object.
    """

    label: str
    """
    The label, as the dataset writes it.
    """

    objects: int = 0
    """
    How many objects carry it.
    """

    agreed: int = 0
    """
    How many of those were answered with something that agrees with it.
    """

    false_positives: int = 0
    """
    How many objects carrying another label were answered with something meaning this one.
    """

    @property
    def missed(self) -> int:
        """
        :return: How many objects carrying it were not answered as it.
        """
        return self.objects - self.agreed

    @property
    def scores(self) -> ClassificationScores:
        """
        :return: The scores of finding this label.
        """
        return ClassificationScores.from_counts(
            true_positives=self.agreed,
            false_positives=self.false_positives,
            false_negatives=self.missed,
        )

    def added_to(self, other: LabelTally) -> LabelTally:
        """
        :param other: The same label's tally in another room.
        :return: Both rooms' counts together.
        """
        return LabelTally(
            label=self.label,
            objects=self.objects + other.objects,
            agreed=self.agreed + other.agreed,
            false_positives=self.false_positives + other.false_positives,
        )


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
    def rooms(self) -> ComparedRooms:
        """
        :return: This room as a set of one, which is what its scores are counted over.
        """
        return ComparedRooms(comparisons=[self])

    @property
    def agreed(self) -> int:
        """
        :return: How many objects the two sides ended up naming the same.
        """
        return self.rooms.agreed

    @property
    def tallies(self) -> Dict[str, LabelTally]:
        """
        :return: Per label of the room, how often it was found, missed and taken for
            another object: an object carrying it whose answer agrees is found, one whose
            answer does not is missed, and an object carrying another label whose answer
            names it is a false positive.
        """
        return {
            label: LabelTally(
                label=label,
                objects=sum(1 for one in self.objects if one.truth == label),
                agreed=sum(
                    1 for one in self.objects if one.truth == label and one.agrees
                ),
                false_positives=sum(
                    1
                    for one in self.objects
                    if one.truth != label and label in one.means
                ),
            )
            for label in sorted({one.truth for one in self.objects})
        }

    @property
    def per_object(self) -> ClassificationScores:
        """
        :return: The scores over objects: precision over the objects that were answered,
            recall over every object. Recall is the share that agrees.
        """
        return self.rooms.per_object

    @property
    def per_label(self) -> Dict[str, ClassificationScores]:
        """
        :return: Per label of the room, the scores of finding it.
        """
        return {label: tally.scores for label, tally in self.tallies.items()}

    @property
    def per_label_average(self) -> ClassificationScores:
        """
        :return: The scores averaged over the room's labels, so that a rare label weighs
            as much as a common one.
        """
        return self.rooms.per_label_average

    def decided(self) -> LabelComparison:
        """
        :return: The room with the objects whose answer is far from their label in meaning
            left out, so that what is scored is only what the matcher could judge.
        """
        return replace(self, objects=[one for one in self.objects if not one.far_apart])

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
            "| | precision | recall | F1 |",
            "|---|---:|---:|---:|",
            self.scores_row("per object", self.per_object),
            self.scores_row("per label, averaged", self.per_label_average),
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

    @staticmethod
    def scores_row(name: str, scores: ClassificationScores) -> str:
        """
        :param name: What the scores are over.
        :param scores: The scores.
        :return: One row of the page's table of scores.
        """
        return (
            f"| {name} | {scores.precision:.3f} | {scores.recall:.3f} "
            f"| {scores.f1_score:.3f} |"
        )

    def write_beside(self, directory: Path) -> Path:
        """
        Write the comparison into the run, as a page and as data, under the matcher that
        made it.

        :param directory: The run to write into.
        :return: The page written.
        """
        matcher = ComparisonMatcher(self.matcher)
        written = Path(directory) / matcher.page
        written.write_text(self.as_markdown())
        (Path(directory) / matcher.record).write_text(
            json.dumps(self.to_json(), indent=2)
        )
        return written


# %% several rooms, added up


@dataclass(frozen=True)
class LabelGroup:
    """
    The labels carried by a range of object counts, for asking whether rare labels fare
    differently from common ones.
    """

    fewest: int
    """
    The fewest objects a label of the group is carried by.
    """

    most: Optional[int] = None
    """
    The most objects a label of the group is carried by, or None where there is no most.
    """

    @property
    def name(self) -> str:
        """
        :return: The range, as a reader writes it.
        """
        if self.most is None:
            return f"{self.fewest} or more"
        if self.fewest == self.most:
            return f"{self.fewest}"
        return f"{self.fewest} to {self.most}"


LABEL_GROUPS = (
    LabelGroup(fewest=1, most=1),
    LabelGroup(fewest=2, most=5),
    LabelGroup(fewest=6, most=20),
    LabelGroup(fewest=21),
)
"""
How the labels of a summary are grouped by how many objects carry them.
"""

MOST_MISSED_SHOWN = 10
"""
How many of the most-missed labels a summary lists.
"""


@dataclass
class ComparedRooms:
    """
    Several compared rooms, added up as one vocabulary.

    A label seen in several rooms is one label: its counts are added across the rooms
    before it is scored, and the average over labels is over the distinct labels.
    """

    comparisons: List[LabelComparison] = field(default_factory=list)
    """
    The rooms, each compared by the same matcher.
    """

    @property
    def objects(self) -> int:
        """
        :return: How many objects the rooms hold.
        """
        return sum(len(one.objects) for one in self.comparisons)

    @property
    def agreed(self) -> int:
        """
        :return: How many objects agree with their label.
        """
        return sum(1 for one in self.comparisons for each in one.objects if each.agrees)

    @property
    def answered(self) -> int:
        """
        :return: How many objects were given an answer at all.
        """
        return sum(
            1
            for one in self.comparisons
            for each in one.objects
            if each.predicted is not None
        )

    @property
    def far_apart(self) -> int:
        """
        :return: How many objects were answered with something far from their label in
            meaning.
        """
        return sum(
            1 for one in self.comparisons for each in one.objects if each.far_apart
        )

    @property
    def per_object(self) -> ClassificationScores:
        """
        :return: The scores over objects: precision over the objects that were answered,
            recall over every object.
        """
        return ClassificationScores.from_counts(
            true_positives=self.agreed,
            false_positives=self.answered - self.agreed,
            false_negatives=self.objects - self.agreed,
        )

    @property
    def tallies(self) -> Dict[str, LabelTally]:
        """
        :return: Per distinct label, its counts added across every room.
        """
        added: Dict[str, LabelTally] = {}
        for comparison in self.comparisons:
            for label, tally in comparison.tallies.items():
                added[label] = added[label].added_to(tally) if label in added else tally
        return added

    @property
    def false_positives(self) -> int:
        """
        :return: How many times an answer was taken for a label other than its object's.
        """
        return sum(tally.false_positives for tally in self.tallies.values())

    @property
    def per_label_average(self) -> ClassificationScores:
        """
        :return: The scores averaged over the distinct labels, every label weighing the
            same.
        """
        return ClassificationScores.mean_of(
            [tally.scores for tally in self.tallies.values()]
        )

    def decided(self) -> ComparedRooms:
        """
        :return: The rooms with every far-apart object left out.
        """
        return ComparedRooms(comparisons=[one.decided() for one in self.comparisons])

    def labels_carried_by(self, group: LabelGroup) -> List[LabelTally]:
        """
        :param group: How many objects a label may be carried by.
        :return: The labels carried by that many objects across the rooms.
        """
        return [
            tally
            for tally in self.tallies.values()
            if group.fewest <= tally.objects
            and (group.most is None or tally.objects <= group.most)
        ]

    def most_missed(self, count: int) -> List[LabelTally]:
        """
        :param count: How many labels to return.
        :return: The labels missed most often, the most missed first.
        """
        return sorted(self.tallies.values(), key=lambda tally: -tally.missed)[:count]

    def by_scene(self) -> Dict[str, ComparedRooms]:
        """
        :return: The rooms grouped by the building each was cut from.
        """
        grouped: Dict[str, List[LabelComparison]] = defaultdict(list)
        for comparison in self.comparisons:
            grouped[comparison.scene].append(comparison)
        return {
            scene: ComparedRooms(comparisons=grouped[scene])
            for scene in sorted(grouped)
        }


# %% every number of a set of runs, on one page


@dataclass
class ScoreSummary:
    """
    Every number a set of compared runs comes to under one matcher, per building and over
    all of them.
    """

    matcher: ComparisonMatcher
    """
    What reconciled the two vocabularies.
    """

    rooms: ComparedRooms
    """
    Every room compared.
    """

    def as_markdown(self) -> str:
        """
        :return: The scores over every object and over the objects the matcher could
            judge, then how labels fare by how many objects carry them, then the labels
            missed most.
        """
        scenes = {**self.rooms.by_scene(), "all": self.rooms}
        lines = [
            f"# Scores by the {self.matcher.value} matcher",
            "",
            f"{len(self.rooms.comparisons)} rooms.",
            "",
            "## Over every object",
            "",
            "| scene | rooms | objects | agree | P / R / F1 per object | labels "
            "| P / R / F1 per label | false positives | far apart |",
            "|---|---:|---:|---:|---|---:|---|---:|---:|",
        ]
        lines += [
            f"| {scene} | {len(rooms.comparisons)} | {rooms.objects} "
            f"| {rooms.agreed} ({self.share(rooms.agreed, rooms.objects)}) "
            f"| {self.scores(rooms.per_object)} | {len(rooms.tallies)} "
            f"| {self.scores(rooms.per_label_average)} | {rooms.false_positives} "
            f"| {rooms.far_apart} ({self.share(rooms.far_apart, rooms.objects)}) |"
            for scene, rooms in scenes.items()
        ]
        lines += [
            "",
            "## Over the objects not far apart",
            "",
            "| scene | objects | agree | P / R / F1 per object | labels "
            "| P / R / F1 per label |",
            "|---|---:|---:|---|---:|---|",
        ]
        for scene, rooms in scenes.items():
            decided = rooms.decided()
            lines.append(
                f"| {scene} | {decided.objects} "
                f"| {decided.agreed} ({self.share(decided.agreed, decided.objects)}) "
                f"| {self.scores(decided.per_object)} | {len(decided.tallies)} "
                f"| {self.scores(decided.per_label_average)} |"
            )
        lines += [
            "",
            "## Labels by how many objects carry them",
            "",
            "| objects per label | labels | objects | agree |",
            "|---|---:|---:|---:|",
        ]
        for group in LABEL_GROUPS:
            tallies = self.rooms.labels_carried_by(group)
            objects = sum(tally.objects for tally in tallies)
            agreed = sum(tally.agreed for tally in tallies)
            lines.append(
                f"| {group.name} | {len(tallies)} | {objects} "
                f"| {agreed} ({self.share(agreed, objects)}) |"
            )
        missed = self.rooms.objects - self.rooms.agreed
        lines += [
            "",
            f"## The labels missed most, of {missed} misses",
            "",
            "| label | missed | of objects | share of all misses, running |",
            "|---|---:|---:|---:|",
        ]
        running = 0
        for tally in self.rooms.most_missed(MOST_MISSED_SHOWN):
            running += tally.missed
            lines.append(
                f"| {tally.label} | {tally.missed} | {tally.objects} "
                f"| {self.share(running, missed)} |"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def share(part: int, whole: int) -> str:
        """
        :param part: A count.
        :param whole: What it is a count of.
        :return: The share, as a percentage.
        """
        return f"{ClassificationScores.ratio(part, whole):.1%}"

    @staticmethod
    def scores(scores: ClassificationScores) -> str:
        """
        :param scores: Precision, recall and F1.
        :return: The three, as one cell.
        """
        return f"{scores.precision:.3f} / {scores.recall:.3f} / {scores.f1_score:.3f}"


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
    labels = sorted({one.label for one in room.objects})
    meanings = {
        predicted: [
            label for label in labels if names_the_same(predicted, label, matcher)
        ]
        for predicted in set(answered.values())
        if predicted is not None
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
                means=meanings.get(answered.get(one.segment), []),
                agrees=names_the_same(answered.get(one.segment), one.label, matcher),
                far_apart=names_far_apart(
                    answered.get(one.segment), one.label, matcher
                ),
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


def names_far_apart(
    predicted: Optional[str], truth: str, matcher: Optional[Matcher]
) -> bool:
    """
    :param predicted: What the run answered about one object, or None where it answered
        nothing.
    :param truth: What the dataset's annotator called that same object.
    :param matcher: What reconciles the two vocabularies, or None to compare the words as
        they stand.
    :return: Whether the answer is too far from the label in meaning for the matcher to
        judge. Never, where nothing was answered or nothing reads meanings.
    """
    if predicted is None or matcher is None:
        return False
    return matcher.far_apart(predicted, truth)


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for comparing runs against their rooms.
    """
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="The run directories"
    )
    parser.add_argument(
        "--matchers",
        nargs="+",
        choices=[one.value for one in ComparisonMatcher],
        default=[ComparisonMatcher.MEANING.value, ComparisonMatcher.HEAD_NOUN.value],
        help="What reconciles the two vocabularies, one comparison per matcher",
    )
    parser.add_argument(
        "--summary", type=Path, help="Where to write every run added up, per matcher"
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Compare runs against the rooms they were given, once per matcher.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once every comparison is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    meaning = EmbeddingMatcher()
    summaries = []
    for chosen in [ComparisonMatcher(one) for one in parsed.matchers]:
        matcher = chosen.built(meaning)
        compared = []
        for run in parsed.runs:
            comparison = compare_a_run(run, matcher, matcher_name=chosen.value)
            comparison.write_beside(run)
            compared.append(comparison)
            logger.info(
                "%s, %s: %s of %s objects agree, %s far apart",
                run.name,
                chosen.value,
                comparison.agreed,
                len(comparison.objects),
                comparison.rooms.far_apart,
            )
        summaries.append(
            ScoreSummary(matcher=chosen, rooms=ComparedRooms(comparisons=compared))
        )
    if parsed.summary is not None:
        parsed.summary.write_text("\n".join(one.as_markdown() for one in summaries))
        logger.info("written to %s", parsed.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
