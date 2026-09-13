"""
Deciding when a class of our ontology and a label of somebody else's dataset name the
same kind of thing.

The two vocabularies were written by different people for different purposes, so
counting them against each other without reconciling them first counts every difference
of wording as a mistake. What is checked here is that the reconciliation is the wording
alone: it closes the gap between ``kitchen cabinet`` and ``Cabinet``, and it does not
close the gap between a light fixture and the ``Decor`` our ontology coarsens it to,
which is a disagreement about the world and not about words.

The question is asked of one pair at a time. Each object of an HM3D room carries the
dataset's own object number, so what the run answered and what the annotator wrote are
already about the same object and there is nothing to match up first.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import Dict, Sequence, Tuple

from experiments.warsaw.evaluation.label_vocabulary import (
    EmbeddingMatcher,
    LexicalMatcher,
    spoken_class_name,
)

# %% saying a class name the way a dataset would


def test_a_class_name_is_said_as_words():
    """
    Our classes are written as one camel-cased word and theirs as several plain ones.
    """
    assert spoken_class_name("CoffeeMachine") == "coffee machine"
    assert spoken_class_name("TrashCan") == "trash can"
    assert spoken_class_name("Sink") == "sink"


# %% what counts as the same kind of thing


@pytest.fixture
def matcher() -> LexicalMatcher:
    """
    :return: The matcher, at its own defaults.
    """
    return LexicalMatcher()


def test_a_qualified_label_matches_the_class_it_qualifies(matcher):
    """
    HM3D says ``kitchen cabinet`` where the ontology says ``Cabinet``, and counting
    those apart would measure the annotator's wording rather than the answer.
    """
    assert matcher.means_the_same("kitchen cabinet", "cabinet")


def test_a_word_matches_the_same_word_inflected(matcher):
    """
    ``shelving`` and ``shelf`` are one thing said twice, which the stems settle.
    """
    assert matcher.means_the_same("shelving", "shelf")


def test_sharing_one_word_of_two_is_not_enough(matcher):
    """
    A coffee table is not a coffee machine.

    Sharing a word is the commonest way a purely lexical match goes wrong, so agreement
    is measured over all the words rather than taken from any one of them.
    """
    assert not matcher.means_the_same("coffee table", "coffee machine")


def test_a_name_means_itself(matcher):
    """
    The commonest case of all, and the one an exact answer has to keep scoring.
    """
    assert matcher.means_the_same("coffee table", "coffee table")


def test_a_different_idea_of_the_thing_is_left_unmatched(matcher):
    """
    The ontology has no light fixture and answers ``Decor``.

    That is a disagreement about how coarsely the world is carved, and closing it here
    would hide exactly what the evaluation exists to show.
    """
    assert not matcher.means_the_same("light fixture", "decor")


# %% a name is judged against its own label and no other


def test_a_nearer_name_elsewhere_in_the_room_does_not_take_the_answer():
    """
    The failure this replaced.

    A room holding both ``kitchen cabinet`` and ``cabinet`` used to score an answer of
    ``cabinet`` against whichever of them it resembled most, so a kitchen cabinet
    answered ``cabinet`` was counted wrong because another object in the room happened
    to be labelled ``cabinet``.

    The two sides are already about the same object, so only that object's own label is
    ever asked about.
    """
    matcher = matching_by_meaning(
        cabinet=turned(0), kitchen_cabinet=turned(20), sofa=turned(80)
    )
    assert matcher.means_the_same("cabinet", "kitchen cabinet")
    assert matcher.means_the_same("cabinet", "cabinet")
    assert not matcher.means_the_same("cabinet", "sofa")


# %% an encoder standing in for the one that has to be downloaded


@dataclass
class MeaningsGivenInAdvance:
    """
    An encoder whose meanings are decided by the test rather than by a model.

    Named for what it does rather than for the model it replaces: what is being checked
    is the rule applied to similarities, not the similarities themselves, and a test that
    downloaded a model to find that out would be testing the model.
    """

    placed: Dict[str, Tuple[float, float]]
    """
    Per name, where it sits on the unit circle.
    """

    def __call__(self, names: Sequence[str]) -> np.ndarray:
        """
        :param names: The names to place.
        :return: Their places, as unit vectors.
        """
        return np.array([self.placed[one] for one in names], dtype=float)


def turned(degrees: float) -> Tuple[float, float]:
    """
    :param degrees: How far from the first name to place another.
    :return: That place on the unit circle, so the two agree by the cosine of the angle.
    """
    return (np.cos(np.radians(degrees)), np.sin(np.radians(degrees)))


def matching_by_meaning(**placed: Tuple[float, float]) -> EmbeddingMatcher:
    """
    :param placed: Per name, where it sits, with underscores standing for spaces.
    :return: A matcher reading those meanings.
    """
    return EmbeddingMatcher(
        encoder=MeaningsGivenInAdvance(
            placed={one.replace("_", " "): where for one, where in placed.items()}
        )
    )


# %% what meaning settles that wording cannot


def test_two_names_meaning_the_same_match_without_sharing_a_word():
    """
    A fridge is a refrigerator, and no amount of looking at the letters says so.
    """
    matcher = matching_by_meaning(fridge=turned(0), refrigerator=turned(15))
    assert matcher.means_the_same("fridge", "refrigerator")


def test_two_names_merely_near_each_other_do_not_match():
    """
    A picture hanging on a wall is not the wall.

    Wording alone matched those two, because one name contains the other's word; meaning
    keeps them apart.
    """
    matcher = matching_by_meaning(wall_decor=turned(0), picture=turned(75))
    assert not matcher.means_the_same("wall decor", "picture")


def test_a_middling_likeness_needs_the_wording_to_agree_as_well():
    """
    Between the two thresholds the earlier study asked for a shared word as well, which
    is what stops a merely related word being read as the same one.
    """
    shared = matching_by_meaning(kitchen_counter=turned(0), counter_top=turned(50))
    assert shared.means_the_same("kitchen counter", "counter top")

    unshared = matching_by_meaning(kitchen_counter=turned(0), worktop=turned(50))
    assert not unshared.means_the_same("kitchen counter", "worktop")


def test_how_near_two_meanings_are_decides_it():
    """
    Near enough is one thing; further off is another, whatever else the room holds.

    The angles straddle :attr:`EmbeddingMatcher.settles_it`, which is the only thing that
    decides a pair now: ten degrees apart is a likeness of 0.98 and seventy is 0.34.
    """
    matcher = matching_by_meaning(
        stovetop=turned(0), cooktop=turned(10), oven=turned(70), sink=turned(85)
    )
    assert matcher.means_the_same("stovetop", "cooktop")
    assert not matcher.means_the_same("stovetop", "oven")
    assert not matcher.means_the_same("stovetop", "sink")


# %% the model itself, when it is there to be asked


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="the encoder has to be downloaded, which CI does not do",
)
def test_the_downloaded_encoder_reads_a_cooktop_as_a_stovetop():
    """
    The same rule against the real encoder, so the stand-in above is known to stand for
    something.
    """
    assert EmbeddingMatcher().means_the_same("cooktop", "stovetop")
    assert not EmbeddingMatcher().means_the_same("cooktop", "sink")
