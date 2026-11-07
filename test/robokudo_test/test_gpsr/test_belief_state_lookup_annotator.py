"""
Tests for answering a recall query with the remembered persons and objects.
"""

from __future__ import annotations

# %% Imports
import json

import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.belief_state import (
    BeliefEntry,
    BeliefState,
    EntityType,
)
from robokudo.annotators.gpsr.belief_state_lookup_annotator import (
    BeliefStateLookupAnnotator,
)
from typing_extensions import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene

# %% remembered things


@pytest.fixture
def remembered(belief_state: BeliefState) -> BeliefState:
    """
    :param belief_state: The empty belief state.
    :return: A belief state that remembers two persons, a cup and a bowl.
    """
    first_person = BeliefEntry(1, EntityType.PERSON, "human")
    second_person = BeliefEntry(2, EntityType.PERSON, "human")
    cup = BeliefEntry(3, EntityType.OBJECT, "cup")
    cup.yolo_label = "cup"
    bowl = BeliefEntry(4, EntityType.OBJECT, "bowl_white")
    bowl.yolo_label = "white_bowl"
    bowl.semantic_labels = ["Bowl"]
    for entry in (first_person, second_person, cup, bowl):
        belief_state.entries[entry.id] = entry
    return belief_state


def answered_entries(scene: GpsrScene) -> dict[str, Any] | list[dict[str, Any]]:
    """
    :param scene: The scene that received the answer.
    :return: The entries the recall query was answered with.
    """
    return json.loads(scene.sent_answer().text_result)


def no_entries_answer() -> dict[str, str]:
    """
    :return: The answer to a recall query that finds nothing.
    """
    return {"error": "no entries found"}


# %% recall by id


def test_id_recalls_its_single_entry(scene: GpsrScene, remembered: BeliefState) -> None:
    """
    ID recalls its single entry.
    """
    scene.query.obj.type = "recall"
    scene.query.obj.uid = "3"
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert answered_entries(scene) == remembered.find_by_id(3).to_dict()


def test_id_surrounded_by_spaces_recalls_its_entry(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    ID surrounded by spaces recalls its entry.
    """
    scene.query.obj.uid = " 3 "
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == remembered.find_by_id(3).to_dict()


def test_unknown_id_recalls_nothing(scene: GpsrScene, remembered: BeliefState) -> None:
    """
    Unknown ID recalls nothing.
    """
    scene.query.obj.uid = "99"
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == no_entries_answer()


def test_id_that_is_no_number_recalls_nothing(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    An ID that is not a number recalls nothing.
    """
    scene.query.obj.uid = "abc"
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == no_entries_answer()


def test_id_takes_precedence_over_the_description(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    ID takes precedence over the description.
    """
    scene.query.obj.uid = "3"
    scene.query.obj.description = ["human"]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == remembered.find_by_id(3).to_dict()


# %% recall by description


@pytest.mark.parametrize("person_word", ["human", "person"])
def test_person_word_recalls_every_person(
    scene: GpsrScene, remembered: BeliefState, person_word: str
) -> None:
    """
    The descriptions "human" and "person" recall every person.
    """
    scene.query.obj.description = [person_word]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == [
        person.to_dict() for person in remembered.find_persons()
    ]


def test_class_recalls_its_entry(scene: GpsrScene, remembered: BeliefState) -> None:
    """
    Class recalls its entry.
    """
    scene.query.obj.description = ["cup"]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == remembered.find_by_id(3).to_dict()


def test_detector_label_recalls_its_entry(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    Detector label recalls its entry.
    """
    scene.query.obj.description = ["white_bowl"]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == remembered.find_by_id(4).to_dict()


def test_semantic_label_recalls_its_entry(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    Semantic label recalls its entry.
    """
    scene.query.obj.description = ["Bowl"]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == remembered.find_by_id(4).to_dict()


def test_class_takes_precedence_over_a_detector_label(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    Class takes precedence over a detector label.
    """
    cup = remembered.find_by_id(3)
    mug_detected_as_cup = BeliefEntry(5, EntityType.OBJECT, "mug")
    mug_detected_as_cup.yolo_label = cup.classname
    remembered.entries[mug_detected_as_cup.id] = mug_detected_as_cup
    scene.query.obj.description = [cup.classname]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == cup.to_dict()


def test_detector_label_takes_precedence_over_a_semantic_label(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    Detector label takes precedence over a semantic label.
    """
    bowl = remembered.find_by_id(4)
    dish_labeled_as_bowl = BeliefEntry(5, EntityType.OBJECT, "dish")
    dish_labeled_as_bowl.semantic_labels = [bowl.yolo_label]
    remembered.entries[dish_labeled_as_bowl.id] = dish_labeled_as_bowl
    scene.query.obj.description = [bowl.yolo_label]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == bowl.to_dict()


def test_unknown_description_recalls_nothing(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    Unknown description recalls nothing.
    """
    scene.query.obj.description = ["spaceship"]
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == no_entries_answer()


# %% recall of everything


def test_empty_query_recalls_every_entry(
    scene: GpsrScene, remembered: BeliefState
) -> None:
    """
    Empty query recalls every entry.
    """
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert answered_entries(scene) == [entry.to_dict() for entry in remembered]


def test_empty_memory_recalls_nothing(scene: GpsrScene) -> None:
    """
    Empty memory recalls nothing.
    """
    annotator = BeliefStateLookupAnnotator()
    scene.place_in_pipeline(annotator)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert answered_entries(scene) == no_entries_answer()
