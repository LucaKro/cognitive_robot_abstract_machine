"""
Reading a scene's labels as classes without asking a model: the protocol's own classes,
the class whose name is spelled most alike, and the class whose name means most alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from experiments.warsaw.articulate3d.exceptions import LabelWithSeveralClassesError
from experiments.warsaw.articulate3d.label_readings import (
    ClassVocabulary,
    NearestByEmbedding,
    NearestBySpelling,
    TextEmbedding,
    TrueClasses,
)

from .scenes import measured_kitchen_corner

# %% the classes a reading may answer with


@pytest.fixture(scope="module")
def vocabulary(taxonomy):
    """
    :return: The classes a run may give an object.
    """
    return ClassVocabulary.from_taxonomy(taxonomy)


def test_a_mixin_is_no_answer(vocabulary):
    assert "HasHandles" not in vocabulary.class_names


def test_a_category_is_no_answer(vocabulary, taxonomy):
    categories = [one["name"] for one in taxonomy["classes"] if one.get("abstract")]

    assert categories
    assert not set(categories) & set(vocabulary.class_names)


def test_a_class_of_the_ontology_is_an_answer(vocabulary):
    assert "Cabinet" in vocabulary.class_names


def test_a_class_name_is_read_as_words():
    assert ClassVocabulary.words_of("ElectricalDevice") == "electrical device"


# %% the protocol's own classes


def test_the_true_classes_are_the_protocol_s_classes_of_each_label(ground_truth):
    read = TrueClasses().classes_of(measured_kitchen_corner(), ground_truth)

    assert read == {
        "cabinet": "Cabinet",
        "door": "Door",
        "handle": "Handle",
        "drawer": "Drawer",
        "switch_base": "ElectricalDevice",
        "control": None,
    }


def test_a_label_whose_parts_have_two_true_classes_is_refused(ground_truth):
    """
    A reading by label gives every part with one label one class, so it cannot stand for
    a protocol that classified one of them otherwise on sight.
    """
    reclassified = replace(
        ground_truth,
        parts=[
            replace(one, semantic_class="Knob") if one.part_id == 5 else one
            for one in ground_truth.parts
        ],
    )

    with pytest.raises(LabelWithSeveralClassesError):
        TrueClasses().classes_of(measured_kitchen_corner(), reclassified)


# %% the class spelled most alike


def test_a_label_spelled_as_a_class_is_read_as_that_class(vocabulary, ground_truth):
    read = NearestBySpelling(vocabulary=vocabulary).classes_of(
        measured_kitchen_corner(), ground_truth
    )

    assert (read["door"], read["drawer"]) == ("Door", "Drawer")


# %% the class meaning most alike


@dataclass
class EmbeddingByWordList(TextEmbedding):
    """
    Each text embedded as the one direction its known word stands for.
    """

    directions: dict[str, list[float]] = field(default_factory=dict)
    """
    The direction each known text points in.
    """

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray([self.directions[text] for text in texts], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_a_label_is_read_as_the_class_whose_name_means_most_alike(ground_truth):
    embedding = EmbeddingByWordList(
        directions={
            "cabinet": [1.0, 0.0, 0.0],
            "door": [0.0, 1.0, 0.0],
            "handle": [0.0, 0.1, 1.0],
            "drawer": [0.9, 0.1, 0.0],
            "switch base": [0.0, 0.9, 0.1],
            "control": [0.1, 0.0, 1.0],
        }
    )
    reading = NearestByEmbedding(
        vocabulary=ClassVocabulary(class_names=["Cabinet", "Door", "Handle"]),
        embedding=embedding,
    )

    read = reading.classes_of(measured_kitchen_corner(), ground_truth)

    assert (read["drawer"], read["switch_base"], read["control"]) == (
        "Cabinet",
        "Door",
        "Handle",
    )
