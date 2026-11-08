"""
Reading a scene's labels as classes of the ontology without asking a model.

A run reads each label once and gives every part carrying it that class. The readings
here do the same without a model: the protocol's own classes, which is what a perfect
reading would give; the class whose name is spelled most alike; and the class whose name
means most alike, by a sentence embedding.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cached_property
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from typing_extensions import Any

from experiments.warsaw.articulate3d.exceptions import LabelWithSeveralClassesError
from experiments.warsaw.articulate3d.ground_truth import SceneGroundTruth
from experiments.warsaw.articulate3d.measured_scene import MeasuredScene
from experiments.warsaw.articulate3d.score import TaxonomyKey

# %% the classes a reading may answer with


@dataclass
class ClassVocabulary:
    """
    The classes a run may give an object: those of the ontology that are neither mixins
    nor categories.
    """

    class_names: list[str] = field(default_factory=list)
    """
    The classes, by name.
    """

    @classmethod
    def from_taxonomy(cls, taxonomy: dict[str, Any]) -> ClassVocabulary:
        """
        :param taxonomy: The ontology as a run is given it.
        :return: The classes it offers as answers.
        """
        return cls(
            class_names=[
                one[TaxonomyKey.NAME]
                for one in taxonomy[TaxonomyKey.CLASSES]
                if not one.get(TaxonomyKey.MIXIN) and not one.get(TaxonomyKey.ABSTRACT)
            ]
        )

    @staticmethod
    def words_of(class_name: str) -> str:
        """
        :param class_name: A class name, written in camel case.
        :return: The words it is made of, in lower case.
        """
        return re.sub(r"(?<!^)(?=[A-Z])", " ", class_name).lower()

    @cached_property
    def class_words(self) -> list[str]:
        """
        :return: The words of each class name, in the order of the names.
        """
        return [self.words_of(one) for one in self.class_names]


# %% a reading of labels


class LabelReading(ABC):
    """
    A way of reading every label of a scene as a class.
    """

    @abstractmethod
    def classes_of(
        self, scene: MeasuredScene, ground_truth: SceneGroundTruth
    ) -> dict[str, str | None]:
        """
        :param scene: The measured scene.
        :param ground_truth: The scene's ground truth, which only the reading standing for
            the truth looks at.
        :return: Per label, the class every part carrying it is read as, or nothing.
        """


@dataclass
class TrueClasses(LabelReading):
    """
    The protocol's class of each label: what a perfect reading would give.
    """

    def classes_of(
        self, scene: MeasuredScene, ground_truth: SceneGroundTruth
    ) -> dict[str, str | None]:
        """
        A label none of whose parts the ground truth keeps is read as no class.

        :raises LabelWithSeveralClassesError: If the parts carrying one label have
            different true classes.
        """
        true_class = {one.part_id: one.semantic_class for one in ground_truth.parts}
        label_of_segment = scene.label_of_segment
        found: dict[str, set[str | None]] = {}
        for part in scene.converted.parts:
            classes = found.setdefault(label_of_segment[part.segment], set())
            if part.part_id in true_class:
                classes.add(true_class[part.part_id])
        for label, classes in found.items():
            if len(classes) > 1:
                raise LabelWithSeveralClassesError(
                    scene_id=scene.converted.scene_id,
                    label=label,
                    classes=sorted(classes, key=str),
                )
        return {label: next(iter(classes), None) for label, classes in found.items()}


@dataclass
class NearestClass(LabelReading, ABC):
    """
    A reading that gives each label the class of the vocabulary its words are nearest.
    """

    vocabulary: ClassVocabulary
    """
    The classes a label may be read as.
    """

    def classes_of(
        self, scene: MeasuredScene, ground_truth: SceneGroundTruth
    ) -> dict[str, str | None]:
        labels = sorted(set(scene.label_of_segment.values()))
        return dict(
            zip(labels, self.nearest([label.replace("_", " ") for label in labels]))
        )

    @abstractmethod
    def nearest(self, texts: list[str]) -> list[str]:
        """
        :param texts: The words of each label.
        :return: For each, the name of the class its words are nearest.
        """


@dataclass
class NearestBySpelling(NearestClass):
    """
    The class whose name is spelled most like the label.
    """

    def nearest(self, texts: list[str]) -> list[str]:
        return [
            max(
                self.vocabulary.class_names,
                key=lambda name: SequenceMatcher(
                    None, text, self.vocabulary.words_of(name)
                ).ratio(),
            )
            for text in texts
        ]


# %% the class whose name means most alike


class TextEmbedding(ABC):
    """
    A way of placing texts in a space where texts meaning alike lie close.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        :param texts: The texts to place.
        :return: One vector of unit length per text, one row each.
        """


@dataclass
class SentenceEmbedding(TextEmbedding):
    """
    A Sentence-BERT model, its token vectors averaged over each text.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """
    The model on the Hugging Face hub, fetched once and kept in its cache.
    """

    @cached_property
    def tokenizer(self) -> Any:
        """
        :return: The model's tokenizer.
        """
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def model(self) -> Any:
        """
        :return: The model, ready to embed.
        """
        return AutoModel.from_pretrained(self.model_name).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        averaged = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(averaged, dim=1).numpy()


@dataclass
class NearestByEmbedding(NearestClass):
    """
    The class whose name means most like the label.
    """

    embedding: TextEmbedding = field(default_factory=SentenceEmbedding)
    """
    How the label and the class names are placed to be compared.
    """

    @cached_property
    def class_vectors(self) -> np.ndarray:
        """
        :return: The vector of each class name, in the order of the names.
        """
        return self.embedding.embed(self.vocabulary.class_words)

    def nearest(self, texts: list[str]) -> list[str]:
        similarity = self.embedding.embed(texts) @ self.class_vectors.T
        return [self.vocabulary.class_names[one] for one in similarity.argmax(axis=1)]
