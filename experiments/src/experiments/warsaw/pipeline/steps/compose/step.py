"""
Settle what a proposed class is a kind of, as a question of its own.

A run composes a class by naming it and its superclass in one answer, and the two come
out with very different reliability: ``Ceiling`` was proposed by nine runs out of nine
under that name, while ``SoapDispenser`` has been built from ``Furniture``, from
``IsStorageSpace`` and from ``Cabinet``, and ``Tap`` from a handle and a joint five
times and from ``Aperture`` once. Telling a model to choose better did not work, even
where the prompt named the very pair it then composed.

The subsumption is therefore asked on its own, after the name is settled and before the
class is written: one question per class a run wants, carrying no pictures, since what
is being asked is whether one name is a kind of another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from semantic_digital_twin.adapters.vision_language_model.client import ModelResponse
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    MessagePart,
    TextPart,
)
from typing_extensions import Any, Dict, List, Optional

from krrood.ormatic.utils import classproperty

from experiments.warsaw.pipeline.asking import Prompt, QuestionAboutTheOntology
from experiments.warsaw.pipeline.records import SuperclassAnswer

# %% what a class is a kind of


@dataclass(kw_only=True)
class SuperclassChoice(QuestionAboutTheOntology[SuperclassAnswer]):
    """
    What one proposed class is a kind of, and what it can hold.
    """

    class_name: str
    """
    The class a run wants, which the ontology does not have.
    """

    proposed_bases: List[str] = field(default_factory=list)
    """
    What the step that named it proposed to build it from, shown rather than hidden: it
    is usually right, and the question is whether it is.
    """

    labels: List[str] = field(default_factory=list)
    """
    What the annotator called the objects the class was proposed for, which is the only
    evidence of what they are that this question carries.
    """

    @classproperty
    def prompt(cls) -> Prompt:
        """
        :return: The prompt this question is put with, both halves of it.
        """
        return Prompt.SUPERCLASS

    @property
    def key(self) -> str:
        return self.class_name

    @property
    def shows_pictures(self) -> bool:
        """
        :return: False. Whether one name is a kind of another is not something a render
            answers, and rendering for it would undo the saving that makes asking it
            affordable.
        """
        return False

    def message(self) -> List[MessagePart]:
        return [
            TextPart(
                self.templates.render_document(
                    self.message_template,
                    taxonomy=json.dumps(self.taxonomy),
                    class_name=self.class_name,
                    proposed_bases=self.proposed_bases,
                    labels=self.labels,
                )
            )
        ]

    def read(self, response: ModelResponse) -> SuperclassAnswer:
        answered: Dict[str, Any] = response.parse_json()
        return SuperclassAnswer(
            class_name=self.class_name,
            superclass=answered.get("superclass"),
            mixins=list(answered.get("mixins") or []),
            confidence=answered.get("confidence"),
            reason=answered.get("reason", ""),
        )

    def refusal(self, refused: ModelRefusedError) -> SuperclassAnswer:
        return SuperclassAnswer(class_name=self.class_name, problems=[str(refused)])

    def problems_with(self, answer: SuperclassAnswer) -> List[str]:
        """
        Say what is wrong with an answer, if anything.

        A superclass that is not in the ontology, or is the class being placed, cannot
        be built from, and would come back as a generated class deriving from nothing
        two steps later.

        :param answer: What came back.
        :return: What is wrong with it, empty when it is usable.
        """
        known = {node["name"] for node in self.taxonomy["classes"]}
        mixins = {node["name"] for node in self.taxonomy["part_whole_mixins"]}
        problems = []
        if not answer.superclass:
            problems.append("name a superclass from classes[]")
        elif answer.superclass == self.class_name:
            problems.append(f"{answer.superclass} cannot be its own superclass")
        elif answer.superclass not in known and answer.superclass not in mixins:
            problems.append(f"{answer.superclass} is not in the ontology")
        for mixin in answer.mixins:
            if mixin not in mixins:
                problems.append(f"{mixin} is not one of the part_whole_mixins")
        return problems

    def bases(self, answer: SuperclassAnswer) -> Optional[List[str]]:
        """
        :param answer: What came back.
        :return: What to build the class from, or None where nothing usable came back and
            what the earlier step proposed should stand.
        """
        if answer.problems or not answer.superclass:
            return None
        return [answer.superclass] + [
            mixin for mixin in answer.mixins if mixin != answer.superclass
        ]
