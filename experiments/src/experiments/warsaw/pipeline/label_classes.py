"""
The classes a run's vocabulary answers name.

An answer is either a class of the ontology or a class proposed by naming the superclass
and mixins it is composed of. The second does not exist until the last step of the run
generates it, and what it can hold is decided by exactly that composition: a
``KitchenIsland`` composed with ``HasDrawers`` admits the drawers overlapping it as
parts, one without it admits nothing, and every later step turns on that.

So a proposal is *composed* here rather than looked up, and the composed class stands in
for it wherever the run asks the ontology a question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    compose_class,
    relations_of,
)
from typing_extensions import Any, Dict, Optional, Type

from experiments.warsaw.pipeline.records import LabelAnswer, Vocabulary
from experiments.warsaw.pipeline.reporting import Reporting


@dataclass
class VocabularyClasses(Reporting):
    """
    What each of a scene's labels stands for, as a class rather than as a name.
    """

    vocabulary: Vocabulary
    """
    What was answered about each label.
    """

    known: Dict[str, Type] = field(default_factory=dict)
    """
    The ontology's classes by name.
    """

    proposal_marker: str = "proposed_for_label"
    """
    What marks a class in a widened taxonomy as one this run proposed rather than one
    the ontology already holds.
    """

    def composed(self, answer: LabelAnswer) -> Optional[Type]:
        """
        Build the class one proposal names.

        :param answer: The answer proposing it.
        :return: The class, or None where it cannot be built. The vocabulary step
            already reports a composition it cannot build, so there is nothing to add
            here.
        """
        superclass = self.known.get(answer.superclass)
        if superclass is None:
            return None
        mixins = [self.known[one] for one in answer.mixins if one in self.known]
        try:
            return compose_class(answer.class_name, superclass, mixins)
        except TypeError:
            # A composition is a model's proposal, so one that cannot be built is an
            # answer to report rather than a state the run should not have reached.
            return None

    def by_label(self) -> Dict[str, Optional[Type]]:
        """
        :return: Per label, the class it stands for, or None where it stands for none.
        """
        classes: Dict[str, Optional[Type]] = {}
        for label, answer in self.vocabulary.labels.items():
            if not answer.class_name or answer.problems:
                classes[label] = None
                if answer.class_name:
                    self.logger.info(
                        "  %s: leaving unmapped, %s", label, answer.problems[0]
                    )
            elif not answer.is_new_class:
                classes[label] = self.known.get(answer.class_name)
            else:
                classes[label] = self.composed(answer)
        return classes

    def widened(self, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Put the classes this run proposed beside the ones it inherited.

        The exported taxonomy is the one that is committed. A class the vocabulary step
        proposed exists only in this run's own directory until the last step generates
        it, so a body was described as "labelled kitchen_island, which was read as
        KitchenIsland" while ``KitchenIsland`` appeared nowhere in the ontology it was
        being asked to choose a name from -- and naming an existing class instead is
        then the only coherent answer left. Once it was ``CounterTop``, which accepts no
        drawer, and the eight drawers the island's own composition was built to hold
        could not be mounted.

        A proposal is an answer about the same object from the step before, so it
        belongs where the model is looking rather than in a clause it cannot act on. It
        is marked as proposed, and disagreeing with it stays available -- the split can
        leave a body that is no longer what its label said -- but it becomes a choice
        rather than the only way out.

        :param taxonomy: The exported taxonomy.
        :return: A taxonomy holding a class for each proposal that can be composed. The
            one it was given is not changed, so the file it was read from still holds
            what every run reads.
        """
        already = {node["name"] for node in taxonomy["classes"]}
        added = []
        for label, answer in sorted(self.vocabulary.proposals.items()):
            if answer.class_name in already:
                continue
            composed = self.composed(answer)
            if composed is None:
                continue
            node: Dict[str, Any] = {
                "name": answer.class_name,
                "bases": [base.__name__ for base in composed.__bases__],
                self.proposal_marker: label,
            }
            relations = relations_of(composed)
            if relations:
                node["relations"] = [relation.to_json() for relation in relations]
            added.append(node)
            already.add(answer.class_name)

        if not added:
            return taxonomy
        return {
            **taxonomy,
            "classes": taxonomy["classes"] + added,
            "note": taxonomy["note"] + " " + self.proposal_note(),
        }

    def proposal_note(self) -> str:
        """
        :return: What a model is told about the classes marked as proposed.
        """
        return (
            f"A class marked '{self.proposal_marker}' is not in the ontology yet: an "
            "earlier step read that label as this class and it is built when the run "
            "ends. Name it when it fits the object, exactly as you would one already "
            "there."
        )
