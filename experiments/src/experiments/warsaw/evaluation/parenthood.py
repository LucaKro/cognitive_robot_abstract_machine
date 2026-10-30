"""
Which whole each part of a run ended up in, and why some ended up in none.

Counting relations by kind says whether a run builds the right sort of world. It cannot
say that a drawer went into no cabinet at all because the cabinet stopped existing, and
that turns out to be the commonest way a relation goes missing: a drawer takes every face
of the cabinet it belongs to, the cabinet is left with nothing and dropped, and the drawer
has nothing left to go into. The relation was never decided wrongly. It stopped being
expressible.

So every part is read twice over. A part that was given a whole is judged on the *kind* of
relation it was given, which needs no correspondence between the two worlds -- only
whether the modelled world holds that kind at all. A part that was given none is
attributed to whichever step lost it, which needs nothing of the modelled world beyond
knowing that its class is one that has a whole.

..note:: Neither half says a part went into the *right* cabinet. That question needs the
    two worlds related to each other; these are what can be answered before that.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Dict, Iterable, List, Optional

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.structure import (
    ClassifiedObject,
    RelationPattern,
    TypedRelation,
    UNCLASSIFIED,
    UNNAMED_FIELD,
    relation_patterns_in,
)
from experiments.warsaw.pipeline.records import SplitRecord

# %% why a part was given no whole


class MissingParentReason(StrEnum):
    """
    What became of the whole a part should have been given.

    Ordered as they are decided: an earlier one is the reason whatever a later one saw was
    going to happen anyway.
    """

    LOST_IN_THE_SPLIT = "lost in the split"
    """
    The part took every face of the object that would have held it, which left that object
    with nothing and dropped it.
    """

    MOUNT_REFUSED = "mount refused"
    """
    A whole was decided on and the world would not hold the part that way.
    """

    NOT_CHOSEN = "not chosen"
    """
    The split carried a pairing for the part past it and no relation came of it.
    """

    NEVER_OFFERED = "never offered"
    """
    Nothing ever proposed a whole for the part at all.
    """


@dataclass(frozen=True)
class MissingParent(JsonRecord):
    """
    One part of a run that was given no whole, and what became of the whole.
    """

    part: str
    """
    The part that was given none.
    """

    part_class: str
    """
    What the run decided the part is.
    """

    reason: MissingParentReason
    """
    What became of its whole.
    """

    whole: Optional[str] = None
    """
    The object at the other end, where there is one: the whole that was proposed, or, for
    a part lost in the split, the object it consumed. That object is usually the whole it
    belonged to, but a scan can label two unrelated things over one surface, so it is what
    vanished rather than a relation that was decided.
    """


# %% the relations a run did assert


@dataclass(frozen=True)
class JudgedRelation(JsonRecord):
    """
    One relation a run asserted, and whether the modelled world holds that kind.
    """

    whole: str
    """
    The object at the holding end.
    """

    part: str
    """
    The object at the held end.
    """

    pattern: RelationPattern
    """
    The kind of relation it is, said in classes.
    """

    modelled: bool
    """
    Whether the modelled world holds relations of that kind.
    """


# %% both halves, over one run


@dataclass(frozen=True)
class ParenthoodComparison(JsonRecord):
    """
    Every part of a run: the relation it was given, or what became of the one it was not.

    The two lists are disjoint and together cover every object whose class the modelled
    world holds as a part, so a low number can be taken apart into the reasons it is low.
    """

    relations: List[JudgedRelation] = field(default_factory=list)
    """
    The relations the run asserted, each judged on its kind.
    """

    missing: List[MissingParent] = field(default_factory=list)
    """
    The parts that were given no whole, each attributed to what became of it.
    """

    @classmethod
    def between(
        cls,
        predicted_objects: Iterable[ClassifiedObject],
        predicted_relations: Iterable[TypedRelation],
        modelled_objects: Iterable[ClassifiedObject],
        modelled_relations: Iterable[TypedRelation],
        split: SplitRecord,
    ) -> ParenthoodComparison:
        """
        Judge every part of a run against what the modelled world holds.

        :param predicted_objects: The run's objects in scope.
        :param predicted_relations: The relations the run asserted, in scope.
        :param modelled_objects: The modelled world's objects in scope.
        :param modelled_relations: The modelled world's relations in scope.
        :param split: What the run's split recorded, which says what became of a whole
            that is no longer there.
        :return: Both halves, over every part.
        """
        predicted_objects = list(predicted_objects)
        predicted_relations = list(predicted_relations)
        modelled_patterns = relation_patterns_in(
            modelled_relations, list(modelled_objects)
        )
        class_of = {
            one.name: (one.classes[0] if one.classes else UNCLASSIFIED)
            for one in predicted_objects
        }
        relations = [
            JudgedRelation(
                whole=one.whole,
                part=one.part,
                pattern=cls._pattern_of(one, class_of),
                modelled=cls._pattern_of(one, class_of) in modelled_patterns,
            )
            for one in predicted_relations
        ]

        held = {one.part for one in predicted_relations}
        part_classes = {pattern.part_class for pattern in modelled_patterns}
        missing = [
            cls._what_became_of_the_whole(one.name, class_of[one.name], split)
            for one in predicted_objects
            if class_of[one.name] in part_classes and one.name not in held
        ]
        return cls(relations=relations, missing=missing)

    @staticmethod
    def _pattern_of(
        relation: TypedRelation, class_of: Dict[str, str]
    ) -> RelationPattern:
        """
        :param relation: The relation to say in classes.
        :param class_of: What each object of its graph stands as.
        :return: The kind of relation it is.
        """
        return RelationPattern(
            whole_class=class_of.get(relation.whole, UNCLASSIFIED),
            field_name=relation.field_name or UNNAMED_FIELD,
            part_class=class_of.get(relation.part, UNCLASSIFIED),
            relation=relation.relation,
        )

    @staticmethod
    def _what_became_of_the_whole(
        part: str, part_class: str, split: SplitRecord
    ) -> MissingParent:
        """
        Say which step lost the whole a part should have been given.

        The object a part is said to have emptied is the one it took most faces of and was
        the largest taker of: taking thirty of something's faces is not standing in for it.

        :param part: The part that was given no whole.
        :param part_class: What the run decided the part is.
        :param split: What the run's split recorded.
        :return: The part, with what became of its whole.
        """
        emptied = [
            segment
            for segment in split.emptied
            if segment.taken_by and segment.taken_by[0].name == part
        ]
        if emptied:
            consumed = max(emptied, key=lambda segment: segment.taken_by[0].faces)
            return MissingParent(
                part=part,
                part_class=part_class,
                reason=MissingParentReason.LOST_IN_THE_SPLIT,
                whole=consumed.name,
            )

        refused = [one for one in split.refused if one.pairing.part == part]
        if refused:
            return MissingParent(
                part=part,
                part_class=part_class,
                reason=MissingParentReason.MOUNT_REFUSED,
                whole=refused[0].pairing.whole,
            )

        offered = [one for one in split.pairings if one.part == part]
        if offered:
            return MissingParent(
                part=part,
                part_class=part_class,
                reason=MissingParentReason.NOT_CHOSEN,
                whole=offered[0].whole,
            )

        return MissingParent(
            part=part, part_class=part_class, reason=MissingParentReason.NEVER_OFFERED
        )

    # %% what it comes to

    @property
    def relations_of_a_kind_the_modelled_world_holds(self) -> int:
        """
        :return: How many of the run's relations are of a kind the modelled world holds.
        """
        return sum(1 for one in self.relations if one.modelled)

    @property
    def missing_by_reason(self) -> Dict[MissingParentReason, int]:
        """
        :return: How many parts each reason accounts for, commonest first.
        """
        counted = Counter(one.reason for one in self.missing)
        return {reason: counted[reason] for reason, _ in counted.most_common()}
