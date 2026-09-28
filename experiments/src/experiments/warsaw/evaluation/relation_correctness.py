"""
Whether each relation a run asserted is one the modelled world holds.

The objects a run can find are settled by what the scan was segmented into; what the
pipeline adds is the claim that *this* part belongs in *that* whole, through *that*
ontology field. So the relations are what it should be judged on, and judged one at a time
rather than counted.

Counting does not answer it. Agreement on the *kind* of a relation saturates as soon as
the two vocabularies match, and a run that puts every drawer in the wrong cabinet builds
exactly as many relations of exactly the same kinds as one that puts them all right.

A run that skips a level is not wrong about where a part is. The modelled world puts a
drawer inside a cabinet inside a side; a run that hangs the drawer straight off the island
has still put it somewhere it really is. So a relation is correct when the modelled world
holds the part anywhere inside the matched whole, and the field it was held through is
reported separately, since that is the finer claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Dict, List, Optional, Sequence, Set

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.matching import ObjectCorrespondences
from experiments.warsaw.evaluation.structure import TypedRelation

# %% what became of one relation


class RelationVerdict(StrEnum):
    """
    What the modelled world made of one relation a run asserted.
    """

    CORRECT = "the part is inside that whole"
    """
    The modelled world holds the part somewhere inside the matched whole.
    """

    WRONG_WHOLE = "the part is somewhere else"
    """
    Both ends have counterparts and the modelled world does not hold one inside the
    other.
    """

    ONE_OBJECT = "both ends are the same modelled object"
    """
    The run found one modelled object as several, and this relation is between two of
    them. The modelled world holds no opinion about the inside of one of its own objects.
    """

    NOT_JUDGED = "an end has no counterpart"
    """
    One or both ends were paired with nothing modelled, so the relation can be called
    neither right nor wrong.
    """


@dataclass(frozen=True)
class JudgedAssertion(JsonRecord):
    """
    One relation a run asserted, and what the modelled world made of it.
    """

    whole: str
    """
    The object the run put the part in.
    """

    part: str
    """
    The object it put there.
    """

    field_name: str
    """
    The ontology field it held it through.
    """

    verdict: RelationVerdict
    """
    What the modelled world made of it.
    """

    modelled_whole: Optional[str] = None
    """
    Where the modelled world puts the part's counterpart, where it puts it anywhere.
    """

    held_directly: bool = False
    """
    Whether the modelled world holds the part in that very whole, rather than in
    something further out.
    """

    same_field: bool = False
    """
    Whether the modelled world holds it directly, through the same field.
    """


# %% every relation a run asserted


@dataclass(frozen=True)
class RelationCorrectness(JsonRecord):
    """
    Every relation a run asserted, judged against the modelled world one at a time.
    """

    judged: List[JudgedAssertion] = field(default_factory=list)
    """
    Each relation the run asserted, with what became of it.
    """

    @classmethod
    def between(
        cls,
        predicted_relations: Sequence[TypedRelation],
        modelled_relations: Sequence[TypedRelation],
        correspondences: ObjectCorrespondences,
    ) -> RelationCorrectness:
        """
        Judge every relation a run asserted.

        :param predicted_relations: The relations the run asserted.
        :param modelled_relations: The relations the modelled world holds.
        :param correspondences: Which reconstructed object is which modelled one.
        :return: Each of them, judged.
        """
        stands_for = {one.predicted: one.modelled for one in correspondences.matched}
        held_by = {}
        held_through = {}
        for relation in modelled_relations:
            held_by.setdefault(relation.part, relation.whole)
            held_through.setdefault(relation.part, relation.field_name)
        return cls(
            judged=[
                cls._judge(one, stands_for, held_by, held_through)
                for one in predicted_relations
            ]
        )

    @classmethod
    def _judge(
        cls,
        asserted: TypedRelation,
        stands_for: Dict[str, str],
        held_by: Dict[str, str],
        held_through: Dict[str, str],
    ) -> JudgedAssertion:
        """
        :param asserted: One relation the run asserted.
        :param stands_for: Which modelled object each reconstructed one is.
        :param held_by: What directly holds each modelled part.
        :param held_through: The field each modelled part is held through.
        :return: The relation, judged.
        """
        whole = stands_for.get(asserted.whole)
        part = stands_for.get(asserted.part)
        if whole is None or part is None:
            return JudgedAssertion(
                whole=asserted.whole,
                part=asserted.part,
                field_name=asserted.field_name,
                verdict=RelationVerdict.NOT_JUDGED,
            )
        if whole == part:
            return JudgedAssertion(
                whole=asserted.whole,
                part=asserted.part,
                field_name=asserted.field_name,
                verdict=RelationVerdict.ONE_OBJECT,
                modelled_whole=whole,
            )
        inside = cls._everything_holding(part, held_by)
        directly = held_by.get(part) == whole
        return JudgedAssertion(
            whole=asserted.whole,
            part=asserted.part,
            field_name=asserted.field_name,
            verdict=(
                RelationVerdict.CORRECT
                if whole in inside
                else RelationVerdict.WRONG_WHOLE
            ),
            modelled_whole=held_by.get(part),
            held_directly=directly,
            same_field=directly and held_through.get(part) == asserted.field_name,
        )

    @staticmethod
    def _everything_holding(part: str, held_by: Dict[str, str]) -> Set[str]:
        """
        :param part: A modelled object.
        :param held_by: What directly holds each modelled part.
        :return: Everything it is inside, at any depth, stopping where a world holds a
            part in a circle rather than walking it forever.
        """
        inside = set()
        whole = held_by.get(part)
        while whole is not None and whole not in inside:
            inside.add(whole)
            whole = held_by.get(whole)
        return inside

    # %% what it reads as

    @property
    def asserted(self) -> int:
        """
        :return: How many relations the run asserted.
        """
        return len(self.judged)

    @property
    def judgeable(self) -> int:
        """
        :return: How many of them had a counterpart at both ends, which is how many could
            be called right or wrong at all.
        """
        return sum(
            1
            for one in self.judged
            if one.verdict in (RelationVerdict.CORRECT, RelationVerdict.WRONG_WHOLE)
        )

    @property
    def correct(self) -> int:
        """
        :return: How many put the part somewhere the modelled world really holds it.
        """
        return sum(1 for one in self.judged if one.verdict is RelationVerdict.CORRECT)

    @property
    def share_correct(self) -> Optional[float]:
        """
        :return: What fraction of the relations that could be judged were correct, or
            nothing where none could be.
        """
        if self.judgeable == 0:
            return None
        return self.correct / self.judgeable

    @property
    def held_directly(self) -> int:
        """
        :return: How many put the part in the very whole the modelled world holds it in,
            rather than in something further out.
        """
        return sum(1 for one in self.judged if one.held_directly)

    @property
    def same_field(self) -> int:
        """
        :return: How many held the part through the field the modelled world holds it
            through, which is the finer claim the ontology is what makes.
        """
        return sum(1 for one in self.judged if one.same_field)

    @property
    def wrongly_placed(self) -> List[JudgedAssertion]:
        """
        :return: The relations that put a part somewhere it is not.
        """
        return [
            one for one in self.judged if one.verdict is RelationVerdict.WRONG_WHOLE
        ]
