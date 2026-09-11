"""
Whether a part ended up in the right whole.

Every other measure counts structure without saying where any of it went: a run that
puts every drawer in the wrong cabinet counts exactly the same as one that puts them all
right.

The question is asked of pairs of parts rather than of each part and its whole, because
naming the whole would mean deciding which predicted cabinet is which modelled one, and
a run that finds one cabinet as five fragments has no such answer. Asking instead
whether two parts that belong together ended up together needs no whole to be identified
at all.

The two directions say different things and are worth reading apart. A pair the run put
together that the model holds apart is a part in the wrong place. A pair the model holds
together that the run split is usually the whole arriving in pieces, which is a
segmentation result rather than a placement one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations

from typing_extensions import Dict, List, Optional, Sequence

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.matching import ObjectCorrespondences
from experiments.warsaw.evaluation.structure import Tally, TypedRelation

# %% which of a part's wholes it is judged by


class Depth(StrEnum):
    """
    Which of a part's wholes a comparison is made at.
    """

    OUTERMOST = "outermost"
    """
    The piece of furniture a part ends up in, found by walking to the top of what holds
    it.

    A modelled world nests more deeply than the question does: an apartment puts a drawer
    inside a cabinet inside a side, while a run builds one island holding its cabinets,
    its countertop and its drawers at a single level. Judging a part by the whole that
    directly holds it scores that flattening as misplacement.
    """

    IMMEDIATE = "immediate"
    """
    The whole that directly holds a part, which is what says whether one whole arrived
    split into pieces.
    """


# %% how many pairs each world keeps together


@dataclass(frozen=True)
class SiblingPairs(Tally):
    """
    How many pairs of parts each world keeps together, and how many of them are the same
    pairs.
    """

    in_both: int = 0
    """
    How many pairs both worlds keep together.
    """

    @property
    def agreeing(self) -> int:
        """
        :return: How many pairs both worlds keep together, which is known here rather
            than bounded: the parts have been paired up, so a pair of one world is the
            same pair as one of the other or it is not.
        """
        return self.in_both


# %% a pair the two worlds disagree about


@dataclass(frozen=True)
class DisagreedPair(JsonRecord):
    """
    Two parts one world keeps together and the other does not.
    """

    one: str
    """
    One of the two, named as the run names it.
    """

    other: str
    """
    The other, named as the run names it.
    """

    whole: str
    """
    What holds them together in the world that does.
    """


# %% what the run put where


@dataclass(frozen=True)
class SiblingAgreement(JsonRecord):
    """
    Which parts the run kept together, against which the modelled world holds together.
    """

    depth: Depth = Depth.OUTERMOST
    """
    Which of a part's wholes this was judged at.
    """

    considered: List[str] = field(default_factory=list)
    """
    The run's parts this was asked of: those paired with a modelled object, and held by
    something in at least one of the two worlds.
    """

    pairs: SiblingPairs = field(
        default_factory=lambda: SiblingPairs(modelled=0, predicted=0, in_both=0)
    )
    """
    How many pairs each world keeps together, and how many they agree on.
    """

    put_together: List[DisagreedPair] = field(default_factory=list)
    """
    The pairs the run put in one whole that the modelled world holds in different ones,
    which is a part in the wrong place.
    """

    left_apart: List[DisagreedPair] = field(default_factory=list)
    """
    The pairs the modelled world holds together that the run split, which is usually the
    whole arriving in pieces.
    """

    @classmethod
    def between(
        cls,
        predicted_relations: Sequence[TypedRelation],
        modelled_relations: Sequence[TypedRelation],
        correspondences: ObjectCorrespondences,
        depth: Depth = Depth.OUTERMOST,
    ) -> SiblingAgreement:
        """
        Judge where a run put its parts, against where the modelled world holds them.

        :param predicted_relations: The relations the run asserted.
        :param modelled_relations: The relations the modelled world holds.
        :param correspondences: Which reconstructed object is which modelled one.
        :param depth: Which of a part's wholes to judge it by.
        :return: What the two worlds agree and disagree about.
        """
        stands_for = {one.predicted: one.modelled for one in correspondences.matched}
        predicted_whole = cls._whole_of(predicted_relations, depth)
        modelled_whole = cls._whole_of(modelled_relations, depth)
        considered = sorted(
            name
            for name in stands_for
            if name in predicted_whole or stands_for[name] in modelled_whole
        )

        together_in_both, put_together, left_apart = 0, [], []
        for one, other in combinations(considered, 2):
            run_holds = cls._same_whole(predicted_whole, one, other)
            model_holds = cls._same_whole(
                modelled_whole, stands_for[one], stands_for[other]
            )
            if run_holds is not None and model_holds is not None:
                together_in_both += 1
            elif run_holds is not None:
                put_together.append(
                    DisagreedPair(one=one, other=other, whole=run_holds)
                )
            elif model_holds is not None:
                left_apart.append(
                    DisagreedPair(one=one, other=other, whole=model_holds)
                )

        return cls(
            depth=depth,
            considered=considered,
            pairs=SiblingPairs(
                modelled=together_in_both + len(left_apart),
                predicted=together_in_both + len(put_together),
                in_both=together_in_both,
            ),
            put_together=put_together,
            left_apart=left_apart,
        )

    @classmethod
    def _whole_of(
        cls, relations: Sequence[TypedRelation], depth: Depth
    ) -> Dict[str, str]:
        """
        :param relations: The relations one world holds.
        :param depth: Which of a part's wholes to take.
        :return: The whole each part is judged by, by the name of the part.
        """
        held = {}
        for relation in relations:
            held.setdefault(relation.part, relation.whole)
        if depth is Depth.IMMEDIATE:
            return held
        return {part: cls._outermost(part, held) for part in held}

    @staticmethod
    def _outermost(part: str, held: Dict[str, str]) -> str:
        """
        :param part: The part to follow up.
        :param held: What directly holds each part of one world.
        :return: The whole at the top of what holds it, stopping where a world holds a
            part in a circle rather than walking it forever.
        """
        seen = {part}
        whole = held[part]
        while whole in held and whole not in seen:
            seen.add(whole)
            whole = held[whole]
        return whole

    @staticmethod
    def _same_whole(held_by: Dict[str, str], one: str, other: str) -> Optional[str]:
        """
        :param held_by: What holds each part of one world.
        :param one: One part of it.
        :param other: Another.
        :return: What holds them both, or nothing where one world does not hold them
            together.
        """
        whole = held_by.get(one)
        if whole is not None and whole == held_by.get(other):
            return whole
        return None
