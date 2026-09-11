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
from itertools import combinations

from typing_extensions import Dict, List, Optional, Sequence

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.matching import ObjectCorrespondences
from experiments.warsaw.evaluation.structure import Tally, TypedRelation

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
    ) -> SiblingAgreement:
        """
        Judge where a run put its parts, against where the modelled world holds them.

        :param predicted_relations: The relations the run asserted.
        :param modelled_relations: The relations the modelled world holds.
        :param correspondences: Which reconstructed object is which modelled one.
        :return: What the two worlds agree and disagree about.
        """
        stands_for = {one.predicted: one.modelled for one in correspondences.matched}
        predicted_whole = cls._whole_of(predicted_relations)
        modelled_whole = cls._whole_of(modelled_relations)
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
            considered=considered,
            pairs=SiblingPairs(
                modelled=together_in_both + len(left_apart),
                predicted=together_in_both + len(put_together),
                in_both=together_in_both,
            ),
            put_together=put_together,
            left_apart=left_apart,
        )

    @staticmethod
    def _whole_of(relations: Sequence[TypedRelation]) -> Dict[str, str]:
        """
        :param relations: The relations one world holds.
        :return: What holds each part, by the name of the part.
        """
        held = {}
        for relation in relations:
            held.setdefault(relation.part, relation.whole)
        return held

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
