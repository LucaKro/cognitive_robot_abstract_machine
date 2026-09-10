"""
Deciding which reconstructed object is which modelled one.

Every later measure rests on this: a class is only right or wrong about some particular
object, and a relation is only right if both of its ends are. So the correspondence is
worked out once, written down with what it cost, and read back by everything that
scores.

The assignment is the cheapest one-to-one pairing over a table of costs, so it depends
on nothing but that table. What fills the table is separate and says what "alike" means:
two objects are compared by what they are and how big they are, both of which two
unrelated frames agree on. A pairing that costs more than the declared limit is no
pairing, so an object with nothing like it in the other graph is left unmatched rather
than forced onto the least bad partner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing_extensions import List, Optional, Protocol, Sequence

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.size import ObjectSize

# %% what a pairing is decided from


class MatchableObject(Protocol):
    """
    An object one graph offers for pairing with an object of the other.
    """

    name: str
    """
    What identifies it within its own graph.
    """

    @property
    def classes(self) -> List[str]:
        """
        :return: The classes it stands as.
        """

    @property
    def size(self) -> Optional[ObjectSize]:
        """
        :return: How big it is, where it was measured.
        """


# %% what a pairing cost is made of


@dataclass(frozen=True)
class MatchCost(JsonRecord):
    """
    What one pairing costs, and what each part of the comparison contributed.

    The parts are kept rather than only their total, because a correspondence worked out
    from sizes alone and one confirmed by the classes agreeing are not equally
    trustworthy and should not read alike afterwards.
    """

    disagreeing_class: float
    """
    What the two objects standing as different classes contributed.
    """

    differing_size: float
    """
    What their difference in size contributed.
    """

    @property
    def total(self) -> float:
        """
        :return: What the pairing costs altogether.
        """
        return self.disagreeing_class + self.differing_size


@dataclass(frozen=True)
class HowToCompare(JsonRecord):
    """
    What counts as alike, and how much a pairing may cost before it is no pairing.

    Every number here is declared with the result rather than chosen after seeing one,
    so a correspondence can be recomputed and a different setting can be argued with.
    """

    class_disagreement: float = 0.6
    """
    What two objects standing as different classes costs.
    """

    size_difference: float = 1.0
    """
    What a wholly unalike size costs, in proportion to how unalike it is.
    """

    most_a_pairing_may_cost: float = 0.8
    """
    Above this a pairing is refused, leaving both objects unmatched.
    """

    def cost_of(
        self, predicted: MatchableObject, modelled: MatchableObject
    ) -> MatchCost:
        """
        Say what pairing two objects would cost.

        Classes disagreeing is not enough on its own to refuse a pairing, because a
        correspondence that only ever pairs objects already agreeing on their class
        cannot measure a classification mistake: every one of them would be reported as
        an object nobody found.

        :param predicted: The reconstructed object.
        :param modelled: The modelled object.
        :return: What the pairing costs and where the cost came from.
        """
        agreeing = bool(set(predicted.classes) & set(modelled.classes))
        both_measured = predicted.size is not None and modelled.size is not None
        return MatchCost(
            disagreeing_class=0.0 if agreeing else self.class_disagreement,
            differing_size=(
                self.size_difference * predicted.size.difference_from(modelled.size)
                if both_measured
                else self.size_difference
            ),
        )


# %% what the pairing came to


@dataclass(frozen=True)
class Correspondence(JsonRecord):
    """
    One reconstructed object paired with the modelled object it stands for.
    """

    predicted: str
    """
    The reconstructed object.
    """

    modelled: str
    """
    The modelled object it was paired with.
    """

    cost: MatchCost
    """
    What the pairing cost, kept so a weak one can be told from a firm one.
    """

    better_than_the_next_by: float
    """
    How much dearer this object's second-best partner would have been.

    Near nought means the other graph held something else just as alike, so which of
    them was paired is close to arbitrary and anything read from this pairing is too. It
    is the honest measure of a correspondence drawn from evidence that cannot tell two
    similar objects apart.
    """

    @property
    def classes_agree(self) -> bool:
        """
        :return: Whether the two objects stand as the same class, which is what makes
            this pairing a correct classification rather than only a correct detection.
        """
        return self.cost.disagreeing_class == 0.0


@dataclass(frozen=True)
class ObjectCorrespondences(JsonRecord):
    """
    Which reconstructed object is which modelled one, and what was left over.
    """

    how_compared: HowToCompare
    """
    The settings this correspondence was worked out under.
    """

    matched: List[Correspondence] = field(default_factory=list)
    """
    The pairings, cheapest first.
    """

    unmatched_predicted: List[str] = field(default_factory=list)
    """
    Reconstructed objects nothing modelled was near enough to pair with.
    """

    unmatched_modelled: List[str] = field(default_factory=list)
    """
    Modelled objects the reconstruction has nothing near enough for.
    """

    @classmethod
    def between(
        cls,
        predicted: Sequence[MatchableObject],
        modelled: Sequence[MatchableObject],
        how_compared: Optional[HowToCompare] = None,
    ) -> ObjectCorrespondences:
        """
        Pair each reconstructed object with at most one modelled object.

        The pairing chosen is the one whose costs add up to least over the whole scene,
        rather than each object taking its own favourite: two handles that both prefer
        the same drawer cannot both have it, and settling that greedily depends on which
        was asked first.

        :param predicted: The reconstructed objects in scope.
        :param modelled: The modelled objects in scope.
        :param how_compared: What counts as alike, or the declared defaults.
        :return: The correspondence and everything it left unpaired.
        """
        how_compared = how_compared or HowToCompare()
        if not predicted or not modelled:
            return cls(
                how_compared=how_compared,
                unmatched_predicted=[one.name for one in predicted],
                unmatched_modelled=[one.name for one in modelled],
            )
        costs = [
            [how_compared.cost_of(one, other) for other in modelled]
            for one in predicted
        ]
        totals = np.array([[cost.total for cost in row] for row in costs])
        chosen_rows, chosen_columns = linear_sum_assignment(totals)
        matched = [
            Correspondence(
                predicted=predicted[row].name,
                modelled=modelled[column].name,
                cost=costs[row][column],
                better_than_the_next_by=cls._lead_over_the_next_best(
                    totals[row], column
                ),
            )
            for row, column in zip(chosen_rows, chosen_columns)
            if costs[row][column].total <= how_compared.most_a_pairing_may_cost
        ]
        paired_predicted = {correspondence.predicted for correspondence in matched}
        paired_modelled = {correspondence.modelled for correspondence in matched}
        return cls(
            how_compared=how_compared,
            matched=sorted(matched, key=lambda pairing: pairing.cost.total),
            unmatched_predicted=[
                one.name for one in predicted if one.name not in paired_predicted
            ],
            unmatched_modelled=[
                one.name for one in modelled if one.name not in paired_modelled
            ],
        )

    @staticmethod
    def _lead_over_the_next_best(costs_for_one: np.ndarray, chosen: int) -> float:
        """
        :param costs_for_one: What pairing one object with each of the others would cost.
        :param chosen: The one it was paired with.
        :return: How much dearer the next best would have been, or nought where there was
            nothing else to choose between.
        """
        others = np.delete(costs_for_one, chosen)
        if others.size == 0:
            return 0.0
        return float(others.min() - costs_for_one[chosen])
