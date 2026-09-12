from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

"""
How a rated set of pose candidates is drawn from.
"""


@dataclass
class CostmapSamplingStrategy(ABC):
    """
    How a costmap picks which of its entries to offer as candidates.

    A costmap rates every entry it holds; a strategy decides what that rating is used
    for.
    """

    @abstractmethod
    def choose(self, ratings: np.ndarray, count: int) -> np.ndarray:
        """
        Pick which entries to offer.

        :param ratings: The flattened costmap, one rating per entry.
        :param count: How many entries to pick.
        :return: The indices to offer, in the order they should be offered.
        """


@dataclass
class HighestRatedFirst(CostmapSamplingStrategy):
    """
    Offers the highest rated entries first, in order, leaving nothing to chance.

    Lets a caller take the first candidate that passes its own checks. A caller that can
    only afford to judge a handful never sees past what the map rates highest, though --
    for a ring, that is its own radius, one angle at a time.
    """

    def choose(self, ratings: np.ndarray, count: int) -> np.ndarray:
        highest = np.argpartition(ratings, -count)[-count:]
        return highest[np.argsort(ratings[highest])[::-1]]


@dataclass
class RandomCostmapSamplingStrategy(CostmapSamplingStrategy, ABC):
    """
    Base for the strategies that draw entries at random.
    """

    seed: Optional[int] = field(default=None, kw_only=True)
    """
    Fixes the draw, so a run can be repeated exactly.

    ``None`` draws afresh every time.
    """

    random_generator: np.random.Generator = field(init=False, repr=False)
    """
    Source of randomness, kept off numpy's global state so one draw cannot disturb
    another.
    """

    def __post_init__(self) -> None:
        self.random_generator = np.random.default_rng(self.seed)


@dataclass
class WeightedByRating(RandomCostmapSamplingStrategy):
    """
    Draws entries at random, an entry's rating being its chance of being drawn.

    Treats the map as the distribution its shape describes, so what it rates highest is
    merely likeliest and the rest of the region still comes up.
    """

    def choose(self, ratings: np.ndarray, count: int) -> np.ndarray:
        total = ratings.sum()
        if total == 0:
            return self.random_generator.choice(ratings.size, count, replace=False)
        return self.random_generator.choice(
            ratings.size, count, replace=False, p=ratings / total
        )


@dataclass
class UniformlyAtRandom(RandomCostmapSamplingStrategy):
    """
    Draws entries at random, all of them equally likely, ignoring their ratings.

    For a caller that wants the region covered rather than the part of it the map rates
    highest.
    """

    def choose(self, ratings: np.ndarray, count: int) -> np.ndarray:
        return self.random_generator.choice(ratings.size, count, replace=False)
