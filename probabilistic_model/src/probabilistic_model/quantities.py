from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from random_events.variable import Continuous
from typing_extensions import Dict, Mapping, Self, Tuple

from probabilistic_model.exceptions import (
    RepeatedVariableError,
    VariableNotInQuantitiesError,
)

QuantityPair = Tuple[Continuous, Continuous]
"""
Two quantities whose relationship an entry describes.
"""

# %% the variables an array is laid out by


@dataclass(frozen=True)
class Quantities:
    """
    An ordered set of continuous random variables, and the layout of every array over
    them.

    Arrays are built here from the variables rather than from row and column counts, so
    an array cannot end up describing a different set of quantities than the thing it
    belongs to. Naming a quantity outside the set is then the only way left to get it
    wrong, and that is what :meth:`index_of` rejects.

    ..note:: Building each array here walks a mapping on every call, where taking the
        arrays ready-made would not. At the sizes these layouts are used at that stays
        far below the rest of a control cycle, but it is the first thing to undo if one
        ever shows up in a profile: what consumes these arrays does plain numpy
        arithmetic, so the mappings can go back to being arrays the caller builds.
    """

    variables: Tuple[Continuous, ...]
    """
    The quantities, in the order they index every array over them.
    """

    @classmethod
    def of(cls, *variables: Continuous) -> Self:
        """
        :param variables: The quantities to lay arrays out by, in the order they are to
            be laid out in. That order is the caller's own and is kept as given, since a
            domain's own ordering is rarely alphabetical.
        :return: Them, in the order given.
        :raises RepeatedVariableError: If one of them is named more than once.
        """
        for position, variable in enumerate(variables):
            if variable in variables[:position]:
                raise RepeatedVariableError(variable=variable)
        return cls(variables=variables)

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    def __contains__(self, variable: Continuous) -> bool:
        return variable in self.variables

    def index_of(self, variable: Continuous) -> int:
        """
        :param variable: The quantity to locate.
        :return: The row every array over these quantities holds it in.
        :raises VariableNotInQuantitiesError: If it is not one of them.
        """
        if variable not in self.variables:
            raise VariableNotInQuantitiesError(
                variable=variable, quantities=list(self.variables)
            )
        return self.variables.index(variable)

    def vector(self, values: Mapping[Continuous, float]) -> npt.NDArray[np.float64]:
        """
        Build one number per quantity.

        :param values: The number for each quantity that has one; the rest are zero.
        :return: Them, in this layout.
        :raises VariableNotInQuantitiesError: If an entry names a quantity outside this
            layout.
        """
        built = np.zeros(len(self))
        for variable, value in values.items():
            built[self.index_of(variable)] = value
        return built

    def matrix(self, entries: Mapping[QuantityPair, float]) -> npt.NDArray[np.float64]:
        """
        Build one number per ordered pair of quantities.

        :param entries: The number for each pair that has one; the rest are zero. The
            first quantity of a pair is the row, the second the column.
        :return: Them, in this layout.
        :raises VariableNotInQuantitiesError: If an entry names a quantity outside this
            layout.
        """
        built = np.zeros((len(self), len(self)))
        for (row, column), value in entries.items():
            built[self.index_of(row), self.index_of(column)] = value
        return built

    def symmetric_matrix(
        self, entries: Mapping[QuantityPair, float]
    ) -> npt.NDArray[np.float64]:
        """
        Build one number per unordered pair of quantities, for a covariance.

        Two quantities vary together by one number rather than two, so each pair given
        fills its mirror as well and a caller states it once.

        :param entries: The number for each pair that has one; the rest are zero.
        :return: Them, in this layout.
        :raises VariableNotInQuantitiesError: If an entry names a quantity outside this
            layout.
        """
        built = self.matrix(entries)
        for (row, column), value in entries.items():
            built[self.index_of(column), self.index_of(row)] = value
        return built

    @property
    def unchanged(self) -> Dict[QuantityPair, float]:
        """
        :return: The transition of quantities expected to stay as they are.
        """
        return {(variable, variable): 1.0 for variable in self.variables}
