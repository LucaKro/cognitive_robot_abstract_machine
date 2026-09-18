from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, TYPE_CHECKING

from krrood.exceptions import DataclassException
from random_events.variable import Variable

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class IntractableError(DataclassException):
    """
    Exception raised when an inference is intractable for a model.

    For instance, the mode of a non-deterministic model.
    """

    model: ProbabilisticModel

    def error_message(self) -> str:
        return f"Inference is intractable for {self.model}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UndefinedOperationError(DataclassException):
    """
    Exception raised when an operation is not defined for a model.

    For instance, invoking the CDF of a model that contains symbolic variables.
    """

    model: ProbabilisticModel

    def error_message(self) -> str:
        return f"Operation is not defined for {self.model}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ShapeMismatchError(DataclassException, ValueError):
    """
    Exception raised when the shape of two objects does not match.
    """

    received_shape: Any
    """
    The first object to compare.
    """

    expected_shape: Any
    """
    The second object to compare.
    """

    def error_message(self) -> str:
        return f"Expected shape {self.expected_shape}, received shape {self.received_shape}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class VariableNotInQuantitiesError(DataclassException):
    """
    Exception raised when a variable is named that the layout is not over.

    Every array is laid out by a fixed, ordered set of variables, so a variable outside
    it has no row to be read from or written to.
    """

    variable: Variable
    """
    The variable that was named.
    """

    quantities: List[Variable]
    """
    The variables the layout is over.
    """

    def error_message(self) -> str:
        return (
            f"{self.variable} is not one of the quantities "
            f"{[str(variable) for variable in self.quantities]}."
        )

    def suggest_correction(self) -> str:
        return "Name one of the quantities the layout was built over."


@dataclass
class RepeatedVariableError(DataclassException):
    """
    Exception raised when a variable is named more than once in one layout.

    A variable occupies exactly one row, so a second mention would claim a row that can
    never be read back.
    """

    variable: Variable
    """
    The variable that was named more than once.
    """

    def error_message(self) -> str:
        return f"{self.variable} was named more than once."

    def suggest_correction(self) -> str:
        return "Name each quantity once."


@dataclass
class MeanAndCovarianceDisagreeError(DataclassException):
    """
    Exception raised when a distribution's mean and covariance are laid out by different
    quantities, in which case neither says anything about the other.
    """

    mean_quantities: List[Variable]
    """
    The variables the mean is laid out by.
    """

    covariance_quantities: List[Variable]
    """
    The variables the covariance is laid out by.
    """

    def error_message(self) -> str:
        return (
            f"The mean is about {[str(variable) for variable in self.mean_quantities]} "
            f"and the covariance about "
            f"{[str(variable) for variable in self.covariance_quantities]}."
        )

    def suggest_correction(self) -> str:
        return "Lay both out by the same quantities."
