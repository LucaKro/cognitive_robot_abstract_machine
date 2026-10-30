"""
Handing work to an interpreter that starts after the ontology or the ORM was rewritten.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.semantic_annotations.mixins import HasDoors
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Handle,
)

from experiments.warsaw.exceptions import (
    AmendmentNotInForceError,
    SubprocessStepFailedError,
)
from experiments.warsaw.pipeline.new_interpreter import NewInterpreter
from experiments.warsaw.pipeline.steps.amend.amended_classes_check import (
    AmendedClassesCheck,
    AmendedPair,
)

# %% handing the work over


def checking(tmp_path, whole: str) -> NewInterpreter:
    """
    :param whole: The class to check holds the doors mixin.
    :return: The amendment check, ready to be carried out in an interpreter of its own.
    """
    check = AmendedClassesCheck(
        pairs=[AmendedPair(whole=whole, mixin=HasDoors.__name__)]
    )
    return NewInterpreter(
        entry=AmendedClassesCheck,
        directory=tmp_path,
        what="checking the amended classes came back amended",
        arguments=check.arguments(),
    )


def test_work_that_finishes_returns_what_it_printed(tmp_path):
    """
    The check prints nothing when every class holds its mixin.
    """
    assert checking(tmp_path, Cabinet.__name__).carry_out() == ""


def test_work_that_fails_is_reported_with_its_own_failure(tmp_path):
    """
    The arguments reach the interpreter, and what went wrong there comes back rather
    than being lost with the process.
    """
    with pytest.raises(SubprocessStepFailedError) as raised:
        checking(tmp_path, Handle.__name__).carry_out()
    assert AmendmentNotInForceError.__name__ in raised.value.output
