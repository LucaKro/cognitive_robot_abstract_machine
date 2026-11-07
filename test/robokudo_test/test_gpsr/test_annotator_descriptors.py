"""
Tests for GPSR annotator configuration.
"""

from __future__ import annotations

# %% Imports
import pytest
from robokudo.annotators.gpsr.filter_objects_by_area import FilterObjectsByArea
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator


# %% descriptor ownership
@pytest.mark.parametrize(
    "annotator_type",
    [
        HumanSittingAnnotator,
        HumanStandingAnnotator,
        HumanLyingAnnotator,
        HumanWavingAnnotator,
        FilterObjectsByArea,
    ],
)
def test_default_descriptors_are_independent(
    annotator_type: type[BaseAnnotator],
) -> None:
    """
    Each annotator receives its own default configuration.
    """
    first = annotator_type()
    second = annotator_type()

    assert first.descriptor is not second.descriptor
    assert first.descriptor.parameters is not second.descriptor.parameters


def test_supplied_descriptor_is_preserved() -> None:
    """
    An explicitly configured descriptor belongs to the annotator.
    """
    descriptor = HumanSittingAnnotator.Descriptor()
    annotator = HumanSittingAnnotator(descriptor=descriptor)

    assert annotator.descriptor is descriptor
