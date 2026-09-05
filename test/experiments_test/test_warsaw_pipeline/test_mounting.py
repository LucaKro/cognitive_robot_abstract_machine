"""
Carrying a decided mount out through the channel the ontology says mounts it.

A mug resting on a countertop is mounted with ``add_object``, not with ``add``, and a
mount through the wrong channel raises rather than building the wrong world quietly. The
channel is written into the split's record and read back out of it, so the two have to
mean the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from experiments.warsaw.pipeline.steps.annotate import MountAnnotations
from experiments.warsaw.scene_split import Pairing
from semantic_digital_twin.exceptions import CannotBeAPartOf
from semantic_digital_twin.semantic_annotations.taxonomy_export import MountKind
from typing_extensions import Any, List


@dataclass
class RecordingWhole:
    """
    Something that holds, which records the channel it was reached through.

    Named for the behaviour it exercises rather than for any one annotation class: what
    is checked is the routing, and every class that holds offers the same three methods.
    """

    mounted: List[Any] = field(default_factory=list)
    """
    What was mounted, as ``(channel, part, field)``.
    """

    def add(self, part, *, field_name: str = "") -> None:
        """
        :param part: The structural part to hold.
        :param field_name: The field to hold it in.
        """
        self.mounted.append((MountKind.PART, part, field_name))

    def add_object(self, part) -> None:
        """
        :param part: The object to store inside this.
        """
        self.mounted.append((MountKind.CONTAINS, part, ""))

    def add_supporting_surface(self, part) -> None:
        """
        :param part: The surface this rests on.
        """
        self.mounted.append((MountKind.SUPPORTS, part, ""))


@dataclass
class RefusingWhole:
    """
    Something that will not hold what it is offered.
    """

    def add(self, part, *, field_name: str = "") -> None:
        """
        :param part: The part it refuses.
        :raises CannotBeAPartOf: Always.
        """
        raise CannotBeAPartOf(self, part)


@pytest.mark.parametrize(
    "kind, expected_field",
    [
        (MountKind.PART, "drawers"),
        (MountKind.CONTAINS, ""),
        (MountKind.SUPPORTS, ""),
    ],
)
def test_a_pairing_is_mounted_through_the_channel_it_names(kind, expected_field):
    """
    The channel is what the ontology said mounts this relation when the pair was
    measured.
    """
    whole, part = RecordingWhole(), object()
    MountAnnotations.mount_one(
        whole,
        part,
        Pairing(whole="a", part="b", field_name="drawers", kind=kind),
    )
    assert whole.mounted == [(kind, part, expected_field)]


def test_a_structural_part_is_routed_to_the_field_it_was_measured_for():
    """
    A whole holds several drawers and several doors, so which field takes this one was
    decided when the candidates were gathered.
    """
    whole, part = RecordingWhole(), object()
    MountAnnotations.mount_one(
        whole, part, Pairing(whole="a", part="b", field_name="doors")
    )
    assert whole.mounted == [(MountKind.PART, part, "doors")]


def test_a_whole_that_will_not_hold_the_part_says_so_rather_than_holding_it():
    """
    The class a body was given may simply not admit the part another step said it holds,
    and that is an answer about this pairing rather than a run that should not have got
    this far.
    """
    with pytest.raises(CannotBeAPartOf):
        MountAnnotations.mount_one(
            RefusingWhole(),
            object(),
            Pairing(whole="a", part="b", field_name="drawers"),
        )
