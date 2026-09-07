"""
A written-down annotation class an amendment can be worked out against.

The amendment reads the file a class is declared in, so the class it edits has to be one
that really is written in a file rather than built at run time.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasDoors, HasDrawers
from semantic_digital_twin.semantic_annotations.semantic_annotations import Furniture


@dataclass
class Sideboard(Furniture, HasDoors):
    """
    A class declared with bases, one of which is a mixin, so another can be added.
    """


@dataclass
class Pedestal(Furniture, HasDrawers):
    """
    A class that already has the mixin an amendment would give it.
    """
