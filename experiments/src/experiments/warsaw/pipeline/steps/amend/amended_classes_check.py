"""
Check that amended classes really hold what they were amended to hold.

Run as a program in an interpreter of its own, since the classes in the one asking were
built before the edit and a dataclass collects its fields once:

    python -m experiments.warsaw.pipeline.steps.amend.amended_classes_check <run directory> <pairs>
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import List, Sequence

from experiments.warsaw.exceptions import AmendmentNotInForceError


@dataclass(frozen=True)
class AmendedPair:
    """
    One class and the mixin it was amended to derive from.
    """

    whole: str
    """
    The name of the amended class.
    """

    mixin: str
    """
    The name of the mixin it was amended with.
    """


@dataclass
class AmendedClassesCheck:
    """
    The amended classes, read anew, asked whether each derives from its mixin.
    """

    pairs: List[AmendedPair]
    """
    The amendments to check.
    """

    def arguments(self) -> List[str]:
        """
        :return: What hands this check to an interpreter of its own, after the run's
            directory.
        """
        return [json.dumps([asdict(pair) for pair in self.pairs])]

    @classmethod
    def from_arguments(cls, arguments: Sequence[str]) -> AmendedClassesCheck:
        """
        Read back a check from what its interpreter was handed after the run's
        directory.

        :param arguments: What :meth:`arguments` gave.
        :return: The check they describe.
        """
        [pairs] = arguments
        return cls(pairs=[AmendedPair(**pair) for pair in json.loads(pairs)])

    def carry_out(self) -> None:
        """
        Check every pair against the classes as this interpreter reads them.

        :raises AmendmentNotInForceError: If a class does not derive from its mixin.
        """
        known = annotation_classes(SemanticAnnotation)
        missing = [
            pair
            for pair in self.pairs
            if not issubclass(known[pair.whole], known[pair.mixin])
        ]
        if missing:
            raise AmendmentNotInForceError(pairs=missing)


if __name__ == "__main__":
    # The first argument is the run's directory, which every such interpreter is handed
    # and this check does not need.
    AmendedClassesCheck.from_arguments(sys.argv[2:]).carry_out()
