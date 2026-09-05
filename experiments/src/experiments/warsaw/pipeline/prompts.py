"""
What a model is told it is doing, one file per question.

A prompt is a long document rather than a value, and one written into the middle of a
Python file is one nobody reads before changing the code around it. Each lives in a file
of its own, beside the others, so the five of them can be read together and compared.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class Prompt(StrEnum):
    """
    One thing a model can be told it is doing.

    The value is the file the words are kept in.
    """

    VOCABULARY = "vocabulary.md"
    """
    Which class of the ontology a scene's label means.
    """

    OWNERSHIP = "ownership.md"
    """
    Whose surface a face several labels claim is.
    """

    MEMBERSHIP = "membership.md"
    """
    Which whole a part belongs to.
    """

    CLASSIFICATION = "classification.md"
    """
    What each of a split scene's bodies is.
    """

    TAXONOMY_AMENDMENT = "taxonomy_amendment.md"
    """
    Whether a class of the ontology is missing a structural part.
    """

    def read(self, directory: Path | None = None) -> str:
        """
        :param directory: Where the prompts are kept, by default beside this module.
        :return: The words themselves.
        """
        kept = directory or Path(__file__).resolve().parent / "prompts"
        return (kept / self.value).read_text().strip()
