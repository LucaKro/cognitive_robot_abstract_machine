"""
Read the ontology out into a run, and build the tables it asks for in the run's schema.

Run as a program in an interpreter of its own, which starts after the ORM was rebuilt:

    python -m experiments.warsaw.pipeline.database.ontology_export <run directory> [options]

The export has to happen before the ORM is imported. The generated interface names every
mapped class, robots and their fingers included, and importing it puts all of them into
the annotation hierarchy the export then walks: the same ontology came out as 441 classes
rather than 139, and every question would have carried three hundred robot parts for a
model to choose a countertop from.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.semantic_annotations.taxonomy_export import export_taxonomy
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import List, Sequence

from experiments.warsaw.pipeline.records import PreparedOntology
from experiments.warsaw.pipeline.run import Run, RunFile


class OntologyExportOption(StrEnum):
    """
    The command-line options an export is handed what it was told with.
    """

    INCLUDE_SUMMARIES = "--include-summaries"
    """
    Give each class the first sentence of its docstring.
    """

    CATEGORIES_ARE_ANSWERS = "--categories-are-answers"
    """
    Let a class the ontology declares a category be an answer about an object.
    """


@dataclass
class OntologyExport:
    """
    The ontology read out into a run, and the run's tables built by the ORM that will
    write to them.
    """

    run: Run
    """
    The run the ontology is read out into.
    """

    include_summaries: bool = False
    """
    Whether each class is written with the first sentence of its docstring.
    """

    categories_are_answers: bool = False
    """
    Whether a class the ontology declares a category may be named as an answer.
    """

    def arguments(self) -> List[str]:
        """
        :return: The options that hand this export to an interpreter of its own, after
            the run's directory.
        """
        chosen = {
            OntologyExportOption.INCLUDE_SUMMARIES: self.include_summaries,
            OntologyExportOption.CATEGORIES_ARE_ANSWERS: self.categories_are_answers,
        }
        return [option.value for option, on in chosen.items() if on]

    @classmethod
    def from_arguments(cls, arguments: Sequence[str]) -> OntologyExport:
        """
        Read back an export from what its interpreter was handed.

        :param arguments: The run's directory, then the options :meth:`arguments` gave.
        :return: The export they describe.
        """
        parser = ArgumentParser()
        parser.add_argument("directory", type=Path)
        for option in OntologyExportOption:
            parser.add_argument(option.value, dest=option.name, action="store_true")
        parsed = parser.parse_args(list(arguments))
        chosen = vars(parsed)
        return cls(
            run=Run(directory=parsed.directory),
            include_summaries=chosen[OntologyExportOption.INCLUDE_SUMMARIES.name],
            categories_are_answers=chosen[
                OntologyExportOption.CATEGORIES_ARE_ANSWERS.name
            ],
        )

    def carry_out(self) -> PreparedOntology:
        """
        Export the taxonomy, build the tables, and record what came of both.

        :return: What was read out and what was built.
        """
        taxonomy = export_taxonomy(
            SemanticAnnotation,
            self.run.path(RunFile.TAXONOMY),
            include_summaries=self.include_summaries,
            categories_are_answers=self.categories_are_answers,
        )

        # Imported only now, after the export, for the reason the module gives.
        from semantic_digital_twin.orm.ormatic_interface import Base
        from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker

        Base.metadata.create_all(bind=semantic_digital_twin_sessionmaker()().bind)
        prepared = PreparedOntology(
            classes=len(taxonomy["classes"]),
            mixins=len(taxonomy["part_whole_mixins"]),
            tables=len(Base.metadata.tables),
        )
        self.run.write_record(RunFile.PREPARED_ONTOLOGY, prepared)
        return prepared


if __name__ == "__main__":
    OntologyExport.from_arguments(sys.argv[1:]).carry_out()
