"""
Put the ontology back as it is written, read it out, and give the run a schema.

A run must start from the ontology that is committed, not from what the last run talked
a model into. Classes generated for one scene and mixins one room argued for would
otherwise be in force for the next, where nothing questions them and nobody remembers
they were ever in doubt.

What this leaves behind is the run's own reading of the ontology, in the run's own
directory, and a schema in the database holding the tables that reading asks for. Both
are built by an interpreter that started after the ORM was rewritten, because the one
asking is holding the version from before that.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationFilePaths,
)
from typing_extensions import List, Tuple

from experiments.warsaw.exceptions import OntologyLeftAmendedError
from experiments.warsaw.pipeline.database.ontology_export import OntologyExport
from experiments.warsaw.pipeline.records import PreparedOntology
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.templates import PipelineTemplates
from experiments.warsaw.pipeline.database.run_schema import RunSchema
from experiments.warsaw.pipeline.steps.step import PipelineStep

# %% getting a run ready


@dataclass
class PrepareRun(PipelineStep):
    """
    The ontology as it is committed, read out into a run that has a schema of its own.
    """

    watched_files: Tuple[str, ...] = ("semantic_annotations.py", "mixins.py")
    """
    The ontology's own files a run may amend and must not leave amended.
    """

    templates: PipelineTemplates = field(default_factory=PipelineTemplates)
    """
    Where the file a reset ontology is written from is kept.
    """

    empty_classes_template: str = "empty_generated_classes.py"
    """
    The generated classes as a run finds them: the imports and nothing else.
    """

    @property
    def name(self) -> str:
        return "put the ontology back and give the run a schema"

    @property
    def repository(self) -> Path:
        """
        :return: The checkout the ontology is written in.
        """
        return self.settings.repository

    @property
    def generated_classes_file(self) -> Path:
        """
        :return: The ontology's own file that a run's generated classes are kept out of.
        """
        return Path(SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value)

    def carry_out(self) -> None:
        """
        Reset the ontology, rebuild the ORM, read it out and build the run's tables.
        """
        self.refuse_an_amended_ontology()

        self.logger.info("emptying the classes generated for an earlier scene ...")
        self.reset_generated_classes()
        self.logger.info("  %s", self.generated_classes_file)

        self.logger.info("rebuilding the ORM without them ...")
        self.rebuild_orm()

        schema = RunSchema.for_run(self.run.directory)
        self.logger.info("making the run its own schema in the database ...")
        schema.create()

        self.logger.info("reading the ontology out and building its tables ...")
        prepared = self.export_and_build(schema)
        self.logger.info(
            "  %s classes, %s mixins -> %s",
            prepared.classes,
            prepared.mixins,
            self.run.path(RunFile.TAXONOMY),
        )
        self.logger.info("  %s tables in schema %s", prepared.tables, schema.name)

    def hand_written_changes(self) -> List[str]:
        """
        Report the ontology's own files a run has been left holding changes to.

        :return: The paths that differ from what is committed, empty when none do.
        """
        finished = subprocess.run(
            ["git", "-C", str(self.repository), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        return [
            line[3:]
            for line in finished.stdout.splitlines()
            if any(line.endswith(name) for name in self.watched_files)
        ]

    def refuse_an_amended_ontology(self) -> None:
        """
        Refuse to start against an ontology an earlier run left edited.

        :raises OntologyLeftAmendedError: If it is edited and that was not asked for.
        """
        amended = self.hand_written_changes()
        if not amended:
            return
        if not self.settings.ignore_amendments:
            raise OntologyLeftAmendedError(amended_paths=amended)
        self.logger.warning("running against an amended ontology, as asked:")
        for path in amended:
            self.logger.warning("  %s", path)

    def reset_generated_classes(self) -> None:
        """
        Empty the classes generated for an earlier scene.
        """
        self.generated_classes_file.write_text(
            self.templates.render(self.empty_classes_template)
        )

    def export_and_build(self, schema: RunSchema) -> PreparedOntology:
        """
        Read the ontology out into the run, and build the tables it asks for.

        Both in one new interpreter: the ORM was just rewritten and this one is holding
        the version from before that, and the tables have to be made by the ORM that is
        about to write to them so a run never meets a table another run left standing.

        :param schema: The schema the run writes into.
        :return: What was read out and what was built.
        :raises SubprocessStepFailedError: If either fails.
        """
        self.in_new_interpreter(
            OntologyExport,
            what="reading the ontology out and building the run's tables",
            environment=schema.environment(),
        )
        return self.run.read_record(RunFile.PREPARED_ONTOLOGY, PreparedOntology)
