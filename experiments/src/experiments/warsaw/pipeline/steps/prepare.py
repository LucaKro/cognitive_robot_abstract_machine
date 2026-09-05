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

import semantic_digital_twin
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationFilePaths,
)
from typing_extensions import List, Tuple

from experiments.warsaw.exceptions import (
    OntologyLeftAmendedError,
    SubprocessStepFailedError,
)
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.run_database import RunSchema
from experiments.warsaw.pipeline.steps.step import PipelineStep


@dataclass
class PreparedOntology:
    """
    What reading the ontology out and building its tables produced.
    """

    classes: int
    """
    How many classes the run may name.
    """

    mixins: int
    """
    How many of them a new class can be composed from.
    """

    tables: int
    """
    How many tables were built in the run's schema.
    """


@dataclass
class PrepareRun(PipelineStep):
    """
    The ontology as it is committed, read out into a run that has a schema of its own.
    """

    watched_files: Tuple[str, ...] = ("semantic_annotations.py", "mixins.py")
    """
    The ontology's own files a run may amend and must not leave amended.
    """

    templates: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "templates"
    )
    """
    Where the file a reset ontology is written from is kept.
    """

    empty_classes_template: str = "empty_generated_classes.py.template"
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
        return Path(semantic_digital_twin.__file__).resolve().parents[3]

    @property
    def generated_classes_file(self) -> Path:
        """
        :return: The ontology's own file that a run's generated classes are kept out of.
        """
        return Path(SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value)

    @property
    def orm_generator(self) -> Path:
        """
        :return: The script that rebuilds the ORM from the classes that are left.
        """
        root = Path(semantic_digital_twin.__file__).resolve().parent
        return root.parent.parent / "scripts" / "generate_orm.py"

    def carry_out(self) -> None:
        """
        Reset the ontology, rebuild the ORM, read it out and build the run's tables.
        """
        self.refuse_an_amended_ontology()

        self.logger.info("emptying the classes generated for an earlier scene ...")
        self.reset_generated_classes()
        self.logger.info("  %s", self.generated_classes_file)

        self.logger.info("rebuilding the ORM without them ...")
        self.regenerate_orm()

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
            (self.templates / self.empty_classes_template).read_text()
        )

    def regenerate_orm(self) -> None:
        """
        Rebuild the ORM from the classes that are left.

        The generator reads the interface it is about to replace, and the one standing
        there still names the classes just removed -- so it cannot be imported and the
        rebuild dies on the very staleness it was run to cure. Moved aside, the
        generator builds from the ontology alone; put back if it fails, so a failure
        costs nothing.

        :raises SubprocessStepFailedError: If the rebuild fails.
        """
        interface = (
            Path(semantic_digital_twin.__file__).resolve().parent
            / "orm"
            / "ormatic_interface.py"
        )
        aside = interface.with_suffix(".py.aside")
        if interface.exists():
            interface.replace(aside)

        rebuilt = False
        try:
            self.in_new_interpreter(
                "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')",
                [str(self.orm_generator)],
                what="rebuilding the ORM",
            )
            rebuilt = interface.exists()
        finally:
            if rebuilt:
                aside.unlink(missing_ok=True)
            elif aside.exists():
                aside.replace(interface)

        if not rebuilt:
            raise SubprocessStepFailedError(
                what="rebuilding the ORM",
                output=f"the generator finished but wrote no {interface}",
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
        # The export runs before the ORM is imported, and has to. The generated interface
        # names every mapped class, robots and their fingers included, and importing it
        # puts all of them into the annotation hierarchy the export then walks: the same
        # ontology came out as 441 classes rather than 139, and every question would have
        # carried three hundred robot parts for a model to choose a countertop from.
        program = (
            "import sys\n"
            "from pathlib import Path\n"
            "from semantic_digital_twin.semantic_annotations.taxonomy_export import "
            "export_taxonomy\n"
            "from semantic_digital_twin.world_description.world_entity import "
            "SemanticAnnotation\n"
            "taxonomy = export_taxonomy(SemanticAnnotation, Path(sys.argv[1]))\n"
            "from semantic_digital_twin.orm.ormatic_interface import Base\n"
            "from semantic_digital_twin.orm.utils import "
            "semantic_digital_twin_sessionmaker\n"
            "Base.metadata.create_all(bind=semantic_digital_twin_sessionmaker()().bind)\n"
            "print(len(taxonomy['classes']), len(taxonomy['part_whole_mixins']), "
            "len(Base.metadata.tables))\n"
        )
        printed = self.in_new_interpreter(
            program,
            [str(self.run.path(RunFile.TAXONOMY))],
            what="reading the ontology out and building the run's tables",
            environment=schema.environment(),
        )
        classes, mixins, tables = printed.split()[-3:]
        return PreparedOntology(
            classes=int(classes), mixins=int(mixins), tables=int(tables)
        )
