from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing_extensions import List

from krrood.exceptions import DataclassException


@dataclass
class WarsawSceneNotFoundError(DataclassException, FileNotFoundError):
    """
    Raised when a directory holds no Warsaw scene mesh.
    """

    directory: Path
    """
    The directory that was searched.
    """

    scene_mesh_pattern: str
    """
    The pattern a scene mesh was looked for under.
    """

    def error_message(self) -> str:
        return (
            f"No Warsaw scene mesh matching '{self.scene_mesh_pattern}' in "
            f"'{self.directory}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Point the loader at a scene directory, which holds the one mesh the scene "
            "is written as."
        )


@dataclass
class AmbiguousWarsawSceneError(DataclassException, FileNotFoundError):
    """
    Raised when a directory holds more than one mesh, leaving undecided which of them
    the scene is.
    """

    directory: Path
    """
    The directory that was searched.
    """

    scene_meshes: List[Path]
    """
    The meshes it holds.
    """

    def error_message(self) -> str:
        names = ", ".join(sorted(path.name for path in self.scene_meshes))
        return f"'{self.directory}' holds {len(self.scene_meshes)} meshes: {names}."

    def suggest_correction(self) -> str:
        return (
            "Give the scene a directory of its own, or name its mesh file directly with "
            "WarsawScene.from_file."
        )


@dataclass
class WarsawLabelsMissingError(DataclassException, ValueError):
    """
    Raised when a scene mesh carries no per-face class labels.
    """

    scene_mesh: Path
    """
    The mesh that was read.
    """

    def error_message(self) -> str:
        return f"'{self.scene_mesh}' carries no per-face class labels."

    def suggest_correction(self) -> str:
        return (
            "A scene writes its labels as one integer face property per class. Only the "
            "PLY reader keeps them, under the mesh's '_ply_raw' metadata, so a mesh that "
            "was re-exported or loaded with processing has lost them and has to be "
            "written again."
        )


@dataclass
class WarsawLabelsMisalignedError(DataclassException, ValueError):
    """
    Raised when the faces a world was built from are not the faces the labels were
    written for, which would leave every segment pointing at another object's geometry.
    """

    scene_mesh: Path
    """
    The mesh that was read.
    """

    labelled_faces: int
    """
    How many faces the labels were written for.
    """

    loaded_faces: int
    """
    How many faces the world's scene body ended up with.
    """

    def error_message(self) -> str:
        return (
            f"'{self.scene_mesh}' was labelled for {self.labelled_faces} faces, but the "
            f"world's scene body holds {self.loaded_faces} of them."
        )

    def suggest_correction(self) -> str:
        return (
            "The mesh has to reach the world unprocessed: welding vertices or dropping "
            "degenerate faces renumbers the faces and leaves every label pointing at "
            "another one."
        )


# %% preparing a run


@dataclass
class DatabaseNotConfiguredError(DataclassException, RuntimeError):
    """
    Raised when a step that reads or writes worlds has no database to reach.
    """

    variable: str
    """
    The environment variable the connection is read from.
    """

    def error_message(self) -> str:
        return f"'{self.variable}' is not set, so there is no database to reach."

    def suggest_correction(self) -> str:
        return (
            f"Export '{self.variable}' with a 'postgresql+psycopg://' URI before running "
            "the pipeline, or run it with persistence turned off, which stops it at the "
            "split's report."
        )


@dataclass
class OntologyLeftAmendedError(DataclassException, RuntimeError):
    """
    Raised when a run would start against an ontology an earlier run left edited.

    A run must start from what is committed. Classes generated for one scene and mixins
    one room argued for would otherwise be in force for the next, where nothing
    questions them and nobody remembers they were ever in doubt.
    """

    amended_paths: List[str]
    """
    The ontology's own files that differ from what is committed.
    """

    def error_message(self) -> str:
        return "The ontology's own files are left amended: " + ", ".join(
            sorted(self.amended_paths)
        )

    def suggest_correction(self) -> str:
        return (
            "Put them back by reverting the run that applied them, or start the pipeline "
            "with ignore_amendments set, which runs against them deliberately."
        )


@dataclass
class SubprocessStepFailedError(DataclassException, RuntimeError):
    """
    Raised when work a step handed to a new interpreter did not finish.

    Some work has to happen in an interpreter that started after the ontology or the ORM
    was rewritten, because the one asking is holding the version from before that.
    """

    what: str
    """
    What the interpreter was asked to do.
    """

    output: str
    """
    What it said before it stopped.
    """

    shown: int = 2000
    """
    How much of that output the message quotes, counting back from the end.
    """

    def error_message(self) -> str:
        return f"{self.what} failed:\n{self.output.strip()[-self.shown :]}"

    def suggest_correction(self) -> str:
        return (
            "Read the output above: it is the failure as the interpreter reported it, "
            "not a summary of it."
        )


@dataclass
class RunOutputAlreadyWrittenError(DataclassException, FileExistsError):
    """
    Raised when a step would write over output that is already there.

    Writing beside it is how a run comes to be part one run and part another: the files a
    step does not happen to rewrite stay as an earlier pass left them, and nothing says so.
    """

    directory: Path
    """
    The run the step meant to write into.
    """

    written: List[str]
    """
    The files that are already there.
    """

    def error_message(self) -> str:
        return f"'{self.directory}' already holds " + ", ".join(sorted(self.written))

    def suggest_correction(self) -> str:
        return (
            "Write into a run of its own, or let the step overwrite, which is what a step "
            "run twice in one run needs -- the second time knowing something the first "
            "did not."
        )


# %% the classes a run generates


@dataclass
class GeneratedClassesAlreadyImportedError(DataclassException, ImportError):
    """
    Raised when a run's generated classes cannot be the ones in use.

    By the time the module has been imported from somewhere else, everything holding one
    of those classes holds the wrong one, so pointing at the run's file now would leave
    two versions of the same class in one interpreter.
    """

    module_name: str
    """
    The name the classes are imported under.
    """

    imported_from: str
    """
    Where the module standing under that name was read from.
    """

    wanted: Path
    """
    The run's own file, which was to have been used instead.
    """

    def error_message(self) -> str:
        return (
            f"'{self.module_name}' was already imported from '{self.imported_from}', so "
            f"'{self.wanted}' cannot be the file in use."
        )

    def suggest_correction(self) -> str:
        return (
            "Point at the run's classes before anything imports the ORM or the "
            "annotations, which is why it is done at the top of a step."
        )


# %% reading a world back


@dataclass
class NoWorldRecordedError(DataclassException, LookupError):
    """
    Raised when a step needs the world an earlier step wrote and none was recorded.
    """

    step: str
    """
    The step whose record was read.
    """

    def error_message(self) -> str:
        return f"The {self.step} recorded no world in the database."

    def suggest_correction(self) -> str:
        return (
            "Run the pipeline with persistence turned on: everything after the split "
            "reads a world back, so without it there is nothing to read."
        )


@dataclass
class WorldNotInDatabaseError(DataclassException, LookupError):
    """
    Raised when the world a step was told to read is not in the database it is looking
    in.
    """

    world_id: int
    """
    The id that was looked for.
    """

    def error_message(self) -> str:
        return f"No world in the database with id {self.world_id}."

    def suggest_correction(self) -> str:
        return (
            "Check that the connection points at the schema the run wrote into: a run "
            "writes into a schema of its own, and its worlds are invisible from another."
        )


# %% aligning a database that already holds worlds


@dataclass
class ColumnsCannotBeAddedError(DataclassException, RuntimeError):
    """
    Raised when a column the ORM asks for cannot be added to a table that holds rows.

    The drift a regenerated ORM leaves is additive and can be closed by adding columns,
    which costs no stored world. A column that is required and has no value a stored row
    could have been written with is not guessed at.
    """

    refusals: List[str]
    """
    One sentence per column that cannot be added.
    """

    def error_message(self) -> str:
        return (
            "These columns cannot be added to a database that already holds rows:\n  "
            + "\n  ".join(self.refusals)
        )

    def suggest_correction(self) -> str:
        return (
            "The stored worlds predate them, so closing this is a decision about those "
            "worlds rather than a migration."
        )
