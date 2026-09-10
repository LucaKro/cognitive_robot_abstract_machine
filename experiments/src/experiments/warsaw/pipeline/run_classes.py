"""
Keep the classes a run generates inside that run.

A scene needs classes the taxonomy does not have -- a ``Ceiling``, a ``KitchenIsland``.
Written into the ontology's own package, as they were, they are a tracked file that
every run leaves modified, one careless ``git add`` away from becoming part of the
shared ontology, and importable by the next run's taxonomy export.

They belong to the run that proposed them, so they are written into its directory. The
run's directory is then put at the *front* of the annotations package's search path,
which makes its file the one imported under the name the ORM refers to them by, ahead of
the package's own empty one. The ORM generator finds them by walking that path, so it
maps them without being told anything.

This has to be done before anything imports the ORM or the classes, which is why it is a
call at the top of a step rather than something a step can opt into later.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import semantic_digital_twin
from typing_extensions import Optional

from experiments.warsaw.exceptions import (
    GeneratedClassesAlreadyImportedError,
    SubprocessStepFailedError,
)


@dataclass
class GeneratedClasses:
    """
    The classes one run generated, and the name they are imported under.
    """

    directory: Path
    """
    The run's directory, which is where they are written.
    """

    module_name: str = "semantic_digital_twin.semantic_annotations.generated_classes"
    """
    The name the ORM refers to a generated class by, whatever directory it is written
    in.
    """

    file_name: str = "generated_classes.py"
    """
    What the file is called.
    """

    holding_directory: str = "annotations"
    """
    The directory inside the run that holds the generated classes and nothing else.

    Its own directory rather than the run's, because what is put on the annotations
    package's search path becomes part of that package: with the run's directory there,
    the inspector script a finished run leaves behind is walked by the ORM generator and
    mapped as though it were an annotation class, and a run cannot be annotated twice.
    """

    @property
    def path(self) -> Path:
        """
        :return: Where the run's generated classes are written.
        """
        return self.searched_directory / self.file_name

    @property
    def searched_directory(self) -> Path:
        """
        :return: The directory put on the annotations package's search path.
        """
        return Path(self.directory).resolve() / self.holding_directory

    @property
    def were_generated(self) -> bool:
        """
        :return: Whether the run generated any.
        """
        return self.path.exists()

    @property
    def orm_generator(self) -> Path:
        """
        :return: The script that rebuilds the ORM.
        """
        root = Path(semantic_digital_twin.__file__).resolve().parent
        return root.parent.parent / "scripts" / "generate_orm.py"

    def rebuild_orm(self) -> None:
        """
        Build the ORM anew so it maps the classes this run generated.

        The ORM is one file for the whole repository while generated classes belong to a
        run, so anything rebuilding it without this run's directory on the search path
        -- another run, or the test suite -- leaves it unable to read this run's world.
        It is rebuilt whether or not it already matches, because asking is not cheaper:
        an ORM built for another run's classes raises while being imported rather than
        answering, and an interpreter that has imported a stale one holds it however
        carefully it imports again.

        :raises SubprocessStepFailedError: If the rebuild fails.
        """
        self._in_new_interpreter(
            "import importlib, sys\n"
            "from pathlib import Path\n"
            "from experiments.warsaw.pipeline.run_classes import GeneratedClasses\n"
            "GeneratedClasses(directory=Path(sys.argv[1])).use()\n"
            "specification = importlib.util.spec_from_file_location("
            "'generate_orm', sys.argv[2])\n"
            "generator = importlib.util.module_from_spec(specification)\n"
            "specification.loader.exec_module(generator)\n"
            "generator.generate_orm()\n",
            what="rebuilding the ORM with the run's generated classes",
        )

    def _in_new_interpreter(self, program: str, what: str) -> str:
        """
        Run a program in an interpreter that starts after the ORM was last written.

        :param program: The program to run, given this run's directory and the
            generator.
        :param what: What it is doing, for the failure message.
        :return: What it printed.
        :raises SubprocessStepFailedError: If it did not finish.
        """
        finished = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(Path(self.directory).resolve()),
                str(self.orm_generator),
            ],
            capture_output=True,
            text=True,
        )
        if finished.returncode != 0:
            raise SubprocessStepFailedError(what=what, output=finished.stderr)
        return finished.stdout

    def use(self) -> Optional[ModuleType]:
        """
        Make this run's generated classes the ones this interpreter means.

        :return: The module, once something imports it, or None if the run generated
            none.
        :raises GeneratedClassesAlreadyImportedError: If the classes were already
            imported from somewhere else, since by then everything holding one of them
            holds the wrong one.
        """
        if not self.were_generated:
            return None

        already = sys.modules.get(self.module_name)
        if already is not None:
            imported_from = Path(already.__file__ or "").resolve()
            if imported_from == self.path:
                return already
            raise GeneratedClassesAlreadyImportedError(
                module_name=self.module_name,
                imported_from=str(imported_from),
                wanted=self.path,
            )

        import semantic_digital_twin.semantic_annotations as annotations

        # In front of the package's own file, so a walk of the package finds this one
        # first.
        directory = str(self.searched_directory)
        if directory not in annotations.__path__:
            annotations.__path__.insert(0, directory)
        return sys.modules.get(self.module_name)
