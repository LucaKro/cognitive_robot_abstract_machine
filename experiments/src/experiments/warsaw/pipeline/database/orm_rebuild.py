"""
Rebuild the ORM from the ontology and the classes a run generated.

The rebuild runs in an interpreter of its own, since the one that asks for it is holding
the ORM from before it:

    python -m experiments.warsaw.pipeline.database.orm_rebuild <run directory>

Nothing but this module and the run's generated classes may be imported before the
rebuild: the classes have to be in place before anything imports the ontology.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import semantic_digital_twin

from experiments.warsaw.exceptions import SubprocessStepFailedError
from experiments.warsaw.pipeline.new_interpreter import NewInterpreter
from experiments.warsaw.pipeline.run_classes import GeneratedClasses


@dataclass
class OrmRebuild:
    """
    The ORM regenerated with a run's generated classes mapped.
    """

    directory: Path
    """
    The run's directory, whose generated classes are mapped beside the ontology's own.
    """

    @staticmethod
    def interface() -> Path:
        """
        :return: The generated ORM interface the rebuild replaces.
        """
        return (
            Path(semantic_digital_twin.__file__).resolve().parent
            / "orm"
            / "ormatic_interface.py"
        )

    @staticmethod
    def generator() -> Path:
        """
        :return: The script that generates the ORM interface.
        """
        package = Path(semantic_digital_twin.__file__).resolve().parent
        return package.parent.parent / "scripts" / "generate_orm.py"

    def run_in_new_interpreter(self) -> None:
        """
        Build the ORM anew so it maps the classes this run generated.

        The ORM is one file for the whole repository while generated classes belong to a
        run, so anything rebuilding it without this run's directory on the search path
        -- another run, or the test suite -- leaves it unable to read this run's world.
        It is rebuilt whether or not it already matches, because asking is not cheaper:
        an ORM built for another run's classes raises while being imported rather than
        answering, and an interpreter that has imported a stale one holds it however
        carefully it imports again.

        The standing interface is moved aside first, because the generator reads the one
        it is about to replace and a stale interface therefore kills the rebuild run to
        cure it. It is put back when the rebuild writes nothing, so a failure costs
        nothing.

        :raises RunClassTakenOverByTheOntologyError: If the ontology has since gained a
            class this run generated, which would leave two classes of one name and an
            ORM nothing can import.
        :raises SubprocessStepFailedError: If the rebuild fails or writes no interface.
        """
        GeneratedClasses(directory=self.directory).refuse_classes_taken_over()

        interface = self.interface()
        aside = interface.with_suffix(".py.aside")
        if interface.exists():
            interface.replace(aside)

        rebuilt = False
        try:
            NewInterpreter(
                entry=type(self), directory=self.directory, what="rebuilding the ORM"
            ).carry_out()
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

    def carry_out(self) -> None:
        """
        Make the run's generated classes the ones this interpreter means, then generate
        the ORM from what that leaves in the ontology.
        """
        GeneratedClasses(directory=self.directory).use()
        specification = importlib.util.spec_from_file_location(
            "generate_orm", self.generator()
        )
        generator = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(generator)
        generator.generate_orm()


if __name__ == "__main__":
    OrmRebuild(directory=Path(sys.argv[1])).carry_out()
