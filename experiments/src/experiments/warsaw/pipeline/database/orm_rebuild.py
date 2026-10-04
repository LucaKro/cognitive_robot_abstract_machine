"""
Rebuild the ORM from the ontology and the classes a run generated.

Run as a program in an interpreter of its own, since the one that asks for the rebuild
is holding the ORM from before it:

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

from experiments.warsaw.pipeline.run_classes import GeneratedClasses


@dataclass
class OrmRebuild:
    """
    The ORM regenerated in this interpreter, with a run's generated classes mapped.
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
