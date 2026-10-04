"""
Work done in an interpreter that starts after the ontology or the ORM was rewritten.

The interpreter asking is holding the version from before that, so it cannot do the work
itself however carefully it re-imports.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Dict, Optional, Sequence

from experiments.warsaw.exceptions import SubprocessStepFailedError


@dataclass
class NewInterpreter:
    """
    One piece of work, handed to an interpreter of its own.
    """

    entry: type
    """
    The class doing the work, whose module runs it when run as a program.
    """

    directory: Path
    """
    The run's directory, which the work is handed first.
    """

    what: str
    """
    What the work is, for the failure message.
    """

    environment: Optional[Dict[str, str]] = None
    """
    The environment to run it in, or None for this one's.
    """

    arguments: Sequence[str] = ()
    """
    What else the work is handed, after the run's directory.
    """

    def carry_out(self) -> str:
        """
        Run the work and wait for it to finish.

        :return: What it printed.
        :raises SubprocessStepFailedError: If it did not finish.
        """
        finished = subprocess.run(
            [
                sys.executable,
                "-m",
                self.entry.__module__,
                str(self.directory.resolve()),
                *self.arguments,
            ],
            capture_output=True,
            text=True,
            env=self.environment,
        )
        if finished.returncode != 0:
            raise SubprocessStepFailedError(what=self.what, output=finished.stderr)
        return finished.stdout
