"""
Reading back what a command wrote.

Every command in the pipeline writes records declared in the module being run, and a
module run with ``python -m`` is imported as ``__main__``. A record that takes its class
name from there names a class nothing can import, so the file it was written into can
never be read again -- which is only noticed much later, by whoever tries.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from krrood.adapters.json_field import JSONField

from experiments.warsaw.bases import JsonRecord

# %% a command that writes a record

WRITES_A_RECORD = (
    "test.experiments_test.dataset.warsaw_evaluation.record_writing_command"
)
"""
A module that writes one record while being run as a command.
"""


def written_by_the_command() -> dict:
    """
    Run the command and read the record it wrote.
    """
    repository = Path(__file__).resolve().parents[3]
    finished = subprocess.run(
        [sys.executable, "-m", WRITES_A_RECORD],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(finished.stdout)


# %% what the record says it is


def test_a_record_is_written_under_a_name_that_can_be_imported():
    """
    The name is what a reader resolves the class from, so a record naming ``__main__``
    is a file nobody can read.
    """
    written = written_by_the_command()

    assert written[JSONField.TYPE] == f"{WRITES_A_RECORD}.RecordWrittenByACommand"


def test_a_record_a_command_wrote_can_be_read_back():
    """
    What the name is for, end to end.
    """
    read = JsonRecord.from_json(written_by_the_command())

    assert read.what == "written"
