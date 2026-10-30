"""
A command that writes one record, to show what name the record is written under.

A module run with ``python -m`` is imported as ``__main__``, so a record class declared
beside the command that writes it takes ``__main__`` as its module. Written into a file
that way, the record names a class nothing can import and the file can never be read
back. This stands in for every command in the pipeline that writes its own records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from experiments.warsaw.bases import JsonRecord


@dataclass(frozen=True)
class RecordWrittenByACommand(JsonRecord):
    """
    One record, written by the module that declares it while it is being run.
    """

    what: str = "written"
    """
    Anything at all, so that the record has a field.
    """


def main() -> int:
    """
    Write one record to standard output.

    :return: Zero once it is written.
    """
    print(json.dumps(RecordWrittenByACommand().to_json()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
