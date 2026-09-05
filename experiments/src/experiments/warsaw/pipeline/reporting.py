"""
How the pipeline says what it is doing.

A run is minutes of work over seven steps, and what it says as it goes is the only
account of it until the report is written at the end. That account goes to logging
rather than to standard output, so a caller decides where it lands and a step does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class Reporting:
    """
    Something that says what it is doing.
    """

    @property
    def logger(self) -> logging.Logger:
        """
        :return: Where this says what it is doing, named for the module it is declared in
            so a caller can quieten one part of a run without quietening the rest.
        """
        return logging.getLogger(type(self).__module__)
