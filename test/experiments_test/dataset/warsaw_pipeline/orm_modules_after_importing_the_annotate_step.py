"""
Report which ORM modules importing the annotate step brought with it.

Run as a program in an interpreter of its own, because what is already imported cannot
be un-imported and the answer in a test process is whatever that process did earlier.
"""

from __future__ import annotations

import sys

import experiments.warsaw.pipeline.steps.annotate  # noqa: F401

print(
    "\n".join(name for name in sys.modules if name.endswith("ormatic_interface")),
    end="",
)
