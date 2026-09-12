"""
Telling a model what each class of the ontology means, rather than only its name.

A class reaches a model as its name and its bases. Whether it also carries the first
sentence of its docstring is something a run is told, because it is a real change to
every question the run asks and the runs already made were not asked that way.
"""

from __future__ import annotations

from pathlib import Path

from experiments.warsaw.pipeline.provenance import settings_to_json
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.prepare import PrepareRun
from experiments.warsaw.pipeline.run import Run

# %% what a run is told


def test_the_classes_are_not_described_unless_asked():
    """
    The runs already made were not asked that way, so their questions stay what they
    were.
    """
    assert PipelineSettings().describe_the_classes is False


def test_what_the_run_was_told_is_written_down(tmp_path: Path):
    """
    The taxonomy a run wrote cannot say whether it was asked for summaries when none of
    its classes happened to carry one, so the provenance says instead.
    """
    written = settings_to_json(PipelineSettings(describe_the_classes=True))
    assert written["describe_the_classes"] is True


# %% what the exporting interpreter is asked to do


def describing(asked: bool) -> str:
    """
    :param asked: Whether the run was told to describe its classes.
    :return: The program the step hands to the interpreter that reads the ontology out.
    """
    return PrepareRun(
        settings=PipelineSettings(describe_the_classes=asked), run=Run(directory=Path())
    ).taxonomy_program()


def test_the_exporting_interpreter_is_asked_for_summaries_when_the_run_was():
    """
    The export happens in an interpreter of its own, so what it is asked for has to
    survive the crossing.
    """
    assert "include_summaries=True" in describing(True)


def test_it_is_asked_for_none_when_the_run_was_not():
    """
    And a run that was not told to describe its classes exports what it always did.
    """
    assert "include_summaries=False" in describing(False)
