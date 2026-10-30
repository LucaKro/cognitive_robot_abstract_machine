"""
Refusing to start a run against an ontology an earlier run left edited.

A run amends the ontology's own source and puts it back when it ends. If it did not put
it back -- it was killed, or it failed between the edit and the revert -- the next run
would measure against a taxonomy nobody chose, and its numbers would not be comparable
with any other run's. So a run looks before it starts, and says so rather than starting.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from experiments.warsaw.exceptions import OntologyLeftAmendedError
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.prepare import PrepareRun

# %% a checkout of this test's own


@dataclass
class PrepareRunInCheckout(PrepareRun):
    """
    The step, looking at a checkout a test made rather than the one it is running in.
    """

    checkout: Path = None
    """
    The checkout to look at.
    """

    @property
    def repository(self) -> Path:
        """
        :return: The checkout the test wrote.
        """
        return self.checkout


def a_checkout(tmp_path: Path, files: dict) -> Path:
    """
    :param tmp_path: Where to make it.
    :param files: Per file name, what it holds when it is committed.
    :return: A checkout holding those files, with nothing changed since.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name, content in files.items():
        (checkout / name).write_text(content)
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "test"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "committed"],
    ):
        subprocess.run(command, cwd=checkout, check=True, capture_output=True)
    return checkout


def preparing(checkout: Path, ignore_amendments: bool, finished_run) -> PrepareRun:
    """
    :param checkout: The checkout to look at.
    :param ignore_amendments: Whether a run was told to start anyway.
    :param finished_run: A run for the step to belong to.
    :return: The step, ready to be asked what it finds.
    """
    return PrepareRunInCheckout(
        settings=PipelineSettings(ignore_amendments=ignore_amendments),
        run=finished_run,
        checkout=checkout,
    )


# %% what it finds


def test_an_untouched_ontology_leaves_nothing_to_report(tmp_path, finished_run):
    """
    Nothing changed since the commit is the state a run expects to start from.
    """
    checkout = a_checkout(
        tmp_path, {"semantic_annotations.py": "class Cabinet: pass\n"}
    )

    assert preparing(checkout, False, finished_run).hand_written_changes() == []


def test_an_edited_ontology_file_is_reported(tmp_path, finished_run):
    """
    The file a run failed to put back is exactly what the next run has to be told about.
    """
    checkout = a_checkout(
        tmp_path, {"semantic_annotations.py": "class Cabinet: pass\n"}
    )
    (checkout / "semantic_annotations.py").write_text(
        "class Cabinet(HasDrawers): pass\n"
    )

    assert preparing(checkout, False, finished_run).hand_written_changes() == [
        "semantic_annotations.py"
    ]


def test_a_changed_file_that_is_not_the_ontology_is_left_alone(tmp_path, finished_run):
    """
    Only the ontology's own files are watched; a run does not police the whole checkout.
    """
    checkout = a_checkout(
        tmp_path,
        {"semantic_annotations.py": "class Cabinet: pass\n", "notes.md": "notes\n"},
    )
    (checkout / "notes.md").write_text("changed\n")

    assert preparing(checkout, False, finished_run).hand_written_changes() == []


# %% what it does about it


def test_a_run_refuses_to_start_against_an_amended_ontology(tmp_path, finished_run):
    """
    Measuring against a taxonomy nobody chose is worse than not starting.
    """
    checkout = a_checkout(tmp_path, {"mixins.py": "class HasDrawers: pass\n"})
    (checkout / "mixins.py").write_text("class HasDrawers: pass  # edited\n")

    with pytest.raises(OntologyLeftAmendedError):
        preparing(checkout, False, finished_run).refuse_an_amended_ontology()


def test_a_run_told_to_ignore_the_amendments_starts_anyway(tmp_path, finished_run):
    """
    Deliberately running against an edited ontology is a thing a person can ask for.
    """
    checkout = a_checkout(tmp_path, {"mixins.py": "class HasDrawers: pass\n"})
    (checkout / "mixins.py").write_text("class HasDrawers: pass  # edited\n")

    preparing(checkout, True, finished_run).refuse_an_amended_ontology()


def test_an_untouched_ontology_starts_without_being_asked_about(tmp_path, finished_run):
    """
    The ordinary case raises nothing.
    """
    checkout = a_checkout(tmp_path, {"mixins.py": "class HasDrawers: pass\n"})

    preparing(checkout, False, finished_run).refuse_an_amended_ontology()
