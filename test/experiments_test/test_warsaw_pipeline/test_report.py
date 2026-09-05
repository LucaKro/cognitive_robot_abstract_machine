"""
Gathering a run's numbers into the one page anyone reads afterwards.

A run's numbers are spread over six files and a terminal that has scrolled away, and the
question asked of a run afterwards is usually how much of it went through rather than
what any one step said. A run that stopped early is worth reporting on too.
"""

from __future__ import annotations

from string import Template

from experiments.warsaw.pipeline.report import RunReport
from experiments.warsaw.pipeline.run import Run, RunFile

# %% a run that finished


def test_the_report_names_both_worlds_the_run_wrote(finished_run, split):
    """
    The split world is what the classification and the pairings were decided against;
    the annotated one is what came out.
    """
    report = RunReport(run=finished_run).markdown()
    assert f"**{split.annotated_world_id}** annotated" in report
    assert f"{split.world_id} as it was split" in report


def test_the_report_counts_what_every_step_wrote(
    finished_run, relations, vocabulary, adjudications, split, classifications
):
    """
    Every number is read from the file the step that produced it wrote, so a step that
    changes what it writes changes the report rather than quietly disagreeing with it.
    """
    report = RunReport(run=finished_run).markdown()
    assert f"{len(relations.segments)} labelled objects" in report
    assert f"{len(vocabulary.labels)} labels" in report
    assert f"{len(adjudications.ownership)} class patterns" in report
    assert f"{len(split.bodies)} bodies," in report
    assert f"{len(split.pairings)} pairings carried past the split" in report


def test_the_objects_that_lost_every_face_are_named_with_what_took_them(
    finished_run, split
):
    """
    An ownership answer covers a whole class of objects, so a wrong one empties every
    object of that kind at once.

    A report that merely said ten cabinets are missing would not say which answer to
    look at.
    """
    report = RunReport(run=finished_run).markdown()
    assert f"### {len(split.emptied)} objects lost every face" in report
    for name, took in split.emptied.items():
        assert f"`{name}` -> " in report
        assert f"{next(iter(took))} ({next(iter(took.values()))})" in report


def test_every_class_given_is_counted(finished_run, classifications):
    """
    How often each class was given is what says whether a run discriminated at all.
    """
    report = RunReport(run=finished_run).markdown()
    for name in {
        one.class_name for one in classifications.bodies.values() if one.class_name
    }:
        assert f"`{name}`" in report


def test_the_report_says_how_to_open_what_was_built(finished_run):
    """
    The script the run leaves behind opens its world without anyone knowing an id.
    """
    assert RunFile.INSPECTOR.value in RunReport(run=finished_run).markdown()


# %% a run that stopped early


def test_a_run_that_wrote_nothing_still_reports(tmp_path):
    """
    A run that stopped on its first step is exactly the run someone wants a report of.
    """
    report = RunReport(run=Run.create(tmp_path)).markdown()
    assert "0 labelled objects" in report
    assert "0 bodies," in report


def test_a_run_that_lost_nothing_says_nothing_about_losses(tmp_path):
    """
    A heading with nothing under it reads as a step that failed.
    """
    assert "lost every face" not in RunReport(run=Run.create(tmp_path)).markdown()


# %% what is left behind beside the report


def test_the_inspector_is_written_with_the_worlds_this_run_wrote(tmp_path, split):
    """
    Written by the run that built them, so the ids are the ones it wrote under and
    nobody has to be told them.
    """
    run = Run.create(tmp_path)
    run.write_json(RunFile.SPLIT, split.to_json())
    RunReport(run=run).write_inspector(Template("annotated=$annotated split=$split"))
    assert run.path(RunFile.INSPECTOR).read_text() == (
        f"annotated={split.annotated_world_id} split={split.world_id}"
    )


def test_the_report_is_written_into_the_run_it_is_about(tmp_path):
    """
    A run's account of itself lives with the run, not beside it.
    """
    run = Run.create(tmp_path)
    written = RunReport(run=run).write()
    assert run.path(RunFile.REPORT).read_text() == written
    assert written.startswith(f"# {run.name}")
