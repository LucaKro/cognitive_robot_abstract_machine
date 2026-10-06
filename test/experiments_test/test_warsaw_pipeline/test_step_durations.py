"""
How long each step of a run took, written where the run's other records are.

Most of a run is spent in one or two of its steps, and which ones is not apparent from
the outside: the model calls a run makes are recorded with their own elapsed time, and
they turn out to be a small part of it. What the rest went on is only answerable if the
run says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.warsaw.exceptions import NoSegmentsGivenError
from experiments.warsaw.pipeline.pipeline import WarsawPipeline
from experiments.warsaw.pipeline.records import StepDuration, StepDurations
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.step import PipelineStep

# %% steps standing in for the ones a run carries out


@dataclass
class StepThatFinishes(PipelineStep):
    """
    Stands in for a step that does its work and returns.
    """

    @property
    def name(self) -> str:
        return "finishes"

    def carry_out(self) -> None:
        return None


@dataclass
class StepTheRunCanDoWithout(PipelineStep):
    """
    Stands in for a step whose failure a run is told to carry on from.
    """

    @property
    def name(self) -> str:
        return "can be done without"

    @property
    def is_optional(self) -> bool:
        return True

    def carry_out(self) -> None:
        raise NoSegmentsGivenError()


# %% timing one step


def test_a_step_is_timed_under_its_own_name(tmp_path):
    """
    A duration says nothing without saying what it was spent on.
    """
    pipeline = WarsawPipeline(settings=PipelineSettings())
    step = StepThatFinishes(settings=pipeline.settings, run=Run(directory=tmp_path))

    timed = pipeline.carry_out_step(step)

    assert timed.step == step.name
    assert timed.seconds >= 0.0


def test_a_step_the_run_carried_on_from_is_timed_too(tmp_path):
    """
    A step that failed still spent the time it spent, and a run that lost an hour to one
    should say so rather than leave the hour unaccounted for.
    """
    pipeline = WarsawPipeline(settings=PipelineSettings())
    step = StepTheRunCanDoWithout(
        settings=pipeline.settings, run=Run(directory=tmp_path)
    )

    timed = pipeline.carry_out_step(step)

    assert timed.step == step.name


# %% what the run writes


def test_the_durations_are_written_where_the_run_s_records_are(tmp_path):
    """
    The point of recording them is to read them back against a finished run.
    """
    run = Run(directory=tmp_path)
    written = StepDurations(
        scene="a scene",
        steps=[
            StepDuration(step="measuring", seconds=3120.0),
            StepDuration(step="classifying", seconds=780.0),
        ],
    )

    run.write_record(RunFile.STEP_DURATIONS, written)

    assert run.read_record(RunFile.STEP_DURATIONS, StepDurations) == written


def test_the_durations_add_up_to_what_the_run_spent_in_its_steps():
    """
    What the record is read for is the share one step took of the whole.
    """
    durations = StepDurations(
        scene="a scene",
        steps=[
            StepDuration(step="measuring", seconds=3120.0),
            StepDuration(step="classifying", seconds=780.0),
        ],
    )

    assert durations.total_seconds == 3900.0
