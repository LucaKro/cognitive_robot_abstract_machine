"""
Running the pipeline over many scenes, several times each, one run after another.

Each run is a program of its own, so a failed run is reported and the batch goes on.
What finished is kept in the batch's ledger, so a batch started again carries on where
it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from experiments.warsaw.pipeline.batch import (
    BatchCommand,
    BatchLedger,
    BatchOption,
    PipelineBatch,
    argument_parser,
    one_run_settings,
    record_finished_run,
)

# %% the ledger


def test_a_ledger_reads_back_as_it_was_written(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = BatchLedger().with_finished(tmp_path / "scene", tmp_path / "run")

    ledger.write(path)

    assert BatchLedger.read_or_empty(path) == ledger


def test_a_batch_never_started_has_an_empty_ledger(tmp_path):
    assert BatchLedger.read_or_empty(tmp_path / "ledger.json") == BatchLedger()


def test_the_ledger_counts_the_runs_finished_of_a_scene(tmp_path):
    ledger = (
        BatchLedger()
        .with_finished(tmp_path / "kitchen", tmp_path / "first")
        .with_finished(tmp_path / "kitchen", tmp_path / "second")
        .with_finished(tmp_path / "office", tmp_path / "third")
    )

    assert ledger.finished_in(tmp_path / "kitchen") == 2


def test_a_finished_run_is_added_to_the_ledger_on_disk(tmp_path):
    path = tmp_path / "ledger.json"

    record_finished_run(path, tmp_path / "kitchen", tmp_path / "run")
    record_finished_run(path, tmp_path / "kitchen", tmp_path / "another_run")

    assert BatchLedger.read_or_empty(path).finished_in(tmp_path / "kitchen") == 2


# %% what a batch runs, and in which order


def batch_of(tmp_path, runs_per_scene=2, headless=False):
    """
    :return: A batch over two scenes.
    """
    return PipelineBatch(
        name="tuning",
        scene_directories=[tmp_path / "kitchen", tmp_path / "office"],
        runs_per_scene=runs_per_scene,
        headless=headless,
        runs_directory=tmp_path / "runs",
    )


def test_every_scene_runs_once_before_any_runs_again(tmp_path):
    batch = batch_of(tmp_path)

    assert batch.planned() == [
        tmp_path / "kitchen",
        tmp_path / "office",
        tmp_path / "kitchen",
        tmp_path / "office",
    ]


def test_a_batch_started_again_runs_only_what_did_not_finish(tmp_path):
    batch = batch_of(tmp_path)
    batch.directory.mkdir(parents=True)
    BatchLedger().with_finished(tmp_path / "kitchen", tmp_path / "run").write(
        batch.ledger_path
    )

    assert batch.planned() == [
        tmp_path / "office",
        tmp_path / "kitchen",
        tmp_path / "office",
    ]


def test_each_run_is_its_own_program_told_its_scene_and_the_ledger(tmp_path):
    batch = batch_of(tmp_path, headless=True)

    command = batch.command_for(tmp_path / "kitchen")

    assert command[command.index("--scene") + 1] == str(tmp_path / "kitchen")
    assert command[command.index("--ledger") + 1] == str(batch.ledger_path)
    assert "--headless" in command


def test_a_run_drawing_on_screen_is_not_told_to_be_headless(tmp_path):
    assert "--headless" not in batch_of(tmp_path).command_for(tmp_path / "kitchen")


# %% a batch of the baseline without the ontology


def test_each_run_of_a_batch_without_the_knowledge_is_told_so(tmp_path):
    batch = replace(batch_of(tmp_path), uses_part_whole_knowledge=False)
    [_, _, _, *arguments] = batch.command_for(tmp_path / "kitchen")

    settings = one_run_settings(argument_parser().parse_args(arguments))

    assert settings.use_part_whole_knowledge is False


def test_a_run_of_an_ordinary_batch_uses_the_knowledge(tmp_path):
    [_, _, _, *arguments] = batch_of(tmp_path).command_for(tmp_path / "kitchen")

    settings = one_run_settings(argument_parser().parse_args(arguments))

    assert settings.use_part_whole_knowledge is True


def test_a_batch_is_started_without_the_knowledge_from_the_command_line(tmp_path):
    parsed = argument_parser().parse_args(
        [
            BatchCommand.RUN,
            BatchOption.NAME,
            "baseline",
            BatchOption.RUNS_PER_SCENE,
            "1",
            BatchOption.WITHOUT_PART_WHOLE_KNOWLEDGE,
            str(tmp_path / "kitchen"),
        ]
    )

    assert PipelineBatch.from_arguments(parsed).uses_part_whole_knowledge is False


# %% carrying a batch out


@dataclass
class BatchWhoseKitchenRunsFail(PipelineBatch):
    """
    A batch whose runs of the kitchen fail, and whose other runs finish at once.
    """

    started: list[Path] = field(default_factory=list)
    """
    The scenes a run was started for, in order.
    """

    def run_one(self, scene_directory: Path, log: Path) -> int:
        self.started.append(scene_directory)
        if scene_directory.name == "kitchen":
            return 1
        record_finished_run(
            self.ledger_path, scene_directory, self.runs_directory / "a_run"
        )
        return 0


@pytest.fixture
def failing_kitchen(tmp_path):
    """
    :return: The batch whose kitchen runs fail.
    """
    return BatchWhoseKitchenRunsFail(
        name="tuning",
        scene_directories=[tmp_path / "kitchen", tmp_path / "office"],
        runs_per_scene=2,
        runs_directory=tmp_path / "runs",
    )


def test_a_failed_run_does_not_stop_the_batch(failing_kitchen, tmp_path):
    failing_kitchen.carry_out()

    assert failing_kitchen.started == failing_kitchen.scene_directories * 2


def test_the_batch_reports_the_runs_that_failed(failing_kitchen, tmp_path):
    outcome = failing_kitchen.carry_out()

    assert outcome.failed == [tmp_path / "kitchen", tmp_path / "kitchen"]
    assert outcome.finished == 2


def test_only_the_failed_runs_are_left_to_run_again(failing_kitchen, tmp_path):
    failing_kitchen.carry_out()

    assert failing_kitchen.planned() == [tmp_path / "kitchen", tmp_path / "kitchen"]


def test_each_run_writes_its_output_to_a_log_of_its_own(failing_kitchen):
    logs = [
        failing_kitchen.log_for(index, scene)
        for index, scene in enumerate(failing_kitchen.planned())
    ]

    assert len(set(logs)) == len(logs)
    assert all(one.parent == failing_kitchen.directory for one in logs)
