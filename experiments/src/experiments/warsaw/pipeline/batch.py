"""
Run the pipeline over many scenes, several times each, one run after another.

    python -m experiments.warsaw.pipeline.batch run --name <batch> \
        --runs-per-scene 2 [--headless] <scene directory> [<scene directory> ...]

Every run is told the settings as they stand, with only its scene and whether to draw
on screen set, so a batch runs exactly the configuration the code holds. Each run is a
program of its own: it starts with the ontology and the ORM as committed, leaves nothing
behind in the next run's interpreter, and a run that fails is reported while the batch
goes on. The runs finished are kept in the batch's ledger, so the same batch started
again carries on where it stopped. Run no test session in the same checkout meanwhile:
every run rewrites the generated classes.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from experiments.warsaw.bases import HasLogger, JsonRecord
from experiments.warsaw.pipeline.pipeline import WarsawPipeline
from experiments.warsaw.pipeline.settings import PipelineSettings

# %% what a batch keeps


class BatchFile(StrEnum):
    """
    What a batch writes, beside the runs.
    """

    BATCHES = "batches"
    """
    The directory in the runs directory holding one directory per batch.
    """

    LEDGER = "ledger.json"
    """
    The runs of the batch that finished.
    """


@dataclass(frozen=True)
class FinishedRun(JsonRecord):
    """
    One run of a batch that finished.
    """

    scene_directory: str
    """
    The scene it ran on.
    """

    run_directory: str
    """
    The directory it wrote into.
    """


@dataclass(frozen=True)
class BatchLedger(JsonRecord):
    """
    Every run of a batch that finished, in the order they finished.
    """

    finished: list[FinishedRun] = field(default_factory=list)
    """
    The runs that finished.
    """

    @classmethod
    def read_or_empty(cls, path: Path) -> BatchLedger:
        """
        :param path: Where the ledger is kept.
        :return: The ledger, or an empty one for a batch never started.
        """
        if not Path(path).is_file():
            return cls()
        return cls.from_json(json.loads(Path(path).read_text()))

    def write(self, path: Path) -> None:
        """
        :param path: Where to keep the ledger.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    def with_finished(self, scene_directory: Path, run_directory: Path) -> BatchLedger:
        """
        :param scene_directory: The scene a run ran on.
        :param run_directory: The directory it wrote into.
        :return: The ledger with that run added.
        """
        return replace(
            self,
            finished=[
                *self.finished,
                FinishedRun(
                    scene_directory=str(Path(scene_directory).resolve()),
                    run_directory=str(Path(run_directory).resolve()),
                ),
            ],
        )

    def finished_in(self, scene_directory: Path) -> int:
        """
        :param scene_directory: A scene.
        :return: How many runs of it finished.
        """
        scene = str(Path(scene_directory).resolve())
        return sum(one.scene_directory == scene for one in self.finished)


def record_finished_run(
    ledger_path: Path, scene_directory: Path, run_directory: Path
) -> None:
    """
    Add a run that finished to the ledger kept on disk.

    :param ledger_path: Where the batch's ledger is kept.
    :param scene_directory: The scene the run ran on.
    :param run_directory: The directory it wrote into.
    """
    BatchLedger.read_or_empty(ledger_path).with_finished(
        scene_directory, run_directory
    ).write(ledger_path)


# %% a batch


@dataclass(frozen=True)
class BatchOutcome:
    """
    What became of the runs a batch started.
    """

    finished: int
    """
    How many finished.
    """

    failed: list[Path] = field(default_factory=list)
    """
    The scene of each run that failed, in the order they were run.
    """


@dataclass
class PipelineBatch(HasLogger):
    """
    The pipeline run over many scenes, several times each.
    """

    name: str
    """
    What the batch is called; a batch started again under the same name carries on.
    """

    scene_directories: list[Path]
    """
    The scenes to run on.
    """

    runs_per_scene: int
    """
    How many runs each scene is to have.
    """

    headless: bool = False
    """
    Whether the runs render without a window, for a machine nobody watches.
    """

    runs_directory: Path = field(
        default_factory=lambda: PipelineSettings().runs_directory
    )
    """
    Where the runs write, and where the batch keeps its own directory.
    """

    @property
    def directory(self) -> Path:
        """
        :return: Where the batch keeps its ledger and the output of each run.
        """
        return self.runs_directory / BatchFile.BATCHES / self.name

    @property
    def ledger_path(self) -> Path:
        """
        :return: Where the batch keeps its ledger.
        """
        return self.directory / BatchFile.LEDGER

    def planned(self) -> list[Path]:
        """
        :return: The scene of every run still to make, every scene's next run before
            any scene's run after that, so a batch cut short has run every scene alike.
        """
        ledger = BatchLedger.read_or_empty(self.ledger_path)
        finished = {
            scene: ledger.finished_in(scene) for scene in self.scene_directories
        }
        return [
            scene
            for round_number in range(self.runs_per_scene)
            for scene in self.scene_directories
            if finished[scene] <= round_number
        ]

    def command_for(self, scene_directory: Path) -> list[str]:
        """
        :param scene_directory: The scene to run on.
        :return: The program making one run of it and recording it in the ledger.
        """
        return [
            sys.executable,
            "-m",
            __spec__.name,
            BatchCommand.ONE,
            "--scene",
            str(scene_directory),
            "--ledger",
            str(self.ledger_path),
            *(["--headless"] if self.headless else []),
        ]

    def log_for(self, index: int, scene_directory: Path) -> Path:
        """
        :param index: Where the run comes in the batch, counted from this start.
        :param scene_directory: The scene it runs on.
        :return: Where the run's output is written.
        """
        started = time.strftime("%Y-%m-%d_%H%M%S")
        return self.directory / f"{started}_{index:03d}_{scene_directory.name}.log"

    def run_one(self, scene_directory: Path, log: Path) -> int:
        """
        :param scene_directory: The scene to run on.
        :param log: Where to write what the run prints.
        :return: How the run ended: zero once it finished and is in the ledger.
        """
        with log.open("w") as output:
            return subprocess.run(
                self.command_for(scene_directory),
                stdout=output,
                stderr=subprocess.STDOUT,
            ).returncode

    def carry_out(self) -> BatchOutcome:
        """
        Make every run still to make, one after another.

        :return: How many finished, and which failed.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        planned = self.planned()
        failed = []
        for index, scene in enumerate(planned):
            log = self.log_for(index, scene)
            self.logger.info(
                "%s/%s %s, writing to %s", index + 1, len(planned), scene.name, log
            )
            started = time.monotonic()
            ended = self.run_one(scene, log)
            minutes = (time.monotonic() - started) / 60
            if ended != 0:
                failed.append(scene)
                self.logger.warning("  failed after %.1f minutes, see %s", minutes, log)
                continue
            self.logger.info("  finished in %.1f minutes", minutes)
        outcome = BatchOutcome(finished=len(planned) - len(failed), failed=failed)
        self.logger.info(
            "%s runs finished, %s failed; start the batch again to retry them",
            outcome.finished,
            len(outcome.failed),
        )
        return outcome


# %% command-line entry point


class BatchCommand(StrEnum):
    """
    What the program is asked to do.
    """

    RUN = "run"
    """
    Carry out a batch.
    """

    ONE = "one"
    """
    Make one run of a batch and record it in the batch's ledger.
    """


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for running a batch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser(BatchCommand.RUN, help="Carry out a batch")
    batch.add_argument("--name", required=True, help="What the batch is called")
    batch.add_argument(
        "--runs-per-scene",
        type=int,
        required=True,
        help="How many runs each scene gets",
    )
    batch.add_argument(
        "--headless", action="store_true", help="Render without a window"
    )
    batch.add_argument(
        "scenes", type=Path, nargs="+", help="The scene directories to run on"
    )
    one = commands.add_parser(
        BatchCommand.ONE, help="Make one run of a batch; the batch starts these itself"
    )
    one.add_argument("--scene", type=Path, required=True, help="The scene to run on")
    one.add_argument(
        "--ledger", type=Path, required=True, help="The ledger to record the run in"
    )
    one.add_argument("--headless", action="store_true", help="Render without a window")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Carry out a batch, or make one run of it.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once every run finished.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)
    if parsed.command == BatchCommand.ONE:
        run = WarsawPipeline(
            settings=PipelineSettings(
                scene_directory=parsed.scene, headless=parsed.headless
            )
        ).carry_out()
        record_finished_run(parsed.ledger, parsed.scene, run.directory)
        return 0
    outcome = PipelineBatch(
        name=parsed.name,
        scene_directories=parsed.scenes,
        runs_per_scene=parsed.runs_per_scene,
        headless=parsed.headless,
    ).carry_out()
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
