"""
What every step of the pipeline has in common.

A step is constructed with what the run was told and the directory it writes into, and
carried out. It reads what the steps before it wrote and writes what the steps after it
read, and nothing reaches it on a command line.
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

from semantic_digital_twin.adapters.vision_language_model.client import (
    VisionLanguageModel,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Dict, List, Optional, Type

from experiments.warsaw.exceptions import SubprocessStepFailedError
from experiments.warsaw.pipeline.asking import Questioner
from experiments.warsaw.pipeline.reporting import Reporting
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings


@dataclass
class PipelineStep(Reporting, ABC):
    """
    One step of the pipeline, as it is carried out.
    """

    settings: PipelineSettings
    """
    What the run was told.
    """

    run: Run
    """
    The directory it writes into.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: What to call the step while it runs.
        """

    @property
    def is_optional(self) -> bool:
        """
        :return: Whether the run carries on when this step fails.
        """
        return False

    @abstractmethod
    def carry_out(self) -> None:
        """
        Do the step's work, reading and writing the run's own files.
        """

    def ontology_classes(self) -> Dict[str, Type]:
        """
        The classes the ontology holds, as this run read them out.

        Not every subclass this interpreter happens to have: composing a proposed class
        registers it as a subclass of the annotation root for the rest of the process, and
        the steps of a run share one. Asked the live process, a step is told a class an
        earlier step invented for this scene is part of the ontology -- so nothing
        generates it, and the interpreter that writes the world, which invented nothing,
        cannot find it and leaves those bodies unannotated.

        The exported ontology is the one that is committed and it is written once per run,
        which is what makes it the same answer in every step and every process.

        :return: The ontology's classes by name.
        """
        declared = {
            node["name"] for node in self.run.read_json(RunFile.TAXONOMY)["classes"]
        }
        return {
            name: annotation_class
            for name, annotation_class in annotation_classes(SemanticAnnotation).items()
            if name in declared
        }

    def questioner(self, answers: RunFile) -> Questioner:
        """
        :param answers: Where this step's replies are kept.
        :return: The model, ready to be asked this step's questions.
        """
        return Questioner(
            model=VisionLanguageModel(model=self.settings.model.value),
            answers_directory=self.run.path(answers),
            corrections=self.settings.corrections,
            reuse_answers=self.settings.reuse_answers,
        )

    def in_new_interpreter(
        self,
        program: str,
        arguments: Optional[List[str]] = None,
        what: str = "work handed to a new interpreter",
        environment: Optional[dict] = None,
    ) -> str:
        """
        Do work in an interpreter that started after the ontology or the ORM was
        rewritten.

        The interpreter asking is holding the version from before that, so it cannot do
        the work itself however carefully it re-imports.

        :param program: The program to run.
        :param arguments: What to pass it, after the program itself.
        :param what: What it is doing, for the failure message.
        :param environment: The environment to run it in, by default this one's.
        :return: What it printed.
        :raises SubprocessStepFailedError: If it did not finish.
        """
        finished = subprocess.run(
            [sys.executable, "-c", program, *(arguments or [])],
            capture_output=True,
            text=True,
            env=environment,
        )
        if finished.returncode != 0:
            raise SubprocessStepFailedError(what=what, output=finished.stderr)
        return finished.stdout
