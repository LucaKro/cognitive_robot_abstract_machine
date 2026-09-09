"""
Run the whole pipeline: a labelled scan in, an annotated world out.

    python -m experiments.warsaw.pipeline.pipeline

That is the whole invocation. Everything a run can be told is in
:class:`experiments.warsaw.pipeline.settings.PipelineSettings` -- change a default there,
or construct the pipeline with the settings you want, and nothing is passed on a command
line.

Every step writes into one directory made for this run, named for when it started, and
reads nothing another run concluded. Two runs of the same scene may reach different
answers; neither should quietly inherit half of the other, so a run also writes into a
schema of the database made for it alone.

The steps run in this process. Two of them hand work to a new interpreter, and only those
two: the ontology and the ORM are rewritten during a run, and an interpreter holding the
version from before that cannot do the work however carefully it re-imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import List

from krrood.exceptions import DataclassException
from experiments.warsaw.bases import HasLogger
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.provenance import record_run_provenance
from experiments.warsaw.pipeline.database.run_schema import RunSchema
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.adjudicate.step import AdjudicateOverlaps
from experiments.warsaw.pipeline.steps.amend.step import AmendTaxonomy, RevertAmendments
from experiments.warsaw.pipeline.steps.annotate import AnnotateAndMount
from experiments.warsaw.pipeline.steps.classify.step import ClassifyBodies
from experiments.warsaw.pipeline.steps.evidence import MeasureScene
from experiments.warsaw.pipeline.steps.prepare import PrepareRun
from experiments.warsaw.pipeline.steps.split import SplitScene
from experiments.warsaw.pipeline.steps.step import PipelineStep
from experiments.warsaw.pipeline.steps.vocabulary.step import MapLabelVocabulary

# %% the run, from end to end


@dataclass
class WarsawPipeline(HasLogger):
    """
    A labelled scan turned into an annotated, hierarchical world.
    """

    settings: PipelineSettings = field(default_factory=PipelineSettings)
    """
    What the run is told.
    """

    rule: str = "─" * 78
    """
    What separates one step's account of itself from the next.
    """

    def run_steps(self, run: Run) -> List[PipelineStep]:
        """
        Work out what this run is to do, in order.

        The scene is measured twice: once knowing no classes, which is what the vocabulary
        question is built from, and once knowing them, which is what says how the labels
        may hold one another and what is left open.

        :param run: The directory the run writes into.
        :return: The steps, after the preparation that had to happen first.
        """
        planned: List[PipelineStep] = [
            MeasureScene(settings=self.settings, run=run, exemplar_renders=True),
            MapLabelVocabulary(settings=self.settings, run=run),
            MeasureScene(
                settings=self.settings,
                run=run,
                knowing_the_vocabulary=True,
                question_renders=1000,
                overwrite=True,
            ),
        ]
        if self.settings.ask_about_the_ontology:
            planned.append(AmendTaxonomy(settings=self.settings, run=run))

        planned += [
            AdjudicateOverlaps(settings=self.settings, run=run),
            SplitScene(settings=self.settings, run=run),
        ]
        if not self.settings.persist:
            return planned

        planned += [
            ClassifyBodies(settings=self.settings, run=run),
            AnnotateAndMount(settings=self.settings, run=run),
        ]
        if self.settings.amend_the_ontology:
            planned.append(RevertAmendments(settings=self.settings, run=run))
        return planned

    def carry_out(self) -> Run:
        """
        Prepare a run and carry it out.

        :return: The run, once every step it depends on has finished.
        :raises SubprocessStepFailedError: If a step the run depends on fails.
        """
        run = Run.create(self.settings.runs_directory)
        record_run_provenance(
            settings=self.settings,
            run=run,
            repository=Path(__file__).resolve().parents[5],
        )
        schema = RunSchema.for_run(run.directory)

        self.announce("preparing")
        PrepareRun(settings=self.settings, run=run).carry_out()

        # Every step runs in this process or is started by it, so pointing this one at the
        # run's schema points all of them at it, and none has to be told which.
        schema.use()

        planned = self.run_steps(run)
        self.logger.info(
            "%s over %s steps into %s, writing to schema %s",
            self.settings.model.name,
            len(planned),
            run.directory.name,
            schema.name,
        )

        for number, step in enumerate(planned, start=1):
            self.announce(f"{number}/{len(planned)}  {step.name}")
            self.carry_out_step(step)

        self.announce("done")
        for made in (RunFile.REPORT, RunFile.INSPECTOR):
            if run.holds(made):
                self.logger.info("  %s", run.path(made))
        return run

    def carry_out_step(self, step: PipelineStep) -> None:
        """
        Carry out one step, letting an optional one fail.

        :param step: The step to carry out.
        :raises DataclassException: Whatever the step raised, when the run depends on
            it.
        """
        if not step.is_optional:
            step.carry_out()
            return
        try:
            step.carry_out()
        except DataclassException as failure:
            # An optional step is one the run was told it could do without, so its failure
            # is a thing to report rather than a state the run should not have reached.
            # Only the project's own exceptions say that; anything else is a bug, and a
            # bug is not something to carry on from.
            self.logger.warning("%s failed, carrying on: %s", step.name, failure)

    def announce(self, what: str) -> None:
        """
        :param what: What is about to happen.
        """
        self.logger.info("\n%s\n%s\n%s", self.rule, what, self.rule)


# %% running it with nothing to be told


def main() -> None:
    """
    Run the pipeline with the settings as they stand.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    WarsawPipeline().carry_out()


if __name__ == "__main__":
    main()
