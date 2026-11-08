"""
Model-free baselines of the whole pipeline, judged as a run is judged.

A baseline reads every label as a class and chooses every part's whole without asking a
model, builds the graph a run writes, and is scored by :mod:`score` on the same metrics.
Chained, the baselines show what each stage contributes: geometry alone, the ontology's
rule, and, against a run, the model.

Run ``python -m experiments.warsaw.articulate3d.baselines --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    build_taxonomy,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.articulate3d.ground_truth import SceneGroundTruth
from experiments.warsaw.articulate3d.label_readings import (
    ClassVocabulary,
    LabelReading,
    NearestByEmbedding,
    NearestBySpelling,
    TrueClasses,
)
from experiments.warsaw.articulate3d.measured_scene import MeasuredScene, SceneToJudge
from experiments.warsaw.articulate3d.score import (
    ClassLineage,
    SceneScore,
    ScoreOverRuns,
)
from experiments.warsaw.articulate3d.whole_choices import (
    FirstCandidateOfTheRule,
    LargerObjectTouchedMost,
    WholeChoice,
)
from experiments.warsaw.evaluation.graph import EvaluationGraph, EvaluationNode
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene

# %% one baseline


@dataclass
class Baseline:
    """
    A reading of the labels and a choice of wholes, standing in for a run.
    """

    reading: LabelReading
    """
    How each label is read as a class.
    """

    choice: WholeChoice
    """
    How each part's whole is chosen.
    """

    def graph(
        self, scene: MeasuredScene, ground_truth: SceneGroundTruth
    ) -> EvaluationGraph:
        """
        :param scene: The measured scene.
        :param ground_truth: Its ground truth, which only the true classes look at.
        :return: The graph a run would write: one node per segment, with its class and
            the whole it hangs from.
        """
        classes = self.reading.classes_of(scene, ground_truth)
        wholes = self.choice.wholes_of(scene, classes)
        return EvaluationGraph(
            nodes=[
                EvaluationNode(
                    name=one.name,
                    input_label=one.class_name,
                    predicted_class=classes.get(one.class_name),
                    faces=one.faces,
                    body_id=None,
                    annotation_applied=classes.get(one.class_name) is not None,
                    parent=wholes.get(one.name),
                )
                for one in scene.segments
            ]
        )

    def score(
        self,
        scene: MeasuredScene,
        ground_truth: SceneGroundTruth,
        lineage: ClassLineage,
    ) -> SceneScore:
        """
        :param scene: The measured scene.
        :param ground_truth: Its ground truth.
        :param lineage: What each class of the ontology derives from.
        :return: The baseline on the scene, judged as a run of it would be.
        """
        return SceneScore.between(
            run_name=scene.converted.scene_id,
            ground_truth=ground_truth,
            bodies=scene.bodies(ground_truth),
            graph=self.graph(scene, ground_truth),
            lineage=lineage,
        )


# %% the baselines compared


class BaselineName(StrEnum):
    """
    The baselines the command judges.
    """

    EMBEDDING_AND_GEOMETRY = "embedding-and-geometry"
    """
    Classes by Sentence-BERT, wholes by geometry alone.
    """

    SPELLING_AND_RULE = "spelling-and-rule"
    """
    Classes by the most alike spelled name, wholes by the pipeline's rule.
    """

    EMBEDDING_AND_RULE = "embedding-and-rule"
    """
    Classes by Sentence-BERT, wholes by the pipeline's rule: the pipeline without a model.
    """

    TRUE_CLASSES_AND_RULE = "true-classes-and-rule"
    """
    The protocol's classes, wholes by the pipeline's rule: the most the rule gives
    without a model.
    """


@dataclass
class Baselines:
    """
    The readings and choices the named baselines are put together from.
    """

    rule: FirstCandidateOfTheRule
    """
    The pipeline's rule.
    """

    geometry: LargerObjectTouchedMost
    """
    The choice by geometry alone.
    """

    vocabulary: ClassVocabulary
    """
    The classes a label may be read as.
    """

    @cached_property
    def by_embedding(self) -> NearestByEmbedding:
        """
        :return: The reading by Sentence-BERT, whose model is loaded once and only when
            a baseline uses it.
        """
        return NearestByEmbedding(vocabulary=self.vocabulary)

    def named(self, name: BaselineName) -> Baseline:
        """
        :param name: A baseline.
        :return: The baseline it names.
        """
        match name:
            case BaselineName.EMBEDDING_AND_GEOMETRY:
                return Baseline(reading=self.by_embedding, choice=self.geometry)
            case BaselineName.SPELLING_AND_RULE:
                return Baseline(
                    reading=NearestBySpelling(vocabulary=self.vocabulary),
                    choice=self.rule,
                )
            case BaselineName.EMBEDDING_AND_RULE:
                return Baseline(reading=self.by_embedding, choice=self.rule)
            case BaselineName.TRUE_CLASSES_AND_RULE:
                return Baseline(reading=TrueClasses(), choice=self.rule)


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for judging the baselines.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measured", type=Path, required=True, help="The directory of measured scenes"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="The directory holding every scene's ground truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write each baseline's summary, named after the baseline",
    )
    parser.add_argument(
        "--baseline",
        type=BaselineName,
        choices=list(BaselineName),
        action="append",
        help="A baseline to judge; every one by default",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="The scenes to judge; every measured scene by default",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Judge each baseline on every measured scene, with the pipeline's settings, and write
    what each came to over all of them.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once every summary is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    parsed.output.mkdir(parents=True, exist_ok=True)
    load_annotation_modules()
    taxonomy = build_taxonomy(SemanticAnnotation)
    lineage = ClassLineage.from_taxonomy(taxonomy, {})
    settings = PipelineSettings()
    baselines = Baselines(
        rule=FirstCandidateOfTheRule(
            step=MeasureScene(settings=settings, run=Run(directory=parsed.output)),
            ontology=annotation_classes(SemanticAnnotation),
        ),
        geometry=LargerObjectTouchedMost(settings=settings),
        vocabulary=ClassVocabulary.from_taxonomy(taxonomy),
    )
    scenes = SceneToJudge.all_in(parsed.measured, parsed.ground_truth, parsed.scenes)
    for name in parsed.baseline or list(BaselineName):
        baseline = baselines.named(name)
        summary = ScoreOverRuns.over(
            [
                baseline.score(scene.measured, scene.ground_truth, lineage)
                for scene in scenes
            ]
        )
        (parsed.output / f"{name}.json").write_text(
            json.dumps(summary.to_json(), indent=2)
        )
        logger.info("== %s", name)
        for line in summary.lines():
            logger.info("   %s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
