"""
Gather every number a run can be judged by into one file.

The comparisons have different requirements. Counting what the two graphs are made of,
and judging each relation on its kind, need only the graphs. Saying that *this* drawer
went into the right cabinet needs the two worlds related to each other, which waits on
landmarks picked by hand.

A command that quietly ran only what it could would read as though the rest had passed,
so what was left out is written beside what was not, each with what it waits on.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import List, Optional

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.composition import CompositionComparison
from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.evaluation.ground_truth import GroundTruthGraph
from experiments.warsaw.evaluation.matching import HowToCompare, ObjectCorrespondences
from experiments.warsaw.evaluation.parenthood import (
    JudgedRelation,
    MissingParent,
    MissingParentReason,
    ParenthoodComparison,
)
from experiments.warsaw.evaluation.placement import (
    placed_modelled_objects,
    placed_run_bodies,
)
from experiments.warsaw.evaluation.relation_correctness import RelationCorrectness
from experiments.warsaw.evaluation.scope import ComparisonRole, ComparisonScope
from experiments.warsaw.evaluation.siblings import Depth, SiblingAgreement
from experiments.warsaw.evaluation.structure import StructuralComparison
from experiments.warsaw.pipeline.records import SplitRecord
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.templates import PipelineTemplates

# %% a comparison that could not be made


@dataclass(frozen=True)
class LeftOut(JsonRecord):
    """
    One comparison that was not made, and what it waits on.
    """

    comparison: str
    """
    What was not compared.
    """

    because: str
    """
    What it waits on.
    """


NEEDS_AN_ALIGNMENT = (
    "the two worlds are in unrelated frames; see evaluation/landmarks/README.md"
)
"""
What every comparison naming one object against another is waiting on.
"""

HOW_TO_PAIR_PLACED_OBJECTS = HowToCompare(distance_apart=1.0, size_difference=0.3)
"""
What counts as the same object once an alignment puts both worlds in one frame.

Where a thing sits is the only evidence that separates one drawer of a run of drawers
from the next, so it is worth as much as everything else put together. Size is worth
less here than it is between two graphs that share no frame: the scan sees the front of
a cabinet and the modelled world is a solid box, so their extents do not compare for
anything with a carcass, and size is left as a tie-breaker between candidates in the
same place rather than as grounds to refuse one.
"""

# %% everything a run was judged by


@dataclass(frozen=True)
class Evaluation(JsonRecord):
    """
    Every number one run was judged by, and every comparison that was not made.
    """

    run: str
    """
    The run the numbers are of.
    """

    scene: str
    """
    The modelled world they are against.
    """

    structure: StructuralComparison
    """
    What the two graphs are made of, counted without pairing anything up.
    """

    parenthood: ParenthoodComparison
    """
    Which whole each part ended up in, and what became of the ones that ended up in
    none.
    """

    composition: CompositionComparison
    """
    How much structure each graph has per class, and how much of it is held by
    something.
    """

    relations_correct: Optional[RelationCorrectness] = None
    """
    Each relation the run asserted, judged against the modelled world one at a time.

    The objects a run can find are settled by the segmentation; the relations are what
    the pipeline adds, so this is what it is judged on.
    """

    placement: Optional[SiblingAgreement] = None
    """
    Whether each part ended up in the right piece of furniture, judged by the outermost
    whole and read as a precision: of the parts the run put together, how many belong
    together.

    Its recall says nothing, because the outermost whole of a modelled apartment is a
    whole side of a kitchen and keeping all of that together is not something a run is
    asked to do.
    """

    splitting: Optional[SiblingAgreement] = None
    """
    Whether a whole arrived in one piece, judged by the whole that directly holds each
    part and read as a recall: of the parts the modelled world holds together, how many
    the run kept together.
    """

    matching: Optional[ObjectCorrespondences] = None
    """
    Which reconstructed object is which modelled one, where that could be decided.
    """

    left_out: List[LeftOut] = field(default_factory=list)
    """
    The comparisons that were not made.
    """

    @classmethod
    def of(
        cls,
        run: Run,
        ground_truth: GroundTruthGraph,
        scope: Optional[ComparisonScope] = None,
        alignment: Optional[List[List[float]]] = None,
        predicted: Optional[EvaluationGraph] = None,
    ) -> Evaluation:
        """
        Judge one finished run against a modelled world.

        :param run: The run to judge.
        :param ground_truth: The modelled world to judge it against.
        :param scope: What the comparison covers, or nothing to compare everything.
        :param alignment: The transform landmarks fitted from the scene file to the
            modelled world, without which nothing can be said about placement.
        :param predicted: What to judge instead of what the run wrote, which is how a
            run with one part of it left out is scored the same way as the run itself.
        :return: The numbers, and what was left out.
        """
        predicted = predicted or run.read_record(
            RunFile.EVALUATION_GRAPH, EvaluationGraph
        )
        split = run.read_record(RunFile.SPLIT, SplitRecord)

        predicted_nodes = cls._in_scope(scope, predicted.nodes)
        modelled_nodes = cls._in_scope(scope, ground_truth.nodes)
        predicted_edges = cls._between(
            scope, [edge for edge in predicted.edges if edge.accepted], predicted.nodes
        )
        modelled_edges = cls._between(scope, ground_truth.edges, ground_truth.nodes)

        matching, placement, splitting, relations_correct = cls._where_the_parts_went(
            run, ground_truth, scope, alignment, predicted_edges, modelled_edges
        )
        return cls(
            run=run.directory.name,
            scene=ground_truth.scene,
            structure=StructuralComparison.between(
                predicted_objects=predicted_nodes,
                predicted_relations=predicted_edges,
                modelled_objects=modelled_nodes,
                modelled_relations=modelled_edges,
            ),
            parenthood=ParenthoodComparison.between(
                predicted_objects=predicted_nodes,
                predicted_relations=predicted_edges,
                modelled_objects=modelled_nodes,
                modelled_relations=modelled_edges,
                split=split,
            ),
            composition=CompositionComparison.between(
                predicted_objects=predicted_nodes,
                predicted_relations=predicted_edges,
                modelled_objects=modelled_nodes,
                modelled_relations=modelled_edges,
            ),
            relations_correct=relations_correct,
            placement=placement,
            splitting=splitting,
            matching=matching,
            left_out=(
                []
                if alignment is not None
                else [
                    LeftOut(comparison="object matching", because=NEEDS_AN_ALIGNMENT),
                    LeftOut(comparison="placement", because=NEEDS_AN_ALIGNMENT),
                ]
            ),
        )

    @classmethod
    def _where_the_parts_went(
        cls,
        run: Run,
        ground_truth: GroundTruthGraph,
        scope: Optional[ComparisonScope],
        alignment: Optional[List[List[float]]],
        predicted_edges: List,
        modelled_edges: List,
    ) -> tuple:
        """
        Relate the two worlds object by object, where an alignment allows it.

        :param run: The run to judge.
        :param ground_truth: The modelled world to judge it against.
        :param scope: What the comparison covers, or nothing.
        :param alignment: The fitted transform, or nothing where none was picked.
        :param predicted_edges: The relations the run asserted, in scope.
        :param modelled_edges: The relations the modelled world holds, in scope.
        :return: The correspondence, both agreements and every relation judged, or
            nothing for each.
        """
        if alignment is None:
            return None, None, None, None
        matching = ObjectCorrespondences.between(
            predicted=cls._in_scope(scope, placed_run_bodies(run, alignment)),
            modelled=cls._in_scope(scope, placed_modelled_objects(ground_truth)),
            how_compared=HOW_TO_PAIR_PLACED_OBJECTS,
        )
        judged = {
            depth: SiblingAgreement.between(
                predicted_relations=predicted_edges,
                modelled_relations=modelled_edges,
                correspondences=matching,
                depth=depth,
            )
            for depth in Depth
        }
        return (
            matching,
            judged[Depth.OUTERMOST],
            judged[Depth.IMMEDIATE],
            RelationCorrectness.between(
                predicted_relations=predicted_edges,
                modelled_relations=modelled_edges,
                correspondences=matching,
            ),
        )

    @staticmethod
    def _in_scope(scope: Optional[ComparisonScope], nodes: List) -> List:
        """
        :param scope: What the comparison covers, or nothing.
        :param nodes: The objects of one graph.
        :return: Those compared as objects.
        """
        if scope is None:
            return list(nodes)
        return scope.nodes_playing(ComparisonRole.INSTANCE, nodes)

    @staticmethod
    def _between(scope: Optional[ComparisonScope], edges: List, nodes: List) -> List:
        """
        :param scope: What the comparison covers, or nothing.
        :param edges: The relations of one graph.
        :param nodes: The objects of that same graph.
        :return: Those both of whose ends are compared as objects.
        """
        if scope is None:
            return list(edges)
        return scope.edges_between_instances(edges, nodes)

    # %% what it reads as

    @property
    def relations_of_an_unmodelled_kind(self) -> List[JudgedRelation]:
        """
        :return: The relations the run asserted that the modelled world holds no kind of.
        """
        return [one for one in self.parenthood.relations if not one.modelled]

    @property
    def parts_lost_in_the_split(self) -> List[MissingParent]:
        """
        :return: The parts whose whole was dropped for having nothing left.
        """
        return [
            one
            for one in self.parenthood.missing
            if one.reason is MissingParentReason.LOST_IN_THE_SPLIT
        ]

    @property
    def classes_worth_reporting(self) -> List:
        """
        :return: The classes either graph holds as a part of something, which are the ones
            whose share held says anything.
        """
        return [
            one
            for one in self.composition.by_class.values()
            if one.modelled_held or one.predicted_held
        ]

    def markdown(self) -> str:
        """
        :return: The numbers as one page, carrying what was left out.
        """
        return PipelineTemplates(
            directory=Path(__file__).resolve().parent / "templates"
        ).render(
            "evaluation.md.jinja",
            run=self.run,
            scene=self.scene,
            structure=self.structure,
            parenthood=self.parenthood,
            composition=self.composition,
            relations_correct=self.relations_correct,
            placement=self.placement,
            splitting=self.splitting,
            matching=self.matching,
            held=self.classes_worth_reporting,
            unmodelled=self.relations_of_an_unmodelled_kind,
            lost=self.parts_lost_in_the_split,
            left_out=self.left_out,
        )


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for judging a run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A finished run's directory")
    parser.add_argument(
        "--ground-truth", type=Path, required=True, help="The modelled world's graph"
    )
    parser.add_argument("--scope", type=Path, help="What the comparison covers")
    parser.add_argument(
        "--alignment",
        type=Path,
        help="The fitted landmark transform, without which placement cannot be judged",
    )
    parser.add_argument(
        "--output", type=Path, help="Where to write, by default into the run"
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Judge a finished run and write every number it comes to.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the numbers are written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)

    judged = Evaluation.of(
        run=Run(directory=parsed.run),
        ground_truth=GroundTruthGraph.from_json(
            json.loads(parsed.ground_truth.read_text())
        ),
        scope=ComparisonScope.read(parsed.scope) if parsed.scope else None,
        alignment=(
            json.loads(parsed.alignment.read_text())["matrix"]
            if parsed.alignment
            else None
        ),
    )

    output = parsed.output or parsed.run
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(judged.to_json(), indent=2))
    (output / "evaluation.md").write_text(judged.markdown())
    logging.getLogger(__name__).info(judged.markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
