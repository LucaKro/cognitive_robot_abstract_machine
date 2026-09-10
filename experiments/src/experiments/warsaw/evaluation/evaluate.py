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
from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.evaluation.ground_truth import GroundTruthGraph
from experiments.warsaw.evaluation.parenthood import (
    JudgedRelation,
    MissingParent,
    MissingParentReason,
    ParenthoodComparison,
)
from experiments.warsaw.evaluation.scope import ComparisonRole, ComparisonScope
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
    ) -> Evaluation:
        """
        Judge one finished run against a modelled world.

        :param run: The run to judge.
        :param ground_truth: The modelled world to judge it against.
        :param scope: What the comparison covers, or nothing to compare everything.
        :return: The numbers, and what was left out.
        """
        predicted = run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph)
        split = run.read_record(RunFile.SPLIT, SplitRecord)

        predicted_nodes = cls._in_scope(scope, predicted.nodes)
        modelled_nodes = cls._in_scope(scope, ground_truth.nodes)
        predicted_edges = cls._between(
            scope, [edge for edge in predicted.edges if edge.accepted], predicted.nodes
        )
        modelled_edges = cls._between(scope, ground_truth.edges, ground_truth.nodes)

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
            left_out=[
                LeftOut(comparison="object matching", because=NEEDS_AN_ALIGNMENT),
                LeftOut(comparison="sibling agreement", because=NEEDS_AN_ALIGNMENT),
            ],
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
    )

    output = parsed.output or parsed.run
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(judged.to_json(), indent=2))
    (output / "evaluation.md").write_text(judged.markdown())
    logging.getLogger(__name__).info(judged.markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
