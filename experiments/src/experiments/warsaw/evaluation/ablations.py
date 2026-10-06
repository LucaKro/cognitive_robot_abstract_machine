"""
What each part of a run was worth, by leaving it out and scoring again.

A finished run keeps every intermediate record and every model call, so anything that
only decides what happens *after* the model answered can be replayed for nothing. What
is worth replaying is whatever the pipeline does that a far cheaper rule could have done
instead: the memberships geometry forced on its own, the split's rule that an emptied
object is dropped, and the model's ruling on faces two labels both claim.

Each of these changes one thing and leaves the rest of the run untouched, so what comes
out is comparable with the run as it stands rather than a differently-built world.
"""

from __future__ import annotations

import argparse
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path

from typing_extensions import Dict, List, Optional, Sequence

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.evaluate import Evaluation
from experiments.warsaw.evaluation.graph import (
    EvaluationEdge,
    EvaluationGraph,
)
from experiments.warsaw.evaluation.ground_truth import GroundTruthGraph
from experiments.warsaw.evaluation.scope import ComparisonScope
from experiments.warsaw.pipeline.records import (
    Adjudication,
    Adjudications,
    EmptiedSegment,
    ForcedMembership,
    OwnershipAnswer,
    Relations,
    SplitRecord,
)
from experiments.warsaw.pipeline.run import Run, RunFile

# %% what a relation means where only geometry said so

HELD_AS_A_PART = "part"
"""
What a membership means where nothing recorded what it means.

Geometry says that two objects meet and which field holds them, not what holding them
that way means, and being part of something is what the ontology makes of a field unless
it says otherwise.
"""

# %% one part of the pipeline left out


@dataclass(frozen=True)
class Ablation(ABC):
    """
    One part of a run left out, so that what it was worth can be read off.
    """

    @property
    @abstractmethod
    def left_out(self) -> str:
        """
        :return: What this leaves out, as it reads in a report.
        """

    @abstractmethod
    def relations_of(self, run: Run, graph: EvaluationGraph) -> List[EvaluationEdge]:
        """
        :param run: The finished run to replay.
        :param graph: What that run built.
        :return: The relations it would have asserted without the part left out.
        """

    def applied_to(self, run: Run, graph: EvaluationGraph) -> EvaluationGraph:
        """
        :param run: The finished run to replay.
        :param graph: What that run built.
        :return: The same run with this part of it left out.
        """
        return self.applied_to_graph(graph, self.relations_of(run, graph))

    @staticmethod
    def applied_to_graph(
        graph: EvaluationGraph, relations: Sequence[EvaluationEdge]
    ) -> EvaluationGraph:
        """
        Put a different set of relations on a run, leaving everything else as it was.

        :param graph: What the run built.
        :param relations: The relations to put on it instead.
        :return: The run with those relations.
        """
        return replace(graph, edges=list(relations))


# %% the relations the model decided


@dataclass(frozen=True)
class WithoutTheModelsRelations(Ablation):
    """
    The run as it would have been having asked the model nothing about relations.

    Kept are the memberships geometry forced on its own, where a part met exactly one
    candidate and there was nothing to choose between. The classes are left alone:
    without them there is nothing to compare against a ground truth stated in classes,
    so this is a baseline for the pipeline's relation decisions and not for its
    classification.
    """

    @property
    def left_out(self) -> str:
        """
        :return: What this leaves out.
        """
        return "the model's relation decisions"

    def relations_of(self, run: Run, graph: EvaluationGraph) -> List[EvaluationEdge]:
        """
        :param run: The finished run to replay.
        :param graph: What that run built.
        :return: Only the memberships geometry forced.
        """
        return self.forced_only(
            run.read_record(RunFile.RELATIONS, Relations).forced, graph
        )

    @staticmethod
    def forced_only(
        forced: Sequence[ForcedMembership], graph: EvaluationGraph
    ) -> List[EvaluationEdge]:
        """
        :param forced: The memberships geometry left no choice about.
        :param graph: What the run built, which says what it made of each field.
        :return: Those memberships as relations.
        """
        kinds = {one.field_name: one.relation for one in graph.edges}
        return [
            EvaluationEdge(
                whole=one.whole,
                part=one.part,
                relation=kinds.get(one.field_name, HELD_AS_A_PART),
                field_name=one.field_name,
                accepted=True,
            )
            for one in forced
        ]


# %% the wholes the split emptied


@dataclass(frozen=True)
class KeepingEmptiedWholes(Ablation):
    """
    The run as it would have been had the split kept objects it emptied.

    An object whose every face was taken by its neighbours is dropped, which is why a
    drawer that consumed the cabinet holding it ends up belonging to nothing: the cabinet
    is gone and the relation stops being expressible.
    """

    @property
    def left_out(self) -> str:
        """
        :return: What this leaves out.
        """
        return "the split's rule that an emptied whole is dropped"

    def relations_of(self, run: Run, graph: EvaluationGraph) -> List[EvaluationEdge]:
        """
        :param run: The finished run to replay.
        :param graph: What that run built.
        :return: Its relations, with the emptied wholes put back.
        """
        return self.with_the_emptied_put_back(
            run.read_record(RunFile.SPLIT, SplitRecord).emptied, graph
        )

    @staticmethod
    def with_the_emptied_put_back(
        emptied: Sequence[EmptiedSegment], graph: EvaluationGraph
    ) -> List[EvaluationEdge]:
        """
        :param emptied: The segments the split left with nothing, and who took them.
        :param graph: What the run built.
        :return: Its relations, and one more for each emptied whole, to whichever part
            took most of it.
        """
        put_back = []
        for segment in emptied:
            if not segment.taken_by:
                continue
            took_most = max(segment.taken_by, key=lambda one: one.faces)
            put_back.append(
                EvaluationEdge(
                    whole=segment.name,
                    part=took_most.name,
                    relation=HELD_AS_A_PART,
                    field_name="",
                    accepted=True,
                )
            )
        return list(graph.edges) + put_back


# %% the contested faces the model ruled on


@dataclass(frozen=True)
class AdjudicationAgreement(JsonRecord):
    """
    How often the model's ruling on contested faces was the obvious one anyway.

    The cheap default is that the larger of the claimants owns what both claim. Where
    the model said the same, the call bought nothing; where it overruled the default is
    what the step is worth.
    """

    decided: int = 0
    """
    How many contested patterns the model ruled on.
    """

    agreeing: int = 0
    """
    How many of those rulings the default would have made as well.
    """

    differing: List[str] = field(default_factory=list)
    """
    The patterns where the model overruled the default.
    """

    @property
    def share_the_default_would_have_got(self) -> float:
        """
        :return: What fraction of the rulings needed no asking, or one where nothing was
            ruled on.
        """
        if self.decided == 0:
            return 1.0
        return self.agreeing / self.decided

    @classmethod
    def between(
        cls, answers: Sequence[Adjudication], graph: EvaluationGraph
    ) -> AdjudicationAgreement:
        """
        Compare every ruling about contested faces against the larger claimant.

        A run's adjudications hold two kinds of answer, whose faces are whose and which
        whole a part belongs to, and only the first has a cheap default to be held
        against.

        :param answers: What the model settled, of either kind.
        :param graph: What the run built, which is what says how large a label is.
        :return: How often the two agreed.
        """
        faces = cls._faces_per_label(graph)
        about_faces = [one for one in answers if isinstance(one, OwnershipAnswer)]
        differing = [
            one.name
            for one in about_faces
            if one.owner != max(one.pattern, key=lambda label: faces.get(label, 0))
        ]
        return cls(
            decided=len(about_faces),
            agreeing=len(about_faces) - len(differing),
            differing=differing,
        )

    @staticmethod
    def _faces_per_label(graph: EvaluationGraph) -> Dict[str, int]:
        """
        :param graph: What the run built.
        :return: How many faces each label came to across every body carrying it.
        """
        faces: Dict[str, int] = {}
        for node in graph.nodes:
            faces[node.input_label] = faces.get(node.input_label, 0) + node.faces
        return faces

    @classmethod
    def of(cls, run: Run) -> AdjudicationAgreement:
        """
        :param run: The finished run to replay.
        :return: How often its adjudication was the obvious ruling anyway.
        """
        return cls.between(
            run.read_record(RunFile.ADJUDICATIONS, Adjudications).answered,
            run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph),
        )


# %% every ablation of a run at once

ABLATIONS = (WithoutTheModelsRelations(), KeepingEmptiedWholes())
"""
The parts of a run that can be left out without asking the model anything again.
"""


@dataclass(frozen=True)
class AblatedRun(JsonRecord):
    """
    One run scored as it stands and again with each part of it left out.
    """

    run: str
    """
    The run that was ablated.
    """

    as_it_stands: Evaluation
    """
    The run scored with nothing left out.
    """

    without: Dict[str, Evaluation] = field(default_factory=dict)
    """
    The run scored again for each part left out, by what that part is.
    """

    adjudication: AdjudicationAgreement = field(default_factory=AdjudicationAgreement)
    """
    How often the model's ruling on contested faces was the obvious one anyway.
    """

    @classmethod
    def of(
        cls,
        run: Run,
        ground_truth: GroundTruthGraph,
        scope: Optional[ComparisonScope] = None,
        alignment: Optional[List[List[float]]] = None,
    ) -> AblatedRun:
        """
        Score a run, and score it again with each part of it left out.

        :param run: The finished run to ablate.
        :param ground_truth: The modelled world to judge it against.
        :param scope: What the comparison covers, or nothing to compare everything.
        :param alignment: The fitted landmark transform, without which nothing can be
            said about placement.
        :return: Every scoring, side by side.
        """
        graph = run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph)
        judged = {
            ablation.left_out: Evaluation.of(
                run=run,
                ground_truth=ground_truth,
                scope=scope,
                alignment=alignment,
                predicted=ablation.applied_to(run, graph),
            )
            for ablation in ABLATIONS
        }
        return cls(
            run=run.directory.name,
            as_it_stands=Evaluation.of(
                run=run, ground_truth=ground_truth, scope=scope, alignment=alignment
            ),
            without=judged,
            adjudication=AdjudicationAgreement.of(run),
        )


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for ablating a run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A finished run's directory")
    parser.add_argument(
        "--ground-truth", type=Path, required=True, help="The modelled world's graph"
    )
    parser.add_argument("--scope", type=Path, help="What the comparison covers")
    parser.add_argument("--alignment", type=Path, help="The fitted landmark transform")
    parser.add_argument(
        "--output", type=Path, help="Where to write, by default into the run"
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Score a run with each part of it left out, and write what each was worth.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the numbers are written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)

    ablated = AblatedRun.of(
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
    (output / "ablations.json").write_text(json.dumps(ablated.to_json(), indent=2))
    logging.getLogger(__name__).info("wrote %s", (output / "ablations.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
