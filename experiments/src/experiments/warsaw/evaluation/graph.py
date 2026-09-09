"""A database-independent snapshot of the semantic graph produced by a run."""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.pipeline.records import Classifications, SplitRecord

# %% graph nodes and edges


@dataclass(frozen=True)
class EvaluationNode(JsonRecord):
    """One reconstructed body and the semantic class assigned to it."""

    name: str
    """The stable body name used by all pipeline artefacts."""

    input_label: str
    """The rough semantic label supplied with the reconstruction."""

    predicted_class: str | None
    """The SDT class selected by the pipeline, if any."""

    faces: int
    """How many mesh faces survived the split for this body."""

    body_id: str | None
    """The database identity of the underlying body, when persisted."""

    annotation_applied: bool
    """Whether the selected class was instantiated in the final SDT."""


@dataclass(frozen=True)
class EvaluationEdge(JsonRecord):
    """One proposed hierarchy relation and whether the final SDT accepted it."""

    whole: str
    """The body at the containing end of the relation."""

    part: str
    """The body at the contained end of the relation."""

    relation: str
    """The semantic relation: part, contains, or supports."""

    field_name: str
    """The ontology field through which the mount was attempted."""

    accepted: bool
    """Whether the relation is present in the final SDT."""

    refusal_reason: str | None = None
    """Why the relation was rejected, when it was not accepted."""


# %% one run's portable graph


@dataclass(frozen=True)
class EvaluationGraph(JsonRecord):
    """All reconstructed objects and hierarchy decisions needed for evaluation."""

    nodes: list[EvaluationNode] = field(default_factory=list)
    """Every body made by the split, including bodies left unannotated."""

    edges: list[EvaluationEdge] = field(default_factory=list)
    """Every relation attempted by the final SDT construction step."""

    source_world_id: int | None = None
    """The database ID of the flat split world."""

    annotated_world_id: int | None = None
    """The database ID of the final annotated world."""

    @classmethod
    def from_run_products(
        cls,
        *,
        split: SplitRecord,
        classifications: Classifications,
        annotated_names: set[str],
    ) -> EvaluationGraph:
        """Build a graph from files and outcomes already produced by the pipeline.

        :param split: Bodies, proposed pairings, and refused mounts.
        :param classifications: The class selected for every body the model answered.
        :param annotated_names: Bodies whose annotations were instantiated successfully.
        :return: A portable graph that does not depend on the run's database schema.
        """
        answers = {
            answer.name: answer
            for answer in classifications.bodies
            if answer.name is not None
        }
        refusal_by_pairing = {
            (
                refusal.pairing.whole,
                refusal.pairing.part,
                refusal.pairing.field_name,
                refusal.pairing.kind.value,
            ): refusal.reason
            for refusal in split.refused
        }
        nodes = [
            EvaluationNode(
                name=body.name,
                input_label=body.label,
                predicted_class=(
                    answers[body.name].class_name if body.name in answers else None
                ),
                faces=body.faces,
                body_id=body.body_id,
                annotation_applied=body.name in annotated_names,
            )
            for body in split.bodies
        ]
        edges = []
        for pairing in split.pairings:
            key = (
                pairing.whole,
                pairing.part,
                pairing.field_name,
                pairing.kind.value,
            )
            refusal_reason = refusal_by_pairing.get(key)
            edges.append(
                EvaluationEdge(
                    whole=pairing.whole,
                    part=pairing.part,
                    relation=pairing.kind.value,
                    field_name=pairing.field_name,
                    accepted=refusal_reason is None,
                    refusal_reason=refusal_reason,
                )
            )
        return cls(
            nodes=nodes,
            edges=edges,
            source_world_id=split.world_id,
            annotated_world_id=split.annotated_world_id,
        )
