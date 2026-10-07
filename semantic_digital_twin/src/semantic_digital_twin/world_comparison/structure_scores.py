"""
How well a reconstructed world relates its annotations the way the ground truth world
does.

The relations compared are the part-whole relations between annotations: a drawer that
is one of a cabinet's drawers, a handle that is a drawer's handle. Each is a triplet of
whole, field and part, as in the triplet recall of scene graph evaluation, and a ground
truth triplet is found when the reconstruction relates the counterparts of its whole and
part through the same field.

A relation can be missed because an annotation at either end has no counterpart, or
because the counterparts are not related that way. The recall between paired annotations
keeps the second apart from the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.adapters.json_serializer import list_like_classes
from krrood.class_diagrams.class_diagram import WrappedClass

from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.world_comparison.semantic_scores import SemanticEvaluation
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

# %% relations


@dataclass
class PartWholeRelation:
    """
    One annotation being a part of another, through one of the whole's part-whole
    fields.
    """

    whole: SemanticAnnotation
    """
    The annotation the part belongs to.
    """

    field_name: str
    """
    The name of the whole's field holding the part.
    """

    part: SemanticAnnotation
    """
    The annotation that is a part of the whole.
    """


@dataclass
class FoundRelation:
    """
    A ground truth relation together with the reconstructed relation that stands for it.
    """

    ground_truth_relation: PartWholeRelation
    """
    The relation of the ground truth world.
    """

    reconstructed_relation: PartWholeRelation
    """
    The relation of the reconstructed world between the counterparts of its whole and
    part, through the same field.
    """


# %% evaluation


@dataclass
class StructureEvaluation:
    """
    Which part-whole relations of the ground truth world the reconstructed world has,
    which it misses, and which it adds.
    """

    semantic_evaluation: SemanticEvaluation
    """
    The pairing of annotations the relations were compared through.
    """

    ground_truth_relations: list[PartWholeRelation] = field(default_factory=list)
    """
    The relations between the ground truth annotations that take part.
    """

    reconstructed_relations: list[PartWholeRelation] = field(default_factory=list)
    """
    The relations between the reconstructed annotations that take part.
    """

    found_relations: list[FoundRelation] = field(default_factory=list)
    """
    The ground truth relations the reconstruction has.
    """

    missed_relations: list[PartWholeRelation] = field(default_factory=list)
    """
    The ground truth relations the reconstruction does not have.
    """

    extra_relations: list[PartWholeRelation] = field(default_factory=list)
    """
    The reconstructed relations that stand for no ground truth relation.
    """

    relations_between_paired_annotations: list[PartWholeRelation] = field(
        default_factory=list
    )
    """
    The ground truth relations whose whole and part both have a reconstructed
    counterpart, so only the relation itself can be wrong.
    """

    @property
    def relation_recall(self) -> float | None:
        """
        :return: The share of the ground truth relations that were found; ``None``
            without any.
        """
        return self._share(len(self.found_relations), len(self.ground_truth_relations))

    @property
    def relation_precision(self) -> float | None:
        """
        :return: The share of the reconstructed relations that stand for a ground truth
            relation; ``None`` without any.
        """
        return self._share(len(self.found_relations), len(self.reconstructed_relations))

    @property
    def relation_recall_between_paired_annotations(self) -> float | None:
        """
        :return: The share of the relations between paired annotations that were found,
            so annotations without a counterpart do not count against it; ``None``
            without any.
        """
        return self._share(
            len(self.found_relations), len(self.relations_between_paired_annotations)
        )

    @staticmethod
    def _share(count: int, total: int) -> float | None:
        """
        :return: The count as a share of the total, ``None`` when the total is 0.
        """
        return count / total if total else None


# %% scoring


@dataclass
class StructureScorer:
    """
    Compares the part-whole relations of two worlds through their pairing of
    annotations.

    Only relations between annotations that take part in the semantic evaluation are
    compared, since a relation to an annotation that stands on no surface could never be
    found.
    """

    def score(self, semantic_evaluation: SemanticEvaluation) -> StructureEvaluation:
        """
        :param semantic_evaluation: The pairing of the two worlds' annotations.
        :return: The relations found, missed and added.
        """
        counterpart_of = {
            id(
                score.match.ground_truth_annotation
            ): score.match.reconstructed_annotation
            for score in semantic_evaluation.scores
        }
        evaluation = StructureEvaluation(
            semantic_evaluation=semantic_evaluation,
            ground_truth_relations=self._relations_among(
                [
                    score.match.ground_truth_annotation
                    for score in semantic_evaluation.scores
                ]
                + semantic_evaluation.unmatched_ground_truth_annotations
            ),
            reconstructed_relations=self._relations_among(
                [
                    score.match.reconstructed_annotation
                    for score in semantic_evaluation.scores
                ]
                + semantic_evaluation.extra_annotations
            ),
        )
        unfound = {
            self._key_of(relation.whole, relation.field_name, relation.part): relation
            for relation in evaluation.reconstructed_relations
        }
        for relation in evaluation.ground_truth_relations:
            whole = counterpart_of.get(id(relation.whole))
            part = counterpart_of.get(id(relation.part))
            if whole is None or part is None:
                evaluation.missed_relations.append(relation)
                continue
            evaluation.relations_between_paired_annotations.append(relation)
            reconstructed_relation = unfound.pop(
                self._key_of(whole, relation.field_name, part), None
            )
            if reconstructed_relation is None:
                evaluation.missed_relations.append(relation)
            else:
                evaluation.found_relations.append(
                    FoundRelation(
                        ground_truth_relation=relation,
                        reconstructed_relation=reconstructed_relation,
                    )
                )
        evaluation.extra_relations = list(unfound.values())
        return evaluation

    @staticmethod
    def _relations_among(
        annotations: list[SemanticAnnotation],
    ) -> list[PartWholeRelation]:
        """
        :return: The part-whole relations whose whole and part are both among the
            annotations.
        """
        taking_part = {id(annotation) for annotation in annotations}
        relations = []
        for whole in annotations:
            for wrapped_field in WrappedClass(type(whole)).fields_with_metadata(
                IsPartWholeRelationship
            ):
                value = getattr(whole, wrapped_field.field.name)
                parts = value if isinstance(value, list_like_classes) else [value]
                relations.extend(
                    PartWholeRelation(
                        whole=whole, field_name=wrapped_field.field.name, part=part
                    )
                    for part in parts
                    if id(part) in taking_part
                )
        return relations

    @staticmethod
    def _key_of(
        whole: SemanticAnnotation, field_name: str, part: SemanticAnnotation
    ) -> tuple[int, str, int]:
        """
        :return: What identifies a relation within one world.
        """
        return id(whole), field_name, id(part)
