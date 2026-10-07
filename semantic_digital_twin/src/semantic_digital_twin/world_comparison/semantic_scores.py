"""
How well the semantic annotations of a reconstructed world say what the ground truth
world's annotations say.

Annotations are paired by where they stand, never by their class, so a wrong class is
scored as wrong rather than left without a partner: an annotation stands on its root
body, and two annotations are paired when the bodies they stand on are matched to each
other. Paired classes are then compared exactly and over a taxonomy, so naming a fridge
a cabinet is partly right and naming it an apple is not.

The taxonomy scores are the hierarchical precision and recall of Kiritchenko et al.
(2006): each class stands for the set of its ancestors, and precision and recall are
taken over those sets. The ancestors are the semantic classes alone, since the mixins
nearly every annotation inherits would make any two classes look alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType

from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_comparison.matching import (
    BodyCorrespondence,
    pairs_sharing_the_most,
)
from semantic_digital_twin.world_comparison.surface_samples import WorldSurfaces
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% taxonomy


@dataclass
class SemanticTaxonomy:
    """
    The semantic classes over which annotation classes are compared.

    A class's ancestors are itself and those of its base classes that belong to the
    taxonomy, so mixins outside it (such as having a root body or doors) are not counted
    as something two classes have in common.
    """

    classes: list[type[SemanticAnnotation]]
    """
    The semantic classes of the taxonomy.
    """

    @classmethod
    def of_module(cls, module: ModuleType) -> SemanticTaxonomy:
        """
        :param module: A module defining semantic annotation classes.
        :return: The taxonomy of the semantic annotation classes the module defines, not
            those it imports.
        """
        return cls(
            classes=[
                member
                for member in vars(module).values()
                if isinstance(member, type)
                and issubclass(member, SemanticAnnotation)
                and member is not SemanticAnnotation
                and member.__module__ == module.__name__
            ]
        )

    @cached_property
    def _class_set(self) -> frozenset[type[SemanticAnnotation]]:
        """
        :return: The classes of the taxonomy, for looking them up.
        """
        return frozenset(self.classes)

    def ancestors_of(
        self, annotation_class: type[SemanticAnnotation]
    ) -> set[type[SemanticAnnotation]]:
        """
        :param annotation_class: The class of an annotation.
        :return: The class itself, whether or not it belongs to the taxonomy, and every
            class of the taxonomy it derives from.
        """
        return {annotation_class} | {
            base for base in annotation_class.__mro__ if base in self._class_set
        }


# %% scores


@dataclass
class PrecisionAndRecall:
    """
    How much of what was claimed is right (precision), and how much of what is right was
    claimed (recall).
    """

    precision: float | None
    """
    The share of the claims that are right, from 0 to 1; ``None`` without any claim.
    """

    recall: float | None
    """
    The share of what is right that was claimed, from 0 to 1; ``None`` when nothing is
    right to claim.
    """

    @property
    def f_score(self) -> float | None:
        """
        :return: The harmonic mean of precision and recall, high only when both are; 0
            when both are 0, ``None`` when either is undefined.
        """
        if self.precision is None or self.recall is None:
            return None
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass
class AnnotationMatch:
    """
    A ground truth annotation and the reconstructed annotation standing where it stands.
    """

    ground_truth_annotation: SemanticAnnotation
    """
    The annotation of the ground truth world.
    """

    reconstructed_annotation: SemanticAnnotation
    """
    The annotation of the reconstructed world.
    """

    shared_samples: int
    """
    How many samples of the reconstructed annotation's bodies lie on the ground truth
    annotation's bodies they are matched to.
    """


@dataclass
class SemanticScore:
    """
    How well the class of a reconstructed annotation agrees with the class of the ground
    truth annotation it is paired with.
    """

    match: AnnotationMatch
    """
    The pair of annotations compared.
    """

    same_class: bool
    """
    Whether both annotations are of exactly the same class.
    """

    shared_ancestors: int
    """
    How many ancestors the two classes have in common.
    """

    ground_truth_ancestors: int
    """
    How many ancestors the ground truth annotation's class has.
    """

    reconstructed_ancestors: int
    """
    How many ancestors the reconstructed annotation's class has.
    """

    @property
    def hierarchical(self) -> PrecisionAndRecall:
        """
        :return: The share of the reconstructed class's ancestors that the ground truth
            class has too (precision), and the reverse (recall).
        """
        return PrecisionAndRecall(
            precision=self.shared_ancestors / self.reconstructed_ancestors,
            recall=self.shared_ancestors / self.ground_truth_ancestors,
        )


@dataclass
class SemanticEvaluation:
    """
    How well the annotations of a reconstructed world agree with those of the ground
    truth world, pair by pair and over both worlds.
    """

    correspondence: BodyCorrespondence
    """
    The correspondence of bodies the annotations were paired through.
    """

    taxonomy: SemanticTaxonomy
    """
    The taxonomy the classes were compared over.
    """

    scores: list[SemanticScore] = field(default_factory=list)
    """
    One score per pair of annotations.
    """

    unmatched_ground_truth_annotations: list[SemanticAnnotation] = field(
        default_factory=list
    )
    """
    The ground truth annotations no reconstructed annotation stands for.
    """

    extra_annotations: list[SemanticAnnotation] = field(default_factory=list)
    """
    The reconstructed annotations that stand for no ground truth annotation.
    """

    @property
    def exact(self) -> PrecisionAndRecall:
        """
        :return: The share of the reconstructed annotations that are paired with one of
            exactly their class (precision), and the share of the ground truth
            annotations that are (recall).
        """
        same_class = sum(score.same_class for score in self.scores)
        return self._agreement(
            same_class,
            len(self.scores) + len(self.extra_annotations),
            len(self.scores) + len(self.unmatched_ground_truth_annotations),
        )

    @property
    def hierarchical(self) -> PrecisionAndRecall:
        """
        :return: The hierarchical precision and recall over all annotations of both
            worlds: the shared ancestors of the pairs, divided by the ancestors of every
            reconstructed annotation (precision) or every ground truth annotation
            (recall), so unpaired annotations count against them.
        """
        shared = sum(score.shared_ancestors for score in self.scores)
        reconstructed = sum(
            score.reconstructed_ancestors for score in self.scores
        ) + sum(
            len(self.taxonomy.ancestors_of(type(annotation)))
            for annotation in self.extra_annotations
        )
        ground_truth = sum(score.ground_truth_ancestors for score in self.scores) + sum(
            len(self.taxonomy.ancestors_of(type(annotation)))
            for annotation in self.unmatched_ground_truth_annotations
        )
        return self._agreement(shared, reconstructed, ground_truth)

    @staticmethod
    def _agreement(
        agreeing: int, reconstructed: int, ground_truth: int
    ) -> PrecisionAndRecall:
        """
        :return: The agreeing amount as a share of the reconstructed and of the ground
            truth amount, ``None`` where that amount is 0.
        """
        return PrecisionAndRecall(
            precision=agreeing / reconstructed if reconstructed else None,
            recall=agreeing / ground_truth if ground_truth else None,
        )


# %% scoring


@dataclass
class SemanticScorer:
    """
    Pairs the annotations of two worlds through their body correspondence and compares
    their classes.

    An annotation takes part when it stands on a body with a visual surface: its root
    body, or, for an annotation whose root is not such a body, those of its bodies that
    are. Annotations on regions alone, such as the passage of a door, stand on no
    surface and take no part.
    """

    taxonomy: SemanticTaxonomy
    """
    The taxonomy the classes are compared over.
    """

    def score(self, correspondence: BodyCorrespondence) -> SemanticEvaluation:
        """
        :param correspondence: The correspondence of bodies between the two worlds.
        :return: The scores of the annotation pairs, and the annotations left without a
            partner.
        """
        ground_truth_surfaces = correspondence.overlap_table.ground_truth_surfaces
        reconstructed_surfaces = correspondence.overlap_table.reconstructed_surfaces
        ground_truth_annotations = self._annotations_taking_part(
            correspondence.ground_truth_world, ground_truth_surfaces
        )
        reconstructed_annotations = self._annotations_taking_part(
            correspondence.reconstructed_world, reconstructed_surfaces
        )
        shared = correspondence.matched_samples_between(
            [
                self._bodies_standing_under(annotation, ground_truth_surfaces)
                for annotation in ground_truth_annotations
            ],
            [
                self._bodies_standing_under(annotation, reconstructed_surfaces)
                for annotation in reconstructed_annotations
            ],
        )
        pairs = pairs_sharing_the_most(shared)
        paired_ground_truth = {column for _, column in pairs}
        paired_reconstructed = {row for row, _ in pairs}
        return SemanticEvaluation(
            correspondence=correspondence,
            taxonomy=self.taxonomy,
            scores=[
                self._score_of(
                    AnnotationMatch(
                        ground_truth_annotation=ground_truth_annotations[column],
                        reconstructed_annotation=reconstructed_annotations[row],
                        shared_samples=int(round(shared[row, column])),
                    )
                )
                for row, column in pairs
            ],
            unmatched_ground_truth_annotations=[
                annotation
                for index, annotation in enumerate(ground_truth_annotations)
                if index not in paired_ground_truth
            ],
            extra_annotations=[
                annotation
                for index, annotation in enumerate(reconstructed_annotations)
                if index not in paired_reconstructed
            ],
        )

    def _annotations_taking_part(
        self, world: World, surfaces: WorldSurfaces
    ) -> list[SemanticAnnotation]:
        """
        :return: The annotations of the world that stand on a body with a surface.
        """
        return [
            annotation
            for annotation in world.semantic_annotations
            if self._bodies_standing_under(annotation, surfaces)
        ]

    @staticmethod
    def _bodies_standing_under(
        annotation: SemanticAnnotation, surfaces: WorldSurfaces
    ) -> list[Body]:
        """
        :return: The root body of the annotation if it has a surface, otherwise those of
            the annotation's bodies that have one.
        """
        if (
            isinstance(annotation, HasRootKinematicStructureEntity)
            and isinstance(annotation.root, Body)
            and surfaces.has_surface(annotation.root)
        ):
            return [annotation.root]
        return [body for body in annotation.bodies if surfaces.has_surface(body)]

    def _score_of(self, match: AnnotationMatch) -> SemanticScore:
        """
        :return: How well the classes of the paired annotations agree.
        """
        ground_truth_class = type(match.ground_truth_annotation)
        reconstructed_class = type(match.reconstructed_annotation)
        ground_truth_ancestors = self.taxonomy.ancestors_of(ground_truth_class)
        reconstructed_ancestors = self.taxonomy.ancestors_of(reconstructed_class)
        return SemanticScore(
            match=match,
            same_class=ground_truth_class is reconstructed_class,
            shared_ancestors=len(ground_truth_ancestors & reconstructed_ancestors),
            ground_truth_ancestors=len(ground_truth_ancestors),
            reconstructed_ancestors=len(reconstructed_ancestors),
        )
