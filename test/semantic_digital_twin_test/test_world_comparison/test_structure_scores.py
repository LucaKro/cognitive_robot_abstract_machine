import pytest

from semantic_digital_twin.semantic_annotations import semantic_annotations
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_comparison.matching import BodyMatcher
from semantic_digital_twin.world_comparison.semantic_scores import (
    SemanticScorer,
    SemanticTaxonomy,
)
from semantic_digital_twin.world_comparison.structure_scores import (
    PartWholeRelation,
    StructureEvaluation,
    StructureScorer,
)
from semantic_digital_twin.world_comparison.surface_samples import SurfaceSampler
from .worlds import AnnotatedCabinetScene

# %% fixtures


@pytest.fixture
def matcher() -> BodyMatcher:
    return BodyMatcher(
        distance_tolerance=0.02,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.01, seed=0),
    )


def evaluate(
    ground_truth_scene: AnnotatedCabinetScene,
    reconstructed_scene: AnnotatedCabinetScene,
    matcher: BodyMatcher,
) -> StructureEvaluation:
    correspondence = matcher.match(
        ground_truth_scene.create_world(),
        reconstructed_scene.create_world(),
        HomogeneousTransformationMatrix(),
    )
    semantic_evaluation = SemanticScorer(
        taxonomy=SemanticTaxonomy.of_module(semantic_annotations)
    ).score(correspondence)
    return StructureScorer().score(semantic_evaluation)


def described(relations: list[PartWholeRelation]) -> list[tuple[str, str, str]]:
    return sorted(
        (
            relation.whole.root.name.name,
            relation.field_name,
            relation.part.root.name.name,
        )
        for relation in relations
    )


# %% relations


def test_relations_are_the_part_whole_fields_of_the_annotations(matcher):
    evaluation = evaluate(AnnotatedCabinetScene(), AnnotatedCabinetScene(), matcher)
    assert described(evaluation.ground_truth_relations) == [
        ("cabinet", "drawers", "drawer"),
        ("drawer", "handle", "handle"),
    ]


def test_identical_relations_are_all_found(matcher):
    evaluation = evaluate(AnnotatedCabinetScene(), AnnotatedCabinetScene(), matcher)

    assert len(evaluation.found_relations) == 2
    assert evaluation.missed_relations == []
    assert evaluation.extra_relations == []
    assert evaluation.relation_recall == 1.0
    assert evaluation.relation_precision == 1.0


# %% relations the reconstruction gets wrong


def test_part_standing_alone_misses_its_relation(matcher):
    evaluation = evaluate(
        AnnotatedCabinetScene(),
        AnnotatedCabinetScene(handle_is_part=False),
        matcher,
    )
    assert described(evaluation.missed_relations) == [("drawer", "handle", "handle")]
    assert evaluation.relation_recall == 0.5
    assert evaluation.relation_recall_between_paired_annotations == 0.5
    assert evaluation.relation_precision == 1.0


def test_relation_through_another_field_is_missed_and_extra(matcher):
    evaluation = evaluate(
        AnnotatedCabinetScene(),
        AnnotatedCabinetScene(drawer_type=Door, drawer_field="doors"),
        matcher,
    )
    assert described(evaluation.missed_relations) == [("cabinet", "drawers", "drawer")]
    assert described(evaluation.extra_relations) == [("cabinet", "doors", "drawer")]
    assert described(
        [found.ground_truth_relation for found in evaluation.found_relations]
    ) == [("drawer", "handle", "handle")]


def test_relation_to_a_missed_part_counts_against_recall_only_overall(matcher):
    evaluation = evaluate(
        AnnotatedCabinetScene(), AnnotatedCabinetScene(has_handle=False), matcher
    )
    assert described(evaluation.missed_relations) == [("drawer", "handle", "handle")]
    assert evaluation.relation_recall == 0.5
    assert evaluation.relation_recall_between_paired_annotations == 1.0


def test_world_without_relations_has_no_relation_scores(matcher):
    evaluation = evaluate(
        AnnotatedCabinetScene(drawer_type=None),
        AnnotatedCabinetScene(drawer_type=None),
        matcher,
    )
    assert evaluation.relation_recall is None
    assert evaluation.relation_precision is None
    assert evaluation.relation_recall_between_paired_annotations is None
