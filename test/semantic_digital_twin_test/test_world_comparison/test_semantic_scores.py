from dataclasses import dataclass

import pytest

from semantic_digital_twin.semantic_annotations import semantic_annotations
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Cabinet,
    Door,
    Drawer,
    EntryWay,
    Fridge,
    Furniture,
    Handle,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_comparison.matching import BodyMatcher
from semantic_digital_twin.world_comparison.semantic_scores import (
    SemanticEvaluation,
    SemanticScore,
    SemanticScorer,
    SemanticTaxonomy,
)
from semantic_digital_twin.world_comparison.surface_samples import SurfaceSampler
from .worlds import AnnotatedCabinetScene, box, world_of

# %% fixtures


@pytest.fixture
def matcher() -> BodyMatcher:
    return BodyMatcher(
        distance_tolerance=0.02,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.01, seed=0),
    )


@pytest.fixture
def taxonomy() -> SemanticTaxonomy:
    return SemanticTaxonomy.of_module(semantic_annotations)


def evaluate(
    ground_truth_world: World,
    reconstructed_world: World,
    matcher: BodyMatcher,
    taxonomy: SemanticTaxonomy,
    ground_truth_root_T_reconstructed_root: HomogeneousTransformationMatrix = None,
) -> SemanticEvaluation:
    correspondence = matcher.match(
        ground_truth_world,
        reconstructed_world,
        ground_truth_root_T_reconstructed_root or HomogeneousTransformationMatrix(),
    )
    return SemanticScorer(taxonomy=taxonomy).score(correspondence)


def score_on(evaluation: SemanticEvaluation, root_body: str) -> SemanticScore:
    [score] = [
        score
        for score in evaluation.scores
        if score.match.ground_truth_annotation.root.name.name == root_body
    ]
    return score


def root_names(annotations) -> list[str]:
    return sorted(annotation.root.name.name for annotation in annotations)


@dataclass(eq=False)
class BuiltInCabinet(Cabinet):
    """
    A cabinet class that is not part of the taxonomy.
    """


# %% taxonomy


def test_taxonomy_holds_the_semantic_classes_only(taxonomy):
    assert taxonomy.ancestors_of(Fridge) == {Fridge, Cabinet, Furniture}
    assert taxonomy.ancestors_of(Drawer) == {Drawer, Furniture}
    assert taxonomy.ancestors_of(Handle) == {Handle}


def test_class_outside_the_taxonomy_is_its_own_ancestor(taxonomy):
    assert taxonomy.ancestors_of(BuiltInCabinet) == {
        BuiltInCabinet,
        Cabinet,
        Furniture,
    }


# %% identical annotations


def test_identical_annotations_agree_exactly(matcher, taxonomy):
    scene = AnnotatedCabinetScene()
    evaluation = evaluate(scene.create_world(), scene.create_world(), matcher, taxonomy)

    assert len(evaluation.scores) == 3
    assert all(score.same_class for score in evaluation.scores)
    assert evaluation.exact.precision == 1.0
    assert evaluation.exact.recall == 1.0
    assert evaluation.hierarchical.f_score == 1.0
    assert evaluation.unmatched_ground_truth_annotations == []
    assert evaluation.extra_annotations == []


def test_alignment_carries_the_annotations_into_the_ground_truth_frame(
    matcher, taxonomy
):
    root_T_scene = HomogeneousTransformationMatrix.from_xyz_rpy(x=5.0, y=-2.0, yaw=0.3)
    evaluation = evaluate(
        AnnotatedCabinetScene().create_world(),
        AnnotatedCabinetScene(root_T_scene=root_T_scene).create_world(),
        matcher,
        taxonomy,
        root_T_scene.inverse(),
    )
    assert evaluation.exact.recall == 1.0
    assert evaluation.exact.precision == 1.0


# %% annotations of another class


def test_annotations_pair_by_where_they_stand_not_by_class(matcher, taxonomy):
    evaluation = evaluate(
        AnnotatedCabinetScene().create_world(),
        AnnotatedCabinetScene(handle_type=Apple, handle_is_part=False).create_world(),
        matcher,
        taxonomy,
    )
    score = score_on(evaluation, "handle")
    assert type(score.match.reconstructed_annotation) is Apple
    assert not score.same_class
    assert evaluation.extra_annotations == []


def test_superclass_in_place_of_a_subclass_costs_recall_only(matcher, taxonomy):
    evaluation = evaluate(
        AnnotatedCabinetScene(cabinet_type=Fridge).create_world(),
        AnnotatedCabinetScene(cabinet_type=Cabinet).create_world(),
        matcher,
        taxonomy,
    )
    score = score_on(evaluation, "cabinet")
    assert not score.same_class
    assert score.hierarchical.precision == 1.0
    assert score.hierarchical.recall == pytest.approx(2 / 3)


def test_unrelated_classes_share_no_ancestor(matcher, taxonomy):
    evaluation = evaluate(
        AnnotatedCabinetScene().create_world(),
        AnnotatedCabinetScene(handle_type=Apple, handle_is_part=False).create_world(),
        matcher,
        taxonomy,
    )
    score = score_on(evaluation, "handle")
    assert score.hierarchical.precision == 0.0
    assert score.hierarchical.recall == 0.0
    assert score.hierarchical.f_score == 0.0


# %% annotations without a counterpart


def test_body_without_an_annotation_leaves_the_annotation_unmatched(matcher, taxonomy):
    evaluation = evaluate(
        AnnotatedCabinetScene().create_world(),
        AnnotatedCabinetScene(drawer_type=None).create_world(),
        matcher,
        taxonomy,
    )
    assert root_names(evaluation.unmatched_ground_truth_annotations) == ["drawer"]
    assert evaluation.exact.recall == pytest.approx(2 / 3)
    assert evaluation.exact.precision == 1.0


def test_annotation_the_ground_truth_lacks_is_extra(matcher, taxonomy):
    evaluation = evaluate(
        AnnotatedCabinetScene(drawer_type=None).create_world(),
        AnnotatedCabinetScene().create_world(),
        matcher,
        taxonomy,
    )
    assert root_names(evaluation.extra_annotations) == ["drawer"]
    assert evaluation.exact.precision == pytest.approx(2 / 3)
    assert evaluation.exact.recall == 1.0


def test_annotation_of_a_missed_body_is_unmatched(matcher, taxonomy):
    evaluation = evaluate(
        AnnotatedCabinetScene().create_world(),
        AnnotatedCabinetScene(has_handle=False).create_world(),
        matcher,
        taxonomy,
    )
    assert root_names(evaluation.unmatched_ground_truth_annotations) == ["handle"]


def test_annotation_without_a_surface_takes_no_part(matcher, taxonomy):
    scene = AnnotatedCabinetScene(drawer_type=Door, drawer_field="doors")
    ground_truth_world = scene.create_world()
    assert ground_truth_world.get_semantic_annotations_by_type(EntryWay)

    evaluation = evaluate(ground_truth_world, scene.create_world(), matcher, taxonomy)

    compared = [
        score.match.ground_truth_annotation for score in evaluation.scores
    ] + evaluation.unmatched_ground_truth_annotations
    assert not any(isinstance(annotation, EntryWay) for annotation in compared)
    assert evaluation.exact.recall == 1.0


# %% the whole world


def test_hierarchical_scores_count_every_annotation_of_both_worlds(matcher, taxonomy):
    evaluation = evaluate(
        AnnotatedCabinetScene(cabinet_type=Fridge).create_world(),
        AnnotatedCabinetScene(cabinet_type=Cabinet, drawer_type=None).create_world(),
        matcher,
        taxonomy,
    )
    # Ground truth ancestors: Fridge 3, Drawer 2, Handle 1; reconstructed: Cabinet 2,
    # Handle 1. Shared: Cabinet and Furniture with the fridge, Handle with the handle.
    assert evaluation.hierarchical.precision == 1.0
    assert evaluation.hierarchical.recall == pytest.approx(3 / 6)
    assert evaluation.hierarchical.f_score == pytest.approx(2 / 3)


def test_worlds_without_annotations_have_no_semantic_scores(matcher, taxonomy):
    world = world_of(box("box", 0.3, 0.0))
    evaluation = evaluate(world, world_of(box("box", 0.3, 0.0)), matcher, taxonomy)

    assert evaluation.scores == []
    assert evaluation.exact.precision is None
    assert evaluation.exact.recall is None
    assert evaluation.hierarchical.f_score is None
