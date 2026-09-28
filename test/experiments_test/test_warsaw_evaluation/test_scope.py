"""
Declaring what a comparison covers, and applying it to both graphs alike.

The point of a declared scope is that it moves no score by itself: it is written once,
read beside both graphs, and applied to each identically. So what is tested here is that
one file places a reconstructed body and a modelled entity the same way, and that
nothing left out drags relations back in behind it.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.warsaw.evaluation.graph import EvaluationEdge, EvaluationNode
from experiments.warsaw.evaluation.ground_truth import GroundTruthEdge, GroundTruthNode
from experiments.warsaw.evaluation.scope import (
    ClassDecision,
    ComparisonRole,
    ComparisonScope,
    EntityDecision,
)

# %% a scope and the two kinds of graph it is read beside


def declared_scope() -> ComparisonScope:
    """
    Declare the kinds of decision the IAI apartment comparison makes.
    """
    return ComparisonScope(
        scene="iai_apartment",
        classes=[
            ClassDecision(
                semantic_class="Ceiling",
                role=ComparisonRole.EXCLUDED,
                reason="The modelled world holds no ceiling body to correspond to.",
            ),
            ClassDecision(
                semantic_class="Wall",
                role=ComparisonRole.AREA,
                reason="One merged body against many reconstructed segments.",
            ),
        ],
        entities=[
            EntityDecision(
                entity="apartment/tap_body",
                role=ComparisonRole.EXCLUDED,
                reason="The ontology has no class for a tap.",
            ),
            EntityDecision(
                entity="wall_7",
                role=ComparisonRole.INSTANCE,
                reason="A worked example of an object named against its class.",
            ),
        ],
    )


def predicted(name: str, predicted_class: str | None) -> EvaluationNode:
    """
    Build one reconstructed body.
    """
    return EvaluationNode(
        name=name,
        input_label="label",
        predicted_class=predicted_class,
        faces=10,
        body_id=None,
        annotation_applied=True,
    )


def modelled(
    name: str, semantic_classes: list[str], faces: int = 10
) -> GroundTruthNode:
    """
    Build one entity of a modelled world.
    """
    return GroundTruthNode(
        name=name,
        source_id="id",
        parent=None,
        semantic_classes=semantic_classes,
        faces=faces,
        world_transform=[[1.0]],
        bounds=None,
    )


# %% placing an object


def test_a_class_decision_places_every_object_of_that_class():
    """
    Scope is declared over the vocabulary, not over the objects one run happened to
    produce, so it holds for a run that has not been made yet.
    """
    scope = declared_scope()

    assert scope.role_of(predicted("ceiling_3", "Ceiling")) is ComparisonRole.EXCLUDED
    assert scope.role_of(predicted("ceiling_9", "Ceiling")) is ComparisonRole.EXCLUDED


def test_an_object_no_decision_mentions_is_compared_as_an_object():
    """
    The file records departures from comparing everything, not the whole vocabulary.
    """
    scope = declared_scope()

    assert scope.role_of(predicted("cabinet_5", "Cabinet")) is ComparisonRole.INSTANCE


def test_an_entity_decision_places_an_object_its_class_cannot():
    """
    A body the modelled world leaves unnamed carries no class to decide by, which is
    exactly the case the tap is.
    """
    scope = declared_scope()

    assert scope.role_of(modelled("apartment/tap_body", [])) is ComparisonRole.EXCLUDED


def test_a_decision_about_one_object_wins_over_the_one_about_its_class():
    """
    A decision naming the object was written knowing what class it carries.
    """
    scope = declared_scope()

    assert scope.role_of(predicted("wall_7", "Wall")) is ComparisonRole.INSTANCE
    assert scope.role_of(predicted("wall_2", "Wall")) is ComparisonRole.AREA


def test_the_same_scope_places_a_reconstructed_and_a_modelled_object_alike():
    """
    Applying a scope to one side only would move the score without either graph having
    changed, so the two graphs have to answer it the same way.
    """
    scope = declared_scope()

    assert scope.role_of(predicted("wall_2", "Wall")) is scope.role_of(
        modelled("apartment/wall_coloksu_wall2", ["Wall"])
    )
    assert scope.role_of(predicted("cabinet_5", "Cabinet")) is scope.role_of(
        modelled("apartment/cabinet5", ["Cabinet"])
    )


def test_an_object_the_run_never_classified_is_compared_as_an_object():
    """
    A body with no class is still a body the reconstruction found, and dropping it would
    hide a classification failure instead of scoring it.
    """
    scope = declared_scope()

    assert scope.role_of(predicted("body_12", None)) is ComparisonRole.INSTANCE


# %% collecting what plays each part


def test_objects_are_collected_by_what_they_do_in_the_comparison():
    """
    Instances and area are scored by different measures, so they are asked for apart.
    """
    scope = declared_scope()
    nodes = [
        predicted("ceiling_3", "Ceiling"),
        predicted("wall_2", "Wall"),
        predicted("cabinet_5", "Cabinet"),
    ]

    assert [
        node.name for node in scope.nodes_playing(ComparisonRole.INSTANCE, nodes)
    ] == ["cabinet_5"]
    assert [node.name for node in scope.nodes_playing(ComparisonRole.AREA, nodes)] == [
        "wall_2"
    ]
    assert [
        node.name for node in scope.nodes_playing(ComparisonRole.EXCLUDED, nodes)
    ] == ["ceiling_3"]


# %% relations reaching out of scope


def test_a_relation_reaching_something_left_out_is_left_out_with_it():
    """
    Such a relation cannot be right or wrong in the comparison, and scoring it would
    charge a graph for the scope rather than for what it built.
    """
    scope = declared_scope()
    nodes = [
        predicted("cabinet_5", "Cabinet"),
        predicted("door_2", "Door"),
        predicted("ceiling_3", "Ceiling"),
    ]
    edges = [
        EvaluationEdge(
            whole="cabinet_5",
            part="door_2",
            relation="part",
            field_name="doors",
            accepted=True,
        ),
        EvaluationEdge(
            whole="cabinet_5",
            part="ceiling_3",
            relation="part",
            field_name="",
            accepted=True,
        ),
    ]

    kept = scope.edges_between_instances(edges, nodes)

    assert [(edge.whole, edge.part) for edge in kept] == [("cabinet_5", "door_2")]


def test_a_relation_reaching_something_compared_as_area_is_left_out_too():
    """
    Area is not matched one to one, so nothing can stand at the other end of the
    relation to agree or disagree with it.
    """
    scope = declared_scope()
    nodes = [
        modelled("apartment/walls", ["Wall"]),
        modelled("aperture_1", ["Aperture"]),
    ]
    edges = [
        GroundTruthEdge(
            whole="apartment/walls",
            part="aperture_1",
            relation="part",
            field_name="apertures",
        )
    ]

    assert scope.edges_between_instances(edges, nodes) == []


# %% reading the declared scope


def test_a_declared_scope_reads_back_as_what_was_written(tmp_path: Path):
    """
    The scope is checked in beside the graphs and recorded with every result.
    """
    scope = declared_scope()
    written = tmp_path / "scope.json"
    written.write_text(json.dumps(scope.to_json(), indent=2))

    assert ComparisonScope.read(written) == scope


# %% objects no reconstruction could have found


def test_an_object_with_no_geometry_is_left_out_whatever_its_class():
    """
    World models carry joints and hierarchy on bodies with no shapes.

    Counting those as ground truth would make every one of them a miss the run could not
    have avoided.
    """
    scope = declared_scope()
    joint_body = modelled("apartment/cabinet3_door_out_fancy", ["Door"], faces=0)

    assert joint_body.faces == 0
    assert scope.role_of(joint_body) is ComparisonRole.EXCLUDED


def test_a_decision_about_one_object_wins_over_it_having_no_geometry():
    """
    A decision naming the object was written knowing what the world holds for it.
    """
    scope = ComparisonScope(
        scene="iai_apartment",
        entities=[
            EntityDecision(
                entity="apartment/region",
                role=ComparisonRole.AREA,
                reason="A region has a pose and an extent but no shapes of its own.",
            )
        ],
    )

    assert (
        scope.role_of(modelled("apartment/region", [], faces=0)) is ComparisonRole.AREA
    )
