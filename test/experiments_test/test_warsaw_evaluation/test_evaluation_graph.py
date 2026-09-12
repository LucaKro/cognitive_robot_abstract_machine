"""Portable graph snapshots of the SDT produced by a pipeline run."""

from __future__ import annotations

from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.pipeline.records import (
    BodyAnswer,
    Classifications,
    RefusedMount,
    SplitBody,
    SplitRecord,
)
from experiments.warsaw.scene_split import Pairing
from semantic_digital_twin.semantic_annotations.taxonomy_export import MountKind

# %% preserving nodes and relation outcomes


def test_graph_snapshot_keeps_accepted_and_refused_relations() -> None:
    """Evaluation can distinguish a proposed edge from one present in the final SDT."""
    accepted = Pairing(
        whole="cabinet_1",
        part="drawer_1",
        field_name="drawers",
        kind=MountKind.PART,
    )
    refused = Pairing(
        whole="drawer_1",
        part="handle_1",
        field_name="handles",
        kind=MountKind.PART,
    )
    split = SplitRecord(
        scene="apartment.glb",
        bodies=[
            SplitBody(name="cabinet_1", label="cabinet", faces=100, body_id="b1"),
            SplitBody(name="drawer_1", label="drawer", faces=40, body_id="b2"),
            SplitBody(name="handle_1", label="handle", faces=10, body_id="b3"),
        ],
        pairings=[accepted, refused],
        refused=[RefusedMount(pairing=refused, reason="class does not admit part")],
        world_id=7,
        annotated_world_id=8,
    )
    classifications = Classifications(
        scene="apartment.glb",
        model="scripted/model",
        bodies=[
            BodyAnswer(name="cabinet_1", label="cabinet", class_name="Cabinet"),
            BodyAnswer(name="drawer_1", label="drawer", class_name="Drawer"),
            BodyAnswer(name="handle_1", label="handle", class_name="Handle"),
        ],
    )

    graph = EvaluationGraph.from_run_products(
        split=split,
        classifications=classifications,
        annotated_names={"cabinet_1", "drawer_1"},
    )

    assert graph.source_world_id == 7
    assert graph.annotated_world_id == 8
    assert [node.name for node in graph.nodes] == [
        "cabinet_1",
        "drawer_1",
        "handle_1",
    ]
    assert graph.nodes[0].predicted_class == "Cabinet"
    assert graph.nodes[0].annotation_applied is True
    assert graph.nodes[2].annotation_applied is False
    assert graph.edges[0].accepted is True
    assert graph.edges[0].relation == "part"
    assert graph.edges[1].accepted is False
    assert graph.edges[1].refusal_reason == "class does not admit part"
    written = graph.to_json()
    assert written["nodes"][0]["predicted_class"] == "Cabinet"
    assert written["edges"][1]["accepted"] is False


def test_unclassified_body_remains_visible_in_snapshot() -> None:
    """A missed annotation is measurable rather than disappearing from evaluation."""
    split = SplitRecord(
        scene="apartment.glb",
        bodies=[SplitBody(name="unknown_1", label="object", faces=12, body_id="b1")],
    )

    graph = EvaluationGraph.from_run_products(
        split=split,
        classifications=Classifications(scene="apartment.glb", model="scripted/model"),
        annotated_names=set(),
    )

    assert graph.nodes[0].predicted_class is None
    assert graph.nodes[0].annotation_applied is False
