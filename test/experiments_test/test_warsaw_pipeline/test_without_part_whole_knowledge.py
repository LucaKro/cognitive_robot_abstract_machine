"""
A run told not to use the ontology's part-whole knowledge.

It is the baseline for what that knowledge is worth: the model still reads the labels as
classes of the ontology and still decides which whole each part belongs to, but it is
told only the classes' names, every object is asked which of its neighbours it belongs
to, if any, and the graph is written from its answers rather than mounted.
"""

from __future__ import annotations

from pathlib import Path

from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.pipeline.database.ontology_export import OntologyExport
from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.provenance import settings_to_json
from experiments.warsaw.pipeline.records import (
    Adjudications,
    LabelAnswer,
    LabelRequest,
    MembershipAnswer,
    Relations,
    RelationStatus,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.adjudicate.step import MembershipDecision
from experiments.warsaw.pipeline.steps.classify.step import (
    BodiesByLabel,
    ClassifyBodies,
)
from experiments.warsaw.pipeline.steps.evidence import MeasureScene
from experiments.warsaw.pipeline.steps.split import SplitScene
from experiments.warsaw.pipeline.steps.vocabulary.step import (
    ClassProposalQuestion,
    LabelQuestion,
)
from experiments.warsaw.scene_split import Pairing

from .test_classes_by_label import names, segment
from .test_label_classes import apart
from .test_open_questions import described, memberships_of

# %% a run told not to use it


def unknowing() -> PipelineSettings:
    """
    :return: The settings of a run told not to use the ontology's part-whole knowledge.
    """
    return PipelineSettings(use_part_whole_knowledge=False)


def unknowing_step(tmp_path: Path) -> MeasureScene:
    """
    :return: The measuring step of such a run.
    """
    return MeasureScene(settings=unknowing(), run=Run(directory=tmp_path))


# %% the taxonomy the model is shown


def test_the_taxonomy_keeps_the_classes_and_drops_what_they_can_hold(taxonomy):
    mixins = {node["name"] for node in taxonomy["classes"] if node.get("mixin")}

    names_only = OntologyExport.without_part_whole_knowledge(taxonomy)

    assert names_only["part_whole_mixins"] == []
    assert names_only["classes"] == [
        {
            "name": node["name"],
            "bases": [base for base in node["bases"] if base not in mixins]
            or [taxonomy["root_name"]],
            **({"abstract": True} if node.get("abstract") else {}),
        }
        for node in taxonomy["classes"]
        if not node.get("mixin")
    ]


def test_an_export_told_to_leave_the_knowledge_out_reads_back_the_same(tmp_path):
    export = OntologyExport(
        run=Run(directory=tmp_path), keeps_part_whole_knowledge=False
    )

    read_back = OntologyExport.from_arguments([str(tmp_path), *export.arguments()])

    assert read_back.keeps_part_whole_knowledge is False


# %% what a label is asked as


def label_question(taxonomy, question_type=LabelQuestion) -> LabelQuestion:
    """
    :return: A question about a label of a run told not to use the knowledge.
    """
    return question_type(
        label=LabelRequest(label="cabinet", instances=3, exemplar="cabinet_2"),
        every_label=["cabinet", "drawer"],
        taxonomy=OntologyExport.without_part_whole_knowledge(taxonomy),
        renders_directory=Path(),
        knows_part_whole_relations=False,
    )


def test_a_label_is_asked_about_without_mentioning_mixins(taxonomy):
    assert "mixin" not in label_question(taxonomy).system_prompt


def test_a_class_is_proposed_without_mentioning_mixins(taxonomy):
    question = label_question(taxonomy, ClassProposalQuestion)

    assert "mixin" not in question.system_prompt


# %% what the ontology is asked about a pair


def test_the_ontology_is_not_asked_about_a_pair(tmp_path, known):
    view = unknowing_step(tmp_path).ontology_view(known["Cabinet"], known["Drawer"])

    assert view.status is RelationStatus.UNJUDGED
    assert view.mounts == []


def test_a_pair_is_judged_by_no_alternative_reading(tmp_path, known):
    read = VocabularyClasses(
        vocabulary=Vocabulary(
            model="",
            scene="",
            labels=[
                LabelAnswer(
                    label="cabinet", class_name="Cabinet", alternatives=["Drawer"]
                )
            ],
        ),
        known=known,
    )

    assert unknowing_step(tmp_path).alternative_classes(read) == {}


def test_a_slice_of_the_ontology_names_only_the_classes(tmp_path, known):
    sliced = unknowing_step(tmp_path).ontology_slice(
        ["drawer_1", "cabinet_1"],
        {"drawer_1": "drawer", "cabinet_1": "cabinet"},
        {"drawer": known["Drawer"], "cabinet": known["Cabinet"]},
    )

    assert sliced.read_as == {"drawer_1": "Drawer", "cabinet_1": "Cabinet"}
    assert (sliced.classes, sliced.admits) == ([], [])
    assert sliced.knows_part_whole_relations is False


# %% which neighbour each object belongs to


def test_every_object_is_asked_which_of_its_neighbours_it_belongs_to(tmp_path, known):
    memberships = memberships_of(unknowing_step(tmp_path), known, ("rail_1", 3, 0.0))

    assert memberships.forced == []
    assert {
        one.part: (one.candidate_names, one.may_belong_to_none)
        for one in memberships.to_ask_about
    } == {"blinds_1": (["rail_1"], True), "rail_1": (["blinds_1"], True)}


def test_an_object_touching_a_neighbour_is_not_offered_one_only_nearby(tmp_path, known):
    step = unknowing_step(tmp_path)
    gap = step.settings.largest_gap_to_a_whole

    memberships = memberships_of(
        step, known, ("rail_1", 3, 0.0), ("rail_2", 0, gap / 2)
    )

    [blinds] = [one for one in memberships.to_ask_about if one.part == "blinds_1"]
    assert blinds.candidate_names == ["rail_1"]


def test_a_neighbour_past_the_gap_is_not_offered(tmp_path, known):
    step = unknowing_step(tmp_path)
    gap = step.settings.largest_gap_to_a_whole

    memberships = memberships_of(step, known, ("rail_1", 0, gap * 2))

    assert (memberships.forced, memberships.to_ask_about) == ([], [])


def test_a_question_about_a_neighbour_names_no_field_to_hold_it(tmp_path, known):
    [question, _] = memberships_of(
        unknowing_step(tmp_path), known, ("rail_1", 3, 0.0)
    ).to_ask_about

    said = MembershipDecision(
        asked=question,
        labels={"blinds_1": "blinds", "rail_1": "rail"},
        renders_directory=tmp_path,
    ).message()[0]

    assert "would hold it" not in said.text


# %% answers that contradict each other


def test_the_least_confident_answer_of_a_cycle_is_not_mounted(tmp_path):
    answers = [
        MembershipAnswer(
            name="door_1", part="door_1", whole="cabinet_1", confidence=0.9
        ),
        MembershipAnswer(
            name="cabinet_1", part="cabinet_1", whole="door_1", confidence=0.6
        ),
        MembershipAnswer(
            name="handle_1", part="handle_1", whole="door_1", confidence=0.8
        ),
    ]
    step = SplitScene(settings=unknowing(), run=Run(directory=tmp_path))

    carried = step.named_pairings(
        Adjudications(model="", scene="", answered=answers), Relations(scene="")
    )

    assert [(one.part, one.whole) for one in carried] == [
        ("door_1", "cabinet_1"),
        ("handle_1", "door_1"),
    ]


# %% naming each body


def test_a_body_chooses_among_every_reading_of_its_label(tmp_path):
    """
    The handle base touches only a door, which the ontology would say no door attachment
    hangs from; a run not told what hangs from what keeps both readings open.
    """
    run = Run(directory=tmp_path)
    vocabulary = Vocabulary(
        model="",
        scene="",
        labels=[
            LabelAnswer(
                label="handle_base",
                class_name="Backplate",
                alternatives=["DoorAttachment"],
            ),
            LabelAnswer(label="door", class_name="Door"),
        ],
    )
    run.write_record(RunFile.VOCABULARY, vocabulary)
    run.write_json(
        RunFile.TAXONOMY,
        {
            "classes": [
                {"name": name}
                for answer in vocabulary.labels
                for name in answer.candidates
            ]
        },
    )
    run.write_record(
        RunFile.RELATIONS,
        Relations(
            scene="a door",
            segments=[
                described("handle_base_1", "handle_base"),
                described("door_1", "door"),
            ],
            pairs=[apart("handle_base_1", "door_1", 0.0)],
        ),
    )
    step = ClassifyBodies(settings=unknowing(), run=run)

    sorted_out = BodiesByLabel.of(
        [segment("handle_base")], step.readings_to_choose_from(), settle=True
    )

    assert names(sorted_out) == ([], ["handle_base_1"], [])


# %% the graph, written from the answers


def test_a_run_reads_back_that_the_knowledge_was_not_used(tmp_path):
    run = Run(directory=tmp_path)
    run.write_json(RunFile.PROVENANCE, {"settings": settings_to_json(unknowing())})

    assert run.uses_part_whole_knowledge is False


def test_a_run_that_says_nothing_used_the_knowledge(tmp_path):
    assert Run(directory=tmp_path).uses_part_whole_knowledge is True


def test_each_body_hangs_below_the_whole_it_was_paired_with():
    pairings = [
        Pairing(whole="cabinet_1", part="door_1", field_name=""),
        Pairing(whole="door_1", part="handle_1", field_name=""),
    ]

    parents = EvaluationGraph.paired_parents(
        pairings, {"cabinet_1", "door_1", "handle_1", "wall_1"}
    )

    assert parents == {
        "cabinet_1": None,
        "door_1": "cabinet_1",
        "handle_1": "door_1",
        "wall_1": None,
    }
