"""
Leaving one part of the pipeline out, so that what it was worth is a number.

A finished run stores every intermediate record, so a step that only decides what
happens after the model answered can be replayed without asking anything again. What
each of these has to get right is that it changes exactly one thing and leaves the rest
of the run alone.
"""

from __future__ import annotations

from experiments.warsaw.evaluation.ablations import (
    AdjudicationAgreement,
    KeepingEmptiedWholes,
    WithoutTheModelsRelations,
)
from experiments.warsaw.evaluation.graph import (
    EvaluationEdge,
    EvaluationGraph,
    EvaluationNode,
)
from experiments.warsaw.pipeline.records import (
    EmptiedSegment,
    ForcedMembership,
    MembershipAnswer,
    OwnershipAnswer,
    TakenFaces,
)

# %% a run to ablate


def node(name: str, label: str, faces: int) -> EvaluationNode:
    """
    One body a run built.
    """
    return EvaluationNode(
        name=name,
        input_label=label,
        predicted_class=label.capitalize(),
        faces=faces,
        body_id=name,
        annotation_applied=True,
    )


def graph_of_a_cabinet_and_two_drawers() -> EvaluationGraph:
    """
    A run holding a cabinet with two drawers, one relation of which geometry forced.
    """
    return EvaluationGraph(
        nodes=[
            node("cabinet_1", "cabinet", 900),
            node("drawer_1", "drawer", 300),
            node("drawer_2", "drawer", 300),
        ],
        edges=[
            EvaluationEdge(
                whole="cabinet_1",
                part="drawer_1",
                relation="part",
                field_name="drawers",
                accepted=True,
            ),
            EvaluationEdge(
                whole="cabinet_1",
                part="drawer_2",
                relation="part",
                field_name="drawers",
                accepted=True,
            ),
        ],
    )


# %% the relations the model decided


def test_only_the_memberships_geometry_forced_survive():
    """
    The baseline the model has to beat: what a run would have asserted having asked
    nothing, which is the memberships where a part met exactly one candidate.
    """
    forced = [
        ForcedMembership(
            whole="cabinet_1", part="drawer_1", field_name="drawers", shared_faces=40
        )
    ]

    relations = WithoutTheModelsRelations.forced_only(
        forced, graph_of_a_cabinet_and_two_drawers()
    )

    assert [(one.whole, one.part) for one in relations] == [("cabinet_1", "drawer_1")]


def test_a_forced_membership_keeps_the_kind_the_run_gave_that_field():
    """
    Geometry says two things meet and which field holds them, not what the relation
    means, so the kind is taken from what the run itself made of that field rather than
    guessed.
    """
    forced = [
        ForcedMembership(
            whole="cabinet_1", part="drawer_1", field_name="drawers", shared_faces=40
        )
    ]

    [relation] = WithoutTheModelsRelations.forced_only(
        forced, graph_of_a_cabinet_and_two_drawers()
    )

    assert relation.relation == "part"


def test_the_classes_are_left_alone_because_nothing_else_states_them():
    """
    This ablates the model's relation decisions and not its classification: without the
    classes there is nothing to compare against a ground truth stated in classes.
    """
    graph = graph_of_a_cabinet_and_two_drawers()

    ablated = WithoutTheModelsRelations().applied_to_graph(graph, [])

    assert [one.predicted_class for one in ablated.nodes] == [
        one.predicted_class for one in graph.nodes
    ]


# %% the wholes the split emptied


def test_a_part_that_consumed_its_whole_gets_that_whole_back():
    """
    The split drops an object whose every face was taken, which is why a drawer that
    consumed its own cabinet ends up belonging to nothing.

    Putting the whole back is what measures that rule rather than arguing about it.
    """
    emptied = [
        EmptiedSegment(
            name="cabinet_9",
            taken_by=[TakenFaces(name="drawer_3", faces=200)],
        )
    ]
    graph = graph_of_a_cabinet_and_two_drawers()

    relations = KeepingEmptiedWholes.with_the_emptied_put_back(emptied, graph)

    assert ("cabinet_9", "drawer_3") in [(one.whole, one.part) for one in relations]


def test_the_relations_the_run_already_had_are_kept():
    """
    Only the dropped wholes are being put back; everything the run decided stands.
    """
    emptied = [
        EmptiedSegment(
            name="cabinet_9", taken_by=[TakenFaces(name="drawer_3", faces=1)]
        )
    ]
    graph = graph_of_a_cabinet_and_two_drawers()

    relations = KeepingEmptiedWholes.with_the_emptied_put_back(emptied, graph)

    assert len(relations) == len(graph.edges) + 1


def test_the_whole_goes_back_to_whichever_part_took_most_of_it():
    """
    Several bodies may have taken a share; the one that took most is the one it was
    really part of, and handing it to all of them would invent relations the run never
    had grounds for.
    """
    emptied = [
        EmptiedSegment(
            name="cabinet_9",
            taken_by=[
                TakenFaces(name="handle_4", faces=20),
                TakenFaces(name="drawer_3", faces=200),
            ],
        )
    ]
    graph = graph_of_a_cabinet_and_two_drawers()

    put_back = [
        one
        for one in KeepingEmptiedWholes.with_the_emptied_put_back(emptied, graph)
        if one.whole == "cabinet_9"
    ]

    assert [one.part for one in put_back] == ["drawer_3"]


# %% the contested faces the model ruled on


def test_the_model_agreeing_with_the_larger_claimant_needed_no_asking():
    """
    The cheap default is that the larger of two claimants owns what both claim.

    Where the model says the same, the call bought nothing.
    """
    answers = [
        OwnershipAnswer(
            name="cabinet__drawer", pattern=["cabinet", "drawer"], owner="cabinet"
        )
    ]

    agreement = AdjudicationAgreement.between(
        answers, graph_of_a_cabinet_and_two_drawers()
    )

    assert agreement.decided == 1
    assert agreement.agreeing == 1
    assert agreement.differing == []


def test_the_model_overruling_the_larger_claimant_is_what_it_was_for():
    """
    These are the calls that changed something, and counting them is the whole point:

    they are what the step is worth.
    """
    answers = [
        OwnershipAnswer(
            name="cabinet__drawer", pattern=["cabinet", "drawer"], owner="drawer"
        )
    ]

    agreement = AdjudicationAgreement.between(
        answers, graph_of_a_cabinet_and_two_drawers()
    )

    assert agreement.agreeing == 0
    assert agreement.differing == ["cabinet__drawer"]


def test_rulings_about_where_a_part_belongs_are_not_counted_as_rulings_about_faces():
    """
    A run's adjudications hold two kinds of answer -- whose contested faces are whose,
    and which whole a part belongs to -- and only the first has a cheap default to be
    compared against.

    Counting both would measure the step against a rule that does not apply to half of
    it.
    """
    answers = [
        OwnershipAnswer(
            name="cabinet__drawer", pattern=["cabinet", "drawer"], owner="cabinet"
        ),
        MembershipAnswer(name="drawer_1", part="drawer_1", whole="cabinet_1"),
    ]

    agreement = AdjudicationAgreement.between(
        answers, graph_of_a_cabinet_and_two_drawers()
    )

    assert agreement.decided == 1
