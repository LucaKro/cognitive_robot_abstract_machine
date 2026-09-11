"""
Deciding which reconstructed object is which modelled one.

The pairing has to be settled over the whole scene rather than object by object, and it
has to leave things unpaired: an object the other graph does not hold is the finding,
not a nuisance to be paired off with whatever is least unlike it. It also has to be able
to pair two objects that disagree about their class, or a classification mistake could
never be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from typing_extensions import List, Optional

from experiments.warsaw.evaluation.matching import (
    HowToCompare,
    ObjectCorrespondences,
)
from experiments.warsaw.evaluation.size import ObjectSize

# %% objects to pair


@dataclass(frozen=True)
class Object:
    """
    One object of either graph, standing in for a node of it.
    """

    name: str
    """
    What identifies it within its own graph.
    """

    semantic_class: Optional[str] = None
    """
    What it stands as, or nothing where its graph names it nothing.
    """

    measured: Optional[ObjectSize] = None
    """
    How big it is, where it was measured.
    """

    placed: Optional[List[float]] = None
    """
    Where its middle is, where both graphs are in one frame.
    """

    spans: Optional[List[List[float]]] = None
    """
    What it spans, as low and high corners, where it was measured.
    """

    @property
    def classes(self) -> List[str]:
        """
        :return: The classes it stands as.
        """
        return [] if self.semantic_class is None else [self.semantic_class]

    @property
    def size(self) -> Optional[ObjectSize]:
        """
        :return: How big it is.
        """
        return self.measured

    @property
    def centre(self) -> Optional[List[float]]:
        """
        :return: Where its middle is, or nothing where the two graphs share no frame.
        """
        return self.placed

    @property
    def bounds(self) -> Optional[List[List[float]]]:
        """
        :return: What it spans, or nothing where that was never measured.
        """
        return self.spans


def sized(*extents: float) -> ObjectSize:
    """
    Build a size from the sides of its box, with an area that follows from them.
    """
    length, width, height = sorted(extents, reverse=True)
    return ObjectSize(
        extents=[length, width, height],
        surface_area=2 * (length * width + length * height + width * height),
    )


def drawer(name: str) -> Object:
    """
    A drawer-sized object called a drawer.
    """
    return Object(name=name, semantic_class="Drawer", measured=sized(0.6, 0.5, 0.2))


def handle(name: str) -> Object:
    """
    A handle-sized object called a handle.
    """
    return Object(name=name, semantic_class="Handle", measured=sized(0.15, 0.03, 0.03))


# %% pairing what is alike


def test_objects_alike_in_class_and_size_are_paired():
    """
    The plain case: one drawer in each graph, and nothing else they could be.
    """
    correspondences = ObjectCorrespondences.between(
        [drawer("drawer_1")], [drawer("apartment/cabinet1_drawer_top")]
    )

    assert [(one.predicted, one.modelled) for one in correspondences.matched] == [
        ("drawer_1", "apartment/cabinet1_drawer_top")
    ]
    assert correspondences.matched[0].classes_agree


def test_an_object_the_other_graph_does_not_hold_is_left_unpaired():
    """
    Pairing it off with whatever is least unlike it would hide the finding: the run
    built something the modelled world never claimed.
    """
    correspondences = ObjectCorrespondences.between(
        [drawer("drawer_1"), handle("handle_1")],
        [drawer("apartment/cabinet1_drawer_top")],
    )

    assert correspondences.unmatched_predicted == ["handle_1"]
    assert correspondences.unmatched_modelled == []


def test_a_modelled_object_the_run_never_found_is_left_unpaired():
    """
    The same the other way round, which is what a missed detection looks like.
    """
    correspondences = ObjectCorrespondences.between(
        [drawer("drawer_1")],
        [drawer("apartment/drawer_a"), handle("apartment/handle_a")],
    )

    assert correspondences.unmatched_modelled == ["apartment/handle_a"]


def test_nothing_is_paired_when_one_graph_holds_nothing():
    """
    A scene the run found nothing in is every modelled object missed, not an error.
    """
    correspondences = ObjectCorrespondences.between([], [drawer("apartment/drawer_a")])

    assert correspondences.matched == []
    assert correspondences.unmatched_modelled == ["apartment/drawer_a"]


# %% pairing over the whole scene rather than object by object


def test_the_pairing_chosen_is_the_cheapest_over_the_whole_scene():
    """
    Two objects may both be nearest the same partner, and only one can have it.

    Letting whichever was asked first take it can cost the other a pairing it should
    have had, so the whole assignment is settled at once.
    """
    small = Object(name="small", semantic_class="Drawer", measured=sized(0.4, 0.4, 0.2))
    large = Object(name="large", semantic_class="Drawer", measured=sized(0.8, 0.6, 0.3))
    modelled_small = Object(
        name="modelled_small", semantic_class="Drawer", measured=sized(0.42, 0.4, 0.2)
    )
    modelled_large = Object(
        name="modelled_large", semantic_class="Drawer", measured=sized(0.78, 0.6, 0.3)
    )

    correspondences = ObjectCorrespondences.between(
        [small, large], [modelled_large, modelled_small]
    )

    assert {(one.predicted, one.modelled) for one in correspondences.matched} == {
        ("small", "modelled_small"),
        ("large", "modelled_large"),
    }


# %% pairing across a disagreement about class


def test_two_objects_disagreeing_about_their_class_can_still_be_paired():
    """
    A correspondence that only ever pairs objects already agreeing about their class
    cannot measure a classification mistake: every one would be reported as an object
    nobody found.
    """
    called_a_cupboard = Object(
        name="cabinet_5", semantic_class="Cupboard", measured=sized(0.6, 0.6, 2.1)
    )
    modelled_cabinet = Object(
        name="apartment/cabinet5",
        semantic_class="Cabinet",
        measured=sized(0.6, 0.6, 2.1),
    )

    [paired] = ObjectCorrespondences.between(
        [called_a_cupboard], [modelled_cabinet]
    ).matched

    assert (paired.predicted, paired.modelled) == ("cabinet_5", "apartment/cabinet5")
    assert not paired.classes_agree


def test_a_pairing_is_refused_when_nothing_about_the_two_is_alike():
    """
    Disagreeing about the class is survivable and disagreeing about everything is not,
    or a handle would be reported as a cupboard the run got wrong.
    """
    correspondences = ObjectCorrespondences.between(
        [handle("handle_1")],
        [
            Object(
                name="apartment/cabinet5",
                semantic_class="Cabinet",
                measured=sized(0.6, 0.6, 2.1),
            )
        ],
    )

    assert correspondences.matched == []
    assert correspondences.unmatched_predicted == ["handle_1"]
    assert correspondences.unmatched_modelled == ["apartment/cabinet5"]


# %% what the correspondence records


def test_a_pairing_records_what_it_cost_and_where_the_cost_came_from():
    """
    A pairing settled on sizes alone and one the classes confirm are not equally
    trustworthy, and should not read alike afterwards.
    """
    [paired] = ObjectCorrespondences.between(
        [drawer("drawer_1")], [drawer("apartment/drawer_a")]
    ).matched

    assert paired.cost.disagreeing_class == 0.0
    assert paired.cost.differing_size == 0.0
    assert paired.cost.total == 0.0


def test_an_object_that_was_never_measured_is_still_offered_a_pairing():
    """
    A size nobody measured is not evidence against a pairing, only the absence of
    evidence for one, so the class still has to be able to carry it.
    """
    unmeasured = Object(name="drawer_1", semantic_class="Drawer")
    modelled = Object(name="apartment/drawer_a", semantic_class="Drawer")
    lenient = HowToCompare(most_a_pairing_may_cost=1.0)

    correspondences = ObjectCorrespondences.between(
        [unmeasured], [modelled], how_compared=lenient
    )

    assert [one.modelled for one in correspondences.matched] == ["apartment/drawer_a"]
    assert correspondences.matched[0].cost.differing_size == 1.0


def test_the_correspondence_records_the_settings_it_was_worked_out_under():
    """
    A result is only arguable if it says what counted as alike when it was made.
    """
    how_compared = HowToCompare(most_a_pairing_may_cost=0.5)

    correspondences = ObjectCorrespondences.between(
        [drawer("drawer_1")], [drawer("apartment/drawer_a")], how_compared=how_compared
    )

    assert correspondences.how_compared == how_compared


def test_a_correspondence_reads_back_as_what_was_written():
    """
    It is written down once and read by everything that scores afterwards.
    """
    correspondences = ObjectCorrespondences.between(
        [drawer("drawer_1"), handle("handle_1")],
        [drawer("apartment/drawer_a")],
    )

    assert ObjectCorrespondences.from_json(correspondences.to_json()) == correspondences


# %% how far a pairing can be trusted


def test_a_pairing_with_nothing_else_like_it_leads_the_next_best_by_a_wide_margin():
    """
    Where the other graph holds one plainly alike object and one plainly unlike one, the
    pairing is not a matter of opinion and should not read as one.
    """
    [paired] = ObjectCorrespondences.between(
        [drawer("drawer_1")],
        [drawer("apartment/drawer_a"), handle("apartment/handle_a")],
    ).matched

    assert paired.better_than_the_next_by > 0.5


def test_a_pairing_among_objects_alike_to_each_other_leads_the_next_best_by_nothing():
    """
    Twenty-five handles all the same size cannot be told apart by what they are and how
    big they are.

    The pairing still has to be made, but it has to say that it could as well have been
    the other one.
    """
    correspondences = ObjectCorrespondences.between(
        [handle("handle_1")],
        [handle("apartment/handle_a"), handle("apartment/handle_b")],
    )

    assert len(correspondences.matched) == 1
    assert correspondences.matched[0].better_than_the_next_by == 0.0


def test_a_pairing_that_was_the_only_one_on_offer_leads_by_nothing():
    """
    There was no second best to be better than, which is not the same as being sure.
    """
    [paired] = ObjectCorrespondences.between(
        [drawer("drawer_1")], [drawer("apartment/drawer_a")]
    ).matched

    assert paired.better_than_the_next_by == 0.0


# %% pairing by where things are


def test_where_an_object_is_decides_between_two_that_are_otherwise_alike():
    """
    Two drawers of one size are indistinguishable by class and size, which is the case
    every run of a kitchen is full of.

    Once an alignment puts both worlds in one frame, where each sits is the only
    evidence that tells them apart.
    """
    how_compared = HowToCompare(distance_apart=1.0)
    near = replace(drawer("apartment/drawer_near"), placed=[0.0, 0.0, 0.6])
    far = replace(drawer("apartment/drawer_far"), placed=[0.0, 0.0, 2.6])

    correspondences = ObjectCorrespondences.between(
        [replace(drawer("drawer_1"), placed=[0.02, 0.0, 0.6])],
        [far, near],
        how_compared,
    )

    assert [one.modelled for one in correspondences.matched] == [
        "apartment/drawer_near"
    ]


def test_two_graphs_that_share_no_frame_are_not_charged_for_being_apart():
    """
    Where nothing has been aligned there is no such evidence, and charging for its
    absence would refuse every pairing in a scene that has no landmarks picked yet.
    """
    correspondences = ObjectCorrespondences.between(
        [drawer("drawer_1")], [drawer("apartment/drawer_a")], HowToCompare()
    )

    assert correspondences.matched[0].cost.far_apart == 0.0
    assert correspondences.matched[0].cost.total == 0.0


def test_a_scanned_front_counts_as_being_at_the_object_it_lies_on():
    """
    A scan sees the front of a cabinet and the modelled world is a solid box, so their
    middles sit half a carcass apart while the front lies flat against the box.

    Measuring to the box rather than between the middles is what stops that being read
    as a pairing between two different objects.
    """
    how_compared = HowToCompare(distance_apart=1.0)
    carcass = Object(
        name="apartment/cabinet1",
        semantic_class="Cabinet",
        placed=[0.0, 0.3, 1.0],
        spans=[[-0.3, 0.0, 0.0], [0.3, 0.6, 2.0]],
    )
    front = Object(name="cabinet_1", semantic_class="Cabinet", placed=[0.0, 0.0, 1.0])

    assert how_compared.cost_of(front, carcass).far_apart == 0.0
