"""
Colouring one overlap so a reader can see what the split did to it.

A figure of the split has to show three things at once: which faces several labels
claim, who ended up with them, and what was lost. The colouring is what carries that, so
it is what is tested here, apart from any run or any mesh on disk.
"""

from __future__ import annotations

import numpy as np
import trimesh

from experiments.warsaw.evaluation.overlap_case import (
    CONTESTED,
    CONTESTED_FACES,
    ELSEWHERE,
    FACES_ELSEWHERE,
    OverlapCase,
    case_mesh,
    case_scene,
)

# %% one overlap, small enough to count by hand

CABINET = "cabinet_1"
DRAWER = "drawer_1"
HANDLE = "handle_1"


def a_cabinet_losing_its_front() -> OverlapCase:
    """
    A cabinet claiming five faces, of which the drawer claims two and the handle one.
    """
    return OverlapCase(
        claims={
            CABINET: np.array([0, 1, 2, 3, 4]),
            DRAWER: np.array([2, 3, 5]),
            HANDLE: np.array([3]),
        },
        bodies={
            CABINET: np.array([0, 1, 4]),
            DRAWER: np.array([2, 5]),
            HANDLE: np.array([3]),
        },
    )


# %% which faces the case covers


class TestWhatTheCaseCovers:
    """
    The faces a case is drawn from, and which of them several labels claim.
    """

    def test_the_faces_are_every_face_any_claimant_claims(self):
        case = a_cabinet_losing_its_front()
        assert list(case.faces) == [0, 1, 2, 3, 4, 5]

    def test_the_contested_faces_are_those_claimed_more_than_once(self):
        case = a_cabinet_losing_its_front()
        assert list(case.contested) == [2, 3]


# %% what the colours say


class TestColouringBeforeTheSplit:
    """
    Before the split, a face says whether it is claimed once or by several labels.
    """

    def test_a_face_one_claimant_claims_takes_that_claimant_s_colour(self):
        case = a_cabinet_losing_its_front()
        colours = case.before()
        assert list(colours[list(case.faces).index(0)]) == list(case.colour_of(CABINET))
        assert list(colours[list(case.faces).index(5)]) == list(case.colour_of(DRAWER))

    def test_a_contested_face_is_coloured_as_contested(self):
        case = a_cabinet_losing_its_front()
        colours = case.before()
        for face in case.contested:
            assert list(colours[list(case.faces).index(face)]) == list(CONTESTED)


class TestColouringAfterTheSplit:
    """
    After the split, a face says which body kept it, and a face that went elsewhere says
    so.
    """

    def test_a_kept_face_takes_the_colour_of_the_body_that_kept_it(self):
        case = a_cabinet_losing_its_front()
        colours = case.after()
        assert list(colours[list(case.faces).index(2)]) == list(case.colour_of(DRAWER))
        assert list(colours[list(case.faces).index(3)]) == list(case.colour_of(HANDLE))

    def test_a_face_none_of_the_drawn_bodies_kept_is_coloured_as_elsewhere(self):
        case = OverlapCase(
            claims={CABINET: np.array([0, 1])}, bodies={CABINET: np.array([0])}
        )
        colours = case.after()
        assert list(colours[list(case.faces).index(1)]) == list(ELSEWHERE)

    def test_a_claimant_left_without_faces_is_named_as_emptied(self):
        case = OverlapCase(
            claims={CABINET: np.array([0]), DRAWER: np.array([0])},
            bodies={DRAWER: np.array([0])},
        )
        assert case.emptied == [CABINET]


# %% what is written out


class TestTheMeshThatIsWritten:
    """
    The mesh a case is cut out of carries one colour per face of the case.
    """

    def test_the_mesh_holds_the_case_s_faces_in_their_colours(self):
        case = a_cabinet_losing_its_front()
        mesh = case_mesh(trimesh.creation.box(), case.faces, case.before())
        assert len(mesh.faces) == len(case.faces)
        assert np.array_equal(mesh.visual.face_colors, case.before())


# %% the objects a modelling tool opens


class TestTheObjectsOfACase:
    """
    A case is written as one named object per claimant, so a viewer can pick them apart
    and each carries a colour of its own rather than a colour per face.
    """

    def test_before_the_split_the_contested_faces_are_an_object_of_their_own(self):
        case = a_cabinet_losing_its_front()
        groups = case.before_groups()
        assert list(groups[CONTESTED_FACES]) == list(case.contested)

    def test_before_the_split_a_claimant_holds_everything_it_claims(self):
        case = a_cabinet_losing_its_front()
        groups = case.before_groups()
        assert list(groups[CABINET]) == [0, 1, 2, 3, 4]
        assert list(groups[DRAWER]) == [2, 3, 5]
        assert list(groups[HANDLE]) == [3]

    def test_after_the_split_every_claimant_holds_what_it_kept(self):
        case = a_cabinet_losing_its_front()
        groups = case.after_groups()
        assert list(groups[DRAWER]) == [2, 5]
        assert list(groups[HANDLE]) == [3]

    def test_after_the_split_the_faces_that_went_elsewhere_are_their_own_object(self):
        case = OverlapCase(
            claims={CABINET: np.array([0, 1])}, bodies={CABINET: np.array([0])}
        )
        assert list(case.after_groups()[FACES_ELSEWHERE]) == [1]

    def test_an_emptied_claimant_is_not_written_as_an_empty_object(self):
        case = OverlapCase(
            claims={CABINET: np.array([0]), DRAWER: np.array([0])},
            bodies={DRAWER: np.array([0])},
        )
        assert CABINET not in case.after_groups()

    def test_each_object_of_the_scene_carries_its_own_colour(self):
        case = a_cabinet_losing_its_front()
        scene = case_scene(trimesh.creation.box(), case.before_groups(), case.palette)
        assert set(scene.geometry) == set(case.before_groups())
        drawn = scene.geometry[CONTESTED_FACES]
        assert len(drawn.faces) == len(case.contested)
        assert list(drawn.visual.material.baseColorFactor) == list(CONTESTED)

    def test_an_object_carries_its_colour_as_a_material_as_well(self):
        case = a_cabinet_losing_its_front()
        scene = case_scene(trimesh.creation.box(), case.before_groups(), case.palette)
        material = scene.geometry[CABINET].visual.material
        assert list(material.baseColorFactor) == list(case.colour_of(CABINET))

    def test_the_claims_can_be_drawn_pulled_apart_so_they_do_not_coincide(self):
        case = a_cabinet_losing_its_front()
        together = case_scene(
            trimesh.creation.box(), case.before_groups(), case.palette
        )
        apart = case_scene(
            trimesh.creation.box(),
            case.before_groups(),
            case.palette,
            pulled_apart=0.5,
        )
        moved = apart.geometry[DRAWER].bounds[0] - together.geometry[DRAWER].bounds[0]
        assert not np.allclose(moved, 0.0)
