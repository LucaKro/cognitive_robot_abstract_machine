"""
Leaving a body alone when its class cannot be made from a body, rather than losing the
run -- where the run was told to.

Every annotation is made from the body it was answered about and nothing else. Most
classes need nothing else: a cabinet is made with no drawers and has them mounted into it
afterwards, which is the only order the run could work in, since what belongs in which
cabinet is not known when the cabinet is made. A few are constituted by other annotations
instead -- a room is its floor, a double door is its two doors -- and there is nothing a
body alone can give them. One answer naming such a class used to raise out of the step and
cost the run its world, its report and its evaluation graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from experiments.warsaw.pipeline.provenance import settings_to_json
from experiments.warsaw.pipeline.records import BodyAnswer, Classifications
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.annotate import (
    MountAnnotations,
    fields_beyond_the_body,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Aperture,
    Cabinet,
    Handle,
    Room,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% classes a body is not enough for


@dataclass(eq=False)
class AnnotationAboutNoBody(SemanticAnnotation):
    """
    An annotation taking no body at all, which refuses the one keyword the step passes.
    """


@dataclass(eq=False)
class AnnotationNeedingAnotherAnnotation(HasRootBody):
    """
    An annotation that takes a body and is still not made by one, because it is
    constituted by something the run cannot hand it.
    """

    held: Handle = field(kw_only=True)
    """
    The annotation it cannot exist without.
    """


# %% what a body does and does not supply


def test_a_class_needing_only_a_body_needs_nothing_else():
    """
    A cabinet's drawers are mounted into it after it is made, so they are structure it
    may have rather than something it cannot be made without.
    """
    assert fields_beyond_the_body(Cabinet) == []
    assert fields_beyond_the_body(Handle) == []


def test_a_class_constituted_by_another_annotation_names_what_it_needs():
    """
    Named rather than merely refused, so the run's report says why the body was left.
    """
    assert fields_beyond_the_body(AnnotationNeedingAnotherAnnotation) == ["held"]


def test_a_class_taking_no_body_names_the_body_itself():
    """
    Such a class does not merely need more than the body; it will not take the body.
    """
    assert fields_beyond_the_body(AnnotationAboutNoBody) == ["root"]


def test_a_room_is_short_of_both_a_body_and_its_floor():
    """
    The ontology's own case, and it fails on both counts at once: a room takes no body,
    and it is constituted by a floor.
    """
    assert fields_beyond_the_body(Room) == ["floor", "root"]


def test_a_class_rooted_on_a_region_is_not_made_from_a_body():
    """
    A window is an aperture, and an aperture is a hole rather than a thing: it is rooted
    on a region, of which a body is not one.

    Nothing refuses the body when the annotation is made -- a field takes what it is
    given -- so this stays wrong and quiet until something reads the area a body does
    not have, which is the mount. Caught here instead.
    """
    assert fields_beyond_the_body(Aperture) == ["root"]


# %% the step carrying on


@pytest.fixture
def one_body_world() -> World:
    """
    :return: A world holding a single body under its root.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
    return world


def named(world: World, class_name: str) -> Classifications:
    """
    :param world: The world whose body was named.
    :param class_name: What it was answered to be.
    :return: That answer, as the step reads it.
    """
    return Classifications(
        scene="",
        model="",
        bodies=[
            BodyAnswer(
                name=str(next(iter(world.bodies)).name.name), class_name=class_name
            )
        ],
    )


def told_to_skip(tmp_path: Path, skipping: bool) -> MountAnnotations:
    """
    :param tmp_path: A directory for the run.
    :param skipping: What the run was told about classes a body cannot make.
    :return: The step, reading that back out of the run's own files.
    """
    run = Run(directory=tmp_path)
    run.write_json(
        RunFile.PROVENANCE,
        {
            "settings": settings_to_json(
                PipelineSettings(skip_classes_a_body_cannot_make=skipping)
            )
        },
    )
    return MountAnnotations(directory=tmp_path)


def test_the_scans_are_left_as_they_were(one_body_world, tmp_path: Path):
    """
    Skipping is not the default. A scan answers as it always has, so its numbers stay
    comparable with the ones already reported, and an answer naming a class a body
    cannot make still raises rather than quietly costing a body.
    """
    assert PipelineSettings().skip_classes_a_body_cannot_make is False
    with pytest.raises(TypeError):
        told_to_skip(tmp_path, skipping=False).annotate(
            one_body_world, named(one_body_world, Room.__name__)
        )


def test_a_body_named_as_a_room_is_left_alone_when_the_run_was_told_to(
    one_body_world, tmp_path: Path
):
    """
    Told to skip, the answer costs that one body rather than every body after it.
    """
    annotated = told_to_skip(tmp_path, skipping=True).annotate(
        one_body_world, named(one_body_world, Room.__name__)
    )
    assert annotated == {}


def test_a_body_named_as_something_a_body_makes_is_still_annotated(
    one_body_world, tmp_path: Path
):
    """
    Skipping refuses what cannot be made and nothing else, so everything the step
    annotated before it is annotated still.
    """
    annotated = told_to_skip(tmp_path, skipping=True).annotate(
        one_body_world, named(one_body_world, Handle.__name__)
    )
    assert [type(one).__name__ for one in annotated.values()] == [Handle.__name__]


def test_a_run_that_says_nothing_does_not_skip(one_body_world, tmp_path: Path):
    """
    A run written before there was anything to say reads as a run that was not told to
    skip, rather than as one that was.
    """
    assert MountAnnotations(directory=tmp_path).skip_classes_a_body_cannot_make is False
