"""
Tests for keeping only the objects of the semantic type a query asks for.
"""

from __future__ import annotations

# %% Imports
from py_trees.common import Status
from robokudo.annotators.gpsr.filter_object_by_semantic_annotation import (
    FilterObjectBySemanticAnnotation,
)
from robokudo.types.annotation import Classification
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene

# %% queried semantic type


def test_only_objects_of_the_queried_type_are_kept(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Only objects of the queried type are kept.
    """
    mug_type = semantic_type_mapping["mug_highgrip_blue"]
    bowl_type = semantic_type_mapping["bowl_white"]
    scene.query.obj.type = mug_type
    annotator = FilterObjectBySemanticAnnotation()
    cas = scene.place_in_pipeline(annotator)
    mug = scene.add_object(Classification(classname=mug_type))
    scene.add_object(Classification(classname=bowl_type))

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == [mug]


def test_queried_type_is_read_from_the_description_of_an_object_query(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Queried type is read from the description of an object query.
    """
    mug_type = semantic_type_mapping["mug_highgrip_blue"]
    scene.query.obj.type = "object"
    scene.query.obj.description = [mug_type]
    annotator = FilterObjectBySemanticAnnotation()
    cas = scene.place_in_pipeline(annotator)
    mug = scene.add_object(Classification(classname=mug_type))
    scene.add_object(Classification(classname=semantic_type_mapping["bowl_white"]))

    annotator.update()

    assert cas.annotations == [mug]


def test_object_whose_later_classification_matches_is_kept(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Object whose later classification matches is kept.
    """
    mug_type = semantic_type_mapping["mug_highgrip_blue"]
    scene.query.obj.type = mug_type
    annotator = FilterObjectBySemanticAnnotation()
    cas = scene.place_in_pipeline(annotator)
    mug = scene.add_object(
        Classification(classname="mug_highgrip_blue"),
        Classification(classname=mug_type),
    )

    annotator.update()

    assert cas.annotations == [mug]


def test_annotations_other_than_matching_objects_are_removed(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Annotations other than matching objects are removed.
    """
    mug_type = semantic_type_mapping["mug_highgrip_blue"]
    scene.query.obj.type = mug_type
    annotator = FilterObjectBySemanticAnnotation()
    cas = scene.place_in_pipeline(annotator)
    mug = scene.add_object(Classification(classname=mug_type))
    cas.annotations.append(Classification(classname=mug_type))

    annotator.update()

    assert cas.annotations == [mug]


# %% nothing to filter by


def test_unknown_type_keeps_every_object(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Unknown type keeps every object.
    """
    scene.query.obj.type = "object"
    scene.query.obj.description = ["spaceship"]
    annotator = FilterObjectBySemanticAnnotation()
    cas = scene.place_in_pipeline(annotator)
    mug = scene.add_object(
        Classification(classname=semantic_type_mapping["mug_highgrip_blue"])
    )

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == [mug]


def test_object_query_without_description_keeps_every_object(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Object query without description keeps every object.
    """
    scene.query.obj.type = "object"
    annotator = FilterObjectBySemanticAnnotation()
    cas = scene.place_in_pipeline(annotator)
    mug = scene.add_object(
        Classification(classname=semantic_type_mapping["mug_highgrip_blue"])
    )

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == [mug]


# %% no object of the queried type


def test_no_object_of_the_queried_type_leaves_an_empty_scene(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    No object of the queried type leaves an empty scene.
    """
    scene.query.obj.type = semantic_type_mapping["mug_highgrip_blue"]
    annotator = FilterObjectBySemanticAnnotation()
    cas = scene.place_in_pipeline(annotator)
    scene.add_object(Classification(classname=semantic_type_mapping["bowl_white"]))

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == []
