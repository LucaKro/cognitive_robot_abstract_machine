"""
Tests for giving each detected object the semantic type its detector class maps to.
"""

from __future__ import annotations

# %% Imports
from py_trees.common import Status
from robokudo.annotators.gpsr.type_mapping_annotator import TypeMappingAnnotator
from robokudo.types.annotation import Classification, PositionAnnotation
from typing_extensions import TYPE_CHECKING

from ._data_files import GpsrFileName

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene

# %% mapped classes


def test_mapped_type_is_put_first(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    The semantic type of a detected class is placed before that class.
    """
    annotator = TypeMappingAnnotator()
    scene.place_in_pipeline(annotator)
    detected = Classification(classname="mug_highgrip_blue")
    cup = scene.add_object(detected)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert [classification.classname for classification in cup.annotations] == [
        semantic_type_mapping["mug_highgrip_blue"],
        "mug_highgrip_blue",
    ]
    assert cup.annotations[0].source == annotator.name


def test_already_mapped_type_is_not_added_twice(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Already mapped type is not added twice.
    """
    annotator = TypeMappingAnnotator()
    scene.place_in_pipeline(annotator)
    detected = Classification(classname="mug_highgrip_blue")
    mapped = Classification(classname=semantic_type_mapping["mug_highgrip_blue"])
    cup = scene.add_object(detected, mapped)

    annotator.update()

    assert cup.annotations == [detected, mapped]


def test_other_annotations_follow_the_classifications(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Other annotations move behind the classifications.
    """
    annotator = TypeMappingAnnotator()
    scene.place_in_pipeline(annotator)
    position = PositionAnnotation(translation=[1.0, 0.0, 2.0])
    detected = Classification(classname="mug_highgrip_blue")
    cup = scene.add_object(position, detected)

    annotator.update()

    assert cup.annotations[1:] == [detected, position]


# %% unmapped classes


def test_class_without_mapping_is_left_alone(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Class without mapping is left alone.
    """
    annotator = TypeMappingAnnotator()
    scene.place_in_pipeline(annotator)
    detected = Classification(classname="toy_robot")
    toy = scene.add_object(detected)

    annotator.update()

    assert toy.annotations == [detected]


def test_object_without_classification_is_left_alone(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Object without classification is left alone.
    """
    annotator = TypeMappingAnnotator()
    scene.place_in_pipeline(annotator)
    position = PositionAnnotation(translation=[1.0, 0.0, 2.0])
    unknown = scene.add_object(position)

    annotator.update()

    assert unknown.annotations == [position]


def test_only_the_first_classification_is_mapped(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Only the first classification is mapped.
    """
    annotator = TypeMappingAnnotator()
    scene.place_in_pipeline(annotator)
    unmapped = Classification(classname="toy_robot")
    mappable = Classification(classname="mug_highgrip_blue")
    toy = scene.add_object(unmapped, mappable)

    annotator.update()

    assert toy.annotations == [unmapped, mappable]


def test_annotations_outside_of_objects_are_left_alone(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Annotations outside of objects are left alone.
    """
    annotator = TypeMappingAnnotator()
    cas = scene.place_in_pipeline(annotator)
    loose_classification = Classification(classname="mug_highgrip_blue")
    cas.annotations.append(loose_classification)

    annotator.update()

    assert cas.annotations == [loose_classification]


def test_without_mapping_file_nothing_is_mapped(
    scene: GpsrScene, semantic_type_mapping: dict[str, str]
) -> None:
    """
    Without mapping file nothing is mapped.
    """
    annotator = TypeMappingAnnotator(mapping_file_name=GpsrFileName.MISSING_MAPPING)
    scene.place_in_pipeline(annotator)
    detected = Classification(classname="mug_highgrip_blue")
    cup = scene.add_object(detected)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cup.annotations == [detected]
