"""
Tests for keeping only the persons who hold the posture and make the gesture a query
asks for.
"""

from __future__ import annotations

# %% Imports
from py_trees.common import Status
from robokudo.annotators.gpsr.filter_human_by_activity_annotator import (
    FilterHumanByActivityAnnotator,
)
from robokudo.annotators.gpsr.human_activity_annotator import (
    HumanActivityAnnotator,
    HumanGesture,
)
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.types.annotation import Classification
from robokudo.types.human import ActivityAnnotation
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene

# %% posture and gesture filters


def test_without_filter_every_person_is_kept(scene: GpsrScene) -> None:
    """
    Without filter every person is kept.
    """
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    standing = scene.add_human(ActivityAnnotation(type="Standing"))
    sitting = scene.add_human(ActivityAnnotation(type="Sitting"))

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == [standing, sitting]


def test_posture_filter_ignores_letter_case(scene: GpsrScene) -> None:
    """
    Posture filter ignores letter case.
    """
    scene.query.obj.attribute = ["sitting"]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    scene.add_human(ActivityAnnotation(type="Standing"))
    sitting = scene.add_human(ActivityAnnotation(type="Sitting"))

    annotator.update()

    assert cas.annotations == [sitting]


def test_gesture_filter_alone_keeps_persons_making_the_gesture(
    scene: GpsrScene,
) -> None:
    """
    Gesture filter alone keeps persons making the gesture.
    """
    scene.query.obj.attribute = ["", "waving"]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    waving = scene.add_human(
        ActivityAnnotation(type="Standing"),
        ActivityAnnotation(type=HumanGesture.WAVING),
    )
    scene.add_human(
        ActivityAnnotation(type="Standing"), ActivityAnnotation(type=HumanGesture.NONE)
    )

    annotator.update()

    assert cas.annotations == [waving]


def test_both_filters_keep_persons_matching_both(scene: GpsrScene) -> None:
    """
    Both filters keep persons matching both.
    """
    scene.query.obj.attribute = ["sitting", "waving"]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    sitting_and_waving = scene.add_human(
        ActivityAnnotation(type="Sitting"), ActivityAnnotation(type=HumanGesture.WAVING)
    )
    scene.add_human(
        ActivityAnnotation(type="Standing"),
        ActivityAnnotation(type=HumanGesture.WAVING),
    )
    scene.add_human(
        ActivityAnnotation(type="Sitting"), ActivityAnnotation(type=HumanGesture.NONE)
    )

    annotator.update()

    assert cas.annotations == [sitting_and_waving]


def test_filters_ignore_surrounding_spaces(scene: GpsrScene) -> None:
    """
    Filters ignore surrounding spaces.
    """
    scene.query.obj.attribute = [" sitting "]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    sitting = scene.add_human(ActivityAnnotation(type="Sitting"))

    annotator.update()

    assert cas.annotations == [sitting]


def clip_lying_label() -> str:
    """
    :return: The posture label CLIP reports for a lying person.
    """
    return next(
        label
        for label in HumanActivityAnnotator.Descriptor.Parameters().posture_labels
        if label.lower() == HumanLyingAnnotator.posture
    )


def test_laying_query_keeps_a_person_clip_sees_lying(scene: GpsrScene) -> None:
    """
    Laying query keeps a person CLIP sees lying.
    """
    scene.query.obj.attribute = [HumanLyingAnnotator.query_word]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    lying = scene.add_human(ActivityAnnotation(type=clip_lying_label()))
    scene.add_human(ActivityAnnotation(type="Standing"))

    annotator.update()

    assert cas.annotations == [lying]


def test_laying_query_keeps_a_person_the_lying_rule_decided(scene: GpsrScene) -> None:
    """
    Laying query keeps a person the lying rule decided.
    """
    scene.query.obj.attribute = [HumanLyingAnnotator.query_word]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    lying = scene.add_human(
        ActivityAnnotation(
            type=HumanLyingAnnotator.posture, source=HumanLyingAnnotator.__name__
        )
    )

    annotator.update()

    assert cas.annotations == [lying]


# %% nobody matches


def test_nobody_matching_leaves_no_person(scene: GpsrScene) -> None:
    """
    Nobody matching leaves no person.
    """
    scene.query.obj.attribute = ["lying"]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    scene.add_human(ActivityAnnotation(type="Standing"))

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == []


def test_annotations_other_than_persons_are_removed(scene: GpsrScene) -> None:
    """
    Annotations other than persons are removed.
    """
    scene.query.obj.attribute = ["sitting"]
    annotator = FilterHumanByActivityAnnotator()
    cas = scene.place_in_pipeline(annotator)
    sitting = scene.add_human(ActivityAnnotation(type="Sitting"))
    scene.add_object(Classification(classname="chair"))

    annotator.update()

    assert cas.annotations == [sitting]
