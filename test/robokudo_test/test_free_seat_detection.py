"""
Tests for the pipeline detecting free seats.
"""

# %% imports

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.query import QueryAnnotator
from robokudo.annotators.seat_occupancy import (
    SeatOccupancyAnnotator,
    SeatQueryAnswerAnnotator,
)
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.descriptors.analysis_engines.free_seat_detection import AnalysisEngine

if TYPE_CHECKING:
    from rclpy.node import Node

# %% human detection


def test_humans_are_only_detected_with_body_keypoints(node: Node) -> None:
    """
    The object detection leaves persons out, so every human hypothesis comes from the
    human detection and carries the body keypoints its posture is judged on.
    """
    pipeline = AnalysisEngine().implementation()

    (object_detection,) = [
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, YoloAnnotator)
    ]
    parameters = object_detection.descriptor.parameters
    assert parameters.person_class_name in parameters.ignored_class_names


# %% posture


def test_knee_at_hip_height_counts_as_sitting(node: Node) -> None:
    """
    The sitting rule of the pipeline also decides persons with hidden ankles, whose
    knees are at hip height, as sitting.
    """
    pipeline = AnalysisEngine().implementation()

    (sitting_rule,) = [
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, HumanSittingAnnotator)
    ]
    assert sitting_rule.descriptor.parameters.knee_at_hip_height_is_sitting


# %% query answering


def test_seats_are_annotated_between_query_and_answer(node: Node) -> None:
    """
    The pipeline waits for a query before reading a camera image and answers only after
    the seats are annotated with their occupancy.
    """
    pipeline = AnalysisEngine().implementation()

    behaviour_types = [type(behaviour) for behaviour in pipeline.children]
    assert behaviour_types.index(QueryAnnotator) < behaviour_types.index(
        CollectionReaderAnnotator
    )
    assert behaviour_types[-3:] == [
        SeatOccupancyAnnotator,
        SeatQueryAnswerAnnotator,
        ActionServerCheck,
    ]
