"""
Seats, sofas and humans in the camera frame, and running an annotator on them.
"""

# %% imports

from __future__ import annotations

import numpy as np
import open3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.sitting import HumanPosture
from robokudo.annotators.seat_detection import SeatDetection, SeatRegion
from robokudo.annotators.sofa_split import SofaSplitAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification, PositionAnnotation
from robokudo.types.core import Annotation
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import (
    AnalyzableAnnotation,
    HumanHypothesis,
    ObjectHypothesis,
    OccupiedAnnotation,
)
from .._marked_image import MarkedRegionImage

# %% scene setup


def located(
    hypothesis: AnalyzableAnnotation, x: float, z: float
) -> AnalyzableAnnotation:
    """
    Give a hypothesis a position in the camera frame, at the height of the camera.

    :return: The given hypothesis.
    """
    hypothesis.annotations.append(PositionAnnotation(translation=[x, 0.0, z]))
    return hypothesis


def cloud_across_view(
    *across_view: float, depth: float = 0.0
) -> open3d.geometry.PointCloud:
    """
    :param across_view: Positions in meters of the points across the camera view.
    :param depth: Distance in meters of all points from the camera.
    :return: A point cloud with one point at each of the given positions.
    """
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector([[x, 0.0, depth] for x in across_view])
    return cloud


def seat(
    class_name: str | None = None,
    width: float | None = None,
    depth: float = 0.0,
    confidence: float = 1.0,
) -> ObjectHypothesis:
    """
    :param class_name: The class of the hypothesis, the first seat class if None.
    :param width: Width in meters of the point cloud of the hypothesis across the
        camera view, centered in it, half the maximum seat width if None.
    :param depth: Distance in meters of the point cloud from the camera.
    :param confidence: Confidence of the classification of the hypothesis.
    :return: An object hypothesis classified as the given class, in the region of
        :class:`MarkedRegionImage`.
    """
    seat_detection = SeatDetection()
    if class_name is None:
        class_name = seat_detection.seat_class_names[0]
    if width is None:
        width = seat_detection.maximum_seat_width / 2
    hypothesis = ObjectHypothesis(
        points=cloud_across_view(*np.linspace(-width / 2, width / 2, 101), depth=depth)
    )
    hypothesis.roi.roi = MarkedRegionImage().region
    hypothesis.annotations.append(
        Classification(classname=class_name, confidence=confidence)
    )
    return hypothesis


def sofa(width: float, depth: float) -> ObjectHypothesis:
    """
    :param width: Width in meters of the sofa across the camera view, centered in it.
    :param depth: Distance in meters of the sofa from the camera.
    :return: An object hypothesis of the first sofa class.
    """
    sofa_class_name = SofaSplitAnnotator.Descriptor.Parameters().sofa_class_names[0]
    return seat(sofa_class_name, width=width, depth=depth)


def seat_region(x: float, z: float) -> SeatRegion:
    """
    :return: A seat region at the given position, in the region of
        :class:`MarkedRegionImage`.
    """
    region = SeatRegion()
    region.roi.roi = MarkedRegionImage().region
    located(region, x=x, z=z)
    return region


def human(*activities: HumanPosture) -> HumanHypothesis:
    """
    :param activities: The activities the human is annotated with; none for a human
        whose posture is unclear.
    :return: A human hypothesis annotated with the given activities.
    """
    hypothesis = HumanHypothesis()
    hypothesis.annotations.extend(
        ActivityAnnotation(type=activity) for activity in activities
    )
    return hypothesis


# %% annotation


def annotate(annotator: BaseAnnotator, *annotations: Annotation) -> BaseAnnotator:
    """
    Run an annotator on a CAS holding the given annotations and the color image of
    :class:`MarkedRegionImage`.

    :return: The given annotator after its update.
    """
    annotate_in_order([annotator], *annotations)
    return annotator


def annotate_in_order(
    annotators: list[BaseAnnotator], *annotations: Annotation
) -> None:
    """
    Run annotators one after the other on a shared CAS holding the given annotations and
    the color image of :class:`MarkedRegionImage`.
    """
    pipeline = Pipeline()
    pipeline.add_children(annotators)
    pipeline.setup()
    pipeline.cas.set(CASViews.COLOR_IMAGE, MarkedRegionImage().color_image())
    pipeline.cas.annotations.extend(annotations)
    for annotator in annotators:
        assert annotator.update() == Status.SUCCESS


def occupancies(hypothesis: AnalyzableAnnotation) -> list[bool]:
    """
    :return: The occupancy values the hypothesis is annotated with.
    """
    return [
        annotation.occupied
        for annotation in CAS.filter_by_type(OccupiedAnnotation, hypothesis.annotations)
    ]


def created_seat_regions(annotator: BaseAnnotator) -> list[SeatRegion]:
    """
    :return: The seat regions in the CAS of the annotator, in the order they were added.
    """
    return annotator.get_cas().filter_annotations_by_type(SeatRegion)
