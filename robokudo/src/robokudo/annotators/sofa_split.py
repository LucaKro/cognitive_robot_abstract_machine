"""
Splitting of detected sofas into seats for a single person.
"""

# %% imports

from __future__ import annotations

import numpy as np
from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.seat_detection import SeatDetection, SeatRegion
from robokudo.cas import CASViews
from robokudo.types.annotation import PositionAnnotation
from robokudo.types.cv import Rect
from robokudo.vis.box_marking import BoxMarking

if TYPE_CHECKING:
    from robokudo.types.scene import ObjectHypothesis

# %% annotator


class SofaSplitAnnotator(BaseAnnotator):
    """
    Splits every detected sofa into seats and adds them to the CAS as
    :class:`~robokudo.annotators.seat_detection.SeatRegion`.

    A sofa is an object hypothesis with points classified as one of the sofa classes,
    or as one of the seat classes but wider than a single seat. Its point cloud is split
    across the camera view into :attr:`Descriptor.Parameters.seats_per_sofa` equally
    wide seats. The visualization image frames every seat region.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Descriptor for the SofaSplitAnnotator.
        """

        class Parameters:
            """
            Parameters for the SofaSplitAnnotator.
            """

            def __init__(self) -> None:
                """
                Set every parameter to its default.
                """
                self.seat_detection: SeatDetection = SeatDetection()
                """
                The rules that decide which object hypotheses are single seats.
                """
                self.sofa_class_names: list[str] = ["couch"]
                """
                Class names of the object hypotheses that are sofas.
                """
                self.seats_per_sofa: int = 2
                """
                Number of seats every sofa is split into.
                """

    @classproperty
    def seat_marking(cls) -> BoxMarking:
        """
        How a sofa seat is marked in the visualization image.
        """
        return BoxMarking(label="sofa seat", color=(0, 200, 200))

    def __init__(
        self,
        name: str = "SofaSplitAnnotator",
        descriptor: SofaSplitAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of the annotator in the pipeline.
        :param descriptor: Configuration of the annotator, the default one if None.
        """
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Add the seats of every sofa to the CAS.

        :return: SUCCESS after processing.
        """
        seat_regions = [
            seat_region for sofa in self.sofas() for seat_region in self.split(sofa)
        ]
        self.get_cas().annotations.extend(seat_regions)

        if self.descriptor.parameters.global_with_visualization:
            self._visualize(seat_regions)
        self.feedback_message = f"{len(seat_regions)} sofa seats"
        return Status.SUCCESS

    def _visualize(self, seat_regions: list[SeatRegion]) -> None:
        """
        Frame every seat region in a copy of the color image and output the image.

        :param seat_regions: The seat regions added in this update.
        """
        image = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        for seat_region in seat_regions:
            self.seat_marking.draw(image, seat_region.roi.roi)
        self.get_annotator_output_struct().set_image(image)

    def sofas(self) -> list[ObjectHypothesis]:
        """
        :return: The object hypotheses with points classified as one of the sofa
            classes, or as one of the seat classes but wider than a single seat.
        """
        parameters: SofaSplitAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        seat_detection = parameters.seat_detection
        sofas = []
        for hypothesis in seat_detection.objects_classified_as(
            self.get_cas(),
            parameters.sofa_class_names + seat_detection.seat_class_names,
        ):
            width = seat_detection.width_across_view(hypothesis)
            if width is None:
                continue
            if (
                seat_detection.classified_as(hypothesis, parameters.sofa_class_names)
                or width > seat_detection.maximum_seat_width
            ):
                sofas.append(hypothesis)
        return sofas

    def split(self, sofa: ObjectHypothesis) -> list[SeatRegion]:
        """
        Split a sofa across the camera view into equally wide seat regions.

        Each seat region is positioned at the middle of its part across the view, at the
        median height and depth of the sofa points. Its image region is the matching
        part of the image region of the sofa.

        :param sofa: The sofa to split, with points.
        :return: The seat regions of the sofa, from left to right.
        """
        parameters: SofaSplitAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        seats_per_sofa = parameters.seats_per_sofa
        points = np.asarray(sofa.points.points)
        height, depth = np.median(points[:, 1]), np.median(points[:, 2])
        left, right = parameters.seat_detection.across_view_bounds(sofa)
        seat_bounds = np.linspace(left, right, seats_per_sofa + 1)
        image_left, top, image_right, bottom = sofa.roi.roi.get_corner_points()
        image_bounds = np.linspace(image_left, image_right, seats_per_sofa + 1)

        seat_regions = []
        for index in range(seats_per_sofa):
            seat_region = SeatRegion(source=self.name)
            seat_region.roi.roi = Rect.from_corner_points(
                image_bounds[index], top, image_bounds[index + 1], bottom
            )
            seat_region.annotations.append(
                PositionAnnotation(
                    source=self.name,
                    translation=[
                        float((seat_bounds[index] + seat_bounds[index + 1]) / 2),
                        float(height),
                        float(depth),
                    ],
                )
            )
            seat_regions.append(seat_region)
        return seat_regions
