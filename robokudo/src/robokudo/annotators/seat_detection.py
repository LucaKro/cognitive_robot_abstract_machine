"""
Recognition of seats among the detected objects.

Provides the region type for the seats split from larger furniture and the rules that
decide which object hypotheses are seats for a single person.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CAS
from robokudo.types.annotation import Classification
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis, RegionHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt

# %% seat regions


@dataclass
class SeatRegion(RegionHypothesis):
    """
    A region of a piece of furniture on which one person can sit, such as one seat of a
    sofa.
    """


# %% seat detection


@dataclass
class SeatDetection:
    """
    The rules that decide which object hypotheses are seats for a single person.

    A single seat is an object hypothesis classified as one of the seat classes with at
    least :attr:`minimum_seat_confidence` whose point cloud is not wider than
    :attr:`maximum_seat_width` across the camera view.
    """

    seat_class_names: list[str] = field(default_factory=lambda: ["chair"])
    """
    Class names of the object hypotheses that are seats.
    """

    minimum_seat_confidence: float = 0.65
    """
    Confidence below which a seat class detection is not a single seat.
    """

    maximum_seat_width: float = 1.0
    """
    Width in meters across the camera view beyond which a seat holds more than one
    person.
    """

    width_outlier_share: float = 0.05
    """
    Share of the points on each side across the camera view left out when measuring the
    width of a hypothesis.
    """

    def single_seats(self, cas: CAS) -> list[ObjectHypothesis]:
        """
        :param cas: The CAS holding the object hypotheses.
        :return: The object hypotheses in the CAS that are single seats.
        """
        return [
            hypothesis
            for hypothesis in self.objects_classified_as(cas, self.seat_class_names)
            if self.is_single_seat(hypothesis)
        ]

    def is_single_seat(self, hypothesis: ObjectHypothesis) -> bool:
        """
        :param hypothesis: The hypothesis to check.
        :return: Whether the hypothesis is classified as one of the seat classes with
            at least the minimum seat confidence and has points that are not wider
            than a single seat.
        """
        width = self.width_across_view(hypothesis)
        return (
            self.seat_confidence(hypothesis) >= self.minimum_seat_confidence
            and width is not None
            and width <= self.maximum_seat_width
        )

    def seat_confidence(self, hypothesis: ObjectHypothesis) -> float:
        """
        :param hypothesis: The hypothesis to check.
        :return: The highest confidence of the classifications of the hypothesis as
            one of the seat classes, 0 if it has none.
        """
        return max(
            (
                classification.confidence
                for classification in CAS.filter_by_type(
                    Classification, hypothesis.annotations
                )
                if classification.classname in self.seat_class_names
            ),
            default=0.0,
        )

    def objects_classified_as(
        self, cas: CAS, class_names: list[str]
    ) -> list[ObjectHypothesis]:
        """
        :param cas: The CAS holding the object hypotheses.
        :param class_names: The class names to look for.
        :return: The object hypotheses in the CAS classified as one of the given
            classes, humans excluded.
        """
        return [
            hypothesis
            for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis)
            if not isinstance(hypothesis, HumanHypothesis)
            and self.classified_as(hypothesis, class_names)
        ]

    @staticmethod
    def classified_as(hypothesis: ObjectHypothesis, class_names: list[str]) -> bool:
        """
        :param hypothesis: The hypothesis to check.
        :param class_names: The class names to look for.
        :return: Whether the hypothesis is classified as one of the given classes.
        """
        return any(
            classification.classname in class_names
            for classification in CAS.filter_by_type(
                Classification, hypothesis.annotations
            )
        )

    def across_view_bounds(
        self, hypothesis: ObjectHypothesis
    ) -> npt.NDArray[np.float64] | None:
        """
        :param hypothesis: The hypothesis to measure.
        :return: The left and right bound in meters of the point cloud of the
            hypothesis across the camera view, without the outlying share of points on
            each side, or None if it has no points.
        """
        if hypothesis.points is None or not hypothesis.points.has_points():
            return None
        across_view = np.asarray(hypothesis.points.points)[:, 0]
        return np.quantile(
            across_view, [self.width_outlier_share, 1 - self.width_outlier_share]
        )

    def width_across_view(self, hypothesis: ObjectHypothesis) -> float | None:
        """
        :param hypothesis: The hypothesis to measure.
        :return: The width in meters of the hypothesis across the camera view, or None
            if it has no points.
        """
        bounds = self.across_view_bounds(hypothesis)
        if bounds is None:
            return None
        left, right = bounds
        return float(right - left)
