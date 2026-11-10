"""
Labeled, colored frames marking regions of an image.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass

import cv2
from krrood.ormatic.utils import classproperty
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from robokudo.types.cv import Rect

# %% box marking


@dataclass(frozen=True)
class BoxMarking:
    """
    A frame in one color with a label above it, marking a region of an image.
    """

    label: str
    """
    Text written above the frame.
    """

    color: tuple[int, int, int]
    """
    Color of the frame and the label, in OpenCV's blue, green, red order.
    """

    @classproperty
    def line_thickness(cls) -> int:
        """
        Thickness in pixels of the frame and the label strokes.
        """
        return 2

    @classproperty
    def font_scale(cls) -> float:
        """
        Scale of the label font relative to its base size.
        """
        return 0.5

    @classproperty
    def label_gap(cls) -> int:
        """
        Distance in pixels between the label and the top of the frame.
        """
        return 5

    def draw(self, image: npt.NDArray[np.uint8], box: Rect) -> None:
        """
        Draw this marking around a box into an image.

        :param image: The image to draw into, changed in place.
        :param box: The region to mark, in image pixels.
        """
        left, top, right, bottom = box.get_corner_points()
        cv2.rectangle(
            image, (left, top), (right, bottom), self.color, self.line_thickness
        )
        cv2.putText(
            image,
            self.label,
            (left, top - self.label_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.color,
            self.line_thickness,
        )
