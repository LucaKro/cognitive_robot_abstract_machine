"""
A blank color image with one region, for checking how annotators mark that region.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from robokudo.types.cv import Rect

# %% marked image


@dataclass
class MarkedRegionImage:
    """
    A black color image and the region in it an annotator is expected to mark.
    """

    height: int = 480
    """
    Height of the image in pixels.
    """

    width: int = 640
    """
    Width of the image in pixels.
    """

    region: Rect = field(
        default_factory=lambda: Rect.from_corner_points(100, 50, 400, 450)
    )
    """
    The region to be marked.
    """

    def color_image(self) -> npt.NDArray[np.uint8]:
        """
        :return: A black color image.
        """
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def frame_color(self, image: npt.NDArray[np.uint8]) -> tuple[int, int, int]:
        """
        :param image: An image in which the region may be marked.
        :return: The color of the image on the left edge of the region.
        """
        left, top, _, bottom = self.region.get_corner_points()
        blue, green, red = image[(top + bottom) // 2, left]
        return int(blue), int(green), int(red)
