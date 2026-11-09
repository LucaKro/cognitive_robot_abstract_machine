"""
Tests for marking image regions with labeled, colored frames.
"""

# %% imports

from __future__ import annotations

import numpy as np

from robokudo.vis.box_marking import BoxMarking
from ._marked_image import MarkedRegionImage

# %% drawing


def test_frame_is_drawn_in_marking_color() -> None:
    """
    The frame around the region is drawn in the color of the marking.
    """
    scene = MarkedRegionImage()
    marking = BoxMarking(label="seat", color=(10, 20, 30))
    image = scene.color_image()

    marking.draw(image, scene.region)

    assert scene.frame_color(image) == marking.color


def test_label_is_drawn_above_frame() -> None:
    """
    The label is drawn in the color of the marking above the region.
    """
    scene = MarkedRegionImage()
    marking = BoxMarking(label="seat", color=(10, 20, 30))
    image = scene.color_image()

    marking.draw(image, scene.region)

    left, top, right, _ = scene.region.get_corner_points()
    above_frame = image[: top - marking.line_thickness, left:right]
    assert np.any(np.all(above_frame == marking.color, axis=-1))


def test_inside_of_region_stays_untouched() -> None:
    """
    Drawing leaves the inside of the region unchanged.
    """
    scene = MarkedRegionImage()
    marking = BoxMarking(label="seat", color=(10, 20, 30))
    image = scene.color_image()

    marking.draw(image, scene.region)

    left, top, right, bottom = scene.region.get_corner_points()
    border = 2 * marking.line_thickness
    inside = image[top + border : bottom - border, left + border : right - border]
    np.testing.assert_array_equal(inside, np.zeros_like(inside))
