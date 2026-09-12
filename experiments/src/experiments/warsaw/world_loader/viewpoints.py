"""
Where a scene is looked at from, and how large the pictures are drawn.

None of this knows what a scene is. A viewpoint is a corner to stand in and an angle to
look from, and a size is a size, so both are settled before anything is loaded and both
are read by the settings a run is given.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image
from typing_extensions import Optional, Tuple

# %% where to stand and where to look


class Viewpoint(StrEnum):
    """
    One of the four directions the predefined cameras look at the scene from.

    They stand a quarter turn apart, offset so that none of them looks straight down an
    axis of the room and every one sees two of its walls.
    """

    def __new__(cls, name: str, azimuth_degrees: float) -> Viewpoint:
        member = str.__new__(cls, name)
        member._value_ = name
        member.azimuth_degrees = azimuth_degrees
        return member

    FRONT_LEFT = ("front_left", 45.0)
    """
    From the front left corner.
    """

    BACK_LEFT = ("back_left", 135.0)
    """
    From the back left corner.
    """

    BACK_RIGHT = ("back_right", 225.0)
    """
    From the back right corner.
    """

    FRONT_RIGHT = ("front_right", 315.0)
    """
    From the front right corner.
    """

    @property
    def azimuth(self) -> float:
        """
        :return: The direction it looks at the scene from, in radians.
        """
        return np.radians(self.azimuth_degrees)


class ViewpointChoice(StrEnum):
    """
    How the one viewpoint a render is kept from is picked.

    Every viewpoint is drawn to decide between them, so choosing costs the renders it
    then discards.
    """

    ALONE = "alone"
    """
    By what is visible of the segments on their own, which takes seconds and does not
    count what stands in front of them.
    """

    IN_ROOM = "in-room"
    """
    By what is visible of them in the scene around them, which counts what stands in
    front of them and takes minutes per region.
    """

    ALL = "all"
    """
    Choose nothing and keep every viewpoint.
    """


# %% what a render shows


class PictureKind(StrEnum):
    """
    What one of a question's renders shows.

    A render is written as ``<subject>__<kind>_<viewpoint>.png``, so the kind is read
    back out of the filename when the question is put to a model.
    """

    CONTEXT = "context"
    """
    Where in the room the subject is.
    """

    PLAIN = "plain"
    """
    The subject alone, in the colors it was scanned in.
    """

    CLOSEUP = "closeup"
    """
    The subject alone, painted, which is exactly the faces in question.
    """

    @classmethod
    def of_render(cls, filename: str) -> Optional[PictureKind]:
        """
        :param filename: A render's name.
        :return: What it shows, or None if its name does not say.
        """
        tail = filename.rsplit("__", 1)[-1].split("_", 1)[0]
        return cls(tail) if tail in set(cls) else None


# %% how large to draw


@dataclass
class RenderSizes:
    """
    How large a scene's renders are drawn.

    The two always travel together: what a picture is kept at, and what it is drawn at
    when it is made only to be compared against another and then thrown away.
    """

    kept: Tuple[int, int] = (1024, 768)
    """
    The size of the renders that are written out.
    """

    deciding: Optional[Tuple[int, int]] = None
    """
    The size of a render made only to choose between viewpoints, if not the kept size.

    Which viewpoint shows more of something is a question about proportions, and
    proportions survive being asked small -- but only asking it small can answer it
    differently, and the renders turned out not to be the cost they looked like, so it
    is not done unless asked for.
    """


# %% a render with nothing in it


def is_one_color(render: bytes) -> bool:
    """
    Say whether a render came back a single flat color.

    :param render: A render, as PNG bytes.
    :return: Whether every one of its pixels is the same color.
    """
    pixels = np.asarray(Image.open(io.BytesIO(render)).convert("RGB"))
    return len(np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0)) == 1


# %% telling two renders apart


def changed_pixels(one: bytes, other: bytes) -> int:
    """
    Count how many pixels two renders of the same pose differ in.

    Comparing a render against the same view painted in the scene's own colors counts
    exactly the pixels the highlight is responsible for, which is what "how much of this
    is visible from here" means. Matching the highlight's color instead would need a
    tolerance, since a renderer shades one color across a range of them.

    :param one: One render, as PNG bytes.
    :param other: The other render of the same pose.
    :return: How many pixels differ.
    """
    first = np.asarray(Image.open(io.BytesIO(one)).convert("RGB"))
    second = np.asarray(Image.open(io.BytesIO(other)).convert("RGB"))
    if first.shape != second.shape:
        return 0
    return int((first != second).any(axis=-1).sum())
