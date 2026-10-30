"""
How big an object is, measured so that two frames can be compared without relating them.

A reconstruction and a modelled world are built in unrelated frames: the same cabinet
stands somewhere else, turned some other way, in each of them. Anything measured along
the axes of one frame therefore says nothing in the other, and an axis-aligned bounding
box is the obvious trap -- a cupboard 0.6 wide and 2.1 tall reads as 2.1 wide once the
frame is turned on its side.

What survives a rigid transform is the shape itself: the sides of the smallest box that
can be turned to fit around the object, and the area of its surface. Both are in metres,
so they compare across the two sources as long as each is metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh
from typing_extensions import List, Optional

from experiments.warsaw.bases import JsonRecord

# %% a size two frames agree on


@dataclass(frozen=True)
class ObjectSize(JsonRecord):
    """
    How big one object is, in terms no frame can change.
    """

    extents: List[float]
    """
    The sides of the smallest box that fits around it, longest first, in metres.
    """

    surface_area: float
    """
    The area of its surface in square metres.
    """

    @classmethod
    def of(cls, mesh: Optional[trimesh.Trimesh]) -> Optional[ObjectSize]:
        """
        Measure a mesh.

        :param mesh: The geometry to measure, or nothing where the object has none.
        :return: Its size, or nothing where there was no geometry to measure.
        """
        if mesh is None or len(mesh.faces) == 0:
            return None
        return cls(
            extents=sorted(
                (float(side) for side in mesh.bounding_box_oriented.primitive.extents),
                reverse=True,
            ),
            surface_area=float(mesh.area),
        )

    def difference_from(self, other: ObjectSize) -> float:
        """
        How unlike two objects are in size, from nought for alike to one for unalike.

        Each side is compared with the matching side of the other box as a ratio, so a
        centimetre matters on a handle and not on a wall, and the worst-matching side
        decides. Sizes are never a reason to rule a pair out on their own, so the result
        is bounded rather than growing without limit.

        :param other: The size to compare against.
        :return: Their difference, between nought and one.
        """
        ratios = [
            min(side, other_side) / max(side, other_side)
            for side, other_side in zip(self.extents, other.extents)
            if max(side, other_side) > 0.0
        ]
        if not ratios:
            return 1.0
        return 1.0 - min(ratios)
