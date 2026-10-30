"""
Pipeline steps that cut the shape of a body into bodies of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import BodyWithoutShapeError
from semantic_digital_twin.part_finding.splitting import BodyCutter, ShapePiece
from semantic_digital_twin.pipeline.pipeline import Step

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% splitting a body into pieces


@dataclass
class SplitBodyIntoPieces(Step):
    """
    Cuts sets of faces off the shape of one body into bodies of their own, each fixed to
    it; the body keeps the faces no piece takes.

    Splits a scanned or reconstructed room, one mesh, into its labelled objects or
    parts.
    """

    body_name: PrefixedName
    """
    The name of the body whose shape is cut.
    """

    pieces: list[ShapePiece]
    """
    The faces to cut off, each with the body they are cut into.
    """

    cutter: BodyCutter = field(default_factory=BodyCutter)
    """
    Cuts the shape and gives the pieces their collision shapes.
    """

    def _apply(self, world: World) -> World:
        body = world.get_body_by_name(self.body_name)
        colored = self.cutter.shape_to_cut(body)
        if colored is None:
            raise BodyWithoutShapeError(body_name=body.name)
        self.cutter.cut(body, colored, self.pieces).apply()
        return world
