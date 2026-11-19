"""
Classes that implement an interface no alternative mapping covers, some on their own
and some next to a parent chain of their own, and classes whose fields are typed with
that interface.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

from krrood.patterns.interface import Interface

# %% the interface


class ReferenceFrame(Interface):
    """
    Something that offsets can be measured from.
    """

    @abstractmethod
    def describe_origin(self) -> str:
        """
        :return: A sentence naming where offsets measured from this frame start.
        """


# %% an implementation without stored ancestors


@dataclass(kw_only=True)
class LandmarkFrame(ReferenceFrame):
    """
    A frame fixed at a named landmark.
    """

    landmark_name: str
    """
    The name of the landmark the frame sits at.
    """

    def describe_origin(self) -> str:
        return f"at {self.landmark_name}"


# %% implementations with a parent chain of their own


@dataclass(kw_only=True)
class TrackedPart:
    """
    A part of a machine whose wear is tracked.
    """

    part_number: str
    """
    The number the manufacturer gives the part.
    """


@dataclass(kw_only=True)
class FramedPart(TrackedPart, ReferenceFrame):
    """
    A tracked part that offsets can also be measured from.
    """

    def describe_origin(self) -> str:
        return f"at part {self.part_number}"


@dataclass(kw_only=True)
class FramedJoint(FramedPart):
    """
    A framed part that connects two other parts.
    """

    stiffness: float
    """
    How strongly the joint resists bending, in newton metres per radian.
    """


# %% classes whose fields are typed with the interface


@dataclass(kw_only=True)
class FramedOffset:
    """
    An offset measured from a frame, or from nowhere in particular.
    """

    distance: float
    """
    The length of the offset, in metres.
    """

    reference_frame: ReferenceFrame | None = None
    """
    The frame the offset is measured from, or *None* where it is measured from no frame.
    """


@dataclass(kw_only=True)
class FrameChain:
    """
    Frames that are passed through one after another.
    """

    frames: list[ReferenceFrame] = field(default_factory=list)
    """
    The frames, in the order they are passed through.
    """
