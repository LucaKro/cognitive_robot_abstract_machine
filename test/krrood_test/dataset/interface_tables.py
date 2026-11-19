"""
Classes that implement an interface no alternative mapping covers, some on their own
and some next to a parent chain of their own, and classes whose fields are typed with
that interface.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Generic, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

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


# %% a generic interface


TagT = TypeVar("TagT")
"""
The type of the tag a tagged frame carries.
"""


class TaggedFrame(Generic[TagT], SubClassSafeGeneric, Interface):
    """
    A frame that carries a tag, whose type each implementation chooses.

    .. note:: Its generic base makes :func:`dataclasses.is_dataclass` true for it,
        although it is no dataclass.
    """

    @abstractmethod
    def tag(self) -> TagT:
        """
        :return: The tag of the frame.
        """


@dataclass(kw_only=True)
class StationFrame(TaggedFrame[str]):
    """
    A frame at a station, tagged with the station's code.
    """

    station_code: str
    """
    The code of the station.
    """

    def tag(self) -> str:
        return self.station_code


@dataclass(kw_only=True)
class TaggedRoute:
    """
    A route that starts at a tagged frame.
    """

    start: TaggedFrame
    """
    The frame the route starts at.
    """
