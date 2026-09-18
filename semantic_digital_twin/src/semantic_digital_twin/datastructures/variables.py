from __future__ import annotations

from enum import Enum
from functools import cached_property

from krrood.ormatic.utils import classproperty
from random_events.variable import Continuous
from sortedcontainers import SortedSet
from typing_extensions import Tuple


class SpatialVariables(Enum):
    """
    Enum for spatial variables used in the semantic digital twin.

    Used in the context of random events.
    """

    x = Continuous(name="x")
    y = Continuous(name="y")
    z = Continuous(name="z")
    roll = Continuous(name="roll")
    pitch = Continuous(name="pitch")
    yaw = Continuous(name="yaw")

    @classproperty
    def xy(cls):
        return SortedSet([cls.x.value, cls.y.value])

    @classproperty
    def xz(cls):
        return SortedSet([cls.x.value, cls.z.value])

    @classproperty
    def yz(cls):
        return SortedSet([cls.y.value, cls.z.value])

    @classproperty
    def position(cls) -> Tuple[Continuous, ...]:
        """
        :return: Where something is, along each axis.
        """
        return cls.x.value, cls.y.value, cls.z.value

    @classproperty
    def rotation(cls) -> Tuple[Continuous, ...]:
        """
        :return: How something is turned, about each axis.
        """
        return cls.roll.value, cls.pitch.value, cls.yaw.value

    @classproperty
    def pose(cls) -> Tuple[Continuous, ...]:
        """
        :return: The six degrees of freedom of a pose, in the order an array over them
            is laid out.

        ..note:: Ordered rather than a :class:`SortedSet`, because the position of a
            variable here is the row and column it occupies in a matrix over them.
        """
        return cls.position + cls.rotation
