from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from collections.abc import Sequence

from typing_extensions import Self

from giskardpy.motion_statechart.exceptions import CovarianceNotSixBySixError

# %% the axes a pose covariance is expressed over


class PoseAxis(IntEnum):
    """
    The degrees of freedom a pose covariance relates, in the order the ROS pose
    covariance stores them.

    The members double as indices into :attr:`PoseCovariance.values`.
    """

    POSITION_X = 0
    """
    Translation along the x axis.
    """

    POSITION_Y = 1
    """
    Translation along the y axis.
    """

    POSITION_Z = 2
    """
    Translation along the z axis.
    """

    ROTATION_X = 3
    """
    Rotation about the x axis.
    """

    ROTATION_Y = 4
    """
    Rotation about the y axis.
    """

    ROTATION_Z = 5
    """
    Rotation about the z axis.
    """

    @classmethod
    def position_axes(cls) -> tuple[Self, ...]:
        """
        :return: The three translational axes.
        """
        return cls.POSITION_X, cls.POSITION_Y, cls.POSITION_Z

    @classmethod
    def rotation_axes(cls) -> tuple[Self, ...]:
        """
        :return: The three rotational axes.
        """
        return cls.ROTATION_X, cls.ROTATION_Y, cls.ROTATION_Z


# %% the covariance itself


@dataclass
class PoseCovariance:
    """
    How uncertain a reported pose is, as the covariance between its six degrees of
    freedom.

    .. note:: The covariance is expressed in the frame the pose itself was reported in;
        this type does not carry that frame.
    """

    values: np.ndarray
    """
    The covariance as a six by six matrix indexed by :class:`PoseAxis`.
    """

    @classmethod
    def from_row_major(cls, values: Sequence[float]) -> Self:
        """
        Read a covariance stored as one flat row-major sequence, the way a ROS pose
        covariance stores it.

        :param values: The 36 entries of the matrix, row by row.
        :return: The covariance they describe.
        :raises CovarianceNotSixBySixError: If there are not exactly 36 entries.
        """
        entries = np.asarray(values, dtype=np.float64).reshape(-1)
        expected_size = len(PoseAxis) * len(PoseAxis)
        if entries.size != expected_size:
            raise CovarianceNotSixBySixError(
                given_size=int(entries.size), expected_size=expected_size
            )
        return cls(values=entries.reshape(len(PoseAxis), len(PoseAxis)))

    def variance_of(self, axis: PoseAxis) -> float:
        """
        :param axis: The degree of freedom to read.
        :return: How uncertain that one degree of freedom is.
        """
        return float(self.values[axis, axis])

    @property
    def position_variance(self) -> float:
        """
        :return: The uncertainty of the position alone, summed over its three axes.
        """
        return sum(self.variance_of(axis) for axis in PoseAxis.position_axes())

    @property
    def rotation_variance(self) -> float:
        """
        :return: The uncertainty of the orientation alone, summed over its three axes.
        """
        return sum(self.variance_of(axis) for axis in PoseAxis.rotation_axes())

    @property
    def total_variance(self) -> float:
        """
        :return: The uncertainty of the whole pose, summed over all six axes.
        """
        return self.position_variance + self.rotation_variance


# %% reading the most recent covariance


@dataclass
class PoseCovarianceSource(ABC):
    """
    Something that reports how uncertain the pose it most recently observed was.

    Implemented by whatever receives poses from outside the process, so that a node
    reading the uncertainty does not depend on where the poses come from.
    """

    @property
    @abstractmethod
    def pose_covariance(self) -> PoseCovariance | None:
        """
        :return: The covariance of the most recently observed pose, or ``None`` while
            nothing has been observed yet.
        """
