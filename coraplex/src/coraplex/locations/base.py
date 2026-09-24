from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Callable, Iterator, Iterable

from coraplex.locations.sampling import CandidateDraw
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class Location(Iterable[Pose], ABC):
    """
    A region of poses the robot can be sent to, iterated as the pose candidates drawn
    from it.
    """

    draw: CandidateDraw = field(default_factory=CandidateDraw, kw_only=True)
    """
    The terms this location's candidates are drawn on.
    """

    @abstractmethod
    def candidates(self, draw: CandidateDraw) -> Iterator[Pose]:
        """
        Draw pose candidates from this location.

        Every location says what it does with the terms it is given, so none of them is
        chosen on a caller's behalf.

        :param draw: The terms to draw the candidates on.
        :return: The pose candidates, in the order they should be tried.
        """

    def ground(self) -> Pose:
        """
        :return: The first pose candidate of this location.
        """
        return next(iter(self))

    def __iter__(self) -> Iterator[Pose]:
        """
        :return: The candidates drawn on :attr:`draw`.

        .. warning::
            Must stay a generator, so nothing is drawn before the first ``next``. EQL's
            ``variable`` calls :func:`iter` on its domain while the plan is built.
        """
        yield from self.candidates(self.draw)


@dataclass
class DeferredLocation(Iterable[Pose]):
    """
    Lazily rebuilds a concrete :class:`Location` from current world state on each
    iteration, so it reflects the world at the moment the location is consumed
    (execution time) rather than when the plan was constructed.

    .. warning::
        :meth:`__iter__` must stay a generator (``yield from``). Returning
        ``iter(self.location_factory())`` would invoke the factory eagerly, because EQL's
        ``variable`` wraps the domain in :func:`filter`, which calls :func:`iter` on its
        argument at plan-construction time. A generator defers the factory call to the
        first ``next``, which only happens once the underspecified action is grounded.
    """

    location_factory: Callable[[], Location]
    """
    Builds a fresh :class:`Location` from the current world state.
    """

    def __iter__(self) -> Iterator[Pose]:
        yield from self.location_factory()
