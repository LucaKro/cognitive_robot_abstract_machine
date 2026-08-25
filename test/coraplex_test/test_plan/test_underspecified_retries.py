"""
Tests for which outcomes make an underspecified step try its next candidate, and for the
world each candidate is tried in.

A candidate that cannot be carried out must not take the whole plan down while other
candidates are still on offer, nor leave the world it stopped in for the next one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from giskardpy.data_types.exceptions import MotionTimeout
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.qp.exceptions import InfeasibleException
from typing_extensions import List, Optional, Type

from coraplex.plans.executables import UnderspecifiedExecutable
from coraplex.plans.failures import EmptyUnderspecified, PlanFailure

# %% a step whose candidates fail on demand


@dataclass
class FailingOnDemand:
    """
    Stands in for the executable a grounded candidate parses into.
    """

    raises: Optional[Exception] = None
    """
    What executing this candidate raises, or ``None`` when it succeeds.
    """

    def execute(self) -> None:
        if self.raises is not None:
            raise self.raises


@dataclass
class CandidateStub:
    """
    Stands in for one grounded candidate of an underspecified step.
    """

    executable: FailingOnDemand

    def parse(self) -> FailingOnDemand:
        return self.executable


@dataclass
class StepWithCandidates:
    """
    Stands in for the underspecified node, handing out its candidates in order.
    """

    outcomes: List[Optional[Exception]]
    """
    What each candidate raises when it is executed, in the order they are offered.
    """

    current_candidate: Optional[CandidateStub] = field(default=None, init=False)
    tried: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)

    def advance(self) -> bool:
        if self.tried >= len(self.outcomes):
            return False
        self.current_candidate = CandidateStub(
            FailingOnDemand(self.outcomes[self.tried])
        )
        self.tried += 1
        return True

    def stop_grounding(self) -> None:
        self.stopped = True


# %% the world a candidate moves


@dataclass
class WorldState:
    """
    Stands in for the state array a candidate's motion writes to.
    """

    data: np.ndarray


@dataclass
class ResetToWhereTheAttemptStarted:
    """
    Stands in for the world's reset-state context: it puts the state back on the way out
    unless the block that used it asked to keep what it did.
    """

    world: MovableWorld
    state_when_entered: Optional[np.ndarray] = field(default=None, init=False)
    keeps_what_the_block_did: bool = field(default=False, init=False)

    def __enter__(self) -> ResetToWhereTheAttemptStarted:
        self.state_when_entered = self.world.state.data.copy()
        return self

    def keep(self) -> None:
        self.keeps_what_the_block_did = True

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self.keeps_what_the_block_did:
            return
        self.world.state.data[:] = self.state_when_entered


@dataclass
class MovableWorld:
    """
    Stands in for the world a step's candidates move, putting its state back the way the
    real one does.
    """

    state: WorldState

    def reset_state_context(self) -> ResetToWhereTheAttemptStarted:
        return ResetToWhereTheAttemptStarted(self)


@dataclass
class ContextOf:
    """
    Stands in for the plan context an executable reads its world from.
    """

    world: MovableWorld


@dataclass
class MovingThenFailing:
    """
    Stands in for a candidate's executable that drives the world somewhere before it
    fails, the way a cancelled motion leaves the arm partway.
    """

    world: MovableWorld
    drives_to: float
    raises: Optional[Exception] = None

    def execute(self) -> None:
        self.world.state.data[:] = self.drives_to
        if self.raises is not None:
            raise self.raises


@dataclass
class StepWithMovingCandidates:
    """
    Stands in for an underspecified node whose candidates move the world as they run.
    """

    world: MovableWorld
    drives_to: List[float]
    outcomes: List[Optional[Exception]]

    current_candidate: Optional[CandidateStub] = field(default=None, init=False)
    tried: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)
    states_candidates_started_from: List[np.ndarray] = field(
        default_factory=list, init=False
    )

    def advance(self) -> bool:
        if self.tried >= len(self.outcomes):
            return False
        self.states_candidates_started_from.append(self.world.state.data.copy())
        self.current_candidate = CandidateStub(
            MovingThenFailing(
                self.world, self.drives_to[self.tried], self.outcomes[self.tried]
            )
        )
        self.tried += 1
        return True

    def stop_grounding(self) -> None:
        self.stopped = True


def _context_of_a_still_world() -> ContextOf:
    """
    :return: A context whose world no candidate moves, for the tests that only care
        which outcomes are retried.
    """
    return ContextOf(MovableWorld(WorldState(np.zeros(1))))


# %% which outcomes are retried


@pytest.mark.parametrize(
    "outcome",
    [
        PlanFailure(),
        CollisionViolatedError(violated_collisions=[], thresholds=[]),
        InfeasibleException(),
        MotionTimeout(tick_budget=1),
    ],
    ids=["plan_failure", "collision", "infeasible", "timeout"],
)
def test_a_candidate_that_cannot_be_carried_out_is_followed_by_the_next(outcome):
    """
    Every way a motion fails to be carried out leaves the remaining candidates a chance.
    """
    step = StepWithCandidates([outcome, None])

    UnderspecifiedExecutable(node=step, context=_context_of_a_still_world()).execute()

    assert step.tried == 2
    assert step.stopped


def test_a_step_whose_candidates_all_fail_reports_it():
    """
    Once nothing is left to try, the step fails rather than swallowing the exhaustion.
    """
    step = StepWithCandidates([PlanFailure(), PlanFailure()])

    with pytest.raises(EmptyUnderspecified):
        UnderspecifiedExecutable(
            node=step, context=_context_of_a_still_world()
        ).execute()

    assert step.tried == 2
    assert not step.stopped


def test_an_unrelated_error_is_not_retried():
    """
    A candidate that fails for a reason the plan cannot route around takes the plan
    down, rather than being retried until the candidates run out.
    """
    step = StepWithCandidates([ValueError("not a motion outcome"), None])

    with pytest.raises(ValueError):
        UnderspecifiedExecutable(
            node=step, context=_context_of_a_still_world()
        ).execute()

    assert step.tried == 1


# %% the world the next candidate is tried in


def test_a_failed_candidate_leaves_the_world_as_the_next_one_finds_it():
    """
    A candidate that fails has already moved the robot part of the way, and the next one
    must not be judged from where it stopped.

    Left there, a hand a cancelled motion parked inside a buffer zone fails a candidate
    that would have worked from the pose the step started at.
    """
    world = MovableWorld(WorldState(np.zeros(1)))
    step = StepWithMovingCandidates(
        world, drives_to=[7.0, 3.0], outcomes=[PlanFailure(), None]
    )

    UnderspecifiedExecutable(node=step, context=ContextOf(world)).execute()

    assert [state[0] for state in step.states_candidates_started_from] == [0.0, 0.0]
    assert step.stopped


def test_the_candidate_that_works_keeps_the_world_it_moved():
    """
    Only a failed candidate is undone.

    The one that works is the step's outcome, so what it moved has to stand for the
    steps that follow it.
    """
    world = MovableWorld(WorldState(np.zeros(1)))
    step = StepWithMovingCandidates(
        world, drives_to=[7.0, 3.0], outcomes=[PlanFailure(), None]
    )

    UnderspecifiedExecutable(node=step, context=ContextOf(world)).execute()

    assert world.state.data[0] == 3.0
