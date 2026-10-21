from __future__ import annotations

import logging
import time
from abc import ABC
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, fields
from itertools import count

from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
)

from coraplex.datastructures.enums import ActionTrialVisualization
from coraplex.plans.context_extensions import StatementGrounding
from coraplex.plans.plan_transformation import PlanRewriting, PlanTransformation
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.plan_transformations import WriteTheOdometryInsteadOfDriving
from coraplex.robot_plans.robot_specific_plan_transformations import (
    KeepStretchInFrontOfTheContainerItCloses,
)
from cramph.visualization import RvizVisualization
from cramph.context import StatechartContext
from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    ChoosingNode,
    CompositeNodeChoosingItsChild,
    ControlStructureNode,
    Parallel,
    Sequence,
    StatementNode,
)
from cramph.executor import ExecutorExtension
from cramph.monitors import CountTicks
from cramph.exceptions import ExecutionFailure
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled

if TYPE_CHECKING:
    from coraplex.plans.executors import PlanExecutor
    from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

# %% preparing a chosen child


@dataclass
class PlanChildChooser(ChildChooser[ChoosingNode], ABC):
    """
    Chooses children that are part of a plan: each is rewritten by the plan
    transformations of its statechart, and given up on once it stops approaching its
    goal, so the node that chose it can choose the next one.
    """

    def prepare_chosen_child(
        self,
        node: ChoosingNode,
        child: StatechartNode,
    ) -> None:
        """
        Rewrites `child` with the plan transformations of `context`, before the monitor
        watching it for a stall collects what it watches, see
        :meth:`~cramph.composites.ChildChooser.prepare_chosen_child`.
        """
        rewriting = node.context.get_extension(PlanRewriting)
        if rewriting is not None:
            rewriting.rewrite(child)

    def failure_monitors_for(
        self,
        node: ChoosingNode,
        child: StatechartNode,
    ) -> list[StatechartNode]:
        """
        Differs from :meth:`~cramph.composites.ChildChooser.failure_monitors_for` in
        giving up on `child` once nothing below it approaches its goal.

        :return: A monitor firing once nothing below `child` approaches its goal.
        """
        return [Stalled(monitored_node=child)]


@dataclass(eq=False, repr=False, kw_only=True)
class TriedStatement(CompositeNodeChoosingItsChild):
    """
    Stands in for a statement in a trial, choosing the candidate tried for it the way
    the statement chooses the candidate it runs, so the candidate starts at the same
    point of the trial as it does for real, see :class:`TriedCandidateChooser`.
    """


@dataclass
class TriedCandidateChooser(PlanChildChooser[TriedStatement]):
    """
    Hands every :class:`TriedStatement` of a trial the candidate tried for it, prepared
    and watched for a stall like a candidate the statement runs for real.

    A candidate that fails leaves its stand-in nothing to run next, so it fails the
    trial.
    """

    candidates: dict[TriedStatement, Action] = field(default_factory=dict)
    """
    The candidate tried for each stand-in, removed once it is handed out.
    """

    def choose_child(self, node: TriedStatement) -> StatechartNode | None:
        candidate = self.candidates.pop(node, None)
        if candidate is None:
            return None
        return Sequence(name=f"{candidate.name}/steps", nodes=[candidate])


@dataclass
class TriedNode:
    """
    What a trial runs in the copy: a node of the plan, rebuilt onto the copy with its
    statements standing in as :class:`TriedStatement` nodes, and the chooser handing
    them their candidates.
    """

    node: StatechartNode
    """
    The node run in the copy.
    """

    chooser: TriedCandidateChooser
    """
    Hands every stand-in in :attr:`node` the candidate tried for it.
    """


@dataclass
class ActionTrial:
    """
    Tries grounded actions against a copy of the world, to check that a candidate can
    succeed before it is attempted for real.

    One copy serves every candidate: after each attempt its model is rolled back and its
    state restored, and when the executor's world has changed since, the copy replays
    those model and state changes instead of being taken anew. Collision rules changed
    after the copy was taken are not carried over.

    Trials never publish to a synchronizer and always run simulated, in an executor of
    :attr:`trial_executor_type` over the copy, with the context extensions of
    :attr:`executor` rebound to the copy, and the base put where a candidate drives it
    by :attr:`plan_adaptations`.
    While the executor is debugging, the copy is shown in RViz under its own frame
    prefix and marker topic.
    """

    executor: PlanExecutor
    """
    The executor the candidates are grounded for.

    Only ever read from: a trial never mutates its world, and the candidates themselves
    are left untouched too, so they can still run for real afterwards.
    """

    trial_executor_type: type[PlanExecutor]
    """
    The executor that runs a candidate in the copy, which simulates the robot.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=lambda: [
            WriteTheOdometryInsteadOfDriving(),
            KeepStretchInFrontOfTheContainerItCloses(),
        ],
        kw_only=True,
    )
    """
    The plan transformations adapting every candidate to being tried, in place of the
    plan adaptations of :attr:`trial_executor_type`.
    """

    copy_marker_alpha: float = field(default=0.9, kw_only=True)
    """
    The opacity the copy is drawn with while debugging, so it can be told apart from the
    world it copies where the two overlap.
    """

    _copied_world: World | None = field(default=None, init=False, repr=False)
    """
    The copy candidates are tried against.
    """

    _source_versions: tuple[int, int] | None = field(
        default=None, init=False, repr=False
    )
    """
    The model and state versions of the executor's world when the copy last matched it,
    used to notice that it has moved on and the copy has to be caught up.
    """

    _replayed_modification_blocks: int = field(default=0, init=False, repr=False)
    """
    How many of the modification blocks of the executor's world the copy already holds.
    """

    @property
    def world(self) -> World:
        """
        :return: The world the candidates are grounded in.
        """
        return self.executor.context.world

    def succeeds(self, action: Action) -> bool:
        """
        Run `action` against the copy and restore the copy afterwards.

        The action is rebuilt from its own parameters, rebound onto the copy, because an
        action that modifies the model (attaching a grasped body, say) requires the
        entities it is given to belong to the world being modified, and because a
        statechart node belongs to one statechart only.

        The version to roll back to is read here rather than when the copy is taken, so
        each attempt undoes only its own modifications.

        :param action: The grounded action to try out.
        :return: True if `action` runs to completion without raising an
            :class:`~cramph.exceptions.ExecutionFailure`.
        """

        def build(world: World) -> TriedNode:
            statement = TriedStatement(name=str(action))
            return TriedNode(
                node=statement,
                chooser=TriedCandidateChooser(
                    candidates={statement: self._rebind_action_to_world(action, world)}
                ),
            )

        return self._runs_to_completion(build, str(action))

    def block_succeeds(
        self,
        block: StatechartNode,
        candidates: dict[StatementNode, Action],
    ) -> bool:
        """
        Run `block` against the copy, every statement in it replaced by its candidate,
        and restore the copy afterwards.

        The candidates are tried together with every node that runs beside them, the way
        the block runs for real, rather than each on its own, see :meth:`_rebuild_onto`.

        :param block: The node whose statements are grounded together.
        :param candidates: The candidate each statement of `block` runs.
        :return: True if `block` runs to completion without raising an
            :class:`~cramph.exceptions.ExecutionFailure`.
        """

        def build(world: World) -> TriedNode:
            chooser = TriedCandidateChooser()
            return TriedNode(
                node=self._rebuild_onto(block, world, candidates, chooser),
                chooser=chooser,
            )

        return self._runs_to_completion(
            build, f"{block.name} with {', '.join(map(str, candidates.values()))}"
        )

    def _runs_to_completion(
        self,
        build: Callable[[World], TriedNode],
        description: str,
    ) -> bool:
        """
        :param build: Builds what is tried against the copy it is given.
        :param description: What is tried, for the log.
        :return: True if the node `build` returns runs to completion against the copy
            without raising an :class:`~cramph.exceptions.ExecutionFailure`.
        """
        logger.info(f"Trial of {description} started")
        started = time.monotonic()
        world = self._copy()
        tried = build(world)
        version = world.get_world_model_manager().version

        with world.reset_state_context():
            try:
                executor = self._trial_executor(world)
                executor.context.require_extension(ChildChooserAccess).choosers = [
                    tried.chooser
                ]
                statechart = Statechart(context=executor.context)
                # The node runs in a sequence of its own, as a statement or a block of
                # a plan does, and only after a first tick: what a statechart starts
                # with starts as it compiles, a tick before anything starts in a running
                # one, which would let a stand-in choose its candidate a tick earlier
                # than the statement does for real.
                statechart.add_node(
                    Sequence(
                        nodes=[CountTicks(name="trial start", ticks=1), tried.node]
                    )
                )
                executor.compile(statechart)
                executor.execute()
                logger.info(
                    f"Trial of {description} succeeded after "
                    f"{time.monotonic() - started:.2f} s"
                )
                return True
            except ExecutionFailure as failure:
                logger.info(
                    f"Trial of {description} failed after "
                    f"{time.monotonic() - started:.2f} s: {failure}"
                )
                return False
            finally:
                # Undo the model changes before leaving the reset context restores the
                # state, which needs the degrees of freedom it was snapshotted with.
                world.rollback_to_version(version)

    def _trial_executor(self, world: World) -> PlanExecutor:
        """
        :param world: The copy a candidate is tried in.
        :return: An executor running a candidate in `world`, configured like
            :attr:`executor`, the robot and everything else its context extensions refer
            to rebound to `world`, adapting the candidate with :attr:`plan_adaptations`,
            and showing `world` in RViz while it is debugging.
        """
        return self.trial_executor_type(
            StatechartContext.with_extensions(
                world,
                [
                    world.rebind_world_entities(extension)
                    for extension in self.executor.supplied_context_extensions
                ],
            ),
            robot_adaptations=self.executor.robot_adaptations,
            plan_adaptations=self.plan_adaptations,
            collision_avoidance=self.executor.collision_avoidance,
            extensions=self._trial_visualizations(),
        )

    def _trial_visualizations(self) -> list[ExecutorExtension]:
        """
        :return: The RViz visualization of the copy, through the ROS node of
            :attr:`executor`, under its own frame prefix and marker topic, while the
            executor is debugging; empty otherwise.
        """
        if not self.executor.debug:
            return []
        return [
            RvizVisualization(
                ros_node=self.executor.ros_node,
                collision_visualization=True,
                frame_prefix=ActionTrialVisualization.FRAME_PREFIX,
                marker_topic=ActionTrialVisualization.MARKER_TOPIC,
                marker_alpha=self.copy_marker_alpha,
                # Hidden between trials rather than deleted, so the next one does not
                # load every mesh again.
                hides_markers_when_stopped=True,
            )
        ]

    @classmethod
    def _rebind_action_to_world(cls, action: Action, world: World) -> Action:
        """
        :param action: The grounded action to try out.
        :param world: The copy to try it against.
        :return: A new action with the parameters of `action`, referring to `world`.
            The actions among those parameters are built anew the same way, since each
            of them becomes a node of the statechart the trial runs.
        """
        return type(action)(
            **{
                name: (
                    cls._rebind_action_to_world(value, world)
                    if isinstance(value, Action)
                    else world.rebind_world_entities(value)
                )
                for name, value in action.designator_parameter.items()
            }
        )

    @classmethod
    def _rebuild_onto(
        cls,
        node: StatechartNode,
        world: World,
        candidates: dict[StatementNode, Action],
        chooser: TriedCandidateChooser,
    ) -> StatechartNode | None:
        """
        :param node: A node of the plan, with everything below it.
        :param world: The copy to try it against.
        :param candidates: The candidate each statement below `node` runs.
        :param chooser: Hands the stand-ins of the statements their candidates; every
            stand-in made here is added to it.
        :return: A new node referring to `world`: a statement replaced by a
            :class:`TriedStatement` choosing its candidate, so the candidate starts when
            it would for real, an action rebuilt from its own parameters, and any other
            node from the fields it is constructed with, the nodes among them rebuilt
            the same way. None for a statement without a candidate, and for a control
            structure left with nothing to hold.
        """
        if isinstance(node, StatementNode):
            if node not in candidates:
                return None
            statement = TriedStatement(name=node.name)
            chooser.candidates[statement] = cls._rebind_action_to_world(
                candidates[node], world
            )
            return statement
        if isinstance(node, Action):
            return cls._rebind_action_to_world(node, world)
        constructor_arguments = {
            constructor_field.name: cls._rebuild_field_onto(
                getattr(node, constructor_field.name), world, candidates, chooser
            )
            for constructor_field in fields(node)
            if constructor_field.init
        }
        if (
            isinstance(node, ControlStructureNode)
            and not constructor_arguments["nodes"]
        ):
            return None
        return type(node)(**constructor_arguments)

    @classmethod
    def _rebuild_field_onto(
        cls,
        value: Any,
        world: World,
        candidates: dict[StatementNode, Action],
        chooser: TriedCandidateChooser,
    ) -> Any:
        """
        :param value: The value of a field a node is constructed with.
        :param world: The copy to try the node against.
        :param candidates: The candidate each statement below the node runs.
        :param chooser: Hands the stand-ins of the statements their candidates.
        :return: `value` referring to `world`, a node or a list of nodes rebuilt by
            :meth:`_rebuild_onto`, leaving out the nodes it returns None for.
        """
        if isinstance(value, StatechartNode):
            return cls._rebuild_onto(value, world, candidates, chooser)
        if isinstance(value, list) and any(
            isinstance(element, StatechartNode) for element in value
        ):
            return [
                rebuilt
                for element in value
                if (rebuilt := cls._rebuild_onto(element, world, candidates, chooser))
                is not None
            ]
        return world.rebind_world_entities(value)

    def _copy(self) -> World:
        """
        :return: The copy to try candidates against, caught up with the executor's world
            if that has changed since the copy last matched it.
        """
        versions = (
            self.world.get_world_model_manager().version,
            self.world.state.version,
        )
        if self._copied_world is None:
            self._take_copy()
        elif self._source_versions != versions:
            self._catch_up()
        self._source_versions = versions
        return self._copied_world

    def _take_copy(self) -> None:
        """
        Copy the executor's world.
        """
        self._copied_world = deepcopy(self.world)
        self._replayed_modification_blocks = len(
            self.world.get_world_model_manager().model_modification_blocks
        )

    def _catch_up(self) -> None:
        """
        Bring the copy up to date with the executor's world: replay the modifications
        made to it since, the way copying it replays all of them, and take over its
        state.

        The copy's own modifications are all rolled back by then, so it still matches
        the world as it was when it last caught up.
        """
        modification_blocks = (
            self.world.get_world_model_manager().model_modification_blocks
        )
        with self._copied_world.modify_world():
            for block in modification_blocks[self._replayed_modification_blocks :]:
                block.update_references_for_world_and_apply(world=self._copied_world)
            self._copied_world.state.merge_state(self.world.state)
        self._replayed_modification_blocks = len(modification_blocks)

    def discard(self) -> None:
        """
        Release the copy, so the next trial takes a fresh one.
        """
        self._copied_world = None
        self._source_versions = None


# %% grounding the statements that run side by side together


@dataclass
class RankedCandidates:
    """
    The candidates of a statement in the order it yields them, pulled from it only as
    far as they are asked for.
    """

    node: StatementNode
    """
    The node whose statement yields the candidates.
    """

    grounding: StatementGrounding
    """
    How the statement is grounded.
    """

    pulled: list[Action] = field(default_factory=list, init=False)
    """
    The candidates pulled so far, in rank order.
    """

    exhausted: bool = field(default=False, init=False)
    """
    Whether the statement yields no candidate beyond :attr:`pulled`.
    """

    def at(self, rank: int) -> Action | None:
        """
        :param rank: The position of the candidate in the order the statement yields
            them, the first at 0.
        :return: The candidate at `rank`, or None if the statement yields fewer.
        """
        while len(self.pulled) <= rank and not self.exhausted:
            candidate = self.node.next_candidate(
                self.grounding.query_backend, self.grounding.candidates_to_try
            )
            if candidate is None:
                self.exhausted = True
            else:
                self.pulled.append(candidate)
        return self.pulled[rank] if rank < len(self.pulled) else None


def combinations_closest_first(
    rankings: list[RankedCandidates],
) -> Iterator[tuple[Action, ...]]:
    """
    Every combination of one candidate of each ranking, those with the lowest sum of
    ranks first, so the candidates every statement ranks first are combined before any
    combination takes a later one.

    :param rankings: The candidates of each statement.
    :return: The combinations, each holding one candidate per ranking, in its order.
    """
    for rank_sum in count():
        for ranks in _ranks_summing_to(rank_sum, len(rankings)):
            combination = tuple(
                ranking.at(rank) for ranking, rank in zip(rankings, ranks)
            )
            if all(candidate is not None for candidate in combination):
                yield combination
        # Every ranking was asked for its candidate at rank_sum above, so each is
        # exhausted once rank_sum outgrew it.
        if all(ranking.exhausted for ranking in rankings) and rank_sum >= sum(
            len(ranking.pulled) - 1 for ranking in rankings
        ):
            return


def _ranks_summing_to(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """
    :param total: What the ranks add up to.
    :param parts: How many ranks there are.
    :return: Every tuple of `parts` ranks that add up to `total`.
    """
    if parts == 0:
        if total == 0:
            yield ()
        return
    for first in range(total + 1):
        for rest in _ranks_summing_to(total - first, parts - 1):
            yield first, *rest


@dataclass
class BlockGrounding:
    """
    The candidates the statements of one run of a parallel were grounded to together.
    """

    start_tick: int
    """
    The tick the run of the parallel started on.
    """

    candidates: dict[StatementNode, Action | None]
    """
    The candidate each statement runs once it asks for its child, None for every
    statement if no combination succeeded its trial.
    """


# %% grounding underspecified actions while the statechart runs


@dataclass
class UnderspecifiedChildChooser(PlanChildChooser[StatementNode]):
    """
    Grounds the statement of every :class:`~cramph.composites.StatementNode` of a
    statechart into the action it runs next, in a sequence of its own for the nodes a
    plan transformation puts beside it.

    A grounded action is tried out first, so one that fails its trial is discarded
    without ever touching the real world, and a bad parameterization cannot poison a
    later attempt.

    The statements of a parallel are grounded together when it starts, because what
    runs beside a candidate changes the motion it runs in: every combination of their
    candidates is tried as the parallel, with the actions beside them, the combinations
    of the candidates each statement ranks first before the rest, and the first that
    succeeds is what they run.
    A statement is grounded on its own when it runs in no parallel, when it asks again
    after a child of its own failed, and when it started after its parallel did, which
    leaves no state to try the parallel from.
    """

    executor: PlanExecutor
    """
    The executor whose statechart holds the nodes, which also tries their candidates.
    """

    trial_executor_type: type[PlanExecutor]
    """
    The executor the candidates are tried in, see :class:`ActionTrial`.
    """

    trial: ActionTrial = field(init=False)
    """
    The trial every node tries its candidates against, so they all try them in one copy
    of the world.
    """

    _block_groundings: dict[Parallel, BlockGrounding] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The latest grounding of each parallel whose statements were grounded together.
    """

    _leftover_candidates: dict[StatementNode, deque[Action]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The candidates a statement yielded while it was grounded together with the others of
    its parallel but does not run, tried first should its child fail.
    """

    def __post_init__(self) -> None:
        self.trial = ActionTrial(
            executor=self.executor, trial_executor_type=self.trial_executor_type
        )

    def choose_child(self, node: StatementNode) -> StatechartNode | None:
        """
        :return: The sequence holding the grounded action `node` runs next, or None
            once its statement ran out of actions that succeed their trial.
        """
        grounding = node.context.require_extension(StatementGrounding)
        block_grounding = self._grounding_of_block_around(node, grounding)
        if block_grounding is not None and node in block_grounding.candidates:
            candidate = block_grounding.candidates.pop(node)
            if candidate is None:
                node.stop_grounding()
                return None
            return Sequence(name=f"{candidate.name}/steps", nodes=[candidate])
        return self._choose_on_its_own(node, grounding)

    def _choose_on_its_own(
        self, node: StatementNode, grounding: StatementGrounding
    ) -> StatechartNode | None:
        """
        :return: The sequence holding the next grounded action of `node` that succeeds
            its trial on its own, or None once its statement ran out of them.
        """
        leftovers = self._leftover_candidates.pop(node, deque())
        if node.latest_child is None:
            # Left from an earlier run, grounded against a world that has moved on.
            leftovers.clear()
        proposal = self._next_proposal(node, grounding, leftovers)
        while proposal is not None:
            if self.trial.succeeds(proposal):
                if leftovers:
                    self._leftover_candidates[node] = leftovers
                return Sequence(name=f"{proposal.name}/steps", nodes=[proposal])
            proposal = self._next_proposal(node, grounding, leftovers)
        node.stop_grounding()
        return None

    @staticmethod
    def _next_proposal(
        node: StatementNode,
        grounding: StatementGrounding,
        leftovers: deque[Action],
    ) -> Action | None:
        """
        :return: The next of `leftovers`, or the next candidate of `node`'s statement
            once they are used up.
        """
        if leftovers:
            return leftovers.popleft()
        return node.next_candidate(grounding.query_backend, grounding.candidates_to_try)

    def _grounding_of_block_around(
        self, node: StatementNode, grounding: StatementGrounding
    ) -> BlockGrounding | None:
        """
        :return: The grounding of the current run of the parallel `node` runs in,
            grounded now if `node` is the first of its statements to ask on the run's
            start, or None if `node` runs in no parallel, asks after a child of its own
            failed, or started after its parallel when that was not grounded.
        """
        block = self._block_around(node)
        if block is None or node.latest_child is not None:
            return None
        start_tick = self._start_tick_of_current_run(block)
        block_grounding = self._block_groundings.get(block)
        if block_grounding is not None and block_grounding.start_tick == start_tick:
            return block_grounding
        if self._start_tick_of_current_run(node) != start_tick:
            return None
        block_grounding = BlockGrounding(
            start_tick=start_tick,
            candidates=self._ground_together(block, grounding),
        )
        self._block_groundings[block] = block_grounding
        return block_grounding

    @staticmethod
    def _block_around(node: StatementNode) -> Parallel | None:
        """
        :return: The outermost parallel above `node`, which holds everything that runs
            beside it, or None if it runs in none. A parallel that is an action is left
            out, because it is tried by rebuilding it from its own parameters, which
            grounds what it runs anew.
        """
        blocks = [
            ancestor
            for ancestor in node.path
            if isinstance(ancestor, Parallel) and not isinstance(ancestor, Action)
        ]
        return blocks[-1] if blocks else None

    @staticmethod
    def _start_tick_of_current_run(node: StatechartNode) -> int:
        """
        :return: The tick the current run of `node` started on.
        """
        run = node.statechart.state_history.get_current_run_ticks_of_node(node)
        # A tick is recorded only once its nodes chose their children, so a node that
        # started on the tick being chosen for has no run recorded yet.
        return node.context.tick_count if run is None else run.start_tick

    def _ground_together(
        self, block: Parallel, grounding: StatementGrounding
    ) -> dict[StatementNode, Action | None]:
        """
        Try the combinations of the candidates of the statements of `block` as the whole
        of `block`, the closest first, see :func:`combinations_closest_first`.

        :return: The candidate each statement runs, from the first combination that
            succeeded its trial, or None for each if none did.
        """
        statements = self._statements_of(block)
        if not statements:
            return {}
        rankings = [
            RankedCandidates(node=statement, grounding=grounding)
            for statement in statements
        ]
        for combination in combinations_closest_first(rankings):
            candidates = dict(zip(statements, combination))
            if self.trial.block_succeeds(block, candidates):
                for ranking, chosen in zip(rankings, combination):
                    self._leftover_candidates[ranking.node] = deque(
                        candidate
                        for candidate in ranking.pulled
                        if candidate is not chosen
                    )
                return candidates
        for statement in statements:
            statement.stop_grounding()
        return dict.fromkeys(statements)

    def _statements_of(self, node: StatechartNode) -> list[StatementNode]:
        """
        :return: The statements below `node` that have not chosen a child in their
            current run, leaving out those in an action, which the action's own trial
            grounds.
        """
        statements = []
        for child in node.children:
            if isinstance(child, StatementNode):
                if (
                    child.latest_child is None
                    and not child.life_cycle_state.is_terminal
                ):
                    statements.append(child)
            elif not isinstance(child, Action):
                statements.extend(self._statements_of(child))
        return statements

    def cleanup(self) -> None:
        """
        Release the trial's world copy and what was grounded together.
        """
        self.trial.discard()
        self._block_groundings.clear()
        self._leftover_candidates.clear()
