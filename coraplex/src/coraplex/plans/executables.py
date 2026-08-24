from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

from typing_extensions import List, Dict, ClassVar, Optional, TYPE_CHECKING

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import (
    MotionDidNotFinish,
    UnknownExecutionType,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from coraplex.plans.plan_node import MotionNode, UnderspecifiedNode
    from coraplex.datastructures.dataclasses import Context

logger = logging.getLogger(__name__)


@dataclass
class Executable:
    """
    Base class for executable units.
    """

    execution_list: List[Executable] = field(default_factory=list)
    """
    List of executables that comprises this executable.
    """

    context: Context = field(kw_only=True)
    """
    Coraplex context which should be used to execute this executable.
    """

    def execute(self) -> None:
        """
        Executes the unit.
        """
        for executable in self.execution_list:
            executable.execute()

    @property
    def giskard_executables(self) -> List[GiskardExecutable]:
        """
        The motion state charts run under this executable, in execution order.

        An action body that ends up in more than one chart is held as a composite of
        composites, so the charts it starts and finishes in are not all at the top
        level.
        """
        return [
            giskard_executable
            for executable in self.execution_list
            for giskard_executable in executable.giskard_executables
        ]


@dataclass
class GiskardExecutable(Executable):
    """
    Executable for everything that can be added to a Motion state chart, this includes
    the motions, pre -and postconditions and the pause and interrupt calls.
    """

    motion_nodes: List[MotionNode] = field(kw_only=True)
    """
    The motions this executable runs, in execution order.
    """

    execution_type: ClassVar[Optional[ExecutionType]] = None
    """
    The execution type used for all giskard executables, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
    """

    collision_avoidance: ClassVar[bool] = False
    """
    Whether an :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCo
    llisionAvoidance` is added to the motion state chart, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
    """

    _current_motion_state_chart: MotionStatechart = field(init=False, default=None)
    """
    Currently build motion state chart, internal only for managing the building the msc.
    """

    _motion_mappings: Optional[Dict[MotionNode, Task]] = field(init=False, default=None)
    """
    The tasks built from :attr:`motion_nodes`, once they have been.
    """

    @property
    def giskard_executables(self) -> List[GiskardExecutable]:
        return [self]

    @property
    def motion_mappings(self) -> Dict[MotionNode, Task]:
        """
        The giskard task of every motion this executable runs, in execution order.

        Built on first use rather than when the plan is parsed, so that a motion is
        turned into a task against the world it will run in: everything before it has
        run by then, and a chart built earlier describes a world that no longer exists --
        a hand that has since taken hold of something, for one.
        """
        if self._motion_mappings is None:
            self._motion_mappings = {
                node: node.motion.motion_chart for node in self.motion_nodes
            }
        return self._motion_mappings

    @property
    def motion_state_chart(self) -> MotionStatechart:
        """
        Giskard's motion state chart constructed from the motions of this executable.
        """
        self._current_motion_state_chart = MotionStatechart()
        if self.execution_type == ExecutionType.REAL:
            self._current_motion_state_chart.add_node(
                seq := Sequence(list(self.motion_mappings.values()))
            )
            self._current_motion_state_chart.add_node(EndMotion.when_true(seq))
            return self._current_motion_state_chart

        tasks = list(self.motion_mappings.values())
        for task in tasks:
            self._current_motion_state_chart.add_node(task)
        end_trigger = tasks[-1].observation_variable

        if self.execution_type == ExecutionType.SIMULATED:
            skip_end_conditions = self._add_pause_interrupt(tasks)

            # The motion is done when the last task finished or the first skipped
            # (interrupted) task is reached.
            if skip_end_conditions:
                end_trigger = trinary_logic_or(end_trigger, *skip_end_conditions)

        if GiskardExecutable.collision_avoidance:
            self._current_motion_state_chart.add_node(ExternalCollisionAvoidance())

        end_motion = EndMotion()
        end_motion.start_condition = end_trigger
        self._current_motion_state_chart.add_node(end_motion)
        return self._current_motion_state_chart

    def _add_pause_interrupt(self, tasks: List[Task]) -> List[ObservationStateValues]:
        """
        Wire the tasks as an interruptible/pausable sequence.

        Each task carries two monitors bound to its originating plan node:

        - a pause monitor feeding the task's pause_condition, so the *active*
          motion is held (and later resumed) when its plan node is paused;
        - an interrupt monitor gating the *next* task's start. An interrupt lets
          the currently active motion finish but prevents the subsequent ones
          from starting ("finish active, skip rest"). When a not-yet-started task
          is reached while interrupted, the motion ends there.

        :param tasks: The list of tasks that are were added to the motion state chart
        :returns: List of skip conditions for the case if a task is interrupted
        """
        from coraplex.plans.condition_nodes import PlanNodeStatusMonitor

        skip_end_conditions = []
        plan_nodes = list(self.motion_nodes)
        for index, (plan_node, task) in enumerate(zip(plan_nodes, tasks)):
            # A task is done once its own goal is observed (as giskard's Sequence does),
            # which hands the robot to the task after it. Only the last task hands it to
            # nobody, so only there can a motion keep its goal in force while the chart
            # winds down.
            holds_on = (
                task is tasks[-1]
                and plan_node.designator.holds_its_goal_until_the_motion_ends
            )
            if not holds_on:
                task.end_condition = task.observation_variable

            pause_monitor = PlanNodeStatusMonitor(
                predicate=lambda node=plan_node: node.is_paused,
                name=f"paused#{index}",
            )
            self._current_motion_state_chart.add_node(pause_monitor)
            task.pause_condition = pause_monitor.observation_variable

            interrupt_monitor = PlanNodeStatusMonitor(
                predicate=lambda node=plan_node: node.is_interrupted,
                name=f"interrupted#{index}",
            )
            self._current_motion_state_chart.add_node(interrupt_monitor)
            if index > 0:
                previous_done = tasks[index - 1].observation_variable
                # start only once the previous motion finished and this one is not
                # interrupted ...
                task.start_condition = trinary_logic_and(
                    previous_done,
                    trinary_logic_not(interrupt_monitor.observation_variable),
                )
                # ... otherwise, if we reach it while interrupted, the sequence ends.
                skip_end_conditions.append(
                    trinary_logic_and(
                        previous_done, interrupt_monitor.observation_variable
                    )
                )
        return skip_end_conditions

    @property
    def is_interrupted(self) -> bool:
        return any(node.is_interrupted for node in self.motion_nodes)

    @property
    def is_paused(self) -> bool:
        return any(node.is_paused for node in self.motion_nodes)

    def execute(self) -> None:
        """
        Builds the motion state chart from the motions and executes it according to the
        execution type.
        """
        if len(self.motion_nodes) == 0:
            return

        match GiskardExecutable.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_simulation()
            case ExecutionType.REAL:
                self._execute_real()
            case ExecutionType.NO_EXECUTION:
                return
            case _:
                raise UnknownExecutionType(GiskardExecutable.execution_type)

    def _execute_simulation(self) -> None:
        """
        Compiles the motion state chart and ticks it in the world of the context until
        it is done.
        """
        executor = Ros2Executor(
            context=MotionStatechartContext(
                world=self.context.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
            ros_node=self.context.ros_node,
        )
        motion_state_chart = self.motion_state_chart
        executor.compile(motion_state_chart)

        counter = 0
        while counter < len(self.motion_nodes) * 2000:
            # Interrupting and pausing are handled inside the motion state chart by
            # per-task monitors (see motion_state_chart): an interrupt ends the
            # motion via EndMotion, a pause holds the active task via its
            # pause_condition. While paused we simply do not tick, so the pause does
            # not consume the tick budget.
            if self.is_paused:
                time.sleep(0.01)
                continue

            executor.tick()
            counter += 1
            if executor.motion_statechart.is_end_motion():
                break

        executor.set_velocity_acceleration_jerk_to_zero()
        executor.motion_statechart.cleanup_nodes(context=executor.context)
        executor.context.cleanup()

        if not executor.motion_statechart.is_end_motion():
            failed_nodes = [
                node
                for node in motion_state_chart.nodes
                if node.life_cycle_state
                not in [LifeCycleValues.DONE, LifeCycleValues.NOT_STARTED]
            ]
            logger.error(f"Failed Nodes: {failed_nodes}")
            raise MotionDidNotFinish(failed_nodes)

    def _execute_real(self) -> None:
        """
        Executes the motion state chart on the real robot via giskard while monitoring
        for interrupts.
        """
        self.context.giskard_wrapper.execute(self.motion_state_chart)


@dataclass
class ModelChangeExecutable(Executable):
    """
    Executable that re-attaches a body to a new parent in the world model while keeping
    its current global pose.
    """

    body: Body = field(kw_only=True)
    """
    The body that is re-attached.
    """

    new_parent: Body = field(kw_only=True)
    """
    The body the moved body is attached to afterwards.
    """

    giskard_idle_settle_delta: timedelta = field(
        default=timedelta(seconds=0.3), kw_only=True
    )
    """
    Time to wait after publishing the model change on the real robot.

    Giskard only applies buffered world updates, and only republishes tf, while its
    behavior tree is idle between goals (tree tick period is 50ms); this delay gives it
    a few idle ticks to catch up before the next motion goal is sent, instead of relying
    on however much idle time happens to fall out of the surrounding plan's timing.
    """

    def execute(self) -> None:
        """
        Re-parent the body to ``new_parent`` while preserving its global pose.
        """
        obj_transform = self.context.world.compute_forward_kinematics(
            self.new_parent, self.body
        )
        with self.context.world.modify_world():
            self.context.world.remove_connection(self.body.parent_connection)
            # TODO: this shouldn't be fixed but 6DOF
            connection = FixedConnection(
                parent=self.new_parent,
                child=self.body,
                parent_T_connection_expression=obj_transform,
            )

            # connection = Connection6DoF.create_with_dofs(
            #     parent=self.new_parent, child=self.body, world=self.context.world, parent_T_connection_expression=obj_transform
            # )
            self.context.world.add_connection(connection)
            # connection.origin = obj_transform
        if GiskardExecutable.execution_type == ExecutionType.REAL:
            time.sleep(self.giskard_idle_settle_delta.total_seconds())


@dataclass
class UnderspecifiedExecutable(Executable):
    """
    Executable for an underspecified node whose resolution is deferred to execution
    time.

    Because it is not a :class:`GiskardExecutable`, it acts as a boundary in the
    execution list: every preceding executable runs (and mutates the world) before it
    is reached. Only then is the underspecified statement grounded, so the query sees
    the correct world state (e.g. the torso already raised, the object already in the
    gripper). Candidates are tried in order until one executes without raising a
    :class:`~pycram.plans.failures.PlanFailure`; if the generator is exhausted,
    :class:`~pycram.plans.failures.EmptyUnderspecified` is raised.

    A candidate that fails may already have moved the robot part of the way, so the world it
    stopped in is put back before the next one is tried. Judged from where a cancelled
    motion left the arm -- inside a buffer zone it never chose to enter -- a candidate
    that would have worked is rejected.
    """

    node: UnderspecifiedNode = field(kw_only=True)
    """
    The underspecified node that is grounded when this executable is reached.
    """

    def execute(self) -> None:
        from coraplex.plans.failures import (
            MOTION_DID_NOT_WORK_OUT,
            PlanFailure,
            EmptyUnderspecified,
        )

        while self.node.advance():
            with self.context.world.reset_state_context() as attempt:
                try:
                    self.node.current_candidate.parse().execute()
                except (PlanFailure, *MOTION_DID_NOT_WORK_OUT):
                    continue
                attempt.keep()
                self.node.stop_grounding()
                return
        raise EmptyUnderspecified()
