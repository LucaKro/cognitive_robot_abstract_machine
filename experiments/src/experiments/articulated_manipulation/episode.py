"""
Running a task through one episode of the drawer-opening benchmark: the physics is built
from the true scene, the task's controller runs in a world built from the scene it
believes in, in lockstep with the physics, and the disturbances strike the physics
alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoSim

from experiments.articulated_manipulation.cabinet_physics import CabinetPhysics
from experiments.articulated_manipulation.cabinet_scene import CabinetScene
from experiments.articulated_manipulation.disturbance_protocol import (
    Disturbance,
    EpisodeSetup,
)
from experiments.articulated_manipulation.metrics import (
    EpisodeOutcome,
    EpisodeVerdict,
)
from experiments.control_loop_experiments.control_loop_profiler import (
    EXECUTOR_CONTROL_CYCLE_PHASES,
    ControlLoopProfiler,
)

# %% the task under evaluation


class EvaluatedTask(ABC):
    """
    A way of opening the cabinet's moving part that the benchmark evaluates.
    """

    @property
    @abstractmethod
    def recovery_transitions_authored(self) -> int:
        """
        How many transitions the task's author wrote to recover from a specific
        disturbance.
        """

    @abstractmethod
    def motion_statechart(self, scene: CabinetScene) -> MotionStatechart:
        """
        :param scene: The scene the robot believes in.
        :return: The motion that opens the part, which ends once the task believes the
            part is open.
        """


# %% disturbing the physics


@dataclass
class StruckDisturbance:
    """
    A disturbance that has struck and is not yet released.
    """

    disturbance: Disturbance
    """
    The disturbance.
    """

    struck_at: timedelta
    """
    When it struck, in simulated time.
    """


@dataclass
class DisturbanceSchedule:
    """
    The disturbances of one episode, each struck once it is due and released once its
    duration has passed.
    """

    pending: list[Disturbance]
    """
    The disturbances that have not struck yet.
    """

    struck: list[StruckDisturbance] = field(default_factory=list)
    """
    The disturbances that have struck and are not yet released.
    """

    def advance(self, physics: CabinetPhysics) -> None:
        """
        Release every disturbance whose duration has passed, and strike every one that
        is due.

        :param physics: The ground truth of the running episode.
        """
        for struck in list(self.struck):
            if physics.simulated_time - struck.struck_at >= struck.disturbance.duration:
                struck.disturbance.release(physics)
                self.struck.remove(struck)
        for disturbance in list(self.pending):
            if disturbance.trigger.is_due(physics):
                disturbance.strike(physics)
                self.pending.remove(disturbance)
                self.struck.append(
                    StruckDisturbance(
                        disturbance=disturbance, struck_at=physics.simulated_time
                    )
                )


# %% one episode


@dataclass
class Episode:
    """
    One run of a task under one condition of the benchmark.
    """

    setup: EpisodeSetup
    """
    What the episode runs under.
    """

    task: EvaluatedTask
    """
    The task evaluated.
    """

    timeout: timedelta = timedelta(seconds=30)
    """
    How much simulated time the task has to report that it is done.

    Placeholder.
    """

    control_frequency: float = 50.0
    """
    The frequency the task's controller runs at, in Hz.
    """

    required_opened_fraction: float = 0.8
    """
    The share of its travel the part has to be open in the physics to count as open.

    Placeholder.
    """

    def run(self) -> EpisodeOutcome:
        """
        :return: How the episode ended.
        """
        truth = self.setup.true_specification.to_domain_object()
        belief = self.setup.believed_specification.to_domain_object()
        for scene in (truth, belief):
            self.setup.arm_start_pose.apply_to(scene.robot.left_arm)
        simulation = MujocoSim(
            world=belief.world, ground_truth=truth.world, headless=True
        )
        physics = CabinetPhysics(
            simulation=simulation,
            scene=truth,
            specification=self.setup.true_specification,
        )
        simulation.start_stepped_simulation()
        try:
            return self._drive(belief, physics)
        finally:
            simulation.stop_simulation()

    def _drive(self, belief: CabinetScene, physics: CabinetPhysics) -> EpisodeOutcome:
        """
        Run the task in lockstep with the physics until it reports that it is done or
        runs out of time.

        :param belief: The scene the robot believes in.
        :param physics: The ground truth of the started simulation.
        :return: How the episode ended.
        """
        executor = Executor(
            context=MotionStatechartContext(
                world=belief.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=self.control_frequency
                ),
            ),
            pacer=SteppedSimulationPacer(physics.simulation),
        )
        schedule = DisturbanceSchedule(pending=list(self.setup.condition.disturbances))
        completion_time = None
        with ControlLoopProfiler(
            scenario_name=self.setup.condition.kind.name,
            control_dt=1 / self.control_frequency,
            phase_definitions=EXECUTOR_CONTROL_CYCLE_PHASES,
        ) as profiler:
            executor.compile(self.task.motion_statechart(belief))
            while physics.simulated_time < self.timeout:
                executor.tick()
                executor.pacer.sleep()
                schedule.advance(physics)
                if executor.motion_statechart.is_end_motion():
                    completion_time = physics.simulated_time
                    break
        opened_fraction = physics.opened_fraction
        return EpisodeOutcome(
            setup=self.setup,
            verdict=EpisodeVerdict.judge(
                reported_done=completion_time is not None,
                opened_fraction=opened_fraction,
                required_opened_fraction=self.required_opened_fraction,
            ),
            completion_time=completion_time,
            opened_fraction=opened_fraction,
            control_cycle_durations=profiler.profile.control_cycle.inclusive_durations,
        )
