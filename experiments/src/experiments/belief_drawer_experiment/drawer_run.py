"""
One run of the belief experiment: a motion performed under one condition, and what it
did, measured against the world rather than against the belief.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from random_events.variable import Continuous
from typing_extensions import Optional

from experiments.belief_drawer_experiment.drawer_scenario import (
    ArmConfiguration,
    DrawerCondition,
    DrawerWorld,
    ScriptedLikelihood,
)
from experiments.experiment_definitions import ExperimentResult
from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.grasp import GraspBelief
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% what a caller has to say about the belief


@dataclass(frozen=True)
class BeliefSettings:
    """
    The parameters of the grasp belief a run filters its likelihood with.

    They say how fast a grasp should be forgotten and how confident is confident enough,
    which is what this experiment exists to measure, so they are stated by whoever runs
    it rather than defaulted here.
    """

    prior_uncertainty: float
    """
    How uncertain the prior is, in the log-odds the belief is estimated in.
    """

    forgetting_half_life: float
    """
    How many seconds an open gripper takes to carry the estimate halfway back to the
    prior.
    """

    drift: float
    """
    How much uncertainty a second adds on its own.
    """

    false_below: float
    """
    The probability of a grasp under which the belief observes that there is none.
    """


# %% what one run did

MILLIMETRES_PER_METRE = 1000.0
"""
What the world's metres are converted to before a run's distances are reported.

A drawer travels tenths of a metre and a grip is left behind by thousandths of one, and
:class:`~experiments.experiment_definitions.MeanAndStandardDeviation` reports to two
decimals, so metres would round both to nothing.
"""


@dataclass
class DrawerRunOutcome(ExperimentResult):
    """
    What one run did, read off the world it ran in.

    Nothing here is read off the posterior except :attr:`grasp_probability`, which is
    reported so a reader can see why a run behaved as it did and is never what a run is
    judged by.
    """

    condition: DrawerCondition
    """
    How the run weighed its grip and what its grasp was doing.
    """

    arm_configuration_name: str
    """
    Which posture the arm started from.
    """

    cabinet_yaw: float
    """
    How far the cabinet was turned about the vertical, in radians.
    """

    sample_size: int
    """
    How many rays the likelihood was reported out of, which is how far a reading is
    trusted.
    """

    mechanism_travel: float
    """
    How far the drawer ended up open, in millimetres.
    """

    arm_travel: float
    """
    How far the arm's joints moved in total over the run, in radians.
    """

    grip_offset: float
    """
    How far the gripper ended up from the handle, in millimetres.
    """

    gripper_touches_handle: bool
    """
    Whether the gripper was still in contact with the handle when the run ended.
    """

    control_cycles: int
    """
    How many control cycles the run took.
    """

    reached_its_goals: bool
    """
    Whether the motion ended by reaching its goals rather than by running out of
    cycles.
    """

    grasp_probability: Optional[float]
    """
    How likely the belief held a grasp to be when the run ended, or nothing where the
    run had no belief.
    """


# %% performing one run


@dataclass
class DrawerRun:
    """
    One condition, performed once, at one arm posture and one cabinet angle.
    """

    condition: DrawerCondition
    """
    How the grip is weighed and what the grasp is doing.
    """

    arm_configuration: ArmConfiguration
    """
    The posture the arm starts from.
    """

    cabinet_yaw: float
    """
    How far the cabinet is turned about the vertical, in radians.
    """

    belief_settings: BeliefSettings
    """
    The parameters of the belief the grip's weight follows.
    """

    sample_size: int = field(default=100, kw_only=True)
    """
    How many rays the likelihood is reported out of.
    """

    holding_share_of_hits: float = field(default=1.0, kw_only=True)
    """
    The share of rays reported while a grasp holds, defaulting to every ray hitting.
    """

    failing_share_of_hits: float = field(default=0.0, kw_only=True)
    """
    The share of rays reported while a grasp fails, defaulting to none hitting.
    """

    control_cycle_limit: int = field(default=400, kw_only=True)
    """
    How many control cycles a run may take before it counts as not having reached its
    goals.
    """

    def execute(self) -> DrawerRunOutcome:
        """
        Build the world, run the motion in it, and measure what it did.

        :return: What the run did.
        """
        scenario = DrawerWorld.of(self.cabinet_yaw, self.arm_configuration)
        belief = self._create_belief(scenario)
        statechart = self._create_statechart(scenario, belief)

        arm_positions_before = scenario.arm_positions
        executor = Executor(MotionStatechartContext(world=scenario.world))
        executor.compile(motion_statechart=statechart)
        reached_its_goals = self._tick_until_end(executor)

        return DrawerRunOutcome(
            condition=self.condition,
            arm_configuration_name=self.arm_configuration.name,
            cabinet_yaw=self.cabinet_yaw,
            sample_size=self.sample_size,
            mechanism_travel=scenario.mechanism_travel * MILLIMETRES_PER_METRE,
            arm_travel=float(
                np.abs(scenario.arm_positions - arm_positions_before).sum()
            ),
            grip_offset=scenario.grip_offset * MILLIMETRES_PER_METRE,
            gripper_touches_handle=scenario.gripper_touches_handle,
            control_cycles=int(executor.control_cycles),
            reached_its_goals=reached_its_goals,
            grasp_probability=self._published_probability(executor, belief),
        )

    def _tick_until_end(self, executor: Executor) -> bool:
        """
        Run the motion until it ends or the cycle limit is reached.

        :param executor: The executor holding the compiled motion.
        :return: Whether the motion ended by reaching its goals.

        ..note:: Running out of cycles is an outcome this experiment records rather than
            an illegal state, and :meth:`Executor.tick_until_end` reports it by raising,
            so the limit is read back out of the exception it raises.
        """
        try:
            executor.tick_until_end(timeout=self.control_cycle_limit)
        except TimeoutError:
            return False
        return True

    def _create_belief(self, scenario: DrawerWorld) -> Optional[GraspBelief]:
        """
        Build the belief the grip's weight follows, where the condition has one.

        :param scenario: The world the run is performed in.
        :return: The belief, or nothing where the condition weighs its grip by no
            evidence at all.
        """
        share_of_hits = self.condition.share_of_hits(
            self.holding_share_of_hits, self.failing_share_of_hits
        )
        if share_of_hits is None:
            return None
        return GraspBelief(
            name="grasp belief",
            grasp=Continuous("grasp_log_odds"),
            likelihood_source=ScriptedLikelihood(
                name="scripted likelihood",
                share_of_hits=share_of_hits,
                sample_size=self.sample_size,
            ),
            gripper_open=scenario.gripper_opening,
            prior_uncertainty=self.belief_settings.prior_uncertainty,
            forgetting_half_life=self.belief_settings.forgetting_half_life,
            drift=self.belief_settings.drift,
            false_below=self.belief_settings.false_below,
        )

    def _create_statechart(
        self, scenario: DrawerWorld, belief: Optional[GraspBelief]
    ) -> MotionStatechart:
        """
        Describe the motion: reach the handle, then open the drawer while holding it.

        Nothing else competes for the arm. What the grip's weight has to win or lose
        against is the goal driving the drawer, and the conflict between them is that the
        drawer slides further than the arm can follow.

        :param scenario: The world the run is performed in.
        :param belief: The belief the grip's weight follows, where there is one.
        :return: The statechart the run executes.
        """
        statechart = MotionStatechart()
        if belief is not None:
            statechart.add_node(belief.likelihood_source)
            statechart.add_node(belief)

        opening = Sequence(
            [
                CartesianPose(
                    name="reach the handle",
                    root_link=scenario.world.root,
                    tip_link=scenario.gripper,
                    goal_pose=Pose(reference_frame=scenario.handle),
                ),
                Open(
                    tip_link=scenario.gripper,
                    environment_link=scenario.handle,
                    grasp_belief=belief,
                ),
            ]
        )
        statechart.add_node(opening)
        statechart.add_node(EndMotion.when_true(opening))
        return statechart

    @staticmethod
    def _published_probability(
        executor: Executor, belief: Optional[GraspBelief]
    ) -> Optional[float]:
        """
        :param executor: The executor whose context holds the published values.
        :param belief: The belief that published one, where there is one.
        :return: How likely a grasp was when the run ended, or nothing where there was
            no belief.
        """
        if belief is None:
            return None
        return float(executor.context.float_variable_data.get_value(belief.probability))
