"""
Judge whether weighing a carry by the grasp belief changes what the robot does, by
running the same pick-and-place under four conditions in a physics simulation.

Run the whole sweep with::

    python -m experiments.simulated_grasp.belief_experiment

Two things vary. The carry either carries its full weight whatever the evidence says,
which is what a goal does today, or it carries as much of it as the grasp is believed
in. And the block either sits where the motion expects it or sits clear of the gripper,
so that the fingers close on nothing while the robot has every reason to think they
did not.

Every outcome is read from the physics — whether the block left the table, whether both
fingers were on it, how far it ended from where it was to be put down, and how far the
hand ended from over that place. The belief's own probability is reported beside them so
a reader can see why a run behaved as it did, and is never what a run is judged by.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, field
from enum import StrEnum

from random_events.variable import Continuous
from typing_extensions import Any, Dict, List, Optional

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from experiments.simulated_grasp.contact_likelihood import (
    ContactLikelihood,
    ContactSensor,
    SimulatedContactSensor,
)
from experiments.simulated_grasp.grasp_attempt import (
    GraspOutcome,
    GraspPhase,
    PLACED_ACROSS_THE_TABLE,
    PhysicalGrasp,
)
from experiments.simulated_grasp.panda_world import (
    OPEN_FINGER_OFFSET,
    READY_POSTURE,
    PandaPartName,
    PandaWorld,
)
from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.grasp import GraspBelief
from giskardpy.motion_statechart.beliefs.grasp_weighted_tasks import (
    GraspWeightedCartesianPosition,
)
from giskardpy.motion_statechart.data_types import DefaultWeights, LifeCycleValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood.exceptions import DataclassException
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types.spatial_types import Point3

# %% what one run varies

MISSED_BY = 0.10
"""
How far across the table the block sits from where the motion expects it, in a condition
where the grasp is to fail, in metres.

Wider than the gripper's own opening, so the fingers close well clear of the block
rather than on its edge, and the physics reports no contact at all rather than a
marginal one.
"""

CARRY_THRESHOLD = 0.03
"""
How close a step that carries the block has to come to its goal before it counts as
arrived, in metres.

Wider than a step with nothing pulling against it, because a carry is performed against
the alternative of putting the arm back where it started: even a carry worth two and a
half thousand times that alternative settles about fourteen millimetres short of its
goal on this arm, which a threshold of one centimetre never reaches.
"""

RETURN_TO_READY = "return to ready"
"""
What the task competing with the carry is called.
"""


class PickupCondition(StrEnum):
    """
    How a run weighs the carry, and what the gripper closed on before it began.
    """

    UNCONDITIONAL_ON_THE_BLOCK = "unconditional_on_the_block"
    """
    The carry keeps its full weight, and the gripper did close on the block.

    This is what the stack does today when everything goes right.
    """

    BELIEVED_ON_THE_BLOCK = "believed_on_the_block"
    """
    The carry's weight follows the belief, and the gripper did close on the block.

    Nothing should change against the unconditional run, which is what makes weighing by
    a belief safe to opt into.
    """

    UNCONDITIONAL_ON_NOTHING = "unconditional_on_nothing"
    """
    The carry keeps its full weight, and the gripper closed on nothing.

    The arm performs the whole carry holding air, because a weight that reads no
    evidence cannot represent a grasp that failed.
    """

    BELIEVED_ON_NOTHING = "believed_on_nothing"
    """
    The carry's weight follows the belief, and the gripper closed on nothing.

    This is the
    cell the experiment exists to decide: whether the carry gives way on its own.
    """

    @property
    def weighs_the_carry_by_belief(self) -> bool:
        """
        :return: Whether the steps that only make sense while the block is held weigh
            themselves by how likely it is that it is.
        """
        return self in (
            PickupCondition.BELIEVED_ON_THE_BLOCK,
            PickupCondition.BELIEVED_ON_NOTHING,
        )

    @property
    def closes_on_the_block(self) -> bool:
        """
        :return: Whether the block is where the motion expects it, so that the gripper
            closes on it rather than beside it.
        """
        return self in (
            PickupCondition.UNCONDITIONAL_ON_THE_BLOCK,
            PickupCondition.BELIEVED_ON_THE_BLOCK,
        )

    @property
    def block_offset(self) -> float:
        """
        :return: How far across the table the block sits from where the motion expects
            it, in metres.
        """
        return 0.0 if self.closes_on_the_block else MISSED_BY


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


# %% one run


@dataclass
class BelievedPickup(PhysicalGrasp):
    """
    The pick-and-place, performed under one condition, with a belief about the grasp
    kept throughout and a posture to fall back to while carrying.

    The belief is kept under every condition, so the only thing that differs between an
    unconditional run and a believed one is whether the carry reads it. What the carry
    has to be worth more than is returning to the posture the arm started in: with
    nothing to lose against, a weight changes nothing at all.
    """

    condition: PickupCondition = field(kw_only=True)
    """
    How the carry is weighed, and what the gripper closes on.
    """

    belief_settings: BeliefSettings = field(kw_only=True)
    """
    The parameters of the belief the carry's weight follows.
    """

    sample_size: int = field(kw_only=True)
    """
    How many independent samples one cycle of measured contact is trusted as.
    """

    return_to_ready_weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    What going back to the arm's starting posture is worth while the block is being
    carried, which is what the carry gives way to once it is believed in less than this.
    """

    carry_threshold: float = field(default=CARRY_THRESHOLD, kw_only=True)
    """
    How close a step that carries the block has to come to its goal before it counts as
    arrived, in metres.
    """

    _belief: Optional[GraspBelief] = field(default=None, init=False, repr=False)
    """
    The belief this run keeps, built with the statechart.
    """

    _carrying: Optional[MotionStatechartNode] = field(
        default=None, init=False, repr=False
    )
    """
    The step that takes the block off the table, whose first cycle is when the carry is
    decided.
    """

    _probability_when_the_carry_began: Optional[float] = field(
        default=None, init=False, repr=False
    )
    """
    How likely a grasp was when the carry began, kept because that is the moment the
    carry's weight decided whether to perform it.
    """

    def __post_init__(self) -> None:
        """
        Put the block where the condition says it is, rather than where a caller does.
        """
        self.block_offset = self.condition.block_offset

    def _create_statechart(self, scenario: PandaWorld) -> MotionStatechart:
        self._belief = self._create_belief(scenario)
        return super()._create_statechart(scenario)

    def _create_belief(self, scenario: PandaWorld) -> GraspBelief:
        """
        Build the belief about whether the block is held, fed by what the physics
        reports between the fingers.

        :param scenario: The world the run is performed in.
        :return: The belief.
        """
        return GraspBelief(
            name="grasp belief",
            grasp=Continuous("grasp_log_odds"),
            likelihood_source=ContactLikelihood(
                name="measured likelihood",
                contacts=self._contact_sensor(),
                held_body=PandaPartName.BLOCK,
                fingers=[PandaPartName.LEFT_FINGER, PandaPartName.RIGHT_FINGER],
                sample_size=self.sample_size,
            ),
            gripper_open=JointState.from_mapping(
                {finger: OPEN_FINGER_OFFSET for finger in scenario.fingers}
            ),
            prior_uncertainty=self.belief_settings.prior_uncertainty,
            forgetting_half_life=self.belief_settings.forgetting_half_life,
            drift=self.belief_settings.drift,
            false_below=self.belief_settings.false_below,
        )

    def _contact_sensor(self) -> ContactSensor:
        """
        :return: Where contact between the fingers and the block is read from, which is
            the physics the run is judged by.
        """
        return SimulatedContactSensor(simulation=self._simulation)

    def _position_task(
        self, scenario: PandaWorld, phase: GraspPhase, goal_point: Point3
    ) -> CartesianPosition:
        if not phase.carries_the_block:
            return super()._position_task(scenario, phase, goal_point)
        arguments: Dict[str, Any] = dict(
            name=f"{phase}/position",
            root_link=scenario.world.root,
            tip_link=scenario.tool_frame,
            goal_point=goal_point,
            threshold=self.carry_threshold,
        )
        if not self.condition.weighs_the_carry_by_belief:
            return CartesianPosition(**arguments)
        return GraspWeightedCartesianPosition(**arguments, grasp_belief=self._belief)

    def _competing_nodes(
        self, scenario: PandaWorld, steps: Dict[GraspPhase, MotionStatechartNode]
    ) -> List[MotionStatechartNode]:
        self._carrying = steps[GraspPhase.LIFTING]
        return [
            self._belief.likelihood_source,
            self._belief,
            self._return_to_ready(scenario, steps),
        ]

    def _return_to_ready(
        self, scenario: PandaWorld, steps: Dict[GraspPhase, MotionStatechartNode]
    ) -> JointPositionList:
        """
        Build what the carry has to outweigh: going back to the posture the arm started
        in.

        It runs from the moment the grip has settled until the block is over where it is
        to be put down, so that it is the alternative to carrying rather than an
        alternative to reaching, to lowering the block or to letting go.

        :param scenario: The world the run is performed in.
        :param steps: The motion's own steps, by the phase each performs.
        :return: The task.
        """
        returning = JointPositionList(
            name=RETURN_TO_READY,
            goal_state=JointState.from_mapping(dict(zip(scenario.arm, READY_POSTURE))),
            weight=self.return_to_ready_weight,
        )
        returning.start_condition = steps[GraspPhase.GRIPPING].is_succeeded
        returning.end_condition = steps[GraspPhase.ABOVE_THE_TARGET].is_succeeded
        return returning

    def _observe_cycle(self, executor: Executor) -> None:
        """
        Keep how likely a grasp was on the first cycle of the carry.

        :param executor: The executor that has just ticked.
        """
        if self._probability_when_the_carry_began is not None:
            return
        if self._carrying.life_cycle_state != LifeCycleValues.RUNNING:
            return
        self._probability_when_the_carry_began = float(
            executor.context.float_variable_data.get_value(self._belief.probability)
        )

    def _grasp_probability(self, executor: Executor) -> Optional[float]:
        return self._probability_when_the_carry_began


# %% one cell of the grid, aggregated over its runs


@dataclass
class NoRunsToAggregateError(DataclassException):
    """
    Raised when a row is built over no runs, which has nothing to report.
    """

    condition: PickupCondition
    """
    The condition whose runs are missing.
    """

    sample_size: int
    """
    The evidence strength whose runs are missing.
    """

    def error_message(self) -> str:
        return (
            f"No run of {self.condition} at a sample size of {self.sample_size} was "
            f"performed, so there is nothing to aggregate."
        )

    def suggest_correction(self) -> str:
        return "Perform at least one run of a condition before aggregating it."


@dataclass(frozen=True)
class PickupCell:
    """
    One condition at one evidence strength, which is what a row of the comparison
    reports.
    """

    condition: PickupCondition
    """
    How the carry is weighed and what the gripper closed on.
    """

    sample_size: int
    """
    How many independent samples one cycle of measured contact is trusted as.
    """


@dataclass
class PickupConditionResult(ExperimentResult):
    """
    What one condition did at one evidence strength, over every place the block was put.
    """

    condition: PickupCondition
    """
    How the carry was weighed and what the gripper closed on.
    """

    sample_size: int
    """
    How many independent samples one cycle of measured contact was trusted as, which an
    unconditional carry does not read.
    """

    runs: int
    """
    How many runs the row aggregates.
    """

    runs_that_lifted_the_block: int
    """
    In how many runs the block left the table.
    """

    block_travelled: MeanAndStandardDeviation
    """
    How far the block ended from where it started, in millimetres.
    """

    gripper_offset_from_the_target: MeanAndStandardDeviation
    """
    How far the hand ended from over the place the block was to be put down, in
    millimetres, which is what says whether the arm performed the carry at all.
    """

    grasp_probability: MeanAndStandardDeviation
    """
    How likely the belief held a grasp to be when the runs ended.
    """

    runs_that_reached_their_goals: int
    """
    In how many runs the motion ended by reaching its goals rather than by running out
    of cycles.
    """

    @classmethod
    def from_outcomes(
        cls, cell: PickupCell, outcomes: List[GraspOutcome]
    ) -> PickupConditionResult:
        """
        Aggregate every run of one cell into a single row.

        :param cell: The condition and evidence strength the runs were performed at.
        :param outcomes: What each of those runs did.
        :return: The row reporting them.
        :raises NoRunsToAggregateError: If there is not a single run to aggregate.
        """
        if not outcomes:
            raise NoRunsToAggregateError(
                condition=cell.condition, sample_size=cell.sample_size
            )
        return cls(
            condition=cell.condition,
            sample_size=cell.sample_size,
            runs=len(outcomes),
            runs_that_lifted_the_block=sum(
                outcome.block_was_lifted for outcome in outcomes
            ),
            block_travelled=MeanAndStandardDeviation.from_measurements(
                [outcome.travelled for outcome in outcomes]
            ),
            gripper_offset_from_the_target=MeanAndStandardDeviation.from_measurements(
                [outcome.gripper_offset_from_the_target for outcome in outcomes]
            ),
            grasp_probability=MeanAndStandardDeviation.from_measurements(
                [outcome.grasp_probability for outcome in outcomes]
            ),
            runs_that_reached_their_goals=sum(
                outcome.reached_its_goals for outcome in outcomes
            ),
        )


# %% the grid


DEFAULT_BLOCK_DISTANCES: List[float] = [0.48, 0.53, 0.58]
"""
How far in front of the arm the block is put, in metres.

Spread so that a result holds over the arm's configuration rather than at one geometry,
and kept inside the span where the arm completes the carry with nothing pulling against
it: past about six tenths of a metre it runs out of reach on the way across the table,
and a carry that fails for that reason would be the experiment's own doing rather than
the belief's.
"""

DEFAULT_SAMPLE_SIZES: List[int] = [2, 20, 200, 2000]
"""
How many independent samples one cycle of measured contact is trusted as.

A belief-weighted carry is worth ``WEIGHT_ABOVE_COLLISION_AVOIDANCE`` times the
probability of a grasp, so it stops outweighing a return to the arm's starting posture
only once that probability falls below one in two and a half thousand. How low a
measurement of nothing touching can drive it depends on how much that measurement is
trusted, which is why the evidence strength is swept rather than fixed.
"""

CONTROL_CYCLE_LIMIT = 500
"""
How many control cycles a run may take before it counts as not having reached its goals.

A carry that gives way never finishes its steps, so a run that backs off spends the
whole limit, which is an outcome rather than a fault.
"""


@dataclass
class PickupSweep:
    """
    Every condition, at every evidence strength and every place the block is put.
    """

    belief_settings: BeliefSettings
    """
    The parameters of the belief every run keeps.
    """

    conditions: List[PickupCondition] = field(
        default_factory=lambda: list(PickupCondition)
    )
    """
    The conditions that are compared.
    """

    block_distances: List[float] = field(
        default_factory=lambda: list(DEFAULT_BLOCK_DISTANCES)
    )
    """
    How far in front of the arm the block is put, in metres.
    """

    sample_sizes: List[int] = field(default_factory=lambda: list(DEFAULT_SAMPLE_SIZES))
    """
    How many independent samples one cycle of measured contact is trusted as.
    """

    control_cycle_limit: int = CONTROL_CYCLE_LIMIT
    """
    How many control cycles a run may take before it counts as not having reached its
    goals.
    """

    rows: List[PickupConditionResult] = field(default_factory=list)
    """
    What every cell run so far did.
    """

    @property
    def cells(self) -> List[PickupCell]:
        """
        :return: Every cell of the grid, one per row of the comparison.
        """
        return [
            PickupCell(condition=condition, sample_size=sample_size)
            for condition in self.conditions
            for sample_size in self._sample_sizes_for(condition)
        ]

    def _sample_sizes_for(self, condition: PickupCondition) -> List[int]:
        """
        :param condition: The condition the runs are performed under.
        :return: The evidence strengths it is run at. A carry that does not read the
            belief runs the same way at every one of them, so it is run at only the
            first — which is what makes it the baseline for the believed runs rather
            than a second set of numbers to compare against.
        """
        if not condition.weighs_the_carry_by_belief:
            return self.sample_sizes[:1]
        return self.sample_sizes

    def runs(self, cell: PickupCell) -> List[BelievedPickup]:
        """
        :param cell: The condition and evidence strength to perform.
        :return: One run per place the block is put.
        """
        return [
            self.run(cell, block_distance) for block_distance in self.block_distances
        ]

    def run(self, cell: PickupCell, block_distance: float) -> BelievedPickup:
        """
        :param cell: The condition and evidence strength to perform.
        :param block_distance: How far in front of the arm the block is put, in metres.
        :return: The run.
        """
        return BelievedPickup(
            condition=cell.condition,
            belief_settings=self.belief_settings,
            sample_size=cell.sample_size,
            block_distance=block_distance,
            target=PandaWorld.resting_place(block_distance, PLACED_ACROSS_THE_TABLE),
            control_cycle_limit=self.control_cycle_limit,
        )

    def record(self, directory: pathlib.Path, sample_size: int) -> List[pathlib.Path]:
        """
        Run every condition once, at one evidence strength and one place, and record
        what each one looked like.

        :param directory: Where the recordings are written.
        :param sample_size: How much one cycle of measured contact is trusted, which
            only the believed conditions read.
        :return: Where each recording was written, in the order the conditions are
            compared.
        """
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for condition in self.conditions:
            run = self.run(
                PickupCell(condition=condition, sample_size=sample_size),
                self.block_distances[len(self.block_distances) // 2],
            )
            run.video_path = directory / f"{condition}.mp4"
            print(f"recording {condition}")
            run.execute()
            written.append(run.video_path)
        return written

    def execute(self) -> None:
        """
        Perform every run of the grid and keep what each cell did.
        """
        for cell in self.cells:
            outcomes = []
            for run in self.runs(cell):
                print(
                    f"running {cell.condition} at {cell.sample_size} samples, "
                    f"block {run.block_distance:.2f} m out"
                )
                outcomes.append(run.execute())
            self.rows.append(PickupConditionResult.from_outcomes(cell, outcomes))

    def as_table(self) -> ExperimentsTable:
        """
        :return: One row per condition and evidence strength, in the order they were
            run.
        """
        return ExperimentsTable(list(self.rows))

    def render_figure(self) -> str:
        """
        :return: The table a reader sees, captioned with what it shows.
        """
        return TypstRenderer(self.as_table()).render_figure(
            "What weighing a carry by the grasp belief changed, over "
            f"{len(self.block_distances)} places the block was put per condition. "
            "Every column but the last is measured in the physics rather than read off "
            "the belief."
        )


# %% command line


def parse_arguments() -> argparse.Namespace:
    """
    Describe what the sweep can be run with.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-uncertainty",
        type=float,
        default=1.0,
        help="how uncertain the belief's prior is, in log-odds",
    )
    parser.add_argument(
        "--forgetting-half-life",
        type=float,
        default=0.5,
        help="seconds an open gripper takes to carry the estimate halfway back",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.05,
        help="how much uncertainty a second adds on its own",
    )
    parser.add_argument(
        "--false-below",
        type=float,
        default=0.2,
        help="probability of a grasp under which the belief reports there is none",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SIZES,
        help="how many samples one cycle of measured contact is trusted as",
    )
    parser.add_argument(
        "--control-cycle-limit",
        type=int,
        default=CONTROL_CYCLE_LIMIT,
        help="how many cycles a run may take before it counts as not having finished",
    )
    parser.add_argument(
        "--write-manifest-to",
        default=None,
        help="directory the rows are also written to as JSON",
    )
    parser.add_argument(
        "--record-to",
        default=None,
        help="directory to record each condition into, in place of running the sweep",
    )
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> None:
    """
    Run every condition of the sweep and print the comparison.

    :param arguments: What the sweep was asked to run with.
    """
    sweep = PickupSweep(
        belief_settings=BeliefSettings(
            prior_uncertainty=arguments.prior_uncertainty,
            forgetting_half_life=arguments.forgetting_half_life,
            drift=arguments.drift,
            false_below=arguments.false_below,
        ),
        sample_sizes=arguments.sample_sizes,
        control_cycle_limit=arguments.control_cycle_limit,
    )
    if arguments.record_to is not None:
        for path in sweep.record(
            pathlib.Path(arguments.record_to), arguments.sample_sizes[-1]
        ):
            print(f"recorded to {path}")
        return
    sweep.execute()
    print()
    print(sweep.render_figure())
    if arguments.write_manifest_to is not None:
        path = sweep.as_table().write_manifest(
            pathlib.Path(arguments.write_manifest_to), "belief_pickup_experiment.json"
        )
        print(f"rows written to {path}")


if __name__ == "__main__":
    main(parse_arguments())
