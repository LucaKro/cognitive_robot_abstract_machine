"""
Judge whether weighing the ``Open`` goal's grip by the grasp belief changes what the
robot does, by running three conditions over a grid of arm postures, cabinet angles and
evidence strengths.

Run the whole sweep with::

    python -m experiments.belief_drawer_experiment.sweep

Every outcome is read off the world the motion ran in — how far the drawer opened, how
far the arm moved, how far the gripper ended from the handle, and whether the two are
still in contact. The belief's own probability is reported beside them so a reader can
see why a run behaved as it did, and is never what a run is judged by.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
from dataclasses import dataclass, field

from typing_extensions import List

from experiments.belief_drawer_experiment.drawer_run import (
    BeliefSettings,
    DrawerRun,
    DrawerRunOutcome,
)
from experiments.belief_drawer_experiment.drawer_scenario import (
    ArmConfiguration,
    DrawerCondition,
)
from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from krrood.exceptions import DataclassException

# %% one cell of the grid, aggregated over its runs


@dataclass
class NoRunsToAggregateError(DataclassException):
    """
    Raised when a row is built over no runs, which has nothing to report.
    """

    condition: DrawerCondition
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


@dataclass
class DrawerConditionResult(ExperimentResult):
    """
    What one condition did at one evidence strength, over every arm posture and cabinet
    angle it was run at.
    """

    condition: DrawerCondition
    """
    How the grip was weighed and what the grasp was doing.
    """

    sample_size: int
    """
    How many rays the likelihood was reported out of, which an unconditional grip does
    not read.
    """

    runs: int
    """
    How many runs the row aggregates.
    """

    mechanism_travel: MeanAndStandardDeviation
    """
    How far the drawer ended up open, in millimetres.
    """

    arm_travel: MeanAndStandardDeviation
    """
    How far the arm's joints moved in total, in radians.
    """

    grip_offset: MeanAndStandardDeviation
    """
    How far the gripper ended up from the handle, in millimetres.
    """

    runs_still_touching_the_handle: int
    """
    In how many runs the gripper was still in contact with the handle at the end.
    """

    runs_that_reached_their_goals: int
    """
    In how many runs the motion ended by reaching its goals rather than by running out
    of cycles.
    """

    @classmethod
    def from_outcomes(
        cls,
        condition: DrawerCondition,
        sample_size: int,
        outcomes: List[DrawerRunOutcome],
    ) -> DrawerConditionResult:
        """
        Aggregate every run of one condition at one evidence strength into a single row.

        :param condition: The condition the runs were performed under.
        :param sample_size: The evidence strength they were performed at.
        :param outcomes: What each of those runs did.
        :return: The row reporting them.
        :raises NoRunsToAggregateError: If there is not a single run to aggregate.
        """
        if not outcomes:
            raise NoRunsToAggregateError(condition=condition, sample_size=sample_size)
        return cls(
            condition=condition,
            sample_size=sample_size,
            runs=len(outcomes),
            mechanism_travel=MeanAndStandardDeviation.from_measurements(
                [outcome.mechanism_travel for outcome in outcomes]
            ),
            arm_travel=MeanAndStandardDeviation.from_measurements(
                [outcome.arm_travel for outcome in outcomes]
            ),
            grip_offset=MeanAndStandardDeviation.from_measurements(
                [outcome.grip_offset for outcome in outcomes]
            ),
            runs_still_touching_the_handle=sum(
                outcome.gripper_touches_handle for outcome in outcomes
            ),
            runs_that_reached_their_goals=sum(
                outcome.reached_its_goals for outcome in outcomes
            ),
        )


# %% the grid


DEFAULT_ARM_CONFIGURATIONS: List[ArmConfiguration] = [
    ArmConfiguration(name="straight", shoulder=0.0, elbow=0.0, wrist=0.0),
    ArmConfiguration(name="elbow_out", shoulder=0.4, elbow=-0.8, wrist=0.0),
    ArmConfiguration(name="elbow_in", shoulder=-0.3, elbow=0.6, wrist=0.0),
]
"""
The postures the arm is started from, spanning the two elbow solutions and the stretched
configuration between them.
"""

DEFAULT_CABINET_YAWS: List[float] = [0.0, -0.35, 0.35]
"""
The angles the cabinet is turned by, in radians, following the published drawer
tutorial's own use of a yawed cabinet as the configuration that decides an outcome.
"""

DEFAULT_SAMPLE_SIZES: List[int] = [100, 1000, 10000]
"""
How many rays a reading is reported out of.

A belief-scaled grip carries ``WEIGHT_ABOVE_COLLISION_AVOIDANCE`` times the probability
of a grasp, so it stops outranking the mechanism's ``WEIGHT_BELOW_COLLISION_AVOIDANCE``
only below a probability of one in two and a half thousand. How low a failed grasp can
drive that probability depends on how many rays said so, which is why the evidence
strength is swept rather than fixed.
"""


@dataclass
class DrawerSweep:
    """
    Every condition, at every arm posture, cabinet angle and evidence strength.
    """

    belief_settings: BeliefSettings
    """
    The parameters of the belief every weighted run filters its likelihood with.
    """

    conditions: List[DrawerCondition] = field(
        default_factory=lambda: list(DrawerCondition)
    )
    """
    The conditions that are compared.
    """

    arm_configurations: List[ArmConfiguration] = field(
        default_factory=lambda: list(DEFAULT_ARM_CONFIGURATIONS)
    )
    """
    The postures the arm is started from.
    """

    cabinet_yaws: List[float] = field(
        default_factory=lambda: list(DEFAULT_CABINET_YAWS)
    )
    """
    The angles the cabinet is turned by, in radians.
    """

    sample_sizes: List[int] = field(default_factory=lambda: list(DEFAULT_SAMPLE_SIZES))
    """
    How many rays a reading is reported out of.
    """

    control_cycle_limit: int = 400
    """
    How many control cycles a run may take before it counts as not having reached its
    goals.
    """

    outcomes: List[DrawerRunOutcome] = field(default_factory=list)
    """
    What every run performed so far did.
    """

    @property
    def runs(self) -> List[DrawerRun]:
        """
        :return: Every run the sweep performs, one per cell of the grid.
        """
        return [
            DrawerRun(
                condition=condition,
                arm_configuration=arm_configuration,
                cabinet_yaw=cabinet_yaw,
                belief_settings=self.belief_settings,
                sample_size=sample_size,
                control_cycle_limit=self.control_cycle_limit,
            )
            for condition in self.conditions
            for sample_size in self._sample_sizes_for(condition)
            for arm_configuration, cabinet_yaw in itertools.product(
                self.arm_configurations, self.cabinet_yaws
            )
        ]

    def _sample_sizes_for(self, condition: DrawerCondition) -> List[int]:
        """
        :param condition: The condition the runs are performed under.
        :return: The evidence strengths it is run at. A grip that reads no likelihood
            runs the same way at every one of them, so it is run at only the first —
            which is what makes it the baseline for both other conditions at once
            rather than a third set of numbers to compare against.
        """
        if not condition.weighs_the_grip_by_belief:
            return self.sample_sizes[:1]
        return self.sample_sizes

    def execute(self) -> None:
        """
        Perform every run of the grid and keep what each one did.
        """
        for run in self.runs:
            print(
                f"running {run.condition} at {run.arm_configuration.name}, "
                f"yaw {run.cabinet_yaw:+.2f}, {run.sample_size} rays"
            )
            self.outcomes.append(run.execute())

    def as_table(self) -> ExperimentsTable:
        """
        :return: One row per condition and evidence strength, in the order they were
            run.
        """
        return ExperimentsTable(
            [
                DrawerConditionResult.from_outcomes(
                    condition, sample_size, cell_outcomes
                )
                for (
                    condition,
                    sample_size,
                ), cell_outcomes in self._outcomes_by_cell().items()
            ]
        )

    def _outcomes_by_cell(self) -> dict:
        """
        :return: The runs of each condition and evidence strength, keyed by the two
            together, in the order they were first seen.
        """
        cells: dict = {}
        for outcome in self.outcomes:
            cells.setdefault((outcome.condition, outcome.sample_size), []).append(
                outcome
            )
        return cells

    def render_figure(self) -> str:
        """
        :return: The table a reader sees, captioned with what it shows.
        """
        return TypstRenderer(self.as_table()).render_figure(
            "What weighing the Open goal's grip by the grasp belief changed, over "
            f"{len(self.arm_configurations)} arm postures and "
            f"{len(self.cabinet_yaws)} cabinet angles per condition. Every column is "
            "measured against the world rather than against the belief."
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
        help="how many rays a reading is reported out of",
    )
    parser.add_argument(
        "--control-cycle-limit",
        type=int,
        default=400,
        help="how many cycles a run may take before it counts as not having finished",
    )
    parser.add_argument(
        "--write-manifest-to",
        default=None,
        help="directory the rows are also written to as JSON",
    )
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> None:
    """
    Run every condition of the sweep and print the comparison.

    :param arguments: What the sweep was asked to run with.
    """
    sweep = DrawerSweep(
        belief_settings=BeliefSettings(
            prior_uncertainty=arguments.prior_uncertainty,
            forgetting_half_life=arguments.forgetting_half_life,
            drift=arguments.drift,
            false_below=arguments.false_below,
        ),
        sample_sizes=arguments.sample_sizes,
        control_cycle_limit=arguments.control_cycle_limit,
    )
    sweep.execute()
    print()
    print(sweep.render_figure())
    if arguments.write_manifest_to is not None:
        path = sweep.as_table().write_manifest(
            pathlib.Path(arguments.write_manifest_to), "belief_drawer_experiment.json"
        )
        print(f"rows written to {path}")


if __name__ == "__main__":
    main(parse_arguments())
