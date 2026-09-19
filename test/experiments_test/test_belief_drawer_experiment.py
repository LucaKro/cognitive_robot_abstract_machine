"""
Tests for the three-condition drawer sweep that judges whether weighing the Open goal's
grip by the grasp belief changes what the robot does.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.belief_drawer_experiment.drawer_run import (
    MILLIMETRES_PER_METRE,
    BeliefSettings,
    DrawerRun,
    DrawerRunOutcome,
)
from experiments.belief_drawer_experiment.drawer_scenario import (
    ArmConfiguration,
    DrawerCondition,
    DrawerWorld,
    ScriptedLikelihood,
)
from experiments.belief_drawer_experiment.sweep import (
    DrawerConditionResult,
    DrawerSweep,
    NoRunsToAggregateError,
)
from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.grasp import GraspBelief
from giskardpy.motion_statechart.beliefs.grasp_weighted_tasks import (
    GraspWeightedCartesianPose,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose

# %% what the tests run with

STRAIGHT = ArmConfiguration(name="straight", shoulder=0.0, elbow=0.0, wrist=0.0)

BELIEF_SETTINGS = BeliefSettings(
    prior_uncertainty=1.0, forgetting_half_life=0.5, drift=0.05, false_below=0.2
)


def drawer_run(condition: DrawerCondition, **overrides) -> DrawerRun:
    """
    :param condition: The condition the run is performed under.
    :param overrides: Further arguments for the run, overriding its defaults.
    :return: A run of that condition from the straight posture at a square cabinet.
    """
    return DrawerRun(
        condition=condition,
        arm_configuration=STRAIGHT,
        cabinet_yaw=0.0,
        belief_settings=BELIEF_SETTINGS,
        **overrides,
    )


def grip_task(run: DrawerRun) -> CartesianPose:
    """
    :param run: The run whose motion is inspected.
    :return: The child of its Open goal that keeps the gripper on the handle.
    """
    scenario = DrawerWorld.of(run.cabinet_yaw, run.arm_configuration)
    statechart = run._create_statechart(scenario, run._create_belief(scenario))
    context = MotionStatechartContext(world=scenario.world)
    opening = next(node for node in statechart.nodes if node.name == "Sequence")
    opening.expand(context)
    open_goal = next(node for node in opening.nodes if isinstance(node, Open))
    open_goal.expand(context)
    return next(node for node in open_goal.nodes if isinstance(node, CartesianPose))


def outcome(
    condition: DrawerCondition = DrawerCondition.UNCONDITIONAL_GRIP,
    sample_size: int = 100,
    mechanism_travel: float = 0.0,
    arm_travel: float = 0.0,
    grip_offset: float = 0.0,
    gripper_touches_handle: bool = True,
    control_cycles: int = 1,
    reached_its_goals: bool = True,
) -> DrawerRunOutcome:
    """
    A run's outcome built by hand, so aggregation can be checked against known numbers.
    """
    return DrawerRunOutcome(
        condition=condition,
        arm_configuration_name=STRAIGHT.name,
        cabinet_yaw=0.0,
        sample_size=sample_size,
        mechanism_travel=mechanism_travel,
        arm_travel=arm_travel,
        grip_offset=grip_offset,
        gripper_touches_handle=gripper_touches_handle,
        control_cycles=control_cycles,
        reached_its_goals=reached_its_goals,
        grasp_probability=None,
    )


# %% what each condition builds


def test_an_unconditional_grip_measures_no_grasp_at_all():
    """
    Stock ``Open`` reads no likelihood, so there is nothing for a scripted share of hits
    to reach.

    That is what lets one run of it stand as the baseline for a holding and a failing
    grasp alike.
    """
    assert (
        DrawerCondition.UNCONDITIONAL_GRIP.share_of_hits(holding=1.0, failing=0.0)
        is None
    )


def test_the_two_weighted_conditions_differ_only_in_what_the_rays_report():
    """
    A believed and a failing grasp are the same motion under the same belief; what
    separates them is the evidence the belief is corrected with.
    """
    assert DrawerCondition.BELIEVED_GRIP.share_of_hits(holding=1.0, failing=0.0) == 1.0
    assert DrawerCondition.FAILING_GRIP.share_of_hits(holding=1.0, failing=0.0) == 0.0
    assert DrawerCondition.BELIEVED_GRIP.weighs_the_grip_by_belief
    assert DrawerCondition.FAILING_GRIP.weighs_the_grip_by_belief


def test_an_unconditional_run_builds_no_belief():
    """
    A run that weighs its grip by nothing builds nothing to weigh it by, so the
    comparison against it is against stock ``Open`` rather than against a belief that
    happens to read one.
    """
    run = drawer_run(DrawerCondition.UNCONDITIONAL_GRIP)
    scenario = DrawerWorld.of(run.cabinet_yaw, run.arm_configuration)

    assert run._create_belief(scenario) is None
    assert type(grip_task(run)) is CartesianPose


@pytest.mark.parametrize(
    "condition", [DrawerCondition.BELIEVED_GRIP, DrawerCondition.FAILING_GRIP]
)
def test_a_weighted_run_builds_a_grip_that_follows_its_belief(condition):
    """
    Both weighted conditions hand the ``Open`` goal a belief, so the grip the goal
    builds is the one whose weight follows it.
    """
    run = drawer_run(condition)
    scenario = DrawerWorld.of(run.cabinet_yaw, run.arm_configuration)

    belief = run._create_belief(scenario)

    assert isinstance(belief, GraspBelief)
    assert belief.likelihood_source.sample_size == run.sample_size
    assert type(grip_task(run)) is GraspWeightedCartesianPose


def test_the_scripted_likelihood_publishes_the_share_it_was_given():
    """
    The share of hits a condition dictates is what reaches the belief, every cycle,
    rather than whatever a raycast would have sampled.
    """
    share_of_hits = 0.37
    source = ScriptedLikelihood(name="likelihood", share_of_hits=share_of_hits)
    statechart = MotionStatechart()
    statechart.add_node(source)
    world = DrawerWorld.of(0.0, STRAIGHT).world
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=statechart)

    executor.tick()

    assert (
        executor.context.float_variable_data.get_value(source.likelihood)
        == share_of_hits
    )


# %% the grid


def test_the_sweep_runs_every_posture_and_angle_for_every_condition():
    """
    Each condition is run once per arm posture and cabinet angle, so the conditions are
    compared on the same geometry rather than on different ones.
    """
    sweep = DrawerSweep(
        belief_settings=BELIEF_SETTINGS,
        arm_configurations=[STRAIGHT, ArmConfiguration("bent", 0.4, -0.8, 0.0)],
        cabinet_yaws=[0.0, -0.35],
        sample_sizes=[100],
    )

    runs = sweep.runs

    assert len(runs) == len(DrawerCondition) * 2 * 2
    for condition in DrawerCondition:
        geometries = {
            (run.arm_configuration.name, run.cabinet_yaw)
            for run in runs
            if run.condition is condition
        }
        assert geometries == {
            (STRAIGHT.name, 0.0),
            (STRAIGHT.name, -0.35),
            ("bent", 0.0),
            ("bent", -0.35),
        }


def test_the_sweep_runs_an_unconditional_grip_at_one_evidence_strength_only():
    """
    A grip that reads no likelihood runs the same way however many rays reported it, so
    repeating it per evidence strength would only restate the same numbers.
    """
    sweep = DrawerSweep(
        belief_settings=BELIEF_SETTINGS,
        arm_configurations=[STRAIGHT],
        cabinet_yaws=[0.0],
        sample_sizes=[100, 1000, 10000],
    )

    sample_sizes = {
        run.condition: sorted(
            {
                other.sample_size
                for other in sweep.runs
                if other.condition is run.condition
            }
        )
        for run in sweep.runs
    }

    assert sample_sizes[DrawerCondition.UNCONDITIONAL_GRIP] == [100]
    assert sample_sizes[DrawerCondition.BELIEVED_GRIP] == [100, 1000, 10000]
    assert sample_sizes[DrawerCondition.FAILING_GRIP] == [100, 1000, 10000]


# %% aggregating the runs of one cell


def test_a_row_reports_the_spread_of_the_runs_it_aggregates():
    """
    A row is the mean and standard deviation of its runs, so two postures that behave
    differently are visible rather than averaged into a single number.
    """
    outcomes = [outcome(mechanism_travel=100.0), outcome(mechanism_travel=200.0)]

    row = DrawerConditionResult.from_outcomes(
        DrawerCondition.FAILING_GRIP, sample_size=1000, outcomes=outcomes
    )

    assert row.runs == 2
    assert row.mechanism_travel.mean == 150.0
    assert row.mechanism_travel.standard_deviation == pytest.approx(
        np.std([100.0, 200.0], ddof=1), abs=1e-4
    )
    assert row.condition is DrawerCondition.FAILING_GRIP
    assert row.sample_size == 1000


def test_a_row_counts_the_runs_that_kept_hold_and_the_runs_that_finished():
    """
    Whether a grip kept hold is a count of runs rather than an average, because contact
    is something a run either had or did not.
    """
    outcomes = [
        outcome(gripper_touches_handle=True, reached_its_goals=True),
        outcome(gripper_touches_handle=False, reached_its_goals=True),
        outcome(gripper_touches_handle=False, reached_its_goals=False),
    ]

    row = DrawerConditionResult.from_outcomes(
        DrawerCondition.FAILING_GRIP, sample_size=100, outcomes=outcomes
    )

    assert row.runs_still_touching_the_handle == 1
    assert row.runs_that_reached_their_goals == 2


def test_aggregating_no_runs_says_which_cell_is_missing():
    """
    A cell nothing was run at has nothing to report, and saying which cell it is beats a
    row of zeros that reads like a measurement.
    """
    with pytest.raises(NoRunsToAggregateError) as raised:
        DrawerConditionResult.from_outcomes(
            DrawerCondition.BELIEVED_GRIP, sample_size=1000, outcomes=[]
        )

    assert raised.value.condition is DrawerCondition.BELIEVED_GRIP
    assert raised.value.sample_size == 1000


def test_the_table_has_one_row_per_condition_and_evidence_strength():
    """
    Runs are compared per cell of the grid, so the runs of one condition at one evidence
    strength become one row rather than being pooled across strengths.
    """
    sweep = DrawerSweep(belief_settings=BELIEF_SETTINGS)
    sweep.outcomes = [
        outcome(DrawerCondition.UNCONDITIONAL_GRIP, sample_size=100),
        outcome(DrawerCondition.FAILING_GRIP, sample_size=100),
        outcome(DrawerCondition.FAILING_GRIP, sample_size=1000),
        outcome(DrawerCondition.FAILING_GRIP, sample_size=1000),
    ]

    rows = sweep.as_table().experiments

    assert [(row.condition, row.sample_size, row.runs) for row in rows] == [
        (DrawerCondition.UNCONDITIONAL_GRIP, 100, 1),
        (DrawerCondition.FAILING_GRIP, 100, 1),
        (DrawerCondition.FAILING_GRIP, 1000, 2),
    ]


def test_a_row_is_recorded_flat_with_its_condition_by_name():
    """
    The manifest beside the table carries the same columns the table presents, with the
    condition written as the member it is rather than as a number.
    """
    sweep = DrawerSweep(belief_settings=BELIEF_SETTINGS)
    sweep.outcomes = [outcome(DrawerCondition.BELIEVED_GRIP, mechanism_travel=42.0)]

    [row] = sweep.as_table().as_json_rows()

    assert row["condition"] == DrawerCondition.BELIEVED_GRIP.name
    assert row["mechanism_travel"]["mean"] == 42.0
    assert set(row) == set(DrawerConditionResult.get_column_names())


# %% what a run measures


def test_a_run_that_runs_out_of_cycles_records_that_rather_than_raising():
    """
    A motion that never finishes is an outcome the sweep reports, not a failure that
    ends it, so the cell it belongs to still gets its row.
    """
    result = drawer_run(
        DrawerCondition.UNCONDITIONAL_GRIP, control_cycle_limit=3
    ).execute()

    assert result.reached_its_goals is False
    assert result.control_cycles == 3


def test_a_run_reports_its_distances_in_millimetres():
    """
    A drawer travels tenths of a metre and a grip is left behind by thousandths of one,
    and the table reports to two decimals, so the world's metres are converted once.
    """
    run = drawer_run(DrawerCondition.UNCONDITIONAL_GRIP, control_cycle_limit=3)
    scenario = DrawerWorld.of(run.cabinet_yaw, run.arm_configuration)

    result = run.execute()

    assert result.grip_offset == pytest.approx(
        scenario.grip_offset * MILLIMETRES_PER_METRE, abs=1.0
    )


def test_an_unconditional_run_reports_no_probability():
    """
    A run without a belief has no probability to report, and reporting none is what
    keeps a reader from comparing it against one.
    """
    result = drawer_run(
        DrawerCondition.UNCONDITIONAL_GRIP, control_cycle_limit=3
    ).execute()

    assert result.grasp_probability is None


# %% the measurements the experiment exists to make


@pytest.mark.slow
def test_a_confirmed_grasp_opens_the_drawer_exactly_as_stock_open_does():
    """
    Weighing the grip by a belief that keeps being confirmed changes nothing, which is
    what an opt-in change has to be able to claim.
    """
    unconditional = drawer_run(DrawerCondition.UNCONDITIONAL_GRIP).execute()
    believed = drawer_run(DrawerCondition.BELIEVED_GRIP).execute()

    assert believed.mechanism_travel == pytest.approx(
        unconditional.mechanism_travel, abs=1.0
    )
    assert believed.gripper_touches_handle is unconditional.gripper_touches_handle
    assert believed.grasp_probability > 0.9


@pytest.mark.slow
def test_a_failed_grasp_only_lets_go_once_it_outweighs_the_mechanism():
    """
    The deciding measurement.

    A belief-scaled grip carries ``WEIGHT_ABOVE_COLLISION_AVOIDANCE`` times the
    probability of a grasp, so it keeps outranking the mechanism until that probability
    falls below one in two and a half thousand. A hundred rays reporting no hit at all
    do not reach that, and the grip keeps dragging the arm after a drawer it is not
    holding; ten thousand do, and it gives way.
    """
    weakly_evidenced = drawer_run(
        DrawerCondition.FAILING_GRIP, sample_size=100
    ).execute()
    strongly_evidenced = drawer_run(
        DrawerCondition.FAILING_GRIP, sample_size=10000
    ).execute()

    assert weakly_evidenced.gripper_touches_handle is True
    assert strongly_evidenced.gripper_touches_handle is False
    assert strongly_evidenced.mechanism_travel > weakly_evidenced.mechanism_travel
    assert strongly_evidenced.arm_travel < weakly_evidenced.arm_travel
