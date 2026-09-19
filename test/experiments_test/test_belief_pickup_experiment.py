"""
Tests for :mod:`experiments.simulated_grasp.belief_experiment`.

How a condition is put together, what a measured likelihood reports and how a run reads
it are all answerable without physics. Whether weighing the carry by the belief changes
what the robot does is not, so those tests start MuJoCo and follow the same continuous-
integration gating as every other simulator-backed test in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from typing_extensions import Set

from ..pytest_environment import runs_in_continuous_integration

from experiments.simulated_grasp.belief_experiment import (
    BeliefSettings,
    BelievedPickup,
    MISSED_BY,
    NoRunsToAggregateError,
    PickupCell,
    PickupCondition,
    PickupConditionResult,
    PickupSweep,
    RETURN_TO_READY,
)
from experiments.simulated_grasp.contact_likelihood import (
    ContactLikelihood,
    ContactSensor,
    FINGERS_OF_A_PARALLEL_GRIPPER,
)
from experiments.simulated_grasp.grasp_attempt import (
    GraspOutcome,
    GraspPhase,
    PLACED_ACROSS_THE_TABLE,
)
from experiments.simulated_grasp.panda_world import (
    BLOCK_SIDE,
    OPEN_FINGER_OFFSET,
    PandaPartName,
    PandaWorld,
)
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

requires_mujoco = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

BELIEF_SETTINGS = BeliefSettings(
    prior_uncertainty=1.0, forgetting_half_life=0.5, drift=0.05, false_below=0.2
)
"""
What every run below filters its likelihood with, so that a difference between two runs
is never the belief's own parameters.
"""

# %% contact the test decides, in place of a simulation


@dataclass
class RecordedContacts(ContactSensor):
    """
    Reports the contact a test states, so that what is read off it can be asserted
    without a physics engine deciding it.
    """

    touching: Set[str] = field(default_factory=set)
    """
    What the body in question is currently touching.
    """

    def bodies_touching(self, body_name: str) -> Set[str]:
        return set(self.touching)


class PickupWithRecordedContact(BelievedPickup):
    """
    A run whose grasp is reported by the test rather than measured, so that how a
    condition wires its statechart can be exercised without starting a simulation.
    """

    def _contact_sensor(self) -> ContactSensor:
        return RecordedContacts()


# %% what a condition is


def test_only_the_believed_conditions_weigh_the_carry_by_the_belief():
    """
    The two axes of the comparison are independent: whether the carry reads the belief
    says nothing about whether the gripper closed on anything.
    """
    weighed = {
        condition
        for condition in PickupCondition
        if condition.weighs_the_carry_by_belief
    }

    assert weighed == {
        PickupCondition.BELIEVED_ON_THE_BLOCK,
        PickupCondition.BELIEVED_ON_NOTHING,
    }


def test_only_the_conditions_named_for_it_close_on_the_block():
    """
    The other axis, read the same way.
    """
    closing = {
        condition for condition in PickupCondition if condition.closes_on_the_block
    }

    assert closing == {
        PickupCondition.UNCONDITIONAL_ON_THE_BLOCK,
        PickupCondition.BELIEVED_ON_THE_BLOCK,
    }


def test_a_grasp_that_is_to_fail_puts_the_block_clear_of_the_open_gripper():
    """
    A failing grasp is a block the motion aims at and misses, so the block has to sit
    beyond the fingers rather than on their edge, where contact would be marginal and
    the physics would have to decide the experiment.
    """
    assert MISSED_BY > OPEN_FINGER_OFFSET + BLOCK_SIDE / 2


def test_a_condition_that_closes_on_the_block_leaves_it_where_it_is_expected():
    """
    The two conditions that succeed must run the scene the confirmed baseline runs, or
    they are not a baseline for anything.
    """
    offsets = {
        condition.block_offset
        for condition in PickupCondition
        if condition.closes_on_the_block
    }

    assert offsets == {0.0}


def test_a_condition_that_closes_on_nothing_moves_the_block_out_of_the_way():
    """
    The failing half of the comparison is a block that is not where the motion aims, so
    a condition that never moved it would run the succeeding scene under another name.
    """
    offsets = {
        condition.block_offset
        for condition in PickupCondition
        if not condition.closes_on_the_block
    }

    assert offsets == {MISSED_BY}


# %% what the physics is read as


def test_the_measured_likelihood_is_the_share_of_fingers_against_the_block():
    """
    The belief layer filters a share of samples that found the body between the fingers.

    In a simulation that share is measured, and one finger of two is half of it.
    """
    contacts = RecordedContacts()
    likelihood = ContactLikelihood(
        name="measured likelihood",
        contacts=contacts,
        held_body=PandaPartName.BLOCK,
        fingers=[PandaPartName.LEFT_FINGER, PandaPartName.RIGHT_FINGER],
    )

    measured = []
    for touching in (
        set(),
        {PandaPartName.LEFT_FINGER},
        {PandaPartName.LEFT_FINGER, PandaPartName.RIGHT_FINGER},
    ):
        contacts.touching = touching
        measured.append(likelihood.share_of_fingers_holding)

    assert measured == [0.0, 0.5, 1.0]


def test_anything_else_touching_the_block_is_not_read_as_a_grasp():
    """
    The hand's own shell and the table touch the block too, and neither is a finger
    holding it.
    """
    likelihood = ContactLikelihood(
        name="measured likelihood",
        contacts=RecordedContacts(touching={PandaPartName.HAND, PandaPartName.TABLE}),
        held_body=PandaPartName.BLOCK,
        fingers=[PandaPartName.LEFT_FINGER, PandaPartName.RIGHT_FINGER],
    )

    assert likelihood.share_of_fingers_holding == 0.0


def test_a_measured_reading_is_worth_the_fingers_it_was_taken_over():
    """
    Contact is measured rather than sampled, so a reading carries no sample count of its
    own and is worth its face value until a caller says otherwise.
    """
    likelihood = ContactLikelihood(
        name="measured likelihood",
        contacts=RecordedContacts(),
        held_body=PandaPartName.BLOCK,
        fingers=[PandaPartName.LEFT_FINGER, PandaPartName.RIGHT_FINGER],
    )

    assert likelihood.sample_size == FINGERS_OF_A_PARALLEL_GRIPPER
    assert FINGERS_OF_A_PARALLEL_GRIPPER == len(likelihood.fingers)


# %% how a run wires the belief into the motion


@pytest.fixture(scope="module")
def panda() -> PandaWorld:
    """
    Reading the arm costs a few seconds of mesh work, and nothing below changes it.
    """
    return PandaWorld.of()


@dataclass
class CompiledMotion:
    """
    A condition's motion, compiled, and read back through the constraints the solver is
    actually given.
    """

    statechart: MotionStatechart
    """
    The motion that was compiled.
    """

    def weight_variables_of(self, phase: GraspPhase) -> Set[str]:
        """
        :param phase: The step whose position task is read.
        :return: The names of everything the weight of that task's constraints depends
            on.
        """
        task = next(
            node
            for node in self.statechart.nodes
            if str(node.name) == f"{phase}/position"
        )
        constraints = self.statechart.combine_constraint_collections_of_nodes()
        return {
            str(variable)
            for constraint in constraints.equality_constraints
            if constraint.name.startswith(task.unique_name)
            for variable in constraint.quadratic_weight.free_variables()
        }


def compiled(condition: PickupCondition, panda: PandaWorld) -> CompiledMotion:
    """
    Build and compile the statechart one condition runs, with no physics behind it.

    :param condition: The condition to build.
    :param panda: The world to build it in.
    :return: The compiled motion.
    """
    run = PickupWithRecordedContact(
        condition=condition, belief_settings=BELIEF_SETTINGS, sample_size=200
    )
    statechart = run._create_statechart(panda)
    executor = Executor(context=MotionStatechartContext(world=panda.world))
    executor.compile(motion_statechart=statechart)
    return CompiledMotion(statechart=statechart)


def probability_variable(panda: PandaWorld) -> str:
    """
    :param panda: The world a run is built in.
    :return: The name of the variable a belief publishes how likely a grasp is to, taken
        from a built belief rather than spelled out a second time here.
    """
    run = PickupWithRecordedContact(
        condition=PickupCondition.BELIEVED_ON_NOTHING,
        belief_settings=BELIEF_SETTINGS,
        sample_size=200,
    )
    statechart = run._create_statechart(panda)
    executor = Executor(context=MotionStatechartContext(world=panda.world))
    executor.compile(motion_statechart=statechart)
    return str(run._belief.probability)


def test_a_believed_carry_weighs_every_step_of_itself_by_the_belief(panda):
    """
    The weight only reaches the solver through the constraints the program is built
    from, so each carrying step is read on its own: a union over both would keep passing
    with one of them reverted.
    """
    motion = compiled(PickupCondition.BELIEVED_ON_NOTHING, panda)
    probability = probability_variable(panda)

    for phase in (GraspPhase.LIFTING, GraspPhase.ABOVE_THE_TARGET):
        assert probability in motion.weight_variables_of(phase), phase


def test_an_unconditional_carry_weighs_no_step_of_itself_by_the_belief(panda):
    """
    The belief is kept under every condition so that the only difference between two
    runs is whether the carry reads it.

    An unconditional carry must not.
    """
    motion = compiled(PickupCondition.UNCONDITIONAL_ON_NOTHING, panda)
    probability = probability_variable(panda)

    for phase in (GraspPhase.LIFTING, GraspPhase.ABOVE_THE_TARGET):
        assert probability not in motion.weight_variables_of(phase), phase


def test_the_steps_that_do_not_carry_the_block_are_never_weighed_by_the_belief(panda):
    """
    Reaching for the block, lowering it and letting go of it are worth performing
    whether or not it is held, so a belief that has ruled the grasp out must not stop
    the arm from finishing.
    """
    motion = compiled(PickupCondition.BELIEVED_ON_NOTHING, panda)
    probability = probability_variable(panda)

    for phase in (
        GraspPhase.ABOVE_THE_BLOCK,
        GraspPhase.AT_THE_BLOCK,
        GraspPhase.AT_THE_TARGET,
        GraspPhase.RETREATING,
    ):
        assert probability not in motion.weight_variables_of(phase), phase


def test_the_carry_is_performed_against_returning_to_the_starting_posture(panda):
    """
    A weight only means anything relative to another one.

    With nothing to lose against, the solver drives a carry's error to zero whatever it
    is worth, and scaling it would change nothing at all.
    """
    motion = compiled(PickupCondition.BELIEVED_ON_NOTHING, panda)
    returning = [
        node for node in motion.statechart.nodes if str(node.name) == RETURN_TO_READY
    ]

    assert len(returning) == 1
    assert returning[0].weight == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE


def test_returning_to_the_starting_posture_commands_every_arm_joint(panda):
    """
    A posture short of a joint leaves that joint wherever the carry put it, which is not
    the posture the arm started in.
    """
    motion = compiled(PickupCondition.BELIEVED_ON_NOTHING, panda)
    returning = next(
        node for node in motion.statechart.nodes if str(node.name) == RETURN_TO_READY
    )

    assert len(returning.goal_state) == len(panda.arm)


# %% the grid


def test_a_carry_that_reads_no_belief_is_run_at_one_evidence_strength():
    """
    It runs the same way at every one of them, which is what makes it the baseline for
    the believed runs rather than a second set of numbers to compare against.
    """
    sweep = PickupSweep(belief_settings=BELIEF_SETTINGS, sample_sizes=[2, 20, 200])

    strengths = {
        condition: [
            cell.sample_size for cell in sweep.cells if cell.condition is condition
        ]
        for condition in PickupCondition
    }

    assert strengths[PickupCondition.UNCONDITIONAL_ON_NOTHING] == [2]
    assert strengths[PickupCondition.BELIEVED_ON_NOTHING] == [2, 20, 200]


def test_every_place_the_block_is_put_is_run_once_per_cell():
    """
    A cell is aggregated over the places the block was put, so a result holds over the
    arm's configuration rather than at one geometry.
    """
    sweep = PickupSweep(belief_settings=BELIEF_SETTINGS, block_distances=[0.5, 0.6])
    cell = PickupCell(condition=PickupCondition.BELIEVED_ON_NOTHING, sample_size=200)

    assert [run.block_distance for run in sweep.runs(cell)] == [0.5, 0.6]


def test_the_block_is_carried_the_same_distance_wherever_it_was_put():
    """
    The place the block is put down follows the place it started, so moving the block
    changes where the arm reaches and not how far it carries.
    """
    sweep = PickupSweep(belief_settings=BELIEF_SETTINGS, block_distances=[0.5, 0.6])
    cell = PickupCell(condition=PickupCondition.BELIEVED_ON_NOTHING, sample_size=200)

    for run in sweep.runs(cell):
        placed = (
            run.target.to_np()[:2]
            - PandaWorld.resting_place(run.block_distance, 0.0).to_np()[:2]
        )
        assert placed == pytest.approx([0.0, PLACED_ACROSS_THE_TABLE])


def test_a_cell_with_no_runs_has_nothing_to_report():
    """
    Aggregating nothing would report a row of empty means rather than say that the runs
    behind it are missing.
    """
    cell = PickupCell(condition=PickupCondition.BELIEVED_ON_NOTHING, sample_size=200)

    with pytest.raises(NoRunsToAggregateError):
        PickupConditionResult.from_outcomes(cell, [])


def _outcome(lifted: bool, gripper_offset: float) -> GraspOutcome:
    """
    :param lifted: Whether the block left the table.
    :param gripper_offset: How far the hand ended from over the target, in millimetres.
    :return: A run that did that, with everything else it reports left at nothing.
    """
    return GraspOutcome(
        block_was_lifted=lifted,
        highest_lift=0.0,
        held_by_both_fingers=lifted,
        hand_touched_the_block=False,
        placement_error=0.0,
        travelled=0.0,
        gripper_offset_from_the_target=gripper_offset,
        grasp_probability=0.5,
        control_cycles=1,
        reached_its_goals=lifted,
    )


def test_a_row_counts_the_runs_that_lifted_the_block_rather_than_averaging_them():
    """
    Whether a block was picked up is a yes or a no per run, and a mean of those would
    report a block half lifted.
    """
    cell = PickupCell(condition=PickupCondition.BELIEVED_ON_NOTHING, sample_size=200)

    row = PickupConditionResult.from_outcomes(
        cell, [_outcome(True, 1.0), _outcome(False, 250.0), _outcome(False, 250.0)]
    )

    assert row.runs == 3
    assert row.runs_that_lifted_the_block == 1
    assert row.gripper_offset_from_the_target.mean == pytest.approx(167.0)


# %% what the physics says the belief changed


@pytest.fixture(scope="module")
def outcomes() -> dict:
    """
    Every condition, run once, at the one evidence strength the comparison is decided
    at.
    """
    return {
        condition: BelievedPickup(
            condition=condition, belief_settings=BELIEF_SETTINGS, sample_size=200
        ).execute()
        for condition in PickupCondition
    }


@requires_mujoco
def test_the_arm_completes_the_carry_wherever_the_sweep_puts_the_block():
    """
    Every distance the sweep runs at has to be one the arm can carry from with nothing
    pulling against it, or a carry that gives way says only that the block was out of
    reach.
    """
    sweep = PickupSweep(belief_settings=BELIEF_SETTINGS)
    cell = PickupCell(
        condition=PickupCondition.UNCONDITIONAL_ON_THE_BLOCK,
        sample_size=sweep.sample_sizes[0],
    )

    for run in sweep.runs(cell):
        outcome = run.execute()
        assert outcome.reached_its_goals, run.block_distance
        assert outcome.gripper_offset_from_the_target < 20.0, run.block_distance


@requires_mujoco
def test_a_grasp_that_misses_leaves_the_block_where_it_was(outcomes):
    """
    The block is placed clear of the gripper, so the fingers close past it and nothing
    the arm does afterwards touches it.

    Without this the two axes of the comparison are not independent.
    """
    for condition in PickupCondition:
        if condition.closes_on_the_block:
            continue
        assert not outcomes[condition].block_was_lifted, condition
        assert outcomes[condition].travelled < 1.0, condition


@requires_mujoco
def test_an_unconditional_carry_performs_itself_holding_nothing(outcomes):
    """
    A weight that reads no evidence cannot represent a grasp that failed, so the arm
    takes an empty gripper across the table and reports having finished.
    """
    outcome = outcomes[PickupCondition.UNCONDITIONAL_ON_NOTHING]

    assert outcome.gripper_offset_from_the_target < 20.0
    assert outcome.reached_its_goals


@requires_mujoco
def test_a_believed_carry_gives_way_when_the_grasp_missed(outcomes):
    """
    The deciding measurement: with the grasp ruled out by what the physics reports, the
    carry stops being worth more than putting the arm back where it started, and the arm
    never crosses the table.
    """
    outcome = outcomes[PickupCondition.BELIEVED_ON_NOTHING]

    assert outcome.gripper_offset_from_the_target > 100.0
    assert not outcome.reached_its_goals


@requires_mujoco
def test_a_believed_carry_is_the_unconditional_one_while_the_grasp_holds(outcomes):
    """
    Weighing a carry by the belief is safe to opt into: with the grasp confirmed, every
    number the run reports is the one the unconditional run reports.
    """
    believed = outcomes[PickupCondition.BELIEVED_ON_THE_BLOCK]
    unconditional = outcomes[PickupCondition.UNCONDITIONAL_ON_THE_BLOCK]

    assert believed.block_was_lifted and unconditional.block_was_lifted
    assert believed.reached_its_goals and unconditional.reached_its_goals
    assert believed.travelled == pytest.approx(unconditional.travelled, abs=1.0)
    assert believed.placement_error == pytest.approx(
        unconditional.placement_error, abs=1.0
    )


@requires_mujoco
def test_the_belief_reports_the_grasp_the_physics_actually_made(outcomes):
    """
    The belief is what the weight reads, so a result only means anything if it tracked
    what happened: confident where the fingers closed on the block, ruled out where they
    closed on nothing.
    """
    for condition in PickupCondition:
        probability = outcomes[condition].grasp_probability
        if condition.closes_on_the_block:
            assert probability > 0.5, condition
        else:
            assert probability < 0.1, condition
