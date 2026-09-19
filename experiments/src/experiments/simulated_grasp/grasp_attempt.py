"""
Picking the block up and putting it down somewhere else, with the physics deciding
whether the gripper is actually holding it.

The controller ticks against the world and the physics steps one control cycle between
two ticks, so every command becomes a servo's set point and the arm reaches its goal
only as far as the contacts allow. What the attempt reports is read from the simulation,
never from the world the controller planned in.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np
from typing_extensions import Dict, List, Optional, Set

from experiments.experiment_definitions import ExperimentResult
from experiments.simulated_grasp.panda_world import (
    BLOCK_DISTANCE,
    GRASP_HEIGHT,
    GRIPPED_FINGER_OFFSET,
    OPEN_FINGER_OFFSET,
    OVERVIEW_CAMERA,
    PandaPartName,
    PandaWorld,
)
from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion, MotionStatechartNode
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountSimulationTimeSeconds,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.mujoco_video_recording import (
    RecordedVideo,
    VideoResolution,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types.spatial_types import Point3, Vector3

# %% the numbers a run is made with

CONTROL_FREQUENCY = 50
"""
How many times a second the controller is ticked, in hertz.
"""

CONTROL_CYCLE_LIMIT = 900
"""
How many control cycles an attempt is given before it is reported as unfinished.
"""

MILLIMETRES_PER_METRE = 1000.0
"""
What a distance is reported in, so a result reads in the units the differences occur at.
"""

APPROACH_HEIGHT = 0.13
"""
How far above the block the gripper is brought before it descends, in metres.
"""

CARRY_HEIGHT = 0.25
"""
How far above a resting place the block is carried, in metres.
"""

SETTLING_TIME = 1.0
"""
How long the gripper is held still while it closes or opens, in seconds.

The fingers are driven by servos, so closing them is not instantaneous and a lift
started too early would slide off a block that is not yet held.
"""

LIFTED_OFF_THE_TABLE = 0.05
"""
How far above its resting height the block counts as lifted, in metres.
"""

PLACED_ACROSS_THE_TABLE = 0.25
"""
How far across the table the block is carried, in metres.
"""


class GraspPhase(StrEnum):
    """
    The steps an attempt runs through, in order.
    """

    ABOVE_THE_BLOCK = "above the block"
    AT_THE_BLOCK = "at the block"
    CLOSING = "closing the gripper"
    GRIPPING = "settling the grip"
    LIFTING = "lifting the block"
    ABOVE_THE_TARGET = "above the target"
    AT_THE_TARGET = "at the target"
    OPENING = "opening the gripper"
    RELEASING = "settling the release"
    RETREATING = "retreating"

    @property
    def carries_the_block(self) -> bool:
        """
        :return: Whether this step moves the block from where it was picked up to where
            it is going. Reaching for it, lowering it and letting go of it are worth
            performing whether or not it is held; taking it across the table is not.
        """
        return self in (GraspPhase.LIFTING, GraspPhase.ABOVE_THE_TARGET)


# %% what an attempt reports


@dataclass
class GraspOutcome(ExperimentResult):
    """
    What one attempt did, read from the physics rather than from the controller's world.
    """

    block_was_lifted: bool
    """
    Whether the block left the table while it was carried.
    """

    highest_lift: float
    """
    How far above its resting height the block reached, in millimetres.
    """

    held_by_both_fingers: bool
    """
    Whether both fingers were touching the block at the moment it was highest.
    """

    hand_touched_the_block: bool
    """
    Whether anything but a finger ever touched the block.

    The hand's own shell reaches lower than its fingertips, so a grip taken too deep
    presses the palm onto the block before the fingers ever close on it.
    """

    placement_error: float
    """
    How far the block ended from where it was to be placed, in millimetres.
    """

    travelled: float
    """
    How far the block ended from where it started, in millimetres.
    """

    gripper_offset_from_the_target: float
    """
    How far the hand ended from over the place the block was to be put down, in
    millimetres, which is what says whether the arm performed the carry at all.
    """

    grasp_probability: Optional[float]
    """
    How likely a belief held a grasp to be when the attempt ended, or nothing where the
    attempt kept no belief.

    Reported so a reader can see why an attempt behaved as it did, and never what one is
    judged by.
    """

    control_cycles: int
    """
    How many control cycles the attempt took.
    """

    reached_its_goals: bool
    """
    Whether the motion ended by reaching its goals rather than by running out of cycles.
    """


# %% running one attempt


@dataclass
class PhysicalGrasp:
    """
    One attempt at picking the block up and putting it down at another place on the
    table.
    """

    target: Point3 = field(
        default_factory=lambda: PandaWorld.resting_place(
            BLOCK_DISTANCE, PLACED_ACROSS_THE_TABLE
        )
    )
    """
    Where on the table the block is to be put down.
    """

    control_cycle_limit: int = CONTROL_CYCLE_LIMIT
    """
    How many control cycles the attempt is given.
    """

    closes_the_gripper: bool = True
    """
    Whether the gripper closes on the block at all.

    An attempt that leaves it open runs exactly the same motion and is what says a
    lifted block was lifted by the grip rather than carried along by anything else.
    """

    block_distance: float = BLOCK_DISTANCE
    """
    How far in front of the arm the block sits, in metres.
    """

    block_offset: float = 0.0
    """
    How far across the table the block actually sits from where the motion expects it,
    in metres.

    The motion aims at where the scene says the block is, so an offset wide enough to
    clear the gripper is a grasp that closes on nothing while the robot has every reason
    to think it succeeded.
    """

    video_path: Optional[Path] = None
    """
    Where a recording of the attempt is written, or nothing to run without rendering.
    """

    video_resolution: VideoResolution = field(
        default_factory=lambda: VideoResolution(width=1280, height=720)
    )
    """
    How large the recorded frames are.
    """

    # %% init False

    _scenario: Optional[PandaWorld] = field(default=None, init=False, repr=False)
    """
    The world the attempt is made in, built by :meth:`execute`.
    """

    _simulation: Optional[MujocoSim] = field(default=None, init=False, repr=False)
    """
    The physics the attempt is judged by, started by :meth:`execute`.
    """

    _frames: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    """
    The frames captured so far, empty unless :attr:`video_path` is set.
    """

    def execute(self) -> GraspOutcome:
        """
        Run the attempt.

        :return: What it did.
        """
        self._scenario = PandaWorld.of(
            block_distance=self.block_distance, block_offset=self.block_offset
        )
        started_at = self._scenario.block_position.copy()
        resting_height = started_at[2]

        if self.video_path is not None:
            self._use_headless_rendering()
        self._simulation = MujocoSim(world=self._scenario.world, headless=True)
        if self.video_path is not None:
            self._make_room_to_render_into()
        self._simulation.start_stepped_simulation()
        statechart = self._create_statechart(self._scenario)
        highest = resting_height
        held_when_highest = False
        hand_ever_touched = False
        try:
            executor = Executor(
                context=MotionStatechartContext(
                    world=self._scenario.world,
                    qp_controller_config=QPControllerConfig(
                        target_frequency=CONTROL_FREQUENCY
                    ),
                ),
                pacer=SteppedSimulationPacer(self._simulation),
            )
            executor.compile(motion_statechart=statechart)
            reached_its_goals = False
            for _ in range(self.control_cycle_limit):
                if statechart.is_end_motion():
                    reached_its_goals = True
                    break
                executor.tick()
                executor.pacer.sleep()
                self._observe_cycle(executor)
                self._capture_frame()
                hand_ever_touched = hand_ever_touched or self.hand_touches_the_block
                height = self.block_position[2]
                if height > highest:
                    highest = height
                    held_when_highest = self.fingers_touching_the_block == {
                        PandaPartName.LEFT_FINGER,
                        PandaPartName.RIGHT_FINGER,
                    }
            self._simulation.step_simulation(timedelta(seconds=SETTLING_TIME))
            self._capture_frame()
            ended_at = self.block_position.copy()
            hand_ended_at = self.hand_position.copy()
            grasp_probability = self._grasp_probability(executor)
            control_cycles = int(executor.control_cycles)
        finally:
            self._simulation.stop_simulation()

        self._write_video()
        return GraspOutcome(
            block_was_lifted=bool(highest - resting_height >= LIFTED_OFF_THE_TABLE),
            highest_lift=(highest - resting_height) * MILLIMETRES_PER_METRE,
            held_by_both_fingers=held_when_highest,
            hand_touched_the_block=hand_ever_touched,
            placement_error=float(np.linalg.norm(ended_at - self.target.to_np()[:3]))
            * MILLIMETRES_PER_METRE,
            travelled=float(np.linalg.norm(ended_at - started_at))
            * MILLIMETRES_PER_METRE,
            gripper_offset_from_the_target=float(
                np.linalg.norm(hand_ended_at[:2] - self.target.to_np()[:2])
            )
            * MILLIMETRES_PER_METRE,
            grasp_probability=grasp_probability,
            control_cycles=control_cycles,
            reached_its_goals=reached_its_goals,
        )

    @property
    def block_position(self) -> np.ndarray:
        """
        :return: Where the block's centre is in the physics.
        """
        return np.array(
            self._simulation.simulator.get_body_position(
                body_name=PandaPartName.BLOCK
            ).result
        )

    @property
    def hand_position(self) -> np.ndarray:
        """
        :return: Where the hand is in the physics, which is where the arm actually got
            to rather than where it was commanded.
        """
        return np.array(
            self._simulation.simulator.get_body_position(
                body_name=PandaPartName.HAND
            ).result
        )

    @property
    def fingers_touching_the_block(self) -> Set[str]:
        """
        :return: Which of the gripper's fingers the physics reports in contact with the
            block.
        """
        touching = self._simulation.simulator.get_contact_bodies(
            body_name=PandaPartName.BLOCK
        ).result
        return {
            finger
            for finger in (PandaPartName.LEFT_FINGER, PandaPartName.RIGHT_FINGER)
            if finger in touching
        }

    @property
    def hand_touches_the_block(self) -> bool:
        """
        :return: Whether the physics reports the hand itself, rather than a finger,
            against the block.
        """
        return (
            PandaPartName.HAND
            in self._simulation.simulator.get_contact_bodies(
                body_name=PandaPartName.BLOCK
            ).result
        )

    def _create_statechart(self, scenario: PandaWorld) -> MotionStatechart:
        """
        Describe the motion: reach the block from above, close on it, carry it to the
        target, put it down and let go.

        :param scenario: The world the attempt is made in.
        :return: The statechart the attempt executes.

        ..note:: Built once the physics is running, so that anything running alongside
            the motion can read the simulation the motion is judged by.
        """
        block = scenario.expected_block_position
        steps: Dict[GraspPhase, MotionStatechartNode] = {
            GraspPhase.ABOVE_THE_BLOCK: self._reach(
                scenario, GraspPhase.ABOVE_THE_BLOCK, block, APPROACH_HEIGHT
            ),
            GraspPhase.AT_THE_BLOCK: self._reach(
                scenario, GraspPhase.AT_THE_BLOCK, block, GRASP_HEIGHT
            ),
            GraspPhase.GRIPPING: CountSimulationTimeSeconds(
                name=GraspPhase.GRIPPING, seconds=SETTLING_TIME
            ),
            GraspPhase.LIFTING: self._reach(
                scenario, GraspPhase.LIFTING, block, CARRY_HEIGHT
            ),
            GraspPhase.ABOVE_THE_TARGET: self._reach(
                scenario, GraspPhase.ABOVE_THE_TARGET, self.target, CARRY_HEIGHT
            ),
            GraspPhase.AT_THE_TARGET: self._reach(
                scenario, GraspPhase.AT_THE_TARGET, self.target, GRASP_HEIGHT
            ),
            GraspPhase.RELEASING: CountSimulationTimeSeconds(
                name=GraspPhase.RELEASING, seconds=SETTLING_TIME
            ),
            GraspPhase.RETREATING: self._reach(
                scenario, GraspPhase.RETREATING, self.target, CARRY_HEIGHT
            ),
        }
        previous = None
        for step in steps.values():
            step.end_condition = step.observation_variable
            if previous is not None:
                step.start_condition = previous.is_succeeded
            previous = step

        closing = self._grip(scenario, GraspPhase.CLOSING, GRIPPED_FINGER_OFFSET)
        opening = self._grip(scenario, GraspPhase.OPENING, OPEN_FINGER_OFFSET)
        closing.start_condition = steps[GraspPhase.AT_THE_BLOCK].is_succeeded
        closing.end_condition = steps[GraspPhase.AT_THE_TARGET].is_succeeded
        opening.start_condition = steps[GraspPhase.AT_THE_TARGET].is_succeeded

        statechart = MotionStatechart()
        for node in (
            list(steps.values())
            + [closing, opening]
            + self._competing_nodes(scenario, steps)
        ):
            statechart.add_node(node)
        statechart.add_node(EndMotion.when_true(steps[GraspPhase.RETREATING]))
        return statechart

    def _competing_nodes(
        self, scenario: PandaWorld, steps: Dict[GraspPhase, MotionStatechartNode]
    ) -> List[MotionStatechartNode]:
        """
        Whatever else runs alongside the motion.

        :param scenario: The world the attempt is made in.
        :param steps: The motion's own steps, by the phase each performs, so that
            something running alongside them can say when it does.
        :return: Nothing, for an attempt that is only the motion.
        """
        return []

    def _observe_cycle(self, executor: Executor) -> None:
        """
        Read whatever this attempt watches for while it runs.

        :param executor: The executor that has just ticked.
        """

    def _grasp_probability(self, executor: Executor) -> Optional[float]:
        """
        :param executor: The executor whose context holds whatever was published.
        :return: Nothing, for an attempt that keeps no belief about its grasp.
        """
        return None

    def _reach(
        self, scenario: PandaWorld, phase: GraspPhase, place: Point3, height: float
    ) -> Parallel:
        """
        Bring the frame between the fingertips over a place on the table, with the palm
        facing down.

        Only the direction the fingers point in is constrained, not how the hand is
        turned about it: the block's diagonal is narrower than the gripper's opening, so
        the fingers straddle it whatever way the hand ends up facing, and leaving that
        free spares the wrist a half-turn it would otherwise have to wind up.

        :param scenario: The world the attempt is made in.
        :param phase: Which step of the attempt this is.
        :param place: The point on the table to reach over.
        :param height: How far above it, in metres.
        :return: The step.
        """
        world = scenario.world
        goal = place.to_np()[:3]
        return Parallel(
            name=phase,
            nodes=[
                self._position_task(
                    scenario,
                    phase,
                    Point3(
                        goal[0], goal[1], goal[2] + height, reference_frame=world.root
                    ),
                ),
                AlignPlanes(
                    name=f"{phase}/approach",
                    root_link=world.root,
                    tip_link=scenario.hand,
                    tip_normal=Vector3.Z(reference_frame=scenario.hand),
                    goal_normal=Vector3(0.0, 0.0, -1.0, reference_frame=world.root),
                ),
            ],
        )

    def _position_task(
        self, scenario: PandaWorld, phase: GraspPhase, goal_point: Point3
    ) -> CartesianPosition:
        """
        Bring the frame between the fingertips to a point.

        :param scenario: The world the attempt is made in.
        :param phase: Which step of the attempt this is.
        :param goal_point: Where that frame is to end up.
        :return: The task, carrying its weight whatever happens.
        """
        return CartesianPosition(
            name=f"{phase}/position",
            root_link=scenario.world.root,
            tip_link=scenario.tool_frame,
            goal_point=goal_point,
        )

    def _grip(
        self, scenario: PandaWorld, phase: GraspPhase, offset: float
    ) -> JointPositionList:
        """
        Command both fingers to the same distance from the palm's centre.

        :param scenario: The world the attempt is made in.
        :param phase: Which step of the attempt this is.
        :param offset: How far from the palm's centre each finger is commanded, in
            metres.
        :return: The step.
        """
        commanded = offset if self.closes_the_gripper else OPEN_FINGER_OFFSET
        return JointPositionList(
            name=phase,
            goal_state=JointState.from_mapping(
                {finger: commanded for finger in scenario.fingers}
            ),
        )

    @staticmethod
    def _use_headless_rendering() -> None:
        """
        Ask MuJoCo for a graphics backend that needs no window.

        Its default picks one that cannot make a context where no display is attached,
        which is every machine this runs on.
        """
        if os.environ.get("MUJOCO_GL", "").lower() not in ("egl", "osmesa"):
            os.environ["MUJOCO_GL"] = "osmesa"

    def _make_room_to_render_into(self) -> None:
        """
        Size the off-screen buffer MuJoCo renders into to the frames being asked for.

        A scene states that size in its own ``<visual>`` block, and the simulation is
        not read from a scene but rebuilt from what the world holds, which carries
        shapes and colours and no such block. Left alone the buffer keeps MuJoCo's own
        default, and a larger frame is refused rather than cropped.
        """
        display = self._simulation.simulator._mj_model.vis.global_
        display.offwidth = max(display.offwidth, self.video_resolution.width)
        display.offheight = max(display.offheight, self.video_resolution.height)

    def _capture_frame(self) -> None:
        """
        Keep a frame of what the scene looks like now, if a recording was asked for.
        """
        if self.video_path is None:
            return
        self._frames.append(
            self._simulation.simulator.capture_rgb(
                camera_name=OVERVIEW_CAMERA,
                height=self.video_resolution.height,
                width=self.video_resolution.width,
            ).result
        )

    def _write_video(self) -> None:
        """
        Encode the captured frames, if a recording was asked for.
        """
        if self.video_path is None:
            return
        RecordedVideo(frames=self._frames, frames_per_second=CONTROL_FREQUENCY).write(
            self.video_path
        )
        self._frames = []


# %% running it from the command line


def main() -> None:
    """
    Run one attempt and report what the physics said about it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Where to write a recording of the attempt.",
    )
    parser.add_argument(
        "--leave-the-gripper-open",
        action="store_true",
        help="Run the same motion without closing the gripper.",
    )
    parser.add_argument(
        "--block-offset",
        type=float,
        default=0.0,
        help="How far across the table the block sits from where the motion expects it.",
    )
    arguments = parser.parse_args()
    outcome = PhysicalGrasp(
        video_path=arguments.video,
        closes_the_gripper=not arguments.leave_the_gripper_open,
        block_offset=arguments.block_offset,
    ).execute()
    for name, value in zip(
        GraspOutcome.get_column_names(), outcome.get_column_values()
    ):
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
