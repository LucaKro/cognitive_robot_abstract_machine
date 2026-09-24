"""
The conditions the drawer-opening benchmark runs a task under, after AICON's paper A: a
wrong prior on where the cabinet stands and on its joint, the cabinet moved and the part
pushed while the task runs, and sweeps over the cabinet's yaw and the arm's start pose.

Every episode is seeded, so a condition replays exactly.

.. warning:: The magnitudes of the prior-error levels, the disturbances and the sweeps are
    placeholders. They are not taken from paper A.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto

import numpy
from typing_extensions import TYPE_CHECKING

from experiments.articulated_manipulation.cabinet_scene import (
    ArticulatedPart,
    CabinetSceneSpecification,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose2D

from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

if TYPE_CHECKING:
    from experiments.articulated_manipulation.cabinet_physics import CabinetPhysics
    from semantic_digital_twin.robots.robot_parts import Arm

# %% prior error


class PriorErrorLevel(Enum):
    """
    How wrong the robot's prior about the cabinet is.
    """

    NONE = auto()
    """
    The prior is right.
    """

    LOW = auto()
    """
    The prior is slightly wrong.
    """

    MEDIUM = auto()
    """
    The prior is noticeably wrong.
    """

    HIGH = auto()
    """
    The prior is badly wrong.
    """


@dataclass
class PriorErrorMagnitude:
    """
    How wrong the believed cabinet is at one :class:`PriorErrorLevel`.
    """

    location: float
    """
    How far the believed centre of the cabinet's front lies from the true one, in
    metres.
    """

    joint_axis: float
    """
    How far the believed joint axis is turned from the true one, in radians.
    """


@dataclass
class PriorError:
    """
    How the scene the robot believes in differs from the true one.
    """

    front_offset: Pose2D = field(default_factory=Pose2D)
    """
    Where the believed centre of the cabinet's front lies in the frame of the true one.
    """

    joint_axis_deviation: float = 0.0
    """
    How far the believed joint axis is turned from the true one, in radians (see
    :attr:`CabinetSceneSpecification.mechanism_axis_deviation`).
    """

    @classmethod
    def sample(
        cls, location: float, joint_axis: float, random: numpy.random.Generator
    ) -> PriorError:
        """
        :param location: How far the believed front lies from the true one, in metres,
            in a random direction on the table.
        :param joint_axis: How far the believed joint axis is turned from the true one,
            in radians, in a random sense.
        :param random: Where the directions are drawn from.
        :return: A prior error of exactly the given magnitudes.
        """
        direction = random.uniform(0.0, 2 * math.pi)
        sense = random.choice([-1.0, 1.0])
        return cls(
            front_offset=Pose2D(
                x=location * math.cos(direction), y=location * math.sin(direction)
            ),
            joint_axis_deviation=sense * joint_axis,
        )

    def believed(self, truth: CabinetSceneSpecification) -> CabinetSceneSpecification:
        """
        :param truth: The true scene.
        :return: The scene the robot believes in.
        """
        return dataclasses.replace(
            truth,
            table_T_cabinet_front=Pose2D.from_pose(
                (
                    truth.table_T_cabinet_front.to_homogeneous_matrix()
                    @ self.front_offset.to_homogeneous_matrix()
                ).to_pose()
            ),
            mechanism_axis_deviation=truth.mechanism_axis_deviation
            + self.joint_axis_deviation,
        )


# %% disturbances


class DisturbanceTrigger(ABC):
    """
    When a disturbance strikes, judged from the ground truth.
    """

    @abstractmethod
    def is_due(self, physics: CabinetPhysics) -> bool:
        """
        :param physics: The ground truth of the running episode.
        :return: Whether the disturbance strikes now.
        """


@dataclass
class AfterSimulatedTime(DisturbanceTrigger):
    """
    Strikes once a fixed amount of simulated time has passed.
    """

    elapsed: timedelta
    """
    How much simulated time passes before the disturbance strikes.
    """

    def is_due(self, physics: CabinetPhysics) -> bool:
        return physics.simulated_time >= self.elapsed


@dataclass
class OnceOpenedBy(DisturbanceTrigger):
    """
    Strikes once the moving part is physically open by a share of its travel.
    """

    opened_fraction: float
    """
    The share of the part's travel, between zero and one.
    """

    def is_due(self, physics: CabinetPhysics) -> bool:
        return physics.opened_fraction >= self.opened_fraction


@dataclass
class Disturbance(ABC):
    """
    Something that happens to the physics behind the controller's back.
    """

    trigger: DisturbanceTrigger
    """
    When the disturbance strikes.
    """

    duration: timedelta = timedelta()
    """
    How long the disturbance lasts before it is released.
    """

    @abstractmethod
    def strike(self, physics: CabinetPhysics) -> None:
        """
        Disturb the physics.

        :param physics: The ground truth of the running episode.
        """

    def release(self, physics: CabinetPhysics) -> None:
        """
        End the disturbance once its duration has passed.

        :param physics: The ground truth of the running episode.
        """


@dataclass
class CabinetMoved(Disturbance):
    """
    The cabinet is shoved to a new place on the table.
    """

    displacement: Pose2D = field(default_factory=Pose2D)
    """
    Where the centre of the cabinet's front ends up, in the frame it stood in before.
    """

    def strike(self, physics: CabinetPhysics) -> None:
        physics.move_cabinet(self.displacement)


@dataclass
class PartPushed(Disturbance):
    """
    The moving part is pushed along its joint, such as pulled out of the hand or shut
    again.
    """

    force: float = 0.0
    """
    The force along the joint, or the torque about it for a door; a negative one closes
    the part.
    """

    def strike(self, physics: CabinetPhysics) -> None:
        physics.push_part(self.force)

    def release(self, physics: CabinetPhysics) -> None:
        physics.push_part(0.0)


# %% conditions


class ConditionKind(Enum):
    """
    Which of the benchmark's conditions an episode runs under.
    """

    UNDISTURBED = auto()
    """
    The prior is right and nothing happens.
    """

    LOCATION_PRIOR_ERROR = auto()
    """
    The robot believes the cabinet stands elsewhere.
    """

    JOINT_PRIOR_ERROR = auto()
    """
    The robot believes the joint's axis is turned.
    """

    CABINET_MOVED = auto()
    """
    The cabinet is shoved while the robot works.
    """

    PULLED_FROM_THE_HAND = auto()
    """
    The part is pulled shut while the robot opens it.
    """

    CLOSED_AGAIN = auto()
    """
    The part is pushed shut once it is nearly open.
    """

    CABINET_YAW = auto()
    """
    The cabinet stands turned on the table, and the robot knows it.
    """

    ARM_START_POSE = auto()
    """
    The arm starts away from its park pose.
    """


@dataclass
class Condition:
    """
    One condition of the benchmark: what the robot believes wrongly, what happens while
    it works, and how the scene is set up.
    """

    kind: ConditionKind
    """
    Which condition this is.
    """

    location_error: PriorErrorLevel = PriorErrorLevel.NONE
    """
    How wrong the robot believes the cabinet's location.
    """

    joint_error: PriorErrorLevel = PriorErrorLevel.NONE
    """
    How wrong the robot believes the joint's axis.
    """

    disturbances: list[Disturbance] = field(default_factory=list)
    """
    What happens to the physics while the robot works.
    """

    cabinet_yaw: float = 0.0
    """
    How far the cabinet is turned on the table, in radians.
    """

    arm_start_deviation: float = 0.0
    """
    How far each joint of the arm may start from its park pose, in radians.
    """


# %% episodes


@dataclass
class ArmStartPose:
    """
    A seeded deviation of the arm's start configuration from its park pose.
    """

    deviation: float = 0.0
    """
    How far each joint may start from its park pose, in radians.
    """

    seed: int = 0
    """
    The seed the offsets are drawn with.
    """

    def apply_to(self, arm: Arm) -> None:
        """
        Move each of the arm's joints away from where it stands by its offset.

        :param arm: The arm, standing in its park pose.
        """
        joints = [
            connection
            for connection in arm.connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
        for joint, offset in zip(joints, self.offsets(joints)):
            joint.position += offset

    def offsets(self, joints: list[ActiveConnection1DOF]) -> list[float]:
        """
        :param joints: The arm's joints, in a fixed order.
        :return: How far each joint starts from its park pose.
        """
        random = numpy.random.default_rng(self.seed)
        return [
            float(offset)
            for offset in random.uniform(
                -self.deviation, self.deviation, size=len(joints)
            )
        ]


@dataclass
class EpisodeSetup:
    """
    Everything one episode of a condition needs, drawn with its own seed.
    """

    condition: Condition
    """
    The condition the episode runs under.
    """

    seed: int
    """
    The seed the episode's random parts were drawn with.
    """

    true_specification: CabinetSceneSpecification
    """
    The scene the physics is built from.
    """

    prior_error: PriorError
    """
    How the scene the robot believes in differs from the true one.
    """

    arm_start_pose: ArmStartPose
    """
    Where the arm starts.
    """

    @property
    def believed_specification(self) -> CabinetSceneSpecification:
        """
        The scene the robot's world is built from.
        """
        return self.prior_error.believed(self.true_specification)


# %% the protocol


@dataclass
class DisturbanceProtocol:
    """
    The benchmark's conditions and how many seeded episodes each of them runs.
    """

    articulated_part: ArticulatedPart = ArticulatedPart.DRAWER
    """
    Which part of the cabinet the task opens.
    """

    episodes_per_condition: int = 10
    """
    How many episodes each condition runs.
    """

    prior_error_magnitudes: dict[PriorErrorLevel, PriorErrorMagnitude] = field(
        default_factory=lambda: {
            PriorErrorLevel.NONE: PriorErrorMagnitude(location=0.0, joint_axis=0.0),
            PriorErrorLevel.LOW: PriorErrorMagnitude(
                location=0.02, joint_axis=math.radians(5)
            ),
            PriorErrorLevel.MEDIUM: PriorErrorMagnitude(
                location=0.05, joint_axis=math.radians(10)
            ),
            PriorErrorLevel.HIGH: PriorErrorMagnitude(
                location=0.1, joint_axis=math.radians(20)
            ),
        }
    )
    """
    How wrong the prior is at each level. Placeholders, not taken from paper A.
    """

    cabinet_displacement: Pose2D = field(
        default_factory=lambda: Pose2D(x=0.05, y=0.05, yaw=math.radians(10))
    )
    """
    Where the shoved cabinet's front ends up, in the frame it stood in before.
    Placeholder.
    """

    cabinet_moved_after: timedelta = timedelta(seconds=2)
    """
    How much simulated time passes before the cabinet is shoved. Placeholder.
    """

    pull_force: float = -100.0
    """
    The force that pulls the part out of the hand, and the one that shuts it again.
    Placeholder.
    """

    pull_duration: timedelta = timedelta(seconds=0.5)
    """
    How long the part is pulled or pushed shut. Placeholder.
    """

    pulled_once_opened_by: float = 0.2
    """
    The share of its travel the part is open when it is pulled out of the hand.
    Placeholder.
    """

    closed_once_opened_by: float = 0.6
    """
    The share of its travel the part is open when it is pushed shut again, short of the
    share an episode requires to succeed. Placeholder.
    """

    cabinet_yaws: list[float] = field(
        default_factory=lambda: [math.radians(angle) for angle in (-20, -10, 10, 20)]
    )
    """
    The cabinet yaws the yaw sweep runs, in radians. Placeholders.
    """

    arm_start_deviations: list[float] = field(
        default_factory=lambda: [math.radians(angle) for angle in (5, 10, 20)]
    )
    """
    How far each arm joint may start from its park pose in the arm-pose sweep, in
    radians. Placeholders.
    """

    def conditions(self) -> list[Condition]:
        """
        :return: Every condition of the benchmark.
        """
        erroneous_levels = [
            level for level in PriorErrorLevel if level is not PriorErrorLevel.NONE
        ]
        return [
            Condition(kind=ConditionKind.UNDISTURBED),
            *[
                Condition(kind=ConditionKind.LOCATION_PRIOR_ERROR, location_error=level)
                for level in erroneous_levels
            ],
            *[
                Condition(kind=ConditionKind.JOINT_PRIOR_ERROR, joint_error=level)
                for level in erroneous_levels
            ],
            Condition(
                kind=ConditionKind.CABINET_MOVED,
                disturbances=[
                    CabinetMoved(
                        trigger=AfterSimulatedTime(self.cabinet_moved_after),
                        displacement=self.cabinet_displacement,
                    )
                ],
            ),
            Condition(
                kind=ConditionKind.PULLED_FROM_THE_HAND,
                disturbances=[self._part_pushed_shut(self.pulled_once_opened_by)],
            ),
            Condition(
                kind=ConditionKind.CLOSED_AGAIN,
                disturbances=[self._part_pushed_shut(self.closed_once_opened_by)],
            ),
            *[
                Condition(kind=ConditionKind.CABINET_YAW, cabinet_yaw=yaw)
                for yaw in self.cabinet_yaws
            ],
            *[
                Condition(
                    kind=ConditionKind.ARM_START_POSE, arm_start_deviation=deviation
                )
                for deviation in self.arm_start_deviations
            ],
        ]

    def episode_setups(self, seed: int) -> list[EpisodeSetup]:
        """
        :param seed: The seed the whole run is drawn with.
        :return: The setup of every episode of every condition, the same for the same
            seed.
        """
        random = numpy.random.default_rng(seed)
        return [
            self._episode_setup(condition, int(random.integers(2**31)))
            for condition in self.conditions()
            for _ in range(self.episodes_per_condition)
        ]

    def _episode_setup(self, condition: Condition, seed: int) -> EpisodeSetup:
        """
        :param condition: The condition the episode runs under.
        :param seed: The episode's own seed.
        :return: The episode's setup.
        """
        random = numpy.random.default_rng(seed)
        true_specification = CabinetSceneSpecification(
            articulated_part=self.articulated_part
        )
        true_specification = dataclasses.replace(
            true_specification,
            table_T_cabinet_front=Pose2D(
                x=true_specification.table_T_cabinet_front.x,
                y=true_specification.table_T_cabinet_front.y,
                yaw=condition.cabinet_yaw,
            ),
        )
        return EpisodeSetup(
            condition=condition,
            seed=seed,
            true_specification=true_specification,
            prior_error=PriorError.sample(
                location=self.prior_error_magnitudes[condition.location_error].location,
                joint_axis=self.prior_error_magnitudes[
                    condition.joint_error
                ].joint_axis,
                random=random,
            ),
            arm_start_pose=ArmStartPose(
                deviation=condition.arm_start_deviation,
                seed=int(random.integers(2**31)),
            ),
        )

    def _part_pushed_shut(self, once_opened_by: float) -> PartPushed:
        """
        :param once_opened_by: The share of its travel the part is open when it is
            pushed.
        :return: The part pushed shut with :attr:`pull_force`.
        """
        return PartPushed(
            trigger=OnceOpenedBy(once_opened_by),
            duration=self.pull_duration,
            force=self.pull_force,
        )
