"""
RoboSimStudio as the physics a CRAM plan is performed in.

:class:`RoboSimStudioSimulation` is what
:class:`~coraplex.execution_environment.PhysicallySimulatedRobot` steps between two
controller ticks: the controller's command lands in the world state, the servos take it
as their set point, RoboSimStudio advances the physics, and the groceries' simulated
poses are written back into the world. Nothing ties a grocery to the gripper; whether it
comes along is up to the friction of the finger pads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import newton
import numpy as np
import robosimstudio
import warp
from robosimstudio.sim.sim import Sim
from scipy.spatial.transform import Rotation

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
)

from experiments.robosimstudio_tiago_demo.scene import (
    BaseJoint,
    PhysicsChoices,
    PhysicsScene,
)
from experiments.robosimstudio_tiago_demo.world import Grocery, TiagoApartment

# %% arrangement


class RoboSimStudioName(StrEnum):
    """
    Names the scene is built with in RoboSimStudio.
    """

    SCENE = "cram_tiago_apartment"
    WORLD_ASSET = "cram_world"
    EMPTY_CELL = "void"
    MUJOCO_BACKEND = "mujoco"
    EVERY_ASSET = "*"
    PROCESSOR = "cpu"


@dataclass(frozen=True)
class PhysicsArrangement:
    """
    How RoboSimStudio steps the physics: MuJoCo's own solver on the processor, at the
    rate of CRAM's controller.
    """

    control_frequency: float = 100.0
    """
    How often per simulated second the servos take a new set point, in hertz.
    """

    physics_steps_per_control_step: int = 6
    """
    How many physics steps one control step is divided into; RoboSimStudio requires an
    even number.
    """

    maximum_contacts: int = 4096
    """
    How many contacts the solver has room for.
    """

    maximum_constraint_rows: int = 8192
    """
    How many constraint rows the solver has room for.
    """

    def to_specification(self) -> robosimstudio.ArrangementSpec:
        """
        :return: The arrangement as RoboSimStudio takes it.
        """
        return robosimstudio.ArrangementSpec(
            control_hz=self.control_frequency,
            substeps=self.physics_steps_per_control_step,
            backends=(
                (
                    RoboSimStudioName.EVERY_ASSET,
                    robosimstudio.backend(
                        RoboSimStudioName.MUJOCO_BACKEND,
                        use_mujoco_cpu=True,
                        nconmax=self.maximum_contacts,
                        njmax=self.maximum_constraint_rows,
                    ),
                ),
            ),
            collision=robosimstudio.MUJOCO_500HZ.collision,
        )


# %% joint coordinates


@dataclass
class JointCoordinates:
    """
    Where each joint's coordinates start in the simulation's state, and its servo set
    points in the simulation's control, by joint name.
    """

    position_start: dict[str, int]
    """
    The index of each joint's first coordinate.
    """

    target_start: dict[str, int]
    """
    The index of each joint's first servo set point.
    """

    @classmethod
    def from_model(cls, model: newton.Model) -> JointCoordinates:
        """
        :param model: The simulated model.
        :return: The coordinates of every joint of ``model``.
        """
        names = [label.rsplit("/", 1)[-1] for label in model.joint_label]
        return cls(
            position_start=dict(zip(names, map(int, model.joint_q_start.numpy()))),
            target_start=dict(zip(names, map(int, model.joint_target_q_start.numpy()))),
        )

    @classproperty
    def free_joint_position_size(cls) -> int:
        """
        How many coordinates a free joint spends on its position; its orientation, a
        quaternion ordered x, y, z, w, follows.
        """
        return 3

    @classproperty
    def free_joint_size(cls) -> int:
        """
        How many coordinates a free joint has.
        """
        return 7

    def free_joint_slice(self, body_name: str) -> slice:
        """
        :param body_name: The body a free joint moves.
        :return: The coordinates of that free joint.
        """
        [start] = [
            start
            for name, start in self.position_start.items()
            if name.endswith(body_name)
        ]
        return slice(start, start + self.free_joint_size)

    def read_free_joint(self, positions: np.ndarray, body_name: str) -> np.ndarray:
        """
        :param positions: The simulation's joint coordinates.
        :param body_name: The body a free joint moves.
        :return: The pose of that body, as a homogeneous matrix.
        """
        coordinates = positions[self.free_joint_slice(body_name)]
        pose = np.eye(4)
        pose[:3, 3] = coordinates[: self.free_joint_position_size]
        pose[:3, :3] = Rotation.from_quat(
            coordinates[self.free_joint_position_size :]
        ).as_matrix()
        return pose

    def write_free_joint(
        self, positions: np.ndarray, body_name: str, pose: np.ndarray
    ) -> None:
        """
        :param positions: The simulation's joint coordinates, changed in place.
        :param body_name: The body a free joint moves.
        :param pose: The pose to put that body at, as a homogeneous matrix.
        """
        positions[self.free_joint_slice(body_name)] = [
            *pose[:3, 3],
            *Rotation.from_matrix(pose[:3, :3]).as_quat(),
        ]


# %% simulation


class SimulationObserver(ABC):
    """
    Something that looks at the simulation after every control step, such as a recorder.
    """

    @abstractmethod
    def observe(self, simulation: RoboSimStudioSimulation) -> None:
        """
        Look at ``simulation`` as it stands after a control step.

        :param simulation: The simulation that just stepped.
        """


@dataclass
class RoboSimStudioSimulation:
    """
    The demonstration's world simulated by RoboSimStudio, stepped by CRAM's controller.
    """

    world: World
    """
    The world the plan is performed in.
    """

    robot: Tiago
    """
    The robot whose servos the simulation drives.
    """

    simulator: Sim
    """
    The RoboSimStudio simulation.
    """

    arrangement: PhysicsArrangement
    """
    How :attr:`simulator` steps the physics.
    """

    servoed_joints: list[str]
    """
    The joints of :attr:`robot` driven by a servo, by name, the base joints aside.
    """

    observers: list[SimulationObserver] = field(default_factory=list)
    """
    What looks at the simulation after every control step.
    """

    coordinates: JointCoordinates = field(init=False)
    """
    Where each joint's coordinates are in :attr:`simulator`'s state.
    """

    def __post_init__(self):
        self.coordinates = JointCoordinates.from_model(self.simulator.model)

    @classmethod
    def build(
        cls,
        apartment: TiagoApartment,
        scene_file: Path,
        cameras: tuple[robosimstudio.CameraSpec, ...] = (),
        arrangement: PhysicsArrangement = PhysicsArrangement(),
        choices: PhysicsChoices | None = None,
    ) -> RoboSimStudioSimulation:
        """
        Export ``apartment`` and build its RoboSimStudio simulation.

        :param apartment: The world to simulate.
        :param scene_file: Where to write the exported scene to.
        :param cameras: The cameras to render the simulation through.
        :param arrangement: How to step the physics.
        :param choices: The physics choices to export with, or ``None`` for the
            defaults.
        :return: The simulation, not started yet.
        """
        warp.set_device(RoboSimStudioName.PROCESSOR)
        scene = PhysicsScene.export(apartment, scene_file, choices)
        simulation_scene = robosimstudio.Scene(
            cell=RoboSimStudioName.EMPTY_CELL,
            name=RoboSimStudioName.SCENE,
            physics=arrangement.to_specification(),
        )
        simulation_scene.add(
            robosimstudio.StaticSpec(
                name=RoboSimStudioName.WORLD_ASSET,
                asset=robosimstudio.AssetSpec(path=str(scene_file)),
                world_xform=True,
            )
        )
        for camera in cameras:
            simulation_scene.add(camera)
        return cls(
            world=apartment.world,
            robot=apartment.robot,
            simulator=simulation_scene.build(),
            arrangement=arrangement,
            servoed_joints=[
                joint
                for joint in scene.choices.servo_gains
                if joint not in list(BaseJoint)
            ],
        )

    @property
    def base_pose(self) -> list[float]:
        """
        The commanded pose of the robot's base: x, y and yaw in the apartment.
        """
        drive = self.robot.root.parent_connection
        return [
            self.world.state[dof.id].position for dof in (drive.x, drive.y, drive.yaw)
        ]

    @property
    def time(self) -> timedelta:
        """
        How much simulated time has passed since the simulation was built.
        """
        return timedelta(
            seconds=self.simulator.step_count / self.arrangement.control_frequency
        )

    def start_stepped_simulation(self) -> None:
        """
        Put every joint and grocery of the simulation where the world has it, at rest,
        and hand the servos their set points.
        """
        state = self.simulator.state
        positions = state.joint_q.numpy()
        for connection in self.world.connections:
            name = connection.name.name
            if isinstance(connection, ActiveConnection1DOF) and (
                name in self.coordinates.position_start
            ):
                positions[self.coordinates.position_start[name]] = connection.position
        base_start = self.coordinates.position_start[BaseJoint.merged_label]
        positions[base_start : base_start + len(BaseJoint)] = self.base_pose
        for grocery in Grocery:
            self.coordinates.write_free_joint(
                positions,
                grocery,
                self.world.get_body_by_name(grocery).global_transform.to_np(),
            )
        state.joint_q.assign(positions)
        state.joint_qd.assign(np.zeros_like(state.joint_qd.numpy()))
        newton.eval_fk(self.simulator.model, state.joint_q, state.joint_qd, state)
        self.command_servos()

    def step_simulation(self, duration: timedelta) -> None:
        """
        Advance the physics by ``duration`` towards the set points the world holds, and
        write the groceries' simulated poses back into the world.

        :param duration: How much simulated time to advance.
        """
        control_steps = duration.total_seconds() * self.arrangement.control_frequency
        for _ in range(round(control_steps)):
            self.command_servos()
            self.simulator.run()
            self.read_groceries()
            for observer in self.observers:
                observer.observe(self)

    def command_servos(self) -> None:
        """
        Hand every servo the position the world holds for its joint.
        """
        targets = self.simulator.control.joint_target_q.numpy()
        for joint in self.servoed_joints:
            targets[self.coordinates.target_start[joint]] = (
                self.world.get_connection_by_name(joint).position
            )
        base_start = self.coordinates.target_start[BaseJoint.merged_label]
        targets[base_start : base_start + len(BaseJoint)] = self.base_pose
        self.simulator.control.joint_target_q.assign(targets)

    def simulated_pose(self, grocery: Grocery) -> np.ndarray:
        """
        :param grocery: The grocery to look up.
        :return: Where the simulation has the grocery, as a homogeneous matrix in the
            apartment frame.
        """
        return self.coordinates.read_free_joint(
            self.simulator.state.joint_q.numpy(), grocery
        )

    def simulated_joint_position(self, joint: str) -> float:
        """
        :param joint: The name of a single-coordinate joint.
        :return: Where the simulation has that joint.
        """
        return float(
            self.simulator.state.joint_q.numpy()[self.coordinates.position_start[joint]]
        )

    def read_groceries(self) -> None:
        """
        Write each grocery's simulated pose into the world.
        """
        with self.world.batch_state_changes():
            for grocery in Grocery:
                self.world.get_body_by_name(grocery).parent_connection.origin = (
                    HomogeneousTransformationMatrix(
                        self.simulated_pose(grocery), reference_frame=self.world.root
                    )
                )
