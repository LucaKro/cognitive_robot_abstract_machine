"""
The ground truth of a simulated cabinet scene: what the physics says, read directly from
MuJoCo rather than from the controller's world, and the ways the physics is disturbed
behind the controller's back.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import timedelta

from krrood.exceptions import DataclassException
from physics_simulators.base_simulator import SimulatorCallbackResult
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.spatial_types.spatial_types import Pose2D

from experiments.articulated_manipulation.cabinet_scene import (
    CabinetScene,
    CabinetSceneSpecification,
)

# %% failures

SUCCESSFUL_RESULT_TYPES = frozenset(
    {
        SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
        SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL,
        SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
    }
)
"""
The results of a simulator callback that did what it was asked.
"""


@dataclass
class PhysicsRequestFailedError(DataclassException):
    """
    Raised when the physics refuses to be read or disturbed.
    """

    result: SimulatorCallbackResult
    """
    What the simulator answered.
    """

    def error_message(self) -> str:
        return f"The physics refused the request: {self.result.info}"


# %% the ground truth


@dataclass
class CabinetPhysics:
    """
    The ground truth of a running simulation of a cabinet scene.
    """

    simulation: MujocoSim
    """
    The running simulation.
    """

    scene: CabinetScene
    """
    The true scene the physics was built from, which names the cabinet's body and its
    moving part's joint in the physics.
    """

    specification: CabinetSceneSpecification
    """
    The true scene's description; its cabinet follows every time the cabinet is moved.
    """

    @property
    def simulated_time(self) -> timedelta:
        """
        How much simulated time has passed since the simulation started.
        """
        return timedelta(seconds=self.simulation.simulator.current_simulation_time)

    @property
    def opening(self) -> float:
        """
        How far the moving part is open in the physics, in its joint's coordinate.
        """
        return self._request(
            self.simulation.simulator.get_joint_value(self.scene.mechanism.name.name)
        ).result

    @property
    def opened_fraction(self) -> float:
        """
        The share of its travel the moving part is open in the physics.
        """
        return self.opening / self.scene.mechanism.dof.limits.upper.position

    def move_cabinet(self, displacement: Pose2D) -> None:
        """
        Shove the cabinet to a new place on the table.

        :param displacement: Where the centre of the cabinet's front ends up, in the
            frame it stood in before.
        """
        self.specification = dataclasses.replace(
            self.specification,
            table_T_cabinet_front=Pose2D.from_pose(
                (
                    self.specification.table_T_cabinet_front.to_homogeneous_matrix()
                    @ displacement.to_homogeneous_matrix()
                ).to_pose()
            ),
        )
        world_T_cabinet = self.specification.world_T_cabinet()
        x, y, z, w = world_T_cabinet.to_quaternion().to_np()
        self._request(
            self.simulation.simulator.set_fixed_body_pose(
                body_name=self.scene.cabinet.root.name.name,
                position=world_T_cabinet.to_position().to_np()[:3],
                quaternion=[w, x, y, z],
            )
        )

    def push_part(self, force: float) -> None:
        """
        Push the moving part along its joint until it is pushed again.

        :param force: The force along the joint, or the torque about it for a door; a
            negative one closes the part, and zero stops pushing.
        """
        self._request(
            self.simulation.simulator.set_joint_applied_force(
                joint_name=self.scene.mechanism.name.name, force=force
            )
        )

    @staticmethod
    def _request(result: SimulatorCallbackResult) -> SimulatorCallbackResult:
        """
        :param result: What the simulator answered a request with.
        :return: The answer, if the request succeeded.
        :raises PhysicsRequestFailedError: If the simulator refused the request.
        """
        if result.type not in SUCCESSFUL_RESULT_TYPES:
            raise PhysicsRequestFailedError(result)
        return result
