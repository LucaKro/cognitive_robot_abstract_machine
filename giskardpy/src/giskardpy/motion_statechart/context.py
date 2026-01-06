from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
    AuxiliaryVariable,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World


@dataclass
class BuildContext:
    world: World
    auxiliary_variable_manager: AuxiliaryVariableManager
    collision_scene: CollisionWorldSynchronizer
    qp_controller_config: QPControllerConfig
    control_cycle_variable: AuxiliaryVariable

    @classmethod
    def empty(cls) -> Self:
        return cls(
            world=World(),
            auxiliary_variable_manager=None,
            collision_scene=None,
            qp_controller_config=None,
            control_cycle_variable=None,
        )


@dataclass
class ExecutionContext:
    world: World
    external_collision_data_data: np.ndarray
    self_collision_data_data: np.ndarray
    auxiliar_variables_data: np.ndarray
    control_cycle_counter: int
    dt: float
