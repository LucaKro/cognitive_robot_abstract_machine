"""
RoboSimStudio stepped by the demonstration's world, and the world reading it back.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("robosimstudio")

from semantic_digital_twin.robots.tiago import TiagoJoint

from experiments.robosimstudio_tiago_demo.scene import BaseJoint
from experiments.robosimstudio_tiago_demo.simulation import RoboSimStudioSimulation
from experiments.robosimstudio_tiago_demo.world import Grocery, TiagoApartment

# %% a started simulation


@pytest.fixture(scope="module")
def simulation(
    tmp_path_factory: pytest.TempPathFactory,
) -> RoboSimStudioSimulation:
    """
    A started simulation of a fresh world, its groceries given half a second to come to
    rest on the counter.
    """
    simulation = RoboSimStudioSimulation.build(
        TiagoApartment.build(),
        tmp_path_factory.mktemp("robosimstudio_simulation") / "scene.xml",
    )
    simulation.start_stepped_simulation()
    simulation.step_simulation(timedelta(seconds=0.5))
    return simulation


# %% stepping


def test_the_base_stays_where_the_world_has_it(simulation: RoboSimStudioSimulation):
    start = simulation.coordinates.position_start[BaseJoint.merged_label]
    simulated = simulation.simulator.state.joint_q.numpy()[
        start : start + len(BaseJoint)
    ]

    np.testing.assert_allclose(simulated, simulation.base_pose, atol=0.01)


def test_the_world_holds_the_groceries_where_the_simulation_has_them(
    simulation: RoboSimStudioSimulation,
):
    for grocery in Grocery:
        np.testing.assert_allclose(
            simulation.world.get_body_by_name(grocery).global_transform.to_np(),
            simulation.simulated_pose(grocery),
            atol=1e-6,
        )


def test_a_joint_follows_its_commanded_position(simulation: RoboSimStudioSimulation):
    torso = simulation.world.get_connection_by_name(TiagoJoint.TORSO_LIFT)
    target = (torso.dof.limits.lower.position + torso.dof.limits.upper.position) / 2
    torso.position = target

    simulation.step_simulation(timedelta(seconds=2.0))

    assert simulation.simulated_joint_position(TiagoJoint.TORSO_LIFT) == pytest.approx(
        target, abs=0.01
    )
