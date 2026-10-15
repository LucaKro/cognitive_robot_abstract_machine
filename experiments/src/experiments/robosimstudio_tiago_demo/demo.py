"""
The TIAGo carries the milk from the counter to the kitchen island, planned and driven by
CRAM and simulated by RoboSimStudio.

Run it with::

    python -m experiments.robosimstudio_tiago_demo.demo --video milk.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np

from coraplex.execution_environment import PhysicallySimulatedRobot

from experiments.robosimstudio_tiago_demo.descriptions import DescriptionPackages
from experiments.robosimstudio_tiago_demo.plan import MilkTransport
from experiments.robosimstudio_tiago_demo.recording import (
    DemonstrationCamera,
    LiveViewer,
    VideoRecorder,
)
from experiments.robosimstudio_tiago_demo.simulation import RoboSimStudioSimulation
from experiments.robosimstudio_tiago_demo.world import Grocery, TiagoApartment

# %% outcome


@dataclass
class DemonstrationOutcome:
    """
    Where the milk ended up, against where the plan was to put it.
    """

    milk_position: np.ndarray
    """
    Where the simulation has the milk once it has come to rest, in the apartment frame.
    """

    target_position: np.ndarray
    """
    Where the plan puts the milk down, in the apartment frame.
    """

    @property
    def distance_to_target(self) -> float:
        """
        How far the milk stands from where it was to be put down, on the island's
        surface.
        """
        return float(np.linalg.norm(self.milk_position[:2] - self.target_position[:2]))

    def __str__(self) -> str:
        return (
            f"milk at {np.round(self.milk_position, 3).tolist()}, target "
            f"{self.target_position.tolist()}, "
            f"{self.distance_to_target * 100:.1f} cm off on the island"
        )


# %% demonstration


class OutputFile(StrEnum):
    """
    The files a run writes to its output directory.
    """

    SCENE = "tiago_apartment.xml"


@dataclass
class RoboSimStudioTiagoDemonstration:
    """
    Builds the world, simulates it in RoboSimStudio and performs the plan in it.
    """

    output_directory: Path = field(
        default_factory=lambda: Path("robosimstudio_tiago_demo_output")
    )
    """
    Where the exported scene, and the video, are written to.
    """

    description_packages: DescriptionPackages = field(
        default_factory=DescriptionPackages
    )
    """
    Where the descriptions the world is built from are fetched to.
    """

    transport: MilkTransport = field(default_factory=MilkTransport)
    """
    What the TIAGo does.
    """

    settling_time: timedelta = timedelta(seconds=0.5)
    """
    How long the groceries are given to come to rest, before the plan starts and after
    it ends.
    """

    def build_simulation(self) -> RoboSimStudioSimulation:
        """
        :return: The simulation of the demonstration's world, not started yet.
        """
        self.description_packages.make_resolvable()
        return RoboSimStudioSimulation.build(
            TiagoApartment.build(),
            self.output_directory / OutputFile.SCENE,
            DemonstrationCamera.specifications(),
        )

    def perform(self, simulation: RoboSimStudioSimulation) -> DemonstrationOutcome:
        """
        Let the groceries settle, carry the milk to the island, and let it settle there.

        :param simulation: The simulation to perform the plan in.
        :return: Where the milk ended up.
        """
        apartment = TiagoApartment(world=simulation.world, robot=simulation.robot)
        simulation.start_stepped_simulation()
        simulation.step_simulation(self.settling_time)
        plan = self.transport.plan(apartment, self.transport.context(apartment))
        with PhysicallySimulatedRobot(simulation=simulation):
            plan.perform()
        simulation.step_simulation(self.settling_time)
        target = self.transport.place_target
        return DemonstrationOutcome(
            milk_position=simulation.simulated_pose(Grocery.MILK)[:3, 3],
            target_position=np.array([target.x, target.y, target.z]),
        )


# %% command line


class Option(StrEnum):
    """
    The command line options of the demonstration.
    """

    VIDEO = "--video"
    VIEWER_PORT = "--viewer-port"
    OUTPUT_DIRECTORY = "--output-directory"
    CACHE_DIRECTORY = "--cache-directory"


def parse_arguments() -> argparse.Namespace:
    """
    :return: The command line arguments.
    """
    defaults = RoboSimStudioTiagoDemonstration()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        Option.VIDEO,
        type=Path,
        default=None,
        help="render the run into this video, relative to the output directory",
    )
    parser.add_argument(
        Option.VIEWER_PORT,
        type=int,
        default=None,
        help="serve RoboSimStudio's viewer on this port while the plan runs",
    )
    parser.add_argument(
        Option.OUTPUT_DIRECTORY,
        type=Path,
        default=defaults.output_directory,
        help="where the exported scene and the video are written to",
    )
    parser.add_argument(
        Option.CACHE_DIRECTORY,
        type=Path,
        default=defaults.description_packages.cache_directory,
        help="where the robot and apartment descriptions are fetched to",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the demonstration as the command line asks.
    """
    arguments = parse_arguments()
    demonstration = RoboSimStudioTiagoDemonstration(
        output_directory=arguments.output_directory,
        description_packages=DescriptionPackages(arguments.cache_directory),
    )
    simulation = demonstration.build_simulation()
    recorder = None
    if arguments.video is not None:
        recorder = VideoRecorder(arguments.output_directory / arguments.video)
        simulation.observers.append(recorder)
    if arguments.viewer_port is not None:
        simulation.observers.append(LiveViewer.serve(simulation, arguments.viewer_port))
    outcome = demonstration.perform(simulation)
    if recorder is not None:
        recorder.close()
    print(outcome)


if __name__ == "__main__":
    main()
