"""
Ways to watch the demonstration: a video rendered by RoboSimStudio, and RoboSimStudio's
own viewer in the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import robosimstudio
from imageio.core.format import Format
from PIL import Image, ImageDraw
from robosimstudio.sim.draw import Viewer

from semantic_digital_twin.robots.tiago import TiagoJoint
from semantic_digital_twin.world_description.geometry import Color

from experiments.robosimstudio_tiago_demo.simulation import (
    RoboSimStudioSimulation,
    SimulationObserver,
)
from experiments.robosimstudio_tiago_demo.world import Grocery

# %% cameras


def camera_framing(
    name: str,
    vertical_field_of_view: float,
    eye: tuple[float, float, float],
    look_at: tuple[float, float, float],
    width: int = 640,
    height: int = 400,
    far_clip: float = 30.0,
) -> robosimstudio.CameraSpec:
    """
    :param name: The name the camera's pictures are rendered under.
    :param vertical_field_of_view: The vertical field of view, in degrees.
    :param eye: Where the camera stands, in the apartment frame.
    :param look_at: The point the camera is aimed at, in the apartment frame.
    :param width: The width of the picture, in pixels.
    :param height: The height of the picture, in pixels.
    :param far_clip: How far the camera sees, in metres.
    :return: The camera.
    """
    return robosimstudio.CameraSpec(
        name=name,
        width=width,
        height=height,
        fov_y=vertical_field_of_view,
        eye=eye,
        look_at=look_at,
        far=far_clip,
    )


class DemonstrationCamera(Enum):
    """
    The cameras the video is rendered through.
    """

    OVERVIEW = camera_framing("overview", 62.0, (3.5, -0.7, 3.3), (3.0, 2.3, 0.7))
    COUNTER = camera_framing("counter", 50.0, (3.3, 2.7, 1.8), (2.2, 1.9, 1.0))
    ISLAND = camera_framing("island", 50.0, (2.0, 2.9, 1.8), (3.0, 2.15, 0.95))

    @classmethod
    def close_ups(cls) -> list[DemonstrationCamera]:
        """
        :return: The cameras close to where the robot works.
        """
        return [cls.COUNTER, cls.ISLAND]

    @classmethod
    def specifications(cls) -> tuple[robosimstudio.CameraSpec, ...]:
        """
        :return: Every camera, as RoboSimStudio takes it.
        """
        return tuple(camera.value for camera in cls)


# %% video


@dataclass
class VideoRecorder(SimulationObserver):
    """
    Renders the simulation into a video in real time: the overview next to the close-up
    nearest the robot, under a caption with what the plan commands and what the physics
    does.
    """

    file_path: Path
    """
    Where the video is written to.
    """

    control_steps_per_frame: int = 4
    """
    How many control steps pass between two frames.
    """

    caption_height: int = 26
    """
    The height of the caption bar above the pictures, in pixels.
    """

    caption_margin: int = 10
    """
    How far the caption text stands off the bar's left edge, in pixels.
    """

    caption_background: Color = field(default_factory=lambda: Color(0.08, 0.08, 0.08))
    """
    The color of the caption bar.
    """

    caption_color: Color = field(default_factory=Color)
    """
    The color of the caption text.
    """

    codec: str = "libx264"
    """
    The codec the video is encoded with.
    """

    quality: int = 8
    """
    The encoding quality of the video, from 0 to 10.
    """

    block_size: int = 8
    """
    The block size the frame dimensions are rounded to for the encoder.
    """

    _writer: Format.Writer | None = field(init=False, default=None)
    """
    The encoder, opened on the first frame.
    """

    def observe(self, simulation: RoboSimStudioSimulation) -> None:
        if simulation.simulator.step_count % self.control_steps_per_frame:
            return
        if self._writer is None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                str(self.file_path),
                fps=simulation.arrangement.control_frequency
                / self.control_steps_per_frame,
                codec=self.codec,
                quality=self.quality,
                macro_block_size=self.block_size,
            )
        base = np.array(simulation.base_pose[:2])
        close_up = min(
            DemonstrationCamera.close_ups(),
            key=lambda camera: np.linalg.norm(
                np.array(camera.value.look_at[:2]) - base
            ),
        )
        pictures = [
            simulation.simulator.render(camera.value.name)["rgb"]
            for camera in (DemonstrationCamera.OVERVIEW, close_up)
        ]
        frame = Image.fromarray(np.concatenate(pictures, axis=1))
        draw = ImageDraw.Draw(frame)
        draw.rectangle(
            (0, 0, frame.width, self.caption_height),
            fill=self.caption_background.to_hex(),
        )
        draw.text(
            (self.caption_margin, self.caption_height // 3),
            self.caption(simulation),
            fill=self.caption_color.to_hex(),
        )
        self._writer.append_data(np.asarray(frame))

    @staticmethod
    def caption(simulation: RoboSimStudioSimulation) -> str:
        """
        :param simulation: The simulation to describe.
        :return: The simulated time, the milk's height and the right gripper as
            commanded and as simulated.
        """
        finger = TiagoJoint.RIGHT_GRIPPER_FINGER
        commanded = simulation.world.get_connection_by_name(finger).position
        milk_height = simulation.simulated_pose(Grocery.MILK)[2, 3]
        return (
            "CRAM plans and drives the TIAGo, RoboSimStudio simulates it; the milk is "
            f"held by friction alone | t = {simulation.time.total_seconds():5.1f} s | "
            f"milk height {milk_height:.3f} m | gripper commanded {commanded:.2f} rad, "
            f"simulated {simulation.simulated_joint_position(finger):.2f} rad"
        )

    def close(self) -> None:
        """
        Finish the video.
        """
        if self._writer is not None:
            self._writer.close()


# %% viewer


@dataclass
class LiveViewer(SimulationObserver):
    """
    RoboSimStudio's viewer, served in the browser while the plan is performed.
    """

    viewer: Viewer
    """
    The viewer the simulation is shown in.
    """

    control_steps_per_update: int = 5
    """
    How many control steps pass between two updates of the viewer.
    """

    @classmethod
    def serve(
        cls,
        simulation: RoboSimStudioSimulation,
        port: int,
        title: str = "CRAM TIAGo in RoboSimStudio",
    ) -> LiveViewer:
        """
        Serve the viewer of ``simulation``.

        :param simulation: The simulation to show.
        :param port: The port the viewer is served on.
        :param title: The title of the viewer's page.
        :return: The viewer, showing the simulation as it stands.
        """
        viewer = robosimstudio.viewer(simulation.simulator, port=port, label=title)
        viewer.show()
        return cls(viewer=viewer)

    def observe(self, simulation: RoboSimStudioSimulation) -> None:
        if simulation.simulator.step_count % self.control_steps_per_update:
            return
        self.viewer.show()
