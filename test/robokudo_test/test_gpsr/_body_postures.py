"""
Body keypoints of persons holding a known posture, in the image and in the camera frame.

Image positions are pixels with *y* growing downward. Camera frame positions are meters
with *x* to the image right, *y* downward and *z* away from the camera.
"""

from __future__ import annotations

# %% Imports
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
import open3d as o3d
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.types.human import (
    BodyKeypoint,
    BodyKeypointCloud,
    KeypointAnnotation,
    KeypointKind,
)

# %% body sides


class BodySide(StrEnum):
    """
    A side of the body, from the person's point of view.
    """

    LEFT = "LEFT"
    """
    The left side of the person.
    """

    RIGHT = "RIGHT"
    """
    The right side of the person.
    """

    @property
    def eye(self) -> PoseKeypoint:
        """
        :return: The eye on this side.
        """
        return PoseKeypoint[f"{self.value}_EYE"]

    @property
    def shoulder(self) -> PoseKeypoint:
        """
        :return: The shoulder on this side.
        """
        return PoseKeypoint[f"{self.value}_SHOULDER"]

    @property
    def elbow(self) -> PoseKeypoint:
        """
        :return: The elbow on this side.
        """
        return PoseKeypoint[f"{self.value}_ELBOW"]

    @property
    def wrist(self) -> PoseKeypoint:
        """
        :return: The wrist on this side.
        """
        return PoseKeypoint[f"{self.value}_WRIST"]


# %% postures in the image


@dataclass
class BodyPosture:
    """
    The image positions of the body keypoints of one person.
    """

    keypoints: dict[PoseKeypoint, BodyKeypoint]
    """
    Position of each body keypoint, missing keypoints are reported with no confidence.
    """

    @classmethod
    def from_positions(
        cls, positions: dict[PoseKeypoint, tuple[float, float]], confidence: float = 0.9
    ) -> BodyPosture:
        """
        :param positions: The image position of every keypoint.
        :param confidence: The confidence of all keypoints.
        :return: A posture whose keypoints are all detected with the same confidence.
        """
        return cls(
            keypoints={
                keypoint: BodyKeypoint(x=x, y=y, confidence=confidence)
                for keypoint, (x, y) in positions.items()
            }
        )

    @classmethod
    def standing(cls) -> BodyPosture:
        """
        :return: A person standing upright facing the camera, arms down.
        """
        return cls.from_positions(
            {
                PoseKeypoint.NOSE: (320, 100),
                PoseKeypoint.LEFT_EYE: (315, 95),
                PoseKeypoint.RIGHT_EYE: (325, 95),
                PoseKeypoint.LEFT_EAR: (310, 98),
                PoseKeypoint.RIGHT_EAR: (330, 98),
                PoseKeypoint.LEFT_SHOULDER: (300, 150),
                PoseKeypoint.RIGHT_SHOULDER: (340, 150),
                PoseKeypoint.LEFT_ELBOW: (295, 200),
                PoseKeypoint.RIGHT_ELBOW: (345, 200),
                PoseKeypoint.LEFT_WRIST: (295, 250),
                PoseKeypoint.RIGHT_WRIST: (345, 250),
                PoseKeypoint.LEFT_HIP: (305, 260),
                PoseKeypoint.RIGHT_HIP: (335, 260),
                PoseKeypoint.LEFT_KNEE: (305, 350),
                PoseKeypoint.RIGHT_KNEE: (335, 350),
                PoseKeypoint.LEFT_ANKLE: (305, 420),
                PoseKeypoint.RIGHT_ANKLE: (335, 420),
            }
        )

    @classmethod
    def sitting(cls) -> BodyPosture:
        """
        :return: A person sitting on a chair, thighs horizontal and shins vertical.
        """
        return cls.from_positions(
            {
                PoseKeypoint.NOSE: (320, 150),
                PoseKeypoint.LEFT_EYE: (315, 145),
                PoseKeypoint.RIGHT_EYE: (325, 145),
                PoseKeypoint.LEFT_EAR: (310, 148),
                PoseKeypoint.RIGHT_EAR: (330, 148),
                PoseKeypoint.LEFT_SHOULDER: (300, 200),
                PoseKeypoint.RIGHT_SHOULDER: (340, 200),
                PoseKeypoint.LEFT_ELBOW: (300, 250),
                PoseKeypoint.RIGHT_ELBOW: (340, 250),
                PoseKeypoint.LEFT_WRIST: (300, 280),
                PoseKeypoint.RIGHT_WRIST: (340, 280),
                PoseKeypoint.LEFT_HIP: (305, 300),
                PoseKeypoint.RIGHT_HIP: (335, 300),
                PoseKeypoint.LEFT_KNEE: (365, 310),
                PoseKeypoint.RIGHT_KNEE: (395, 310),
                PoseKeypoint.LEFT_ANKLE: (365, 400),
                PoseKeypoint.RIGHT_ANKLE: (395, 400),
            }
        )

    @classmethod
    def lying(cls) -> BodyPosture:
        """
        :return: A person lying on their side with the head to the image left, both
            shoulders on top of each other.
        """
        return cls.from_positions(
            {
                PoseKeypoint.NOSE: (100, 296),
                PoseKeypoint.LEFT_EYE: (95, 294),
                PoseKeypoint.RIGHT_EYE: (95, 298),
                PoseKeypoint.LEFT_EAR: (110, 296),
                PoseKeypoint.RIGHT_EAR: (110, 298),
                PoseKeypoint.LEFT_SHOULDER: (150, 300),
                PoseKeypoint.RIGHT_SHOULDER: (150, 304),
                PoseKeypoint.LEFT_ELBOW: (200, 302),
                PoseKeypoint.RIGHT_ELBOW: (200, 306),
                PoseKeypoint.LEFT_WRIST: (250, 304),
                PoseKeypoint.RIGHT_WRIST: (250, 308),
                PoseKeypoint.LEFT_HIP: (300, 310),
                PoseKeypoint.RIGHT_HIP: (300, 314),
                PoseKeypoint.LEFT_KNEE: (400, 315),
                PoseKeypoint.RIGHT_KNEE: (400, 319),
                PoseKeypoint.LEFT_ANKLE: (500, 320),
                PoseKeypoint.RIGHT_ANKLE: (500, 324),
            }
        )

    def moved(self, keypoint: PoseKeypoint, x: float, y: float) -> BodyPosture:
        """
        :param keypoint: The keypoint to move.
        :param x: The new image column.
        :param y: The new image row.
        :return: A copy of this posture with one keypoint at another image position.
        """
        keypoints = dict(self.keypoints)
        keypoints[keypoint] = replace(keypoints[keypoint], x=x, y=y)
        return BodyPosture(keypoints=keypoints)

    def moved_to_height(self, y: float, *keypoints: PoseKeypoint) -> BodyPosture:
        """
        :param y: The new image row.
        :param keypoints: The keypoints to move.
        :return: A copy of this posture in which the given keypoints are at another
            image height.
        """
        changed = dict(self.keypoints)
        for keypoint in keypoints:
            changed[keypoint] = replace(changed[keypoint], y=y)
        return BodyPosture(keypoints=changed)

    def scaled(self, factor: float) -> BodyPosture:
        """
        :param factor: The factor on every image position.
        :return: A copy of this posture shrunk or grown about the image origin, as the
            same person seen from another distance.
        """
        return BodyPosture(
            keypoints={
                keypoint: replace(
                    position, x=position.x * factor, y=position.y * factor
                )
                for keypoint, position in self.keypoints.items()
            }
        )

    def with_confidence(
        self, confidence: float, *keypoints: PoseKeypoint
    ) -> BodyPosture:
        """
        :param confidence: The new confidence.
        :param keypoints: The keypoints to change.
        :return: A copy of this posture in which the given keypoints are detected with
            another confidence.
        """
        changed = dict(self.keypoints)
        for keypoint in keypoints:
            changed[keypoint] = replace(changed[keypoint], confidence=confidence)
        return BodyPosture(keypoints=changed)

    def keypoint_annotation(self) -> KeypointAnnotation:
        """
        :return: The image keypoints of this posture, as a pose model reports them.
        """
        return KeypointAnnotation(
            keypoints=[
                self.keypoints.get(keypoint, BodyKeypoint(x=0, y=0, confidence=0.0))
                for keypoint in PoseKeypoint
            ],
            kind=KeypointKind.IMAGE_POSITION,
        )


# %% bodies in the camera frame


@dataclass
class Forearm:
    """
    The elbow and the wrist of one arm, in the camera frame.
    """

    elbow: list[float]
    """
    Position of the elbow.
    """

    wrist: list[float]
    """
    Position of the wrist.
    """


@dataclass
class BodyInSpace:
    """
    The camera frame positions of the body keypoints of one person standing two meters
    in front of the camera.
    """

    positions: dict[PoseKeypoint, list[float]] = field(
        default_factory=lambda: {
            PoseKeypoint.LEFT_SHOULDER: [0.2, 0.0, 2.0],
            PoseKeypoint.RIGHT_SHOULDER: [-0.2, 0.0, 2.0],
            PoseKeypoint.LEFT_ELBOW: [0.25, 0.25, 2.0],
            PoseKeypoint.RIGHT_ELBOW: [-0.25, 0.25, 2.0],
            PoseKeypoint.LEFT_WRIST: [0.25, 0.5, 2.0],
            PoseKeypoint.RIGHT_WRIST: [-0.25, 0.5, 2.0],
            PoseKeypoint.LEFT_HIP: [0.15, 0.5, 2.0],
            PoseKeypoint.RIGHT_HIP: [-0.15, 0.5, 2.0],
        }
    )
    """
    Position of each measured keypoint, by default with both arms hanging down.
    """

    def with_arm(self, side: BodySide, forearm: Forearm) -> BodyInSpace:
        """
        :param side: The side of the arm.
        :param forearm: The new forearm.
        :return: A copy of this body with the elbow and wrist of the given side at the
            forearm's positions.
        """
        return self.with_positions(
            {side.elbow: forearm.elbow, side.wrist: forearm.wrist}
        )

    def with_arm_held_out(
        self, side: BodySide, lowered_by: float = 0.0, forearm_length: float = 0.25
    ) -> BodyInSpace:
        """
        :param side: The side of the arm.
        :param lowered_by: Angle of the forearm below the horizontal, in degrees.
        :param forearm_length: Distance from the elbow to the wrist, in meters.
        :return: A copy of this body holding the forearm of the given side sideways away
            from the body, the elbow beside the shoulder.
        """
        away_from_the_body = 1.0 if side is BodySide.LEFT else -1.0
        elbow = np.array([away_from_the_body * 0.45, 0.1, 2.0])
        radians = math.radians(lowered_by)
        direction = np.array(
            [away_from_the_body * math.cos(radians), math.sin(radians), 0.0]
        )
        return self.with_arm(
            side,
            Forearm(elbow=list(elbow), wrist=list(elbow + forearm_length * direction)),
        )

    def with_positions(self, positions: dict[PoseKeypoint, list[float]]) -> BodyInSpace:
        """
        :param positions: The new position of every keypoint to change.
        :return: A copy of this body with the given keypoints at other positions.
        """
        return BodyInSpace(positions={**self.positions, **positions})

    def without(self, *keypoints: PoseKeypoint) -> BodyInSpace:
        """
        :param keypoints: The keypoints to leave out.
        :return: A copy of this body in which the given keypoints were not measured.
        """
        return BodyInSpace(
            positions={
                keypoint: position
                for keypoint, position in self.positions.items()
                if keypoint not in keypoints
            }
        )

    def torso_center(self) -> np.ndarray:
        """
        :return: The mean of the shoulders and hips.
        """
        return np.mean(
            [
                self.positions[keypoint]
                for keypoint in (
                    PoseKeypoint.LEFT_SHOULDER,
                    PoseKeypoint.RIGHT_SHOULDER,
                    PoseKeypoint.LEFT_HIP,
                    PoseKeypoint.RIGHT_HIP,
                )
            ],
            axis=0,
        )

    def keypoint_annotation(self) -> KeypointAnnotation:
        """
        :return: The keypoint clouds of this body, each measured keypoint as a cloud of
            one point and every other keypoint as an empty cloud.
        """
        clouds = []
        for keypoint in PoseKeypoint:
            cloud = o3d.geometry.PointCloud()
            if keypoint in self.positions:
                cloud.points = o3d.utility.Vector3dVector([self.positions[keypoint]])
            clouds.append(
                BodyKeypointCloud(
                    keypoint=BodyKeypoint(x=0, y=0, confidence=0.9), points=cloud
                )
            )
        return KeypointAnnotation(keypoints=clouds, kind=KeypointKind.POINT_CLOUD)
