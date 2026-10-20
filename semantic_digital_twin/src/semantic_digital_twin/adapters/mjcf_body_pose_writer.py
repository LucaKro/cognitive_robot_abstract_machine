from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.exceptions import (
    BodyNotInMJCFFileError,
    UnsupportedMJCFOrientationError,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% MJCF vocabulary


class MJCFTag(StrEnum):
    """
    The MJCF elements that place bodies.
    """

    WORLD_BODY = "worldbody"
    """
    The element every body of the scene is nested in.
    """

    BODY = "body"
    """
    A body, posed relative to the frame it is nested in.
    """

    FRAME = "frame"
    """
    A frame that shifts the bodies nested in it without being a body itself.
    """


class MJCFAttribute(StrEnum):
    """
    The attributes of MJCF elements that name and place bodies.
    """

    NAME = "name"
    """
    The name of the element.
    """

    POSITION = "pos"
    """
    The position, as ``x y z``.
    """

    QUATERNION = "quat"
    """
    The orientation, as a quaternion ``w x y z``.
    """

    EULER = "euler"
    """
    The orientation, as Euler angles.
    """

    AXIS_ANGLE = "axisangle"
    """
    The orientation, as a rotation axis and angle.
    """

    XY_AXES = "xyaxes"
    """
    The orientation, as its x- and y-axis.
    """

    Z_AXIS = "zaxis"
    """
    The orientation, as the shortest rotation onto its z-axis.
    """


# %% poses as MJCF writes them


@dataclass
class MJCFPose:
    """
    The pose of an MJCF element relative to the frame it is nested in.
    """

    position: np.ndarray
    """
    The position ``x y z``, in meters.
    """

    quaternion: np.ndarray
    """
    The unit quaternion ``w x y z`` of the orientation.
    """

    @classmethod
    def from_element(cls, element: ElementTree.Element) -> MJCFPose:
        """
        :param element: An element giving its orientation by ``quat``, or not at all.
        :return: the pose the element gives, identity where it omits an attribute.
        """
        attributes = element.attrib
        unsupported = {
            MJCFAttribute.EULER,
            MJCFAttribute.AXIS_ANGLE,
            MJCFAttribute.XY_AXES,
            MJCFAttribute.Z_AXIS,
        } & attributes.keys()
        if unsupported:
            written_attributes = " ".join(f'{k}="{v}"' for k, v in attributes.items())
            raise UnsupportedMJCFOrientationError(
                element_tag=f"<{element.tag} {written_attributes}>",
                orientation_attribute=unsupported.pop(),
            )
        quaternion = np.array(
            attributes.get(MJCFAttribute.QUATERNION, "1 0 0 0").split(), dtype=float
        )
        return cls(
            position=np.array(
                attributes.get(MJCFAttribute.POSITION, "0 0 0").split(), dtype=float
            ),
            quaternion=quaternion / np.linalg.norm(quaternion),
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> MJCFPose:
        """
        :param matrix: A homogeneous transformation matrix.
        :return: the pose the matrix describes.
        """
        x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return cls(position=matrix[:3, 3].copy(), quaternion=np.array([w, x, y, z]))

    @property
    def matrix(self) -> np.ndarray:
        """
        The homogeneous transformation matrix of the pose.
        """
        w, x, y, z = self.quaternion
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        matrix[:3, 3] = self.position
        return matrix

    def rounded(self, decimals: int) -> MJCFPose:
        """
        :param decimals: The number of decimals to keep.
        :return: this pose with every value rounded, and without negative zeros.
        """
        # adding 0.0 turns -0.0 into 0.0
        return MJCFPose(
            position=np.round(self.position, decimals) + 0.0,
            quaternion=np.round(self.quaternion, decimals) + 0.0,
        )

    def is_close_to(self, other: MJCFPose, tolerance: float) -> bool:
        """
        :param other: The pose to compare to.
        :param tolerance: The largest difference per value that still counts as equal.
        :return: whether both poses agree up to the tolerance, where a quaternion and
            its negation describe the same orientation.
        """
        other_quaternion = other.quaternion
        if np.dot(self.quaternion, other_quaternion) < 0:
            other_quaternion = -other_quaternion
        return np.allclose(
            self.position, other.position, atol=tolerance
        ) and np.allclose(self.quaternion, other_quaternion, atol=tolerance)

    @property
    def has_identity_orientation(self) -> bool:
        """
        Whether the pose does not rotate.
        """
        return np.allclose(np.abs(self.quaternion), [1, 0, 0, 0])


@dataclass
class MJCFBodyPlacement:
    """
    Where a ``<body>`` sits in an MJCF file.
    """

    parent_name: str
    """
    The name of the body the ``<body>`` is nested in.
    """

    parent_T_frame: np.ndarray
    """
    The pose of the frame the ``pos`` and ``quat`` of the ``<body>`` are expressed in,
    relative to its parent; the identity unless ``<frame>`` elements lie in between.
    """


# %% writing body poses


@dataclass
class MJCFBodyPoseWriter:
    """
    Writes the poses that bodies of a world have relative to their parents in an MJCF
    file into the ``pos`` and ``quat`` of their ``<body>`` elements in that file.

    Only the start tags of bodies whose pose changed are rewritten, so the rest of the
    file keeps its formatting and comments.
    """

    file_path: str
    """
    The MJCF file the world was parsed from.
    """

    world: World
    """
    The world whose body poses are written.
    """

    decimals: int = 6
    """
    The number of decimals the written values are rounded to.
    """

    @classproperty
    def world_body_name(cls) -> str:
        """
        The name :class:`~semantic_digital_twin.adapters.mjcf.MJCFParser` gives the body
        of the ``<worldbody>``.
        """
        return "world"

    def write(self, bodies: list[Body]) -> list[Body]:
        """
        Write the pose of each body relative to its parent in the file, where it changed.

        :param bodies: The bodies whose poses are written.
        :return: the bodies whose pose changed and was written.
        """
        text = Path(self.file_path).read_text()
        placements = self._placements(
            ElementTree.fromstring(text).find(MJCFTag.WORLD_BODY),
            self.world_body_name,
            np.eye(4),
        )
        written_bodies = []
        for body in bodies:
            if body.name.name not in placements:
                raise BodyNotInMJCFFileError(
                    body_name=body.name.name, file_path=self.file_path
                )
            start_tag = self._find_start_tag(text, body.name.name)
            start_tag_with_pose = self._start_tag_with_pose(
                start_tag.group(0), self._pose_in_file(body, placements[body.name.name])
            )
            if start_tag_with_pose == start_tag.group(0):
                continue
            text = (
                text[: start_tag.start()]
                + start_tag_with_pose
                + text[start_tag.end() :]
            )
            written_bodies.append(body)
        Path(self.file_path).write_text(text)
        return written_bodies

    def _placements(
        self,
        element: ElementTree.Element,
        parent_name: str,
        parent_T_frame: np.ndarray,
    ) -> dict[str, MJCFBodyPlacement]:
        """
        :param element: The element whose nested bodies are collected.
        :param parent_name: The name of the body the element belongs to.
        :param parent_T_frame: The pose of the frame of the element, relative to that
            body.
        :return: the placement of every named body nested in the element, by name.
        """
        placements = {}
        for child in element:
            if child.tag == MJCFTag.FRAME:
                frame_T_child_frame = MJCFPose.from_element(child).matrix
                placements |= self._placements(
                    child, parent_name, parent_T_frame @ frame_T_child_frame
                )
            elif child.tag == MJCFTag.BODY:
                placements[child.get(MJCFAttribute.NAME)] = MJCFBodyPlacement(
                    parent_name=parent_name, parent_T_frame=parent_T_frame
                )
                placements |= self._placements(
                    child, child.get(MJCFAttribute.NAME), np.eye(4)
                )
        return placements

    @staticmethod
    def _find_start_tag(text: str, body_name: str) -> re.Match:
        """
        :param text: The content of the file.
        :param body_name: The name of a body the file has.
        :return: the start tag of that body.
        """
        return re.search(
            rf"<{MJCFTag.BODY}\b[^>]*\s{MJCFAttribute.NAME}\s*=\s*[\"']{re.escape(body_name)}[\"'][^>]*>",
            text,
        )

    def _pose_in_file(self, body: Body, placement: MJCFBodyPlacement) -> MJCFPose:
        """
        :param body: The body whose pose is written.
        :param placement: Where the body sits in the file.
        :return: the pose to write into the start tag of the body.
        """
        parent = self.world.get_body_by_name(placement.parent_name)
        parent_T_body = self.world.compute_forward_kinematics_np(parent, body)
        frame_T_body = np.linalg.inv(placement.parent_T_frame) @ parent_T_body
        return MJCFPose.from_matrix(frame_T_body).rounded(self.decimals)

    def _start_tag_with_pose(self, start_tag: str, pose: MJCFPose) -> str:
        """
        :param start_tag: The start tag of a body, as written in the file.
        :param pose: The pose to give the body.
        :return: the start tag unchanged if it already gives the pose up to the rounding,
            otherwise the start tag with the pose written in.
        """
        closing = "/>" if start_tag.endswith("/>") else ">"
        element = ElementTree.fromstring(start_tag[: -len(closing)] + "/>")
        tolerance = 0.5 * 10.0**-self.decimals
        if pose.is_close_to(MJCFPose.from_element(element), tolerance):
            return start_tag
        start_tag = self._with_attribute(
            start_tag, closing, MJCFAttribute.POSITION, pose.position
        )
        if (
            MJCFAttribute.QUATERNION in element.attrib
            or not pose.has_identity_orientation
        ):
            start_tag = self._with_attribute(
                start_tag, closing, MJCFAttribute.QUATERNION, pose.quaternion
            )
        return start_tag

    @staticmethod
    def _with_attribute(
        start_tag: str, closing: str, attribute: MJCFAttribute, values: np.ndarray
    ) -> str:
        """
        :param start_tag: A start tag.
        :param closing: How the start tag ends, ``>`` or ``/>``.
        :param attribute: The attribute to set.
        :param values: The values to set it to.
        :return: the start tag with the attribute replaced, or appended if it had none.
        """
        written_values = " ".join(
            np.format_float_positional(value, trim="-") for value in values
        )
        written_attribute = f'{attribute}="{written_values}"'
        existing_attribute = re.compile(rf"\s{attribute}\s*=\s*[\"'][^\"']*[\"']")
        if existing_attribute.search(start_tag):
            return existing_attribute.sub(" " + written_attribute, start_tag, count=1)
        return f"{start_tag[: -len(closing)].rstrip()} {written_attribute}{closing}"
