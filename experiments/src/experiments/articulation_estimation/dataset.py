"""
The RBO dataset of articulated objects and interactions by Martín-Martín, Eppner and
Brock (arXiv:1806.06465), Zenodo record 1036660 under CC BY 4.0: poses of the parts of
14 articulated objects, tracked by motion capture at 100 Hz while people move them.

Files are downloaded from its Zenodo record on first use; of each interaction only the
poses of the tracked parts are kept.
"""

from __future__ import annotations

# %% imports
import csv
import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from krrood.patterns.classproperty import classproperty
from krrood.patterns.path_enumeration import PathEnumeration

from semantic_digital_twin.articulation_estimation.observations import PoseObservations


# %% names


class JointKind(StrEnum):
    """
    The kinds of joint the dataset's object models name.
    """

    REVOLUTE = "REVOLUTE"
    PRISMATIC = "PRISMATIC"


class DatasetFileName(StrEnum):
    """
    The files of the dataset that the experiment reads.
    """

    INTERACTION_INDEX = "interactions_index.csv"
    RIGID_BODY_POSES = "rb_poses_array.csv"


class DatasetDirectory(PathEnumeration):
    """
    The directories, below the dataset's own, that hold the downloaded files.
    """

    MODELS = Path("models")
    INTERACTIONS = Path("interactions")


class RecordFileTemplate(StrEnum):
    """
    The names of the record's files and of the directories in its archives, with the
    fields to fill in.
    """

    MODEL_ARCHIVE = "{object_name}.tar.gz"
    INTERACTION_ARCHIVE = "{interaction_name}_o.tar.gz"
    MODEL_CONFIGURATION_DIRECTORY = "configuration_{recording_date}"
    MODEL_CONFIGURATION_FILES = "*_in.yaml"


class RigidBodyPosesColumn(StrEnum):
    """
    The columns of an interaction's poses of the marker sets, with the slot of the
    marker set and the field of its pose to fill in.
    """

    TIME = "%time"
    MARKER_SET_IDENTIFIER = "field.markers{slot}.id"
    MARKER_SET_POSE = "field.markers{slot}.pose.{pose_field}"


class PoseField(StrEnum):
    """
    The fields of a pose in the order the poses of the marker sets are read: the
    position, then the quaternion (x, y, z, w).
    """

    POSITION_X = "position.x"
    POSITION_Y = "position.y"
    POSITION_Z = "position.z"
    ORIENTATION_X = "orientation.x"
    ORIENTATION_Y = "orientation.y"
    ORIENTATION_Z = "orientation.z"
    ORIENTATION_W = "orientation.w"


class InteractionIndexColumn(StrEnum):
    """
    The columns of the interaction index that the experiment reads.
    """

    NAME = "Name"
    OBJECT = "Object"
    RECORDING_DATE = "recording date(marker set id)"


# %% object models


@dataclass(kw_only=True)
class ObjectJoint:
    """
    A joint of an object model: between the parts two marker sets track.
    """

    parent_marker_set: int
    """
    The identifier of the marker set on the parent part.
    """

    child_marker_set: int
    """
    The identifier of the marker set on the child part.
    """

    kind: JointKind
    """
    The kind of joint the model gives it.
    """

    @classmethod
    def from_json(cls, *, data: dict[str, object]) -> Self:
        """
        :param data: A joint as an object model's ``joints`` list holds it.
        :return: The joint.
        """
        return cls(
            parent_marker_set=int(data["rb1_id"]),
            child_marker_set=int(data["rb2_id"]),
            kind=JointKind(data["type"]),
        )


@dataclass(kw_only=True)
class ObjectModel:
    """
    The model of an object for one recording date, as far as the experiment reads it.
    """

    joints: list[ObjectJoint]
    """
    The joints between the object's parts.
    """

    @classmethod
    def from_json(cls, *, data: dict[str, object]) -> Self:
        """
        :param data: The contents of a model configuration file.
        :return: The model.
        """
        return cls(
            joints=[ObjectJoint.from_json(data=joint) for joint in data["joints"]]
        )


# %% interactions


@dataclass(kw_only=True)
class Interaction:
    """
    One recording of a person moving an object.
    """

    name: str
    """
    The name of the recording, such as ``microwave01``.
    """

    object_name: str
    """
    The name of the object moved.
    """

    recording_date: str
    """
    The day of the recording, which selects the configuration of the object model.
    """

    @classmethod
    def from_json(cls, *, data: dict[str, str]) -> Self:
        """
        :param data: A row of the interaction index.
        :return: The interaction.
        """
        return cls(
            name=data[InteractionIndexColumn.NAME],
            object_name=data[InteractionIndexColumn.OBJECT],
            recording_date=data[InteractionIndexColumn.RECORDING_DATE],
        )


@dataclass(kw_only=True)
class JointMotion:
    """
    The observed poses of a joint's child part relative to its parent part during one
    interaction, at the times both parts were tracked.
    """

    interaction: Interaction
    """
    The recording.
    """

    joint_index: int
    """
    The position of the joint in the object model's list of joints.
    """

    joint: ObjectJoint
    """
    The joint.
    """

    times: npt.NDArray[np.float64]
    """
    The time of each observation, in seconds since the first.
    """

    observations: PoseObservations
    """
    The poses of the child part relative to the parent part.
    """


@dataclass(kw_only=True)
class MarkerSetPoses:
    """
    The poses of every marker set an interaction tracked, in the motion capture frame.
    """

    times: npt.NDArray[np.float64]
    """
    The time of each row of the recording, in seconds since the first.
    """

    world_T_marker_set_stacks: dict[int, npt.NDArray[np.float64]]
    """
    For each marker set, its pose at each row, shape (n, 4, 4), not a number where the
    marker set was not seen.
    """

    @classmethod
    def from_csv(cls, *, path: Path) -> Self:
        """
        :param path: An interaction's ``rb_poses_array.csv``. The order of the marker sets
            can change from row to row, so each row is read by marker set identifier.
        :return: The poses in the file.
        """
        with path.open(newline="") as file:
            rows = csv.reader(file)
            header = next(rows)
            data = list(rows)
        column_of = {name: index for index, name in enumerate(header)}
        identifier_pattern = re.escape(
            RigidBodyPosesColumn.MARKER_SET_IDENTIFIER
        ).replace(re.escape("{slot}"), r"(\d+)")
        slots = sorted(
            {
                int(match.group(1))
                for name in header
                if (match := re.fullmatch(identifier_pattern, name))
            }
        )
        values: dict[int, npt.NDArray[np.float64]] = {}
        for row_index, row in enumerate(data):
            for slot in slots:
                identifier_column = column_of[
                    RigidBodyPosesColumn.MARKER_SET_IDENTIFIER.format(slot=slot)
                ]
                if identifier_column >= len(row) or row[identifier_column] == "":
                    continue
                marker_set = int(row[identifier_column])
                values.setdefault(marker_set, np.full((len(data), 7), np.nan))[
                    row_index
                ] = [
                    float(
                        row[
                            column_of[
                                RigidBodyPosesColumn.MARKER_SET_POSE.format(
                                    slot=slot, pose_field=pose_field
                                )
                            ]
                        ]
                    )
                    for pose_field in PoseField
                ]
        nanoseconds = np.array(
            [int(row[column_of[RigidBodyPosesColumn.TIME]]) for row in data]
        )
        return cls(
            times=(nanoseconds - nanoseconds[0]) / cls.nanoseconds_per_second,
            world_T_marker_set_stacks={
                marker_set: cls.poses_from_positions_and_quaternions(values=entries)
                for marker_set, entries in values.items()
            },
        )

    @classproperty
    def nanoseconds_per_second(cls) -> float:
        """
        The recordings' time stamps are in nanoseconds: :math:`10^9` per second.
        """
        return 1e9

    @classproperty
    def quaternion_norm_tolerance(cls) -> float:
        """
        How far a recorded quaternion's length may lie from one for its pose to count as
        seen: :math:`10^{-3}`, since the recordings round them.
        """
        return 1e-3

    @classmethod
    def poses_from_positions_and_quaternions(
        cls, *, values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :param values: Per row a position and a quaternion (x, y, z, w), shape (n, 7).
        :return: The poses, shape (n, 4, 4), not a number in rows whose quaternion is
            missing or not of unit length.
        """
        quaternion_norms = np.linalg.norm(np.nan_to_num(values[:, 3:]), axis=1)
        seen = np.isclose(quaternion_norms, 1.0, atol=cls.quaternion_norm_tolerance)
        poses = np.full((len(values), 4, 4), np.nan)
        poses[seen] = np.eye(4)
        poses[seen, :3, :3] = Rotation.from_quat(values[seen, 3:]).as_matrix()
        poses[seen, :3, 3] = values[seen, :3]
        return poses

    def joint_motion(
        self, *, interaction: Interaction, joint_index: int, joint: ObjectJoint
    ) -> JointMotion | None:
        """
        :return: The motion of the joint at the rows where both of its marker sets were
            seen, or nothing if one of them never was or fewer than two rows remain.
        """
        stacks = self.world_T_marker_set_stacks
        if (
            joint.parent_marker_set not in stacks
            or joint.child_marker_set not in stacks
        ):
            return None
        world_T_parent = stacks[joint.parent_marker_set]
        world_T_child = stacks[joint.child_marker_set]
        seen = ~np.isnan(world_T_parent[:, 0, 0]) & ~np.isnan(world_T_child[:, 0, 0])
        if np.sum(seen) < PoseObservations.required_observation_count:
            return None
        return JointMotion(
            interaction=interaction,
            joint_index=joint_index,
            joint=joint,
            times=self.times[seen] - self.times[seen][0],
            observations=PoseObservations(
                parent_T_child_stack=np.linalg.inv(world_T_parent[seen])
                @ world_T_child[seen]
            ),
        )


# %% the dataset


@dataclass(kw_only=True)
class RBODataset:
    """
    The dataset in a local directory, filled from its Zenodo record as files are needed.
    """

    directory: Path
    """
    Where the downloaded files are kept.
    """

    record_files_url: str = "https://zenodo.org/records/1036660/files"
    """
    The address of the record's files.
    """

    model_archive_exceptions: dict[str, str] = field(
        default_factory=lambda: {"tripod": "tripod.tar"}
    )
    """
    The model archives not named by :attr:`RecordFileTemplate.MODEL_ARCHIVE`, by object:

    the tripod's is named ``tripod.tar``, though it is compressed too.
    """

    @property
    def interactions(self) -> list[Interaction]:
        """
        The recordings the dataset's index lists.
        """
        path = self.directory / DatasetFileName.INTERACTION_INDEX
        if not path.exists():
            self.download_to(
                url=self.file_url(key=DatasetFileName.INTERACTION_INDEX), path=path
            )
        with path.open(newline="") as file:
            return [Interaction.from_json(data=row) for row in csv.DictReader(file)]

    def joints(self, *, interaction: Interaction) -> list[ObjectJoint]:
        """
        :return: The joints of the object model configured for the interaction's
            recording date.
        """
        model_directory = (
            self.directory / DatasetDirectory.MODELS / interaction.object_name
        )
        if not model_directory.exists():
            self.download_model(object_name=interaction.object_name)
        configuration = next(
            (
                model_directory
                / RecordFileTemplate.MODEL_CONFIGURATION_DIRECTORY.format(
                    recording_date=interaction.recording_date
                )
            ).glob(RecordFileTemplate.MODEL_CONFIGURATION_FILES)
        )
        with configuration.open() as file:
            return ObjectModel.from_json(data=yaml.safe_load(file)).joints

    def joint_motions(self, *, interaction: Interaction) -> list[JointMotion]:
        """
        :return: The motion of each joint of the object whose two parts were both
            tracked during the interaction.
        """
        path = (
            self.directory
            / DatasetDirectory.INTERACTIONS
            / interaction.name
            / (DatasetFileName.RIGID_BODY_POSES)
        )
        if not path.exists():
            self.download_rigid_body_poses(interaction=interaction, path=path)
        poses = MarkerSetPoses.from_csv(path=path)
        motions = [
            poses.joint_motion(
                interaction=interaction, joint_index=joint_index, joint=joint
            )
            for joint_index, joint in enumerate(self.joints(interaction=interaction))
        ]
        return [motion for motion in motions if motion is not None]

    def file_url(self, *, key: str) -> str:
        """
        :return: The download address of a file of the record.
        """
        return f"{self.record_files_url}/{key}?download=1"

    def download_to(self, *, url: str, path: Path) -> None:
        """
        Downloads a file, writing it under a temporary name first so that an interrupted
        download leaves no file behind that looks complete.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial.rename(path)

    def download_model(self, *, object_name: str) -> None:
        """
        Downloads and unpacks an object's model: its joints per recording date and its
        meshes, see :attr:`model_archive_exceptions`.
        """
        key = self.model_archive_exceptions.get(
            object_name,
            RecordFileTemplate.MODEL_ARCHIVE.format(object_name=object_name),
        )
        models = self.directory / DatasetDirectory.MODELS
        partial = models / f"{object_name}.partial"
        with urllib.request.urlopen(self.file_url(key=key)) as response:
            with tarfile.open(fileobj=response, mode="r|*") as archive:
                archive.extractall(partial, filter="data")
        (partial / object_name).rename(models / object_name)
        shutil.rmtree(partial)

    def download_rigid_body_poses(
        self, *, interaction: Interaction, path: Path
    ) -> None:
        """
        Streams an interaction's archive, most of which are images, and keeps only its
        poses of the marker sets, stopping as soon as they are read.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        url = self.file_url(
            key=RecordFileTemplate.INTERACTION_ARCHIVE.format(
                interaction_name=interaction.name
            )
        )
        with urllib.request.urlopen(url) as response:
            with tarfile.open(fileobj=response, mode="r|*") as archive:
                member = next(
                    member
                    for member in archive
                    if Path(member.name).name == DatasetFileName.RIGID_BODY_POSES
                )
                with archive.extractfile(member) as source, partial.open("wb") as file:
                    shutil.copyfileobj(source, file)
        partial.rename(path)
