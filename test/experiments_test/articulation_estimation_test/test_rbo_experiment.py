"""
Tests of the RBO articulation estimation experiment on a small synthetic dataset written
in the dataset's own layout, with the downloads mocked.
"""

from __future__ import annotations

# %% Imports
import csv
import io
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from numpy.testing import assert_allclose
from scipy.spatial.transform import RigidTransform, Rotation

from experiments.articulation_estimation import dataset
from experiments.articulation_estimation.dataset import (
    DatasetDirectory,
    DatasetFileName,
    InteractionIndexColumn,
    JointKind,
    PoseField,
    RBODataset,
    RecordFileTemplate,
    RigidBodyPosesColumn,
)
from experiments.articulation_estimation.evaluation import FullFit, JointAxis
from experiments.articulation_estimation.runner import RBOExperiment
from semantic_digital_twin.articulation_estimation.observations import ObservationNoise


# %% A synthetic dataset
@dataclass(kw_only=True)
class SyntheticJoint:
    """
    A joint between two marker sets, moved through known positions and recorded in the
    layout of the RBO dataset.
    """

    kind: JointKind
    """
    The kind of the joint.
    """

    screw_axis_entries: npt.NDArray[np.float64]
    """
    The rotation and then the translation per unit of position, in the parent's frame.
    """

    positions: npt.NDArray[np.float64]
    """
    The position at each row of the recording.
    """

    parent_marker_set: int = 3
    """
    The identifier of the parent's marker set.
    """

    child_marker_set: int = 7
    """
    The identifier of the child's marker set.
    """

    world_T_parent: npt.NDArray[np.float64] = field(
        default_factory=lambda: RigidTransform.from_exp_coords(
            [0.1, -0.2, 0.3, 1.0, 0.5, 0.8]
        ).as_matrix()
    )
    """
    Where the parent stands in the motion capture frame.
    """

    reference_parent_T_child: npt.NDArray[np.float64] = field(
        default_factory=lambda: RigidTransform.from_exp_coords(
            [0.0, 0.2, -0.1, 0.4, 0.1, 0.0]
        ).as_matrix()
    )
    """
    The pose of the child at position zero.
    """

    @property
    def parent_T_child_stack(self) -> npt.NDArray[np.float64]:
        """
        The true pose of the child at each row.
        """
        return (
            RigidTransform.from_exp_coords(
                np.outer(self.positions, self.screw_axis_entries)
            ).as_matrix()
            @ self.reference_parent_T_child
        )

    def write_rigid_body_poses(
        self, *, path: Path, noise: ObservationNoise, seed: int, missing_row: int
    ) -> None:
        """
        Writes the recording with the marker sets in alternating slots, the child's
        missing in ``missing_row``, and each pose disturbed by ``noise``.
        """
        random_numbers = np.random.default_rng(seed)
        world_T_child = self.world_T_parent @ self.parent_T_child_stack
        world_T_child[:, :3, 3] += random_numbers.normal(
            scale=noise.position_standard_deviation, size=(len(self.positions), 3)
        )
        world_T_child[:, :3, :3] = (
            Rotation.from_matrix(world_T_child[:, :3, :3])
            * Rotation.from_rotvec(
                random_numbers.normal(
                    scale=noise.orientation_standard_deviation,
                    size=(len(self.positions), 3),
                )
            )
        ).as_matrix()
        header = [RigidBodyPosesColumn.TIME] + [
            column
            for slot in (0, 1)
            for column in [RigidBodyPosesColumn.MARKER_SET_IDENTIFIER.format(slot=slot)]
            + [
                RigidBodyPosesColumn.MARKER_SET_POSE.format(
                    slot=slot, pose_field=pose_field
                )
                for pose_field in PoseField
            ]
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for row_index, world_T_child_pose in enumerate(world_T_child):
                marker_sets = [
                    (self.parent_marker_set, self.world_T_parent),
                    (self.child_marker_set, world_T_child_pose),
                ]
                if row_index % 2:
                    marker_sets.reverse()
                row = [str(1_000_000_000 + 10_000_000 * row_index)]
                for marker_set, pose in marker_sets:
                    if marker_set == self.child_marker_set and row_index == missing_row:
                        row += [""] * (1 + len(PoseField))
                        continue
                    row += [str(marker_set)] + [
                        repr(float(value))
                        for value in np.concatenate(
                            [pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
                        )
                    ]
                writer.writerow(row)


@dataclass(kw_only=True)
class SyntheticDataset:
    """
    Recordings of one object with one joint, written to a directory in the dataset's
    layout.
    """

    directory: Path
    """
    The dataset's directory.
    """

    joint: SyntheticJoint
    """
    The object's joint.
    """

    object_name: str = "door"
    """
    The object's name.
    """

    recording_date: str = "2020-01-01"
    """
    The day of all recordings.
    """

    noise: ObservationNoise = field(
        default_factory=lambda: ObservationNoise(
            position_standard_deviation=5e-4, orientation_standard_deviation=2e-3
        )
    )
    """
    The noise of the recorded poses.
    """

    missing_row: int = 5
    """
    The row in which the child's marker set was not seen.
    """

    def interaction_names(self, *, count: int) -> list[str]:
        """
        :return: The names of ``count`` recordings.
        """
        return [f"{self.object_name}{index:02d}" for index in range(1, count + 1)]

    def write(self, *, interaction_count: int) -> RBODataset:
        """
        :return: The dataset with ``interaction_count`` recordings of the joint, each with
            its own noise.
        """
        with (self.directory / DatasetFileName.INTERACTION_INDEX).open(
            "w", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(InteractionIndexColumn))
            writer.writeheader()
            for name in self.interaction_names(count=interaction_count):
                writer.writerow(
                    {
                        InteractionIndexColumn.NAME: name,
                        InteractionIndexColumn.OBJECT: self.object_name,
                        InteractionIndexColumn.RECORDING_DATE: self.recording_date,
                    }
                )
        configuration = (
            self.directory
            / DatasetDirectory.MODELS
            / self.object_name
            / RecordFileTemplate.MODEL_CONFIGURATION_DIRECTORY.format(
                recording_date=self.recording_date
            )
            / f"{self.object_name}_in.yaml"
        )
        configuration.parent.mkdir(parents=True)
        configuration.write_text(yaml.safe_dump(self.model_contents))
        for seed, name in enumerate(self.interaction_names(count=interaction_count)):
            self.joint.write_rigid_body_poses(
                path=self.directory
                / DatasetDirectory.INTERACTIONS
                / name
                / DatasetFileName.RIGID_BODY_POSES,
                noise=self.noise,
                seed=seed,
                missing_row=self.missing_row,
            )
        return RBODataset(directory=self.directory)

    @property
    def model_contents(self) -> dict[str, list[dict[str, object]]]:
        """
        The object model as its configuration file holds it.
        """
        return {
            "joints": [
                {
                    "rb1_id": self.joint.parent_marker_set,
                    "rb2_id": self.joint.child_marker_set,
                    "type": str(self.joint.kind),
                }
            ]
        }


def revolute_joint() -> SyntheticJoint:
    """
    :return: A door turning 1.2 rad about a vertical axis 0.3 m from the parent's origin.
    """
    return SyntheticJoint(
        kind=JointKind.REVOLUTE,
        screw_axis_entries=np.array([0.0, 0.0, 1.0, 0.0, -0.3, 0.0]),
        positions=np.linspace(0.0, 1.2, 120),
    )


def prismatic_joint() -> SyntheticJoint:
    """
    :return: A drawer sliding 0.3 m.
    """
    return SyntheticJoint(
        kind=JointKind.PRISMATIC,
        screw_axis_entries=np.array([0.0, 0.0, 0.0, 0.6, 0.8, 0.0]),
        positions=np.linspace(0.0, 0.3, 120),
    )


# %% Reading the dataset
def test_joint_motion_reads_each_row_by_marker_set_identifier(tmp_path: Path):
    """
    The marker sets swap slots from row to row, and a row without the child is left out:

    the poses read are the true ones.
    """
    synthetic = SyntheticDataset(
        directory=tmp_path,
        joint=revolute_joint(),
        noise=ObservationNoise(
            position_standard_deviation=0.0, orientation_standard_deviation=0.0
        ),
    )
    rbo = synthetic.write(interaction_count=1)

    (motion,) = rbo.joint_motions(interaction=rbo.interactions[0])

    expected = np.delete(
        synthetic.joint.parent_T_child_stack, synthetic.missing_row, axis=0
    )
    assert_allclose(motion.observations.parent_T_child_stack, expected, atol=1e-9)
    assert motion.times[0] == 0
    assert motion.joint.kind == JointKind.REVOLUTE


# %% Downloads
@dataclass(kw_only=True)
class RecordedDownloads:
    """
    Stands in for the record's server: serves archives from memory and remembers the
    addresses asked for.
    """

    archives: dict[str, bytes]
    """
    The contents of each archive, by the name of its file in the record.
    """

    requested_urls: list[str] = field(default_factory=list)
    """
    The addresses asked for, in order.
    """

    def open_url(self, url: str) -> io.BytesIO:
        """
        :return: The archive the address names.
        """
        self.requested_urls.append(url)
        key = url.rsplit("/", 1)[1].split("?")[0]
        return io.BytesIO(self.archives[key])


def gzipped_archive(*, files: dict[str, bytes]) -> bytes:
    """
    :return: A compressed tar archive holding ``files`` by path, in order.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, contents in files.items():
            information = tarfile.TarInfo(name=name)
            information.size = len(contents)
            archive.addfile(information, io.BytesIO(contents))
    return buffer.getvalue()


@pytest.mark.parametrize("object_name", ["door", "tripod"])
def test_download_model_unpacks_the_object_model(
    object_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    The model is downloaded on first use from the record's archive of the object, whose
    name the tripod does not follow.
    """
    synthetic = SyntheticDataset(
        directory=tmp_path, joint=revolute_joint(), object_name=object_name
    )
    rbo = RBODataset(directory=tmp_path)
    key = rbo.model_archive_exceptions.get(
        object_name, RecordFileTemplate.MODEL_ARCHIVE.format(object_name=object_name)
    )
    configuration = RecordFileTemplate.MODEL_CONFIGURATION_DIRECTORY.format(
        recording_date=synthetic.recording_date
    )
    downloads = RecordedDownloads(
        archives={
            key: gzipped_archive(
                files={
                    f"{object_name}/{configuration}/{object_name}_in.yaml": yaml.safe_dump(
                        synthetic.model_contents
                    ).encode()
                }
            )
        }
    )
    monkeypatch.setattr(dataset.urllib.request, "urlopen", downloads.open_url)

    rbo.download_model(object_name=object_name)

    assert downloads.requested_urls == [rbo.file_url(key=key)]
    model_files = list((tmp_path / DatasetDirectory.MODELS / object_name).rglob("*"))
    assert [path.name for path in model_files if path.is_file()] == [
        f"{object_name}_in.yaml"
    ]


def test_download_keeps_only_the_poses_of_an_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Of an interaction's archive only the poses of the marker sets are kept.
    """
    rbo = RBODataset(directory=tmp_path)
    poses = b"recorded poses"
    downloads = RecordedDownloads(
        archives={
            RecordFileTemplate.INTERACTION_ARCHIVE.format(
                interaction_name="door01"
            ): gzipped_archive(
                files={
                    "door01/images/0001.png": b"an image",
                    f"door01/{DatasetFileName.RIGID_BODY_POSES}": poses,
                }
            )
        }
    )
    monkeypatch.setattr(dataset.urllib.request, "urlopen", downloads.open_url)
    synthetic = SyntheticDataset(directory=tmp_path, joint=revolute_joint())
    path = tmp_path / DatasetDirectory.INTERACTIONS / "door01" / "poses.csv"

    rbo.download_rigid_body_poses(
        interaction=dataset.Interaction(
            name="door01",
            object_name=synthetic.object_name,
            recording_date=synthetic.recording_date,
        ),
        path=path,
    )

    assert path.read_bytes() == poses
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


# %% The full fit
@pytest.mark.parametrize(
    "joint, travel",
    [(revolute_joint(), 1.2), (prismatic_joint(), 0.3)],
    ids=["revolute", "prismatic"],
)
def test_excursion_ignores_an_outlier(
    joint: SyntheticJoint, travel: float, tmp_path: Path
):
    """
    A pose that jumped far away leaves the excursion at the angle turned or the distance
    slid.
    """
    rbo = SyntheticDataset(directory=tmp_path, joint=joint).write(interaction_count=1)
    (motion,) = rbo.joint_motions(interaction=rbo.interactions[0])
    motion.observations.parent_T_child_stack[60, :3, 3] += [0.5, -0.4, 0.3]

    fit = FullFit.from_motion(
        motion=motion,
        initial_noise=ObservationNoise(
            position_standard_deviation=1e-3, orientation_standard_deviation=5e-3
        ),
    )

    assert fit.excursion[-1] == pytest.approx(travel, abs=0.01)


# %% The experiment
@pytest.mark.parametrize(
    "joint", [revolute_joint(), prismatic_joint()], ids=["revolute", "prismatic"]
)
def test_experiment_recovers_a_joint_recorded_three_times(
    joint: SyntheticJoint, tmp_path: Path
):
    """
    Each of three recordings of the same joint agrees with the reference the other two
    make.
    """
    rbo = SyntheticDataset(directory=tmp_path, joint=joint).write(interaction_count=3)

    results = RBOExperiment(dataset=rbo, process_count=1).run()

    assert len(results) == 3
    for result in results:
        assert result.kind == joint.kind
        assert result.full_fit_errors_to_reference.direction_degrees < 0.5
        assert len(result.partial_fits) == 3


# %% Axes
@pytest.mark.parametrize("child_distance", [0.0, 0.5, 5.0])
def test_prismatic_direction_is_the_child_sliding_direction(child_distance: float):
    """
    A fit of a slide that turns a little gives the direction the child moves in, however
    far the child lies from the parent's origin.
    """
    sliding_direction = np.array([0.6, 0.8, 0.0])
    rotation = np.array([0.0, 0.0, 0.05])
    child_position = child_distance * np.array([0.3, -0.4, 0.866])
    entries = np.concatenate(
        [rotation, sliding_direction - np.cross(rotation, child_position)]
    )

    axis = JointAxis.from_screw_axis_entries(
        kind=JointKind.PRISMATIC, entries=entries, child_position=child_position
    )

    assert_allclose(axis.direction, sliding_direction)
