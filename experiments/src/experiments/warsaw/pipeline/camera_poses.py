"""
Where the camera stood for every picture a run shows a model.

A run's pictures are its evidence, and the poses that produced them are worked out while
rendering and are otherwise dropped when the render returns. Anything that later wants
to measure against a picture -- reprojecting what a model said into the scene, or asking
what was in view and what was hidden -- needs the pose, and recomputing it afterwards
means rebuilding the world, the framing and three camera constants that live only as
defaults.

Kept beside the pictures rather than in the run's other records, because the pictures
are written a render at a time into a directory of their own and the pose belongs with
the picture it belongs to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing_extensions import Dict, List, Sequence

# %% where the poses are kept

CAMERA_POSES_FILE = "camera_poses.json"
"""
What the record is called, in every directory a run writes pictures into.
"""

# %% the poses themselves


@dataclass(frozen=True)
class CameraPoses:
    """
    Where the camera stood for each picture in one directory, and what it saw with.
    """

    frame: str
    """
    What the poses are relative to.
    """

    field_of_view: List[float]
    """
    How wide the camera saw, in degrees across and down.

    Kept because it is a loader default rather than a setting, so nothing else a run
    writes records it, and a pose without it builds no camera.
    """

    poses: Dict[str, List[List[float]]] = field(default_factory=dict)
    """
    The camera pose each picture was taken from, by the name it was written as.
    """

    @classmethod
    def beside(cls, directory: Path) -> CameraPoses:
        """
        Read the poses recorded in a directory.

        :param directory: The directory the pictures were written into.
        :return: What is recorded there, or an empty record where no render has reached
            it yet.
        """
        written = Path(directory) / CAMERA_POSES_FILE
        if not written.exists():
            return cls(frame="", field_of_view=[], poses={})
        held = json.loads(written.read_text())
        return cls(
            frame=held["frame"],
            field_of_view=held["field_of_view"],
            poses=held["poses"],
        )

    @classmethod
    def record(
        cls,
        directory: Path,
        frame: str,
        field_of_view: Sequence[float],
        poses: Dict[str, np.ndarray],
    ) -> Path:
        """
        Add the poses of some pictures to what a directory already records.

        Added rather than written over: one directory is filled by a render per label,
        each writing as it goes, so replacing would leave every picture but the last
        unaccounted for.

        :param directory: The directory the pictures were written into.
        :param frame: What the poses are relative to.
        :param field_of_view: How wide the camera saw, in degrees across and down.
        :param poses: The pose each picture was taken from, by the name it was written
            as.
        :return: The file the record was written to.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        already = cls.beside(directory)
        recorded = cls(
            frame=frame,
            field_of_view=[float(one) for one in field_of_view],
            poses={
                **already.poses,
                **{
                    name: np.asarray(pose, dtype=np.float64).tolist()
                    for name, pose in poses.items()
                },
            },
        )
        written = directory / CAMERA_POSES_FILE
        written.write_text(json.dumps(recorded.as_json(), indent=2))
        return written

    def as_json(self) -> Dict:
        """
        :return: The record as a plain document.
        """
        return {
            "frame": self.frame,
            "field_of_view": self.field_of_view,
            "poses": self.poses,
        }

    def pose_of(self, picture: str) -> np.ndarray:
        """
        :param picture: The name a picture was written as.
        :return: The camera pose it was taken from.
        """
        return np.asarray(self.poses[picture], dtype=np.float64)
