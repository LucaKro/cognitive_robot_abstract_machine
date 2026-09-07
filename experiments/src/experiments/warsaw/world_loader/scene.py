"""
A scanned scene, and the objects its labels mark out in it.

A scene arrives as one mesh whose faces carry one integer property per class, naming
which object of that class each face belongs to. The same face can carry several, since
a drawer front is labelled both as the drawer and as the cabinet holding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import trimesh
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from typing_extensions import Dict, Iterable, Iterator, List

from experiments.warsaw.exceptions import (
    AmbiguousWarsawSceneError,
    WarsawLabelsMissingError,
    WarsawSceneNotFoundError,
)

# %% where a scan keeps its labels


class PlyPayload(StrEnum):
    """
    Where a PLY file keeps the labels a scan wrote onto its faces.

    The path into the payload is the file format's, not ours, so it is named here rather
    than spelled at the one place that walks it.
    """

    RAW = "_ply_raw"
    """
    The header and data of the file, as the reader kept them.
    """

    FACE = "face"
    """
    The element the labels are written per.
    """

    DATA = "data"
    """
    The rows themselves, one per face.
    """

    GEOMETRY = "vertex_indices"
    """
    The face property holding a face's geometry rather than one of its labels.
    """


# %% the objects a scan labels


def segment_label(segments: Iterable[LabelSegment], maximum_length: int = 120) -> str:
    """
    Name the segments a render highlights, so its filename says what is colored in it.

    :param segments: The segments highlighted in the render.
    :param maximum_length: How many characters of names the filename can hold.
    :return: Their names joined by dashes, cut short of that length.
    """
    names = [str(segment.name).replace(" ", "_") for segment in segments]
    if len("-".join(names)) <= maximum_length:
        return "-".join(names)

    kept: List[str] = []
    length = 0
    for name in names:
        if length + len(name) + 1 > maximum_length:
            break
        kept.append(name)
        length += len(name) + 1
    return "-".join(kept + [f"and_{len(names) - len(kept)}_more"])


@dataclass
class LabelSegment:
    """
    One object a Warsaw scene labels: the faces one of its classes marks as one
    instance.

    A face can belong to segments of several classes at once, since a scene labels, for
    example, a drawer's front both as ``drawer`` and as the ``cabinet`` holding it.
    """

    class_name: str
    """
    The class that labels this object.
    """

    instance: int
    """
    Which object of that class this is.
    """

    face_indices: np.ndarray
    """
    Which of the scene mesh's faces this object is made of.
    """

    @property
    def name(self) -> PrefixedName:
        """
        :return: The name identifying this object among the scene's objects.
        """
        return PrefixedName(f"{self.class_name}_{self.instance}")

    def __len__(self) -> int:
        """
        :return: How many of the scene's faces this object is made of.
        """
        return len(self.face_indices)


# %% the scan itself


@dataclass
class WarsawScene:
    """
    A Warsaw scene: one mesh whose faces carry, per class, the instance they belong to.

    The scene is stored as a single mesh with one integer face property per class.
    Instance :attr:`unsegmented` marks the faces a class does not cover.
    """

    mesh_path: Path
    """
    The file the scene was read from.
    """

    mesh: trimesh.Trimesh
    """
    The scene's geometry, carrying the colors it was scanned in.
    """

    face_labels: Dict[str, np.ndarray]
    """
    Per class, the instance each face belongs to.
    """

    unsegmented: int = 0
    """
    The instance marking every face a class does not cover.
    """

    world_T_source: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=-np.pi / 2
        )
    )
    """
    Turns the scene from the frame it is written in into the world's.

    The scene measures height down its own y axis, so a floor is written at a greater y
    than the ceiling above it. This rolls that axis onto the world's upward z.
    """

    @classmethod
    def from_directory(
        cls, directory: Path, scene_mesh_pattern: str = "*.ply"
    ) -> WarsawScene:
        """
        Read the scene a directory holds.

        :param directory: The directory holding the scene's mesh.
        :param scene_mesh_pattern: How that mesh is named.
        :raises WarsawSceneNotFoundError: If the directory holds no mesh.
        :raises AmbiguousWarsawSceneError: If it holds more than one.
        """
        directory = Path(directory)
        scene_meshes = sorted(directory.glob(scene_mesh_pattern))
        if not scene_meshes:
            raise WarsawSceneNotFoundError(
                directory=directory, scene_mesh_pattern=scene_mesh_pattern
            )
        if len(scene_meshes) > 1:
            raise AmbiguousWarsawSceneError(
                directory=directory, scene_meshes=scene_meshes
            )
        return cls.from_file(scene_meshes[0])

    @classmethod
    def from_file(cls, scene_mesh_path: Path) -> WarsawScene:
        """
        Read the scene one mesh file holds.

        :param scene_mesh_path: The mesh to read.
        :raises WarsawLabelsMissingError: If the mesh carries no per-face class labels.
        """
        scene_mesh_path = Path(scene_mesh_path)
        # Processing welds vertices and drops degenerate faces, which renumbers the
        # faces and would leave every label pointing at another face than the one it
        # was written for.
        mesh = trimesh.load(scene_mesh_path, process=False)
        return cls(
            mesh_path=scene_mesh_path,
            mesh=mesh,
            face_labels=cls._read_face_labels(mesh, scene_mesh_path),
        )

    @staticmethod
    def _read_face_labels(
        mesh: trimesh.Trimesh,
        scene_mesh_path: Path,
        geometry_property: str = PlyPayload.GEOMETRY,
    ) -> Dict[str, np.ndarray]:
        """
        Read the instance each face belongs to, per class.

        :param mesh: The mesh the scene was read from.
        :param scene_mesh_path: The file it was read from, for the error message.
        :param geometry_property: The face property holding a face's geometry rather
            than one of its labels.
        :return: Per class, the instance each face belongs to.
        :raises WarsawLabelsMissingError: If the mesh carries no labels.
        """
        raw = mesh.metadata.get(PlyPayload.RAW.value) or {}
        face = raw.get(PlyPayload.FACE.value) or {}
        faces = face.get(PlyPayload.DATA.value)
        named = faces.dtype.names if faces is not None else None
        class_names = [name for name in (named or ()) if name != geometry_property]
        if not class_names:
            raise WarsawLabelsMissingError(scene_mesh=scene_mesh_path)
        return {name: np.asarray(faces[name]) for name in class_names}

    @property
    def class_names(self) -> List[str]:
        """
        :return: The classes the scene is labelled by, in the order it declares them.
        """
        return list(self.face_labels)

    def segments(self) -> Iterator[LabelSegment]:
        """
        :return: Every object the scene labels, in class order.
        """
        for class_name, instances in self.face_labels.items():
            for instance in np.unique(instances):
                if instance == self.unsegmented:
                    continue
                yield LabelSegment(
                    class_name=class_name,
                    instance=int(instance),
                    face_indices=np.flatnonzero(instances == instance),
                )
