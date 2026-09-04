from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from krrood.exceptions import DataclassException


@dataclass
class WarsawSceneNotFoundError(DataclassException, FileNotFoundError):
    """
    Raised when a directory holds no Warsaw scene mesh.
    """

    directory: Path
    """
    The directory that was searched.
    """

    scene_mesh_pattern: str
    """
    The pattern a scene mesh was looked for under.
    """

    def error_message(self) -> str:
        return (
            f"No Warsaw scene mesh matching '{self.scene_mesh_pattern}' in "
            f"'{self.directory}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Point the loader at a scene directory, which holds the one mesh the scene "
            "is written as."
        )


@dataclass
class AmbiguousWarsawSceneError(DataclassException, FileNotFoundError):
    """
    Raised when a directory holds more than one mesh, leaving undecided which of them
    the scene is.
    """

    directory: Path
    """
    The directory that was searched.
    """

    scene_meshes: List[Path]
    """
    The meshes it holds.
    """

    def error_message(self) -> str:
        names = ", ".join(sorted(path.name for path in self.scene_meshes))
        return f"'{self.directory}' holds {len(self.scene_meshes)} meshes: {names}."

    def suggest_correction(self) -> str:
        return (
            "Give the scene a directory of its own, or name its mesh file directly with "
            "WarsawScene.from_file."
        )


@dataclass
class WarsawLabelsMissingError(DataclassException, ValueError):
    """
    Raised when a scene mesh carries no per-face class labels.
    """

    scene_mesh: Path
    """
    The mesh that was read.
    """

    def error_message(self) -> str:
        return f"'{self.scene_mesh}' carries no per-face class labels."

    def suggest_correction(self) -> str:
        return (
            "A scene writes its labels as one integer face property per class. Only the "
            "PLY reader keeps them, under the mesh's '_ply_raw' metadata, so a mesh that "
            "was re-exported or loaded with processing has lost them and has to be "
            "written again."
        )


@dataclass
class WarsawLabelsMisalignedError(DataclassException, ValueError):
    """
    Raised when the faces a world was built from are not the faces the labels were
    written for, which would leave every segment pointing at another object's geometry.
    """

    scene_mesh: Path
    """
    The mesh that was read.
    """

    labelled_faces: int
    """
    How many faces the labels were written for.
    """

    loaded_faces: int
    """
    How many faces the world's scene body ended up with.
    """

    def error_message(self) -> str:
        return (
            f"'{self.scene_mesh}' was labelled for {self.labelled_faces} faces, but the "
            f"world's scene body holds {self.loaded_faces} of them."
        )

    def suggest_correction(self) -> str:
        return (
            "The mesh has to reach the world unprocessed: welding vertices or dropping "
            "degenerate faces renumbers the faces and leaves every label pointing at "
            "another one."
        )
