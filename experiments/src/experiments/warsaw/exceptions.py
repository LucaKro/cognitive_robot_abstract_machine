from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    scene_mesh_file: str
    """
    The name of the scene mesh file that was looked for.
    """

    def error_message(self) -> str:
        return f"No Warsaw scene mesh '{self.scene_mesh_file}' in '{self.directory}'."

    def suggest_correction(self) -> str:
        return (
            "Point the loader at a scene directory, which holds the scene mesh beside "
            "one segmentation file per class."
        )


@dataclass
class WarsawSegmentationMissingError(DataclassException, FileNotFoundError):
    """
    Raised when a class the scene mesh declares has no segmentation file.
    """

    class_name: str
    """
    The class declared by the scene mesh.
    """

    segmentation_file: str
    """
    The segmentation file that class was expected to have.
    """

    def error_message(self) -> str:
        return (
            f"The scene declares the class '{self.class_name}' but holds no "
            f"'{self.segmentation_file}' to segment it by."
        )

    def suggest_correction(self) -> str:
        return "Export the scene again, so that every declared class is segmented."
