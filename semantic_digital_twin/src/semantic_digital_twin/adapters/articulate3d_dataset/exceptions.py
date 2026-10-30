from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% scene lookup


@dataclass
class Articulate3DSceneNotFoundError(DataclassException, LookupError):
    """
    Raised when the Articulate3D dataset directory holds no annotations for a requested
    scene.
    """

    scene_id: str
    """
    The scene that was requested.
    """

    directory: Path
    """
    The Articulate3D dataset directory that was searched.
    """

    def error_message(self) -> str:
        return f"No Articulate3D annotations for scene '{self.scene_id}' in '{self.directory}'."

    def suggest_correction(self) -> str:
        return (
            "Point the loader at the directory holding Articulate3D's "
            "'<scene_id>_parts.json' and '<scene_id>_artic.json' files, and request a "
            "scene listed in its 'train.txt' or 'val.txt'."
        )
