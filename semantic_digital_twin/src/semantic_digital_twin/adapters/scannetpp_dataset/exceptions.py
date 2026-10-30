from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% scene lookup


@dataclass
class ScanNetPPSceneNotFoundError(DataclassException, LookupError):
    """
    Raised when the ScanNet++ data directory holds no folder for a requested scene.
    """

    scene_id: str
    """
    The scene that was requested.
    """

    directory: Path
    """
    The data directory that was searched.
    """

    def error_message(self) -> str:
        return f"No ScanNet++ scene '{self.scene_id}' in '{self.directory}'."

    def suggest_correction(self) -> str:
        return (
            "Point the loader at ScanNet++'s 'data' directory, which holds one folder "
            "per scene, and download the scene with ScanNet++'s download script."
        )
