from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from semantic_digital_twin.adapters.scannetpp_dataset.exceptions import (
    ScanNetPPSceneNotFoundError,
)
from semantic_digital_twin.adapters.scannetpp_dataset.schema import (
    ScanNetPPObject,
    ScanNetPPScene,
    ScanNetPPSceneFile,
)

# %% loader


@dataclass
class ScanNetPPDatasetLoader:
    """
    Loader for scenes of the ScanNet++ dataset (https://kaldir.vc.in.tum.de/scannetpp/),
    laser scans of real indoor rooms with annotated objects.

    .. note::
        ScanNet++ is only available after accepting its terms of use, so this loader
        reads scenes already downloaded with ScanNet++'s own download script.
    """

    directory: Path
    """
    ScanNet++'s ``data`` directory, holding one folder per scene.
    """

    def load(self, scene_id: str) -> ScanNetPPScene:
        """
        :param scene_id: The scene's identifier, e.g. ``"f9f95681fd"``.
        :return: The scene's mesh and annotated objects.
        :raises ScanNetPPSceneNotFoundError: if :attr:`directory` holds no folder for
            the scene.
        """
        scene_directory = self.directory / scene_id
        if not scene_directory.is_dir():
            raise ScanNetPPSceneNotFoundError(
                scene_id=scene_id, directory=self.directory
            )

        # The segments, and Articulate3D's parts, refer to vertices by their position in
        # the file. Processing merges duplicate vertices, which renumbers the rest and
        # attaches labels to the wrong geometry.
        mesh_path = scene_directory / ScanNetPPSceneFile.MESH
        mesh = trimesh.load_mesh(mesh_path, process=False)
        segments = json.loads(
            (scene_directory / ScanNetPPSceneFile.SEGMENTS).read_text()
        )
        annotations = json.loads(
            (scene_directory / ScanNetPPSceneFile.SEGMENT_ANNOTATIONS).read_text()
        )
        return ScanNetPPScene(
            scene_id=scene_id,
            mesh=mesh,
            mesh_path=mesh_path,
            vertex_segment_ids=np.asarray(segments["segIndices"]),
            objects=[
                ScanNetPPObject.from_json(segment_group)
                for segment_group in annotations["segGroups"]
            ],
        )
