"""
What a scene looks like, kept beside the mesh that says what it is.

The pipeline's mesh has to be welded and has to carry one integer property per class,
and neither can hold a texture: welding merges the vertices a UV seam splits, and a face
property has nowhere to put an image. Baking the texture onto vertices instead loses most
of it -- an HM3D room carries a colour every 6 cm where its source carries one every 3 mm
-- and the renderer then flattens even that to one colour per triangle.

So the appearance travels as a file of its own, over the same faces in the same order, and
only the pictures read it. Everything that measures, labels, splits or mounts goes on
reading the labelled mesh and is untouched.

A scene that ships no appearance mesh -- every Warsaw scan -- is drawn exactly as it
always was.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from typing_extensions import List, Optional, Sequence

APPEARANCE_FILE = "appearance.glb"
"""
What the appearance mesh is called, beside the scene's own mesh.
"""

PIECE_NAME = "piece_{:04d}"
"""
How a piece is named, so that walking the pieces in the order their names sort in walks
the faces in the order the labelled mesh numbers them.

The order is the file's own rather than a table beside it, since a table and a file are
two things to keep agreeing and the names are already written down.
"""

# %% taking faces out of a textured geometry


def slice_keeping_appearance(
    geometry: trimesh.Trimesh, faces: np.ndarray
) -> trimesh.Trimesh:
    """
    Take some faces of a textured geometry, keeping its texture and its own normals.

    ``submesh`` keeps the texture but drops the vertex normals, and trimesh then
    computes flat ones from the faces, which is what makes a coarse triangle read as a
    facet rather than as part of a smooth surface.

    :param geometry: The geometry to take from.
    :param faces: Which of its faces to keep.
    :return: Those faces, with the vertices they need, their UVs and their normals.
    """
    used = np.unique(geometry.faces[faces])
    renumbered = np.zeros(len(geometry.vertices), dtype=np.int64)
    renumbered[used] = np.arange(len(used))
    return trimesh.Trimesh(
        vertices=geometry.vertices[used],
        faces=renumbered[geometry.faces[faces]],
        vertex_normals=geometry.vertex_normals[used],
        visual=trimesh.visual.TextureVisuals(
            uv=geometry.visual.uv[used], material=geometry.visual.material
        ),
        process=False,
    )


# %% the appearance of one scene


@dataclass
class AppearanceMesh:
    """
    The pieces a scene's pictures are drawn from, in the order its faces are numbered.

    Several pieces rather than one mesh, because a room's faces come from source
    geometries with textures of their own and merging them would mean repacking those
    into one atlas.
    """

    pieces: List[trimesh.Trimesh]
    """
    The source geometries, sliced to the scene's faces, in face order.
    """

    @property
    def first_face_of_piece(self) -> np.ndarray:
        """
        :return: Where each piece's faces begin in the scene's numbering, with the total
            at the end.
        """
        return np.cumsum([0] + [len(one.faces) for one in self.pieces])

    @property
    def face_count(self) -> int:
        """
        :return: How many faces it draws, which is how many the labelled mesh has.
        """
        return int(self.first_face_of_piece[-1])

    @classmethod
    def write(cls, pieces: Sequence[trimesh.Trimesh], directory: Path) -> Path:
        """
        Write pieces beside a scene, named so their order survives the file.

        :param pieces: The sliced source geometries, in face order.
        :param directory: The scene directory to write into.
        :return: The file written.
        """
        scene = trimesh.Scene()
        for index, piece in enumerate(pieces):
            scene.add_geometry(piece, geom_name=PIECE_NAME.format(index))
        path = Path(directory) / APPEARANCE_FILE
        scene.export(path)
        return path

    @classmethod
    def beside(cls, directory: Path) -> Optional[AppearanceMesh]:
        """
        Read the appearance mesh a scene directory holds, if it holds one.

        :param directory: The scene directory.
        :return: It, or None where the scene says nothing about how it looks.
        """
        path = Path(directory) / APPEARANCE_FILE
        if not path.is_file():
            return None
        loaded = trimesh.load(path, process=False)
        geometries = (
            loaded.geometry
            if hasattr(loaded, "geometry")
            else {PIECE_NAME.format(0): loaded}
        )
        return cls(pieces=[geometries[name] for name in sorted(geometries)])

    def faces_alone(self, faces: np.ndarray) -> trimesh.Scene:
        """
        Draw some of the scene's faces and nothing else.

        :param faces: Which faces, numbered as the labelled mesh numbers them.
        :return: A scene of the pieces those faces fall in, each sliced to them.
        """
        starts = self.first_face_of_piece
        wanted = np.asarray(faces, dtype=np.int64)
        drawn = trimesh.Scene()
        for index, piece in enumerate(self.pieces):
            here = wanted[(wanted >= starts[index]) & (wanted < starts[index + 1])]
            if not len(here):
                continue
            drawn.add_geometry(
                slice_keeping_appearance(piece, here - starts[index]),
                geom_name=PIECE_NAME.format(index),
            )
        return drawn
