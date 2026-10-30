"""
HM3D scenes small enough to check by hand, written the way the dataset writes them.

A scene is two textured GLBs over the same triangles -- one painting each object a flat
colour, one painting what the room looks like -- and a table saying what those colours
mean. Everything here writes real files, because reading them is what is being tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from typing_extensions import Dict, List, Sequence, Tuple

# %% the triangles


def squares(how_many: int) -> Dict[str, float]:
    """
    Name a row of unit squares, written as separate geometries and each with its own
    vertices, so the edges they meet along only exist once their vertices are welded.

    :param how_many: How many squares to write.
    :return: Per geometry name, how far along x that square starts.
    """
    return {f"chunk{index:03d}_square": float(index) for index in range(how_many)}


def square(offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param offset: How far along x the square starts.
    :return: A unit square in the z=0 plane, as two triangles.
    """
    vertices = np.array(
        [
            [offset, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 1.0, 1.0, 0.0],
            [offset, 1.0, 0.0],
        ]
    )
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


# %% the textures


@dataclass(frozen=True)
class Palette:
    """
    A texture of flat patches, one per colour, and where to look each colour up.
    """

    colors: Sequence[Tuple[int, int, int]]
    """
    The patches, left to right along the top row of a one-row texture.
    """

    @property
    def image(self) -> Image.Image:
        """
        :return: The texture itself.
        """
        return Image.fromarray(np.array([list(self.colors)], dtype=np.uint8))

    def place(self, color: Tuple[int, int, int]) -> Tuple[float, float]:
        """
        :param color: The patch to look up.
        :return: Texture coordinates landing in the middle of it.
        """
        return ((self.colors.index(color) + 0.5) / len(self.colors), 0.5)


def painted(
    vertices: np.ndarray,
    faces: np.ndarray,
    palette: Palette,
    color: Tuple[int, int, int],
    as_triangle_soup: bool = False,
) -> trimesh.Trimesh:
    """
    :param vertices: The geometry's vertices.
    :param faces: Its triangles.
    :param palette: The texture to paint it from.
    :param color: The patch every one of its vertices is placed on.
    :param as_triangle_soup: Whether to give every corner a vertex of its own, which is
        how the semantic mesh is written and why it holds three times the vertices the
        textured one does over the very same triangles.
    :return: The geometry, painted that colour through the texture.
    """
    if as_triangle_soup:
        vertices = vertices[faces].reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=np.tile(palette.place(color), (len(vertices), 1)), image=palette.image
        ),
    )


# %% a whole scene on disk


@dataclass
class WrittenScene:
    """
    A scene written to disk, and what was written, so a test can check against it.
    """

    annotated_directory: Path
    """
    Where the semantic mesh and the annotation table were written.
    """

    textured_directory: Path
    """
    Where the mesh painted in the room's own colours was written.
    """

    instance_colors: List[Tuple[int, int, int]]
    """
    Per geometry, the colour the semantic mesh paints it.
    """

    room_colors: List[Tuple[int, int, int]]
    """
    Per geometry, the colour the textured mesh paints it.
    """


def write_scene(
    directory: Path,
    scene_name: str,
    instance_colors: Sequence[Tuple[int, int, int]],
    room_colors: Sequence[Tuple[int, int, int]],
    rows: Sequence[str],
    with_textured_mesh: bool = True,
) -> WrittenScene:
    """
    Write one scene the way an HM3D release holds it.

    :param directory: The release's root.
    :param scene_name: What the scene's directory is called in both trees.
    :param instance_colors: Per geometry, the colour the semantic mesh paints it.
    :param room_colors: Per geometry, the colour the textured mesh paints it.
    :param rows: The annotation table's rows, without its header.
    :param with_textured_mesh: Whether to write the mesh painted in the room's colours.
    :return: Where it was written, and what was written.
    """
    from experiments.warsaw.habitat.annotations import HEADER
    from experiments.warsaw.habitat.scene import ReleaseTree

    identifier = scene_name.split("-", 1)[1]
    annotated = directory / ReleaseTree.ANNOTATED.value / scene_name
    textured_directory = directory / ReleaseTree.TEXTURED.value / scene_name
    annotated.mkdir(parents=True, exist_ok=True)
    textured_directory.mkdir(parents=True, exist_ok=True)

    instances = Palette(colors=tuple(dict.fromkeys(instance_colors)))
    room = Palette(colors=tuple(dict.fromkeys(room_colors)))
    semantic: Dict[str, trimesh.Trimesh] = {}
    visual: Dict[str, trimesh.Trimesh] = {}
    for index, (name, offset) in enumerate(squares(len(instance_colors)).items()):
        vertices, faces = square(offset)
        semantic[name] = painted(
            vertices, faces, instances, instance_colors[index], as_triangle_soup=True
        )
        visual[name] = painted(vertices, faces, room, room_colors[index])

    trimesh.Scene(semantic).export(annotated / f"{identifier}.semantic.glb")
    (annotated / f"{identifier}.semantic.txt").write_text(
        "\n".join([HEADER, *rows]) + "\n"
    )
    if with_textured_mesh:
        trimesh.Scene(visual).export(textured_directory / f"{identifier}.glb")

    return WrittenScene(
        annotated_directory=annotated,
        textured_directory=textured_directory,
        instance_colors=list(instance_colors),
        room_colors=list(room_colors),
    )
