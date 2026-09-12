"""
An HM3D scene: its triangles, and which annotated object each of them belongs to.

A scene is held as two GLB files over the same triangles -- one painted so that every
annotated object is a flat colour, one painted in the colours the building was
photographed in -- beside the table saying what those colours mean. Reading it means
recovering the object each face belongs to from the first, and the room's own look from
the second.

Two things about these files decide how that has to be done, and both were measured
rather than assumed. The colours are reached through a texture, which has to be read
once per face rather than blended per vertex or a fifth of the faces end up a colour no
annotation claims. And the triangles are written with a vertex per corner, so nothing
touches anything until the vertices are welded -- and touching is the only evidence
these files give that two objects meet. ``HABITAT_HM3D.md`` has the measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import trimesh
from typing_extensions import Dict, List, Tuple

from experiments.warsaw.exceptions import (
    HabitatMeshesDisagreeError,
    HabitatSceneNotFoundError,
)
from experiments.warsaw.habitat.annotations import AnnotatedObject, SceneAnnotations

NOTHING = 0
"""
The object a face belongs to when no annotation claims it.

Most of these carry the colour black, which is HM3D's own mark for a face it never
annotated; on scene 00800 they are 22,556 faces of 395,018.
"""


class ReleaseTree(StrEnum):
    """
    The trees an HM3D release keeps its scenes in, each holding one directory per scene.
    """

    ANNOTATED = "hm3d-minival-semantic-annots-v0.2"
    """
    The mesh painted one flat colour per object, and the table saying what they mean.
    """

    TEXTURED = "hm3d-minival-glb-v0.2"
    """
    The same triangles, painted in the colours the building was photographed in.
    """


class SceneSuffix(StrEnum):
    """
    How a scene's three files are named, after the identifier they all start with.
    """

    ANNOTATIONS = ".semantic.txt"
    """
    The table naming every annotated object.
    """

    SEMANTIC_MESH = ".semantic.glb"
    """
    The mesh painted one flat colour per object.
    """

    TEXTURED_MESH = ".glb"
    """
    The mesh painted in the colours the building was photographed in.
    """


# %% reading a texture


def sampled(texture: np.ndarray, places: np.ndarray) -> np.ndarray:
    """
    Read a texture at each of a number of places, taking the nearest texel and blending
    nothing.

    The colour of a face is what says which object it belongs to, so a colour read
    between two of the texture's patches names no object at all. Reading each place on
    its own, without interpolation, is what keeps the colours the ones the table lists.

    :param texture: The texture, as rows of pixels from the top down.
    :param places: Texture coordinates, one row per place.
    :return: The colour at each place, as red, green and blue.
    """
    height, width = texture.shape[:2]
    across = np.clip((places[:, 0] * width).astype(int), 0, width - 1)
    down = np.clip(((1.0 - places[:, 1]) * height).astype(int), 0, height - 1)
    return texture[down, across, :3]


def texture_of(geometry: trimesh.Trimesh) -> np.ndarray:
    """
    :param geometry: A geometry of a GLB.
    :return: The texture its material paints it with.
    """
    return np.asarray(geometry.visual.material.baseColorTexture)


def face_colors_of(geometry: trimesh.Trimesh) -> np.ndarray:
    """
    :param geometry: A geometry of a GLB.
    :return: The colour its texture paints each of its faces, read at the face's own
        middle so that it falls inside one patch.
    """
    return sampled(
        texture_of(geometry), geometry.visual.uv[geometry.faces].mean(axis=1)
    )


def vertex_colors_of(geometry: trimesh.Trimesh) -> np.ndarray:
    """
    :param geometry: A geometry of a GLB.
    :return: The colour its texture paints each of its vertices.
    """
    return sampled(texture_of(geometry), geometry.visual.uv)


def placed_by_chunk(
    scene: trimesh.Scene, files: HabitatSceneFiles
) -> Dict[str, trimesh.Trimesh]:
    """
    Put every geometry of a GLB where the file's graph places it, keyed by its chunk.

    :param scene: The GLB, as it was loaded.
    :param files: Where it was read from, for the error message.
    :return: Per chunk, the geometry standing where the file puts it.
    :raises HabitatMeshesDisagreeError: If two geometries are the same chunk, which
        would leave undecided which of them a colour belongs to.
    """
    placements: Dict[str, np.ndarray] = {}
    for node in scene.graph.nodes_geometry:
        where, geometry_name = scene.graph[node]
        placements[geometry_name] = where

    placed: Dict[str, trimesh.Trimesh] = {}
    for geometry_name, geometry in scene.geometry.items():
        chunk = chunk_of(geometry_name)
        if chunk in placed:
            raise HabitatMeshesDisagreeError(
                semantic_mesh=files.semantic_mesh,
                textured_mesh=files.textured_mesh,
                complaint=f"{chunk} names more than one geometry",
            )
        where = placements.get(geometry_name)
        if where is None or np.allclose(where, np.eye(4)):
            placed[chunk] = geometry
            continue
        standing = geometry.copy()
        standing.apply_transform(where)
        placed[chunk] = standing
    return placed


def chunk_of(geometry_name: str) -> str:
    """
    Name the piece of the building a geometry is, which is what the two meshes agree on.

    The rest of the name describes what was reconstructed there and the two files spell
    it differently -- one writes ``..._filled001_t`` where the other writes
    ``..._filled001_type002_1`` -- but the chunk is the same piece in both, and it is
    unique within a file.

    :param geometry_name: The name a GLB gives one of its geometries.
    :return: The chunk it belongs to.
    """
    return geometry_name.split("_", 1)[0]


# %% where a release keeps its scenes


@dataclass
class HabitatSceneFiles:
    """
    The three files one HM3D scene is written as.
    """

    annotations: Path
    """
    The table naming every annotated object.
    """

    semantic_mesh: Path
    """
    The mesh painted one flat colour per object.
    """

    textured_mesh: Path
    """
    The same triangles, painted in the colours the building was photographed in.
    """

    @property
    def missing(self) -> List[Path]:
        """
        :return: The ones that are not there.
        """
        return [
            path
            for path in (self.annotations, self.semantic_mesh, self.textured_mesh)
            if not path.is_file()
        ]


@dataclass
class HabitatDataset:
    """
    An HM3D release on disk, and the scenes it holds.
    """

    root: Path
    """
    The directory holding the release's trees.
    """

    @property
    def scene_names(self) -> List[str]:
        """
        :return: Every scene the release annotates, in order.
        """
        annotated = self.root / ReleaseTree.ANNOTATED.value
        if not annotated.is_dir():
            return []
        return sorted(one.name for one in annotated.iterdir() if one.is_dir())

    def file(self, scene_name: str, tree: ReleaseTree, suffix: SceneSuffix) -> Path:
        """
        :param scene_name: The scene's directory, as the release names it.
        :param tree: Which of the release's trees the file is under.
        :param suffix: Which of the scene's files it is.
        :return: Where the release keeps it.
        """
        identifier = scene_name.split("-", 1)[-1]
        return self.root / tree.value / scene_name / f"{identifier}{suffix.value}"

    def scene(self, scene_name: str) -> HabitatSceneFiles:
        """
        :param scene_name: The scene's directory, as the release names it.
        :return: The three files it is written as.
        :raises HabitatSceneNotFoundError: If any of them is missing.
        """
        files = HabitatSceneFiles(
            annotations=self.file(
                scene_name, ReleaseTree.ANNOTATED, SceneSuffix.ANNOTATIONS
            ),
            semantic_mesh=self.file(
                scene_name, ReleaseTree.ANNOTATED, SceneSuffix.SEMANTIC_MESH
            ),
            textured_mesh=self.file(
                scene_name, ReleaseTree.TEXTURED, SceneSuffix.TEXTURED_MESH
            ),
        )
        if files.missing:
            raise HabitatSceneNotFoundError(
                root=self.root,
                scene_name=scene_name,
                missing=files.missing,
                holds=self.scene_names,
            )
        return files


# %% the scene itself


@dataclass
class HabitatScene:
    """
    One HM3D scene, read: its triangles, its colours, and what each face belongs to.
    """

    files: HabitatSceneFiles
    """
    Where it was read from.
    """

    annotations: SceneAnnotations
    """
    Every object it annotates.
    """

    mesh: trimesh.Trimesh
    """
    The whole building as one mesh, its vertices welded and its colours the room's own.
    """

    face_objects: np.ndarray
    """
    Per face, the object it belongs to, :data:`NOTHING` where no annotation claims it.
    """

    @classmethod
    def read(cls, files: HabitatSceneFiles) -> HabitatScene:
        """
        Read a scene from the files it is written as.

        :param files: The scene's three files.
        :return: The scene.
        :raises HabitatMeshesDisagreeError: If the two meshes are not the same
            triangles.
        """
        annotations = SceneAnnotations.from_file(files.annotations)
        semantic = trimesh.load(str(files.semantic_mesh), process=False)
        textured = trimesh.load(str(files.textured_mesh), process=False)
        paired = cls._paired(
            placed_by_chunk(semantic, files), placed_by_chunk(textured, files), files
        )

        # The geometry is taken from the textured mesh and the labels from the semantic
        # one. They are the same triangles in the same order, but the semantic mesh
        # writes every corner a vertex of its own -- 8,742 where the textured one has
        # 2,851 over the same 2,914 faces -- so a colour per vertex only lines up with
        # the mesh it was read from.
        pieces, colors, instances = [], [], []
        for painted, seen in paired:
            pieces.append(
                trimesh.Trimesh(vertices=seen.vertices, faces=seen.faces, process=False)
            )
            colors.append(vertex_colors_of(seen))
            instances.append(face_colors_of(painted))

        mesh = trimesh.util.concatenate(pieces)
        mesh.visual.vertex_colors = np.vstack(colors)
        faces_before = len(mesh.faces)
        # Welding is what gives two objects an edge to meet along: these files write
        # every triangle with vertices of its own, so nothing touches anything until
        # the ones written twice are merged.
        mesh.merge_vertices()
        if len(mesh.faces) != faces_before:
            raise HabitatMeshesDisagreeError(
                semantic_mesh=files.semantic_mesh,
                textured_mesh=files.textured_mesh,
                complaint=(
                    f"welding the vertices left {len(mesh.faces)} faces of "
                    f"{faces_before}, so the colours read per face no longer name them"
                ),
            )

        return cls(
            files=files,
            annotations=annotations,
            mesh=mesh,
            face_objects=cls._objects_of(np.vstack(instances), annotations),
        )

    @staticmethod
    def _paired(
        semantic: Dict[str, trimesh.Trimesh],
        textured: Dict[str, trimesh.Trimesh],
        files: HabitatSceneFiles,
    ) -> List[Tuple[trimesh.Trimesh, trimesh.Trimesh]]:
        """
        Put each piece of the semantic mesh beside the same piece of the textured one.

        The two files describe the same reconstruction but spell the rest of a
        geometry's name differently -- one writes ``..._filled001_t`` where the other
        writes ``..._filled001_type002_1`` -- so the chunk is what they agree on.

        :param semantic: The mesh painted one flat colour per object, by chunk.
        :param textured: The mesh painted in the building's own colours, by chunk.
        :param files: Where they were read from, for the error message.
        :return: The pairs, in the semantic mesh's order.
        :raises HabitatMeshesDisagreeError: If a piece is in one and not the other, or
            if a pair is not the same triangles.
        """
        paired = []
        for chunk, painted in semantic.items():
            if chunk not in textured:
                raise HabitatMeshesDisagreeError(
                    semantic_mesh=files.semantic_mesh,
                    textured_mesh=files.textured_mesh,
                    complaint=(
                        f"{chunk} is in the semantic mesh and not the textured one"
                    ),
                )
            seen = textured[chunk]
            if painted.faces.shape != seen.faces.shape or not np.allclose(
                painted.triangles, seen.triangles, atol=1e-5
            ):
                raise HabitatMeshesDisagreeError(
                    semantic_mesh=files.semantic_mesh,
                    textured_mesh=files.textured_mesh,
                    complaint=f"{chunk} is different triangles in the two meshes",
                )
            paired.append((painted, seen))
        return paired

    @staticmethod
    def _objects_of(
        face_colors: np.ndarray, annotations: SceneAnnotations
    ) -> np.ndarray:
        """
        :param face_colors: The colour the semantic mesh paints each face.
        :param annotations: What the table says those colours mean.
        :return: Per face, the object it belongs to.
        """
        claimed = np.full(len(face_colors), NOTHING, dtype=np.int64)
        for color, annotated in annotations.by_color().items():
            claimed[np.all(face_colors == np.array(color), axis=1)] = (
                annotated.object_id
            )
        return claimed

    def objects_in_room(self, room_id: int) -> List[AnnotatedObject]:
        """
        :param room_id: The room to gather.
        :return: The objects assigned to it that the mesh actually has faces for.
        """
        present = set(np.unique(self.face_objects).tolist())
        return [
            one for one in self.annotations.in_room(room_id) if one.object_id in present
        ]

    def faces_in_room(self, room_id: int) -> np.ndarray:
        """
        :param room_id: The room to gather.
        :return: The faces its objects are made of, in order.
        """
        wanted = np.array(
            [one.object_id for one in self.objects_in_room(room_id)], dtype=np.int64
        )
        return np.flatnonzero(np.isin(self.face_objects, wanted))
