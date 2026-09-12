"""
Write one room of an HM3D scene as a scene the Warsaw pipeline already reads.

The pipeline reads a directory holding one mesh whose faces carry, per class, the
instance each face belongs to. Nothing in that is particular to the scans it was built
for, so HM3D is reached by writing a scene of that shape rather than by teaching the
pipeline a second way of reading one: every step downstream is left exactly as it is.

A room is the unit, not a whole scene. Every room of every annotated minival scene
carries its own floor, walls and ceiling, so a room cut out on its own is already a
closed room -- while a whole scene is seven to eleven times a kitchen and would put a
hundred and more labels into the one question that asks what they all mean.

python -m experiments.warsaw.habitat.convert --dataset <release> --scene
00800-TEEsavR23oF

names the rooms that scene has, and adding ``--room 12 --output <directory>`` writes
one.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing_extensions import Dict, List, Optional

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.habitat.annotations import UNASSIGNED_ROOM
from experiments.warsaw.habitat.scene import HabitatDataset, HabitatScene
from experiments.warsaw.world_loader.scene import SceneFrame, SourceFrame

SCENE_MESH = "mesh_all_classes.ply"
"""
What the mesh is called, as the scans beside it call theirs.
"""

CONVERTED_ROOM_FILE = "room.json"
"""
What a converted room says about where it came from.
"""

NOT_A_NAME = re.compile(r"[^0-9a-zA-Z_]+")
"""
Everything a mesh cannot carry in the name of a face property.
"""


def label_property(label: str) -> str:
    """
    Name a label the way a mesh can carry it.

    HM3D's labels are written as people say them, ``bath mat`` and ``cans of paint``,
    while a mesh names a face property the way an identifier is named. Across all 364
    labels of the annotated minival scenes this leaves 364 distinct names, so no two
    labels are turned into one.

    :param label: The label as the annotator wrote it.
    :return: The name a mesh can carry it under.
    """
    return NOT_A_NAME.sub("_", label.strip()).strip("_")


# %% what was written


@dataclass
class ConvertedObject(JsonRecord):
    """
    One object of a converted room, as the dataset has it and as the pipeline will.
    """

    object_id: int
    """
    Which object of the whole scene this is, as HM3D numbers them.
    """

    label: str
    """
    What the annotator called it, in their own words.
    """

    segment: str
    """
    What every step of a run calls it.
    """

    faces: int
    """
    How many of the room's faces it is made of.
    """


@dataclass
class ConvertedRoom(JsonRecord):
    """
    Where a converted room came from, which nothing in the mesh itself says.
    """

    scene: str = ""
    """
    The scene it was cut out of, as the release names it.
    """

    room_id: int = UNASSIGNED_ROOM
    """
    Which room of that scene it is.
    """

    objects: List[ConvertedObject] = field(default_factory=list)
    """
    Every object it holds.
    """

    absent: List[int] = field(default_factory=list)
    """
    The objects the table assigns to this room that the mesh carries no faces for.

    Kept because a room is quietly smaller than its table says -- 35 of scene 00800's
    660 objects are named and never drawn -- and a later comparison against the dataset
    would otherwise count them as objects the pipeline lost.
    """

    @classmethod
    def beside(cls, directory: Path) -> ConvertedRoom:
        """
        :param directory: The converted room's directory.
        :return: What it says about where it came from.
        """
        return cls.from_json(
            json.loads((Path(directory) / CONVERTED_ROOM_FILE).read_text())
        )

    def write_beside(self, directory: Path) -> Path:
        """
        :param directory: The converted room's directory.
        :return: The file written.
        """
        written = Path(directory) / CONVERTED_ROOM_FILE
        written.write_text(json.dumps(self.to_json(), indent=2))
        return written


# %% writing a room


def write_room(scene: HabitatScene, room_id: int, output: Path) -> ConvertedRoom:
    """
    Write one room of a scene as a directory the pipeline can be pointed at.

    Only the faces the room's objects are made of are kept. HM3D leaves about a
    twentieth of a building's faces annotated by nothing, and which room one of those
    belongs to is not something the dataset says, so keeping them would mean guessing.

    :param scene: The scene the room is part of.
    :param room_id: Which room to write.
    :param output: The directory to write it into.
    :return: What was written.
    """
    faces = scene.faces_in_room(room_id)
    objects = scene.objects_in_room(room_id)
    drawn = {one.object_id for one in objects}
    piece = scene.mesh.submesh([faces], append=True)

    of_object = {one.object_id: one for one in objects}
    here = scene.face_objects[faces]
    labels: Dict[str, np.ndarray] = {}
    for object_id in np.unique(here):
        annotated = of_object[int(object_id)]
        instances = labels.setdefault(
            label_property(annotated.label), np.zeros(len(here), dtype=np.uint32)
        )
        instances[here == object_id] = annotated.object_id

    for name, instances in labels.items():
        piece.face_attributes[name] = instances

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    piece.export(output / SCENE_MESH)
    SceneFrame(source=SourceFrame.UPRIGHT).write_beside(output)

    written = ConvertedRoom(
        scene=scene.files.semantic_mesh.parent.name,
        room_id=room_id,
        objects=[
            ConvertedObject(
                object_id=one.object_id,
                label=one.label,
                segment=f"{label_property(one.label)}_{one.object_id}",
                faces=int((here == one.object_id).sum()),
            )
            for one in objects
        ],
        absent=[
            one.object_id
            for one in scene.annotations.in_room(room_id)
            if one.object_id not in drawn
        ],
    )
    written.write_beside(output)
    return written


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for writing an HM3D room as a scene.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="The HM3D release's root directory"
    )
    parser.add_argument(
        "--scene", required=True, help="The scene, as the release names its directory"
    )
    parser.add_argument(
        "--room", type=int, help="Which room to write; left out, the rooms are named"
    )
    parser.add_argument("--output", type=Path, help="The directory to write it into")
    return parser


def name_the_rooms(scene: HabitatScene) -> None:
    """
    Say which rooms a scene has and how much is in each, so one can be chosen.

    :param scene: The scene to look through.
    """
    logger = logging.getLogger(__name__)
    logger.info("%s rooms", len(scene.annotations.room_ids))
    for room_id in scene.annotations.room_ids:
        logger.info(
            "  room %-3s %6s faces  %4s objects  %3s labels%s",
            room_id,
            len(scene.faces_in_room(room_id)),
            len(scene.objects_in_room(room_id)),
            len(scene.annotations.labels_in_room(room_id)),
            "  (the objects assigned to no room)" if room_id == UNASSIGNED_ROOM else "",
        )


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Write one room of an HM3D scene as a scene the pipeline reads.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the room is written, or once its rooms have been named.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)

    files = HabitatDataset(root=parsed.dataset).scene(parsed.scene)
    logger.info("reading %s ...", files.semantic_mesh.parent)
    scene = HabitatScene.read(files)
    logger.info(
        "%s faces, %s objects annotated",
        len(scene.mesh.faces),
        len(scene.annotations.objects),
    )

    if parsed.room is None or parsed.output is None:
        name_the_rooms(scene)
        logger.info("give --room and --output to write one of them")
        return 0

    written = write_room(scene=scene, room_id=parsed.room, output=parsed.output)
    logger.info(
        "wrote room %s of %s: %s objects over %s labels, to %s",
        written.room_id,
        written.scene,
        len(written.objects),
        len({one.segment.rsplit("_", 1)[0] for one in written.objects}),
        parsed.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
