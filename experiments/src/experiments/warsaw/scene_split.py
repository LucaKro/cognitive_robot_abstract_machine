"""
Cut a scene's one body into one body per labelled object.

A scanned scene arrives as a single mesh whose faces carry labels, and the same face can
carry several: a drawer front is the drawer and the cabinet holding it. A body cannot be
built from that, since a face belongs to exactly one body's geometry, so the faces
several labels claim have to be given to one of them first. Who they belong to is decided
elsewhere -- it is not something the geometry says -- and this carries the decisions out.

The decision is made once per set of claimants rather than once per pair of them, so
applying it is a subtraction and never a contradiction: the owner keeps the set's faces
and every other claimant loses exactly those.

What comes out is flat: every object is a sibling under the root, wearing the name its
label segment had. The hierarchy is built afterwards by mounting the parts into their
wholes, which needs the objects to exist first and moves them without moving them in the
world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.semantic_annotations.part_whole import admissible_relations
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

ROOT_BODY_NAME = "root_body"
"""
What the body every object hangs from is called.
"""


@dataclass
class Ownership:
    """
    Who a set of contested faces belongs to.
    """

    names: Tuple[str, ...]
    """
    The segments claiming them.
    """

    owner: str
    """
    The one they belong to.
    """

    faces: np.ndarray
    """
    The faces in question.
    """

    settled_by_ontology: bool = False
    """
    Whether the ontology decided it rather than a model being asked.
    """


@dataclass
class Pairing:
    """
    One mount to carry out once the bodies exist.
    """

    whole: str
    """
    The object that holds.
    """

    part: str
    """
    The object it holds.
    """

    field_name: str
    """
    The field it is held in.
    """

    kind: str = "part"
    """
    Which channel mounts it: ``part``, ``contains`` or ``supports``.
    """

    def to_json(self) -> Dict[str, object]:
        """
        :return: The pairing as JSON-ready data.
        """
        return {
            "whole": self.whole,
            "part": self.part,
            "field": self.field_name,
            "kind": self.kind,
        }


@dataclass
class SplitFaces:
    """
    Which faces every object is left with, and what it cost.
    """

    faces: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    Per object, the faces that are its alone.
    """

    emptied: List[str] = field(default_factory=list)
    """
    The segments left with no faces at all, which no body can be built from.
    """

    lost_to: Dict[str, Dict[str, int]] = field(default_factory=dict)
    """
    Per segment, how many faces each owner took from it.

    An object that ends up with nothing has to be answerable for: without this, a wrong
    answer about a class of objects reads as ten objects that quietly failed to exist.
    """

    contested: Dict[int, List[str]] = field(default_factory=dict)
    """
    Per face still claimed by more than one object, who claims it.

    Applying one decision per set of claimants cannot leave any, so anything here means a
    set was reached by no decision rather than that two decisions disagreed.
    """


def owner_by_ontology(
    names: Sequence[str], classes: Dict[str, Optional[type]]
) -> Optional[str]:
    """
    Say which claimant a set of faces belongs to, where the taxonomy already decides it.

    A part keeps the surface it is made of: a drawer front is the drawer's, and the
    cabinet is what is left over. So where every claimant but one can hold the others as
    parts, the one that cannot is the one the faces are.

    :param names: The segments claiming the faces.
    :param classes: Per segment name, the class it was read as.
    :return: The claimant that no other can hold as a part, or None where that is not
        exactly one of them.
    """
    parts = []
    for name in names:
        mine = classes.get(name)
        if mine is None:
            return None
        others = [classes.get(other) for other in names if other != name]
        if any(other is None for other in others):
            return None
        if not any(
            relation.whole is mine
            for other in others
            for relation in admissible_relations(mine, other)
        ):
            parts.append(name)
    return parts[0] if len(parts) == 1 else None


def exclusive_faces(
    segments: Dict[str, np.ndarray], ownerships: Sequence[Ownership]
) -> SplitFaces:
    """
    Give every contested face to the object it was decided to belong to.

    :param segments: Per segment, the faces it claims.
    :param ownerships: Who each set of contested faces belongs to.
    :return: What every object is left with, and what it cost.
    """
    split = SplitFaces(
        faces={name: np.asarray(faces) for name, faces in segments.items()}
    )

    for ownership in ownerships:
        for name in ownership.names:
            if name == ownership.owner or name not in split.faces:
                continue
            before = len(split.faces[name])
            split.faces[name] = np.setdiff1d(split.faces[name], ownership.faces)
            taken = before - len(split.faces[name])
            if taken:
                split.lost_to.setdefault(name, {})
                split.lost_to[name][ownership.owner] = (
                    split.lost_to[name].get(ownership.owner, 0) + taken
                )

    split.emptied = sorted(name for name, faces in split.faces.items() if not len(faces))
    for name in split.emptied:
        del split.faces[name]

    if split.faces:
        claimed = np.concatenate(list(split.faces.values())).astype(np.int64)
        still = np.flatnonzero(np.bincount(claimed) > 1)
        for name, kept in split.faces.items():
            for face in kept[np.isin(kept, still)]:
                split.contested.setdefault(int(face), []).append(name)
        split.contested = {
            face: sorted(names) for face, names in split.contested.items()
        }
    return split


def pairings(candidates: Sequence[Pairing], split: SplitFaces) -> List[Pairing]:
    """
    Report the mounts that still have both ends.

    The overlap that says a handle is on *this* drawer is gone the moment the faces stop
    being shared, so what it said has to be carried past the split rather than measured
    again from the bodies. A mount naming an object that lost every face has no end to
    hang from and is dropped here rather than raising at mount time.

    :param candidates: The mounts the decisions named.
    :param split: What the split left.
    :return: Those whose whole and part both became bodies.
    """
    return [
        pairing
        for pairing in candidates
        if pairing.whole in split.faces and pairing.part in split.faces
    ]


def split_world(
    mesh,
    faces: Dict[str, np.ndarray],
    source_to_world,
    directory: Optional[Path] = None,
    file_type: str = "obj",
) -> World:
    """
    Build a world of one body per object from a scene's mesh.

    Each body's geometry is written to its file already in world coordinates and already
    centred on itself, and its connection carries where that centre sits. Nothing is left
    for a later step to adjust, because the steps that would --
    :class:`TransformGeometry` and :class:`CenterLocalGeometryAndPreserveWorldPose` --
    move the vertices of the mesh *object* a shape has loaded, and a shape backed by a
    file loads it again from that file when a world is read back. The connection keeps
    the adjustment, the file never had it, and the geometry comes back moved by it twice
    over: a body centred at ten metres reads back at twenty.

    :param mesh: The scene's mesh, whose faces the objects index.
    :param faces: Per object, the faces that are its alone.
    :param source_to_world: The transform from the file's coordinates to the world's.
    :param directory: Where to write the bodies' meshes, defaulting to a place that is
        removed when the process ends -- which will not do for a world to be persisted.
    :param file_type: The format to write them in. Not PLY: the collision detector reads
        only .obj, .stl and .dae, and a body it cannot read stops the world being built.
    :return: The world, flat, one body per object under a single root.
    """
    if directory is not None:
        # Each mesh goes in a directory of its own made under this one, and making it
        # needs this one to be there.
        Path(directory).mkdir(parents=True, exist_ok=True)

    to_world = source_to_world.to_np()
    world = World()
    root = Body(name=PrefixedName(ROOT_BODY_NAME))
    with world.modify_world():
        world.add_body(root)
        for name, kept in faces.items():
            piece = mesh.submesh([kept], append=True)
            piece.apply_transform(to_world)
            low, high = piece.bounds
            centre = (low + high) / 2.0
            piece.apply_translation(-centre)

            shapes = ShapeCollection(
                [Mesh.from_trimesh(piece, directory=directory, file_type=file_type)]
            )
            body = Body(name=PrefixedName(name), collision=shapes, visual=shapes)
            world.add_body(body)
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=body,
                    name=PrefixedName(f"root_to_{name}"),
                    parent_T_connection_expression=(
                        HomogeneousTransformationMatrix.from_point_rotation_matrix(
                            Point3(*centre)
                        )
                    ),
                )
            )
    return world
