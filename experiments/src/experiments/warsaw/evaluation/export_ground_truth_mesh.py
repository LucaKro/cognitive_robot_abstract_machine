"""
Write a modelled world as one mesh, so points can be picked on it by hand.

Relating a reconstruction to a modelled world starts with the same physical spot found
in each, clicked in a viewer. That needs both sides to be a file a viewer opens, and a
world built by a class is not one: it exists only once something has built it.

One mesh rather than one per body, because what is being done with it is clicking
corners of the room, and a viewer with two hundred objects in its tree makes that harder
rather than easier.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import trimesh
from semantic_digital_twin.adapters.world_mesh_exporter import (
    GeometrySource,
    WorldMeshExtractor,
    WorldMeshSnapshot,
)
from typing_extensions import List, Optional

from experiments.warsaw.evaluation.ground_truth import (
    world_from_provider,
    world_from_urdf,
)

# %% a world as one mesh


def one_mesh(snapshot: WorldMeshSnapshot) -> trimesh.Trimesh:
    """
    Put every body of a world into a single mesh, each where the world places it.

    :param snapshot: The world's bodies and their geometry.
    :return: All of it as one mesh, in the world's own frame.
    :raises ValueError: If no body of the world carries any geometry.
    """
    pieces = []
    for body in snapshot.body_meshes:
        if body.local_mesh is None or not len(body.local_mesh.faces):
            continue
        piece = trimesh.Trimesh(
            vertices=np.asarray(body.local_mesh.vertices),
            faces=np.asarray(body.local_mesh.faces),
            process=False,
        )
        piece.apply_transform(np.asarray(body.world_transform, dtype=np.float64))
        pieces.append(piece)
    if not pieces:
        raise ValueError("No body of this world carries geometry to write.")
    return trimesh.util.concatenate(pieces)


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for writing a modelled world as one mesh.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    world = parser.add_mutually_exclusive_group(required=True)
    world.add_argument("--urdf", type=Path, help="The modelled world as a file")
    world.add_argument(
        "--world-provider", help="What builds it, as module.path:ClassName"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the mesh"
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Write a modelled world as one mesh for picking points on.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the mesh is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)

    world = (
        world_from_urdf(parsed.urdf)
        if parsed.urdf is not None
        else world_from_provider(parsed.world_provider)
    )
    whole = one_mesh(
        WorldMeshExtractor(
            geometry_source=GeometrySource.VISUAL_WITH_COLLISION_FALLBACK
        ).extract(world)
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    whole.export(parsed.output)
    logging.getLogger(__name__).info(
        "wrote %s faces spanning %s to %s",
        len(whole.faces),
        np.round(whole.bounds, 2).tolist(),
        parsed.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
