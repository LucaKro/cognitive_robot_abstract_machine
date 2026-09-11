"""
Put the reconstruction and the modelled world in one scene, in one frame.

Every number the evaluation reports is a count of something. Whether an object is
actually where the model says it is can only be seen, and seeing it means both worlds
open at once, in the same frame, with each body still saying which world it came from
and what it is.

The reconstruction is the one that moves: it sits in its own frame and, on the
kitchenlab scan, is not even metric. The transform comes from landmarks picked by hand
on the scan file, and a run's bodies are a roll away from that file, so it is re-
expressed before it is applied. The overlay is only ever as good as that fit -- read the
residuals before believing a mismatch you see here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray
from semantic_digital_twin.adapters.world_mesh_exporter import (
    GeometrySource,
    WorldMeshExporter,
    WorldMeshExtractor,
)
from typing_extensions import List, Optional

from experiments.warsaw.evaluation.alignment import run_bodies_to_ground_truth
from experiments.warsaw.evaluation.export_world_mesh import named_by_class
from experiments.warsaw.evaluation.ground_truth import (
    world_from_provider,
    world_from_urdf,
)
from experiments.warsaw.painting import Coloring, paint
from experiments.warsaw.pipeline.run import RunFile

# %% a fit applied in the wrong frame


class ReconstructionNotUprightError(ValueError):
    """
    The reconstruction stands far taller than the world it is overlaid on.

    Landmarks are picked on the scan file while a run's bodies sit in the world the
    loader rolls that scan upright into, and a fit applied without re-expressing it
    therefore arrives a quarter turn out. That looks like a badly picked landmark set,
    so it is named here instead of being written out to be puzzled over.
    """


TALLER_THAN_THE_WORLD_ALLOWED = 1.0
"""
How much taller than the modelled world the reconstruction may stand, in metres.

A scan reaching somewhat above or below what was modelled is ordinary: it sees ceilings,
skirting and clutter nobody modelled. A quarter turn is not ordinary, and it shows up
here because the roll swaps a horizontal extent -- several metres across any room --
onto the vertical.
"""

UPWARD = 2
"""
The axis both worlds measure height along once the reconstruction has been rolled.
"""

# %% what each world is called in the scene

RECONSTRUCTION = "run"
"""
What the bodies a run built are grouped under, so a click says which world it is.
"""

MODELLED = "modelled"
"""
What the bodies of the world being compared against are grouped under.
"""

MODELLED_COLOUR = (150, 150, 150, 110)
"""
One flat translucent grey for the whole modelled world.

Overlaid geometry is read by eye before it is read by name, and the run is already
coloured per class; giving the modelled world its own colouring would leave the two
indistinguishable where they sit on top of one another.
"""

# %% putting the two together


def overlaid(
    reconstruction: trimesh.Scene,
    modelled: trimesh.Scene,
    reconstruction_to_modelled: NDArray[np.float64],
) -> trimesh.Scene:
    """
    Put both worlds in one scene, in the modelled world's frame.

    :param reconstruction: What the run built, in its own frame.
    :param modelled: The world it is compared against, whose frame is the one used.
    :param reconstruction_to_modelled: The fitted transform, scale included.
    :return: One scene holding both, each body named for its world.
    """
    together = trimesh.Scene()
    stands = _gather(
        together, reconstruction, RECONSTRUCTION, reconstruction_to_modelled
    )
    room = _gather(together, modelled, MODELLED, np.eye(4), colour=MODELLED_COLOUR)
    _refuse_a_frame_mismatch(stands, room)
    return together


def _refuse_a_frame_mismatch(
    reconstruction: NDArray[np.float64], modelled: NDArray[np.float64]
) -> None:
    """
    Refuse a reconstruction that cannot be standing in the world it is overlaid on.

    :param reconstruction: What the moved reconstruction spans, as low and high corners.
    :param modelled: What the modelled world spans.
    :raises ReconstructionNotUprightError: If it stands too much taller to be upright.
    """
    stands = reconstruction[1][UPWARD] - reconstruction[0][UPWARD]
    room = modelled[1][UPWARD] - modelled[0][UPWARD]
    if stands <= room + TALLER_THAN_THE_WORLD_ALLOWED:
        return
    raise ReconstructionNotUprightError(
        f"The reconstruction stands {stands:.2f} m tall in a world {room:.2f} m tall, "
        f"so the two are not in the same frame. A fit picked on the scan file has to be "
        f"re-expressed for a run's bodies; see evaluation/landmarks/README.md."
    )


def _gather(
    together: trimesh.Scene,
    world: trimesh.Scene,
    called: str,
    transform: NDArray[np.float64],
    colour: Optional[tuple] = None,
) -> NDArray[np.float64]:
    """
    Copy one world's bodies into the shared scene, moved and named for that world.

    Walked by node rather than by geometry: a GLB carries the name worth reading on the
    node, while the geometry under it is called something like ``body_1``, and several
    nodes may share one geometry.

    :param together: The scene being built.
    :param world: The world to copy in.
    :param called: What that world is named in the result.
    :param transform: What to move it by.
    :param colour: One flat colour for all of it, or nothing to keep its own.
    :return: What it spans once moved, as low and high corners.
    """
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for node in world.graph.nodes_geometry:
        placed, geometry_name = world.graph[node]
        mesh = world.geometry[geometry_name].copy()
        mesh.apply_transform(transform @ placed)
        if colour is not None:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, face_colors=np.tile(colour, (len(mesh.faces), 1))
            )
        low, high = np.minimum(low, mesh.bounds[0]), np.maximum(high, mesh.bounds[1])
        together.add_geometry(
            mesh, node_name=f"{called}/{node}", geom_name=f"{called}/{node}"
        )
    return np.vstack([low, high])


def scene_of_a_run(directory: Path) -> trimesh.Scene:
    """
    Read back the scene a finished run left behind.

    Read from the run's own ``world_mesh`` rather than from the database, so this works
    on a run whose generated classes the ontology has since taken over.

    :param directory: The run's directory.
    :return: Its bodies, each named for what the run decided it is.
    """
    return trimesh.load(
        Path(directory) / RunFile.WORLD_MESH.value / WorldMeshExporter.scene_file_name
    )


def scene_of_a_modelled_world(
    urdf: Optional[Path] = None, world_provider: Optional[str] = None
) -> trimesh.Scene:
    """
    Build the scene of a modelled world, from a file or from what builds it.

    :param urdf: The URDF to read, if the world is a file.
    :param world_provider: What builds it, as ``module.path:ClassName``.
    :return: Its bodies, each named for what it is.
    """
    world = (
        world_from_urdf(urdf)
        if urdf is not None
        else world_from_provider(world_provider)
    )
    paint(world, Coloring.BY_CLASS)
    snapshot = named_by_class(
        WorldMeshExtractor(
            geometry_source=GeometrySource.VISUAL_WITH_COLLISION_FALLBACK
        ).extract(world)
    )
    return WorldMeshExporter().to_scene(snapshot)


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for overlaying two worlds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A finished run's directory")
    parser.add_argument(
        "--alignment", type=Path, required=True, help="The fitted transform to apply"
    )
    modelled = parser.add_mutually_exclusive_group(required=True)
    modelled.add_argument("--urdf", type=Path, help="The modelled world as a file")
    modelled.add_argument(
        "--world-provider", help="What builds it, as module.path:ClassName"
    )
    parser.add_argument(
        "--output", type=Path, help="Where to write, by default into the run"
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Write one scene holding a run and the world it is compared against.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the scene is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)
    fitted = json.loads(parsed.alignment.read_text())

    together = overlaid(
        reconstruction=scene_of_a_run(parsed.run),
        modelled=scene_of_a_modelled_world(parsed.urdf, parsed.world_provider),
        reconstruction_to_modelled=run_bodies_to_ground_truth(fitted["matrix"]),
    )
    output = parsed.output or Path(parsed.run) / RunFile.WORLD_MESH.value
    output.mkdir(parents=True, exist_ok=True)
    written = output / "overlay.glb"
    together.export(written)
    logging.getLogger(__name__).info(
        "wrote %s bodies to %s -- the fit it uses has an error of %.3f m",
        len(together.geometry),
        written,
        fitted["root_mean_square_error"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
