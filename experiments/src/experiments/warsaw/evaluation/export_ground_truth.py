"""
The command that writes one scene's ground truth for evaluation.

It is a module of its own because a module run with ``-m`` is imported as ``__main__``,
and a record class defined in it would be written into every file it produces under that
name. Nothing else could then read those files back, so the records live beside the
graph they describe and only the command lives here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from semantic_digital_twin.adapters.world_mesh_exporter import GeometrySource
from typing_extensions import List, Optional

from experiments.warsaw.evaluation.ground_truth import (
    GroundTruthCorrections,
    GroundTruthGraph,
    world_from_provider,
    world_from_urdf,
)
from experiments.warsaw.pipeline.provenance import inspect_source

# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for exporting ground truth.
    """
    parser = argparse.ArgumentParser(
        description="Write a modelled world's semantic graph for evaluation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("urdf", type=Path, nargs="?", help="The URDF file to read")
    source.add_argument(
        "--world-provider",
        help="What builds the world, as module.path:ClassName, for a scene modelled as "
        "Python rather than as a file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the ground-truth graph",
    )
    parser.add_argument(
        "--geometry-source",
        type=GeometrySource,
        choices=list(GeometrySource),
        default=GeometrySource.VISUAL_WITH_COLLISION_FALLBACK,
        help="Which of an entity's shape collections to measure",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        help="Ground truth a person supplied for this scene, applied to the result",
    )
    parser.add_argument(
        "--no-infer-semantics",
        dest="infer_semantics",
        action="store_false",
        help="Read the world's own classes without inferring any",
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Write the ground-truth graph of a modelled world.

    :param arguments: Command-line arguments without the program name.
    :return: Zero after the output is written.
    """
    parsed = argument_parser().parse_args(arguments)
    if parsed.urdf is not None:
        world = world_from_urdf(parsed.urdf, infer_semantics=parsed.infer_semantics)
        scene = str(parsed.urdf.expanduser().resolve())
    else:
        # A world built by a class names its own classes as it builds, so there is
        # nothing to infer and the reference is what identifies the scene.
        world = world_from_provider(parsed.world_provider)
        scene = parsed.world_provider
    graph = GroundTruthGraph.from_world(
        world, scene=scene, geometry_source=parsed.geometry_source
    )
    if parsed.corrections is not None:
        graph = GroundTruthCorrections.read(parsed.corrections).applied_to(graph)
    source, _ = inspect_source(Path(__file__).resolve().parents[5])
    graph = replace(graph, source_commit=source.commit, source_dirty=source.dirty)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(graph.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
