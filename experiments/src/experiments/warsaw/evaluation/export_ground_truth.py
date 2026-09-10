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
    parser.add_argument("urdf", type=Path, help="The URDF file to read")
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
    world = world_from_urdf(parsed.urdf, infer_semantics=parsed.infer_semantics)
    graph = GroundTruthGraph.from_world(
        world,
        scene=str(parsed.urdf.expanduser().resolve()),
        geometry_source=parsed.geometry_source,
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
