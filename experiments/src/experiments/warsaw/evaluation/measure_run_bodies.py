"""
The command that measures the bodies a run built, so they can be paired with a world.

A run records which faces of the scene each of its bodies is made of, and the faces
themselves stay in the scene mesh, which is far too large to keep per run. Measuring
therefore happens once, afterwards, against the scene the run recorded having read, and
the sizes are written into a graph of its own rather than back over the run's: a run is
what it produced, and nothing later should edit it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
from typing_extensions import Dict, List, Optional

from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.evaluation.size import ObjectSize
from experiments.warsaw.pipeline.run import Run, RunFile

# %% measuring what a run built


def sizes_of_run_bodies(run: Run) -> Dict[str, ObjectSize]:
    """
    Measure every body a run built, from the scene it was cut out of.

    :param run: The finished run to measure.
    :return: The size of each body that has any faces, by the name the run gave it.
    """
    scene_path = Path(run.read_json(RunFile.PROVENANCE)["settings"]["scene_directory"])
    [scene_mesh] = sorted(scene_path.glob("*.ply"))
    scene = trimesh.load(scene_mesh, process=False)
    faces_of = np.load(run.path(RunFile.SPLIT_FACES))
    measured = {}
    for name in faces_of.files:
        body = scene.submesh([faces_of[name]], append=True, repair=False)
        size = ObjectSize.of(body)
        if size is not None:
            measured[name] = size
    return measured


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for measuring a run's bodies.
    """
    parser = argparse.ArgumentParser(
        description="Measure the bodies of a finished run for comparison."
    )
    parser.add_argument("run", type=Path, help="The run directory to measure")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the run's graph with its bodies measured",
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Write a run's graph with every body measured.

    :param arguments: Command-line arguments without the program name.
    :return: Zero after the output is written.
    """
    parsed = argument_parser().parse_args(arguments)
    run = Run(directory=parsed.run)
    graph = run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph)
    measured = sizes_of_run_bodies(run)
    graph = replace(
        graph,
        nodes=[replace(node, size=measured.get(node.name)) for node in graph.nodes],
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(graph.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
