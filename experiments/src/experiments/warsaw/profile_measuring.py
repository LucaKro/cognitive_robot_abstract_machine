"""
Time the phases of loading a scene and measuring how its labelled objects meet.

    python -m experiments.warsaw.profile_measuring --scene <directory>

Measuring a scene was most of what a run cost, and which part of it was expensive was not
apparent from the outside: loading a 594 MB scan turned out to be six seconds while
finding each segment's nearest took three quarters of an hour. A step's own duration,
which a run records, is too coarse to say that.

Writing a profile to a file is what makes two of them comparable, which is the only way to
say whether a change to any of these phases was worth making.

..note:: This reaches for the parts of :mod:`experiments.warsaw.segment_relations` that
    the module keeps to itself. Seeing inside a function is what a profile is for, and
    timing only what is public would report one number for the phase that matters.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing_extensions import Dict, Iterator, List, Optional

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.segment_relations import (
    SegmentDistances,
    _claim_slots,
    _edges_between_segments,
    _nearest_neighbours,
    _shared_faces,
)
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

# %% the phases a scene is measured in


class Phase(StrEnum):
    """
    One piece of work between a scene directory and a measured scene.

    Named for what is being worked out rather than for the call that works it out, so
    that a phase keeps its name when the code under it is rewritten.
    """

    LOAD_THE_SCENE = "load the scene"
    """
    Read the scan and build the world holding it.
    """

    LABEL_SEGMENTS = "find the labelled objects"
    """
    Read which faces each labelled object is made of.
    """

    FACE_GEOMETRY = "measure the faces"
    """
    Work out every face's centre and area.
    """

    CLAIM_SLOTS = "see which object claims each face"
    """
    Gather, per face, the objects whose labels cover it.
    """

    FACE_ADJACENCY = "find which faces meet"
    """
    Work out the mesh's own edges between neighbouring faces.
    """

    SHARED_FACES = "count the faces two objects share"
    """
    Count, per pair, the faces both of them claim.
    """

    EDGES_BETWEEN = "count the edges two objects meet along"
    """
    Count, per pair, the mesh edges where one's faces meet the other's.
    """

    SEARCH_TREES = "build the search trees"
    """
    Build, per object, a tree over its face centres.
    """

    COMPONENTS = "count each object's separate pieces"
    """
    Work out how many connected pieces each object falls into.
    """

    NEAREST = "find each object's nearest others"
    """
    Measure how far apart the objects are and keep the nearest few.
    """


# %% what a phase cost


@dataclass
class PhaseCost(JsonRecord):
    """
    How long one phase took.
    """

    phase: str
    """
    The phase, as :class:`Phase` names it.
    """

    seconds: float
    """
    How long it took.
    """


@dataclass
class MeasuringCosts(JsonRecord):
    """
    What each phase of loading and measuring one scene cost.
    """

    scene: str
    """
    The scene directory this was measured on.
    """

    nearest: int
    """
    How many nearest neighbours each object was asked for, which is what the last and
    largest phase is asked to do.
    """

    phases: List[PhaseCost] = field(default_factory=list)
    """
    Each phase, in the order it was carried out.
    """

    @property
    def total_seconds(self) -> float:
        """
        :return: How long the whole of it took.
        """
        return sum(one.seconds for one in self.phases)

    def ranked(self) -> List[PhaseCost]:
        """
        :return: The phases, costliest first, which is what the profile is read for.
        """
        return sorted(self.phases, key=lambda one: -one.seconds)

    def as_table(self) -> str:
        """
        :return: The phases as lines to read, costliest first.
        """
        lines = [f"{one.seconds / 60:8.2f} min  {one.phase}" for one in self.ranked()]
        lines.append(f"{self.total_seconds / 60:8.2f} min  in all")
        return "\n".join(lines)


# %% timing them


@dataclass
class PhaseClock:
    """
    Times each phase as it is carried out.
    """

    costs: List[PhaseCost] = field(default_factory=list)
    """
    What each phase that has finished took.
    """

    @contextmanager
    def timing(self, phase: Phase) -> Iterator[None]:
        """
        Time one phase.

        :param phase: The phase about to be carried out.
        """
        started = time.monotonic()
        yield
        self.costs.append(
            PhaseCost(phase=phase.value, seconds=time.monotonic() - started)
        )


def time_the_phases(scene: Path, nearest: int = 5) -> MeasuringCosts:
    """
    Load a scene and measure it, timing each phase separately.

    This repeats what :func:`experiments.warsaw.segment_relations.segment_evidence` does
    rather than calling it, because what is wanted is the parts and not the whole.

    :param scene: The directory holding the scene's labelled mesh.
    :param nearest: How many nearest neighbours each object reports.
    :return: What each phase cost.
    """
    clock = PhaseClock()

    with clock.timing(Phase.LOAD_THE_SCENE):
        loader = WarsawWorldLoader(input_directory=Path(scene))
    with clock.timing(Phase.LABEL_SEGMENTS):
        segments = loader.label_segments
    mesh = loader.scene_mesh
    with clock.timing(Phase.FACE_GEOMETRY):
        centres = mesh.triangles_center
        mesh.area_faces
    with clock.timing(Phase.CLAIM_SLOTS):
        slots, counts = _claim_slots(
            [segment.face_indices for segment in segments], len(mesh.faces)
        )
    with clock.timing(Phase.FACE_ADJACENCY):
        adjacency = mesh.face_adjacency
    labelled = counts > 0
    labelled_adjacency = adjacency[
        labelled[adjacency[:, 0]] & labelled[adjacency[:, 1]]
    ]
    with clock.timing(Phase.SHARED_FACES):
        _shared_faces(slots, counts)
    with clock.timing(Phase.EDGES_BETWEEN):
        _, internal_edges = _edges_between_segments(labelled_adjacency, slots)

    points: List[np.ndarray] = []
    trees: List[cKDTree] = []
    corners = np.zeros((len(segments), 2, 3))
    with clock.timing(Phase.SEARCH_TREES):
        for index, segment in enumerate(segments):
            own = centres[segment.face_indices]
            points.append(own)
            trees.append(cKDTree(own))
            corners[index] = [own.min(axis=0), own.max(axis=0)]
    with clock.timing(Phase.COMPONENTS):
        for index, segment in enumerate(segments):
            trimesh.graph.connected_components(
                labelled_adjacency[
                    internal_edges.get(index, np.empty(0, dtype=np.int64))
                ],
                nodes=segment.face_indices,
            )
    with clock.timing(Phase.NEAREST):
        _nearest_neighbours(
            SegmentDistances(trees=trees, points=points), corners, nearest
        )

    return MeasuringCosts(scene=str(scene), nearest=nearest, phases=clock.costs)


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for timing a scene's measurement.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, required=True, help="The scene directory to measure"
    )
    parser.add_argument(
        "--nearest",
        type=int,
        default=5,
        help="How many nearest neighbours each object reports",
    )
    parser.add_argument(
        "--output", type=Path, help="Where to write the profile, to compare against"
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Time a scene's measurement and report what each phase cost.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the profile has been taken.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)

    costs = time_the_phases(parsed.scene, nearest=parsed.nearest)
    logging.getLogger(__name__).info("%s", costs.as_table())
    if parsed.output is not None:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(json.dumps(costs.to_json(), indent=2))
        logging.getLogger(__name__).info("written to %s", parsed.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
