"""
Cut one overlap out of a scan and colour it, before the split and after it.

A figure of the split needs the same few objects twice: as the scan labels them, where a
drawer front is the drawer and the cabinet at once, and as the run left them, where
every face belongs to one body. This writes both as meshes a modelling tool can open, so
the camera and the lighting are the only work left.

Which faces belong to whom is read from the scan and from the run, never decided here.
`paper-figures/FIGURE_CASES.md` names the cases worth drawing.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import trimesh
from typing_extensions import Dict, List, Sequence, Tuple

from experiments.warsaw.painting import spread_colours
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

logger = logging.getLogger(__name__)

# %% the colours that are not a claimant's own

OPAQUE = 255
"""
The alpha every colour is written with.
"""

CONTESTED = (235, 45, 45, OPAQUE)
"""
What a face several labels claim is painted before the split, in red, since it is the
thing the picture is about.
"""

ELSEWHERE = (90, 90, 90, OPAQUE)
"""
What a face none of the drawn objects kept is painted after the split, in grey: it went
to an object the case does not draw.
"""


def as_colour(red: float, green: float, blue: float) -> Tuple[int, int, int, int]:
    """
    :param red: How much red, from zero to one.
    :param green: How much green.
    :param blue: How much blue.
    :return: The same colour as bytes, opaque.
    """
    return (int(red * 255), int(green * 255), int(blue * 255), OPAQUE)


# %% one overlap, before and after


@dataclass
class OverlapCase:
    """
    The faces of a few objects that claim one another's surface, and what became of
    them.
    """

    claims: Dict[str, np.ndarray]
    """
    Per segment, the faces the scan's labels claim for it, before anything is decided.
    """

    bodies: Dict[str, np.ndarray]
    """
    Per body, the faces the run left it with.

    A claimant the split emptied is absent.
    """

    @cached_property
    def faces(self) -> np.ndarray:
        """
        :return: Every face any claimant claims, in order and each once.
        """
        return np.unique(np.concatenate(list(self.claims.values())))

    @cached_property
    def contested(self) -> np.ndarray:
        """
        :return: The faces more than one claimant claims.
        """
        claimed = np.concatenate(list(self.claims.values()))
        face, times = np.unique(claimed, return_counts=True)
        return face[times > 1]

    @property
    def emptied(self) -> List[str]:
        """
        :return: The claimants the split left without a single face.
        """
        return [
            name
            for name in self.claims
            if name not in self.bodies or not len(self.bodies[name])
        ]

    @cached_property
    def palette(self) -> Dict[str, Tuple[int, int, int, int]]:
        """
        :return: One colour per claimant, told apart by name order so that the two
            pictures of a case agree about who is which colour.
        """
        names = sorted(self.claims)
        return {
            name: as_colour(colour.R, colour.G, colour.B)
            for name, colour in zip(names, spread_colours(len(names)))
        }

    def colour_of(self, name: str) -> Tuple[int, int, int, int]:
        """
        :param name: A claimant.
        :return: The colour it is drawn in.
        """
        return self.palette[name]

    def before(self) -> np.ndarray:
        """
        :return: A colour per face of :attr:`faces`, as the scan labels them: the
            claimant's own colour where one claims it, and :data:`CONTESTED` where
            several do.
        """
        colours = np.tile(np.array(CONTESTED, dtype=np.uint8), (len(self.faces), 1))
        contested = set(self.contested.tolist())
        for name, claimed in self.claims.items():
            for face in claimed:
                if int(face) in contested:
                    continue
                colours[self.row_of(face)] = self.colour_of(name)
        return colours

    def after(self) -> np.ndarray:
        """
        :return: A colour per face of :attr:`faces`, as the run left them: the colour of
            the body that kept the face, and :data:`ELSEWHERE` where none of the drawn
            bodies did.
        """
        colours = np.tile(np.array(ELSEWHERE, dtype=np.uint8), (len(self.faces), 1))
        for name, kept in self.bodies.items():
            for face in kept:
                colours[self.row_of(face)] = self.colour_of(name)
        return colours

    def row_of(self, face: int) -> int:
        """
        :param face: A face of the scene.
        :return: Which row of the case's colours it is.
        """
        return int(np.searchsorted(self.faces, face))

    @classmethod
    def read(cls, run: Run, names: Sequence[str]) -> OverlapCase:
        """
        Read one case out of a finished run and the scan it was made from.

        :param run: The run to read.
        :param names: The segments to draw, by the names the scan gave them.
        :return: What those segments claimed and what they were left with.
        """
        loader = cls.loader_of(run)
        claimed = {
            str(segment.name): segment.face_indices for segment in loader.label_segments
        }
        missing = [name for name in names if name not in claimed]
        if missing:
            raise UnknownSegmentError(missing)
        kept = np.load(run.path(RunFile.SPLIT_FACES))
        return cls(
            claims={name: claimed[name] for name in names},
            bodies={name: kept[name] for name in names if name in kept},
        )

    @staticmethod
    def loader_of(run: Run) -> WarsawWorldLoader:
        """
        :param run: A finished run.
        :return: A loader over the scene that run read.
        """
        scene = Path(run.read_json(RunFile.SPLIT)["scene"])
        return WarsawWorldLoader(input_directory=scene.parent)


class UnknownSegmentError(KeyError):
    """
    Segments asked for that the scan does not label.
    """


# %% cutting the case out of the scene


def case_mesh(
    scene: trimesh.Trimesh, faces: np.ndarray, colours: np.ndarray
) -> trimesh.Trimesh:
    """
    Cut the case's faces out of the scene and paint them.

    :param scene: The scan's mesh.
    :param faces: The faces to keep, in the order the colours are given in.
    :param colours: One colour per face.
    :return: A mesh of those faces alone, each in its colour.
    """
    cut = scene.submesh([faces], append=True)
    cut.visual.face_colors = colours
    return cut


@dataclass
class CaseExport:
    """
    Writing a case out as the two meshes a figure is drawn from.
    """

    run: Run
    """
    The finished run to read the split from.
    """

    names: List[str]
    """
    The segments to draw.
    """

    output: Path
    """
    The directory to write into.
    """

    @cached_property
    def case(self) -> OverlapCase:
        """
        :return: What the run and the scan say about those segments.
        """
        return OverlapCase.read(self.run, self.names)

    def write(self) -> List[Path]:
        """
        Write the case before and after the split.

        :return: The files written.
        """
        scene = OverlapCase.loader_of(self.run).scene_mesh
        self.output.mkdir(parents=True, exist_ok=True)
        written = []
        for stage, colours in (
            ("before", self.case.before()),
            ("after", self.case.after()),
        ):
            path = self.output / f"{stage}.ply"
            case_mesh(scene, self.case.faces, colours).export(path)
            written.append(path)
        self.report()
        return written

    def report(self) -> None:
        """
        Say what the case holds, so the figure's caption can be written from it.
        """
        for name in self.names:
            before = len(self.case.claims[name])
            after = len(self.case.bodies.get(name, []))
            logger.info("%s: %s faces before, %s after", name, before, after)
        logger.info("contested faces: %s", len(self.case.contested))
        logger.info("emptied: %s", ", ".join(self.case.emptied) or "none")


def main() -> None:
    """
    Write one overlap case of a run as two meshes.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("run", type=Path, help="The run directory to read")
    parser.add_argument(
        "--segments",
        nargs="+",
        required=True,
        help="The segments to draw, by the names the scan gave them",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the meshes"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = CaseExport(
        run=Run(arguments.run), names=list(arguments.segments), output=arguments.output
    ).write()
    for path in written:
        logger.info("written to %s", path)


if __name__ == "__main__":
    main()
