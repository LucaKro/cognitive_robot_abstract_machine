"""
A converted Articulate3D scene, measured once as the pipeline measures it.

Measuring a scanned room is most of what working out a scene's candidate wholes costs,
and it needs the converted mesh, which is large. A measured scene keeps what was
measured and which part each segment was written from, so baselines and the rule's
ceiling can be worked out on every scene from a small file.

Run ``python -m experiments.warsaw.articulate3d.measured_scene --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from experiments.warsaw.articulate3d.convert import ConvertedScene, SceneConversion
from experiments.warsaw.articulate3d.ground_truth import (
    GroundTruthFile,
    SceneGroundTruth,
)
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol, ProtocolFile
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.segment_relations import (
    PairEvidence,
    SegmentDescriptor,
    SegmentRelations,
    segment_evidence,
)
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

# %% where a measured scene is written


class MeasuredSceneFile(StrEnum):
    """
    How a scene's measurement file is named.
    """

    SUFFIX = "_measured.json"
    """
    What follows the scene id.
    """

    def path(self, directory: Path, scene_id: str) -> Path:
        """
        :param directory: The directory holding measured scenes.
        :param scene_id: The scene.
        :return: Where the scene's measurement is written.
        """
        return Path(directory) / f"{scene_id}{self.value}"


# %% the measured scene


@dataclass
class MeasuredScene(JsonRecord):
    """
    What the pipeline measures of a converted scene, and which part each segment is.
    """

    converted: ConvertedScene
    """
    The conversion the scene was measured from, naming the segment of each part.
    """

    segments: list[SegmentDescriptor] = field(default_factory=list)
    """
    Every segment, measured on its own.
    """

    pairs: list[PairEvidence] = field(default_factory=list)
    """
    Every pair of segments standing in some measurable relation.
    """

    @classmethod
    def measure(cls, directory: Path, settings: PipelineSettings) -> MeasuredScene:
        """
        :param directory: A converted scene's directory.
        :param settings: The settings saying how far the measurement reaches.
        :return: The scene, measured as a run of it would measure it.
        """
        measured = segment_evidence(
            WarsawWorldLoader(
                input_directory=directory, render_sizes=settings.render_sizes
            ),
            nearest=settings.nearest,
        )
        return cls(
            converted=ConvertedScene.beside(directory),
            segments=list(measured.descriptors.values()),
            pairs=measured.pairs,
        )

    @property
    def measured(self) -> SegmentRelations:
        """
        :return: The measurement, in the form the pipeline's steps read it.
        """
        return SegmentRelations(
            descriptors={one.name: one for one in self.segments}, pairs=self.pairs
        )

    @property
    def label_of_segment(self) -> dict[str, str]:
        """
        :return: Per segment, the label it carries.
        """
        return {one.name: one.class_name for one in self.segments}

    def bodies(self, ground_truth: SceneGroundTruth) -> dict[str, np.ndarray]:
        """
        :param ground_truth: The scene's ground truth.
        :return: The faces of each segment written from a part the ground truth keeps,
            which under disjoint labels are exactly that part's faces.
        """
        segment_of_part = self.converted.segment_of_part
        return {
            segment_of_part[one.part_id]: np.asarray(one.faces, dtype=int)
            for one in ground_truth.parts
            if one.part_id in segment_of_part
        }

    @classmethod
    def read(cls, path: Path) -> MeasuredScene:
        """
        :param path: A measured scene's file.
        :return: What it holds.
        """
        return cls.from_json(json.loads(Path(path).read_text()))

    def write(self, path: Path) -> None:
        """
        :param path: Where to write the measured scene.
        """
        Path(path).write_text(json.dumps(self.to_json()))


# %% a measured scene with its ground truth


@dataclass
class SceneToJudge:
    """
    A measured scene, together with the ground truth it is judged against.
    """

    measured: MeasuredScene
    """
    The scene as the pipeline measures it.
    """

    ground_truth: SceneGroundTruth
    """
    Its ground truth.
    """

    @classmethod
    def all_in(
        cls,
        measured_directory: Path,
        ground_truth_directory: Path,
        scene_ids: list[str] | None = None,
    ) -> list[SceneToJudge]:
        """
        :param measured_directory: The directory of measured scenes.
        :param ground_truth_directory: The directory holding every scene's ground truth.
        :param scene_ids: The scenes to judge; every measured scene by default.
        :return: Each scene, in the order of its id.
        """
        if scene_ids is None:
            scene_ids = [
                path.name.removesuffix(MeasuredSceneFile.SUFFIX)
                for path in Path(measured_directory).glob(
                    f"*{MeasuredSceneFile.SUFFIX}"
                )
            ]
        return [
            cls(
                measured=MeasuredScene.read(
                    MeasuredSceneFile.SUFFIX.path(measured_directory, scene_id)
                ),
                ground_truth=SceneGroundTruth.read(
                    GroundTruthFile.SUFFIX.path(ground_truth_directory, scene_id)
                ),
            )
            for scene_id in sorted(scene_ids)
        ]


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for measuring scenes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="The directory holding Articulate3D's annotation files",
    )
    parser.add_argument(
        "--scannetpp",
        type=Path,
        required=True,
        help="ScanNet++'s data directory, holding the meshes the annotations were made on",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ProtocolFile.ARTICULATE3D,
        help="The frozen protocol; the one kept with this package by default",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the measurements"
    )
    parser.add_argument("scenes", nargs="+", help="The scenes to measure")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Convert each scene as a run's input, measure it, and keep only the measurement.

    A scene already measured into the output is left as it is.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once every scene is measured.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    protocol = EvaluationProtocol.read(parsed.protocol)
    settings = PipelineSettings()
    parsed.output.mkdir(parents=True, exist_ok=True)
    for scene_id in parsed.scenes:
        path = MeasuredSceneFile.SUFFIX.path(parsed.output, scene_id)
        if path.is_file():
            logger.info("%s: measured already", scene_id)
            continue
        with tempfile.TemporaryDirectory(dir=parsed.output) as converted:
            directory = Path(converted) / scene_id
            SceneConversion().write_from_dataset(
                scene_id, parsed.dataset, parsed.scannetpp, protocol, directory
            )
            measured = MeasuredScene.measure(directory, settings)
        measured.write(path)
        logger.info(
            "%s: %s segments, %s pairs",
            scene_id,
            len(measured.segments),
            len(measured.pairs),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
