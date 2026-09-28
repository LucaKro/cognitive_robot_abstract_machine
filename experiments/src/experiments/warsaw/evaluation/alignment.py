"""Estimate reconstruction-to-ground-truth alignment from named landmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# %% invalid landmark sets


class LandmarkAlignmentError(ValueError):
    """A landmark set cannot determine a coordinate-frame alignment."""


class InsufficientLandmarksError(LandmarkAlignmentError):
    """Fewer than three landmarks were supplied."""


class DegenerateLandmarksError(LandmarkAlignmentError):
    """The landmarks do not span enough directions to determine rotation."""


# %% corresponding points


@dataclass(frozen=True)
class Landmark:
    """One point identified in both the reconstruction and ground-truth frame."""

    name: str
    """A human-readable description of the physical point."""

    reconstruction: tuple[float, float, float]
    """The point in reconstruction coordinates."""

    ground_truth: tuple[float, float, float]
    """The same physical point in ground-truth coordinates."""


@dataclass(frozen=True)
class LandmarkResidual:
    """Distance left at one landmark after alignment."""

    name: str
    """The landmark being measured."""

    distance: float
    """Its Euclidean error in ground-truth units."""


# %% fitted alignment


@dataclass(frozen=True)
class LandmarkAlignment:
    """A similarity transform from reconstruction to ground-truth coordinates."""

    rotation: NDArray[np.float64]
    """The proper three-dimensional rotation matrix."""

    translation: NDArray[np.float64]
    """The translation applied after rotation and scale."""

    scale: float
    """The uniform scale applied before translation."""

    residuals: tuple[LandmarkResidual, ...]
    """The post-fit error at every supplied landmark."""

    root_mean_square_error: float
    """The root mean square landmark residual."""

    @classmethod
    def fit(
        cls,
        landmarks: Sequence[Landmark],
        *,
        estimate_scale: bool = False,
    ) -> LandmarkAlignment:
        """Fit a proper rotation, translation, and optional uniform scale.

        :param landmarks: Corresponding points in the two frames.
        :param estimate_scale: Whether to estimate a uniform unit conversion.
        :return: The least-squares similarity transform.
        :raises InsufficientLandmarksError: If fewer than three points are supplied.
        :raises DegenerateLandmarksError: If either point set is collinear.
        """
        if len(landmarks) < 3:
            raise InsufficientLandmarksError(
                "At least three non-collinear landmarks are required."
            )

        reconstruction = np.asarray(
            [landmark.reconstruction for landmark in landmarks], dtype=np.float64
        )
        ground_truth = np.asarray(
            [landmark.ground_truth for landmark in landmarks], dtype=np.float64
        )
        reconstruction_center = reconstruction.mean(axis=0)
        ground_truth_center = ground_truth.mean(axis=0)
        centered_reconstruction = reconstruction - reconstruction_center
        centered_ground_truth = ground_truth - ground_truth_center
        if (
            np.linalg.matrix_rank(centered_reconstruction) < 2
            or np.linalg.matrix_rank(centered_ground_truth) < 2
        ):
            raise DegenerateLandmarksError(
                "Landmarks must include at least three non-collinear points in both frames."
            )

        covariance = centered_reconstruction.T @ centered_ground_truth
        left_singular_vectors, singular_values, right_singular_vectors = np.linalg.svd(
            covariance
        )
        rotation = right_singular_vectors.T @ left_singular_vectors.T
        if np.linalg.det(rotation) < 0:
            right_singular_vectors[-1, :] *= -1
            singular_values[-1] *= -1
            rotation = right_singular_vectors.T @ left_singular_vectors.T

        scale = 1.0
        if estimate_scale:
            squared_distance = float(np.square(centered_reconstruction).sum())
            scale = float(singular_values.sum() / squared_distance)
        translation = ground_truth_center - scale * rotation @ reconstruction_center

        transformed = scale * reconstruction @ rotation.T + translation
        distances = np.linalg.norm(transformed - ground_truth, axis=1)
        residuals = tuple(
            LandmarkResidual(name=landmark.name, distance=float(distance))
            for landmark, distance in zip(landmarks, distances)
        )
        return cls(
            rotation=rotation,
            translation=translation,
            scale=scale,
            residuals=residuals,
            root_mean_square_error=float(np.sqrt(np.square(distances).mean())),
        )

    @property
    def homogeneous_matrix(self) -> NDArray[np.float64]:
        """Return the transform as a homogeneous four-by-four matrix."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def transform_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform reconstruction points into ground-truth coordinates.

        :param points: An array whose rows are three-dimensional points.
        :return: The points expressed in the ground-truth frame.
        """
        points = np.asarray(points, dtype=np.float64)
        return self.scale * points @ self.rotation.T + self.translation

    def to_json(self, *, source_frame: str, target_frame: str) -> dict[str, Any]:
        """Describe the fitted transform and all evidence used to assess it.

        :param source_frame: The coordinate frame transformed by this result.
        :param target_frame: The coordinate frame produced by this result.
        :return: A JSON-ready transform with aggregate and per-landmark residuals.
        """
        return {
            "schema_version": 1,
            "source_frame": source_frame,
            "target_frame": target_frame,
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "matrix": self.homogeneous_matrix.tolist(),
            "landmark_count": len(self.residuals),
            "root_mean_square_error": self.root_mean_square_error,
            "residuals": [
                {"name": residual.name, "distance": residual.distance}
                for residual in self.residuals
            ],
        }


# %% portable landmark input


@dataclass(frozen=True)
class LandmarkFile:
    """Corresponding points selected in reconstruction and ground-truth viewers."""

    reconstruction_frame: str
    """A human-readable name for the reconstruction coordinate frame."""

    ground_truth_frame: str
    """A human-readable name for the ground-truth coordinate frame."""

    landmarks: tuple[Landmark, ...]
    """The same physical points measured in both frames."""

    estimate_scale: bool = False
    """Whether the two coordinate frames may use different units."""

    @classmethod
    def read(cls, path: Path) -> LandmarkFile:
        """Read the versioned landmark interchange format.

        :param path: A JSON file containing named corresponding points.
        :return: The landmark set ready to fit.
        :raises ValueError: If the file uses an unsupported schema version.
        """
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported landmark schema version: {data.get('schema_version')!r}"
            )
        landmarks = tuple(
            Landmark(
                name=item["name"],
                reconstruction=tuple(item["reconstruction"]),
                ground_truth=tuple(item["ground_truth"]),
            )
            for item in data["landmarks"]
        )
        return cls(
            reconstruction_frame=data["reconstruction_frame"],
            ground_truth_frame=data["ground_truth_frame"],
            landmarks=landmarks,
            estimate_scale=bool(data.get("estimate_scale", False)),
        )

    def fit(self) -> LandmarkAlignment:
        """Fit the transform requested by this landmark file."""
        return LandmarkAlignment.fit(self.landmarks, estimate_scale=self.estimate_scale)


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for landmark alignment."""
    parser = argparse.ArgumentParser(
        description="Fit reconstruction-to-ground-truth alignment from landmarks."
    )
    parser.add_argument("landmarks", type=Path, help="Versioned landmark JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the fitted transform and residuals",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Fit a landmark file and write an auditable coordinate transform.

    :param arguments: Command-line arguments without the program name.
    :return: Zero after the output is written.
    """
    parsed = argument_parser().parse_args(arguments)
    landmarks = LandmarkFile.read(parsed.landmarks)
    alignment = landmarks.fit()
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(
        json.dumps(
            alignment.to_json(
                source_frame=landmarks.reconstruction_frame,
                target_frame=landmarks.ground_truth_frame,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
