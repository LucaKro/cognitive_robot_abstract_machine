"""
A stand-in for the face recognition library that finds faces where the test places them,
so no face model is loaded.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

import numpy as np

# %% face finder


@dataclass
class FaceLocation:
    """
    A face in the image the detector searches, in the order the face recognition library
    reports it.
    """

    top: int
    """
    Upper edge of the face, in pixels.
    """

    right: int
    """
    Right edge of the face, in pixels.
    """

    bottom: int
    """
    Lower edge of the face, in pixels.
    """

    left: int
    """
    Left edge of the face, in pixels.
    """

    def as_tuple(self) -> tuple[int, int, int, int]:
        """
        :return: The face as the face recognition library reports it.
        """
        return self.top, self.right, self.bottom, self.left


@dataclass
class PlacedFaces:
    """
    Face recognition stand-in that reports the placed faces in every image and records
    the size of the images it searched.
    """

    faces: list[FaceLocation] = field(default_factory=list)
    """
    The faces reported in every image.
    """

    encoding: np.ndarray = field(default_factory=lambda: np.arange(128.0))
    """
    The encoding reported for every face.
    """

    searched_image_sizes: list[tuple[int, int]] = field(default_factory=list)
    """
    Width and height of every searched image, in search order.
    """

    def face_locations(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        :param image: The image to search.
        :return: The placed faces.
        """
        self.searched_image_sizes.append((image.shape[1], image.shape[0]))
        return [face.as_tuple() for face in self.faces]

    def face_encodings(
        self, image: np.ndarray, known_face_locations: list[tuple[int, int, int, int]]
    ) -> list[np.ndarray]:
        """
        :param image: The image of the faces.
        :param known_face_locations: The faces to encode.
        :return: The encoding of every given face.
        """
        return [self.encoding for _ in known_face_locations]
