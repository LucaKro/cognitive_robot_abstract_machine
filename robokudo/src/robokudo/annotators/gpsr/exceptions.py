"""
Exceptions raised by GPSR annotators.
"""

# %% Imports
from dataclasses import dataclass

from krrood.exceptions import DataclassException


# %% object classification
@dataclass
class MissingObjectClassification(DataclassException):
    """
    An object has no classification for the query answer.
    """

    def error_message(self) -> str:
        return "An object has no classification."

    def suggest_correction(self) -> str:
        return "Ensure each detected object has a classification."


# %% face storage
@dataclass
class FaceImageDirectoryMissing(DataclassException, RuntimeError):
    """
    Face classification has no configured image directory.
    """

    def error_message(self) -> str:
        return "The face image directory is not configured."

    def suggest_correction(self) -> str:
        return "Configure a face image directory."


@dataclass
class FaceLabelsMismatch(DataclassException, RuntimeError):
    """
    The numbers of face images and labels differ.
    """

    def error_message(self) -> str:
        return "Face image names and labels have different lengths."

    def suggest_correction(self) -> str:
        return "Provide one label for each face image."


@dataclass
class FaceImageScalingFailed(DataclassException):
    """
    The color image cannot be scaled for face point generation.
    """

    def error_message(self) -> str:
        return "The color image cannot be scaled to the depth image."

    def suggest_correction(self) -> str:
        return "Check the camera's color-to-depth ratio and image configuration."


@dataclass
class UnequalColorDepthRatios(DataclassException, RuntimeError):
    """
    The color-to-depth ratios differ between image axes.
    """

    def error_message(self) -> str:
        return "The horizontal and vertical color-to-depth ratios differ."

    def suggest_correction(self) -> str:
        return "Provide equal horizontal and vertical color-to-depth ratios."


@dataclass
class FaceStoragePathMissing(DataclassException):
    """
    The directory for storing face images is not configured.
    """

    def error_message(self) -> str:
        return "The face image directory is not configured."

    def suggest_correction(self) -> str:
        return "Configure a face image directory."


@dataclass
class InvalidFaceHypothesis(DataclassException):
    """
    A face storage operation received an invalid human hypothesis.
    """

    actual_type: type
    """Type supplied to the face storage operation."""

    def error_message(self) -> str:
        return f"Face storage received {self.actual_type.__name__} instead of a human hypothesis."

    def suggest_correction(self) -> str:
        return "Provide a human hypothesis."


@dataclass
class NoAvailableFaceIdentifier(DataclassException):
    """
    No face image identifier is available in the configured directory.
    """

    def error_message(self) -> str:
        return "No face image identifier is available."

    def suggest_correction(self) -> str:
        return "Free an existing face image identifier before storing another image."


# %% area filter
@dataclass
class InvalidAreaPose(DataclassException):
    """
    An area filter received an unsupported position annotation.
    """

    actual_type: type
    """Type supplied to the area filter."""

    def error_message(self) -> str:
        return f"The area filter cannot transform {self.actual_type.__name__}."

    def suggest_correction(self) -> str:
        return "Provide a position or pose annotation."
