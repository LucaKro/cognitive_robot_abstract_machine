"""
Tests for reporting errors while storing face images.
"""

from __future__ import annotations

# %% Imports
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.exceptions import (
    FaceStoragePathMissing,
    InvalidFaceHypothesis,
    NoAvailableFaceIdentifier,
)
from robokudo.annotators.gpsr.store_unknown_faces import StoreFaces
from robokudo.utils.error_handling import get_blackboard_exception
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# %% face storage errors
def test_missing_face_directory_is_reported_to_blackboard() -> None:
    """
    A missing face directory stops face storage with a specific error.
    """
    annotator = StoreFaces()

    status = annotator.compute()

    assert status == Status.FAILURE
    assert isinstance(get_blackboard_exception(), FaceStoragePathMissing)


def test_face_storage_rejects_an_invalid_hypothesis() -> None:
    """
    A non-human hypothesis cannot be checked for a face classification.
    """
    annotator = StoreFaces()

    with pytest.raises(InvalidFaceHypothesis):
        annotator._has_classification(object())


def test_face_storage_reports_exhausted_identifiers(tmp_path: Path) -> None:
    """
    All occupied face identifiers prevent another face from being stored.
    """
    for identifier in range(1000):
        (tmp_path / f"{identifier}.png").touch()
    annotator = StoreFaces()

    with pytest.raises(NoAvailableFaceIdentifier):
        annotator._get_id(tmp_path)
