"""
The data files of the GPSR tests, and the names of the files they let annotators write
or look for.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration

# %% test data files


class GpsrDataFile(PathEnumeration):
    """
    The test data files, next to the GPSR tests.
    """

    SEMANTIC_TYPE_MAPPING = Path(__file__).parent / "semantic_type_mapping.json"
    """
    A mapping from detector classes to semantic types, in the format of the mapping
    the semantic annotators read.
    """

    FURNITURE_CLASS_NAMES = Path(__file__).parent / "furniture_class_names.json"
    """
    The class names of a furniture detector, by class id.
    """


# %% names of files the tests let annotators handle


class GpsrFileName(StrEnum):
    """
    Names of files the tests let annotators write or look for.
    """

    BELIEF_STATE_DEBUG = "belief_state.json"
    """
    The debug file the belief state annotator under test writes.
    """

    MISSING_MAPPING = "missing_mapping.json"
    """
    A mapping file no package holds.
    """
