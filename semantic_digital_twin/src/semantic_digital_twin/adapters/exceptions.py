from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException

# %% writing body poses into MJCF files


@dataclass
class BodyNotInMJCFFileError(DataclassException):
    """
    Raised when a body's pose is to be written into an MJCF file that has no ``<body>``
    of that name.
    """

    body_name: str = field(kw_only=True)
    """
    The name of the body that is missing from the file.
    """

    file_path: str = field(kw_only=True)
    """
    The MJCF file the body is missing from.
    """

    def error_message(self) -> str:
        return f'{self.file_path} has no <body name="{self.body_name}">.'

    def suggest_correction(self) -> str:
        return "Write only bodies of the world parsed from this file."


@dataclass
class UnsupportedMJCFOrientationError(DataclassException):
    """
    Raised when an MJCF element a pose is written for, or expressed in, gives its
    orientation by another attribute than ``quat``.
    """

    element_tag: str = field(kw_only=True)
    """
    The start tag of the element, as written in the file.
    """

    orientation_attribute: str = field(kw_only=True)
    """
    The attribute the element gives its orientation by.
    """

    def error_message(self) -> str:
        return (
            f"{self.element_tag} gives its orientation by "
            f'"{self.orientation_attribute}", only "quat" is supported.'
        )

    def suggest_correction(self) -> str:
        return f'Convert "{self.orientation_attribute}" to "quat" in the file.'
