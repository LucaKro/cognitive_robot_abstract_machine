"""
The parts a question to a vision-language model is built from.

A question is a sequence of parts rather than one string, because the models these are
put to read pictures beside the words and the two have to keep their order: a caption
saying what the next picture shows is worth nothing if it arrives after it.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Dict


class PartKind(StrEnum):
    """
    What a message part is, as the chat completions schema names it.
    """

    TEXT = "text"
    """
    Words.
    """

    IMAGE = "image_url"
    """
    A picture, carried inline rather than by URL so nothing has to be hosted.
    """


@dataclass
class MessagePart(ABC):
    """
    One part of what a model is shown.
    """

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """
        :return: The part as the chat completions schema takes it.
        """


@dataclass
class TextPart(MessagePart):
    """
    Words for the model to read.
    """

    text: str
    """
    What to say.
    """

    def to_json(self) -> Dict[str, Any]:
        return {"type": PartKind.TEXT.value, "text": self.text}


@dataclass
class ImagePart(MessagePart):
    """
    A picture the model is shown, carried inline as a data URL.
    """

    image: bytes
    """
    The PNG, as bytes.
    """

    @classmethod
    def from_file(cls, path: Path) -> ImagePart:
        """
        :param path: The PNG to show.
        :return: It, ready to be shown.
        """
        return cls(image=Path(path).read_bytes())

    def to_json(self) -> Dict[str, Any]:
        encoded = base64.b64encode(self.image).decode("utf-8")
        return {
            "type": PartKind.IMAGE.value,
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }
