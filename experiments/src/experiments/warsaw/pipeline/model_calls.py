"""Compact, reproducible records of every request made to a model."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.utils import get_full_class_name
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    MessagePart,
    PartKind,
    TextPart,
)

from experiments.warsaw.bases import JsonRecord

# %% the parts sent to a model


@dataclass(frozen=True)
class ModelCallPart(JsonRecord):
    """One message part, retaining text and identifying image content."""

    kind: PartKind
    """Whether the part contains text or an image."""

    text: str | None = None
    """The exact text sent, when this is a text part."""

    image_sha256: str | None = None
    """A stable digest of the exact image bytes, when this is an image part."""

    image_bytes: int | None = None
    """The image size in bytes, when this is an image part."""

    @classmethod
    def of(cls, part: MessagePart) -> ModelCallPart:
        """Make a compact record from a model message part.

        :param part: The text or image sent to the model.
        :return: The part without duplicating base64 image data in every trace.
        :raises TypeError: If the caller supplies an unknown message-part type.
        """
        if isinstance(part, TextPart):
            return cls(kind=PartKind.TEXT, text=part.text)
        if isinstance(part, ImagePart):
            return cls(
                kind=PartKind.IMAGE,
                image_sha256=sha256(part.image).hexdigest(),
                image_bytes=len(part.image),
            )
        raise TypeError(f"Unsupported model message part: {type(part).__name__}")


# %% one complete attempt


@dataclass(frozen=True)
class ModelCallTrace:
    """The request, raw response, and validation result for one attempt."""

    question: str
    """The stable key of the question being answered."""

    attempt: int
    """The one-based attempt number for this question."""

    requested_model: str
    """The model identifier configured for the request."""

    system_prompt: str
    """The exact system prompt sent with the request."""

    message_parts: list[ModelCallPart]
    """The ordered text and image identities sent as the user message."""

    response: dict[str, Any]
    """The complete response payload returned by the provider or read from disk."""

    problems: list[str]
    """Validation problems found after reading this response."""

    reused: bool
    """Whether the response was loaded from a previously kept answer."""

    started_at: str
    """The UTC time at which this attempt began."""

    elapsed_seconds: float
    """Wall-clock time spent waiting for this response."""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready record while preserving the raw provider payload."""
        return {
            JSON_TYPE_NAME: get_full_class_name(type(self)),
            "question": self.question,
            "attempt": self.attempt,
            "requested_model": self.requested_model,
            "system_prompt": self.system_prompt,
            "message_parts": [part.to_json() for part in self.message_parts],
            "response": self.response,
            "problems": self.problems,
            "reused": self.reused,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
        }
