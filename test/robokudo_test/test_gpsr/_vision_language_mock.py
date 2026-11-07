"""
Stand-ins for a llama.cpp vision language model and its image projector, so no weights
are loaded.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from typing_extensions import Any

# %% model


@dataclass
class RecordingVisionLanguageModel:
    """
    Vision language model stand-in that answers every question with the same text and
    records the conversations it is handed.
    """

    answer: str = "blue jeans, white t-shirt"
    """
    The text the model answers with.
    """

    received_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    """
    The messages of every call, in call order.
    """

    received_temperatures: list[float] = field(default_factory=list)
    """
    The sampling temperature of every call, in call order.
    """

    def create_chat_completion(
        self, messages: list[dict[str, Any]], temperature: float
    ) -> dict[str, Any]:
        """
        Record the conversation and answer it.

        :param messages: The conversation.
        :param temperature: The sampling temperature, as the real model takes it.
        :return: The answer in the shape llama.cpp returns it.
        """
        self.received_messages.append(messages)
        self.received_temperatures.append(temperature)
        return {"choices": [{"message": {"content": self.answer}}]}


# %% loaders


@dataclass
class ModelLoader:
    """
    Stands in for the llama.cpp model class: hands out the recording model and records
    how it was loaded.
    """

    loaded: RecordingVisionLanguageModel = field(
        default_factory=RecordingVisionLanguageModel
    )
    """
    The model every load returns.
    """

    load_arguments: list[dict[str, Any]] = field(default_factory=list)
    """
    The keyword arguments of every load, in load order.
    """

    def __call__(self, **arguments: Any) -> RecordingVisionLanguageModel:
        """
        :param arguments: The keyword arguments of the load.
        :return: The recording model.
        """
        self.load_arguments.append(arguments)
        return self.loaded


@dataclass
class ProjectorLoader:
    """
    Stands in for the llama.cpp chat handler that loads the image projector, and records
    which projector it was handed.
    """

    projector_paths: list[str] = field(default_factory=list)
    """
    The projector file of every load, in load order.
    """

    def __call__(self, clip_model_path: str, verbose: bool) -> str:
        """
        :param clip_model_path: The path of the projector.
        :param verbose: Whether to log, as the real loader takes it.
        :return: The projector path, standing in for the loaded handler.
        """
        self.projector_paths.append(clip_model_path)
        return clip_model_path
