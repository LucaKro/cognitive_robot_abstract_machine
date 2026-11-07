"""
Stand-ins for the CLIP model and processor that score every person with probabilities
the test chooses, so no weights are downloaded.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

# %% processor


@dataclass
class RecordingProcessor:
    """
    CLIP processor stand-in that records the crops and texts it is handed.
    """

    received_images: list[list[Image.Image]] = field(default_factory=list)
    """
    The crops of every call, in call order.
    """

    def __call__(
        self,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor | list[str]]:
        """
        Record the crops and prepare them for the model.

        :param text: The texts to score.
        :param images: The crops of the persons.
        :param return_tensors: The tensor format, as the real processor takes it.
        :param padding: Whether to pad, as the real processor takes it.
        :return: The texts and one placeholder pixel row per crop.
        """
        self.received_images.append(list(images))
        return {"texts": text, "pixel_values": torch.zeros((len(images), 1))}


# %% model


@dataclass
class ImageTextLogits:
    """
    The part of a CLIP model output the activity annotator reads.
    """

    logits_per_image: torch.Tensor
    """
    Similarity of every image to every text.
    """


@dataclass
class ScoringModel:
    """
    CLIP model stand-in that reports the same probabilities for every person.
    """

    posture_texts: tuple[str, ...]
    """
    The texts that describe postures, which tell a posture call from a gesture call.
    """

    posture_probabilities: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0]
    )
    """
    Probability of each posture text.
    """

    gesture_probabilities: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0]
    )
    """
    Probability of each gesture text.
    """

    image_embedding: list[float] = field(default_factory=lambda: [3.0, 4.0])
    """
    The embedding every person crop gets, before normalization.
    """

    call_count: int = 0
    """
    How often the model scored images.
    """

    def __call__(self, texts: list[str], pixel_values: torch.Tensor) -> ImageTextLogits:
        """
        Score every crop against the posture or the gesture texts.

        :param texts: The texts to score.
        :param pixel_values: One placeholder pixel row per crop.
        :return: Logits that give every crop the chosen probabilities.
        """
        self.call_count += 1
        probabilities = (
            self.posture_probabilities
            if tuple(texts) == self.posture_texts
            else self.gesture_probabilities
        )
        logits = torch.log(torch.tensor(probabilities, dtype=torch.float32))
        return ImageTextLogits(logits_per_image=logits.repeat(len(pixel_values), 1))

    def get_image_features(
        self, pixel_values: torch.Tensor
    ) -> BaseModelOutputWithPooling:
        """
        :param pixel_values: One placeholder pixel row per crop.
        :return: The chosen embedding for every crop.
        """
        return BaseModelOutputWithPooling(
            pooler_output=torch.tensor([self.image_embedding] * len(pixel_values))
        )

    @staticmethod
    def seen_probabilities(probabilities: list[float]) -> np.ndarray:
        """
        :param probabilities: The probabilities of the texts.
        :return: The probabilities as the annotator sees them after normalizing the
            reported logits.
        """
        logits = torch.log(torch.tensor(probabilities, dtype=torch.float32))
        return logits.softmax(dim=0).numpy()


# %% stand-in pair


@dataclass
class ScoringModelAndProcessor:
    """
    A model scoring every person with chosen probabilities and a processor recording
    the crops it is handed, loaded in place of pretrained CLIP.
    """

    model: ScoringModel
    """
    The stand-in CLIP model.
    """

    processor: RecordingProcessor = field(default_factory=RecordingProcessor)
    """
    The stand-in CLIP processor.
    """


# %% loading


@dataclass
class PretrainedLoader:
    """
    Stand-in for a pretrained model class, handing out a stand-in instead of loading
    weights.
    """

    loaded: ScoringModel | RecordingProcessor
    """
    What every load hands out.
    """

    def from_pretrained(self, model_name: str) -> ScoringModel | RecordingProcessor:
        """
        :param model_name: The name of the requested model.
        :return: The stand-in, whichever model was asked for.
        """
        return self.loaded
