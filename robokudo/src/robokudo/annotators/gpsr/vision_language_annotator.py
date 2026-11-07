"""
Asks a local vision language model about every detected person.
"""

from __future__ import annotations

# %% Imports
import base64
from enum import IntEnum
from io import BytesIO

import cv2
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from PIL import Image
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.file_loader import FileLoader

# %% layers on the GPU


class GpuLayers(IntEnum):
    """
    How many layers of the model llama.cpp puts on the GPU.
    """

    ALL = -1
    """
    Every layer, as far as llama.cpp was built with GPU support.
    """

    NONE = 0
    """
    No layer, so the model runs on the CPU.
    """


# %% vision language annotator


class VisionLanguageAnnotator(ThreadedAnnotator):
    """
    Asks a local vision language model a question about the crop of every person and
    annotates the person with the answer.

    Persons with an empty answer are left without annotation.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the vision language annotator.
        """

        class Parameters:
            """
            Parameters of the vision language annotator.
            """

            def __init__(self) -> None:
                """
                Start with the default model files and question.
                """
                self.ros_package_name: str = "robokudo_suturo"
                """
                The ROS package that holds the model files.
                """
                self.model_file: str = "weights/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
                """
                The language model, relative to the ROS package.
                """
                self.projector_file: str = (
                    "weights/Qwen2.5-VL-3B-Instruct-mmproj-F16.gguf"
                )
                """
                The image projector of the model, relative to the ROS package.
                """
                self.context_length: int = 4096
                """
                How many tokens the model reads and writes per question.
                """
                self.use_gpu: bool = True
                """
                Whether llama.cpp puts the model on the GPU, as far as it was built with
                GPU support.
                """
                self.temperature: float = 0.0001
                """
                Sampling temperature of the answers.
                """
                self.system_prompt: str = (
                    "You are a robot vision system. Answer questions about people in "
                    "images in one short sentence. When asked about appearance, always "
                    'use the format: "[color] [item]". Example: blue jeans, white '
                    "t-shirt, brown hair. Never explain, never greet, just answer."
                )
                """
                The instructions the model gets before every question.
                """
                self.question: str = (
                    "What clothing is this person wearing and what colors?"
                )
                """
                The question asked about every person.
                """

        parameters = Parameters()
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "VisionLanguageAnnotator",
        descriptor: VisionLanguageAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name, descriptor)
        self.model: Llama | None = None
        """
        The vision language model, loaded at the first question.
        """

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """
        Overrides :meth:`ThreadedAnnotator.compute` to ask the model about every person
        and annotate each answer.

        :return: :attr:`Status.FAILURE` without persons, otherwise
            :attr:`Status.SUCCESS`.
        """
        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        if not human_hypotheses:
            self.feedback_message = "No humans found"
            return Status.FAILURE

        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        image = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        for human_hypothesis in human_hypotheses:
            region = human_hypothesis.roi.roi
            crop = image.crop(
                (
                    region.pos.x,
                    region.pos.y,
                    min(image.width, region.pos.x + region.width),
                    region.pos.y + region.height,
                )
            )
            answer = self.ask(crop)
            if not answer:
                continue
            human_hypothesis.annotations.append(
                ActivityAnnotation(type=answer, source=self.get_class_name())
            )
            self.feedback_message = answer
        return Status.SUCCESS

    def ask(self, image: Image.Image) -> str:
        """
        Ask the configured question about an image.

        :param image: The crop of the person.
        :return: The answer of the model without surrounding spaces.
        """
        parameters = self.descriptor.parameters
        if self.model is None:
            self.model = self.load_model()
        encoded = BytesIO()
        image.save(encoded, format="JPEG")
        image_url = "data:image/jpeg;base64," + base64.b64encode(
            encoded.getvalue()
        ).decode("utf-8")
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": parameters.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": parameters.question},
                    ],
                },
            ],
            temperature=parameters.temperature,
        )
        return response["choices"][0]["message"]["content"].strip()

    def load_model(self) -> Llama:
        """
        :return: The configured model with its image projector.
        """
        parameters = self.descriptor.parameters
        model_path = FileLoader.get_path_to_file_in_ros_package(
            ros_pkg_name=parameters.ros_package_name,
            relative_path=parameters.model_file,
        )
        projector_path = FileLoader.get_path_to_file_in_ros_package(
            ros_pkg_name=parameters.ros_package_name,
            relative_path=parameters.projector_file,
        )
        return Llama(
            model_path=str(model_path),
            chat_handler=Qwen25VLChatHandler(
                clip_model_path=str(projector_path), verbose=False
            ),
            n_ctx=parameters.context_length,
            n_gpu_layers=GpuLayers.ALL if parameters.use_gpu else GpuLayers.NONE,
            verbose=False,
        )
