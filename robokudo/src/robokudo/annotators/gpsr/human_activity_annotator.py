"""
This module implements a RoboKudo Annotator: The HumanActivityAnnotator.

This annotator uses a pretrained CLIP model to classify human activities with a zeroshot
approach. It distinguishes between mutually exclusive postures (e.g. sitting, standing)
and gestures (raising, waving, pointing).

Gesture direction (left/right) is not determined here.

Author: Ali Talha Sükür Created at: 2026-04-14
"""

from __future__ import annotations

# %% Imports
from timeit import default_timer

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from py_trees.common import Status
from transformers import CLIPModel, CLIPProcessor

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Encoding
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

# %% CLIP activity annotator


class HumanActivityAnnotator(ThreadedAnnotator):
    """
    Annotates every person with the posture and the gesture CLIP finds most probable,
    and with the CLIP embedding of the person.

    The posture is decided on the whole person, the gesture on the upper body, widened
    to both sides. The most probable posture is always annotated; a gesture only when
    its probability reaches the gesture threshold, otherwise the person is annotated
    with ``No gesture``.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the CLIP activity annotator.
        """

        class Parameters:
            """
            Parameters of the CLIP activity annotator.
            """

            def __init__(self) -> None:
                self.posture_classes: tuple[str, ...] = (
                    "a person standing upright with arms at their sides",
                    "a person sitting upright with torso vertical",
                    "a person lying down with their body mostly horizontal",
                    "a person crouching or squatting low",
                )
                """
                The texts CLIP compares the crop of the whole person with, one per
                posture.
                """
                self.posture_labels: tuple[str, ...] = (
                    "Standing",
                    "Sitting",
                    "Lying",
                    "Crouching",
                )
                """
                The posture annotated for each text of :attr:`posture_classes`, in the
                same order.
                """

                self.gesture_classes: tuple[str, ...] = (
                    "a person waving hello with an open palm",
                    "a person with their arm extended out pointing at something",
                    "a person with one arm raised up in the air",
                    "a person standing with both arms down at their sides",
                )
                """
                The texts CLIP compares the crop of the upper body with, one per
                gesture.
                """
                self.gesture_labels: tuple[str, ...] = (
                    "Waving",
                    "Pointing",
                    "Raising",
                    "No gesture",
                )
                """
                The gesture annotated for each text of :attr:`gesture_classes`, in the
                same order.
                """

                self.gesture_threshold: float = 0.65
                """
                Smallest probability of a gesture for it to be annotated; a person
                without such a gesture is annotated with ``No gesture``.
                """
                self.save_top_k: int = 1
                """
                How many gestures reaching the threshold are annotated per person, the
                most probable first.
                """

                self.clip_model: str = "openai/clip-vit-large-patch14"
                """
                Name of the pretrained CLIP model.
                """
                self.clip_processor: str = "openai/clip-vit-large-patch14"
                """
                Name of the pretrained CLIP processor.
                """

                self.upper_body_padding_fraction: float = 0.2
                """
                Fraction of the width of a person added on each side of the upper body
                crop, as far as the image reaches.
                """
                self.upper_body_height_fraction: float = 0.6
                """
                Fraction of the height of a person, from the top, that the upper body
                crop covers.
                """

        parameters = Parameters()
        """
        Default parameters on the class; each descriptor instance gets its own.
        """

    def __init__(
        self,
        name: str = "HumanActivityAnnotator",
        descriptor: HumanActivityAnnotator.Descriptor = Descriptor(),
    ) -> None:
        self.parameters: HumanActivityAnnotator.Descriptor.Parameters = (
            descriptor.parameters
        )
        """
        The parameters of the descriptor.
        """
        super().__init__(name, descriptor)
        self.model: CLIPModel = CLIPModel.from_pretrained(self.parameters.clip_model)
        """
        The CLIP model, loaded once when the annotator is created.
        """
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            self.parameters.clip_processor
        )
        """
        The CLIP processor, loaded once when the annotator is created.
        """

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        start_timer = default_timer()
        self.analyze_human_hypotheses()
        self.feedback_message = (
            f"Processing took {(default_timer() - start_timer):.4f}s"
        )
        return Status.SUCCESS

    def cutout_human_rois(
        self, human_hypotheses: list[HumanHypothesis], upper_body: bool = False
    ) -> list[Image.Image]:
        """
        Cut every person out of the color image, in RGB order.

        :param human_hypotheses: The persons to cut out.
        :param upper_body: Whether to cut out the widened, contrast-stretched upper
            body instead of the whole person.
        :return: One crop per person, in the order of the persons.
        """
        # the CAS holds the color image in OpenCV's BGR order, CLIP expects RGB
        image = Image.fromarray(
            cv2.cvtColor(self.get_cas().get(CASViews.COLOR_IMAGE), cv2.COLOR_BGR2RGB)
        )

        image_width, _ = image.size
        crops = []
        for human in human_hypotheses:
            x = human.roi.roi.pos.x
            y = human.roi.roi.pos.y
            width = human.roi.roi.width
            height = human.roi.roi.height

            if upper_body:
                padding = int(width * self.parameters.upper_body_padding_fraction)
                left = max(0, x - padding)
                right = min(image_width, x + width + padding)
                bottom = y + int(height * self.parameters.upper_body_height_fraction)
            else:
                left = x
                right = x + width
                bottom = y + height

            crop = image.crop((left, y, right, bottom))

            if upper_body:
                crop = ImageOps.autocontrast(crop)

            crops.append(crop)

        return crops

    def analyze_human_hypotheses(self) -> None:
        """
        Run CLIP on every person in the CAS and annotate each with its posture, its
        gesture and its image embedding.
        """
        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        if not human_hypotheses:
            return

        whole_person_crops = self.cutout_human_rois(human_hypotheses)
        upper_body_crops = self.cutout_human_rois(human_hypotheses, upper_body=True)

        posture_inputs = self.processor(
            text=list(self.parameters.posture_classes),
            images=whole_person_crops,
            return_tensors="pt",
            padding=True,
        )
        gesture_inputs = self.processor(
            text=list(self.parameters.gesture_classes),
            images=upper_body_crops,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            posture_probabilities = (
                self.model(**posture_inputs)
                .logits_per_image.softmax(dim=1)
                .detach()
                .numpy()
            )
            gesture_probabilities = (
                self.model(**gesture_inputs)
                .logits_per_image.softmax(dim=1)
                .detach()
                .numpy()
            )
            raw_features = self.model.get_image_features(
                pixel_values=posture_inputs["pixel_values"]
            ).pooler_output
            image_features = (
                (raw_features / raw_features.norm(dim=-1, keepdim=True))
                .detach()
                .cpu()
                .numpy()
            )

        for human, posture_distribution, gesture_distribution, image_feature in zip(
            human_hypotheses,
            posture_probabilities,
            gesture_probabilities,
            image_features,
        ):
            self.annotate_person(
                human, posture_distribution, gesture_distribution, image_feature
            )

        self.visualize_humans(human_hypotheses)

    def annotate_person(
        self,
        hh: HumanHypothesis,
        posture_dist: np.ndarray,
        gesture_dist: np.ndarray,
        img_feat: np.ndarray,
    ) -> None:
        """
        Annotate one person with the CLIP image embedding, the most probable posture,
        and the most probable gestures that reach the threshold, or ``No gesture`` if
        none does.

        :param hh: The person.
        :param posture_dist: Probability of each posture text.
        :param gesture_dist: Probability of each gesture text.
        :param img_feat: The normalized CLIP embedding of the crop of the whole person.
        """
        clip_encoding = Encoding()
        clip_encoding.encoding = img_feat
        clip_encoding.source = type(self).__name__
        hh.annotations.append(clip_encoding)

        posture_label = self.parameters.posture_labels[posture_dist.argmax()]
        posture_annotation = ActivityAnnotation()
        posture_annotation.type = posture_label
        posture_annotation.source = type(self).__name__
        hh.annotations.append(posture_annotation)

        for index, (label, score) in enumerate(
            zip(self.parameters.posture_labels, posture_dist)
        ):
            self.rk_logger.info(
                f"  Posture HH {hh.id}: {label} = {score:.2f}"
                + ("  <-- WINNER" if index == posture_dist.argmax() else "")
            )

        self.rk_logger.info(f"  Gesture scores HH {hh.id}:")
        for label, score in zip(self.parameters.gesture_labels, gesture_dist):
            self.rk_logger.info(
                f"    {label} = {score:.2f}"
                + (
                    "  <-- THRESHOLD"
                    if score >= self.parameters.gesture_threshold
                    else ""
                )
            )

        gesture_count = 0
        for index in gesture_dist.argsort()[::-1]:
            if gesture_count >= self.parameters.save_top_k:
                break
            label = self.parameters.gesture_labels[index]
            if label == "No gesture":
                continue
            if gesture_dist[index] >= self.parameters.gesture_threshold:
                gesture_annotation = ActivityAnnotation()
                gesture_annotation.type = label
                gesture_annotation.source = type(self).__name__
                hh.annotations.append(gesture_annotation)
                self.rk_logger.info(
                    f"Gesture for HH {hh.id}: {label} ({gesture_dist[index]:.2f})"
                )
                gesture_count += 1

        if gesture_count == 0:
            no_gesture_annotation = ActivityAnnotation()
            no_gesture_annotation.type = "No gesture"
            no_gesture_annotation.source = type(self).__name__
            hh.annotations.append(no_gesture_annotation)
            self.rk_logger.info(f"Gesture for HH {hh.id}: No gesture")

    def visualize_humans(self, human_hypotheses: list[HumanHypothesis]) -> None:
        """
        Draw the box of every person with its posture and gestures into the output
        image.

        :param human_hypotheses: The persons to draw.
        """
        image = Image.fromarray(self.get_cas().get_copy(CASViews.COLOR_IMAGE))
        draw = ImageDraw.Draw(image)

        for human in human_hypotheses:
            x, y = human.roi.roi.pos.x, human.roi.roi.pos.y
            width, height = human.roi.roi.width, human.roi.roi.height
            draw.rectangle((x, y, x + width, y + height), outline=(0, 255, 0), width=2)

            activities = [
                annotation
                for annotation in human.annotations
                if isinstance(annotation, ActivityAnnotation)
            ]
            postures = [
                activity
                for activity in activities
                if activity.type in self.parameters.posture_labels
            ]
            gestures = [
                activity
                for activity in activities
                if activity.type in self.parameters.gesture_labels
            ]

            if postures:
                draw.text((x, y - 20), postures[0].type, fill=(0, 255, 0))
            if gestures:
                draw.text(
                    (x, y - 40),
                    ", ".join(gesture.type for gesture in gestures),
                    fill=(255, 255, 0),
                )

        self.get_annotator_output_struct().set_image(np.array(image))
