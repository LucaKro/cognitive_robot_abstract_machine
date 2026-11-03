from __future__ import annotations

# %% imports
import json
from enum import StrEnum

import py_trees
import robokudo.analysis_engine
import robokudo.annotators.core
from typing_extensions import TYPE_CHECKING
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.query import GenerateQueryResult
from robokudo.cas import CASViews
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator

from robokudo.annotators.gpsr.filter_humans_by_object_annotator import (
    FilterHumansByObjectAnnotator,
)
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.furniture_direction import FurnitureDirection

if TYPE_CHECKING:
    from pathlib import Path


# %% furniture
class Furniture(StrEnum):
    """
    Object classes supported as locations for person queries.
    """

    CHAIR = "chair"
    COUCH = "couch"
    BED = "bed"
    DINING_TABLE = "dining table"
    TOILET = "toilet"
    TV = "tv"
    MICROWAVE = "microwave"
    OVEN = "oven"
    SINK = "sink"
    REFRIGERATOR = "refrigerator"


# %% furniture detector configuration
class YoloFurnitureSetup(robokudo.annotators.core.BaseAnnotator):
    """
    Configure furniture detection and person filtering from the query.
    """

    def __init__(
        self,
        yolo_params: YoloAnnotator.Descriptor.Parameters,
        filter_params: FilterHumansByObjectAnnotator.Descriptor.Parameters,
    ) -> None:
        """
        :param yolo_params: Configuration of the furniture detector.
        :param filter_params: Configuration of the person filter.
        """
        super().__init__(name="YoloFurnitureSetup")
        self.yolo_params = yolo_params
        """
        Configuration of the furniture detector.
        """
        self.filter_params = filter_params
        """
        Configuration of the person filter.
        """
        # fallback when the query carries no direction
        self.default_direction = filter_params.direction
        """
        Direction used when a query does not specify one.
        """

    # TODO: find a way, to make it cleaner/ better
    # Maybe map the Yolo stuff via an annotator and the location in used annotator not here!
    def update(self) -> py_trees.common.Status:
        cas = self.get_cas()
        if cas.contains(CASViews.QUERY):
            query = cas.get(CASViews.QUERY)
            location = query.obj.location
            if location in Furniture:
                # CRAM's YoloAnnotator has no positive class filter, so every other class is ignored instead
                class_names = json.loads(
                    self.yolo_params.file_in_ros_package(
                        self.yolo_params.class_names_path
                    ).read_text()
                )
                self.yolo_params.ignored_class_names = [
                    name for name in class_names.values() if name != location
                ]
            if location:
                self.filter_params.target_type = location
            # optional direction from the query extracted under Attribute
            direction = next(
                (
                    attribute
                    for attribute in (query.obj.attribute or [])
                    if any(
                        attribute == direction.value for direction in FurnitureDirection
                    )
                ),
                None,
            )
            self.filter_params.direction = (
                direction if direction else self.default_direction
            )
        return py_trees.common.Status.SUCCESS


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Find people near the furniture named in the query.
    """

    def __init__(
        self, *args: object, model_directory: Path | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_directory = model_directory
        """
        Directory containing detector class names, if set.
        """

    def implementation(
        self,
        name: str | None = None,
        direction: str | None = None,
        box_x: float | None = None,
        box_y: float | None = None,
        box_z: float | None = None,
        *args: object,
        **kwargs: object,
    ) -> py_trees.composites.Sequence:
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.precision_mode = True
        yolo_descriptor.parameters.ros_package_name = "robokudo_suturo"
        yolo_descriptor.parameters.data_directory = self.model_directory
        yolo_descriptor.parameters.pretrained_model = "yolov8n.pt"
        yolo_descriptor.parameters.class_names_path = (
            "data/json/id2name_yolov8Model.json"
        )
        yolo_descriptor.parameters.threshold = 0.6

        filter_descriptor = FilterHumansByObjectAnnotator.Descriptor()
        if direction is not None:
            filter_descriptor.parameters.direction = direction
        if box_x is not None:
            filter_descriptor.parameters.box_x = box_x
        if box_y is not None:
            filter_descriptor.parameters.box_y = box_y
        if box_z is not None:
            filter_descriptor.parameters.box_z = box_z

        # CRAM's HumanAndPoseAnnotator works without depth by default, the person's position needs it
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True

        seq = py_trees.composites.Sequence("subtree_human_near_object", memory=True)
        seq.add_children(
            [
                YoloFurnitureSetup(
                    yolo_descriptor.parameters, filter_descriptor.parameters
                ),
                YoloAnnotator(descriptor=yolo_descriptor),
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                ClusterPositionAnnotator(),
                FilterHumansByObjectAnnotator(descriptor=filter_descriptor),
                BeliefStateAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return seq
