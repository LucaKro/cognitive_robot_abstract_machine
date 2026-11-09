"""
The GPSR subtree that finds the persons near the queried furniture.
"""

from __future__ import annotations

# %% Imports
import json
from enum import StrEnum

import py_trees
import robokudo.analysis_engine
import robokudo.annotators.core
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


# %% furniture
class Furniture(StrEnum):
    """
    Object classes supported as locations for person queries.
    """

    CHAIR = "chair"
    """
    The detector class of chairs.
    """
    COUCH = "couch"
    """
    The detector class of couches.
    """
    BED = "bed"
    """
    The detector class of beds.
    """
    DINING_TABLE = "dining table"
    """
    The detector class of dining tables.
    """
    TOILET = "toilet"
    """
    The detector class of toilets.
    """
    TELEVISION = "tv"
    """
    The detector class of televisions.
    """
    MICROWAVE = "microwave"
    """
    The detector class of microwaves.
    """
    OVEN = "oven"
    """
    The detector class of ovens.
    """
    SINK = "sink"
    """
    The detector class of sinks.
    """
    REFRIGERATOR = "refrigerator"
    """
    The detector class of refrigerators.
    """


# %% furniture detector configuration
class YoloFurnitureSetup(robokudo.annotators.core.BaseAnnotator):
    """
    Configures furniture detection and person filtering from the query.
    """

    def __init__(
        self,
        yolo_parameters: YoloAnnotator.Descriptor.Parameters,
        filter_parameters: FilterHumansByObjectAnnotator.Descriptor.Parameters,
    ) -> None:
        """
        :param yolo_parameters: Configuration of the furniture detector.
        :param filter_parameters: Configuration of the person filter.
        """
        super().__init__(name="YoloFurnitureSetup")
        self.yolo_parameters = yolo_parameters
        """
        Configuration of the furniture detector.
        """
        self.filter_parameters = filter_parameters
        """
        Configuration of the person filter.
        """
        self.default_direction = filter_parameters.direction
        """
        Direction used when a query does not specify one.
        """

    # TODO: Move the mapping from the query to the detector into the annotators it configures.
    def update(self) -> py_trees.common.Status:
        """
        Point the detector and the person filter at the queried furniture and direction.

        :return: :attr:`Status.SUCCESS`.
        """
        cas = self.get_cas()
        if cas.contains(CASViews.QUERY):
            query = cas.get(CASViews.QUERY)
            location = query.obj.location
            if location in Furniture:
                class_names = json.loads(
                    self.yolo_parameters.file_in_ros_package(
                        self.yolo_parameters.class_names_path
                    ).read_text()
                )
                self.yolo_parameters.ignored_class_names = [
                    name for name in class_names.values() if name != location
                ]
            if location:
                self.filter_parameters.target_type = location
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
            self.filter_parameters.direction = (
                direction if direction else self.default_direction
            )
        return py_trees.common.Status.SUCCESS


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Finds persons near the furniture named in the query.
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
        """
        :param name: Name of this subtree.
        :param direction: Side of the furniture to search, if set.
        :param box_x: Extent of the search box along x in meters, if set.
        :param box_y: Extent of the search box along y in meters, if set.
        :param box_z: Height of the search box in meters, if set.
        :return: The sequence that answers with the persons near the queried furniture.
        """
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.precision_mode = True
        yolo_descriptor.parameters.ros_package_name = "robokudo_suturo"
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

        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True

        sequence = py_trees.composites.Sequence(
            "subtree_human_near_object", memory=True
        )
        sequence.add_children(
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
        return sequence
