from __future__ import annotations

# %% Imports
from enum import StrEnum
import os
from pathlib import Path

from py_trees.composites import Sequence
from typing_extensions import TYPE_CHECKING

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.gpsr.gpsr_results import QueryType
from robokudo.annotators.gpsr.sitting import HumanPosture
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import QueryAnnotator
from robokudo.descriptors.analysis_engines.gpsr.subtree_belief_lookup import (
    Subtree as Subtree_BeliefLookup,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_door_open_vocabulary_pipeline import (
    Subtree as Subtree_DoorOpenVocabularyPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_door_pipeline import (
    Subtree as Subtree_DoorPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_faces import (
    Subtree as Subtree_FacesPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_activity_pipeline import (
    Subtree as Subtree_Human_Activity,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_attribute_pipeline import (
    Subtree as Subtree_HumanAttribute,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_in_room import (
    Subtree as Subtree_HumanInRoomPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    Furniture,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    Subtree as Subtree_HumanNearObjectPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_posture_pipeline import (
    Subtree as Subtree_HumanPosturePipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_obj_in_room import (
    Subtree as Subtree_ObjInRoomPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection_pipeline import (
    Subtree as Subtree_ObjectDetectionPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_open_vocabulary_pipeline import (
    Subtree as Subtree_OpenVocabularyPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_pointing_pipeline import (
    Subtree as Subtree_PointingPipeline,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_waving_raising_pipeline import (
    Subtree as Subtree_WavingRaisingPipeline,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    from robokudo_msgs.action import Query


# %% query routes
class ModelDirectoryVariable(StrEnum):
    """
    Environment variable for the shared model checkout.
    """

    DIRECTORY = "GPSR_MODELS_DIRECTORY"


class GpsrRoute(StrEnum):
    """
    Routes available to the GPSR query scheduler.
    """

    WAVING_RAISING = "WavingRaisingPipeline"
    HUMAN_POSTURE = "HumanPosturePipeline"
    OBJECT_DETECTION = "ObjectDetectionPipeline"
    POINTING = "PointingPipeline"
    OPEN_VOCABULARY = "OpenVocabularyPipeline"
    OPEN_VOCABULARY_IN_ROOM = "OpenVocabularyPipelineWithRegion"
    HUMAN_ACTIVITY_FILTERED = "HumanActivityFiltered"
    HUMAN_ACTIVITY = "HumanActivity"
    HUMAN_ATTRIBUTE = "Attribute"
    HUMAN_NEAR_OBJECT = "HumanNearObjectPipeline"
    HUMAN_IN_ROOM = "HumanInRoomPipeline"
    HUMAN_ACTIVITY_IN_ROOM = "HumanActivityInRoomPipeline"
    OBJECT_IN_ROOM = "ObjInRoomPipeline"
    FACE = "FacePipeline"
    BELIEF_LOOKUP = "BeliefStateLookupPipeline"
    DOOR = "DoorPipeline"
    DOOR_OPEN_VOCABULARY = "DoorOpenVocabularyPipeline"


# %% GPSR analysis engine


class AnalysisEngine(AnalysisEngineInterface):
    """
    The GPSR perception: it reads the camera and the queries and runs, for every query,
    the subtree that answers that kind of query.

    The route of a query follows from its type and, for persons and objects, from its
    attribute, color, description and location.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_directory = Path(
            os.environ.get(
                ModelDirectoryVariable.DIRECTORY,
                Path.home() / "robokudo_suturo_weights",
            )
        )
        """
        Directory containing the shared perception models and mappings.
        """
        self.cr: CollectionReaderAnnotator | None = None
        """
        The collection reader that reads the camera, created when the tree is built.
        """

    def name(self) -> str:
        return "gpsr"

    def configure_collection_reader(self) -> None:
        """
        Create the collection reader for the camera configuration ``hsr_ros2``.
        """
        self.cr = CollectionReaderAnnotator(
            descriptor=CollectionReaderDescriptorFactory.create_descriptor("hsr_ros2")
        )

    @catch_and_raise_to_blackboard
    def implementation(self) -> Pipeline:
        self.configure_collection_reader()

        # The actual attributes for GPSR can be found in the CommandGenerator. See
        # https://github.com/RoboCupAtHome/CommandGenerator/tree/master for details. Check the rulebook to see
        # if the repository is the current correct one for GPSR.
        waving_raising_attributes = ["waving", "raising right arm", "raising left arm"]
        human_posture_attributes = [posture.value for posture in HumanPosture]
        pointing_attributes = ["pointing right", "pointing left"]
        clothes_attributes = ["t shirt", "shirt", "blouse", "sweater", "coat", "jacket"]
        color_attributes = ["blue", "yellow", "black", "white", "red", "orange", "gray"]
        object_attributes = [
            "biggest",
            "largest",
            "smallest",
            "heaviest",
            "lightest",
            "thinnest",
        ]
        rooms = ["iai_kitchen", "living_room", "bedroom", "office", "hallway"]

        def filter_query_input(values: list[str]) -> str | None:
            """
            :param values: The values of one field of the query.
            :return: The only value of the field, or *None* if it holds none or
                several.
            """
            if len(values) == 1:
                return values[0]
            return None

        def task_selection(query: Query.Goal) -> str | None:
            """
            :return: The name of the route that answers the query, or *None* if no
                route does.
            """
            if not query.obj.type:
                return None

            attribute = filter_query_input(query.obj.attribute)
            color = filter_query_input(query.obj.color)
            description = filter_query_input(query.obj.description)

            # TODO: Add a human query case that applies filtering.
            if query.obj.type == QueryType.HUMAN:
                if color in color_attributes and description in clothes_attributes:
                    if query.obj.location in rooms:
                        return GpsrRoute.OPEN_VOCABULARY_IN_ROOM.value
                    return GpsrRoute.OPEN_VOCABULARY.value
                if query.obj.location in rooms:
                    if (
                        attribute
                        in waving_raising_attributes
                        + human_posture_attributes
                        + pointing_attributes
                    ):
                        return GpsrRoute.HUMAN_ACTIVITY_IN_ROOM.value
                    return GpsrRoute.HUMAN_IN_ROOM.value
                if query.obj.location in Furniture:
                    return GpsrRoute.HUMAN_NEAR_OBJECT.value
                if attribute in waving_raising_attributes:
                    return GpsrRoute.WAVING_RAISING.value
                if attribute in human_posture_attributes:
                    return GpsrRoute.HUMAN_POSTURE.value
                if attribute in pointing_attributes:
                    return GpsrRoute.POINTING.value
                if attribute == "Attribute":
                    return GpsrRoute.HUMAN_ATTRIBUTE.value
                if len(query.obj.attribute) >= 2:
                    return GpsrRoute.HUMAN_ACTIVITY_FILTERED.value
                return GpsrRoute.HUMAN_ACTIVITY.value
            if query.obj.type == QueryType.RECALL:
                return GpsrRoute.BELIEF_LOOKUP.value
            if query.obj.type == QueryType.FACE:
                return GpsrRoute.FACE.value
            if query.obj.type == QueryType.DOOR:
                return GpsrRoute.DOOR.value
            if query.obj.type == QueryType.DOOR_OPEN_VOCABULARY:
                return GpsrRoute.DOOR_OPEN_VOCABULARY.value
            if query.obj.type == QueryType.OBJECT:
                if query.obj.location in rooms:
                    return GpsrRoute.OBJECT_IN_ROOM.value
                if attribute in object_attributes:
                    return GpsrRoute.OPEN_VOCABULARY.value
                if attribute:
                    return None
                return GpsrRoute.OBJECT_DETECTION.value
            return None

        task_trees = {
            GpsrRoute.WAVING_RAISING.value: Subtree_WavingRaisingPipeline().implementation(),
            GpsrRoute.HUMAN_POSTURE.value: Subtree_HumanPosturePipeline().implementation(),
            GpsrRoute.OBJECT_DETECTION.value: Subtree_ObjectDetectionPipeline(
                model_directory=self.model_directory
            ).implementation(),
            GpsrRoute.POINTING.value: Subtree_PointingPipeline().implementation(),
            GpsrRoute.OPEN_VOCABULARY.value: Subtree_OpenVocabularyPipeline(
                object_attributes=object_attributes,
                color_attributes=color_attributes,
                clothes_attributes=clothes_attributes,
            ).implementation(),
            GpsrRoute.OPEN_VOCABULARY_IN_ROOM.value: Subtree_OpenVocabularyPipeline(
                object_attributes=object_attributes,
                color_attributes=color_attributes,
                clothes_attributes=clothes_attributes,
                with_region_filter=True,
            ).implementation(),
            GpsrRoute.HUMAN_ACTIVITY_FILTERED.value: Subtree_Human_Activity(
                with_filter=True
            ).implementation(),
            GpsrRoute.HUMAN_ACTIVITY.value: Subtree_Human_Activity().implementation(),
            GpsrRoute.HUMAN_ATTRIBUTE.value: Subtree_HumanAttribute(
                model_directory=self.model_directory
            ).implementation(),
            GpsrRoute.HUMAN_NEAR_OBJECT.value: Subtree_HumanNearObjectPipeline(
                model_directory=self.model_directory
            ).implementation(),
            GpsrRoute.HUMAN_IN_ROOM.value: Subtree_HumanInRoomPipeline().implementation(),
            GpsrRoute.HUMAN_ACTIVITY_IN_ROOM.value: Subtree_HumanInRoomPipeline(
                with_activity=True
            ).implementation(),
            GpsrRoute.OBJECT_IN_ROOM.value: Subtree_ObjInRoomPipeline(
                model_directory=self.model_directory
            ).implementation(),
            GpsrRoute.FACE.value: Subtree_FacesPipeline().implementation(),
            GpsrRoute.BELIEF_LOOKUP.value: Subtree_BeliefLookup().implementation(),
            GpsrRoute.DOOR.value: Subtree_DoorPipeline(
                model_directory=self.model_directory
            ).implementation(),
            GpsrRoute.DOOR_OPEN_VOCABULARY.value: Subtree_DoorOpenVocabularyPipeline().implementation(),
        }

        task_scheduling = Sequence("Task Scheduling", memory=True)
        task_scheduling.add_child(
            QueryBasedScheduler(tasks=task_trees, filter_fn=task_selection)
        )

        gpsr_pipeline = Pipeline("GPSR")
        gpsr_pipeline.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                self.cr,
                ImagePreprocessorAnnotator(),
                task_scheduling,
            ]
        )
        return gpsr_pipeline
