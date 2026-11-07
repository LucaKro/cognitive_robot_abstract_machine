"""
The GPSR perception analysis engine and the routes of its queries.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum

from py_trees.composites import Sequence
from typing_extensions import TYPE_CHECKING

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.gpsr.gpsr_results import QueryType
from robokudo.annotators.gpsr.sitting import HumanPosture
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import QueryAnnotator
from robokudo.descriptors.analysis_engines.gpsr.subtree_belief_lookup import (
    Subtree as BeliefLookupSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_door_open_vocabulary_pipeline import (
    Subtree as DoorOpenVocabularySubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_door_pipeline import (
    Subtree as DoorSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_faces import (
    Subtree as FaceSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_activity_pipeline import (
    Subtree as HumanActivitySubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_attribute_pipeline import (
    Subtree as HumanAttributeSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_in_room import (
    Subtree as HumanInRoomSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    Furniture,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    Subtree as HumanNearObjectSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_posture_pipeline import (
    Subtree as HumanPostureSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_in_room import (
    Subtree as ObjectInRoomSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection_pipeline import (
    Subtree as ObjectDetectionPipelineSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_open_vocabulary_pipeline import (
    Subtree as OpenVocabularySubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_pointing_pipeline import (
    Subtree as PointingSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_waving_raising_pipeline import (
    Subtree as WavingRaisingSubtree,
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
class GpsrRoute(StrEnum):
    """
    Routes available to the GPSR query scheduler.
    """

    WAVING_RAISING = "WavingRaisingPipeline"
    """
    Persons waving or raising an arm.
    """
    HUMAN_POSTURE = "HumanPosturePipeline"
    """
    Persons in the queried posture.
    """
    OBJECT_DETECTION = "ObjectDetectionPipeline"
    """
    Objects found by the trained detector.
    """
    POINTING = "PointingPipeline"
    """
    Persons pointing in the queried direction.
    """
    OPEN_VOCABULARY = "OpenVocabularyPipeline"
    """
    Clothes and objects found by text.
    """
    OPEN_VOCABULARY_IN_ROOM = "OpenVocabularyPipelineWithRegion"
    """
    Clothes and objects found by text in the queried room.
    """
    HUMAN_ACTIVITY_FILTERED = "HumanActivityFiltered"
    """
    Persons with the queried activity.
    """
    HUMAN_ACTIVITY = "HumanActivity"
    """
    The activities of all persons.
    """
    HUMAN_ATTRIBUTE = "Attribute"
    """
    Persons described by the vision language model.
    """
    HUMAN_NEAR_OBJECT = "HumanNearObjectPipeline"
    """
    Persons near the queried furniture.
    """
    HUMAN_IN_ROOM = "HumanInRoomPipeline"
    """
    Persons in the queried room.
    """
    HUMAN_ACTIVITY_IN_ROOM = "HumanActivityInRoomPipeline"
    """
    Persons with the queried activity in the queried room.
    """
    OBJECT_IN_ROOM = "ObjectInRoomPipeline"
    """
    Objects in the queried room.
    """
    FACE = "FacePipeline"
    """
    Faces of persons.
    """
    BELIEF_LOOKUP = "BeliefStateLookupPipeline"
    """
    Remembered persons and objects.
    """
    DOOR = "DoorPipeline"
    """
    Doors found by the trained detector.
    """
    DOOR_OPEN_VOCABULARY = "DoorOpenVocabularyPipeline"
    """
    Doors found by text.
    """


# %% GPSR analysis engine


class AnalysisEngine(AnalysisEngineInterface):
    """
    The GPSR perception: it reads the camera and the queries. For every query, it runs
    the subtree that answers that kind of query.

    The route of a query follows from its type. For persons and objects it also depends
    on attribute, color, description and location.
    """

    def __init__(self) -> None:
        """
        Start without a collection reader, which is created when the tree is built.
        """
        super().__init__()
        self.collection_reader: CollectionReaderAnnotator | None = None
        """
        The collection reader that reads the camera, created when the tree is built.
        """

    def name(self) -> str:
        """
        :return: The name of this analysis engine.
        """
        return "gpsr"

    def configure_collection_reader(self) -> None:
        """
        Create the collection reader for the camera configuration ``hsr_ros2``.
        """
        self.collection_reader = CollectionReaderAnnotator(
            descriptor=CollectionReaderDescriptorFactory.create_descriptor("hsr_ros2")
        )

    @catch_and_raise_to_blackboard
    def implementation(self) -> Pipeline:
        """
        :return: The pipeline that reads the camera and runs the route of every query.
        """
        self.configure_collection_reader()

        # The GPSR attributes come from the CommandGenerator. The rulebook names its
        # current repository, now https://github.com/RoboCupAtHome/CommandGenerator.
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

        def get_single_value(values: list[str]) -> str | None:
            """
            :param values: The values of one field of the query.
            :return: The only value of the field, or *None* if it holds none or
                several.
            """
            if len(values) == 1:
                return values[0]
            return None

        def select_route(query: Query.Goal) -> str | None:
            """
            :param query: The query to route.
            :return: The name of the route that answers the query, or *None* if no route
                does.
            """
            if not query.obj.type:
                return None

            attribute = get_single_value(query.obj.attribute)
            color = get_single_value(query.obj.color)
            description = get_single_value(query.obj.description)

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
            GpsrRoute.WAVING_RAISING.value: WavingRaisingSubtree().implementation(),
            GpsrRoute.HUMAN_POSTURE.value: HumanPostureSubtree().implementation(),
            GpsrRoute.OBJECT_DETECTION.value: ObjectDetectionPipelineSubtree().implementation(),
            GpsrRoute.POINTING.value: PointingSubtree().implementation(),
            GpsrRoute.OPEN_VOCABULARY.value: OpenVocabularySubtree(
                object_attributes=object_attributes,
                color_attributes=color_attributes,
                clothes_attributes=clothes_attributes,
            ).implementation(),
            GpsrRoute.OPEN_VOCABULARY_IN_ROOM.value: OpenVocabularySubtree(
                object_attributes=object_attributes,
                color_attributes=color_attributes,
                clothes_attributes=clothes_attributes,
                with_room_filter=True,
            ).implementation(),
            GpsrRoute.HUMAN_ACTIVITY_FILTERED.value: HumanActivitySubtree(
                with_filter=True
            ).implementation(),
            GpsrRoute.HUMAN_ACTIVITY.value: HumanActivitySubtree().implementation(),
            GpsrRoute.HUMAN_ATTRIBUTE.value: HumanAttributeSubtree().implementation(),
            GpsrRoute.HUMAN_NEAR_OBJECT.value: HumanNearObjectSubtree().implementation(),
            GpsrRoute.HUMAN_IN_ROOM.value: HumanInRoomSubtree().implementation(),
            GpsrRoute.HUMAN_ACTIVITY_IN_ROOM.value: HumanInRoomSubtree(
                with_activity=True
            ).implementation(),
            GpsrRoute.OBJECT_IN_ROOM.value: ObjectInRoomSubtree().implementation(),
            GpsrRoute.FACE.value: FaceSubtree().implementation(),
            GpsrRoute.BELIEF_LOOKUP.value: BeliefLookupSubtree().implementation(),
            GpsrRoute.DOOR.value: DoorSubtree().implementation(),
            GpsrRoute.DOOR_OPEN_VOCABULARY.value: DoorOpenVocabularySubtree().implementation(),
        }

        task_scheduling = Sequence("Task Scheduling", memory=True)
        task_scheduling.add_child(
            QueryBasedScheduler(tasks=task_trees, filter_fn=select_route)
        )

        gpsr_pipeline = Pipeline("GPSR")
        gpsr_pipeline.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                self.collection_reader,
                ImagePreprocessorAnnotator(),
                task_scheduling,
            ]
        )
        return gpsr_pipeline
