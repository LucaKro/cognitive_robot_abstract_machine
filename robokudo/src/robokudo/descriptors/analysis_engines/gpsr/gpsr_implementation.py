from __future__ import annotations

# %% Imports
from py_trees.composites import Sequence
from typing_extensions import TYPE_CHECKING

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
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
    FURNITURE,
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
        human_posture_attributes = ["sitting", "standing", "laying"]
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
            if len(values) > 1:
                return None
            elif len(values) == 1:
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

            # TODO query.obj.type = human base case so anpassen, dass wir noch ein fall haben, dass er filtern soll
            if query.obj.type == "human":
                if color in color_attributes and description in clothes_attributes:
                    if query.obj.location in rooms:
                        return "OpenVocabularyPipelineWithRegion"
                    return "OpenVocabularyPipeline"
                elif query.obj.location in rooms:
                    if (
                        attribute
                        in waving_raising_attributes
                        + human_posture_attributes
                        + pointing_attributes
                    ):
                        return "HumanActivityInRoomPipeline"
                    return "HumanInRoomPipeline"
                elif query.obj.location in FURNITURE:
                    return "HumanNearObjectPipeline"
                elif attribute in waving_raising_attributes:
                    return "WavingRaisingPipeline"
                elif attribute in human_posture_attributes:
                    return "HumanPosturePipeline"
                elif attribute in pointing_attributes:
                    return "PointingPipeline"
                elif attribute == "Attribute":
                    return "Attribute"
                elif len(query.obj.attribute) >= 2:
                    return "HumanActivityFiltered"
                else:
                    return "HumanActivity"
            elif query.obj.type == "recall":
                return "BeliefStateLookupPipeline"
            elif query.obj.type == "face":
                return "FacePipeline"
            elif query.obj.type == "door":
                return "DoorPipeline"
            elif query.obj.type == "doorgsam":
                return "DoorOpenVocabularyPipeline"
            elif query.obj.type == "object":
                if query.obj.location in rooms:
                    return "ObjInRoomPipeline"
                elif attribute in object_attributes:
                    return "OpenVocabularyPipeline"
                elif attribute and attribute not in object_attributes:
                    return None
                else:
                    return "ObjectDetectionPipeline"
            else:
                return None

        task_trees = {
            "WavingRaisingPipeline": Subtree_WavingRaisingPipeline().implementation(),
            "HumanPosturePipeline": Subtree_HumanPosturePipeline().implementation(),
            "ObjectDetectionPipeline": Subtree_ObjectDetectionPipeline().implementation(),
            "PointingPipeline": Subtree_PointingPipeline().implementation(),
            "OpenVocabularyPipeline": Subtree_OpenVocabularyPipeline(
                object_attributes=object_attributes,
                color_attributes=color_attributes,
                clothes_attributes=clothes_attributes,
            ).implementation(),
            "OpenVocabularyPipelineWithRegion": Subtree_OpenVocabularyPipeline(
                object_attributes=object_attributes,
                color_attributes=color_attributes,
                clothes_attributes=clothes_attributes,
                with_region_filter=True,
            ).implementation(),
            "HumanActivityFiltered": Subtree_Human_Activity(
                with_filter=True
            ).implementation(),
            "HumanActivity": Subtree_Human_Activity().implementation(),
            "Attribute": Subtree_HumanAttribute().implementation(),
            "HumanNearObjectPipeline": Subtree_HumanNearObjectPipeline().implementation(),
            "HumanInRoomPipeline": Subtree_HumanInRoomPipeline().implementation(),
            "HumanActivityInRoomPipeline": Subtree_HumanInRoomPipeline(
                with_activity=True
            ).implementation(),
            "ObjInRoomPipeline": Subtree_ObjInRoomPipeline().implementation(),
            "FacePipeline": Subtree_FacesPipeline(
                ros_pkg_path="robokudo_suturo"
            ).implementation(),
            "BeliefStateLookupPipeline": Subtree_BeliefLookup().implementation(),
            "DoorPipeline": Subtree_DoorPipeline().implementation(),
            "DoorOpenVocabularyPipeline": Subtree_DoorOpenVocabularyPipeline().implementation(),
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
