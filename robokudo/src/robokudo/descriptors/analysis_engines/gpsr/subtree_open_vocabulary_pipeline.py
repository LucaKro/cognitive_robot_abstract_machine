"""
The GPSR subtree that finds persons by the color and kind of their clothes, and objects
by an attribute, with open-vocabulary detection.
"""

from __future__ import annotations

# %% Imports
from py_trees.composites import Sequence

from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.filter_objects_by_region import FilterObjectsByRegion
from robokudo.annotators.gpsr.gpsr_results import (
    GenerateOpenVocabularyResult,
    QueryType,
)
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from robokudo_msgs.action import Query

# %% subtree


class Subtree(SubtreeInterface):
    """
    Detects the clothes a query describes on persons, or objects with the attribute a
    query names, and answers with the matching persons or objects.

    The detector searches for the text ``"<color> <clothes>"`` for persons and
    ``"<attribute> object"`` for objects.
    """

    def __init__(
        self,
        object_attributes: list[str],
        color_attributes: list[str],
        clothes_attributes: list[str],
        with_room_filter: bool = False,
        object_detection_threshold: float = 0.1,
        clothes_detection_threshold: float = 0.3,
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        :param object_attributes: The object attributes the subtree searches for.
        :param color_attributes: The clothes colors the subtree searches for.
        :param clothes_attributes: The kinds of clothes the subtree searches for.
        :param with_room_filter: Whether only persons in the queried room are answered.
        :param object_detection_threshold: Smallest detector score of an object.
        :param clothes_detection_threshold: Smallest detector score of a piece of
            clothing.
        """
        super().__init__(*args, **kwargs)
        self.object_attributes = object_attributes
        """
        The object attributes the subtree searches for.
        """
        self.color_attributes = color_attributes
        """
        The clothes colors the subtree searches for.
        """
        self.clothes_attributes = clothes_attributes
        """
        The kinds of clothes the subtree searches for.
        """
        self.with_room_filter = with_room_filter
        """
        Whether only persons in the queried room are answered.
        """
        self.object_detection_threshold = object_detection_threshold
        """
        Smallest detector score of an object.
        """
        self.clothes_detection_threshold = clothes_detection_threshold
        """
        Smallest detector score of a piece of clothing.
        """

    def implementation(self, *args: object, **kwargs: object) -> Sequence:
        """
        :return: The sequence that searches for what the query describes and answers with it.
        """
        detector = OpenVocabularyObjectDetectionAnnotator(
            descriptor=OpenVocabularyObjectDetectionAnnotator.Descriptor()
        )

        def select_branch(query: Query.Goal) -> str:
            """
            Let the detector search for what the query describes.

            :param query: The query to prepare the detector for.
            :return: The branch that prepares the detection for the queried type.
            """
            if query.obj.type == QueryType.OBJECT:
                if query.obj.attribute[0] in self.object_attributes:
                    detector.classes = [f"{query.obj.attribute[0]} {query.obj.type}"]
                    detector.descriptor.parameters.detection_threshold = (
                        self.object_detection_threshold
                    )
                return "ObjectPipeline"
            if query.obj.type == QueryType.HUMAN:
                if (
                    query.obj.color[0] in self.color_attributes
                    and query.obj.description[0] in self.clothes_attributes
                ):
                    detector.classes = [
                        f"{query.obj.color[0]} {query.obj.description[0]}"
                    ]
                    detector.descriptor.parameters.detection_threshold = (
                        self.clothes_detection_threshold
                    )
                return "HumanPipeline"
            return "InvalidQuery"

        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        human_detection = Sequence("Human Detection", memory=True)
        human_detection.add_child(HumanAndPoseAnnotator(descriptor=human_descriptor))
        branches = {
            "ObjectPipeline": Sequence("Object Detection", memory=True),
            "HumanPipeline": human_detection,
        }

        sequence = Sequence("subtree_open_vocabulary_pipeline", memory=True)
        sequence.add_children(
            [
                QueryBasedScheduler(tasks=branches, filter_fn=select_branch),
                detector,
                ClusterPositionAnnotator(),
            ]
        )
        if self.with_room_filter:
            sequence.add_child(
                FilterObjectsByRegion(descriptor=FilterObjectsByRegion.Descriptor())
            )
        sequence.add_children(
            [
                BeliefStateAnnotator(),
                GenerateOpenVocabularyResult(with_room_filter=self.with_room_filter),
            ]
        )
        return sequence
