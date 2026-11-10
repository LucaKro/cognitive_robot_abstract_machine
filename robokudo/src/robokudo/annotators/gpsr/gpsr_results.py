"""
Answers of the GPSR perception queries.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum

from builtin_interfaces.msg import Time
from py_trees.common import Status
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from std_msgs.msg import Header

from typing_extensions import Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefEntry
from robokudo.annotators.gpsr.exceptions import MissingObjectClassification
from robokudo.annotators.gpsr.human_activity_annotator import (
    HumanActivityAnnotator,
    HumanGesture,
)
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator
from robokudo.annotators.gpsr.vision_language_annotator import VisionLanguageAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import (
    Classification,
    LocationAnnotation,
    PositionAnnotation,
)
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.query import QueryHandler
from robokudo.utils.type_conversion import (
    get_geometry_msgs_pose_stamped_from_position_annotation,
)


# %% query types
class QueryType(StrEnum):
    """
    Types accepted by GPSR perception queries.
    """

    HUMAN = "human"
    """
    A query about persons.
    """
    HUMANS = "humans"
    """
    A query about several persons.
    """
    OBJECT = "object"
    """
    A query about objects.
    """
    RECALL = "recall"
    """
    A query about remembered persons and objects.
    """
    FACE = "face"
    """
    A query about faces.
    """
    DOOR = "door"
    """
    A query about doors, answered with the trained door detector.
    """
    DOOR_OPEN_VOCABULARY = "door_open_vocabulary"
    """
    A query about doors, answered with open vocabulary detection.
    """


# %% camera header
def create_camera_header(cas: CAS) -> Header:
    """
    :param cas: The CAS holding the camera observation.
    :return: The header of the current camera image: its acquisition time and the frame
        of the camera.
    """
    observation = cas.require_camera_observation()
    return Header(
        frame_id=str(observation.camera.root.name),
        stamp=Time(
            sec=observation.timestamp_nanoseconds // 1_000_000_000,
            nanosec=observation.timestamp_nanoseconds % 1_000_000_000,
        ),
    )


# %% person answers
class GenerateResult(BaseAnnotator):
    """
    Builds answers for detected hypotheses.
    """

    def add_identity(
        self,
        object_designator: ObjectDesignator,
        human_hypothesis: HumanHypothesis,
    ) -> None:
        """
        Add the person's belief identifier to the answer when available.

        :param object_designator: The answer that receives the identifier.
        :param human_hypothesis: The person of the answer.
        """
        belief_entries = self.get_cas().filter_by_type(
            BeliefEntry, human_hypothesis.annotations
        )
        if belief_entries:
            object_designator.uid = str(belief_entries[0].id)

    def add_positions(
        self, object_designator: ObjectDesignator, positions: list[PositionAnnotation]
    ) -> None:
        """
        Add observed positions and their sources to the answer.

        :param object_designator: The answer that receives the positions.
        :param positions: The observed positions.
        """
        for position in positions:
            header = create_camera_header(self.get_cas())
            pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(
                position, header
            )
            object_designator.pose.append(pose_stamped)
            object_designator.pose_source.append(position.source)


# %% posture answers
class GenerateHumanPostureResult(GenerateResult):
    """
    Answers queries for persons whose unambiguous posture matches the query.
    """

    def __init__(
        self, name: str = "GenerateHumanPostureResult", *args: Any, **kwargs: Any
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        """
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to answer with every person whose only
        posture is the queried one.

        :return: :attr:`Status.SUCCESS`.
        """
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        keypoint_rule_sources = {
            HumanSittingAnnotator.__name__,
            HumanStandingAnnotator.__name__,
            HumanLyingAnnotator.__name__,
        }

        for human_hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, human_hypothesis.annotations
            )
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, human_hypothesis.annotations
            )

            # a person's posture is known when exactly one keypoint rule fires
            keypoint_postures = {
                activity.type
                for activity in activities
                if activity.source in keypoint_rule_sources
            }
            if len(keypoint_postures) != 1:
                self.rk_logger.info(
                    f"Posture of human hypothesis {human_hypothesis.id}: undecided, rules fired: {sorted(keypoint_postures)}"
                )
                continue
            posture = keypoint_postures.pop()
            if posture == HumanLyingAnnotator.posture:
                posture = HumanLyingAnnotator.query_word
            self.rk_logger.info(
                f"Posture of human hypothesis {human_hypothesis.id}: {posture}"
            )
            if posture == query.obj.attribute[0]:
                object_designator.attribute.append(posture)

            if object_designator.attribute:
                self.add_identity(object_designator, human_hypothesis)
                self.add_positions(object_designator, positions)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = "Query result generated."
        return Status.SUCCESS


# %% gesture answers
class GenerateWavingRaisingResult(GenerateResult):
    """
    Answers queries for matching waving and arm raising activities.
    """

    def __init__(
        self, name: str = "ObjectDetectionResult", *args: Any, **kwargs: Any
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        """
        super().__init__(*args, **kwargs)

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to answer with every person showing the
        queried waving or arm raising.

        :return: :attr:`Status.SUCCESS`.
        """
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for human_hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, human_hypothesis.annotations
            )
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, human_hypothesis.annotations
            )

            for activity in activities:
                if activity.source == HumanWavingAnnotator.__name__:
                    if activity.type == query.obj.attribute[0]:
                        object_designator.description.append(activity.type)

            if object_designator.description:
                self.add_identity(object_designator, human_hypothesis)
                self.add_positions(object_designator, positions)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS


# %% pointing answers
class GeneratePointingResult(GenerateResult):
    """
    Answers queries for persons pointing in the requested direction.
    """

    def __init__(self, name: str = "PointingResult", *args: Any, **kwargs: Any) -> None:
        """
        :param name: Name of this annotator in the tree.
        """
        super().__init__(*args, **kwargs)

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to answer with every person pointing in
        the queried direction.

        :return: :attr:`Status.SUCCESS`.
        """
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for human_hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, human_hypothesis.annotations
            )
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, human_hypothesis.annotations
            )

            pointing_directions = [
                activity.type
                for activity in activities
                if activity.source == HumanPointingAnnotator.__name__
            ]
            for direction in pointing_directions:
                object_designator.description.append(f"Behavior suggests {direction}.")

            if object_designator.description:
                if query.obj.attribute[0] in pointing_directions:
                    self.add_identity(object_designator, human_hypothesis)
                    self.add_positions(object_designator, positions)
                else:
                    continue

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS


# %% activity answers
class GenerateHumanActivityResult(GenerateResult):
    """
    Answers queries with observed postures and gestures, optionally filtered by room.
    """

    def __init__(
        self, name: str = "GenerateHumanActivityResult", *args: Any, **kwargs: Any
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        """
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to answer with the postures and gestures
        observed for each person.

        :return: :attr:`Status.SUCCESS`.
        """
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        if query.obj.location:
            human_hypotheses = [
                human_hypothesis
                for human_hypothesis in human_hypotheses
                if any(
                    isinstance(annotation, LocationAnnotation)
                    and annotation.name == query.obj.location
                    for annotation in human_hypothesis.annotations
                )
            ]
        query_result = []
        result = Query.Result()

        for human_hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, human_hypothesis.annotations
            )
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, human_hypothesis.annotations
            )

            pointing_directions = [
                activity.type
                for activity in activities
                if activity.source == HumanPointingAnnotator.__name__
            ]

            for activity in activities:
                if activity.source == HumanActivityAnnotator.__name__:
                    if (
                        activity.type
                        in HumanActivityAnnotator.Descriptor.Parameters().posture_labels
                    ):
                        object_designator.attribute.append(activity.type)
                    elif activity.type == HumanGesture.POINTING and pointing_directions:
                        # a keypoint rule found a concrete direction, the ambiguous CLIP pointing label is dropped
                        continue
                    elif activity.type == HumanGesture.NONE and (
                        pointing_directions
                        or any(
                            activity.source == HumanWavingAnnotator.__name__
                            for activity in activities
                        )
                    ):
                        # a keypoint rule found a gesture, the CLIP no gesture label is dropped
                        continue
                    else:
                        object_designator.description.append(activity.type)
                elif activity.source == HumanPointingAnnotator.__name__:
                    object_designator.description.append(activity.type)
                elif activity.source == HumanWavingAnnotator.__name__:
                    object_designator.description.append(activity.type)

            if object_designator.attribute or object_designator.description:
                self.add_identity(object_designator, human_hypothesis)
                self.add_positions(object_designator, positions)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = "Query result generated."
        return Status.SUCCESS


# %% open vocabulary answers
class GenerateOpenVocabularyResult(GenerateResult):
    """
    Answers a query with the detections of the
    :class:`OpenVocabularyObjectDetectionAnnotator`.

    For persons it answers every person that contains a detection, with the detection's
    class as description. For objects it answers every detection.
    """

    def __init__(
        self,
        name: str = "OpenVocabularyResult",
        with_room_filter: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param with_room_filter: Whether a queried location limits the answered persons.
        """
        super().__init__(*args, **kwargs)
        self.with_room_filter = with_room_filter
        """
        Whether a queried location limits the answered persons.
        """

    def compute_containment_ratio(
        self, inner: ObjectHypothesis, outer: ObjectHypothesis
    ) -> float:
        """
        :param inner: The inner hypothesis (the one that is contained).
        :param outer: The outer hypothesis (the one that contains).
        :return: The share of the inner region that lies inside of the outer region.
        """
        inner_left = inner.roi.roi.pos.x
        inner_right = inner.roi.roi.pos.x + inner.roi.roi.width
        inner_top = inner.roi.roi.pos.y
        inner_bottom = inner.roi.roi.pos.y + inner.roi.roi.height
        outer_left = outer.roi.roi.pos.x
        outer_right = outer.roi.roi.pos.x + outer.roi.roi.width
        outer_top = outer.roi.roi.pos.y
        outer_bottom = outer.roi.roi.pos.y + outer.roi.roi.height

        intersection_left = max(inner_left, outer_left)
        intersection_right = min(inner_right, outer_right)
        intersection_top = max(inner_top, outer_top)
        intersection_bottom = min(inner_bottom, outer_bottom)

        intersection_width = max(0, intersection_right - intersection_left + 1)
        intersection_height = max(0, intersection_bottom - intersection_top + 1)
        intersection_area = intersection_width * intersection_height
        inner_area = (inner_right - inner_left) * (inner_bottom - inner_top)

        self.rk_logger.info(
            f"The outer hypothesis contains the inner hypothesis to {min(intersection_area / inner_area, 100)} of its area"
        )
        return intersection_area / inner_area

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to answer the query with the open
        vocabulary detections it asks for.

        :return: :attr:`Status.FAILURE` for a query that is no person or object query,
            otherwise :attr:`Status.SUCCESS`.
        :raises MissingObjectClassification: If an object has no classification.
        """
        query = self.get_cas().get(CASViews.QUERY)

        if query.obj.type == QueryType.HUMAN or query.obj.type == QueryType.HUMANS:
            human_hypotheses = self.get_cas().filter_annotations_by_type(
                HumanHypothesis
            )
            object_hypotheses = self.get_cas().filter_annotations_by_type(
                ObjectHypothesis
            )

            if self.with_room_filter and query.obj.location:
                human_hypotheses = [
                    human_hypothesis
                    for human_hypothesis in human_hypotheses
                    if any(
                        isinstance(annotation, LocationAnnotation)
                        and annotation.name == query.obj.location
                        for annotation in human_hypothesis.annotations
                    )
                ]

            query_result = []
            result = Query.Result()

            for human_hypothesis in human_hypotheses:
                for object_hypothesis in object_hypotheses:
                    if not isinstance(object_hypothesis, HumanHypothesis):
                        detection_classification = self.get_cas().filter_by_type(
                            Classification, object_hypothesis.annotations
                        )[0]
                        if (
                            detection_classification.source
                            == OpenVocabularyObjectDetectionAnnotator.__name__
                        ):
                            containment_ratio = self.compute_containment_ratio(
                                object_hypothesis, human_hypothesis
                            )
                            if containment_ratio > 0.8:
                                object_designator = ObjectDesignator()
                                object_designator.type = query.obj.type
                                object_designator.description.append(
                                    detection_classification.classname
                                )
                                self.add_identity(object_designator, human_hypothesis)

                                positions = self.get_cas().filter_by_type(
                                    PositionAnnotation, human_hypothesis.annotations
                                )
                                self.add_positions(object_designator, positions)

                                query_result.append(object_designator)

        elif query.obj.type == QueryType.OBJECT:
            object_hypotheses = self.get_cas().filter_annotations_by_type(
                ObjectHypothesis
            )
            query_result = []
            result = Query.Result()

            for object_hypothesis in object_hypotheses:
                classifications = self.get_cas().filter_by_type(
                    Classification, object_hypothesis.annotations
                )
                if not classifications:
                    raise MissingObjectClassification()
                detection_classification = classifications[0]
                if (
                    detection_classification.source
                    == OpenVocabularyObjectDetectionAnnotator.__name__
                ):
                    object_designator = ObjectDesignator()
                    object_designator.type = query.obj.type
                    object_designator.description.append(
                        detection_classification.classname
                    )

                    positions = self.get_cas().filter_by_type(
                        PositionAnnotation, object_hypothesis.annotations
                    )
                    self.add_positions(object_designator, positions)

                    query_result.append(object_designator)
        else:
            return Status.FAILURE

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS


# %% vision language answers
class GenerateVisionLanguageResult(GenerateResult):
    """
    Answers a query with every person the :class:`VisionLanguageAnnotator` described,
    each part of the description between commas as one attribute.
    """

    def __init__(self, name: str = "GenerateVisionLanguageResult") -> None:
        """
        :param name: Name of this annotator in the tree.
        """
        super().__init__(name=name)

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to answer with every described person and
        the parts of the description.

        :return: :attr:`Status.SUCCESS`.
        """
        query = self.get_cas().get(CASViews.QUERY)
        query_result = []
        for human_hypothesis in self.get_cas().filter_annotations_by_type(
            HumanHypothesis
        ):
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            for activity in self.get_cas().filter_by_type(
                ActivityAnnotation, human_hypothesis.annotations
            ):
                if activity.source != VisionLanguageAnnotator.__name__:
                    continue
                object_designator.attribute.extend(
                    part.strip()
                    for part in activity.type.strip("[] ").split(",")
                    if part.strip()
                )
            if not object_designator.attribute:
                continue
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, human_hypothesis.annotations
            )
            self.add_identity(object_designator, human_hypothesis)
            self.add_positions(object_designator, positions)
            query_result.append(object_designator)
        QueryHandler.send_answer(Query.Result(res=query_result))
        self.feedback_message = "Query result generated."
        return Status.SUCCESS
