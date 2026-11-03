from __future__ import annotations

# %% imports
from builtin_interfaces.msg import Time
from py_trees.common import Status
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from std_msgs.msg import Header

from typing_extensions import Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefEntry
from robokudo.annotators.gpsr.exceptions import MissingObjectClassification
from robokudo.annotators.gpsr.human_activity_annotator import HumanActivityAnnotator
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator
from robokudo.annotators.gpsr.vision_language_annotator import VisionLanguageAnnotator
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


# %% camera header
def camera_header(cas: CAS) -> Header:
    """
    The header of the current camera image: its acquisition time and the frame of the
    camera.
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
    Build answers for detected hypotheses.
    """

    def add_identity(
        self,
        object_designator: ObjectDesignator,
        hypothesis: HumanHypothesis,
    ) -> None:
        """
        Add the person's belief identifier to the answer when available.
        """
        belief_ids = self.get_cas().filter_by_type(BeliefEntry, hypothesis.annotations)
        if belief_ids:
            object_designator.uid = str(belief_ids[0].id)

    def add_positions(
        self, object_designator: ObjectDesignator, positions: list[PositionAnnotation]
    ) -> None:
        """
        Add observed positions and their sources to the answer.
        """
        for position in positions:
            header = camera_header(self.get_cas())
            pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(
                position, header
            )
            object_designator.pose.append(pose_stamped)
            object_designator.pose_source.append(position.source)


# %% posture answers
class GenerateHumanPostureResults(GenerateResult):
    """
    Answer queries for people whose unambiguous posture matches the query.
    """

    def __init__(
        self, name: str = "GenerateHumanPostureResults", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        keypoint_rule_sources = {
            HumanSittingAnnotator.__name__,
            HumanStandingAnnotator.__name__,
            HumanLyingAnnotator.__name__,
        }

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, hypothesis.annotations
            )
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, hypothesis.annotations
            )

            # a person's posture is known when exactly one keypoint rule fires
            keypoint_postures = {
                activity.type
                for activity in activities
                if activity.source in keypoint_rule_sources
            }
            if len(keypoint_postures) != 1:
                self.rk_logger.info(
                    f"Posture of HH {hypothesis.id}: undecided, rules fired: {sorted(keypoint_postures)}"
                )
                continue
            posture = keypoint_postures.pop()
            if posture == HumanLyingAnnotator.posture:
                posture = HumanLyingAnnotator.query_word
            self.rk_logger.info(f"Posture of HH {hypothesis.id}: {posture}")
            if posture == query.obj.attribute[0]:
                object_designator.attribute.append(posture)

            if object_designator.attribute:
                self.add_identity(object_designator, hypothesis)
                self.add_positions(object_designator, positions)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = "Query result generated."
        return Status.SUCCESS


# %% gesture answers
class GenerateWavingRaisingResult(GenerateResult):
    """
    Answer queries for matching waving and arm raising activities.
    """

    def __init__(
        self, name: str = "ObjectDetectionResult", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, hypothesis.annotations
            )
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, hypothesis.annotations
            )

            for activity in activities:
                if activity.source == "HumanWavingAnnotator":
                    if activity.type == query.obj.attribute[0]:
                        object_designator.description.append(activity.type)

            if object_designator.description:
                self.add_identity(object_designator, hypothesis)
                self.add_positions(object_designator, positions)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS


# %% pointing answers
class GeneratePointingResult(GenerateResult):
    """
    Answer queries for people pointing in the requested direction.
    """

    def __init__(self, name: str = "PointingResult", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, hypothesis.annotations
            )
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, hypothesis.annotations
            )

            pointing_directions = [
                activity.type
                for activity in activities
                if activity.source == "HumanPointingAnnotator"
            ]
            for direction in pointing_directions:
                object_designator.description.append(f"Behavior suggests {direction}.")

            if object_designator.description:
                if query.obj.attribute[0] in pointing_directions:
                    self.add_identity(object_designator, hypothesis)
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
    Answer queries with observed postures and gestures, optionally filtered by room.
    """

    def __init__(
        self, name: str = "GenerateHumanActivityResult", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        if query.obj.location:
            human_hypotheses = [
                hypothesis
                for hypothesis in human_hypotheses
                if any(
                    isinstance(annotation, LocationAnnotation)
                    and annotation.name == query.obj.location
                    for annotation in hypothesis.annotations
                )
            ]
        query_result = []
        result = Query.Result()

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            positions = self.get_cas().filter_by_type(
                PositionAnnotation, hypothesis.annotations
            )
            activities = self.get_cas().filter_by_type(
                ActivityAnnotation, hypothesis.annotations
            )

            pointing_refinements = [
                activity.type
                for activity in activities
                if activity.source == "HumanPointingAnnotator"
            ]

            for activity in activities:
                if activity.source == HumanActivityAnnotator.__name__:
                    if (
                        activity.type
                        in HumanActivityAnnotator.Descriptor.Parameters().posture_labels
                    ):
                        object_designator.attribute.append(activity.type)
                    elif activity.type == "Pointing" and pointing_refinements:
                        # skeleton already has a concrete direction so we skip the ambiguous CLIP pointing label
                        continue
                    elif activity.type == "No gesture" and (
                        pointing_refinements
                        or any(
                            activity.source == "HumanWavingAnnotator"
                            for activity in activities
                        )
                    ):
                        # skeleton found a real gesture so the CLIP no gesture label is dropped
                        continue
                    else:
                        object_designator.description.append(activity.type)
                elif activity.source == "HumanPointingAnnotator":
                    object_designator.description.append(activity.type)
                elif activity.source == "HumanWavingAnnotator":
                    object_designator.description.append(activity.type)

            if object_designator.attribute or object_designator.description:
                self.add_identity(object_designator, hypothesis)
                self.add_positions(object_designator, positions)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = "Query result generated."
        return Status.SUCCESS


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
        with_region_filter: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.with_region_filter = with_region_filter

    def containment_ratio(
        self, inner: ObjectHypothesis, outer: ObjectHypothesis
    ) -> float:
        """
        Calculate the containment ratio of two hypotheses.

        :param inner: The inner hypothesis (the one that is contained).
        :param outer: The outer hypothesis (the one that contains).
        """
        inner_x_min = inner.roi.roi.pos.x
        inner_x_max = inner.roi.roi.pos.x + inner.roi.roi.width
        inner_y_min = inner.roi.roi.pos.y
        inner_y_max = inner.roi.roi.pos.y + inner.roi.roi.height
        outer_x_min = outer.roi.roi.pos.x
        outer_x_max = outer.roi.roi.pos.x + outer.roi.roi.width
        outer_y_min = outer.roi.roi.pos.y
        outer_y_max = outer.roi.roi.pos.y + outer.roi.roi.height

        xA = max(inner_x_min, outer_x_min)
        yA = max(inner_y_min, outer_y_min)
        xB = min(inner_x_max, outer_x_max)
        yB = min(inner_y_max, outer_y_max)

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        inner_area = (inner_x_max - inner_x_min) * (inner_y_max - inner_y_min)

        self.rk_logger.info(
            f"The outer hypothesis contains the inner hypothesis to {min(interArea / inner_area, 100)} of its area"
        )
        return interArea / inner_area

    def intersection_over_union(
        self, detection: ObjectHypothesis, human: HumanHypothesis
    ) -> float:
        """
        Calculate the intersection over union of two hypotheses.

        :param detection: The first hypothesis (the one that is contained).
        :param human: The second hypothesis (the one that contains).
        """
        box1_x_min = detection.roi.roi.pos.x
        box1_x_max = detection.roi.roi.pos.x + detection.roi.roi.width
        box1_y_min = detection.roi.roi.pos.y
        box1_y_max = detection.roi.roi.pos.y + detection.roi.roi.height
        box2_x_min = human.roi.roi.pos.x
        box2_x_max = human.roi.roi.pos.x + human.roi.roi.width
        box2_y_min = human.roi.roi.pos.y
        box2_y_max = human.roi.roi.pos.y + human.roi.roi.height

        xA = max(box1_x_min, box2_x_min)
        yA = max(box1_y_min, box2_y_min)
        xB = min(box1_x_max, box2_x_max)
        yB = min(box1_y_max, box2_y_max)

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        box1_area = (box1_x_max - box1_x_min) * (box1_y_max - box1_y_min)
        box2_area = (box2_x_max - box2_x_min) * (box2_y_max - box2_y_min)

        iou = interArea / float(box1_area + box2_area - interArea)

        self.rk_logger.info(
            f"Intersection over union between {detection} and {human}: {iou}"
        )
        return iou

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)

        # HumanPipeline
        if query.obj.type == "human" or query.obj.type == "humans":
            human_hypotheses = self.get_cas().filter_annotations_by_type(
                HumanHypothesis
            )
            object_hypotheses = self.get_cas().filter_annotations_by_type(
                ObjectHypothesis
            )

            if self.with_region_filter and query.obj.location:
                human_hypotheses = [
                    h
                    for h in human_hypotheses
                    if any(
                        isinstance(a, LocationAnnotation)
                        and a.name == query.obj.location
                        for a in h.annotations
                    )
                ]

            query_result = []
            result = Query.Result()

            for human_hypothesis in human_hypotheses:
                for object_hypothesis in object_hypotheses:
                    # We dont want to compare human with human or object with object
                    if not isinstance(object_hypothesis, HumanHypothesis):
                        detection_classification = self.get_cas().filter_by_type(
                            Classification, object_hypothesis.annotations
                        )[0]
                        if (
                            detection_classification.source
                            == OpenVocabularyObjectDetectionAnnotator.__name__
                        ):
                            containment_ratio = self.containment_ratio(
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

        # ObjectPipeline
        elif query.obj.type == "object":
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


class GenerateVisionLanguageResult(GenerateResult):
    """
    Answers a query with every person the :class:`VisionLanguageAnnotator` described,
    each part of the description between commas as one attribute.
    """

    def __init__(self, name: str = "GenerateVisionLanguageResult") -> None:
        super().__init__(name=name)

    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)
        answered = []
        for hypothesis in self.get_cas().filter_annotations_by_type(HumanHypothesis):
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            for activity in self.get_cas().filter_by_type(
                ActivityAnnotation, hypothesis.annotations
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
                PositionAnnotation, hypothesis.annotations
            )
            self.add_identity(object_designator, hypothesis)
            self.add_positions(object_designator, positions)
            answered.append(object_designator)
        QueryHandler.send_answer(Query.Result(res=answered))
        self.feedback_message = "Query result generated."
        return Status.SUCCESS
