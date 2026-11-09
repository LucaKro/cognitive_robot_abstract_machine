"""
A camera at a known place in the world, the GPSR query asked of it, and the hypotheses
the annotators under test read.
"""

from __future__ import annotations

# %% Imports
import copy
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from py_trees.blackboard import Blackboard
from robokudo.annotators.gpsr.human_activity_annotator import HumanActivityAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.identifier import BBIdentifier
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import LocationAnnotation, PositionAnnotation
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import Point2D, Rect
from robokudo.types.human import ActivityAnnotation, KeypointAnnotation
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis
from robokudo_msgs.action import Query
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
)
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator

# %% scene


@dataclass
class GpsrScene:
    """
    A CAS holding a black color image, an empty point cloud, a camera at a known place
    in the world and the GPSR query the annotators under test answer.
    """

    query: Query.Goal | None = field(default_factory=Query.Goal)
    """
    The query the annotators under test read from the CAS, or *None* if the CAS holds no
    query.
    """

    resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(width=640, height=480)
    )
    """
    Resolution of the color image in the CAS.
    """

    world_T_camera: HomogeneousTransformationMatrix | None = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.0,
            pos_y=0.0,
            pos_z=0.0,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
        )
    )
    """
    Pose of the camera in the world, or *None* if the camera pose is unknown.
    """

    focal_length: float = 500.0
    """
    Focal length of the camera in pixels, along both image axes.
    """

    hypothesis_region: Rect = field(
        default_factory=lambda: Rect(pos=Point2D(x=10, y=10), width=50, height=100)
    )
    """
    Region of the color image every added hypothesis covers.
    """

    timestamp_nanoseconds: int = 1_500_000_123
    """
    Acquisition time of the camera image.
    """

    camera_body: Body = field(default_factory=lambda: Body(name=PrefixedName("camera")))
    """
    The body the camera is mounted on, whose name is the frame of the camera image.
    """

    world_body: Body = field(default_factory=lambda: Body(name=PrefixedName("map")))
    """
    The root of the world the camera pose is given in.
    """

    cas: CAS | None = None
    """
    The CAS of the pipeline the scene was placed in.
    """

    def color_image(self) -> np.ndarray:
        """
        :return: A black color image of the scene's resolution.
        """
        return np.zeros(
            (self.resolution.height, self.resolution.width, 3), dtype=np.uint8
        )

    def camera_model(self) -> PinholeCameraModel:
        """
        :return: A pinhole camera of the scene's resolution looking through the image
            center.
        """
        return PinholeCameraModel(
            image_resolution=self.resolution,
            focal_length_x=self.focal_length,
            focal_length_y=self.focal_length,
            principal_point=Point2(
                x=self.resolution.width / 2.0, y=self.resolution.height / 2.0
            ),
        )

    def camera_observation(self) -> CameraObservation:
        """
        :return: The observation of the scene's camera, posed in the world when its pose
            is known.
        """
        camera_model = self.camera_model()
        world_T_camera = None
        if self.world_T_camera is not None:
            world_T_camera = HomogeneousTransformationMatrix(
                data=self.world_T_camera,
                reference_frame=self.world_body,
                child_frame=self.camera_body,
            )
        return CameraObservation(
            camera=Camera(
                name=self.camera_body.name,
                root=self.camera_body,
                camera_model=camera_model,
            ),
            effective_camera_model=camera_model,
            world_T_camera=world_T_camera,
            timestamp_nanoseconds=self.timestamp_nanoseconds,
        )

    def place_in_pipeline(self, annotator: BaseAnnotator) -> CAS:
        """
        Set the annotator up in a pipeline whose CAS holds the scene.

        :param annotator: The annotator to set up.
        :return: The CAS the annotator reads and writes.
        """
        annotator.setup()
        pipeline = Pipeline("Sequence")
        pipeline.add_child(annotator)
        pipeline.setup()
        self.cas = annotator.get_cas()
        self.cas.set(CASViews.COLOR_IMAGE, self.color_image())
        self.cas.set(CASViews.CLOUD, o3d.geometry.PointCloud())
        if self.query is not None:
            self.cas.set_ref(CASViews.QUERY, self.query)
        self.cas.camera_observation = self.camera_observation()
        return self.cas

    def add_human(
        self, *annotations: Any, position: list[float] | None = None
    ) -> HumanHypothesis:
        """
        Add a human carrying the given annotations to the CAS.

        :param annotations: The annotations of the human.
        :param position: Position of the human in the camera frame, if one is known.
        :return: The added human.
        """
        human_hypothesis = HumanHypothesis()
        self._fill(human_hypothesis, annotations, position)
        return human_hypothesis

    def add_object(
        self, *annotations: Any, position: list[float] | None = None
    ) -> ObjectHypothesis:
        """
        Add an object carrying the given annotations to the CAS.

        :param annotations: The annotations of the object.
        :param position: Position of the object in the camera frame, if one is known.
        :return: The added object.
        """
        hypothesis = ObjectHypothesis()
        self._fill(hypothesis, annotations, position)
        return hypothesis

    def _fill(
        self,
        hypothesis: ObjectHypothesis,
        annotations: tuple[Any, ...],
        position: list[float] | None,
    ) -> None:
        """
        Give the hypothesis its annotations and position and add it to the CAS.

        :param hypothesis: The hypothesis to fill.
        :param annotations: The annotations of the hypothesis.
        :param position: The position of the hypothesis, if any.
        """
        hypothesis.roi.roi = copy.deepcopy(self.hypothesis_region)
        hypothesis.annotations.extend(annotations)
        if position is not None:
            hypothesis.annotations.append(PositionAnnotation(translation=position))
        self.cas.annotations.append(hypothesis)

    def decide_activities(
        self, annotator: BaseAnnotator, keypoints: KeypointAnnotation
    ) -> list[str]:
        """
        Run the annotator on one person with the given keypoints.

        :param annotator: The annotator to run.
        :param keypoints: The keypoints of the person.
        :return: The activities the annotator annotated the person with.
        """
        self.place_in_pipeline(annotator)
        human_hypothesis = self.add_human(keypoints)
        annotator.update()
        return self.activities_of(human_hypothesis)

    @staticmethod
    def clip_activity(activity: str) -> ActivityAnnotation:
        """
        :param activity: The posture or gesture.
        :return: A posture or gesture as the CLIP activity annotator reports it.
        """
        return ActivityAnnotation(type=activity, source=HumanActivityAnnotator.__name__)

    @staticmethod
    def activities_of(
        hypothesis: ObjectHypothesis, source: type[BaseAnnotator] | None = None
    ) -> list[str]:
        """
        :param hypothesis: The hypothesis to read.
        :param source: The annotator whose activities are returned, or *None* for the
            activities of every annotator.
        :return: The activities the hypothesis was annotated with.
        """
        return [
            activity.type
            for activity in CAS.filter_by_type(
                ActivityAnnotation, hypothesis.annotations
            )
            if source is None or activity.source == source.__name__
        ]

    @staticmethod
    def locations_of(hypothesis: ObjectHypothesis) -> list[str]:
        """
        :param hypothesis: The hypothesis to read.
        :return: The names of the locations the hypothesis was marked with.
        """
        return [
            location.name
            for location in CAS.filter_by_type(
                LocationAnnotation, hypothesis.annotations
            )
        ]

    @staticmethod
    def sent_answer() -> Query.Result:
        """
        :return: The answer the annotators under test sent to the query.
        """
        return Blackboard().get(BBIdentifier.QUERY_ANSWER)

    @staticmethod
    def has_answered() -> bool:
        """
        :return: Whether an annotator under test sent an answer to the query.
        """
        return Blackboard().exists(BBIdentifier.QUERY_ANSWER)

    @staticmethod
    def has_failed() -> bool:
        """
        :return: Whether an annotator under test raised an exception to the blackboard.
        """
        return Blackboard().exists(BBIdentifier.BLACKBOARD_EXCEPTION_NAME)
