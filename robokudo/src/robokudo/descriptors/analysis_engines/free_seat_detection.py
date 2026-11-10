"""
Analysis engine answering queries for the free seats in view of the HSR head camera.

On every query, the pipeline reads one camera image, detects chairs, sofas and humans,
judges the posture of the humans and answers with every seat within reach and whether a
sitting human occupies it.
"""

# %% imports

from __future__ import annotations

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import QueryAnnotator
from robokudo.annotators.seat_occupancy import (
    SeatOccupancyAnnotator,
    SeatQueryAnswerAnnotator,
)
from robokudo.annotators.sofa_split import SofaSplitAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.descriptors.camera_configs.config_hsr_ros2 import HsrRos2CameraConfig
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline

# %% analysis engine


class AnalysisEngine(AnalysisEngineInterface):
    """
    Analysis engine answering queries for the seats in view of the HSR head camera and
    whether a human occupies them.
    """

    def name(self) -> str:
        """
        :return: The name identifier of this analysis engine
        """
        return "free_seat_detection"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline that waits for a query, reads one camera image, detects seats
        and humans, locates them in 3D, annotates every seat within reach, of chairs and
        sofas, with whether it is occupied and answers with the seats.

        Every query starts the seat detection, whatever it asks for.

        :return: The configured pipeline
        """
        hsr_config = CollectionReaderDescriptorFactory.create_descriptor(
            HsrRos2CameraConfig.registry_name
        )

        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.pretrained_model = "chairs.pt"
        yolo_descriptor.parameters.threshold = 0.4
        yolo_descriptor.parameters.precision_mode = True
        yolo_descriptor.parameters.ignored_class_names = [
            yolo_descriptor.parameters.person_class_name
        ]

        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True

        sitting_descriptor = HumanSittingAnnotator.Descriptor()
        sitting_descriptor.parameters.knee_at_hip_height_is_sitting = True

        seq = Pipeline("FreeSeatDetectionPipeline")
        seq.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=hsr_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                YoloAnnotator(descriptor=yolo_descriptor),
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                ClusterPositionAnnotator(),
                HumanStandingAnnotator(),
                HumanSittingAnnotator(descriptor=sitting_descriptor),
                SofaSplitAnnotator(),
                SeatOccupancyAnnotator(),
                SeatQueryAnswerAnnotator(),
                ActionServerCheck(),
            ]
        )
        return seq
