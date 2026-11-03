from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.descriptors.camera_configs.config_hsr import HsrCameraConfig
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """
    Analysis engine detecting seats and the humans sitting on them with the HSR head
    camera.
    """

    def name(self) -> str:
        """
        :return: The name identifier of this analysis engine
        """
        return "free_seat_detection"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline detecting seats and humans and locating them in 3D.

        :return: The configured pipeline
        """
        hsr_config = CollectionReaderDescriptorFactory.create_descriptor(
            HsrCameraConfig.registry_name,
            tf_from="head_rgbd_sensor_rgb_frame",
        )

        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.pretrained_model = "chairs.pt"
        yolo_descriptor.parameters.threshold = 0.4
        yolo_descriptor.parameters.precision_mode = True

        seq = Pipeline("FreeSeatDetectionPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=hsr_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                YoloAnnotator(descriptor=yolo_descriptor),
                HumanAndPoseAnnotator(),
                ClusterPositionAnnotator(),
            ]
        )
        return seq
