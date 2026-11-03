from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.human_activity_annotator import HumanActivityAnnotator
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.utils.file_loader import FileLoader
from robokudo.annotators.gpsr.gpsr_results import GenerateHumanActivityResult
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.check_for_human_hypothesis import CheckForHumanHypothesis
from robokudo.annotators.gpsr.filter_human_by_activity_annotator import (
    FilterHumanByActivityAnnotator,
)
from robokudo.annotators.gpsr.filter_objects_by_region import FilterObjectsByRegion
from robokudo.annotators.gpsr.generate_all_annotations import GenerateAllQueryResult


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Find people in the queried room, optionally by activity.
    """

    def __init__(
        self, with_activity: bool = False, *args: object, **kwargs: object
    ) -> None:
        """:param with_activity: Whether to filter by the queried activity."""
        super().__init__(*args, **kwargs)
        self.with_activity = with_activity
        """
        Whether to filter by the queried activity.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        # the pose model lies next to our weights, outside CRAM
        human_descriptor.parameters.models_directory = (
            FileLoader.get_path_to_file_in_ros_package(
                ros_pkg_name="robokudo_suturo", relative_path="weights"
            )
        )
        if self.with_activity:
            human_descriptor.parameters.generate_clouds_for_keypoints = True

        filter_region_descriptor = FilterObjectsByRegion.Descriptor()

        seq = py_trees.composites.Sequence("subtree_human_in_room", memory=True)
        seq.add_child(HumanAndPoseAnnotator(descriptor=human_descriptor))
        seq.add_child(CheckForHumanHypothesis())

        if self.with_activity:
            waving_descriptor = HumanWavingAnnotator.Descriptor()
            waving_descriptor.parameters.check_activity = True

            pointing_descriptor = HumanPointingAnnotator.Descriptor()
            pointing_descriptor.parameters.pointing_length_meters = 5
            pointing_descriptor.parameters.known = True
            pointing_descriptor.parameters.min_arm_extension = 0.25

            seq.add_children(
                [
                    HumanActivityAnnotator(),
                    ClusterPositionAnnotator(),
                    HumanWavingAnnotator(descriptor=waving_descriptor),
                    HumanPointingAnnotator(descriptor=pointing_descriptor),
                    FilterObjectsByRegion(descriptor=filter_region_descriptor),
                    BeliefStateAnnotator(),
                    FilterHumanByActivityAnnotator(),
                    GenerateHumanActivityResult(),
                ]
            )
        else:
            seq.add_children(
                [
                    ClusterPositionAnnotator(),
                    FilterObjectsByRegion(descriptor=filter_region_descriptor),
                    BeliefStateAnnotator(),
                    GenerateAllQueryResult(),
                ]
            )

        return seq
