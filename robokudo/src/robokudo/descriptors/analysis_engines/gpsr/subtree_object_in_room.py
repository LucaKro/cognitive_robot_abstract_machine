"""
The GPSR subtree that detects objects and marks those in the queried room.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.filter_objects_by_region import FilterObjectsByRegion
from robokudo.annotators.gpsr.generate_all_annotations import GenerateAllQueryResult
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection import (
    Subtree as ObjectDetectionSubtree,
)


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detects objects and marks those in the queried room.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        :param name: Name of this subtree.
        :return: The sequence that answers with the objects in the queried room.
        """
        filter_room_descriptor = FilterObjectsByRegion.Descriptor()

        sequence = py_trees.composites.Sequence("subtree_object_in_room", memory=True)
        sequence.add_children(
            [
                ObjectDetectionSubtree(
                    model="ultimate_model_v11_1.26.4",
                    outlier_removal_name="OutlierRemovalOnObjectHypothesisInRoom",
                ).implementation(),
                FilterObjectsByRegion(descriptor=filter_room_descriptor),
                BeliefStateAnnotator(),
                GenerateAllQueryResult(),
            ]
        )
        return sequence
