from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
from typing_extensions import TYPE_CHECKING
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.filter_objects_by_region import FilterObjectsByRegion
from robokudo.annotators.gpsr.generate_all_annotations import GenerateAllQueryResult
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection import (
    Subtree as ObjectDetectionSubtree,
)

if TYPE_CHECKING:
    from pathlib import Path


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detect objects and mark those in the queried room.
    """

    def __init__(self, model_directory: Path | None = None) -> None:
        """:param model_directory: Directory containing detector files and mappings."""
        super().__init__()
        self.model_directory = model_directory
        """
        Directory containing detector files and mappings, if set.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        filter_region_descriptor = FilterObjectsByRegion.Descriptor()

        seq = py_trees.composites.Sequence("subtree_obj_in_room", memory=True)
        seq.add_children(
            [
                ObjectDetectionSubtree(
                    model="ultimate_model_v11_1.26.4",
                    outlier_removal_name="OutlierRemovalOnObjectHypothesisInRoom",
                    model_directory=self.model_directory,
                ).implementation(),
                FilterObjectsByRegion(descriptor=filter_region_descriptor),
                BeliefStateAnnotator(),
                GenerateAllQueryResult(),
            ]
        )
        return seq
