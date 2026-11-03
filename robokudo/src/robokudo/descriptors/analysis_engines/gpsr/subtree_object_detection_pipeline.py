from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
from typing_extensions import TYPE_CHECKING
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection import (
    Subtree as ObjectDetectionSubtree,
)
from robokudo.annotators.query import GenerateQueryResult

if TYPE_CHECKING:
    from pathlib import Path


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detect objects and answer the query with the observed hypotheses.
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

        seq = py_trees.composites.Sequence(
            "subtree_object_detection_pipeline", memory=True
        )
        seq.add_children(
            [
                ObjectDetectionSubtree(
                    ros_pkg="robokudo_suturo",
                    model="ultimate_model_v11_1.26.4",
                    with_semantic_filter=True,
                    model_directory=self.model_directory,
                ).implementation(),
                BeliefStateAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return seq
