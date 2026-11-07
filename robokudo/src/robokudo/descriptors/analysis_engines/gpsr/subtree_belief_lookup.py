"""
The GPSR subtree that answers recall queries from the belief state.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.gpsr.belief_state_lookup_annotator import (
    BeliefStateLookupAnnotator,
)


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Answers queries from the stored belief state.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        :param name: Name of this subtree.
        :return: The sequence that answers the query from the belief state.
        """
        sequence = py_trees.composites.Sequence("subtree_belief_lookup", memory=True)
        sequence.add_child(BeliefStateLookupAnnotator())
        return sequence
