from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.gpsr.belief_state_lookup_annotator import (
    BeliefStateLookupAnnotator,
)


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Answer queries from the stored belief state.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        seq = py_trees.composites.Sequence("subtree_belief_lookup", memory=True)
        seq.add_child(BeliefStateLookupAnnotator())
        return seq
