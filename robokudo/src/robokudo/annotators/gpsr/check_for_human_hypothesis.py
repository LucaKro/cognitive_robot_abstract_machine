from __future__ import annotations

# %% Imports
from py_trees.common import Status
from robokudo_msgs.action import Query

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.query import QueryHandler

# %% human check


class CheckForHumanHypothesis(BaseAnnotator):
    """
    Checks whether the CAS holds a :class:`HumanHypothesis`.

    Without one it answers the query with nobody and stops the pipeline.
    """

    def __init__(self, name: str = "CheckForHumanHypothesis") -> None:
        super().__init__(name=name)

    def update(self) -> Status:
        annotations = self.get_cas().annotations
        for annotation in annotations:
            if isinstance(annotation, HumanHypothesis):
                return Status.SUCCESS
        QueryHandler.send_answer(Query.Result())
        self.feedback_message = "No human detected, answered with nobody"
        return Status.FAILURE
