"""
Check that stops a human pipeline with an empty answer when no human was detected.
"""

from __future__ import annotations

# %% Imports
from py_trees.common import Status
from robokudo_msgs.action import Query

from robokudo.annotators.cas_check import CASCheckAnnotationTypeExists
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.query import QueryHandler

# %% human check


class CheckForHumanHypothesis(CASCheckAnnotationTypeExists):
    """
    Checks whether the CAS holds a :class:`HumanHypothesis`.

    Without one it answers the query with nobody and stops the pipeline.
    """

    def __init__(self, name: str = "CheckForHumanHypothesis") -> None:
        """
        :param name: Name of this annotator in the tree.
        """
        super().__init__(name=name, annotation_type=HumanHypothesis)

    def update(self) -> Status:
        """
        Extends :meth:`CASCheckFunc.update` by answering the query when no human exists.

        :return: :attr:`Status.SUCCESS` if a human exists, otherwise
            :attr:`Status.FAILURE`.
        """
        status = super().update()
        if status == Status.FAILURE:
            QueryHandler.send_answer(Query.Result())
            self.feedback_message = "No human detected, answered with nobody"
        return status
