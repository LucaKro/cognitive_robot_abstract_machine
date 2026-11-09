"""
Answers to recall queries about remembered persons and objects.
"""

from __future__ import annotations

# %% Imports
import json

from py_trees.common import Status
from robokudo_msgs.action import Query
from typing_extensions import Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.cas import CASViews
from robokudo.utils.query import QueryHandler

# %% recall


class BeliefStateLookupAnnotator(BaseAnnotator):
    """
    Answers a recall query in the text result with one JSON object, a list, or an error.

    A query uid selects that entry. The description "human" or "person" selects all
    persons. Any other description selects entries by class, else by detector label,
    else by semantic label. With neither uid nor description, every entry is answered.
    """

    def __init__(
        self, name: str = "BeliefStateLookupAnnotator", *args: Any, **kwargs: Any
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        """
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to send the entries matching the query as
        the answer.

        :return: :attr:`Status.SUCCESS`.
        """
        query = self.get_cas().get(CASViews.QUERY)
        uid = query.obj.uid.strip()
        result = Query.Result()
        state = BeliefStateAnnotator._state

        if query.obj.description:
            category = query.obj.description[0].strip()
        else:
            category = ""

        if uid:
            try:
                entry = state.find_by_id(int(uid))
                if entry:
                    entries = [entry]
                else:
                    entries = []
            except ValueError:
                entries = []
        elif category in ("human", "person"):
            entries = list(state.find_persons())
        elif category:
            entries = (
                state.find_by_classname(category)
                or state.find_by_yolo_label(category)
                or state.find_by_semantic_label(category)
            )
        else:
            entries = list(state)

        if not entries:
            result.text_result = json.dumps({"error": "no entries found"})
            QueryHandler.send_answer(result)
            self.feedback_message = "No entries found."
            return Status.SUCCESS

        payload = []
        for entry in entries:
            payload.append(entry.to_dict())

        if len(payload) == 1:
            answer_content = payload[0]
        else:
            answer_content = payload
        result.text_result = json.dumps(answer_content)
        QueryHandler.send_answer(result)
        self.feedback_message = f"Returned {len(entries)} entry/entries."
        return Status.SUCCESS
