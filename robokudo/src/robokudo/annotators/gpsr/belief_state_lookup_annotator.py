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
    Answers a recall query with remembered persons and objects, as JSON in the text
    result of the answer.

    - With a ``uid``, the answer is the entry with that id.
    - With a description, the answer is every person for "human" or "person", and
      otherwise the entries whose class, detector label or semantic label is the
      description, tried in that order.
    - Without either, the answer is every entry.

    A single entry is answered as one JSON object, several as a list. When nothing
    matches, the answer is an error object.
    """

    def __init__(
        self, name: str = "BeliefStateLookupAnnotator", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self) -> Status:
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
            entries = list(state.persons())
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
