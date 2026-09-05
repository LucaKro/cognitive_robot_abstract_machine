"""
Putting an unusable answer back to the model instead of acting on it.

Four steps ask a model something, and all four correct an answer the same way: read what
came back, say what is wrong with it, and ask again with that. What is checked here is
the correcting itself -- how often it happens, what the model is told, and what is kept
when even the last attempt is unusable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from experiments.warsaw.pipeline.asking import Question, Questioner
from semantic_digital_twin.adapters.vision_language_model.client import ModelResponse
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    MessagePart,
    TextPart,
)
from typing_extensions import Any, Dict, List


@dataclass
class ScriptedAnswers:
    """
    A model answering from a list written in advance, recording what it was asked.
    """

    replies: List[str]
    """
    What to answer, one per question, the last repeating once they run out.
    """

    asked: List[List[MessagePart]] = field(default_factory=list)
    """
    Every message it was sent, in order.
    """

    def ask(self, content, system: str) -> ModelResponse:
        """
        :param content: The question.
        :param system: What it is told it is doing.
        :return: The next scripted answer.
        """
        self.asked.append(list(content))
        spoken = self.replies[min(len(self.asked) - 1, len(self.replies) - 1)]
        return ModelResponse.from_json({"choices": [{"message": {"content": spoken}}]})


@dataclass
class NamingOneThing(Question[Dict[str, Any]]):
    """
    A question answered by one name, which has to be one that was offered.
    """

    allowed: List[str] = field(default_factory=list)
    """
    The names the answer may give.
    """

    @property
    def key(self) -> str:
        return "the-one-question"

    @property
    def system_prompt(self) -> str:
        return "Name one of them."

    def message(self) -> List[MessagePart]:
        return [TextPart("Which one is it?")]

    def read(self, response: ModelResponse) -> Dict[str, Any]:
        return response.parse_json()

    def refusal(self, refused: ModelRefusedError) -> Dict[str, Any]:
        return {}

    def problems_with(self, answer: Dict[str, Any]) -> List[str]:
        if answer.get("name") in self.allowed:
            return []
        return [f"{answer.get('name')!r} is not one of {', '.join(self.allowed)}"]


@pytest.fixture
def question():
    """
    :return: A question with two acceptable answers.
    """
    return NamingOneThing(allowed=["drawer", "cabinet"])


# %% how often a question is asked


def test_a_usable_answer_is_asked_once(tmp_path, question):
    """
    Nothing is corrected when nothing is wrong.
    """
    model = ScriptedAnswers(replies=['{"name": "drawer"}'])
    answered = Questioner(model=model, answers_directory=tmp_path).answer(question)
    assert answered.is_usable
    assert answered.attempts == 1
    assert len(model.asked) == 1


def test_an_unusable_answer_is_put_back_once_by_default(tmp_path, question):
    """
    One correction is what a run allows unless it is told otherwise.
    """
    model = ScriptedAnswers(replies=['{"name": "sink"}', '{"name": "cabinet"}'])
    answered = Questioner(model=model, answers_directory=tmp_path).answer(question)
    assert answered.is_usable
    assert answered.answer["name"] == "cabinet"
    assert answered.attempts == 2


def test_no_corrections_means_the_answer_stands_as_it_came(tmp_path, question):
    """
    A run told to correct nothing asks each question exactly once.
    """
    model = ScriptedAnswers(replies=['{"name": "sink"}', '{"name": "cabinet"}'])
    answered = Questioner(
        model=model, answers_directory=tmp_path, corrections=0
    ).answer(question)
    assert not answered.is_usable
    assert len(model.asked) == 1


def test_an_answer_that_stays_unusable_is_kept_with_what_is_wrong_with_it(
    tmp_path, question
):
    """
    A question nobody could answer costs that question rather than the run, and what was
    wrong is written down beside it.
    """
    model = ScriptedAnswers(replies=['{"name": "sink"}'])
    answered = Questioner(model=model, answers_directory=tmp_path).answer(question)
    assert not answered.is_usable
    assert answered.problems == ["'sink' is not one of drawer, cabinet"]
    assert answered.attempts == 2


# %% what the model is told when it is asked again


def test_the_correction_says_what_was_wrong(tmp_path, question):
    """
    Saying what was wrong is what makes a second attempt worth more than a first.
    """
    model = ScriptedAnswers(replies=['{"name": "sink"}', '{"name": "cabinet"}'])
    Questioner(model=model, answers_directory=tmp_path).answer(question)
    correction = model.asked[1][-1].text
    assert "'sink' is not one of drawer, cabinet" in correction


def test_the_question_itself_is_asked_again_unchanged(tmp_path, question):
    """
    A correction is added to the question rather than replacing it.
    """
    model = ScriptedAnswers(replies=['{"name": "sink"}', '{"name": "cabinet"}'])
    Questioner(model=model, answers_directory=tmp_path).answer(question)
    assert model.asked[1][0] == model.asked[0][0]


# %% a reply that holds no answer at all


def test_a_refusal_is_corrected_like_any_other_unusable_answer(tmp_path, question):
    """
    A model that answers with prose has refused, and the refusal is what it is told
    about.
    """
    model = ScriptedAnswers(
        replies=["I am not sure what that is.", '{"name": "drawer"}']
    )
    answered = Questioner(model=model, answers_directory=tmp_path).answer(question)
    assert answered.is_usable
    assert answered.attempts == 2


def test_a_refusal_that_stands_leaves_the_blank_answer_the_question_names(
    tmp_path, question
):
    """
    What a refusal costs is decided by the question, not by the asking.
    """
    model = ScriptedAnswers(replies=["I am not sure what that is."])
    answered = Questioner(model=model, answers_directory=tmp_path).answer(question)
    assert answered.answer == {}
    assert not answered.is_usable


# %% keeping and re-reading the replies


def test_every_reply_is_kept_as_it_came_back(tmp_path, question):
    """
    The reply is what a run is re-read from, so it is kept whole rather than parsed
    away.
    """
    model = ScriptedAnswers(replies=['{"name": "drawer"}'])
    Questioner(model=model, answers_directory=tmp_path).answer(question)
    kept = json.loads((tmp_path / "the-one-question.json").read_text())
    assert kept["choices"][0]["message"]["content"] == '{"name": "drawer"}'


def test_a_kept_reply_is_read_back_instead_of_asking_again(tmp_path, question):
    """
    Re-reading a run costs nothing, which is the point of keeping the replies.
    """
    (tmp_path / "the-one-question.json").write_text(
        json.dumps({"choices": [{"message": {"content": '{"name": "cabinet"}'}}]})
    )
    model = ScriptedAnswers(replies=['{"name": "drawer"}'])
    answered = Questioner(
        model=model, answers_directory=tmp_path, reuse_answers=True
    ).answer(question)
    assert answered.answer["name"] == "cabinet"
    assert model.asked == []


def test_a_correction_is_asked_rather_than_read_back(tmp_path, question):
    """
    A kept reply is the one that was wrong, so re-reading it would correct nothing.
    """
    (tmp_path / "the-one-question.json").write_text(
        json.dumps({"choices": [{"message": {"content": '{"name": "sink"}'}}]})
    )
    model = ScriptedAnswers(replies=['{"name": "cabinet"}'])
    answered = Questioner(
        model=model, answers_directory=tmp_path, reuse_answers=True
    ).answer(question)
    assert answered.is_usable
    assert len(model.asked) == 1
