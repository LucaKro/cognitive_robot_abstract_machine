"""
Reading an answer out of a reply that was written by a model rather than by a schema.

A model asked for JSON answers with it fenced, with a sentence in front of it, with the
array alone where an object was asked for, and sometimes with nothing at all. What is
checked here is which of those still carry an answer and which are refusals, because a
refusal read as an answer is a body that quietly gets no class.
"""

from __future__ import annotations

import json

import pytest

from semantic_digital_twin.adapters.vision_language_model.client import (
    ModelResponse,
    ServiceFailure,
    VisionLanguageModel,
)
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ApiKeyMissingError,
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    PartKind,
    TextPart,
)


def response_from(replies, name: str) -> ModelResponse:
    """
    :param replies: The directory holding replies as a model returned them.
    :param name: Which reply to read.
    :return: It, ready to be read.
    """
    return ModelResponse.from_json(json.loads((replies / name).read_text()))


# %% answers that are wrapped rather than absent


def test_a_fenced_answer_is_read(replies):
    """
    A model asked for JSON answers with it inside a code fence.
    """
    assert response_from(replies, "fenced.json").parse_json() == {
        "class": "Drawer",
        "is_new_class": False,
    }


def test_an_answer_with_prose_around_it_is_read(replies):
    """
    A sentence in front of the answer, and one after it, leave the answer readable.
    """
    assert response_from(replies, "prose_wrapped.json").parse_json() == {
        "class": "Drawer",
        "is_new_class": False,
    }


def test_an_array_answer_is_read(replies):
    """
    Asked for an object holding an array, a model will answer with the array alone.
    """
    assert response_from(replies, "array_only.json").parse_json() == [
        {"name": "drawer_1", "class": "Drawer"}
    ]


def test_a_real_reply_is_read(replies):
    """
    A reply as the service actually returned one carries the answer it was asked for.
    """
    answered = response_from(replies, "vocabulary_kitchen_island.json").parse_json()
    assert answered["class"] == "KitchenIsland"
    assert answered["is_new_class"] is True


# %% replies that hold no answer


@pytest.mark.parametrize(
    "reply", ["empty.json", "null_content.json", "prose_only.json"]
)
def test_a_reply_without_json_is_a_refusal(replies, reply):
    """
    A model that answers with nothing, or with prose, has refused as surely as one that
    says so -- and a refusal has to raise rather than come back as a blank answer.
    """
    with pytest.raises(ModelRefusedError):
        response_from(replies, reply).parse_json()


def test_a_null_content_reads_as_empty_text(replies):
    """
    A reply can carry a null content, and a caller reading it as text should get text.
    """
    assert response_from(replies, "null_content.json").text == ""


# %% the message a question is sent as


def test_words_are_sent_as_words():
    """
    A text part names itself the way the chat completions schema does.
    """
    assert TextPart("hello").to_json() == {
        "type": PartKind.TEXT.value,
        "text": "hello",
    }


def test_a_picture_is_carried_inline(tmp_path):
    """
    A picture is carried in the message rather than by URL, so nothing has to be hosted
    for a model to see it.
    """
    picture = tmp_path / "render.png"
    picture.write_bytes(b"not really a png")
    sent = ImagePart.from_file(picture).to_json()
    assert sent["type"] == PartKind.IMAGE.value
    assert sent["image_url"]["url"].startswith("data:image/png;base64,")


# %% what is worth asking again after


def test_the_service_being_busy_is_worth_asking_again_after():
    """
    A rate limit is the service being busy, not the question being wrong.
    """
    assert (
        ServiceFailure.RATE_LIMITED in VisionLanguageModel(model="any").asks_again_after
    )


def test_a_question_the_service_rejects_is_not_asked_again():
    """
    Asking a question the service will not answer a second time buys nothing.
    """
    asks_again_after = VisionLanguageModel(model="any").asks_again_after
    assert not any(one == 404 for one in asks_again_after)


def test_asking_without_a_credential_says_which_one_is_missing(monkeypatch):
    """
    A run that would spend a hundred questions should stop on the first one, saying what
    to set rather than what failed.
    """
    model = VisionLanguageModel(model="any")
    monkeypatch.delenv(model.api_key_variable, raising=False)
    with pytest.raises(ApiKeyMissingError) as raised:
        model.ask([TextPart("anything")], system="anything")
    assert raised.value.variable == model.api_key_variable
