"""
Bounding how long one question may take, rather than how many times it is asked.

``requests`` takes a ``timeout``, and it is not a deadline: it bounds each socket
operation, so a service that sends a byte now and then resets it and the request waits
for as long as the service cares to hold the connection. Two runs were killed after
fifty and thirty minutes with no attempt written, both at the same question, which the
five attempts of three minutes the client allows cannot account for.

So the attempts are bounded by a wall-clock budget as well as by a count, and a question
that has spent its budget raises rather than waiting on a connection that may never
close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import requests

from semantic_digital_twin.adapters.vision_language_model.client import (
    VisionLanguageModel,
)

# %% a service that never answers


@dataclass
class ServiceThatHoldsTheLine:
    """
    A service standing in for one that keeps a connection open without answering.

    It raises what ``requests`` raises when a read times out, which the client treats as
    worth asking again -- correctly, since a read timeout usually is. What is under test
    is that asking again does not go on for ever.
    """

    asked: int = 0
    """
    How many times it was asked.
    """

    slept: list = field(default_factory=list)
    """
    How long the client waited between attempts.
    """

    def __call__(self, *arguments, **keywords):
        self.asked += 1
        raise requests.exceptions.Timeout("the service is still thinking")


# %% giving up


def test_a_question_stops_being_asked_once_its_budget_is_spent(monkeypatch) -> None:
    """
    The failure this bounds: attempts alone do not bound time, because each attempt's
    timeout is per read rather than per request.
    """
    service = ServiceThatHoldsTheLine()
    monkeypatch.setattr(requests, "post", service)
    monkeypatch.setattr(
        "semantic_digital_twin.adapters.vision_language_model.client.time.sleep",
        lambda seconds: service.slept.append(seconds),
    )
    spent = iter([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    monkeypatch.setattr(
        "semantic_digital_twin.adapters.vision_language_model.client.perf_counter",
        lambda: next(spent),
    )

    model = VisionLanguageModel(model="a-model", seconds_per_question=250)
    with pytest.raises(requests.exceptions.Timeout):
        model.ask([], system="")

    assert service.asked < model.maximum_attempts


def test_the_budget_does_not_cut_a_question_that_is_answering(monkeypatch) -> None:
    """
    A question answered inside its budget is untouched, so nothing a run already does
    becomes slower or more fragile.
    """

    class Answered:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            pass

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: Answered())
    model = VisionLanguageModel(model="a-model")
    assert model.ask([], system="") is not None
