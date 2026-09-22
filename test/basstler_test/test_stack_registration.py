"""
Tests for stack_registration.py: registering a plan item's pull request as the top layer
of its GitHub stack through the Stacks REST API.

Nothing here reaches the network: every request goes to an in-memory stand-in for the
stacks endpoints.
"""

from __future__ import annotations

import json
import re
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from basstler import stack_registration as stack_registration_module
from basstler.build_dashboard import Item, ItemStatus, Plan, Track, Wave
from basstler.github_api import GitHubApi
from basstler.plan_stack import plan_stack
from basstler.stack_registration import (
    MissingPullRequestError,
    RegisteredStack,
    StackContinuesAboveError,
    StackRegistry,
)

REPOSITORY = "owner/repo"


# %% fixtures


@dataclass
class ResponseStandIn:
    """
    Stands in for what :func:`urllib.request.urlopen` returns, as a context manager.
    """

    body: bytes
    """
    What ``read()`` hands back.
    """

    def read(self) -> bytes:
        """:return: The body."""
        return self.body

    def __enter__(self) -> ResponseStandIn:
        return self

    def __exit__(self, *exception_information: object) -> None:
        return None


@dataclass
class StacksEndpointStandIn:
    """
    Answers the three Stacks REST calls registration makes from an in-memory list of
    stacks, recording every request.
    """

    stacks: list[list[int]] = field(default_factory=list)
    """
    Each stack's pull request numbers, bottom first; a stack's number is its index + 1.
    """

    requests: list[tuple[str, str, Any]] = field(default_factory=list)
    """
    Every request as method, path and decoded body.
    """

    def __call__(self, request) -> ResponseStandIn:
        path = request.full_url.removeprefix("https://api.github.com")
        payload = json.loads(request.data) if request.data else None
        self.requests.append((request.get_method(), path, payload))

        listing = re.fullmatch(rf"/repos/{REPOSITORY}/stacks\?pull_request=(\d+)", path)
        if listing:
            number = int(listing.group(1))
            found = [
                self.representation(index)
                for index, pull_requests in enumerate(self.stacks)
                if number in pull_requests
            ]
            return ResponseStandIn(json.dumps(found).encode())
        if path == f"/repos/{REPOSITORY}/stacks":
            self.stacks.append(list(payload["pull_requests"]))
            return self.respond_with(len(self.stacks) - 1)
        addition = re.fullmatch(rf"/repos/{REPOSITORY}/stacks/(\d+)/add", path)
        if addition:
            index = int(addition.group(1)) - 1
            self.stacks[index].extend(payload["pull_requests"])
            return self.respond_with(index)
        raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)

    def representation(self, index: int) -> dict[str, Any]:
        """:return: Stack *index* as the API represents it."""
        return {
            "number": index + 1,
            "pull_requests": [{"number": number} for number in self.stacks[index]],
        }

    def respond_with(self, index: int) -> ResponseStandIn:
        """:return: Stack *index* as a response."""
        return ResponseStandIn(json.dumps(self.representation(index)).encode())

    @property
    def writes(self) -> list[tuple[str, str, Any]]:
        """
        Every request that was not a read.
        """
        return [request for request in self.requests if request[0] != "GET"]


def make_item(
    identifier: str,
    pull_request_number: int | None = None,
    depends_on: list[str] | None = None,
) -> Item:
    """
    :param identifier: The item's id, which is also its branch.
    :param pull_request_number: The item's tracked pull request.
    :param depends_on: The items it builds on.
    :return: An in-progress item on track-1.
    """
    return Item(
        title=identifier.title(),
        branch=identifier,
        track="track-1",
        status=ItemStatus.IN_PROGRESS,
        id=identifier,
        pull_request_number=pull_request_number,
        depends_on=depends_on or [],
    )


def make_plan(*items: Item) -> Plan:
    """:return: A one-track plan holding *items*."""
    return Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository=REPOSITORY,
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=list(items),
    )


THREE_LAYER_PLAN = make_plan(
    make_item("bottom", 10),
    make_item("middle", 11, depends_on=["bottom"]),
    make_item("top", depends_on=["middle"]),
)
"""
A plan whose ``top`` item rests on two unlanded layers with pull requests.
"""


def registry_for(endpoint: StacksEndpointStandIn) -> StackRegistry:
    """:return: A registry sending every request to *endpoint*."""
    return StackRegistry(GitHubApi(token="secret", opener=endpoint))


# %% registering


def test_a_single_layer_is_no_stack_and_sends_nothing():
    endpoint = StacksEndpointStandIn()
    plan = make_plan(make_item("only"))

    registered = registry_for(endpoint).register(plan_stack(plan, "only"), 20)

    assert registered is None
    assert endpoint.requests == []


def test_layers_in_no_stack_yet_are_created_as_one_bottom_first():
    endpoint = StacksEndpointStandIn()

    registered = registry_for(endpoint).register(
        plan_stack(THREE_LAYER_PLAN, "top"), 12
    )

    assert endpoint.writes == [
        ("POST", f"/repos/{REPOSITORY}/stacks", {"pull_requests": [10, 11, 12]})
    ]
    assert registered == RegisteredStack(number=1, pull_requests=[10, 11, 12])


def test_a_stack_topped_by_the_layer_below_is_extended():
    endpoint = StacksEndpointStandIn(stacks=[[10, 11]])

    registered = registry_for(endpoint).register(
        plan_stack(THREE_LAYER_PLAN, "top"), 12
    )

    assert endpoint.writes == [
        ("POST", f"/repos/{REPOSITORY}/stacks/1/add", {"pull_requests": [12]})
    ]
    assert registered == RegisteredStack(number=1, pull_requests=[10, 11, 12])


def test_registering_again_changes_nothing():
    endpoint = StacksEndpointStandIn(stacks=[[10, 11, 12]])

    registered = registry_for(endpoint).register(
        plan_stack(THREE_LAYER_PLAN, "top"), 12
    )

    assert endpoint.writes == []
    assert registered == RegisteredStack(number=1, pull_requests=[10, 11, 12])


def test_a_stack_that_already_continues_above_the_layer_below_is_refused():
    endpoint = StacksEndpointStandIn(stacks=[[10, 11, 30]])

    with pytest.raises(StackContinuesAboveError) as raised:
        registry_for(endpoint).register(plan_stack(THREE_LAYER_PLAN, "top"), 12)

    assert raised.value.above == [30]
    assert endpoint.writes == []


def test_a_lower_layer_without_a_pull_request_is_refused_before_any_request():
    endpoint = StacksEndpointStandIn()
    plan = make_plan(make_item("bottom"), make_item("top", depends_on=["bottom"]))

    with pytest.raises(MissingPullRequestError) as raised:
        registry_for(endpoint).register(plan_stack(plan, "top"), 12)

    assert raised.value.identifier == "bottom"
    assert endpoint.requests == []


# %% the command


def test_the_command_prints_the_registered_stack(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "test-plan",
                "title": "Test Plan",
                "description": "d",
                "default_repository": REPOSITORY,
                "waves": [{"id": "wave-1", "name": "Wave One"}],
                "tracks": [{"id": "track-1", "name": "Track One", "wave": "wave-1"}],
                "items": [
                    {
                        "id": "bottom",
                        "title": "Bottom",
                        "branch": "bottom-branch",
                        "track": "track-1",
                        "status": "in_progress",
                        "pull_request_number": 10,
                    },
                    {
                        "id": "top",
                        "title": "Top",
                        "branch": "top-branch",
                        "track": "track-1",
                        "status": "in_progress",
                        "depends_on": ["bottom"],
                    },
                ],
            }
        )
    )
    endpoint = StacksEndpointStandIn()

    exit_code = stack_registration_module.main(
        ["--plan", str(plan_path), "--item", "top", "--pull-request-number", "11"],
        api=GitHubApi(token="secret", opener=endpoint),
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "number": 1,
        "pull_requests": [10, 11],
    }
