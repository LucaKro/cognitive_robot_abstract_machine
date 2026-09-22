#!/usr/bin/env python3
"""
Register a plan item's pull request as the top layer of its GitHub stack.

Goes through the Stacks REST API only, so it runs wherever GitHub's REST API does,
including a cloud session whose proxy refuses the GraphQL API ``gh stack link`` needs.
The pull requests of the layers below come from the plan's ``pull_request_number``
fields; the item's own is passed in, since it is created just before this runs.

Usage:
    python3 -m basstler.stack_registration --plan /tmp/plan.yaml --item <item-id> \\
        --pull-request-number <number> [--trunk main]

Prints one JSON document: ``number`` (the stack's number, ``null`` when nothing unlanded
lies below the item and there is no stack) and ``pull_requests`` (the stack, bottom
first).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from basstler.build_dashboard import Plan, PlanValidationError, validate_plan
from basstler.github_api import (
    GitHubApi,
    GitHubApiRequestFailedError,
    GitHubCredentialUnavailableError,
)
from basstler.plan_stack import (
    DEFAULT_TRUNK,
    ForeignRepositoryError,
    NonLinearStackError,
    PlanStack,
    UnknownItemError,
    plan_stack,
)

# %% failures


@dataclass
class MissingPullRequestError(ValueError):
    """
    Raised when a layer below the item has no pull request recorded in the plan, so the
    stack cannot name it.
    """

    identifier: str
    """The layer without a pull request."""

    def __str__(self) -> str:
        """:return: Which layer, and what to record."""
        return (
            f"item {self.identifier!r} is a layer below but has no pull_request_number "
            "in the plan; record its pull request first"
        )


@dataclass
class StackContinuesAboveError(ValueError):
    """
    Raised when the layer below already has pull requests stacked on top of it, so the
    item cannot become the top layer.
    """

    below: int
    """The pull request directly below the item."""

    above: list[int]
    """
    The pull requests the stack already carries above it.
    """

    def __str__(self) -> str:
        """:return: Which pull requests are in the way."""
        return (
            f"pull request #{self.below} already has {', '.join(f'#{n}' for n in self.above)} "
            "stacked on it; a stack is a single line"
        )


# %% the stack on GitHub


@dataclass(frozen=True)
class RegisteredStack:
    """
    A stack as GitHub holds it.
    """

    number: int
    """
    The stack's number in its repository.
    """

    pull_requests: list[int]
    """
    Its pull requests, bottom first.
    """

    @classmethod
    def from_json(cls, representation: dict[str, Any]) -> RegisteredStack:
        """
        :param representation: A stack as the Stacks REST API returns it.
        :return: The stack.
        """
        return cls(
            number=representation["number"],
            pull_requests=[
                entry["number"] for entry in representation["pull_requests"]
            ],
        )

    def as_document(self) -> dict[str, object]:
        """:return: What the command prints."""
        return {"number": self.number, "pull_requests": self.pull_requests}


@dataclass(frozen=True)
class StackRegistry:
    """
    The stacks of pull requests in GitHub repositories.
    """

    api: GitHubApi
    """
    The REST client every call goes through.
    """

    def stack_containing(
        self, repository: str, pull_request: int
    ) -> RegisteredStack | None:
        """
        :param repository: The ``owner/name`` the pull request lives in.
        :param pull_request: A pull request number.
        :return: The stack holding it, or ``None`` when it is in none.
        """
        found = self.api.get(f"/repos/{repository}/stacks?pull_request={pull_request}")
        return RegisteredStack.from_json(found[0]) if found else None

    def register(
        self, stack: PlanStack, pull_request_number: int
    ) -> RegisteredStack | None:
        """
        Put the item's pull request on top of the stack its layers below form, creating
        that stack when none of them is in one yet. Registering again changes nothing.

        :param stack: The item's stack in the plan.
        :param pull_request_number: The item's own pull request.
        :return: The stack on GitHub, or ``None`` when the item is a single layer.
        :raises MissingPullRequestError: If a layer below has no pull request recorded.
        :raises StackContinuesAboveError: If the layer below already has others on it.
        """
        below_layers = stack.layers[:-1]
        if not below_layers:
            return None
        for layer in below_layers:
            if layer.pull_request_number is None:
                raise MissingPullRequestError(layer.identifier)
        below = [layer.pull_request_number for layer in below_layers]

        existing = self.stack_containing(stack.repository, below[-1])
        if existing is None:
            created = self.api.post(
                f"/repos/{stack.repository}/stacks",
                {"pull_requests": [*below, pull_request_number]},
            )
            return RegisteredStack.from_json(created)
        if pull_request_number in existing.pull_requests:
            return existing
        above = existing.pull_requests[existing.pull_requests.index(below[-1]) + 1 :]
        if above:
            raise StackContinuesAboveError(below=below[-1], above=above)
        extended = self.api.post(
            f"/repos/{stack.repository}/stacks/{existing.number}/add",
            {"pull_requests": [pull_request_number]},
        )
        return RegisteredStack.from_json(extended)


# %% the command


def main(arguments: Sequence[str] | None = None, api: GitHubApi | None = None) -> int:
    """
    Register an item's pull request in its stack, as the module docstring describes.

    :param arguments: The command line, defaulting to the process's.
    :param api: The REST client, defaulting to one authenticated from the environment.
    :return: The exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, help="Path to plan.yaml")
    parser.add_argument("--item", required=True, help="The item's id")
    parser.add_argument(
        "--pull-request-number", required=True, type=int, help="The item's pull request"
    )
    parser.add_argument(
        "--trunk", default=DEFAULT_TRUNK, help="The branch the stack targets"
    )
    parsed = parser.parse_args(arguments)

    raw_plan = yaml.safe_load(Path(parsed.plan).read_text())
    try:
        validate_plan(raw_plan)
        stack = plan_stack(Plan.from_mapping(raw_plan), parsed.item, parsed.trunk)
        registry = StackRegistry(api or GitHubApi.from_environment())
        registered = registry.register(stack, parsed.pull_request_number)
    except (
        PlanValidationError,
        UnknownItemError,
        NonLinearStackError,
        ForeignRepositoryError,
        MissingPullRequestError,
        StackContinuesAboveError,
        GitHubCredentialUnavailableError,
        GitHubApiRequestFailedError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    document = (
        registered.as_document()
        if registered is not None
        else {"number": None, "pull_requests": [parsed.pull_request_number]}
    )
    print(json.dumps(document))
    return 0


if __name__ == "__main__":
    sys.exit(main())
