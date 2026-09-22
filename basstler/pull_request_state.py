#!/usr/bin/env python3
"""
The live state of exactly the pull requests a plan's items reference, in the shape
:mod:`basstler.build_dashboard` and :mod:`basstler.check_dependency_readiness` read.

Only the four fields those readers use survive - see build_dashboard's module docstring
for the shape. A pull request's full API representation runs to tens of kilobytes, and a
session that fetched every pull request of the repository and transcribed the result
spent most of a /plan-item-resolve run on that alone, before touching any code. Here the
fetching and the reduction happen in one call, and nothing but the reduced file and a
one-line summary reaches the session.

Usage:
    python3 -m basstler.pull_request_state --plan /tmp/plan.yaml --output /tmp/pr_data.json

Prints a one-line JSON summary: ``{"pull_requests": <written>, "not_found":
["owner/repo#n", ...]}``. A pull request that does not exist is left out of the file,
which the dashboard reads as ``not_found``.
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from basstler.build_dashboard import Plan, PlanValidationError, validate_plan
from basstler.github_api import GitHubApi

PullRequestStatesByRepository = dict[str, dict[str, dict[str, Any]]]
"""
``pr_data.json``: repository, then pull request number as a string, then its state.
"""


# %% which pull requests


@dataclass(frozen=True, order=True)
class PullRequestReference:
    """
    One pull request, named by where it lives.
    """

    repository: str
    """
    ``owner/repo``.
    """

    number: int
    """
    Its number in that repository.
    """

    def __str__(self) -> str:
        """:return: The ``owner/repo#number`` form GitHub itself uses."""
        return f"{self.repository}#{self.number}"


def referenced_pull_requests(plan: Plan) -> list[PullRequestReference]:
    """
    :param plan: The plan to read.
    :return: Every pull request an item names, each once, in a stable order.
    """
    return sorted(
        {
            PullRequestReference(
                item.repository or plan.default_repository, item.pull_request_number
            )
            for item in plan.items
            if item.pull_request_number is not None
        }
    )


# %% where their state comes from


class PullRequestSource(ABC):
    """
    Answers what one pull request looks like right now.
    """

    @abstractmethod
    def pull_request(self, reference: PullRequestReference) -> Mapping[str, Any] | None:
        """
        :param reference: The pull request to read.
        :return: Its API representation, or ``None`` if it does not exist.
        """


@dataclass(frozen=True)
class GitHubPullRequestSource(PullRequestSource):
    """
    Reads pull requests from the REST API.
    """

    api: GitHubApi
    """
    The client the reads go through.
    """

    def pull_request(self, reference: PullRequestReference) -> Mapping[str, Any] | None:
        return self.api.get(f"/repos/{reference.repository}/pulls/{reference.number}")


def default_pull_request_source() -> PullRequestSource:
    """
    :return: A source reading from GitHub with this environment's credential.
    """
    return GitHubPullRequestSource(GitHubApi.from_environment())


# %% what is kept


def pull_request_state(representation: Mapping[str, Any]) -> dict[str, Any]:
    """
    :param representation: A pull request as the API returns it.
    :return: The four fields the dashboard reads. ``merged_at`` is always present,
        ``None`` included, because the dashboard refuses a closed entry without it.
    """
    return {
        "state": representation["state"],
        "draft": bool(representation.get("draft", False)),
        "merged_at": representation.get("merged_at"),
        "labels": [label["name"] for label in representation.get("labels", [])],
    }


@dataclass
class PullRequestStates:
    """
    What was found for a plan's pull requests.
    """

    by_repository: PullRequestStatesByRepository = field(default_factory=dict)
    """
    The content of ``pr_data.json``.
    """

    not_found: list[PullRequestReference] = field(default_factory=list)
    """
    Pull requests an item names that do not exist.
    """

    def summary(self) -> dict[str, Any]:
        """:return: The one-line report a session reads instead of the file."""
        return {
            "pull_requests": sum(
                len(pull_requests) for pull_requests in self.by_repository.values()
            ),
            "not_found": [str(reference) for reference in self.not_found],
        }


def collect_pull_request_states(
    plan: Plan, source: PullRequestSource
) -> PullRequestStates:
    """
    :param plan: The plan whose pull requests to read.
    :param source: Where each pull request's state comes from.
    :return: The state of every pull request the plan's items name.
    """
    states = PullRequestStates()
    for reference in referenced_pull_requests(plan):
        representation = source.pull_request(reference)
        if representation is None:
            states.not_found.append(reference)
            continue
        states.by_repository.setdefault(reference.repository, {})[
            str(reference.number)
        ] = pull_request_state(representation)
    return states


# %% the command


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Fetch the plan's pull request states, write them, and print the summary.

    See the module docstring for the command-line contract.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, help="Path to plan.yaml")
    parser.add_argument("--output", required=True, help="Where to write pr_data.json")
    parsed = parser.parse_args(arguments)

    raw_plan = yaml.safe_load(Path(parsed.plan).read_text())
    try:
        validate_plan(raw_plan)
    except PlanValidationError as error:
        print(str(error), file=sys.stderr)
        return 1

    states = collect_pull_request_states(
        Plan.from_mapping(raw_plan), default_pull_request_source()
    )
    Path(parsed.output).write_text(json.dumps(states.by_repository, indent=2))
    print(json.dumps(states.summary()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
