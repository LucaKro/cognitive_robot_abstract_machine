#!/usr/bin/env python3
"""
Where a plan item's branch sits in a GitHub stack of pull requests.

Stacks are GitHub's own (``gh stack``): GitHub keeps each layer's base, rebases the layers
above a changed one with ``gh stack rebase --upstack``, and retargets them when one lands.
The plan only says which items build on which; this module turns an item's unlanded
dependencies into the chain of branches below it.

A GitHub stack is a single line of branches. An item whose unlanded dependencies fork
(two of them, neither building on the other) cannot be one layer of a stack, and is
refused rather than stacked on an arbitrary one of them.

Usage:
    python3 -m basstler.plan_stack --plan /tmp/plan.yaml --item <item-id> [--trunk main]

Prints one JSON document: ``base`` (the branch to create the item's branch from and open
its pull request against) and ``branches`` (the stack, bottom first, ending with the
item's own; just the item's own when it is a plain pull request against the trunk).
:mod:`basstler.stack_registration` registers the stack on GitHub once the item's pull
request exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from basstler.build_dashboard import (
    Item,
    ItemStatus,
    Plan,
    PlanValidationError,
    validate_plan,
)

DEFAULT_TRUNK = "main"
"""
The branch a stack's bottom layer targets unless told otherwise.
"""


@dataclass
class UnknownItemError(LookupError):
    """
    Raised when the plan has no item of the requested id.
    """

    identifier: str
    """The id that was looked up."""

    known: list[str]
    """Every id the plan does have."""

    def __str__(self) -> str:
        """:return: The unknown id and the ones that exist."""
        return f"no item {self.identifier!r}; the plan has: {', '.join(self.known)}"


@dataclass
class NonLinearStackError(ValueError):
    """
    Raised when an item rests on more than one unlanded dependency at once, which a stack
    of pull requests cannot express.
    """

    identifier: str
    """The item whose dependencies fork."""

    dependencies: list[str]
    """Its unlanded dependencies."""

    def __str__(self) -> str:
        """:return: Which item and which dependencies."""
        return (
            f"item {self.identifier!r} rests on {len(self.dependencies)} unlanded items at once "
            f"({', '.join(self.dependencies)}); a stack is a single line. Land or merge one "
            "of them first, or change the plan's depends_on."
        )


@dataclass
class ForeignRepositoryError(ValueError):
    """
    Raised when a dependency lives in another repository; stacks do not span repositories.
    """

    identifier: str
    """The dependency in the other repository."""

    repository: str
    """The repository it lives in."""

    def __str__(self) -> str:
        """:return: Which dependency and where it lives."""
        return (
            f"dependency {self.identifier!r} lives in {self.repository}; "
            "a stack cannot span repositories"
        )


@dataclass(frozen=True)
class PlanStack:
    """
    The stack an item's branch belongs to.
    """

    trunk: str
    """The branch the bottom layer targets."""

    repository: str
    """The repository every layer's pull request lives in."""

    layers: list[Item]
    """The stack's items, bottom first, ending with the item itself."""

    @property
    def branches(self) -> list[str]:
        """The stack's branches, bottom first, ending with the item's own."""
        return [layer.branch for layer in self.layers]

    @property
    def base(self) -> str:
        """The branch directly below the item's: its branch point and pull request base."""
        return self.branches[-2] if len(self.branches) > 1 else self.trunk

    def as_document(self) -> dict[str, object]:
        """:return: What the command prints."""
        return {"base": self.base, "branches": self.branches}


def is_landed(item: Item) -> bool:
    """
    :param item: A dependency.
    :return: Whether its work is on the trunk already, and so is no layer of the stack.
    """
    return item.status == ItemStatus.DONE


def plan_stack(plan: Plan, identifier: str, trunk: str = DEFAULT_TRUNK) -> PlanStack:
    """
    Follow an item's unlanded dependencies down to the trunk.

    :param plan: The plan.
    :param identifier: The item's id.
    :param trunk: The branch the bottom layer targets.
    :return: The item's stack.
    :raises UnknownItemError: If the plan has no such item.
    :raises NonLinearStackError: If some layer rests on more than one unlanded item.
    :raises ForeignRepositoryError: If a layer lives in another repository.
    """
    items = {item.identifier: item for item in plan.items}
    if identifier not in items:
        raise UnknownItemError(identifier, sorted(items))
    top = items[identifier]
    repository = top.repository or plan.default_repository

    layers: list[Item] = []
    current: Item | None = top
    while current is not None:
        layers.append(current)
        unlanded = [
            items[name] for name in current.depends_on if not is_landed(items[name])
        ]
        if len(unlanded) > 1:
            raise NonLinearStackError(
                current.identifier, [item.identifier for item in unlanded]
            )
        current = unlanded[0] if unlanded else None
        if current is not None:
            dependency_repository = current.repository or plan.default_repository
            if dependency_repository != repository:
                raise ForeignRepositoryError(current.identifier, dependency_repository)
    return PlanStack(trunk=trunk, repository=repository, layers=list(reversed(layers)))


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Print an item's stack. See the module docstring.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, help="Path to plan.yaml")
    parser.add_argument("--item", required=True, help="The item's id")
    parser.add_argument(
        "--trunk", default=DEFAULT_TRUNK, help="The branch the stack targets"
    )
    parsed = parser.parse_args(arguments)

    raw_plan = yaml.safe_load(Path(parsed.plan).read_text())
    try:
        validate_plan(raw_plan)
    except PlanValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    try:
        stack = plan_stack(Plan.from_mapping(raw_plan), parsed.item, parsed.trunk)
    except (UnknownItemError, NonLinearStackError, ForeignRepositoryError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(stack.as_document()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
