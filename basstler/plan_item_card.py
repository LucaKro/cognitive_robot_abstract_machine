#!/usr/bin/env python3
"""
The few lines about the current branch's plan item that session-start.sh writes into
CLAUDE.local.md.

CLAUDE.local.md is part of every request a session makes. Copying the whole manifest
and roadmap into it put every other item's history and the plan's whole rationale in
front of a session working one small item, turn after turn. The card carries this item
only; the brief (:mod:`basstler.plan_item_brief`) and the roadmap are one command away
when a session actually needs them.

Usage:
    python3 -m basstler.plan_item_card --plan /tmp/plan.yaml --branch <branch>

Prints the card as Markdown, or exits with :data:`EXIT_ITEM_NOT_TRACKED` if no item
tracks the branch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from basstler.build_dashboard import Item, Plan, PlanValidationError, validate_plan
from basstler.plan_item_brief import shortened

EXIT_ITEM_NOT_TRACKED = 2
"""
Exit code when no item of the plan tracks the branch - distinct from a malformed plan,
so session-start.sh can tell the two apart.
"""


@dataclass
class ItemNotTrackedError(LookupError):
    """
    Raised when no item of the plan tracks the branch.
    """

    branch: str
    """The branch that was looked up."""

    def __str__(self) -> str:
        """:return: Which branch had no item."""
        return f"no item tracks branch {self.branch!r}"


def item_for_branch(plan: Plan, branch: str) -> Item:
    """
    :param plan: The plan to search.
    :param branch: The checked-out branch.
    :return: The item whose ``branch`` is *branch*.
    :raises ItemNotTrackedError: If there is none.
    """
    for item in plan.items:
        if item.branch == branch:
            return item
    raise ItemNotTrackedError(branch)


def render_item_card(plan: Plan, item: Item) -> str:
    """
    :param plan: The plan the item belongs to.
    :param item: The item to describe.
    :return: The card as Markdown.
    """
    tracks = {track.id: track for track in plan.tracks}
    waves = {wave.id: wave for wave in plan.waves}
    track = tracks.get(item.track)
    wave = waves.get(track.wave) if track else None
    where = track.name if track else item.track
    if wave:
        where += f" ({wave.name})"

    lines = [
        f"## Plan item {item.identifier}: {item.title}",
        f"- plan: {plan.title} (`{plan.id}`) · status: {item.status.value} · track: {where}",
        f"- depends on: {', '.join(item.depends_on) if item.depends_on else 'nothing'}",
    ]
    if item.pull_request_number is not None:
        repository = item.repository or plan.default_repository
        lines.append(f"- pull request: {repository}#{item.pull_request_number}")
    else:
        lines.append("- pull request: not opened yet")
    if item.notes:
        lines.append(f"- notes: {shortened(item.notes)}")
    lines.extend(f"- blocker: {blocker}" for blocker in item.blockers)
    return "\n".join(lines) + "\n"


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Print the card for the item tracking a branch. See the module docstring.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, help="Path to plan.yaml")
    parser.add_argument("--branch", required=True, help="The checked-out branch")
    parsed = parser.parse_args(arguments)

    raw_plan = yaml.safe_load(Path(parsed.plan).read_text())
    try:
        validate_plan(raw_plan)
    except PlanValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    plan = Plan.from_mapping(raw_plan)
    try:
        item = item_for_branch(plan, parsed.branch)
    except ItemNotTrackedError as error:
        print(str(error), file=sys.stderr)
        return EXIT_ITEM_NOT_TRACKED
    print(render_item_card(plan, item), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
