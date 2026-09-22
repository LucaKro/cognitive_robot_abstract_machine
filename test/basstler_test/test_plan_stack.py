"""
Tests for plan_stack.py: an item's unlanded dependencies as the chain of branches below it
in a GitHub stack, and the ``gh stack link`` command registering it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from basstler import plan_stack as plan_stack_module
from basstler.build_dashboard import Item, ItemStatus, Plan, Track, Wave
from basstler.plan_stack import (
    ForeignRepositoryError,
    NonLinearStackError,
    UnknownItemError,
    plan_stack,
)

REPOSITORY = "owner/repo"


def make_item(identifier: str, status: ItemStatus = ItemStatus.IN_PROGRESS, **fields) -> Item:
    """
    :param identifier: The item's id, which is also its branch.
    :param status: The item's status.
    :return: An item on track-1.
    """
    return Item(title=identifier.title(), branch=identifier, track="track-1",
                status=status, id=identifier, **fields)


def make_plan(*items: Item) -> Plan:
    """
    :return: A one-track plan holding *items*.
    """
    return Plan(
        id="test-plan", title="Test Plan", description="desc",
        default_repository=REPOSITORY,
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=list(items),
    )


def test_an_item_with_nothing_unlanded_below_it_is_a_plain_pull_request():
    plan = make_plan(
        make_item("base", ItemStatus.DONE),
        make_item("top", ItemStatus.NOT_STARTED, depends_on=["base"]),
    )

    stack = plan_stack(plan, "top")

    assert stack.branches == ["top"]
    assert stack.base == "main"
    assert stack.link_command is None


def test_the_chain_runs_bottom_first_and_skips_landed_layers():
    plan = make_plan(
        make_item("landed", ItemStatus.DONE),
        make_item("bottom", depends_on=["landed"]),
        make_item("middle", ItemStatus.BLOCKED, depends_on=["bottom"]),
        make_item("top", ItemStatus.NOT_STARTED, depends_on=["middle", "landed"]),
    )

    stack = plan_stack(plan, "top", trunk="develop")

    assert stack.branches == ["bottom", "middle", "top"]
    assert stack.base == "middle"
    assert stack.link_command == "gh stack link --base develop bottom middle top"


def test_two_unlanded_dependencies_at_once_are_refused():
    plan = make_plan(
        make_item("left"),
        make_item("right"),
        make_item("top", ItemStatus.NOT_STARTED, depends_on=["left", "right"]),
    )

    with pytest.raises(NonLinearStackError) as raised:
        plan_stack(plan, "top")

    assert raised.value.dependencies == ["left", "right"]


def test_a_dependency_in_another_repository_is_refused():
    plan = make_plan(
        make_item("elsewhere", repository="other/repo"),
        make_item("top", ItemStatus.NOT_STARTED, depends_on=["elsewhere"]),
    )

    with pytest.raises(ForeignRepositoryError):
        plan_stack(plan, "top")


def test_an_unknown_item_names_the_ones_that_exist():
    with pytest.raises(UnknownItemError) as raised:
        plan_stack(make_plan(make_item("only")), "missing")

    assert raised.value.known == ["only"]


def test_the_command_prints_the_stack_as_json(tmp_path: Path, capsys: pytest.CaptureFixture):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump({
        "schema_version": 1, "id": "test-plan", "title": "Test Plan", "description": "d",
        "default_repository": REPOSITORY,
        "waves": [{"id": "wave-1", "name": "Wave One"}],
        "tracks": [{"id": "track-1", "name": "Track One", "wave": "wave-1"}],
        "items": [
            {"id": "bottom", "title": "Bottom", "branch": "bottom-branch", "track": "track-1",
             "status": "in_progress"},
            {"id": "top", "title": "Top", "branch": "top-branch", "track": "track-1",
             "status": "not_started", "depends_on": ["bottom"]},
        ],
    }))

    assert plan_stack_module.main(["--plan", str(plan_path), "--item", "top"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "base": "bottom-branch",
        "branches": ["bottom-branch", "top-branch"],
        "link": "gh stack link --base main bottom-branch top-branch",
    }


def test_the_command_reports_a_forked_dependency_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump({
        "schema_version": 1, "id": "test-plan", "title": "Test Plan", "description": "d",
        "default_repository": REPOSITORY,
        "waves": [{"id": "wave-1", "name": "Wave One"}],
        "tracks": [{"id": "track-1", "name": "Track One", "wave": "wave-1"}],
        "items": [
            {"id": "left", "title": "L", "branch": "left", "track": "track-1", "status": "in_progress"},
            {"id": "right", "title": "R", "branch": "right", "track": "track-1", "status": "in_progress"},
            {"id": "top", "title": "T", "branch": "top", "track": "track-1",
             "status": "not_started", "depends_on": ["left", "right"]},
        ],
    }))

    assert plan_stack_module.main(["--plan", str(plan_path), "--item", "top"]) == 1
    assert "single line" in capsys.readouterr().err
