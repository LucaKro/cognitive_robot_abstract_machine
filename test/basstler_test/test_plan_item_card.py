"""
Tests for plan_item_card.py: the few lines about the current branch's plan item that
session-start.sh writes into CLAUDE.local.md instead of the whole manifest and roadmap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from basstler import plan_item_card
from basstler.build_dashboard import Item, ItemStatus, Plan, Track, Wave
from basstler.plan_item_brief import MAXIMUM_BODY_CHARACTERS
from basstler.plan_item_card import ItemNotTrackedError, item_for_branch, render_item_card

REPOSITORY = "owner/repo"


def make_plan() -> Plan:
    """
    :return: A plan with a dependency chain, one item carrying a pull request and notes.
    """
    return Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository=REPOSITORY,
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            Item(title="Base", branch="base-branch", track="track-1",
                 status=ItemStatus.DONE, id="base"),
            Item(title="Current work", branch="current-branch", track="track-1",
                 status=ItemStatus.IN_PROGRESS, id="current", pull_request_number=17,
                 depends_on=["base"], notes="n" * (MAXIMUM_BODY_CHARACTERS * 2),
                 blockers=["waiting on review"]),
        ],
    )


def test_the_item_is_found_by_its_branch():
    assert item_for_branch(make_plan(), "current-branch").identifier == "current"


def test_a_branch_no_item_tracks_is_refused():
    with pytest.raises(ItemNotTrackedError):
        item_for_branch(make_plan(), "someone-elses-branch")


def test_the_card_carries_the_item_and_nothing_about_other_items():
    plan = make_plan()
    card = render_item_card(plan, item_for_branch(plan, "current-branch"))

    assert "current: Current work" in card
    assert "in_progress" in card and "Track One (Wave One)" in card
    assert "depends on: base" in card
    assert f"{REPOSITORY}#17" in card
    assert "waiting on review" in card
    assert "Base" not in card.replace("base", "")


def test_long_notes_are_cut_short():
    plan = make_plan()
    card = render_item_card(plan, item_for_branch(plan, "current-branch"))

    assert "n" * MAXIMUM_BODY_CHARACTERS + "…" in card
    assert "n" * (MAXIMUM_BODY_CHARACTERS + 1) not in card


def test_the_command_exits_distinctly_when_no_item_tracks_the_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump({
        "schema_version": 1, "id": "p", "title": "P", "description": "d",
        "default_repository": REPOSITORY,
        "waves": [{"id": "w", "name": "W"}],
        "tracks": [{"id": "t", "name": "T", "wave": "w"}],
        "items": [{"id": "a", "title": "A", "branch": "a-branch", "track": "t",
                   "status": "not_started"}],
    }))

    assert plan_item_card.main(["--plan", str(plan_path), "--branch", "a-branch"]) == 0
    assert "a: A" in capsys.readouterr().out
    assert plan_item_card.main(["--plan", str(plan_path), "--branch", "other"]) == (
        plan_item_card.EXIT_ITEM_NOT_TRACKED
    )
