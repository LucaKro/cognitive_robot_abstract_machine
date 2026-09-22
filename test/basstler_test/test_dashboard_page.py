"""
Tests for what build_dashboard.py puts into the published page and what it leaves beside
it: the stylesheet and script as separate, content-named files, the roadmap's plan-wide
sections only, and a link from each item to its own history where GitHub renders it.

Every byte of the page is read back by the session that publishes it, so what does not
change from one publish to the next stays out of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from basstler import build_dashboard
from basstler.build_dashboard import (
    DashboardRenderer,
    Item,
    ItemStatus,
    Plan,
    Track,
    Wave,
    dashboard_assets,
)

ROADMAP = """# Roadmap

## Why this plan exists

Plan-wide rationale.

## `first`

Kickoff record of the first item.

## `first` — resolution

Resolution record of the first item.
"""

ROADMAP_URL = "https://github.com/owner/repo/blob/notes/plans/test-plan/roadmap.md"


def make_renderer(roadmap_url: str | None = ROADMAP_URL) -> DashboardRenderer:
    """
    :param roadmap_url: Where GitHub renders the roadmap, if known.
    :return: A renderer for a plan with one item that has history and one that has none.
    """
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            Item(title="First", branch="first", track="track-1",
                 status=ItemStatus.IN_PROGRESS, id="first"),
            Item(title="Second", branch="second", track="track-1",
                 status=ItemStatus.NOT_STARTED, id="second"),
        ],
    )
    return DashboardRenderer(
        plan=plan,
        roadmap_text=ROADMAP,
        pull_requests_by_repository={},
        tracking_url=None,
        roadmap_url=roadmap_url,
    )


def test_the_stylesheet_and_script_are_linked_by_their_content_names_not_inlined():
    page, _ = make_renderer().render()

    assert "<style>" not in page
    assert "<script>" not in page
    for asset in dashboard_assets():
        assert f'"{asset.name}"' in page


def test_an_asset_name_changes_exactly_when_its_content_does():
    first, second = dashboard_assets(), dashboard_assets()

    assert [asset.name for asset in first] == [asset.name for asset in second]
    assert len({asset.name for asset in first}) == len(first)


def test_only_the_plan_wide_sections_of_the_roadmap_are_in_the_page():
    page, _ = make_renderer().render()

    assert "Plan-wide rationale." in page
    assert "Kickoff record of the first item." not in page
    assert "Resolution record of the first item." not in page


def test_an_item_links_to_its_first_section_where_github_renders_the_roadmap():
    page, _ = make_renderer().render()

    assert f'href="{ROADMAP_URL}#first"' in page
    assert "history (2)" in page


def test_without_a_roadmap_url_an_item_only_says_how_much_history_it_has():
    page, _ = make_renderer(roadmap_url=None).render()

    assert "#first" not in page
    assert "2 roadmap sections" in page


def test_the_command_writes_the_assets_beside_the_page_and_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump({
        "schema_version": 1, "id": "test-plan", "title": "Test Plan", "description": "d",
        "default_repository": "owner/repo",
        "waves": [{"id": "wave-1", "name": "Wave One"}],
        "tracks": [{"id": "track-1", "name": "Track One", "wave": "wave-1"}],
        "items": [{"id": "first", "title": "First", "branch": "first", "track": "track-1",
                   "status": "in_progress"}],
    }))
    roadmap_path = tmp_path / "roadmap.md"
    roadmap_path.write_text(ROADMAP)
    pull_request_data_path = tmp_path / "pr_data.json"
    pull_request_data_path.write_text("{}")
    output_directory = tmp_path / "published"
    output_directory.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "build_dashboard", "--plan", str(plan_path), "--roadmap", str(roadmap_path),
        "--pr-data", str(pull_request_data_path),
        "--output", str(output_directory / "dashboard.html"),
    ])

    assert build_dashboard.main() == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["assets"] == [asset.name for asset in dashboard_assets()]
    for asset in dashboard_assets():
        assert (output_directory / asset.name).read_text() == asset.content
