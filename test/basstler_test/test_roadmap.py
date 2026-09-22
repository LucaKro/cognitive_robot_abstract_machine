"""
Tests for roadmap.py: reading a plan's roadmap by section - which sections are plan-wide,
which belong to an item, and where GitHub renders each one.
"""

from __future__ import annotations

from basstler.build_dashboard import Item, ItemStatus, Plan, Track, Wave
from basstler.roadmap import (
    MAXIMUM_OWN_HISTORY_CHARACTERS,
    ItemHistory,
    github_heading_anchor,
    item_histories,
    level_two_anchors,
    plan_wide_markdown,
    render_roadmap_selection,
    roadmap_section,
    select_roadmap_sections,
)


def make_item(identifier: str) -> Item:
    """:return: An item whose id, branch and title are all *identifier*."""
    return Item(title=identifier, branch=identifier, track="track-1",
                status=ItemStatus.IN_PROGRESS, id=identifier)


def make_plan(items: list[Item]) -> Plan:
    """:return: A one-track plan holding *items*."""
    return Plan(id="test-plan", title="Test Plan", description="desc",
                default_repository="owner/repo",
                waves=[Wave(id="wave-1", name="Wave One")],
                tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
                items=items)


# %% selecting what an item's brief carries


ROADMAP = """# Test plan roadmap

Intro paragraph.

## Why this plan exists

Plan-wide rationale.

## `item`

Kickoff notes for the item.

## `other`

Kickoff notes for another item.

## `item` — first review round

Review of the item.
"""


def test_the_roadmap_contributes_plan_wide_sections_and_the_items_own_only():
    plan = make_plan([make_item("item"), make_item("other")])

    selection = select_roadmap_sections(ROADMAP, plan, plan.items[0])
    text = render_roadmap_selection(selection)

    assert "Plan-wide rationale." in text and "Intro paragraph." in text
    assert "Kickoff notes for the item." in text and "Review of the item." in text
    assert "Kickoff notes for another item." not in text
    assert "## `other`" in text


def test_an_item_is_recognized_by_its_branch_in_a_heading():
    item = Item(title="Renamed", branch="the-branch", track="track-1",
                status=ItemStatus.IN_PROGRESS, id="the-id")
    plan = make_plan([item])

    selection = select_roadmap_sections("## `the-branch` — resolution\n\nResolved.\n", plan, item)

    assert [section.heading for section in selection.own] == ["## `the-branch` — resolution"]


def test_an_over_long_history_keeps_its_newest_sections_whole():
    sections = "".join(
        f"## `item` — round {index}\n\n{'y' * (MAXIMUM_OWN_HISTORY_CHARACTERS // 3)}\n\n"
        for index in range(6)
    )
    plan = make_plan([make_item("item")])

    selection = select_roadmap_sections(sections, plan, plan.items[0])

    assert selection.own[-1].heading == "## `item` — round 5"
    assert sum(len(section.text) for section in selection.own) <= MAXIMUM_OWN_HISTORY_CHARACTERS
    assert "## `item` — round 0" in selection.omitted_own_headings


def test_one_section_can_be_pulled_by_its_heading():
    assert roadmap_section(ROADMAP, "## `other`").strip() == (
        "## `other`\n\nKickoff notes for another item."
    )



# %% where GitHub renders each section


def test_an_anchor_drops_punctuation_and_joins_words_with_hyphens():
    assert github_heading_anchor("`grasp-belief-node` — first review round", {}) == (
        "grasp-belief-node--first-review-round"
    )


def test_a_repeated_heading_is_numbered_the_way_github_numbers_it():
    used: dict[str, int] = {}
    assert [github_heading_anchor("Notes", used) for _ in range(3)] == ["notes", "notes-1", "notes-2"]


def test_headings_of_every_level_count_and_fenced_lines_do_not():
    roadmap = (
        "# Notes\n\n"
        "```\n## Notes\n```\n\n"
        "## Notes\n"
    )

    assert level_two_anchors(roadmap) == {"## Notes": "notes-1"}


def test_an_items_history_is_counted_and_located_by_its_first_section():
    plan = make_plan([make_item("item"), make_item("other")])

    assert item_histories(ROADMAP, plan) == {
        "item": ItemHistory(section_count=2, first_anchor="item"),
        "other": ItemHistory(section_count=1, first_anchor="other"),
    }


def test_the_plan_wide_text_leaves_every_items_sections_out():
    plan = make_plan([make_item("item"), make_item("other")])

    text = plan_wide_markdown(ROADMAP, plan)

    assert "Plan-wide rationale." in text and "Intro paragraph." in text
    assert "Kickoff notes" not in text and "Review of the item." not in text
