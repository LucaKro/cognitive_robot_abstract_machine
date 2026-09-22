"""
Reading a plan's roadmap by section.

Kickoffs, resolutions, review rounds and restacks append their records to a plan's
roadmap.md, so it grows without bound - one real plan's reached 236 KB, nearly all of it
per-item history. Everything that reads it for one purpose goes through here: the item
brief carries the plan-wide sections and one item's own history, and the dashboard carries
the plan-wide sections and links each item to its history as GitHub renders it.

A section belongs to an item when its level-two heading starts with that item's id or
branch in backticks, which is how the bootstrap heads every section it appends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basstler.build_dashboard import Item, Plan

MAXIMUM_OWN_HISTORY_CHARACTERS = 24_000
"""
Most of the item's own roadmap history carried whole; older sections are listed by
heading instead.
"""

ITEM_SECTION_HEADING = re.compile(r"^## `([^`]+)`")
"""
How a roadmap section about one item is headed: its id or branch in backticks, as the
bootstrap writes it.
"""

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
"""
A Markdown heading line, as GitHub gives it an anchor.
"""

FENCE = re.compile(r"^\s*(```|~~~)")
"""
The start or end of a fenced code block, inside which a ``#`` line is not a heading.
"""


# %% sections


@dataclass(frozen=True)
class RoadmapSection:
    """
    One level-two section of a roadmap, or the preamble before the first one.
    """

    heading: str
    """The ``## ...`` line, or empty for the preamble."""

    text: str
    """The whole section, heading included."""


@dataclass
class RoadmapSelection:
    """
    What of the roadmap an item's brief carries, and what it only names.
    """

    plan_wide: list[RoadmapSection] = field(default_factory=list)
    """Sections about no single item: the plan's rationale, decisions, conventions."""

    own: list[RoadmapSection] = field(default_factory=list)
    """The item's own most recent sections, within the history budget."""

    omitted_own_headings: list[str] = field(default_factory=list)
    """The item's older sections that did not fit."""

    other_headings: list[str] = field(default_factory=list)
    """Every other item's sections."""


@dataclass
class UnknownRoadmapSectionError(LookupError):
    """
    Raised when no roadmap section has the requested heading.
    """

    heading: str
    """The heading asked for."""

    def __str__(self) -> str:
        """:return: Which heading was not found."""
        return f"no roadmap section headed {self.heading!r}"


def roadmap_sections(roadmap_text: str) -> list[RoadmapSection]:
    """
    :param roadmap_text: A whole roadmap.
    :return: Its preamble, if any, then each level-two section, in order.
    """
    sections: list[RoadmapSection] = []
    heading, lines = "", []
    for line in roadmap_text.splitlines(keepends=True):
        if line.startswith("## "):
            if heading or "".join(lines).strip():
                sections.append(RoadmapSection(heading, "".join(lines)))
            heading, lines = line.rstrip("\n"), [line]
        else:
            lines.append(line)
    if heading or "".join(lines).strip():
        sections.append(RoadmapSection(heading, "".join(lines)))
    return sections


def section_subject(section: RoadmapSection, plan: Plan) -> Item | None:
    """
    :return: The item a section is about, recognized by id or branch in its heading, or
        ``None`` for a section about no single item.
    """
    match = ITEM_SECTION_HEADING.match(section.heading)
    if match is None:
        return None
    named = match.group(1)
    return next(
        (item for item in plan.items if named in (item.identifier, item.branch)), None
    )


def select_roadmap_sections(roadmap_text: str, plan: Plan, item: Item) -> RoadmapSelection:
    """
    :param roadmap_text: The plan's whole roadmap.
    :param plan: The plan, to recognize which item a section is about.
    :param item: The item the brief is for.
    :return: The plan-wide sections, the item's newest sections within
        :data:`MAXIMUM_OWN_HISTORY_CHARACTERS`, and the headings of everything else.
    """
    selection = RoadmapSelection()
    own_sections: list[RoadmapSection] = []
    for section in roadmap_sections(roadmap_text):
        subject = section_subject(section, plan)
        if subject is None:
            selection.plan_wide.append(section)
        elif subject is item:
            own_sections.append(section)
        else:
            selection.other_headings.append(section.heading)

    remaining = MAXIMUM_OWN_HISTORY_CHARACTERS
    for index in range(len(own_sections) - 1, -1, -1):
        section = own_sections[index]
        if len(section.text) > remaining:
            selection.omitted_own_headings = [
                older.heading for older in own_sections[: index + 1]
            ]
            break
        selection.own.insert(0, section)
        remaining -= len(section.text)
    return selection


def roadmap_section(roadmap_text: str, heading: str) -> str:
    """
    :param roadmap_text: The plan's whole roadmap.
    :param heading: A section's ``## ...`` line, as the brief lists it.
    :return: That section.
    :raises UnknownRoadmapSectionError: If no section has that heading.
    """
    for section in roadmap_sections(roadmap_text):
        if section.heading == heading.strip():
            return section.text
    raise UnknownRoadmapSectionError(heading)


def render_roadmap_selection(selection: RoadmapSelection) -> str:
    """
    :param selection: What of the roadmap to carry.
    :return: It as Markdown, with the sections left out named at the end.
    """
    parts = ["# Roadmap: plan-wide sections\n"]
    parts += [section.text.rstrip() + "\n" for section in selection.plan_wide]
    parts.append("# Roadmap: this item's own history\n")
    parts += [section.text.rstrip() + "\n" for section in selection.own] or ["None yet.\n"]
    if selection.omitted_own_headings:
        parts.append("Older sections of this item, not included:")
        parts += [f"- {heading}" for heading in selection.omitted_own_headings]
        parts.append("")
    if selection.other_headings:
        parts.append(
            "Other items' sections, not included - pull one with "
            "`--roadmap <path> --section \"<heading>\"` if it bears on this item:"
        )
        parts += [f"- {heading}" for heading in selection.other_headings]
        parts.append("")
    return "\n".join(parts)


# %% where GitHub renders each section


def github_heading_anchor(heading_text: str, used: dict[str, int]) -> str:
    """
    The anchor GitHub gives a heading when it renders Markdown: lower-cased, stripped of
    punctuation, spaces turned into hyphens, and numbered when it repeats.

    :param heading_text: The heading's text, without its leading ``#`` characters.
    :param used: Anchors handed out so far in the document, updated in place.
    :return: The anchor, without ``#``.
    """
    slug = re.sub(r"[^\w\- ]", "", heading_text.strip().lower()).replace(" ", "-")
    repeat = used.get(slug, 0)
    used[slug] = repeat + 1
    return slug if repeat == 0 else f"{slug}-{repeat}"


def level_two_anchors(roadmap_text: str) -> dict[str, str]:
    """
    :param roadmap_text: A whole roadmap.
    :return: Each level-two heading line mapped to its GitHub anchor. Every heading of
        every level counts towards GitHub's numbering of repeats, and a ``#`` line inside
        a fenced code block is not a heading.
    """
    anchors: dict[str, str] = {}
    used: dict[str, int] = {}
    inside_fence = False
    for line in roadmap_text.splitlines():
        if FENCE.match(line):
            inside_fence = not inside_fence
            continue
        if inside_fence:
            continue
        match = MARKDOWN_HEADING.match(line)
        if match is None:
            continue
        anchor = github_heading_anchor(match.group(2), used)
        if len(match.group(1)) == 2:
            anchors.setdefault(line.rstrip(), anchor)
    return anchors


@dataclass(frozen=True)
class ItemHistory:
    """
    Where an item's own sections are in the roadmap.
    """

    section_count: int
    """How many sections are about the item."""

    first_anchor: str
    """The GitHub anchor of the first of them."""


def item_histories(roadmap_text: str, plan: Plan) -> dict[str, ItemHistory]:
    """
    :param roadmap_text: The plan's whole roadmap.
    :param plan: The plan, to recognize which item a section is about.
    :return: Each item that has sections, keyed by its identifier.
    """
    anchors = level_two_anchors(roadmap_text)
    counts: dict[str, int] = {}
    first: dict[str, str] = {}
    for section in roadmap_sections(roadmap_text):
        subject = section_subject(section, plan)
        if subject is None:
            continue
        counts[subject.identifier] = counts.get(subject.identifier, 0) + 1
        first.setdefault(subject.identifier, anchors.get(section.heading, ""))
    return {
        identifier: ItemHistory(count, first[identifier])
        for identifier, count in counts.items()
    }


def plan_wide_markdown(roadmap_text: str, plan: Plan) -> str:
    """
    :param roadmap_text: The plan's whole roadmap.
    :param plan: The plan, to recognize which sections are about one item.
    :return: The preamble and every section about no single item, in order.
    """
    return "".join(
        section.text
        for section in roadmap_sections(roadmap_text)
        if section_subject(section, plan) is None
    )
