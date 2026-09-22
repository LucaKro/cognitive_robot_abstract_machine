#!/usr/bin/env python3
"""
A compact account of one plan item, for plan-item-kickoff and plan-item-resolve to read
instead of raw API payloads.

Before this, gathering an item's context meant reading its pull request's full
representation, every check run, every comment, the whole diff and every tracking-issue
comment - over 100k tokens for one item before any code was touched. The brief keeps what
decides the next step: the item's recorded state, whether its dependencies are ready,
which checks fail, which review threads are unresolved, which files changed and by how
much, the tracking-issue comments that mention it, and which files landed siblings in its
track touched. A session then opens only the files the brief points at.

The roadmap is handled the same way. Kickoffs, resolutions, review rounds and restacks
append their records to it, so it grows without bound - one real plan's reached 236 KB,
nearly all of it other items' history. With ``--roadmap`` the brief carries the
plan-wide sections and this item's own sections (newest first within a budget), and
lists every other section by heading so one can be pulled with ``--section``.

Usage:
    python3 -m basstler.plan_item_brief --plan /tmp/plan.yaml --roadmap /tmp/roadmap.md \
        --item <item-id>
    python3 -m basstler.plan_item_brief --roadmap /tmp/roadmap.md --section "<heading>"

Prints the brief, or the one section, as Markdown.
"""

from __future__ import annotations

import argparse
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from basstler.build_dashboard import (
    Item,
    ItemStatus,
    LiveState,
    Plan,
    PlanValidationError,
    classify_live_state,
    load_pull_requests_by_repository,
    validate_plan,
)
from basstler.check_dependency_readiness import dependency_readiness
from basstler.github_api import GitHubApi
from basstler.pull_request_state import (
    GitHubPullRequestSource,
    PullRequestReference,
    PullRequestSource,
    collect_pull_request_states,
)

MAXIMUM_BODY_CHARACTERS = 400
"""
Longest comment or thread body quoted before it is cut short.
"""

MAXIMUM_LISTED_CHANGED_FILES = 40
"""
Most changed files listed for the item's own pull request; the rest are counted.
"""

MAXIMUM_LISTED_SIBLING_FILES = 15
"""
Most changed files listed for each landed sibling.
"""

MAXIMUM_CONVERSATION_COMMENTS = 3
"""
How many of the most recent pull request conversation comments are quoted.
"""

FAILING_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "startup_failure", "stale"}
)
"""
Check run conclusions that mean the check did not pass.
"""

COMPLETED = "completed"
"""
The check run status of a run that has finished.
"""

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


# %% failures


@dataclass
class UnknownPlanItemError(ValueError):
    """
    Raised when the requested item is not in the plan.
    """

    requested: str
    """The identifier asked for."""

    available: list[str]
    """Every item identifier the plan has."""

    def __str__(self) -> str:
        """:return: What was asked for and what exists, so the right one can be picked."""
        return f"no item {self.requested!r}; the plan has: {', '.join(self.available)}"


# %% what is gathered


@dataclass(frozen=True)
class ChangedFile:
    """
    One file a pull request changes.
    """

    path: str
    status: str
    """``added``, ``modified``, ``removed`` or ``renamed``."""
    additions: int
    deletions: int


@dataclass(frozen=True)
class CheckRun:
    """
    One check on a pull request's head commit.
    """

    name: str
    status: str
    """``queued``, ``in_progress`` or ``completed``."""
    conclusion: str | None
    """The outcome, once :attr:`status` is ``completed``."""
    url: str | None

    @property
    def is_pending(self) -> bool:
        """Whether it has not finished."""
        return self.status != COMPLETED

    @property
    def is_failing(self) -> bool:
        """Whether it finished without passing."""
        return not self.is_pending and self.conclusion in FAILING_CONCLUSIONS


@dataclass(frozen=True)
class ReviewThread:
    """
    One unresolved review thread, represented by its first comment.
    """

    path: str
    line: int | None
    author: str
    body: str
    comment_count: int
    is_outdated: bool
    """Whether the lines it was left on have since changed."""


@dataclass(frozen=True)
class Comment:
    """
    One issue or pull request conversation comment.
    """

    author: str
    body: str
    created_at: str


@dataclass(frozen=True)
class PullRequestDetails:
    """
    What decides the next step on one pull request.
    """

    reference: PullRequestReference
    title: str
    state: str
    draft: bool
    mergeable_state: str | None
    """GitHub's summary of whether it can merge; ``dirty`` means it conflicts."""
    head_branch: str
    base_branch: str
    changed_files: list[ChangedFile]
    check_runs: list[CheckRun]
    unresolved_review_threads: list[ReviewThread]
    conversation_comments: list[Comment]


@dataclass(frozen=True)
class DependencyEntry:
    """
    One of the item's dependencies and whether it is ready to build on.
    """

    identifier: str
    title: str | None
    live_state: LiveState | None
    """``None`` when the dependency names no item in the plan."""
    is_ready: bool

    @classmethod
    def from_readiness(cls, entry: dict[str, Any]) -> DependencyEntry:
        """
        :param entry: One result of :func:`check_dependency_readiness.dependency_readiness`.
        :return: The same, typed.
        """
        live_state = entry["live_state"]
        return cls(
            identifier=entry["identifier"],
            title=entry["title"],
            live_state=None if live_state is None else LiveState(live_state),
            is_ready=entry["is_ready"],
        )


@dataclass(frozen=True)
class LandedSibling:
    """
    An item of the same track that already landed, and the files it changed.
    """

    item: Item
    reference: PullRequestReference
    changed_files: list[ChangedFile]


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


@dataclass
class PlanItemBrief:
    """
    Everything the brief reports on one item.
    """

    plan: Plan
    item: Item
    dependencies: list[DependencyEntry] = field(default_factory=list)
    pull_request_reference: PullRequestReference | None = None
    pull_request: PullRequestDetails | None = None
    """``None`` both when the item has no pull request and when it does not exist."""
    tracking_comments: list[Comment] = field(default_factory=list)
    """Comments on the plan's tracking issue that mention the item."""
    tracking_comment_total: int = 0
    landed_siblings: list[LandedSibling] = field(default_factory=list)
    roadmap: RoadmapSelection | None = None
    """What of the roadmap is carried, if a roadmap was given."""


# %% where the details come from


class PullRequestDetailsSource(ABC):
    """
    Answers what a pull request, its files and an issue's comments look like.
    """

    @abstractmethod
    def details(self, reference: PullRequestReference) -> PullRequestDetails | None:
        """:return: The pull request's details, or ``None`` if it does not exist."""

    @abstractmethod
    def changed_files(self, reference: PullRequestReference) -> list[ChangedFile]:
        """:return: Every file the pull request changes."""

    @abstractmethod
    def issue_comments(self, reference: PullRequestReference) -> list[Comment]:
        """:return: Every conversation comment on the issue or pull request."""


REVIEW_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        nodes {
          isResolved
          isOutdated
          path
          line
          comments(first: 1) { totalCount nodes { author { login } body } }
        }
      }
    }
  }
}
"""
"""
Review threads carry their resolution only through GraphQL; the REST comments do not.
"""


def author_login(author: dict[str, Any] | None) -> str:
    """:return: The login, or ``ghost`` for a deleted account, as GitHub shows it."""
    return author["login"] if author else "ghost"


@dataclass(frozen=True)
class GitHubPullRequestDetailsSource(PullRequestDetailsSource):
    """
    Reads pull request details from the REST and GraphQL APIs.
    """

    api: GitHubApi

    def details(self, reference: PullRequestReference) -> PullRequestDetails | None:
        base = f"/repos/{reference.repository}"
        representation = self.api.get(f"{base}/pulls/{reference.number}")
        if representation is None:
            return None
        check_runs = self.api.get(
            f"{base}/commits/{representation['head']['sha']}/check-runs?per_page=100"
        ) or {"check_runs": []}
        return PullRequestDetails(
            reference=reference,
            title=representation["title"],
            state=representation["state"],
            draft=bool(representation.get("draft", False)),
            mergeable_state=representation.get("mergeable_state"),
            head_branch=representation["head"]["ref"],
            base_branch=representation["base"]["ref"],
            changed_files=self.changed_files(reference),
            check_runs=[
                CheckRun(
                    name=run["name"],
                    status=run["status"],
                    conclusion=run.get("conclusion"),
                    url=run.get("details_url") or run.get("html_url"),
                )
                for run in check_runs["check_runs"]
            ],
            unresolved_review_threads=self._unresolved_review_threads(reference),
            conversation_comments=self.issue_comments(reference),
        )

    def changed_files(self, reference: PullRequestReference) -> list[ChangedFile]:
        return [
            ChangedFile(
                path=entry["filename"],
                status=entry["status"],
                additions=entry["additions"],
                deletions=entry["deletions"],
            )
            for entry in self.api.get_all(
                f"/repos/{reference.repository}/pulls/{reference.number}/files"
            )
        ]

    def issue_comments(self, reference: PullRequestReference) -> list[Comment]:
        return [
            Comment(
                author=author_login(entry.get("user")),
                body=entry.get("body") or "",
                created_at=entry["created_at"],
            )
            for entry in self.api.get_all(
                f"/repos/{reference.repository}/issues/{reference.number}/comments"
            )
        ]

    def _unresolved_review_threads(
        self, reference: PullRequestReference
    ) -> list[ReviewThread]:
        owner, name = reference.repository.split("/", 1)
        data = self.api.graphql(
            REVIEW_THREADS_QUERY,
            {"owner": owner, "name": name, "number": reference.number},
        )
        threads = []
        for node in data["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            if node["isResolved"]:
                continue
            first_comments = node["comments"]["nodes"]
            first = first_comments[0] if first_comments else {"author": None, "body": ""}
            threads.append(
                ReviewThread(
                    path=node["path"],
                    line=node.get("line"),
                    author=author_login(first.get("author")),
                    body=first.get("body") or "",
                    comment_count=node["comments"]["totalCount"],
                    is_outdated=node["isOutdated"],
                )
            )
        return threads


# %% building the brief


def mentions(body: str, item: Item) -> bool:
    """:return: Whether *body* names the item by id, branch or title."""
    lowered = body.lower()
    return any(
        term.lower() in lowered
        for term in (item.identifier, item.branch, item.title)
        if term
    )


def build_brief(
    plan: Plan,
    item_identifier: str,
    state_source: PullRequestSource,
    details_source: PullRequestDetailsSource,
) -> PlanItemBrief:
    """
    :param plan: The already-validated plan.
    :param item_identifier: The item's ``id``, or ``branch`` if it has none.
    :param state_source: Where the dependency check reads pull request state from.
    :param details_source: Where pull request details and comments come from.
    :return: The brief.
    :raises UnknownPlanItemError: If the item is not in the plan.
    """
    items_by_identifier = {item.identifier: item for item in plan.items}
    item = items_by_identifier.get(item_identifier)
    if item is None:
        raise UnknownPlanItemError(item_identifier, sorted(items_by_identifier))

    pull_requests_by_repository = load_pull_requests_by_repository(
        collect_pull_request_states(plan, state_source).by_repository
    )
    brief = PlanItemBrief(
        plan=plan,
        item=item,
        dependencies=[
            DependencyEntry.from_readiness(entry)
            for entry in dependency_readiness(
                plan, item.identifier, pull_requests_by_repository
            )
        ],
    )

    if item.pull_request_number is not None:
        brief.pull_request_reference = PullRequestReference(
            item.repository or plan.default_repository, item.pull_request_number
        )
        brief.pull_request = details_source.details(brief.pull_request_reference)

    if plan.tracking_issue is not None:
        comments = details_source.issue_comments(
            PullRequestReference(plan.default_repository, plan.tracking_issue)
        )
        brief.tracking_comment_total = len(comments)
        brief.tracking_comments = [
            comment for comment in comments if mentions(comment.body, item)
        ]

    for sibling in plan.items:
        if sibling is item or sibling.track != item.track:
            continue
        if sibling.pull_request_number is None:
            continue
        repository = sibling.repository or plan.default_repository
        live_state = classify_live_state(
            sibling.pull_request_number, repository, pull_requests_by_repository
        )
        if sibling.status is not ItemStatus.DONE and live_state is not LiveState.MERGED:
            continue
        reference = PullRequestReference(repository, sibling.pull_request_number)
        brief.landed_siblings.append(
            LandedSibling(sibling, reference, details_source.changed_files(reference))
        )
    return brief


# %% the roadmap


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


# %% rendering it


def shortened(body: str) -> str:
    """:return: *body* on one line, cut at :data:`MAXIMUM_BODY_CHARACTERS`."""
    flattened = " ".join(body.split())
    if len(flattened) <= MAXIMUM_BODY_CHARACTERS:
        return flattened
    return flattened[:MAXIMUM_BODY_CHARACTERS] + "…"


def file_line(changed: ChangedFile) -> str:
    """:return: One changed file with its line counts."""
    return f"`{changed.path}` ({changed.status}, +{changed.additions} −{changed.deletions})"


def render_pull_request(details: PullRequestDetails) -> list[str]:
    """:return: The pull request section's lines."""
    lines = [f"## Pull request {details.reference}: {details.title}"]
    state = details.state + (" (draft)" if details.draft else "")
    mergeable = details.mergeable_state or "unknown"
    if mergeable == "dirty":
        mergeable += " (conflicts with its base)"
    lines.append(
        f"- state: {state} · mergeable: {mergeable} · "
        f"`{details.head_branch}` → `{details.base_branch}`"
    )

    runs = details.check_runs
    failing = [run for run in runs if run.is_failing]
    pending = [run for run in runs if run.is_pending]
    passed = len(runs) - len(failing) - len(pending)
    lines.append(
        f"- checks: {len(runs)} total: {passed} passed, {len(failing)} failing, "
        f"{len(pending)} pending"
    )
    lines.extend(f"  - failing: {run.name} ({run.conclusion}) {run.url or ''}".rstrip() for run in failing)

    threads = details.unresolved_review_threads
    lines.append(f"- unresolved review threads: {len(threads)}")
    for thread in threads:
        where = f"{thread.path}:{thread.line}" if thread.line else thread.path
        outdated = ", outdated" if thread.is_outdated else ""
        lines.append(
            f"  - `{where}` @{thread.author} ({thread.comment_count} comments{outdated}): "
            f"{shortened(thread.body)}"
        )

    comments = details.conversation_comments
    lines.append(f"- conversation: {len(comments)} comments")
    for comment in comments[-MAXIMUM_CONVERSATION_COMMENTS:]:
        lines.append(f"  - @{comment.author} ({comment.created_at}): {shortened(comment.body)}")

    files = details.changed_files
    additions = sum(changed.additions for changed in files)
    deletions = sum(changed.deletions for changed in files)
    lines.append(f"- changed files: {len(files)} (+{additions} −{deletions})")
    lines.extend(f"  - {file_line(changed)}" for changed in files[:MAXIMUM_LISTED_CHANGED_FILES])
    if len(files) > MAXIMUM_LISTED_CHANGED_FILES:
        lines.append(f"  - … and {len(files) - MAXIMUM_LISTED_CHANGED_FILES} more")
    return lines


def render_brief(brief: PlanItemBrief) -> str:
    """
    :param brief: What was gathered.
    :return: The brief as Markdown.
    """
    item, plan = brief.item, brief.plan
    tracks = {track.id: track for track in plan.tracks}
    waves = {wave.id: wave for wave in plan.waves}
    track = tracks.get(item.track)
    wave = waves.get(track.wave) if track else None

    lines = [f"# {item.identifier}: {item.title}", ""]
    lines.append(
        f"- status: {item.status.value} · track: {track.name if track else item.track}"
        + (f" ({wave.name})" if wave else "")
        + f" · branch: `{item.branch}`"
    )
    if track and track.description:
        lines.append(f"- track: {track.description}")
    if item.session:
        lines.append(f"- previous session: {item.session}")
    if item.notes:
        lines += ["", "## Notes", item.notes]
    if item.blockers:
        lines += ["", "## Blockers", *[f"- {blocker}" for blocker in item.blockers]]

    if brief.dependencies:
        lines += ["", "## Dependencies"]
        for dependency in brief.dependencies:
            state = dependency.live_state.value if dependency.live_state else "unknown item"
            ready = "ready" if dependency.is_ready else "NOT ready"
            lines.append(f"- {dependency.identifier}: {state}, {ready}")

    lines.append("")
    if brief.pull_request_reference is None:
        lines.append("No pull request yet.")
    elif brief.pull_request is None:
        lines.append(f"Pull request {brief.pull_request_reference} does not exist.")
    else:
        lines += render_pull_request(brief.pull_request)

    if plan.tracking_issue is not None:
        lines += [
            "",
            f"## Tracking issue #{plan.tracking_issue}: "
            f"{len(brief.tracking_comments)} of {brief.tracking_comment_total} comments "
            f"mention this item",
        ]
        lines.extend(
            f"- @{comment.author} ({comment.created_at}): {shortened(comment.body)}"
            for comment in brief.tracking_comments
        )

    if brief.landed_siblings:
        lines += ["", "## Landed items in this track"]
        for sibling in brief.landed_siblings:
            listed = sibling.changed_files[:MAXIMUM_LISTED_SIBLING_FILES]
            names = ", ".join(f"`{changed.path}`" for changed in listed)
            more = len(sibling.changed_files) - len(listed)
            lines.append(
                f"- {sibling.item.identifier} ({sibling.reference}): {names}"
                + (f", … and {more} more" if more > 0 else "")
            )

    if brief.roadmap is not None:
        lines += ["", render_roadmap_selection(brief.roadmap).rstrip()]

    lines += [
        "",
        "## Reading further",
        f"Open only what the above points at: `git fetch origin {item.branch}` then "
        f"`git show origin/{item.branch}:<path>`, or `git diff` a single path.",
    ]
    return "\n".join(lines) + "\n"


# %% the command


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Print the brief for one item. See the module docstring for the contract.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", help="Path to plan.yaml")
    parser.add_argument("--item", help="The item id (or branch)")
    parser.add_argument("--roadmap", help="Path to roadmap.md")
    parser.add_argument("--section", help="Print only this roadmap section")
    parsed = parser.parse_args(arguments)

    if parsed.section is not None:
        if parsed.roadmap is None:
            parser.error("--section needs --roadmap")
        try:
            print(roadmap_section(Path(parsed.roadmap).read_text(), parsed.section), end="")
        except UnknownRoadmapSectionError as error:
            print(str(error), file=sys.stderr)
            return 1
        return 0
    if parsed.plan is None or parsed.item is None:
        parser.error("--plan and --item are required unless --section is given")

    raw_plan = yaml.safe_load(Path(parsed.plan).read_text())
    try:
        validate_plan(raw_plan)
    except PlanValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    plan = Plan.from_mapping(raw_plan)

    api = GitHubApi.from_environment()
    try:
        brief = build_brief(
            plan,
            parsed.item,
            GitHubPullRequestSource(api),
            GitHubPullRequestDetailsSource(api),
        )
    except UnknownPlanItemError as error:
        print(str(error), file=sys.stderr)
        return 1
    if parsed.roadmap is not None:
        brief.roadmap = select_roadmap_sections(
            Path(parsed.roadmap).read_text(), plan, brief.item
        )
    print(render_brief(brief), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
