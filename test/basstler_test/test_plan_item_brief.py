"""
Tests for plan_item_brief.py: the compact account of one plan item that
plan-item-kickoff and plan-item-resolve read instead of raw pull request, check run,
comment and diff payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from basstler.build_dashboard import Item, ItemStatus, LiveState, Plan, Track, Wave
from basstler.plan_item_brief import (
    MAXIMUM_BODY_CHARACTERS,
    MAXIMUM_LISTED_CHANGED_FILES,
    ChangedFile,
    CheckRun,
    Comment,
    PullRequestDetails,
    PullRequestDetailsSource,
    ReviewThread,
    UnknownPlanItemError,
    build_brief,
    render_brief,
)
from basstler.pull_request_state import PullRequestReference, PullRequestSource

REPOSITORY = "owner/repo"
TRACKING_ISSUE = 7


# %% building plans and sources


def make_item(
    identifier: str,
    pull_request_number: int | None = None,
    status: ItemStatus = ItemStatus.IN_PROGRESS,
    track: str = "track-1",
    depends_on: tuple[str, ...] = (),
) -> Item:
    """
    :return: An item whose id, branch and title are all *identifier*.
    """
    return Item(
        title=identifier,
        branch=identifier,
        track=track,
        status=status,
        id=identifier,
        pull_request_number=pull_request_number,
        depends_on=list(depends_on),
    )


def make_plan(items: list[Item], tracking_issue: int | None = TRACKING_ISSUE) -> Plan:
    """
    :return: A plan with two tracks in one wave, holding *items*.
    """
    return Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository=REPOSITORY,
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[
            Track(id="track-1", name="Track One", wave="wave-1"),
            Track(id="track-2", name="Track Two", wave="wave-1"),
        ],
        items=items,
        tracking_issue=tracking_issue,
    )


def details(number: int, **overrides: Any) -> PullRequestDetails:
    """
    :return: An open, non-draft pull request with no checks, threads, comments or files,
        with *overrides* applied.
    """
    values: dict[str, Any] = dict(
        reference=PullRequestReference(REPOSITORY, number),
        title=f"pull request {number}",
        state="open",
        draft=False,
        mergeable_state="clean",
        head_branch=f"branch-{number}",
        base_branch="main",
        changed_files=[],
        check_runs=[],
        unresolved_review_threads=[],
        conversation_comments=[],
    )
    values.update(overrides)
    return PullRequestDetails(**values)


@dataclass
class StateSource(PullRequestSource):
    """
    Answers the four-field state the dependency check reads.
    """

    states: Mapping[int, dict[str, Any]] = field(default_factory=dict)

    def pull_request(self, reference: PullRequestReference) -> dict[str, Any] | None:
        return self.states.get(reference.number)


@dataclass
class DetailsSource(PullRequestDetailsSource):
    """
    Answers from fixed details, file lists and issue comments.
    """

    pull_requests: Mapping[int, PullRequestDetails] = field(default_factory=dict)
    files: Mapping[int, list[ChangedFile]] = field(default_factory=dict)
    comments: Mapping[int, list[Comment]] = field(default_factory=dict)

    def details(self, reference: PullRequestReference) -> PullRequestDetails | None:
        return self.pull_requests.get(reference.number)

    def changed_files(self, reference: PullRequestReference) -> list[ChangedFile]:
        return list(self.files.get(reference.number, []))

    def issue_comments(self, reference: PullRequestReference) -> list[Comment]:
        return list(self.comments.get(reference.number, []))


def brief_text(plan: Plan, item: str, **sources: Any) -> str:
    """
    :return: The rendered brief for *item*.
    """
    return render_brief(
        build_brief(
            plan,
            item,
            sources.get("states", StateSource()),
            sources.get("details", DetailsSource()),
        )
    )


# %% the item itself


def test_an_unknown_item_is_refused_naming_what_exists():
    plan = make_plan([make_item("real")])

    with pytest.raises(UnknownPlanItemError) as raised:
        build_brief(plan, "imagined", StateSource(), DetailsSource())

    assert raised.value.available == ["real"]


def test_an_item_without_a_pull_request_says_so():
    text = brief_text(make_plan([make_item("fresh")]), "fresh")

    assert "No pull request yet." in text


# %% dependencies


def test_dependencies_carry_their_live_state_and_readiness():
    plan = make_plan(
        [
            make_item("base", 1),
            make_item("draft-base", 2),
            make_item("item", depends_on=("base", "draft-base")),
        ]
    )
    states = StateSource(
        {
            1: {"state": "closed", "draft": False, "merged_at": "2026-09-01T10:00:00Z", "labels": []},
            2: {"state": "open", "draft": True, "merged_at": None, "labels": []},
        }
    )

    brief = build_brief(plan, "item", states, DetailsSource())

    assert [(entry.identifier, entry.live_state, entry.is_ready) for entry in brief.dependencies] == [
        ("base", LiveState.MERGED, True),
        ("draft-base", LiveState.OPEN_DRAFT, False),
    ]


# %% the pull request


def test_only_failing_checks_are_named_and_the_rest_are_counted():
    check_runs = [
        CheckRun("unit tests", status="completed", conclusion="success", url="https://ci/1"),
        CheckRun("lint", status="completed", conclusion="failure", url="https://ci/2"),
        CheckRun("integration", status="in_progress", conclusion=None, url="https://ci/3"),
    ]
    plan = make_plan([make_item("item", 17)])
    source = DetailsSource({17: details(17, check_runs=check_runs)})

    text = brief_text(plan, "item", details=source)

    assert "3 total: 1 passed, 1 failing, 1 pending" in text
    assert "lint" in text and "https://ci/2" in text
    assert "unit tests" not in text


def test_every_unresolved_thread_is_listed_with_where_it_points():
    thread = ReviewThread(
        path="basstler/stack.py", line=12, author="reviewer", body="rename this",
        comment_count=3, is_outdated=False,
    )
    plan = make_plan([make_item("item", 17)])
    source = DetailsSource({17: details(17, unresolved_review_threads=[thread])})

    text = brief_text(plan, "item", details=source)

    assert "`basstler/stack.py:12`" in text
    assert "rename this" in text


def test_changed_files_are_listed_with_line_counts_up_to_a_limit():
    files = [
        ChangedFile(path=f"file_{index}.py", status="modified", additions=index, deletions=1)
        for index in range(MAXIMUM_LISTED_CHANGED_FILES + 5)
    ]
    plan = make_plan([make_item("item", 17)])
    source = DetailsSource({17: details(17, changed_files=files)})

    text = brief_text(plan, "item", details=source)

    assert "`file_0.py` (modified, +0 −1)" in text
    assert f"`file_{MAXIMUM_LISTED_CHANGED_FILES}.py`" not in text
    assert "… and 5 more" in text


def test_long_comment_bodies_are_cut_short():
    comment = Comment(author="someone", body="x" * (MAXIMUM_BODY_CHARACTERS * 3), created_at="2026-09-01")
    plan = make_plan([make_item("item", 17)])
    source = DetailsSource({17: details(17, conversation_comments=[comment])})

    text = brief_text(plan, "item", details=source)

    assert "x" * MAXIMUM_BODY_CHARACTERS + "…" in text
    assert "x" * (MAXIMUM_BODY_CHARACTERS + 1) not in text


# %% the tracking issue


def test_only_tracking_issue_comments_that_mention_the_item_are_quoted():
    comments = [
        Comment(author="a", body="Moving ITEM to wave 2", created_at="2026-09-01"),
        Comment(author="b", body="Unrelated structural change", created_at="2026-09-02"),
    ]
    plan = make_plan([make_item("item")])
    source = DetailsSource(comments={TRACKING_ISSUE: comments})

    text = brief_text(plan, "item", details=source)

    assert "1 of 2 comments mention this item" in text
    assert "Moving ITEM to wave 2" in text
    assert "Unrelated structural change" not in text


def test_a_plan_without_a_tracking_issue_has_no_tracking_section():
    text = brief_text(make_plan([make_item("item")], tracking_issue=None), "item")

    assert "Tracking issue" not in text


# %% landed siblings


def test_landed_items_of_the_same_track_are_listed_with_their_files_only():
    plan = make_plan(
        [
            make_item("landed", 3, status=ItemStatus.DONE),
            make_item("other-track", 4, status=ItemStatus.DONE, track="track-2"),
            make_item("item"),
        ]
    )
    source = DetailsSource(
        files={
            3: [ChangedFile("basstler/landed.py", "added", 40, 0)],
            4: [ChangedFile("basstler/elsewhere.py", "added", 10, 0)],
        }
    )

    text = brief_text(plan, "item", details=source)

    assert "`basstler/landed.py`" in text
    assert "elsewhere.py" not in text
