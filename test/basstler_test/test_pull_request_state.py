"""
Tests for pull_request_state.py and the github_api.py client it reads through: fetching
the live state of exactly the pull requests a plan references, reduced to the fields the
dashboard and the dependency check read.
"""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from basstler import pull_request_state
from basstler.build_dashboard import (
    Item,
    ItemStatus,
    LiveState,
    Plan,
    Track,
    Wave,
    classify_live_state,
    load_pull_requests_by_repository,
)
from basstler.github_api import (
    CREDENTIAL_VARIABLES,
    GitHubApi,
    GitHubApiRequestFailedError,
    GitHubCredentialUnavailableError,
    GitHubGraphQLError,
    resolve_github_token,
)
from basstler.pull_request_state import (
    GitHubPullRequestSource,
    PullRequestReference,
    PullRequestSource,
    collect_pull_request_states,
    referenced_pull_requests,
)

from .executable_stubs import ExecutableStubDirectory, path_hiding_executable

DEFAULT_REPOSITORY = "owner/repo"
OTHER_REPOSITORY = "other/repo"


# %% building plans and API representations


def make_plan(items: list[Item]) -> Plan:
    """
    :param items: The plan's items.
    :return: A one-wave, one-track plan holding *items*, on :data:`DEFAULT_REPOSITORY`.
    """
    return Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository=DEFAULT_REPOSITORY,
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=items,
    )


def make_item(
    identifier: str,
    pull_request_number: int | None,
    repository: str | None = None,
) -> Item:
    """
    :param identifier: The item's id and branch.
    :param pull_request_number: Its pull request, if it has one.
    :param repository: A repository overriding the plan's default.
    :return: An in-progress item.
    """
    return Item(
        title=identifier,
        branch=identifier,
        track="track-1",
        status=ItemStatus.IN_PROGRESS,
        id=identifier,
        pull_request_number=pull_request_number,
        repository=repository,
    )


def api_representation(
    number: int,
    state: str = "open",
    draft: bool = False,
    merged_at: str | None = None,
    labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    """
    A pull request as the REST API returns it: the four fields the dashboard reads,
    padded with the kind of bulk the real representation carries around them.

    :return: The representation.
    """
    return {
        "number": number,
        "state": state,
        "draft": draft,
        "merged_at": merged_at,
        "labels": [{"id": index, "name": name, "color": "ffffff"} for index, name in enumerate(labels)],
        "body": "a long description " * 200,
        "head": {"ref": f"branch-{number}", "repo": {"full_name": DEFAULT_REPOSITORY}},
        "user": {"login": "someone", "avatar_url": "https://example.invalid/avatar"},
    }


@dataclass
class RecordingPullRequestSource(PullRequestSource):
    """
    Answers from a fixed set of representations and records every request, so a test can
    assert both what came back and what was asked for.
    """

    representations: Mapping[PullRequestReference, dict[str, Any]]
    """
    What each pull request that exists looks like; any other reference does not exist.
    """

    requested: list[PullRequestReference] = field(default_factory=list)
    """
    Every reference asked for, in order.
    """

    def pull_request(self, reference: PullRequestReference) -> dict[str, Any] | None:
        self.requested.append(reference)
        return self.representations.get(reference)


# %% which pull requests are fetched


def test_only_the_pull_requests_items_reference_are_requested_each_once():
    plan = make_plan(
        [
            make_item("first", 3),
            make_item("second", 1),
            make_item("same-pull-request", 3),
            make_item("not-started", None),
            make_item("elsewhere", 1, repository=OTHER_REPOSITORY),
        ]
    )
    source = RecordingPullRequestSource(representations={})

    collect_pull_request_states(plan, source)

    assert source.requested == referenced_pull_requests(plan)
    assert len(source.requested) == len(set(source.requested))
    assert set(source.requested) == {
        PullRequestReference(DEFAULT_REPOSITORY, 1),
        PullRequestReference(DEFAULT_REPOSITORY, 3),
        PullRequestReference(OTHER_REPOSITORY, 1),
    }


# %% what is kept of each


def test_keeps_only_the_fields_the_dashboard_reads():
    reference = PullRequestReference(DEFAULT_REPOSITORY, 7)
    source = RecordingPullRequestSource(
        {reference: api_representation(7, draft=True, labels=("bug", "merged"))}
    )

    states = collect_pull_request_states(make_plan([make_item("item", 7)]), source)

    assert states.by_repository == {
        DEFAULT_REPOSITORY: {
            "7": {"state": "open", "draft": True, "merged_at": None, "labels": ["bug", "merged"]}
        }
    }


@pytest.mark.parametrize(
    ("representation", "expected_live_state"),
    [
        (api_representation(5, draft=True), LiveState.OPEN_DRAFT),
        (api_representation(5), LiveState.OPEN_READY),
        (api_representation(5, state="closed", merged_at="2026-09-01T10:00:00Z"), LiveState.MERGED),
        (api_representation(5, state="closed"), LiveState.CLOSED_UNMERGED),
    ],
)
def test_the_output_is_read_back_by_the_dashboard_as_the_same_live_state(
    representation: dict[str, Any], expected_live_state: LiveState
):
    """
    A closed, unmerged pull request is the case that matters most: the dashboard refuses
    a closed entry without an explicit ``merged_at``, rather than reading it as unmerged.
    """
    reference = PullRequestReference(DEFAULT_REPOSITORY, 5)
    source = RecordingPullRequestSource({reference: representation})

    states = collect_pull_request_states(make_plan([make_item("item", 5)]), source)
    read_back = load_pull_requests_by_repository(json.loads(json.dumps(states.by_repository)))

    assert classify_live_state(5, DEFAULT_REPOSITORY, read_back) is expected_live_state


def test_a_pull_request_that_does_not_exist_is_left_out_and_reads_as_not_found():
    missing = PullRequestReference(DEFAULT_REPOSITORY, 99)
    source = RecordingPullRequestSource(representations={})

    states = collect_pull_request_states(make_plan([make_item("item", 99)]), source)
    read_back = load_pull_requests_by_repository(states.by_repository)

    assert states.not_found == [missing]
    assert classify_live_state(99, DEFAULT_REPOSITORY, read_back) is LiveState.NOT_FOUND


# %% the REST client


@dataclass
class ResponseStandIn:
    """
    Stands in for what :func:`urllib.request.urlopen` returns, as a context manager.
    """

    body: bytes

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> ResponseStandIn:
        return self

    def __exit__(self, *exception_information: object) -> None:
        return None


@dataclass
class RecordingOpener:
    """
    Answers every request with one fixed response or error, recording the requests.
    """

    body: bytes = b"{}"
    status: int | None = None
    requests: list = field(default_factory=list)

    def __call__(self, request) -> ResponseStandIn:
        self.requests.append(request)
        if self.status is not None:
            raise urllib.error.HTTPError(
                request.full_url, self.status, "refused", {}, io.BytesIO(b"denied")
            )
        return ResponseStandIn(self.body)


def test_the_client_reads_the_pull_request_endpoint_with_the_token():
    opener = RecordingOpener(body=json.dumps(api_representation(4)).encode())
    source = GitHubPullRequestSource(GitHubApi(token="secret", opener=opener))

    representation = source.pull_request(PullRequestReference(DEFAULT_REPOSITORY, 4))

    assert representation["number"] == 4
    (request,) = opener.requests
    assert request.full_url.endswith(f"/repos/{DEFAULT_REPOSITORY}/pulls/4")
    assert request.get_header("Authorization") == "Bearer secret"


def test_a_missing_resource_reads_as_absent_rather_than_failing():
    api = GitHubApi(token="secret", opener=RecordingOpener(status=404))

    assert api.get("/repos/owner/repo/pulls/1") is None


def test_any_other_refusal_fails_naming_the_call():
    api = GitHubApi(token="secret", opener=RecordingOpener(status=403))

    with pytest.raises(GitHubApiRequestFailedError) as raised:
        api.get("/repos/owner/repo/pulls/1")

    assert raised.value.status == 403
    assert raised.value.path == "/repos/owner/repo/pulls/1"


# %% where the token comes from


@pytest.fixture
def no_token_variables(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """
    :return: *monkeypatch*, with every credential variable removed from the environment.
    """
    for variable in CREDENTIAL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


def test_the_first_credential_variable_wins(no_token_variables: pytest.MonkeyPatch):
    first, second = CREDENTIAL_VARIABLES
    no_token_variables.setenv(first, "from-first")
    no_token_variables.setenv(second, "from-second")

    assert resolve_github_token() == "from-first"


def test_a_logged_in_gh_supplies_the_token_when_no_variable_is_set(
    no_token_variables: pytest.MonkeyPatch, stub_bin: ExecutableStubDirectory
):
    stub_bin.install("gh")
    no_token_variables.setenv("PATH", stub_bin.ahead_of(str(Path("/usr/bin"))))
    no_token_variables.setenv("STUB_GH_AUTH_TOKEN", "from-gh")

    assert resolve_github_token() == "from-gh"


def test_no_variable_and_no_logged_in_gh_is_refused(
    no_token_variables: pytest.MonkeyPatch, stub_bin: ExecutableStubDirectory
):
    stub_bin.install("gh")
    no_token_variables.setenv("PATH", stub_bin.ahead_of(str(Path("/usr/bin"))))
    no_token_variables.delenv("STUB_GH_AUTH_TOKEN", raising=False)

    with pytest.raises(GitHubCredentialUnavailableError):
        resolve_github_token()


def test_no_variable_and_no_gh_at_all_is_refused(
    no_token_variables: pytest.MonkeyPatch, tmp_path: Path
):
    no_token_variables.setenv("PATH", path_hiding_executable("gh", tmp_path))

    with pytest.raises(GitHubCredentialUnavailableError):
        resolve_github_token()


# %% the command line


def test_the_command_writes_the_file_and_summarizes_what_it_could_not_find(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    manifest = {
        "schema_version": 1,
        "id": "test-plan",
        "title": "Test Plan",
        "description": "desc",
        "default_repository": DEFAULT_REPOSITORY,
        "waves": [{"id": "wave-1", "name": "Wave One"}],
        "tracks": [{"id": "track-1", "name": "Track One", "wave": "wave-1"}],
        "items": [
            {"id": "found", "title": "found", "branch": "found", "track": "track-1",
             "status": "in_progress", "pull_request_number": 2},
            {"id": "gone", "title": "gone", "branch": "gone", "track": "track-1",
             "status": "in_progress", "pull_request_number": 8},
        ],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(manifest))
    output_path = tmp_path / "pr_data.json"
    source = RecordingPullRequestSource(
        {PullRequestReference(DEFAULT_REPOSITORY, 2): api_representation(2)}
    )
    monkeypatch.setattr(pull_request_state, "default_pull_request_source", lambda: source)

    exit_code = pull_request_state.main(["--plan", str(plan_path), "--output", str(output_path)])

    assert exit_code == 0
    assert json.loads(output_path.read_text()) == {
        DEFAULT_REPOSITORY: {"2": {"state": "open", "draft": False, "merged_at": None, "labels": []}}
    }
    assert json.loads(capsys.readouterr().out) == {
        "pull_requests": 1,
        "not_found": [str(PullRequestReference(DEFAULT_REPOSITORY, 8))],
    }


# %% paging and GraphQL


@dataclass
class PagedOpener:
    """
    Answers ``page=N`` requests from a list of pages and GraphQL posts with one body,
    recording every request.
    """

    pages: list[list[Any]] = field(default_factory=list)
    graphql_body: dict[str, Any] = field(default_factory=dict)
    requests: list = field(default_factory=list)

    def __call__(self, request) -> ResponseStandIn:
        self.requests.append(request)
        if request.full_url.endswith("/graphql"):
            return ResponseStandIn(json.dumps(self.graphql_body).encode())
        page = int(request.full_url.rsplit("page=", 1)[1])
        answered = self.pages[page - 1] if page <= len(self.pages) else []
        return ResponseStandIn(json.dumps(answered).encode())


def test_every_page_is_read_until_a_short_one():
    opener = PagedOpener(pages=[[1, 2], [3, 4], [5]])
    api = GitHubApi(token="secret", opener=opener, page_size=2)

    assert api.get_all("/repos/owner/repo/pulls/1/files") == [1, 2, 3, 4, 5]
    assert len(opener.requests) == 3


def test_a_graphql_query_returns_its_data_and_sends_the_variables():
    opener = PagedOpener(graphql_body={"data": {"answer": 42}})
    api = GitHubApi(token="secret", opener=opener)

    assert api.graphql("query { answer }", {"number": 7}) == {"answer": 42}
    (request,) = opener.requests
    assert json.loads(request.data) == {"query": "query { answer }", "variables": {"number": 7}}


def test_graphql_errors_fail_rather_than_reading_as_empty_data():
    api = GitHubApi(
        token="secret",
        opener=PagedOpener(graphql_body={"data": None, "errors": [{"message": "bad field"}]}),
    )

    with pytest.raises(GitHubGraphQLError) as raised:
        api.graphql("query { nope }", {})

    assert raised.value.messages == ["bad field"]
