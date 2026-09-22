"""
Authenticated calls to the GitHub REST API.

A web session carries a token in ``GH_TOKEN`` or ``GITHUB_TOKEN`` but no ``gh``; a local
machine usually has ``gh`` logged in and neither variable set.
:func:`resolve_github_token` accepts either, so no caller has to know which of the two
it runs in.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

GITHUB_API_ROOT = "https://api.github.com"
"""
Base URL every REST call is built on.
"""

CREDENTIAL_VARIABLES = ("GH_TOKEN", "GITHUB_TOKEN")
"""
Environment variables read, in order, for the token the API calls authenticate with.
"""

HTTP_NOT_FOUND = 404
"""
The status the API answers a resource that does not exist with.
"""

# %% failures


@dataclass
class GitHubCredentialUnavailableError(RuntimeError):
    """
    Raised when neither a credential variable nor a logged-in ``gh`` supplies a token.
    """

    variables: tuple[str, ...]
    """
    The environment variables that were consulted.
    """

    def __str__(self) -> str:
        """:return: What was looked for, so the caller can supply it."""
        return (
            f"no GitHub token: set one of {', '.join(self.variables)}, or log in with "
            f"'gh auth login'"
        )


@dataclass
class GitHubApiRequestFailedError(RuntimeError):
    """
    Raised when the API refuses a call for any reason other than the resource not
    existing.
    """

    status: int
    """
    The HTTP status the API answered with.
    """

    path: str
    """
    The API path called, without the host.
    """

    detail: str
    """
    The body of the refusal, which usually says why.
    """

    def __str__(self) -> str:
        """:return: The call and the refusal, so the cause can be read off directly."""
        return f"{self.path} was refused with {self.status}: {self.detail}"


@dataclass
class GitHubGraphQLError(RuntimeError):
    """
    Raised when a GraphQL query is answered with errors, which the API does with a
    successful status and ``null`` data rather than a refusal.
    """

    messages: list[str]
    """
    What each reported error says.
    """

    def __str__(self) -> str:
        """:return: Every reported error, in order."""
        return "GraphQL query failed: " + "; ".join(self.messages)


# %% the token


def resolve_github_token() -> str:
    """
    :return: The first credential variable that is set, else the token of a logged-in
        ``gh``.
    :raises GitHubCredentialUnavailableError: If neither supplies one.
    """
    for variable in CREDENTIAL_VARIABLES:
        token = os.environ.get(variable)
        if token:
            return token
    if shutil.which("gh") is None:
        raise GitHubCredentialUnavailableError(CREDENTIAL_VARIABLES)
    completed = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, check=False
    )
    token = completed.stdout.strip()
    if completed.returncode != 0 or not token:
        raise GitHubCredentialUnavailableError(CREDENTIAL_VARIABLES)
    return token


# %% the client


ResponseOpener = Callable[[urllib.request.Request], Any]
"""
Something that opens a request the way :func:`urllib.request.urlopen` does, returning a
context-managed response with ``read()``.
"""


@dataclass(frozen=True)
class GitHubApi:
    """
    Calls the REST API with one token.
    """

    token: str
    """
    The credential every request authenticates with.
    """

    opener: ResponseOpener = urllib.request.urlopen
    """
    How a request is sent; replaceable so a test never reaches the network.
    """

    root: str = GITHUB_API_ROOT
    """
    The API's base URL.
    """

    page_size: int = 100
    """
    How many entries a list endpoint is asked for per page, the API's maximum.
    """

    @classmethod
    def from_environment(cls) -> GitHubApi:
        """
        :return: A client authenticated with whatever credential this environment has.
        :raises GitHubCredentialUnavailableError: If it has none.
        """
        return cls(resolve_github_token())

    def get(self, path: str) -> Any | None:
        """
        :param path: The API path, starting with a slash.
        :return: The decoded response, or ``None`` when the resource does not exist.
        :raises GitHubApiRequestFailedError: If the API refuses the call for any other
            reason.
        """
        return self._send(path)

    def post(self, path: str, payload: dict[str, Any]) -> Any | None:
        """
        :param path: The API path, starting with a slash.
        :param payload: The JSON body to send.
        :return: The decoded response, or ``None`` when the resource does not exist.
        :raises GitHubApiRequestFailedError: If the API refuses the call for any other
            reason.
        """
        return self._send(path, payload)

    def get_all(self, path: str) -> list[Any]:
        """
        Read every page of a list endpoint.

        :param path: The API path, starting with a slash, without paging parameters.
        :return: Every entry, in the API's order; empty if the resource does not exist.
        """
        separator = "&" if "?" in path else "?"
        collected: list[Any] = []
        page = 1
        while True:
            fetched = self._send(
                f"{path}{separator}per_page={self.page_size}&page={page}"
            )
            if not fetched:
                return collected
            collected.extend(fetched)
            if len(fetched) < self.page_size:
                return collected
            page += 1

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        :param query: The GraphQL document.
        :param variables: Its variables.
        :return: The response's ``data``.
        :raises GitHubGraphQLError: If the response reports errors.
        """
        response = self._send("/graphql", {"query": query, "variables": variables})
        if response.get("errors"):
            raise GitHubGraphQLError(
                [error.get("message", str(error)) for error in response["errors"]]
            )
        return response["data"]

    def _send(self, path: str, payload: dict[str, Any] | None = None) -> Any | None:
        """
        :param path: The API path, starting with a slash.
        :param payload: A JSON body to post, absent for a read.
        :return: The decoded response, or ``None`` when the resource does not exist.
        :raises GitHubApiRequestFailedError: If the API refuses the call for any other
            reason.
        """
        request = urllib.request.Request(
            f"{self.root}{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
        try:
            with self.opener(request) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as refused:
            if refused.code == HTTP_NOT_FOUND:
                return None
            raise GitHubApiRequestFailedError(
                status=refused.code,
                path=path,
                detail=refused.read().decode(errors="replace"),
            ) from refused
