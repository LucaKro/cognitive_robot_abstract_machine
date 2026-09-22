"""
A GitHub repository, as every part of the package names and addresses one.
"""

from __future__ import annotations

from dataclasses import dataclass

GITHUB_WEB_ROOT = "https://github.com"
"""
Where GitHub serves repositories to a browser.
"""

GITHUB_HOST = "github.com"
"""
GitHub's host, as remote URLs name it.
"""

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
"""
Hosts of the local proxy a cloud session reaches GitHub through; its remote URLs carry
the repository as their last two path segments but never name GitHub.
"""


@dataclass
class MalformedRepositoryError(ValueError):
    """
    Raised when a repository reference is not in ``owner/name`` form.
    """

    text: str
    """The value that could not be parsed."""

    def __str__(self) -> str:
        """:return: What was expected and what arrived instead."""
        return f"expected a repository as 'owner/name', got {self.text!r}"


@dataclass(frozen=True)
class Repository:
    """
    A GitHub repository, identified the way GitHub itself writes it.
    """

    owner: str
    """The user or organization the repository belongs to."""

    name: str
    """The repository's own name."""

    @classmethod
    def parse(cls, text: str) -> Repository:
        """
        :param text: An ``owner/name`` reference.
        :return: The repository it names.
        :raises MalformedRepositoryError: If *text* is not ``owner/name``.
        """
        owner, separator, name = text.partition("/")
        if not (owner and separator and name) or "/" in name:
            raise MalformedRepositoryError(text)
        return cls(owner, name)

    @classmethod
    def from_remote_url(cls, url: str) -> Repository | None:
        """
        Read the repository a git remote URL points at.

        Accepts HTTPS and SSH remotes on GitHub, and the local proxy a cloud session is
        given, whose URL never names GitHub but carries the repository as its last two
        path segments.

        :param url: The remote URL.
        :return: The repository, or ``None`` if the URL names no GitHub repository (a
            local path, or another host).
        """
        reference = url.removesuffix(".git").rstrip("/")
        if "://" in reference:
            _, _, host_and_path = reference.partition("://")
            authority, _, path = host_and_path.partition("/")
        elif ":" in reference:
            authority, _, path = reference.partition(":")
        else:
            return None
        host = authority.rpartition("@")[2].partition(":")[0].lower()
        is_github = host == GITHUB_HOST or host.endswith("." + GITHUB_HOST)
        if not (is_github or host in LOOPBACK_HOSTS):
            return None
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) < 2:
            return None
        return cls(segments[-2], segments[-1])

    @property
    def full_name(self) -> str:
        """The ``owner/name`` form the ``gh`` CLI and GitHub's own interface use."""
        return f"{self.owner}/{self.name}"

    def __str__(self) -> str:
        """:return: :attr:`full_name`."""
        return self.full_name

    @property
    def web_url(self) -> str:
        """The repository's page."""
        return f"{GITHUB_WEB_ROOT}/{self.full_name}"

    @property
    def labels_url(self) -> str:
        """The page where labels are created by hand."""
        return f"{self.web_url}/labels"

    def blob_url(self, branch: str, path: str) -> str:
        """
        :param branch: The branch the file is on.
        :param path: The file's path in the repository.
        :return: The page GitHub renders the file at.
        """
        return f"{self.web_url}/blob/{branch}/{path}"
