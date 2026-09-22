"""
Tests for roadmap_location.py: where GitHub renders a plan's roadmap on the notes branch,
so the dashboard can link each item's history instead of carrying it.
"""

from __future__ import annotations

import pytest

from basstler import roadmap_location
from basstler.setup_steps import Repository

from .scratch_repository import ScratchRepository


@pytest.mark.parametrize(
    "remote_url",
    ["git@github.com:owner/repo.git", "https://github.com/owner/repo", "https://github.com/owner/repo.git"],
)
def test_a_github_remote_gives_the_rendered_files_address(remote_url: str):
    repository = Repository.from_remote_url(remote_url)

    assert repository.blob_url("claude/personal-notes", ".claude/personal/plans/p/roadmap.md") == (
        "https://github.com/owner/repo/blob/claude/personal-notes/.claude/personal/plans/p/roadmap.md"
    )


def test_the_command_resolves_the_notes_remote_by_name(
    scratch_repository: ScratchRepository, capsys: pytest.CaptureFixture
):
    scratch_repository.run_git("remote", "add", "fork", "git@github.com:owner/repo.git")

    exit_code = roadmap_location.main([
        "--project-root", str(scratch_repository.project_root),
        "--notes-remote", "fork", "--notes-branch", "notes", "--path", "plans/p/roadmap.md",
    ])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "https://github.com/owner/repo/blob/notes/plans/p/roadmap.md"


def test_a_remote_that_is_not_on_github_gives_no_address(
    scratch_repository: ScratchRepository, capsys: pytest.CaptureFixture
):
    exit_code = roadmap_location.main([
        "--project-root", str(scratch_repository.project_root),
        "--notes-remote", "/some/local/path", "--notes-branch", "notes", "--path", "x.md",
    ])

    assert exit_code == roadmap_location.EXIT_NOT_ON_GITHUB
    assert capsys.readouterr().out == ""
