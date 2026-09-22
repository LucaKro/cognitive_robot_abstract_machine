"""
Tests for install-gh-stack.sh: making ``gh stack`` runnable in a cloud session, and the
one line session-start.sh reports about it. Nothing here reaches the network: ``gh`` and
``curl`` are stubs, and a download is only ever attempted against the curl stub.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from .constants import SCRUBBED_ENVIRONMENT_PREFIXES, ToolingDirectory
from .executable_stubs import ExecutableStubDirectory, path_hiding_executable
from .script_runner import BashScriptRunner

INSTALLER = ToolingDirectory.HOOKS.path / "install-gh-stack.sh"


@pytest.fixture
def stub_bin(tmp_path: Path) -> ExecutableStubDirectory:
    """
    :return: A stub directory carrying ``gh`` and ``curl``.
    """
    stubs = ExecutableStubDirectory.create(tmp_path)
    stubs.install("gh")
    stubs.install("curl")
    return stubs


def run_installer(
    tmp_path: Path, search_path: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    """
    :param tmp_path: Where the tools directory goes.
    :param search_path: The ``PATH`` the script runs with.
    :param environment: Further variables for the run.
    :return: The finished script.
    """
    return BashScriptRunner(
        project_root=tmp_path,
        removed_variable_prefixes=SCRUBBED_ENVIRONMENT_PREFIXES,
        script_path=INSTALLER,
    ).run(
        PATH=search_path,
        GH_STACK_TOOLS_DIRECTORY=str(tmp_path / "tools"),
        **environment,
    )


def test_a_runnable_gh_stack_is_reported_ready_and_nothing_is_installed(
    tmp_path: Path, stub_bin: ExecutableStubDirectory
):
    result = run_installer(
        tmp_path, stub_bin.ahead_of(os.environ["PATH"]),
        STUB_GH_STACK_RUNS="1", CLAUDE_CODE_REMOTE="true",
    )

    assert result.returncode == 0
    assert result.stdout == "ready (gh 2.90.0)\n"
    assert not (tmp_path / "tools").exists()


def test_outside_a_cloud_session_it_only_says_how_to_install(
    tmp_path: Path, stub_bin: ExecutableStubDirectory
):
    result = run_installer(tmp_path, stub_bin.ahead_of(os.environ["PATH"]))

    assert result.returncode == 1
    assert result.stdout.startswith("unavailable - install gh >= 2.90.0")
    assert "gh extension install github/gh-stack" in result.stdout
    assert not (tmp_path / "tools").exists()


def test_a_failed_gh_download_is_reported_not_fatal(
    tmp_path: Path, stub_bin: ExecutableStubDirectory
):
    result = run_installer(
        tmp_path, stub_bin.ahead_of(os.environ["PATH"]),
        STUB_GH_VERSION="2.80.0", CLAUDE_CODE_REMOTE="true",
    )

    assert result.returncode == 1
    assert result.stdout.startswith("unavailable - ")
    assert result.stdout.count("\n") == 1


def test_without_go_the_extension_is_reported_unbuildable(
    tmp_path: Path, stub_bin: ExecutableStubDirectory
):
    search_path = stub_bin.ahead_of(path_hiding_executable("go", tmp_path))

    result = run_installer(tmp_path, search_path, GH_STACK_INSTALL="1")

    assert result.returncode == 1
    assert result.stdout == "unavailable - gh-stack has to be built from source and there is no go\n"
