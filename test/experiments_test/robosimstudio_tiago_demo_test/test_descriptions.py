"""
Fetching the description packages and making them resolvable without ROS.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.robosimstudio_tiago_demo.descriptions import (
    DescriptionPackages,
    DescriptionRepository,
    EnvironmentVariable,
)
from semantic_digital_twin.adapters.package_resolver import ROSPackagePathLocator

# %% resolving


def test_fetched_packages_resolve_through_the_package_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    packages = DescriptionPackages(tmp_path)
    for directory in packages.package_directories:
        directory.mkdir(parents=True)
    monkeypatch.setenv(EnvironmentVariable.ROS_PACKAGE_PATH, "")

    packages.make_resolvable()

    assert [
        ROSPackagePathLocator().resolve(directory.name)
        for directory in packages.package_directories
    ] == [str(directory) for directory in packages.package_directories]


# %% fetching


def git(repository: Path, *arguments: str) -> str:
    """
    Run git in ``repository``.

    :return: What git printed.
    """
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository_with_two_packages(tmp_path: Path) -> tuple[Path, str]:
    """
    A local repository holding the packages ``wanted`` and ``unwanted``.

    :return: The repository and its only commit.
    """
    repository = tmp_path / "origin" / "descriptions"
    for package in ("wanted", "unwanted"):
        (repository / package).mkdir(parents=True)
        (repository / package / "package.xml").write_text(package)
    git(repository, "init", "--quiet")
    git(repository, "add", ".")
    git(
        repository,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "--message=packages",
    )
    return repository, git(repository, "rev-parse", "HEAD")


def test_a_fetch_checks_out_only_the_needed_packages_at_the_pinned_commit(
    tmp_path: Path, repository_with_two_packages: tuple[Path, str]
):
    origin, commit = repository_with_two_packages
    repository = DescriptionRepository(
        url=origin.as_uri(), commit=commit, packages=("wanted",)
    )
    cache_directory = tmp_path / "cache"

    repository.fetch(cache_directory)

    checkout = cache_directory / repository.directory_name
    assert repository.is_fetched(cache_directory)
    assert not (checkout / "unwanted").exists()
    assert git(checkout, "rev-parse", "HEAD") == commit
