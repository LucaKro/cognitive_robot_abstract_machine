"""
The ROS description packages the demonstration's world is built from, fetched at the
commits the demonstration was made with, so it runs without a ROS workspace.

Run this module to fetch them ahead of time::

    python -m experiments.robosimstudio_tiago_demo.descriptions
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

# %% git


class GitArgument(StrEnum):
    """
    The git subcommands and options a fetch runs with.
    """

    EXECUTABLE = "git"
    IN_DIRECTORY = "-C"
    INIT = "init"
    FETCH = "fetch"
    SPARSE_CHECKOUT = "sparse-checkout"
    SET = "set"
    CHECKOUT = "checkout"
    QUIET = "--quiet"
    LATEST_COMMIT_ONLY = "--depth=1"
    WITHOUT_FILE_CONTENTS = "--filter=blob:none"
    BY_PATTERN = "--no-cone"


class EnvironmentVariable(StrEnum):
    """
    The environment variables the demonstration sets.
    """

    ROS_PACKAGE_PATH = "ROS_PACKAGE_PATH"


# %% repositories


@dataclass(frozen=True)
class DescriptionRepository:
    """
    A git repository holding ROS description packages, pinned to one commit.
    """

    url: str
    """
    Where the repository is cloned from.
    """

    commit: str
    """
    The commit the demonstration was made with.
    """

    packages: tuple[str, ...] = ()
    """
    The package directories the demonstration needs, relative to the repository root;
    empty when the repository is itself the package.
    """

    @property
    def directory_name(self) -> str:
        """
        The name of the directory the repository is fetched into.
        """
        return Path(self.url).stem

    def package_directories(self, cache_directory: Path) -> list[Path]:
        """
        :param cache_directory: The directory the repository is fetched under.
        :return: The directory of each needed package.
        """
        root = (cache_directory / self.directory_name).resolve()
        if not self.packages:
            return [root]
        return [root / package for package in self.packages]

    def is_fetched(self, cache_directory: Path) -> bool:
        """
        :param cache_directory: The directory the repository is fetched under.
        :return: Whether every needed package is already there.
        """
        return all(
            directory.is_dir()
            for directory in self.package_directories(cache_directory)
        )

    def fetch(self, cache_directory: Path) -> None:
        """
        Fetch the pinned commit, checking out only the needed packages.

        :param cache_directory: The directory to fetch the repository under.
        """
        root = cache_directory / self.directory_name
        root.mkdir(parents=True, exist_ok=True)
        commands = [
            [GitArgument.INIT, GitArgument.QUIET],
            [
                GitArgument.FETCH,
                GitArgument.QUIET,
                GitArgument.LATEST_COMMIT_ONLY,
                GitArgument.WITHOUT_FILE_CONTENTS,
                self.url,
                self.commit,
            ],
        ]
        if self.packages:
            commands.append(
                [
                    GitArgument.SPARSE_CHECKOUT,
                    GitArgument.SET,
                    GitArgument.BY_PATTERN,
                    *(f"/{package}/" for package in self.packages),
                ]
            )
        commands.append([GitArgument.CHECKOUT, GitArgument.QUIET, self.commit])
        for command in commands:
            subprocess.run(
                [GitArgument.EXECUTABLE, GitArgument.IN_DIRECTORY, str(root), *command],
                check=True,
            )


class DescriptionRepositories(Enum):
    """
    The repositories holding the apartment's meshes and the TIAGo's description.
    """

    MAPS = DescriptionRepository(
        url="https://github.com/code-iai/iai_maps.git",
        commit="66746efda6ef81945c2295a18459c5d134032574",
        packages=("iai_apartment", "iai_kitchen"),
    )
    TIAGO = DescriptionRepository(
        url="https://github.com/code-iai/iai_tiago_description.git",
        commit="095ffd882a68cbabd9f7c82ce6fd56fbe4e48c53",
    )


# %% package path


@dataclass
class DescriptionPackages:
    """
    The description packages of the demonstration, kept in one cache directory.
    """

    cache_directory: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "robosimstudio_tiago_demo"
    )
    """
    The directory the repositories are fetched under.
    """

    def fetch_missing(self) -> None:
        """
        Fetch every repository whose packages are not in the cache yet.
        """
        for repository in DescriptionRepositories:
            if not repository.value.is_fetched(self.cache_directory):
                repository.value.fetch(self.cache_directory)

    @property
    def package_directories(self) -> list[Path]:
        """
        The directory of every needed package.
        """
        return [
            directory
            for repository in DescriptionRepositories
            for directory in repository.value.package_directories(self.cache_directory)
        ]

    def make_resolvable(self) -> None:
        """
        Fetch what is missing and put every package on the package path, so that
        ``package://`` locations in the descriptions resolve to the cache.
        """
        self.fetch_missing()
        existing = os.environ.get(EnvironmentVariable.ROS_PACKAGE_PATH, "")
        os.environ[EnvironmentVariable.ROS_PACKAGE_PATH] = os.pathsep.join(
            [*map(str, self.package_directories), *filter(None, [existing])]
        )


if __name__ == "__main__":
    DescriptionPackages().fetch_missing()
