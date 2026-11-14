"""
The base every code rule checker extends, and the violations checkers report.
"""

from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from typing_extensions import Iterator, Sequence, override

from krrood.development_tools.exceptions import GitCommandFailed

# %% violations


@dataclass(kw_only=True, frozen=True)
class CodeRuleViolation:
    """
    One place in a source file that breaks a rule.
    """

    path: Path
    """
    The file breaking the rule.
    """

    first_line: int
    """
    The first line of the offending code, counting from one.
    """

    last_line: int
    """
    The last line of the offending code, counting from one.
    """

    description: str
    """
    What is wrong and how to fix it, as one sentence fragment.
    """

    def touches(self, *, lines: set[int]) -> bool:
        """
        :param lines: Line numbers in the file.
        :return: Whether any of them falls within the offending code.
        """
        return any(self.first_line <= line <= self.last_line for line in lines)

    def render(self) -> str:
        """
        :return: The violation as it is printed.
        """
        return f"{self.path}:{self.first_line}: {self.description}"


# %% checkers


@dataclass(kw_only=True, frozen=True)
class CodeRuleChecker(ABC):
    """
    Finds the code that breaks one agent rule, the rule that names this checker.

    Checkers are frozen so that a rule holding one stays hashable.
    """

    @abstractmethod
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        """
        :param path: A Python source file.
        :return: Every place in it that breaks the rule.
        """


# %% the change a check is restricted to


class GitArgument(StrEnum):
    """
    The git subcommands and options listing a change without surrounding context.
    """

    GIT = "git"
    """
    The git program.
    """

    DIFFERENCE = "diff"
    """
    List the differences between two versions of the files.
    """

    CACHED = "--cached"
    """
    Compare the staged files with the last commit.
    """

    NO_CONTEXT = "--unified=0"
    """
    List the changed lines alone, without the lines around them.
    """

    NO_COLOR = "--no-color"
    """
    Print the differences without terminal colours.
    """

    NAME_ONLY = "--name-only"
    """
    List only the paths of the changed files.
    """

    RELATIVE = "--relative"
    """
    List the paths relative to the working directory, as the other listings do.
    """

    MERGE_BASE = "merge-base"
    """
    Find the newest commit two commits share in their history.
    """

    HEAD = "HEAD"
    """
    The commit the working tree is checked out from.
    """

    EXCLUDING_DELETED = "--diff-filter=d"
    """
    Leave out the files the change deletes.
    """

    LIST_FILES = "ls-files"
    """
    List the files of the repository.
    """

    UNTRACKED = "--others"
    """
    List the files git does not track.
    """

    EXCLUDE_IGNORED = "--exclude-standard"
    """
    Leave out the files the repository's ignore rules match.
    """

    PATHS_FOLLOW = "--"
    """
    Read every following argument as a path, never as an option or revision.
    """

    PYTHON_FILES = "*.py"
    """
    The path pattern matching every Python file, at any depth.
    """


class DifferencePattern(StrEnum):
    """
    The patterns read from the output of ``git diff``.
    """

    HUNK_HEADER = r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@"
    """
    The header of a hunk, capturing the first line it covers in the new file and, when
    it is not one, how many lines.
    """


@dataclass(kw_only=True, frozen=True)
class ChangedLines(ABC):
    """
    The lines a change adds or alters in the files of a git repository.
    """

    repository: Path
    """
    The directory of the git repository the change is made in.
    """

    @property
    @abstractmethod
    def compared_version(self) -> tuple[str, ...]:
        """
        The arguments of ``git diff`` naming what the change is compared with.
        """

    def find_changed_lines(self, *, path: Path) -> set[int]:
        """
        :param path: A file in the repository.
        :return: The line numbers, in the changed file, that the change adds or alters.
        :raises GitCommandFailed: If git cannot list the change.
        """
        difference = self.run_git(
            arguments=(
                GitArgument.DIFFERENCE,
                *self.compared_version,
                GitArgument.NO_CONTEXT,
                GitArgument.NO_COLOR,
                GitArgument.PATHS_FOLLOW,
                str(path),
            )
        )
        lines: set[int] = set()
        for hunk in re.finditer(
            DifferencePattern.HUNK_HEADER, difference, re.MULTILINE
        ):
            start = int(hunk[1])
            count = 1 if hunk[2] is None else int(hunk[2])
            lines.update(range(start, start + count))
        return lines

    def run_git(self, *, arguments: tuple[str, ...]) -> str:
        """
        :param arguments: The git subcommand and its arguments.
        :return: What git printed.
        :raises GitCommandFailed: If git exits with an error.
        """
        result = subprocess.run(
            [GitArgument.GIT, *arguments],
            cwd=self.repository,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GitCommandFailed(arguments=arguments, standard_error=result.stderr)
        return result.stdout


@dataclass(kw_only=True, frozen=True)
class StagedLines(ChangedLines):
    """
    The lines a commit being prepared adds or changes.
    """

    @property
    @override
    def compared_version(self) -> tuple[str, ...]:
        return (GitArgument.CACHED,)


@dataclass(kw_only=True, frozen=True)
class LinesChangedSince(ChangedLines):
    """
    The lines the working tree adds or alters since it forked from a revision, staged or
    not.
    """

    revision: str
    """
    The git revision the working tree forked from, such as a commit or a branch.
    """

    @cached_property
    def merge_base(self) -> str:
        """
        The commit the working tree is compared with: the newest one :attr:`revision`
        and ``HEAD`` share, so what :attr:`revision` changed after it does not count.

        :raises GitCommandFailed: If git cannot resolve the revision.
        """
        return self.run_git(
            arguments=(GitArgument.MERGE_BASE, self.revision, GitArgument.HEAD)
        ).strip()

    @property
    @override
    def compared_version(self) -> tuple[str, ...]:
        return (self.merge_base,)

    @override
    def find_changed_lines(self, *, path: Path) -> set[int]:
        """
        Unlike :meth:`ChangedLines.find_changed_lines`, every line of a file git does
        not track counts as changed.
        """
        untracked = self.run_git(
            arguments=(
                GitArgument.LIST_FILES,
                GitArgument.UNTRACKED,
                GitArgument.EXCLUDE_IGNORED,
                GitArgument.PATHS_FOLLOW,
                str(path),
            )
        )
        if untracked:
            return set(range(1, len(path.read_text().splitlines()) + 1))
        return super().find_changed_lines(path=path)

    def find_changed_files(self) -> list[Path]:
        """
        :return: The Python files below :attr:`repository` the working tree adds or
            modifies since :attr:`merge_base`, tracked or not, leaving out those the
            repository ignores.
        :raises GitCommandFailed: If git cannot resolve the revision.
        """
        tracked = self.run_git(
            arguments=(
                GitArgument.DIFFERENCE,
                GitArgument.NAME_ONLY,
                GitArgument.RELATIVE,
                GitArgument.EXCLUDING_DELETED,
                self.merge_base,
                GitArgument.PATHS_FOLLOW,
                GitArgument.PYTHON_FILES,
            )
        )
        untracked = self.run_git(
            arguments=(
                GitArgument.LIST_FILES,
                GitArgument.UNTRACKED,
                GitArgument.EXCLUDE_IGNORED,
                GitArgument.PATHS_FOLLOW,
                GitArgument.PYTHON_FILES,
            )
        )
        return [self.repository / name for name in (tracked + untracked).splitlines()]


# %% checking files


@dataclass(kw_only=True, frozen=True)
class CodeRulesCheck:
    """
    Runs a set of checkers over a set of files.
    """

    checkers: tuple[CodeRuleChecker, ...]
    """
    The checkers to run.
    """

    changed_lines: ChangedLines | None = None
    """
    The change whose code alone is checked; none to check all of it.
    """

    def violations(self, *, paths: Sequence[Path]) -> list[CodeRuleViolation]:
        """
        :param paths: The Python files to check.
        :return: Their violations, file by file, top to bottom.
        """
        violations: list[CodeRuleViolation] = []
        for path in paths:
            changed = (
                None
                if self.changed_lines is None
                else self.changed_lines.find_changed_lines(path=path)
            )
            found = [
                violation
                for checker in self.checkers
                for violation in checker.violations(path=path)
                if changed is None or violation.touches(lines=changed)
            ]
            violations.extend(sorted(found, key=lambda violation: violation.first_line))
        return violations
