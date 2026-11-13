"""
Tests of the checkers that enforce a rule through ruff's lint rules.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path

import pytest
from typing_extensions import Callable

from krrood.development_tools.agents import ruff_rules
from krrood.development_tools.agents.rule import AgentRule
from krrood.development_tools.agents.ruff_rules import (
    CheckedModules,
    PytestPathPattern,
    RuffRule,
    RuffRuleChecker,
    RuffSetting,
)
from krrood.development_tools.agents.shipped_rules.code_style import CodeStyleRules
from krrood.development_tools.agents.shipped_rules.design_principles import (
    ExceptionModellingRules,
)
from krrood.development_tools.agents.shipped_rules.imports import ImportsRules
from krrood.development_tools.agents.shipped_rules.testing import WritingTestsRules
from krrood.development_tools.agents.shipped_rules.type_hints import TypeHintsRules
from krrood.development_tools.exceptions import RuffCheckFailed

from .dataset_files import DatasetFile, DatasetMarker, RuffSnippet

# %% file names a copied snippet is checked under


class CopiedModuleName(StrEnum):
    """
    The names a snippet is copied under, inside and outside a test suite.
    """

    OUTSIDE_TESTS = "module.py"
    """
    A module pytest does not collect.
    """

    TEST = "test_module.py"
    """
    A module pytest collects as a test.
    """


# %% helpers


def marked_lines(path: Path) -> list[int]:
    """
    :return: The lines of *path* marked with :attr:`DatasetMarker.RUFF_FINDING`.
    """
    return [
        number
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.endswith(DatasetMarker.RUFF_FINDING)
    ]


def reported_lines(checker: RuffRuleChecker, path: Path) -> list[int]:
    """
    :return: The first line of each violation *checker* reports in *path*, top to bottom.
    """
    return sorted(violation.first_line for violation in checker.violations(path=path))


# %% telling test modules apart


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("test", "module.py"),
        Path("tests", "package", "module.py"),
        Path("test_module.py"),
        Path("module_test.py"),
        Path("conftest.py"),
    ],
)
def test_a_path_in_a_test_suite_is_a_test_module(
    repository_root: Path, relative_path: Path
) -> None:
    """
    A test directory on the path, or a file named the way pytest collects, marks a test.
    """
    assert (
        PytestPathPattern.is_test_module(path=repository_root / relative_path) is True
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("src", "module.py"),
        Path("testing", "module.py"),
        Path("contest.py"),
    ],
)
def test_a_path_outside_a_test_suite_is_no_test_module(
    repository_root: Path, relative_path: Path
) -> None:
    """
    Names that merely resemble the test conventions do not mark a test.
    """
    assert (
        PytestPathPattern.is_test_module(path=repository_root / relative_path) is False
    )


def test_directories_above_the_working_directory_do_not_mark_a_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A repository checked out inside a directory named ``test`` holds no test only by that.
    """
    working_directory = tmp_path / "test" / "repository"
    working_directory.mkdir(parents=True)
    monkeypatch.chdir(working_directory)

    assert (
        PytestPathPattern.is_test_module(path=working_directory / "module.py") is False
    )


# %% reporting what ruff finds


def test_only_the_selected_lint_rules_are_reported(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
) -> None:
    """
    The snippet also asserts, which the checker leaves alone since it selects only
    mutable defaults.
    """
    path = copy_into_temporary_directory(
        RuffSnippet.MUTABLE_DEFAULT, CopiedModuleName.OUTSIDE_TESTS
    )
    checker = RuffRuleChecker(lint_rules=(RuffRule.MUTABLE_ARGUMENT_DEFAULT,))

    assert reported_lines(checker, path) == marked_lines(path)


def test_a_violation_names_the_lint_rule_ruff_reports(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
) -> None:
    """
    The description ends in the code of the lint rule, so it can be looked up.
    """
    path = copy_into_temporary_directory(
        RuffSnippet.MUTABLE_DEFAULT, CopiedModuleName.OUTSIDE_TESTS
    )
    checker = RuffRuleChecker(lint_rules=(RuffRule.MUTABLE_ARGUMENT_DEFAULT,))

    [violation] = checker.violations(path=path)

    assert violation.description.endswith(f"(ruff {RuffRule.MUTABLE_ARGUMENT_DEFAULT})")


@pytest.mark.parametrize(
    "settings, reports_the_marked_lines",
    [
        ((), False),
        ((RuffSetting.RELATIVE_IMPORTS_BANNED,), True),
    ],
)
def test_a_setting_configures_its_lint_rule(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
    settings: tuple[RuffSetting, ...],
    reports_the_marked_lines: bool,
) -> None:
    """
    By default ruff bans only relative imports from parent packages; the setting bans
    imports from the same package too.
    """
    path = copy_into_temporary_directory(
        RuffSnippet.RELATIVE_IMPORTS, CopiedModuleName.OUTSIDE_TESTS
    )
    checker = RuffRuleChecker(
        lint_rules=(RuffRule.RELATIVE_IMPORTS,), settings=settings
    )
    expected_lines = marked_lines(path) if reports_the_marked_lines else []

    assert reported_lines(checker, path) == expected_lines


@pytest.mark.parametrize(
    "modules, file_name",
    [
        (CheckedModules.OUTSIDE_TESTS, CopiedModuleName.TEST),
        (CheckedModules.TESTS, CopiedModuleName.OUTSIDE_TESTS),
    ],
)
def test_a_module_the_rule_does_not_cover_is_not_checked(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
    modules: CheckedModules,
    file_name: str,
) -> None:
    """
    An ``assert`` goes unreported where the checker does not look.
    """
    path = copy_into_temporary_directory(RuffSnippet.ASSERTION, file_name)
    checker = RuffRuleChecker(lint_rules=(RuffRule.ASSERT,), modules=modules)

    assert reported_lines(checker, path) == []


def test_a_failing_ruff_run_raises(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A program that rejects ruff's arguments stands in for a ruff that cannot run.
    """
    path = copy_into_temporary_directory(
        RuffSnippet.ASSERTION, CopiedModuleName.OUTSIDE_TESTS
    )
    monkeypatch.setattr(ruff_rules, "find_ruff_bin", lambda: sys.executable)
    checker = RuffRuleChecker(lint_rules=(RuffRule.ASSERT,))

    with pytest.raises(RuffCheckFailed):
        list(checker.violations(path=path))


# %% the shipped rules ruff enforces


@pytest.mark.parametrize(
    "rule, snippet, file_name",
    [
        (
            CodeStyleRules.NO_MUTABLE_DEFAULT_ARGUMENTS,
            RuffSnippet.MUTABLE_DEFAULT,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
        (
            ExceptionModellingRules.NO_ASSERT_OUTSIDE_TESTS,
            RuffSnippet.ASSERTION,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
        (
            TypeHintsRules.ANNOTATE_EVERYTHING,
            RuffSnippet.MISSING_TYPE_HINTS,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
        (
            TypeHintsRules.BUILTIN_GENERICS,
            RuffSnippet.TYPING_CONSTRUCTS,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
        (
            TypeHintsRules.FUTURE_ANNOTATIONS,
            RuffSnippet.MISSING_FUTURE_IMPORT,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
        (
            TypeHintsRules.FUTURE_ANNOTATIONS,
            RuffSnippet.QUOTED_ANNOTATIONS,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
        (
            ImportsRules.ABSOLUTE_IMPORTS,
            RuffSnippet.RELATIVE_IMPORTS,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
        (
            WritingTestsRules.NO_UNITTEST_TEST_CASES,
            RuffSnippet.UNITTEST_CASE,
            CopiedModuleName.TEST,
        ),
    ],
)
def test_a_shipped_rule_reports_the_code_breaking_it(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
    rule: AgentRule,
    snippet: RuffSnippet,
    file_name: str,
) -> None:
    """
    Each rule's checker reports exactly the marked lines of the snippet breaking it.
    """
    path = copy_into_temporary_directory(snippet, file_name)

    assert reported_lines(rule.checker, path) == marked_lines(path)


@pytest.mark.parametrize(
    "rule, snippet, file_name",
    [
        (
            ExceptionModellingRules.NO_ASSERT_OUTSIDE_TESTS,
            RuffSnippet.ASSERTION,
            CopiedModuleName.TEST,
        ),
        (
            ImportsRules.ABSOLUTE_IMPORTS,
            RuffSnippet.RELATIVE_IMPORTS,
            CopiedModuleName.TEST,
        ),
        (
            WritingTestsRules.NO_UNITTEST_TEST_CASES,
            RuffSnippet.UNITTEST_CASE,
            CopiedModuleName.OUTSIDE_TESTS,
        ),
    ],
)
def test_a_shipped_rule_leaves_the_modules_it_exempts_alone(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
    rule: AgentRule,
    snippet: RuffSnippet,
    file_name: str,
) -> None:
    """
    Tests may assert and import relatively; only tests are held to pytest's style.
    """
    path = copy_into_temporary_directory(snippet, file_name)

    assert reported_lines(rule.checker, path) == []
