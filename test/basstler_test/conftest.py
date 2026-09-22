"""
Fixtures shared by every suite in this directory, and the one ``sys.path`` entry they
need.

``basstler`` is a plain top-level directory on the repository root, importable with no
install, so the root just has to be on ``sys.path`` when this directory is run from an
arbitrary working directory. That single entry replaces the three conftests this suite
was merged from, each of which inserted its own directory - and one of which reached
across into two others, which is the path hackery the package exists to end.

This suite runs in the lightweight ``test_basstler`` CI job with ``--confcutdir`` pointed
here, so the repository-root ``test/conftest.py`` - which imports the robotics stack that
job does not install - is never loaded for it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest  # noqa: E402

from .executable_stubs import ExecutableStubDirectory  # noqa: E402
from .scratch_repository import ScratchRepository  # noqa: E402
from .upstream_reviews_replay import RecordedResponse, ReplayingClient  # noqa: E402


@pytest.fixture
def scratch_repository(tmp_path: Path) -> ScratchRepository:
    """
    An initialized scratch project root and its bare notes remote, with nothing
    committed and no personal-notes branch published yet.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch repository.
    """
    return ScratchRepository.create(tmp_path)


@pytest.fixture
def stub_bin(tmp_path: Path) -> ExecutableStubDirectory:
    """
    An empty directory meant to be placed first on a test subprocess's PATH, into which
    a test installs whichever stubbed executable it needs.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The stub directory.
    """
    return ExecutableStubDirectory.create(tmp_path)


@pytest.fixture
def paginated_client() -> ReplayingClient:
    """:return: A client replaying both pages of the recorded review threads."""
    return ReplayingClient(
        [
            RecordedResponse.PULL_REQUEST_PAGE_ONE.load(),
            RecordedResponse.PULL_REQUEST_PAGE_TWO.load(),
        ]
    )
