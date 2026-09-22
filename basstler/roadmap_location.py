#!/usr/bin/env python3
"""
Where GitHub renders a plan's roadmap on the notes branch.

The dashboard links each item to its history there instead of carrying the whole roadmap:
GitHub renders the file with an anchor per heading, and linking costs nothing to publish.

Usage:
    python3 -m basstler.roadmap_location --project-root <root> --notes-remote <remote> \\
        --notes-branch <branch> --path <path/to/roadmap.md>

Prints the URL, or nothing and exits with :data:`EXIT_NOT_ON_GITHUB` when the notes
remote is not a GitHub repository.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from basstler.setup_steps import resolve_repository

EXIT_NOT_ON_GITHUB = 2
"""
Exit code when the notes remote resolves to no GitHub repository.
"""


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Print where GitHub renders the file. See the module docstring.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project-root", required=True, help="The clone to read remotes from")
    parser.add_argument("--notes-remote", required=True, help="The notes remote, a name or URL")
    parser.add_argument("--notes-branch", required=True, help="The notes branch")
    parser.add_argument("--path", required=True, help="The file's path on that branch")
    parsed = parser.parse_args(arguments)

    repository = resolve_repository(Path(parsed.project_root), parsed.notes_remote)
    if repository is None:
        return EXIT_NOT_ON_GITHUB
    print(repository.blob_url(parsed.notes_branch, parsed.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
