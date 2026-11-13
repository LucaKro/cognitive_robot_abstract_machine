#!/usr/bin/env bash
# Commits the paths given as github-actions[bot] and pushes the commit to the notes
# branch, taking in what was pushed there meanwhile; commits nothing when none of them
# changed. Run by the notes branch's workflows in a checkout of the notes branch.
#
# Usage: commit-notes-branch.sh <message> <notes branch> <path>...
set -euo pipefail

MESSAGE="$1"
NOTES_BRANCH="$2"
shift 2

export GIT_AUTHOR_NAME="github-actions[bot]"
export GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
export GIT_COMMITTER_NAME="${GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_AUTHOR_EMAIL}"

git add --all -- "$@"
if git diff --cached --quiet; then
  exit 0
fi
git commit --quiet --message "${MESSAGE}"
for attempt in 1 2 3; do
  if git push --quiet origin "HEAD:refs/heads/${NOTES_BRANCH}"; then
    exit 0
  fi
  git pull --quiet --rebase origin "${NOTES_BRANCH}"
done
exit 1
