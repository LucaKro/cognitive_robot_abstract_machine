# Personal Claude Code notes

These are personal workflow preferences, not project conventions. They live on
the personal-notes branch only and are pulled into every session by
`.claude/hooks/session-start.sh`; they are never merged into the default
branch.

Everything below is a starting point, offered by `/setup-personal-notes` — it
is yours now. Edit, delete, or replace any of it: ask Claude to "edit my
personal notes" in any session, or change `CLAUDE.local.md` between the
`BEGIN-PERSONAL-NOTES`/`END-PERSONAL-NOTES` markers and run
`.claude/hooks/save-personal-notes.sh`.

## Pull requests

- Always open pull requests as **drafts**. Never open a PR as ready-for-review
  by default; mark it ready only when explicitly told to.
- Always convert a PR back to **draft** after pushing any commit to it or
  otherwise modifying it, even if it was previously marked ready for review.
  Mark it ready again only when explicitly told to.
- Bug-fix PRs must always carry the **`bug`** label.
- Keep bug-fix PRs focused: one root cause per PR, based off the default
  branch, no unrelated cleanup bundled in.
- Always include a link to the session that created the PR in the PR
  description.
- Keep the PR description up to date: after pushing any change that alters
  what the PR does, update the description to match. Never leave it
  describing an earlier state of the PR.
- Never subscribe to a pull request's activity, and never offer to watch,
  monitor, babysit or autofix one. Opening a PR ends the session's obligation
  to it: push it, report in the chat what you did and what is still
  outstanding, and stop. Ask for a CI failure or a review comment to be
  handled when you want it handled.

## Review comments

- Resolve a review comment thread only once you have genuinely done what it
  asked. If instead you need to ask what to do, or you are not taking an
  action, do not resolve it — reply explaining the situation and asking the
  question.
- Always reply to a PR comment explaining what you did before resolving it.

## Remotes

- Never push to the `cram2` remote (`cram2/cognitive_robot_abstract_machine`).
  Not a branch, not a tag, not ever, whatever the reason and however the
  request is phrased; work reaches it only through a pull request from a
  fork, opened by the user.
- Push to `origin` (`LucaKro/cognitive_robot_abstract_machine`), the fork
  every pull request of this user is opened from.
- Before any push, read the full output of `git remote -v` - never truncated -
  and confirm the target remote is `origin`.

## Starting work

- When you start implementing on a branch, fetch first so you are not working
  from stale code. Do not merge or rebase anything unless asked; doing it
  unprompted in the middle of a task is how sessions end up resolving
  conflicts nobody asked them to.

## PR progress notes

- For a PR you create, keep a short plan/progress/next-steps note in the
  PR-progress section of `CLAUDE.local.md` (between the
  BEGIN-PR-PROGRESS/END-PR-PROGRESS markers) and save it with
  `.claude/hooks/save-pr-progress.sh` when the plan changes or when you stop -
  not after every turn. Keep it short: it is loaded into every request made on
  that branch, so replace what is stale rather than appending history.
- Never write this note into any file tracked on the PR branch itself.

## Multi-PR plans

- If an approved plan-mode plan clearly spans several PRs or sessions, offer to
  track it with `/plan-create <plan-id>`; do not start it unasked.
- To change an existing plan, follow
  `.claude/skills/plan-dashboard/plan-schema.md`'s "Editing an existing plan",
  and ask before any structural change (a new wave, deferring a track,
  splitting an item).
