---
name: plan-create
description: Create (or migrate an existing bespoke roadmap doc into) a new multi-PR/multi-session plan under .claude/personal/plans/<plan-id>/plan.yaml on the personal-notes branch, cross-checked against live GitHub, then bootstrap and publish it. Invoke as "/plan-create <plan-id>". Use when the user asks to start tracking something as a plan, set up a new plan/roadmap, or migrate an existing roadmap doc into the plan-dashboard system.
allowed-tools: Bash, Read, Write, Grep, Glob, AskUserQuestion, Skill, mcp__github__issue_write, mcp__github__create_pull_request
---

# Plan Create

Drafts a new plan's `plan.yaml` and `roadmap.md`, checks them against live
GitHub, saves them to the personal-notes branch and publishes the dashboard.
The schema is in `.claude/skills/plan-dashboard/plan-schema.md`; read it before
drafting.

## 1. The plan id

```bash
source .claude/hooks/resolve-personal-notes-config.sh
git fetch "${NOTES_REMOTE}" "${NOTES_BRANCH}" --quiet
git cat-file -e "FETCH_HEAD:${PLANS_DIR}/<plan-id>/plan.yaml" && echo "exists"
```

If no id was given, ask for one (short, kebab-case): it is the directory name and
index key for good. If the plan already exists, stop and point at editing it
instead. If the notes branch is missing, offer `/setup-personal-notes`.

## 2. Where the content comes from

Ask which applies; they can combine:

- **An existing document** - a roadmap file, notes in the conversation, or a plan
  approved in plan mode earlier in this session. Read it in full, and keep its
  detail: structured facts (branch, pull request, status, blockers) become items,
  and everything else (rationale, history, conventions) goes into `roadmap.md`.
- **Existing branches or pull requests to track** - check them live in step 4
  rather than trusting a description of their state.
- **From scratch** - ask for the title, a one-line description, the repository,
  and whether the work has sequential waves and parallel tracks or is one flat
  list. Do not impose waves and tracks on work that has none.

## 3. Draft the structure

Waves, tracks, dependencies and statuses are the user's judgment calls: ask
(`AskUserQuestion`) about anything costly to redo rather than inventing it. For
each item, check whether it is really a change to an unlanded item rather than a
new one, per `${SCOPE_DECISION_DOCUMENT}`.

## 4. Validate and check against GitHub

Write the draft to `/tmp/plan.yaml` and run:

```bash
python3 -m "${PULL_REQUEST_STATE_MODULE}" --plan /tmp/plan.yaml --output /tmp/pr_data.json
```

It validates the manifest first - a non-zero exit prints exactly what is wrong -
then writes each named pull request's state. Set every item's `status` from that
(merged → `done`, open → `in_progress`, closed unmerged → `deferred` with a note),
and note any disagreement with the source material in the item's `notes`.

## 5. The tracking issue

Ask whether the plan wants one: it is where structural changes to the plan are
recorded for other sessions to read. If yes, create an issue titled
`[plan-tracking] <plan-id>` explaining that it is not a work item, and record its
number as `tracking_issue`. If issues are disabled (a `410`), open a draft pull
request of the same title from an empty-commit branch `plan-tracking-<plan-id>`
instead, and record that number.

## 6. Save

```bash
bash "${SAVE_PLAN_SCRIPT}" <plan-id> --manifest /tmp/plan.yaml --roadmap /tmp/roadmap.md
```

This pushes both files and regenerates the branch index in one commit.

## 7. Publish and report

Run `/plan-dashboard <plan-id>`. Do not refresh the master index or replace a
migrated source document unasked; offer to.

Report the plan id, item/wave/track counts, the dashboard and tracking-issue
links, and - explicitly - every judgment call you made and every disagreement
between the source and GitHub.
