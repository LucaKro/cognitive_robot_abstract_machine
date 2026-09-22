# Gathering a tracked plan item's context

Shared by `plan-item-kickoff` and `plan-item-resolve`: both first answer *what is
already known and already decided about this item?*

## 1. Load the plan

```bash
source .claude/hooks/resolve-personal-notes-config.sh
git fetch "${NOTES_REMOTE}" "${NOTES_BRANCH}" --quiet
git show "FETCH_HEAD:${PLANS_DIR}/<plan-id>/plan.yaml" > /tmp/plan.yaml
```

If that fails because the notes branch or plan does not exist, list the plan ids
under `${PLANS_DIR}` and stop. If the notes branch itself is missing, offer
`/setup-personal-notes`.

## 2. Read the brief

```bash
python3 -m "${PLAN_ITEM_BRIEF_MODULE}" --plan /tmp/plan.yaml --item <item-id>
```

One call gives the item's recorded state (status, notes, blockers, track, prior
session), whether each dependency is ready to build on, and - when it has a pull
request - its failing checks by name, unresolved review threads, recent
conversation, changed files with line counts, the tracking-issue comments that
mention the item, and which files landed siblings in its track changed. If the
item id is wrong it lists the ones that exist.

This is the item's GitHub context. Do not fetch pull requests, check runs,
comments or diffs yourself on top of it; open individual files only where the
brief points (`git fetch origin <branch>` then `git show origin/<branch>:<path>`).

`blockers` is often the most direct statement of what is wrong, and `notes`
routinely carries design calls settled long before this run. A dependency the
brief reports **NOT ready** decides what a branch can be based on: flag it.

## 3. Read the roadmap once

If the roadmap is not already in `CLAUDE.local.md`, read it:

```bash
git show "FETCH_HEAD:${PLANS_DIR}/<plan-id>/roadmap.md"
```

Read it in full, not by grepping for the item: decisions and conventions often
sit in sections that do not name every item, and they bind this item just the
same. `AGENTS.md` is already in your context; do not re-read it.

## 4. Before asking, check it is not already answered

A design call, naming convention or scope boundary is very often already decided
in the brief or the roadmap. Only ask what is genuinely still open, and say what
you checked.
