---
name: plan-item-resolve
description: Gather everything available about one already-underway tracked plan item (its plan.yaml entry, roadmap.md history, the real state of its branch/PR - conflicts, CI, review comments, and any unresolved review threads on its upstream pull request - and any relevant discussion on its plan's tracking issue), then resolve whatever is stalling it in whichever execution mode is in force - presenting a plan for approval, carrying it out directly on the item's existing branch, or asking which. Invoke as "/plan-item-resolve <plan-id> <item-id>". Use when resolving a blocked, in-progress, or deferred item from a plan-dashboard's "Resolve"/"Resume"/"Reconsider" link, or when the user asks to "resolve", "unblock", "resume", or "reconsider" a specific tracked item.
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion, Skill, EnterPlanMode, ExitPlanMode, mcp__github__update_pull_request
---

# Plan Item Resolve

Generic, plan-agnostic — nothing here may hardcode a specific plan id,
item, or branch. Unlike `plan-item-kickoff` (for an item that hasn't
started), this skill is for an item that already has real state - a
branch, a PR, prior review, a recorded blocker - and needs that state
understood before deciding what to do next.

**Step 1 is research only — it creates nothing and writes no code.**
Step 2 resolves the item's execution mode, and that mode decides what
happens next: `plan` presents the plan and stops until the user approves it,
`auto` carries it out on the item's existing branch without asking, and
`ask` puts the choice to the user. `${EXECUTION_MODES_DOCUMENT}` states what
each mode means and what it still obliges.

This skill never creates a branch or a pull request — the item already has
both, and an item that has neither belongs to `plan-item-kickoff`. Every
invocation starts fresh in the current session; it does not try to detect or
resume any other session.

## 1. Gather the item's context

Follow `${PLAN_ITEM_GATHERING_DOCUMENT}`: load the plan, read the item's brief,
read the roadmap once. The brief already carries the live state of the work - failing
checks by name, unresolved review threads, recent conversation, changed files and the
tracking-issue comments that mention the item - which is where a stall's cause almost
always is. Name the exact failing check or review thread rather than saying "CI is
failing".

Then add the one thing the brief cannot see: if the fork pull request carries the
`in_review_label` from `basstler/upstream.toml` (`in-review`), or the
item's `notes`/`status` say it is under upstream review, the branch also has an
upstream pull request whose review threads live there. Invoke `/upstream-reviews` for
the item's `branch` and read every unresolved thread it reports. If that fails, mention
it when drafting the plan (step 3) and continue.

Read the item branch's actual file contents only where the brief points, before
proposing changes to them.

## 2. Resolve how this item gets resolved

Follow `${EXECUTION_MODES_DOCUMENT}`: run its `resolve --skill resolve`
call, and if the answer is `ask`, put its question — with a recommendation
drawn from what step 1 just turned up, and the reasons behind it.

Do this only once step 1 is done. What decides the recommendation here
is whether the cause of the stall is actually identified: a named failing
check or review comment with an obvious fix argues for going ahead, and a
blocker whose real cause is still a guess argues for planning first.

Pass `--requested <mode>` when the user named a mode in the invocation
itself.

## 3. Draft the plan

Apply `${PLAN_ITEM_GATHERING_DOCUMENT}`'s last section first: anything you
are about to raise as an open question is very often already answered by the
material step 1 gathered — here including the pull request's own review
threads and the tracking issue's discussion.

Also ask whether the item should still exist separately: follow
`${SCOPE_DECISION_DOCUMENT}`. If nothing substantial would remain once the
overlapping edits are removed, folding it into that item is often the
resolution — an item stuck behind its own parent is sometimes stuck because it
was never really a separate item. The same goes when two items turn out to have
built the same thing, which that document's purpose comparison is there to
catch.

Draft a concrete plan to resolve the item: what's actually wrong (cite the
specific failing check, review comment, unresolved upstream review thread,
blocker text, or regressed dependency that's the real cause — never a vague
"something's blocking this"), what changes it requires, in which files, in
what order, and how each part will be verified. Cite where each part of the
plan came from so it can be sanity-checked against the source. Flag
explicitly, never silently paper over:

- Any dependency that regressed or still isn't safe to build on.
- Any conflict between what `blockers`/`notes` says and what the PR's own
  review threads or the tracking issue actually say.
- Any conflict between the fork PR's state and the upstream review: a fork
  PR that is green and out of draft while its upstream pull request has
  unresolved threads is exactly the stall this skill exists to surface.
- Whether upstream review state was read at all, when the item looked
  promoted but `/upstream-reviews` could not be run.
- Anything genuinely unresolved after the check above — say so rather
  than filling the gap with an assumption.

In `plan` mode, present it via `ExitPlanMode` and stop there — nothing below
happens until the user approves it, and whether to carry the approved plan
out in this session or a fresh one is their call. In `auto` mode, don't ask:
the plan is settled the moment it is drafted, and the flags above go into
the record described below instead of into a question.

Either way, write no code in this step — its only output is the plan itself.

## 4. Carry it out — `auto` mode only

Work the plan on the item's existing branch, honoring the standing
conventions step 1 cross-checked: tests first per TDD, commits in the user's
own git identity, no assistant author or co-author trailer.

`${EXECUTION_MODES_DOCUMENT}` states what stays owed while doing it, and the
bar for stopping to ask anyway. Two things are this skill's own, because the
item is already underway rather than being started:

- **The record is an update, not a first draft.** Append what the resolution
  turned out to be to `roadmap.md` — especially a blocker whose recorded
  cause turned out to be wrong — and refresh the PR-progress note and the
  pull request description rather than writing them from scratch. Update the
  item's `blockers`, `notes` and `status` where the resolution changed what
  the item means, then republish the dashboard with `/plan-dashboard`.
- **Conflicts and moved dependencies go through the stack.** A conflict with a
  lower layer, a fix that belongs in a lower layer, or a `main` that moved on is
  handled with `${STACKS_DOCUMENT}`'s commands, not by merging branches into each
  other by hand.
- **The pull request goes back to draft after the push**, per the user's own
  convention, unless they marked it ready themselves — in which case the
  item was finished and this skill should not have been resolving it.

Finish by reporting what was wrong, what was changed, and what was decided —
in `auto` mode this report is the user's first look at the resolution.
