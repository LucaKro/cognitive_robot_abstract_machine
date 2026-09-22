---
name: plan-item-kickoff
description: Gather everything available about one tracked plan item (its plan.yaml entry, roadmap.md history/design context, its dependency chain's live GitHub state, and patterns from already-landed sibling items in the same track), then start it in whichever execution mode is in force - presenting a plan for approval, implementing it directly, or asking which - opening the item's branch and draft pull request and recording its manifest state before implementation begins. Invoke as "/plan-item-kickoff <plan-id> <item-id>". Use when starting work on a specific item from a plan-dashboard's "Start now" link, or when the user asks to "start", "kick off", or "plan out" a specific tracked item.
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion, Skill, EnterPlanMode, ExitPlanMode, mcp__github__create_pull_request, mcp__github__update_pull_request
---

# Plan Item Kickoff

Generic, plan-agnostic — nothing here may hardcode a specific plan id, item,
or branch. Gathers everything a session doing this item's work would
actually want before starting, then starts it.

**Step 1 is research only — it creates nothing and writes no code.**
Step 2 resolves the item's execution mode, and that mode decides what
happens next: `plan` presents the plan and stops until the user approves it,
`auto` records the plan and implements it without asking, and `ask` puts the
choice to the user. `${EXECUTION_MODES_DOCUMENT}` states what each mode
means and what it still obliges; this file only says where in the sequence
each one applies.

Step 4 opens the item's branch and draft pull request and records its
manifest state, before any implementation begins. It runs in both modes, so
the item stops reading as `not_started` the moment it isn't.

## 1. Gather the item's context

Follow `${PLAN_ITEM_GATHERING_DOCUMENT}`: load the plan, read the item's brief,
read the roadmap once.

Then learn the shape the item should take from what already landed: the brief lists the
files each landed item in the same track changed. Open the few that show the pattern this
item should follow (file layout, test conventions) with `git show origin/<default
branch>:<path>`, rather than reading whole diffs. If the item's own branch already
exists, read what is on it before proposing anything: the plan must build on real state.

## 2. Resolve how this item gets started

Follow `${EXECUTION_MODES_DOCUMENT}`: run its `resolve --skill kickoff`
call, and if the answer is `ask`, put its question — with a recommendation
drawn from what step 1 just turned up, and the reasons behind it.

Do this only once step 1 is done. The recommendation is worth something
only if it can cite whether the item's `notes` and `roadmap.md` section
actually settle the design calls, whether every dependency came back ready,
and whether a sibling PR already established the pattern — none of which is
known before then.

Pass `--requested <mode>` when the user named a mode in the invocation
itself; otherwise let the setting decide, rather than inferring a mode from
how well-specified the item looks.

## 3. Draft the plan

Apply `${PLAN_ITEM_GATHERING_DOCUMENT}`'s last section first: anything you
are about to raise as an open question is very often already answered by the
material step 1 gathered.

Then check the item is still the right unit of work, now that you know which
files it will touch: follow `${SCOPE_DECISION_DOCUMENT}`. If nothing
substantial would remain once the overlapping edits are removed, propose
folding rather than opening a branch that will have to be folded later.
Otherwise carry on, but say which files overlap so the two branches do not
build the same thing twice.

Draft a concrete implementation plan: what changes, in which files, in what
order, and how each part will be verified (tests first, per TDD). Cite where
each part of the plan came from (the item's own notes, a specific sibling
PR, a `roadmap.md` section) so it can be sanity-checked against the source
instead of just trusted. Flag explicitly, never silently paper over:

- Any dependency that isn't actually ready to build on yet.
- Any conflict between the item's `notes` and what a sibling PR or
  `roadmap.md` actually says.
- Anything the gathered context left genuinely unresolved after the check
  above — say so rather than filling the gap with an assumption.

In `plan` mode, present it via `ExitPlanMode` and stop there — nothing below
happens until the user approves it. In `auto` mode, don't ask: the plan is
settled the moment it is drafted, and the flags above go into the roadmap
section and the pull request description instead of into a question.

Either way, do not touch git, create a branch, or write any code in this
step — its only output is the plan itself.

## 4. Bootstrap the item — before implementing, not after

The moment the plan is settled, the branch, the draft pull request, the
item's `branch`/`session`/`pull_request_number` fields and its roadmap
section are all derivable, and none of them depends on a line of the
implementation. Doing them at the end instead means the manifest says
`not_started` with no branch for the entire length of the work, which every
dashboard, kickoff and resolve run downstream reads as truth.

So run this first, before the first edit:

Find where the branch goes, per `${STACKS_DOCUMENT}`, then create the branch and
its draft pull request yourself and hand the number over:

```bash
source .claude/hooks/resolve-personal-notes-config.sh
python3 -m "${PLAN_STACK_MODULE}" --plan /tmp/plan.yaml --item <item-id>
# -> {"base": <base-branch>, "branches": [...]}
git checkout -b <branch> origin/<base-branch>
git commit --allow-empty -m "Bootstrap <item-id>"
git push -u origin <branch>
gh api "repos/<repository>/pulls" -f title="<title>" -f body="<body>" \
    -f head=<branch> -f base=<base-branch> -F draft=true --jq .number
python3 -m "${STACK_REGISTRATION_MODULE}" --plan /tmp/plan.yaml --item <item-id> \
    --pull-request-number <number>    # top layer of its stack; nothing for a single layer
python3 -m "${PLAN_ITEM_BOOTSTRAP_MODULE}" open \
    --plan <plan-id> --item <item-id> \
    --branch <branch> --base <base-branch> \
    --session <this session's url> \
    --pull-request-number <number>
python3 -m "${PLAN_ITEM_BOOTSTRAP_MODULE}" record \
    --plan <plan-id> --item <item-id> \
    --status in_progress --roadmap-section <file>
```

`open` before `record`: the pull request number does not exist until the pull
request does. `open` writes the branch, session and pull request number onto
the item and flips it to `in_progress`; `record` appends the settled plan to
`roadmap.md`. Both print a one-line JSON report led by `status` and
`exit_code`.

**Why a session creates the pull request rather than the script.** `gh` runs on
your token, so the pull request is yours. Use `gh api` rather than `gh pr
create`: the latter goes through GitHub's GraphQL API, which a cloud session's
proxy refuses. The script
can create one — with `--pull-request-title`/`--pull-request-body` instead of
`--pull-request-number`, verified live — but a pull request it creates is
attributed to the app its requests are proxied through rather than to the
person whose work it is, the same authorship problem `AGENTS.md` rules out for
commits. Creating it yourself keeps your identity on it. The creating path is
there for an unattended run whose credential is a real one; if you use it,
`open` publishes the branch too, so the three git commands above are yours to
skip.

The branch name is this skill's judgment: whatever this session is designated
to develop on, else the item's `branch`. The base is `plan_stack`'s, and a
refusal from it (two unlanded dependencies at once) is a question for the user,
not a base to pick. If the item's `branch` differs from the one you created,
pass the real one to `open`. `<repository>` is the item's `repository`, else the
plan's `default_repository`.

**Write the settled plan down in both places it belongs**, rather than
leaving it only in the conversation that produced it — in `auto` mode this is
the only record it will ever have:

- **`roadmap.md`**, via `record`'s `--roadmap-section` — the durable record of
  what was decided and *why*, including any assumption or open question the
  plan carries. Not a restatement of the diff.
- **The PR-progress note** — the plan, what is done, and what is next, kept
  current as the work goes. Write it between `CLAUDE.local.md`'s
  `BEGIN-PR-PROGRESS`/`END-PR-PROGRESS` markers and run
  `.claude/hooks/save-pr-progress.sh` to push it. Do this as soon as the
  branch exists, not at the end: a note written afterwards is a summary, and
  the point of it is that another session can pick the work up mid-flight.

Then republish the dashboard yourself:

```
/plan-dashboard <plan-id>
```

Both operations end here rather than doing it, because only a live session
can call the `Artifact` tool — the script's report hands the command back
rather than pretending it ran. Do not skip it: a published dashboard that is
older than the manifest behind it is the exact staleness this step exists to
close.

In `plan` mode the skill ends here. Whether to implement the approved plan
in this session or a fresh one is the user's call, made after they see it.

## 5. Implement it — `auto` mode only

Work the plan step 3 settled, honoring the standing conventions step 1
cross-checked: tests first per TDD, commits in the user's own git identity,
no assistant author or co-author trailer.

`${EXECUTION_MODES_DOCUMENT}` states what stays owed while doing it — the
pull request stays a draft, its description and the PR-progress note keep
matching what the work actually does, and decisions get recorded as they are
made rather than reconstructed at review time. It also states the bar for
stopping to ask anyway: a decision that changes the item's recorded scope or
contract, is not easily revertible, or departs from the settled plan in a way
the item's own description would not lead a reviewer to expect. Below that
bar, decide and record it.

If the work turns out to need more than this one pull request, that is a
structural change to the plan: it goes to the plan's `tracking_issue` and to
the user, not into a quietly widened branch.

Finish by reporting what was built and what was decided along the way — in
`auto` mode this report is the user's first look at the work, so it stands in
for the plan they never approved.
