# Stacks of plan items

Shared by `plan-item-kickoff` and `plan-item-resolve`. An item that builds on an
unlanded item is a layer of a GitHub stack of pull requests (`gh stack`), on the
fork, targeting the fork's `main`. GitHub keeps each layer's base, shows the stack
on every pull request in it, and retargets the layers above one that lands. Nothing
in `basstler` maintains stacks; these commands are the whole procedure.

If `gh stack` is not available, the session-start summary says why. Stop and
report it rather than hand-rebasing the stack; `gh stack <command> --help` is the
reference for anything below.

A cloud session's proxy refuses GitHub's GraphQL API, which `gh stack link`,
`gh stack submit`, `gh stack checkout <pull request number>` and `gh pr create`
depend on. Everything below avoids them: pull requests and stacks go through the
REST API, and `gh stack` only handles the local branches.

## Where an item's branch goes

```bash
python3 -m "${PLAN_STACK_MODULE}" --plan /tmp/plan.yaml --item <item-id>
```

Prints `base` (branch the item's branch from, and open its pull request against)
and `branches` (the stack, bottom first, ending with the item's own). A non-zero
exit means the item rests on two unlanded items at once, which a stack cannot
express: raise it with the user; the plan's `depends_on` or the order of landing
has to change.

Once the item's pull request exists, register it as the stack's top layer:

```bash
python3 -m "${STACK_REGISTRATION_MODULE}" --plan /tmp/plan.yaml --item <item-id> \
    --pull-request-number <number>
```

It creates the stack from the layers below when none of them is in one yet, and
extends it otherwise; running it again changes nothing. It needs each layer
below to have its `pull_request_number` in the plan.

## A change to a lower layer

A stack this clone has not seen yet has to be adopted first, from the branches
`plan_stack` prints:

```bash
git fetch origin
git branch --track <branch> origin/<branch>    # for each branch not local yet
gh stack init --base main <branches, bottom first>
```

Then commit on the lower layer's branch and carry it up through every layer above:

```bash
gh stack checkout <lower layer's branch>
# ... commit ...
gh stack rebase --upstack       # conflicts stop it: fix, git add, gh stack rebase --continue
gh stack push
```

`gh stack rebase --abort` returns every branch to where it was. Resolve each
conflict once, in the layer it arises in; do not merge lower layers into upper
ones by hand.

## Keeping up with upstream

The fork's `main` follows `cram2/main`; the stack follows the fork's `main`:

```bash
gh api -X POST repos/<fork owner>/<fork name>/merge-upstream -f branch=main
gh stack sync                            # fetch, rebase the stack onto it, push
```

Run it when the brief shows a layer behind or conflicting, not on every session.
In a cloud session `gh stack sync` warns that each branch "has no PR": it looks
them up over GraphQL, and the warning does not affect the sync.

## Releasing to upstream

Opening pull requests on `cram2` is the user's call and the user's action, never a
session's. Once a layer has been opened there, label its fork pull request
`in-review` (`in_review_label` in `basstler/upstream.toml`), so resolve sessions
read its upstream review threads with `/upstream-reviews`.
