# Stacks of plan items

Shared by `plan-item-kickoff` and `plan-item-resolve`. An item that builds on an
unlanded item is a layer of a GitHub stack of pull requests (`gh stack`), on the
fork, targeting the fork's `main`. GitHub keeps each layer's base, shows the stack
on every pull request in it, and retargets the layers above one that lands. Nothing
in `basstler` maintains stacks; these commands are the whole procedure.

If `gh stack` is not available, the session-start summary says why. Stop and
report it rather than hand-rebasing the stack; `gh stack <command> --help` is the
reference for anything below.

## Where an item's branch goes

```bash
python3 -m "${PLAN_STACK_MODULE}" --plan /tmp/plan.yaml --item <item-id>
```

Prints `base` (branch the item's branch from, and open its pull request against),
`branches` (the stack, bottom first) and `link` (the command that registers the
stack, or `null` when nothing unlanded is below the item). A non-zero exit means
the item rests on two unlanded items at once, which a stack cannot express: raise
it with the user; the plan's `depends_on` or the order of landing has to change.

## A change to a lower layer

Commit on the lower layer's branch, then carry it up through every layer above:

```bash
gh stack checkout <lower pull request number>
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
gh repo sync <fork owner>/<fork name>    # fast-forwards the fork's main from cram2
gh stack sync                            # fetch, rebase the stack onto it, push
```

Run it when the brief shows a layer behind or conflicting, not on every session.

## Releasing to upstream

Opening pull requests on `cram2` is the user's call and the user's action, never a
session's. Once a layer has been opened there, label its fork pull request
`in-review` (`in_review_label` in `basstler/upstream.toml`), so resolve sessions
read its upstream review threads with `/upstream-reviews`.
