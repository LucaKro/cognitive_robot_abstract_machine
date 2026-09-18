# PR progress: `claude/probability-concepts-probabilistic-model-m0i2n0`

Plan item `probability-concepts-in-probabilistic-model` (aicon-belief-integration,
wave 2, track *belief core*). PR #11, draft. Based on #10's branch
`claude/belief-integration-gaussian-zi1o82`, not `main` — re-base once #10 lands.
Mode: `auto`; the settled plan and both rounds are in the plan's `roadmap.md`.

## Status: first review round addressed and pushed (`6da3fc22`). CI not yet run on it.

## Round 1 — "an actual datastructure instead of just np array"

Two threads, both from the author, on `Mean.values` and `Covariance.values`.
Fourth time this ask has come up on this plan (#10 twice, #9 once); first time aimed
at what the type *holds* rather than at what callers hand it.

- `Mean` holds `estimates` (per quantity), `Covariance` holds `uncertainty` (per
  ordered pair). Neither holds an array.
- `as_array` / `from_array` are the only two places numpy appears on either type.
- `Covariance.from_array` keeps both directions of a pair apart — symmetrizing on the
  way in would have made the symmetry test true by construction. Checked by removing
  the symmetrization and watching that test fail.
- Thread 1 **resolved**. Thread 2 **left open**: the remaining arrays are
  `ProbabilisticModel`'s abstract `log_likelihood`/`sample` signatures, and whether to
  change those across the whole package is the author's call, not this item's.

## Verified locally

- 272 passed, collectible `probabilistic_model` suite (12 layout + 68 distribution).
- 29 passed on #10's belief tests — only `.values` → `.as_array` where they genuinely
  want the matrix (inverse, transpose, eigenvalues).
- 20 passed on `test_dependency_declarations.py`.
- CI was 23/23 green on `f431efa7`, the commit before these fixes.

## Next

- Watch CI on `6da3fc22`.
- Thread 2 needs the author's answer on the base-class signatures.
- After #10 lands: rebase onto `main`.

## Open

- **Dashboard still not republished.** `Artifact` treats this account's plan dashboard
  as a public third-party artifact: a read returns only a summary, so the publish
  keeps refusing with "you haven't viewed the latest version". `force:true` would
  discard another session's publish. Needs the user's call.
- Tracking-issue (#7) subscription refused by this session's permission mode.
- **#12 (`estimator-node-base`) heads-up**, now three items: import
  `Quantities`/`Reading` from `probabilistic_model`, use the renamed exceptions, and
  read a covariance matrix via `.as_array`. All three are in #11's description.
