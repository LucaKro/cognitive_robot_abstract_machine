# PR progress: `claude/probability-concepts-probabilistic-model-m0i2n0`

Plan item `probability-concepts-in-probabilistic-model` (aicon-belief-integration,
wave 2, track *belief core*). PR #11, draft. Based on #10's branch
`claude/belief-integration-gaussian-zi1o82`, not `main` — re-base once #10 lands.
Mode: `auto`; the settled plan is in the plan's `roadmap.md` under this item.

## Status: implemented and pushed (`f431efa7`), awaiting first CI run and author review.

## What landed

1. `probabilistic_model/quantities.py` — `Quantities`, in its own module so
   `pose-covariance-on-shared-quantities` can use the layout without a Gaussian.
2. `probabilistic_model/distributions/multivariate_gaussian.py` — `Mean`,
   `Covariance`, `Reading`, `MultivariateGaussianDistribution`,
   `TruncatedMultivariateGaussianDistribution`.
3. `probability_of_simple_event` via `scipy.stats.multivariate_normal.cdf(...,
   lower_limit=...)`; `scipy>=1.10` pinned.
4. `log_truncated` answers with the truncated type, not `Self`.
5. Three exceptions renamed off "belief" into `probabilistic_model/exceptions.py`;
   `UnknownBeliefError`/`DuplicateBeliefError` stayed in giskardpy.
6. `GaussianBelief` swapped onto the distribution; interface unchanged. Needed two
   extra distribution operations so no probability arithmetic stayed in giskardpy:
   `apply_linear_map` and `apply_added_uncertainty`.
7. `probabilistic_model` declared in giskardpy's `[project] dependencies` and
   `[dependency-groups] workspace`.
8. `scripts/format_docstrings.py` run on every modified file.

## Verified locally (a first for this plan)

- 266 passed across the collectible `probabilistic_model` suite (74 new).
- 29 passed on #10's belief tests, unchanged but for imports/renames.
- 20 passed on `test_dependency_declarations.py`, including the giskardpy check that
  caught #10 — confirmed load-bearing by removing the declaration.
- Key assertions mutation-checked: dropping the correlation, not narrowing the
  conditional covariance, and removing the dependency declaration each fail only
  their own tests.

## Next

- Watch the first CI run. ORM regeneration is still CI's to confirm.
- After #10 lands: rebase onto `main`.

## Open

- **Dashboard not republished.** `Artifact` treats this account's plan dashboard as a
  public third-party artifact, so a read returns only a summary and the publish
  refuses with "you haven't viewed the latest version". Did not use `force:true`,
  which would discard #12's session's publish. Needs the user's call.
- Tracking-issue (#7) subscription was refused by this session's permission mode.
- **#12 (`estimator-node-base`) heads-up**: it stacks on #10 too and has no
  implementation commits yet. It should import `Quantities`/`Reading` from
  `probabilistic_model` and use the renamed exceptions. Spelled out in #11's
  description.
