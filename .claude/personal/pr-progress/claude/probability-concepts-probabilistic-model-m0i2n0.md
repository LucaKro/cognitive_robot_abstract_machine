# PR progress: `claude/probability-concepts-probabilistic-model-m0i2n0`

Plan item `probability-concepts-in-probabilistic-model` (aicon-belief-integration,
wave 2, track *belief core*). PR #11, draft. Based on #10's branch
`claude/belief-integration-gaussian-zi1o82`, not `main` — re-base once #10 lands.
Mode: `auto`, so nothing here was approved in advance; the settled plan is in the
plan's `roadmap.md` under this item.

## Plan

1. `probabilistic_model/quantities.py` — `Quantities` moved out of giskardpy. Its own
   module, because `pose-covariance-on-shared-quantities` needs the layout without a
   Gaussian.
2. `probabilistic_model/distributions/multivariate_gaussian.py` —
   `MultivariateGaussianDistribution(ProbabilisticModel)` with `Mean`, `Covariance`
   and `Reading`. `log_conditional` is the Kalman update; a measurement update is
   conditioning the joint over quantities and readings.
3. `probability_of_simple_event` via `scipy.stats.multivariate_normal.cdf(...,
   lower_limit=...)`, summed over the boxes a union-valued interval makes. Needs
   `scipy>=1.10` pinned in `probabilistic_model/pyproject.toml`.
4. `TruncatedMultivariateGaussianDistribution` — what `log_truncated` answers with,
   since a truncated correlated Gaussian is not Gaussian.
5. Exceptions renamed off "belief" into `probabilistic_model/exceptions.py`;
   `UnknownBeliefError`/`DuplicateBeliefError` stay in giskardpy.
6. Swap `GaussianBelief`'s internals; public interface unchanged.
7. Declare `probabilistic_model` in giskardpy's `[project] dependencies` and
   `[dependency-groups] workspace` — the lesson #10's CI taught.
8. `scripts/format_docstrings.py` on every modified file.

Tests first throughout, per `AGENTS.md`.

## Done

- Setup check green (installed `markdown`/`nh3`); all three PR labels exist.
- Item gathered: plan.yaml, roadmap.md in full, #10's diff/description/comments.
- Dependency `belief-context-and-gaussian` reports `open_ready`.
- Scope check run: `beliefs/gaussian.py` absent from `main`, introduced by #10 only.
- Branch + draft PR #11 opened; manifest flipped to `in_progress`; roadmap section
  recorded.
- Verified locally: `scipy`'s `lower_limit` rectangle probability matches Monte Carlo;
  the existing 114 `probabilistic_model` distribution tests pass in this container.

## Next

- Write the failing tests for `Quantities` in its new home, then move it.
- Then the distribution, then the truncated one, then the giskardpy swap.

## Open

- Tracking-issue (#7) subscription was refused by this session's permission mode.
- The giskardpy half is CI's to verify: the root `test/conftest.py` imports
  `urdf_parser_py`, which is not on PyPI.
