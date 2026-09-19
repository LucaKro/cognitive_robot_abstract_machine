# PR progress: `claude/probability-concepts-probabilistic-model-m0i2n0`

Plan item `probability-concepts-in-probabilistic-model` (aicon-belief-integration,
wave 2, track *belief core*). PR #11, based on #10's branch. Mode: `auto`; every round
is in the plan's `roadmap.md`.

## Status: second maintainer review round, in progress.

## What the stall actually is

Everything the earlier records point at is clear: CI is **23 of 23 green** on
`8930c6f8`, `mergeable_state` is `clean`, `needs-resolution` is gone, the dependency
(#10) is `open_ready`, and the fork PR carries no `in-review` label so there is no
upstream review to read.

The stall is a **second review round from `tomsch420` on 2026-09-19, 06:15-06:48** -
eleven new threads on `multivariate_gaussian.py`, posted after the previous round
closed. The recorded state (`plan.yaml`'s *"Both review threads are resolved ... the
branch has rejoined promotion"*) predates it and is stale.

Carry forward, again and more sharply: **the recorded state is only ever as fresh as the
round that wrote it.** Two rounds in a row on this item have been stalled by a review
posted after the record said it was clear.

## The plan

### Implement (clearly correct, in scope)

1. `support` uses `ProbabilisticModel.universal_simple_event()` rather than rebuilding
   it from `reals()` - r4052429466.
2. `scipy_distribution`'s `-> Any` is the wrong type hint - r4052434804.
3. `_over_rows` -> `_marginal_over_variable_indices` - r4052438074.
4. `apply_linear_map`'s docstring warns that it changes what the variables *mean*, which
   the variables themselves do not reflect - r4052460610.

### Move, per the user's call on the Kalman seam

5. `apply_linear_map` and `apply_added_covariance` move out of the distribution into
   `GaussianBelief.predict`, where the sensor-facing wrapper is - r4052462622. The
   measurement update (`conditional_on_measurement`) **stays** in `probabilistic_model`:
   that `conditional(point)` *is* the Kalman update in closed form is this item's whole
   recorded premise. Both halves of that thread answered on it.

### The truncated distribution's correctness cluster

The maintainer calls three things on `TruncatedMultivariateGaussianDistribution`
outright incorrect, and they resolve together once truncation is narrowed:

6. Truncation is supported **only over a box** - a simple event with exactly one
   interval per variable. Anything else needs a circuit and this class is not what
   should be used - r4052470661.
7. `log_mode` is *not* intractable when the mean is cut away: a strictly log-concave
   density on a convex box has exactly one maximum - r4052463820, r4052465083.
8. `log_conditional` *does* exist: condition the untruncated Gaussian, then confine the
   result to the slice the box makes at that value - r4052466049.
9. `sample` draws exactly through `scipy.stats.truncnorm` where the variables do not
   co-vary, instead of rejecting - r4052469095, r4052483009. Rejection stays only for
   the correlated case, which is also a partial answer to the older open thread
   r4050610620.

### Reply only - his call, not mine

10. `_point_mass_at` - r4052444869 and r4052448442 ask for the circuit not to be built
    here and perhaps for an error, which **contradicts his own earlier thread**
    r4050566432 (*"its defined to be the dirac impulse in that case"*), acted on in
    `8930c6f8`. Put back to him rather than flipped a second time.
11. Triangular covariance storage - r4052454927 is explicitly optional (*"if you want to
    go for gigachad storage"*) and undoes the plain-arrays shape he asked for in the
    same review. Replied, not done.
12. Discrete variables via an encoding - r4050505434, unchanged and still his call.

## Two tests pin behaviour the review says is wrong

`test_the_mode_is_intractable_once_the_mean_is_cut_away` and
`test_conditioning_a_truncated_distribution_is_not_answered` assert exactly what 7 and 8
overturn. `AGENTS.md`'s *"never modify the test"* guards against weakening an assertion
to dodge a failure; this is the case `pose-covariance-on-shared-quantities` already
recorded as different - the assertion itself is the thing being corrected. Replaced by
tests of the answers, and said so in the round's record.

## Flagged, not silently papered over

- **#15's premise is gone and its record still relies on it.** `plan.yaml` for
  `pose-covariance-on-shared-quantities` says `SpatialVariables.pose` *is* a
  `Quantities`; `8930c6f8` deleted `Quantities` and its module. This item's own
  PR-progress note already recorded #15 as broken by it. The two records disagree and
  #15 needs its own resolve round.
- **The tracking-issue subscription was refused** by this session's permission mode, as
  on every earlier round.

## Next

- Verify against the pre-change baseline in this container: 66 distribution tests and
  29 belief tests, both green on `8930c6f8` before any edit.
- Push, reply on every thread, and leave 10-12 open for the maintainer.
- CI on the pushed commit.
