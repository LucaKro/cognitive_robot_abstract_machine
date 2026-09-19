# PR progress: `claude/probability-concepts-probabilistic-model-m0i2n0`

Plan item `probability-concepts-in-probabilistic-model` (aicon-belief-integration,
wave 2, track *belief core*). PR #11, based on #10's branch. Mode: `auto`; every round
is in the plan's `roadmap.md`.

## Status: second maintainer review round answered and pushed (`0c081d59`). 8 of 12 threads resolved.

## What the stall was

Not CI (23/23 green on `8930c6f8`), not a conflict (`clean`), not a label, not the
dependency (#10 `open_ready`), and no `in-review` label so no upstream review to read.
**Eleven new threads from `tomsch420`, 06:15-06:48**, after the record said clear.

Second round running that this has happened. Carry it forward: **the recorded state is
only ever as fresh as the round that wrote it**, and a dashboard reading checks, labels
and mergeability cannot see a review posted since.

## What landed

One change and its consequences, not eleven fixes. **Truncation is supported only over a
box** (`EventIsNotABoxError` otherwise) — and that is what made the two recorded
intractables answerable:

- `log_mode`: strictly log-concave density on a convex box has exactly one maximum. The
  mean while the box holds it, otherwise the nearest point *in the distribution's own
  metric*; an open end nudged with `nextafter`, as the univariate truncated Gaussian
  already does.
- `log_conditional`: the Gaussian conditional confined to the slice the box makes.
- `sample`: exact via `truncnorm` where quantities do not co-vary. Rejection only where
  they do — per-variable truncnorms would sample the *wrong* distribution for a
  correlated Gaussian.

Plus `universal_simple_event()` for `support`, a real type on `scipy_distribution`, and
`_over_rows` → `_marginal_over_variable_indices`.

**Prediction moved to `GaussianBelief.predict`; the measurement update stayed.** Settled
with the user. His own objection is the reason: a general linear map makes `x` into
`0.5x + 2y` and the variable still says `x`. A Kalman transition does not — same
quantities, one cycle later.

## Verification

Baseline vs change, same container, both run twice: probabilistic_model 396→400,
motion_statechart 114→115 passed with **byte-identical failure sets**, version_test 21
both. Eight mutations each failed only the tests naming them; the sampling one *hangs*
rather than fails, which is the proof rejection cannot reach that box.

The symmetry test needed rewriting before it bit: `F P Fᵀ` stays exactly symmetric for
many matrices, and drift washes out within ~20 cycles. It asserts on every cycle now.

## Open for the maintainer (not blocked on us)

1. **`_point_mass_at`** — his two comments contradict his own earlier thread that asked
   for exactly this construction. Asked rather than flipped twice.
2. **Triangular covariance storage** — optional, and pulls against the plain-arrays shape
   he asked for in the same review.
3. **`conditional_on_measurement`** — explained; offer to move it out stands.
4. **Discrete-variable encoding** + rejection-sampler bound — unchanged from round 1.

## Next

- CI on `0c081d59`: 22/23 green, `semantic_digital_twin/scripts/test_exercises.sh` still
  running (untouched by this diff).
- **#15 needs its own resolve round**: its premise was collapsing `PoseCovariance` onto
  `Quantities`, which `8930c6f8` deleted. Flagged in its `plan.yaml` notes.
- **#11 is out of draft and stays there** unless told otherwise — un-drafting is this
  repo's record of author review, and re-drafting withdraws it from the promotion queue.
  Unlike the restack that set that precedent, this round *did* change production code, so
  the standing "re-draft after any push" rule is reported rather than applied unasked.
