## belief-core (PR #25, draft, stacked on #22)

Plan (auto mode; full record in the plan roadmap under `belief-core`):
package giskardpy.motion_statechart.beliefs, ORM-excluded by package;
giskardpy declares probabilistic_model + random_events.
- Belief[PredictionT, EvidenceT] base, Statistic StrEnum, VariableStatistic key.
- GaussianBelief on #22's distribution: predict(LinearPrediction) written here,
  update(Reading list) via product_with_gaussian_likelihood.
- BinaryBelief (BinaryTransition, BinaryEvidence), plain probability.
- BeliefContext(ContextExtension): add / belief_of / from_context.
- EstimatorNode: prior/prediction/measure abstract; FloatVariable per statistic;
  observes TRUE iff evidence this cycle.

Testing: fixture-free tests under test/giskardpy_test/test_motion_statechart/
test_beliefs, run with --noconftest (root conftest needs ROS msgs for ORM).
Env recipe: pip deps + `pip install -e <member> --no-deps` for workspace members,
root package with --ignore-requires-python, urdf_parser_py/xacro copied from sdist.

Done: branch, draft PR #25 (stack #26), manifest in_progress, roadmap section;
implementation 23df0ae6 - 33 tests pass locally, mutation-checked; PR body current.
Decided: BinaryBelief update is atomic; plain Generic (no SubClassSafeGeneric -
params only type signatures, package ORM-excluded).
Next: CI on 23df0ae6 (ORM regeneration is CI-only); author review. #22 lands first.
