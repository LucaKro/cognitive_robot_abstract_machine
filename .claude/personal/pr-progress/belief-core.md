## belief-core (PR #25, draft, stacked on #22)

Review round 1 (4 threads, all answered + resolved): user chose to move the
filter steps into probabilistic_model within this PR (reverses original note).
Now (e9ed7221):
- probabilistic_model: MultivariateGaussianDistribution.linear_gaussian_transition
  (Kalman predict; _joint_with_observation reuses _linearly_transformed),
  SymbolicDistribution.markov_transition(MultinomialDistribution) and
  product_with_likelihood; ImpossibleEvidenceError.
- giskardpy.motion_statechart.beliefs: BeliefContext (ProbabilisticModel by
  variable: add/replace/distribution_of), EstimatorNode[ModelT]
  (create_initial_distribution/predict/update->None if no evidence; publishes
  mean/variance of numeric, probability per value of symbolic vars;
  PublishedValue StrEnum names them; UnpublishedValueError).
This branch edits #22's multivariate_gaussian.py + its tests: carry #22 changes up
with gh stack, never merge by hand.

Testing: giskardpy belief tests fixture-free, run with --noconftest. Env recipe:
pip deps + `pip install -e <member> --no-deps`, root package with
--ignore-requires-python, urdf_parser_py/xacro copied from sdist.

Next: CI on e9ed7221; author review round 2. #22 lands first.
