## belief-core (PR #25, draft, stacked on #22)

Review rounds 1+2 done (all 7 threads answered + resolved). Head 46135e11.
- probabilistic_model: LinearGaussianModel(matrix, offset, covariance) (+ without_offset);
  MultivariateGaussianDistribution.linear_gaussian_transition(model),
  product_with_gaussian_likelihood(observation_model, observed) (#22 signature changed,
  user's choice), closed-form expectation/variance (publish 925us -> 45us);
  SymbolicDistribution.markov_transition(MultinomialDistribution), product_with_likelihood;
  ImpossibleEvidenceError.
- giskardpy.motion_statechart.beliefs: BeliefContext (ProbabilisticModel by variable),
  EstimatorNode[ModelT] (create_initial_distribution/predict/update->None if no evidence;
  expectation_/variance_/probability_variable; PublishedProbability pre-builds events).
This branch edits #22's multivariate_gaussian.py + tests: carry #22 changes up with
gh stack, never merge by hand.

Testing: giskardpy belief tests fixture-free, --noconftest. Env recipe: pip deps +
`pip install -e <member> --no-deps`, root with --ignore-requires-python,
urdf_parser_py/xacro copied from sdist. plan_item_brief crashes here (GraphQL 403):
read PR state over REST (`gh api .../pulls/25/ccr/review_threads`).

Next: CI on 46135e11; author review round 3. #22 lands first.
