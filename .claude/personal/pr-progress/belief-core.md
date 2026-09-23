## belief-core (PR #25, draft, stacked on #22)

Review rounds 1+2 done (7 threads resolved). #22 merged in at cd21dea7 (user request),
onto #22's maintainer-reviewed API: Covariance class, `variables` field,
product_with_gaussian_likelihood(other Gaussian over some variables) = the update
(round-2 observation-model signature dropped).
- probabilistic_model (ours): LinearGaussianModel(matrix, offset, Covariance) +
  MultivariateGaussianDistribution.linear_gaussian_transition; closed-form
  expectation/variance via Covariance.variances; SymbolicDistribution.markov_transition,
  product_with_likelihood; ImpossibleEvidenceError.
- giskardpy.motion_statechart.beliefs: BeliefContext, EstimatorNode[ModelT]
  (expectation_/variance_/probability_variable; PublishedProbability pre-builds events).
Keep carrying #22 up by merging when asked; this branch edits #22's files.

Testing: belief tests fixture-free, --noconftest (336 pass). Env recipe: pip deps +
`pip install -e <member> --no-deps`, root with --ignore-requires-python,
urdf_parser_py/xacro copied from sdist. plan_item_brief crashes here (GraphQL 403):
read PR state over REST.

Next: CI on cd21dea7; author review. #22 lands first.
