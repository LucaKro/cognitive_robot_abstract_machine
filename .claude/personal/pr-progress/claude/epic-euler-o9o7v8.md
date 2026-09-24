PR #22 (fork, draft) = upstream cram2#672, same branch. cram2/main merged in (27df0019).
Upstream review rounds: 1 in d10d969f, Gibbs sampling in 53251b4e, 2 in b0288a2b, 3 in 802c11b1
(Truncated class in its own module truncated_multivariate_gaussian.py, burn_in_period_length,
box_contains via SimpleEvent.contains, MultivariateGaussianDistribution.precision computed once).
Round 2's "validate is not resolved" thread was unclear - asked the user what Tom means.
Never reply on the upstream PR; answers go to the user in chat.
#11 / #25 (belief-core) still use the old API (from_mean_and_covariance, old product,
Truncated import path) - need syncing.
Pre-existing bug, not fixed: ProbabilisticCircuit.log_conditional returns 0.0 when the
root is simplified away (separate bug PR).
Next: user review, then upstream re-review.
