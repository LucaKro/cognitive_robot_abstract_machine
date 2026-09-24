PR #22 (fork, draft) = upstream cram2#672, same branch. cram2/main merged in (27df0019).
Tom's review round 1 applied in d10d969f, Gibbs sampling in 53251b4e, round 2 in b0288a2b:
from_mean_and_covariance removed (constructor + Covariance.from_matrix), shapes in docstrings,
sweeps_per_sample a field, truncated CDF via mvn.cdf(lower_limit), vectorised contains.
Round 2's "validate is not resolved" thread is unclear - asked the user what Tom means.
Never reply on the upstream PR; answers go to the user in chat.
#11 / #25 (belief-core) still use the old API (from_mean_and_covariance, old product) - need syncing.
Pre-existing bug, not fixed: ProbabilisticCircuit.log_conditional returns 0.0 when the
root is simplified away (separate bug PR).
Next: user review, then upstream re-review.
