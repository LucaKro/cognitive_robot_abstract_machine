PR #22 (fork, draft) = upstream cram2#672, same branch. cram2/main merged in (27df0019).
Tom's upstream review (18 threads) applied in d10d969f: Covariance class, plain `variables`
field, validate(), product_with_gaussian_likelihood(other), Event.is_box and
SimpleInterval.nearest_contained_value in random_events, scipy moments/logpdf/cdf,
lsq_linear mode. Never reply on the upstream PR; answers go to the user in chat.
#11 / #25 (belief-core) still use the old product signature - they need syncing.
Pre-existing bug, not fixed: ProbabilisticCircuit.log_conditional returns 0.0 when the
root is simplified away (separate bug PR).
Next: user review, then upstream re-review.
