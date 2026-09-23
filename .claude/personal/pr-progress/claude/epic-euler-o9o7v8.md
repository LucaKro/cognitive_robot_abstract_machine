PR #22 (draft, base main): MultivariateGaussianDistribution extracted from #11
(179c0a36), then circuits made to hold it as a leaf + naming aligned with PM
(f5aaed48): MultivariateLeaf, LeafUnit.replace_by_mixture, leaf columns in the
distribution's own order, mean_of/variance_of dropped for expectation/variance,
normalizing_constant, rejection_sample, "interval" not "stretch".
#11 has NOT been synced with f5aaed48 (its GaussianBelief still calls mean_of etc.).
Open upstream on #11 (awaiting tomsch420): discrete variables; scipy method question.
Pre-existing bug found, not fixed: ProbabilisticCircuit.log_conditional returns 0.0
log-density when the root is simplified away (reproduces on main) - separate bug PR.
Next: user review.
