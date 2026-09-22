PR #22 (draft, base main): MultivariateGaussianDistribution extracted from #11,
with all of Tom's third-round review applied (commit 179c0a36). Contains only
probabilistic_model files: multivariate_gaussian.py, exceptions.py, pyproject scipy
pin, test_multivariate_gaussian.py. Wording uses "variable", not "quantity".
Open upstream on #11 (awaiting tomsch420): discrete variables -> circuit?;
which scipy method for correlated truncated sampling. Circuits cannot hold a
multivariate leaf yet (leaf() univariate-only, no __deepcopy__) - possible follow-up.
Next: user review; once landed, rebase #11 onto it.
