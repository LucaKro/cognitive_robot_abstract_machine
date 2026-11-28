Step 1 of #173 (calibrated fit uncertainty), uncommitted; decide open choices by experiment.
Done: uncertainty.py (sandwich x n/(n-10) x AR(1) tau; AxisLineEstimate, SlidingDirectionEstimate,
UncertaintyCalibration), fits.py estimates, experiments calibration.py + UncertaintyErrors, tests, docs.
Rejected (measured): Newey-West HAC, Sokal tau, power law, Gaussian-likelihood calibration.
Calibration (factor, floors, block shrinkage 0.9) measured and in library + RBO doc; tests pass; reported.
Step 1 tails can't be calibrated: they are fits in a wrong minimum (40 % of early fits on 7 joints).
Scratch experiment (prior_fusion.py): prior as an error term inside the fit (weight tau*factor) beats
fusing afterwards: revolute 10 % 10.85 [79.6] -> 2.07 [6.9] deg (correct), 4.18 [8.4] (5 deg off);
without tau*factor no gain. Cauchy prior term scale 5 escapes overconfident priors by 50 % at ~no cost.
Next: Luca decides whether step 2 = prior inside the fit; commit only on his word.
