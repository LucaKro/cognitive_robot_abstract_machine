Step 1 of #173 (calibrated fit uncertainty), uncommitted; decide open choices by experiment.
Done: uncertainty.py (sandwich x n/(n-10) x AR(1) tau; AxisLineEstimate, SlidingDirectionEstimate,
UncertaintyCalibration), fits.py estimates, experiments calibration.py + UncertaintyErrors, tests, docs.
Rejected (measured): Newey-West HAC, Sokal tau, power law, Gaussian-likelihood calibration.
Calibration (factor, floors, block shrinkage 0.9) measured and in library + RBO doc; tests pass; reported.
Next: Luca reviews; commit only on his word; then step 2 (prior interface, heavy-tailed fusion).
