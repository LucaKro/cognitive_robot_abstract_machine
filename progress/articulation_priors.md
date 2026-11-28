Step 1 of #173 (calibrated fit uncertainty), uncommitted; decide open choices by experiment.
Done: uncertainty.py (sandwich x n/(n-10) x AR(1) tau; AxisLineEstimate, SlidingDirectionEstimate,
UncertaintyCalibration), fits.py estimates, experiments calibration.py + UncertaintyErrors, tests, docs.
Rejected (measured): Newey-West HAC, Sokal tau, power law, Gaussian-likelihood calibration.
Next: RBO run -> calibration numbers into library + RBO doc; report all decisions; commit only on Luca's word.
