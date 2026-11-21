#130: pushed 96db4e9 (2026-11-21): sampled-pair guess (30 pairs, errors capped at 3 sigma), soft_l1 at 3 sigma, scipy RigidTransform instead of CompiledScrewDisplacement (scipy>=1.16), NotHomogeneousTransformationError; 164 passed; description updated.
Pushed a9434ad (docs passage shortened); all threads resolved.
2026-11-21: #117 step 3 and step 6 rewritten (one screw fit, sampled pairs, soft_l1, annotation priors as a term, drawer-as-slight-turn for behaves_like). Pushed 8055a9b: examples page rewritten for ScrewMotionFit; description updated.
Next: #130 can leave draft once Luca has reviewed 8055a9b (ready only on his word). Tolerant behaves_like + commit: on hold (recorded in #117 "On hold", with the rule and numbers), not for #132 either. Then #132/#133.
Tests: --orm-build=always after changing fit fields; needs the conftest.
