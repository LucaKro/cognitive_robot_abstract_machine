#130: pushed 96db4e9 (2026-11-21): sampled-pair guess (30 pairs, errors capped at 3 sigma), soft_l1 at 3 sigma, scipy RigidTransform instead of CompiledScrewDisplacement (scipy>=1.16), NotHomogeneousTransformationError; 164 passed; description updated.
Pushed a9434ad (docs passage shortened); all threads resolved.
2026-11-21: #117 step 3 and step 6 rewritten (one screw fit, sampled pairs, soft_l1, annotation priors as a term, drawer-as-slight-turn for behaves_like). Uncommitted: examples/articulation_estimation.md rewritten for ScrewMotionFit (runs, builds alone in scratchpad aebook); description draft pr130_changes.md updated for it, push describe after the commit.
Next: on Luca's word, commit the page, push, describe; then #130 can leave draft. After that: behaves_like tolerances proposal (discuss first), then #132/#133.
Tests: --orm-build=always after changing fit fields; needs the conftest.
