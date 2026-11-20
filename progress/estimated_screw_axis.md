#132, step 4 of #117. Stack: main <- #142 (both book pages; #145 merged into it) <- #120 <- #130 <- #132 <- #133.
2026-11-20: merged #130 (4d6f317), Twist -> ScrewAxis, ScrewAxisEstimateParameter is ROTATION_PER_UNIT_POSITION / TRANSLATION_PER_UNIT_POSITION, checks use is_normalized / passes_through_origin; tests pass, description updated.
Next: Luca reviews #142, #120, #130, #132; then #133 needs the same merge (`sync --on 132`) and rename.
