Draft PR #120 (fork), on main since #121 merged upstream (cram2#1009): steps 1+2 of plan #117.
Pushed 870c11f (position_for_translation + docs page screw_axis_connections.md).
Stacked on it: draft #130 (branch screw_axis_estimation), step 3 part A: estimator core
(semantic_digital_twin.articulation_estimation: fits by least squares, BIC selection,
create_connection in the canonical frame) + docs page articulation_estimation.md. 18 tests pass.
Step 3 part B (estimated ScrewAxisConnection, branch estimated_screw_axis_connection, no commits):
waiting for Luca's decision on #117 (comment 6996475244): Connection forbids non-constant
parent_T_connection, so recommended: conjugate the motion by an estimated axis frame A.
On hold (Stelter): Giskard k-scaling, safe exploration. Labels new-feature/api-break missing on fork.
