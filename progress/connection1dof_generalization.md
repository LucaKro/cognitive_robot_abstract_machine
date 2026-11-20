Draft PR #120 (fork): steps 1+2 of plan #117. Stack 152: main <- #142 <- #145 (docs) <- #120 <- 
#130 screw_axis_estimation (estimator core + docs page), #132 estimated_screw_axis (symbolic
screw axis over passive DOFs alpha/beta, commit/reopen, position_is_periodic), then two
alternatives for the axis frame: #133 estimated_axis_frame_conjugated (A*Screw*A^-1, 9 DOFs,
recommended) and branch estimated_axis_frame_in_transforms (pushed, no PR: workflow refuses two
PRs on #132, GitHubWorkflow#319; PR text in scratchpad transforms_body.md). Summary on #117
(comment 6997428382). Estimator related work: artifact JxnrixFmuRj8hDwV1AunqZ; recommended next:
robust loss, covariance for commit rule, geometric BIC. Luca to choose the alternative.
On hold (Stelter): Giskard k-scaling, safe exploration. Labels new-feature/api-break missing on fork.
#120's description starts with Luca's video, https://github.com/user-attachments/assets/face134e-9bee-4bf8-af35-216207ac347b:
keep it as the first line of every --why-file (describe drops anything outside its markers).
Stacked 2026-11-20, all pushed; estimated_axis_frame_in_transforms not updated (214 behind #145).
2026-11-20: #120 now holds screw_axis as krrood ScrewAxis (not Twist), with ScrewAxis.is_normalized/
passes_through_origin; Twist.screw_axis_direction removed (use unit_direction). Pushed a17e0d9,
description updated. #130 synced + pushed (44de521). #117 plan rewritten 2026-11-20 (spatial types, steps 1-6; step 6 =
uncertainty via probabilistic_model numpy parts, no jax), update comment 7050674669. #132/#133 not yet synced: they need the same rename, and #132's
normalization check should call is_normalized.
GARMI demo passes on #120 (earlier failure was a local iai_maps edit). FK expressions identical to main.
