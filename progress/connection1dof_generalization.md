Draft PR #120 (fork) on main: steps 1+2 of plan #117. Stack 131 on it:
#130 screw_axis_estimation (estimator core + docs page), #132 estimated_screw_axis (symbolic
screw axis over passive DOFs alpha/beta, commit/reopen, position_is_periodic), then two
alternatives for the axis frame: #133 estimated_axis_frame_conjugated (A*Screw*A^-1, 9 DOFs,
recommended) and branch estimated_axis_frame_in_transforms (pushed, no PR: workflow refuses two
PRs on #132, GitHubWorkflow#319; PR text in scratchpad transforms_body.md). Summary on #117
(comment 6997428382). Estimator related work: artifact JxnrixFmuRj8hDwV1AunqZ; recommended next:
robust loss, covariance for commit rule, geometric BIC. Luca to choose the alternative.
On hold (Stelter): Giskard k-scaling, safe exploration. Labels new-feature/api-break missing on fork.
