## PR #588 (cram2) - rip_grasp_descriptions

User committed/pushed up to b7764d3a4 ("Faster demo now"). Demo CI Run Script 363 s there; coraplex
tests 627 s (main 1007 s).

2026-10-03 later rounds, uncommitted, for review (user: NEVER commit, no git ops on the checkout;
decisions in scratchpad DECISIONS_FOR_REVIEW.md of session 1536c00c, items 9-12):
- ReachabilityLocation.candidates skips candidates farther than arm.approximate_length() (filter,
  same draws/order; user rejected farthest_reaching_distance).
- IsAmongTheClosestGraspsTo ties rank by list order (user: keep).
- One trial world per Plan (Plan.action_trial), caught up instead of re-copied; released when the
  root node ends. RobotDemonstration.debug=False applied in run(); bullet demo main(debug=False).
- Verified: coraplex suite 683 passed/14 skipped, 9 notebooks, demo 193 s alone (default mode).
