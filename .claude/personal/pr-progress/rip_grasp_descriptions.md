## PR #588 (cram2) - rip_grasp_descriptions

User committed/pushed up to b7764d3a4 ("Faster demo now": drawer opened once before the transport,
opening_ratio fix, bowl from closest grasps). Demo CI Run Script 363 s; coraplex tests 627 s (main 1007 s).

2026-10-03 round 2, uncommitted, for review (user: NEVER commit, no git ops on the checkout;
decisions in scratchpad DECISIONS_FOR_REVIEW.md of session 1536c00c):
- ReachabilityLocation.candidates skips candidates farther (along the floor) than
  arm.approximate_length(); same draws, same order (user rejected farthest_reaching_distance and
  map-shaping caps reshuffle draws). Demo 257 s alone (was 331 s).
- IsAmongTheClosestGraspsTo ties rank by list order (ties let N+1 grasps through).
- Verified: coraplex suite 678 passed/14 skipped, 9 notebooks, demo 257 s.
Remaining: spoon place (8 candidates) fails on orientation within reach distance.
